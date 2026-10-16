import gc
import hashlib
import json
import time
from dataclasses import dataclass

import torch


def _json_safe_copy(value, field_name):
    try:
        return json.loads(
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-safe plain data") from exc


def _action_digest(actions):
    serialized = json.dumps(
        list(actions),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, value in named_tensors:
        digest.update(str(name).encode("utf-8"))
        if value is None:
            digest.update(b"<none>")
            continue
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BufferCommitRequest:
    version: int
    mode: str
    actions: tuple
    action_digest: str
    iteration: int | None
    reason: str
    mutation_kind: str
    commit_enabled: bool
    committed_def_path: str
    refresh_mode: str
    rebuild_mode: str
    projection_metadata: dict
    runtime_refresh_summary: dict
    effective_config_reference: str = ""
    sizing_actions: tuple = ()
    sizing_action_digest: str = ""
    sizing_cell_ids: object = None
    placement_node_x: object = None
    placement_node_y: object = None
    placement_snapshot_digest: str = ""

    @property
    def is_noop(self):
        return not self.actions and not self.sizing_actions

    def to_summary(self):
        return {
            "request_version": int(self.version),
            "mode": str(self.mode),
            "action_count": len(self.actions),
            "action_digest": str(self.action_digest),
            "sizing_action_count": len(self.sizing_actions),
            "sizing_action_digest": str(self.sizing_action_digest),
            "iteration": self.iteration,
            "reason": str(self.reason),
            "mutation_kind": str(self.mutation_kind),
            "commit_enabled": bool(self.commit_enabled),
            "committed_def_path": str(self.committed_def_path),
            "refresh_mode": str(self.refresh_mode),
            "rebuild_mode": str(self.rebuild_mode),
            "is_noop": bool(self.is_noop),
            "projection_metadata": dict(self.projection_metadata),
            "runtime_refresh_summary": dict(self.runtime_refresh_summary),
            "effective_config_reference": str(self.effective_config_reference),
            "placement_coordinate_count": (
                0
                if self.placement_node_x is None
                else int(torch.as_tensor(self.placement_node_x).numel())
            ),
            "placement_snapshot_digest": str(self.placement_snapshot_digest),
        }


class BufferingOptimizationLane:
    """Canonical lifecycle holder for relaxed buffering state."""

    shared_typed_candidate_gradients = True

    def __init__(
        self,
        config,
        *,
        buffer_optimization_state=None,
        buffer_relaxed_timing_payload=None,
        placedb=None,
    ):
        self.config = config
        self.buffer_optimization_state = buffer_optimization_state
        self.buffer_relaxed_timing_payload = buffer_relaxed_timing_payload
        self.placedb = placedb
        self.data_collections = None
        self.model = None
        self.params = None
        self.last_projection_result = None
        self.last_runtime_refresh_summary = None
        self.last_commit_request = None
        self.segment_capacity_bundle = None
        self.segment_capacity_controller = None
        self._candidate_gc_restore_required = False
        self.timing_loss = None
        self._summary = {
            "status": "initialized",
            "mode": getattr(config, "mode", None),
            "objective_boundary": "PlaceObj.timing_obj",
            "timing_integration_mode": "dynamic_net_provider",
            "state_source": "prebuilt_or_attached",
        }
        self._update_state_kind_summary()

    def _payload_metadata(self):
        payload = self.buffer_relaxed_timing_payload
        if payload is None and self.data_collections is not None:
            payload = getattr(self.data_collections, "buffer_relaxed_timing_payload", None)
        if isinstance(payload, dict):
            return dict(payload.get("metadata", {}) or {})
        return {}

    def _state_kind(self):
        metadata_kind = self._payload_metadata().get("state_kind")
        if metadata_kind:
            return str(metadata_kind)
        state = self.buffer_optimization_state
        if state is None and self.data_collections is not None:
            state = getattr(self.data_collections, "buffer_optimization_state", None)
        if state is not None and hasattr(state, "z_param"):
            return "segment_count"
        if state is not None:
            return "candidate"
        return None

    def _update_state_kind_summary(self):
        state_kind = self._state_kind()
        if state_kind is not None:
            self._summary["state_kind"] = state_kind
        return state_kind

    def _has_relaxed_timing_state(self):
        state = self.buffer_optimization_state
        payload = self.buffer_relaxed_timing_payload
        if self.data_collections is not None:
            state = state or getattr(self.data_collections, "buffer_optimization_state", None)
            payload = payload or getattr(
                self.data_collections,
                "buffer_relaxed_timing_payload",
                None,
            )
        return state is not None and payload is not None

    def _sync_params(self, params):
        if params is None:
            return
        setattr(params, "relaxed_buffer_timing_integration_mode", "dynamic_net_provider")
        setattr(params, "enable_relaxed_buffer_timing", self._has_relaxed_timing_state())

    def prepare(self, model=None, params=None):
        runtime_profile_enabled = bool(
            getattr(params, "buffering_runtime_profile", False)
        )
        prepare_started_at = time.perf_counter()
        runtime_profile = {} if runtime_profile_enabled else None

        def record_runtime_stage(name, started_at):
            if runtime_profile is not None:
                runtime_profile[name] = (
                    time.perf_counter() - started_at
                ) * 1000.0

        self.model = model
        self.params = params
        if (
            model is not None
            and self.buffer_optimization_state is None
            and self.buffer_relaxed_timing_payload is None
        ):
            from .buffering_state_builder import build_buffering_state_for_model

            runtime_stage_started_at = time.perf_counter()
            suspend_candidate_gc = (
                str(getattr(self.config, "mode", "")) == "candidate"
                and str(getattr(self.config, "candidate_strategy", ""))
                == "discrete_net_gradient"
                and gc.isenabled()
            )
            if suspend_candidate_gc:
                gc.disable()
            try:
                build_result = build_buffering_state_for_model(
                    model,
                    params,
                    self.config,
                )
            except Exception:
                if suspend_candidate_gc:
                    gc.enable()
                raise
            self._candidate_gc_restore_required = bool(
                suspend_candidate_gc and build_result.status == "ok"
            )
            if suspend_candidate_gc and not self._candidate_gc_restore_required:
                gc.enable()
            if self._candidate_gc_restore_required:
                build_result.metadata["python_gc_suspended_until_projection"] = True
                payload = build_result.buffer_relaxed_timing_payload
                if isinstance(payload, dict):
                    payload.setdefault("metadata", {})[
                        "python_gc_suspended_until_projection"
                    ] = True
            record_runtime_stage("state_build_ms", runtime_stage_started_at)
            self._summary["state_build_status"] = build_result.status
            if build_result.reason is not None:
                self._summary["state_build_reason"] = build_result.reason
            if build_result.metadata:
                self._summary["state_build_metadata"] = dict(build_result.metadata)
            if build_result.status == "ok":
                self.buffer_optimization_state = build_result.buffer_optimization_state
                self.buffer_relaxed_timing_payload = (
                    build_result.buffer_relaxed_timing_payload
                )
                self._summary["state_source"] = build_result.metadata.get(
                    "state_source",
                    "buffering_state_builder",
                )
                self._summary["candidate_count"] = build_result.candidate_count
                self._summary["net_count"] = build_result.net_count
                self._summary.update(dict(build_result.metadata or {}))
        else:
            if runtime_profile is not None:
                runtime_profile["state_build_ms"] = 0.0
        if (
            bool(getattr(self.config, "segment_capacity_enabled", False))
            and self.segment_capacity_controller is None
        ):
            runtime_stage_started_at = time.perf_counter()
            self._prepare_segment_capacity_controller(model, params)
            record_runtime_stage("segment_capacity_build_ms", runtime_stage_started_at)
        elif runtime_profile is not None:
            runtime_profile["segment_capacity_build_ms"] = 0.0
        runtime_stage_started_at = time.perf_counter()
        self._sync_params(params)
        record_runtime_stage("parameter_sync_ms", runtime_stage_started_at)
        self._summary["prepared"] = True
        self._summary["relaxed_timing_enabled"] = self._has_relaxed_timing_state()
        self._update_state_kind_summary()
        if runtime_profile is not None:
            runtime_profile["total_ms"] = (
                time.perf_counter() - prepare_started_at
            ) * 1000.0
            runtime_profile["accounted_ms"] = sum(
                float(runtime_profile.get(name, 0.0) or 0.0)
                for name in (
                    "state_build_ms",
                    "segment_capacity_build_ms",
                    "parameter_sync_ms",
                )
            )
            runtime_profile["unattributed_ms"] = max(
                0.0,
                runtime_profile["total_ms"] - runtime_profile["accounted_ms"],
            )
            state_build_profile = self._summary.get("state_build_runtime_profile")
            if isinstance(state_build_profile, dict):
                runtime_profile["state_build_profile"] = dict(state_build_profile)
            self._summary["lane_prepare_runtime_profile"] = runtime_profile
        return self

    def _prepare_segment_capacity_controller(self, model, params):
        if self._state_kind() != "segment_count":
            raise ValueError("segment capacity control requires SegmentCountState")
        if self.placedb is None:
            raise ValueError("segment capacity control requires PlaceDB geometry")
        payload = self.buffer_relaxed_timing_payload
        if not isinstance(payload, dict):
            raise ValueError("segment capacity control requires relaxed timing payload")
        legal_table = payload.get("buffer_legal_table")
        if not isinstance(legal_table, dict):
            raise ValueError("segment capacity control requires buffer legal table")
        from .segment_capacity import build_segment_capacity_bundle_for_model

        bundle = build_segment_capacity_bundle_for_model(
            self._current_state(),
            model=model,
            placedb=self.placedb,
            params=params,
            legal_table=legal_table,
            fixed_bsu_index=self._payload_metadata().get(
                "fixed_bsu_index",
                getattr(self.config, "fixed_bsu_index", None),
            ),
            selected_grid_factor=getattr(self.config, "segment_capacity_grid", "auto"),
        )
        self.segment_capacity_bundle = bundle
        self.segment_capacity_controller = bundle.selected
        self._summary.update(bundle.to_summary())
        self._summary["segment_capacity_geometry"] = dict(
            self.segment_capacity_controller.geometry_summary
        )

    def attach_state(self, data_collections):
        self.data_collections = data_collections
        attached = False
        if self.buffer_optimization_state is not None:
            if (
                self._state_kind() != "segment_count"
                and hasattr(data_collections, "set_buffer_optimization_state")
            ):
                data_collections.set_buffer_optimization_state(self.buffer_optimization_state)
            else:
                data_collections.buffer_optimization_state = self.buffer_optimization_state
            attached = True
        if self.buffer_relaxed_timing_payload is not None:
            data_collections.buffer_relaxed_timing_payload = self.buffer_relaxed_timing_payload
            if self._state_kind() == "segment_count":
                payload = self.buffer_relaxed_timing_payload
                install_state = getattr(
                    data_collections,
                    "install_buffer_segment_timing_state",
                    None,
                )
                if callable(install_state):
                    install_state(
                        self.buffer_optimization_state,
                        payload,
                        reason="buffering_lane_attach_segment_state",
                    )
                else:
                    data_collections.buffer_segment_count_state = self.buffer_optimization_state
                    data_collections.buffer_segment_count_payload = payload
                    data_collections.buffer_segment_count_nets = payload.get("nets", ())
                    data_collections.buffer_segment_count_per_size_input_cap = payload.get(
                        "per_size_input_cap"
                    )
                    data_collections.buffer_segment_count_per_size_delay = payload.get(
                        "per_size_delay"
                    )
                    data_collections.buffer_segment_count_per_size_output_slew = payload.get(
                        "per_size_output_slew"
                    )
            attached = True
        self._summary["state_attached"] = bool(attached)
        self._summary["relaxed_timing_enabled"] = self._has_relaxed_timing_state()
        self._update_state_kind_summary()
        self._sync_params(getattr(self, "params", None))
        return self

    def refresh_candidate_geometry(
        self,
        *,
        model=None,
        params=None,
        reason="live_geometry_refresh",
        active_only=False,
    ):
        """Rebuild candidate timing data while retaining virtual actions.

        Candidate activation is remapped by the stable edge/position key.  The
        rebuilt payload is the only source used by the dynamic provider; old
        packed coordinates and RC tensors are discarded.
        """

        state = self._current_state()
        if state is None or not hasattr(state, "activation_param"):
            return {"status": "skipped", "reason": "non_candidate_state"}
        model = model if model is not None else self.model
        params = params if params is not None else self.params
        if model is None or params is None:
            return {"status": "skipped", "reason": "missing_model_or_params"}

        from .candidate_geometry import (
            refresh_candidate_geometry,
            remap_candidate_state,
        )
        from .buffering_state_builder import build_buffering_state_for_model

        live_topology = dict(
            getattr(self.data_collections, "buffering_timing_topology", {}) or {}
        )
        previous_metadata = dict(
            (getattr(self, "buffer_relaxed_timing_payload", {}) or {}).get(
                "metadata", {}
            )
            or {}
        )
        geometry_generation = int(
            previous_metadata.get("candidate_geometry_generation", 0) or 0
        ) + 1
        build_options = {}
        old_records = state.candidate_records
        if active_only:
            activation = state.activation_param.detach()
            active = activation > 0 if activation.ndim == 1 else activation.sum(dim=1) > 0
            rows = torch.nonzero(active).flatten().cpu().tolist()
            if not rows:
                return {"status": "ok", "reason": "no_active_candidates", "candidate_count": 0}
            net_ids = {int(state.candidate_records[index]["net_id"]) for index in rows}
            # Keep inactive split points on retained nets: removing them changes
            # the distributed RC slew calculation even when their BU is zero.
            old_records = tuple(row for row in state.candidate_records if int(row["net_id"]) in net_ids)
            build_options["net_ids"] = sorted(net_ids)
        # ``buffering_nets`` is a generation-local RC tree cache.  Once a
        # live topology has been published, reusing it would repack the new
        # candidate records against the previous vertex domain.  Force the
        # state builder through its live-topology adapter for this refresh.
        if live_topology:
            for name in ("buffering_nets", "buffer_nets"):
                if hasattr(self.data_collections, name):
                    setattr(self.data_collections, name, ())
        old_refresh = refresh_candidate_geometry(old_records, live_topology)
        if active_only:
            build_options["candidate_records"] = old_refresh.candidate_records
        build_result = build_buffering_state_for_model(model, params, self.config, **build_options)
        if build_result.status != "ok":
            raise RuntimeError(
                "candidate geometry refresh failed to rebuild state: "
                f"{build_result.reason or build_result.metadata}"
            )
        new_state = build_result.buffer_optimization_state
        new_refresh = refresh_candidate_geometry(
            new_state,
            live_topology,
            candidate_generation=live_topology.get("topology_generation", 0),
            geometry_generation=geometry_generation,
        )
        new_state.candidate_records = new_refresh.candidate_records
        remap = remap_candidate_state(
            state,
            new_state,
            invalidated_keys=new_refresh.invalidated_candidate_keys,
        )
        if remap["new_active_count"] != remap["old_active_count"]:
            raise RuntimeError(
                "candidate geometry refresh lost accepted virtual buffers: "
                f"{remap}"
            )
        self.buffer_optimization_state = new_state
        self.buffer_relaxed_timing_payload = build_result.buffer_relaxed_timing_payload
        metadata = dict(self.buffer_relaxed_timing_payload.get("metadata", {}) or {})
        metadata.update(
            {
                "candidate_geometry_refresh_reason": str(reason),
                "candidate_topology_generation": int(
                    new_refresh.candidate_generation
                ),
                "candidate_geometry_generation": int(
                    new_refresh.geometry_generation
                ),
                "candidate_geometry_digest": new_refresh.geometry_digest,
                "candidate_topology_digest": new_refresh.topology_digest,
                "candidate_geometry_invalidated_count": len(
                    new_refresh.invalidated_candidate_keys
                ),
                "candidate_geometry_refresh_status": "ok",
                "candidate_geometry_old_digest": old_refresh.geometry_digest,
                "candidate_activation_remap": dict(remap),
                "candidate_state_remap_failed": list(
                    remap.get("candidate_state_remap_failed", ()) or ()
                ),
            }
        )
        self.buffer_relaxed_timing_payload["metadata"] = metadata
        self._summary.update(build_result.metadata)
        self._summary["candidate_count"] = build_result.candidate_count
        self._summary["net_count"] = build_result.net_count
        self.attach_state(self.data_collections)
        self.register_model_parameters(model)
        model._relaxed_buffer_dynamic_net_provider = None
        model._relaxed_buffer_dynamic_net_provider_key = None
        model._timing_geometry_cache = None
        self._summary["candidate_geometry_refresh"] = dict(metadata)
        return {
            "status": "ok",
            "reason": str(reason),
            "candidate_generation": int(new_refresh.candidate_generation),
            "geometry_generation": int(new_refresh.geometry_generation),
            "candidate_count": len(new_refresh.candidate_records),
            "invalidated_candidate_count": len(new_refresh.invalidated_candidate_keys),
            "geometry_digest": new_refresh.geometry_digest,
            "topology_digest": new_refresh.topology_digest,
            "activation_remap": dict(remap),
        }

    def register_model_parameters(self, model):
        if model is not None and hasattr(model, "register_buffer_optimization_state"):
            model.register_buffer_optimization_state(self.buffer_optimization_state)
        return self

    def optimizer_parameters(self):
        state = self.buffer_optimization_state
        if state is None and self.data_collections is not None:
            state = getattr(self.data_collections, "buffer_optimization_state", None)
        if state is None:
            return []
        params = []
        if self._state_kind() == "segment_count":
            names = ("z_param",)
            if getattr(state, "fixed_bsu_index", None) is None:
                names += ("bsu_index_param",)
        else:
            names = ("bu_logits", "bsu_index_param")
        for name in names:
            value = getattr(state, name, None)
            if value is not None and id(value) not in {id(item) for item in params}:
                params.append(value)
        return params

    def objective(self, model, pos=None):
        if pos is None:
            data_collections = getattr(model, "data_collections", None)
            positions = getattr(data_collections, "pos", None)
            if positions is None:
                raise ValueError("buffering objective requires model.data_collections.pos")
            pos = positions[0]
        timing_result = model.timing_obj(pos)
        try:
            wns, tns, worst_slack, total_slack = timing_result
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "buffering objective expects timing_obj to return "
                "(wns, tns, worst_slack, total_slack)"
            ) from exc
        model._diff_opt_last_timing_metrics = {
            "wns": float(wns.detach()) if hasattr(wns, "detach") else float(wns),
            "tns": float(tns.detach()) if hasattr(tns, "detach") else float(tns),
        }
        timing_loss = (
            -tns if self.timing_loss is None
            else self.timing_loss(wns, tns, worst_slack, total_slack)
        )
        capacity_metrics = {}
        if self.segment_capacity_controller is not None:
            loss, capacity_metrics = self.segment_capacity_controller.compose_objective(
                timing_loss
            )
        else:
            loss = timing_loss
        metrics = {
            "objective_boundary": "PlaceObj.timing_obj",
            "loss": loss.detach() if hasattr(loss, "detach") else loss,
            "timing_loss": (
                timing_loss.detach() if hasattr(timing_loss, "detach") else timing_loss
            ),
            "wns": wns,
            "tns": tns,
            "worst_slack": worst_slack,
            "total_slack": total_slack,
        }
        metrics.update(capacity_metrics)
        return loss, metrics

    def before_step(self, iteration, loss, metrics):
        metrics.update(self._gradient_metrics())
        return None

    def after_step(self, iteration, loss, metrics):
        state = self._current_state()
        if state is None:
            return None
        projected = False
        if hasattr(state, "project_"):
            state.project_()
            projected = True
        elif hasattr(state, "project_bsu_index_"):
            state.project_bsu_index_()
            projected = True
        if projected:
            metrics.update(self._parameter_range_metrics(state))
            metrics["projected_after_step"] = True
        if self.segment_capacity_controller is not None:
            metrics.update(
                self.segment_capacity_controller.update_dual_after_step(
                    iteration,
                    objective_metrics=metrics,
                )
            )
        return None

    def _current_state(self):
        state = self.buffer_optimization_state
        if state is None and self.data_collections is not None:
            state = getattr(self.data_collections, "buffer_optimization_state", None)
        return state

    def _tensor_grad_norm(self, tensor):
        grad = getattr(tensor, "grad", None)
        if grad is None:
            return None
        finite = grad.detach()
        if finite.numel() == 0:
            return 0.0
        return float(torch.linalg.vector_norm(finite.float()).detach().cpu().item())

    def _gradient_metrics(self):
        state = self._current_state()
        if state is None:
            return {}
        metrics = {}
        if hasattr(state, "bu_logits"):
            metrics["grad_bu_norm"] = self._tensor_grad_norm(state.bu_logits)
        if hasattr(state, "z_param"):
            metrics["grad_z_norm"] = self._tensor_grad_norm(state.z_param)
        if hasattr(state, "bsu_index_param"):
            metrics["grad_bsu_norm"] = self._tensor_grad_norm(state.bsu_index_param)
        return metrics

    def _parameter_range_metrics(self, state):
        metrics = {}
        for prefix, tensor in (
            ("z", getattr(state, "z_param", None)),
            ("bsu", getattr(state, "bsu_index_param", None)),
        ):
            if tensor is None:
                continue
            value = tensor.detach()
            if value.numel() == 0:
                metrics[f"{prefix}_param_min"] = None
                metrics[f"{prefix}_param_max"] = None
                continue
            metrics[f"{prefix}_param_min"] = float(value.min().cpu().item())
            metrics[f"{prefix}_param_max"] = float(value.max().cpu().item())
        return metrics

    def _integer_projection_stats(self, tensor, low, high):
        value = tensor.detach()
        rounded = torch.round(value).clamp(float(low), float(high))
        changed = rounded != value
        stats = {
            "count": int(value.numel()),
            "changed_count": int(changed.sum().detach().cpu().item()),
        }
        if value.numel() == 0:
            stats.update(
                {
                    "nonzero_count_before": 0,
                    "nonzero_count_after": 0,
                    "sum_before": 0.0,
                    "sum_after": 0.0,
                    "min_before": None,
                    "min_after": None,
                    "max_before": None,
                    "max_after": None,
                    "ge_0p5_count_before": 0,
                    "ge_0p5_count_after": 0,
                }
            )
            return rounded, stats
        stats.update(
            {
                "nonzero_count_before": int((value != 0).sum().detach().cpu().item()),
                "nonzero_count_after": int((rounded != 0).sum().detach().cpu().item()),
                "sum_before": float(value.sum().detach().cpu().item()),
                "sum_after": float(rounded.sum().detach().cpu().item()),
                "min_before": float(value.min().detach().cpu().item()),
                "min_after": float(rounded.min().detach().cpu().item()),
                "max_before": float(value.max().detach().cpu().item()),
                "max_after": float(rounded.max().detach().cpu().item()),
                "ge_0p5_count_before": int((value >= 0.5).sum().detach().cpu().item()),
                "ge_0p5_count_after": int((rounded >= 0.5).sum().detach().cpu().item()),
            }
        )
        return rounded, stats

    def _reset_optimizer_state_for_params(self, optimizer, params):
        if optimizer is None:
            return {
                "status": "skipped",
                "reason": "missing_optimizer",
                "reset_param_count": 0,
                "reset_tensor_state_count": 0,
            }
        reset_tensor_state_count = 0
        reset_param_count = 0
        for param in tuple(params or ()):
            state = getattr(optimizer, "state", {}).get(param)
            if not isinstance(state, dict):
                continue
            param_reset = False
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
                    reset_tensor_state_count += 1
                    param_reset = True
            if param_reset:
                reset_param_count += 1
        return {
            "status": "ok",
            "reset_param_count": int(reset_param_count),
            "reset_tensor_state_count": int(reset_tensor_state_count),
        }

    def project_segment_count_optimizer_state_to_integer(
        self,
        *,
        project_bsu=False,
        reset_optimizer_state=True,
        optimizer=None,
        iteration=None,
    ):
        state = self._current_state()
        if state is None or not hasattr(state, "z_param"):
            return {
                "status": "skipped",
                "reason": "not_segment_count_state",
                "type": "periodic_integer_projection",
                "iteration": None if iteration is None else int(iteration),
            }
        projected_params = [state.z_param]
        with torch.no_grad():
            z_projected, z_stats = self._integer_projection_stats(
                state.z_param,
                0,
                int(state.max_repeater_count),
            )
            state.z_param.copy_(z_projected)
            result = {
                "status": "ok",
                "type": "periodic_integer_projection",
                "iteration": None if iteration is None else int(iteration),
                "project_z": True,
                "project_bsu": bool(project_bsu),
                "reset_optimizer_state": bool(reset_optimizer_state),
                "z_changed_count": int(z_stats["changed_count"]),
                "z_nonzero_count_before": int(z_stats["nonzero_count_before"]),
                "z_nonzero_count_after": int(z_stats["nonzero_count_after"]),
                "z_sum_before": float(z_stats["sum_before"]),
                "z_sum_after": float(z_stats["sum_after"]),
                "z_max_before": z_stats["max_before"],
                "z_max_after": z_stats["max_after"],
                "z_ge_0p5_count_before": int(z_stats["ge_0p5_count_before"]),
                "z_ge_0p5_count_after": int(z_stats["ge_0p5_count_after"]),
            }
            if bool(project_bsu):
                bsu_projected, bsu_stats = self._integer_projection_stats(
                    state.bsu_index_param,
                    0,
                    int(state.legal_buffer_count) - 1,
                )
                state.bsu_index_param.copy_(bsu_projected)
                projected_params.append(state.bsu_index_param)
                result.update(
                    {
                        "bsu_changed_count": int(bsu_stats["changed_count"]),
                        "bsu_min_before": bsu_stats["min_before"],
                        "bsu_max_before": bsu_stats["max_before"],
                        "bsu_min_after": bsu_stats["min_after"],
                        "bsu_max_after": bsu_stats["max_after"],
                    }
                )
        if bool(reset_optimizer_state):
            result["optimizer_state_reset"] = self._reset_optimizer_state_for_params(
                optimizer,
                projected_params,
            )
        else:
            result["optimizer_state_reset"] = {
                "status": "skipped",
                "reason": "disabled",
                "reset_param_count": 0,
                "reset_tensor_state_count": 0,
            }
        trace = self._summary.setdefault("periodic_integer_projection_trace", [])
        trace.append(dict(result))
        self._summary["periodic_integer_projection_count"] = len(trace)
        self._summary["periodic_integer_projection_enabled"] = True
        self._summary["periodic_integer_projection_project_bsu"] = bool(project_bsu)
        self._summary["periodic_integer_projection_reset_optimizer_state"] = bool(
            reset_optimizer_state
        )
        return result

    def should_project(self, iteration):
        return True

    def project(self, **kwargs):
        from .projection import project_buffering_lane_state

        if "top_k" not in kwargs and getattr(self.config, "max_selected_actions", None) is not None:
            kwargs["top_k"] = int(self.config.max_selected_actions)
        projection_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"top_k", "threshold", "live_projection_context"}
        }
        self.last_projection_result = project_buffering_lane_state(
            self,
            **projection_kwargs,
        )
        if kwargs:
            metadata = dict(getattr(self.last_projection_result, "metadata", {}) or {})
            metadata.update(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key in {"iteration", "final"}
                }
            )
            object.__setattr__(self.last_projection_result, "metadata", metadata)
        self._summary["last_projected_buffer_count"] = (
            self.last_projection_result.projected_buffer_count
        )
        if self.segment_capacity_controller is not None and bool(kwargs.get("final", False)):
            fidelity = self.segment_capacity_controller.close_projection_fidelity(
                self.last_projection_result.selected_actions
            )
            artifact_paths = self.segment_capacity_controller.write_artifacts(
                getattr(self.config, "output_dir", "results")
            )
            self._summary["segment_capacity_projection_fidelity"] = fidelity
            self._summary["segment_capacity_artifacts"] = artifact_paths
        projection_metadata = dict(
            getattr(self.last_projection_result, "metadata", {}) or {}
        )
        if projection_metadata:
            self._summary["projection"] = projection_metadata
            if "segment_count" in projection_metadata:
                self._summary["segment_count"] = projection_metadata["segment_count"]
            if "projected_candidate_count" in projection_metadata:
                self._summary["projected_candidate_count"] = projection_metadata[
                    "projected_candidate_count"
                ]
            if "min_z_to_insert" in projection_metadata:
                self._summary["segment_projection_min_z_to_insert"] = (
                    projection_metadata["min_z_to_insert"]
                )
            if "require_setup_criticality" in projection_metadata:
                self._summary["segment_projection_require_setup_criticality"] = (
                    projection_metadata["require_setup_criticality"]
                )
            if "criticality_filtered_candidate_count" in projection_metadata:
                self._summary["criticality_filtered_candidate_count"] = (
                    projection_metadata["criticality_filtered_candidate_count"]
                )
            if "z_value_distribution" in projection_metadata:
                self._summary["z_value_distribution"] = projection_metadata[
                    "z_value_distribution"
                ]
            if "top_candidate_diagnostics" in projection_metadata:
                self._summary["projection_top_candidate_diagnostics"] = (
                    projection_metadata["top_candidate_diagnostics"]
                )
            if "selected_candidate_diagnostics" in projection_metadata:
                self._summary["projection_selected_candidate_diagnostics"] = (
                    projection_metadata["selected_candidate_diagnostics"]
                )
        return self.last_projection_result

    def refresh_runtime_state(self, projection_result=None):
        from .runtime_refresh import refresh_buffering_runtime_state

        if projection_result is None:
            projection_result = self.last_projection_result
        self.last_runtime_refresh_summary = refresh_buffering_runtime_state(
            self.data_collections,
            projection_result,
        )
        self._summary["runtime_refresh"] = dict(self.last_runtime_refresh_summary)
        return self.last_runtime_refresh_summary

    def build_commit_request(
        self,
        projection_result=None,
        *,
        iteration=None,
        reason="buffering_projection",
        runtime_refresh=None,
        commit_enabled=None,
        committed_def_path=None,
        mutation_kind="topology-changing",
        refresh_mode="topo",
        rebuild_mode="topo",
        effective_config_reference="",
        sizing_actions=None,
        sizing_cell_ids=None,
        placement_node_x=None,
        placement_node_y=None,
    ):
        if projection_result is None:
            projection_result = self.last_projection_result
        raw_actions = tuple(
            getattr(projection_result, "selected_actions", ()) or ()
        )
        actions = tuple(_json_safe_copy(list(raw_actions), "selected_actions"))
        metadata = _json_safe_copy(
            dict(getattr(projection_result, "metadata", {}) or {}),
            "projection metadata",
        )
        refresh_summary = _json_safe_copy(
            dict(runtime_refresh or {}),
            "runtime refresh summary",
        )
        resolved_sizing_actions = tuple(
            _json_safe_copy(list(sizing_actions or ()), "sizing actions")
        )
        resolved_sizing_cell_ids = (
            None
            if sizing_cell_ids is None
            else torch.as_tensor(sizing_cell_ids).detach().cpu().long().contiguous().clone()
        )
        resolved_placement_x = (
            None
            if placement_node_x is None
            else torch.as_tensor(placement_node_x).detach().cpu().contiguous().clone()
        )
        resolved_placement_y = (
            None
            if placement_node_y is None
            else torch.as_tensor(placement_node_y).detach().cpu().contiguous().clone()
        )
        if (resolved_placement_x is None) != (resolved_placement_y is None):
            raise ValueError("final placement snapshot requires both x and y")
        if (
            resolved_placement_x is not None
            and resolved_placement_x.numel() != resolved_placement_y.numel()
        ):
            raise ValueError("final placement x/y snapshot lengths differ")
        resolved_commit_enabled = (
            bool(getattr(self.config, "commit_enabled", False))
            if commit_enabled is None
            else bool(commit_enabled)
        )
        resolved_def_path = (
            str(getattr(self.config, "committed_def_path", "") or "")
            if committed_def_path is None
            else str(committed_def_path or "")
        )
        request = BufferCommitRequest(
            version=1,
            mode=str(getattr(projection_result, "mode", getattr(self.config, "mode", "unknown"))),
            actions=actions,
            action_digest=_action_digest(actions),
            iteration=None if iteration is None else int(iteration),
            reason=str(reason),
            mutation_kind=str(mutation_kind),
            commit_enabled=resolved_commit_enabled,
            committed_def_path=resolved_def_path,
            refresh_mode=str(refresh_mode),
            rebuild_mode=str(rebuild_mode),
            projection_metadata=metadata,
            runtime_refresh_summary=refresh_summary,
            effective_config_reference=str(effective_config_reference or ""),
            sizing_actions=resolved_sizing_actions,
            sizing_action_digest=_action_digest(resolved_sizing_actions),
            sizing_cell_ids=resolved_sizing_cell_ids,
            placement_node_x=resolved_placement_x,
            placement_node_y=resolved_placement_y,
            placement_snapshot_digest=(
                ""
                if resolved_placement_x is None
                else _tensor_digest(
                    (
                        ("node_x", resolved_placement_x),
                        ("node_y", resolved_placement_y),
                    )
                )
            ),
        )
        self.last_commit_request = request
        self._summary["commit_request"] = request.to_summary()
        return request

    def summarize(self):
        payload_metadata = self._payload_metadata()
        if payload_metadata:
            for key in (
                "state_kind",
                "segment_count",
                "affected_net_count",
                "full_design_scope",
                "max_repeater_count",
                "segment_count_timing_backend_requested",
                "segment_count_timing_backend_used",
                "segment_count_timing_backend_fallback_reason",
                "segment_count_timing_backend_fallback_is_explicit",
                "segment_count_timing_backend",
                "segment_transfer_backend_requested",
                "segment_transfer_backend",
                "segment_transfer_backend_used",
                "fixed_bsu_index",
                "fixed_buffer_master",
                "fixed_buffer_selection_source",
                "buffer_size_optimization",
            ):
                if key in payload_metadata and key not in self._summary:
                    self._summary[key] = payload_metadata[key]
        if self.segment_capacity_bundle is not None:
            self._summary.update(self.segment_capacity_bundle.to_summary())
        if self.segment_capacity_controller is not None:
            self._summary["segment_capacity_runtime"] = {
                "timing_scale": self.segment_capacity_controller.timing_scale,
                "trace_count": len(self.segment_capacity_controller.trace),
                "final_relaxed_feasibility": (
                    None
                    if not self.segment_capacity_controller.trace
                    else self.segment_capacity_controller.trace[-1].get(
                        "segment_capacity_max_violation"
                    )
                ),
            }
        return dict(self._summary)
