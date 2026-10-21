"""Candidate-domain reuse and geometry refresh for buffering lanes."""

import time

import torch


class CandidateStateRefreshMixin:
    """Maintain candidate geometry without rebuilding unchanged topology."""

    def _candidate_topology_signature(self):
        return self._candidate_topology_digest()

    def _candidate_structure_signature(self):
        return self._candidate_topology_digest(include_geometry=False)

    def _candidate_topology_digest(self, *, include_geometry=True):
        topology = getattr(
            self.data_collections,
            "buffering_timing_topology",
            None,
        )
        if not topology:
            return None
        from .candidate_records_cache import candidate_topology_signature

        topology_op = getattr(
            getattr(self.model, "op_collections", None),
            "steiner_topo_op",
            None,
        )
        topology_epoch = None
        if topology_op is not None:
            frozen_nets = getattr(topology_op, "frozen_nets", None)
            topology_epoch = {
                "topology_generation": int(
                    getattr(topology_op, "topology_generation", 0) or 0
                ),
                "frozen_topology_generation": getattr(
                    topology_op,
                    "frozen_topology_generation",
                    None,
                ),
                "frozen_net_ids": tuple(
                    int(net_id)
                    for net_id in getattr(frozen_nets, "net_ids", ()) or ()
                ),
            }
        return candidate_topology_signature(
            topology,
            topology_epoch=topology_epoch,
            include_geometry=include_geometry,
        )

    def _record_candidate_geometry_profile(
        self,
        *,
        reason,
        active_only,
        candidate_records_cache_hit,
        runtime_profile,
    ):
        if runtime_profile is None:
            return
        history = self._summary.setdefault(
            "candidate_geometry_runtime_profile_history",
            [],
        )
        history.append(
            {
                "reason": str(reason),
                "active_only": bool(active_only),
                "candidate_records_cache_hit": bool(
                    candidate_records_cache_hit
                ),
                "runtime_profile": dict(runtime_profile),
            }
        )

    def _refresh_candidate_records_native(
        self,
        *,
        dirty_net_ids=None,
        all_candidates=False,
        update_full_domain=True,
    ):
        """Refresh candidate record coordinates without repacking topology."""

        from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
            apply_candidate_geometry_rows,
            refresh_candidate_geometry_rows,
        )

        payload = self.buffer_relaxed_timing_payload
        topology = dict(
            getattr(self.data_collections, "buffering_timing_topology", {}) or {}
        )
        node_x_dbu = topology.get("node_x_dbu")
        node_y_dbu = topology.get("node_y_dbu")
        positions = getattr(self.data_collections, "pos", None)
        geometry_for_pos = getattr(self.model, "_timing_geometry_for_pos", None)
        if positions is not None and callable(geometry_for_pos):
            _, live_x, live_y = geometry_for_pos(positions[0])
            scale = float(getattr(self.params, "scale_factor", 1.0) or 1.0)
            shift = getattr(self.params, "shift_factor", (0.0, 0.0)) or (0.0, 0.0)
            node_x_dbu = (
                live_x.detach().double() / scale + float(shift[0])
            ).round()
            node_y_dbu = (
                live_y.detach().double() / scale + float(shift[1])
            ).round()
        if node_x_dbu is None or node_y_dbu is None:
            raise ValueError(
                "incremental candidate geometry requires live dbu coordinates"
            )
        result = refresh_candidate_geometry_rows(
            payload=payload,
            dirty_net_ids=dirty_net_ids,
            node_x_dbu=node_x_dbu,
            node_y_dbu=node_y_dbu,
            all_candidates=all_candidates,
        )
        dbu = (
            getattr(self.data_collections, "buffering_timing_topology_dbu", None)
            or topology.get("dbu")
            or getattr(self.placedb, "dbu", None)
        )
        if dbu is None:
            raise ValueError(
                "incremental candidate geometry requires the topology dbu"
            )
        records = apply_candidate_geometry_rows(
            self._current_state().candidate_records,
            result,
            dbu=float(dbu),
        )
        self._current_state().candidate_records = records
        if update_full_domain:
            self._full_candidate_records = records
        self._last_candidate_incremental_refresh = result
        return result

    def refresh_candidate_geometry(
        self,
        *,
        model=None,
        params=None,
        reason="live_geometry_refresh",
        active_only=False,
        dirty_net_ids=None,
        refresh_records_all=False,
        force_rebuild=False,
    ):
        """Refresh candidate timing data while retaining virtual actions.

        An unchanged candidate tree structure reuses the packed state and
        refreshes only requested record geometry through the native operator.
        A structural topology change falls back to a payload rebuild and
        remaps activation by the stable edge/position key.
        """

        state = self._current_state()
        if state is None or not hasattr(state, "activation_param"):
            return {"status": "skipped", "reason": "non_candidate_state"}
        model = model if model is not None else self.model
        params = params if params is not None else self.params
        if model is None or params is None:
            return {"status": "skipped", "reason": "missing_model_or_params"}

        runtime_profile_enabled = bool(
            getattr(params, "buffering_runtime_profile", False)
        )
        runtime_profile = {} if runtime_profile_enabled else None
        refresh_started_at = time.perf_counter()

        def record_runtime_stage(name, started_at):
            if runtime_profile is not None:
                runtime_profile[name] = (
                    time.perf_counter() - started_at
                ) * 1000.0

        from .buffering_state_builder import build_buffering_state_for_model
        from .candidate_geometry import (
            CandidateRefreshResult,
            candidate_key,
            refresh_candidate_geometry,
            remap_candidate_state,
        )

        runtime_stage_started_at = time.perf_counter()
        live_structure_signature = self._candidate_structure_signature()
        live_topology_signature = self._candidate_topology_signature()
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
        native_geometry_result = None
        missing_native_columns = False
        structural_domain_reused = bool(
            not force_rebuild
            and self._full_candidate_records is not None
            and (
                active_only
                or self._candidate_state_contains_full_domain
            )
            and live_structure_signature is not None
            and live_structure_signature
            == self._full_candidate_structure_signature
        )
        if structural_domain_reused:
            should_refresh_records = bool(
                refresh_records_all
                or (dirty_net_ids is not None and len(dirty_net_ids) > 0)
            )
            if should_refresh_records:
                runtime_stage_started_at = time.perf_counter()
                from dreamplace.ops.net_subgraph_timing.candidate_maintenance import (
                    CandidateGeometryRefreshError,
                )

                try:
                    native_geometry_result = self._refresh_candidate_records_native(
                        dirty_net_ids=()
                        if refresh_records_all
                        else dirty_net_ids,
                        all_candidates=bool(refresh_records_all),
                        update_full_domain=bool(
                            not active_only
                            or self._candidate_state_contains_full_domain
                        ),
                    )
                except (
                    CandidateGeometryRefreshError,
                    ImportError,
                    KeyError,
                    ValueError,
                ) as exc:
                    missing_native_columns = True
                    self._summary["candidate_incremental_geometry_fallback"] = str(exc)
                record_runtime_stage(
                    "native_candidate_geometry_ms",
                    runtime_stage_started_at,
                )
            if not missing_native_columns:
                state.timing_active_only = bool(active_only)
                self._candidate_state_contains_full_domain = True
                metadata = dict(
                    (self.buffer_relaxed_timing_payload or {}).get("metadata", {})
                    or {}
                )
                topology_digest = live_structure_signature or ""
                metadata.update(
                    {
                        "candidate_geometry_refresh_reason": str(reason),
                        "candidate_topology_generation": int(
                            live_topology.get("topology_generation", 0) or 0
                        ),
                        "candidate_geometry_generation": int(geometry_generation),
                        "candidate_geometry_digest": live_topology_signature,
                        "candidate_topology_digest": topology_digest,
                        "candidate_geometry_invalidated_count": 0,
                        "candidate_geometry_refresh_status": "ok",
                        "candidate_activation_remap": {
                            "status": "retained_in_place",
                            "old_active_count": int(
                                state.activation_param.detach().sum().cpu().item()
                            ),
                            "new_active_count": int(
                                state.activation_param.detach().sum().cpu().item()
                            ),
                        },
                        "candidate_domain_reused": True,
                        "candidate_domain_reused_count": len(
                            state.candidate_records
                        ),
                        "candidate_full_domain_count": len(
                            self._full_candidate_records
                        ),
                        "candidate_records_cache_hit": True,
                        "candidate_state_build_skipped": True,
                        "candidate_new_geometry_reused": True,
                        "candidate_incremental_geometry": (
                            None
                            if native_geometry_result is None
                            else {
                                "counts": dict(
                                    native_geometry_result.get("counts", {}) or {}
                                ),
                                "metadata": dict(
                                    native_geometry_result.get("metadata", {}) or {}
                                ),
                            }
                        ),
                        "candidate_state_remap_failed": [],
                    }
                )
                self.buffer_relaxed_timing_payload["metadata"] = metadata
                self._full_candidate_topology_signature = (
                    live_topology_signature
                )
                self._full_candidate_geometry_digest = live_topology_signature
                self._full_candidate_refresh_topology_digest = topology_digest
                self._full_candidate_candidate_generation = int(
                    live_topology.get("topology_generation", 0) or 0
                )
                self.attach_state(self.data_collections)
                self.register_model_parameters(model)
                model._relaxed_buffer_dynamic_net_provider = None
                model._relaxed_buffer_dynamic_net_provider_key = None
                model._timing_geometry_cache = None
                self._summary["candidate_geometry_refresh"] = dict(metadata)
                self._summary["candidate_incremental_refresh_count"] = int(
                    self._summary.get("candidate_incremental_refresh_count", 0)
                ) + 1
                runtime_stage_started_at = time.perf_counter()
                record_runtime_stage(
                    "publish_and_invalidate_ms",
                    runtime_stage_started_at,
                )
                if runtime_profile is not None:
                    runtime_profile["total_ms"] = (
                        time.perf_counter() - refresh_started_at
                    ) * 1000.0
                    runtime_profile["candidate_state_build_skipped"] = True
                    runtime_profile["native_candidate_geometry"] = dict(
                        (
                            native_geometry_result or {}
                        ).get("metadata", {})
                        or {}
                    )
                    metadata["candidate_geometry_runtime_profile"] = runtime_profile
                    self._summary["candidate_geometry_runtime_profile"] = (
                        runtime_profile
                    )
                    self._record_candidate_geometry_profile(
                        reason=reason,
                        active_only=active_only,
                        candidate_records_cache_hit=True,
                        runtime_profile=runtime_profile,
                    )
                return {
                    "status": "ok",
                    "reason": str(reason),
                    "candidate_generation": int(
                        live_topology.get("topology_generation", 0) or 0
                    ),
                    "geometry_generation": int(geometry_generation),
                    "candidate_count": len(self._full_candidate_records),
                    "invalidated_candidate_count": 0,
                    "geometry_digest": metadata["candidate_geometry_digest"],
                    "topology_digest": topology_digest,
                    "activation_remap": dict(
                        metadata["candidate_activation_remap"]
                    ),
                    "candidate_domain_reused": True,
                    "candidate_records_cache_hit": True,
                    "candidate_state_build_skipped": True,
                    "runtime_profile": runtime_profile,
                }
        build_options = {}
        old_records = (
            self._full_candidate_records
            if not active_only and self._full_candidate_records is not None
            else state.candidate_records
        )
        if active_only:
            activation = state.activation_param.detach()
            active = activation > 0 if activation.ndim == 1 else activation.sum(dim=1) > 0
            rows = torch.nonzero(active).flatten().cpu().tolist()
            if not rows:
                return {"status": "ok", "reason": "no_active_candidates", "candidate_count": 0}
            net_ids = {int(state.candidate_records[index]["net_id"]) for index in rows}
            # Keep inactive split points on retained nets: removing them changes
            # the distributed RC slew calculation even when their BU is zero.
            old_records = tuple(
                row
                for row in state.candidate_records
                if int(row["net_id"]) in net_ids
            )
            build_options["net_ids"] = sorted(net_ids)
        # ``buffering_nets`` is a generation-local RC tree cache.  Once a
        # live topology has been published, reusing it would repack the new
        # candidate records against the previous vertex domain.  Force the
        # state builder through its live-topology adapter for this refresh.
        if live_topology:
            for name in ("buffering_nets", "buffer_nets"):
                if hasattr(self.data_collections, name):
                    setattr(self.data_collections, name, ())
        record_runtime_stage("refresh_inputs_ms", runtime_stage_started_at)
        runtime_stage_started_at = time.perf_counter()
        candidate_records_cache_hit = bool(
            not active_only
            and old_records is not None
            and live_topology_signature is not None
            and live_topology_signature == self._full_candidate_topology_signature
            and self._full_candidate_geometry_digest
        )
        if candidate_records_cache_hit:
            old_refresh = CandidateRefreshResult(
                candidate_records=tuple(old_records),
                candidate_generation=int(
                    self._full_candidate_candidate_generation
                    if self._full_candidate_candidate_generation is not None
                    else live_topology.get("topology_generation", 0) or 0
                ),
                geometry_generation=int(geometry_generation),
                geometry_digest=self._full_candidate_geometry_digest,
                topology_digest=self._full_candidate_refresh_topology_digest
                or "",
            )
        else:
            old_refresh = refresh_candidate_geometry(
                old_records,
                live_topology,
                geometry_generation=geometry_generation,
            )
        record_runtime_stage("old_geometry_refresh_ms", runtime_stage_started_at)
        candidate_domain_reused = not old_refresh.invalidated_candidate_keys
        if candidate_domain_reused:
            # The candidate identity is net/edge/ratio, which remains stable
            # while the diff-opt topology is frozen. Reuse the complete domain
            # and refresh only its geometry; regenerating millions of rows can
            # change sparse edge ordering and silently drop accepted buffers.
            build_options["candidate_records"] = old_refresh.candidate_records
        if not active_only and candidate_domain_reused:
            self._full_candidate_records = old_refresh.candidate_records
        runtime_stage_started_at = time.perf_counter()
        build_result = build_buffering_state_for_model(model, params, self.config, **build_options)
        record_runtime_stage("state_rebuild_ms", runtime_stage_started_at)
        if build_result.status != "ok":
            raise RuntimeError(
                "candidate geometry refresh failed to rebuild state: "
                f"{build_result.reason or build_result.metadata}"
            )
        new_state = build_result.buffer_optimization_state
        runtime_stage_started_at = time.perf_counter()
        # The builder consumes the supplied candidate sequence in order and
        # only filters rows, so an unchanged count proves that the exact
        # cached sequence survived without comparing every row again.
        domain_matches_old_refresh = candidate_domain_reused and (
            build_result.candidate_count == len(old_refresh.candidate_records)
        )
        if not domain_matches_old_refresh and candidate_domain_reused:
            domain_matches_old_refresh = (
                len(new_state.candidate_records)
                == len(old_refresh.candidate_records)
                and all(
                    candidate_key(built) == candidate_key(refreshed)
                    and int(built.get("candidate_id", -1))
                    == int(refreshed.get("candidate_id", -1))
                    for built, refreshed in zip(
                        new_state.candidate_records,
                        old_refresh.candidate_records,
                        strict=True,
                    )
                )
            )
        if domain_matches_old_refresh:
            new_state.candidate_records = old_refresh.candidate_records
            new_refresh = old_refresh
        else:
            new_refresh = refresh_candidate_geometry(
                new_state,
                live_topology,
                candidate_generation=live_topology.get(
                    "topology_generation",
                    0,
                ),
                geometry_generation=geometry_generation,
            )
            new_state.candidate_records = new_refresh.candidate_records
        record_runtime_stage("new_geometry_refresh_ms", runtime_stage_started_at)
        runtime_stage_started_at = time.perf_counter()
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
        if not active_only:
            self._full_candidate_records = tuple(new_state.candidate_records)
            self._candidate_state_contains_full_domain = True
            self._full_candidate_topology_signature = live_topology_signature
            self._full_candidate_structure_signature = live_structure_signature
            self._full_candidate_geometry_digest = new_refresh.geometry_digest
            self._full_candidate_refresh_topology_digest = (
                new_refresh.topology_digest
            )
            self._full_candidate_candidate_generation = (
                new_refresh.candidate_generation
            )
        else:
            self._full_candidate_structure_signature = live_structure_signature
        record_runtime_stage("state_remap_ms", runtime_stage_started_at)
        runtime_stage_started_at = time.perf_counter()
        self.buffer_optimization_state = new_state
        self.buffer_relaxed_timing_payload = build_result.buffer_relaxed_timing_payload
        self._candidate_state_contains_full_domain = not bool(active_only)
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
                "candidate_domain_reused": candidate_domain_reused,
                "candidate_domain_reused_count": (
                    len(build_options["candidate_records"])
                    if candidate_domain_reused
                    else 0
                ),
                "candidate_records_cache_hit": candidate_records_cache_hit,
                "candidate_state_remap_failed": list(
                    remap.get("candidate_state_remap_failed", ()) or ()
                ),
                "candidate_new_geometry_reused": bool(
                    domain_matches_old_refresh
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
        record_runtime_stage("publish_and_invalidate_ms", runtime_stage_started_at)
        return_values = {
            "candidate_generation": int(new_refresh.candidate_generation),
            "geometry_generation": int(new_refresh.geometry_generation),
            "candidate_count": len(new_refresh.candidate_records),
            "invalidated_candidate_count": len(
                new_refresh.invalidated_candidate_keys
            ),
            "geometry_digest": new_refresh.geometry_digest,
            "topology_digest": new_refresh.topology_digest,
        }
        state_build_profile = (build_result.metadata or {}).get(
            "state_build_runtime_profile"
        )
        release_started_at = time.perf_counter()
        del state, old_refresh, new_refresh, new_state, build_result
        release_ms = (time.perf_counter() - release_started_at) * 1000.0
        if runtime_profile is not None:
            runtime_profile["total_ms"] = (
                time.perf_counter() - refresh_started_at
            ) * 1000.0
            accounted_names = (
                "refresh_inputs_ms",
                "old_geometry_refresh_ms",
                "state_rebuild_ms",
                "new_geometry_refresh_ms",
                "state_remap_ms",
                "publish_and_invalidate_ms",
            )
            runtime_profile["accounted_ms"] = sum(
                float(runtime_profile.get(name, 0.0) or 0.0)
                for name in accounted_names
            )
            runtime_profile["unattributed_ms"] = max(
                0.0,
                float(runtime_profile["total_ms"])
                - float(runtime_profile["accounted_ms"]),
            )
            runtime_profile["release_temporaries_ms"] = release_ms
            if isinstance(state_build_profile, dict):
                runtime_profile["state_build"] = dict(state_build_profile)
            metadata["candidate_geometry_runtime_profile"] = runtime_profile
            self.buffer_relaxed_timing_payload["metadata"] = metadata
            self._summary["candidate_geometry_runtime_profile"] = runtime_profile
            self._record_candidate_geometry_profile(
                reason=reason,
                active_only=active_only,
                candidate_records_cache_hit=candidate_records_cache_hit,
                runtime_profile=runtime_profile,
            )
        self._summary["candidate_geometry_refresh"] = dict(metadata)
        return {
            "status": "ok",
            "reason": str(reason),
            **return_values,
            "activation_remap": dict(remap),
            "candidate_domain_reused": candidate_domain_reused,
            "candidate_records_cache_hit": candidate_records_cache_hit,
            "runtime_profile": runtime_profile,
        }
