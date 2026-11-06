"""Standalone diff-sizing with selectable placement/GR RC."""

import copy
import json
import math
import time
from pathlib import Path

from dreamplace.flows.diff_opt_spec import TimingCoefficients
from dreamplace.flows.diff_sizing_spec import (
    DiffSizingFlowSpec,
    parse_diff_sizing_flows,
)
from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.ops.routability.cooptimization_area import SizingWindowGeometry

_COEFFICIENT_FIELDS = {
    "wns": "timing_wns_coeff",
    "tns": "timing_tns_coeff",
    "slew": "timing_slew_weight",
    "cap": "timing_cap_weight",
}


def _stage_rounds(params):
    stages = list(getattr(params, "global_place_stages", None) or ())
    if not stages or not isinstance(stages[0], dict):
        return 0
    try:
        return int(stages[0].get("iteration", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _legacy_phase(params, rounds=None):
    return DiffSizingFlowSpec(
        action="up",
        rounds=_stage_rounds(params) if rounds is None else int(rounds),
        coefficients=TimingCoefficients(
            wns=float(getattr(params, "timing_wns_coeff", 500.0)),
            tns=float(getattr(params, "timing_tns_coeff", 5.0)),
            slew=float(getattr(params, "timing_slew_weight", 1.0)),
            cap=float(getattr(params, "timing_cap_weight", 1.0)),
        ),
        selection_percent=float(
            getattr(
                params,
                "discrete_gradient_topk_shared_budget_percent",
                1.0,
            )
        ),
    )


def _resolve_phases(params):
    flows = parse_diff_sizing_flows(
        getattr(params, "diff_sizing_optimization_flows", None)
    )
    return (flows if flows is not None else (_legacy_phase(params),))


def _validate_action_profile(params):
    mode = str(
        getattr(params, "timing_action_gradient_mode", "lut_action")
        or "lut_action"
    )
    if mode != "lut_action":
        raise ValueError("diff_sizing action path requires timing_action_gradient_mode=lut_action")
    if str(getattr(params, "flow_kind", "")) != "sizing":
        raise ValueError("diff_sizing action path requires flow_kind=sizing")
    timing_rc_mode = str(getattr(params, "timing_rc_mode", "gr") or "gr")
    if timing_rc_mode not in {"placement", "gr"}:
        raise ValueError(
            "diff_sizing action path requires timing_rc_mode=placement or gr"
        )
    if str(getattr(params, "placement_sizing_mode", "")) != "size_only":
        raise ValueError("diff_sizing action path requires placement_sizing_mode=size_only")
    if int(getattr(params, "diff_sizing_continuous_steps", 0) or 0) != 0:
        raise ValueError(
            "lut_action standalone diff_sizing requires "
            "diff_sizing_continuous_steps=0"
        )
    phases = _resolve_phases(params)
    rounds = sum(spec.rounds for spec in phases)
    if rounds <= 0:
        raise ValueError("diff_sizing action path requires a positive stage iteration count")
    return phases


def _build_timing_model(engine):
    """Build the zero-iteration timing model without moving placement cells."""

    params = engine.params
    saved = {
        "flow_kind": params.flow_kind,
        "global_place_flag": params.global_place_flag,
        "legalize_flag": params.legalize_flag,
        "detailed_place_flag": params.detailed_place_flag,
        "global_place_stages": copy.deepcopy(params.global_place_stages),
    }
    stages = copy.deepcopy(params.global_place_stages)
    for stage in stages:
        stage["iteration"] = 0
    params.global_place_stages = stages
    params.flow_kind = "placement"
    params.global_place_flag = 1
    params.legalize_flag = 0
    params.detailed_place_flag = 0
    try:
        engine.place()
    finally:
        for name, value in saved.items():
            setattr(params, name, value)
    model = getattr(engine.placer, "last_global_place_model", None)
    if model is None:
        raise RuntimeError("diff_sizing did not construct a timing model")
    return model


def _legalize_and_measure(engine, params):
    """Legalize the committed masters and refresh terminal timing."""

    from dreamplace.NonLinearPlace import NonLinearPlace

    terminal_params = copy.deepcopy(params)
    terminal_params.flow_kind = "placement"
    terminal_params.global_place_flag = 0
    terminal_params.legalize_flag = 1
    terminal_params.detailed_place_flag = 0
    terminal_params.differentiable_timing_obj = 0
    terminal_params.diff_timing_driven_placement = 0
    terminal_params.timing_opt_enabled = 0
    terminal_params.timing_action_gradient_mode = "off"
    terminal_params.result_dir = str(Path(params.result_dir) / "terminal")
    Path(terminal_params.result_dir).mkdir(parents=True, exist_ok=True)

    engine.placedb.cooptimization_geometry = None
    placer = NonLinearPlace(terminal_params, engine.placedb, None)
    engine.placer = placer
    engine.rsmt, engine.hpwl, engine.metrics = placer(terminal_params, engine.placedb)
    if not placer.op_collections.legality_check_op(placer.pos[0]):
        raise RuntimeError("diff_sizing terminal placement is not legal")
    _, _, engine.metrics = placer.run_sta_only(terminal_params, engine.placedb)
    timing = engine.metrics["timing_stage_summary"]["sta"]
    if not all(math.isfinite(float(timing[key])) for key in ("wns", "tns")):
        raise RuntimeError("diff_sizing terminal STA produced non-finite WNS/TNS")
    return dict(timing)


class DiffSizingController:
    """Run one fixed-budget sizing phase from the four-action backward."""

    def __init__(self, params, placedb, data, ops, refresh_topology):
        self.params = params
        self.placedb = placedb
        self.data = data
        self.ops = ops
        self.owner = DiscreteTimingOwner(
            params,
            placedb,
            data,
            ops,
            refresh_topology,
            refresh_candidates_after_sizing=False,
        )

    def run(self, model, pos, phases):
        if isinstance(phases, int):
            phases = (_legacy_phase(self.params, phases),)
        phases = tuple(phases)
        if not phases:
            raise ValueError("diff_sizing requires at least one action phase")
        started_at = time.perf_counter()
        phase_records = []
        all_rounds = []
        geometry = SizingWindowGeometry(self.data, self.placedb, pos)
        for index, spec in enumerate(phases, 1):
            previous_rounds = getattr(self.params, "timing_opt_sizing_rounds", None)
            previous_percent = getattr(
                self.params,
                "discrete_gradient_topk_shared_budget_percent",
                None,
            )
            previous_coefficients = {
                name: getattr(model, field, None)
                for name, field in _COEFFICIENT_FIELDS.items()
            }
            previous_hold_objective = getattr(model, "hold_objective", None)
            self.params.timing_opt_sizing_rounds = int(spec.rounds)
            self.params.discrete_gradient_topk_shared_budget_percent = float(
                spec.selection_percent
            )
            try:
                for name, field in _COEFFICIENT_FIELDS.items():
                    setattr(model, field, float(getattr(spec.coefficients, name)))
                if previous_hold_objective is None and (
                    spec.coefficients.hold_wns or spec.coefficients.hold_tns
                ):
                    raise ValueError(
                        "nonzero diff_sizing hold weights require timing_hold_enabled"
                    )
                if previous_hold_objective is not None:
                    from dataclasses import replace

                    model.hold_objective = replace(
                        previous_hold_objective,
                        wns=float(spec.coefficients.hold_wns),
                        tns=float(spec.coefficients.hold_tns),
                    )
                with self.owner.electrical_window():
                    phase_sizing = self.owner.run_sizing(
                        model,
                        pos,
                        geometry,
                        self.owner.capacity,
                        action_mask=spec.action_mask,
                    )
            finally:
                if previous_rounds is None:
                    if hasattr(self.params, "timing_opt_sizing_rounds"):
                        delattr(self.params, "timing_opt_sizing_rounds")
                else:
                    self.params.timing_opt_sizing_rounds = previous_rounds
                if previous_percent is None:
                    if hasattr(
                        self.params,
                        "discrete_gradient_topk_shared_budget_percent",
                    ):
                        delattr(
                            self.params,
                            "discrete_gradient_topk_shared_budget_percent",
                        )
                else:
                    self.params.discrete_gradient_topk_shared_budget_percent = (
                        previous_percent
                    )
                for name, field in _COEFFICIENT_FIELDS.items():
                    setattr(model, field, previous_coefficients[name])
                model.hold_objective = previous_hold_objective
            for record in phase_sizing.get("rounds", ()) or ():
                record["phase"] = index
                record["phase_action"] = spec.action
                all_rounds.append(record)
            phase_records.append(
                {
                    "phase": index,
                    "action": spec.action,
                    "rounds": int(spec.rounds),
                    "selection_percent": float(spec.selection_percent),
                    "coefficients": spec.coefficients.to_dict(),
                    "sizing": phase_sizing,
                }
            )
        sizing = dict(phase_records[-1]["sizing"])
        sizing["phases"] = phase_records
        sizing["phase_count"] = len(phase_records)
        sizing["rounds"] = all_rounds
        sizing["total_rounds"] = sum(spec.rounds for spec in phases)
        sizing["best_phase"] = len(phase_records)
        sizing["initial_objective"] = phase_records[0]["sizing"]["initial_objective"]
        sizing["best_objective"] = phase_records[-1]["sizing"]["best_objective"]
        sizing["run_sizing_ms"] = float(
            (time.perf_counter() - started_at) * 1000.0
        )
        # Match the diff_opt phase boundary: publish the restored footprint
        # before the native master transaction so downstream geometry and
        # timing see the same state that produced ``sizing``.
        self.owner.publish_geometry(model)
        sizing_commit = self.owner.sync_native_sizing()
        return {
            "sizing": sizing,
            "sizing_commit": sizing_commit,
        }


def _write_summary(params, payload):
    result_dir = Path(params.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / "diff_sizing_summary.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
    return path


def run_diff_sizing(engine):
    """Run standalone diff-sizing through the opt-in four-action controller."""

    started_at = time.perf_counter()
    params = engine.params
    phases = _validate_action_profile(params)
    rounds = sum(spec.rounds for spec in phases)
    timing_rc_mode = str(getattr(params, "timing_rc_mode", "gr") or "gr")
    previous_rounds = getattr(params, "timing_opt_sizing_rounds", None)
    params.timing_opt_sizing_rounds = rounds

    try:
        engine.setup_placedb()
        gr_session = None
        if timing_rc_mode == "gr":
            gr_session = getattr(engine.placedb, "gr_sizing", None)
            if gr_session is None:
                raise RuntimeError(
                    "diff_sizing action path requires an initialized GR timing session"
                )

        model = _build_timing_model(engine)
        data = engine.placer.data_collections
        ops = engine.placer.op_collections
        pos = data.pos[0]
        controller = DiffSizingController(
            params,
            engine.placedb,
            data,
            ops,
            engine.placer._refresh_live_timing_topology,
        )
        operation = controller.run(model, pos, phases)

        refresh = engine._physical_mutation_backend().refresh(
            refresh_mode="full_rebuild",
            rebuild_mode="full_rebuild",
        )
        if str(refresh.get("status") or "") != "ok":
            raise RuntimeError(f"diff_sizing native refresh failed: {refresh}")
        if gr_session is not None:
            # The frozen GR snapshot is the optimization reference.  The
            # refresh rebuilds the placement DB, but the terminal sizing
            # finalizer still needs the original pin-name mapping for its
            # frozen LUT report.
            engine.placedb.gr_sizing = gr_session

        terminal_timing = _legalize_and_measure(engine, params)
        params.timing_action_gradient_mode = "off"
        try:
            result = {
                "status": "ok",
                "flow_kind": "sizing",
                "rsmt": engine.rsmt,
                "hpwl": engine.hpwl,
                "density": float(getattr(params, "target_density", 0.0) or 0.0),
                "iteration": rounds,
                "objective": -1,
                "overflow": -1,
                "max_density": -1,
                "diff_sizing": {
                    "mode": "lut_action",
                    "rounds": rounds,
                    "sizing": operation["sizing"],
                    "sizing_commit": operation["sizing_commit"],
                    "refresh": refresh,
                    "terminal_timing": terminal_timing,
                    "phase_count": operation["sizing"].get("phase_count"),
                    "phases": operation["sizing"].get("phases"),
                    "timing_rc_mode": timing_rc_mode,
                    "action_gradient_mode": "lut_action",
                    "run_sizing_ms": float(
                        (time.perf_counter() - started_at) * 1000.0
                    ),
                },
            }
            if (
                timing_rc_mode == "gr"
                and bool(getattr(params, "diff_sizing_terminal_gr_sta_enabled", 0))
            ):
                from dreamplace.flows.gr_sizing import finalize_gr_sizing

                final = finalize_gr_sizing(engine, result)
            else:
                final = dict(result)
                reason = (
                    "diff_sizing_timing_rc_mode_placement"
                    if timing_rc_mode == "placement"
                    else "diff_sizing_terminal_gr_sta_disabled"
                )
                final["gr_sizing"] = {
                    "status": "skipped",
                    "reason": reason,
                    "terminal_gr_sta_enabled": False,
                }
        finally:
            params.timing_action_gradient_mode = "lut_action"

        summary_path = _write_summary(
            params,
            {
                **result["diff_sizing"],
                "artifact": "diff_sizing",
                "artifact_version": 1,
                "status": final.get("status"),
                "final_timing": (
                    final.get("gr_sizing", {}).get("terminal_timing")
                    or terminal_timing
                ),
                "gr_sizing_artifact": final.get("gr_sizing_artifact"),
            },
        )
        final["diff_sizing_summary_path"] = str(summary_path)
        engine.last_run_result = final
        return final
    finally:
        if previous_rounds is not None:
            params.timing_opt_sizing_rounds = previous_rounds


__all__ = ["DiffSizingController", "run_diff_sizing"]
