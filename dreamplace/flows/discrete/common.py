"""Freeze a phase's timing coefficients and restore the owning GP policy."""


from contextlib import contextmanager
from dataclasses import replace
from dreamplace.flows.config.specs import TIMING_OPT_COEFFICIENT_FIELDS
import torch
from dreamplace.flows.config.specs import DIFF_SIZING_ACTION_MASKS
from dreamplace.flows.config.specs import DEFAULT_TIMING_OPT_COEFFICIENTS
from dreamplace.ops.routability.cooptimization_area import capture_area
from dataclasses import dataclass
import math
import statistics
from dreamplace.ops.buffer_insertion.contract import BufferFamilyContract
from dreamplace.ops.routability.cooptimization_area import AreaState


__all__ = [
    "parse_enabled_flag",
    "segment_counts",
    "sync_owner_native_sizing",
    "timing_area_fits",
    "timing_electrical_window",
    "timing_sizing_coefficients",
    "transfer_segment_counts",
    "virtual_timing_area",
]


@contextmanager
def timing_phase_coefficients(model, fixed=None):
    original = {
        name: float(getattr(model, field))
        for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items()
    }
    weight = model.timing_grad_balance_weight
    hold = getattr(model, "hold_objective", None)
    coefficients = dict(original if fixed is None else fixed)
    if fixed is None and hold is not None:
        coefficients.update(hold_wns=hold.wns, hold_tns=hold.tns)
    if hold is None and (
        coefficients.get("hold_wns", 0) or coefficients.get("hold_tns", 0)
    ):
        raise ValueError("nonzero phase hold weights require timing_hold_enabled")
    try:
        for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
            setattr(model, field, float(coefficients[name]))
        if hold is not None:
            model.hold_objective = replace(
                hold,
                wns=coefficients.get("hold_wns", 0),
                tns=coefficients.get("hold_tns", 0),
            )
        model.timing_grad_balance_weight = weight if fixed is None else 1.0
        yield coefficients | {
            "timing_grad_balance_weight": float(model.timing_grad_balance_weight)
        }
    finally:
        for name, field in TIMING_OPT_COEFFICIENT_FIELDS.items():
            setattr(model, field, original[name])
        if hold is not None:
            model.hold_objective = hold
        model.timing_grad_balance_weight = weight


def parse_enabled_flag(params, attr: str, *, default: bool = True) -> bool:
    """Parse boolean/int/string config flag with consistent string-falsy handling."""
    raw = getattr(params, attr, int(default) if isinstance(default, bool) else default)
    if raw is None:
        return bool(default)
    if isinstance(raw, str):
        return raw.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(int(raw))


def timing_sizing_action_mask(params, action_mask=None):
    """Resolve window/diff-opt defaults while preserving explicit phase masks."""
    if action_mask is not None:
        return action_mask
    action = getattr(params, "timing_sizing_action", "quad")
    if action not in ("up", "down", "quad"):
        raise ValueError("timing_sizing_action must be one of: up, down, quad")
    return DIFF_SIZING_ACTION_MASKS[action]


@contextmanager
def timing_electrical_window(params, data):
    """Temporarily expose the timing lane used by discrete sizing."""

    size, vt = data.get_continuous_size_parameter(), data.vt_logits
    mode, lane = params.placement_sizing_mode, params.timing_objective_lane
    timing_objective_enabled = params.differentiable_timing_obj
    requires_grad = (
        None if size is None else size.requires_grad,
        None if vt is None else vt.requires_grad,
    )
    params.placement_sizing_mode = "size_only"
    params.timing_objective_lane = "timing_slew_cap"
    params.differentiable_timing_obj = 1
    if size is not None:
        size.requires_grad_(requires_grad=True)
    if vt is not None:
        vt.requires_grad_(requires_grad=True)
    try:
        yield
    finally:
        params.placement_sizing_mode = mode
        params.timing_objective_lane = lane
        params.differentiable_timing_obj = timing_objective_enabled
        if size is not None:
            size.requires_grad_(requires_grad[0])
            size.grad = None
        if vt is not None:
            vt.requires_grad_(requires_grad[1])
            vt.grad = None


@contextmanager
def timing_sizing_coefficients(params, model):
    """Apply sizing-window setup/DRV/hold weights and restore the live GP policy."""

    policy = getattr(
        params,
        "timing_opt_coefficients",
        DEFAULT_TIMING_OPT_COEFFICIENTS,
    )
    fixed = None if policy["mode"] == "inherit" else {
        name: value for name, value in policy.items() if name != "mode"
    }
    with timing_phase_coefficients(model, fixed) as effective:
        yield {"mode": policy["mode"], **effective}


def virtual_timing_area(owner, model, state=None):
    """Return the area represented by candidate or segment virtual buffers."""

    if owner.lane is None and state is None:
        return 0.0
    state = owner.lane._current_state() if state is None else state
    if state is None:
        return 0.0
    if getattr(state, "is_typed_candidate", False):
        cell_ids = state.drive_type_cell_ids.to(
            device=owner.data.flat_libcell_width.device,
        )
        widths = owner.data.flat_libcell_width[cell_ids].double()
        heights = owner.data.flat_libcell_height[cell_ids].double()
        counts = state.activation_param.detach().double().sum(dim=0).to(widths.device)
        padding = 2.0 * float(getattr(owner.params, "cell_padding_x", 0.0))
        return float(((widths + padding) * heights * counts).sum())
    width, height = model._segment_virtual_buffer_size()
    if hasattr(state, "z_param"):
        count = state.z_param.detach().sum()
    elif hasattr(state, "activation_param"):
        count = state.activation_param.detach().sum()
    else:
        return 0.0
    return float(count) * width * height


def timing_area_fits(owner, model, capacity):
    """Evaluate current physical plus virtual timing area."""

    return capture_area(
        owner.data,
        owner.placedb,
        owner.virtual_area(model),
        capacity=capacity,
    ).fits()


def segment_counts(state):
    """Return positive virtual segment counts keyed by stable topology IDs."""

    if not hasattr(state, "z_param"):
        return {}
    rows = state.positive_segment_indices()
    return {
        (int(net), int(parent), int(child)): float(count)
        for net, parent, child, count in zip(
            state.segment_net_id[rows].detach().cpu().tolist(),
            state.parent_node_id[rows].detach().cpu().tolist(),
            state.child_node_id[rows].detach().cpu().tolist(),
            state.z_param[rows].detach().cpu().tolist(),
            strict=True,
        )
    }


def transfer_segment_counts(state, counts):
    """Restore retained segment counts by their topology identity."""

    if not counts or not hasattr(state, "z_param"):
        return
    candidate_keys = torch.stack(
        (
            state.segment_net_id,
            state.parent_node_id,
            state.child_node_id,
        ),
        dim=1,
    ).long()
    query_keys = torch.tensor(
        list(counts),
        dtype=torch.long,
        device=candidate_keys.device,
    )
    _, inverse = torch.unique(
        torch.cat((candidate_keys, query_keys)),
        dim=0,
        return_inverse=True,
    )
    candidate_ids = inverse[: state.num_segments]
    query_ids = inverse[state.num_segments :]
    row_by_key = torch.full(
        (int(inverse.max()) + 1,),
        -1,
        dtype=torch.long,
        device=inverse.device,
    )
    row_by_key[candidate_ids] = torch.arange(
        state.num_segments,
        device=inverse.device,
    )
    rows = row_by_key[query_ids]
    if bool((rows < 0).any()):
        raise RuntimeError("retained buffer segment lost its identity at window preparation")
    with torch.no_grad():
        state.z_param[rows] = torch.tensor(
            list(counts.values()),
            dtype=state.z_param.dtype,
            device=state.z_param.device,
        )


def sync_owner_native_sizing(owner):
    """Commit owner master changes to the ECC native DB and Python timing carrier."""

    from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
    from dreamplace.ops.placeio_ecc.timing_carrier import apply_inplace_carrier_sizing
    from dreamplace.ops.routability.gpugr_context import invalidate_route_state

    if str(owner.params.place_io_engine) != "ecc":
        raise ValueError("inflation S5B1 requires the ECC native backend")
    targets = owner.data.inst_cell_id.detach().cpu()
    changed_ids = torch.nonzero(targets != owner.native_cell_ids).flatten().tolist()
    changed = len(changed_ids)
    if not changed:
        return {"ok": True, "accepted_count": 0, "requested_count": 0}
    names = [
        name.decode() if isinstance(name, bytes) else str(name)
        for name in owner.placedb.flat_libcell_names
    ]
    summary = PlaceIOFunction.apply_sizing(
        owner.placedb.rawdb,
        targets.numpy(),
        names,
        current_cell_ids=owner.native_cell_ids.numpy(),
    )
    if not summary.get("ok") or int(summary.get("accepted_count", 0)) != changed:
        raise RuntimeError(f"S5B1 native sizing transaction failed: {summary}")
    if getattr(owner.placedb.rawdb, "pydb", None) is not None:
        resizes = [
            {"node_id": int(node_id), "target_cell_index": int(targets[node_id])}
            for node_id in changed_ids
        ]
        apply_inplace_carrier_sizing(owner.placedb.rawdb, resizes)
    owner.native_cell_ids.copy_(targets)
    invalidate_route_state(owner.placedb)
    return summary


@dataclass
class SizingState:
    size: torch.Tensor
    vt: torch.Tensor
    masters: torch.Tensor
    oscillation: dict
    pin_geometry: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, data, oscillation):
        return cls(
            data.get_continuous_size_parameter().detach().clone(),
            data.vt_logits.detach().clone(),
            data.inst_cell_id.detach().clone(),
            oscillation.snapshot(),
            {
                name: getattr(data, name).detach().clone()
                for name in (
                    "pin_offset_x",
                    "pin_offset_y",
                    "original_pin_offset_x",
                    "original_pin_offset_y",
                )
                if hasattr(data, name)
            },
        )

    def restore_parameters(self, data):
        changed = torch.nonzero(data.inst_cell_id != self.masters).flatten()
        with torch.no_grad():
            data.get_continuous_size_parameter().copy_(self.size)
            data.vt_logits.copy_(self.vt)
        return changed

    def restore_pin_geometry(self, data, placedb):
        # Master sync reconstructs library offsets. Restore the imported
        # geometry used to evaluate the snapshot, including inflation.
        changed = False
        with torch.no_grad():
            for name, values in self.pin_geometry.items():
                changed |= not torch.equal(getattr(data, name), values)
                getattr(data, name).copy_(values)
                if name in {"pin_offset_x", "pin_offset_y"}:
                    getattr(placedb, name)[:] = values.detach().cpu().numpy()
            for name in ("original_pin_offset_x", "original_pin_offset_y"):
                if name not in self.pin_geometry and hasattr(data, name):
                    delattr(data, name)
        return changed


def objective_snapshot(model, loss):
    terms = dict(getattr(model, "last_timing_objective_terms", {}) or {})
    raw_terms = dict(terms.get("raw_terms", {}) or {})
    weighted_terms = dict(terms.get("weighted_terms", {}) or {})
    report_aligned = dict(terms.get("report_aligned_terms", {}) or {})
    timing_op = getattr(getattr(model, "op_collections", None), "timing_propagation_op", None)
    slew = report_aligned.get("slew_violation_ns")
    if slew is None and raw_terms.get("slew") is not None:
        slew = float(raw_terms["slew"]) / 1000.0
    if slew is None and timing_op is not None:
        slew = getattr(timing_op, "last_total_slew_violation", None)
    cap = report_aligned.get("cap_violation_pF")
    if cap is None and raw_terms.get("cap") is not None:
        cap = float(raw_terms["cap"]) / 1000.0
    if cap is None and timing_op is not None:
        cap = getattr(timing_op, "last_total_cap_violation", None)
    total_slew = (
        None if timing_op is None else getattr(timing_op, "last_total_slew_violation", None)
    )
    total_cap = None if timing_op is None else getattr(timing_op, "last_total_cap_violation", None)
    timing_metrics = dict(getattr(model, "_diff_opt_last_timing_metrics", {}) or {})
    return {
        "score": float(terms.get("total_loss", loss.detach())),
        "wns": timing_metrics.get("wns"),
        "tns": timing_metrics.get("tns"),
        "slew_violation": None if slew is None else float(slew),
        "cap_violation": None if cap is None else float(cap),
        "total_slew_violation": (None if total_slew is None else float(total_slew)),
        "total_cap_violation": None if total_cap is None else float(total_cap),
        "raw_terms": raw_terms,
        "weighted_terms": weighted_terms,
    }


def phase_metrics(snapshot):
    """Return unweighted electrical diagnostics for an S/B phase boundary."""

    if not snapshot:
        snapshot = {}
    wns = snapshot.get("wns")
    tns = snapshot.get("tns")
    slew = snapshot.get("total_slew_violation", snapshot.get("slew_violation"))
    cap = snapshot.get("total_cap_violation")
    if cap is None and snapshot.get("cap_violation") is not None:
        cap = float(snapshot["cap_violation"])
    return {
        # PlaceObj timing_obj returns WNS/TNS in ps, and scalar violation
        # snapshots use ns and pF so they line up with external STA reports.
        "wns_ns": None if wns is None else float(wns) / 1000.0,
        "tns_ns": None if tns is None else float(tns) / 1000.0,
        "total_slew_violation_ns": None if slew is None else float(slew),
        "total_cap_violation_pf": None if cap is None else float(cap),
        "total_cap_violation_ff": None if cap is None else float(cap) * 1000.0,
    }


def probe_action_interpolation(evaluate, context, instance, action):
    """Compare small positive beta steps with the base-point adjoint.

    The normal deferred operator queries targets during backward at beta=0.
    A finite difference must explicitly evaluate target LUTs in forward.
    """

    previous = getattr(context, "evaluate_targets", False)
    saved_beta = context.beta.detach().clone()
    records = []
    try:
        context.evaluate_targets = True
        with torch.no_grad():
            context.beta.zero_()
            steps = (
                (0.0, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0)
                if context.beta.dtype == torch.float64
                else (0.0, 0.001, 0.01, 0.1, 1.0)
            )
            for step in steps:
                context.beta[instance, action] = step
                metrics = evaluate()
                records.append({"beta": step, **metrics})
    finally:
        with torch.no_grad():
            context.beta.copy_(saved_beta)
        context.evaluate_targets = previous
    base = records[0]
    for row in records[1:]:
        for metric in ("tns", "wns"):
            row["delta_" + metric] = row[metric] - base[metric]
            row["slope_" + metric] = row["delta_" + metric] / row["beta"]
    return records


DEFAULT_MAX_DENSITY_RATIO = 0.90
DEFAULT_WINDOW_REMAINING_AREA_FRACTION = 0.25
_FITS_FLOAT_TOLERANCE = 1.0e-2


def _buffer_family_cell_ids(placedb, params) -> list[int]:
    if placedb is None:
        return []
    if params is not None:
        metadata = getattr(placedb, "pydb", None)
        if metadata is None:
            metadata = placedb
        artifact = BufferFamilyContract.from_buffering_params(
            metadata,
            params,
            backend=str(getattr(params, "place_io_engine", "unknown") or "unknown"),
        ).build_artifact()
        legal_ids = [int(cid) for cid in artifact.get("legal_cell_ids", ()) or ()]
        if legal_ids:
            return legal_ids

    flat_info = getattr(placedb, "flat_libcell_info", None)
    if flat_info is None:
        return []
    candidate_raw = getattr(placedb, "buffer_main_type_candidate_indices", None)
    main_ids = (
        {int(v) for v in candidate_raw if int(v) >= 0}
        if candidate_raw is not None
        else set()
    )
    primary_main = int(getattr(placedb, "buffer_main_type_index", -1) or -1)
    if primary_main >= 0:
        main_ids.add(primary_main)
    if not main_ids:
        return []
    return [
        idx
        for idx, row in enumerate(flat_info)
        if len(row) >= 2 and int(round(float(row[1]))) in main_ids
    ]


def _legal_buffer_cell_areas(data, placedb, params) -> list[float]:
    cell_ids = _buffer_family_cell_ids(placedb, params)
    if not cell_ids:
        return []
    widths = getattr(data, "flat_libcell_width", None)
    heights = getattr(data, "flat_libcell_height", None)
    if widths is None or heights is None:
        widths = getattr(placedb, "flat_libcell_width", None)
        heights = getattr(placedb, "flat_libcell_height", None)
    if widths is None or heights is None:
        return []
    num_w = len(widths)
    num_h = len(heights)
    padding_x = float(getattr(params, "cell_padding_x", 0.0) or 0.0)
    areas = []
    for cid in cell_ids:
        if not (0 <= cid < num_w and 0 <= cid < num_h):
            return []
        w = float(widths[cid]) + 2.0 * padding_x
        h = float(heights[cid])
        area = w * h
        if math.isfinite(area) and area > 0.0:
            areas.append(area)
    return sorted(areas)


def resolve_median_buffer_area(data, placedb, params) -> float | None:
    """Return median padded physical area of legal buffer cells in PlaceDB units."""
    areas = _legal_buffer_cell_areas(data, placedb, params)
    if not areas:
        return None
    return float(statistics.median(areas))


def resolve_proposal_buffer_area(data, placedb, params) -> float | None:
    """Return lower-quartile padded buffer area for multi-tier candidate pool sizing."""
    areas = _legal_buffer_cell_areas(data, placedb, params)
    if not areas:
        return None
    return float(areas[len(areas) // 4])


@dataclass(frozen=True)
class S5B1WindowAreaBudget:
    """Shared per-window 25% remaining-area budget across S5 and B1."""

    enabled: bool
    entry_movable: float
    placement_capacity: float
    max_density_ratio: float
    max_remaining_fraction: float
    safe_ceiling: float
    remaining_area: float
    max_window_area_delta: float
    window_capacity: float
    fits_capacity: float
    median_buffer_area: float | None
    window_buffer_equivalent_count: int
    proposal_buffer_area: float | None = None

    @classmethod
    def for_window(
        cls,
        entry: AreaState,
        *,
        median_buffer_area: float | None,
        proposal_buffer_area: float | None = None,
        max_density_ratio: float = DEFAULT_MAX_DENSITY_RATIO,
        max_remaining_fraction: float = DEFAULT_WINDOW_REMAINING_AREA_FRACTION,
    ) -> "S5B1WindowAreaBudget":
        cap = float(entry.capacity)
        mov = float(entry.movable)
        if (
            median_buffer_area is None
            or not math.isfinite(median_buffer_area)
            or median_buffer_area <= 0.0
        ):
            return cls(
                enabled=False,
                entry_movable=mov,
                placement_capacity=cap,
                max_density_ratio=float(max_density_ratio),
                max_remaining_fraction=float(max_remaining_fraction),
                safe_ceiling=cap,
                remaining_area=max(0.0, cap - mov),
                max_window_area_delta=max(0.0, cap - mov),
                window_capacity=cap,
                fits_capacity=cap,
                median_buffer_area=None,
                window_buffer_equivalent_count=0,
                proposal_buffer_area=None,
            )

        safe_ceiling = min(cap, float(max_density_ratio) * cap)
        remaining_area = max(0.0, safe_ceiling - mov)
        max_delta = float(max_remaining_fraction) * remaining_area
        window_cap = min(cap, mov + max_delta)
        fits_cap = min(cap, window_cap + _FITS_FLOAT_TOLERANCE)
        equiv_count = (
            int(math.ceil(max_delta / float(median_buffer_area)))
            if max_delta > 0.0
            else 0
        )
        prop_area = (
            float(proposal_buffer_area)
            if proposal_buffer_area is not None
            and math.isfinite(proposal_buffer_area)
            and proposal_buffer_area > 0.0
            else float(median_buffer_area)
        )
        return cls(
            enabled=True,
            entry_movable=mov,
            placement_capacity=cap,
            max_density_ratio=float(max_density_ratio),
            max_remaining_fraction=float(max_remaining_fraction),
            safe_ceiling=safe_ceiling,
            remaining_area=remaining_area,
            max_window_area_delta=max_delta,
            window_capacity=window_cap,
            fits_capacity=fits_cap,
            median_buffer_area=float(median_buffer_area),
            window_buffer_equivalent_count=equiv_count,
            proposal_buffer_area=prop_area,
        )

    def remaining_window_allowance(self, current_movable: float) -> float:
        return max(0.0, self.window_capacity - float(current_movable))

    def round_allowance(
        self,
        current_movable: float,
        *,
        round_idx: int,
        total_rounds: int,
    ) -> float:
        """Return round budget = remaining window allowance split over remaining rounds."""
        rem = self.remaining_window_allowance(current_movable)
        rounds_left = max(1, int(total_rounds) - int(round_idx))
        return rem / float(rounds_left)

    def round_capacity(
        self,
        current_movable: float,
        *,
        round_idx: int,
        total_rounds: int,
    ) -> float:
        """Return hard area ceiling for one buffering round with unused-budget rollover."""
        if not self.enabled:
            return self.window_capacity
        cur = float(current_movable)
        allowance = self.round_allowance(
            cur,
            round_idx=round_idx,
            total_rounds=total_rounds,
        )
        return min(self.window_capacity, cur + allowance)

    def sizing_budget_percent(
        self,
        current_movable: float,
        *,
        sizeable_count: int,
        fallback_percent: float,
    ) -> float:
        fallback = float(fallback_percent)
        if not self.enabled or sizeable_count <= 0 or self.median_buffer_area is None:
            return fallback
        allowance = self.remaining_window_allowance(current_movable)
        if allowance <= 0.0:
            # Keep baseline candidate window open for zero-area VT swaps and
            # area-reducing downsizings while window_capacity blocks +area moves.
            return fallback
        equiv_count = max(1, int(math.ceil(allowance / self.median_buffer_area)))
        dynamic_percent = 100.0 * float(equiv_count) / float(sizeable_count)
        return min(100.0, max(fallback, dynamic_percent))

    def buffering_selection_fraction(
        self,
        current_movable: float,
        *,
        affected_net_count: int,
        fallback_fraction: float,
        round_idx: int | None = None,
        total_rounds: int | None = None,
    ) -> float | None:
        fallback = float(fallback_fraction)
        unit_area = self.proposal_buffer_area or self.median_buffer_area
        if not self.enabled or unit_area is None:
            return fallback
        allowance = (
            self.round_allowance(
                current_movable,
                round_idx=round_idx,
                total_rounds=total_rounds,
            )
            if round_idx is not None and total_rounds is not None
            else self.remaining_window_allowance(current_movable)
        )
        if allowance <= 0.0:
            return None
        if affected_net_count <= 0:
            return fallback
        equiv_count = max(1, int(math.ceil(allowance / unit_area)))
        return min(1.0, float(equiv_count) / float(affected_net_count))

    def to_summary(
        self,
        *,
        after_sizing_movable: float,
        after_window_movable: float,
    ) -> dict:
        sz_mov = float(after_sizing_movable)
        end_mov = float(after_window_movable)
        return {
            "enabled": self.enabled,
            "entry_movable": self.entry_movable,
            "placement_capacity": self.placement_capacity,
            "max_density_ratio": self.max_density_ratio,
            "max_remaining_fraction": self.max_remaining_fraction,
            "safe_ceiling": self.safe_ceiling,
            "remaining_area": self.remaining_area,
            "max_window_area_delta": self.max_window_area_delta,
            "window_capacity": self.window_capacity,
            "median_buffer_area": self.median_buffer_area,
            "window_buffer_equivalent_count": self.window_buffer_equivalent_count,
            "after_sizing_movable": sz_mov,
            "sizing_consumed_area": sz_mov - self.entry_movable,
            "buffering_remaining_allowance": self.remaining_window_allowance(sz_mov),
            "after_window_movable": end_mov,
            "window_consumed_area": end_mov - self.entry_movable,
        }


@contextmanager
def bind_window_area_budget(owner, budget: S5B1WindowAreaBudget):
    """Bind a window area budget to an S5B1 owner and its optional LUT delegate."""
    prev_owner = getattr(owner, "_active_area_budget", None)
    delegate = getattr(owner, "_lut_action_owner", None)
    prev_delegate = (
        getattr(delegate, "_active_area_budget", None) if delegate is not None else None
    )
    owner._active_area_budget = budget
    if delegate is not None:
        delegate._active_area_budget = budget
    try:
        yield budget
    finally:
        owner._active_area_budget = prev_owner
        if delegate is not None:
            delegate._active_area_budget = prev_delegate


def effective_sizing_budget_percent(owner, model) -> float:
    """Resolve current round's sizing candidate budget percent for ``owner``."""
    fallback = float(owner.params.discrete_gradient_topk_shared_budget_percent)
    budget = getattr(owner, "_active_area_budget", None)
    if budget is None or not budget.enabled:
        return fallback
    data = owner.data
    sizeable_mask = getattr(data, "inst_is_sizeable", None)
    if sizeable_mask is None:
        return fallback
    sizeable_count = int(sizeable_mask.sum().item())
    current_movable = capture_area(
        data,
        owner.placedb,
        owner.virtual_area(model),
        capacity=budget.placement_capacity,
    ).movable
    return budget.sizing_budget_percent(
        current_movable,
        sizeable_count=sizeable_count,
        fallback_percent=fallback,
    )


def effective_fits_capacity(owner, capacity: float) -> float:
    """Apply float32/float64 summation tolerance when checking window capacity."""
    cap = float(capacity)
    budget = getattr(owner, "_active_area_budget", None)
    if budget is not None and budget.enabled and cap == budget.window_capacity:
        return budget.fits_capacity
    return cap


def effective_buffering_selection_fraction(
    owner,
    model,
    *,
    affected_net_count: int,
    fallback_fraction: float,
    round_idx: int | None = None,
    total_rounds: int | None = None,
    current_movable: float | None = None,
) -> float | None:
    """Resolve B1's remaining window-allowance selection fraction (`None` when exhausted)."""
    fallback = float(fallback_fraction)
    budget = getattr(owner, "_active_area_budget", None)
    if budget is None or not budget.enabled:
        return fallback
    if current_movable is None:
        current_movable = capture_area(
            owner.data,
            owner.placedb,
            owner.virtual_area(model),
            capacity=budget.placement_capacity,
        ).movable
    return budget.buffering_selection_fraction(
        float(current_movable),
        affected_net_count=affected_net_count,
        fallback_fraction=fallback,
        round_idx=round_idx,
        total_rounds=total_rounds,
    )
