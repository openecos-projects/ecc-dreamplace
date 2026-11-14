"""Remaining-area budget policy for S5B1 placement timing windows."""

import math
import statistics
from contextlib import contextmanager
from dataclasses import dataclass

from dreamplace.ops.buffer_insertion.contract import BufferFamilyContract
from dreamplace.ops.routability.cooptimization_area import AreaState, capture_area

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
