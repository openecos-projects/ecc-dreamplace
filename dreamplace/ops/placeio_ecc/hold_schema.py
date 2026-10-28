"""Validate the independent min-analysis extension of the current ECC schema."""

import math
from dataclasses import dataclass

import numpy as np

from .sparse_timing_schema import SparseTimingView, read_sparse_timing_view
from .timing_schema import validate_timing_schema

HOLD_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class HoldContext:
    start_pin: int
    native_start_pin: int
    launch_pin: int
    endpoint_pin: int
    capture_pin: int
    launch_clock: str
    capture_clock: str
    path_tag: str
    source_type: int
    check_type: int
    data_transition: int
    launch_transition: int
    capture_transition: int
    capture_pin_transition: int
    arrival_ps: float
    required_ps: float
    data_slew_ps: float
    launch_time_ps: float
    launch_clock_arrival_ps: float
    launch_clock_slew_ps: float
    capture_edge_ps: float
    capture_arrival_ps: float
    capture_slew_ps: float
    check_time_ps: float
    uncertainty_ps: float
    cppr_ps: float
    normal_required_ps: float = 0.0
    capture_edge_index: int = -1
    selected_check_arc_idx: int = -1

    @property
    def required_base_ps(self):
        return self.capture_edge_ps + self.capture_arrival_ps + self.uncertainty_ps - self.cppr_ps

    @property
    def runtime_required_base_ps(self):
        """Required time after removing the live Liberty check value."""
        return self.required_ps - self.check_time_ps

    @property
    def launch_key(self):
        return self.launch_clock, self.path_tag, self.launch_transition, self.source_type


@dataclass(frozen=True)
class HoldLaunch:
    start_pin: int
    launch_pin: int
    launch_clock: str
    source_type: int
    data_transition: int
    launch_transition: int
    arrival_ps: float
    slew_ps: float
    launch_time_ps: float
    launch_slew_ps: float
    path_tag: str = ""

    @property
    def launch_key(self):
        return self.launch_clock, self.path_tag, self.launch_transition, self.source_type


@dataclass(frozen=True)
class HoldTimingData:
    endpoint_valid: np.ndarray
    endpoint_reason: np.ndarray
    check_valid: np.ndarray
    check_reason: np.ndarray
    startpoint_valid: np.ndarray
    input_arrival: np.ndarray
    input_slew: np.ndarray
    contexts: tuple[HoldContext, ...]
    launches: tuple[HoldLaunch, ...]
    min_pin_caps: tuple[np.ndarray, np.ndarray]
    lib_arc_check_types: np.ndarray
    output_caps: tuple[np.ndarray, np.ndarray]
    lib_arc_is_latch_data: np.ndarray
    sparse_view: SparseTimingView | None = None


def _qualification(pydb, valid_name, reason_name, count):
    for name in (valid_name, reason_name):
        if not hasattr(pydb, name):
            raise RuntimeError(f"ECC hold schema is missing {name}; rebuild/reinstall ecc-tools")
    valid = np.asarray(getattr(pydb, valid_name))
    reason = np.asarray(getattr(pydb, reason_name))
    if count == 0 and valid.size == reason.size == 0:
        valid = valid.reshape(0, 2).astype(bool)
        reason = reason.reshape(0, 2).astype(np.int32)
    if valid.shape != (count, 2) or reason.shape != (count, 2):
        raise RuntimeError(f"ECC min timing qualification rows do not align: {valid_name}")
    if valid.dtype != np.bool_ or not np.issubdtype(reason.dtype, np.integer):
        raise RuntimeError(f"ECC min timing qualification has invalid types: {valid_name}")
    if np.any((reason < 0) | (reason > 8)) or not np.array_equal(valid, reason == 0):
        raise RuntimeError(
            f"ECC min timing qualification has inconsistent reason codes: {reason_name}"
        )
    if np.any(reason == 6):
        raise RuntimeError(
            f"ECC hold analysis cannot represent a path-specific constraint: {reason_name}"
        )
    if np.any(reason == 7):
        raise RuntimeError(f"ECC hold analysis is missing a real min path state: {reason_name}")
    return valid.astype(bool, copy=True), reason.astype(np.int32, copy=True)


def _read_context(row, pin_count, clocks):
    try:
        context = HoldContext(**dict(row))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"ECC hold context fields do not match hold schema v{HOLD_SCHEMA_VERSION}"
        ) from error
    for name in ("start_pin", "native_start_pin", "launch_pin", "endpoint_pin", "capture_pin"):
        value = getattr(context, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(f"ECC hold context {name} must be an integer pin identity")
        if name == "capture_pin" and context.check_type == 0 and value == -1:
            continue
        if name == "launch_pin" and context.source_type == 1 and value == -1:
            continue
        if not 0 <= value < pin_count:
            raise RuntimeError(f"ECC hold context {name} is outside the design pin domain")
    if context.source_type not in (1, 2) or context.check_type not in (0, 2, 4):
        raise RuntimeError("ECC hold context has an invalid source/check class")
    if any(
        getattr(context, name) not in (-1, 1)
        for name in ("data_transition", "launch_transition", "capture_transition")
    ):
        raise RuntimeError("ECC hold context has an invalid transition")
    if context.launch_clock not in clocks or context.capture_clock not in clocks:
        raise RuntimeError("ECC hold context uses an undeclared launch/capture clock")
    for name in HoldContext.__dataclass_fields__:
        if name.endswith("_ps") and not math.isfinite(float(getattr(context, name))):
            raise RuntimeError(f"ECC hold context has a nonfinite {name}")
    if context.check_type and not math.isclose(
        context.required_ps,
        context.required_base_ps + context.check_time_ps,
        rel_tol=1e-9,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "ECC hold required time disagrees with capture/check/uncertainty/CPPR components"
        )
    return context


def read_hold_timing_schema(pydb):
    """Reject max-fallback payloads and return qualified, context-paired min data."""
    validate_timing_schema(pydb)
    if int(getattr(pydb, "hold_schema_version", 0)) != HOLD_SCHEMA_VERSION:
        raise RuntimeError(
            f"ECC hold requires hold schema v{HOLD_SCHEMA_VERSION}; rebuild/reinstall ecc-tools"
        )
    endpoints = np.asarray(pydb.end_points, dtype=np.int64)
    checks = np.asarray(pydb.endpoints_timing_check_arcs, dtype=np.int64).reshape(-1, 8)
    pin_count = len(pydb.pin_names)
    valid, reason = _qualification(
        pydb, "endpoints_min_valid", "endpoints_min_reason", len(endpoints)
    )
    check_valid, check_reason = _qualification(
        pydb, "endpoints_timing_check_min_valid", "endpoints_timing_check_min_reason", len(checks)
    )
    valid_by_pin = np.zeros((pin_count, 2), dtype=bool)
    valid_by_pin[endpoints] = valid
    if len(checks):
        invalid = (check_valid & ~valid_by_pin[checks[:, 1]]).any(axis=1)
        invalid |= check_valid.any(axis=1) & ~np.isin(checks[:, 6], (2, 4))
        if invalid.any():
            row = int(np.flatnonzero(invalid)[0])
            source, sink = map(int, checks[row, :2])
            names = [
                pydb.pin_names[pin] if 0 <= pin < pin_count else str(pin)
                for pin in (source, sink)
            ]
            raise RuntimeError(
                "ECC min check qualifies an absent endpoint or a max-only check class: "
                f"row={row}, {names[0]} -> {names[1]}, type={checks[row, 6]}, "
                f"check_valid={check_valid[row].tolist()}, "
                f"endpoint_valid={valid_by_pin[sink].tolist()}"
            )
    if not hasattr(pydb, "hold_contexts"):
        raise RuntimeError("ECC hold schema is missing hold_contexts")
    clocks = set(map(str, pydb.timing_clock_names))
    if not hasattr(pydb, "flat_libarc_check_types"):
        raise RuntimeError("ECC hold schema is missing library check-class identities")
    check_types = np.asarray(pydb.flat_libarc_check_types)
    if check_types.shape != (len(pydb.flat_libarc_info),) or not np.issubdtype(
        check_types.dtype, np.integer
    ):
        raise RuntimeError("ECC library check-class identities are misaligned")
    if len(checks) and np.any((checks[:, 3] < 0) | (checks[:, 3] >= len(check_types))):
        raise RuntimeError("ECC min check references an invalid library arc identity")
    if len(checks) and np.any(check_types[checks[:, 3]] != checks[:, 6]):
        raise RuntimeError("ECC check rows disagree with their library check class")
    if not hasattr(pydb, "flat_libarc_is_latch_data"):
        raise RuntimeError(
            "ECC hold schema is missing latch arc roles; rebuild/reinstall ecc-tools"
        )
    latch_data = np.asarray(pydb.flat_libarc_is_latch_data)
    if latch_data.shape != check_types.shape or (latch_data.size and latch_data.dtype != np.bool_):
        raise RuntimeError("ECC library latch arc roles are misaligned or not boolean")
    latch_data = latch_data.astype(bool, copy=True)
    latch_data.setflags(write=False)
    contexts = tuple(_read_context(row, pin_count, clocks) for row in pydb.hold_contexts)
    qualified_checks = {
        (int(arc[0]), int(arc[1]), int(arc[6]), edge)
        for arc, lanes in zip(checks, check_valid, strict=True)
        for edge in (0, 1)
        if lanes[edge]
    }
    covered = np.zeros_like(valid_by_pin)
    for context in contexts:
        edge = 0 if context.data_transition == 1 else 1
        if not valid_by_pin[context.endpoint_pin, edge]:
            raise RuntimeError("ECC hold context refers to an unqualified endpoint/edge")
        if context.check_type and (
            context.capture_pin, context.endpoint_pin, context.check_type, edge
        ) not in qualified_checks:
            raise RuntimeError("ECC hold context has no matching qualified min check")
        covered[context.endpoint_pin, edge] = True
    if not np.array_equal(covered[endpoints], valid):
        missing = np.flatnonzero((covered[endpoints] != valid).any(axis=1))
        samples = [
            f"{pydb.pin_names[endpoints[row]]}: qualified={valid[row].tolist()}, "
            f"covered={covered[endpoints[row]].tolist()}"
            for row in missing[:5]
        ]
        raise RuntimeError(
            "ECC hold contexts do not cover the qualified endpoint/edge domain: "
            f"{len(missing)} endpoints; " + "; ".join(samples)
        )
    context_checks = {
        (c.capture_pin, c.endpoint_pin, c.check_type, c.data_transition, c.capture_transition)
        for c in contexts if c.check_type
    }
    for arc, edges in zip(checks, check_valid, strict=True):
        source, sink, _, _, _, trigger, check_type, _ = map(int, arc)
        for edge, valid_edge in enumerate(edges):
            capture_edges = (-1, 1) if trigger == 0 else (trigger,)
            if valid_edge and not any(
                (source, sink, check_type, 1 if edge == 0 else -1, capture) in context_checks
                for capture in capture_edges
            ):
                raise RuntimeError("qualified min check group has no matching capture context")
    if not hasattr(pydb, "hold_launches"):
        raise RuntimeError("ECC hold schema is missing real min launch seeds")
    launches = []
    for row in pydb.hold_launches:
        try:
            seed = HoldLaunch(**dict(row))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"ECC hold launch fields do not match hold schema v{HOLD_SCHEMA_VERSION}"
            ) from error
        if seed.source_type not in (1, 2) or seed.launch_transition not in (-1, 1):
            raise RuntimeError("ECC hold launch has an invalid source/clock edge")
        # PI/input launches and register Q launches both carry independent
        # rise/fall min seeds.  A register source is not a single logical edge:
        # its CK->Q propagation produces both data transitions.
        edges = (-1, 1)
        if seed.data_transition not in edges or seed.launch_clock not in clocks:
            raise RuntimeError("ECC hold launch has an invalid data edge/clock")
        for name in ("start_pin", "launch_pin"):
            value = getattr(seed, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise RuntimeError("ECC hold launch pin identities must be integers")
            if name == "launch_pin" and seed.source_type == 1 and value == -1:
                continue
            if not 0 <= value < pin_count:
                raise RuntimeError("ECC hold launch pin is outside the design domain")
        if any(
            not math.isfinite(float(getattr(seed, name)))
            for name in HoldLaunch.__dataclass_fields__
            if name.endswith("_ps")
        ):
            raise RuntimeError("ECC hold launch has nonfinite timing seeds")
        launches.append(seed)
    sparse_view = None
    if hasattr(pydb, "timing_min_sparse") and pydb.timing_min_sparse:
        sparse_view = read_sparse_timing_view(pydb, analysis="min")
        sparse_view.require_supported()
        arrays = sparse_view.arrays
        for context, state in zip(contexts, arrays["context_state"], strict=True):
            edge = 0 if context.data_transition == 1 else 1
            if not (
                arrays["state_pin"][state] == context.endpoint_pin
                and sparse_view.domains[arrays["state_domain"][state]] == context.launch_clock
                and arrays["state_source_type"][state] == context.source_type
                and arrays["state_valid"][state, edge]
            ):
                raise RuntimeError(
                    "ECC hold context disagrees with its native sparse arrival state"
                )
    else:
        launch_keys = {seed.launch_key for seed in launches}
        if any(context.launch_key not in launch_keys for context in contexts):
            raise RuntimeError("ECC hold context has no matching min launch domain")
    starts = len(pydb.start_points)
    try:
        start_valid = np.asarray(pydb.startpoints_min_valid)
        if starts == 0 and start_valid.size == 0:
            start_valid = start_valid.reshape(0, 2).astype(bool)
        if start_valid.dtype != np.bool_:
            raise RuntimeError("ECC min input validity must be boolean")
        start_valid = start_valid.copy()
        arrivals = np.column_stack((pydb.min_inrdelays, pydb.min_infdelays)).astype(np.float64)
        slews = np.column_stack((pydb.min_inrtrans, pydb.min_inftrans)).astype(np.float64)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("ECC min input seeds are missing or malformed") from error
    if any(array.shape != (starts, 2) for array in (start_valid, arrivals, slews)):
        raise RuntimeError("ECC min input seeds do not align with start_points")
    if not np.all(np.isfinite(arrivals[start_valid])) or not np.all(
        np.isfinite(slews[start_valid])
    ):
        raise RuntimeError("ECC min input seeds contain nonfinite valid values")
    for array in (valid, reason, check_valid, check_reason, start_valid, arrivals, slews):
        array.setflags(write=False)
    pin_caps = []
    for name in ("flat_lib_pin_min_rcap", "flat_lib_pin_min_fcap"):
        if not hasattr(pydb, name):
            raise RuntimeError(f"ECC hold schema is missing {name}")
        array = np.asarray(getattr(pydb, name), dtype=np.float64).copy()
        if (
            array.shape != (len(pydb.flat_lib_pin_cap),)
            or not np.isfinite(array).all()
            or np.any(array < 0)
        ):
            raise RuntimeError("ECC min library pin capacitances are malformed")
        array.setflags(write=False)
        pin_caps.append(array)
    output_caps = []
    for name in ("min_outcaps_rise", "min_outcaps_fall"):
        if not hasattr(pydb, name):
            raise RuntimeError(f"ECC hold schema is missing {name}; rebuild/reinstall ecc-tools")
        array = np.asarray(getattr(pydb, name), dtype=np.float64).copy()
        if array.shape != (len(endpoints),) or not np.isfinite(array).all() or np.any(array < 0):
            raise RuntimeError("ECC min output loads do not align with endpoint identities")
        array.setflags(write=False)
        output_caps.append(array)
    return HoldTimingData(
        valid,
        reason,
        check_valid,
        check_reason,
        start_valid,
        arrivals,
        slews,
        contexts,
        tuple(launches),
        tuple(pin_caps),
        check_types.copy(),
        tuple(output_caps),
        latch_data,
        sparse_view,
    )
