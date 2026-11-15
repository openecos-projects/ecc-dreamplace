"""Star timing-driven placement (TDP) operators ported from AutoDMP R0_reference."""

from .clock_weight import (
    capture_setup_clock_pair_coverage,
    launch_capture_clock_pair_adjoint,
)
from .delay_adjoint import pair_delay_adjoint_from_pin_grads
from .rc_calibration import (
    ClockRCCalibrationState,
    StarRCCalibrationState,
    XPLACE_CALIBRATION_POLICY,
    calibration_cap_digest,
    calibration_policy_digest,
    hpwl_wirelength_by_net,
    next_lower_overflow_milestone,
    pair_wirelength_by_net,
    tree_wirelength_by_net,
)
from .rowsum_curvature import (
    InputAccumulator,
    apply_satisfied_slack_release,
    clock_rowsum_hessian,
    elmore_rowsum_diagonal,
    linear_length_diagonal,
    quadratic_rowsum_diagonal,
    rowsum_curvature_load,
    rowsum_hessian,
    timing_transport_factor,
    transport_momentum,
)
from .slope_calibration import (
    SlopeCalibration,
    clamp_load_length_weight,
    homothety_teacher_slopes,
    net_roots,
    net_scale,
)
from .slack_beta import (
    compute_delay_sigmoid_retention_beta,
    compute_slack_retention_beta,
    estimate_slack_drift_tau,
    soft_slack_severity,
    summarize_drift_tau_history,
)
from .star_pin_pair import StarPinPair
from .weight_accumulate import (
    accumulate_efftdp_pair_weights,
    momentum_pair_weights,
)

__all__ = [
    "ClockRCCalibrationState",
    "InputAccumulator",
    "SlopeCalibration",
    "StarPinPair",
    "StarRCCalibrationState",
    "XPLACE_CALIBRATION_POLICY",
    "accumulate_efftdp_pair_weights",
    "apply_satisfied_slack_release",
    "calibration_cap_digest",
    "calibration_policy_digest",
    "capture_setup_clock_pair_coverage",
    "clamp_load_length_weight",
    "clock_rowsum_hessian",
    "elmore_rowsum_diagonal",
    "homothety_teacher_slopes",
    "hpwl_wirelength_by_net",
    "launch_capture_clock_pair_adjoint",
    "linear_length_diagonal",
    "momentum_pair_weights",
    "net_roots",
    "net_scale",
    "next_lower_overflow_milestone",
    "pair_delay_adjoint_from_pin_grads",
    "pair_wirelength_by_net",
    "quadratic_rowsum_diagonal",
    "rowsum_curvature_load",
    "rowsum_hessian",
    "timing_transport_factor",
    "transport_momentum",
    "tree_wirelength_by_net",
]
