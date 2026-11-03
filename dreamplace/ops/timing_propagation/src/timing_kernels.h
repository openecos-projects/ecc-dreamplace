#pragma once

#include "utility/src/torch.h"

#include <vector>

namespace dreamplace {

// Differentiable recovery constraints for SDC capture deadlines. ATen owns
// autograd; no backend STA reference slew or checked RAT is required.
std::vector<at::Tensor> recovery_rat_from_capture_deadline(
    at::Tensor rise_required, at::Tensor fall_required,
    at::Tensor endpoint_pins, at::Tensor rise_deadlines, at::Tensor fall_deadlines,
    at::Tensor recovery_pins, at::Tensor rise_checks, at::Tensor fall_checks,
    at::Tensor recovery_valid, at::Tensor setup_pins, at::Tensor setup_valid,
    bool smooth, double tau);

// CPU hard-mode level kernels used by the incremental propagation path.  The
// tensors are ordinary contiguous CPU views; the functions deliberately do
// not participate in autograd.  The existing Python TP and differentiable
// native operators remain unchanged.
std::vector<at::Tensor> cell_aat_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rAAT,
    at::Tensor pin_fAAT,
    at::Tensor pin_rtran,
    at::Tensor pin_ftran,
    at::Tensor pin_net_cap_rise,
    at::Tensor pin_net_cap_fall,
    std::vector<at::Tensor> tables,
    bool smooth_pairwise,
    double alpha);

std::vector<at::Tensor> net_aat_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rAAT,
    at::Tensor pin_fAAT,
    at::Tensor pin_rtran,
    at::Tensor pin_ftran,
    at::Tensor pin_net_delay_rise,
    at::Tensor pin_net_delay_fall,
    at::Tensor pin_net_impulse_rise,
    at::Tensor pin_net_impulse_fall,
    double epsilon);

std::vector<at::Tensor> cell_rat_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rRAT,
    at::Tensor pin_fRAT,
    at::Tensor cell_arc_rr_delays,
    at::Tensor cell_arc_fr_delays,
    at::Tensor cell_arc_rf_delays,
    at::Tensor cell_arc_ff_delays);

std::vector<at::Tensor> net_rat_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rRAT,
    at::Tensor pin_fRAT,
    at::Tensor pin_net_delay_rise,
    at::Tensor pin_net_delay_fall);

std::vector<at::Tensor> setup_rat_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rRAT,
    at::Tensor pin_fRAT,
    at::Tensor clk_pin_rtran,
    at::Tensor clk_pin_ftran,
    at::Tensor pin_rtran,
    at::Tensor pin_ftran,
    std::vector<at::Tensor> tables,
    at::Tensor valid,
    bool smooth_pairwise,
    double alpha);

std::vector<at::Tensor> recovery_rat_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rRAT,
    at::Tensor pin_fRAT,
    at::Tensor pin_rtran,
    at::Tensor pin_ftran,
    at::Tensor clock_rtran,
    at::Tensor clock_ftran,
    at::Tensor check_data_rtran,
    at::Tensor check_data_ftran,
    std::vector<at::Tensor> tables,
    at::Tensor valid);

std::vector<at::Tensor> clk2q_hard_level(
    at::Tensor arcs,
    at::Tensor pin_rAAT,
    at::Tensor pin_fAAT,
    at::Tensor pin_rtran,
    at::Tensor pin_ftran,
    at::Tensor pin_net_cap_rise,
    at::Tensor pin_net_cap_fall,
    at::Tensor clock_rtran,
    at::Tensor clock_ftran,
    std::vector<at::Tensor> tables,
    bool smooth_pairwise,
    double alpha);

}  // namespace dreamplace
