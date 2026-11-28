#include "timing_kernels.h"
#include "timing_hard_math.h"

#include "cell_aat_math.h"
#include "net_aat_math.h"
#include "timing_cpu_checks.h"
#include "utility/src/cpu_tensor.h"
#include "utility/src/torch.h"

#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace dreamplace {

std::vector<at::Tensor> recovery_rat_from_capture_deadline(
    at::Tensor rise_required, at::Tensor fall_required,
    at::Tensor endpoint_pins, at::Tensor rise_deadlines, at::Tensor fall_deadlines,
    at::Tensor recovery_pins, at::Tensor rise_checks, at::Tensor fall_checks,
    at::Tensor recovery_valid, at::Tensor setup_pins, at::Tensor setup_valid,
    bool smooth, double tau) {
  const auto pins = rise_required.numel();
  TORCH_CHECK(rise_required.dim() == 1 && rise_required.is_floating_point(),
              "recovery required times must be floating vectors");
  for (const auto& value : {fall_required, rise_deadlines, fall_deadlines,
                            rise_checks, fall_checks}) {
    TORCH_CHECK(value.dim() == 1 && value.device() == rise_required.device()
                    && value.scalar_type() == rise_required.scalar_type(),
                "recovery deadlines/checks must match required-time dtype and device");
  }
  TORCH_CHECK(fall_required.numel() == pins && rise_deadlines.numel() == endpoint_pins.numel()
                  && fall_deadlines.numel() == endpoint_pins.numel()
                  && rise_checks.numel() == recovery_pins.numel()
                  && fall_checks.numel() == recovery_pins.numel(),
              "recovery deadline/check vector sizes do not match pin domains");
  for (const auto& index : {endpoint_pins, recovery_pins, setup_pins}) {
    TORCH_CHECK(index.dim() == 1 && index.scalar_type() == at::kLong
                    && index.device() == rise_required.device(),
                "recovery endpoint indices must be long vectors on the timing device");
    TORCH_CHECK(index.numel() == 0 || (index.min().item<int64_t>() >= 0
                    && index.max().item<int64_t>() < pins),
                "recovery endpoint index outside the timing pin domain");
  }
  for (const auto& item : {std::make_pair(recovery_valid, recovery_pins.numel()),
                           std::make_pair(setup_valid, setup_pins.numel())}) {
    TORCH_CHECK(item.first.scalar_type() == at::kBool
                    && item.first.device() == rise_required.device()
                    && item.first.dim() == 2 && item.first.size(0) == item.second
                    && item.first.size(1) == 2,
                "recovery/setup validity must have shape [rows, 2]");
  }
  TORCH_CHECK(!smooth || (std::isfinite(tau) && tau > 0),
              "smooth recovery constraints require a positive finite tau");
  const auto counts = at::zeros_like(rise_required, rise_required.options().dtype(at::kLong));
  const auto seeds = counts.scatter_add(0, endpoint_pins, at::ones_like(endpoint_pins));
  TORCH_CHECK(recovery_pins.numel() == 0
                  || seeds.index_select(0, recovery_pins).min().item<int64_t>() > 0,
              "recovery pin has no exported capture deadline");
  std::vector<at::Tensor> result;
  for (int edge = 0; edge < 2; ++edge) {
    const auto& base = edge == 0 ? rise_required : fall_required;
    const auto& deadline = edge == 0 ? rise_deadlines : fall_deadlines;
    const auto& check = edge == 0 ? rise_checks : fall_checks;
    const auto selected = at::nonzero(recovery_valid.select(1, edge)).reshape({-1});
    if (selected.numel() == 0) { result.push_back(base); continue; }
    const auto index = recovery_pins.index_select(0, selected);
    const auto capture = at::zeros_like(base).index_copy(0, endpoint_pins, deadline);
    const auto values = capture.index_select(0, index) - check.index_select(0, selected);
    const auto touched = counts.scatter_add(0, index, at::ones_like(index)) > 0;
    const auto shift = at::full_like(base, std::numeric_limits<double>::infinity())
                           .scatter_reduce(0, index, values, "amin", false).detach();
    at::Tensor reduced;
    if (smooth) {
      const auto exponent = at::exp((shift.index_select(0, index) - values) / tau);
      const auto sum = at::zeros_like(base).scatter_add(0, index, exponent);
      reduced = shift - tau * at::log(sum.clamp_min(1.0e-30));
    } else {
      reduced = at::full_like(base, std::numeric_limits<double>::infinity())
                    .scatter_reduce(0, index, values, "amin", false);
    }
    reduced = at::where(touched, reduced, base);
    const auto setup_count = counts.scatter_add(
        0, setup_pins, setup_valid.select(1, edge).to(at::kLong));
    const auto combined = smooth ? -tau * at::logaddexp(-base / tau, -reduced / tau)
                                 : at::minimum(base, reduced);
    result.push_back(at::where(touched & (setup_count > 0), combined, reduced));
  }
  return result;
}

namespace {

void requireCpuTensor(const at::Tensor& value, const at::Tensor& reference,
                      at::ScalarType dtype, const char* name) {
  TORCH_CHECK(value.device().is_cpu() && value.is_contiguous(), name,
              " must be a contiguous CPU tensor");
  TORCH_CHECK(value.scalar_type() == dtype, name, " dtype mismatch");
  TORCH_CHECK(value.device() == reference.device(), name, " device mismatch");
}

void requirePinVector(const at::Tensor& value, const at::Tensor& reference,
                      const char* name) {
  requireCpuTensor(value, reference, reference.scalar_type(), name);
  TORCH_CHECK(value.dim() == 1 && value.sizes() == reference.sizes(), name,
              " must have the same one-dimensional shape as the pin state");
}

template <typename scalar_t>
TimingLutTable<scalar_t> makeTable(const std::vector<at::Tensor>& tables,
                                   int offset) {
  return {
      tables[offset].data_ptr<scalar_t>(),
      tables[offset + 1].data_ptr<scalar_t>(),
      tables[offset + 2].data_ptr<scalar_t>(),
      tables[offset + 3].data_ptr<int64_t>(),
      tables[offset + 4].data_ptr<int64_t>(),
      static_cast<int>(tables[offset].size(0)),
      static_cast<int>(tables[offset].size(1)),
      static_cast<int>(tables[offset + 1].size(1)),
      static_cast<int>(tables[offset + 2].size(1)),
  };
}


template <typename scalar_t>
void aggregateCellUpdates(const at::Tensor& arcs, const at::Tensor& updates,
                          at::Tensor& rise_arrival, at::Tensor& fall_arrival,
                          at::Tensor& rise_slew, at::Tensor& fall_slew) {
  timing::aggregateCellRows<scalar_t>(
      arcs.data_ptr<int64_t>(), updates.data_ptr<scalar_t>(),
      arcs.size(0), arcs.size(1), rise_arrival.data_ptr<scalar_t>(),
      fall_arrival.data_ptr<scalar_t>(), rise_slew.data_ptr<scalar_t>(),
      fall_slew.data_ptr<scalar_t>());
}

template <typename scalar_t>
void aggregateNetUpdates(const at::Tensor& arcs, const at::Tensor& updates,
                         at::Tensor& rise_arrival, at::Tensor& fall_arrival,
                         at::Tensor& rise_slew, at::Tensor& fall_slew) {
  timing::aggregateNetRows<scalar_t>(
      arcs.data_ptr<int64_t>(), updates.data_ptr<scalar_t>(),
      arcs.size(0), arcs.size(1), rise_arrival.data_ptr<scalar_t>(),
      fall_arrival.data_ptr<scalar_t>(), rise_slew.data_ptr<scalar_t>(),
      fall_slew.data_ptr<scalar_t>());
}

template <typename scalar_t>
void launchCellHardLevel(const at::Tensor& arcs, const at::Tensor& pin_rAAT,
                         const at::Tensor& pin_fAAT, const at::Tensor& pin_rtran,
                         const at::Tensor& pin_ftran,
                         const at::Tensor& pin_net_cap_rise,
                         const at::Tensor& pin_net_cap_fall,
                         const std::vector<at::Tensor>& tables,
                         at::Tensor& updates, at::Tensor& derivatives,
                         at::Tensor& rise_arrival, at::Tensor& fall_arrival,
                         at::Tensor& rise_slew, at::Tensor& fall_slew,
                         bool smooth_pairwise, double alpha) {
  const auto delay_rise = makeTable<scalar_t>(tables, 0);
  const auto delay_fall = makeTable<scalar_t>(tables, 5);
  const auto tran_rise = makeTable<scalar_t>(tables, 10);
  const auto tran_fall = makeTable<scalar_t>(tables, 15);
  const auto* arc_ptr = arcs.data_ptr<int64_t>();
  const int count = static_cast<int>(arcs.size(0));
  const int pins = static_cast<int>(pin_rAAT.numel());
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int index = 0; index < count; ++index) {
    cellForwardRow<scalar_t>(
        index, arc_ptr, static_cast<int>(arcs.size(1)), count, pins,
        pin_rAAT.data_ptr<scalar_t>(), pin_fAAT.data_ptr<scalar_t>(),
        pin_rtran.data_ptr<scalar_t>(), pin_ftran.data_ptr<scalar_t>(),
        pin_net_cap_rise.data_ptr<scalar_t>(), pin_net_cap_fall.data_ptr<scalar_t>(),
        delay_rise, delay_fall, tran_rise, tran_fall,
        updates.data_ptr<scalar_t>(), derivatives.data_ptr<scalar_t>(),
        smooth_pairwise, scalar_t(alpha));
  }
  aggregateCellUpdates<scalar_t>(arcs, updates, rise_arrival, fall_arrival,
                                 rise_slew, fall_slew);
}

template <typename scalar_t>
void launchNetHardLevel(const at::Tensor& arcs, const at::Tensor& pin_rAAT,
                        const at::Tensor& pin_fAAT, const at::Tensor& pin_rtran,
                        const at::Tensor& pin_ftran,
                        const at::Tensor& pin_net_delay_rise,
                        const at::Tensor& pin_net_delay_fall,
                        const at::Tensor& pin_net_impulse_rise,
                        const at::Tensor& pin_net_impulse_fall, double epsilon,
                        at::Tensor& updates, at::Tensor& jacobian,
                        at::Tensor& rise_arrival, at::Tensor& fall_arrival,
                        at::Tensor& rise_slew, at::Tensor& fall_slew) {
  const auto* arc_ptr = arcs.data_ptr<int64_t>();
  const int count = static_cast<int>(arcs.size(0));
  const int pins = static_cast<int>(pin_rAAT.numel());
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int index = 0; index < count; ++index) {
    netForwardRow<scalar_t>(
        index, arc_ptr, static_cast<int>(arcs.size(1)), count, pins,
        pin_rAAT.data_ptr<scalar_t>(), pin_fAAT.data_ptr<scalar_t>(),
        pin_rtran.data_ptr<scalar_t>(), pin_ftran.data_ptr<scalar_t>(),
        pin_net_delay_rise.data_ptr<scalar_t>(), pin_net_delay_fall.data_ptr<scalar_t>(),
        pin_net_impulse_rise.data_ptr<scalar_t>(), pin_net_impulse_fall.data_ptr<scalar_t>(),
        updates.data_ptr<scalar_t>(), jacobian.data_ptr<scalar_t>(), scalar_t(epsilon));
  }
  aggregateNetUpdates<scalar_t>(arcs, updates, rise_arrival, fall_arrival,
                                rise_slew, fall_slew);
}


template <typename scalar_t>
void launchCellRatHardLevel(const at::Tensor& arcs, const at::Tensor& pin_rRAT,
                            const at::Tensor& pin_fRAT,
                            const at::Tensor& cell_arc_rr_delays,
                            const at::Tensor& cell_arc_fr_delays,
                            const at::Tensor& cell_arc_rf_delays,
                            const at::Tensor& cell_arc_ff_delays,
                            at::Tensor& rise_required,
                            at::Tensor& fall_required) {
  timing::cellRatRows<scalar_t>(
      arcs.data_ptr<int64_t>(), arcs.size(0), arcs.size(1),
      pin_rRAT.data_ptr<scalar_t>(), pin_fRAT.data_ptr<scalar_t>(),
      cell_arc_rr_delays.data_ptr<scalar_t>(), cell_arc_fr_delays.data_ptr<scalar_t>(),
      cell_arc_rf_delays.data_ptr<scalar_t>(), cell_arc_ff_delays.data_ptr<scalar_t>(),
      rise_required.data_ptr<scalar_t>(), fall_required.data_ptr<scalar_t>());
}

template <typename scalar_t>
void launchNetRatHardLevel(const at::Tensor& arcs, const at::Tensor& pin_rRAT,
                           const at::Tensor& pin_fRAT,
                           const at::Tensor& pin_net_delay_rise,
                           const at::Tensor& pin_net_delay_fall,
                           at::Tensor& rise_required,
                           at::Tensor& fall_required) {
  timing::netRatRows<scalar_t>(
      arcs.data_ptr<int64_t>(), arcs.size(0), arcs.size(1),
      pin_rRAT.data_ptr<scalar_t>(), pin_fRAT.data_ptr<scalar_t>(),
      pin_net_delay_rise.data_ptr<scalar_t>(), pin_net_delay_fall.data_ptr<scalar_t>(),
      rise_required.data_ptr<scalar_t>(), fall_required.data_ptr<scalar_t>());
}

template <typename scalar_t>
void launchSetupRatHardLevel(
    const at::Tensor& arcs, const at::Tensor& pin_rRAT,
    const at::Tensor& pin_fRAT, const at::Tensor& clk_pin_rtran,
    const at::Tensor& clk_pin_ftran, const at::Tensor& pin_rtran,
    const at::Tensor& pin_ftran, const std::vector<at::Tensor>& tables,
    const at::Tensor& valid, bool smooth_pairwise, double alpha,
    at::Tensor& rise_required, at::Tensor& fall_required) {
  timing::setupRatRows<scalar_t>(
      arcs.data_ptr<int64_t>(), arcs.size(0), arcs.size(1),
      pin_rRAT.data_ptr<scalar_t>(), pin_fRAT.data_ptr<scalar_t>(),
      clk_pin_rtran.data_ptr<scalar_t>(), clk_pin_ftran.data_ptr<scalar_t>(),
      pin_rtran.data_ptr<scalar_t>(), pin_ftran.data_ptr<scalar_t>(),
      makeTable<scalar_t>(tables, 0), makeTable<scalar_t>(tables, 5),
      valid.defined() && valid.numel() != 0 ? valid.data_ptr<bool>() : nullptr,
      smooth_pairwise, alpha,
      rise_required.data_ptr<scalar_t>(), fall_required.data_ptr<scalar_t>());
}

template <typename scalar_t>
void launchRecoveryRatHardLevel(
    const at::Tensor& arcs, const at::Tensor& pin_rRAT,
    const at::Tensor& pin_fRAT, const at::Tensor& pin_rtran,
    const at::Tensor& pin_ftran, const at::Tensor& clock_rtran,
    const at::Tensor& clock_ftran, const at::Tensor& check_data_rtran,
    const at::Tensor& check_data_ftran, const std::vector<at::Tensor>& tables,
    const at::Tensor& valid, at::Tensor& rise_required,
    at::Tensor& fall_required) {
  timing::recoveryRatRows<scalar_t>(
      arcs.data_ptr<int64_t>(), arcs.size(0), arcs.size(1), pin_rRAT.numel(),
      pin_rRAT.data_ptr<scalar_t>(), pin_fRAT.data_ptr<scalar_t>(),
      pin_rtran.data_ptr<scalar_t>(), pin_ftran.data_ptr<scalar_t>(),
      clock_rtran.data_ptr<scalar_t>(), clock_ftran.data_ptr<scalar_t>(),
      check_data_rtran.data_ptr<scalar_t>(), check_data_ftran.data_ptr<scalar_t>(),
      makeTable<scalar_t>(tables, 0), makeTable<scalar_t>(tables, 5),
      valid.data_ptr<bool>(), rise_required.data_ptr<scalar_t>(),
      fall_required.data_ptr<scalar_t>());
}

template <typename scalar_t>
void launchClk2qHardLevel(
    const at::Tensor& arcs, const at::Tensor& pin_rAAT,
    const at::Tensor& pin_fAAT, const at::Tensor& pin_rtran,
    const at::Tensor& pin_ftran, const at::Tensor& pin_net_cap_rise,
    const at::Tensor& pin_net_cap_fall, const at::Tensor& clock_rtran,
    const at::Tensor& clock_ftran, const std::vector<at::Tensor>& tables,
    bool smooth_pairwise, double alpha, at::Tensor& rise_arrival,
    at::Tensor& fall_arrival, at::Tensor& rise_slew, at::Tensor& fall_slew) {
  timing::clk2qRows<scalar_t>(
      arcs.data_ptr<int64_t>(), arcs.size(0), arcs.size(1), pin_rAAT.numel(),
      pin_net_cap_rise.data_ptr<scalar_t>(), pin_net_cap_fall.data_ptr<scalar_t>(),
      clock_rtran.data_ptr<scalar_t>(), clock_ftran.data_ptr<scalar_t>(),
      makeTable<scalar_t>(tables, 0), makeTable<scalar_t>(tables, 5),
      makeTable<scalar_t>(tables, 10), makeTable<scalar_t>(tables, 15),
      smooth_pairwise, alpha, rise_arrival.data_ptr<scalar_t>(),
      fall_arrival.data_ptr<scalar_t>(), rise_slew.data_ptr<scalar_t>(),
      fall_slew.data_ptr<scalar_t>());
}

}  // namespace

std::vector<at::Tensor> cell_aat_hard_level(
    at::Tensor arcs, at::Tensor pin_rAAT, at::Tensor pin_fAAT,
    at::Tensor pin_rtran, at::Tensor pin_ftran, at::Tensor pin_net_cap_rise,
    at::Tensor pin_net_cap_fall, std::vector<at::Tensor> tables,
    bool smooth_pairwise, double alpha) {
  requireCpuTensor(pin_rAAT, pin_rAAT, pin_rAAT.scalar_type(), "pin_rAAT");
  requireCpuTensor(arcs, pin_rAAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5,
              "cell AAT arcs need at least five columns");
  for (const auto& value : {pin_fAAT, pin_rtran, pin_ftran,
                            pin_net_cap_rise, pin_net_cap_fall}) {
    requirePinVector(value, pin_rAAT, "cell AAT pin state");
  }
  TORCH_CHECK(tables.size() == 20, "cell AAT expects four groups of five LUT tensors");
  for (int offset = 0; offset < 20; offset += 5) {
    for (int index = 0; index < 5; ++index) {
      const auto& value = tables[offset + index];
      const auto expected = index < 3 ? pin_rAAT.scalar_type() : at::kLong;
      requireCpuTensor(value, pin_rAAT, expected, "cell AAT LUT");
      TORCH_CHECK(value.dim() == (index < 3 ? 2 : 1),
                  "cell AAT LUT tensor rank mismatch");
    }
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  checkCpuArcIndices(arcs, pin_rAAT.numel());

  auto rise_arrival = pin_rAAT.contiguous().clone();
  auto fall_arrival = pin_fAAT.contiguous().clone();
  auto rise_slew = pin_rtran.contiguous().clone();
  auto fall_slew = pin_ftran.contiguous().clone();
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 8}, pin_rAAT.options());
  auto derivatives = fullyWrittenCpuTensor({arcs.size(0), 48}, pin_rAAT.options());

  AT_DISPATCH_FLOATING_TYPES(pin_rAAT.scalar_type(), "cellAatHardLevel", [&] {
    launchCellHardLevel<scalar_t>(
        arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, pin_net_cap_rise,
        pin_net_cap_fall, tables, updates, derivatives, rise_arrival,
        fall_arrival, rise_slew, fall_slew, smooth_pairwise, alpha);
  });

  return {rise_arrival, fall_arrival, rise_slew, fall_slew,
          updates.select(1, 4).contiguous(), updates.select(1, 5).contiguous(),
          updates.select(1, 6).contiguous(), updates.select(1, 7).contiguous()};
}

std::vector<at::Tensor> net_aat_hard_level(
    at::Tensor arcs, at::Tensor pin_rAAT, at::Tensor pin_fAAT,
    at::Tensor pin_rtran, at::Tensor pin_ftran, at::Tensor pin_net_delay_rise,
    at::Tensor pin_net_delay_fall, at::Tensor pin_net_impulse_rise,
    at::Tensor pin_net_impulse_fall, double epsilon) {
  requireCpuTensor(pin_rAAT, pin_rAAT, pin_rAAT.scalar_type(), "pin_rAAT");
  requireCpuTensor(arcs, pin_rAAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 2,
              "net AAT arcs need at least two columns");
  for (const auto& value : {pin_fAAT, pin_rtran, pin_ftran,
                            pin_net_delay_rise, pin_net_delay_fall,
                            pin_net_impulse_rise, pin_net_impulse_fall}) {
    requirePinVector(value, pin_rAAT, "net AAT pin state");
  }
  TORCH_CHECK(epsilon > 0.0, "net AAT epsilon must be positive");
  checkCpuArcIndices(arcs, pin_rAAT.numel());

  auto rise_arrival = pin_rAAT.contiguous().clone();
  auto fall_arrival = pin_fAAT.contiguous().clone();
  auto rise_slew = pin_rtran.contiguous().clone();
  auto fall_slew = pin_ftran.contiguous().clone();
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 4}, pin_rAAT.options());
  auto jacobian = fullyWrittenCpuTensor({arcs.size(0), 6}, pin_rAAT.options());

  AT_DISPATCH_FLOATING_TYPES(pin_rAAT.scalar_type(), "netAatHardLevel", [&] {
    launchNetHardLevel<scalar_t>(
        arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, pin_net_delay_rise,
        pin_net_delay_fall, pin_net_impulse_rise, pin_net_impulse_fall,
        epsilon, updates, jacobian, rise_arrival, fall_arrival, rise_slew,
        fall_slew);
  });

  return {rise_arrival, fall_arrival, rise_slew, fall_slew};
}

std::vector<at::Tensor> cell_rat_hard_level(
    at::Tensor arcs, at::Tensor pin_rRAT, at::Tensor pin_fRAT,
    at::Tensor cell_arc_rr_delays, at::Tensor cell_arc_fr_delays,
    at::Tensor cell_arc_rf_delays, at::Tensor cell_arc_ff_delays) {
  requireCpuTensor(pin_rRAT, pin_rRAT, pin_rRAT.scalar_type(), "pin_rRAT");
  requireCpuTensor(arcs, pin_rRAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5,
              "cell RAT arcs need at least five columns");
  for (const auto& value : {pin_fRAT, cell_arc_rr_delays,
                            cell_arc_fr_delays, cell_arc_rf_delays,
                            cell_arc_ff_delays}) {
    requireCpuTensor(value, pin_rRAT, pin_rRAT.scalar_type(), "cell RAT state");
  }
  TORCH_CHECK(pin_fRAT.dim() == 1 && pin_fRAT.sizes() == pin_rRAT.sizes(),
              "cell RAT pin state shape mismatch");
  for (const auto& value : {cell_arc_rr_delays, cell_arc_fr_delays,
                            cell_arc_rf_delays, cell_arc_ff_delays}) {
    TORCH_CHECK(value.dim() == 1 && value.size(0) == arcs.size(0),
                "cell RAT delay shape mismatch");
  }
  checkCpuArcIndices(arcs, pin_rRAT.numel());
  auto rise_required = pin_rRAT.contiguous().clone();
  auto fall_required = pin_fRAT.contiguous().clone();
  AT_DISPATCH_FLOATING_TYPES(pin_rRAT.scalar_type(), "cellRatHardLevel", [&] {
    launchCellRatHardLevel<scalar_t>(
        arcs, pin_rRAT, pin_fRAT, cell_arc_rr_delays, cell_arc_fr_delays,
        cell_arc_rf_delays, cell_arc_ff_delays, rise_required, fall_required);
  });
  return {rise_required, fall_required};
}

std::vector<at::Tensor> net_rat_hard_level(
    at::Tensor arcs, at::Tensor pin_rRAT, at::Tensor pin_fRAT,
    at::Tensor pin_net_delay_rise, at::Tensor pin_net_delay_fall) {
  requireCpuTensor(pin_rRAT, pin_rRAT, pin_rRAT.scalar_type(), "pin_rRAT");
  requireCpuTensor(arcs, pin_rRAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 2,
              "net RAT arcs need at least two columns");
  for (const auto& value : {pin_fRAT, pin_net_delay_rise,
                            pin_net_delay_fall}) {
    requirePinVector(value, pin_rRAT, "net RAT state");
  }
  checkCpuArcIndices(arcs, pin_rRAT.numel());
  auto rise_required = pin_rRAT.contiguous().clone();
  auto fall_required = pin_fRAT.contiguous().clone();
  AT_DISPATCH_FLOATING_TYPES(pin_rRAT.scalar_type(), "netRatHardLevel", [&] {
    launchNetRatHardLevel<scalar_t>(
        arcs, pin_rRAT, pin_fRAT, pin_net_delay_rise, pin_net_delay_fall,
        rise_required, fall_required);
  });
  return {rise_required, fall_required};
}

std::vector<at::Tensor> setup_rat_hard_level(
    at::Tensor arcs, at::Tensor pin_rRAT, at::Tensor pin_fRAT,
    at::Tensor clk_pin_rtran, at::Tensor clk_pin_ftran, at::Tensor pin_rtran,
    at::Tensor pin_ftran, std::vector<at::Tensor> tables, at::Tensor valid,
    bool smooth_pairwise, double alpha) {
  requireCpuTensor(pin_rRAT, pin_rRAT, pin_rRAT.scalar_type(), "pin_rRAT");
  requireCpuTensor(arcs, pin_rRAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5,
              "setup RAT arcs need at least five columns");
  for (const auto& value : {pin_fRAT, pin_rtran, pin_ftran}) {
    requirePinVector(value, pin_rRAT, "setup RAT state");
  }
  requireCpuTensor(clk_pin_rtran, pin_rRAT, pin_rRAT.scalar_type(),
                   "setup RAT clock rise slew");
  requireCpuTensor(clk_pin_ftran, pin_rRAT, pin_rRAT.scalar_type(),
                   "setup RAT clock fall slew");
  TORCH_CHECK(clk_pin_rtran.dim() == 1 && clk_pin_ftran.dim() == 1
                  && clk_pin_rtran.numel() == clk_pin_ftran.numel(),
              "setup RAT clock slew shape mismatch");
  TORCH_CHECK(tables.size() == 10,
              "setup RAT expects rise/fall setup LUT groups");
  for (int offset = 0; offset < 10; offset += 5) {
    for (int index = 0; index < 5; ++index) {
      const auto& value = tables[offset + index];
      const auto expected = index < 3 ? pin_rRAT.scalar_type() : at::kLong;
      requireCpuTensor(value, pin_rRAT, expected, "setup RAT LUT");
      TORCH_CHECK(value.dim() == (index < 3 ? 2 : 1),
                  "setup RAT LUT tensor rank mismatch");
    }
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  if (valid.defined() && valid.numel() != 0) {
    requireCpuTensor(valid, pin_rRAT, at::kBool, "setup RAT validity");
    TORCH_CHECK(valid.dim() == 2 && valid.size(0) == arcs.size(0)
                    && valid.size(1) == 2,
                "setup RAT validity shape mismatch");
  }
  const auto* arc_ptr = arcs.data_ptr<int64_t>();
  for (int64_t index = 0; index < arcs.size(0); ++index) {
    const auto* arc = arc_ptr + index * arcs.size(1);
    TORCH_CHECK(arc[0] >= 0 && arc[0] < clk_pin_rtran.numel(),
                "setup RAT clock pin index out of range");
    TORCH_CHECK(arc[1] >= 0 && arc[1] < pin_rRAT.numel(),
                "setup RAT data pin index out of range");
  }
  auto rise_required = pin_rRAT.contiguous().clone();
  auto fall_required = pin_fRAT.contiguous().clone();
  AT_DISPATCH_FLOATING_TYPES(pin_rRAT.scalar_type(), "setupRatHardLevel", [&] {
    launchSetupRatHardLevel<scalar_t>(
        arcs, pin_rRAT, pin_fRAT, clk_pin_rtran, clk_pin_ftran, pin_rtran,
        pin_ftran, tables, valid, smooth_pairwise, alpha, rise_required,
        fall_required);
  });
  return {rise_required, fall_required};
}

std::vector<at::Tensor> recovery_rat_hard_level(
    at::Tensor arcs, at::Tensor pin_rRAT, at::Tensor pin_fRAT,
    at::Tensor pin_rtran, at::Tensor pin_ftran, at::Tensor clock_rtran,
    at::Tensor clock_ftran, at::Tensor check_data_rtran,
    at::Tensor check_data_ftran, std::vector<at::Tensor> tables,
    at::Tensor valid) {
  requireCpuTensor(pin_rRAT, pin_rRAT, pin_rRAT.scalar_type(), "pin_rRAT");
  requireCpuTensor(arcs, pin_rRAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 7,
              "recovery RAT arcs need at least seven columns");
  for (const auto& value : {pin_fRAT, pin_rtran, pin_ftran}) {
    requirePinVector(value, pin_rRAT, "recovery RAT state");
  }
  for (const auto& value : {clock_rtran, clock_ftran}) {
    requireCpuTensor(value, pin_rRAT, pin_rRAT.scalar_type(),
                     "recovery RAT clock slew");
    TORCH_CHECK(value.dim() == 1, "recovery RAT clock slew must be one-dimensional");
  }
  for (const auto& value : {check_data_rtran, check_data_ftran}) {
    requireCpuTensor(value, pin_rRAT, pin_rRAT.scalar_type(),
                     "recovery RAT check data slew");
    TORCH_CHECK(value.dim() == 1 && value.size(0) == arcs.size(0),
                "recovery RAT check data slew shape mismatch");
  }
  TORCH_CHECK(tables.size() == 10,
              "recovery RAT expects rise/fall check LUT groups");
  for (int offset = 0; offset < 10; offset += 5) {
    for (int index = 0; index < 5; ++index) {
      const auto& value = tables[offset + index];
      const auto expected = index < 3 ? pin_rRAT.scalar_type() : at::kLong;
      requireCpuTensor(value, pin_rRAT, expected, "recovery RAT LUT");
      TORCH_CHECK(value.dim() == (index < 3 ? 2 : 1),
                  "recovery RAT LUT tensor rank mismatch");
    }
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  requireCpuTensor(valid, pin_rRAT, at::kBool, "recovery RAT validity");
  TORCH_CHECK(valid.dim() == 2 && valid.size(0) == arcs.size(0)
                  && valid.size(1) == 2,
              "recovery RAT validity shape mismatch");
  const auto* arc_ptr = arcs.data_ptr<int64_t>();
  for (int64_t index = 0; index < arcs.size(0); ++index) {
    const auto* arc = arc_ptr + index * arcs.size(1);
    TORCH_CHECK(arc[0] >= 0 && arc[0] < clock_rtran.numel(),
                "recovery RAT clock pin index out of range");
    TORCH_CHECK(arc[1] >= 0 && arc[1] < pin_rRAT.numel(),
                "recovery RAT endpoint pin index out of range");
  }
  auto rise_required = pin_rRAT.contiguous().clone();
  auto fall_required = pin_fRAT.contiguous().clone();
  AT_DISPATCH_FLOATING_TYPES(pin_rRAT.scalar_type(), "recoveryRatHardLevel", [&] {
    launchRecoveryRatHardLevel<scalar_t>(
        arcs, pin_rRAT, pin_fRAT, pin_rtran, pin_ftran, clock_rtran,
        clock_ftran, check_data_rtran, check_data_ftran, tables, valid,
        rise_required, fall_required);
  });
  return {rise_required, fall_required};
}

std::vector<at::Tensor> clk2q_hard_level(
    at::Tensor arcs, at::Tensor pin_rAAT, at::Tensor pin_fAAT,
    at::Tensor pin_rtran, at::Tensor pin_ftran, at::Tensor pin_net_cap_rise,
    at::Tensor pin_net_cap_fall, at::Tensor clock_rtran,
    at::Tensor clock_ftran, std::vector<at::Tensor> tables,
    bool smooth_pairwise, double alpha) {
  requireCpuTensor(pin_rAAT, pin_rAAT, pin_rAAT.scalar_type(), "pin_rAAT");
  requireCpuTensor(arcs, pin_rAAT, at::kLong, "arcs");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 6,
              "clk2q arcs need at least six columns");
  for (const auto& value : {pin_fAAT, pin_rtran, pin_ftran,
                            pin_net_cap_rise, pin_net_cap_fall}) {
    requirePinVector(value, pin_rAAT, "clk2q pin state");
  }
  for (const auto& value : {clock_rtran, clock_ftran}) {
    requireCpuTensor(value, pin_rAAT, pin_rAAT.scalar_type(),
                     "clk2q clock slew");
    TORCH_CHECK(value.dim() == 1, "clk2q clock slew must be one-dimensional");
  }
  TORCH_CHECK(clock_rtran.numel() == clock_ftran.numel(),
              "clk2q clock slew shape mismatch");
  TORCH_CHECK(tables.size() == 20, "clk2q expects four cell LUT groups");
  for (int offset = 0; offset < 20; offset += 5) {
    for (int index = 0; index < 5; ++index) {
      const auto& value = tables[offset + index];
      const auto expected = index < 3 ? pin_rAAT.scalar_type() : at::kLong;
      requireCpuTensor(value, pin_rAAT, expected, "clk2q LUT");
      TORCH_CHECK(value.dim() == (index < 3 ? 2 : 1),
                  "clk2q LUT tensor rank mismatch");
    }
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  const auto* arc_ptr = arcs.data_ptr<int64_t>();
  for (int64_t index = 0; index < arcs.size(0); ++index) {
    const auto* arc = arc_ptr + index * arcs.size(1);
    TORCH_CHECK(arc[0] >= 0 && arc[0] < clock_rtran.numel(),
                "clk2q clock pin index out of range");
    TORCH_CHECK(arc[1] >= 0 && arc[1] < pin_rAAT.numel(),
                "clk2q output pin index out of range");
  }
  auto rise_arrival = pin_rAAT.contiguous().clone();
  auto fall_arrival = pin_fAAT.contiguous().clone();
  auto rise_slew = pin_rtran.contiguous().clone();
  auto fall_slew = pin_ftran.contiguous().clone();
  AT_DISPATCH_FLOATING_TYPES(pin_rAAT.scalar_type(), "clk2qHardLevel", [&] {
    launchClk2qHardLevel<scalar_t>(
        arcs, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, pin_net_cap_rise,
        pin_net_cap_fall, clock_rtran, clock_ftran, tables, smooth_pairwise,
        alpha, rise_arrival, fall_arrival, rise_slew, fall_slew);
  });
  return {rise_arrival, fall_arrival, rise_slew, fall_slew};
}

}  // namespace dreamplace
