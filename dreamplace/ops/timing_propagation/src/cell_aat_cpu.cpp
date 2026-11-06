#include "utility/src/cpu_tensor.h"
#include "timing_cpu_checks.h"
#include "utility/src/torch.h"
#include "lut_query.h"
#include "timing_scatter_groups.h"

#include "cell_aat_math.h"
#include <ATen/Parallel.h>
#include <omp.h>

template <typename scalar_t>
void cellAatForwardCpuLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* derivatives, bool smooth, scalar_t alpha, bool early) {
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int i = 0; i < count; ++i) {
    cellForwardRow<scalar_t>(i, arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft, updates, derivatives, smooth, alpha, early);
  }
}
template <typename scalar_t>
void cellAatValuesCpuLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, bool smooth, scalar_t alpha) {
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int i = 0; i < count; ++i) {
    cellValueRow<scalar_t>(i, arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft, updates, smooth, alpha);
  }
}

template <typename scalar_t>
void cellAatActionValuesCpuLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const int64_t* target_rows, const int64_t* target_senses,
    const bool* valid_actions, int action_count,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* deltas, bool* finite_actions,
    bool need_delays, bool smooth, scalar_t alpha) {
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int i = 0; i < count; ++i) {
    const int64_t* arc = arcs + i * columns;
    const int64_t in = arc[0], out = arc[1];
    scalar_t* current = updates + i * 8;
    if (need_delays) {
      cellValueRowAt<scalar_t>(in, out, arc[3], arc[4], pins,
          ra, fa, rs, fs, rc, fc, rd, fd, rt, ft,
          current, smooth, alpha, columns > 8 ? arc[8] : 15);
    } else {
      cellValueRow4At<scalar_t>(in, out, arc[3], arc[4], pins,
          ra, fa, rs, fs, rc, fc, rd, fd, rt, ft,
          current, smooth, alpha, columns > 8 ? arc[8] : 15);
      for (int k = 4; k < 8; ++k) current[k] = 0;
    }
    for (int action = 0; action < action_count; ++action) {
      const int64_t key = i * action_count + action;
      scalar_t* delta = deltas + key * 8;
      for (int k = 0; k < 8; ++k) delta[k] = 0;
      finite_actions[key] = false;
      if (!valid_actions[key]) continue;
      scalar_t target[4];
      cellValueRow4At<scalar_t>(in, out, target_rows[key], target_senses[key], pins,
          ra, fa, rs, fs, rc, fc, rd, fd, rt, ft,
          target, smooth, alpha, columns > 8 ? arc[8] : 15);
      bool finite = true;
      for (int k = 0; k < 4; ++k) {
        finite = finite && std::isfinite(target[k]) && std::isfinite(current[k]);
      }
      if (finite) {
        finite_actions[key] = true;
        for (int k = 0; k < 4; ++k) delta[k] = target[k] - current[k];
      }
    }
  }
}

template <typename scalar_t>
void cellAatActionBackwardCpuLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const int64_t* target_rows, const int64_t* target_senses,
    const bool* valid_actions, const int64_t* inst_ids, int action_count,
    const scalar_t* current, const scalar_t* gradient,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc,
    const scalar_t* cap_delta_r, const scalar_t* cap_delta_f,
    TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* grad_beta, bool* finite_actions, int64_t num_instances,
    bool smooth, scalar_t alpha) {
  const int thread_count = at::get_num_threads();
  const int64_t grad_size = num_instances * action_count;
  std::vector<scalar_t> local_grad(
      static_cast<size_t>(thread_count) * static_cast<size_t>(grad_size), 0);
  #pragma omp parallel for if(count >= 2048) num_threads(thread_count) schedule(static)
  for (int i = 0; i < count; ++i) {
    const int64_t inst = inst_ids[i];
    if (inst < 0 || inst >= num_instances) continue;
    const scalar_t* g = gradient + i * 8;
    if (g[0] == 0 && g[1] == 0 && g[2] == 0 && g[3] == 0) continue;
    const int64_t* arc = arcs + i * columns;
    const int64_t in = arc[0], out = arc[1];
    const scalar_t* cur = current + i * 8;
    for (int action = 0; action < action_count; ++action) {
      const int64_t key = i * action_count + action;
      if (!valid_actions[key]) continue;
      scalar_t target[4];
      const scalar_t target_rc = rc[out] + cap_delta_r[key];
      const scalar_t target_fc = fc[out] + cap_delta_f[key];
      cellValueRow4AtLoad<scalar_t>(in, out, target_rows[key], target_senses[key], pins,
          ra, fa, rs, fs, target_rc, target_fc, rd, fd, rt, ft,
          target, smooth, alpha, columns > 8 ? arc[8] : 15);
      bool finite = true;
      for (int k = 0; k < 4; ++k) finite = finite && std::isfinite(target[k]);
      finite_actions[key] = finite;
      if (!finite) continue;
      scalar_t dot = 0;
      for (int k = 0; k < 4; ++k) dot += g[k] * (target[k] - cur[k]);
      scalar_t* thread_grad = local_grad.data()
          + static_cast<size_t>(omp_get_thread_num()) * static_cast<size_t>(grad_size);
      thread_grad[inst * action_count + action] += dot;
    }
  }
  #pragma omp parallel for if(grad_size >= 2048) num_threads(thread_count) schedule(static)
  for (int64_t index = 0; index < grad_size; ++index) {
    scalar_t sum = 0;
    for (int thread = 0; thread < thread_count; ++thread) {
      sum += local_grad[static_cast<size_t>(thread) * static_cast<size_t>(grad_size)
          + static_cast<size_t>(index)];
    }
    grad_beta[index] = sum;
  }
}

template <typename scalar_t>
void cellAatGroupedBackwardCpuLauncher(
    const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int pins) {
  #pragma omp parallel for if(pins >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int pin = 0; pin < pins; ++pin) {
    cellGroupedBackwardRow<scalar_t>(pin, input_order, input_offsets, output_order, output_offsets, jac, gradient, ra, fa, rs, fs, rc, fc, pins);
  }
}

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cpu() && value.device() == reference.device() && value.is_contiguous(),
              "cell AAT expects contiguous tensors on one CPU device");
  TORCH_CHECK(value.scalar_type() == dtype, "cell AAT dtype mismatch");
}

template <typename scalar_t>
TimingLutTable<scalar_t> table(const std::vector<at::Tensor>& values, int offset) {
  return {values[offset].data_ptr<scalar_t>(), values[offset + 1].data_ptr<scalar_t>(),
      values[offset + 2].data_ptr<scalar_t>(), values[offset + 3].data_ptr<int64_t>(),
      values[offset + 4].data_ptr<int64_t>(), static_cast<int>(values[offset].size(0)),
      static_cast<int>(values[offset].size(1)), static_cast<int>(values[offset + 1].size(1)),
      static_cast<int>(values[offset + 2].size(1))};
}
}  // namespace

std::vector<at::Tensor> cell_aat_forward_cpu(
    at::Tensor arcs, at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rc, at::Tensor fc, std::vector<at::Tensor> tables, bool smooth, double alpha, bool early) {
  requireTensor(arcs, ra, at::kLong);
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5, "cell AAT arcs need at least five columns");
  for (const auto& pin : {ra, fa, rs, fs, rc, fc}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "cell AAT pin shape mismatch");
  }
  TORCH_CHECK(tables.size() == 20 && alpha > 0, "cell AAT LUT/temperature mismatch");
  for (int i = 0; i < 20; ++i) {
    requireTensor(tables[i], ra, i % 5 < 3 ? ra.scalar_type() : at::kLong);
    TORCH_CHECK(tables[i].dim() == (i % 5 < 3 ? 2 : 1), "cell AAT LUT shape mismatch");
    TORCH_CHECK(tables[i].size(0) == tables[i - i % 5].size(0), "cell AAT LUT row mismatch");
  }
  checkCpuArcIndices(arcs, ra.numel());
  for (int offset = 0; offset < 20; offset += 5) {
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 8}, ra.options());
  auto jac = fullyWrittenCpuTensor({arcs.size(0), 8, 6}, ra.options());
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "cellAatForward", [&] {
    cellAatForwardCpuLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(), fs.data_ptr<scalar_t>(),
        rc.data_ptr<scalar_t>(), fc.data_ptr<scalar_t>(), table<scalar_t>(tables, 0),
        table<scalar_t>(tables, 5), table<scalar_t>(tables, 10), table<scalar_t>(tables, 15),
        updates.data_ptr<scalar_t>(), jac.data_ptr<scalar_t>(), smooth, scalar_t(alpha), early);
  });
  return {updates, jac};
}

at::Tensor cell_aat_values_cpu(
    at::Tensor arcs, at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rc, at::Tensor fc, std::vector<at::Tensor> tables, bool smooth, double alpha) {
  requireTensor(arcs, ra, at::kLong);
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5, "cell AAT arcs need at least five columns");
  for (const auto& pin : {ra, fa, rs, fs, rc, fc}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "cell AAT pin shape mismatch");
  }
  TORCH_CHECK(tables.size() == 20 && alpha > 0, "cell AAT LUT/temperature mismatch");
  for (int i = 0; i < 20; ++i) {
    requireTensor(tables[i], ra, i % 5 < 3 ? ra.scalar_type() : at::kLong);
    TORCH_CHECK(tables[i].dim() == (i % 5 < 3 ? 2 : 1), "cell AAT LUT shape mismatch");
    TORCH_CHECK(tables[i].size(0) == tables[i - i % 5].size(0), "cell AAT LUT row mismatch");
  }
  checkCpuArcIndices(arcs, ra.numel());
  for (int offset = 0; offset < 20; offset += 5) {
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 8}, ra.options());
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "cellAatValues", [&] {
    cellAatValuesCpuLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(), fs.data_ptr<scalar_t>(),
        rc.data_ptr<scalar_t>(), fc.data_ptr<scalar_t>(), table<scalar_t>(tables, 0),
        table<scalar_t>(tables, 5), table<scalar_t>(tables, 10), table<scalar_t>(tables, 15),
        updates.data_ptr<scalar_t>(), smooth, scalar_t(alpha));
  });
  return updates;
}

std::vector<at::Tensor> cell_aat_action_values_cpu(
    at::Tensor arcs, at::Tensor target_arc_ids, at::Tensor target_senses,
    at::Tensor valid_arcs,
    at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rc, at::Tensor fc, std::vector<at::Tensor> tables,
    bool need_delays, bool smooth, double alpha) {
  requireTensor(arcs, ra, at::kLong);
  requireTensor(target_arc_ids, ra, at::kLong);
  requireTensor(target_senses, ra, at::kLong);
  requireTensor(valid_arcs, ra, at::kBool);
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5, "cell AAT arcs need at least five columns");
  TORCH_CHECK(target_arc_ids.dim() == 2 && target_arc_ids.size(0) == arcs.size(0),
              "cell action target row shape mismatch");
  TORCH_CHECK(target_senses.sizes() == target_arc_ids.sizes(),
              "cell action target sense shape mismatch");
  TORCH_CHECK(valid_arcs.sizes() == target_arc_ids.sizes(),
              "cell action validity shape mismatch");
  const int action_count = static_cast<int>(target_arc_ids.size(1));
  for (const auto& pin : {ra, fa, rs, fs, rc, fc}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "cell AAT pin shape mismatch");
  }
  TORCH_CHECK(tables.size() == 20 && alpha > 0, "cell AAT LUT/temperature mismatch");
  for (int i = 0; i < 20; ++i) {
    requireTensor(tables[i], ra, i % 5 < 3 ? ra.scalar_type() : at::kLong);
    TORCH_CHECK(tables[i].dim() == (i % 5 < 3 ? 2 : 1), "cell AAT LUT shape mismatch");
    TORCH_CHECK(tables[i].size(0) == tables[i - i % 5].size(0), "cell AAT LUT row mismatch");
  }
  checkCpuArcIndices(arcs, ra.numel());
  for (int offset = 0; offset < 20; offset += 5) {
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  const int64_t row_count = tables[0].size(0);
  auto selected_targets = target_arc_ids.masked_select(valid_arcs);
  TORCH_CHECK(
      selected_targets.numel() == 0 ||
          (selected_targets.min().item<int64_t>() >= 0 &&
           selected_targets.max().item<int64_t>() < row_count),
      "cell action target row is out of range");
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 8}, ra.options());
  auto deltas = fullyWrittenCpuTensor({arcs.size(0), action_count, 8}, ra.options());
  auto finite_actions = fullyWrittenCpuTensor(
      {arcs.size(0), action_count}, ra.options().dtype(at::kBool));
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "cellAatActionValues", [&] {
    cellAatActionValuesCpuLauncher<scalar_t>(
        arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        target_arc_ids.data_ptr<int64_t>(), target_senses.data_ptr<int64_t>(),
        valid_arcs.data_ptr<bool>(), action_count,
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(),
        fs.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(), fc.data_ptr<scalar_t>(),
        table<scalar_t>(tables, 0), table<scalar_t>(tables, 5),
        table<scalar_t>(tables, 10), table<scalar_t>(tables, 15),
        updates.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(),
        finite_actions.data_ptr<bool>(),
        need_delays, smooth, scalar_t(alpha));
  });
  return {updates, deltas, finite_actions};
}


std::vector<at::Tensor> cell_aat_action_backward_cpu(
    at::Tensor arcs, at::Tensor target_arc_ids, at::Tensor target_senses,
    at::Tensor valid_arcs, at::Tensor inst_ids, at::Tensor current,
    at::Tensor gradient, at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rc, at::Tensor fc, at::Tensor cap_delta_r, at::Tensor cap_delta_f,
    std::vector<at::Tensor> tables,
    int64_t num_instances, bool smooth, double alpha) {
  requireTensor(arcs, ra, at::kLong);
  requireTensor(target_arc_ids, ra, at::kLong);
  requireTensor(target_senses, ra, at::kLong);
  requireTensor(valid_arcs, ra, at::kBool);
  requireTensor(inst_ids, ra, at::kLong);
  requireTensor(current, ra, ra.scalar_type());
  requireTensor(gradient, ra, ra.scalar_type());
  requireTensor(cap_delta_r, ra, ra.scalar_type());
  requireTensor(cap_delta_f, ra, ra.scalar_type());
  TORCH_CHECK(cap_delta_r.sizes() == target_arc_ids.sizes(),
              "cell action rise cap delta shape mismatch");
  TORCH_CHECK(cap_delta_f.sizes() == target_arc_ids.sizes(),
              "cell action fall cap delta shape mismatch");
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5, "cell AAT arcs need at least five columns");
  TORCH_CHECK(target_arc_ids.dim() == 2 && target_arc_ids.size(0) == arcs.size(0),
              "cell action target row shape mismatch");
  TORCH_CHECK(target_senses.sizes() == target_arc_ids.sizes(),
              "cell action target sense shape mismatch");
  TORCH_CHECK(valid_arcs.sizes() == target_arc_ids.sizes(),
              "cell action validity shape mismatch");
  TORCH_CHECK(inst_ids.dim() == 1 && inst_ids.size(0) == arcs.size(0),
              "cell action instance shape mismatch");
  TORCH_CHECK(current.dim() == 2 && current.size(0) == arcs.size(0) && current.size(1) == 8,
              "cell action current shape mismatch");
  TORCH_CHECK(gradient.sizes() == current.sizes(), "cell action gradient shape mismatch");
  TORCH_CHECK(num_instances >= 0, "cell action instance count must be non-negative");
  const int action_count = static_cast<int>(target_arc_ids.size(1));
  for (const auto& pin : {ra, fa, rs, fs, rc, fc}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "cell AAT pin shape mismatch");
  }
  TORCH_CHECK(tables.size() == 20 && alpha > 0, "cell AAT LUT/temperature mismatch");
  for (int i = 0; i < 20; ++i) {
    requireTensor(tables[i], ra, i % 5 < 3 ? ra.scalar_type() : at::kLong);
    TORCH_CHECK(tables[i].dim() == (i % 5 < 3 ? 2 : 1), "cell AAT LUT shape mismatch");
    TORCH_CHECK(tables[i].size(0) == tables[i - i % 5].size(0), "cell AAT LUT row mismatch");
  }
  checkCpuArcIndices(arcs, ra.numel());
  for (int offset = 0; offset < 20; offset += 5) {
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  const int64_t row_count = tables[0].size(0);
  auto selected_targets = target_arc_ids.masked_select(valid_arcs);
  TORCH_CHECK(
      selected_targets.numel() == 0 ||
          (selected_targets.min().item<int64_t>() >= 0 &&
           selected_targets.max().item<int64_t>() < row_count),
      "cell action target row is out of range");
  auto grad_beta = at::zeros({num_instances, action_count}, ra.options());
  auto finite_actions = valid_arcs.clone();
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "cellAatActionBackward", [&] {
    cellAatActionBackwardCpuLauncher<scalar_t>(
        arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        target_arc_ids.data_ptr<int64_t>(), target_senses.data_ptr<int64_t>(),
        valid_arcs.data_ptr<bool>(), inst_ids.data_ptr<int64_t>(), action_count,
        current.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(),
        fs.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(), fc.data_ptr<scalar_t>(),
        cap_delta_r.data_ptr<scalar_t>(), cap_delta_f.data_ptr<scalar_t>(),
        table<scalar_t>(tables, 0), table<scalar_t>(tables, 5),
        table<scalar_t>(tables, 10), table<scalar_t>(tables, 15),
        grad_beta.data_ptr<scalar_t>(), finite_actions.data_ptr<bool>(),
        num_instances, smooth, scalar_t(alpha));
  });
  return {grad_beta, finite_actions};
}

std::vector<at::Tensor> cell_aat_backward_cpu(
    at::Tensor arcs, at::Tensor jac, at::Tensor gradient, int64_t pins,
    std::vector<at::Tensor> prepared_groups) {
  requireTensor(arcs, jac, at::kLong);
  requireTensor(gradient, jac, jac.scalar_type());
  requireTensor(jac, jac, jac.scalar_type());
  TORCH_CHECK(jac.dim() == 3 && jac.size(0) == arcs.size(0) && jac.size(1) == 8 && jac.size(2) == 6,
              "cell AAT Jacobian shape mismatch");
  TORCH_CHECK(gradient.dim() == 2 && gradient.size(0) == arcs.size(0) && gradient.size(1) == 8,
              "cell AAT gradient shape mismatch");
  checkCpuArcIndices(arcs, pins);
  auto inputs = timingScatterGroups(arcs.select(1, 0), pins, prepared_groups);
  auto outputs = timingScatterGroups(arcs.select(1, 1), pins, prepared_groups, 2);
  std::vector<at::Tensor> result;
  for (int i = 0; i < 6; ++i) {
    result.push_back(fullyWrittenCpuTensor({pins}, jac.options()));
  }
  AT_DISPATCH_FLOATING_TYPES(jac.scalar_type(), "cellAatBackward", [&] {
      cellAatGroupedBackwardCpuLauncher<scalar_t>(inputs.order.data_ptr<int64_t>(), inputs.offsets.data_ptr<int64_t>(),
          outputs.order.data_ptr<int64_t>(), outputs.offsets.data_ptr<int64_t>(),
          jac.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), result[0].data_ptr<scalar_t>(),
          result[1].data_ptr<scalar_t>(), result[2].data_ptr<scalar_t>(), result[3].data_ptr<scalar_t>(),
          result[4].data_ptr<scalar_t>(), result[5].data_ptr<scalar_t>(), pins);

  });
  return result;
}
