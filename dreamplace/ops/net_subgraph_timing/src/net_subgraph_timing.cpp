#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "segment_count_packer.h"

namespace py = pybind11;

namespace {

void check_cpu_1d_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::vector<int64_t> tensor_to_i64_vector(const at::Tensor& tensor, const char* name) {
  check_cpu_1d_contiguous(tensor, name);
  std::vector<int64_t> values(static_cast<size_t>(tensor.numel()));
  if (tensor.scalar_type() == at::kInt) {
    const int32_t* data = tensor.data_ptr<int32_t>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      values[static_cast<size_t>(i)] = static_cast<int64_t>(data[i]);
    }
  } else if (tensor.scalar_type() == at::kLong) {
    const int64_t* data = tensor.data_ptr<int64_t>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      values[static_cast<size_t>(i)] = data[i];
    }
  } else {
    TORCH_CHECK(false, name, " must be int32 or int64");
  }
  return values;
}

void check_float_like(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalar_type) {
  check_cpu_1d_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " dtype mismatch");
}

void check_cpu_2d_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_float_2d_like(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalar_type) {
  check_cpu_2d_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " dtype mismatch");
}

void check_cpu_3d_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dim() == 3, name, " must be 3-D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_float_3d_like(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalar_type) {
  check_cpu_3d_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " dtype mismatch");
}

void check_integer_1d_contiguous(const at::Tensor& tensor, const char* name) {
  check_cpu_1d_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::kLong || tensor.scalar_type() == at::kInt,
      name,
      " must be int32 or int64");
}

template <typename scalar_t>
int64_t find_axis_hi(
    const scalar_t* axis,
    const int64_t count,
    const scalar_t point) {
  if (count <= 1) {
    return 0;
  }
  if (point < axis[0]) {
    return 1;
  }
  if (point >= axis[count - 1]) {
    return count - 1;
  }
  int64_t lo = 0;
  int64_t hi = count;
  while (hi - lo > 1) {
    const int64_t mid = (hi + lo) / 2;
    if (point >= axis[mid]) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return hi;
}

template <typename scalar_t>
scalar_t interp_size_1d(
    const scalar_t* values,
    const int64_t size_count,
    const scalar_t bsu) {
  if (size_count <= 1) {
    return values[0];
  }
  const scalar_t clipped = std::min(
      std::max(bsu, static_cast<scalar_t>(0)),
      static_cast<scalar_t>(size_count - 1));
  const int64_t lo = static_cast<int64_t>(std::floor(clipped));
  const int64_t hi = std::min<int64_t>(lo + 1, size_count - 1);
  const scalar_t alpha = clipped - static_cast<scalar_t>(lo);
  return (static_cast<scalar_t>(1) - alpha) * values[lo] + alpha * values[hi];
}

template <typename scalar_t>
scalar_t interp_size_1d_slope(
    const scalar_t* values,
    const int64_t size_count,
    const scalar_t bsu) {
  if (size_count <= 1) {
    return static_cast<scalar_t>(0);
  }
  if (bsu < static_cast<scalar_t>(0) ||
      bsu > static_cast<scalar_t>(size_count - 1)) {
    return static_cast<scalar_t>(0);
  }
  const scalar_t clipped = std::min(
      std::max(bsu, static_cast<scalar_t>(0)),
      static_cast<scalar_t>(size_count - 1));
  const int64_t lo = static_cast<int64_t>(std::floor(clipped));
  const int64_t hi = std::min<int64_t>(lo + 1, size_count - 1);
  return values[hi] - values[lo];
}

template <typename scalar_t>
scalar_t lookup_lut_3d(
    const scalar_t* table,
    const int64_t size_count,
    const int64_t slew_count,
    const int64_t load_count,
    const scalar_t* slew_axis,
    const scalar_t* load_axis,
    const scalar_t bsu,
    const scalar_t input_slew,
    const scalar_t output_load) {
  TORCH_CHECK(size_count > 0, "LUT size axis must be non-empty");
  TORCH_CHECK(slew_count > 0, "LUT slew axis must be non-empty");
  TORCH_CHECK(load_count > 0, "LUT load axis must be non-empty");
  const int64_t slew_hi = find_axis_hi(slew_axis, slew_count, input_slew);
  const int64_t slew_lo = slew_count <= 1 ? 0 : slew_hi - 1;
  const int64_t load_hi = find_axis_hi(load_axis, load_count, output_load);
  const int64_t load_lo = load_count <= 1 ? 0 : load_hi - 1;
  const scalar_t slew_alpha =
      slew_count <= 1
          ? static_cast<scalar_t>(0)
          : (input_slew - slew_axis[slew_lo]) /
                (slew_axis[slew_hi] - slew_axis[slew_lo]);
  const scalar_t load_alpha =
      load_count <= 1
          ? static_cast<scalar_t>(0)
          : (output_load - load_axis[load_lo]) /
                (load_axis[load_hi] - load_axis[load_lo]);

  const scalar_t clipped_bsu = std::min(
      std::max(bsu, static_cast<scalar_t>(0)),
      static_cast<scalar_t>(size_count - 1));
  const int64_t size_lo = static_cast<int64_t>(std::floor(clipped_bsu));
  const int64_t size_hi = std::min<int64_t>(size_lo + 1, size_count - 1);
  const scalar_t size_alpha = clipped_bsu - static_cast<scalar_t>(size_lo);

  auto at = [&](const int64_t size, const int64_t slew, const int64_t load) {
    return table[(size * slew_count + slew) * load_count + load];
  };
  auto interp_size = [&](const int64_t size) {
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t v0 =
        (static_cast<scalar_t>(1) - load_alpha) * v00 + load_alpha * v01;
    const scalar_t v1 =
        (static_cast<scalar_t>(1) - load_alpha) * v10 + load_alpha * v11;
    return (static_cast<scalar_t>(1) - slew_alpha) * v0 + slew_alpha * v1;
  };

  const scalar_t lo_value = interp_size(size_lo);
  const scalar_t hi_value = interp_size(size_hi);
  return (static_cast<scalar_t>(1) - size_alpha) * lo_value +
         size_alpha * hi_value;
}

template <typename scalar_t>
struct LutValueAndGrad {
  scalar_t value;
  scalar_t grad_bsu;
  scalar_t grad_slew;
  scalar_t grad_load;
};

template <typename scalar_t>
LutValueAndGrad<scalar_t> lookup_lut_3d_with_grad(
    const scalar_t* table,
    const int64_t size_count,
    const int64_t slew_count,
    const int64_t load_count,
    const scalar_t* slew_axis,
    const scalar_t* load_axis,
    const scalar_t bsu,
    const scalar_t input_slew,
    const scalar_t output_load) {
  TORCH_CHECK(size_count > 0, "LUT size axis must be non-empty");
  TORCH_CHECK(slew_count > 0, "LUT slew axis must be non-empty");
  TORCH_CHECK(load_count > 0, "LUT load axis must be non-empty");
  const int64_t slew_hi = find_axis_hi(slew_axis, slew_count, input_slew);
  const int64_t slew_lo = slew_count <= 1 ? 0 : slew_hi - 1;
  const int64_t load_hi = find_axis_hi(load_axis, load_count, output_load);
  const int64_t load_lo = load_count <= 1 ? 0 : load_hi - 1;
  const scalar_t slew_den =
      slew_count <= 1 ? static_cast<scalar_t>(1) : slew_axis[slew_hi] - slew_axis[slew_lo];
  const scalar_t load_den =
      load_count <= 1 ? static_cast<scalar_t>(1) : load_axis[load_hi] - load_axis[load_lo];
  const scalar_t slew_alpha =
      slew_count <= 1 ? static_cast<scalar_t>(0) : (input_slew - slew_axis[slew_lo]) / slew_den;
  const scalar_t load_alpha =
      load_count <= 1 ? static_cast<scalar_t>(0) : (output_load - load_axis[load_lo]) / load_den;

  const scalar_t clipped_bsu = std::min(
      std::max(bsu, static_cast<scalar_t>(0)),
      static_cast<scalar_t>(size_count - 1));
  const int64_t size_lo = static_cast<int64_t>(std::floor(clipped_bsu));
  const int64_t size_hi = std::min<int64_t>(size_lo + 1, size_count - 1);
  const scalar_t size_alpha = clipped_bsu - static_cast<scalar_t>(size_lo);

  auto at = [&](const int64_t size, const int64_t slew, const int64_t load) {
    return table[(size * slew_count + slew) * load_count + load];
  };
  auto interp_size = [&](const int64_t size) {
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t v0 =
        (static_cast<scalar_t>(1) - load_alpha) * v00 + load_alpha * v01;
    const scalar_t v1 =
        (static_cast<scalar_t>(1) - load_alpha) * v10 + load_alpha * v11;
    return (static_cast<scalar_t>(1) - slew_alpha) * v0 + slew_alpha * v1;
  };
  auto interp_size_grad_slew = [&](const int64_t size) {
    if (slew_count <= 1) {
      return static_cast<scalar_t>(0);
    }
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t lo_load =
        (static_cast<scalar_t>(1) - load_alpha) * v00 + load_alpha * v01;
    const scalar_t hi_load =
        (static_cast<scalar_t>(1) - load_alpha) * v10 + load_alpha * v11;
    return (hi_load - lo_load) / slew_den;
  };
  auto interp_size_grad_load = [&](const int64_t size) {
    if (load_count <= 1) {
      return static_cast<scalar_t>(0);
    }
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t lo_slew =
        (static_cast<scalar_t>(1) - slew_alpha) * v00 + slew_alpha * v10;
    const scalar_t hi_slew =
        (static_cast<scalar_t>(1) - slew_alpha) * v01 + slew_alpha * v11;
    return (hi_slew - lo_slew) / load_den;
  };

  const scalar_t lo_value = interp_size(size_lo);
  const scalar_t hi_value = interp_size(size_hi);
  const scalar_t value =
      (static_cast<scalar_t>(1) - size_alpha) * lo_value + size_alpha * hi_value;
  const scalar_t grad_bsu =
      (bsu < static_cast<scalar_t>(0) || bsu > static_cast<scalar_t>(size_count - 1))
          ? static_cast<scalar_t>(0)
          : hi_value - lo_value;
  const scalar_t grad_slew =
      (static_cast<scalar_t>(1) - size_alpha) * interp_size_grad_slew(size_lo) +
      size_alpha * interp_size_grad_slew(size_hi);
  const scalar_t grad_load =
      (static_cast<scalar_t>(1) - size_alpha) * interp_size_grad_load(size_lo) +
      size_alpha * interp_size_grad_load(size_hi);
  return {value, grad_bsu, grad_slew, grad_load};
}

template <typename scalar_t>
py::dict segment_transfer_forward_impl(
    const at::Tensor& input_arrival,
    const at::Tensor& input_slew,
    const at::Tensor& downstream_load,
    const at::Tensor& edge_resistance,
    const at::Tensor& edge_capacitance,
    const at::Tensor& repeater_count,
    const at::Tensor& split_fractions,
    const at::Tensor& bsu_index,
    const at::Tensor& upstream_retained_cap,
    const at::Tensor& buffer_input_cap_by_size,
    const at::Tensor& buffer_slew_axis,
    const at::Tensor& buffer_load_axis,
    const at::Tensor& buffer_delay_lut,
    const at::Tensor& buffer_output_slew_lut) {
  const int64_t sample_count = input_arrival.numel();
  const int64_t nmax = split_fractions.size(1) - 1;
  const int64_t size_count = buffer_input_cap_by_size.numel();
  const int64_t slew_count = buffer_slew_axis.numel();
  const int64_t load_count = buffer_load_axis.numel();

  auto options = input_arrival.options();
  auto upstream_visible_load = at::empty({sample_count}, options);
  auto segment_delay = at::empty({sample_count}, options);
  auto output_arrival = at::empty({sample_count}, options);
  auto output_slew = at::empty({sample_count}, options);
  auto first_buffer_input_slew = at::zeros({sample_count}, options);
  auto first_buffer_output_load = at::zeros({sample_count}, options);
  auto first_buffer_delay = at::zeros({sample_count}, options);
  auto first_buffer_output_slew = at::zeros({sample_count}, options);

  const scalar_t* input_arrival_ptr = input_arrival.data_ptr<scalar_t>();
  const scalar_t* input_slew_ptr = input_slew.data_ptr<scalar_t>();
  const scalar_t* downstream_load_ptr = downstream_load.data_ptr<scalar_t>();
  const scalar_t* edge_resistance_ptr = edge_resistance.data_ptr<scalar_t>();
  const scalar_t* edge_capacitance_ptr = edge_capacitance.data_ptr<scalar_t>();
  const scalar_t* split_ptr = split_fractions.data_ptr<scalar_t>();
  const scalar_t* bsu_ptr = bsu_index.data_ptr<scalar_t>();
  const scalar_t* retained_ptr = upstream_retained_cap.data_ptr<scalar_t>();
  const scalar_t* cap_by_size_ptr = buffer_input_cap_by_size.data_ptr<scalar_t>();
  const scalar_t* slew_axis_ptr = buffer_slew_axis.data_ptr<scalar_t>();
  const scalar_t* load_axis_ptr = buffer_load_axis.data_ptr<scalar_t>();
  const scalar_t* delay_lut_ptr = buffer_delay_lut.data_ptr<scalar_t>();
  const scalar_t* output_slew_lut_ptr = buffer_output_slew_lut.data_ptr<scalar_t>();
  const int64_t* count_i64_ptr =
      repeater_count.scalar_type() == at::kLong ? repeater_count.data_ptr<int64_t>() : nullptr;
  const int32_t* count_i32_ptr =
      repeater_count.scalar_type() == at::kInt ? repeater_count.data_ptr<int32_t>() : nullptr;

  scalar_t* upstream_visible_load_ptr = upstream_visible_load.data_ptr<scalar_t>();
  scalar_t* segment_delay_ptr = segment_delay.data_ptr<scalar_t>();
  scalar_t* output_arrival_ptr = output_arrival.data_ptr<scalar_t>();
  scalar_t* output_slew_ptr = output_slew.data_ptr<scalar_t>();
  scalar_t* first_input_slew_ptr = first_buffer_input_slew.data_ptr<scalar_t>();
  scalar_t* first_output_load_ptr = first_buffer_output_load.data_ptr<scalar_t>();
  scalar_t* first_delay_ptr = first_buffer_delay.data_ptr<scalar_t>();
  scalar_t* first_output_slew_ptr = first_buffer_output_slew.data_ptr<scalar_t>();

  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10 * delay;
    return std::sqrt(slew * slew + delta * delta);
  };

  for (int64_t sample = 0; sample < sample_count; ++sample) {
    const int64_t count =
        count_i64_ptr ? count_i64_ptr[sample] : static_cast<int64_t>(count_i32_ptr[sample]);
    TORCH_CHECK(count >= 0 && count <= nmax, "repeater_count out of range");
    const scalar_t resistance = edge_resistance_ptr[sample];
    const scalar_t capacitance = edge_capacitance_ptr[sample];
    const scalar_t load = downstream_load_ptr[sample];
    const scalar_t no_buffer_delay = resistance * (load + static_cast<scalar_t>(0.5) * capacitance);
    const scalar_t no_buffer_slew = wire_slew(input_slew_ptr[sample], no_buffer_delay);
    if (count == 0) {
      upstream_visible_load_ptr[sample] = load + capacitance;
      segment_delay_ptr[sample] = no_buffer_delay;
      output_arrival_ptr[sample] = input_arrival_ptr[sample] + no_buffer_delay;
      output_slew_ptr[sample] = no_buffer_slew;
      continue;
    }

    const scalar_t bsu = bsu_ptr[sample];
    const scalar_t buffer_input_cap =
        interp_size_1d(cap_by_size_ptr, size_count, bsu);
    scalar_t total_delay = static_cast<scalar_t>(0);
    scalar_t slew = input_slew_ptr[sample];
    for (int64_t step = 0; step < count; ++step) {
      const scalar_t step_fraction = split_ptr[sample * (nmax + 1) + step];
      const scalar_t step_resistance = resistance * step_fraction;
      const scalar_t step_capacitance = capacitance * step_fraction;
      const scalar_t wire_delay =
          step_resistance *
          (buffer_input_cap + static_cast<scalar_t>(0.5) * step_capacitance);
      total_delay += wire_delay;
      slew = wire_slew(slew, wire_delay);
      const bool has_next_buffer = count > (step + 1);
      const scalar_t next_fraction = split_ptr[sample * (nmax + 1) + step + 1];
      const scalar_t next_capacitance = capacitance * next_fraction;
      const scalar_t output_load =
          (has_next_buffer ? buffer_input_cap : load) + next_capacitance;
      const scalar_t buffer_delay = lookup_lut_3d(
          delay_lut_ptr,
          size_count,
          slew_count,
          load_count,
          slew_axis_ptr,
          load_axis_ptr,
          bsu,
          slew,
          output_load);
      const scalar_t buffer_slew = lookup_lut_3d(
          output_slew_lut_ptr,
          size_count,
          slew_count,
          load_count,
          slew_axis_ptr,
          load_axis_ptr,
          bsu,
          slew,
          output_load);
      if (step == 0) {
        first_input_slew_ptr[sample] = slew;
        first_output_load_ptr[sample] = output_load;
        first_delay_ptr[sample] = buffer_delay;
        first_output_slew_ptr[sample] = buffer_slew;
      }
      total_delay += buffer_delay;
      slew = buffer_slew;
    }
    const scalar_t final_fraction = split_ptr[sample * (nmax + 1) + count];
    const scalar_t final_resistance = resistance * final_fraction;
    const scalar_t final_capacitance = capacitance * final_fraction;
    const scalar_t downstream_wire_delay =
        final_resistance *
        (load + static_cast<scalar_t>(0.5) * final_capacitance);
    const scalar_t delay = total_delay + downstream_wire_delay;
    upstream_visible_load_ptr[sample] =
        buffer_input_cap + capacitance * split_ptr[sample * (nmax + 1)] +
        retained_ptr[sample];
    segment_delay_ptr[sample] = delay;
    output_arrival_ptr[sample] = input_arrival_ptr[sample] + delay;
    output_slew_ptr[sample] = wire_slew(slew, downstream_wire_delay);
  }

  py::dict result;
  result["upstream_visible_load"] = upstream_visible_load;
  result["segment_delay"] = segment_delay;
  result["output_arrival"] = output_arrival;
  result["output_slew"] = output_slew;
  result["first_buffer_input_slew"] = first_buffer_input_slew;
  result["first_buffer_output_load"] = first_buffer_output_load;
  result["first_buffer_delay"] = first_buffer_delay;
  result["first_buffer_output_slew"] = first_buffer_output_slew;
  return result;
}

template <typename scalar_t>
py::dict forward_impl(
    const std::vector<int64_t>& topo,
    const std::vector<int64_t>& topo_start,
    const std::vector<int64_t>& parents,
    const std::vector<int64_t>& child_start,
    const std::vector<int64_t>& children,
    const at::Tensor& edge_resistance,
    const at::Tensor& node_capacitance,
    const at::Tensor& edge_capacitance,
    const at::Tensor& driver_arrival,
    const at::Tensor& driver_slew,
    const std::vector<int64_t>& candidate_node_id,
    const at::Tensor& candidate_bu,
    const at::Tensor& buffer_input_cap,
    const at::Tensor& buffer_delay,
    const at::Tensor& buffer_output_slew,
    const std::vector<int64_t>& sink_node_id,
    const at::Tensor& sink_net_index) {
  const int64_t num_nodes = node_capacitance.numel();
  const bool has_edge_cap = edge_capacitance.numel() > 0;
  const bool has_sinks = !sink_node_id.empty();
  const bool has_sink_net_index = sink_net_index.numel() > 0;

  auto effective_node_cap = node_capacitance.clone();
  auto bu_by_node = at::zeros_like(node_capacitance);
  auto buffer_input_cap_by_node = at::zeros_like(node_capacitance);
  auto buffer_delay_by_node = at::zeros_like(node_capacitance);
  auto buffer_output_slew_by_node = at::zeros_like(node_capacitance);

  scalar_t* effective_ptr = effective_node_cap.data_ptr<scalar_t>();
  scalar_t* bu_by_node_ptr = bu_by_node.data_ptr<scalar_t>();
  scalar_t* buffer_input_cap_by_node_ptr = buffer_input_cap_by_node.data_ptr<scalar_t>();
  scalar_t* buffer_delay_by_node_ptr = buffer_delay_by_node.data_ptr<scalar_t>();
  scalar_t* buffer_output_slew_by_node_ptr = buffer_output_slew_by_node.data_ptr<scalar_t>();

  const scalar_t* edge_cap_ptr = has_edge_cap ? edge_capacitance.data_ptr<scalar_t>() : nullptr;
  if (has_edge_cap) {
    for (int64_t child = 0; child < num_nodes; ++child) {
      const int64_t parent = parents[static_cast<size_t>(child)];
      if (parent < 0) {
        continue;
      }
      TORCH_CHECK(parent < num_nodes, "pin_fa parent is out of range");
      const scalar_t half_cap = static_cast<scalar_t>(0.5) * edge_cap_ptr[child];
      effective_ptr[parent] += half_cap;
      effective_ptr[child] += half_cap;
    }
  }

  const scalar_t* candidate_bu_ptr = candidate_bu.data_ptr<scalar_t>();
  const scalar_t* buffer_input_cap_ptr = buffer_input_cap.data_ptr<scalar_t>();
  const scalar_t* buffer_delay_ptr = buffer_delay.data_ptr<scalar_t>();
  const scalar_t* buffer_output_slew_ptr = buffer_output_slew.data_ptr<scalar_t>();
  for (int64_t index = 0; index < static_cast<int64_t>(candidate_node_id.size()); ++index) {
    const int64_t node = candidate_node_id[static_cast<size_t>(index)];
    TORCH_CHECK(node >= 0 && node < num_nodes, "candidate_node_id is out of range");
    bu_by_node_ptr[node] = candidate_bu_ptr[index];
    buffer_input_cap_by_node_ptr[node] = buffer_input_cap_ptr[index];
    buffer_delay_by_node_ptr[node] = buffer_delay_ptr[index];
    buffer_output_slew_by_node_ptr[node] = buffer_output_slew_ptr[index];
  }

  auto lout = effective_node_cap.clone();
  auto lin = effective_node_cap.clone();
  scalar_t* lout_ptr = lout.data_ptr<scalar_t>();
  scalar_t* lin_ptr = lin.data_ptr<scalar_t>();

  const int64_t num_nets = static_cast<int64_t>(topo_start.size()) - 1;
  for (int64_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    const int64_t begin = topo_start[static_cast<size_t>(net_idx)];
    const int64_t end = topo_start[static_cast<size_t>(net_idx + 1)];
    for (int64_t pos = end - 1; pos >= begin; --pos) {
      const int64_t node = topo[static_cast<size_t>(pos)];
      TORCH_CHECK(node >= 0 && node < num_nodes, "topology node is out of range");
      scalar_t children_load = static_cast<scalar_t>(0);
      for (int64_t edge_idx = child_start[static_cast<size_t>(node)];
           edge_idx < child_start[static_cast<size_t>(node + 1)];
           ++edge_idx) {
        const int64_t child = children[static_cast<size_t>(edge_idx)];
        TORCH_CHECK(child >= 0 && child < num_nodes, "child node is out of range");
        children_load += lin_ptr[child];
      }
      const scalar_t node_lout = effective_ptr[node] + children_load;
      const scalar_t bu = bu_by_node_ptr[node];
      lout_ptr[node] = node_lout;
      lin_ptr[node] =
          (static_cast<scalar_t>(1) - bu) * node_lout + bu * buffer_input_cap_by_node_ptr[node];
    }
  }

  auto arrival_in = at::zeros_like(node_capacitance);
  auto arrival_out = at::zeros_like(node_capacitance);
  auto slew_in = at::zeros_like(node_capacitance);
  auto slew_out = at::zeros_like(node_capacitance);

  scalar_t* arrival_in_ptr = arrival_in.data_ptr<scalar_t>();
  scalar_t* arrival_out_ptr = arrival_out.data_ptr<scalar_t>();
  scalar_t* slew_in_ptr = slew_in.data_ptr<scalar_t>();
  scalar_t* slew_out_ptr = slew_out.data_ptr<scalar_t>();

  const scalar_t* edge_res_ptr = edge_resistance.data_ptr<scalar_t>();
  const scalar_t* driver_arrival_ptr = driver_arrival.data_ptr<scalar_t>();
  const scalar_t* driver_slew_ptr = driver_slew.data_ptr<scalar_t>();
  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));

  for (int64_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    const int64_t begin = topo_start[static_cast<size_t>(net_idx)];
    const int64_t end = topo_start[static_cast<size_t>(net_idx + 1)];
    if (begin == end) {
      continue;
    }
    const int64_t root = topo[static_cast<size_t>(begin)];
    TORCH_CHECK(root >= 0 && root < num_nodes, "root node is out of range");
    arrival_in_ptr[root] = driver_arrival_ptr[net_idx];
    arrival_out_ptr[root] = arrival_in_ptr[root];
    slew_in_ptr[root] = driver_slew_ptr[net_idx];
    slew_out_ptr[root] = slew_in_ptr[root];

    for (int64_t pos = begin; pos < end; ++pos) {
      const int64_t node = topo[static_cast<size_t>(pos)];
      const int64_t parent = parents[static_cast<size_t>(node)];
      if (parent >= 0) {
        TORCH_CHECK(parent < num_nodes, "pin_fa parent is out of range");
        const scalar_t wire_delta = edge_res_ptr[node] * lin_ptr[node];
        arrival_in_ptr[node] = arrival_out_ptr[parent] + wire_delta;
        const scalar_t slew_delta = log10 * wire_delta;
        slew_in_ptr[node] = std::sqrt(
            slew_out_ptr[parent] * slew_out_ptr[parent] + slew_delta * slew_delta);
      }

      const scalar_t bu = bu_by_node_ptr[node];
      arrival_out_ptr[node] = arrival_in_ptr[node] + bu * buffer_delay_by_node_ptr[node];
      slew_out_ptr[node] =
          (static_cast<scalar_t>(1) - bu) * slew_in_ptr[node] +
          bu * buffer_output_slew_by_node_ptr[node];
    }
  }

  py::dict result;
  result["lout"] = lout;
  result["lin"] = lin;
  result["effective_node_cap"] = effective_node_cap;
  result["arrival_in"] = arrival_in;
  result["arrival_out"] = arrival_out;
  result["slew_in"] = slew_in;
  result["slew_out"] = slew_out;

  if (has_sinks) {
    auto sink_index = at::empty({static_cast<int64_t>(sink_node_id.size())}, at::TensorOptions().dtype(at::kLong));
    int64_t* sink_index_ptr = sink_index.data_ptr<int64_t>();
    for (int64_t i = 0; i < static_cast<int64_t>(sink_node_id.size()); ++i) {
      const int64_t node = sink_node_id[static_cast<size_t>(i)];
      TORCH_CHECK(node >= 0 && node < num_nodes, "sink_node_id is out of range");
      sink_index_ptr[i] = node;
    }
    result["sink_arrival"] = arrival_out.index_select(0, sink_index);
    result["sink_slew"] = slew_out.index_select(0, sink_index);
    result["sink_load"] = lin.index_select(0, sink_index);
    result["sink_cap"] = effective_node_cap.index_select(0, sink_index);
    if (has_sink_net_index) {
      result["sink_net_index"] = sink_net_index;
    }
  }

  return result;
}

}  // namespace

py::dict forward(
    at::Tensor net_flat_topo_sort,
    at::Tensor net_flat_topo_sort_start,
    at::Tensor pin_fa,
    at::Tensor flat_pin_to_start,
    at::Tensor flat_pin_to,
    at::Tensor edge_resistance,
    at::Tensor node_capacitance,
    at::Tensor edge_capacitance,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor candidate_node_id,
    at::Tensor candidate_bu,
    at::Tensor buffer_input_cap,
    at::Tensor buffer_delay,
    at::Tensor buffer_output_slew,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index) {
  const auto scalar_type = node_capacitance.scalar_type();
  check_float_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_like(node_capacitance, "node_capacitance", scalar_type);
  if (edge_capacitance.numel() > 0) {
    check_float_like(edge_capacitance, "edge_capacitance", scalar_type);
    TORCH_CHECK(edge_capacitance.numel() == node_capacitance.numel(),
                "edge_capacitance size mismatch");
  } else {
    check_cpu_1d_contiguous(edge_capacitance, "edge_capacitance");
  }
  check_float_like(driver_arrival, "driver_arrival", scalar_type);
  check_float_like(driver_slew, "driver_slew", scalar_type);
  check_float_like(candidate_bu, "candidate_bu", scalar_type);
  check_float_like(buffer_input_cap, "buffer_input_cap", scalar_type);
  check_float_like(buffer_delay, "buffer_delay", scalar_type);
  check_float_like(buffer_output_slew, "buffer_output_slew", scalar_type);

  TORCH_CHECK(edge_resistance.numel() == node_capacitance.numel(),
              "edge_resistance size mismatch");
  TORCH_CHECK(candidate_bu.numel() == candidate_node_id.numel(),
              "candidate_bu size mismatch");
  TORCH_CHECK(buffer_input_cap.numel() == candidate_node_id.numel(),
              "buffer_input_cap size mismatch");
  TORCH_CHECK(buffer_delay.numel() == candidate_node_id.numel(),
              "buffer_delay size mismatch");
  TORCH_CHECK(buffer_output_slew.numel() == candidate_node_id.numel(),
              "buffer_output_slew size mismatch");
  TORCH_CHECK(net_flat_topo_sort_start.numel() >= 1,
              "net_flat_topo_sort_start must not be empty");
  TORCH_CHECK(driver_arrival.numel() == net_flat_topo_sort_start.numel() - 1,
              "driver_arrival size mismatch");
  TORCH_CHECK(driver_slew.numel() == net_flat_topo_sort_start.numel() - 1,
              "driver_slew size mismatch");
  TORCH_CHECK(pin_fa.numel() == node_capacitance.numel(), "pin_fa size mismatch");
  TORCH_CHECK(flat_pin_to_start.numel() == node_capacitance.numel() + 1,
              "flat_pin_to_start size mismatch");
  TORCH_CHECK(sink_net_index.numel() == 0 || sink_net_index.numel() == sink_node_id.numel(),
              "sink_net_index size mismatch");

  const std::vector<int64_t> topo = tensor_to_i64_vector(net_flat_topo_sort, "net_flat_topo_sort");
  const std::vector<int64_t> topo_start =
      tensor_to_i64_vector(net_flat_topo_sort_start, "net_flat_topo_sort_start");
  const std::vector<int64_t> parents = tensor_to_i64_vector(pin_fa, "pin_fa");
  const std::vector<int64_t> child_start =
      tensor_to_i64_vector(flat_pin_to_start, "flat_pin_to_start");
  const std::vector<int64_t> children = tensor_to_i64_vector(flat_pin_to, "flat_pin_to");
  const std::vector<int64_t> candidates =
      tensor_to_i64_vector(candidate_node_id, "candidate_node_id");
  const std::vector<int64_t> sinks = tensor_to_i64_vector(sink_node_id, "sink_node_id");
  if (sink_net_index.numel() > 0) {
    tensor_to_i64_vector(sink_net_index, "sink_net_index");
  } else {
    check_cpu_1d_contiguous(sink_net_index, "sink_net_index");
  }

  py::dict result;
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "net_subgraph_timing_forward", [&] {
    result = forward_impl<scalar_t>(
        topo,
        topo_start,
        parents,
        child_start,
        children,
        edge_resistance,
        node_capacitance,
        edge_capacitance,
        driver_arrival,
        driver_slew,
        candidates,
        candidate_bu,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        sinks,
        sink_net_index);
  });
  return result;
}

template <typename scalar_t>
scalar_t interpolate_count_value(
    const scalar_t* values,
    int64_t row_stride,
    scalar_t z,
    int64_t max_count) {
  if (z < static_cast<scalar_t>(0)) {
    z = static_cast<scalar_t>(0);
  }
  if (z > static_cast<scalar_t>(max_count)) {
    z = static_cast<scalar_t>(max_count);
  }
  const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
  const int64_t hi = std::min<int64_t>(lo + 1, max_count);
  const scalar_t alpha = z - static_cast<scalar_t>(lo);
  return (static_cast<scalar_t>(1) - alpha) * values[lo * row_stride] +
         alpha * values[hi * row_stride];
}

template <typename scalar_t>
py::dict segment_count_forward_impl(
    const std::vector<int64_t>& net_topo_start,
    const std::vector<int64_t>& edge_start,
    const std::vector<int64_t>& edge_parent_compact_id,
    const std::vector<int64_t>& edge_child_compact_id,
    const at::Tensor& edge_resistance,
    const at::Tensor& edge_capacitance,
    const at::Tensor& node_capacitance,
    const std::vector<int64_t>& edge_to_segment_id,
    const at::Tensor& driver_arrival,
    const at::Tensor& driver_slew,
    const at::Tensor& z_value,
    const at::Tensor& buffer_input_cap,
    const at::Tensor& buffer_delay,
    const at::Tensor& buffer_output_slew,
    const at::Tensor& parent_cap_fraction,
    const at::Tensor& child_cap_fraction,
    const at::Tensor& segment_sub_resistance_fraction,
    const std::vector<int64_t>& sink_node_id,
    const at::Tensor& sink_net_index,
    const std::vector<int64_t>& sink_node_compact_id) {
  const int64_t num_nodes = node_capacitance.numel();
  const int64_t num_edges = edge_resistance.numel();
  const int64_t num_segments = z_value.numel();
  const int64_t max_count = parent_cap_fraction.size(1) - 1;
  const int64_t sub_stride_count = max_count + 1;
  const int64_t sub_stride_segment = sub_stride_count * sub_stride_count;

  auto effective_node_cap = node_capacitance.clone();
  auto node_load = at::zeros_like(node_capacitance);
  auto node_arrival = at::zeros_like(node_capacitance);
  auto node_slew = at::zeros_like(node_capacitance);
  auto segment_delay = at::zeros_like(z_value);
  auto segment_output_slew = at::zeros_like(z_value);
  auto segment_upstream_cap = at::zeros_like(z_value);

  scalar_t* effective_ptr = effective_node_cap.data_ptr<scalar_t>();
  scalar_t* load_ptr = node_load.data_ptr<scalar_t>();
  scalar_t* arrival_ptr = node_arrival.data_ptr<scalar_t>();
  scalar_t* slew_ptr = node_slew.data_ptr<scalar_t>();
  scalar_t* segment_delay_ptr = segment_delay.data_ptr<scalar_t>();
  scalar_t* segment_slew_ptr = segment_output_slew.data_ptr<scalar_t>();
  scalar_t* segment_upstream_ptr = segment_upstream_cap.data_ptr<scalar_t>();

  const scalar_t* edge_res_ptr = edge_resistance.data_ptr<scalar_t>();
  const scalar_t* edge_cap_ptr = edge_capacitance.data_ptr<scalar_t>();
  const scalar_t* driver_arrival_ptr = driver_arrival.data_ptr<scalar_t>();
  const scalar_t* driver_slew_ptr = driver_slew.data_ptr<scalar_t>();
  const scalar_t* z_ptr = z_value.data_ptr<scalar_t>();
  const scalar_t* input_cap_ptr = buffer_input_cap.data_ptr<scalar_t>();
  const scalar_t* delay_ptr = buffer_delay.data_ptr<scalar_t>();
  const scalar_t* output_slew_ptr = buffer_output_slew.data_ptr<scalar_t>();
  const scalar_t* parent_frac_ptr = parent_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* child_frac_ptr = child_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* sub_frac_ptr = segment_sub_resistance_fraction.data_ptr<scalar_t>();
  std::vector<scalar_t> delay_states(static_cast<size_t>(max_count + 1));
  std::vector<scalar_t> slew_states(static_cast<size_t>(max_count + 1));

  for (int64_t edge = 0; edge < num_edges; ++edge) {
    const int64_t parent = edge_parent_compact_id[static_cast<size_t>(edge)];
    const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
    TORCH_CHECK(parent >= 0 && parent < num_nodes, "edge parent compact id out of range");
    TORCH_CHECK(child >= 0 && child < num_nodes, "edge child compact id out of range");
    const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
    scalar_t parent_fraction = static_cast<scalar_t>(1);
    scalar_t child_fraction = static_cast<scalar_t>(1);
    if (seg >= 0) {
      TORCH_CHECK(seg < num_segments, "edge segment id out of range");
      const scalar_t z = z_ptr[seg];
      parent_fraction = interpolate_count_value<scalar_t>(
          parent_frac_ptr + seg * (max_count + 1), 1, z, max_count);
      child_fraction = interpolate_count_value<scalar_t>(
          child_frac_ptr + seg * (max_count + 1), 1, z, max_count);
    }
    effective_ptr[parent] += static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * parent_fraction;
    effective_ptr[child] += static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * child_fraction;
  }

  const int64_t num_nets = static_cast<int64_t>(net_topo_start.size()) - 1;
  for (int64_t net = 0; net < num_nets; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    for (int64_t compact = end - 1; compact >= begin; --compact) {
      scalar_t children_load = static_cast<scalar_t>(0);
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t edge_input = load_ptr[child];
        if (seg >= 0) {
          const scalar_t z = std::max(
              static_cast<scalar_t>(0),
              std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
          const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
          const int64_t hi = std::min<int64_t>(lo + 1, max_count);
          const scalar_t alpha = z - static_cast<scalar_t>(lo);
          const scalar_t lo_value = (lo == 0) ? load_ptr[child] : input_cap_ptr[seg];
          const scalar_t hi_value = (hi == 0) ? load_ptr[child] : input_cap_ptr[seg];
          edge_input = (static_cast<scalar_t>(1) - alpha) * lo_value + alpha * hi_value;
          segment_upstream_ptr[seg] = edge_input;
        }
        children_load += edge_input;
      }
      load_ptr[compact] = effective_ptr[compact] + children_load;
    }
  }

  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));
  for (int64_t net = 0; net < num_nets; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    if (begin == end) {
      continue;
    }
    arrival_ptr[begin] = driver_arrival_ptr[net];
    slew_ptr[begin] = driver_slew_ptr[net];
    for (int64_t compact = begin; compact < end; ++compact) {
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t edge_delay = static_cast<scalar_t>(0);
        scalar_t child_slew = slew_ptr[compact];
        if (seg < 0) {
          const scalar_t wire_delta = edge_res_ptr[edge] * load_ptr[child];
          edge_delay = wire_delta;
          const scalar_t slew_delta = log10 * wire_delta;
          child_slew = std::sqrt(slew_ptr[compact] * slew_ptr[compact] + slew_delta * slew_delta);
        } else {
          for (int64_t count = 0; count <= max_count; ++count) {
            scalar_t total_delay = static_cast<scalar_t>(0);
            scalar_t slew = slew_ptr[compact];
            const int64_t sub_base = seg * sub_stride_segment + count * sub_stride_count;
            for (int64_t index = 0; index < count; ++index) {
              const scalar_t fraction = sub_frac_ptr[sub_base + index];
              const scalar_t wire_delta = edge_res_ptr[edge] * fraction * input_cap_ptr[seg];
              total_delay += wire_delta + delay_ptr[seg];
              slew = output_slew_ptr[seg];
            }
            const scalar_t final_fraction = sub_frac_ptr[sub_base + count];
            const scalar_t wire_delta = edge_res_ptr[edge] * final_fraction * load_ptr[child];
            total_delay += wire_delta;
            const scalar_t slew_delta = log10 * wire_delta;
            slew = std::sqrt(slew * slew + slew_delta * slew_delta);
            delay_states[static_cast<size_t>(count)] = total_delay;
            slew_states[static_cast<size_t>(count)] = slew;
          }
          const scalar_t z = std::max(
              static_cast<scalar_t>(0),
              std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
          const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
          const int64_t hi = std::min<int64_t>(lo + 1, max_count);
          const scalar_t alpha = z - static_cast<scalar_t>(lo);
          edge_delay = (static_cast<scalar_t>(1) - alpha) * delay_states[static_cast<size_t>(lo)] +
                       alpha * delay_states[static_cast<size_t>(hi)];
          child_slew = (static_cast<scalar_t>(1) - alpha) * slew_states[static_cast<size_t>(lo)] +
                       alpha * slew_states[static_cast<size_t>(hi)];
          segment_delay_ptr[seg] = edge_delay;
          segment_slew_ptr[seg] = child_slew;
        }
        arrival_ptr[child] = arrival_ptr[compact] + edge_delay;
        slew_ptr[child] = child_slew;
      }
    }
  }

  py::dict result;
  result["segment_delay"] = segment_delay;
  result["segment_output_slew"] = segment_output_slew;
  result["segment_upstream_visible_input_cap"] = segment_upstream_cap;
  if (!sink_node_compact_id.empty()) {
    auto sink_index = at::empty({static_cast<int64_t>(sink_node_compact_id.size())}, at::TensorOptions().dtype(at::kLong));
    int64_t* sink_index_ptr = sink_index.data_ptr<int64_t>();
    for (int64_t i = 0; i < static_cast<int64_t>(sink_node_compact_id.size()); ++i) {
      sink_index_ptr[i] = sink_node_compact_id[static_cast<size_t>(i)];
    }
    result["sink_arrival"] = node_arrival.index_select(0, sink_index);
    result["sink_slew"] = node_slew.index_select(0, sink_index);
    result["sink_load"] = node_load.index_select(0, sink_index);
    result["sink_node_id"] = at::empty({static_cast<int64_t>(sink_node_id.size())}, at::TensorOptions().dtype(at::kLong));
    result["sink_net_index"] = sink_net_index;
  }
  result["node_load"] = node_load;
  result["node_arrival"] = node_arrival;
  result["node_slew"] = node_slew;
  result["effective_node_cap"] = effective_node_cap;
  return result;
}

py::dict segment_count_forward(
    at::Tensor net_topo_start,
    at::Tensor flat_topo_node_id,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor node_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor z_value,
    at::Tensor buffer_input_cap,
    at::Tensor buffer_delay,
    at::Tensor buffer_output_slew,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor segment_sub_resistance_fraction,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index,
    at::Tensor sink_node_compact_id) {
  const auto scalar_type = node_capacitance.scalar_type();
  check_float_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_like(node_capacitance, "node_capacitance", scalar_type);
  check_float_like(driver_arrival, "driver_arrival", scalar_type);
  check_float_like(driver_slew, "driver_slew", scalar_type);
  check_float_like(z_value, "z_value", scalar_type);
  check_float_like(buffer_input_cap, "buffer_input_cap", scalar_type);
  check_float_like(buffer_delay, "buffer_delay", scalar_type);
  check_float_like(buffer_output_slew, "buffer_output_slew", scalar_type);
  TORCH_CHECK(parent_cap_fraction.device().is_cpu(), "parent_cap_fraction must be CPU");
  TORCH_CHECK(child_cap_fraction.device().is_cpu(), "child_cap_fraction must be CPU");
  TORCH_CHECK(segment_sub_resistance_fraction.device().is_cpu(),
              "segment_sub_resistance_fraction must be CPU");
  TORCH_CHECK(parent_cap_fraction.scalar_type() == scalar_type,
              "parent_cap_fraction dtype mismatch");
  TORCH_CHECK(child_cap_fraction.scalar_type() == scalar_type,
              "child_cap_fraction dtype mismatch");
  TORCH_CHECK(segment_sub_resistance_fraction.scalar_type() == scalar_type,
              "segment_sub_resistance_fraction dtype mismatch");

  const std::vector<int64_t> topo_start = tensor_to_i64_vector(net_topo_start, "net_topo_start");
  tensor_to_i64_vector(flat_topo_node_id, "flat_topo_node_id");
  const std::vector<int64_t> child_start = tensor_to_i64_vector(edge_start, "edge_start");
  const std::vector<int64_t> parents =
      tensor_to_i64_vector(edge_parent_compact_id, "edge_parent_compact_id");
  const std::vector<int64_t> children =
      tensor_to_i64_vector(edge_child_compact_id, "edge_child_compact_id");
  const std::vector<int64_t> edge_segment =
      tensor_to_i64_vector(edge_to_segment_id, "edge_to_segment_id");
  const std::vector<int64_t> sinks = tensor_to_i64_vector(sink_node_id, "sink_node_id");
  tensor_to_i64_vector(sink_net_index, "sink_net_index");
  const std::vector<int64_t> sink_compact =
      tensor_to_i64_vector(sink_node_compact_id, "sink_node_compact_id");

  py::dict result;
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segment_count_forward", [&] {
    result = segment_count_forward_impl<scalar_t>(
        topo_start,
        child_start,
        parents,
        children,
        edge_resistance,
        edge_capacitance,
        node_capacitance,
        edge_segment,
        driver_arrival,
        driver_slew,
        z_value,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        parent_cap_fraction,
        child_cap_fraction,
        segment_sub_resistance_fraction,
        sinks,
        sink_net_index,
        sink_compact);
  });
  result["sink_node_id"] = sink_node_id;
  return result;
}

#include "segment_transfer_chain.h"

template <typename scalar_t>
py::dict segment_count_transfer_forward_impl(
    const std::vector<int64_t>& net_topo_start,
    const std::vector<int64_t>& edge_start,
    const std::vector<int64_t>& edge_parent_compact_id,
    const std::vector<int64_t>& edge_child_compact_id,
    const at::Tensor& edge_resistance,
    const at::Tensor& edge_capacitance,
    const at::Tensor& node_capacitance,
    const std::vector<int64_t>& edge_to_segment_id,
    const at::Tensor& driver_arrival,
    const at::Tensor& driver_slew,
    const at::Tensor& z_value,
    const at::Tensor& bsu_index,
    const at::Tensor& load_input_cap,
    const at::Tensor& buffer_input_cap_by_size,
    const at::Tensor& buffer_slew_axis,
    const at::Tensor& buffer_load_axis,
    const at::Tensor& buffer_delay_lut,
    const at::Tensor& buffer_output_slew_lut,
    const at::Tensor& parent_cap_fraction,
    const at::Tensor& child_cap_fraction,
    const at::Tensor& segment_sub_resistance_fraction,
    const at::Tensor& segment_retained_upstream_cap,
    const std::vector<int64_t>& sink_node_id,
    const at::Tensor& sink_net_index,
    const std::vector<int64_t>& sink_node_compact_id,
    const at::Tensor& buffer_slew_limits,
    const at::Tensor& buffer_cap_limits,
    const bool count_gradient_enabled) {
  const int64_t num_nodes = node_capacitance.numel();
  const int64_t num_edges = edge_resistance.numel();
  const int64_t num_segments = z_value.numel();
  const int64_t max_count = parent_cap_fraction.size(1) - 1;
  const int64_t sub_stride_count = max_count + 1;
  const int64_t sub_stride_segment = sub_stride_count * sub_stride_count;
  const int64_t size_count = buffer_input_cap_by_size.numel();
  const int64_t slew_count = buffer_slew_axis.numel();
  const int64_t load_count = buffer_load_axis.numel();

  auto effective_node_cap = node_capacitance.clone();
  auto node_load = at::zeros_like(node_capacitance);
  auto node_arrival = at::zeros_like(node_capacitance);
  auto node_slew = at::zeros_like(node_capacitance);
  auto segment_delay = at::zeros_like(z_value);
  auto segment_output_slew = at::zeros_like(z_value);
  auto segment_upstream_cap = at::zeros_like(z_value);
  auto buffer_slew_violation = at::zeros({num_segments, max_count}, z_value.options());
  auto buffer_cap_violation = at::zeros_like(buffer_slew_violation);

  scalar_t* effective_ptr = effective_node_cap.data_ptr<scalar_t>();
  scalar_t* load_ptr = node_load.data_ptr<scalar_t>();
  scalar_t* arrival_ptr = node_arrival.data_ptr<scalar_t>();
  scalar_t* slew_ptr = node_slew.data_ptr<scalar_t>();
  scalar_t* segment_delay_ptr = segment_delay.data_ptr<scalar_t>();
  scalar_t* segment_slew_ptr = segment_output_slew.data_ptr<scalar_t>();
  scalar_t* segment_upstream_ptr = segment_upstream_cap.data_ptr<scalar_t>();
  scalar_t* segment_buffer_slew_ptr = buffer_slew_violation.data_ptr<scalar_t>();
  scalar_t* segment_buffer_cap_ptr = buffer_cap_violation.data_ptr<scalar_t>();
  const scalar_t* slew_limit_ptr = buffer_slew_limits.data_ptr<scalar_t>();
  const scalar_t* cap_limit_ptr = buffer_cap_limits.data_ptr<scalar_t>();

  const scalar_t* edge_res_ptr = edge_resistance.data_ptr<scalar_t>();
  const scalar_t* edge_cap_ptr = edge_capacitance.data_ptr<scalar_t>();
  const scalar_t* driver_arrival_ptr = driver_arrival.data_ptr<scalar_t>();
  const scalar_t* driver_slew_ptr = driver_slew.data_ptr<scalar_t>();
  const scalar_t* z_ptr = z_value.data_ptr<scalar_t>();
  const scalar_t* bsu_ptr = bsu_index.data_ptr<scalar_t>();
  const scalar_t* load_input_cap_ptr = load_input_cap.data_ptr<scalar_t>();
  const scalar_t* cap_by_size_ptr = buffer_input_cap_by_size.data_ptr<scalar_t>();
  const scalar_t* slew_axis_ptr = buffer_slew_axis.data_ptr<scalar_t>();
  const scalar_t* load_axis_ptr = buffer_load_axis.data_ptr<scalar_t>();
  const scalar_t* delay_lut_ptr = buffer_delay_lut.data_ptr<scalar_t>();
  const scalar_t* output_slew_lut_ptr = buffer_output_slew_lut.data_ptr<scalar_t>();
  const scalar_t* parent_frac_ptr = parent_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* child_frac_ptr = child_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* sub_frac_ptr = segment_sub_resistance_fraction.data_ptr<scalar_t>();
  const scalar_t* retained_ptr = segment_retained_upstream_cap.data_ptr<scalar_t>();
  std::vector<scalar_t> delay_states(static_cast<size_t>(max_count + 1));
  std::vector<scalar_t> slew_states(static_cast<size_t>(max_count + 1));
  std::vector<std::vector<scalar_t>> slew_violation_states(static_cast<size_t>(max_count + 1));
  std::vector<std::vector<scalar_t>> cap_violation_states(static_cast<size_t>(max_count + 1));

  for (int64_t edge = 0; edge < num_edges; ++edge) {
    const int64_t parent = edge_parent_compact_id[static_cast<size_t>(edge)];
    const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
    TORCH_CHECK(parent >= 0 && parent < num_nodes, "edge parent compact id out of range");
    TORCH_CHECK(child >= 0 && child < num_nodes, "edge child compact id out of range");
    const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
    scalar_t parent_fraction = static_cast<scalar_t>(1);
    scalar_t child_fraction = static_cast<scalar_t>(1);
    if (seg >= 0) {
      TORCH_CHECK(seg < num_segments, "edge segment id out of range");
      const scalar_t z = z_ptr[seg];
      parent_fraction = interpolate_count_value<scalar_t>(
          parent_frac_ptr + seg * (max_count + 1), 1, z, max_count);
      child_fraction = interpolate_count_value<scalar_t>(
          child_frac_ptr + seg * (max_count + 1), 1, z, max_count);
    }
    effective_ptr[parent] += static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * parent_fraction;
    effective_ptr[child] += static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * child_fraction;
  }

  const int64_t num_nets = static_cast<int64_t>(net_topo_start.size()) - 1;
  for (int64_t net = 0; net < num_nets; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    for (int64_t compact = end - 1; compact >= begin; --compact) {
      scalar_t children_load = static_cast<scalar_t>(0);
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t edge_input = load_ptr[child];
        if (seg >= 0) {
          const scalar_t z = std::max(
              static_cast<scalar_t>(0),
              std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
          const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
          const int64_t hi = std::min<int64_t>(lo + 1, max_count);
          const scalar_t alpha = z - static_cast<scalar_t>(lo);
          const scalar_t cap = load_input_cap_ptr[seg];
          // The root already owns the first pi section's parent half. The
          // buffer input owns its other half, in addition to the Liberty cap.
          const scalar_t lo_value = (lo == 0) ? load_ptr[child] : cap +
              static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * parent_frac_ptr[seg * (max_count + 1) + lo];
          const scalar_t hi_value = (hi == 0) ? load_ptr[child] : cap +
              static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * parent_frac_ptr[seg * (max_count + 1) + hi];
          edge_input = (static_cast<scalar_t>(1) - alpha) * lo_value + alpha * hi_value;
          segment_upstream_ptr[seg] = edge_input;
        }
        children_load += edge_input;
      }
      load_ptr[compact] = effective_ptr[compact] + children_load;
    }
  }

  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10 * delay;
    return std::sqrt(slew * slew + delta * delta);
  };
  for (int64_t net = 0; net < num_nets; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    if (begin == end) {
      continue;
    }
    arrival_ptr[begin] = driver_arrival_ptr[net];
    slew_ptr[begin] = driver_slew_ptr[net];
    for (int64_t compact = begin; compact < end; ++compact) {
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t edge_delay = static_cast<scalar_t>(0);
        scalar_t child_slew = slew_ptr[compact];
        const scalar_t resistance = edge_res_ptr[edge];
        const scalar_t capacitance = edge_cap_ptr[edge];
        if (seg < 0) {
          const scalar_t wire_delta = resistance * load_ptr[child];
          edge_delay = wire_delta;
          child_slew = wire_slew(slew_ptr[compact], wire_delta);
        } else {
          const scalar_t bsu = bsu_ptr[seg];
          const scalar_t buffer_input_cap =
              interp_size_1d(cap_by_size_ptr, size_count, bsu);
          const scalar_t z = std::max(
              static_cast<scalar_t>(0),
              std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
          const scalar_t child_fraction = interpolate_count_value<scalar_t>(
              child_frac_ptr + seg * (max_count + 1), 1, z, max_count);
          const scalar_t downstream_load =
              load_ptr[child] -
              static_cast<scalar_t>(0.5) * capacitance * child_fraction;
          const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
          const int64_t hi = !count_gradient_enabled && z == static_cast<scalar_t>(lo) ?
              lo : std::min<int64_t>(lo + 1, max_count);
          const scalar_t alpha = z - static_cast<scalar_t>(lo);
          const scalar_t slew_limit = interp_size_1d(slew_limit_ptr, size_count, bsu);
          const scalar_t cap_limit = interp_size_1d(cap_limit_ptr, size_count, bsu);
          // Only B needs adjacent circuits; fixed-count integer GP uses one.
          for (int64_t count = lo; count <= hi; ++count) {
            const int64_t sub_base = seg * sub_stride_segment + count * sub_stride_count;
            const auto state = transfer_state_value_and_grad<scalar_t>(
                slew_ptr[compact], downstream_load, resistance, capacitance,
                count, sub_frac_ptr + sub_base, bsu, cap_by_size_ptr, size_count,
                slew_axis_ptr, slew_count, load_axis_ptr, load_count,
                delay_lut_ptr, output_slew_lut_ptr, 0, 0, slew_limit, cap_limit);
            delay_states[static_cast<size_t>(count)] = state.delay;
            slew_states[static_cast<size_t>(count)] = state.output_slew;
            slew_violation_states[static_cast<size_t>(count)] = state.buffer_slew_violation;
            cap_violation_states[static_cast<size_t>(count)] = state.buffer_cap_violation;
          }
          for (int64_t step = 0; step < max_count; ++step) {
            const scalar_t lo_slew = step < lo ? slew_violation_states[lo][step] : 0;
            const scalar_t hi_slew = step < hi ? slew_violation_states[hi][step] : 0;
            const scalar_t lo_cap = step < lo ? cap_violation_states[lo][step] : 0;
            const scalar_t hi_cap = step < hi ? cap_violation_states[hi][step] : 0;
            segment_buffer_slew_ptr[seg * max_count + step] =
                (static_cast<scalar_t>(1) - alpha) * lo_slew + alpha * hi_slew;
            segment_buffer_cap_ptr[seg * max_count + step] =
                (static_cast<scalar_t>(1) - alpha) * lo_cap + alpha * hi_cap;
          }
          edge_delay = (static_cast<scalar_t>(1) - alpha) * delay_states[static_cast<size_t>(lo)] +
                       alpha * delay_states[static_cast<size_t>(hi)];
          child_slew = (static_cast<scalar_t>(1) - alpha) * slew_states[static_cast<size_t>(lo)] +
                       alpha * slew_states[static_cast<size_t>(hi)];
          segment_delay_ptr[seg] = edge_delay;
          segment_slew_ptr[seg] = child_slew;
        }
        arrival_ptr[child] = arrival_ptr[compact] + edge_delay;
        slew_ptr[child] = child_slew;
      }
    }
  }

  py::dict result;
  result["segment_delay"] = segment_delay;
  result["segment_output_slew"] = segment_output_slew;
  result["segment_upstream_visible_input_cap"] = segment_upstream_cap;
  result["buffer_slew_violation"] = buffer_slew_violation;
  result["buffer_cap_violation"] = buffer_cap_violation;
  if (!sink_node_compact_id.empty()) {
    auto sink_index = at::empty({static_cast<int64_t>(sink_node_compact_id.size())}, at::TensorOptions().dtype(at::kLong));
    int64_t* sink_index_ptr = sink_index.data_ptr<int64_t>();
    for (int64_t i = 0; i < static_cast<int64_t>(sink_node_compact_id.size()); ++i) {
      sink_index_ptr[i] = sink_node_compact_id[static_cast<size_t>(i)];
    }
    result["sink_arrival"] = node_arrival.index_select(0, sink_index);
    result["sink_slew"] = node_slew.index_select(0, sink_index);
    result["sink_load"] = node_load.index_select(0, sink_index);
    result["sink_node_id"] = at::empty({static_cast<int64_t>(sink_node_id.size())}, at::TensorOptions().dtype(at::kLong));
    result["sink_net_index"] = sink_net_index;
  }
  result["node_load"] = node_load;
  result["node_arrival"] = node_arrival;
  result["node_slew"] = node_slew;
  result["effective_node_cap"] = effective_node_cap;
  return result;
}

template <typename scalar_t>
py::dict segment_count_backward_impl(
    const std::vector<int64_t>& net_topo_start,
    const std::vector<int64_t>& edge_start,
    const std::vector<int64_t>& edge_parent_compact_id,
    const std::vector<int64_t>& edge_child_compact_id,
    const at::Tensor& edge_resistance,
    const at::Tensor& edge_capacitance,
    const std::vector<int64_t>& edge_to_segment_id,
    const at::Tensor& driver_arrival,
    const at::Tensor& driver_slew,
    const at::Tensor& z_value,
    const at::Tensor& bsu_index,
    const at::Tensor& per_size_input_cap,
    const at::Tensor& per_size_delay,
    const at::Tensor& per_size_output_slew,
    const at::Tensor& buffer_input_cap,
    const at::Tensor& buffer_delay,
    const at::Tensor& buffer_output_slew,
    const at::Tensor& parent_cap_fraction,
    const at::Tensor& child_cap_fraction,
    const at::Tensor& segment_sub_resistance_fraction,
    const std::vector<int64_t>& sink_node_compact_id,
    const at::Tensor& node_load,
    const at::Tensor& node_slew,
    const at::Tensor& effective_node_cap,
    const at::Tensor& grad_segment_delay,
    const at::Tensor& grad_segment_output_slew,
    const at::Tensor& grad_segment_upstream_cap,
    const at::Tensor& grad_sink_arrival,
    const at::Tensor& grad_sink_slew,
    const at::Tensor& grad_sink_load) {
  const int64_t num_nodes = node_load.numel();
  const int64_t num_segments = z_value.numel();
  const int64_t num_nets = driver_arrival.numel();
  const int64_t max_count = parent_cap_fraction.size(1) - 1;
  const int64_t legal_buffer_count = per_size_delay.size(1);
  const int64_t sub_stride_count = max_count + 1;
  const int64_t sub_stride_segment = sub_stride_count * sub_stride_count;

  auto grad_load = at::zeros_like(node_load);
  auto grad_arrival = at::zeros_like(node_load);
  auto grad_slew = at::zeros_like(node_load);
  auto grad_eff_cap = at::zeros_like(node_load);
  auto grad_z = at::zeros_like(z_value);
  auto grad_input_cap = at::zeros_like(z_value);
  auto grad_buffer_delay_tensor = at::zeros_like(z_value);
  auto grad_output_slew_tensor = at::zeros_like(z_value);
  auto grad_bsu = at::zeros_like(z_value);
  auto grad_driver_arrival = at::zeros_like(driver_arrival);
  auto grad_driver_slew = at::zeros_like(driver_slew);

  scalar_t* grad_load_ptr = grad_load.data_ptr<scalar_t>();
  scalar_t* grad_arrival_ptr = grad_arrival.data_ptr<scalar_t>();
  scalar_t* grad_slew_ptr = grad_slew.data_ptr<scalar_t>();
  scalar_t* grad_eff_cap_ptr = grad_eff_cap.data_ptr<scalar_t>();
  scalar_t* grad_z_ptr = grad_z.data_ptr<scalar_t>();
  scalar_t* grad_input_cap_ptr = grad_input_cap.data_ptr<scalar_t>();
  scalar_t* grad_buffer_delay_ptr = grad_buffer_delay_tensor.data_ptr<scalar_t>();
  scalar_t* grad_output_slew_ptr = grad_output_slew_tensor.data_ptr<scalar_t>();
  scalar_t* grad_bsu_ptr = grad_bsu.data_ptr<scalar_t>();
  scalar_t* grad_driver_arrival_ptr = grad_driver_arrival.data_ptr<scalar_t>();
  scalar_t* grad_driver_slew_ptr = grad_driver_slew.data_ptr<scalar_t>();

  const scalar_t* edge_res_ptr = edge_resistance.data_ptr<scalar_t>();
  const scalar_t* edge_cap_ptr = edge_capacitance.data_ptr<scalar_t>();
  const scalar_t* z_ptr = z_value.data_ptr<scalar_t>();
  const scalar_t* bsu_ptr = bsu_index.data_ptr<scalar_t>();
  const scalar_t* per_input_ptr = per_size_input_cap.data_ptr<scalar_t>();
  const scalar_t* per_delay_ptr = per_size_delay.data_ptr<scalar_t>();
  const scalar_t* per_slew_ptr = per_size_output_slew.data_ptr<scalar_t>();
  const scalar_t* input_cap_ptr = buffer_input_cap.data_ptr<scalar_t>();
  const scalar_t* delay_ptr = buffer_delay.data_ptr<scalar_t>();
  const scalar_t* output_slew_ptr = buffer_output_slew.data_ptr<scalar_t>();
  const scalar_t* parent_frac_ptr = parent_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* child_frac_ptr = child_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* sub_frac_ptr = segment_sub_resistance_fraction.data_ptr<scalar_t>();
  const scalar_t* load_ptr = node_load.data_ptr<scalar_t>();
  const scalar_t* slew_ptr = node_slew.data_ptr<scalar_t>();
  const scalar_t* grad_seg_delay_ptr = grad_segment_delay.data_ptr<scalar_t>();
  const scalar_t* grad_seg_slew_ptr = grad_segment_output_slew.data_ptr<scalar_t>();
  const scalar_t* grad_seg_upstream_ptr = grad_segment_upstream_cap.data_ptr<scalar_t>();
  const scalar_t* grad_sink_arrival_ptr = grad_sink_arrival.data_ptr<scalar_t>();
  const scalar_t* grad_sink_slew_ptr = grad_sink_slew.data_ptr<scalar_t>();
  const scalar_t* grad_sink_load_ptr = grad_sink_load.data_ptr<scalar_t>();
  std::vector<scalar_t> delay_states(static_cast<size_t>(max_count + 1));
  std::vector<scalar_t> slew_states(static_cast<size_t>(max_count + 1));

  for (int64_t offset = 0; offset < static_cast<int64_t>(sink_node_compact_id.size()); ++offset) {
    const int64_t compact = sink_node_compact_id[static_cast<size_t>(offset)];
    TORCH_CHECK(compact >= 0 && compact < num_nodes, "sink compact id out of range");
    grad_arrival_ptr[compact] += grad_sink_arrival_ptr[offset];
    grad_slew_ptr[compact] += grad_sink_slew_ptr[offset];
    grad_load_ptr[compact] += grad_sink_load_ptr[offset];
  }

  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));
  const scalar_t epsilon = static_cast<scalar_t>(1e-30);
  for (int64_t net = 0; net < static_cast<int64_t>(net_topo_start.size()) - 1; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    if (begin == end) {
      continue;
    }
    for (int64_t compact = end - 1; compact >= begin; --compact) {
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t grad_edge_delay = grad_arrival_ptr[child];
        scalar_t grad_child_slew = grad_slew_ptr[child];
        grad_arrival_ptr[compact] += grad_arrival_ptr[child];
        if (seg >= 0) {
          grad_edge_delay += grad_seg_delay_ptr[seg];
          grad_child_slew += grad_seg_slew_ptr[seg];
        }
        const scalar_t resistance = edge_res_ptr[edge];
        const scalar_t child_load = load_ptr[child];
        const scalar_t parent_slew = slew_ptr[compact];
        if (seg < 0) {
          const scalar_t wire_delta = resistance * child_load;
          const scalar_t child_slew_value = std::max(slew_ptr[child], epsilon);
          grad_load_ptr[child] += grad_edge_delay * resistance;
          grad_slew_ptr[compact] += grad_child_slew * parent_slew / child_slew_value;
          const scalar_t grad_wire =
              grad_child_slew * (log10 * log10) * wire_delta / child_slew_value;
          grad_load_ptr[child] += grad_wire * resistance;
          continue;
        }

        const scalar_t z = std::max(
            static_cast<scalar_t>(0),
            std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
        const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
        const int64_t hi = std::min<int64_t>(lo + 1, max_count);
        const scalar_t alpha = z - static_cast<scalar_t>(lo);
        const scalar_t input_cap = input_cap_ptr[seg];
        const scalar_t delay = delay_ptr[seg];
        const scalar_t output_slew = output_slew_ptr[seg];

        for (int64_t count = 0; count <= max_count; ++count) {
          scalar_t total_delay = static_cast<scalar_t>(0);
          const int64_t sub_base = seg * sub_stride_segment + count * sub_stride_count;
          scalar_t repeated_fraction_sum = static_cast<scalar_t>(0);
          for (int64_t index = 0; index < count; ++index) {
            const scalar_t fraction = sub_frac_ptr[sub_base + index];
            repeated_fraction_sum += fraction;
            total_delay += resistance * fraction * input_cap + delay;
          }
          const scalar_t final_fraction = sub_frac_ptr[sub_base + count];
          const scalar_t final_wire = resistance * final_fraction * child_load;
          total_delay += final_wire;
          delay_states[static_cast<size_t>(count)] = total_delay;
          const scalar_t base_slew = (count == 0) ? parent_slew : output_slew;
          slew_states[static_cast<size_t>(count)] = std::sqrt(
              base_slew * base_slew + (log10 * final_wire) * (log10 * final_wire));
        }

        grad_z_ptr[seg] += grad_edge_delay *
                           (delay_states[static_cast<size_t>(hi)] -
                            delay_states[static_cast<size_t>(lo)]);
        grad_z_ptr[seg] += grad_child_slew *
                           (slew_states[static_cast<size_t>(hi)] -
                            slew_states[static_cast<size_t>(lo)]);

        for (int64_t count_index = 0; count_index < 2; ++count_index) {
          const int64_t count = (count_index == 0) ? lo : hi;
          const scalar_t weight =
              (count_index == 0) ? (static_cast<scalar_t>(1) - alpha) : alpha;
          if (weight == static_cast<scalar_t>(0)) {
            continue;
          }
          const int64_t sub_base = seg * sub_stride_segment + count * sub_stride_count;
          scalar_t repeated_fraction_sum = static_cast<scalar_t>(0);
          for (int64_t index = 0; index < count; ++index) {
            repeated_fraction_sum += sub_frac_ptr[sub_base + index];
          }
          const scalar_t final_fraction = sub_frac_ptr[sub_base + count];
          const scalar_t final_wire = resistance * final_fraction * child_load;
          const scalar_t grad_delay_state = grad_edge_delay * weight;
          if (count == 0) {
            grad_load_ptr[child] += grad_delay_state * resistance * final_fraction;
          } else {
            grad_input_cap_ptr[seg] +=
                grad_delay_state * resistance * repeated_fraction_sum;
            grad_buffer_delay_ptr[seg] +=
                grad_delay_state * static_cast<scalar_t>(count);
            grad_load_ptr[child] += grad_delay_state * resistance * final_fraction;
          }

          const scalar_t grad_slew_state = grad_child_slew * weight;
          const scalar_t slew_value =
              std::max(slew_states[static_cast<size_t>(count)], epsilon);
          if (count == 0) {
            grad_slew_ptr[compact] += grad_slew_state * parent_slew / slew_value;
          } else {
            grad_output_slew_ptr[seg] += grad_slew_state * output_slew / slew_value;
          }
          const scalar_t grad_wire =
              grad_slew_state * (log10 * log10) * final_wire / slew_value;
          grad_load_ptr[child] += grad_wire * resistance * final_fraction;
        }
      }
    }
    grad_driver_arrival_ptr[net] += grad_arrival_ptr[begin];
    grad_driver_slew_ptr[net] += grad_slew_ptr[begin];
  }

  for (int64_t net = 0; net < static_cast<int64_t>(net_topo_start.size()) - 1; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    for (int64_t compact = begin; compact < end; ++compact) {
      grad_eff_cap_ptr[compact] += grad_load_ptr[compact];
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t grad_edge_input = grad_load_ptr[compact];
        if (seg >= 0) {
          grad_edge_input += grad_seg_upstream_ptr[seg];
        }
        if (seg < 0) {
          grad_load_ptr[child] += grad_edge_input;
          continue;
        }
        const scalar_t z = std::max(
            static_cast<scalar_t>(0),
            std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
        const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
        const int64_t hi = std::min<int64_t>(lo + 1, max_count);
        const scalar_t alpha = z - static_cast<scalar_t>(lo);
        const scalar_t input_cap = input_cap_ptr[seg];
        const scalar_t child_load = load_ptr[child];
        const scalar_t lo_value = (lo == 0) ? child_load : input_cap;
        const scalar_t hi_value = (hi == 0) ? child_load : input_cap;
        grad_z_ptr[seg] += grad_edge_input * (hi_value - lo_value);
        if (lo == 0) {
          grad_load_ptr[child] +=
              grad_edge_input * (static_cast<scalar_t>(1) - alpha);
        } else {
          grad_input_cap_ptr[seg] +=
              grad_edge_input * (static_cast<scalar_t>(1) - alpha);
        }
        if (hi == 0) {
          grad_load_ptr[child] += grad_edge_input * alpha;
        } else {
          grad_input_cap_ptr[seg] += grad_edge_input * alpha;
        }
      }
    }
  }

  for (int64_t edge = 0; edge < static_cast<int64_t>(edge_to_segment_id.size()); ++edge) {
    const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
    if (seg < 0) {
      continue;
    }
    const int64_t parent = edge_parent_compact_id[static_cast<size_t>(edge)];
    const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
    const scalar_t z = std::max(
        static_cast<scalar_t>(0),
        std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
    const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
    const int64_t hi = std::min<int64_t>(lo + 1, max_count);
    const scalar_t parent_lo = parent_frac_ptr[seg * (max_count + 1) + lo];
    const scalar_t parent_hi = parent_frac_ptr[seg * (max_count + 1) + hi];
    const scalar_t child_lo = child_frac_ptr[seg * (max_count + 1) + lo];
    const scalar_t child_hi = child_frac_ptr[seg * (max_count + 1) + hi];
    const scalar_t cap = edge_cap_ptr[edge];
    grad_z_ptr[seg] += grad_eff_cap_ptr[parent] * static_cast<scalar_t>(0.5) * cap *
                       (parent_hi - parent_lo);
    grad_z_ptr[seg] += grad_eff_cap_ptr[child] * static_cast<scalar_t>(0.5) * cap *
                       (child_hi - child_lo);
  }

  const int64_t table_stride = legal_buffer_count;
  for (int64_t seg = 0; seg < num_segments; ++seg) {
    scalar_t bsu = std::max(
        static_cast<scalar_t>(0),
        std::min(bsu_ptr[seg], static_cast<scalar_t>(legal_buffer_count - 1)));
    const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(bsu)));
    const int64_t hi = std::min<int64_t>(lo + 1, legal_buffer_count - 1);
    grad_bsu_ptr[seg] += grad_input_cap_ptr[seg] *
        (per_input_ptr[seg * table_stride + hi] - per_input_ptr[seg * table_stride + lo]);
    grad_bsu_ptr[seg] += grad_buffer_delay_ptr[seg] *
        (per_delay_ptr[seg * table_stride + hi] - per_delay_ptr[seg * table_stride + lo]);
    grad_bsu_ptr[seg] += grad_output_slew_ptr[seg] *
        (per_slew_ptr[seg * table_stride + hi] - per_slew_ptr[seg * table_stride + lo]);
    if (z_ptr[seg] < static_cast<scalar_t>(0) ||
        z_ptr[seg] > static_cast<scalar_t>(max_count)) {
      grad_z_ptr[seg] = static_cast<scalar_t>(0);
    }
    if (bsu_ptr[seg] < static_cast<scalar_t>(0) ||
        bsu_ptr[seg] > static_cast<scalar_t>(legal_buffer_count - 1)) {
      grad_bsu_ptr[seg] = static_cast<scalar_t>(0);
    }
  }

  py::dict result;
  result["grad_z"] = grad_z;
  result["grad_bsu"] = grad_bsu;
  result["grad_driver_arrival"] = grad_driver_arrival;
  result["grad_driver_slew"] = grad_driver_slew;
  return result;
}

py::dict segment_count_backward(
    at::Tensor net_topo_start,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor z_value,
    at::Tensor bsu_index,
    at::Tensor per_size_input_cap,
    at::Tensor per_size_delay,
    at::Tensor per_size_output_slew,
    at::Tensor buffer_input_cap,
    at::Tensor buffer_delay,
    at::Tensor buffer_output_slew,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor segment_sub_resistance_fraction,
    at::Tensor sink_node_compact_id,
    at::Tensor node_load,
    at::Tensor node_slew,
    at::Tensor effective_node_cap,
    at::Tensor grad_segment_delay,
    at::Tensor grad_segment_output_slew,
    at::Tensor grad_segment_upstream_cap,
    at::Tensor grad_sink_arrival,
    at::Tensor grad_sink_slew,
    at::Tensor grad_sink_load) {
  const auto scalar_type = node_load.scalar_type();
  check_float_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_like(driver_arrival, "driver_arrival", scalar_type);
  check_float_like(driver_slew, "driver_slew", scalar_type);
  check_float_like(z_value, "z_value", scalar_type);
  check_float_like(bsu_index, "bsu_index", scalar_type);
  TORCH_CHECK(per_size_input_cap.device().is_cpu(), "per_size_input_cap must be CPU");
  TORCH_CHECK(per_size_input_cap.dim() == 2, "per_size_input_cap must be 2-D");
  TORCH_CHECK(per_size_delay.device().is_cpu(), "per_size_delay must be CPU");
  TORCH_CHECK(per_size_delay.dim() == 2, "per_size_delay must be 2-D");
  TORCH_CHECK(per_size_output_slew.device().is_cpu(), "per_size_output_slew must be CPU");
  TORCH_CHECK(per_size_output_slew.dim() == 2, "per_size_output_slew must be 2-D");
  TORCH_CHECK(per_size_input_cap.scalar_type() == scalar_type, "per_size_input_cap dtype mismatch");
  TORCH_CHECK(per_size_delay.scalar_type() == scalar_type, "per_size_delay dtype mismatch");
  TORCH_CHECK(per_size_output_slew.scalar_type() == scalar_type, "per_size_output_slew dtype mismatch");
  check_float_like(buffer_input_cap, "buffer_input_cap", scalar_type);
  check_float_like(buffer_delay, "buffer_delay", scalar_type);
  check_float_like(buffer_output_slew, "buffer_output_slew", scalar_type);
  TORCH_CHECK(parent_cap_fraction.device().is_cpu(), "parent_cap_fraction must be CPU");
  TORCH_CHECK(child_cap_fraction.device().is_cpu(), "child_cap_fraction must be CPU");
  TORCH_CHECK(segment_sub_resistance_fraction.device().is_cpu(),
              "segment_sub_resistance_fraction must be CPU");
  TORCH_CHECK(parent_cap_fraction.scalar_type() == scalar_type,
              "parent_cap_fraction dtype mismatch");
  TORCH_CHECK(child_cap_fraction.scalar_type() == scalar_type,
              "child_cap_fraction dtype mismatch");
  TORCH_CHECK(segment_sub_resistance_fraction.scalar_type() == scalar_type,
              "segment_sub_resistance_fraction dtype mismatch");
  check_float_like(node_load, "node_load", scalar_type);
  check_float_like(node_slew, "node_slew", scalar_type);
  check_float_like(effective_node_cap, "effective_node_cap", scalar_type);
  check_float_like(grad_segment_delay, "grad_segment_delay", scalar_type);
  check_float_like(grad_segment_output_slew, "grad_segment_output_slew", scalar_type);
  check_float_like(grad_segment_upstream_cap, "grad_segment_upstream_cap", scalar_type);
  check_float_like(grad_sink_arrival, "grad_sink_arrival", scalar_type);
  check_float_like(grad_sink_slew, "grad_sink_slew", scalar_type);
  check_float_like(grad_sink_load, "grad_sink_load", scalar_type);

  const std::vector<int64_t> topo_start = tensor_to_i64_vector(net_topo_start, "net_topo_start");
  const std::vector<int64_t> child_start = tensor_to_i64_vector(edge_start, "edge_start");
  const std::vector<int64_t> parents =
      tensor_to_i64_vector(edge_parent_compact_id, "edge_parent_compact_id");
  const std::vector<int64_t> children =
      tensor_to_i64_vector(edge_child_compact_id, "edge_child_compact_id");
  const std::vector<int64_t> edge_segment =
      tensor_to_i64_vector(edge_to_segment_id, "edge_to_segment_id");
  const std::vector<int64_t> sink_compact =
      tensor_to_i64_vector(sink_node_compact_id, "sink_node_compact_id");

  py::dict result;
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segment_count_backward", [&] {
    result = segment_count_backward_impl<scalar_t>(
        topo_start,
        child_start,
        parents,
        children,
        edge_resistance,
        edge_capacitance,
        edge_segment,
        driver_arrival,
        driver_slew,
        z_value,
        bsu_index,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        parent_cap_fraction,
        child_cap_fraction,
        segment_sub_resistance_fraction,
        sink_compact,
        node_load,
        node_slew,
        effective_node_cap,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_cap,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load);
  });
  return result;
}


template <typename scalar_t>
py::dict segment_count_transfer_backward_impl(
    const std::vector<int64_t>& net_topo_start,
    const std::vector<int64_t>& edge_start,
    const std::vector<int64_t>& edge_parent_compact_id,
    const std::vector<int64_t>& edge_child_compact_id,
    const at::Tensor& edge_resistance,
    const at::Tensor& edge_capacitance,
    const std::vector<int64_t>& edge_to_segment_id,
    const at::Tensor& driver_arrival,
    const at::Tensor& driver_slew,
    const at::Tensor& z_value,
    const at::Tensor& bsu_index,
    const at::Tensor& load_input_cap,
    const at::Tensor& buffer_input_cap_by_size,
    const at::Tensor& buffer_slew_axis,
    const at::Tensor& buffer_load_axis,
    const at::Tensor& buffer_delay_lut,
    const at::Tensor& buffer_output_slew_lut,
    const at::Tensor& parent_cap_fraction,
    const at::Tensor& child_cap_fraction,
    const at::Tensor& segment_sub_resistance_fraction,
    const std::vector<int64_t>& sink_node_compact_id,
    const at::Tensor& node_load,
    const at::Tensor& node_slew,
    const at::Tensor& effective_node_cap,
    const at::Tensor& grad_segment_delay,
    const at::Tensor& grad_segment_output_slew,
    const at::Tensor& grad_segment_upstream_cap,
    const at::Tensor& grad_sink_arrival,
    const at::Tensor& grad_sink_slew,
    const at::Tensor& grad_sink_load,
    const at::Tensor& buffer_slew_limits,
    const at::Tensor& buffer_cap_limits,
    const at::Tensor& grad_buffer_slew_violation,
    const at::Tensor& grad_buffer_cap_violation,
    const bool count_gradient_enabled) {
  const int64_t num_nodes = node_load.numel();
  const int64_t num_segments = z_value.numel();
  const int64_t max_count = parent_cap_fraction.size(1) - 1;
  const int64_t size_count = buffer_input_cap_by_size.numel();
  const int64_t slew_count = buffer_slew_axis.numel();
  const int64_t load_count = buffer_load_axis.numel();
  const int64_t sub_stride_count = max_count + 1;
  const int64_t sub_stride_segment = sub_stride_count * sub_stride_count;

  auto grad_load = at::zeros_like(node_load);
  auto grad_arrival = at::zeros_like(node_load);
  auto grad_slew = at::zeros_like(node_load);
  auto grad_eff_cap = at::zeros_like(node_load);
  auto grad_z = at::zeros_like(z_value);
  auto grad_input_cap = at::zeros_like(z_value);
  auto grad_bsu = at::zeros_like(z_value);
  auto grad_driver_arrival = at::zeros_like(driver_arrival);
  auto grad_driver_slew = at::zeros_like(driver_slew);
  auto grad_edge_resistance = at::zeros_like(edge_resistance);
  auto grad_edge_capacitance = at::zeros_like(edge_capacitance);

  scalar_t* grad_load_ptr = grad_load.data_ptr<scalar_t>();
  scalar_t* grad_arrival_ptr = grad_arrival.data_ptr<scalar_t>();
  scalar_t* grad_slew_ptr = grad_slew.data_ptr<scalar_t>();
  scalar_t* grad_eff_cap_ptr = grad_eff_cap.data_ptr<scalar_t>();
  scalar_t* grad_z_ptr = grad_z.data_ptr<scalar_t>();
  scalar_t* grad_input_cap_ptr = grad_input_cap.data_ptr<scalar_t>();
  scalar_t* grad_bsu_ptr = grad_bsu.data_ptr<scalar_t>();
  scalar_t* grad_driver_arrival_ptr = grad_driver_arrival.data_ptr<scalar_t>();
  scalar_t* grad_driver_slew_ptr = grad_driver_slew.data_ptr<scalar_t>();
  scalar_t* grad_edge_res_ptr = grad_edge_resistance.data_ptr<scalar_t>();
  scalar_t* grad_edge_cap_ptr = grad_edge_capacitance.data_ptr<scalar_t>();

  const scalar_t* edge_res_ptr = edge_resistance.data_ptr<scalar_t>();
  const scalar_t* edge_cap_ptr = edge_capacitance.data_ptr<scalar_t>();
  const scalar_t* z_ptr = z_value.data_ptr<scalar_t>();
  const scalar_t* bsu_ptr = bsu_index.data_ptr<scalar_t>();
  const scalar_t* load_input_cap_ptr = load_input_cap.data_ptr<scalar_t>();
  const scalar_t* cap_by_size_ptr = buffer_input_cap_by_size.data_ptr<scalar_t>();
  const scalar_t* slew_axis_ptr = buffer_slew_axis.data_ptr<scalar_t>();
  const scalar_t* load_axis_ptr = buffer_load_axis.data_ptr<scalar_t>();
  const scalar_t* delay_lut_ptr = buffer_delay_lut.data_ptr<scalar_t>();
  const scalar_t* output_slew_lut_ptr = buffer_output_slew_lut.data_ptr<scalar_t>();
  const scalar_t* parent_frac_ptr = parent_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* child_frac_ptr = child_cap_fraction.data_ptr<scalar_t>();
  const scalar_t* sub_frac_ptr = segment_sub_resistance_fraction.data_ptr<scalar_t>();
  const scalar_t* load_ptr = node_load.data_ptr<scalar_t>();
  const scalar_t* slew_ptr = node_slew.data_ptr<scalar_t>();
  const scalar_t* grad_seg_delay_ptr = grad_segment_delay.data_ptr<scalar_t>();
  const scalar_t* grad_seg_slew_ptr = grad_segment_output_slew.data_ptr<scalar_t>();
  const scalar_t* grad_seg_upstream_ptr = grad_segment_upstream_cap.data_ptr<scalar_t>();
  const scalar_t* grad_sink_arrival_ptr = grad_sink_arrival.data_ptr<scalar_t>();
  const scalar_t* grad_sink_slew_ptr = grad_sink_slew.data_ptr<scalar_t>();
  const scalar_t* grad_sink_load_ptr = grad_sink_load.data_ptr<scalar_t>();
  const scalar_t* slew_limit_ptr = buffer_slew_limits.data_ptr<scalar_t>();
  const scalar_t* cap_limit_ptr = buffer_cap_limits.data_ptr<scalar_t>();
  const scalar_t* grad_buffer_slew_ptr = grad_buffer_slew_violation.data_ptr<scalar_t>();
  const scalar_t* grad_buffer_cap_ptr = grad_buffer_cap_violation.data_ptr<scalar_t>();
  std::vector<scalar_t> delay_states(static_cast<size_t>(max_count + 1));
  std::vector<scalar_t> slew_states(static_cast<size_t>(max_count + 1));
  std::vector<std::vector<scalar_t>> slew_violation_states(static_cast<size_t>(max_count + 1));
  std::vector<std::vector<scalar_t>> cap_violation_states(static_cast<size_t>(max_count + 1));

  for (int64_t offset = 0; offset < static_cast<int64_t>(sink_node_compact_id.size()); ++offset) {
    const int64_t compact = sink_node_compact_id[static_cast<size_t>(offset)];
    TORCH_CHECK(compact >= 0 && compact < num_nodes, "sink compact id out of range");
    grad_arrival_ptr[compact] += grad_sink_arrival_ptr[offset];
    grad_slew_ptr[compact] += grad_sink_slew_ptr[offset];
    grad_load_ptr[compact] += grad_sink_load_ptr[offset];
  }

  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));
  const scalar_t epsilon = static_cast<scalar_t>(1e-30);
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10 * delay;
    return std::sqrt(slew * slew + delta * delta);
  };

  for (int64_t net = 0; net < static_cast<int64_t>(net_topo_start.size()) - 1; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    if (begin == end) {
      continue;
    }
    for (int64_t compact = end - 1; compact >= begin; --compact) {
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t grad_edge_delay = grad_arrival_ptr[child];
        scalar_t grad_child_slew = grad_slew_ptr[child];
        grad_arrival_ptr[compact] += grad_arrival_ptr[child];
        if (seg >= 0) {
          grad_edge_delay += grad_seg_delay_ptr[seg];
          grad_child_slew += grad_seg_slew_ptr[seg];
        }
        const scalar_t resistance = edge_res_ptr[edge];
        const scalar_t child_load = load_ptr[child];
        const scalar_t parent_slew = slew_ptr[compact];
        if (seg < 0) {
          const scalar_t wire_delta = resistance * child_load;
          const scalar_t child_slew_value = std::max(slew_ptr[child], epsilon);
          scalar_t grad_wire_delay = grad_edge_delay;
          grad_slew_ptr[compact] += grad_child_slew * parent_slew / child_slew_value;
          grad_wire_delay +=
              grad_child_slew * (log10 * log10) * wire_delta / child_slew_value;
          grad_load_ptr[child] += grad_wire_delay * resistance;
          grad_edge_res_ptr[edge] += grad_wire_delay * child_load;
          continue;
        }

        const scalar_t z = std::max(
            static_cast<scalar_t>(0),
            std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
        const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
        const int64_t hi = !count_gradient_enabled && z == static_cast<scalar_t>(lo) ?
            lo : std::min<int64_t>(lo + 1, max_count);
        const scalar_t alpha = z - static_cast<scalar_t>(lo);
        const scalar_t child_lo =
            child_frac_ptr[seg * (max_count + 1) + lo];
        const scalar_t child_hi =
            child_frac_ptr[seg * (max_count + 1) + hi];
        const scalar_t child_fraction =
            (static_cast<scalar_t>(1) - alpha) * child_lo + alpha * child_hi;
        const scalar_t downstream_load =
            child_load - static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] *
                             child_fraction;
        const scalar_t bsu = bsu_ptr[seg];
        const scalar_t slew_limit = interp_size_1d(slew_limit_ptr, size_count, bsu);
        const scalar_t cap_limit = interp_size_1d(cap_limit_ptr, size_count, bsu);
        const int64_t sub_base_zero = seg * sub_stride_segment;
        for (int64_t count = lo; count <= hi; ++count) {
          const int64_t sub_base = sub_base_zero + count * sub_stride_count;
          auto state = transfer_state_value_and_grad<scalar_t>(
              parent_slew,
              downstream_load,
              resistance,
              edge_cap_ptr[edge],
              count,
              sub_frac_ptr + sub_base,
              bsu,
              cap_by_size_ptr,
              size_count,
              slew_axis_ptr,
              slew_count,
              load_axis_ptr,
              load_count,
              delay_lut_ptr,
              output_slew_lut_ptr,
              static_cast<scalar_t>(0),
              static_cast<scalar_t>(0), slew_limit, cap_limit);
          delay_states[static_cast<size_t>(count)] = state.delay;
          slew_states[static_cast<size_t>(count)] = state.output_slew;
          slew_violation_states[static_cast<size_t>(count)] = state.buffer_slew_violation;
          cap_violation_states[static_cast<size_t>(count)] = state.buffer_cap_violation;
        }

        grad_z_ptr[seg] += grad_edge_delay *
                           (delay_states[static_cast<size_t>(hi)] -
                            delay_states[static_cast<size_t>(lo)]);
        grad_z_ptr[seg] += grad_child_slew *
                           (slew_states[static_cast<size_t>(hi)] -
                            slew_states[static_cast<size_t>(lo)]);

        for (int64_t step = 0; step < max_count; ++step) {
          const scalar_t lo_slew = step < lo ? slew_violation_states[lo][step] : 0;
          const scalar_t hi_slew = step < hi ? slew_violation_states[hi][step] : 0;
          const scalar_t lo_cap = step < lo ? cap_violation_states[lo][step] : 0;
          const scalar_t hi_cap = step < hi ? cap_violation_states[hi][step] : 0;
          grad_z_ptr[seg] += grad_buffer_slew_ptr[seg * max_count + step] * (hi_slew - lo_slew);
          grad_z_ptr[seg] += grad_buffer_cap_ptr[seg * max_count + step] * (hi_cap - lo_cap);
        }

        scalar_t grad_downstream_load = static_cast<scalar_t>(0);
        for (int64_t count_index = 0; count_index < 2; ++count_index) {
          const int64_t count = (count_index == 0) ? lo : hi;
          const scalar_t weight =
              (count_index == 0) ? (static_cast<scalar_t>(1) - alpha) : alpha;
          if (weight == static_cast<scalar_t>(0)) {
            continue;
          }
          const int64_t sub_base = sub_base_zero + count * sub_stride_count;
          auto vjp = transfer_state_value_and_grad<scalar_t>(
              parent_slew,
              downstream_load,
              resistance,
              edge_cap_ptr[edge],
              count,
              sub_frac_ptr + sub_base,
              bsu,
              cap_by_size_ptr,
              size_count,
              slew_axis_ptr,
              slew_count,
              load_axis_ptr,
              load_count,
              delay_lut_ptr,
              output_slew_lut_ptr,
              grad_edge_delay * weight,
              grad_child_slew * weight, slew_limit, cap_limit,
              grad_buffer_slew_ptr + seg * max_count,
              grad_buffer_cap_ptr + seg * max_count, weight);
          grad_slew_ptr[compact] += vjp.grad_input_slew;
          grad_downstream_load += vjp.grad_downstream_load;
          grad_bsu_ptr[seg] += vjp.grad_bsu;
          grad_edge_res_ptr[edge] += vjp.grad_resistance;
          grad_edge_cap_ptr[edge] += vjp.grad_capacitance;
        }
        grad_load_ptr[child] += grad_downstream_load;
        grad_edge_cap_ptr[edge] -= static_cast<scalar_t>(0.5) *
                                   child_fraction * grad_downstream_load;
        grad_z_ptr[seg] -= static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] *
                           (child_hi - child_lo) * grad_downstream_load;
      }
    }
    grad_driver_arrival_ptr[net] += grad_arrival_ptr[begin];
    grad_driver_slew_ptr[net] += grad_slew_ptr[begin];
  }

  for (int64_t net = 0; net < static_cast<int64_t>(net_topo_start.size()) - 1; ++net) {
    const int64_t begin = net_topo_start[static_cast<size_t>(net)];
    const int64_t end = net_topo_start[static_cast<size_t>(net + 1)];
    for (int64_t compact = begin; compact < end; ++compact) {
      grad_eff_cap_ptr[compact] += grad_load_ptr[compact];
      for (int64_t edge = edge_start[static_cast<size_t>(compact)];
           edge < edge_start[static_cast<size_t>(compact + 1)];
           ++edge) {
        const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
        const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
        scalar_t grad_edge_input = grad_load_ptr[compact];
        if (seg >= 0) {
          grad_edge_input += grad_seg_upstream_ptr[seg];
        }
        if (seg < 0) {
          grad_load_ptr[child] += grad_edge_input;
          continue;
        }
        const scalar_t z = std::max(
            static_cast<scalar_t>(0),
            std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
        const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
        const int64_t hi = std::min<int64_t>(lo + 1, max_count);
        const scalar_t alpha = z - static_cast<scalar_t>(lo);
        const scalar_t input_cap = load_input_cap_ptr[seg];
        const scalar_t child_load = load_ptr[child];
        const scalar_t lo_fraction = (lo == 0) ? static_cast<scalar_t>(0) :
            parent_frac_ptr[seg * (max_count + 1) + lo];
        const scalar_t hi_fraction = (hi == 0) ? static_cast<scalar_t>(0) :
            parent_frac_ptr[seg * (max_count + 1) + hi];
        const scalar_t lo_value = (lo == 0) ? child_load : input_cap +
            static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * lo_fraction;
        const scalar_t hi_value = (hi == 0) ? child_load : input_cap +
            static_cast<scalar_t>(0.5) * edge_cap_ptr[edge] * hi_fraction;
        grad_edge_cap_ptr[edge] += grad_edge_input * static_cast<scalar_t>(0.5) *
            ((static_cast<scalar_t>(1) - alpha) * lo_fraction + alpha * hi_fraction);
        grad_z_ptr[seg] += grad_edge_input * (hi_value - lo_value);
        if (lo == 0) {
          grad_load_ptr[child] +=
              grad_edge_input * (static_cast<scalar_t>(1) - alpha);
        } else {
          grad_input_cap_ptr[seg] +=
              grad_edge_input * (static_cast<scalar_t>(1) - alpha);
        }
        if (hi == 0) {
          grad_load_ptr[child] += grad_edge_input * alpha;
        } else {
          grad_input_cap_ptr[seg] += grad_edge_input * alpha;
        }
      }
    }
  }

  for (int64_t edge = 0; edge < static_cast<int64_t>(edge_to_segment_id.size()); ++edge) {
    const int64_t seg = edge_to_segment_id[static_cast<size_t>(edge)];
    const int64_t parent = edge_parent_compact_id[static_cast<size_t>(edge)];
    const int64_t child = edge_child_compact_id[static_cast<size_t>(edge)];
    if (seg < 0) {
      grad_edge_cap_ptr[edge] += static_cast<scalar_t>(0.5) *
          (grad_eff_cap_ptr[parent] + grad_eff_cap_ptr[child]);
      continue;
    }
    const scalar_t z = std::max(
        static_cast<scalar_t>(0),
        std::min(z_ptr[seg], static_cast<scalar_t>(max_count)));
    const int64_t lo = static_cast<int64_t>(std::floor(static_cast<double>(z)));
    const int64_t hi = std::min<int64_t>(lo + 1, max_count);
    const scalar_t alpha = z - static_cast<scalar_t>(lo);
    const scalar_t parent_lo = parent_frac_ptr[seg * (max_count + 1) + lo];
    const scalar_t parent_hi = parent_frac_ptr[seg * (max_count + 1) + hi];
    const scalar_t child_lo = child_frac_ptr[seg * (max_count + 1) + lo];
    const scalar_t child_hi = child_frac_ptr[seg * (max_count + 1) + hi];
    const scalar_t cap = edge_cap_ptr[edge];
    const scalar_t parent_fraction =
        (static_cast<scalar_t>(1) - alpha) * parent_lo + alpha * parent_hi;
    const scalar_t child_fraction =
        (static_cast<scalar_t>(1) - alpha) * child_lo + alpha * child_hi;
    grad_edge_cap_ptr[edge] += static_cast<scalar_t>(0.5) *
        (grad_eff_cap_ptr[parent] * parent_fraction +
         grad_eff_cap_ptr[child] * child_fraction);
    grad_z_ptr[seg] += grad_eff_cap_ptr[parent] * static_cast<scalar_t>(0.5) * cap *
                       (parent_hi - parent_lo);
    grad_z_ptr[seg] += grad_eff_cap_ptr[child] * static_cast<scalar_t>(0.5) * cap *
                       (child_hi - child_lo);
  }

  for (int64_t seg = 0; seg < num_segments; ++seg) {
    if (z_ptr[seg] < static_cast<scalar_t>(0) ||
        z_ptr[seg] > static_cast<scalar_t>(max_count)) {
      grad_z_ptr[seg] = static_cast<scalar_t>(0);
    }
    if (bsu_ptr[seg] < static_cast<scalar_t>(0) ||
        bsu_ptr[seg] > static_cast<scalar_t>(size_count - 1)) {
      grad_bsu_ptr[seg] = static_cast<scalar_t>(0);
    }
  }

  py::dict result;
  result["grad_z"] = grad_z;
  result["grad_bsu"] = grad_bsu;
  result["grad_load_input_cap"] = grad_input_cap;
  result["grad_driver_arrival"] = grad_driver_arrival;
  result["grad_driver_slew"] = grad_driver_slew;
  result["grad_edge_resistance"] = grad_edge_resistance;
  result["grad_edge_capacitance"] = grad_edge_capacitance;
  result["grad_node_capacitance"] = grad_eff_cap;
  return result;
}

py::dict segment_count_transfer_backward(
    at::Tensor net_topo_start,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor z_value,
    at::Tensor bsu_index,
    at::Tensor load_input_cap,
    at::Tensor buffer_input_cap_by_size,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor segment_sub_resistance_fraction,
    at::Tensor sink_node_compact_id,
    at::Tensor node_load,
    at::Tensor node_slew,
    at::Tensor effective_node_cap,
    at::Tensor grad_segment_delay,
    at::Tensor grad_segment_output_slew,
    at::Tensor grad_segment_upstream_cap,
    at::Tensor grad_sink_arrival,
    at::Tensor grad_sink_slew,
    at::Tensor grad_sink_load,
    at::Tensor buffer_slew_limits,
    at::Tensor buffer_cap_limits,
    at::Tensor grad_buffer_slew_violation,
    at::Tensor grad_buffer_cap_violation,
    bool count_gradient_enabled) {
  const auto scalar_type = node_load.scalar_type();
  check_float_like(buffer_slew_limits, "buffer_slew_limits", scalar_type);
  check_float_like(buffer_cap_limits, "buffer_cap_limits", scalar_type);
  check_float_2d_like(grad_buffer_slew_violation, "grad_buffer_slew_violation", scalar_type);
  check_float_2d_like(grad_buffer_cap_violation, "grad_buffer_cap_violation", scalar_type);
  TORCH_CHECK(buffer_slew_limits.numel() == buffer_input_cap_by_size.numel() &&
              buffer_cap_limits.numel() == buffer_input_cap_by_size.numel(),
              "buffer limits must match legal size count");
  TORCH_CHECK(grad_buffer_slew_violation.size(0) == z_value.numel() &&
              grad_buffer_slew_violation.size(1) == parent_cap_fraction.size(1) - 1 &&
              grad_buffer_cap_violation.sizes() == grad_buffer_slew_violation.sizes(),
              "buffer violation gradients must match segment count");
  check_float_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_like(driver_arrival, "driver_arrival", scalar_type);
  check_float_like(driver_slew, "driver_slew", scalar_type);
  check_float_like(z_value, "z_value", scalar_type);
  check_float_like(bsu_index, "bsu_index", scalar_type);
  check_float_like(load_input_cap, "load_input_cap", scalar_type);
  check_float_like(buffer_input_cap_by_size, "buffer_input_cap_by_size", scalar_type);
  check_float_like(buffer_slew_axis, "buffer_slew_axis", scalar_type);
  check_float_like(buffer_load_axis, "buffer_load_axis", scalar_type);
  check_float_3d_like(buffer_delay_lut, "buffer_delay_lut", scalar_type);
  check_float_3d_like(buffer_output_slew_lut, "buffer_output_slew_lut", scalar_type);
  TORCH_CHECK(parent_cap_fraction.device().is_cpu(), "parent_cap_fraction must be CPU");
  TORCH_CHECK(child_cap_fraction.device().is_cpu(), "child_cap_fraction must be CPU");
  TORCH_CHECK(segment_sub_resistance_fraction.device().is_cpu(),
              "segment_sub_resistance_fraction must be CPU");
  TORCH_CHECK(parent_cap_fraction.scalar_type() == scalar_type,
              "parent_cap_fraction dtype mismatch");
  TORCH_CHECK(child_cap_fraction.scalar_type() == scalar_type,
              "child_cap_fraction dtype mismatch");
  TORCH_CHECK(segment_sub_resistance_fraction.scalar_type() == scalar_type,
              "segment_sub_resistance_fraction dtype mismatch");
  check_float_like(node_load, "node_load", scalar_type);
  check_float_like(node_slew, "node_slew", scalar_type);
  check_float_like(effective_node_cap, "effective_node_cap", scalar_type);
  check_float_like(grad_segment_delay, "grad_segment_delay", scalar_type);
  check_float_like(grad_segment_output_slew, "grad_segment_output_slew", scalar_type);
  check_float_like(grad_segment_upstream_cap, "grad_segment_upstream_cap", scalar_type);
  check_float_like(grad_sink_arrival, "grad_sink_arrival", scalar_type);
  check_float_like(grad_sink_slew, "grad_sink_slew", scalar_type);
  check_float_like(grad_sink_load, "grad_sink_load", scalar_type);

  const std::vector<int64_t> topo_start = tensor_to_i64_vector(net_topo_start, "net_topo_start");
  const std::vector<int64_t> child_start = tensor_to_i64_vector(edge_start, "edge_start");
  const std::vector<int64_t> parents =
      tensor_to_i64_vector(edge_parent_compact_id, "edge_parent_compact_id");
  const std::vector<int64_t> children =
      tensor_to_i64_vector(edge_child_compact_id, "edge_child_compact_id");
  const std::vector<int64_t> edge_segment =
      tensor_to_i64_vector(edge_to_segment_id, "edge_to_segment_id");
  const std::vector<int64_t> sink_compact =
      tensor_to_i64_vector(sink_node_compact_id, "sink_node_compact_id");

  py::dict result;
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segment_count_transfer_backward", [&] {
    result = segment_count_transfer_backward_impl<scalar_t>(
        topo_start,
        child_start,
        parents,
        children,
        edge_resistance,
        edge_capacitance,
        edge_segment,
        driver_arrival,
        driver_slew,
        z_value,
        bsu_index,
        load_input_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        parent_cap_fraction,
        child_cap_fraction,
        segment_sub_resistance_fraction,
        sink_compact,
        node_load,
        node_slew,
        effective_node_cap,
        grad_segment_delay,
        grad_segment_output_slew,
        grad_segment_upstream_cap,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        buffer_slew_limits,
        buffer_cap_limits,
        grad_buffer_slew_violation,
        grad_buffer_cap_violation,
        count_gradient_enabled);
  });
  return result;
}

py::dict segment_count_transfer_forward(
    at::Tensor net_topo_start,
    at::Tensor flat_topo_node_id,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor node_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor z_value,
    at::Tensor bsu_index,
    at::Tensor load_input_cap,
    at::Tensor buffer_input_cap_by_size,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor segment_sub_resistance_fraction,
    at::Tensor segment_retained_upstream_cap,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index,
    at::Tensor sink_node_compact_id,
    at::Tensor buffer_slew_limits,
    at::Tensor buffer_cap_limits,
    bool count_gradient_enabled) {
  const auto scalar_type = node_capacitance.scalar_type();
  check_float_like(buffer_slew_limits, "buffer_slew_limits", scalar_type);
  check_float_like(buffer_cap_limits, "buffer_cap_limits", scalar_type);
  TORCH_CHECK(buffer_slew_limits.numel() == buffer_input_cap_by_size.numel() &&
              buffer_cap_limits.numel() == buffer_input_cap_by_size.numel(),
              "buffer limits must match legal size count");
  check_float_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_like(node_capacitance, "node_capacitance", scalar_type);
  check_float_like(driver_arrival, "driver_arrival", scalar_type);
  check_float_like(driver_slew, "driver_slew", scalar_type);
  check_float_like(z_value, "z_value", scalar_type);
  check_float_like(bsu_index, "bsu_index", scalar_type);
  check_float_like(load_input_cap, "load_input_cap", scalar_type);
  check_float_like(buffer_input_cap_by_size, "buffer_input_cap_by_size", scalar_type);
  check_float_like(buffer_slew_axis, "buffer_slew_axis", scalar_type);
  check_float_like(buffer_load_axis, "buffer_load_axis", scalar_type);
  check_float_3d_like(buffer_delay_lut, "buffer_delay_lut", scalar_type);
  check_float_3d_like(buffer_output_slew_lut, "buffer_output_slew_lut", scalar_type);
  TORCH_CHECK(parent_cap_fraction.device().is_cpu(), "parent_cap_fraction must be CPU");
  TORCH_CHECK(child_cap_fraction.device().is_cpu(), "child_cap_fraction must be CPU");
  TORCH_CHECK(segment_sub_resistance_fraction.device().is_cpu(),
              "segment_sub_resistance_fraction must be CPU");
  TORCH_CHECK(parent_cap_fraction.scalar_type() == scalar_type,
              "parent_cap_fraction dtype mismatch");
  TORCH_CHECK(child_cap_fraction.scalar_type() == scalar_type,
              "child_cap_fraction dtype mismatch");
  TORCH_CHECK(segment_sub_resistance_fraction.scalar_type() == scalar_type,
              "segment_sub_resistance_fraction dtype mismatch");
  check_float_like(
      segment_retained_upstream_cap,
      "segment_retained_upstream_cap",
      scalar_type);
  TORCH_CHECK(buffer_input_cap_by_size.numel() >= 1, "buffer_input_cap_by_size must be non-empty");
  TORCH_CHECK(buffer_slew_axis.numel() >= 1, "buffer_slew_axis must be non-empty");
  TORCH_CHECK(buffer_load_axis.numel() >= 1, "buffer_load_axis must be non-empty");
  TORCH_CHECK(
      buffer_delay_lut.size(0) == buffer_input_cap_by_size.numel() &&
          buffer_delay_lut.size(1) == buffer_slew_axis.numel() &&
          buffer_delay_lut.size(2) == buffer_load_axis.numel(),
      "buffer_delay_lut shape mismatch");
  TORCH_CHECK(
      buffer_output_slew_lut.sizes() == buffer_delay_lut.sizes(),
      "buffer_output_slew_lut shape mismatch");

  const std::vector<int64_t> topo_start = tensor_to_i64_vector(net_topo_start, "net_topo_start");
  tensor_to_i64_vector(flat_topo_node_id, "flat_topo_node_id");
  const std::vector<int64_t> child_start = tensor_to_i64_vector(edge_start, "edge_start");
  const std::vector<int64_t> parents =
      tensor_to_i64_vector(edge_parent_compact_id, "edge_parent_compact_id");
  const std::vector<int64_t> children =
      tensor_to_i64_vector(edge_child_compact_id, "edge_child_compact_id");
  const std::vector<int64_t> edge_segment =
      tensor_to_i64_vector(edge_to_segment_id, "edge_to_segment_id");
  const std::vector<int64_t> sinks = tensor_to_i64_vector(sink_node_id, "sink_node_id");
  tensor_to_i64_vector(sink_net_index, "sink_net_index");
  const std::vector<int64_t> sink_compact =
      tensor_to_i64_vector(sink_node_compact_id, "sink_node_compact_id");

  py::dict result;
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segment_count_transfer_forward", [&] {
    result = segment_count_transfer_forward_impl<scalar_t>(
        topo_start,
        child_start,
        parents,
        children,
        edge_resistance,
        edge_capacitance,
        node_capacitance,
        edge_segment,
        driver_arrival,
        driver_slew,
        z_value,
        bsu_index,
        load_input_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut,
        parent_cap_fraction,
        child_cap_fraction,
        segment_sub_resistance_fraction,
        segment_retained_upstream_cap,
        sinks,
        sink_net_index,
        sink_compact,
        buffer_slew_limits,
        buffer_cap_limits,
        count_gradient_enabled);
  });
  result["sink_node_id"] = sink_node_id;
  return result;
}

py::dict segment_transfer_forward(
    at::Tensor input_arrival,
    at::Tensor input_slew,
    at::Tensor downstream_load,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor repeater_count,
    at::Tensor split_fractions,
    at::Tensor bsu_index,
    at::Tensor upstream_retained_cap,
    at::Tensor buffer_input_cap_by_size,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut) {
  const auto scalar_type = input_arrival.scalar_type();
  check_float_like(input_arrival, "input_arrival", scalar_type);
  check_float_like(input_slew, "input_slew", scalar_type);
  check_float_like(downstream_load, "downstream_load", scalar_type);
  check_float_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_integer_1d_contiguous(repeater_count, "repeater_count");
  check_float_2d_like(split_fractions, "split_fractions", scalar_type);
  check_float_like(bsu_index, "bsu_index", scalar_type);
  check_float_like(upstream_retained_cap, "upstream_retained_cap", scalar_type);
  check_float_like(buffer_input_cap_by_size, "buffer_input_cap_by_size", scalar_type);
  check_float_like(buffer_slew_axis, "buffer_slew_axis", scalar_type);
  check_float_like(buffer_load_axis, "buffer_load_axis", scalar_type);
  check_float_3d_like(buffer_delay_lut, "buffer_delay_lut", scalar_type);
  check_float_3d_like(buffer_output_slew_lut, "buffer_output_slew_lut", scalar_type);

  const int64_t sample_count = input_arrival.numel();
  TORCH_CHECK(input_slew.numel() == sample_count, "input_slew size mismatch");
  TORCH_CHECK(downstream_load.numel() == sample_count, "downstream_load size mismatch");
  TORCH_CHECK(edge_resistance.numel() == sample_count, "edge_resistance size mismatch");
  TORCH_CHECK(edge_capacitance.numel() == sample_count, "edge_capacitance size mismatch");
  TORCH_CHECK(repeater_count.numel() == sample_count, "repeater_count size mismatch");
  TORCH_CHECK(split_fractions.size(0) == sample_count, "split_fractions batch size mismatch");
  TORCH_CHECK(bsu_index.numel() == sample_count, "bsu_index size mismatch");
  TORCH_CHECK(upstream_retained_cap.numel() == sample_count, "upstream_retained_cap size mismatch");
  TORCH_CHECK(split_fractions.size(1) >= 1, "split_fractions must have at least one column");
  TORCH_CHECK(buffer_input_cap_by_size.numel() >= 1, "buffer_input_cap_by_size must be non-empty");
  TORCH_CHECK(buffer_slew_axis.numel() >= 1, "buffer_slew_axis must be non-empty");
  TORCH_CHECK(buffer_load_axis.numel() >= 1, "buffer_load_axis must be non-empty");
  TORCH_CHECK(
      buffer_delay_lut.size(0) == buffer_input_cap_by_size.numel() &&
          buffer_delay_lut.size(1) == buffer_slew_axis.numel() &&
          buffer_delay_lut.size(2) == buffer_load_axis.numel(),
      "buffer_delay_lut shape mismatch");
  TORCH_CHECK(
      buffer_output_slew_lut.sizes() == buffer_delay_lut.sizes(),
      "buffer_output_slew_lut shape mismatch");

  py::dict result;
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segment_transfer_forward", [&] {
    result = segment_transfer_forward_impl<scalar_t>(
        input_arrival,
        input_slew,
        downstream_load,
        edge_resistance,
        edge_capacitance,
        repeater_count,
        split_fractions,
        bsu_index,
        upstream_retained_cap,
        buffer_input_cap_by_size,
        buffer_slew_axis,
        buffer_load_axis,
        buffer_delay_lut,
        buffer_output_slew_lut);
  });
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Buffer-aware net subgraph timing forward operators";
  m.def("forward", &forward, "Fixed-state buffer-aware net subgraph forward (C++)");
  m.def("segment_count_forward", &segment_count_forward, "Segment-count relaxed timing forward (C++)");
  m.def("segment_count_backward", &segment_count_backward, "Segment-count relaxed timing backward (C++)");
  m.def(
      "segment_count_transfer_backward",
      &segment_count_transfer_backward,
      "Segment-count relaxed timing backward with LUT segment transfer (C++)");
  m.def(
      "segment_count_transfer_forward",
      &segment_count_transfer_forward,
      "Segment-count relaxed timing forward with LUT segment transfer (C++)");
  m.def("segment_transfer_forward", &segment_transfer_forward, "Batched segment transfer forward (C++)");
  m.def(
      "pack_segment_count_topology",
      &dreamplace::pack_segment_count_topology,
      "Pack live segment-count topology and geometry (C++ CPU)");
  m.def(
      "pack_candidate_topology",
      &dreamplace::pack_candidate_topology,
      "Pack candidate-buffer topology and expanded synthetic nodes (C++ CPU)");
  m.def(
      "select_packed_segment_count_inputs",
      &dreamplace::select_packed_segment_count_inputs,
      "Select a compact active view from packed segment-count tensors (C++ CPU)");
  m.def(
      "select_packed_candidate_inputs",
      &dreamplace::select_packed_candidate_inputs,
      "Select a compact active view from packed candidate tensors (C++ CPU)");
}
