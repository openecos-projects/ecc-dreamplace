#include "candidate_maintenance.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace dreamplace {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& started_at) {
  return std::chrono::duration<double, std::milli>(
             Clock::now() - started_at)
      .count();
}

void check_cpu_1d_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_integer_1d(const at::Tensor& tensor, const char* name) {
  check_cpu_1d_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong,
      name,
      " must be int32 or int64");
}

void check_floating_1d(const at::Tensor& tensor, const char* name) {
  check_cpu_1d_contiguous(tensor, name);
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble,
      name,
      " must be float32 or float64");
}

int64_t integer_at(const at::Tensor& tensor, int64_t index) {
  return tensor.scalar_type() == at::kInt
      ? static_cast<int64_t>(tensor.data_ptr<int32_t>()[index])
      : tensor.data_ptr<int64_t>()[index];
}

double floating_at(const at::Tensor& tensor, int64_t index) {
  return tensor.scalar_type() == at::kFloat
      ? static_cast<double>(tensor.data_ptr<float>()[index])
      : tensor.data_ptr<double>()[index];
}

template <typename scalar_t>
void fill_float_tensor(at::Tensor& tensor, const std::vector<double>& values) {
  auto* output = tensor.data_ptr<scalar_t>();
  for (size_t index = 0; index < values.size(); ++index) {
    output[index] = static_cast<scalar_t>(values[index]);
  }
}

at::Tensor double_tensor(
    const std::vector<double>& values,
    const at::Tensor& prototype) {
  auto tensor = at::empty(
      {static_cast<int64_t>(values.size())},
      prototype.options().dtype(at::kDouble));
  fill_float_tensor<double>(tensor, values);
  return tensor;
}

at::Tensor long_tensor(
    const std::vector<int64_t>& values,
    const at::Tensor& prototype) {
  auto tensor = at::empty(
      {static_cast<int64_t>(values.size())},
      prototype.options().dtype(at::kLong));
  auto* output = tensor.data_ptr<int64_t>();
  for (size_t index = 0; index < values.size(); ++index) {
    output[index] = values[index];
  }
  return tensor;
}

std::vector<int64_t> normalized_ids(
    const at::Tensor& tensor,
    const char* name) {
  check_integer_1d(tensor, name);
  std::vector<int64_t> result;
  result.reserve(static_cast<size_t>(tensor.numel()));
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    result.push_back(integer_at(tensor, index));
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

}  // namespace

py::dict compute_candidate_dirty_nets(
    const at::Tensor& changed_instance_ids,
    const at::Tensor& moved_instance_ids,
    const at::Tensor& accepted_net_ids,
    const at::Tensor& candidate_net_ids,
    const at::Tensor& flat_node2pin_map,
    const at::Tensor& flat_node2pin_start_map,
    const at::Tensor& pin2net_map) {
  const auto started_at = Clock::now();
  check_integer_1d(flat_node2pin_map, "flat_node2pin_map");
  check_integer_1d(flat_node2pin_start_map, "flat_node2pin_start_map");
  check_integer_1d(pin2net_map, "pin2net_map");
  TORCH_CHECK(
      flat_node2pin_start_map.numel() >= 1,
      "flat_node2pin_start_map must contain at least one offset");
  const int64_t node_count = flat_node2pin_start_map.numel() - 1;
  const int64_t pin_count = flat_node2pin_map.numel();
  TORCH_CHECK(
      pin2net_map.numel() == pin_count,
      "pin2net_map and flat_node2pin_map must have equal lengths");
  // The public boundary accepts int32 as well as int64. Convert the net map
  // once when necessary instead of branching for every incident pin.
  std::vector<int64_t> pin_net;
  if (pin2net_map.scalar_type() == at::kLong) {
    pin_net.assign(
        pin2net_map.data_ptr<int64_t>(),
        pin2net_map.data_ptr<int64_t>() + pin_count);
  } else {
    const int32_t* source = pin2net_map.data_ptr<int32_t>();
    pin_net.resize(static_cast<size_t>(pin_count));
    for (int64_t index = 0; index < pin_count; ++index) {
      pin_net[static_cast<size_t>(index)] = source[index];
    }
  }
  int64_t net_count = 0;
  int64_t unconnected_pin_count = 0;
  for (const int64_t net_id : pin_net) {
    if (net_id == -1) {
      ++unconnected_pin_count;
      continue;
    }
    TORCH_CHECK(
        net_id >= 0,
        "pin2net_map contains an invalid net id: ",
        net_id);
    net_count = std::max<int64_t>(net_count, net_id + 1);
  }

  const auto candidates = normalized_ids(
      candidate_net_ids,
      "candidate_net_ids");
  std::vector<uint8_t> candidate_mask(static_cast<size_t>(net_count), 0);
  for (const int64_t net_id : candidates) {
    TORCH_CHECK(
        net_id >= 0 && net_id < net_count,
        "candidate net id is outside pin2net_map: ",
        net_id);
    candidate_mask[static_cast<size_t>(net_id)] = 1;
  }
  const bool unrestricted = candidates.empty();

  std::vector<int64_t> source_instances = normalized_ids(
      changed_instance_ids,
      "changed_instance_ids");
  const auto moved = normalized_ids(moved_instance_ids, "moved_instance_ids");
  source_instances.insert(
      source_instances.end(),
      moved.begin(),
      moved.end());
  std::sort(source_instances.begin(), source_instances.end());
  source_instances.erase(
      std::unique(source_instances.begin(), source_instances.end()),
      source_instances.end());

  std::vector<uint8_t> dirty_mask(static_cast<size_t>(net_count), 0);
  int64_t incident_pin_count = 0;
  for (const int64_t instance_id : source_instances) {
    TORCH_CHECK(
        instance_id >= 0 && instance_id < node_count,
        "instance id is outside flat_node2pin_map: ",
        instance_id);
    const int64_t begin = integer_at(flat_node2pin_start_map, instance_id);
    const int64_t end = integer_at(flat_node2pin_start_map, instance_id + 1);
    TORCH_CHECK(
        begin >= 0 && begin <= end && end <= pin_count,
        "invalid incident-pin CSR range for instance ",
        instance_id);
    for (int64_t offset = begin; offset < end; ++offset) {
      const int64_t pin_id = integer_at(flat_node2pin_map, offset);
      if (pin_id < 0) {
        continue;
      }
      TORCH_CHECK(
          pin_id < pin_count,
          "pin id is outside pin2net_map: ",
          pin_id);
      const int64_t net_id = pin_net[static_cast<size_t>(pin_id)];
      if (net_id == -1) {
        continue;
      }
      TORCH_CHECK(
          net_id >= 0 && net_id < net_count,
          "net id is outside candidate domain: ",
          net_id);
      ++incident_pin_count;
      if (unrestricted || candidate_mask[static_cast<size_t>(net_id)] != 0) {
        dirty_mask[static_cast<size_t>(net_id)] = 1;
      }
    }
  }

  const auto accepted = normalized_ids(accepted_net_ids, "accepted_net_ids");
  int64_t accepted_candidate_count = 0;
  for (const int64_t net_id : accepted) {
    TORCH_CHECK(
        net_id >= 0 && net_id < net_count,
        "accepted net id is outside candidate domain: ",
        net_id);
    if (unrestricted || candidate_mask[static_cast<size_t>(net_id)] != 0) {
      dirty_mask[static_cast<size_t>(net_id)] = 1;
      ++accepted_candidate_count;
    }
  }

  std::vector<int64_t> dirty_net_ids;
  for (int64_t net_id = 0; net_id < net_count; ++net_id) {
    if (dirty_mask[static_cast<size_t>(net_id)] != 0) {
      dirty_net_ids.push_back(net_id);
    }
  }

  const auto changed = normalized_ids(
      changed_instance_ids,
      "changed_instance_ids");
  py::dict counts;
  counts["changed_instance_count"] = static_cast<int64_t>(changed.size());
  counts["moved_instance_count"] = static_cast<int64_t>(moved.size());
  counts["source_instance_count"] =
      static_cast<int64_t>(source_instances.size());
  counts["incident_pin_count"] = incident_pin_count;
  counts["unconnected_pin_count"] = unconnected_pin_count;
  counts["accepted_net_count"] = static_cast<int64_t>(accepted.size());
  counts["accepted_candidate_net_count"] = accepted_candidate_count;
  counts["dirty_net_count"] = static_cast<int64_t>(dirty_net_ids.size());

  py::dict metadata;
  metadata["backend"] = "native_cpp";
  metadata["unrestricted_candidate_domain"] = unrestricted;
  metadata["candidate_net_count"] = static_cast<int64_t>(candidates.size());
  metadata["native_total_ms"] = elapsed_ms(started_at);

  py::dict result;
  result["dirty_net_ids"] = long_tensor(dirty_net_ids, candidate_net_ids);
  result["counts"] = counts;
  result["metadata"] = metadata;
  return result;
}

py::dict refresh_candidate_geometry(
    const at::Tensor& dirty_net_ids,
    const at::Tensor& candidate_net_id,
    const at::Tensor& candidate_parent_node_id,
    const at::Tensor& candidate_child_node_id,
    const at::Tensor& candidate_tree_node_id,
    const at::Tensor& candidate_is_segment,
    const at::Tensor& candidate_split_ratio,
    const at::Tensor& node_x_dbu,
    const at::Tensor& node_y_dbu,
    bool all_candidates) {
  const auto started_at = Clock::now();
  const std::array<const at::Tensor*, 6> integer_inputs = {
      &candidate_net_id,
      &candidate_parent_node_id,
      &candidate_child_node_id,
      &candidate_tree_node_id,
      &candidate_is_segment,
      &dirty_net_ids,
  };
  for (const at::Tensor* tensor : integer_inputs) {
    check_integer_1d(*tensor, "candidate geometry integer input");
  }
  check_floating_1d(candidate_split_ratio, "candidate_split_ratio");
  check_floating_1d(node_x_dbu, "node_x_dbu");
  check_floating_1d(node_y_dbu, "node_y_dbu");
  TORCH_CHECK(
      node_x_dbu.numel() == node_y_dbu.numel(),
      "node_x_dbu and node_y_dbu must have equal lengths");
  const int64_t candidate_count = candidate_net_id.numel();
  const std::array<const at::Tensor*, 6> candidate_tensors = {
      &candidate_parent_node_id,
      &candidate_child_node_id,
      &candidate_tree_node_id,
      &candidate_is_segment,
      &candidate_split_ratio,
      &candidate_net_id,
  };
  for (const at::Tensor* tensor : candidate_tensors) {
    TORCH_CHECK(
        tensor->numel() == candidate_count,
        "candidate geometry inputs must have equal lengths");
  }

  std::vector<uint8_t> dirty_mask;
  if (!all_candidates) {
    dirty_mask.assign(
        static_cast<size_t>(candidate_count),
        0);
    const auto dirty = normalized_ids(dirty_net_ids, "dirty_net_ids");
    std::unordered_set<int64_t> dirty_set(dirty.begin(), dirty.end());
    // Candidate rows are grouped by net in normal generated payloads. The
    // serial mask fill is still tiny compared with rebuilding candidate state.
    for (int64_t row = 0; row < candidate_count; ++row) {
      if (dirty_set.count(integer_at(candidate_net_id, row)) != 0) {
        dirty_mask[static_cast<size_t>(row)] = 1;
      }
    }
  }

  std::vector<int64_t> rows;
  std::vector<int64_t> nets;
  std::vector<int64_t> invalid_rows;
  std::vector<double> x_values;
  std::vector<double> y_values;
  std::vector<double> segment_lengths;
  std::vector<double> upstream_lengths;
  std::vector<double> downstream_lengths;
  rows.reserve(static_cast<size_t>(candidate_count));
  nets.reserve(static_cast<size_t>(candidate_count));
  x_values.reserve(static_cast<size_t>(candidate_count));
  y_values.reserve(static_cast<size_t>(candidate_count));
  segment_lengths.reserve(static_cast<size_t>(candidate_count));
  upstream_lengths.reserve(static_cast<size_t>(candidate_count));
  downstream_lengths.reserve(static_cast<size_t>(candidate_count));

  const int64_t node_count = node_x_dbu.numel();
  for (int64_t row = 0; row < candidate_count; ++row) {
    if (!all_candidates && dirty_mask[static_cast<size_t>(row)] == 0) {
      continue;
    }
    const bool is_segment = integer_at(candidate_is_segment, row) != 0;
    double x = 0.0;
    double y = 0.0;
    double ratio = 0.0;
    if (is_segment) {
      const int64_t parent = integer_at(candidate_parent_node_id, row);
      const int64_t child = integer_at(candidate_child_node_id, row);
      ratio = floating_at(candidate_split_ratio, row);
      if (parent < 0 || parent >= node_count || child < 0 ||
          child >= node_count || !std::isfinite(ratio) || ratio < 0.0 ||
          ratio > 1.0) {
        invalid_rows.push_back(row);
        continue;
      }
      x = (1.0 - ratio) * floating_at(node_x_dbu, parent) +
          ratio * floating_at(node_x_dbu, child);
      y = (1.0 - ratio) * floating_at(node_y_dbu, parent) +
          ratio * floating_at(node_y_dbu, child);
    } else {
      const int64_t node = integer_at(candidate_tree_node_id, row);
      if (node < 0 || node >= node_count) {
        invalid_rows.push_back(row);
        continue;
      }
      x = floating_at(node_x_dbu, node);
      y = floating_at(node_y_dbu, node);
    }
    if (!std::isfinite(x) || !std::isfinite(y)) {
      invalid_rows.push_back(row);
      continue;
    }
    double length = 0.0;
    if (is_segment) {
      const int64_t parent = integer_at(candidate_parent_node_id, row);
      const int64_t child = integer_at(candidate_child_node_id, row);
      length = std::hypot(
          floating_at(node_x_dbu, child) -
              floating_at(node_x_dbu, parent),
          floating_at(node_y_dbu, child) -
              floating_at(node_y_dbu, parent));
    }
    rows.push_back(row);
    nets.push_back(integer_at(candidate_net_id, row));
    x_values.push_back(x);
    y_values.push_back(y);
    segment_lengths.push_back(length);
    upstream_lengths.push_back(length * ratio);
    downstream_lengths.push_back(length * (1.0 - ratio));
  }

  py::dict counts;
  counts["candidate_count"] = candidate_count;
  counts["refreshed_candidate_count"] =
      static_cast<int64_t>(rows.size());
  counts["invalid_candidate_count"] =
      static_cast<int64_t>(invalid_rows.size());

  py::dict metadata;
  metadata["backend"] = "native_cpp";
  metadata["all_candidates"] = all_candidates;
  metadata["node_count"] = node_count;
  metadata["dirty_net_count"] =
      static_cast<int64_t>(dirty_net_ids.numel());
  metadata["native_total_ms"] = elapsed_ms(started_at);

  py::dict result;
  result["candidate_rows"] = long_tensor(rows, candidate_net_id);
  result["net_ids"] = long_tensor(nets, candidate_net_id);
  result["invalid_candidate_rows"] =
      long_tensor(invalid_rows, candidate_net_id);
  result["x_dbu"] = double_tensor(x_values, candidate_split_ratio);
  result["y_dbu"] = double_tensor(y_values, candidate_split_ratio);
  result["segment_length_dbu"] =
      double_tensor(segment_lengths, candidate_split_ratio);
  result["upstream_length_dbu"] =
      double_tensor(upstream_lengths, candidate_split_ratio);
  result["downstream_length_dbu"] =
      double_tensor(downstream_lengths, candidate_split_ratio);
  result["counts"] = counts;
  result["metadata"] = metadata;
  return result;
}

}  // namespace dreamplace
