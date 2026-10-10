#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace dreamplace {
namespace {

template <typename scalar_t>
void apply_adams_kernel(
    scalar_t* net_weights,
    scalar_t* net_criticality,
    const int64_t* degree_map,
    int64_t ignore_net_degree,
    const int64_t* pin2net_map,
    int64_t num_pins,
    int64_t num_nets,
    const std::vector<std::vector<int64_t>>& paths,
    double max_net_weight) {
  std::vector<uint8_t> net_critical_flag(static_cast<size_t>(num_nets), 0);
  for (const auto& path : paths) {
    for (int64_t pin : path) {
      if (pin < 0 || pin >= num_pins) {
        continue;
      }
      int64_t net_id = pin2net_map[pin];
      if (net_id >= 0 && net_id < num_nets) {
        net_critical_flag[static_cast<size_t>(net_id)] = 1;
      }
    }
  }

  const bool limit_weight = std::isfinite(max_net_weight);
  const scalar_t max_weight = static_cast<scalar_t>(max_net_weight);

  for (int64_t net_id = 0; net_id < num_nets; ++net_id) {
    if (degree_map[net_id] > ignore_net_degree) {
      continue;
    }
    net_criticality[net_id] *= static_cast<scalar_t>(0.5);
    if (net_critical_flag[static_cast<size_t>(net_id)]) {
      net_criticality[net_id] += static_cast<scalar_t>(0.5);
    }
    scalar_t updated = net_weights[net_id] *
        (static_cast<scalar_t>(1.0) + net_criticality[net_id]);
    if (limit_weight && updated > max_weight) {
      updated = max_weight;
    }
    net_weights[net_id] = updated;
  }
}

template <typename scalar_t>
void apply_lilith_kernel(
    scalar_t* net_weights,
    scalar_t* net_criticality,
    const int64_t* degree_map,
    int64_t ignore_net_degree,
    const int64_t* flat_netpin,
    const int64_t* netpin_start,
    int64_t num_nets,
    int64_t num_pins,
    const scalar_t* pin_slack,
    double wns,
    double decay,
    double max_net_weight) {
  std::vector<scalar_t> net_slack(static_cast<size_t>(num_nets), static_cast<scalar_t>(0));
  for (int64_t net_id = 0; net_id < num_nets; ++net_id) {
    int64_t begin = netpin_start[net_id];
    int64_t end = netpin_start[net_id + 1];
    if (begin >= end) {
      net_slack[static_cast<size_t>(net_id)] = static_cast<scalar_t>(0);
      continue;
    }
    scalar_t min_slack = std::numeric_limits<scalar_t>::infinity();
    for (int64_t idx = begin; idx < end; ++idx) {
      int64_t pin = flat_netpin[idx];
      if (pin < 0 || pin >= num_pins) {
        continue;
      }
      scalar_t slack = pin_slack[pin];
      if (slack < min_slack) {
        min_slack = slack;
      }
    }
    if (!std::isfinite(static_cast<double>(min_slack))) {
      min_slack = static_cast<scalar_t>(0);
    }
    net_slack[static_cast<size_t>(net_id)] = min_slack;
  }

  const bool limit_weight = std::isfinite(max_net_weight);
  const scalar_t max_weight = static_cast<scalar_t>(max_net_weight);
  const bool has_negative_wns = std::isfinite(wns) && (wns < 0.0);
  const double decay_d = decay;
  const double one_minus_decay = 1.0 - decay_d;

  for (int64_t net_id = 0; net_id < num_nets; ++net_id) {
    if (degree_map[net_id] > ignore_net_degree) {
      continue;
    }
    if (has_negative_wns) {
      double slack = static_cast<double>(net_slack[static_cast<size_t>(net_id)]);
      double nc = 0.0;
      if (slack < 0.0 && wns != 0.0) {
        nc = std::max(0.0, slack / wns);
      }
      double crit = static_cast<double>(net_criticality[net_id]);
      double updated_crit = std::pow(1.0 + crit, decay_d) *
          std::pow(1.0 + nc, one_minus_decay) - 1.0;
      net_criticality[net_id] = static_cast<scalar_t>(updated_crit);
    }
    scalar_t updated = net_weights[net_id] *
        (static_cast<scalar_t>(1.0) + net_criticality[net_id]);
    if (limit_weight && updated > max_weight) {
      updated = max_weight;
    }
    net_weights[net_id] = updated;
  }
}

template <typename scalar_t>
std::pair<int64_t, int64_t> apply_pin2pin_kernel(
    const std::vector<std::vector<int64_t>>& paths,
    const int64_t* pin2node_map,
    int64_t num_pins,
    const scalar_t* pin_slack,
    double wns,
    double min_weight,
    double max_weight,
    double accumulate_weight,
    py::dict& pin2pin_net_weight) {
  std::pair<int64_t, int64_t> counts{0, 0};
  if (!(std::isfinite(wns) && wns < 0.0)) {
    return counts;
  }

  const bool limit_weight = std::isfinite(max_weight);

  for (const auto& path : paths) {
    if (path.empty()) {
      continue;
    }
    int64_t sink_pin = path.back();
    if (sink_pin < 0 || sink_pin >= num_pins) {
      continue;
    }
    scalar_t path_slack = pin_slack[sink_pin];
    double scale = (wns != 0.0) ? static_cast<double>(path_slack) / wns : 0.0;

    int64_t prev_pin = -1;
    int64_t prev_node = -1;
    for (int64_t pin : path) {
      if (pin < 0 || pin >= num_pins) {
        continue;
      }
      int64_t node = pin2node_map[pin];
      if (prev_pin >= 0) {
        if (node == prev_node) {
          prev_pin = pin;
          prev_node = node;
          continue;
        }
        py::tuple key = py::make_tuple(prev_pin, pin);
        if (pin2pin_net_weight.contains(key)) {
          double value = pin2pin_net_weight[key].cast<double>();
          value += accumulate_weight * scale;
          if (limit_weight && value > max_weight) {
            value = max_weight;
          }
          pin2pin_net_weight[key] = value;
          counts.first += 1;
        } else {
          pin2pin_net_weight[key] = min_weight;
          counts.second += 1;
        }
      }
      prev_pin = pin;
      prev_node = node;
    }
  }

  return counts;
}

}  // namespace

struct PinPairBatch {
  at::Tensor pair_keys;
  at::Tensor pair_weights;
  int64_t raw_pair_count;
  int64_t unique_pair_count;
  int64_t updated_pair_count;
  int64_t clamped_pair_count;
  double aggregation_runtime_ms;
};

PinPairBatch accumulate_pin2pin_pairs(
    at::Tensor path_offsets,
    at::Tensor path_pins,
    at::Tensor endpoint_slacks,
    at::Tensor path_valid,
    at::Tensor pin2node_map,
    at::Tensor existing_pair_keys,
    at::Tensor existing_pair_weights,
    double wns,
    double min_weight,
    double max_weight,
    double accumulate_weight) {
  const auto started_at = std::chrono::steady_clock::now();
  for (const auto& item : std::vector<std::pair<const char*, at::Tensor>>{
           {"path_offsets", path_offsets},
           {"path_pins", path_pins},
           {"endpoint_slacks", endpoint_slacks},
           {"path_valid", path_valid},
           {"pin2node_map", pin2node_map},
           {"existing_pair_keys", existing_pair_keys},
           {"existing_pair_weights", existing_pair_weights},
       }) {
    TORCH_CHECK(item.second.defined(), item.first, " must be defined");
    TORCH_CHECK(item.second.device().is_cpu(), item.first, " must reside on CPU");
  }
  TORCH_CHECK(path_offsets.dim() == 1, "path_offsets must be 1-D");
  TORCH_CHECK(path_pins.dim() == 1, "path_pins must be 1-D");
  TORCH_CHECK(endpoint_slacks.dim() == 1, "endpoint_slacks must be 1-D");
  TORCH_CHECK(path_valid.dim() == 1, "path_valid must be 1-D");
  TORCH_CHECK(pin2node_map.dim() == 1, "pin2node_map must be 1-D");
  TORCH_CHECK(existing_pair_keys.dim() == 2 && existing_pair_keys.size(1) == 2,
              "existing_pair_keys must have shape (N, 2)");
  TORCH_CHECK(existing_pair_weights.dim() == 1,
              "existing_pair_weights must be 1-D");
  TORCH_CHECK(existing_pair_keys.size(0) == existing_pair_weights.numel(),
              "existing pair key and weight lengths must match");
  TORCH_CHECK(endpoint_slacks.is_floating_point(),
              "endpoint_slacks must be floating point");
  TORCH_CHECK(existing_pair_weights.is_floating_point(),
              "existing_pair_weights must be floating point");
  TORCH_CHECK(std::isfinite(wns) && wns < 0.0,
              "wns must be finite and negative");
  TORCH_CHECK(std::isfinite(min_weight) && min_weight >= 0.0,
              "min_weight must be finite and non-negative");
  TORCH_CHECK((std::isfinite(max_weight) || std::isinf(max_weight)) &&
                  max_weight >= min_weight,
              "max_weight must be >= min_weight");
  TORCH_CHECK(std::isfinite(accumulate_weight) && accumulate_weight >= 0.0,
              "accumulate_weight must be finite and non-negative");

  auto offsets = path_offsets.contiguous().to(at::kLong);
  auto pins = path_pins.contiguous().to(at::kLong);
  auto slacks = endpoint_slacks.contiguous().to(at::kDouble);
  auto valid = path_valid.contiguous().to(at::kBool);
  auto pin_nodes = pin2node_map.contiguous().to(at::kLong);
  auto old_keys = existing_pair_keys.contiguous().to(at::kLong);
  auto old_weights = existing_pair_weights.contiguous().to(at::kDouble);

  TORCH_CHECK(offsets.numel() >= 1,
              "path_offsets must contain at least the zero offset");
  const int64_t num_paths = offsets.numel() - 1;
  TORCH_CHECK(slacks.numel() == num_paths && valid.numel() == num_paths,
              "path offsets, endpoint slacks, and valid mask lengths must match");
  const int64_t* offset_ptr = offsets.data_ptr<int64_t>();
  TORCH_CHECK(offset_ptr[0] == 0, "path_offsets must start at zero");
  TORCH_CHECK(offset_ptr[num_paths] == pins.numel(),
              "last path offset must equal path_pins length");
  for (int64_t idx = 1; idx <= num_paths; ++idx) {
    TORCH_CHECK(offset_ptr[idx - 1] <= offset_ptr[idx],
                "path_offsets must be non-decreasing");
  }

  const int64_t num_pins = pin_nodes.numel();
  const int64_t* pin_ptr = pins.data_ptr<int64_t>();
  const int64_t* node_ptr = pin_nodes.data_ptr<int64_t>();
  const double* slack_ptr = slacks.data_ptr<double>();
  const bool* valid_ptr = valid.data_ptr<bool>();

  std::map<std::pair<int64_t, int64_t>, double> weights;
  const int64_t* old_key_ptr = old_keys.data_ptr<int64_t>();
  const double* old_weight_ptr = old_weights.data_ptr<double>();
  for (int64_t idx = 0; idx < old_keys.size(0); ++idx) {
    const int64_t src = old_key_ptr[2 * idx];
    const int64_t dst = old_key_ptr[2 * idx + 1];
    TORCH_CHECK(src >= 0 && src < num_pins && dst >= 0 && dst < num_pins,
                "existing_pair_keys contains out-of-range pin id");
    const double weight = old_weight_ptr[idx];
    TORCH_CHECK(std::isfinite(weight),
                "existing_pair_weights contains non-finite value");
    TORCH_CHECK(weights.emplace(std::make_pair(src, dst), weight).second,
                "existing_pair_keys contains duplicate pair");
  }

  int64_t raw_pair_count = 0;
  int64_t unique_pair_count = 0;
  int64_t updated_pair_count = 0;
  int64_t clamped_pair_count = 0;
  for (int64_t path_idx = 0; path_idx < num_paths; ++path_idx) {
    if (!valid_ptr[path_idx]) {
      continue;
    }
    const int64_t begin = offset_ptr[path_idx];
    const int64_t end = offset_ptr[path_idx + 1];
    TORCH_CHECK(begin < end, "valid path must contain at least one pin");
    const double path_slack = slack_ptr[path_idx];
    TORCH_CHECK(std::isfinite(path_slack) && path_slack < 0.0,
                "valid path endpoint slack must be finite and negative");
    for (int64_t point_idx = begin + 1; point_idx < end; ++point_idx) {
      const int64_t src = pin_ptr[point_idx - 1];
      const int64_t dst = pin_ptr[point_idx];
      TORCH_CHECK(src >= 0 && src < num_pins && dst >= 0 && dst < num_pins,
                  "path_pins contains out-of-range pin id");
      if (node_ptr[src] == node_ptr[dst]) {
        continue;
      }
      raw_pair_count += 1;
      const auto key = std::make_pair(src, dst);
      auto [it, inserted] = weights.emplace(key, min_weight);
      if (inserted) {
        unique_pair_count += 1;
        continue;
      }
      updated_pair_count += 1;
      it->second += accumulate_weight * path_slack / wns;
      if (std::isfinite(max_weight) && it->second > max_weight) {
        it->second = max_weight;
        clamped_pair_count += 1;
      }
    }
  }

  auto result_keys = at::empty(
      {static_cast<int64_t>(weights.size()), 2},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  auto result_weights = at::empty(
      {static_cast<int64_t>(weights.size())},
      at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
  int64_t* result_key_ptr = result_keys.data_ptr<int64_t>();
  double* result_weight_ptr = result_weights.data_ptr<double>();
  int64_t result_idx = 0;
  for (const auto& [key, weight] : weights) {
    result_key_ptr[2 * result_idx] = key.first;
    result_key_ptr[2 * result_idx + 1] = key.second;
    result_weight_ptr[result_idx] = weight;
    result_idx += 1;
  }

  const auto finished_at = std::chrono::steady_clock::now();
  PinPairBatch result;
  result.pair_keys = std::move(result_keys);
  result.pair_weights = std::move(result_weights);
  result.raw_pair_count = raw_pair_count;
  result.unique_pair_count = unique_pair_count;
  result.updated_pair_count = updated_pair_count;
  result.clamped_pair_count = clamped_pair_count;
  result.aggregation_runtime_ms = std::chrono::duration<double, std::milli>(
      finished_at - started_at).count();
  return result;
}

void apply_adams(
    at::Tensor net_weights,
    at::Tensor net_criticality,
    at::Tensor degree_map,
    int64_t ignore_net_degree,
    at::Tensor pin2net_map,
    const std::vector<std::vector<int64_t>>& paths,
    double max_net_weight) {
  TORCH_CHECK(net_weights.device().is_cpu(), "net_weights must reside on CPU");
  TORCH_CHECK(net_criticality.device().is_cpu(), "net_criticality must reside on CPU");
  TORCH_CHECK(degree_map.device().is_cpu(), "degree_map must reside on CPU");
  TORCH_CHECK(pin2net_map.device().is_cpu(), "pin2net_map must reside on CPU");

  TORCH_CHECK(net_weights.is_contiguous(), "net_weights must be contiguous");
  TORCH_CHECK(net_criticality.is_contiguous(), "net_criticality must be contiguous");
  TORCH_CHECK(degree_map.is_contiguous(), "degree_map must be contiguous");
  TORCH_CHECK(pin2net_map.is_contiguous(), "pin2net_map must be contiguous");

  TORCH_CHECK(net_weights.dim() == 1, "net_weights must be 1-D");
  TORCH_CHECK(net_criticality.dim() == 1, "net_criticality must be 1-D");
  TORCH_CHECK(degree_map.dim() == 1, "degree_map must be 1-D");
  TORCH_CHECK(pin2net_map.dim() == 1, "pin2net_map must be 1-D");

  TORCH_CHECK(net_weights.numel() == net_criticality.numel(), "net_weights and net_criticality size mismatch");
  TORCH_CHECK(degree_map.numel() == net_weights.numel(), "degree_map size mismatch");

  TORCH_CHECK(net_weights.scalar_type() == net_criticality.scalar_type(), "net_weights and net_criticality dtype mismatch");
  TORCH_CHECK(degree_map.scalar_type() == at::kLong, "degree_map must be int64");
  TORCH_CHECK(pin2net_map.scalar_type() == at::kLong, "pin2net_map must be int64");

  int64_t num_nets = net_weights.numel();
  int64_t num_pins = pin2net_map.numel();

  auto* degree_ptr = degree_map.data_ptr<int64_t>();
  auto* pin2net_ptr = pin2net_map.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(net_weights.scalar_type(), "dreamplace_apply_adams", [&] {
    apply_adams_kernel<scalar_t>(
        net_weights.data_ptr<scalar_t>(),
        net_criticality.data_ptr<scalar_t>(),
        degree_ptr,
        ignore_net_degree,
        pin2net_ptr,
        num_pins,
        num_nets,
        paths,
        max_net_weight);
  });
}

void apply_lilith(
    at::Tensor net_weights,
    at::Tensor net_criticality,
    at::Tensor degree_map,
    int64_t ignore_net_degree,
    at::Tensor flat_netpin,
    at::Tensor netpin_start,
    at::Tensor pin_slack,
    double wns,
    double decay,
    double max_net_weight) {
  TORCH_CHECK(net_weights.device().is_cpu(), "net_weights must reside on CPU");
  TORCH_CHECK(net_criticality.device().is_cpu(), "net_criticality must reside on CPU");
  TORCH_CHECK(degree_map.device().is_cpu(), "degree_map must reside on CPU");
  TORCH_CHECK(flat_netpin.device().is_cpu(), "flat_netpin must reside on CPU");
  TORCH_CHECK(netpin_start.device().is_cpu(), "netpin_start must reside on CPU");
  TORCH_CHECK(pin_slack.device().is_cpu(), "pin_slack must reside on CPU");

  TORCH_CHECK(net_weights.is_contiguous(), "net_weights must be contiguous");
  TORCH_CHECK(net_criticality.is_contiguous(), "net_criticality must be contiguous");
  TORCH_CHECK(degree_map.is_contiguous(), "degree_map must be contiguous");
  TORCH_CHECK(flat_netpin.is_contiguous(), "flat_netpin must be contiguous");
  TORCH_CHECK(netpin_start.is_contiguous(), "netpin_start must be contiguous");
  TORCH_CHECK(pin_slack.is_contiguous(), "pin_slack must be contiguous");

  TORCH_CHECK(net_weights.dim() == 1, "net_weights must be 1-D");
  TORCH_CHECK(net_criticality.dim() == 1, "net_criticality must be 1-D");
  TORCH_CHECK(degree_map.dim() == 1, "degree_map must be 1-D");
  TORCH_CHECK(flat_netpin.dim() == 1, "flat_netpin must be 1-D");
  TORCH_CHECK(netpin_start.dim() == 1, "netpin_start must be 1-D");
  TORCH_CHECK(pin_slack.dim() == 1, "pin_slack must be 1-D");

  TORCH_CHECK(net_weights.numel() == net_criticality.numel(), "net_weights and net_criticality size mismatch");
  TORCH_CHECK(degree_map.numel() == net_weights.numel(), "degree_map size mismatch");
  TORCH_CHECK(netpin_start.numel() == net_weights.numel() + 1, "netpin_start must have num_nets + 1 elements");

  TORCH_CHECK(net_weights.scalar_type() == net_criticality.scalar_type(), "net_weights and net_criticality dtype mismatch");
  TORCH_CHECK(pin_slack.scalar_type() == net_weights.scalar_type(), "pin_slack dtype must match net_weights");
  TORCH_CHECK(degree_map.scalar_type() == at::kLong, "degree_map must be int64");
  TORCH_CHECK(flat_netpin.scalar_type() == at::kLong, "flat_netpin must be int64");
  TORCH_CHECK(netpin_start.scalar_type() == at::kLong, "netpin_start must be int64");

  int64_t num_nets = net_weights.numel();
  int64_t num_pins = pin_slack.numel();

  auto* degree_ptr = degree_map.data_ptr<int64_t>();
  auto* flat_netpin_ptr = flat_netpin.data_ptr<int64_t>();
  auto* netpin_start_ptr = netpin_start.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(net_weights.scalar_type(), "dreamplace_apply_lilith", [&] {
    apply_lilith_kernel<scalar_t>(
        net_weights.data_ptr<scalar_t>(),
        net_criticality.data_ptr<scalar_t>(),
        degree_ptr,
        ignore_net_degree,
        flat_netpin_ptr,
        netpin_start_ptr,
        num_nets,
        num_pins,
        pin_slack.data_ptr<scalar_t>(),
        wns,
        decay,
        max_net_weight);
  });
}

std::pair<int64_t, int64_t> apply_pin2pin(
    const std::vector<std::vector<int64_t>>& paths,
    at::Tensor pin2node_map,
    at::Tensor pin_slack,
    double wns,
    double min_weight,
    double max_weight,
    double accumulate_weight,
    py::dict pin2pin_net_weight) {
  TORCH_CHECK(pin2node_map.device().is_cpu(), "pin2node_map must reside on CPU");
  TORCH_CHECK(pin_slack.device().is_cpu(), "pin_slack must reside on CPU");

  TORCH_CHECK(pin2node_map.is_contiguous(), "pin2node_map must be contiguous");
  TORCH_CHECK(pin_slack.is_contiguous(), "pin_slack must be contiguous");

  TORCH_CHECK(pin2node_map.dim() == 1, "pin2node_map must be 1-D");
  TORCH_CHECK(pin_slack.dim() == 1, "pin_slack must be 1-D");

  TORCH_CHECK(pin2node_map.scalar_type() == at::kLong, "pin2node_map must be int64");
  TORCH_CHECK(pin_slack.scalar_type() == at::kFloat || pin_slack.scalar_type() == at::kDouble,
              "pin_slack must be float or double");

  int64_t num_pins = pin2node_map.numel();
  TORCH_CHECK(pin_slack.numel() >= num_pins, "pin_slack length must be >= number of pins");

  auto* pin2node_ptr = pin2node_map.data_ptr<int64_t>();
  std::pair<int64_t, int64_t> counts{0, 0};

  AT_DISPATCH_FLOATING_TYPES(pin_slack.scalar_type(), "dreamplace_apply_pin2pin", [&] {
    counts = apply_pin2pin_kernel<scalar_t>(
        paths,
        pin2node_ptr,
        num_pins,
        pin_slack.data_ptr<scalar_t>(),
        wns,
        min_weight,
        max_weight,
        accumulate_weight,
        pin2pin_net_weight);
  });

  return counts;
}

}  // namespace dreamplace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Timing propagation based net weighting operators";

  py::class_<dreamplace::PinPairBatch>(m, "PinPairBatch")
      .def_readonly("pair_keys", &dreamplace::PinPairBatch::pair_keys)
      .def_readonly("pair_weights", &dreamplace::PinPairBatch::pair_weights)
      .def_readonly("raw_pair_count", &dreamplace::PinPairBatch::raw_pair_count)
      .def_readonly("unique_pair_count", &dreamplace::PinPairBatch::unique_pair_count)
      .def_readonly("updated_pair_count", &dreamplace::PinPairBatch::updated_pair_count)
      .def_readonly("clamped_pair_count", &dreamplace::PinPairBatch::clamped_pair_count)
      .def_readonly("aggregation_runtime_ms", &dreamplace::PinPairBatch::aggregation_runtime_ms);

  m.def(
      "accumulate_pin2pin_pairs",
      &dreamplace::accumulate_pin2pin_pairs,
      py::arg("path_offsets"),
      py::arg("path_pins"),
      py::arg("endpoint_slacks"),
      py::arg("path_valid"),
      py::arg("pin2node_map"),
      py::arg("existing_pair_keys"),
      py::arg("existing_pair_weights"),
      py::arg("wns"),
      py::arg("min_weight"),
      py::arg("max_weight"),
      py::arg("accumulate_weight"),
      "Accumulate packed critical paths into deterministic pair tensors.");

  m.def(
      "apply_adams",
      &dreamplace::apply_adams,
      py::arg("net_weights"),
      py::arg("net_criticality"),
      py::arg("degree_map"),
      py::arg("ignore_net_degree"),
      py::arg("pin2net_map"),
      py::arg("paths"),
      py::arg("max_net_weight"),
      "Apply ADAMS net weighting scheme using timing propagation results.");

  m.def(
      "apply_lilith",
      &dreamplace::apply_lilith,
      py::arg("net_weights"),
      py::arg("net_criticality"),
      py::arg("degree_map"),
      py::arg("ignore_net_degree"),
      py::arg("flat_netpin"),
      py::arg("netpin_start"),
      py::arg("pin_slack"),
      py::arg("wns"),
      py::arg("decay"),
      py::arg("max_net_weight"),
      "Apply LILITH net weighting scheme using timing propagation results.");

  m.def(
      "apply_pin2pin",
      &dreamplace::apply_pin2pin,
      py::arg("paths"),
      py::arg("pin2node_map"),
      py::arg("pin_slack"),
      py::arg("wns"),
      py::arg("min_weight"),
      py::arg("max_weight"),
      py::arg("accumulate_weight"),
      py::arg("pin2pin_net_weight"),
      "Apply pin-to-pin attraction accumulation using timing propagation paths.");
}
