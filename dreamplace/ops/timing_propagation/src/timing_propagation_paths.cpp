#include "timing_propagation_paths.h"
#include "../../utility/src/torch.h"

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dreamplace {
namespace {

struct ArcLookupEntry {
  int32_t start;
  int32_t length;
};

inline uint64_t encode_key(int32_t src, int32_t dst) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32) |
         static_cast<uint32_t>(dst);
}

inline uint64_t encode_key64(int64_t src, int64_t dst) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32) |
         static_cast<uint32_t>(dst);
}

std::vector<int64_t> tensor_to_i64_vector(const at::Tensor& tensor,
                                          const char* name,
                                          int64_t expected_dim = 1) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), name, " must reside on CPU");
  TORCH_CHECK(tensor.dim() == expected_dim, name, " must be ", expected_dim, "-D");
  auto tensor_i64 = tensor.contiguous().to(at::ScalarType::Long);
  const int64_t* ptr = tensor_i64.data_ptr<int64_t>();
  return std::vector<int64_t>(ptr, ptr + tensor_i64.numel());
}

at::Tensor i64_tensor_from_vector(const std::vector<int64_t>& values) {
  auto tensor = at::empty(
      {static_cast<int64_t>(values.size())},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), tensor.data_ptr<int64_t>());
  }
  return tensor;
}

std::vector<double> tensor_to_double_vector(const at::Tensor& tensor,
                                            const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), name, " must reside on CPU");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  TORCH_CHECK(tensor.is_floating_point(), name, " must be floating point");
  auto tensor_double = tensor.contiguous().to(at::ScalarType::Double);
  const double* ptr = tensor_double.data_ptr<double>();
  return std::vector<double>(ptr, ptr + tensor_double.numel());
}

at::Tensor double_tensor_from_vector(const std::vector<double>& values) {
  auto tensor = at::empty(
      {static_cast<int64_t>(values.size())},
      at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), tensor.data_ptr<double>());
  }
  return tensor;
}

at::Tensor bool_tensor_from_vector(const std::vector<uint8_t>& values) {
  auto tensor = at::empty(
      {static_cast<int64_t>(values.size())},
      at::TensorOptions().dtype(at::kBool).device(at::kCPU));
  bool* ptr = tensor.data_ptr<bool>();
  for (size_t idx = 0; idx < values.size(); ++idx) {
    ptr[idx] = values[idx] != 0;
  }
  return tensor;
}

inline int64_t query_arc_index(
    const std::unordered_map<uint64_t, ArcLookupEntry>& table,
    const int32_t* arc_indices,
    int32_t src,
    int32_t dst) {
  if (!arc_indices || table.empty()) {
    return -1;
  }
  uint64_t key = encode_key(src, dst);
  auto it = table.find(key);
  if (it == table.end() || it->second.length <= 0) {
    return -1;
  }
  return static_cast<int64_t>(arc_indices[it->second.start]);
}

}  // namespace

SetupCriticalPathExtractor::SetupCriticalPathExtractor(
    const at::Tensor& flat_inst_arcs_by_level,
    const at::Tensor& pin_pred_start,
    const at::Tensor& pin_pred_pin,
    const at::Tensor& pin_pred_arc_id,
    const at::Tensor& start_points,
    int64_t topology_epoch)
    : topology_epoch_(topology_epoch) {
  TORCH_CHECK(flat_inst_arcs_by_level.defined(),
              "flat_inst_arcs_by_level must be defined");
  TORCH_CHECK(flat_inst_arcs_by_level.device().is_cpu(),
              "flat_inst_arcs_by_level must reside on CPU");
  TORCH_CHECK(flat_inst_arcs_by_level.dim() == 2,
              "flat_inst_arcs_by_level must be 2-D");
  TORCH_CHECK(flat_inst_arcs_by_level.size(1) >= 5,
              "flat_inst_arcs_by_level must contain timing sense column");

  auto flat_arcs = flat_inst_arcs_by_level.contiguous().to(at::ScalarType::Long);
  const int64_t* flat_arc_ptr = flat_arcs.data_ptr<int64_t>();
  const int64_t arc_cols = flat_arcs.size(1);
  num_cell_arcs_ = flat_arcs.size(0);
  arc_senses_.resize(static_cast<size_t>(num_cell_arcs_));
  for (int64_t arc_id = 0; arc_id < num_cell_arcs_; ++arc_id) {
    const int64_t sense = flat_arc_ptr[arc_id * arc_cols + 4];
    TORCH_CHECK(sense == -1 || sense == 0 || sense == 1,
                "unsupported timing sense for cell arc ", arc_id, ": ", sense);
    arc_senses_[static_cast<size_t>(arc_id)] = sense;
  }

  predecessor_offsets_ = tensor_to_i64_vector(pin_pred_start, "pin_pred_start");
  predecessor_pins_ = tensor_to_i64_vector(pin_pred_pin, "pin_pred_pin");
  predecessor_arc_ids_ = tensor_to_i64_vector(pin_pred_arc_id, "pin_pred_arc_id");
  TORCH_CHECK(!predecessor_offsets_.empty(),
              "pin_pred_start must contain at least one offset");
  TORCH_CHECK(predecessor_offsets_.front() == 0,
              "pin_pred_start must start at zero");
  TORCH_CHECK(predecessor_offsets_.back() ==
                  static_cast<int64_t>(predecessor_pins_.size()),
              "pin_pred_start last offset must equal predecessor edge count");
  TORCH_CHECK(predecessor_arc_ids_.size() == predecessor_pins_.size(),
              "pin_pred_arc_id length must match pin_pred_pin");
  num_pins_ = static_cast<int64_t>(predecessor_offsets_.size()) - 1;
  for (size_t idx = 1; idx < predecessor_offsets_.size(); ++idx) {
    TORCH_CHECK(predecessor_offsets_[idx - 1] <= predecessor_offsets_[idx],
                "pin_pred_start must be non-decreasing");
  }
  for (size_t edge_idx = 0; edge_idx < predecessor_pins_.size(); ++edge_idx) {
    const int64_t pred = predecessor_pins_[edge_idx];
    const int64_t arc_id = predecessor_arc_ids_[edge_idx];
    TORCH_CHECK(pred >= 0 && pred < num_pins_,
                "pin_pred_pin contains out-of-range pin id");
    TORCH_CHECK(arc_id == -1 || (arc_id >= 0 && arc_id < num_cell_arcs_),
                "pin_pred_arc_id entries must be -1 or valid flat arc ids");
  }

  is_start_pin_.assign(static_cast<size_t>(num_pins_), 0);
  const auto starts = tensor_to_i64_vector(start_points, "start_points");
  for (const int64_t pin : starts) {
    TORCH_CHECK(pin >= 0 && pin < num_pins_,
                "start_points contains out-of-range pin id");
    is_start_pin_[static_cast<size_t>(pin)] = 1;
  }
}

CriticalPathBatch SetupCriticalPathExtractor::extract(
    const at::Tensor& endpoint_pins,
    const at::Tensor& endpoint_test_ids,
    const at::Tensor& endpoint_rise_slack,
    const at::Tensor& endpoint_fall_slack,
    const at::Tensor& pin_rise_aat,
    const at::Tensor& pin_fall_aat,
    const at::Tensor& pin_net_delay_rise,
    const at::Tensor& pin_net_delay_fall,
    const at::Tensor& cell_delay_rr,
    const at::Tensor& cell_delay_fr,
    const at::Tensor& cell_delay_rf,
    const at::Tensor& cell_delay_ff,
    int64_t global_k,
    int64_t max_depth,
    double residual_tolerance_ps) const {
  const auto started_at = std::chrono::steady_clock::now();
  TORCH_CHECK(global_k >= 0, "global_k must be non-negative");
  TORCH_CHECK(std::isfinite(residual_tolerance_ps) && residual_tolerance_ps > 0.0,
              "residual_tolerance_ps must be finite and positive");

  const auto endpoints = tensor_to_i64_vector(endpoint_pins, "endpoint_pins");
  const auto test_ids = tensor_to_i64_vector(endpoint_test_ids, "endpoint_test_ids");
  const auto rise_slacks = tensor_to_double_vector(
      endpoint_rise_slack, "endpoint_rise_slack");
  const auto fall_slacks = tensor_to_double_vector(
      endpoint_fall_slack, "endpoint_fall_slack");
  TORCH_CHECK(endpoints.size() == test_ids.size() &&
                  endpoints.size() == rise_slacks.size() &&
                  endpoints.size() == fall_slacks.size(),
              "endpoint pins, test ids, and slack tensors must have matching lengths");

  std::unordered_set<uint64_t> unique_endpoint_tests;
  for (size_t idx = 0; idx < endpoints.size(); ++idx) {
    const int64_t pin = endpoints[idx];
    const int64_t test_id = test_ids[idx];
    TORCH_CHECK(pin >= 0 && pin < num_pins_,
                "endpoint_pins contains out-of-range pin id");
    TORCH_CHECK(test_id >= -1, "endpoint_test_ids entries must be >= -1");
    const uint64_t key = encode_key64(pin, test_id);
    TORCH_CHECK(unique_endpoint_tests.insert(key).second,
                "endpoint pin/test pairs must be unique");
  }

  const auto rise_aat = tensor_to_double_vector(pin_rise_aat, "pin_rise_aat");
  const auto fall_aat = tensor_to_double_vector(pin_fall_aat, "pin_fall_aat");
  const auto net_delay_rise = tensor_to_double_vector(
      pin_net_delay_rise, "pin_net_delay_rise");
  const auto net_delay_fall = tensor_to_double_vector(
      pin_net_delay_fall, "pin_net_delay_fall");
  TORCH_CHECK(static_cast<int64_t>(rise_aat.size()) == num_pins_ &&
                  static_cast<int64_t>(fall_aat.size()) == num_pins_,
              "pin AAT tensor length must equal number of pins");
  TORCH_CHECK(static_cast<int64_t>(net_delay_rise.size()) >= num_pins_ &&
                  static_cast<int64_t>(net_delay_fall.size()) >= num_pins_,
              "pin net delay tensor length must cover all pins");

  const auto delay_rr = tensor_to_double_vector(cell_delay_rr, "cell_delay_rr");
  const auto delay_fr = tensor_to_double_vector(cell_delay_fr, "cell_delay_fr");
  const auto delay_rf = tensor_to_double_vector(cell_delay_rf, "cell_delay_rf");
  const auto delay_ff = tensor_to_double_vector(cell_delay_ff, "cell_delay_ff");
  TORCH_CHECK(static_cast<int64_t>(delay_rr.size()) == num_cell_arcs_ &&
                  static_cast<int64_t>(delay_fr.size()) == num_cell_arcs_ &&
                  static_cast<int64_t>(delay_rf.size()) == num_cell_arcs_ &&
                  static_cast<int64_t>(delay_ff.size()) == num_cell_arcs_,
              "cell delay tensor length must equal number of flat cell arcs");

  struct EndpointState {
    int64_t pin;
    int64_t test_id;
    int64_t transition;
    double slack;
  };
  std::vector<EndpointState> states;
  states.reserve(endpoints.size() * 2);
  for (size_t idx = 0; idx < endpoints.size(); ++idx) {
    if (std::isfinite(rise_slacks[idx]) && rise_slacks[idx] < 0.0) {
      states.push_back({endpoints[idx], test_ids[idx], 0, rise_slacks[idx]});
    }
    if (std::isfinite(fall_slacks[idx]) && fall_slacks[idx] < 0.0) {
      states.push_back({endpoints[idx], test_ids[idx], 1, fall_slacks[idx]});
    }
  }
  std::sort(states.begin(), states.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.slack, lhs.transition, lhs.pin, lhs.test_id) <
           std::tie(rhs.slack, rhs.transition, rhs.pin, rhs.test_id);
  });
  const int64_t failing_state_count = static_cast<int64_t>(states.size());
  if (global_k > 0 && static_cast<int64_t>(states.size()) > global_k) {
    states.resize(static_cast<size_t>(global_k));
  }

  if (max_depth <= 0) {
    max_depth = std::max<int64_t>(1, num_pins_);
  }

  struct BacktraceResult {
    std::vector<int64_t> pins;
    std::vector<int64_t> transitions;
    std::vector<int64_t> arc_ids;
    uint8_t valid {0};
    int64_t invalid_reason {0};
    double max_residual_ps {0.0};
  };
  std::vector<BacktraceResult> results(states.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t state_idx = 0;
       state_idx < static_cast<int64_t>(states.size());
       ++state_idx) {
    const auto& state = states[static_cast<size_t>(state_idx)];
    auto& result = results[static_cast<size_t>(state_idx)];
    int64_t current_pin = state.pin;
    int64_t current_transition = state.transition;
    result.pins.push_back(current_pin);
    result.transitions.push_back(current_transition);
    std::unordered_set<int64_t> visited;
    visited.insert(current_pin);

    bool reached_start = false;
    for (int64_t depth = 0; depth <= max_depth; ++depth) {
      const bool current_is_start =
          is_start_pin_[static_cast<size_t>(current_pin)] != 0;
      if (depth == max_depth) {
        if (current_is_start) {
          reached_start = true;
        } else {
          result.invalid_reason = 3;
        }
        break;
      }

      const int64_t edge_begin = predecessor_offsets_[static_cast<size_t>(current_pin)];
      const int64_t edge_end = predecessor_offsets_[static_cast<size_t>(current_pin + 1)];
      struct Candidate {
        bool defined {false};
        int64_t pred_pin {-1};
        int64_t pred_transition {-1};
        int64_t arc_id {-2};
        double aat {-std::numeric_limits<double>::infinity()};
        double residual {std::numeric_limits<double>::infinity()};
      } best;
      const double current_aat = current_transition == 0
          ? rise_aat[static_cast<size_t>(current_pin)]
          : fall_aat[static_cast<size_t>(current_pin)];

      auto consider = [&](int64_t pred_pin,
                          int64_t pred_transition,
                          int64_t arc_id,
                          double candidate_aat) {
        if (!std::isfinite(candidate_aat)) {
          return;
        }
        const double residual = std::abs(current_aat - candidate_aat);
        const double epsilon = 1.0e-12;
        const bool better = !best.defined ||
            candidate_aat > best.aat + epsilon ||
            (std::abs(candidate_aat - best.aat) <= epsilon &&
             std::tie(residual, arc_id, pred_pin, pred_transition) <
             std::tie(best.residual, best.arc_id, best.pred_pin,
                      best.pred_transition));
        if (better) {
          best = {true, pred_pin, pred_transition, arc_id,
                  candidate_aat, residual};
        }
      };

      for (int64_t edge_idx = edge_begin; edge_idx < edge_end; ++edge_idx) {
        const int64_t pred_pin = predecessor_pins_[static_cast<size_t>(edge_idx)];
        const int64_t arc_id = predecessor_arc_ids_[static_cast<size_t>(edge_idx)];
        if (arc_id == -1) {
          const double pred_aat = current_transition == 0
              ? rise_aat[static_cast<size_t>(pred_pin)]
              : fall_aat[static_cast<size_t>(pred_pin)];
          const double net_delay = current_transition == 0
              ? net_delay_rise[static_cast<size_t>(current_pin)]
              : net_delay_fall[static_cast<size_t>(current_pin)];
          consider(pred_pin, current_transition, arc_id, pred_aat + net_delay);
          continue;
        }

        const int64_t sense = arc_senses_[static_cast<size_t>(arc_id)];
        if (current_transition == 0) {
          if (sense >= 0) {
            consider(pred_pin, 0, arc_id,
                     rise_aat[static_cast<size_t>(pred_pin)] +
                         delay_rr[static_cast<size_t>(arc_id)]);
          }
          if (sense <= 0) {
            consider(pred_pin, 1, arc_id,
                     fall_aat[static_cast<size_t>(pred_pin)] +
                         delay_fr[static_cast<size_t>(arc_id)]);
          }
        } else {
          if (sense >= 0) {
            consider(pred_pin, 1, arc_id,
                     fall_aat[static_cast<size_t>(pred_pin)] +
                         delay_ff[static_cast<size_t>(arc_id)]);
          }
          if (sense <= 0) {
            consider(pred_pin, 0, arc_id,
                     rise_aat[static_cast<size_t>(pred_pin)] +
                         delay_rf[static_cast<size_t>(arc_id)]);
          }
        }
      }

      // A sequential output is only a path start when its launch state is not
      // explained by an in-graph async control arc for this transition.
      if (current_is_start &&
          (!best.defined || best.residual > residual_tolerance_ps)) {
        reached_start = true;
        break;
      }
      if (!best.defined) {
        result.invalid_reason = 1;
        break;
      }
      if (visited.find(best.pred_pin) != visited.end()) {
        result.invalid_reason = 2;
        break;
      }
      result.max_residual_ps = std::max(result.max_residual_ps, best.residual);
      if (best.residual > residual_tolerance_ps && result.invalid_reason == 0) {
        result.invalid_reason = 4;
      }
      result.pins.push_back(best.pred_pin);
      result.transitions.push_back(best.pred_transition);
      result.arc_ids.push_back(best.arc_id);
      visited.insert(best.pred_pin);
      current_pin = best.pred_pin;
      current_transition = best.pred_transition;
    }

    result.valid = reached_start && result.invalid_reason == 0;
    std::reverse(result.pins.begin(), result.pins.end());
    std::reverse(result.transitions.begin(), result.transitions.end());
    std::reverse(result.arc_ids.begin(), result.arc_ids.end());
  }

  std::vector<int64_t> path_offsets;
  std::vector<int64_t> path_pins;
  std::vector<int64_t> path_transitions;
  std::vector<int64_t> path_arc_ids;
  std::vector<int64_t> selected_pins;
  std::vector<int64_t> selected_test_ids;
  std::vector<int64_t> selected_transitions;
  std::vector<double> selected_slacks;
  std::vector<uint8_t> path_valid;
  std::vector<int64_t> invalid_reasons;
  std::vector<double> max_residuals;
  path_offsets.reserve(results.size() + 1);
  path_offsets.push_back(0);
  int64_t valid_path_count = 0;
  for (size_t idx = 0; idx < results.size(); ++idx) {
    const auto& result = results[idx];
    path_pins.insert(path_pins.end(), result.pins.begin(), result.pins.end());
    path_transitions.insert(
        path_transitions.end(), result.transitions.begin(), result.transitions.end());
    path_arc_ids.insert(path_arc_ids.end(), result.arc_ids.begin(), result.arc_ids.end());
    path_offsets.push_back(static_cast<int64_t>(path_pins.size()));
    selected_pins.push_back(states[idx].pin);
    selected_test_ids.push_back(states[idx].test_id);
    selected_transitions.push_back(states[idx].transition);
    selected_slacks.push_back(states[idx].slack);
    path_valid.push_back(result.valid);
    invalid_reasons.push_back(result.invalid_reason);
    max_residuals.push_back(result.max_residual_ps);
    valid_path_count += result.valid ? 1 : 0;
  }

  const auto finished_at = std::chrono::steady_clock::now();
  CriticalPathBatch batch;
  batch.path_offsets = i64_tensor_from_vector(path_offsets);
  batch.path_pins = i64_tensor_from_vector(path_pins);
  batch.path_transitions = i64_tensor_from_vector(path_transitions);
  batch.path_arc_ids = i64_tensor_from_vector(path_arc_ids);
  batch.endpoint_pins = i64_tensor_from_vector(selected_pins);
  batch.endpoint_test_ids = i64_tensor_from_vector(selected_test_ids);
  batch.endpoint_transitions = i64_tensor_from_vector(selected_transitions);
  batch.endpoint_slacks = double_tensor_from_vector(selected_slacks);
  batch.path_valid = bool_tensor_from_vector(path_valid);
  batch.invalid_reason = i64_tensor_from_vector(invalid_reasons);
  batch.max_residual_ps = double_tensor_from_vector(max_residuals);
  batch.failing_state_count = failing_state_count;
  batch.selected_state_count = static_cast<int64_t>(states.size());
  batch.valid_path_count = valid_path_count;
  batch.invalid_path_count = static_cast<int64_t>(states.size()) - valid_path_count;
  batch.topology_epoch = topology_epoch_;
  batch.extraction_runtime_ms = std::chrono::duration<double, std::milli>(
      finished_at - started_at).count();
  return batch;
}

void CriticalEndpointTraversalPruner::initialize_flat_arcs_and_levels(
    const at::Tensor& flat_inst_arcs_by_level,
    const at::Tensor& flat_inst_arcs_by_level_start,
    const at::Tensor& pin2node_map) {
  TORCH_CHECK(flat_inst_arcs_by_level.defined(), "flat_inst_arcs_by_level must be defined");
  TORCH_CHECK(flat_inst_arcs_by_level.device().is_cpu(),
              "flat_inst_arcs_by_level must reside on CPU");
  TORCH_CHECK(flat_inst_arcs_by_level.dim() == 2,
              "flat_inst_arcs_by_level must be 2-D");
  TORCH_CHECK(flat_inst_arcs_by_level.size(1) >= 2,
              "flat_inst_arcs_by_level must contain at least src/dst pin columns");

  auto flat_arcs = flat_inst_arcs_by_level.contiguous().to(at::ScalarType::Long);
  const int64_t* flat_arc_ptr = flat_arcs.data_ptr<int64_t>();
  const int64_t arc_cols = flat_arcs.size(1);
  num_flat_arcs_ = flat_arcs.size(0);

  level_offsets_ = tensor_to_i64_vector(
      flat_inst_arcs_by_level_start,
      "flat_inst_arcs_by_level_start");
  TORCH_CHECK(level_offsets_.size() >= 1,
              "flat_inst_arcs_by_level_start must contain at least one offset");
  num_levels_ = static_cast<int64_t>(level_offsets_.size()) - 1;
  TORCH_CHECK(level_offsets_.front() == 0,
              "flat_inst_arcs_by_level_start must start at zero");
  TORCH_CHECK(level_offsets_.back() == num_flat_arcs_,
              "flat_inst_arcs_by_level_start last offset must equal number of flat arcs");
  for (size_t i = 1; i < level_offsets_.size(); ++i) {
    TORCH_CHECK(level_offsets_[i - 1] <= level_offsets_[i],
                "flat_inst_arcs_by_level_start must be non-decreasing");
  }

  flat_arc_level_id_.assign(static_cast<size_t>(num_flat_arcs_), -1);
  for (int64_t level = 0; level < num_levels_; ++level) {
    const int64_t begin = level_offsets_[level];
    const int64_t end = level_offsets_[level + 1];
    for (int64_t arc = begin; arc < end; ++arc) {
      flat_arc_level_id_[static_cast<size_t>(arc)] = level;
    }
  }

  std::vector<int64_t> pin_to_node;
  if (pin2node_map.defined() && pin2node_map.numel() > 0) {
    pin_to_node = tensor_to_i64_vector(pin2node_map, "pin2node_map");
  }
  flat_arc_to_inst_id_.assign(static_cast<size_t>(num_flat_arcs_), -1);
  for (int64_t arc = 0; arc < num_flat_arcs_; ++arc) {
    const int64_t src_pin = flat_arc_ptr[arc * arc_cols + 0];
    const int64_t dst_pin = flat_arc_ptr[arc * arc_cols + 1];
    int64_t inst_id = -1;
    if (dst_pin >= 0 && dst_pin < static_cast<int64_t>(pin_to_node.size())) {
      inst_id = pin_to_node[static_cast<size_t>(dst_pin)];
    }
    if (inst_id < 0 && src_pin >= 0 && src_pin < static_cast<int64_t>(pin_to_node.size())) {
      inst_id = pin_to_node[static_cast<size_t>(src_pin)];
    }
    flat_arc_to_inst_id_[static_cast<size_t>(arc)] = inst_id;
  }
}

void CriticalEndpointTraversalPruner::initialize_start_points(
    const at::Tensor& start_points) {
  is_start_pin_.assign(static_cast<size_t>(num_pins_), 0);
  if (start_points.defined() && start_points.numel() > 0) {
    auto starts = tensor_to_i64_vector(start_points, "start_points");
    for (int64_t pin : starts) {
      if (pin >= 0 && pin < num_pins_) {
        is_start_pin_[static_cast<size_t>(pin)] = 1;
      }
    }
  }
}

CriticalEndpointTraversalPruner::CriticalEndpointTraversalPruner(
    const at::Tensor& flat_inst_arcs_by_level,
    const at::Tensor& flat_inst_arcs_by_level_start,
    const at::Tensor& flat_pin_to_graph_reverse,
    const at::Tensor& flat_pin_to_graph_start_reverse,
    const at::Tensor& pin_pair_arc_keys,
    const at::Tensor& flat_pin_pair_arc_start,
    const at::Tensor& flat_pin_pair_arc_indices,
    const at::Tensor& start_points,
    const at::Tensor& pin2node_map) {
  initialize_flat_arcs_and_levels(
      flat_inst_arcs_by_level,
      flat_inst_arcs_by_level_start,
      pin2node_map);

  reverse_offsets_ = tensor_to_i64_vector(
      flat_pin_to_graph_start_reverse,
      "flat_pin_to_graph_start_reverse");
  reverse_edges_ = tensor_to_i64_vector(
      flat_pin_to_graph_reverse,
      "flat_pin_to_graph_reverse");
  TORCH_CHECK(reverse_offsets_.size() >= 1,
              "flat_pin_to_graph_start_reverse must contain at least one offset");
  TORCH_CHECK(reverse_offsets_.front() == 0,
              "flat_pin_to_graph_start_reverse must start at zero");
  TORCH_CHECK(reverse_offsets_.back() == static_cast<int64_t>(reverse_edges_.size()),
              "flat_pin_to_graph_start_reverse last offset must equal reverse edge count");
  for (size_t i = 1; i < reverse_offsets_.size(); ++i) {
    TORCH_CHECK(reverse_offsets_[i - 1] <= reverse_offsets_[i],
                "flat_pin_to_graph_start_reverse must be non-decreasing");
  }
  num_pins_ = static_cast<int64_t>(reverse_offsets_.size()) - 1;
  reverse_edge_arc_ids_.assign(reverse_edges_.size(), -2);
  initialize_start_points(start_points);

  const bool has_arc_mapping =
      pin_pair_arc_keys.defined() &&
      flat_pin_pair_arc_start.defined() &&
      flat_pin_pair_arc_indices.defined() &&
      pin_pair_arc_keys.numel() > 0;
  if (has_arc_mapping) {
    TORCH_CHECK(pin_pair_arc_keys.device().is_cpu(),
                "pin_pair_arc_keys must reside on CPU");
    TORCH_CHECK(pin_pair_arc_keys.dim() == 2 && pin_pair_arc_keys.size(1) == 2,
                "pin_pair_arc_keys must have shape (N, 2)");
    auto keys = pin_pair_arc_keys.contiguous().to(at::ScalarType::Long);
    auto starts = tensor_to_i64_vector(
        flat_pin_pair_arc_start,
        "flat_pin_pair_arc_start");
    auto indices = tensor_to_i64_vector(
        flat_pin_pair_arc_indices,
        "flat_pin_pair_arc_indices");
    const int64_t num_pairs = keys.size(0);
    TORCH_CHECK(static_cast<int64_t>(starts.size()) == num_pairs + 1,
                "flat_pin_pair_arc_start length mismatch with pin_pair_arc_keys");

    const int64_t* key_ptr = keys.data_ptr<int64_t>();
    pair_to_arc_indices_.reserve(static_cast<size_t>(num_pairs));
    for (int64_t pair_id = 0; pair_id < num_pairs; ++pair_id) {
      const int64_t src = key_ptr[2 * pair_id + 0];
      const int64_t dst = key_ptr[2 * pair_id + 1];
      const int64_t begin = starts[static_cast<size_t>(pair_id)];
      const int64_t end = starts[static_cast<size_t>(pair_id + 1)];
      TORCH_CHECK(begin <= end, "flat_pin_pair_arc_start must be non-decreasing");
      TORCH_CHECK(begin >= 0 && end <= static_cast<int64_t>(indices.size()),
                  "flat_pin_pair_arc_start points outside flat_pin_pair_arc_indices");
      auto& arc_ids = pair_to_arc_indices_[encode_key64(src, dst)];
      arc_ids.reserve(arc_ids.size() + static_cast<size_t>(end - begin));
      for (int64_t pos = begin; pos < end; ++pos) {
        const int64_t arc_idx = indices[static_cast<size_t>(pos)];
        TORCH_CHECK(arc_idx >= 0 && arc_idx < num_flat_arcs_,
                    "flat_pin_pair_arc_indices contains an out-of-range arc id");
        arc_ids.push_back(arc_idx);
      }
    }
  }
}

CriticalEndpointTraversalPruner::CriticalEndpointTraversalPruner(
    const at::Tensor& flat_inst_arcs_by_level,
    const at::Tensor& flat_inst_arcs_by_level_start,
    const at::Tensor& pin_pred_start,
    const at::Tensor& pin_pred_pin,
    const at::Tensor& pin_pred_arc_id,
    const at::Tensor& start_points,
    const at::Tensor& pin2node_map) {
  initialize_flat_arcs_and_levels(
      flat_inst_arcs_by_level,
      flat_inst_arcs_by_level_start,
      pin2node_map);

  reverse_offsets_ = tensor_to_i64_vector(pin_pred_start, "pin_pred_start");
  reverse_edges_ = tensor_to_i64_vector(pin_pred_pin, "pin_pred_pin");
  reverse_edge_arc_ids_ = tensor_to_i64_vector(pin_pred_arc_id, "pin_pred_arc_id");
  TORCH_CHECK(reverse_offsets_.size() >= 1,
              "pin_pred_start must contain at least one offset");
  TORCH_CHECK(reverse_offsets_.front() == 0,
              "pin_pred_start must start at zero");
  TORCH_CHECK(reverse_offsets_.back() == static_cast<int64_t>(reverse_edges_.size()),
              "pin_pred_start last offset must equal predecessor edge count");
  TORCH_CHECK(reverse_edge_arc_ids_.size() == reverse_edges_.size(),
              "pin_pred_arc_id length must match pin_pred_pin");
  for (size_t i = 1; i < reverse_offsets_.size(); ++i) {
    TORCH_CHECK(reverse_offsets_[i - 1] <= reverse_offsets_[i],
                "pin_pred_start must be non-decreasing");
  }
  for (int64_t arc_id : reverse_edge_arc_ids_) {
    TORCH_CHECK(arc_id >= -1 && arc_id < num_flat_arcs_,
                "pin_pred_arc_id entries must be -1 or valid flat arc ids");
  }
  num_pins_ = static_cast<int64_t>(reverse_offsets_.size()) - 1;
  initialize_start_points(start_points);
}

TraversalPruningRefreshResult CriticalEndpointTraversalPruner::refresh(
    const at::Tensor& active_endpoint_ids) const {
  TORCH_CHECK(active_endpoint_ids.defined(), "active_endpoint_ids must be defined");
  TORCH_CHECK(active_endpoint_ids.device().is_cpu(),
              "active_endpoint_ids must reside on CPU");
  TORCH_CHECK(active_endpoint_ids.dim() == 1,
              "active_endpoint_ids must be 1-D");

  const auto started_at = std::chrono::steady_clock::now();
  auto endpoints = tensor_to_i64_vector(active_endpoint_ids, "active_endpoint_ids");

  const int64_t endpoint_count = static_cast<int64_t>(endpoints.size());
  unsigned int hardware_threads = std::thread::hardware_concurrency();
  if (hardware_threads == 0) {
    hardware_threads = 2;
  }
  const int64_t max_task_count = std::max<int64_t>(
      1,
      static_cast<int64_t>(std::min<unsigned int>(hardware_threads, 8)));
  const int64_t task_count =
      endpoint_count == 0 ? 0 : std::min<int64_t>(endpoint_count, max_task_count);

  struct LocalTraversalMarks {
    std::vector<uint8_t> kept_arc;
  };

  std::vector<LocalTraversalMarks> local_marks(static_cast<size_t>(task_count));
  const int64_t parallel_task_count = std::max<int64_t>(1, task_count);
#pragma omp parallel for schedule(static) num_threads(parallel_task_count)
  for (int64_t task_id = 0; task_id < task_count; ++task_id) {
    const int64_t begin =
        (endpoint_count * task_id) / std::max<int64_t>(1, task_count);
    const int64_t end =
        (endpoint_count * (task_id + 1)) / std::max<int64_t>(1, task_count);
    std::vector<uint8_t> visited_pin(static_cast<size_t>(num_pins_), 0);
    auto& kept_arc_local = local_marks[static_cast<size_t>(task_id)].kept_arc;
    kept_arc_local.assign(static_cast<size_t>(num_flat_arcs_), 0);
    std::vector<int64_t> stack;
    stack.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)));

    for (int64_t endpoint_idx = begin; endpoint_idx < end; ++endpoint_idx) {
      const int64_t endpoint = endpoints[static_cast<size_t>(endpoint_idx)];
      if (endpoint >= 0 && endpoint < num_pins_ &&
          !visited_pin[static_cast<size_t>(endpoint)]) {
        visited_pin[static_cast<size_t>(endpoint)] = 1;
        stack.push_back(endpoint);
      }
    }

    while (!stack.empty()) {
      const int64_t cur_pin = stack.back();
      stack.pop_back();
      if (cur_pin < 0 || cur_pin >= num_pins_) {
        continue;
      }
      if (is_start_pin_[static_cast<size_t>(cur_pin)]) {
        continue;
      }

      const int64_t pred_begin = reverse_offsets_[static_cast<size_t>(cur_pin)];
      const int64_t pred_end = reverse_offsets_[static_cast<size_t>(cur_pin + 1)];
      for (int64_t edge_idx = pred_begin; edge_idx < pred_end; ++edge_idx) {
        const int64_t pred_pin = reverse_edges_[static_cast<size_t>(edge_idx)];
        if (pred_pin < 0 || pred_pin >= num_pins_) {
          continue;
        }

        const int64_t direct_arc_id =
            edge_idx < static_cast<int64_t>(reverse_edge_arc_ids_.size())
                ? reverse_edge_arc_ids_[static_cast<size_t>(edge_idx)]
                : -2;
        if (direct_arc_id >= 0) {
          kept_arc_local[static_cast<size_t>(direct_arc_id)] = 1;
        } else if (direct_arc_id == -2) {
          auto arc_it = pair_to_arc_indices_.find(encode_key64(pred_pin, cur_pin));
          if (arc_it != pair_to_arc_indices_.end()) {
            for (int64_t arc_idx : arc_it->second) {
              kept_arc_local[static_cast<size_t>(arc_idx)] = 1;
            }
          }
        }

        if (!visited_pin[static_cast<size_t>(pred_pin)]) {
          visited_pin[static_cast<size_t>(pred_pin)] = 1;
          stack.push_back(pred_pin);
        }
      }
    }
  }

  std::vector<uint8_t> kept_arc(static_cast<size_t>(num_flat_arcs_), 0);
  for (const auto& marks : local_marks) {
    for (int64_t arc_idx = 0; arc_idx < num_flat_arcs_; ++arc_idx) {
      if (marks.kept_arc[static_cast<size_t>(arc_idx)]) {
        kept_arc[static_cast<size_t>(arc_idx)] = 1;
      }
    }
  }

  std::vector<int64_t> kept_indices;
  kept_indices.reserve(static_cast<size_t>(num_flat_arcs_));
  std::vector<int64_t> kept_counts_by_level(static_cast<size_t>(num_levels_), 0);
  std::unordered_set<int64_t> active_instances;

  for (int64_t arc_idx = 0; arc_idx < num_flat_arcs_; ++arc_idx) {
    if (!kept_arc[static_cast<size_t>(arc_idx)]) {
      continue;
    }
    kept_indices.push_back(arc_idx);
    const int64_t level = flat_arc_level_id_[static_cast<size_t>(arc_idx)];
    if (level >= 0 && level < num_levels_) {
      kept_counts_by_level[static_cast<size_t>(level)] += 1;
    }
    const int64_t inst_id = flat_arc_to_inst_id_[static_cast<size_t>(arc_idx)];
    if (inst_id >= 0) {
      active_instances.insert(inst_id);
    }
  }

  std::vector<int64_t> kept_level_offsets(static_cast<size_t>(num_levels_) + 1, 0);
  for (int64_t level = 0; level < num_levels_; ++level) {
    kept_level_offsets[static_cast<size_t>(level + 1)] =
        kept_level_offsets[static_cast<size_t>(level)] +
        kept_counts_by_level[static_cast<size_t>(level)];
  }

  const auto finished_at = std::chrono::steady_clock::now();
  const double runtime_ms =
      std::chrono::duration<double, std::milli>(finished_at - started_at).count();

  TraversalPruningRefreshResult result;
  result.kept_flat_arc_indices = i64_tensor_from_vector(kept_indices);
  result.kept_level_offsets = i64_tensor_from_vector(kept_level_offsets);
  result.kept_counts_by_level = i64_tensor_from_vector(kept_counts_by_level);
  result.active_inst_count = static_cast<int64_t>(active_instances.size());
  result.active_arc_count = static_cast<int64_t>(kept_indices.size());
  result.dropped_arc_count = num_flat_arcs_ - result.active_arc_count;
  result.parallel_task_count = task_count;
  result.preparation_runtime_ms = runtime_ms;
  return result;
}

EndpointIncidenceResult CriticalEndpointTraversalPruner::endpoint_incidence(
    const at::Tensor& active_endpoint_ids) const {
  TORCH_CHECK(active_endpoint_ids.defined(), "active_endpoint_ids must be defined");
  TORCH_CHECK(active_endpoint_ids.device().is_cpu(),
              "active_endpoint_ids must reside on CPU");
  TORCH_CHECK(active_endpoint_ids.dim() == 1,
              "active_endpoint_ids must be 1-D");

  const auto started_at = std::chrono::steady_clock::now();
  auto endpoints = tensor_to_i64_vector(active_endpoint_ids, "active_endpoint_ids");
  std::sort(endpoints.begin(), endpoints.end());
  endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());

  std::vector<int64_t> pin_counts(static_cast<size_t>(num_pins_), 0);
  std::vector<int64_t> arc_counts(static_cast<size_t>(num_flat_arcs_), 0);
  std::vector<int64_t> stack;
  stack.reserve(static_cast<size_t>(num_pins_));

  for (const int64_t endpoint : endpoints) {
    if (endpoint < 0 || endpoint >= num_pins_) {
      continue;
    }
    std::vector<uint8_t> visited_pin(static_cast<size_t>(num_pins_), 0);
    std::vector<uint8_t> visited_arc(static_cast<size_t>(num_flat_arcs_), 0);
    visited_pin[static_cast<size_t>(endpoint)] = 1;
    pin_counts[static_cast<size_t>(endpoint)] += 1;
    stack.clear();
    stack.push_back(endpoint);

    while (!stack.empty()) {
      const int64_t cur_pin = stack.back();
      stack.pop_back();
      if (cur_pin < 0 || cur_pin >= num_pins_) {
        continue;
      }
      if (is_start_pin_[static_cast<size_t>(cur_pin)]) {
        continue;
      }

      const int64_t pred_begin = reverse_offsets_[static_cast<size_t>(cur_pin)];
      const int64_t pred_end = reverse_offsets_[static_cast<size_t>(cur_pin + 1)];
      for (int64_t edge_idx = pred_begin; edge_idx < pred_end; ++edge_idx) {
        const int64_t pred_pin = reverse_edges_[static_cast<size_t>(edge_idx)];
        if (pred_pin < 0 || pred_pin >= num_pins_) {
          continue;
        }

        const int64_t direct_arc_id =
            edge_idx < static_cast<int64_t>(reverse_edge_arc_ids_.size())
                ? reverse_edge_arc_ids_[static_cast<size_t>(edge_idx)]
                : -2;
        if (direct_arc_id >= 0 && !visited_arc[static_cast<size_t>(direct_arc_id)]) {
          visited_arc[static_cast<size_t>(direct_arc_id)] = 1;
          arc_counts[static_cast<size_t>(direct_arc_id)] += 1;
        } else if (direct_arc_id == -2) {
          auto arc_it = pair_to_arc_indices_.find(encode_key64(pred_pin, cur_pin));
          if (arc_it != pair_to_arc_indices_.end()) {
            for (int64_t arc_idx : arc_it->second) {
              if (arc_idx >= 0 && arc_idx < num_flat_arcs_ &&
                  !visited_arc[static_cast<size_t>(arc_idx)]) {
                visited_arc[static_cast<size_t>(arc_idx)] = 1;
                arc_counts[static_cast<size_t>(arc_idx)] += 1;
              }
            }
          }
        }

        if (!visited_pin[static_cast<size_t>(pred_pin)]) {
          visited_pin[static_cast<size_t>(pred_pin)] = 1;
          pin_counts[static_cast<size_t>(pred_pin)] += 1;
          stack.push_back(pred_pin);
        }
      }
    }
  }

  int64_t pin_nonzero = 0;
  for (const int64_t count : pin_counts) {
    if (count > 0) {
      pin_nonzero += 1;
    }
  }
  int64_t arc_nonzero = 0;
  for (const int64_t count : arc_counts) {
    if (count > 0) {
      arc_nonzero += 1;
    }
  }

  const auto finished_at = std::chrono::steady_clock::now();
  const double runtime_ms =
      std::chrono::duration<double, std::milli>(finished_at - started_at).count();

  EndpointIncidenceResult result;
  result.pin_endpoint_incidence_count = i64_tensor_from_vector(pin_counts);
  result.arc_endpoint_incidence_count = i64_tensor_from_vector(arc_counts);
  result.active_endpoint_count = static_cast<int64_t>(endpoints.size());
  result.pin_incidence_nonzero_count = pin_nonzero;
  result.arc_incidence_nonzero_count = arc_nonzero;
  result.incidence_runtime_ms = runtime_ms;
  return result;
}

std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
extract_critical_paths(
    const at::Tensor& endpoints,
    int64_t k,
    const at::Tensor& start_points,
    const at::Tensor& pin_rAAT,
    const at::Tensor& pin_fAAT,
    const at::Tensor& pin_rslack,
    const at::Tensor& pin_fslack,
    const at::Tensor& pin_slack,
    const at::Tensor& flat_pin_to_graph_reverse,
    const at::Tensor& flat_pin_to_graph_start_reverse,
    const at::Tensor& pin_pair_arc_keys,
    const at::Tensor& flat_pin_pair_arc_start,
    const at::Tensor& flat_pin_pair_arc_indices,
    int64_t max_depth,
    double slack_epsilon) {
  (void)k;  // currently we only generate one path per endpoint

  TORCH_CHECK(endpoints.device().is_cpu(), "endpoints must reside on CPU");
  TORCH_CHECK(start_points.device().is_cpu(), "start_points must reside on CPU");
  TORCH_CHECK(pin_rAAT.device().is_cpu(), "pin_rAAT must reside on CPU");
  TORCH_CHECK(pin_fAAT.device().is_cpu(), "pin_fAAT must reside on CPU");
  TORCH_CHECK(pin_rslack.device().is_cpu(), "pin_rslack must reside on CPU");
  TORCH_CHECK(pin_fslack.device().is_cpu(), "pin_fslack must reside on CPU");
  TORCH_CHECK(pin_slack.device().is_cpu(), "pin_slack must reside on CPU");
  TORCH_CHECK(flat_pin_to_graph_reverse.device().is_cpu(),
              "flat_pin_to_graph_reverse must reside on CPU");
  TORCH_CHECK(flat_pin_to_graph_start_reverse.device().is_cpu(),
              "flat_pin_to_graph_start_reverse must reside on CPU");

  auto rev_offsets = flat_pin_to_graph_start_reverse.contiguous();
  auto rev_edges = flat_pin_to_graph_reverse.contiguous();
  TORCH_CHECK(rev_offsets.dim() == 1, "flat_pin_to_graph_start_reverse must be 1-D");
  TORCH_CHECK(rev_edges.dim() == 1, "flat_pin_to_graph_reverse must be 1-D");
  TORCH_CHECK(rev_offsets.size(0) >= 1,
              "flat_pin_to_graph_start_reverse must contain at least one entry");

  const int64_t num_pins = rev_offsets.size(0) - 1;
  TORCH_CHECK(num_pins >= 0, "Derived number of pins must be non-negative");

  auto endpoints_cpu = endpoints.contiguous();
  auto start_points_cpu = start_points.contiguous();

  // Convert timing/scalar tensors to double precision for robust comparisons.
  auto pin_rslack_d = pin_rslack.contiguous().to(at::ScalarType::Double);
  auto pin_fslack_d = pin_fslack.contiguous().to(at::ScalarType::Double);
  auto pin_rAAT_d = pin_rAAT.contiguous().to(at::ScalarType::Double);
  auto pin_fAAT_d = pin_fAAT.contiguous().to(at::ScalarType::Double);

  const double* rslack_ptr = pin_rslack_d.data_ptr<double>();
  const double* fslack_ptr = pin_fslack_d.data_ptr<double>();
  const double* rAAT_ptr = pin_rAAT_d.data_ptr<double>();
  const double* fAAT_ptr = pin_fAAT_d.data_ptr<double>();
  const int32_t* rev_offsets_ptr = rev_offsets.data_ptr<int32_t>();
  const int32_t* rev_edges_ptr = rev_edges.data_ptr<int32_t>();

  (void)pin_slack;

  std::vector<uint8_t> is_start(num_pins, 0);
  if (start_points_cpu.numel() > 0) {
    const int32_t* start_ptr = start_points_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < start_points_cpu.numel(); ++i) {
      int32_t pin = start_ptr[i];
      if (pin >= 0 && pin < num_pins) {
        is_start[pin] = 1;
      }
    }
  }

  // Build arc lookup table if available.
  std::unordered_map<uint64_t, ArcLookupEntry> arc_lookup;
  const int32_t* arc_indices_ptr = nullptr;
  const bool has_arc_mapping = pin_pair_arc_keys.defined() &&
                               flat_pin_pair_arc_start.defined() &&
                               flat_pin_pair_arc_indices.defined() &&
                               pin_pair_arc_keys.numel() > 0;
  if (has_arc_mapping) {
    auto keys = pin_pair_arc_keys.contiguous();
    auto starts = flat_pin_pair_arc_start.contiguous();
    auto indices = flat_pin_pair_arc_indices.contiguous();

    TORCH_CHECK(keys.device().is_cpu(), "pin_pair_arc_keys must reside on CPU");
    TORCH_CHECK(starts.device().is_cpu(), "flat_pin_pair_arc_start must reside on CPU");
    TORCH_CHECK(indices.device().is_cpu(), "flat_pin_pair_arc_indices must reside on CPU");
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) == 2,
                "pin_pair_arc_keys must have shape (N, 2)");
    TORCH_CHECK(starts.dim() == 1, "flat_pin_pair_arc_start must be 1-D");
    TORCH_CHECK(indices.dim() == 1, "flat_pin_pair_arc_indices must be 1-D");
    TORCH_CHECK(starts.size(0) == keys.size(0) + 1,
                "flat_pin_pair_arc_start length mismatch with pin_pair_arc_keys");

    const int32_t* key_ptr = keys.data_ptr<int32_t>();
    const int32_t* start_ptr = starts.data_ptr<int32_t>();
    arc_indices_ptr = indices.data_ptr<int32_t>();

    const int64_t num_pairs = keys.size(0);
    arc_lookup.reserve(static_cast<size_t>(num_pairs));
    for (int64_t i = 0; i < num_pairs; ++i) {
      int32_t src = key_ptr[2 * i + 0];
      int32_t dst = key_ptr[2 * i + 1];
      int32_t start = start_ptr[i];
      int32_t end = start_ptr[i + 1];
      if (start < end) {
        arc_lookup.emplace(encode_key(src, dst), ArcLookupEntry{start, end - start});
      }
    }
  }

  if (max_depth <= 0) {
    max_depth = std::max<int64_t>(1, num_pins);
  }
  if (slack_epsilon <= 0.0) {
    slack_epsilon = 1e-3;
  }

  const int64_t num_endpoints = endpoints_cpu.numel();
  std::vector<std::vector<int64_t>> path_results(num_endpoints);
  std::vector<std::vector<int64_t>> arc_results(num_endpoints);

  const int32_t* endpoints_ptr = endpoints_cpu.data_ptr<int32_t>();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t eid = 0; eid < num_endpoints; ++eid) {
    int32_t sink = endpoints_ptr[eid];
    std::vector<int64_t> path;
    std::vector<int64_t> arcs;

    if (sink >= 0 && sink < num_pins) {
      bool use_rise = rslack_ptr[sink] <= fslack_ptr[sink];
      double current_slack = use_rise ? rslack_ptr[sink] : fslack_ptr[sink];
      int32_t current_pin = sink;

      path.push_back(static_cast<int64_t>(current_pin));
      std::unordered_set<int32_t> visited;
      visited.insert(current_pin);

      int64_t steps = 0;
      while (steps < max_depth) {
        if (current_pin < 0 || current_pin >= num_pins) {
          break;
        }
        if (!is_start.empty() && is_start[current_pin]) {
          break;
        }

        int32_t begin = rev_offsets_ptr[current_pin];
        int32_t end = rev_offsets_ptr[current_pin + 1];
        if (begin >= end) {
          break;
        }

        int32_t best_pred = -1;
        int64_t best_arc = -1;
        double best_slack = std::numeric_limits<double>::infinity();
        double best_gap = std::numeric_limits<double>::infinity();
        double best_arrival = -std::numeric_limits<double>::infinity();

        for (int32_t idx = begin; idx < end; ++idx) {
          int32_t pred = rev_edges_ptr[idx];
          if (pred < 0 || pred >= num_pins) {
            continue;
          }
          if (visited.count(pred)) {
            continue;
          }

          double pred_slack = use_rise ? rslack_ptr[pred] : fslack_ptr[pred];
          double slack_gap = std::abs(pred_slack - current_slack);
          double arrival = use_rise ? rAAT_ptr[pred] : fAAT_ptr[pred];

          bool improved = false;
          if (pred_slack < best_slack - slack_epsilon) {
            improved = true;
          } else if (std::abs(pred_slack - best_slack) <= slack_epsilon) {
            if (slack_gap < best_gap - slack_epsilon) {
              improved = true;
            } else if (std::abs(slack_gap - best_gap) <= slack_epsilon &&
                       arrival > best_arrival + slack_epsilon) {
              improved = true;
            }
          }

          if (improved) {
            best_pred = pred;
            best_slack = pred_slack;
            best_gap = slack_gap;
            best_arrival = arrival;
            best_arc = query_arc_index(arc_lookup, arc_indices_ptr, pred, current_pin);
          }
        }

        if (best_pred < 0) {
          break;
        }

        path.push_back(static_cast<int64_t>(best_pred));
        arcs.push_back(best_arc);
        visited.insert(best_pred);
        current_pin = best_pred;
        current_slack = best_slack;
        ++steps;
      }

      std::reverse(path.begin(), path.end());
      std::reverse(arcs.begin(), arcs.end());
    }

    path_results[eid] = std::move(path);
    arc_results[eid] = std::move(arcs);
  }

  return std::make_tuple(std::move(path_results), std::move(arc_results));
}

}  // namespace dreamplace
