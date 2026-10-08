#include "segment_count_packer.h"

#include <ATen/Parallel.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace dreamplace {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& started_at) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started_at).count();
}

void check_cpu_1d_contiguous(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

class IntegerView {
 public:
  IntegerView(const at::Tensor& tensor, const char* name) : tensor_(tensor), name_(name) {
    check_cpu_1d_contiguous(tensor_, name_);
    TORCH_CHECK(
        tensor_.scalar_type() == at::kInt || tensor_.scalar_type() == at::kLong,
        name_,
        " must be int32 or int64");
  }

  int64_t size() const { return tensor_.numel(); }

  int64_t operator[](int64_t index) const {
    TORCH_CHECK(index >= 0 && index < size(), name_, " index out of range");
    if (tensor_.scalar_type() == at::kInt) {
      return static_cast<int64_t>(tensor_.data_ptr<int32_t>()[index]);
    }
    return tensor_.data_ptr<int64_t>()[index];
  }

 private:
  const at::Tensor& tensor_;
  const char* name_;
};

class FloatView {
 public:
  FloatView(const at::Tensor& tensor, const char* name) : tensor_(tensor), name_(name) {
    check_cpu_1d_contiguous(tensor_, name_);
    TORCH_CHECK(
        tensor_.scalar_type() == at::kFloat || tensor_.scalar_type() == at::kDouble,
        name_,
        " must be float32 or float64");
  }

  int64_t size() const { return tensor_.numel(); }

  double operator[](int64_t index) const {
    TORCH_CHECK(index >= 0 && index < size(), name_, " index out of range");
    if (tensor_.scalar_type() == at::kFloat) {
      return static_cast<double>(tensor_.data_ptr<float>()[index]);
    }
    return tensor_.data_ptr<double>()[index];
  }

 private:
  const at::Tensor& tensor_;
  const char* name_;
};

enum class SkipReason : int64_t {
  kNone = 0,
  kSinglePinNet,
  kInvalidDriver,
  kInvalidTopologyRange,
  kMissingTopology,
  kNonRootDriver,
  kMalformedTree,
  kMissingCoordinate,
  kMissingPinCap,
};

const char* skip_reason_name(SkipReason reason) {
  switch (reason) {
    case SkipReason::kNone:
      return "none";
    case SkipReason::kSinglePinNet:
      return "single_pin_net";
    case SkipReason::kInvalidDriver:
      return "invalid_driver_pin";
    case SkipReason::kInvalidTopologyRange:
      return "invalid_topology_range";
    case SkipReason::kMissingTopology:
      return "missing_topology";
    case SkipReason::kNonRootDriver:
      return "driver_is_not_topology_root";
    case SkipReason::kMalformedTree:
      return "malformed_topology_tree";
    case SkipReason::kMissingCoordinate:
      return "missing_coordinate";
    case SkipReason::kMissingPinCap:
      return "missing_sink_pin_capacitance";
  }
  return "unknown";
}

struct NetPlan {
  SkipReason reason = SkipReason::kNone;
  int64_t net_id = -1;
  int64_t driver_pin_id = -1;
  std::vector<int64_t> topo_nodes;
  std::vector<int64_t> edge_start;
  std::vector<int64_t> edge_parent;
  std::vector<int64_t> edge_child;
  std::vector<int64_t> edge_parent_local;
  std::vector<int64_t> edge_child_local;
  std::vector<double> edge_resistance;
  std::vector<double> edge_capacitance;
  std::vector<int64_t> edge_segment_local;
  std::vector<double> node_capacitance;
  std::vector<int64_t> sink_node_id;
  std::vector<int64_t> sink_local_id;
  // Candidate mode follows tensor_builder's child-first compact allocation.
  std::vector<int64_t> compact_to_original;
  std::unordered_map<int64_t, int64_t> compact_local_by_original;
  std::vector<int64_t> topo_compact;
  std::vector<int64_t> compact_parent;
  std::vector<int64_t> compact_flat_pin_to_start;
  std::vector<int64_t> compact_flat_pin_to;
  std::vector<double> compact_node_capacitance;
  std::vector<double> compact_edge_resistance;
  std::vector<double> compact_edge_capacitance;
  std::vector<int64_t> segment_parent;
  std::vector<int64_t> segment_child;
  std::vector<int64_t> segment_parent_x_dbu;
  std::vector<int64_t> segment_parent_y_dbu;
  std::vector<int64_t> segment_child_x_dbu;
  std::vector<int64_t> segment_child_y_dbu;
  std::vector<double> segment_resistance;
  std::vector<double> segment_capacitance;

  bool valid() const { return reason == SkipReason::kNone; }
};

struct NetCounts {
  SkipReason reason = SkipReason::kNone;
  int64_t node_count = 0;
  int64_t edge_count = 0;
  int64_t sink_count = 0;
  int64_t segment_count = 0;
};

struct Inputs {
  IntegerView flat_net2pin;
  IntegerView flat_net2pin_start;
  IntegerView net2driver;
  IntegerView pin2node;
  IntegerView topo;
  IntegerView topo_start;
  IntegerView pin_fa;
  FloatView node_x;
  FloatView node_y;
  FloatView node_x_dbu;
  FloatView node_y_dbu;
  FloatView pin_cap;
  int64_t num_movable_nodes;
  int64_t num_terminals;
  double denominator;
  double r_unit;
  double c_unit;
};

bool finite_coordinate(const Inputs& in, int64_t node_id) {
  return node_id >= 0 && node_id < in.node_x.size() && node_id < in.node_y.size() &&
      node_id < in.node_x_dbu.size() && node_id < in.node_y_dbu.size() &&
      std::isfinite(in.node_x[node_id]) && std::isfinite(in.node_y[node_id]) &&
      std::isfinite(in.node_x_dbu[node_id]) && std::isfinite(in.node_y_dbu[node_id]);
}

NetPlan build_net_plan(const Inputs& in, int64_t net_id) {
  NetPlan plan;
  plan.net_id = net_id;
  if (net_id < 0 || net_id + 1 >= in.flat_net2pin_start.size()) {
    plan.reason = SkipReason::kInvalidDriver;
    return plan;
  }
  const int64_t pin_begin = in.flat_net2pin_start[net_id];
  const int64_t pin_end = in.flat_net2pin_start[net_id + 1];
  if (pin_begin < 0 || pin_end < pin_begin || pin_end > in.flat_net2pin.size()) {
    plan.reason = SkipReason::kInvalidDriver;
    return plan;
  }
  if (pin_end - pin_begin <= 1) {
    plan.reason = SkipReason::kSinglePinNet;
    return plan;
  }
  if (net_id >= in.net2driver.size()) {
    plan.reason = SkipReason::kInvalidDriver;
    return plan;
  }
  plan.driver_pin_id = in.net2driver[net_id];
  std::vector<int64_t> net_pins;
  net_pins.reserve(static_cast<size_t>(pin_end - pin_begin));
  bool driver_found = false;
  for (int64_t offset = pin_begin; offset < pin_end; ++offset) {
    const int64_t pin_id = in.flat_net2pin[offset];
    net_pins.push_back(pin_id);
    driver_found = driver_found || pin_id == plan.driver_pin_id;
  }
  if (plan.driver_pin_id < 0 || !driver_found) {
    plan.reason = SkipReason::kInvalidDriver;
    return plan;
  }
  if (net_id + 1 >= in.topo_start.size()) {
    plan.reason = SkipReason::kMissingTopology;
    return plan;
  }
  const int64_t topo_begin = in.topo_start[net_id];
  const int64_t topo_end = in.topo_start[net_id + 1];
  if (topo_begin < 0 || topo_end < topo_begin || topo_end > in.topo.size()) {
    plan.reason = SkipReason::kInvalidTopologyRange;
    return plan;
  }
  if (topo_begin == topo_end || plan.driver_pin_id >= in.pin_fa.size()) {
    plan.reason = SkipReason::kMissingTopology;
    return plan;
  }
  if (in.pin_fa[plan.driver_pin_id] >= 0) {
    plan.reason = SkipReason::kNonRootDriver;
    return plan;
  }

  std::unordered_set<int64_t> node_set(net_pins.begin(), net_pins.end());
  std::unordered_map<int64_t, int64_t> parent_by_node;
  std::unordered_map<int64_t, std::vector<int64_t>> children;
  for (int64_t offset = topo_begin; offset < topo_end; ++offset) {
    const int64_t node_id = in.topo[offset];
    node_set.insert(node_id);
    if (node_id < 0 || node_id >= in.pin_fa.size()) {
      plan.reason = SkipReason::kMalformedTree;
      return plan;
    }
    const int64_t parent = in.pin_fa[node_id];
    if (parent >= 0) {
      node_set.insert(parent);
      if (node_id == plan.driver_pin_id || parent_by_node.count(node_id) != 0) {
        plan.reason = SkipReason::kMalformedTree;
        return plan;
      }
      parent_by_node.emplace(node_id, parent);
      children[parent].push_back(node_id);
    }
  }
  if (node_set.count(plan.driver_pin_id) == 0 ||
      parent_by_node.size() != node_set.size() - 1) {
    plan.reason = SkipReason::kMalformedTree;
    return plan;
  }
  for (const int64_t node_id : node_set) {
    children.try_emplace(node_id, std::vector<int64_t>{});
  }
  for (auto& item : children) {
    std::sort(item.second.begin(), item.second.end());
  }

  std::vector<int64_t> stack{plan.driver_pin_id};
  std::unordered_set<int64_t> visited;
  while (!stack.empty()) {
    const int64_t node_id = stack.back();
    stack.pop_back();
    if (!visited.insert(node_id).second) {
      plan.reason = SkipReason::kMalformedTree;
      return plan;
    }
    plan.topo_nodes.push_back(node_id);
    const auto& node_children = children[node_id];
    for (auto it = node_children.rbegin(); it != node_children.rend(); ++it) {
      stack.push_back(*it);
    }
  }
  if (visited.size() != node_set.size()) {
    plan.reason = SkipReason::kMalformedTree;
    return plan;
  }

  std::unordered_map<int64_t, int64_t> local_by_node;
  local_by_node.reserve(plan.topo_nodes.size());
  for (int64_t local_id = 0; local_id < static_cast<int64_t>(plan.topo_nodes.size()); ++local_id) {
    const int64_t node_id = plan.topo_nodes[static_cast<size_t>(local_id)];
    if (!finite_coordinate(in, node_id)) {
      plan.reason = SkipReason::kMissingCoordinate;
      return plan;
    }
    local_by_node.emplace(node_id, local_id);
  }

  plan.node_capacitance.reserve(plan.topo_nodes.size());
  std::unordered_set<int64_t> sink_set;
  for (const int64_t pin_id : net_pins) {
    if (pin_id != plan.driver_pin_id) {
      sink_set.insert(pin_id);
    }
  }
  for (const int64_t node_id : plan.topo_nodes) {
    double cap = 0.0;
    if (node_id >= 0 && node_id < in.pin_cap.size()) {
      const double candidate = in.pin_cap[node_id];
      if (std::isfinite(candidate) && candidate >= 0.0) {
        cap = candidate;
      }
    }
    if (sink_set.count(node_id) != 0) {
      if (node_id < 0 || node_id >= in.pin2node.size()) {
        plan.reason = SkipReason::kMissingPinCap;
        return plan;
      }
      const int64_t instance_id = in.pin2node[node_id];
      if (instance_id >= in.num_movable_nodes + in.num_terminals) {
        cap = 0.0;
      } else if (node_id < 0 || node_id >= in.pin_cap.size() ||
                 !std::isfinite(in.pin_cap[node_id]) || in.pin_cap[node_id] < 0.0) {
        plan.reason = SkipReason::kMissingPinCap;
        return plan;
      } else {
        cap = in.pin_cap[node_id];
      }
    }
    plan.node_capacitance.push_back(cap);
  }

  for (const int64_t parent : plan.topo_nodes) {
    plan.edge_start.push_back(static_cast<int64_t>(plan.edge_parent.size()));
    const auto& node_children = children[parent];
    for (const int64_t child : node_children) {
      const int64_t edge_local = static_cast<int64_t>(plan.edge_parent.size());
      plan.edge_parent.push_back(parent);
      plan.edge_child.push_back(child);
      plan.edge_parent_local.push_back(local_by_node.at(parent));
      plan.edge_child_local.push_back(local_by_node.at(child));
      const double length_um =
          (std::abs(in.node_x[parent] - in.node_x[child]) +
           std::abs(in.node_y[parent] - in.node_y[child])) /
          in.denominator;
      const double resistance = length_um * in.r_unit;
      const double capacitance = length_um * in.c_unit;
      plan.edge_resistance.push_back(resistance);
      plan.edge_capacitance.push_back(capacitance);

      const double parent_x = std::nearbyint(in.node_x[parent]);
      const double parent_y = std::nearbyint(in.node_y[parent]);
      const double child_x = std::nearbyint(in.node_x[child]);
      const double child_y = std::nearbyint(in.node_y[child]);
      const double dx = child_x - parent_x;
      const double dy = child_y - parent_y;
      if (dx * dx + dy * dy > 0.0) {
        const int64_t segment_local = static_cast<int64_t>(plan.segment_parent.size());
        plan.edge_segment_local.push_back(segment_local);
        plan.segment_parent.push_back(parent);
        plan.segment_child.push_back(child);
        plan.segment_parent_x_dbu.push_back(
            static_cast<int64_t>(std::nearbyint(in.node_x_dbu[parent])));
        plan.segment_parent_y_dbu.push_back(
            static_cast<int64_t>(std::nearbyint(in.node_y_dbu[parent])));
        plan.segment_child_x_dbu.push_back(
            static_cast<int64_t>(std::nearbyint(in.node_x_dbu[child])));
        plan.segment_child_y_dbu.push_back(
            static_cast<int64_t>(std::nearbyint(in.node_y_dbu[child])));
        plan.segment_resistance.push_back(resistance);
        plan.segment_capacitance.push_back(capacitance);
      } else {
        plan.edge_segment_local.push_back(-1);
      }
      TORCH_INTERNAL_ASSERT(edge_local + 1 == static_cast<int64_t>(plan.edge_segment_local.size()));
    }
  }
  plan.edge_start.push_back(static_cast<int64_t>(plan.edge_parent.size()));

  for (const int64_t pin_id : net_pins) {
    if (pin_id == plan.driver_pin_id) {
      continue;
    }
    const auto local_it = local_by_node.find(pin_id);
    if (local_it == local_by_node.end()) {
      plan.reason = SkipReason::kMalformedTree;
      return plan;
    }
    plan.sink_node_id.push_back(pin_id);
    plan.sink_local_id.push_back(local_it->second);
  }
  return plan;
}

template <typename scalar_t>
void fill_fraction_tables(
    const int64_t* parent_x,
    const int64_t* parent_y,
    const int64_t* child_x,
    const int64_t* child_y,
    int64_t segment_count,
    int64_t max_count,
    scalar_t* parent_fraction,
    scalar_t* child_fraction,
    scalar_t* sub_fraction,
    int64_t thread_count) {
#pragma omp parallel for schedule(static) num_threads(thread_count) if (thread_count > 1)
  for (int64_t segment_id = 0; segment_id < segment_count; ++segment_id) {
    const double px = static_cast<double>(parent_x[segment_id]);
    const double py = static_cast<double>(parent_y[segment_id]);
    const double dx = static_cast<double>(child_x[segment_id] - parent_x[segment_id]);
    const double dy = static_cast<double>(child_y[segment_id] - parent_y[segment_id]);
    const double length_sq = dx * dx + dy * dy;
    const int64_t width = max_count + 1;
    parent_fraction[segment_id * width] = static_cast<scalar_t>(1.0);
    child_fraction[segment_id * width] = static_cast<scalar_t>(1.0);
    sub_fraction[(segment_id * width) * width] = static_cast<scalar_t>(1.0);
    for (int64_t count = 1; count <= max_count; ++count) {
      std::vector<double> cuts;
      cuts.reserve(static_cast<size_t>(count));
      for (int64_t index = 0; index < count; ++index) {
        const double ideal = static_cast<double>(index + 1) / static_cast<double>(count + 1);
        const double qx = std::nearbyint(px + dx * ideal);
        const double qy = std::nearbyint(py + dy * ideal);
        double ratio = length_sq > 0.0 ? ((qx - px) * dx + (qy - py) * dy) / length_sq : ideal;
        cuts.push_back(std::min(1.0, std::max(0.0, ratio)));
      }
      std::sort(cuts.begin(), cuts.end());
      std::vector<double> fractions(static_cast<size_t>(count + 1), 0.0);
      double previous = 0.0;
      for (int64_t index = 0; index < count; ++index) {
        fractions[static_cast<size_t>(index)] = std::max(0.0, cuts[static_cast<size_t>(index)] - previous);
        previous = cuts[static_cast<size_t>(index)];
      }
      fractions[static_cast<size_t>(count)] = std::max(0.0, 1.0 - previous);
      parent_fraction[segment_id * width + count] = static_cast<scalar_t>(fractions.front());
      child_fraction[segment_id * width + count] = static_cast<scalar_t>(fractions.back());
      for (int64_t index = 0; index <= count; ++index) {
        sub_fraction[(segment_id * width + count) * width + index] =
            static_cast<scalar_t>(fractions[static_cast<size_t>(index)]);
      }
    }
  }
}

struct CandidateSpec {
  int64_t index = -1;
  int64_t net_id = -1;
  int64_t parent_node_id = -1;
  int64_t child_node_id = -1;
  int64_t tree_node_id = -1;
  int64_t synthetic_node_id = -1;
  bool is_segment = false;
  double split_ratio = 0.0;
};

struct ExpandedCandidatePlan {
  NetPlan plan;
  std::vector<int64_t> candidate_index;
  std::vector<int64_t> candidate_local_node;
};

ExpandedCandidatePlan expand_candidate_plan(
    const NetPlan& original,
    const std::vector<CandidateSpec>& candidate_specs) {
  ExpandedCandidatePlan result;
  result.plan.net_id = original.net_id;
  result.plan.driver_pin_id = original.driver_pin_id;
  result.candidate_index.reserve(candidate_specs.size());
  result.candidate_local_node.reserve(candidate_specs.size());

  std::unordered_map<int64_t, std::vector<int64_t>> children;
  std::unordered_map<int64_t, std::pair<double, double>> edge_rc;
  std::unordered_map<int64_t, double> node_cap;
  std::unordered_set<int64_t> node_set;
  for (int64_t index = 0; index < static_cast<int64_t>(original.topo_nodes.size()); ++index) {
    const int64_t node = original.topo_nodes[static_cast<size_t>(index)];
    node_set.insert(node);
    node_cap[node] = original.node_capacitance[static_cast<size_t>(index)];
    children[node] = {};
  }
  for (int64_t edge = 0; edge < static_cast<int64_t>(original.edge_parent.size()); ++edge) {
    const int64_t parent = original.edge_parent[static_cast<size_t>(edge)];
    const int64_t child = original.edge_child[static_cast<size_t>(edge)];
    children[parent].push_back(child);
    edge_rc[child] = {
        original.edge_resistance[static_cast<size_t>(edge)],
        original.edge_capacitance[static_cast<size_t>(edge)],
    };
  }

  std::unordered_map<int64_t, std::vector<const CandidateSpec*>> specs_by_edge;
  for (const CandidateSpec& spec : candidate_specs) {
    if (!spec.is_segment) {
      TORCH_CHECK(spec.tree_node_id >= 0, "candidate tree node id must be nonnegative");
      continue;
    }
    TORCH_CHECK(
        spec.parent_node_id >= 0 && spec.child_node_id >= 0,
        "segment candidate parent/child node id must be nonnegative");
    TORCH_CHECK(
        std::isfinite(spec.split_ratio) && spec.split_ratio > 0.0 && spec.split_ratio < 1.0,
        "segment candidate split ratio must lie strictly inside (0, 1)");
    specs_by_edge[spec.parent_node_id].push_back(&spec);
  }

  std::unordered_set<int64_t> synthetic_ids;
  for (auto& entry : specs_by_edge) {
    const int64_t parent = entry.first;
    auto& edge_specs = entry.second;
    std::sort(edge_specs.begin(), edge_specs.end(), [](const CandidateSpec* lhs, const CandidateSpec* rhs) {
      if (lhs->child_node_id != rhs->child_node_id) {
        return lhs->child_node_id < rhs->child_node_id;
      }
      if (lhs->split_ratio != rhs->split_ratio) {
        return lhs->split_ratio < rhs->split_ratio;
      }
      return lhs->index < rhs->index;
    });

    std::unordered_map<int64_t, std::vector<const CandidateSpec*>> by_child;
    for (const CandidateSpec* spec : edge_specs) {
      by_child[spec->child_node_id].push_back(spec);
    }
    for (auto& child_entry : by_child) {
      const int64_t child = child_entry.first;
      auto& child_specs = child_entry.second;
      auto children_it = children.find(parent);
      TORCH_CHECK(children_it != children.end(), "candidate parent node is absent from topology");
      auto child_it = std::find(children_it->second.begin(), children_it->second.end(), child);
      TORCH_CHECK(
          child_it != children_it->second.end(),
          "candidate segment edge is absent from topology: ",
          parent,
          " -> ",
          child);
      std::sort(child_specs.begin(), child_specs.end(), [](const CandidateSpec* lhs, const CandidateSpec* rhs) {
        if (lhs->split_ratio != rhs->split_ratio) {
          return lhs->split_ratio < rhs->split_ratio;
        }
        return lhs->index < rhs->index;
      });

      std::vector<int64_t> split_nodes;
      split_nodes.reserve(child_specs.size());
      for (const CandidateSpec* spec : child_specs) {
        int64_t synthetic = spec->synthetic_node_id;
        if (synthetic >= 0 || !synthetic_ids.insert(synthetic).second) {
          synthetic = -1 - spec->index;
          while (node_set.count(synthetic) != 0 || !synthetic_ids.insert(synthetic).second) {
            --synthetic;
          }
        }
        split_nodes.push_back(synthetic);
        node_set.insert(synthetic);
        node_cap[synthetic] = 0.0;
      }
      *child_it = split_nodes.front();
      for (size_t index = 0; index < split_nodes.size(); ++index) {
        const int64_t to = index + 1 < split_nodes.size() ? split_nodes[index + 1] : child;
        children[split_nodes[index]] = {to};
      }

      const auto original_rc_it = edge_rc.find(child);
      TORCH_CHECK(original_rc_it != edge_rc.end(), "candidate segment edge has no RC payload");
      const double original_r = original_rc_it->second.first;
      const double original_c = original_rc_it->second.second;
      edge_rc.erase(original_rc_it);
      double previous_ratio = 0.0;
      for (size_t index = 0; index < split_nodes.size(); ++index) {
        const double ratio = child_specs[index]->split_ratio;
        const double fraction = ratio - previous_ratio;
        const int64_t to = split_nodes[index];
        edge_rc[to] = {original_r * fraction, original_c * fraction};
        previous_ratio = ratio;
      }
      edge_rc[child] = {original_r * (1.0 - previous_ratio), original_c * (1.0 - previous_ratio)};
    }
  }

  std::vector<int64_t> stack{original.driver_pin_id};
  std::unordered_set<int64_t> visited;
  while (!stack.empty()) {
    const int64_t node = stack.back();
    stack.pop_back();
    TORCH_CHECK(visited.insert(node).second, "candidate-expanded topology contains a cycle");
    result.plan.topo_nodes.push_back(node);
    const auto children_it = children.find(node);
    TORCH_CHECK(children_it != children.end(), "candidate-expanded topology has a missing node");
    for (auto it = children_it->second.rbegin(); it != children_it->second.rend(); ++it) {
      stack.push_back(*it);
    }
  }
  TORCH_CHECK(
      visited.size() == node_set.size(),
      "candidate-expanded topology is disconnected from its driver root");

  std::unordered_map<int64_t, int64_t> local_by_node;
  local_by_node.reserve(result.plan.topo_nodes.size());
  for (const int64_t node : result.plan.topo_nodes) {
    if (local_by_node.count(node) == 0) {
      const int64_t local = static_cast<int64_t>(local_by_node.size());
      local_by_node[node] = local;
      result.plan.compact_to_original.push_back(node);
    }
    for (const int64_t child : children[node]) {
      if (local_by_node.count(child) == 0) {
        const int64_t local = static_cast<int64_t>(local_by_node.size());
        local_by_node[child] = local;
        result.plan.compact_to_original.push_back(child);
      }
    }
  }
  result.plan.compact_local_by_original = local_by_node;
  result.plan.topo_compact.reserve(result.plan.topo_nodes.size());
  for (const int64_t node : result.plan.topo_nodes) {
    result.plan.topo_compact.push_back(local_by_node.at(node));
  }
  result.plan.compact_node_capacitance.reserve(result.plan.compact_to_original.size());
  result.plan.compact_parent.assign(result.plan.compact_to_original.size(), -1);
  result.plan.compact_flat_pin_to_start.push_back(0);
  result.plan.compact_edge_resistance.assign(result.plan.compact_to_original.size(), 0.0);
  result.plan.compact_edge_capacitance.assign(result.plan.compact_to_original.size(), 0.0);
  for (int64_t local = 0;
       local < static_cast<int64_t>(result.plan.compact_to_original.size());
       ++local) {
    const int64_t node = result.plan.compact_to_original[static_cast<size_t>(local)];
    result.plan.compact_node_capacitance.push_back(node_cap[node]);
    for (const int64_t child : children[node]) {
      const int64_t child_local = local_by_node.at(child);
      result.plan.compact_parent[static_cast<size_t>(child_local)] = local;
      result.plan.compact_flat_pin_to.push_back(child_local);
      const auto rc_it = edge_rc.find(child);
      TORCH_CHECK(rc_it != edge_rc.end(), "candidate-expanded topology has a missing edge RC payload");
      result.plan.compact_edge_resistance[static_cast<size_t>(child_local)] =
          rc_it->second.first;
      result.plan.compact_edge_capacitance[static_cast<size_t>(child_local)] =
          rc_it->second.second;
    }
    result.plan.compact_flat_pin_to_start.push_back(
        static_cast<int64_t>(result.plan.compact_flat_pin_to.size()));
  }
  for (const int64_t parent : result.plan.topo_nodes) {
    result.plan.edge_start.push_back(static_cast<int64_t>(result.plan.edge_parent.size()));
    for (const int64_t child : children[parent]) {
      const auto rc_it = edge_rc.find(child);
      TORCH_CHECK(rc_it != edge_rc.end(), "candidate-expanded topology has a missing edge RC payload");
      result.plan.edge_parent.push_back(parent);
      result.plan.edge_child.push_back(child);
      result.plan.edge_parent_local.push_back(local_by_node.at(parent));
      result.plan.edge_child_local.push_back(local_by_node.at(child));
      result.plan.edge_resistance.push_back(rc_it->second.first);
      result.plan.edge_capacitance.push_back(rc_it->second.second);
    }
  }
  result.plan.edge_start.push_back(static_cast<int64_t>(result.plan.edge_parent.size()));

  for (const int64_t sink : original.sink_node_id) {
    const auto local_it = local_by_node.find(sink);
    TORCH_CHECK(local_it != local_by_node.end(), "candidate-expanded topology lost a sink node");
    result.plan.sink_node_id.push_back(sink);
    result.plan.sink_local_id.push_back(local_it->second);
  }
  for (const CandidateSpec& spec : candidate_specs) {
    const int64_t node = spec.is_segment ? [&]() {
      int64_t synthetic = spec.synthetic_node_id;
      if (synthetic >= 0 || node_set.count(synthetic) == 0) {
        synthetic = -1 - spec.index;
        while (node_set.count(synthetic) == 0) {
          --synthetic;
        }
      }
      return synthetic;
    }() : spec.tree_node_id;
    const auto local_it = local_by_node.find(node);
    TORCH_CHECK(
        local_it != local_by_node.end(),
        "candidate node is absent from candidate-expanded topology: ",
        node);
    result.candidate_index.push_back(spec.index);
    result.candidate_local_node.push_back(local_it->second);
  }
  return result;
}

}  // namespace

py::dict pack_segment_count_topology(
    const at::Tensor& flat_net2pin,
    const at::Tensor& flat_net2pin_start,
    const at::Tensor& net2driver,
    const at::Tensor& pin2node,
    const at::Tensor& net_flat_topo_sort,
    const at::Tensor& net_flat_topo_sort_start,
    const at::Tensor& pin_fa,
    const at::Tensor& node_x,
    const at::Tensor& node_y,
    const at::Tensor& node_x_dbu,
    const at::Tensor& node_y_dbu,
    const at::Tensor& pin_capacitance,
    int64_t num_movable_nodes,
    int64_t num_terminals,
    double dbu,
    double scale_factor,
    double r_unit,
    double c_unit,
    int64_t max_repeater_count,
    int64_t num_threads) {
  TORCH_CHECK(max_repeater_count >= 1, "max_repeater_count must be positive");
  TORCH_CHECK(dbu > 0.0, "dbu must be positive");
  TORCH_CHECK(scale_factor > 0.0, "scale_factor must be positive");
  TORCH_CHECK(
      pin_capacitance.scalar_type() == at::kFloat || pin_capacitance.scalar_type() == at::kDouble,
      "pin_capacitance must be float32 or float64");
  const int64_t available_threads = std::max<int64_t>(1, at::get_num_threads());
  const int64_t thread_count = std::max<int64_t>(1, std::min(num_threads, available_threads));

  const Inputs in{
      IntegerView(flat_net2pin, "flat_net2pin"),
      IntegerView(flat_net2pin_start, "flat_net2pin_start"),
      IntegerView(net2driver, "net2driver"),
      IntegerView(pin2node, "pin2node"),
      IntegerView(net_flat_topo_sort, "net_flat_topo_sort"),
      IntegerView(net_flat_topo_sort_start, "net_flat_topo_sort_start"),
      IntegerView(pin_fa, "pin_fa"),
      FloatView(node_x, "node_x"),
      FloatView(node_y, "node_y"),
      FloatView(node_x_dbu, "node_x_dbu"),
      FloatView(node_y_dbu, "node_y_dbu"),
      FloatView(pin_capacitance, "pin_capacitance"),
      num_movable_nodes,
      num_terminals,
      dbu * scale_factor,
      r_unit,
      c_unit,
  };
  TORCH_CHECK(
      in.flat_net2pin_start.size() == in.topo_start.size(),
      "net pin and topology start arrays must describe the same net count");
  const int64_t input_net_count = std::max<int64_t>(0, in.topo_start.size() - 1);

  const auto validate_started_at = Clock::now();
  std::vector<NetCounts> counts(static_cast<size_t>(input_net_count));
#pragma omp parallel for schedule(static) num_threads(thread_count) if (thread_count > 1)
  for (int64_t net_id = 0; net_id < input_net_count; ++net_id) {
    const NetPlan plan = build_net_plan(in, net_id);
    counts[static_cast<size_t>(net_id)] = NetCounts{
        plan.reason,
        static_cast<int64_t>(plan.topo_nodes.size()),
        static_cast<int64_t>(plan.edge_parent.size()),
        static_cast<int64_t>(plan.sink_node_id.size()),
        static_cast<int64_t>(plan.segment_parent.size()),
    };
  }
  const double validate_count_ms = elapsed_ms(validate_started_at);

  const auto prefix_started_at = Clock::now();
  std::vector<int64_t> valid_net_index(static_cast<size_t>(input_net_count), -1);
  std::vector<int64_t> affected_net_index(static_cast<size_t>(input_net_count), -1);
  std::vector<int64_t> node_offset(static_cast<size_t>(input_net_count), 0);
  std::vector<int64_t> edge_offset(static_cast<size_t>(input_net_count), 0);
  std::vector<int64_t> sink_offset(static_cast<size_t>(input_net_count), 0);
  std::vector<int64_t> segment_offset(static_cast<size_t>(input_net_count), 0);
  int64_t valid_net_count = 0;
  int64_t affected_net_count = 0;
  int64_t total_nodes = 0;
  int64_t total_edges = 0;
  int64_t total_sinks = 0;
  int64_t total_segments = 0;
  for (int64_t net_id = 0; net_id < input_net_count; ++net_id) {
    const NetCounts& item = counts[static_cast<size_t>(net_id)];
    if (item.reason != SkipReason::kNone) {
      continue;
    }
    valid_net_index[static_cast<size_t>(net_id)] = valid_net_count++;
    node_offset[static_cast<size_t>(net_id)] = total_nodes;
    edge_offset[static_cast<size_t>(net_id)] = total_edges;
    sink_offset[static_cast<size_t>(net_id)] = total_sinks;
    segment_offset[static_cast<size_t>(net_id)] = total_segments;
    total_nodes += item.node_count;
    total_edges += item.edge_count;
    total_sinks += item.sink_count;
    if (item.segment_count > 0) {
      affected_net_index[static_cast<size_t>(net_id)] = affected_net_count++;
    }
    total_segments += item.segment_count;
  }
  const double prefix_sum_ms = elapsed_ms(prefix_started_at);

  const auto allocate_started_at = Clock::now();
  const auto long_options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  const auto value_options = at::TensorOptions().dtype(pin_capacitance.scalar_type()).device(at::kCPU);
  at::Tensor net_ids = at::empty({valid_net_count}, long_options);
  at::Tensor net_topo_start = at::empty({valid_net_count + 1}, long_options);
  at::Tensor net_edge_start = at::empty({valid_net_count + 1}, long_options);
  at::Tensor net_sink_start = at::empty({valid_net_count + 1}, long_options);
  at::Tensor flat_topo_node_id = at::empty({total_nodes}, long_options);
  at::Tensor edge_start = at::empty({total_nodes + 1}, long_options);
  at::Tensor edge_parent_node_id = at::empty({total_edges}, long_options);
  at::Tensor edge_child_node_id = at::empty({total_edges}, long_options);
  at::Tensor edge_parent_compact_id = at::empty({total_edges}, long_options);
  at::Tensor edge_child_compact_id = at::empty({total_edges}, long_options);
  at::Tensor edge_net_index = at::empty({total_edges}, long_options);
  at::Tensor edge_resistance = at::empty({total_edges}, value_options);
  at::Tensor edge_capacitance = at::empty({total_edges}, value_options);
  at::Tensor node_capacitance = at::empty({total_nodes}, value_options);
  at::Tensor edge_to_segment_id = at::empty({total_edges}, long_options);
  at::Tensor sink_node_id = at::empty({total_sinks}, long_options);
  at::Tensor sink_net_id = at::empty({total_sinks}, long_options);
  at::Tensor sink_node_compact_id = at::empty({total_sinks}, long_options);
  at::Tensor driver_pin_id = at::empty({valid_net_count}, long_options);

  at::Tensor segment_ids = at::arange(total_segments, long_options);
  at::Tensor segment_net_id = at::empty({total_segments}, long_options);
  at::Tensor segment_net_index = at::empty({total_segments}, long_options);
  at::Tensor affected_net_ids = at::empty({affected_net_count}, long_options);
  at::Tensor net_segment_start = at::empty({affected_net_count + 1}, long_options);
  at::Tensor segment_parent_node_id = at::empty({total_segments}, long_options);
  at::Tensor segment_child_node_id = at::empty({total_segments}, long_options);
  at::Tensor segment_parent_x_dbu = at::empty({total_segments}, long_options);
  at::Tensor segment_parent_y_dbu = at::empty({total_segments}, long_options);
  at::Tensor segment_child_x_dbu = at::empty({total_segments}, long_options);
  at::Tensor segment_child_y_dbu = at::empty({total_segments}, long_options);
  at::Tensor segment_edge_resistance = at::empty({total_segments}, value_options);
  at::Tensor segment_edge_capacitance = at::empty({total_segments}, value_options);
  const int64_t fraction_width = max_repeater_count + 1;
  at::Tensor parent_cap_fraction = at::zeros({total_segments, fraction_width}, value_options);
  at::Tensor child_cap_fraction = at::zeros({total_segments, fraction_width}, value_options);
  at::Tensor segment_sub_resistance_fraction =
      at::zeros({total_segments, fraction_width, fraction_width}, value_options);
  const double allocate_ms = elapsed_ms(allocate_started_at);

  auto* net_ids_ptr = net_ids.data_ptr<int64_t>();
  auto* net_topo_start_ptr = net_topo_start.data_ptr<int64_t>();
  auto* net_edge_start_ptr = net_edge_start.data_ptr<int64_t>();
  auto* net_sink_start_ptr = net_sink_start.data_ptr<int64_t>();
  auto* flat_topo_ptr = flat_topo_node_id.data_ptr<int64_t>();
  auto* edge_start_ptr = edge_start.data_ptr<int64_t>();
  auto* edge_parent_ptr = edge_parent_node_id.data_ptr<int64_t>();
  auto* edge_child_ptr = edge_child_node_id.data_ptr<int64_t>();
  auto* edge_parent_compact_ptr = edge_parent_compact_id.data_ptr<int64_t>();
  auto* edge_child_compact_ptr = edge_child_compact_id.data_ptr<int64_t>();
  auto* edge_net_ptr = edge_net_index.data_ptr<int64_t>();
  auto* edge_to_segment_ptr = edge_to_segment_id.data_ptr<int64_t>();
  auto* sink_node_ptr = sink_node_id.data_ptr<int64_t>();
  auto* sink_net_ptr = sink_net_id.data_ptr<int64_t>();
  auto* sink_compact_ptr = sink_node_compact_id.data_ptr<int64_t>();
  auto* driver_ptr = driver_pin_id.data_ptr<int64_t>();
  auto* segment_net_ptr = segment_net_id.data_ptr<int64_t>();
  auto* segment_net_index_ptr = segment_net_index.data_ptr<int64_t>();
  auto* affected_net_ids_ptr = affected_net_ids.data_ptr<int64_t>();
  auto* net_segment_start_ptr = net_segment_start.data_ptr<int64_t>();
  auto* segment_parent_ptr = segment_parent_node_id.data_ptr<int64_t>();
  auto* segment_child_ptr = segment_child_node_id.data_ptr<int64_t>();
  auto* segment_parent_x_ptr = segment_parent_x_dbu.data_ptr<int64_t>();
  auto* segment_parent_y_ptr = segment_parent_y_dbu.data_ptr<int64_t>();
  auto* segment_child_x_ptr = segment_child_x_dbu.data_ptr<int64_t>();
  auto* segment_child_y_ptr = segment_child_y_dbu.data_ptr<int64_t>();

  const auto fill_started_at = Clock::now();
  const auto fill_outputs = [&](auto* edge_r_ptr,
                                auto* edge_c_ptr,
                                auto* node_cap_ptr,
                                auto* segment_r_ptr,
                                auto* segment_c_ptr,
                                auto* parent_fraction_ptr,
                                auto* child_fraction_ptr,
                                auto* sub_fraction_ptr) {
    using scalar_t = typename std::remove_pointer<decltype(edge_r_ptr)>::type;
#pragma omp parallel for schedule(static) num_threads(thread_count) if (thread_count > 1)
    for (int64_t net_id = 0; net_id < input_net_count; ++net_id) {
      const int64_t output_net = valid_net_index[static_cast<size_t>(net_id)];
      if (output_net < 0) {
        continue;
      }
      const NetPlan plan = build_net_plan(in, net_id);
      TORCH_INTERNAL_ASSERT(plan.valid());
      const int64_t node_base = node_offset[static_cast<size_t>(net_id)];
      const int64_t edge_base = edge_offset[static_cast<size_t>(net_id)];
      const int64_t sink_base = sink_offset[static_cast<size_t>(net_id)];
      const int64_t segment_base = segment_offset[static_cast<size_t>(net_id)];
      net_ids_ptr[output_net] = net_id;
      net_topo_start_ptr[output_net] = node_base;
      net_edge_start_ptr[output_net] = edge_base;
      net_sink_start_ptr[output_net] = sink_base;
      driver_ptr[output_net] = plan.driver_pin_id;
      for (int64_t local = 0; local < static_cast<int64_t>(plan.topo_nodes.size()); ++local) {
        flat_topo_ptr[node_base + local] = plan.topo_nodes[static_cast<size_t>(local)];
        node_cap_ptr[node_base + local] = static_cast<scalar_t>(plan.node_capacitance[static_cast<size_t>(local)]);
        edge_start_ptr[node_base + local] =
            edge_base + plan.edge_start[static_cast<size_t>(local)];
      }
      for (int64_t local = 0; local < static_cast<int64_t>(plan.edge_parent.size()); ++local) {
        const int64_t output_edge = edge_base + local;
        edge_parent_ptr[output_edge] = plan.edge_parent[static_cast<size_t>(local)];
        edge_child_ptr[output_edge] = plan.edge_child[static_cast<size_t>(local)];
        edge_parent_compact_ptr[output_edge] = node_base + plan.edge_parent_local[static_cast<size_t>(local)];
        edge_child_compact_ptr[output_edge] = node_base + plan.edge_child_local[static_cast<size_t>(local)];
        edge_net_ptr[output_edge] = output_net;
        edge_r_ptr[output_edge] = static_cast<scalar_t>(plan.edge_resistance[static_cast<size_t>(local)]);
        edge_c_ptr[output_edge] = static_cast<scalar_t>(plan.edge_capacitance[static_cast<size_t>(local)]);
        const int64_t local_segment = plan.edge_segment_local[static_cast<size_t>(local)];
        edge_to_segment_ptr[output_edge] = local_segment < 0 ? -1 : segment_base + local_segment;
      }
      for (int64_t local = 0; local < static_cast<int64_t>(plan.sink_node_id.size()); ++local) {
        sink_node_ptr[sink_base + local] = plan.sink_node_id[static_cast<size_t>(local)];
        sink_net_ptr[sink_base + local] = net_id;
        sink_compact_ptr[sink_base + local] = node_base + plan.sink_local_id[static_cast<size_t>(local)];
      }
      const int64_t compact_affected_net = affected_net_index[static_cast<size_t>(net_id)];
      if (compact_affected_net >= 0) {
        affected_net_ids_ptr[compact_affected_net] = net_id;
        net_segment_start_ptr[compact_affected_net] = segment_base;
      }
      for (int64_t local = 0; local < static_cast<int64_t>(plan.segment_parent.size()); ++local) {
        const int64_t output_segment = segment_base + local;
        segment_net_ptr[output_segment] = net_id;
        segment_net_index_ptr[output_segment] = compact_affected_net;
        segment_parent_ptr[output_segment] = plan.segment_parent[static_cast<size_t>(local)];
        segment_child_ptr[output_segment] = plan.segment_child[static_cast<size_t>(local)];
        segment_parent_x_ptr[output_segment] = plan.segment_parent_x_dbu[static_cast<size_t>(local)];
        segment_parent_y_ptr[output_segment] = plan.segment_parent_y_dbu[static_cast<size_t>(local)];
        segment_child_x_ptr[output_segment] = plan.segment_child_x_dbu[static_cast<size_t>(local)];
        segment_child_y_ptr[output_segment] = plan.segment_child_y_dbu[static_cast<size_t>(local)];
        segment_r_ptr[output_segment] = static_cast<scalar_t>(plan.segment_resistance[static_cast<size_t>(local)]);
        segment_c_ptr[output_segment] = static_cast<scalar_t>(plan.segment_capacitance[static_cast<size_t>(local)]);
      }
    }
    fill_fraction_tables<scalar_t>(
        segment_parent_x_ptr,
        segment_parent_y_ptr,
        segment_child_x_ptr,
        segment_child_y_ptr,
        total_segments,
        max_repeater_count,
        parent_fraction_ptr,
        child_fraction_ptr,
        sub_fraction_ptr,
        thread_count);
  };
  AT_DISPATCH_FLOATING_TYPES(pin_capacitance.scalar_type(), "pack_segment_count_topology", [&] {
    fill_outputs(
        edge_resistance.data_ptr<scalar_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        node_capacitance.data_ptr<scalar_t>(),
        segment_edge_resistance.data_ptr<scalar_t>(),
        segment_edge_capacitance.data_ptr<scalar_t>(),
        parent_cap_fraction.data_ptr<scalar_t>(),
        child_cap_fraction.data_ptr<scalar_t>(),
        segment_sub_resistance_fraction.data_ptr<scalar_t>());
  });
  net_topo_start_ptr[valid_net_count] = total_nodes;
  net_edge_start_ptr[valid_net_count] = total_edges;
  net_sink_start_ptr[valid_net_count] = total_sinks;
  edge_start_ptr[total_nodes] = total_edges;
  net_segment_start_ptr[affected_net_count] = total_segments;
  const double fill_ms = elapsed_ms(fill_started_at);

  py::dict skipped;
  for (int64_t code = static_cast<int64_t>(SkipReason::kSinglePinNet);
       code <= static_cast<int64_t>(SkipReason::kMissingPinCap);
       ++code) {
    const auto reason = static_cast<SkipReason>(code);
    int64_t count = 0;
    for (const NetCounts& item : counts) {
      count += item.reason == reason ? 1 : 0;
    }
    if (count > 0) {
      skipped[skip_reason_name(reason)] = count;
    }
  }

  py::dict prepared;
  prepared["net_ids"] = net_ids;
  prepared["net_topo_start"] = net_topo_start;
  prepared["net_edge_start"] = net_edge_start;
  prepared["net_sink_start"] = net_sink_start;
  prepared["flat_topo_node_id"] = flat_topo_node_id;
  prepared["edge_start"] = edge_start;
  prepared["edge_parent_node_id"] = edge_parent_node_id;
  prepared["edge_child_node_id"] = edge_child_node_id;
  prepared["edge_parent_compact_id"] = edge_parent_compact_id;
  prepared["edge_child_compact_id"] = edge_child_compact_id;
  prepared["edge_net_index"] = edge_net_index;
  prepared["edge_resistance"] = edge_resistance;
  prepared["edge_capacitance"] = edge_capacitance;
  prepared["node_capacitance"] = node_capacitance;
  prepared["edge_to_segment_id"] = edge_to_segment_id;
  prepared["sink_node_id"] = sink_node_id;
  prepared["sink_net_id"] = sink_net_id;
  prepared["sink_node_compact_id"] = sink_node_compact_id;
  prepared["driver_pin_id"] = driver_pin_id;
  prepared["parent_cap_fraction"] = parent_cap_fraction;
  prepared["child_cap_fraction"] = child_cap_fraction;
  prepared["segment_sub_resistance_fraction"] = segment_sub_resistance_fraction;

  py::dict geometry;
  geometry["segment_ids"] = segment_ids;
  geometry["segment_net_id"] = segment_net_id;
  geometry["segment_net_index"] = segment_net_index;
  geometry["net_ids"] = affected_net_ids;
  geometry["net_segment_start"] = net_segment_start;
  geometry["parent_node_id"] = segment_parent_node_id;
  geometry["child_node_id"] = segment_child_node_id;
  geometry["parent_x_dbu"] = segment_parent_x_dbu;
  geometry["parent_y_dbu"] = segment_parent_y_dbu;
  geometry["child_x_dbu"] = segment_child_x_dbu;
  geometry["child_y_dbu"] = segment_child_y_dbu;
  geometry["edge_resistance"] = segment_edge_resistance;
  geometry["edge_capacitance"] = segment_edge_capacitance;

  py::dict metadata;
  metadata["backend"] = "native_cpp";
  metadata["input_net_count"] = input_net_count;
  metadata["valid_net_count"] = valid_net_count;
  metadata["affected_net_count"] = affected_net_count;
  metadata["node_count"] = total_nodes;
  metadata["edge_count"] = total_edges;
  metadata["sink_count"] = total_sinks;
  metadata["segment_count"] = total_segments;
  metadata["thread_count"] = thread_count;
  metadata["validate_count_ms"] = validate_count_ms;
  metadata["prefix_sum_ms"] = prefix_sum_ms;
  metadata["allocate_ms"] = allocate_ms;
  metadata["fill_ms"] = fill_ms;
  metadata["skipped_reasons"] = skipped;

  py::dict result;
  result["prepared_timing_inputs"] = prepared;
  result["packed_segment_geometry"] = geometry;
  result["metadata"] = metadata;
  return result;
}

py::dict pack_candidate_topology(
    const at::Tensor& flat_net2pin,
    const at::Tensor& flat_net2pin_start,
    const at::Tensor& net2driver,
    const at::Tensor& pin2node,
    const at::Tensor& net_flat_topo_sort,
    const at::Tensor& net_flat_topo_sort_start,
    const at::Tensor& pin_fa,
    const at::Tensor& node_x,
    const at::Tensor& node_y,
    const at::Tensor& node_x_dbu,
    const at::Tensor& node_y_dbu,
    const at::Tensor& candidate_net_id,
    const at::Tensor& candidate_parent_node_id,
    const at::Tensor& candidate_child_node_id,
    const at::Tensor& candidate_tree_node_id,
    const at::Tensor& candidate_synthetic_node_id,
    const at::Tensor& candidate_is_segment,
    const at::Tensor& candidate_split_ratio,
    const at::Tensor& pin_capacitance,
    int64_t num_movable_nodes,
    int64_t num_terminals,
    double dbu,
    double scale_factor,
    double r_unit,
    double c_unit,
    int64_t num_threads) {
  TORCH_CHECK(dbu > 0.0, "dbu must be positive");
  TORCH_CHECK(scale_factor > 0.0, "scale_factor must be positive");
  TORCH_CHECK(
      pin_capacitance.scalar_type() == at::kFloat || pin_capacitance.scalar_type() == at::kDouble,
      "pin_capacitance must be float32 or float64");
  const std::array<const at::Tensor*, 7> candidate_tensors = {
      &candidate_net_id,
      &candidate_parent_node_id,
      &candidate_child_node_id,
      &candidate_tree_node_id,
      &candidate_synthetic_node_id,
      &candidate_is_segment,
      &candidate_split_ratio,
  };
  for (size_t index = 0; index < candidate_tensors.size(); ++index) {
    check_cpu_1d_contiguous(*candidate_tensors[index], "candidate input");
    TORCH_CHECK(
        candidate_tensors[index]->numel() == candidate_net_id.numel(),
        "candidate input lengths must match candidate_net_id");
    if (index < 6) {
      TORCH_CHECK(
          candidate_tensors[index]->scalar_type() == at::kInt ||
              candidate_tensors[index]->scalar_type() == at::kLong,
          "candidate integer inputs must be int32 or int64");
    } else {
      TORCH_CHECK(
          candidate_tensors[index]->scalar_type() == at::kFloat ||
              candidate_tensors[index]->scalar_type() == at::kDouble,
          "candidate_split_ratio must be float32 or float64");
    }
  }

  const int64_t available_threads = std::max<int64_t>(1, at::get_num_threads());
  const int64_t thread_count = std::max<int64_t>(1, std::min(num_threads, available_threads));
  const Inputs in{
      IntegerView(flat_net2pin, "flat_net2pin"),
      IntegerView(flat_net2pin_start, "flat_net2pin_start"),
      IntegerView(net2driver, "net2driver"),
      IntegerView(pin2node, "pin2node"),
      IntegerView(net_flat_topo_sort, "net_flat_topo_sort"),
      IntegerView(net_flat_topo_sort_start, "net_flat_topo_sort_start"),
      IntegerView(pin_fa, "pin_fa"),
      FloatView(node_x, "node_x"),
      FloatView(node_y, "node_y"),
      FloatView(node_x_dbu, "node_x_dbu"),
      FloatView(node_y_dbu, "node_y_dbu"),
      FloatView(pin_capacitance, "pin_capacitance"),
      num_movable_nodes,
      num_terminals,
      dbu * scale_factor,
      r_unit,
      c_unit,
  };
  TORCH_CHECK(
      in.flat_net2pin_start.size() == in.topo_start.size(),
      "net pin and topology start arrays must describe the same net count");

  const IntegerView candidate_net(candidate_net_id, "candidate_net_id");
  const IntegerView candidate_parent(candidate_parent_node_id, "candidate_parent_node_id");
  const IntegerView candidate_child(candidate_child_node_id, "candidate_child_node_id");
  const IntegerView candidate_tree(candidate_tree_node_id, "candidate_tree_node_id");
  const IntegerView candidate_synthetic(candidate_synthetic_node_id, "candidate_synthetic_node_id");
  const IntegerView candidate_segment(candidate_is_segment, "candidate_is_segment");
  const FloatView candidate_ratio(candidate_split_ratio, "candidate_split_ratio");
  const int64_t candidate_count = candidate_net.size();

  std::unordered_map<int64_t, std::vector<CandidateSpec>> specs_by_net;
  std::vector<CandidateSpec> all_specs;
  all_specs.reserve(static_cast<size_t>(candidate_count));
  for (int64_t index = 0; index < candidate_count; ++index) {
    const bool is_segment = candidate_segment[index] != 0;
    const CandidateSpec spec{
        index,
        candidate_net[index],
        candidate_parent[index],
        candidate_child[index],
        candidate_tree[index],
        candidate_synthetic[index],
        is_segment,
        candidate_ratio[index],
    };
    all_specs.push_back(spec);
    specs_by_net[spec.net_id].push_back(spec);
  }

  const int64_t input_net_count = std::max<int64_t>(0, in.topo_start.size() - 1);
  std::vector<ExpandedCandidatePlan> expanded_plans;
  expanded_plans.reserve(static_cast<size_t>(input_net_count));
  std::vector<int64_t> valid_input_net_ids;
  std::vector<int64_t> candidate_local_node(static_cast<size_t>(candidate_count), -1);
  std::unordered_set<int64_t> valid_net_set;
  for (int64_t net_id = 0; net_id < input_net_count; ++net_id) {
    const NetPlan original = build_net_plan(in, net_id);
    if (!original.valid()) {
      continue;
    }
    const auto specs_it = specs_by_net.find(net_id);
    const std::vector<CandidateSpec> empty_specs;
    const auto& specs = specs_it == specs_by_net.end() ? empty_specs : specs_it->second;
    expanded_plans.push_back(expand_candidate_plan(original, specs));
    const ExpandedCandidatePlan& expanded = expanded_plans.back();
    for (size_t index = 0; index < expanded.candidate_index.size(); ++index) {
      candidate_local_node[static_cast<size_t>(expanded.candidate_index[index])] =
          expanded.candidate_local_node[index];
    }
    valid_input_net_ids.push_back(net_id);
    valid_net_set.insert(net_id);
  }
  for (const CandidateSpec& spec : all_specs) {
    TORCH_CHECK(
        valid_net_set.count(spec.net_id) != 0,
        "candidate references a net that is not present in the valid timing topology: ",
        spec.net_id);
  }

  int64_t total_nodes = 0;
  int64_t total_edges = 0;
  int64_t total_sinks = 0;
  for (const ExpandedCandidatePlan& expanded : expanded_plans) {
    total_nodes += static_cast<int64_t>(expanded.plan.topo_nodes.size());
    total_edges += static_cast<int64_t>(expanded.plan.edge_parent.size());
    total_sinks += static_cast<int64_t>(expanded.plan.sink_node_id.size());
  }
  const auto long_options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  const auto value_options = at::TensorOptions().dtype(pin_capacitance.scalar_type()).device(at::kCPU);
  at::Tensor net_ids = at::empty({static_cast<int64_t>(expanded_plans.size())}, long_options);
  at::Tensor net_topo_start = at::empty({static_cast<int64_t>(expanded_plans.size()) + 1}, long_options);
  at::Tensor net_edge_start = at::empty({static_cast<int64_t>(expanded_plans.size()) + 1}, long_options);
  at::Tensor net_sink_start = at::empty({static_cast<int64_t>(expanded_plans.size()) + 1}, long_options);
  at::Tensor flat_topo_node_id = at::empty({total_nodes}, long_options);
  at::Tensor packed_net_flat_topo_sort = at::empty({total_nodes}, long_options);
  at::Tensor edge_start = at::empty({total_nodes + 1}, long_options);
  at::Tensor compact_pin_fa = at::full({total_nodes}, -1, long_options);
  at::Tensor flat_pin_to = at::empty({total_edges}, long_options);
  at::Tensor edge_parent_node_id = at::empty({total_edges}, long_options);
  at::Tensor edge_child_node_id = at::empty({total_edges}, long_options);
  at::Tensor edge_parent_compact_id = at::empty({total_edges}, long_options);
  at::Tensor edge_child_compact_id = at::empty({total_edges}, long_options);
  at::Tensor edge_net_index = at::empty({total_edges}, long_options);
  at::Tensor edge_resistance = at::zeros({total_nodes}, value_options);
  at::Tensor edge_capacitance = at::zeros({total_nodes}, value_options);
  at::Tensor node_capacitance = at::empty({total_nodes}, value_options);
  at::Tensor sink_node_id = at::empty({total_sinks}, long_options);
  at::Tensor sink_net_index = at::empty({total_sinks}, long_options);
  at::Tensor sink_pin_id = at::empty({total_sinks}, long_options);
  at::Tensor driver_pin_id = at::empty({static_cast<int64_t>(expanded_plans.size())}, long_options);
  at::Tensor driver_arrival = at::zeros({static_cast<int64_t>(expanded_plans.size())}, value_options);
  at::Tensor driver_slew = at::zeros({static_cast<int64_t>(expanded_plans.size())}, value_options);
  at::Tensor output_candidate_node_id = at::empty({candidate_count}, long_options);
  at::Tensor output_candidate_net_id = at::empty({candidate_count}, long_options);

  auto* net_ids_ptr = net_ids.data_ptr<int64_t>();
  auto* net_topo_start_ptr = net_topo_start.data_ptr<int64_t>();
  auto* net_edge_start_ptr = net_edge_start.data_ptr<int64_t>();
  auto* net_sink_start_ptr = net_sink_start.data_ptr<int64_t>();
  auto* flat_topo_ptr = flat_topo_node_id.data_ptr<int64_t>();
  auto* topo_sort_ptr = packed_net_flat_topo_sort.data_ptr<int64_t>();
  auto* edge_start_ptr = edge_start.data_ptr<int64_t>();
  auto* pin_fa_ptr = compact_pin_fa.data_ptr<int64_t>();
  auto* flat_pin_to_ptr = flat_pin_to.data_ptr<int64_t>();
  auto* edge_parent_ptr = edge_parent_node_id.data_ptr<int64_t>();
  auto* edge_child_ptr = edge_child_node_id.data_ptr<int64_t>();
  auto* edge_parent_compact_ptr = edge_parent_compact_id.data_ptr<int64_t>();
  auto* edge_child_compact_ptr = edge_child_compact_id.data_ptr<int64_t>();
  auto* edge_net_ptr = edge_net_index.data_ptr<int64_t>();
  auto* sink_node_ptr = sink_node_id.data_ptr<int64_t>();
  auto* sink_net_ptr = sink_net_index.data_ptr<int64_t>();
  auto* sink_pin_ptr = sink_pin_id.data_ptr<int64_t>();
  auto* driver_ptr = driver_pin_id.data_ptr<int64_t>();
  auto* candidate_node_ptr = output_candidate_node_id.data_ptr<int64_t>();
  auto* candidate_net_ptr = output_candidate_net_id.data_ptr<int64_t>();
  int64_t node_base = 0;
  int64_t edge_base = 0;
  int64_t sink_base = 0;
  for (size_t output_net_index = 0; output_net_index < expanded_plans.size(); ++output_net_index) {
    const NetPlan& plan = expanded_plans[output_net_index].plan;
    const int64_t net_id = valid_input_net_ids[output_net_index];
    net_ids_ptr[output_net_index] = net_id;
    net_topo_start_ptr[output_net_index] = node_base;
    net_edge_start_ptr[output_net_index] = edge_base;
    net_sink_start_ptr[output_net_index] = sink_base;
    driver_ptr[output_net_index] = plan.driver_pin_id;
    for (int64_t local = 0; local < static_cast<int64_t>(plan.topo_nodes.size()); ++local) {
      const int64_t compact_node = node_base + local;
      flat_topo_ptr[compact_node] = plan.compact_to_original[static_cast<size_t>(local)];
      edge_start_ptr[compact_node] =
          edge_base + plan.compact_flat_pin_to_start[static_cast<size_t>(local)];
      pin_fa_ptr[compact_node] =
          plan.compact_parent[static_cast<size_t>(local)] < 0
          ? -1
          : node_base + plan.compact_parent[static_cast<size_t>(local)];
    }
    for (int64_t local = 0; local < static_cast<int64_t>(plan.topo_compact.size()); ++local) {
      topo_sort_ptr[node_base + local] =
          node_base + plan.topo_compact[static_cast<size_t>(local)];
    }
    for (int64_t local = 0; local < static_cast<int64_t>(plan.compact_flat_pin_to.size()); ++local) {
      flat_pin_to_ptr[edge_base + local] =
          node_base + plan.compact_flat_pin_to[static_cast<size_t>(local)];
    }
    for (int64_t local = 0; local < static_cast<int64_t>(plan.edge_parent.size()); ++local) {
      const int64_t output_edge = edge_base + local;
      edge_parent_ptr[output_edge] = plan.edge_parent[static_cast<size_t>(local)];
      edge_child_ptr[output_edge] = plan.edge_child[static_cast<size_t>(local)];
      edge_parent_compact_ptr[output_edge] =
          node_base + plan.compact_local_by_original.at(
              plan.edge_parent[static_cast<size_t>(local)]);
      edge_child_compact_ptr[output_edge] =
          node_base + plan.compact_local_by_original.at(
              plan.edge_child[static_cast<size_t>(local)]);
      edge_net_ptr[output_edge] = static_cast<int64_t>(output_net_index);
    }
    for (int64_t local = 0; local < static_cast<int64_t>(plan.sink_node_id.size()); ++local) {
      sink_node_ptr[sink_base + local] =
          node_base + plan.sink_local_id[static_cast<size_t>(local)];
      sink_net_ptr[sink_base + local] = net_id;
      sink_pin_ptr[sink_base + local] = plan.sink_node_id[static_cast<size_t>(local)];
    }
    node_base += static_cast<int64_t>(plan.topo_nodes.size());
    edge_base += static_cast<int64_t>(plan.edge_parent.size());
    sink_base += static_cast<int64_t>(plan.sink_node_id.size());
  }
  net_topo_start_ptr[expanded_plans.size()] = node_base;
  net_edge_start_ptr[expanded_plans.size()] = edge_base;
  net_sink_start_ptr[expanded_plans.size()] = sink_base;
  edge_start_ptr[total_nodes] = total_edges;

  // The value tensors use the same scalar type as the input pin-cap table.
  AT_DISPATCH_FLOATING_TYPES(pin_capacitance.scalar_type(), "pack_candidate_topology", [&] {
    auto* node_cap_ptr = node_capacitance.data_ptr<scalar_t>();
    auto* edge_r_ptr = edge_resistance.data_ptr<scalar_t>();
    auto* edge_c_ptr = edge_capacitance.data_ptr<scalar_t>();
    for (size_t output_net_index = 0; output_net_index < expanded_plans.size(); ++output_net_index) {
      const NetPlan& plan = expanded_plans[output_net_index].plan;
      const int64_t current_node_base =
          output_net_index == 0
          ? 0
          : net_topo_start_ptr[output_net_index];
      for (int64_t local = 0; local < static_cast<int64_t>(plan.topo_nodes.size()); ++local) {
        node_cap_ptr[current_node_base + local] =
            static_cast<scalar_t>(plan.compact_node_capacitance[static_cast<size_t>(local)]);
      }
      for (int64_t local = 0;
           local < static_cast<int64_t>(plan.compact_to_original.size());
           ++local) {
        edge_r_ptr[current_node_base + local] =
            static_cast<scalar_t>(plan.compact_edge_resistance[static_cast<size_t>(local)]);
        edge_c_ptr[current_node_base + local] =
            static_cast<scalar_t>(plan.compact_edge_capacitance[static_cast<size_t>(local)]);
      }
    }
  });

  for (int64_t index = 0; index < candidate_count; ++index) {
    TORCH_CHECK(
        candidate_local_node[static_cast<size_t>(index)] >= 0,
        "candidate could not be mapped to compact topology node: ",
        index);
    const int64_t net_id = candidate_net[index];
    const auto net_it = std::lower_bound(valid_input_net_ids.begin(), valid_input_net_ids.end(), net_id);
    TORCH_CHECK(
        net_it != valid_input_net_ids.end() && *net_it == net_id,
        "candidate net id is not present in packed topology: ",
        net_id);
    const int64_t output_net_index = static_cast<int64_t>(std::distance(valid_input_net_ids.begin(), net_it));
    candidate_node_ptr[index] =
        net_topo_start_ptr[output_net_index] + candidate_local_node[static_cast<size_t>(index)];
    candidate_net_ptr[index] = net_id;
  }

  py::dict prepared;
  prepared["net_ids"] = net_ids;
  prepared["net_topo_start"] = net_topo_start;
  prepared["net_edge_start"] = net_edge_start;
  prepared["net_sink_start"] = net_sink_start;
  prepared["flat_topo_node_id"] = flat_topo_node_id;
  prepared["edge_start"] = edge_start;
  prepared["net_flat_topo_sort"] = packed_net_flat_topo_sort;
  prepared["net_flat_topo_sort_start"] = net_topo_start;
  prepared["pin_fa"] = compact_pin_fa;
  prepared["flat_pin_to_start"] = edge_start;
  prepared["edge_parent_node_id"] = edge_parent_node_id;
  prepared["edge_child_node_id"] = edge_child_node_id;
  prepared["edge_parent_compact_id"] = edge_parent_compact_id;
  prepared["edge_child_compact_id"] = edge_child_compact_id;
  prepared["flat_pin_to"] = flat_pin_to;
  prepared["edge_net_index"] = edge_net_index;
  prepared["edge_resistance"] = edge_resistance;
  prepared["edge_capacitance"] = edge_capacitance;
  prepared["node_capacitance"] = node_capacitance;
  prepared["driver_arrival"] = driver_arrival;
  prepared["driver_slew"] = driver_slew;
  prepared["sink_node_id"] = sink_node_id;
  prepared["sink_net_index"] = sink_net_index;
  prepared["sink_pin_id"] = sink_pin_id;
  prepared["driver_pin_id"] = driver_pin_id;
  prepared["candidate_node_id"] = output_candidate_node_id;
  prepared["candidate_net_id"] = output_candidate_net_id;

  py::dict metadata;
  std::vector<int64_t> compact_node_to_original;
  compact_node_to_original.reserve(static_cast<size_t>(total_nodes));
  for (const ExpandedCandidatePlan& expanded : expanded_plans) {
    const NetPlan& plan = expanded.plan;
    compact_node_to_original.insert(
        compact_node_to_original.end(),
        plan.compact_to_original.begin(),
        plan.compact_to_original.end());
  }
  std::vector<int64_t> packed_net_ids = valid_input_net_ids;
  metadata["backend"] = "native_cpp_candidate";
  metadata["topology_source"] = "native_candidate_packer";
  metadata["input_net_count"] = input_net_count;
  metadata["valid_net_count"] = static_cast<int64_t>(expanded_plans.size());
  metadata["node_count"] = total_nodes;
  metadata["edge_count"] = total_edges;
  metadata["sink_count"] = total_sinks;
  metadata["candidate_count"] = candidate_count;
  metadata["thread_count"] = thread_count;
  metadata["net_ids"] = py::cast(packed_net_ids);
  metadata["compact_node_to_original"] = py::cast(compact_node_to_original);

  py::dict result;
  result["prepared_timing_inputs"] = prepared;
  result["metadata"] = metadata;
  return result;
}

py::dict select_packed_segment_count_inputs(
    const py::dict& prepared_timing_inputs,
    const py::dict& packed_segment_geometry,
    const at::Tensor& active_net_ids,
    int64_t num_threads) {
  const auto total_started_at = Clock::now();
  const int64_t available_threads = std::max<int64_t>(1, at::get_num_threads());
  const int64_t thread_count = std::max<int64_t>(1, std::min(num_threads, available_threads));

  auto required = [](const py::dict& values, const char* name) -> at::Tensor {
    TORCH_CHECK(values.contains(name), "missing packed segment-count field: ", name);
    return values[name].cast<at::Tensor>();
  };
  auto output_index = [](const std::vector<int64_t>& values) {
    auto result = at::empty(
        {static_cast<int64_t>(values.size())},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto* ptr = result.data_ptr<int64_t>();
    for (int64_t index = 0; index < static_cast<int64_t>(values.size()); ++index) {
      ptr[index] = values[static_cast<size_t>(index)];
    }
    return result;
  };
  auto select = [&required](const py::dict& values, const char* name, const at::Tensor& index) {
    const auto source = required(values, name);
    TORCH_CHECK(source.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(source.dim() >= 1, name, " must have a leading selection dimension");
    TORCH_CHECK(source.is_contiguous(), name, " must be contiguous");
    return source.index_select(0, index);
  };

  const at::Tensor prepared_net_ids_tensor = required(prepared_timing_inputs, "net_ids");
  const at::Tensor geometry_net_ids_tensor = required(packed_segment_geometry, "net_ids");
  const at::Tensor net_topo_start_tensor = required(prepared_timing_inputs, "net_topo_start");
  const at::Tensor net_edge_start_tensor = required(prepared_timing_inputs, "net_edge_start");
  const at::Tensor net_sink_start_tensor = required(prepared_timing_inputs, "net_sink_start");
  const at::Tensor geometry_segment_start_tensor =
      required(packed_segment_geometry, "net_segment_start");
  const at::Tensor global_edge_segment_tensor =
      required(prepared_timing_inputs, "edge_to_segment_id");
  const IntegerView prepared_net_ids(prepared_net_ids_tensor, "prepared.net_ids");
  const IntegerView geometry_net_ids(geometry_net_ids_tensor, "geometry.net_ids");
  const IntegerView net_topo_start(net_topo_start_tensor, "prepared.net_topo_start");
  const IntegerView net_edge_start(net_edge_start_tensor, "prepared.net_edge_start");
  const IntegerView net_sink_start(net_sink_start_tensor, "prepared.net_sink_start");
  const IntegerView geometry_segment_start(
      geometry_segment_start_tensor, "geometry.net_segment_start");
  const IntegerView global_edge_segment(
      global_edge_segment_tensor, "prepared.edge_to_segment_id");
  const IntegerView query(active_net_ids, "active_net_ids");

  TORCH_CHECK(
      net_topo_start.size() == prepared_net_ids.size() + 1 &&
          net_edge_start.size() == prepared_net_ids.size() + 1 &&
          net_sink_start.size() == prepared_net_ids.size() + 1,
      "prepared net CSR starts must have net_count + 1 entries");
  TORCH_CHECK(
      geometry_segment_start.size() == geometry_net_ids.size() + 1,
      "geometry net_segment_start must have affected_net_count + 1 entries");
  for (int64_t index = 1; index < prepared_net_ids.size(); ++index) {
    TORCH_CHECK(
        prepared_net_ids[index - 1] < prepared_net_ids[index],
        "prepared.net_ids must be unique ascending");
  }
  for (int64_t index = 1; index < geometry_net_ids.size(); ++index) {
    TORCH_CHECK(
        geometry_net_ids[index - 1] < geometry_net_ids[index],
        "geometry.net_ids must be unique ascending");
  }
  for (int64_t index = 0; index < geometry_net_ids.size(); ++index) {
    const int64_t net_id = geometry_net_ids[index];
    int64_t left = 0;
    int64_t right = prepared_net_ids.size();
    while (left < right) {
      const int64_t middle = left + (right - left) / 2;
      if (prepared_net_ids[middle] < net_id) {
        left = middle + 1;
      } else {
        right = middle;
      }
    }
    TORCH_CHECK(
        left < prepared_net_ids.size() && prepared_net_ids[left] == net_id,
        "geometry.net_ids must be a subset of prepared.net_ids");
  }

  const auto match_started_at = Clock::now();
  std::vector<int64_t> canonical_query;
  canonical_query.reserve(static_cast<size_t>(query.size()));
  for (int64_t index = 0; index < query.size(); ++index) {
    canonical_query.push_back(query[index]);
  }
  std::sort(canonical_query.begin(), canonical_query.end());
  canonical_query.erase(
      std::unique(canonical_query.begin(), canonical_query.end()), canonical_query.end());

  struct SelectedNet {
    int64_t geometry_position;
    int64_t prepared_position;
    int64_t node_begin;
    int64_t node_end;
    int64_t edge_begin;
    int64_t edge_end;
    int64_t sink_begin;
    int64_t sink_end;
    int64_t segment_begin;
    int64_t segment_end;
  };
  std::vector<SelectedNet> selected_nets;
  selected_nets.reserve(canonical_query.size());
  std::vector<int64_t> prepared_values;
  prepared_values.reserve(static_cast<size_t>(prepared_net_ids.size()));
  for (int64_t index = 0; index < prepared_net_ids.size(); ++index) {
    prepared_values.push_back(prepared_net_ids[index]);
  }
  std::vector<int64_t> geometry_values;
  geometry_values.reserve(static_cast<size_t>(geometry_net_ids.size()));
  for (int64_t index = 0; index < geometry_net_ids.size(); ++index) {
    geometry_values.push_back(geometry_net_ids[index]);
  }
  for (const int64_t net_id : canonical_query) {
    const auto geometry_it = std::lower_bound(geometry_values.begin(), geometry_values.end(), net_id);
    if (geometry_it == geometry_values.end() || *geometry_it != net_id) {
      continue;
    }
    const auto prepared_it = std::lower_bound(prepared_values.begin(), prepared_values.end(), net_id);
    TORCH_CHECK(
        prepared_it != prepared_values.end() && *prepared_it == net_id,
        "selected geometry net is missing from prepared timing inputs");
    const int64_t geometry_position =
        static_cast<int64_t>(std::distance(geometry_values.begin(), geometry_it));
    const int64_t prepared_position =
        static_cast<int64_t>(std::distance(prepared_values.begin(), prepared_it));
    const SelectedNet item{
        geometry_position,
        prepared_position,
        net_topo_start[prepared_position],
        net_topo_start[prepared_position + 1],
        net_edge_start[prepared_position],
        net_edge_start[prepared_position + 1],
        net_sink_start[prepared_position],
        net_sink_start[prepared_position + 1],
        geometry_segment_start[geometry_position],
        geometry_segment_start[geometry_position + 1],
    };
    TORCH_CHECK(
        item.node_begin >= 0 && item.node_end >= item.node_begin &&
            item.edge_begin >= 0 && item.edge_end >= item.edge_begin &&
            item.sink_begin >= 0 && item.sink_end >= item.sink_begin &&
            item.segment_begin >= 0 && item.segment_end >= item.segment_begin,
        "packed segment-count CSR offsets must be monotonic and non-negative");
    selected_nets.push_back(item);
  }
  const double match_ms = elapsed_ms(match_started_at);

  const auto prefix_started_at = Clock::now();
  std::vector<int64_t> local_net_ids;
  std::vector<int64_t> source_prepared_positions;
  std::vector<int64_t> node_starts{0};
  std::vector<int64_t> edge_starts{0};
  std::vector<int64_t> sink_starts{0};
  std::vector<int64_t> source_node_index;
  std::vector<int64_t> source_edge_index;
  std::vector<int64_t> source_sink_index;
  std::vector<int64_t> source_segment_row_index;
  for (const SelectedNet& item : selected_nets) {
    const int64_t node_count = item.node_end - item.node_begin;
    const int64_t edge_count = item.edge_end - item.edge_begin;
    const int64_t sink_count = item.sink_end - item.sink_begin;
    const int64_t segment_count = item.segment_end - item.segment_begin;
    local_net_ids.push_back(prepared_net_ids[item.prepared_position]);
    source_prepared_positions.push_back(item.prepared_position);
    node_starts.push_back(node_starts.back() + node_count);
    edge_starts.push_back(edge_starts.back() + edge_count);
    sink_starts.push_back(sink_starts.back() + sink_count);
    for (int64_t value = item.node_begin; value < item.node_end; ++value) {
      source_node_index.push_back(value);
    }
    for (int64_t value = item.edge_begin; value < item.edge_end; ++value) {
      source_edge_index.push_back(value);
    }
    for (int64_t value = item.sink_begin; value < item.sink_end; ++value) {
      source_sink_index.push_back(value);
    }
    for (int64_t value = item.segment_begin; value < item.segment_end; ++value) {
      source_segment_row_index.push_back(value);
    }
  }
  const double prefix_sum_ms = elapsed_ms(prefix_started_at);

  const auto allocate_started_at = Clock::now();
  const at::Tensor net_index = output_index(source_prepared_positions);
  const at::Tensor node_index = output_index(source_node_index);
  const at::Tensor edge_index = output_index(source_edge_index);
  const at::Tensor sink_index = output_index(source_sink_index);
  const at::Tensor segment_index = output_index(source_segment_row_index);
  const int64_t total_nodes = static_cast<int64_t>(source_node_index.size());
  const int64_t total_edges = static_cast<int64_t>(source_edge_index.size());
  const int64_t total_sinks = static_cast<int64_t>(source_sink_index.size());
  auto output_edge_start = at::empty(
      {total_nodes + 1}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto output_parent_compact = at::empty(
      {total_edges}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto output_child_compact = at::empty(
      {total_edges}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto output_edge_net_index = at::empty(
      {total_edges}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto output_edge_to_segment = at::empty(
      {total_edges}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto output_sink_compact = at::empty(
      {total_sinks}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  const double allocate_ms = elapsed_ms(allocate_started_at);

  const auto fill_started_at = Clock::now();
  const at::Tensor global_edge_start_tensor = required(prepared_timing_inputs, "edge_start");
  const at::Tensor global_parent_compact_tensor =
      required(prepared_timing_inputs, "edge_parent_compact_id");
  const at::Tensor global_child_compact_tensor =
      required(prepared_timing_inputs, "edge_child_compact_id");
  const at::Tensor global_sink_compact_tensor =
      required(prepared_timing_inputs, "sink_node_compact_id");
  const IntegerView global_edge_start(global_edge_start_tensor, "prepared.edge_start");
  const IntegerView global_parent_compact(
      global_parent_compact_tensor, "prepared.edge_parent_compact_id");
  const IntegerView global_child_compact(
      global_child_compact_tensor, "prepared.edge_child_compact_id");
  const IntegerView global_sink_compact(
      global_sink_compact_tensor, "prepared.sink_node_compact_id");
  TORCH_CHECK(
      global_edge_start.size() >= 1 && global_edge_start.size() ==
          required(prepared_timing_inputs, "flat_topo_node_id").numel() + 1 &&
          global_parent_compact.size() == global_edge_segment.size() &&
          global_child_compact.size() == global_edge_segment.size(),
      "prepared compact CSR arrays are inconsistent");
  auto* edge_start_ptr = output_edge_start.data_ptr<int64_t>();
  auto* parent_compact_ptr = output_parent_compact.data_ptr<int64_t>();
  auto* child_compact_ptr = output_child_compact.data_ptr<int64_t>();
  auto* edge_net_ptr = output_edge_net_index.data_ptr<int64_t>();
  auto* edge_segment_ptr = output_edge_to_segment.data_ptr<int64_t>();
  auto* sink_compact_ptr = output_sink_compact.data_ptr<int64_t>();
  int64_t local_node_base = 0;
  int64_t local_edge_base = 0;
  int64_t local_sink_base = 0;
  int64_t local_segment_base = 0;
  for (int64_t local_net = 0; local_net < static_cast<int64_t>(selected_nets.size()); ++local_net) {
    const SelectedNet& item = selected_nets[static_cast<size_t>(local_net)];
    for (int64_t source_node = item.node_begin; source_node < item.node_end; ++source_node) {
      edge_start_ptr[local_node_base + source_node - item.node_begin] =
          global_edge_start[source_node] - item.edge_begin + local_edge_base;
    }
    for (int64_t source_edge = item.edge_begin; source_edge < item.edge_end; ++source_edge) {
      const int64_t local_edge = local_edge_base + source_edge - item.edge_begin;
      parent_compact_ptr[local_edge] =
          global_parent_compact[source_edge] - item.node_begin + local_node_base;
      child_compact_ptr[local_edge] =
          global_child_compact[source_edge] - item.node_begin + local_node_base;
      edge_net_ptr[local_edge] = local_net;
      const int64_t source_segment = global_edge_segment[source_edge];
      TORCH_CHECK(
          source_segment < 0 ||
              (source_segment >= item.segment_begin && source_segment < item.segment_end),
          "edge_to_segment_id must be -1 or belong to its source geometry net");
      edge_segment_ptr[local_edge] = source_segment < 0
          ? -1
          : local_segment_base + source_segment - item.segment_begin;
    }
    for (int64_t source_sink = item.sink_begin; source_sink < item.sink_end; ++source_sink) {
      sink_compact_ptr[local_sink_base + source_sink - item.sink_begin] =
          global_sink_compact[source_sink] - item.node_begin + local_node_base;
    }
    local_node_base += item.node_end - item.node_begin;
    local_edge_base += item.edge_end - item.edge_begin;
    local_sink_base += item.sink_end - item.sink_begin;
    local_segment_base += item.segment_end - item.segment_begin;
  }
  edge_start_ptr[total_nodes] = total_edges;

  py::dict prepared;
  prepared["net_ids"] = output_index(local_net_ids);
  prepared["net_topo_start"] = output_index(node_starts);
  prepared["net_edge_start"] = output_index(edge_starts);
  prepared["net_sink_start"] = output_index(sink_starts);
  prepared["flat_topo_node_id"] = select(prepared_timing_inputs, "flat_topo_node_id", node_index);
  prepared["edge_start"] = output_edge_start;
  prepared["edge_parent_node_id"] = select(prepared_timing_inputs, "edge_parent_node_id", edge_index);
  prepared["edge_child_node_id"] = select(prepared_timing_inputs, "edge_child_node_id", edge_index);
  prepared["edge_parent_compact_id"] = output_parent_compact;
  prepared["edge_child_compact_id"] = output_child_compact;
  prepared["edge_net_index"] = output_edge_net_index;
  prepared["edge_resistance"] = select(prepared_timing_inputs, "edge_resistance", edge_index);
  prepared["edge_capacitance"] = select(prepared_timing_inputs, "edge_capacitance", edge_index);
  prepared["node_capacitance"] = select(prepared_timing_inputs, "node_capacitance", node_index);
  prepared["edge_to_segment_id"] = output_edge_to_segment;
  prepared["sink_node_id"] = select(prepared_timing_inputs, "sink_node_id", sink_index);
  prepared["sink_net_id"] = select(prepared_timing_inputs, "sink_net_id", sink_index);
  prepared["sink_node_compact_id"] = output_sink_compact;
  prepared["driver_pin_id"] = select(prepared_timing_inputs, "driver_pin_id", net_index);
  prepared["parent_cap_fraction"] = select(prepared_timing_inputs, "parent_cap_fraction", segment_index);
  prepared["child_cap_fraction"] = select(prepared_timing_inputs, "child_cap_fraction", segment_index);
  prepared["segment_sub_resistance_fraction"] =
      select(prepared_timing_inputs, "segment_sub_resistance_fraction", segment_index);
  const double fill_ms = elapsed_ms(fill_started_at);

  py::dict metadata;
  metadata["backend"] = "native_cpp";
  metadata["topology_source"] = "native_packed_active_view_selector";
  metadata["thread_count"] = thread_count;
  metadata["active_net_count"] = static_cast<int64_t>(selected_nets.size());
  metadata["segment_count"] = static_cast<int64_t>(source_segment_row_index.size());
  metadata["native_active_view_match_count_ms"] = match_ms;
  metadata["native_active_view_prefix_sum_ms"] = prefix_sum_ms;
  metadata["native_active_view_allocate_ms"] = allocate_ms;
  metadata["native_active_view_fill_ms"] = fill_ms;
  metadata["native_active_view_total_ms"] = elapsed_ms(total_started_at);

  py::dict result;
  result["prepared_inputs"] = prepared;
  result["source_prepared_net_positions"] = net_index;
  result["source_prepared_edge_index"] = edge_index;
  result["source_segment_row_index"] = segment_index;
  result["metadata"] = metadata;
  return result;
}

py::dict select_packed_candidate_inputs(
    const py::dict& prepared_timing_inputs,
    const at::Tensor& active_net_ids,
    int64_t num_threads) {
  const auto total_started_at = Clock::now();
  const int64_t available_threads = std::max<int64_t>(1, at::get_num_threads());
  const int64_t thread_count = std::max<int64_t>(1, std::min(num_threads, available_threads));

  auto required = [](const py::dict& values, const char* name) -> at::Tensor {
    TORCH_CHECK(values.contains(name), "missing packed candidate field: ", name);
    return values[name].cast<at::Tensor>();
  };
  auto output_index = [](const std::vector<int64_t>& values) {
    auto result = at::empty(
        {static_cast<int64_t>(values.size())},
        at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    auto* ptr = result.data_ptr<int64_t>();
    for (int64_t index = 0; index < static_cast<int64_t>(values.size()); ++index) {
      ptr[index] = values[static_cast<size_t>(index)];
    }
    return result;
  };
  auto select = [&required](
                    const py::dict& values,
                    const char* name,
                    const at::Tensor& index) {
    const auto source = required(values, name);
    TORCH_CHECK(source.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(source.dim() >= 1, name, " must have a leading selection dimension");
    TORCH_CHECK(source.is_contiguous(), name, " must be contiguous");
    return source.index_select(0, index);
  };

  const at::Tensor net_ids_tensor = required(prepared_timing_inputs, "net_ids");
  const at::Tensor net_topo_start_tensor =
      required(prepared_timing_inputs, "net_topo_start");
  const at::Tensor net_edge_start_tensor =
      required(prepared_timing_inputs, "net_edge_start");
  const at::Tensor net_sink_start_tensor =
      required(prepared_timing_inputs, "net_sink_start");
  const at::Tensor topo_tensor =
      required(prepared_timing_inputs, "net_flat_topo_sort");
  const at::Tensor edge_start_tensor = required(prepared_timing_inputs, "edge_start");
  const at::Tensor pin_fa_tensor = required(prepared_timing_inputs, "pin_fa");
  const at::Tensor flat_pin_to_tensor = required(prepared_timing_inputs, "flat_pin_to");
  const at::Tensor candidate_net_tensor =
      required(prepared_timing_inputs, "candidate_net_id");
  const at::Tensor candidate_node_tensor =
      required(prepared_timing_inputs, "candidate_node_id");
  const at::Tensor sink_node_tensor = required(prepared_timing_inputs, "sink_node_id");
  const at::Tensor sink_net_tensor = required(prepared_timing_inputs, "sink_net_index");

  const IntegerView net_ids(net_ids_tensor, "candidate.net_ids");
  const IntegerView net_topo_start(net_topo_start_tensor, "candidate.net_topo_start");
  const IntegerView net_edge_start(net_edge_start_tensor, "candidate.net_edge_start");
  const IntegerView net_sink_start(net_sink_start_tensor, "candidate.net_sink_start");
  const IntegerView topo(topo_tensor, "candidate.net_flat_topo_sort");
  const IntegerView edge_start(edge_start_tensor, "candidate.edge_start");
  const IntegerView pin_fa(pin_fa_tensor, "candidate.pin_fa");
  const IntegerView flat_pin_to(flat_pin_to_tensor, "candidate.flat_pin_to");
  const IntegerView candidate_net(candidate_net_tensor, "candidate.candidate_net_id");
  const IntegerView candidate_node(candidate_node_tensor, "candidate.candidate_node_id");
  const IntegerView sink_node(sink_node_tensor, "candidate.sink_node_id");
  const IntegerView sink_net(sink_net_tensor, "candidate.sink_net_index");
  const IntegerView query(active_net_ids, "active_net_ids");

  TORCH_CHECK(
      net_topo_start.size() == net_ids.size() + 1 &&
          net_edge_start.size() == net_ids.size() + 1 &&
          net_sink_start.size() == net_ids.size() + 1,
      "packed candidate net CSR starts must have net_count + 1 entries");
  TORCH_CHECK(
      candidate_net.size() == candidate_node.size(),
      "packed candidate net/node arrays must have equal length");
  TORCH_CHECK(
      sink_node.size() == sink_net.size(),
      "packed candidate sink node/net arrays must have equal length");
  TORCH_CHECK(
      edge_start.size() == topo.size() + 1 && pin_fa.size() == topo.size(),
      "packed candidate node arrays are inconsistent");
  for (int64_t index = 1; index < net_ids.size(); ++index) {
    TORCH_CHECK(
        net_ids[index - 1] < net_ids[index],
        "packed candidate net_ids must be unique ascending");
  }

  const auto match_started_at = Clock::now();
  std::vector<int64_t> canonical_query;
  canonical_query.reserve(static_cast<size_t>(query.size()));
  for (int64_t index = 0; index < query.size(); ++index) {
    canonical_query.push_back(query[index]);
  }
  std::sort(canonical_query.begin(), canonical_query.end());
  canonical_query.erase(
      std::unique(canonical_query.begin(), canonical_query.end()), canonical_query.end());

  std::vector<int64_t> net_values;
  net_values.reserve(static_cast<size_t>(net_ids.size()));
  for (int64_t index = 0; index < net_ids.size(); ++index) {
    net_values.push_back(net_ids[index]);
  }
  struct SelectedNet {
    int64_t net_id;
    int64_t source_position;
    int64_t node_begin;
    int64_t node_end;
    int64_t edge_begin;
    int64_t edge_end;
    int64_t sink_begin;
    int64_t sink_end;
    int64_t local_net;
    int64_t local_node_base;
    int64_t local_edge_base;
    int64_t local_sink_base;
  };
  std::vector<SelectedNet> selected_nets;
  selected_nets.reserve(canonical_query.size());
  int64_t total_nodes = 0;
  int64_t total_edges = 0;
  int64_t total_sinks = 0;
  for (const int64_t net_id : canonical_query) {
    const auto it = std::lower_bound(net_values.begin(), net_values.end(), net_id);
    if (it == net_values.end() || *it != net_id) {
      continue;
    }
    const int64_t source_position =
        static_cast<int64_t>(std::distance(net_values.begin(), it));
    const SelectedNet item{
        net_id,
        source_position,
        net_topo_start[source_position],
        net_topo_start[source_position + 1],
        net_edge_start[source_position],
        net_edge_start[source_position + 1],
        net_sink_start[source_position],
        net_sink_start[source_position + 1],
        static_cast<int64_t>(selected_nets.size()),
        total_nodes,
        total_edges,
        total_sinks};
    TORCH_CHECK(
        item.node_begin >= 0 && item.node_end >= item.node_begin &&
            item.edge_begin >= 0 && item.edge_end >= item.edge_begin &&
            item.sink_begin >= 0 && item.sink_end >= item.sink_begin,
        "packed candidate CSR offsets must be monotonic and non-negative");
    selected_nets.push_back(item);
    total_nodes += item.node_end - item.node_begin;
    total_edges += item.edge_end - item.edge_begin;
    total_sinks += item.sink_end - item.sink_begin;
  }
  const double match_ms = elapsed_ms(match_started_at);

  const auto prefix_started_at = Clock::now();
  std::unordered_map<int64_t, int64_t> selected_position_by_net;
  selected_position_by_net.reserve(selected_nets.size());
  std::vector<int64_t> local_net_ids;
  std::vector<int64_t> source_net_positions;
  std::vector<int64_t> node_starts{0};
  std::vector<int64_t> source_node_index;
  std::vector<int64_t> source_edge_index;
  std::vector<int64_t> source_sink_index;
  local_net_ids.reserve(selected_nets.size());
  source_net_positions.reserve(selected_nets.size());
  source_node_index.reserve(static_cast<size_t>(total_nodes));
  source_edge_index.reserve(static_cast<size_t>(total_edges));
  source_sink_index.reserve(static_cast<size_t>(total_sinks));
  for (int64_t index = 0; index < static_cast<int64_t>(selected_nets.size()); ++index) {
    const SelectedNet& item = selected_nets[static_cast<size_t>(index)];
    selected_position_by_net[item.net_id] = index;
    local_net_ids.push_back(item.net_id);
    source_net_positions.push_back(item.source_position);
    node_starts.push_back(node_starts.back() + item.node_end - item.node_begin);
    for (int64_t value = item.node_begin; value < item.node_end; ++value) {
      source_node_index.push_back(value);
    }
    for (int64_t value = item.edge_begin; value < item.edge_end; ++value) {
      source_edge_index.push_back(value);
    }
    for (int64_t value = item.sink_begin; value < item.sink_end; ++value) {
      source_sink_index.push_back(value);
    }
  }
  std::vector<int64_t> source_candidate_index;
  std::vector<int64_t> local_candidate_node;
  source_candidate_index.reserve(static_cast<size_t>(candidate_net.size()));
  local_candidate_node.reserve(static_cast<size_t>(candidate_net.size()));
  for (int64_t index = 0; index < candidate_net.size(); ++index) {
    const auto selected_it = selected_position_by_net.find(candidate_net[index]);
    if (selected_it == selected_position_by_net.end()) {
      continue;
    }
    const SelectedNet& item =
        selected_nets[static_cast<size_t>(selected_it->second)];
    const int64_t source_node = candidate_node[index];
    TORCH_CHECK(
        source_node >= item.node_begin && source_node < item.node_end,
        "candidate node is outside its packed net range");
    source_candidate_index.push_back(index);
    local_candidate_node.push_back(
        item.local_node_base + source_node - item.node_begin);
  }
  const double prefix_sum_ms = elapsed_ms(prefix_started_at);

  const auto allocate_started_at = Clock::now();
  const auto long_options = at::TensorOptions().device(at::kCPU).dtype(at::kLong);
  at::Tensor output_topo = at::empty({total_nodes}, long_options);
  at::Tensor output_pin_fa = at::full({total_nodes}, -1, long_options);
  at::Tensor output_edge_start = at::empty({total_nodes + 1}, long_options);
  at::Tensor output_flat_pin_to = at::empty({total_edges}, long_options);
  at::Tensor output_sink_node = at::empty({total_sinks}, long_options);
  at::Tensor output_sink_net_index = at::empty({total_sinks}, long_options);
  const double allocate_ms = elapsed_ms(allocate_started_at);

  const auto fill_started_at = Clock::now();
  auto* output_topo_ptr = output_topo.data_ptr<int64_t>();
  auto* output_pin_fa_ptr = output_pin_fa.data_ptr<int64_t>();
  auto* output_edge_start_ptr = output_edge_start.data_ptr<int64_t>();
  auto* output_flat_pin_to_ptr = output_flat_pin_to.data_ptr<int64_t>();
  auto* output_sink_node_ptr = output_sink_node.data_ptr<int64_t>();
  auto* output_sink_net_ptr = output_sink_net_index.data_ptr<int64_t>();
  for (const SelectedNet& item : selected_nets) {
    for (int64_t source_pos = item.node_begin; source_pos < item.node_end; ++source_pos) {
      const int64_t local_pos = item.local_node_base + source_pos - item.node_begin;
      const int64_t source_node = topo[source_pos];
      TORCH_CHECK(
          source_node >= item.node_begin && source_node < item.node_end,
          "candidate topology node is outside its packed net range");
      output_topo_ptr[local_pos] =
          item.local_node_base + source_node - item.node_begin;
    }
    for (int64_t source_node = item.node_begin; source_node < item.node_end; ++source_node) {
      const int64_t local_node = item.local_node_base + source_node - item.node_begin;
      const int64_t source_parent = pin_fa[source_node];
      output_pin_fa_ptr[local_node] = source_parent < 0
          ? -1
          : item.local_node_base + source_parent - item.node_begin;
      output_edge_start_ptr[local_node] =
          item.local_edge_base + edge_start[source_node] - item.edge_begin;
    }
    for (int64_t source_edge = item.edge_begin; source_edge < item.edge_end; ++source_edge) {
      const int64_t source_child = flat_pin_to[source_edge];
      TORCH_CHECK(
          source_child >= item.node_begin && source_child < item.node_end,
          "candidate child node is outside its packed net range");
      output_flat_pin_to_ptr[
          item.local_edge_base + source_edge - item.edge_begin] =
          item.local_node_base + source_child - item.node_begin;
    }
    for (int64_t source_sink = item.sink_begin; source_sink < item.sink_end; ++source_sink) {
      const int64_t source_node = sink_node[source_sink];
      TORCH_CHECK(
          source_node >= item.node_begin && source_node < item.node_end,
          "candidate sink node is outside its packed net range");
      const int64_t local_sink = item.local_sink_base + source_sink - item.sink_begin;
      output_sink_node_ptr[local_sink] =
          item.local_node_base + source_node - item.node_begin;
      output_sink_net_ptr[local_sink] = item.local_net;
    }
  }
  output_edge_start_ptr[total_nodes] = total_edges;
  const double fill_ms = elapsed_ms(fill_started_at);

  const at::Tensor net_index = output_index(source_net_positions);
  const at::Tensor node_index = output_index(source_node_index);
  const at::Tensor sink_index = output_index(source_sink_index);
  const at::Tensor candidate_index = output_index(source_candidate_index);

  py::dict static_view;
  static_view["net_flat_topo_sort"] = output_topo;
  static_view["net_flat_topo_sort_start"] = output_index(node_starts);
  static_view["pin_fa"] = output_pin_fa;
  static_view["flat_pin_to_start"] = output_edge_start;
  static_view["flat_pin_to"] = output_flat_pin_to;
  static_view["edge_resistance"] =
      select(prepared_timing_inputs, "edge_resistance", node_index);
  static_view["node_capacitance"] =
      select(prepared_timing_inputs, "node_capacitance", node_index);
  static_view["edge_capacitance"] =
      select(prepared_timing_inputs, "edge_capacitance", node_index);
  static_view["sink_node_id"] = output_sink_node;
  static_view["sink_net_index"] = output_sink_net_index;

  py::dict metadata;
  metadata["backend"] = "native_cpp";
  metadata["topology_source"] = "native_packed_candidate_active_view_selector";
  metadata["thread_count"] = thread_count;
  metadata["active_net_count"] = static_cast<int64_t>(selected_nets.size());
  metadata["node_count"] = total_nodes;
  metadata["edge_count"] = total_edges;
  metadata["sink_count"] = total_sinks;
  metadata["candidate_count"] = static_cast<int64_t>(source_candidate_index.size());
  metadata["native_active_view_match_count_ms"] = match_ms;
  metadata["native_active_view_prefix_sum_ms"] = prefix_sum_ms;
  metadata["native_active_view_allocate_ms"] = allocate_ms;
  metadata["native_active_view_fill_ms"] = fill_ms;
  metadata["native_active_view_total_ms"] = elapsed_ms(total_started_at);

  py::dict result;
  result["net_ids"] = output_index(local_net_ids);
  result["net_indices_cpu"] = net_index;
  result["candidate_indices_cpu"] = candidate_index;
  result["sink_indices_cpu"] = sink_index;
  result["static"] = static_view;
  result["candidate_node_id_cpu"] = output_index(local_candidate_node);
  result["sink_pin_id_cpu"] =
      select(prepared_timing_inputs, "sink_pin_id", sink_index);
  result["sink_net_id_cpu"] = select(prepared_timing_inputs, "sink_net_index", sink_index);
  result["metadata"] = metadata;
  return result;
}

}  // namespace dreamplace
