#pragma once

#include <cstdint>
#include <tuple>
#include <vector>
#include <unordered_map>

#include "utility/src/torch.h"

namespace dreamplace {

struct TraversalPruningRefreshResult {
  at::Tensor kept_flat_arc_indices;
  at::Tensor kept_level_offsets;
  at::Tensor kept_counts_by_level;
  int64_t active_inst_count;
  int64_t active_arc_count;
  int64_t dropped_arc_count;
  int64_t parallel_task_count;
  double preparation_runtime_ms;
};

struct EndpointIncidenceResult {
  at::Tensor pin_endpoint_incidence_count;
  at::Tensor arc_endpoint_incidence_count;
  int64_t active_endpoint_count;
  int64_t pin_incidence_nonzero_count;
  int64_t arc_incidence_nonzero_count;
  double incidence_runtime_ms;
};

struct CriticalPathBatch {
  at::Tensor path_offsets;
  at::Tensor path_pins;
  at::Tensor path_transitions;
  at::Tensor path_arc_ids;
  at::Tensor endpoint_pins;
  at::Tensor endpoint_test_ids;
  at::Tensor endpoint_transitions;
  at::Tensor endpoint_slacks;
  at::Tensor path_valid;
  at::Tensor invalid_reason;
  at::Tensor max_residual_ps;
  int64_t failing_state_count;
  int64_t selected_state_count;
  int64_t valid_path_count;
  int64_t invalid_path_count;
  int64_t topology_epoch;
  double extraction_runtime_ms;
};

class SetupCriticalPathExtractor {
 public:
  SetupCriticalPathExtractor(
      const at::Tensor& flat_inst_arcs_by_level,
      const at::Tensor& pin_pred_start,
      const at::Tensor& pin_pred_pin,
      const at::Tensor& pin_pred_arc_id,
      const at::Tensor& start_points,
      int64_t topology_epoch);

  CriticalPathBatch extract(
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
      double residual_tolerance_ps) const;

  int64_t topology_epoch() const { return topology_epoch_; }
  int64_t num_pins() const { return num_pins_; }
  int64_t num_cell_arcs() const { return num_cell_arcs_; }

 private:
  int64_t topology_epoch_ = 0;
  int64_t num_pins_ = 0;
  int64_t num_cell_arcs_ = 0;
  std::vector<int64_t> predecessor_offsets_;
  std::vector<int64_t> predecessor_pins_;
  std::vector<int64_t> predecessor_arc_ids_;
  std::vector<int64_t> arc_senses_;
  std::vector<uint8_t> is_start_pin_;
};

class CriticalEndpointTraversalPruner {
 public:
  CriticalEndpointTraversalPruner(
      const at::Tensor& flat_inst_arcs_by_level,
      const at::Tensor& flat_inst_arcs_by_level_start,
      const at::Tensor& flat_pin_to_graph_reverse,
      const at::Tensor& flat_pin_to_graph_start_reverse,
      const at::Tensor& pin_pair_arc_keys,
      const at::Tensor& flat_pin_pair_arc_start,
      const at::Tensor& flat_pin_pair_arc_indices,
      const at::Tensor& start_points,
      const at::Tensor& pin2node_map);

  CriticalEndpointTraversalPruner(
      const at::Tensor& flat_inst_arcs_by_level,
      const at::Tensor& flat_inst_arcs_by_level_start,
      const at::Tensor& pin_pred_start,
      const at::Tensor& pin_pred_pin,
      const at::Tensor& pin_pred_arc_id,
      const at::Tensor& start_points,
      const at::Tensor& pin2node_map);

  TraversalPruningRefreshResult refresh(
      const at::Tensor& active_endpoint_ids) const;

  EndpointIncidenceResult endpoint_incidence(
      const at::Tensor& active_endpoint_ids) const;

  int64_t num_flat_arcs() const { return num_flat_arcs_; }
  int64_t num_levels() const { return num_levels_; }

 private:
  int64_t num_flat_arcs_ = 0;
  int64_t num_levels_ = 0;
  int64_t num_pins_ = 0;
  std::vector<int64_t> reverse_offsets_;
  std::vector<int64_t> reverse_edges_;
  std::vector<int64_t> reverse_edge_arc_ids_;
  std::vector<int64_t> level_offsets_;
  std::vector<int64_t> flat_arc_level_id_;
  std::vector<int64_t> flat_arc_to_inst_id_;
  std::vector<uint8_t> is_start_pin_;
  std::unordered_map<uint64_t, std::vector<int64_t>> pair_to_arc_indices_;

  void initialize_flat_arcs_and_levels(
      const at::Tensor& flat_inst_arcs_by_level,
      const at::Tensor& flat_inst_arcs_by_level_start,
      const at::Tensor& pin2node_map);

  void initialize_start_points(const at::Tensor& start_points);
};

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
    double slack_epsilon);

} // namespace dreamplace
