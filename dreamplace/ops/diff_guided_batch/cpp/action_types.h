#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dreamplace {
namespace diff_guided_batch {

enum class ActionKind : int {
  kSizing = 0,
  kBufferInsert = 1,
  kBufferRemove = 2,
  kPinSwap = 3,
  kUnknown = 4,
};

inline ActionKind actionKindFromString(const std::string& value)
{
  if (value == "sizing") {
    return ActionKind::kSizing;
  }
  if (value == "buffer_insert") {
    return ActionKind::kBufferInsert;
  }
  if (value == "buffer_remove") {
    return ActionKind::kBufferRemove;
  }
  if (value == "pin_swap") {
    return ActionKind::kPinSwap;
  }
  return ActionKind::kUnknown;
}

inline std::string actionKindName(ActionKind kind)
{
  switch (kind) {
    case ActionKind::kSizing:
      return "sizing";
    case ActionKind::kBufferInsert:
      return "buffer_insert";
    case ActionKind::kBufferRemove:
      return "buffer_remove";
    case ActionKind::kPinSwap:
      return "pin_swap";
    case ActionKind::kUnknown:
      return "unknown";
  }
  return "unknown";
}

struct CompactBridgeAction
{
  int queue_index{-1};
  int64_t action_index{-1};
  int inst_id{-1};
  int endpoint_component{-1};
  int top_path_component{-1};
  int net_component{-1};
  int fanin_fanout_component{-1};
  double predicted_delta_obj{0.0};
  double predicted_improvement{0.0};
};

struct CompactBridgeActionProposal
{
  ActionKind kind{ActionKind::kSizing};
  int64_t action_index{-1};
  int inst_id{-1};
  int endpoint_component{-1};
  int top_path_component{-1};
  int affected_net_id{-1};
  int driver_pin_id{-1};
  int load_pin_id{-1};
  int buffer_master_id{-1};
  int current_size_idx{-1};
  int target_size_idx{-1};
  int seed_size_idx{-1};
  int seed_step{0};
  std::string seed_direction;
  std::string target_master;
  std::string seed_master;
  std::string buffer_master_name;
  std::vector<std::string> legal_master_candidates;
  std::vector<int> legal_cell_id_candidates;
  std::vector<double> legal_timing_coordinate_candidates;
  double predicted_delta_obj{0.0};
  double predicted_improvement{0.0};
  double sensitivity_score{0.0};
  double old_timing_coordinate{0.0};
  double new_timing_coordinate{0.0};
  double local_delta_delay_ps{0.0};
  double local_delta_slew_ps{0.0};
  double local_delta_cap{0.0};
  std::string local_delta_source;
  std::string estimator_source;
  double candidate_location_x{0.0};
  double candidate_location_y{0.0};
};

struct CompactBridgeActionResult
{
  ActionKind kind{ActionKind::kSizing};
  int64_t action_index{-1};
  int inst_id{-1};
  bool accepted{false};
  bool supported{false};
  std::string status;
  std::string reject_reason;
  std::string old_master;
  std::string new_master;
  std::string target_master;
  int old_master_id{-1};
  int new_master_id{-1};
  int current_size_idx{-1};
  int target_size_idx{-1};
  int best_trial_step{0};
  double old_timing_coordinate{0.0};
  double new_timing_coordinate{0.0};
  double actual_delta_tns{0.0};
  double actual_delta_wns{0.0};
  double actual_delta_obj{0.0};
  double local_predicted_net_slack_delta{0.0};
  double local_predicted_weighted_net_tns_delta{0.0};
  std::string local_predicted_slack_delta_source;
  std::string local_predicted_slack_delta_weight_mode;
};

struct CompactBridgeActionProposalBuffers
{
  std::vector<int64_t> action_ids;
  std::vector<int> action_kind_ids;
  std::vector<std::string> action_kinds;
  std::vector<std::string> string_table;
  std::vector<int> primary_inst_ids;
  std::vector<int> current_size_idxs;
  std::vector<int> target_size_idxs;
  std::vector<int> seed_size_idxs;
  std::vector<int> seed_steps;
  std::vector<std::string> seed_directions;
  std::vector<int> seed_direction_ids;
  std::vector<std::string> target_masters;
  std::vector<int> target_master_name_ids;
  std::vector<std::string> seed_masters;
  std::vector<int> seed_master_name_ids;
  std::vector<int> legal_master_candidate_indptr;
  std::vector<std::string> legal_master_candidate_names;
  std::vector<int> legal_master_candidate_name_ids;
  std::vector<int> legal_cell_id_candidate_ids;
  std::vector<double> legal_timing_coordinate_candidates;
  std::vector<double> predicted_delta_objs;
  std::vector<double> predicted_improvements;
  std::vector<double> sensitivity_scores;
  std::vector<double> old_timing_coordinates;
  std::vector<double> new_timing_coordinates;
  std::vector<double> local_delta_delay_ps;
  std::vector<double> local_delta_slew_ps;
  std::vector<double> local_delta_caps;
  std::vector<std::string> local_delta_sources;
  std::vector<std::string> estimator_sources;
  std::vector<int> affected_net_ids;
  std::vector<int> driver_pin_ids;
  std::vector<int> load_pin_ids;
  std::vector<int> buffer_master_ids;
  std::vector<std::string> buffer_master_names;
  std::vector<int> buffer_master_name_ids;
  std::vector<double> candidate_location_xs;
  std::vector<double> candidate_location_ys;
};

}  // namespace diff_guided_batch
}  // namespace dreamplace
