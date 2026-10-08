#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dreamplace {
namespace diff_guided_batch {

enum class ActionKind : int32_t {
  kSizing = 0,
  kBufferInsert = 1,
  kBufferRemove = 2,
  kPinSwap = 3,
};

inline const char* actionKindName(ActionKind kind) {
  switch (kind) {
    case ActionKind::kSizing:
      return "sizing";
    case ActionKind::kBufferInsert:
      return "buffer_insert";
    case ActionKind::kBufferRemove:
      return "buffer_remove";
    case ActionKind::kPinSwap:
      return "pin_swap";
  }
  return "unknown";
}

struct SizingPayload {
  int32_t inst_id = -1;
  int32_t current_master_id = -1;
  int32_t target_master_id = -1;
  int32_t current_size_idx = -1;
  int32_t target_size_idx = -1;
  int32_t size_idx_delta = 0;
};

struct BufferPayload {
  int32_t net_id = -1;
  int32_t driver_pin_id = -1;
  int32_t load_pin_id = -1;
  int32_t buffer_master_id = -1;
  double x = 0.0;
  double y = 0.0;
};

struct ActionProposal {
  int64_t action_id = -1;
  ActionKind action_kind = ActionKind::kSizing;
  double predicted_delta_obj = 0.0;
  double sensitivity_score = 0.0;
  SizingPayload sizing;
  BufferPayload buffer;
};

struct ActionConflictFootprint {
  int64_t action_id = -1;
  ActionKind action_kind = ActionKind::kSizing;
  int32_t primary_inst_id = -1;
  std::vector<int32_t> affected_instance_ids;
  std::vector<int32_t> affected_net_ids;
  std::vector<int32_t> affected_pin_ids;
  std::vector<int32_t> endpoint_component_ids;
  std::vector<int32_t> top_path_component_ids;
  std::vector<int32_t> static_component_ids;
  std::vector<int32_t> physical_bin_ids;
};

struct ActionResult {
  int64_t action_id = -1;
  ActionKind action_kind = ActionKind::kSizing;
  bool supported = false;
  bool accepted = false;
  std::string status;
  std::string reject_reason;
  double actual_delta_tns = 0.0;
  double actual_delta_wns = 0.0;
};

inline bool isSupportedV1(ActionKind kind) {
  return kind == ActionKind::kSizing;
}

inline ActionResult unsupportedResult(const ActionProposal& proposal) {
  ActionResult result;
  result.action_id = proposal.action_id;
  result.action_kind = proposal.action_kind;
  result.supported = false;
  result.accepted = false;
  result.status = "unsupported";
  result.reject_reason = std::string(actionKindName(proposal.action_kind)) +
                         " is reserved and disabled in v1";
  return result;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
