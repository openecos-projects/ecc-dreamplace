#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "diff_guided_batch/cpp/action_types.h"

namespace py = pybind11;

namespace dreamplace {
namespace diff_guided_batch {

struct CompactBridgeConflictPrecompute
{
  std::vector<int> component_id_by_endpoint;
  std::vector<int> component_id_by_top_path;
  std::vector<int> component_id_by_net_neighborhood;
  std::vector<int> component_id_by_fanin_fanout_neighborhood;
  std::vector<int> component_id_by_endpoint_dynamic;
  std::vector<int> component_id_by_top_path_dynamic;
  std::vector<double> local_residual_budget_by_top_path_component;
  std::vector<double> local_residual_budget_by_endpoint_component;
  std::string local_residual_snapshot_source;
};

struct CompactBridgeSelection
{
  std::vector<int> selected_queue_indices;
  std::vector<int> remaining_queue_indices;
  std::vector<int64_t> selected_action_ids;
  std::vector<int64_t> remaining_action_ids;
  int conflict_reject_count{0};
  int residual_budget_reject_count{0};
  int residual_budget_score_reorder_count{0};
  std::string residual_budget_source;
  std::string residual_budget_score_source;
};

struct DiffGuidedBatchConflictRuleMask
{
  bool same_instance{true};
  bool same_endpoint{true};
  bool same_top_path{true};
  bool static_component{true};
};

CompactBridgeActionProposal parseActionProposal(const py::dict& action);
CompactBridgeActionResult parseActionResult(const py::dict& action_result);
CompactBridgeActionProposalBuffers parseActionProposalBuffers(const py::dict& buffers);

std::vector<CompactBridgeActionProposal> parseActionProposals(
    const std::vector<py::dict>& actions);
std::vector<CompactBridgeActionResult> parseActionResults(
    const std::vector<py::dict>& action_results);
std::vector<CompactBridgeActionProposal> buildActionProposalsFromBuffers(
    const CompactBridgeActionProposalBuffers& buffers);
std::vector<CompactBridgeActionProposal> parseActionProposalsFromBuffers(
    const py::dict& buffers);

CompactBridgeConflictPrecompute parseConflictPrecompute(
    const py::dict& conflict_precompute,
    const py::dict& dynamic_conflict_signature);

std::vector<CompactBridgeAction> buildCompactActions(
    const std::vector<CompactBridgeActionProposal>& typed_proposal_pool,
    const std::vector<int>& action_indices,
    const CompactBridgeConflictPrecompute& typed_conflict_precompute);
std::vector<CompactBridgeAction> buildCompactActions(
    const std::vector<py::dict>& seed_queue,
    const py::dict& conflict_precompute,
    const py::dict& dynamic_conflict_signature);

std::vector<int> makeSequentialActionIndices(std::size_t action_count);

void assignActionProposalComponents(
    CompactBridgeActionProposal& proposal,
    const CompactBridgeConflictPrecompute& typed_conflict_precompute);
std::vector<CompactBridgeActionProposal> collectActionProposals(
    const std::vector<CompactBridgeActionProposal>& typed_proposal_pool,
    const std::vector<int>& selected_action_indices,
    const CompactBridgeConflictPrecompute& typed_conflict_precompute);

std::vector<std::string> splitConflictRules(const std::string& value);
bool hasConflictRule(const std::vector<std::string>& rules,
                     const std::string& rule);
DiffGuidedBatchConflictRuleMask conflictRuleMask(const py::dict& config);

CompactBridgeSelection selectCompact(
    const std::vector<CompactBridgeAction>& compact_actions,
    int max_batch_size,
    const DiffGuidedBatchConflictRuleMask& conflict_rules,
    const CompactBridgeConflictPrecompute& conflict_precompute,
    const std::string& residual_budget_filter_mode,
    const std::string& residual_budget_score_mode);

int vectorIndexedIntOrDefault(const std::vector<int>& values,
                              int index,
                              int fallback = -1);

template <typename T>
T vectorValueOrDefault(const std::vector<T>& values,
                       std::size_t index,
                       const T& fallback)
{
  return index < values.size() ? values[index] : fallback;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
