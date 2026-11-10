// SPDX-License-Identifier: BSD-3-Clause
// Native joint tree-level VGDP diagnostic slice.

#pragma once

#include <string>
#include <vector>

#include "rebuffer_driver_tree.hh"
#include "rebuffer_liberty.hh"
#include "rebuffer_repair.hh"
#include "rebuffer_types.hh"

namespace rebuffer {

struct TreeVgdpTrace
{
  int trace_id = -1;
  int node_id = -1;
  std::string operation;
  double cap = 0.0;
  double slack = 0.0;
  double slew_rise = 0.0;
  double slew_fall = 0.0;
  double area = 0.0;
  int buffer_count = 0;
  bool survivor = false;
  std::string reason;
};

struct TreeVgdpCandidate
{
  std::vector<BufferAction> actions;
  double cap = 0.0;
  double slack = 0.0;
  double slew_rise = 0.0;
  double slew_fall = 0.0;
  double area = 0.0;
  int buffer_count = 0;
};

struct TreeVgdpResult
{
  double driver_resistance = 0.0;
  double current_driver_load = 0.0;
  std::vector<TreeVgdpCandidate> candidates;
  std::vector<TreeVgdpTrace> trace;
  int generated_option_count = 0;
  int surviving_option_count = 0;
  int pruned_option_count = 0;
  int tree_node_count = 0;
  int tree_edge_count = 0;
  int tree_branch_count = 0;
  int tree_max_depth = 0;
  int long_wire_edge_count = 0;
  int wire_site_count = 0;
  int electrical_rejected_count = 0;
  int timing_round_count = 0;
  int recovery_attempt_count = 0;
  int recovered_buffer_count = 0;
  std::vector<std::string> round_source_tree_digests;
  std::vector<int> round_source_buffer_counts;
};

bool planCurrentDriverTree(const DesignView& design,
                           const TimingView& timing,
                           const LibertyView& liberty,
                           const CharacterizationResult& characterization,
                           const RepairConfig& config,
                           const CurrentDriverTree& tree,
                           TreeVgdpResult& result,
                           std::string& reason,
                           bool record_trace = true);

}  // namespace rebuffer
