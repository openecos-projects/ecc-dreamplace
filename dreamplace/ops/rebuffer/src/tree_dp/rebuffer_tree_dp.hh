// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Native joint tree-level VGDP diagnostic slice.


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

// --- rebuffer_tree_option.hh ---
#include <limits>

namespace rebuffer {

// Electrical state stops at the next buffer input. Physical terminal loads
// remain in loads for projecting the complete action bundle into placeio_ecc.
struct TreeOption
{
  double cap = 0.0;
  double slack = 0.0;
  double slew_rise = 0.0;
  double slew_fall = 0.0;
  double area = 0.0;
  int buffer_count = 0;
  int node_id = -1;
  double max_load_slew = std::numeric_limits<double>::infinity();
  double max_load_wire = 0.0;
  int fanout = 0;
  std::vector<int> loads;
  std::vector<BufferAction> actions;
};

void pruneTreeOptions(std::vector<TreeOption>& options, int node_id,
                      TreeVgdpResult& result, int& next_trace_id);

}  // namespace rebuffer

// --- rebuffer_tree_annotation.hh ---

namespace rebuffer {

struct TreeRootTiming
{
  double resistance = 0.0;
  double current_load = 0.0;
  double output_cap = 0.0;
};

void appendTreeLoads(std::vector<int>& target, const std::vector<int>& source);
void annotateTreeLoads(const DriverTreeNode& node, const TimingView& timing,
                       int driver, TreeOption& option);
TreeRootTiming annotateTreeRoot(const DesignView& design,
                                const TimingView& timing,
                                const LibertyView& liberty, int driver);
double treeRootCorrection(const TreeRootTiming& root, double cap);

}  // namespace rebuffer

// --- rebuffer_tree_electrical.hh ---

namespace rebuffer {

double treeMarginedLimit(double limit, double margin);
// Fixed RC/drive models only. Nonlinear timing is evaluated by Python TP
// after the complete candidate has been applied to the overlay.
bool treeCanDrive(const TreeOption& option, double resistance,
                  double output_cap, double max_cap, const RepairConfig& config);
bool treeRootFeasible(const TreeOption& option, const TreeRootTiming& root,
                      const TimingView& timing, int driver,
                      const RepairConfig& config);
double treeWireStep(const LibertyView& liberty,
                    const CharacterizationResult& characterization,
                    const RepairConfig& config);

}  // namespace rebuffer

// --- rebuffer_wire_candidates.hh ---

namespace rebuffer {

std::vector<TreeOption> treeWireCandidates(
    const DesignView& design, const LibertyView& liberty,
    const CharacterizationResult& characterization, const RepairConfig& config,
    const CurrentDriverTree& tree, const DriverTreeEdge& edge, int depth,
    const std::vector<TreeOption>& children, TreeVgdpResult& result,
    int& trace_id, std::string& reason);

}  // namespace rebuffer

// --- rebuffer_tree_search.hh ---
#include <map>


namespace rebuffer {

// OpenROAD-style local tree rounds. Each round consumes the previous modeled
// tree; the outer queue evaluates the final complete bundle with Python TP.
bool optimizeCurrentDriverTree(
    const DesignView& design, const TimingView& timing, const LibertyView& liberty,
    const CharacterizationResult& characterization, const RepairConfig& config,
    const CurrentDriverTree& source, TreeVgdpResult& result, std::string& reason,
    std::map<std::string, double>* phase_timings = nullptr,
    bool record_trace = true,
    bool prune_worse_than_unbuffered = false);

}  // namespace rebuffer

// --- rebuffer_tree_bindings.hh ---
#include "pybind11/pybind11.h"

namespace rebuffer {
void registerTreeSearchBindings(pybind11::module_& module);
}
