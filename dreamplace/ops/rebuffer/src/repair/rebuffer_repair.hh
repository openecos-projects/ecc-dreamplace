// SPDX-License-Identifier: BSD-3-Clause
// Native repair_design planner data contracts.

#pragma once

#include <string>
#include <vector>

#include "rebuffer_liberty.hh"
#include "rebuffer_types.hh"

namespace rebuffer {

struct TimingView
{
  std::vector<double> pin_slew_rise;
  std::vector<double> pin_slew_fall;
  std::vector<double> pin_load;
  std::vector<double> pin_cap;
  std::vector<double> pin_max_slew;
  std::vector<double> pin_max_cap;
  std::vector<double> pin_slack;
  // Python TimingPropagation snapshots for root-relative load annotation.
  std::vector<double> pin_arrival_rise;
  std::vector<double> pin_arrival_fall;
  std::vector<double> pin_required_rise;
  std::vector<double> pin_required_fall;
  std::vector<int> clock_pins;
  // Optional Python TimingPropagation topological level for each physical
  // pin.  repair_design consumes this only when its reverse-topological mode
  // is explicitly enabled; the default planner keeps net-id order.
  std::vector<int> pin_rank;
  // Optional path-specific context supplied by the Python native path
  // extractor.  A value below zero means that the planner should use the
  // legacy all-arc fallback.
  int path_driver_input_pin = -1;
  int path_driver_input_transition = -1;
  int path_driver_output_transition = -1;
  int path_driver_lib_arc_id = -1;
  // Optional path-expanded timing data used by the native OpenROAD-style
  // driver ranking.  The vectors are indexed by path edge/pin position:
  // path_arc_delays has one entry per path edge, path_arc_local_ids identifies
  // the selected cell arc (or -1 for a net edge), and path_transitions has one
  // entry per path pin.
  std::vector<double> path_arc_delays;
  std::vector<int> path_arc_local_ids;
  std::vector<int> path_transitions;
  // Current Python TimingPropagation arrivals for the selected path arc.
  // OpenROAD uses these values to correct the bottom-up BufferedNet model at
  // the root. They are optional for non-path repair_design calls.
  double path_driver_input_arrival = 0.0;
  double path_driver_output_arrival = 0.0;
  bool path_driver_arrival_valid = false;
};

struct SteinerView
{
  int steiner_node_base = 0;
  std::vector<int> net_ids;
  std::vector<int> net_flat_topo_sort_start;
  std::vector<int> net_flat_topo_sort;
  std::vector<int> pin_fa;
  std::vector<int> flat_pin_to_start;
  std::vector<int> flat_pin_to;
  std::vector<double> node_x;
  std::vector<double> node_y;
  std::vector<double> edge_resistance;
  std::vector<double> node_capacitance;
};

struct RepairConfig
{
  double slew_margin = 0.0;
  double cap_margin = 0.0;
  double r_unit = 0.914;
  double c_unit = 1.89e-4;
  int max_fanout = 32;
  // Per-net safety bound for the current-driver-tree DP.  repair_design
  // itself has no global action budget: OpenROAD visits every driver vertex
  // and never truncates the plan by a total action count.
  int tree_max_buffers = 1024;
  bool insert_buffers = true;
  bool resize_drivers = true;
  double resize_slew_threshold = 0.03;
  // Low-level defaults stay off so direct native callers keep their existing
  // contract.  The production placement profile enables the OpenROAD order,
  // local state update and load-slew annotation through DEFAULT_CONFIG.
  bool repair_design_reverse_topological = false;
  bool repair_design_local_update = false;
  bool repair_design_load_slew_annotation = false;
  bool repair_design_openroad_driver_resize = false;
  bool repair_design_fanout_repair = false;
  // OpenROAD placement-parasitics repair design normalizes every logic
  // driver to its target load before checking slew/cap violations
  // (Resizer::resizeToTargetSlew / RepairDesign::repairNet).  The low-level
  // native default stays off so direct callers keep the legacy contract;
  // the production placement profile enables it.
  bool repair_design_resize_to_target_slew = false;
  // Isolate the canonical per-arc target load and target-slew delay math.
  // Candidate eligibility, traversal and electrical filters remain legacy.
  bool repair_design_target_cell_math = false;
  // Model OpenROAD makeRepeater(resize=true) by selecting the repeater from
  // the actual post-split downstream load rather than the pre-split tree load.
  // This is deliberately opt-in until the physical commit path is qualified.
  bool repair_design_target_cell_selection = false;
  bool repair_design_openroad_repeater_resize = false;
  double sizing_area_limit = 4.0;
  double sizing_leakage_limit = 4.0;
  // Maximum driver-rooted wire length in microns.  Zero disables the check.
  double repair_design_max_wire_length_um = 0.0;
  bool repair_net_tree = false;
  double slew_rc_factor = 1.864;
  // OpenROAD's Elmore transition shape factor.  The legacy repair_design
  // spelling remains supported; config parsing makes this inherit it unless
  // an explicit slew_shape_factor is supplied.
  double slew_shape_factor = 1.864;
  double min_slack_improvement = 0.0;
  // OpenROAD BufferMove's ordinary repair_timing eligibility.  This is a
  // search-domain rule, separate from max_fanout, which is an electrical
  // design constraint used by repair_design.
  bool enforce_timing_buffer_move_fanout = false;
  // OpenROAD BufferGenerator accepts target fanout in (0, 33).
  int timing_max_fanout = 33;
  // Emit native candidate diagnostics only for an explicit trace run.  The
  // production planner keeps this disabled so candidate serialization does
  // not change the normal action contract or runtime.
  bool emit_candidate_trace = false;
  // Include the validated one-hop downstream cell delay relief in the root
  // score.  Keep this opt-in until BM64 candidate ranking is qualified
  // against Python TimingPropagation; the diagnostic itself is independent
  // of this switch.
  bool use_downstream_slew_delay_bonus = false;
  // Number of non-primary root tree options exported to the caller.  The
  // strict production profile keeps the historical value of four; larger
  // values are an explicit candidate-space diagnostic/qualification knob.
  int max_timing_alternatives = 4;
  // Current-tree search uses fixed RC ERC. Disable only for algebra probes.
  bool tree_enforce_electrical = true;
  double tree_max_wire_length_um = 0.0;
  int tree_timing_rounds = 3;
  int tree_recovery_rounds = 5;
};

struct BufferAction
{
  int net_id = -1;
  int driver_pin = -1;
  std::vector<int> downstream_pins;
  int buffer_cell_index = -1;
  // Segment chosen by the native DP.  ``-1`` means the producer did not
  // expose an explicit segment (for example the single-net rebufferPin path).
  int parent_node_id = -1;
  int child_node_id = -1;
  // Stable physical topology identities for a current-driver-tree action.
  // The graph ids above belong to the native candidate tree and may change
  // when the candidate is materialized.  These ids refer to the imported
  // pydb/Steiner topology and are the ids consumed by the physical overlay.
  // Other repair producers leave them at -1.
  int source_parent_node_id = -1;
  int source_child_node_id = -1;
  // Depth in the buffer tree.  The native apply_buffer_actions sorts by this
  // field and allows overlapping load partitions for depth >= 0.
  int tree_depth = -1;
  int x_dbu = 0;
  int y_dbu = 0;
  double split_ratio = -1.0;
  double downstream_load = 0.0;
  double target_load = 0.0;
  // The source of the insertion resize, before findTargetCell runs again.
  // Export only for the opt-in target-cell selector. Repeating selection
  // from the final cell is a different operation even at the same load.
  int initial_buffer_cell_index = -1;
  double initial_downstream_load = 0.0;
  std::string reason;
};

struct ResizeAction
{
  int node_id = -1;
  int target_cell_index = -1;
};

struct RepairDesignResult
{
  std::vector<BufferAction> actions;
  std::vector<ResizeAction> resizes;
  int driver_slew_violations = 0;
  int driver_cap_violations = 0;
  int load_slew_violations = 0;
  int load_cap_violations = 0;
  int repaired_nets = 0;
};

struct SubtreeLoad
{
  double load = 0.0;
  std::vector<int> sinks;
};

int rebufferLibPinIndex(const DesignView& design, int pin);
int rebufferCellIndexForNode(const DesignView& design, int node);
bool rebufferIsClockPin(const TimingView& timing, int pin);
bool rebufferBuildNetChildren(const SteinerView& steiner,
                              int net_id,
                              std::vector<std::vector<int>>& children,
                              std::string& reason);
bool rebufferValidateLengths(const DesignView& design,
                             const TimingView& timing,
                             const SteinerView& steiner,
                             std::string& reason);
SubtreeLoad rebufferSubtreeLoad(const DesignView& design,
                                const TimingView& timing,
                                const SteinerView& steiner,
                                const RepairConfig& config,
                                const std::vector<std::vector<int>>& children,
                                int node,
                                int driver);
double rebufferEdgeCapacitance(const SteinerView& steiner,
                               const RepairConfig& config,
                               int node);
double rebufferCellOutputSlew(const LibertyCellView& cell,
                              double input_slew_rise,
                              double input_slew_fall,
                              double load_cap);
double rebufferCellOutputSlewRise(const LibertyCellView& cell,
                                  double input_slew_rise,
                                  double load_cap);
double rebufferCellOutputSlewFall(const LibertyCellView& cell,
                                  double input_slew_fall,
                                  double load_cap);
double rebufferCellOutputDelay(const LibertyCellView& cell,
                               double input_slew_rise,
                               double input_slew_fall,
                               double load_cap);
bool rebufferDriverInputSlew(const DesignView& design,
                             const TimingView& timing,
                             const LibertyView& liberty,
                             int driver,
                             double& input_slew_rise,
                             double& input_slew_fall);
double rebufferTargetLoadForCell(const LibertyCellView& cell,
                                 double target_slew_rise,
                                 double target_slew_fall,
                                 bool target_cell_math = false);
int rebufferChooseBufferCell(const LibertyView& liberty,
                             const CharacterizationResult& characterization,
                             double input_slew_rise,
                             double input_slew_fall,
                             double downstream_load,
                             bool target_cell_math = false,
                             bool target_cell_selection = false,
                             int initial_cell = -1,
                             double sizing_area_limit = 4.0,
                             double sizing_leakage_limit = 4.0);
struct RepeaterCellSelection
{
  int cell_index = -1;
  int initial_cell_index = -1;
};

RepeaterCellSelection rebufferChooseRepeaterCell(const LibertyView& liberty,
                               const CharacterizationResult& characterization,
                               double input_slew_rise, double input_slew_fall,
                               double initial_load, double repeater_load,
                               const RepairConfig& config,
                               bool resize = true);

bool repairNetTree(const DesignView& design,
                   const TimingView& timing,
                   const SteinerView& steiner,
                   const LibertyView& liberty,
                   const CharacterizationResult& characterization,
                   const RepairConfig& config,
                   int net_id,
                   double max_cap,
                   int driver,
                   int driver_cell_index,
                   const std::vector<std::vector<int>>& children,
                   std::vector<BufferAction>& actions,
                   std::string& reason,
                   double* root_cap_out = nullptr);

bool planRepairDesign(const DesignView& design,
                      const TimingView& timing,
                      const SteinerView& steiner,
                      const LibertyView& liberty,
                      const CharacterizationResult& characterization,
                      const RepairConfig& config,
                      RepairDesignResult& result,
                      std::string& reason);

}  // namespace rebuffer
