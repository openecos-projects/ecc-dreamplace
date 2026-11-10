// SPDX-License-Identifier: BSD-3-Clause
// Incremental event-driven STA and exact phase-loss gate for discrete sizing.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lut_query.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace dreamplace {

class IncrementalSizingEngine
{
 public:
  explicit IncrementalSizingEngine(const pybind11::dict& config);

  void syncState(
      const pybind11::array_t<int64_t, pybind11::array::c_style | pybind11::array::forcecast>&
          inst_cell_ids,
      const pybind11::object& wire_cap = pybind11::none(),
      const pybind11::object& incoming_resistance = pybind11::none());

  pybind11::dict evaluateRankedCandidates(
      const std::vector<int>& instance_ids,
      const std::vector<int>& target_cell_ids,
      const pybind11::dict& policy);

  pybind11::dict currentMetrics() const;

 private:
  struct LutStorage
  {
    std::vector<double> values, transitions, capacitances;
    std::vector<int64_t> transition_dims, capacitance_dims;
    int rows = 0;
    int value_stride = 0, transition_stride = 0, capacitance_stride = 0;
    TimingLutTable<double> view() const;
  };

  struct StaticRcForest
  {
    int vertex_count = 0;
    std::vector<double> wire_cap;
    std::vector<double> incoming_resistance;
    std::vector<int> parent;
    std::vector<int> child_start;
    std::vector<int> children;
    std::vector<int> topo;
    std::vector<int> topo_start;
    std::vector<int> tree_by_net;
    std::vector<int> pin_to_vertex;

    mutable std::vector<double> scratch_cap;
    mutable std::vector<double> scratch_load;
    mutable std::vector<double> scratch_delay;
    mutable std::vector<double> scratch_ldelay;
    mutable std::vector<double> scratch_beta;
  };

  struct IncomingEdge
  {
    enum class Kind : uint8_t { kClk2q = 0, kCell = 1, kSlewOnlyCell = 2, kNet = 3 };
    Kind kind = Kind::kCell;
    int arc_row = -1;
    int src_state = -1;
    int dst_state = -1;
    int mask = 15;
    std::array<double, 4> net_adj{0.0, 0.0, 1.0, 1.0};
  };

  struct SparseCheckRow
  {
    int state = -1;
    int check_arc_row = -1;
    int transition = 0;
    int endpoint_idx = -1;
    double rule_offset = 0.0;
    double rule_sign = 0.0;
    double clock_slew = 0.0;
  };

  struct UndoCheckpoint
  {
    std::size_t cell_pos = 0;
    std::size_t cell_arc_pos = 0;
    std::size_t setup_arc_pos = 0;
    std::size_t pin_val_pos = 0;
    std::size_t pin_rc_pos = 0;
    std::size_t state_arr_pos = 0;
    std::size_t state_slew_pos = 0;
    std::size_t group_slew_pos = 0;
    std::size_t pin_slew_pos = 0;
    std::size_t endpoint_pos = 0;
  };

  using ArcRow = std::array<int64_t, 8>;
  using LibArcSig = std::array<int64_t, 4>;  // src_sig, dst_sig, sense, trigger

  int mapLibertyArc(int source_cell, int target_cell, int old_arc_id) const;
  void recomputeNetRc(int net_id, bool record_undo, std::vector<int>* changed_rc_pins);
  void propagateFull();
  void propagateEvents(const std::vector<int>& seed_states,
                       std::vector<int>* changed_slew_pins,
                       std::vector<int>* touched_endpoints);
  bool recomputeState(int state_id, bool record_undo, bool* local_slew_changed);
  bool recomputeSlewGroup(int group_id, bool record_undo,
                          std::vector<int>* changed_slew_pins);
  std::pair<double, double> sourceSlewForEdge(int src_state, int dst_level) const;
  double computeEndpointSlack(int endpoint_idx) const;
  UndoCheckpoint captureUndo() const;
  void restoreUndo(const UndoCheckpoint& cp);
  void clearUndo();

  int num_pins_ = 0;
  int num_nets_ = 0;
  int num_instances_ = 0;
  int num_cells_ = 0;
  int clock_end_ = 0;
  bool sparse_mode_ = false;
  bool smooth_aggregation_ = false;
  double aggregation_tau_ps_ = 2.0;
  bool pairwise_smooth_ = false;
  double pairwise_alpha_ = 20.0;
  double slew_epsilon_ = 1e-8;
  bool recovery_uses_live_slew_ = true;
  bool recovery_rat_is_capture_deadline_ = false;

  // Library & instance topology
  std::vector<int> pin_owner_;
  std::vector<int> pin_net_;
  std::vector<int> pin_lib_offset_;
  std::vector<std::vector<int>> net_pins_;
  std::vector<std::vector<int>> node_pins_;
  std::vector<int> inst_cell_id_;
  std::vector<int64_t> cell_arc_start_;
  std::vector<int64_t> cell_pin_start_;
  std::vector<LibArcSig> lib_arc_sigs_;
  std::array<std::vector<double>, 5> lib_pin_values_;
  std::array<std::vector<double>, 5> pin_values_;  // cap, rcap, fcap, max_slew, max_cap

  // Mutable arc tables matching runtime_timing_arcs.py
  std::vector<ArcRow> cell_arcs_;
  std::vector<ArcRow> setup_arcs_;
  std::vector<ArcRow> check_arcs_;
  std::vector<std::vector<int>> node_cell_arcs_;
  std::vector<std::vector<int>> node_setup_arcs_;
  std::array<LutStorage, 6> luts_;

  // Static RC forest
  StaticRcForest rc_forest_;
  std::vector<double> pin_load_r_, pin_load_f_;
  std::vector<double> pin_delay_r_, pin_delay_f_;
  std::vector<double> pin_impulse_r_, pin_impulse_f_;

  // Forward state DAG (unified across Schema 3 sparse and Schema 2 flat)
  int num_states_ = 0;
  int num_slew_groups_ = 0;
  int max_level_ = 0;
  std::vector<int> state_pin_;
  std::vector<int> state_slew_group_;
  std::vector<int> state_level_;
  std::vector<std::array<uint8_t, 2>> state_valid_;
  std::vector<std::array<uint8_t, 2>> state_seed_;
  std::vector<std::array<double, 4>> state_reference_;
  std::vector<uint8_t> group_causal_;

  std::vector<std::vector<IncomingEdge>> state_in_edges_;
  std::vector<std::vector<int>> state_out_states_;
  std::vector<std::vector<int>> group_states_;
  std::vector<std::vector<int>> group_out_states_;
  std::vector<std::vector<int>> pin_states_;
  std::vector<std::vector<int>> pin_net_sink_states_;
  std::vector<std::vector<int>> cell_arc_dst_states_;

  // Live timing state
  std::vector<double> state_ra_, state_fa_;
  std::vector<double> state_local_rs_, state_local_fs_;
  std::vector<double> group_rs_, group_fs_;
  std::vector<double> pin_rs_, pin_fs_;

  // Endpoint reduction
  int num_endpoints_ = 0;
  std::vector<int> endpoint_pins_;
  std::vector<double> endpoint_slacks_;

  // Schema 3 endpoint structures
  std::vector<SparseCheckRow> sparse_checks_;
  std::vector<std::vector<int>> endpoint_sparse_checks_;
  std::vector<std::vector<int>> state_endpoint_indices_;
  std::vector<std::vector<int>> group_endpoint_indices_;

  // Schema 2 flat clock & endpoint constraint structures
  std::vector<double> clock_rtran_, clock_ftran_;
  std::vector<double> flat_endpoint_rRAT_, flat_endpoint_fRAT_;
  std::vector<std::array<uint8_t, 2>> flat_endpoint_valid_;
  std::vector<std::array<uint8_t, 2>> setup_valid_;
  std::vector<std::vector<int>> endpoint_setup_rows_;
  std::vector<ArcRow> recovery_arcs_;
  std::vector<std::array<uint8_t, 2>> recovery_valid_;
  std::vector<double> recovery_clock_rtran_, recovery_clock_ftran_;
  std::vector<double> recovery_ref_rtran_, recovery_ref_ftran_;
  std::vector<double> recovery_deadline_r_, recovery_deadline_f_;
  std::vector<std::array<uint8_t, 2>> recovery_keep_setup_;
  std::vector<std::vector<int>> endpoint_recovery_rows_;
  std::vector<int> pin_to_flat_endpoint_;

  // Reusable event worklist scratch
  std::vector<uint8_t> dirty_state_mask_;
  std::vector<uint8_t> dirty_group_mask_;
  std::vector<uint8_t> touched_endpoint_mask_;

  // Zero-allocation undo stacks
  std::vector<std::pair<int, int>> undo_cells_;
  std::vector<std::pair<int, ArcRow>> undo_cell_arcs_;
  std::vector<std::pair<int, ArcRow>> undo_setup_arcs_;
  std::vector<std::pair<int, std::array<double, 5>>> undo_pin_vals_;
  std::vector<std::pair<int, std::array<double, 6>>> undo_pin_rc_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_state_arr_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_state_slew_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_group_slew_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_pin_slew_;
  std::vector<std::pair<int, double>> undo_endpoints_;
};

}  // namespace dreamplace
