// SPDX-License-Identifier: BSD-3-Clause
// Incremental event-driven STA and exact phase-loss gate for discrete sizing.
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
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

  pybind11::dict evaluateRankedBufferCandidates(
      const pybind11::list& candidates,
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
    std::vector<double> incoming_wire_cap;
    std::vector<int> parent;
    std::vector<int> child_start;
    std::vector<int> children;
    std::vector<int> topo;
    std::vector<int> topo_start;
    std::vector<int> tree_by_net;
    std::vector<int> pin_to_vertex;
    std::vector<int> vertex_net;
    std::vector<std::vector<int>> adj_children;
    std::vector<std::vector<int>> net_topo;

    mutable std::vector<double> scratch_cap;
    mutable std::vector<double> scratch_load;
    mutable std::vector<double> scratch_delay;
    mutable std::vector<double> scratch_ldelay;
    mutable std::vector<double> scratch_beta;
  };

  struct EdgeSegment
  {
    int u = -1;
    int v = -1;
    double r_lo = 0.0;
    double r_hi = 1.0;
  };

  struct BufferActionSpec
  {
    int net_id = -1;
    int parent_vertex = -1;
    std::vector<int> child_vertices;
    double split_ratio = -1.0;
    int target_cell_id = -1;
  };

  struct RankingPolicy
  {
    double wns_coeff = 0.0;
    double tns_coeff = 0.0;
    double timing_weight = 0.0;
    double slew_weight = 0.0;
    double cap_weight = 0.0;
    double leakage_weight = 0.0;
    double loss_tolerance = 0.0;
    double padding_x = 0.0;
    int num_movable = 0;
    double virtual_area = 0.0;
    double capacity = 0.0;
    std::vector<double> cell_width;
    std::vector<double> cell_height;
    std::vector<double> cell_leakage;
    std::vector<uint8_t> inflated_mask;
    std::vector<double> inflated_width;
    std::vector<double> inflated_height;

    double movableNodeArea(int node, int cell) const;
    double rawCellArea(int cell) const;
    double computeLoss(double wns_ps, double tns_ps, double slew_ps,
                       double cap_pf, double leak_mw) const;
  };

  struct RankingEvalState
  {
    std::vector<uint8_t> slew_mask;
    std::vector<uint8_t> cap_mask;
    std::vector<double> slew_viol;
    std::vector<double> cap_viol;
    double current_area = 0.0;
    double current_leakage = 0.0;
    double total_slew_ps = 0.0;
    double total_cap_pf = 0.0;
    double cur_wns = 0.0;
    double cur_tns = 0.0;
    double current_loss = 0.0;
    double initial_loss = 0.0;
    double initial_wns = 0.0;
    double initial_tns = 0.0;
    double initial_slew_ps = 0.0;
    double initial_cap_pf = 0.0;
    double initial_area = 0.0;
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
    std::size_t check_arc_pos = 0;
    std::size_t pin_val_pos = 0;
    std::size_t pin_rc_pos = 0;
    std::size_t state_arr_pos = 0;
    std::size_t state_slew_pos = 0;
    std::size_t group_slew_pos = 0;
    std::size_t pin_slew_pos = 0;
    std::size_t endpoint_pos = 0;
  };

  struct BufferUndoCheckpoint
  {
    UndoCheckpoint timing_cp;
    int num_pins = 0;
    int num_nets = 0;
    int num_instances = 0;
    int vertex_count = 0;
    int tree_count = 0;
    int cell_arc_count = 0;
    int num_states = 0;
    int num_slew_groups = 0;
  };

  using ArcRow = std::array<int64_t, 8>;
  using LibArcSig = std::array<int64_t, 5>;  // src_sig, dst_sig, sense, trigger, check_type

  int mapLibertyArc(int source_cell, int target_cell, int old_arc_id) const;
  void remapNodeArcs(int node, int target, bool record_undo,
                     std::vector<int>* seed_states,
                     std::vector<int>* touched_endpoints);
  RankingPolicy parseRankingPolicy(const pybind11::dict& policy,
                                   bool allow_expanded_pins) const;
  RankingEvalState initRankingEvalState(
      const RankingPolicy& pol,
      const pybind11::dict& policy,
      bool fill_new_pin_masks,
      std::vector<double>* current_node_area = nullptr) const;
  double pinSlewViolationPs(const std::vector<uint8_t>& slew_mask, int pin) const;
  double pinCapViolationPf(const std::vector<uint8_t>& cap_mask, int pin) const;
  std::pair<double, double> computeWnsTns() const;
  bool evaluateTrialPropagationAndLoss(
      const RankingPolicy& pol,
      RankingEvalState& eval,
      double proposed_area,
      double delta_leakage,
      std::vector<int>& dirty_nets,
      std::vector<int>& changed_rc_pins,
      std::vector<int>& changed_slew_pins,
      std::vector<int>& seed_states,
      std::vector<int>& touched_endpoints,
      std::vector<int>& drv_pins,
      std::vector<uint8_t>& drv_seen,
      std::vector<std::tuple<int, double, double>>& drv_undo);
  static void restoreDrvUndo(
      RankingEvalState& eval,
      const std::vector<std::tuple<int, double, double>>& drv_undo);
  void cleanupTrialMasks(
      const std::vector<int>& drv_pins,
      std::vector<uint8_t>& drv_seen,
      const std::vector<int>& touched_endpoints);
  pybind11::dict makeEvalReport(
      const RankingPolicy& pol,
      const RankingEvalState& eval,
      int input_count,
      int evaluated_count,
      int accepted_count,
      int rejected_area_count,
      int rejected_loss_count,
      const std::chrono::steady_clock::time_point& started) const;

  static BufferActionSpec parseBufferActionSpec(const pybind11::handle& h);
  bool collectBufferArcLayout(int target, int& in_off, int& out_off,
                              std::vector<int>& buf_luts) const;
  bool splitRcEdgeSegmentForBuffer(const BufferActionSpec& spec, int v_in, int v_out,
                                   int new_net, int& active_net);
  bool splitRcBranchChildrenForBuffer(const BufferActionSpec& spec, int v_in, int v_out,
                                      int new_net, int& active_net);
  bool wireInsertedBufferStates(int buf_node, int p_in, int p_out, int active_net,
                                int new_net, int drv_pin,
                                const std::vector<int>& moved_sink_pins,
                                const std::vector<int>& new_arc_ids,
                                const std::vector<int>& arc_legals,
                                std::vector<int>* seed_states);
  bool insertSingleBuffer(const BufferActionSpec& spec,
                          std::vector<int>* dirty_nets,
                          std::vector<int>* seed_states,
                          std::vector<int>* new_pins);
  void ensureStrictDownstreamLevels(int start_state);
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
  BufferUndoCheckpoint captureBufferUndo() const;
  void restoreBufferUndo(const BufferUndoCheckpoint& cp);
  void clearBufferUndo();

  int num_pins_ = 0;
  int num_nets_ = 0;
  int num_instances_ = 0;
  int num_cells_ = 0;
  int initial_num_pins_ = 0;
  int initial_num_instances_ = 0;
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
  std::vector<std::array<int, 2>> lib_arc_pin_offsets_;
  std::vector<int> net_driver_pin_;
  std::array<std::vector<double>, 5> lib_pin_values_;
  std::array<std::vector<double>, 5> pin_values_;  // cap, rcap, fcap, max_slew, max_cap

  // Mutable arc tables matching runtime_timing_arcs.py
  std::vector<ArcRow> cell_arcs_;
  std::vector<ArcRow> setup_arcs_;
  std::vector<ArcRow> check_arcs_;
  std::vector<std::vector<int>> node_cell_arcs_;
  std::vector<std::vector<int>> node_setup_arcs_;
  std::vector<std::vector<int>> node_check_arcs_;
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
  std::vector<int> state_source_type_;
  std::vector<int> state_role_;
  std::vector<uint8_t> group_causal_;

  std::vector<std::vector<IncomingEdge>> state_in_edges_;
  std::vector<std::vector<int>> state_out_states_;
  std::vector<std::vector<int>> group_states_;
  std::vector<std::vector<int>> group_out_states_;
  std::vector<std::vector<int>> pin_states_;
  std::vector<std::vector<int>> pin_net_sink_states_;
  std::vector<std::vector<int>> cell_arc_dst_states_;
  std::vector<std::vector<int>> setup_arc_endpoints_;
  std::vector<std::vector<int>> check_arc_endpoints_;

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
  std::vector<int> recovery_check_rows_;
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
  std::vector<std::pair<int, ArcRow>> undo_check_arcs_;
  std::vector<std::pair<int, std::array<double, 5>>> undo_pin_vals_;
  std::vector<std::pair<int, std::array<double, 6>>> undo_pin_rc_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_state_arr_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_state_slew_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_group_slew_;
  std::vector<std::pair<int, std::array<double, 2>>> undo_pin_slew_;
  std::vector<std::pair<int, double>> undo_endpoints_;

  // Buffer topology mutation & undo state
  std::map<std::pair<int, int>, std::vector<EdgeSegment>> edge_segments_;
  std::vector<std::pair<int, int>> undo_pin_net_;
  std::vector<std::pair<int, std::vector<int>>> undo_net_pins_;
  std::vector<std::pair<int, std::tuple<double, double, double, int, int>>> undo_rc_vertex_;
  std::vector<std::pair<int, std::vector<int>>> undo_rc_children_;
  std::vector<std::pair<int, std::vector<int>>> undo_rc_topo_;
  std::vector<std::pair<std::pair<int, int>, std::optional<std::vector<EdgeSegment>>>>
      undo_edge_segments_;
  std::vector<std::pair<int, std::vector<IncomingEdge>>> undo_state_in_edges_;
  std::vector<std::pair<int, std::vector<int>>> undo_state_out_states_;
  std::vector<std::pair<int, std::vector<int>>> undo_group_out_states_;
  std::vector<std::pair<int, int>> undo_state_levels_;
};

}  // namespace dreamplace
