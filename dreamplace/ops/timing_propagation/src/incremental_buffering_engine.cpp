// SPDX-License-Identifier: BSD-3-Clause
// Incremental topology mutation and ranked evaluation for buffer insertion.
#include "flute.hpp"
#include "incremental_sizing_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "pybind11/stl.h"

namespace py = pybind11;

namespace dreamplace {
namespace {

int legalArcMask(int64_t sense, int64_t trigger)
{
  int sm = (sense == 1) ? 9 : ((sense == -1) ? 6 : 15);
  int tm = (trigger == 1) ? 3 : ((trigger == -1) ? 12 : 15);
  return sm & tm;
}

void ensureFluteLutLoaded(const std::string& powv_file, const std::string& post_file)
{
  if (powv_file.empty() || post_file.empty()) {
    throw std::runtime_error(
        "flute_powv_file and flute_post_file must be configured for FLUTE refresh");
  }
  if (!std::filesystem::is_regular_file(powv_file)
      || !std::filesystem::is_regular_file(post_file)) {
    throw std::runtime_error(
        "FLUTE LUT file missing: powv=" + powv_file + ", post=" + post_file);
  }
  static std::once_flag flute_lut_once;
  std::call_once(flute_lut_once, [&]() {
    flute::readLUT(powv_file.c_str(), post_file.c_str());
  });
}

}  // namespace

IncrementalSizingEngine::BufferActionSpec IncrementalSizingEngine::parseBufferActionSpec(
    const py::handle& h)
{
  const auto d = h.cast<py::dict>();
  BufferActionSpec spec;
  spec.net_id = d["net_id"].cast<int>();
  spec.parent_vertex = d["parent_vertex"].cast<int>();
  if (d.contains("child_vertices") && !d["child_vertices"].is_none()) {
    spec.child_vertices = d["child_vertices"].cast<std::vector<int>>();
  } else if (d.contains("child_vertex") && !d["child_vertex"].is_none()) {
    spec.child_vertices = {d["child_vertex"].cast<int>()};
  }
  spec.split_ratio =
      d.contains("split_ratio") && !d["split_ratio"].is_none()
          ? d["split_ratio"].cast<double>()
          : -1.0;
  spec.target_cell_id = d["target_cell_id"].cast<int>();
  if (d.contains("x_dbu") && !d["x_dbu"].is_none()
      && d.contains("y_dbu") && !d["y_dbu"].is_none()) {
    spec.has_xy_dbu = true;
    spec.x_dbu = d["x_dbu"].cast<double>();
    spec.y_dbu = d["y_dbu"].cast<double>();
  }
  return spec;
}

IncrementalSizingEngine::BufferUndoCheckpoint
IncrementalSizingEngine::captureBufferUndo() const
{
  return {captureUndo(),
          num_pins_,
          num_nets_,
          num_instances_,
          rc_forest_.vertex_count,
          static_cast<int>(rc_forest_.net_topo.size()),
          static_cast<int>(cell_arcs_.size()),
          num_states_,
          num_slew_groups_};
}

void IncrementalSizingEngine::restoreBufferUndo(const BufferUndoCheckpoint& cp)
{
  restoreUndo(cp.timing_cp);

  const bool levels_changed = !undo_state_levels_.empty();
  while (!undo_state_levels_.empty()) {
    const auto& [s, lvl] = undo_state_levels_.back();
    state_level_[s] = lvl;
    undo_state_levels_.pop_back();
  }
  while (!undo_group_out_states_.empty()) {
    auto& [g, vec] = undo_group_out_states_.back();
    group_out_states_[g] = std::move(vec);
    undo_group_out_states_.pop_back();
  }
  while (!undo_state_out_states_.empty()) {
    auto& [s, vec] = undo_state_out_states_.back();
    state_out_states_[s] = std::move(vec);
    undo_state_out_states_.pop_back();
  }
  while (!undo_state_in_edges_.empty()) {
    auto& [s, vec] = undo_state_in_edges_.back();
    state_in_edges_[s] = std::move(vec);
    undo_state_in_edges_.pop_back();
  }
  while (!undo_edge_segments_.empty()) {
    auto& [key, opt_segs] = undo_edge_segments_.back();
    if (opt_segs.has_value()) {
      edge_segments_[key] = std::move(*opt_segs);
    } else {
      edge_segments_.erase(key);
    }
    undo_edge_segments_.pop_back();
  }
  while (!undo_rc_topo_.empty()) {
    auto& [t, vec] = undo_rc_topo_.back();
    rc_forest_.net_topo[t] = std::move(vec);
    undo_rc_topo_.pop_back();
  }
  while (!undo_rc_children_.empty()) {
    auto& [v, vec] = undo_rc_children_.back();
    rc_forest_.adj_children[v] = std::move(vec);
    undo_rc_children_.pop_back();
  }
  while (!undo_rc_vertex_.empty()) {
    const auto& [v, tup] = undo_rc_vertex_.back();
    rc_forest_.wire_cap[v] = std::get<0>(tup);
    rc_forest_.incoming_resistance[v] = std::get<1>(tup);
    rc_forest_.incoming_wire_cap[v] = std::get<2>(tup);
    rc_forest_.parent[v] = std::get<3>(tup);
    rc_forest_.vertex_net[v] = std::get<4>(tup);
    undo_rc_vertex_.pop_back();
  }
  while (!undo_net_pins_.empty()) {
    auto& [net, vec] = undo_net_pins_.back();
    net_pins_[net] = std::move(vec);
    undo_net_pins_.pop_back();
  }
  while (!undo_pin_net_.empty()) {
    const auto& [pin, net] = undo_pin_net_.back();
    pin_net_[pin] = net;
    undo_pin_net_.pop_back();
  }

  num_instances_ = cp.num_instances;
  inst_cell_id_.resize(num_instances_);
  node_pins_.resize(num_instances_);
  node_cell_arcs_.resize(num_instances_);
  node_setup_arcs_.resize(num_instances_);
  node_check_arcs_.resize(num_instances_);

  num_pins_ = cp.num_pins;
  pin_owner_.resize(num_pins_);
  pin_net_.resize(num_pins_);
  pin_lib_offset_.resize(num_pins_);
  for (int lane = 0; lane < 5; ++lane) pin_values_[lane].resize(num_pins_);
  rc_forest_.pin_to_vertex.resize(num_pins_);
  pin_load_r_.resize(num_pins_);
  pin_load_f_.resize(num_pins_);
  pin_delay_r_.resize(num_pins_);
  pin_delay_f_.resize(num_pins_);
  pin_impulse_r_.resize(num_pins_);
  pin_impulse_f_.resize(num_pins_);
  pin_rs_.resize(num_pins_);
  pin_fs_.resize(num_pins_);
  pin_states_.resize(num_pins_);
  pin_net_sink_states_.resize(num_pins_);
  if (!sparse_mode_) {
    flat_endpoint_rRAT_.resize(num_pins_);
    flat_endpoint_fRAT_.resize(num_pins_);
    flat_endpoint_valid_.resize(num_pins_);
    pin_to_flat_endpoint_.resize(num_pins_);
    if (!recovery_deadline_r_.empty()) {
      recovery_deadline_r_.resize(num_pins_);
      recovery_deadline_f_.resize(num_pins_);
      recovery_keep_setup_.resize(num_pins_);
    }
  }

  num_nets_ = cp.num_nets;
  net_pins_.resize(num_nets_);
  net_driver_pin_.resize(num_nets_);
  rc_forest_.tree_by_net.resize(num_nets_);

  rc_forest_.vertex_count = cp.vertex_count;
  rc_forest_.wire_cap.resize(cp.vertex_count);
  rc_forest_.incoming_resistance.resize(cp.vertex_count);
  rc_forest_.incoming_wire_cap.resize(cp.vertex_count);
  rc_forest_.parent.resize(cp.vertex_count);
  rc_forest_.vertex_net.resize(cp.vertex_count);
  rc_forest_.adj_children.resize(cp.vertex_count);
  rc_forest_.net_topo.resize(cp.tree_count);
  if (!vertex_x_dbu_.empty()) {
    vertex_x_dbu_.resize(cp.vertex_count);
    vertex_y_dbu_.resize(cp.vertex_count);
  }

  cell_arcs_.resize(cp.cell_arc_count);
  cell_arc_dst_states_.resize(cp.cell_arc_count);

  num_states_ = cp.num_states;
  state_pin_.resize(num_states_);
  state_slew_group_.resize(num_states_);
  state_level_.resize(num_states_);
  state_valid_.resize(num_states_);
  state_seed_.resize(num_states_);
  state_reference_.resize(num_states_);
  state_source_type_.resize(num_states_);
  state_role_.resize(num_states_);
  state_in_edges_.resize(num_states_);
  state_out_states_.resize(num_states_);
  state_endpoint_indices_.resize(num_states_);
  state_ra_.resize(num_states_);
  state_fa_.resize(num_states_);
  state_local_rs_.resize(num_states_);
  state_local_fs_.resize(num_states_);
  dirty_state_mask_.resize(num_states_);

  num_slew_groups_ = cp.num_slew_groups;
  group_causal_.resize(num_slew_groups_);
  group_states_.resize(num_slew_groups_);
  group_out_states_.resize(num_slew_groups_);
  group_endpoint_indices_.resize(num_slew_groups_);
  group_rs_.resize(num_slew_groups_);
  group_fs_.resize(num_slew_groups_);
  dirty_group_mask_.resize(num_slew_groups_);

  if (levels_changed) {
    for (int g = 0; g < num_slew_groups_; ++g) {
      uint8_t causal = 1;
      int max_in = -1;
      for (int u : group_states_[g]) max_in = std::max(max_in, state_level_[u]);
      for (int v : group_out_states_[g]) {
        if (max_in >= state_level_[v]) {
          causal = 0;
          break;
        }
      }
      group_causal_[g] = causal;
    }
  }
}

void IncrementalSizingEngine::clearBufferUndo()
{
  clearUndo();
  undo_pin_net_.clear();
  undo_net_pins_.clear();
  undo_rc_vertex_.clear();
  undo_rc_children_.clear();
  undo_rc_topo_.clear();
  undo_edge_segments_.clear();
  undo_state_in_edges_.clear();
  undo_state_out_states_.clear();
  undo_group_out_states_.clear();
  undo_state_levels_.clear();
}

void IncrementalSizingEngine::ensureStrictDownstreamLevels(int start_state)
{
  std::vector<int> queue{start_state};
  std::vector<int> affected_groups;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const int u = queue[head];
    const int min_dst_level = state_level_[u] + 1;
    auto push_dst = [&](int dst) {
      if (state_level_[dst] < min_dst_level) {
        undo_state_levels_.emplace_back(dst, state_level_[dst]);
        state_level_[dst] = min_dst_level;
        max_level_ = std::max(max_level_, min_dst_level);
        affected_groups.push_back(state_slew_group_[dst]);
        queue.push_back(dst);
      }
    };
    for (int dst : state_out_states_[u]) push_dst(dst);
    for (int dst : group_out_states_[state_slew_group_[u]]) {
      for (const auto& edge : state_in_edges_[dst]) {
        if (edge.src_state == u) {
          push_dst(dst);
          break;
        }
      }
    }
  }
  if (!affected_groups.empty()) {
    std::sort(affected_groups.begin(), affected_groups.end());
    affected_groups.erase(
        std::unique(affected_groups.begin(), affected_groups.end()), affected_groups.end());
    for (int g : affected_groups) {
      uint8_t causal = 1;
      int max_in = -1;
      for (int u : group_states_[g]) max_in = std::max(max_in, state_level_[u]);
      for (int v : group_out_states_[g]) {
        if (max_in >= state_level_[v]) {
          causal = 0;
          break;
        }
      }
      group_causal_[g] = causal;
    }
  }
}

bool IncrementalSizingEngine::collectBufferArcLayout(
    int target, int& in_off, int& out_off, std::vector<int>& buf_luts) const
{
  const int64_t p_start = cell_pin_start_[target];
  const int64_t p_end = cell_pin_start_[target + 1];
  const int n_cell_pins = static_cast<int>(p_end - p_start);
  if (n_cell_pins < 2) return false;

  const int64_t a_start = cell_arc_start_[target];
  const int64_t a_end = cell_arc_start_[target + 1];
  in_off = -1;
  out_off = -1;
  buf_luts.clear();
  for (int64_t lut = a_start; lut < a_end; ++lut) {
    if (lib_arc_sigs_[lut][4] != 0) continue;
    const int io0 = lib_arc_pin_offsets_[lut][0];
    const int io1 = lib_arc_pin_offsets_[lut][1];
    if (io0 < 0 || io0 >= n_cell_pins || io1 < 0 || io1 >= n_cell_pins || io0 == io1) {
      continue;
    }
    if (in_off < 0) {
      in_off = io0;
      out_off = io1;
    } else if (io0 != in_off || io1 != out_off) {
      return false;
    }
    buf_luts.push_back(static_cast<int>(lut));
  }
  return !buf_luts.empty();
}

bool IncrementalSizingEngine::splitRcEdgeSegmentForBuffer(
    const BufferActionSpec& spec, int v_in, int v_out, int new_net, int& active_net)
{
  if (spec.child_vertices.size() != 1) return false;
  const int orig_u = spec.parent_vertex;
  const int orig_v = spec.child_vertices[0];
  const auto ekey = std::make_pair(orig_u, orig_v);
  auto it = edge_segments_.find(ekey);
  if (it == edge_segments_.end()) {
    if (orig_u < 0 || orig_u >= rc_forest_.vertex_count
        || orig_v < 0 || orig_v >= rc_forest_.vertex_count
        || rc_forest_.parent[orig_v] != orig_u) {
      return false;
    }
    const int u_net = rc_forest_.vertex_net[orig_u];
    if (u_net < 0 || rc_forest_.vertex_net[orig_v] != u_net) return false;
    const int drv = net_driver_pin_[u_net];
    if (drv < 0 || drv >= num_pins_ || pin_states_[drv].empty()) return false;
    undo_edge_segments_.emplace_back(ekey, std::nullopt);
    it = edge_segments_.emplace(ekey, std::vector<EdgeSegment>{{orig_u, orig_v, 0.0, 1.0}}).first;
  } else {
    undo_edge_segments_.emplace_back(ekey, it->second);
  }

  auto& segs = it->second;
  const double r = std::clamp(spec.split_ratio, 0.0, 1.0);
  std::size_t seg_idx = segs.size() - 1;
  for (std::size_t i = 0; i < segs.size(); ++i) {
    if (r <= segs[i].r_hi) {
      seg_idx = i;
      break;
    }
  }
  const EdgeSegment seg = segs[seg_idx];
  const int cur_u = seg.u;
  const int cur_v = seg.v;
  if (cur_u < 0 || cur_u >= rc_forest_.vertex_count
      || cur_v < 0 || cur_v >= rc_forest_.vertex_count
      || rc_forest_.parent[cur_v] != cur_u) {
    return false;
  }
  active_net = rc_forest_.vertex_net[cur_u];
  if (active_net < 0 || rc_forest_.vertex_net[cur_v] != active_net) return false;
  const int drv_check = net_driver_pin_[active_net];
  if (drv_check < 0 || drv_check >= num_pins_ || pin_states_[drv_check].empty()) {
    return false;
  }

  auto save_rc_vertex = [&](int v) {
    undo_rc_vertex_.emplace_back(
        v,
        std::make_tuple(rc_forest_.wire_cap[v],
                        rc_forest_.incoming_resistance[v],
                        rc_forest_.incoming_wire_cap[v],
                        rc_forest_.parent[v],
                        rc_forest_.vertex_net[v]));
  };

  const double local_alpha =
      (seg.r_hi > seg.r_lo) ? std::clamp((r - seg.r_lo) / (seg.r_hi - seg.r_lo), 0.0, 1.0)
                            : 0.5;
  const double r_edge = rc_forest_.incoming_resistance[cur_v];
  const double c_edge = rc_forest_.incoming_wire_cap[cur_v];

  save_rc_vertex(cur_u);
  undo_rc_children_.emplace_back(cur_u, rc_forest_.adj_children[cur_u]);
  rc_forest_.wire_cap[cur_u] =
      std::max(0.0, rc_forest_.wire_cap[cur_u] - 0.5 * (1.0 - local_alpha) * c_edge);
  for (int& ch : rc_forest_.adj_children[cur_u]) {
    if (ch == cur_v) {
      ch = v_in;
      break;
    }
  }

  save_rc_vertex(cur_v);
  rc_forest_.wire_cap[cur_v] =
      std::max(0.0, rc_forest_.wire_cap[cur_v] - 0.5 * local_alpha * c_edge);
  rc_forest_.incoming_resistance[cur_v] = (1.0 - local_alpha) * r_edge;
  rc_forest_.incoming_wire_cap[cur_v] = (1.0 - local_alpha) * c_edge;
  rc_forest_.parent[cur_v] = v_out;

  rc_forest_.wire_cap.push_back(0.5 * local_alpha * c_edge);
  rc_forest_.incoming_resistance.push_back(local_alpha * r_edge);
  rc_forest_.incoming_wire_cap.push_back(local_alpha * c_edge);
  rc_forest_.parent.push_back(cur_u);
  rc_forest_.vertex_net.push_back(active_net);
  rc_forest_.adj_children.emplace_back();

  rc_forest_.wire_cap.push_back(0.5 * (1.0 - local_alpha) * c_edge);
  rc_forest_.incoming_resistance.push_back(0.0);
  rc_forest_.incoming_wire_cap.push_back(0.0);
  rc_forest_.parent.push_back(-1);
  rc_forest_.vertex_net.push_back(new_net);
  rc_forest_.adj_children.push_back({cur_v});

  segs[seg_idx] = {cur_u, v_in, seg.r_lo, r};
  segs.insert(segs.begin() + static_cast<std::ptrdiff_t>(seg_idx + 1),
              EdgeSegment{v_out, cur_v, r, seg.r_hi});
  return true;
}

bool IncrementalSizingEngine::splitRcBranchChildrenForBuffer(
    const BufferActionSpec& spec, int v_in, int v_out, int new_net, int& active_net)
{
  if (spec.child_vertices.empty()) return false;
  const int cur_u = spec.parent_vertex;
  if (cur_u < 0 || cur_u >= rc_forest_.vertex_count) return false;
  active_net = rc_forest_.vertex_net[cur_u];
  if (active_net < 0) return false;
  const int drv_check = net_driver_pin_[active_net];
  if (drv_check < 0 || drv_check >= num_pins_ || pin_states_[drv_check].empty()) {
    return false;
  }

  std::vector<int> top_children;
  top_children.reserve(spec.child_vertices.size());
  for (int orig_c : spec.child_vertices) {
    const auto ekey = std::make_pair(cur_u, orig_c);
    int top_c = orig_c;
    auto it = edge_segments_.find(ekey);
    if (it != edge_segments_.end()) {
      if (it->second.empty() || it->second[0].u != cur_u) return false;
      top_c = it->second[0].v;
    }
    if (top_c < 0 || top_c >= rc_forest_.vertex_count
        || rc_forest_.parent[top_c] != cur_u
        || rc_forest_.vertex_net[top_c] != active_net) {
      return false;
    }
    if (std::find(top_children.begin(), top_children.end(), top_c) == top_children.end()) {
      top_children.push_back(top_c);
    }
  }
  if (top_children.empty()) return false;

  for (int orig_c : spec.child_vertices) {
    const auto ekey = std::make_pair(cur_u, orig_c);
    auto it = edge_segments_.find(ekey);
    if (it == edge_segments_.end()) {
      undo_edge_segments_.emplace_back(ekey, std::nullopt);
      edge_segments_[ekey] = {{v_out, orig_c, 0.0, 1.0}};
    } else {
      undo_edge_segments_.emplace_back(ekey, it->second);
      it->second[0].u = v_out;
    }
  }

  auto save_rc_vertex = [&](int v) {
    undo_rc_vertex_.emplace_back(
        v,
        std::make_tuple(rc_forest_.wire_cap[v],
                        rc_forest_.incoming_resistance[v],
                        rc_forest_.incoming_wire_cap[v],
                        rc_forest_.parent[v],
                        rc_forest_.vertex_net[v]));
  };

  save_rc_vertex(cur_u);
  undo_rc_children_.emplace_back(cur_u, rc_forest_.adj_children[cur_u]);
  double moved_half_cap = 0.0;
  for (int tc : top_children) {
    moved_half_cap += 0.5 * rc_forest_.incoming_wire_cap[tc];
    save_rc_vertex(tc);
    rc_forest_.parent[tc] = v_out;
  }
  rc_forest_.wire_cap[cur_u] = std::max(0.0, rc_forest_.wire_cap[cur_u] - moved_half_cap);
  auto& u_ch = rc_forest_.adj_children[cur_u];
  u_ch.erase(
      std::remove_if(u_ch.begin(), u_ch.end(), [&](int ch) {
        return std::find(top_children.begin(), top_children.end(), ch) != top_children.end();
      }),
      u_ch.end());
  u_ch.push_back(v_in);

  rc_forest_.wire_cap.push_back(0.0);
  rc_forest_.incoming_resistance.push_back(0.0);
  rc_forest_.incoming_wire_cap.push_back(0.0);
  rc_forest_.parent.push_back(cur_u);
  rc_forest_.vertex_net.push_back(active_net);
  rc_forest_.adj_children.emplace_back();

  rc_forest_.wire_cap.push_back(moved_half_cap);
  rc_forest_.incoming_resistance.push_back(0.0);
  rc_forest_.incoming_wire_cap.push_back(0.0);
  rc_forest_.parent.push_back(-1);
  rc_forest_.vertex_net.push_back(new_net);
  rc_forest_.adj_children.push_back(std::move(top_children));
  return true;
}

bool IncrementalSizingEngine::wireInsertedBufferStates(
    int /*buf_node*/,
    int p_in,
    int p_out,
    int active_net,
    int new_net,
    int drv_pin,
    const std::vector<int>& moved_sink_pins,
    const std::vector<int>& new_arc_ids,
    const std::vector<int>& arc_legals,
    std::vector<int>* seed_states)
{
  int drv_role = 0;
  if (sparse_mode_) {
    for (int st : pin_states_[drv_pin]) {
      if (state_role_[st] == 1) {
        drv_role = 1;
        break;
      }
    }
  }
  std::vector<int> upstream_states;
  for (int st : pin_states_[drv_pin]) {
    if (sparse_mode_) {
      if (state_role_[st] != drv_role) continue;
      if (state_source_type_[st] != 0 && !state_valid_[st][0] && !state_valid_[st][1]) {
        continue;
      }
    }
    upstream_states.push_back(st);
  }
  if (upstream_states.empty()) return false;

  std::array<double, 4> in_adj{0.0, 0.0, 1.0, 1.0};
  if (sparse_mode_ && drv_pin < initial_num_pins_) {
    bool found_adj = false;
    auto scan_pins = [&](const std::vector<int>& pins_to_scan) {
      for (int snk : pins_to_scan) {
        for (int st_snk : pin_net_sink_states_[snk]) {
          for (const auto& edge : state_in_edges_[st_snk]) {
            if (edge.kind == IncomingEdge::Kind::kNet
                && state_pin_[edge.src_state] == drv_pin) {
              in_adj = edge.net_adj;
              found_adj = true;
              return;
            }
          }
        }
      }
    };
    scan_pins(moved_sink_pins);
    if (!found_adj) scan_pins(net_pins_[active_net]);
  }

  int max_drv_level = 0;
  for (int st : upstream_states) max_drv_level = std::max(max_drv_level, state_level_[st]);
  int min_snk_level = std::numeric_limits<int>::max();
  for (int snk : moved_sink_pins) {
    for (int st_snk : pin_net_sink_states_[snk]) {
      for (const auto& edge : state_in_edges_[st_snk]) {
        if (edge.kind == IncomingEdge::Kind::kNet
            && state_pin_[edge.src_state] == drv_pin) {
          min_snk_level = std::min(min_snk_level, state_level_[st_snk]);
        }
      }
    }
  }
  int lvl_in = max_drv_level + 1;
  int lvl_out = max_drv_level + 2;
  bool need_push_downstream = false;
  if (min_snk_level < std::numeric_limits<int>::max()) {
    const int gap = min_snk_level - max_drv_level;
    if (gap >= 4) {
      lvl_in = max_drv_level + gap / 3;
      lvl_out = max_drv_level + (2 * gap) / 3;
    } else {
      need_push_downstream = true;
    }
  }
  max_level_ = std::max(max_level_, lvl_out);

  const int sg_in = num_slew_groups_++;
  const int sg_out = num_slew_groups_++;
  for (int i = 0; i < 2; ++i) {
    group_causal_.push_back(1);
    group_states_.emplace_back();
    group_out_states_.emplace_back();
    group_endpoint_indices_.emplace_back();
    group_rs_.push_back(0.0);
    group_fs_.push_back(0.0);
    dirty_group_mask_.push_back(0);
  }

  auto append_state = [&](int pin_id, int sg_id, int lvl,
                          std::array<uint8_t, 2> valid_pair, int src_type) -> int {
    const int sid = num_states_++;
    state_pin_.push_back(pin_id);
    state_slew_group_.push_back(sg_id);
    state_level_.push_back(lvl);
    state_valid_.push_back(valid_pair);
    state_seed_.push_back({0, 0});
    state_reference_.push_back({-2e8, -2e8, 0.0, 0.0});
    state_source_type_.push_back(src_type);
    state_role_.push_back(0);
    state_in_edges_.emplace_back();
    state_out_states_.emplace_back();
    state_endpoint_indices_.emplace_back();
    state_ra_.push_back(-2e8);
    state_fa_.push_back(-2e8);
    state_local_rs_.push_back(0.0);
    state_local_fs_.push_back(0.0);
    dirty_state_mask_.push_back(0);
    pin_states_[pin_id].push_back(sid);
    group_states_[sg_id].push_back(sid);
    return sid;
  };

  std::map<int, std::pair<int, std::array<uint8_t, 2>>> drv_to_out_state;
  std::vector<int> saved_out_states;
  std::vector<int> saved_out_groups;
  auto save_state_out_once = [&](int st) {
    if (std::find(saved_out_states.begin(), saved_out_states.end(), st) == saved_out_states.end()) {
      saved_out_states.push_back(st);
      undo_state_out_states_.emplace_back(st, state_out_states_[st]);
    }
  };
  auto save_group_out_once = [&](int g) {
    if (std::find(saved_out_groups.begin(), saved_out_groups.end(), g) == saved_out_groups.end()) {
      saved_out_groups.push_back(g);
      undo_group_out_states_.emplace_back(g, group_out_states_[g]);
    }
  };

  for (int st_u : upstream_states) {
    const auto up_valid = state_valid_[st_u];
    const int src_type = state_source_type_[st_u];
    const int t_in = append_state(p_in, sg_in, lvl_in, up_valid, src_type);
    const int in_mask =
        !sparse_mode_
            ? 9
            : ((src_type == 0) ? 0
                               : ((up_valid[0] ? 1 : 0) | (up_valid[1] ? 8 : 0)));
    state_in_edges_[t_in].push_back({IncomingEdge::Kind::kNet, -1, st_u, t_in, in_mask, in_adj});
    pin_net_sink_states_[p_in].push_back(t_in);
    save_state_out_once(st_u);
    state_out_states_[st_u].push_back(t_in);
    const int g_u = state_slew_group_[st_u];
    save_group_out_once(g_u);
    group_out_states_[g_u].push_back(t_in);
    if (seed_states != nullptr) seed_states->push_back(t_in);

    if (sparse_mode_ && src_type == 0) {
      const std::array<uint8_t, 2> out_valid{0, 0};
      const int t_out = append_state(p_out, sg_out, lvl_out, out_valid, src_type);
      drv_to_out_state[st_u] = {t_out, out_valid};
      bool any_added = false;
      for (std::size_t ai = 0; ai < new_arc_ids.size(); ++ai) {
        if (arc_legals[ai] != 0) {
          state_in_edges_[t_out].push_back(
              {IncomingEdge::Kind::kSlewOnlyCell, new_arc_ids[ai], t_in, t_out,
               arc_legals[ai], {0.0, 0.0, 1.0, 1.0}});
          cell_arc_dst_states_[new_arc_ids[ai]].push_back(t_out);
          any_added = true;
        }
      }
      if (any_added) group_out_states_[sg_in].push_back(t_out);
      if (seed_states != nullptr) seed_states->push_back(t_out);
    } else if (!sparse_mode_) {
      const std::array<uint8_t, 2> out_valid{1, 1};
      const int t_out = append_state(p_out, sg_out, lvl_out, out_valid, 1);
      drv_to_out_state[st_u] = {t_out, out_valid};
      for (int arc_id : new_arc_ids) {
        state_in_edges_[t_out].push_back(
            {IncomingEdge::Kind::kCell, arc_id, t_in, t_out, 15, {0.0, 0.0, 1.0, 1.0}});
        cell_arc_dst_states_[arc_id].push_back(t_out);
      }
      state_out_states_[t_in].push_back(t_out);
      group_out_states_[sg_in].push_back(t_out);
      if (seed_states != nullptr) seed_states->push_back(t_out);
    } else {
      int combined_legal = 0;
      for (int l : arc_legals) combined_legal |= l;
      const uint8_t out_r = static_cast<uint8_t>(
          (up_valid[0] && (combined_legal & 1) != 0)
          || (up_valid[1] && (combined_legal & 4) != 0));
      const uint8_t out_f = static_cast<uint8_t>(
          (up_valid[0] && (combined_legal & 2) != 0)
          || (up_valid[1] && (combined_legal & 8) != 0));
      if (!out_r && !out_f) continue;
      const std::array<uint8_t, 2> out_valid{out_r, out_f};
      const int t_out = append_state(p_out, sg_out, lvl_out, out_valid, src_type);
      drv_to_out_state[st_u] = {t_out, out_valid};
      bool any_added = false;
      for (std::size_t ai = 0; ai < new_arc_ids.size(); ++ai) {
        int cmask = 0;
        for (int ie = 0; ie < 2; ++ie) {
          if (!up_valid[ie]) continue;
          for (int oe = 0; oe < 2; ++oe) {
            const int bit = 1 << (2 * ie + oe);
            if ((arc_legals[ai] & bit) && out_valid[oe]) cmask |= bit;
          }
        }
        if (cmask != 0) {
          state_in_edges_[t_out].push_back(
              {IncomingEdge::Kind::kCell, new_arc_ids[ai], t_in, t_out,
               cmask, {0.0, 0.0, 1.0, 1.0}});
          cell_arc_dst_states_[new_arc_ids[ai]].push_back(t_out);
          any_added = true;
        }
      }
      if (any_added) {
        state_out_states_[t_in].push_back(t_out);
        group_out_states_[sg_in].push_back(t_out);
      }
      if (seed_states != nullptr) seed_states->push_back(t_out);
    }
  }

  for (int snk : moved_sink_pins) {
    for (int st_snk : pin_net_sink_states_[snk]) {
      bool has_drv_edge = false;
      for (const auto& edge : state_in_edges_[st_snk]) {
        if (edge.kind == IncomingEdge::Kind::kNet && state_pin_[edge.src_state] == drv_pin) {
          has_drv_edge = true;
          break;
        }
      }
      if (!has_drv_edge) continue;

      undo_state_in_edges_.emplace_back(st_snk, state_in_edges_[st_snk]);
      std::vector<IncomingEdge> updated_edges;
      updated_edges.reserve(state_in_edges_[st_snk].size());
      for (auto edge : state_in_edges_[st_snk]) {
        if (edge.kind == IncomingEdge::Kind::kNet && state_pin_[edge.src_state] == drv_pin) {
          const int old_src = edge.src_state;
          const int old_g = state_slew_group_[old_src];
          save_state_out_once(old_src);
          auto& s_out = state_out_states_[old_src];
          s_out.erase(std::remove(s_out.begin(), s_out.end(), st_snk), s_out.end());
          save_group_out_once(old_g);

          auto out_it = drv_to_out_state.find(old_src);
          if (out_it != drv_to_out_state.end()) {
            const int t_out = out_it->second.first;
            const auto& out_valid = out_it->second.second;
            int new_mask = edge.mask;
            if (sparse_mode_ && edge.mask != 0) {
              const int avail = (out_valid[0] ? 1 : 0) | (out_valid[1] ? 8 : 0);
              new_mask = edge.mask & avail;
              if (new_mask == 0) continue;
            }
            edge.src_state = t_out;
            edge.mask = new_mask;
            if (snk >= initial_num_pins_) {
              edge.net_adj = {0.0, 0.0, 1.0, 1.0};
            }
            updated_edges.push_back(edge);
            if (std::find(state_out_states_[t_out].begin(),
                          state_out_states_[t_out].end(),
                          st_snk)
                == state_out_states_[t_out].end()) {
              state_out_states_[t_out].push_back(st_snk);
            }
            if (std::find(group_out_states_[sg_out].begin(),
                          group_out_states_[sg_out].end(),
                          st_snk)
                == group_out_states_[sg_out].end()) {
              group_out_states_[sg_out].push_back(st_snk);
            }
          }
        } else {
          updated_edges.push_back(edge);
        }
      }
      state_in_edges_[st_snk] = std::move(updated_edges);
      if (seed_states != nullptr) seed_states->push_back(st_snk);
    }
  }

  for (int g_old : saved_out_groups) {
    auto& g_out = group_out_states_[g_old];
    g_out.erase(
        std::remove_if(g_out.begin(), g_out.end(), [&](int dst) {
          for (const auto& edge : state_in_edges_[dst]) {
            if (edge.src_state >= 0 && state_slew_group_[edge.src_state] == g_old) {
              return false;
            }
          }
          return true;
        }),
        g_out.end());
  }

  if (need_push_downstream) {
    for (int s : pin_states_[p_out]) ensureStrictDownstreamLevels(s);
  }
  return true;
}

bool IncrementalSizingEngine::insertSingleBuffer(
    const BufferActionSpec& spec,
    std::vector<int>* dirty_nets,
    std::vector<int>* seed_states,
    std::vector<int>* new_pins,
    InsertedBufferInfo* out_info)
{
  const int target = spec.target_cell_id;
  if (target < 0 || target >= num_cells_) return false;
  if (spec.net_id < 0 || spec.net_id >= num_nets_) return false;

  int in_off = -1, out_off = -1;
  std::vector<int> buf_luts;
  if (!collectBufferArcLayout(target, in_off, out_off, buf_luts)) return false;

  double buf_x_dbu = spec.x_dbu;
  double buf_y_dbu = spec.y_dbu;
  const bool track_coords = !vertex_x_dbu_.empty();
  if (track_coords && !spec.has_xy_dbu) {
    if (spec.parent_vertex >= 0 && spec.parent_vertex < rc_forest_.vertex_count) {
      buf_x_dbu = vertex_x_dbu_[spec.parent_vertex];
      buf_y_dbu = vertex_y_dbu_[spec.parent_vertex];
      if (spec.child_vertices.size() == 1 && spec.split_ratio >= 0.0) {
        const int child_v = spec.child_vertices[0];
        if (child_v >= 0 && child_v < rc_forest_.vertex_count) {
          const double r = std::clamp(spec.split_ratio, 0.0, 1.0);
          buf_x_dbu = (1.0 - r) * vertex_x_dbu_[spec.parent_vertex] + r * vertex_x_dbu_[child_v];
          buf_y_dbu = (1.0 - r) * vertex_y_dbu_[spec.parent_vertex] + r * vertex_y_dbu_[child_v];
        }
      }
    }
  }

  int active_net = -1;
  const int new_net = num_nets_;
  const int v_in = rc_forest_.vertex_count;
  const int v_out = v_in + 1;
  const bool rc_ok = (spec.split_ratio >= 0.0)
                         ? splitRcEdgeSegmentForBuffer(spec, v_in, v_out, new_net, active_net)
                         : splitRcBranchChildrenForBuffer(spec, v_in, v_out, new_net, active_net);
  if (!rc_ok) return false;

  rc_forest_.vertex_count += 2;
  if (track_coords) {
    vertex_x_dbu_.push_back(buf_x_dbu);
    vertex_x_dbu_.push_back(buf_x_dbu);
    vertex_y_dbu_.push_back(buf_y_dbu);
    vertex_y_dbu_.push_back(buf_y_dbu);
  }
  const int active_tree = rc_forest_.tree_by_net[active_net];
  if (active_tree < 0 || active_tree >= static_cast<int>(rc_forest_.net_topo.size())
      || rc_forest_.net_topo[active_tree].empty()) {
    return false;
  }

  std::vector<int> new_topo{v_out};
  for (std::size_t head = 0; head < new_topo.size(); ++head) {
    const int u = new_topo[head];
    for (int ch : rc_forest_.adj_children[u]) {
      if (rc_forest_.vertex_net[ch] != new_net) {
        undo_rc_vertex_.emplace_back(
            ch,
            std::make_tuple(rc_forest_.wire_cap[ch],
                            rc_forest_.incoming_resistance[ch],
                            rc_forest_.incoming_wire_cap[ch],
                            rc_forest_.parent[ch],
                            rc_forest_.vertex_net[ch]));
        rc_forest_.vertex_net[ch] = new_net;
      }
      new_topo.push_back(ch);
    }
  }

  const int active_root = rc_forest_.net_topo[active_tree][0];
  undo_rc_topo_.emplace_back(active_tree, rc_forest_.net_topo[active_tree]);
  std::vector<int> rebuilt_active_topo{active_root};
  for (std::size_t head = 0; head < rebuilt_active_topo.size(); ++head) {
    for (int ch : rc_forest_.adj_children[rebuilt_active_topo[head]]) {
      rebuilt_active_topo.push_back(ch);
    }
  }
  rc_forest_.net_topo[active_tree] = std::move(rebuilt_active_topo);
  const int new_tree = static_cast<int>(rc_forest_.net_topo.size());
  rc_forest_.net_topo.push_back(std::move(new_topo));

  const int64_t p_start = cell_pin_start_[target];
  const int64_t a_start = cell_arc_start_[target];
  const int drv_pin = net_driver_pin_[active_net];
  const int buf_node = num_instances_++;
  const int p_in = num_pins_++;
  const int p_out = num_pins_++;
  ++num_nets_;

  inst_cell_id_.push_back(target);
  node_pins_.push_back({p_in, p_out});
  node_cell_arcs_.emplace_back();
  node_setup_arcs_.emplace_back();
  node_check_arcs_.emplace_back();

  pin_owner_.push_back(buf_node);
  pin_owner_.push_back(buf_node);
  pin_net_.push_back(active_net);
  pin_net_.push_back(new_net);
  pin_lib_offset_.push_back(in_off);
  pin_lib_offset_.push_back(out_off);
  rc_forest_.pin_to_vertex.push_back(v_in);
  rc_forest_.pin_to_vertex.push_back(v_out);
  for (int lane = 0; lane < 5; ++lane) {
    pin_values_[lane].push_back(lib_pin_values_[lane][p_start + in_off]);
    pin_values_[lane].push_back(lib_pin_values_[lane][p_start + out_off]);
  }
  for (int i = 0; i < 2; ++i) {
    pin_load_r_.push_back(0.0);
    pin_load_f_.push_back(0.0);
    pin_delay_r_.push_back(0.0);
    pin_delay_f_.push_back(0.0);
    pin_impulse_r_.push_back(0.0);
    pin_impulse_f_.push_back(0.0);
    pin_rs_.push_back(0.0);
    pin_fs_.push_back(0.0);
    pin_states_.emplace_back();
    pin_net_sink_states_.emplace_back();
    if (!sparse_mode_) {
      flat_endpoint_rRAT_.push_back(2e8);
      flat_endpoint_fRAT_.push_back(2e8);
      flat_endpoint_valid_.push_back({1, 1});
      pin_to_flat_endpoint_.push_back(-1);
      if (!recovery_deadline_r_.empty()) {
        recovery_deadline_r_.push_back(2e8);
        recovery_deadline_f_.push_back(2e8);
        recovery_keep_setup_.push_back({0, 0});
      }
    }
  }

  undo_net_pins_.emplace_back(active_net, net_pins_[active_net]);
  std::vector<int> kept_pins;
  std::vector<int> moved_sink_pins;
  kept_pins.reserve(net_pins_[active_net].size() + 1);
  for (int pin : net_pins_[active_net]) {
    const int pv = rc_forest_.pin_to_vertex[pin];
    if (pin != drv_pin && pv >= 0 && rc_forest_.vertex_net[pv] == new_net) {
      moved_sink_pins.push_back(pin);
      undo_pin_net_.emplace_back(pin, active_net);
      pin_net_[pin] = new_net;
    } else {
      kept_pins.push_back(pin);
    }
  }
  kept_pins.push_back(p_in);
  net_pins_[active_net] = std::move(kept_pins);

  std::vector<int> new_net_pins;
  new_net_pins.reserve(1 + moved_sink_pins.size());
  new_net_pins.push_back(p_out);
  for (int pin : moved_sink_pins) new_net_pins.push_back(pin);
  net_pins_.push_back(std::move(new_net_pins));
  net_driver_pin_.push_back(p_out);
  rc_forest_.tree_by_net.push_back(new_tree);

  std::vector<int> new_arc_ids;
  new_arc_ids.reserve(buf_luts.size());
  for (int lut : buf_luts) {
    const int arc_id = static_cast<int>(cell_arcs_.size());
    cell_arcs_.push_back(
        {p_in, p_out, target, lut, lib_arc_sigs_[lut][2], lib_arc_sigs_[lut][3],
         lut - a_start, 0});
    cell_arc_dst_states_.emplace_back();
    node_cell_arcs_[buf_node].push_back(arc_id);
    new_arc_ids.push_back(arc_id);
  }

  std::vector<int> arc_legals(new_arc_ids.size(), 15);
  if (sparse_mode_) {
    for (std::size_t i = 0; i < new_arc_ids.size(); ++i) {
      const auto& row = cell_arcs_[new_arc_ids[i]];
      arc_legals[i] = legalArcMask(row[4], row[5]);
    }
  }

  if (!wireInsertedBufferStates(buf_node, p_in, p_out, active_net, new_net, drv_pin,
                                moved_sink_pins, new_arc_ids, arc_legals, seed_states)) {
    return false;
  }

  if (dirty_nets != nullptr) {
    dirty_nets->push_back(active_net);
    dirty_nets->push_back(new_net);
  }
  if (new_pins != nullptr) {
    new_pins->push_back(p_in);
    new_pins->push_back(p_out);
  }
  if (out_info != nullptr) {
    out_info->buf_node = buf_node;
    out_info->p_in = p_in;
    out_info->p_out = p_out;
    out_info->v_in = v_in;
    out_info->v_out = v_out;
    out_info->active_net = active_net;
    out_info->new_net = new_net;
    out_info->target_cell_id = target;
    out_info->x_dbu = buf_x_dbu;
    out_info->y_dbu = buf_y_dbu;
  }
  return true;
}

py::dict IncrementalSizingEngine::evaluateRankedBufferCandidates(
    const py::list& candidates,
    const py::dict& policy)
{
  const auto started = std::chrono::steady_clock::now();

  const RankingPolicy pol = parseRankingPolicy(policy, /*allow_expanded_pins=*/true);
  RankingEvalState eval = initRankingEvalState(pol, policy, /*fill_new_pin_masks=*/true);

  clearBufferUndo();
  std::vector<int> accepted_indices;
  std::vector<int> rejected_area_indices;
  std::vector<py::dict> accepted_actions;
  std::vector<int> accepted_dirty_nets;
  accepted_indices.reserve(candidates.size());
  accepted_actions.reserve(candidates.size());
  int evaluated_count = 0;
  int rejected_area_count = 0;
  int rejected_loss_count = 0;
  int skipped_invalid_count = 0;

  std::vector<uint8_t> drv_seen(num_pins_, 0);
  std::vector<int> drv_pins;
  std::vector<std::tuple<int, double, double>> drv_undo;
  std::vector<int> dirty_nets;
  std::vector<int> changed_rc_pins;
  std::vector<int> changed_slew_pins;
  std::vector<int> seed_states;
  std::vector<int> touched_endpoints;
  std::vector<int> new_pins;

  auto resize_pin_buffers = [&](int old_num_pins) {
    eval.slew_mask.resize(old_num_pins);
    eval.cap_mask.resize(old_num_pins);
    eval.slew_viol.resize(old_num_pins);
    eval.cap_viol.resize(old_num_pins);
    drv_seen.resize(old_num_pins);
  };

  for (std::size_t idx = 0; idx < candidates.size(); ++idx) {
    std::vector<BufferActionSpec> bundle;
    if (py::isinstance<py::list>(candidates[idx]) || py::isinstance<py::tuple>(candidates[idx])) {
      const auto seq = candidates[idx].cast<py::sequence>();
      bundle.reserve(seq.size());
      for (const auto& item : seq) bundle.push_back(parseBufferActionSpec(item));
    } else {
      bundle.push_back(parseBufferActionSpec(candidates[idx]));
    }
    if (bundle.empty()) {
      ++skipped_invalid_count;
      continue;
    }

    bool valid_cells = true;
    double delta_area = 0.0;
    double delta_leakage = 0.0;
    for (const auto& spec : bundle) {
      if (spec.target_cell_id < 0 || spec.target_cell_id >= num_cells_) {
        valid_cells = false;
        break;
      }
      delta_area += pol.rawCellArea(spec.target_cell_id);
      if (pol.leakage_weight != 0.0) {
        delta_leakage += std::max(0.0, pol.cell_leakage[spec.target_cell_id]);
      }
    }
    if (!valid_cells) {
      ++skipped_invalid_count;
      continue;
    }

    const double proposed_area = eval.current_area + delta_area;
    if (proposed_area > pol.capacity) {
      ++rejected_area_count;
      rejected_area_indices.push_back(static_cast<int>(idx));
      continue;
    }

    const auto cp = captureBufferUndo();
    dirty_nets.clear();
    changed_rc_pins.clear();
    changed_slew_pins.clear();
    seed_states.clear();
    touched_endpoints.clear();
    drv_pins.clear();
    drv_undo.clear();
    new_pins.clear();

    try {
      bool inserted_ok = true;
      std::vector<InsertedBufferInfo> trial_infos;
      trial_infos.reserve(bundle.size());
      for (const auto& spec : bundle) {
        InsertedBufferInfo info;
        if (!insertSingleBuffer(spec, &dirty_nets, &seed_states, &new_pins, &info)) {
          inserted_ok = false;
          break;
        }
        trial_infos.push_back(info);
      }
      if (!inserted_ok) {
        restoreBufferUndo(cp);
        ++skipped_invalid_count;
        continue;
      }

      ++evaluated_count;
      eval.slew_mask.resize(num_pins_, 1);
      eval.cap_mask.resize(num_pins_, 0);
      eval.slew_viol.resize(num_pins_, 0.0);
      eval.cap_viol.resize(num_pins_, 0.0);
      drv_seen.resize(num_pins_, 0);
      for (int p : new_pins) {
        eval.slew_mask[p] = 1;
        eval.cap_mask[p] = (pin_values_[0][p] == 0.0) ? 1 : 0;
        if (!drv_seen[p]) {
          drv_seen[p] = 1;
          drv_pins.push_back(p);
        }
      }

      if (evaluateTrialPropagationAndLoss(pol, eval, proposed_area, delta_leakage,
                                          dirty_nets, changed_rc_pins, changed_slew_pins,
                                          seed_states, touched_endpoints, drv_pins, drv_seen,
                                          drv_undo)) {
        accepted_indices.push_back(static_cast<int>(idx));
        for (const auto& info : trial_infos) {
          py::dict rec;
          rec["spec_index"] = static_cast<int>(idx);
          rec["buf_node"] = info.buf_node;
          rec["p_in"] = info.p_in;
          rec["p_out"] = info.p_out;
          rec["v_in"] = info.v_in;
          rec["v_out"] = info.v_out;
          rec["active_net"] = info.active_net;
          rec["new_net"] = info.new_net;
          rec["target_cell_id"] = info.target_cell_id;
          rec["x_dbu"] = info.x_dbu;
          rec["y_dbu"] = info.y_dbu;
          accepted_actions.push_back(std::move(rec));
          accepted_dirty_nets.push_back(info.active_net);
          accepted_dirty_nets.push_back(info.new_net);
        }
        clearBufferUndo();
      } else {
        restoreDrvUndo(eval, drv_undo);
        resize_pin_buffers(cp.num_pins);
        restoreBufferUndo(cp);
        ++rejected_loss_count;
      }
    } catch (...) {
      cleanupTrialMasks(drv_pins, drv_seen, touched_endpoints);
      restoreDrvUndo(eval, drv_undo);
      resize_pin_buffers(cp.num_pins);
      restoreBufferUndo(cp);
      throw;
    }
  }

  clearBufferUndo();
  std::sort(accepted_dirty_nets.begin(), accepted_dirty_nets.end());
  accepted_dirty_nets.erase(
      std::unique(accepted_dirty_nets.begin(), accepted_dirty_nets.end()),
      accepted_dirty_nets.end());
  py::dict result = makeEvalReport(
      pol, eval, static_cast<int>(candidates.size()), evaluated_count,
      static_cast<int>(accepted_indices.size()), rejected_area_count, rejected_loss_count,
      started);
  result["accepted_indices"] = accepted_indices;
  result["rejected_area_indices"] = rejected_area_indices;
  result["accepted_actions"] = accepted_actions;
  result["dirty_nets"] = accepted_dirty_nets;
  result["skipped_invalid_count"] = skipped_invalid_count;
  return result;
}

py::list IncrementalSizingEngine::refreshDirtySubnetSteinerTrees(
    const std::vector<int>& net_ids)
{
  if (static_cast<int>(vertex_x_dbu_.size()) != rc_forest_.vertex_count
      || static_cast<int>(vertex_y_dbu_.size()) != rc_forest_.vertex_count) {
    throw std::runtime_error(
        "refreshDirtySubnetSteinerTrees requires vertex_x_dbu/vertex_y_dbu in rc_forest");
  }
  if (!(rc_dbu_ > 0.0)) {
    throw std::runtime_error(
        "refreshDirtySubnetSteinerTrees requires positive dbu in rc_forest");
  }
  ensureFluteLutLoaded(flute_powv_file_, flute_post_file_);

  std::vector<int> dirty_nets;
  dirty_nets.reserve(net_ids.size());
  for (int nid : net_ids) {
    if (nid >= 0 && nid < num_nets_) dirty_nets.push_back(nid);
  }
  std::sort(dirty_nets.begin(), dirty_nets.end());
  dirty_nets.erase(std::unique(dirty_nets.begin(), dirty_nets.end()), dirty_nets.end());

  py::list rebuilt_subnets;
  if (dirty_nets.empty()) return rebuilt_subnets;

  clearBufferUndo();
  std::vector<int> changed_rc_pins;
  std::vector<int> seed_states;
  std::vector<int> touched_endpoints;

  for (int net_id : dirty_nets) {
    const auto& pins = net_pins_[net_id];
    const int drv_pin = net_driver_pin_[net_id];
    if (pins.empty() || drv_pin < 0 || drv_pin >= num_pins_) continue;
    const int drv_v = rc_forest_.pin_to_vertex[drv_pin];
    if (drv_v < 0) continue;
    const int tree_idx = rc_forest_.tree_by_net[net_id];
    if (tree_idx < 0 || tree_idx >= static_cast<int>(rc_forest_.net_topo.size())) continue;

    // Purge stale edge_segments_ involving any vertex from the previous tree.
    {
      std::unordered_set<int> old_vset(
          rc_forest_.net_topo[tree_idx].begin(),
          rc_forest_.net_topo[tree_idx].end());
      for (auto it = edge_segments_.begin(); it != edge_segments_.end();) {
        if (old_vset.count(it->first.first) || old_vset.count(it->first.second)) {
          it = edge_segments_.erase(it);
        } else {
          ++it;
        }
      }
    }

    // Group pins on this sub-net by quantized DBU coordinate.
    std::map<std::pair<int, int>, std::vector<int>> pin_sites;
    for (int pin : pins) {
      const int vp = rc_forest_.pin_to_vertex[pin];
      if (vp < 0) continue;
      const int qx = static_cast<int>(std::llround(vertex_x_dbu_[vp]));
      const int qy = static_cast<int>(std::llround(vertex_y_dbu_[vp]));
      pin_sites[{qx, qy}].push_back(pin);
    }
    if (pin_sites.empty()) continue;

    std::unordered_map<int, std::vector<int>> adj;
    auto add_undirected = [&](int u, int v) {
      if (u == v || u < 0 || v < 0) return;
      auto& nu = adj[u];
      if (std::find(nu.begin(), nu.end(), v) == nu.end()) nu.push_back(v);
      auto& nv = adj[v];
      if (std::find(nv.begin(), nv.end(), u) == nv.end()) nv.push_back(u);
    };

    std::map<std::pair<int, int>, int> site_canon_vertex;
    std::vector<flute::DType> vx;
    std::vector<flute::DType> vy;
    std::vector<int> site_vertices;
    vx.reserve(pin_sites.size());
    vy.reserve(pin_sites.size());
    site_vertices.reserve(pin_sites.size());

    for (const auto& kv : pin_sites) {
      const auto& pt = kv.first;
      const auto& site_pins = kv.second;
      int canon_v = rc_forest_.pin_to_vertex[site_pins.front()];
      if (std::find(site_pins.begin(), site_pins.end(), drv_pin) != site_pins.end()) {
        canon_v = drv_v;
      }
      site_canon_vertex[pt] = canon_v;
      vx.push_back(static_cast<flute::DType>(pt.first));
      vy.push_back(static_cast<flute::DType>(pt.second));
      site_vertices.push_back(canon_v);
      for (int pin : site_pins) {
        const int vp = rc_forest_.pin_to_vertex[pin];
        if (vp >= 0 && vp != canon_v) {
          add_undirected(canon_v, vp);
        }
      }
    }

    const int num_unique = static_cast<int>(site_vertices.size());
    auto append_steiner_vertex = [&](double x_dbu, double y_dbu) -> int {
      const int sv = rc_forest_.vertex_count++;
      rc_forest_.wire_cap.push_back(0.0);
      rc_forest_.incoming_resistance.push_back(0.0);
      rc_forest_.incoming_wire_cap.push_back(0.0);
      rc_forest_.parent.push_back(-1);
      rc_forest_.vertex_net.push_back(net_id);
      rc_forest_.adj_children.emplace_back();
      vertex_x_dbu_.push_back(x_dbu);
      vertex_y_dbu_.push_back(y_dbu);
      return sv;
    };

    if (num_unique >= 2 && num_unique <= MAXD) {
      flute::Tree ftree = flute::flute(num_unique, vx.data(), vy.data(), ACCURACY);
      std::map<std::pair<int, int>, int> steiner_coord_to_vertex;
      const int branch_count = 2 * ftree.deg - 2;
      if (ftree.branch != nullptr && branch_count > 0) {
        for (int bid = 0; bid < branch_count; ++bid) {
          const auto& b = ftree.branch[bid];
          const std::pair<int, int> pt(static_cast<int>(b.x), static_cast<int>(b.y));
          if (site_canon_vertex.find(pt) == site_canon_vertex.end()
              && steiner_coord_to_vertex.find(pt) == steiner_coord_to_vertex.end()) {
            steiner_coord_to_vertex[pt] =
                append_steiner_vertex(static_cast<double>(pt.first), static_cast<double>(pt.second));
          }
        }
        auto resolve_pt = [&](const std::pair<int, int>& pt) -> int {
          auto it_site = site_canon_vertex.find(pt);
          if (it_site != site_canon_vertex.end()) return it_site->second;
          auto it_st = steiner_coord_to_vertex.find(pt);
          return it_st != steiner_coord_to_vertex.end() ? it_st->second : -1;
        };
        for (int bid = 0; bid < branch_count; ++bid) {
          const auto& b1 = ftree.branch[bid];
          const auto& b2 = ftree.branch[b1.n];
          const std::pair<int, int> p1(static_cast<int>(b1.x), static_cast<int>(b1.y));
          const std::pair<int, int> p2(static_cast<int>(b2.x), static_cast<int>(b2.y));
          if (p1 == p2) continue;
          add_undirected(resolve_pt(p1), resolve_pt(p2));
        }
      }
      if (ftree.branch != nullptr) {
        std::free(ftree.branch);
      }
    } else if (num_unique > MAXD) {
      // Fallback Prim Manhattan MST for oversized nets exceeding FLUTE MAXD.
      std::vector<uint8_t> in_mst(num_unique, 0);
      std::vector<double> min_dist(num_unique, std::numeric_limits<double>::infinity());
      std::vector<int> nearest(num_unique, 0);
      int start_idx = 0;
      for (int i = 0; i < num_unique; ++i) {
        if (site_vertices[i] == drv_v) {
          start_idx = i;
          break;
        }
      }
      min_dist[start_idx] = 0.0;
      for (int step = 0; step < num_unique; ++step) {
        int u = -1;
        for (int i = 0; i < num_unique; ++i) {
          if (!in_mst[i] && (u < 0 || min_dist[i] < min_dist[u])) u = i;
        }
        if (u < 0) break;
        in_mst[u] = 1;
        if (step > 0) add_undirected(site_vertices[nearest[u]], site_vertices[u]);
        for (int v = 0; v < num_unique; ++v) {
          if (in_mst[v]) continue;
          const double d = std::abs(static_cast<double>(vx[v] - vx[u]))
                           + std::abs(static_cast<double>(vy[v] - vy[u]));
          if (d < min_dist[v]) {
            min_dist[v] = d;
            nearest[v] = u;
          }
        }
      }
    }

    // BFS from drv_v to orient directed RC tree and recompute edge R/C.
    std::vector<int> topo;
    topo.reserve(pins.size() + 8);
    std::unordered_set<int> visited;
    auto init_root = [&](int r) {
      visited.insert(r);
      topo.push_back(r);
      rc_forest_.parent[r] = -1;
      rc_forest_.incoming_resistance[r] = 0.0;
      rc_forest_.incoming_wire_cap[r] = 0.0;
      rc_forest_.wire_cap[r] = 0.0;
      rc_forest_.vertex_net[r] = net_id;
      rc_forest_.adj_children[r].clear();
    };
    init_root(drv_v);

    auto link_edge = [&](int p, int c) {
      rc_forest_.parent[c] = p;
      rc_forest_.adj_children[p].push_back(c);
      rc_forest_.adj_children[c].clear();
      rc_forest_.vertex_net[c] = net_id;
      const double len_dbu =
          std::abs(vertex_x_dbu_[c] - vertex_x_dbu_[p])
          + std::abs(vertex_y_dbu_[c] - vertex_y_dbu_[p]);
      const double len = len_dbu / rc_dbu_;
      const double r_val = len * rc_r_unit_;
      const double c_val = len * rc_c_unit_;
      rc_forest_.incoming_resistance[c] = r_val;
      rc_forest_.incoming_wire_cap[c] = c_val;
      rc_forest_.wire_cap[c] = 0.5 * c_val;
      rc_forest_.wire_cap[p] += 0.5 * c_val;
    };

    for (std::size_t head = 0; head < topo.size(); ++head) {
      const int u = topo[head];
      const auto it_adj = adj.find(u);
      if (it_adj == adj.end()) continue;
      for (int nxt : it_adj->second) {
        if (visited.find(nxt) != visited.end()) continue;
        visited.insert(nxt);
        topo.push_back(nxt);
        link_edge(u, nxt);
      }
    }
    // Safety fallback if any net pin remained unreachable from drv_v.
    for (int pin : pins) {
      const int vp = rc_forest_.pin_to_vertex[pin];
      if (vp >= 0 && visited.find(vp) == visited.end()) {
        visited.insert(vp);
        topo.push_back(vp);
        link_edge(drv_v, vp);
      }
    }

    rc_forest_.net_topo[tree_idx] = topo;
    recomputeNetRc(net_id, false, &changed_rc_pins);

    std::unordered_map<int, int> vtx_to_pin;
    py::dict py_pin_to_v;
    std::vector<int> sink_pins;
    sink_pins.reserve(pins.size());
    for (int pin : pins) {
      const int vp = rc_forest_.pin_to_vertex[pin];
      if (vp >= 0 && vtx_to_pin.find(vp) == vtx_to_pin.end()) {
        vtx_to_pin[vp] = pin;
      }
      py_pin_to_v[py::int_(pin)] = vp;
      if (pin != drv_pin) sink_pins.push_back(pin);
    }

    std::vector<int> out_parent(topo.size());
    std::vector<double> out_edge_r(topo.size());
    std::vector<double> out_edge_c(topo.size());
    std::vector<double> out_wire_cap(topo.size());
    std::vector<double> out_x_dbu(topo.size());
    std::vector<double> out_y_dbu(topo.size());
    std::vector<int> out_vtx_pin(topo.size(), -1);
    for (std::size_t k = 0; k < topo.size(); ++k) {
      const int v = topo[k];
      out_parent[k] = rc_forest_.parent[v];
      out_edge_r[k] = rc_forest_.incoming_resistance[v];
      out_edge_c[k] = rc_forest_.incoming_wire_cap[v];
      out_wire_cap[k] = rc_forest_.wire_cap[v];
      out_x_dbu[k] = vertex_x_dbu_[v];
      out_y_dbu[k] = vertex_y_dbu_[v];
      auto it_vp = vtx_to_pin.find(v);
      if (it_vp != vtx_to_pin.end()) out_vtx_pin[k] = it_vp->second;
    }

    py::dict net_rec;
    net_rec["net_id"] = net_id;
    net_rec["root_vertex"] = drv_v;
    net_rec["driver_pin_id"] = drv_pin;
    net_rec["sink_pin_ids"] = sink_pins;
    net_rec["topo"] = topo;
    net_rec["parent"] = out_parent;
    net_rec["edge_r"] = out_edge_r;
    net_rec["edge_c"] = out_edge_c;
    net_rec["wire_cap"] = out_wire_cap;
    net_rec["vertex_x_dbu"] = out_x_dbu;
    net_rec["vertex_y_dbu"] = out_y_dbu;
    net_rec["vertex_pin_id"] = out_vtx_pin;
    net_rec["pin_to_vertex"] = py_pin_to_v;
    rebuilt_subnets.append(std::move(net_rec));
  }

  for (int pin : changed_rc_pins) {
    for (int s : pin_states_[pin]) seed_states.push_back(s);
  }
  for (int net_id : dirty_nets) {
    for (int pin : net_pins_[net_id]) {
      for (int s : pin_states_[pin]) seed_states.push_back(s);
    }
  }
  std::sort(seed_states.begin(), seed_states.end());
  seed_states.erase(std::unique(seed_states.begin(), seed_states.end()), seed_states.end());

  propagateEvents(seed_states, nullptr, &touched_endpoints);
  for (int ep : touched_endpoints) {
    touched_endpoint_mask_[ep] = 0;
    endpoint_slacks_[ep] = computeEndpointSlack(ep);
  }
  clearBufferUndo();
  return rebuilt_subnets;
}

}  // namespace dreamplace
