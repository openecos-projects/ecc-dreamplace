// SPDX-License-Identifier: BSD-3-Clause
// Incremental event-driven STA and exact phase-loss gate for discrete sizing.
#include "incremental_sizing_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#include "cell_aat_math.h"
#include "net_aat_math.h"
#include "pybind11/stl.h"
#include "timing_hard_math.h"

namespace py = pybind11;

namespace dreamplace {
namespace {

void require(bool cond, const std::string& msg)
{
  if (!cond) throw py::value_error(msg);
}

bool bitsChanged(double a, double b)
{
  return std::isnan(a) || std::isnan(b) || std::memcmp(&a, &b, sizeof(double)) != 0;
}

template <typename T>
std::vector<T> copy1D(const py::handle& obj, const std::string& name)
{
  auto arr = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(obj);
  require(bool(arr) && arr.ndim() == 1, name + " must be a 1D array");
  const auto n = static_cast<std::size_t>(arr.shape(0));
  if (n == 0) return {};
  const T* ptr = arr.data();
  return std::vector<T>(ptr, ptr + n);
}

template <std::size_t Cols>
std::vector<std::array<int64_t, Cols>> copyRows(const py::handle& obj, const std::string& name)
{
  auto arr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(obj);
  require(bool(arr) && arr.ndim() == 2, name + " must be a 2D integer array");
  if (arr.shape(0) == 0) return {};
  require(static_cast<std::size_t>(arr.shape(1)) >= Cols,
          name + " has fewer columns than required");
  const auto rows = static_cast<std::size_t>(arr.shape(0));
  const auto stride = static_cast<std::size_t>(arr.shape(1));
  const int64_t* ptr = arr.data();
  std::vector<std::array<int64_t, Cols>> out(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    std::copy_n(ptr + r * stride, Cols, out[r].begin());
  }
  return out;
}

std::vector<std::array<uint8_t, 2>> copyBoolPairs(
    const py::handle& obj, std::size_t expected_rows, const std::string& name)
{
  std::vector<std::array<uint8_t, 2>> out(expected_rows, {1, 1});
  if (obj.is_none() || expected_rows == 0) return out;
  auto arr = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(obj);
  require(bool(arr) && arr.ndim() == 2 && static_cast<std::size_t>(arr.shape(0)) == expected_rows
              && arr.shape(1) == 2,
          name + " must have shape [rows, 2]");
  const bool* ptr = arr.data();
  for (std::size_t r = 0; r < expected_rows; ++r) {
    out[r] = {static_cast<uint8_t>(ptr[r * 2]), static_cast<uint8_t>(ptr[r * 2 + 1])};
  }
  return out;
}

void clk2qSingleRow(
    const std::array<int64_t, 8>& arc, double cap_r, double cap_f,
    const std::vector<double>& clk_r, const std::vector<double>& clk_f,
    TimingLutTable<double> delay_rise, TimingLutTable<double> delay_fall,
    TimingLutTable<double> tran_rise, TimingLutTable<double> tran_fall,
    bool smooth_pairwise, double alpha, double out[4])
{
  const int64_t clock_pin = arc[0];
  const int64_t row = arc[3];
  const bool rising_edge = arc[5] == 1 || arc[5] == 0;
  const bool falling_edge = arc[5] == -1 || arc[5] == 0;
  const double neg_inf = -std::numeric_limits<double>::infinity();
  double rise_update = neg_inf, fall_update = neg_inf;
  double rise_tran = 0.0, fall_tran = 0.0;
  if (arc[4] == 1) {
    if (rising_edge) {
      rise_update = timingLutQuery(delay_rise, row, clk_r[clock_pin], cap_r).value;
      rise_tran = timingLutQuery(tran_rise, row, clk_r[clock_pin], cap_r).value;
    }
    if (falling_edge) {
      fall_update = timingLutQuery(delay_fall, row, clk_f[clock_pin], cap_f).value;
      fall_tran = timingLutQuery(tran_fall, row, clk_f[clock_pin], cap_f).value;
    }
  } else if (arc[4] == -1) {
    if (rising_edge) {
      fall_update = timingLutQuery(delay_fall, row, clk_r[clock_pin], cap_f).value;
      fall_tran = timingLutQuery(tran_fall, row, clk_r[clock_pin], cap_f).value;
    }
    if (falling_edge) {
      rise_update = timingLutQuery(delay_rise, row, clk_f[clock_pin], cap_r).value;
      rise_tran = timingLutQuery(tran_rise, row, clk_f[clock_pin], cap_r).value;
    }
  } else {
    double rr = neg_inf, rf = neg_inf, fr = neg_inf, ff = neg_inf;
    double rr_t = 0.0, rf_t = 0.0, fr_t = 0.0, ff_t = 0.0;
    if (rising_edge) {
      rr = timingLutQuery(delay_rise, row, clk_r[clock_pin], cap_r).value;
      rf = timingLutQuery(delay_fall, row, clk_r[clock_pin], cap_f).value;
      rr_t = timingLutQuery(tran_rise, row, clk_r[clock_pin], cap_r).value;
      rf_t = timingLutQuery(tran_fall, row, clk_r[clock_pin], cap_f).value;
    }
    if (falling_edge) {
      fr = timingLutQuery(delay_rise, row, clk_f[clock_pin], cap_r).value;
      ff = timingLutQuery(delay_fall, row, clk_f[clock_pin], cap_f).value;
      fr_t = timingLutQuery(tran_rise, row, clk_f[clock_pin], cap_r).value;
      ff_t = timingLutQuery(tran_fall, row, clk_f[clock_pin], cap_f).value;
    }
    double wa = 0.0, wb = 0.0;
    rise_update = pairMaximum(rr, fr, smooth_pairwise, alpha, wa, wb);
    fall_update = pairMaximum(rf, ff, smooth_pairwise, alpha, wa, wb);
    rise_tran = pairMaximum(rr_t, fr_t, smooth_pairwise, alpha, wa, wb);
    fall_tran = pairMaximum(rf_t, ff_t, smooth_pairwise, alpha, wa, wb);
  }
  out[0] = rise_update;
  out[1] = fall_update;
  out[2] = rise_tran;
  out[3] = fall_tran;
}

double smoothMergeValues(
    double initial, const double* contribs, std::size_t count, double tau)
{
  double maximum = initial;
  for (std::size_t i = 0; i < count; ++i) {
    const double v = contribs[i];
    maximum = std::isnan(maximum) ? maximum : (std::isnan(v) ? v : std::max(maximum, v));
  }
  if (!std::isfinite(maximum)) return maximum;
  double sum = std::exp((initial - maximum) / tau);
  for (std::size_t i = 0; i < count; ++i) {
    sum += std::exp((contribs[i] - maximum) / tau);
  }
  return maximum + tau * std::log(std::max(sum, 1.0e-30));
}

}  // namespace

TimingLutTable<double> IncrementalSizingEngine::LutStorage::view() const
{
  return {values.data(), transitions.data(), capacitances.data(),
          transition_dims.data(), capacitance_dims.data(),
          rows, value_stride, transition_stride, capacitance_stride};
}

IncrementalSizingEngine::IncrementalSizingEngine(const py::dict& config)
{
  num_pins_ = config["num_pins"].cast<int>();
  num_nets_ = config["num_nets"].cast<int>();
  num_instances_ = config["num_instances"].cast<int>();
  num_cells_ = config["num_cells"].cast<int>();
  initial_num_pins_ = num_pins_;
  initial_num_instances_ = num_instances_;
  require(num_pins_ > 0 && num_nets_ >= 0 && num_instances_ >= 0 && num_cells_ > 0,
          "invalid IncrementalSizingEngine dimensions");

  smooth_aggregation_ = config["aggregation_mode"].cast<std::string>() == "smooth";
  aggregation_tau_ps_ = config["aggregation_tau_ps"].cast<double>();
  pairwise_smooth_ = config["pairwise_smooth"].cast<bool>();
  pairwise_alpha_ = config["pairwise_alpha"].cast<double>();
  slew_epsilon_ = config["slew_epsilon"].cast<double>();

  pin_owner_ = copy1D<int>(config["pin2node_map"], "pin2node_map");
  pin_net_ = copy1D<int>(config["pin2net_map"], "pin2net_map");
  pin_lib_offset_ = copy1D<int>(config["pin_2_libpin_offset"], "pin_2_libpin_offset");
  inst_cell_id_ = copy1D<int>(config["inst_cell_id"], "inst_cell_id");
  cell_arc_start_ = copy1D<int64_t>(config["cell_id_2_arc_id_start"], "cell_id_2_arc_id_start");
  cell_pin_start_ = copy1D<int64_t>(config["cell_id_2_libpin_id_start"], "cell_id_2_libpin_id_start");
  require(static_cast<int>(pin_owner_.size()) == num_pins_
              && static_cast<int>(pin_net_.size()) == num_pins_
              && static_cast<int>(pin_lib_offset_.size()) == num_pins_,
          "pin metadata lengths must equal num_pins");
  require(static_cast<int>(inst_cell_id_.size()) == num_instances_,
          "inst_cell_id length must equal num_instances");
  require(static_cast<int>(cell_arc_start_.size()) == num_cells_ + 1
              && static_cast<int>(cell_pin_start_.size()) == num_cells_ + 1,
          "cell start offset lengths must equal num_cells + 1");

  const auto raw_lib_arcs = copyRows<6>(config["flat_libarc_info"], "flat_libarc_info");
  std::vector<int64_t> raw_check_types(raw_lib_arcs.size(), 0);
  if (config.contains("flat_libarc_check_types") && !config["flat_libarc_check_types"].is_none()) {
    raw_check_types = copy1D<int64_t>(config["flat_libarc_check_types"], "flat_libarc_check_types");
    require(raw_check_types.size() == raw_lib_arcs.size(),
            "flat_libarc_check_types length must equal flat_libarc_info rows");
  }
  lib_arc_sigs_.resize(raw_lib_arcs.size());
  for (std::size_t i = 0; i < raw_lib_arcs.size(); ++i) {
    lib_arc_sigs_[i] = {raw_lib_arcs[i][0], raw_lib_arcs[i][1],
                        raw_lib_arcs[i][4], raw_lib_arcs[i][5], raw_check_types[i]};
  }

  constexpr std::array<const char*, 5> lib_pin_keys{
      "flat_lib_pin_cap", "flat_lib_pin_rcap", "flat_lib_pin_fcap",
      "flat_lib_pin_slew_limit", "flat_lib_pin_cap_limit"};
  constexpr std::array<const char*, 5> base_pin_keys{
      "pin_cap_base", "pin_rcap_base", "pin_fcap_base",
      "pin_slew_limit", "pin_cap_limit"};
  for (int lane = 0; lane < 5; ++lane) {
    lib_pin_values_[lane] = copy1D<double>(config[lib_pin_keys[lane]], lib_pin_keys[lane]);
    pin_values_[lane] = copy1D<double>(config[base_pin_keys[lane]], base_pin_keys[lane]);
    require(static_cast<int>(pin_values_[lane].size()) == num_pins_,
            std::string(base_pin_keys[lane]) + " length mismatch");
  }

  lib_arc_pin_offsets_.assign(raw_lib_arcs.size(), {-1, -1});
  if (config.contains("lib_arc_pin_offsets") && !config["lib_arc_pin_offsets"].is_none()) {
    const auto raw_offsets = copyRows<2>(config["lib_arc_pin_offsets"], "lib_arc_pin_offsets");
    require(raw_offsets.size() == raw_lib_arcs.size(),
            "lib_arc_pin_offsets length must equal flat_libarc_info rows");
    for (std::size_t i = 0; i < raw_offsets.size(); ++i) {
      lib_arc_pin_offsets_[i] = {static_cast<int>(raw_offsets[i][0]),
                                 static_cast<int>(raw_offsets[i][1])};
    }
  } else {
    for (int cell = 0; cell < num_cells_; ++cell) {
      const int64_t p_start = cell_pin_start_[cell];
      const int64_t p_end = cell_pin_start_[cell + 1];
      if (p_end - p_start != 2) continue;
      int in_off = 0, out_off = 1;
      if (lib_pin_values_[0][p_start + 1] > 0.0 && lib_pin_values_[0][p_start] == 0.0) {
        in_off = 1;
        out_off = 0;
      }
      for (int64_t a = cell_arc_start_[cell]; a < cell_arc_start_[cell + 1]; ++a) {
        if (raw_lib_arcs[a][0] != raw_lib_arcs[a][1] && raw_check_types[a] == 0) {
          lib_arc_pin_offsets_[a] = {in_off, out_off};
        }
      }
    }
  }

  net_pins_.resize(num_nets_);
  node_pins_.resize(num_instances_);
  for (int pin = 0; pin < num_pins_; ++pin) {
    const int net = pin_net_[pin];
    if (net >= 0 && net < num_nets_) net_pins_[net].push_back(pin);
    const int owner = pin_owner_[pin];
    if (owner >= 0 && owner < num_instances_ && pin_lib_offset_[pin] >= 0) {
      node_pins_[owner].push_back(pin);
    }
  }

  // Parse 6 LUT tables: r_delay, f_delay, r_trans, f_trans, r_check, f_check.
  const auto lut_list = config["luts"].cast<py::list>();
  require(lut_list.size() == 6, "luts must contain 6 LUT bundles");
  for (int idx = 0; idx < 6; ++idx) {
    const auto bundle = lut_list[idx].cast<py::list>();
    require(bundle.size() == 5, "LUT bundle must contain 5 arrays");
    auto vals = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(bundle[0]);
    auto trans = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(bundle[1]);
    auto caps = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(bundle[2]);
    require(bool(vals) && bool(trans) && bool(caps) && vals.ndim() == 2
                && trans.ndim() == 2 && caps.ndim() == 2,
            "LUT matrices must be 2D arrays");
    auto& dst = luts_[idx];
    dst.rows = static_cast<int>(vals.shape(0));
    dst.value_stride = static_cast<int>(vals.shape(1));
    dst.transition_stride = static_cast<int>(trans.shape(1));
    dst.capacitance_stride = static_cast<int>(caps.shape(1));
    dst.values.assign(vals.data(), vals.data() + vals.size());
    dst.transitions.assign(trans.data(), trans.data() + trans.size());
    dst.capacitances.assign(caps.data(), caps.data() + caps.size());
    dst.transition_dims = copy1D<int64_t>(bundle[3], "trans_dims");
    dst.capacitance_dims = copy1D<int64_t>(bundle[4], "cap_dims");
  }

  cell_arcs_ = copyRows<8>(config["cell_arcs"], "cell_arcs");
  const auto level_start = copy1D<int64_t>(config["cell_level_start"], "cell_level_start");
  require(level_start.size() >= 2, "cell_level_start must have at least 2 entries");
  clock_end_ = static_cast<int>(level_start[1]);
  setup_arcs_ = copyRows<8>(config["setup_arcs"], "setup_arcs");
  check_arcs_ = copyRows<8>(config["check_arcs"], "check_arcs");

  auto index_node_arcs = [&](const std::vector<ArcRow>& arcs,
                             std::vector<std::vector<int>>& out) {
    out.resize(num_instances_);
    for (int i = 0; i < static_cast<int>(arcs.size()); ++i) {
      const int out_pin = static_cast<int>(arcs[i][1]);
      if (out_pin >= 0 && out_pin < num_pins_) {
        const int owner = pin_owner_[out_pin];
        if (owner >= 0 && owner < num_instances_ && arcs[i][3] >= 0) {
          out[owner].push_back(i);
        }
      }
    }
  };
  index_node_arcs(cell_arcs_, node_cell_arcs_);
  index_node_arcs(setup_arcs_, node_setup_arcs_);
  index_node_arcs(check_arcs_, node_check_arcs_);

  // Parse StaticRcForest (unified across placement FLUTE RC and GR RouteSnapshot).
  const auto rc_cfg = config["rc_forest"].cast<py::dict>();
  rc_forest_.wire_cap = copy1D<double>(rc_cfg["wire_cap"], "wire_cap");
  rc_forest_.incoming_resistance =
      copy1D<double>(rc_cfg["incoming_resistance"], "incoming_resistance");
  rc_forest_.parent = copy1D<int>(rc_cfg["parent"], "parent");
  rc_forest_.child_start = copy1D<int>(rc_cfg["child_start"], "child_start");
  rc_forest_.children = copy1D<int>(rc_cfg["children"], "children");
  rc_forest_.topo = copy1D<int>(rc_cfg["topo"], "topo");
  rc_forest_.topo_start = copy1D<int>(rc_cfg["topo_start"], "topo_start");
  rc_forest_.tree_by_net = copy1D<int>(rc_cfg["tree_by_net"], "tree_by_net");
  rc_forest_.pin_to_vertex = copy1D<int>(rc_cfg["pin_to_vertex"], "pin_to_vertex");
  rc_forest_.vertex_count = static_cast<int>(rc_forest_.wire_cap.size());
  require(static_cast<int>(rc_forest_.incoming_resistance.size()) == rc_forest_.vertex_count
              && static_cast<int>(rc_forest_.parent.size()) == rc_forest_.vertex_count
              && static_cast<int>(rc_forest_.child_start.size()) == rc_forest_.vertex_count + 1
              && static_cast<int>(rc_forest_.tree_by_net.size()) == num_nets_
              && static_cast<int>(rc_forest_.pin_to_vertex.size()) == num_pins_,
          "rc_forest array dimensions do not align");

  rc_forest_.adj_children.resize(rc_forest_.vertex_count);
  for (int v = 0; v < rc_forest_.vertex_count; ++v) {
    const int cs = rc_forest_.child_start[v];
    const int ce = rc_forest_.child_start[v + 1];
    rc_forest_.adj_children[v].assign(
        rc_forest_.children.begin() + cs, rc_forest_.children.begin() + ce);
  }
  const int num_rc_trees =
      rc_forest_.topo_start.empty() ? 0 : static_cast<int>(rc_forest_.topo_start.size()) - 1;
  rc_forest_.net_topo.resize(num_rc_trees);
  for (int t = 0; t < num_rc_trees; ++t) {
    const int tb = rc_forest_.topo_start[t];
    const int te = rc_forest_.topo_start[t + 1];
    if (te > tb) {
      rc_forest_.net_topo[t].assign(
          rc_forest_.topo.begin() + tb, rc_forest_.topo.begin() + te);
    }
  }
  rc_forest_.vertex_net.assign(rc_forest_.vertex_count, -1);
  for (int net = 0; net < num_nets_; ++net) {
    const int t = rc_forest_.tree_by_net[net];
    if (t >= 0 && t < num_rc_trees) {
      for (int v : rc_forest_.net_topo[t]) rc_forest_.vertex_net[v] = net;
    }
  }
  if (rc_cfg.contains("incoming_wire_cap") && !rc_cfg["incoming_wire_cap"].is_none()) {
    rc_forest_.incoming_wire_cap =
        copy1D<double>(rc_cfg["incoming_wire_cap"], "incoming_wire_cap");
    require(static_cast<int>(rc_forest_.incoming_wire_cap.size()) == rc_forest_.vertex_count,
            "incoming_wire_cap length mismatch");
  } else {
    rc_forest_.incoming_wire_cap.assign(rc_forest_.vertex_count, 0.0);
    for (int t = 0; t < num_rc_trees; ++t) {
      const auto& topo = rc_forest_.net_topo[t];
      for (int k = static_cast<int>(topo.size()) - 1; k >= 1; --k) {
        const int v = topo[k];
        double child_half = 0.0;
        for (int ch : rc_forest_.adj_children[v]) {
          child_half += 0.5 * rc_forest_.incoming_wire_cap[ch];
        }
        rc_forest_.incoming_wire_cap[v] =
            std::max(0.0, 2.0 * (rc_forest_.wire_cap[v] - child_half));
      }
    }
  }

  rc_forest_.scratch_cap.assign(rc_forest_.vertex_count, 0.0);
  rc_forest_.scratch_load.assign(rc_forest_.vertex_count, 0.0);
  rc_forest_.scratch_delay.assign(rc_forest_.vertex_count, 0.0);
  rc_forest_.scratch_ldelay.assign(rc_forest_.vertex_count, 0.0);
  rc_forest_.scratch_beta.assign(rc_forest_.vertex_count, 0.0);

  pin_load_r_.assign(num_pins_, 0.0);
  pin_load_f_.assign(num_pins_, 0.0);
  pin_delay_r_.assign(num_pins_, 0.0);
  pin_delay_f_.assign(num_pins_, 0.0);
  pin_impulse_r_.assign(num_pins_, 0.0);
  pin_impulse_f_.assign(num_pins_, 0.0);
  for (int net = 0; net < num_nets_; ++net) {
    recomputeNetRc(net, false, nullptr);
  }

  sparse_mode_ = config["sparse_mode"].cast<bool>();
  cell_arc_dst_states_.resize(cell_arcs_.size());
  setup_arc_endpoints_.resize(setup_arcs_.size());
  check_arc_endpoints_.resize(check_arcs_.size());
  pin_states_.resize(num_pins_);
  pin_net_sink_states_.resize(num_pins_);

  if (sparse_mode_) {
    const auto sp = config["sparse"].cast<py::dict>();
    state_pin_ = copy1D<int>(sp["state_pin"], "state_pin");
    state_slew_group_ = copy1D<int>(sp["state_slew_group"], "state_slew_group");
    state_level_ = copy1D<int>(sp["state_level"], "state_level");
    num_states_ = static_cast<int>(state_pin_.size());
    require(static_cast<int>(state_slew_group_.size()) == num_states_
                && static_cast<int>(state_level_.size()) == num_states_,
            "sparse state arrays do not align");
    if (sp.contains("state_source_type") && !sp["state_source_type"].is_none()) {
      state_source_type_ = copy1D<int>(sp["state_source_type"], "state_source_type");
      require(static_cast<int>(state_source_type_.size()) == num_states_,
              "state_source_type length mismatch");
    } else {
      state_source_type_.assign(num_states_, 1);
    }
    if (sp.contains("state_role") && !sp["state_role"].is_none()) {
      state_role_ = copy1D<int>(sp["state_role"], "state_role");
      require(static_cast<int>(state_role_.size()) == num_states_,
              "state_role length mismatch");
    } else {
      state_role_.assign(num_states_, 0);
    }
    num_slew_groups_ = 0;
    max_level_ = 0;
    for (int s = 0; s < num_states_; ++s) {
      num_slew_groups_ = std::max(num_slew_groups_, state_slew_group_[s] + 1);
      max_level_ = std::max(max_level_, state_level_[s]);
      pin_states_[state_pin_[s]].push_back(s);
    }
    state_valid_ = copyBoolPairs(sp["state_valid"], num_states_, "state_valid");
    state_seed_ = copyBoolPairs(sp["state_seed"], num_states_, "state_seed");
    auto ref_arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
        sp["state_reference"]);
    require(bool(ref_arr) && ref_arr.ndim() == 2
                && static_cast<int>(ref_arr.shape(0)) == num_states_ && ref_arr.shape(1) == 4,
            "state_reference shape mismatch");
    state_reference_.resize(num_states_);
    for (int s = 0; s < num_states_; ++s) {
      for (int c = 0; c < 4; ++c) state_reference_[s][c] = ref_arr.at(s, c);
    }

    state_in_edges_.resize(num_states_);
    state_out_states_.resize(num_states_);
    group_states_.resize(num_slew_groups_);
    group_out_states_.resize(num_slew_groups_);
    for (int s = 0; s < num_states_; ++s) {
      group_states_[state_slew_group_[s]].push_back(s);
    }

    auto add_cell_edges = [&](const char* edge_key, const char* mask_key, IncomingEdge::Kind kind) {
      const auto edges = copyRows<3>(sp[edge_key], edge_key);
      const auto masks = copy1D<int64_t>(sp[mask_key], mask_key);
      require(edges.size() == masks.size(), std::string(edge_key) + " mask length mismatch");
      // Sort by destination level preserving original row order within each level,
      // matching SparseTimingSchedule._levels().
      std::vector<std::size_t> order(edges.size());
      for (std::size_t i = 0; i < edges.size(); ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return state_level_[edges[a][2]] < state_level_[edges[b][2]];
      });
      for (std::size_t idx : order) {
        const int arc_id = static_cast<int>(edges[idx][0]);
        const int src = static_cast<int>(edges[idx][1]);
        const int dst = static_cast<int>(edges[idx][2]);
        const int mask = static_cast<int>(masks[idx]);
        state_in_edges_[dst].push_back({kind, arc_id, src, dst, mask, {0.0, 0.0, 1.0, 1.0}});
        if (kind == IncomingEdge::Kind::kCell) {
          state_out_states_[src].push_back(dst);
        }
        group_out_states_[state_slew_group_[src]].push_back(dst);
        cell_arc_dst_states_[arc_id].push_back(dst);
      }
    };
    add_cell_edges("cell_edges", "cell_transition_mask", IncomingEdge::Kind::kCell);
    add_cell_edges("slew_only_cell_edges", "slew_only_cell_transition_mask",
                   IncomingEdge::Kind::kSlewOnlyCell);

    {
      const auto net_edges = copyRows<3>(sp["net_edges"], "net_edges");
      const auto net_masks = copy1D<int64_t>(sp["net_transition_mask"], "net_transition_mask");
      auto adj_arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          sp["net_threshold_adjustments"]);
      require(net_edges.size() == net_masks.size() && bool(adj_arr) && adj_arr.ndim() == 2
                  && static_cast<std::size_t>(adj_arr.shape(0)) == net_edges.size()
                  && adj_arr.shape(1) == 4,
              "net_edges arrays do not align");
      std::vector<std::size_t> order(net_edges.size());
      for (std::size_t i = 0; i < net_edges.size(); ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return state_level_[net_edges[a][2]] < state_level_[net_edges[b][2]];
      });
      for (std::size_t idx : order) {
        const int src = static_cast<int>(net_edges[idx][1]);
        const int dst = static_cast<int>(net_edges[idx][2]);
        const int mask = static_cast<int>(net_masks[idx]);
        std::array<double, 4> adj{adj_arr.at(idx, 0), adj_arr.at(idx, 1),
                                  adj_arr.at(idx, 2), adj_arr.at(idx, 3)};
        state_in_edges_[dst].push_back(
            {IncomingEdge::Kind::kNet, -1, src, dst, mask, adj});
        state_out_states_[src].push_back(dst);
        group_out_states_[state_slew_group_[src]].push_back(dst);
        pin_net_sink_states_[state_pin_[dst]].push_back(dst);
      }
    }

    const auto check_index = copyRows<7>(sp["check_index"], "check_index");
    auto check_vals = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
        sp["check_values"]);
    const auto legal_mask = copy1D<uint8_t>(sp["legal_checks"], "legal_checks");
    require(check_index.size() == legal_mask.size() && bool(check_vals)
                && static_cast<std::size_t>(check_vals.shape(0)) == check_index.size(),
            "sparse check arrays do not align");

    std::map<int, int> pin_to_ep;
    for (std::size_t c = 0; c < check_index.size(); ++c) {
      if (!legal_mask[c]) continue;
      const int st = static_cast<int>(check_index[c][0]);
      const int pin = state_pin_[st];
      auto [it, inserted] = pin_to_ep.emplace(pin, static_cast<int>(endpoint_pins_.size()));
      if (inserted) endpoint_pins_.push_back(pin);
    }
    // Match np.unique(endpoint_pin) sorted endpoint ordering.
    std::sort(endpoint_pins_.begin(), endpoint_pins_.end());
    pin_to_ep.clear();
    for (int e = 0; e < static_cast<int>(endpoint_pins_.size()); ++e) {
      pin_to_ep[endpoint_pins_[e]] = e;
    }
    num_endpoints_ = static_cast<int>(endpoint_pins_.size());
    endpoint_sparse_checks_.resize(num_endpoints_);
    state_endpoint_indices_.resize(num_states_);
    group_endpoint_indices_.resize(num_slew_groups_);

    for (std::size_t c = 0; c < check_index.size(); ++c) {
      if (!legal_mask[c]) continue;
      const int st = static_cast<int>(check_index[c][0]);
      const int check_row = static_cast<int>(check_index[c][2]);
      const int transition = static_cast<int>(check_index[c][3]);
      const int ep = pin_to_ep.at(state_pin_[st]);
      const int id = static_cast<int>(sparse_checks_.size());
      sparse_checks_.push_back(
          {st, check_row, transition, ep,
           check_vals.at(c, 0), check_vals.at(c, 1), check_vals.at(c, 2)});
      endpoint_sparse_checks_[ep].push_back(id);
      state_endpoint_indices_[st].push_back(ep);
      if (check_row >= 0) {
        require(check_row < static_cast<int>(check_arcs_.size()),
                "sparse check_index references out-of-range check_arcs row");
        group_endpoint_indices_[state_slew_group_[st]].push_back(ep);
        check_arc_endpoints_[check_row].push_back(ep);
      }
    }
  } else {
    // Schema 2 flat pin schedule.
    num_states_ = num_pins_;
    num_slew_groups_ = num_pins_;
    state_pin_.resize(num_pins_);
    state_slew_group_.resize(num_pins_);
    for (int p = 0; p < num_pins_; ++p) {
      state_pin_[p] = p;
      state_slew_group_[p] = p;
      pin_states_[p].push_back(p);
    }
    state_valid_.assign(num_pins_, {1, 1});
    state_seed_.assign(num_pins_, {0, 0});
    state_reference_.assign(num_pins_, {-2e8, -2e8, 0.0, 0.0});

    const auto start_pins = copy1D<int64_t>(config["start_pins"], "start_pins");
    const auto start_ra = copy1D<double>(config["start_rAAT"], "start_rAAT");
    const auto start_fa = copy1D<double>(config["start_fAAT"], "start_fAAT");
    const auto start_rs = copy1D<double>(config["start_rtran"], "start_rtran");
    const auto start_fs = copy1D<double>(config["start_ftran"], "start_ftran");
    for (std::size_t i = 0; i < start_pins.size(); ++i) {
      const int p = static_cast<int>(start_pins[i]);
      state_seed_[p] = {1, 1};
      state_reference_[p] = {start_ra[i], start_fa[i], start_rs[i], start_fs[i]};
    }

    const auto clock_pins = copy1D<int64_t>(config["clock_pins"], "clock_pins");
    clock_rtran_ = copy1D<double>(config["clock_rtran"], "clock_rtran");
    clock_ftran_ = copy1D<double>(config["clock_ftran"], "clock_ftran");

    state_in_edges_.resize(num_pins_);
    state_out_states_.resize(num_pins_);
    group_states_.resize(num_pins_);
    group_out_states_.resize(num_pins_);
    for (int p = 0; p < num_pins_; ++p) group_states_[p].push_back(p);

    for (int i = 0; i < clock_end_; ++i) {
      const int dst = static_cast<int>(cell_arcs_[i][1]);
      state_in_edges_[dst].push_back(
          {IncomingEdge::Kind::kClk2q, i, -1, dst, 15, {0.0, 0.0, 1.0, 1.0}});
      cell_arc_dst_states_[i].push_back(dst);
    }
    const auto net_arcs = copyRows<2>(config["net_arcs"], "net_arcs");
    for (int i = clock_end_; i < static_cast<int>(cell_arcs_.size()); ++i) {
      const int src = static_cast<int>(cell_arcs_[i][0]);
      const int dst = static_cast<int>(cell_arcs_[i][1]);
      state_in_edges_[dst].push_back(
          {IncomingEdge::Kind::kCell, i, src, dst, 15, {0.0, 0.0, 1.0, 1.0}});
      if (src != dst) {
        state_out_states_[src].push_back(dst);
        group_out_states_[src].push_back(dst);
      }
      cell_arc_dst_states_[i].push_back(dst);
    }
    for (const auto& na : net_arcs) {
      const int src = static_cast<int>(na[0]);
      const int dst = static_cast<int>(na[1]);
      state_in_edges_[dst].push_back(
          {IncomingEdge::Kind::kNet, -1, src, dst, 9, {0.0, 0.0, 1.0, 1.0}});
      if (src != dst) {
        state_out_states_[src].push_back(dst);
        group_out_states_[src].push_back(dst);
      }
      pin_net_sink_states_[dst].push_back(dst);
    }

    // Topological level assignment over the pin DAG.
    std::vector<int> indeg(num_pins_, 0);
    for (int u = 0; u < num_pins_; ++u) {
      for (int v : state_out_states_[u]) ++indeg[v];
    }
    state_level_.assign(num_pins_, 0);
    std::vector<int> queue;
    queue.reserve(num_pins_);
    for (int u = 0; u < num_pins_; ++u) {
      if (indeg[u] == 0) queue.push_back(u);
    }
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const int u = queue[head];
      for (int v : state_out_states_[u]) {
        state_level_[v] = std::max(state_level_[v], state_level_[u] + 1);
        if (--indeg[v] == 0) queue.push_back(v);
      }
    }
    max_level_ = 0;
    for (int l : state_level_) max_level_ = std::max(max_level_, l);

    // Flat endpoint setup + recovery structures matching PersistentTimingGraph.
    const auto ep_pins = copy1D<int64_t>(config["endpoint_pins"], "endpoint_pins");
    const auto ep_rr = copy1D<double>(config["endpoint_rRAT"], "endpoint_rRAT");
    const auto ep_fr = copy1D<double>(config["endpoint_fRAT"], "endpoint_fRAT");
    flat_endpoint_rRAT_.assign(num_pins_, 2e8);
    flat_endpoint_fRAT_.assign(num_pins_, 2e8);
    for (std::size_t i = 0; i < ep_pins.size(); ++i) {
      const int p = static_cast<int>(ep_pins[i]);
      flat_endpoint_rRAT_[p] = ep_rr[i];
      flat_endpoint_fRAT_[p] = ep_fr[i];
    }
    flat_endpoint_valid_ = copyBoolPairs(config["endpoint_valid"], num_pins_, "endpoint_valid");
    setup_valid_ = copyBoolPairs(config["setup_valid"], setup_arcs_.size(), "setup_valid");

    const std::string scope = config["metric_scope"].cast<std::string>();
    recovery_uses_live_slew_ =
        config["recovery_uses_live_slew"].cast<bool>() || scope != "setup_plus_recovery";
    recovery_rat_is_capture_deadline_ = config["recovery_rat_is_capture_deadline"].cast<bool>();

    std::vector<uint8_t> has_setup(num_pins_, 0), has_recovery(num_pins_, 0);
    for (const auto& arc : setup_arcs_) has_setup[arc[1]] = 1;

    std::vector<double> clk_r_by_pin(num_pins_, 0.0), clk_f_by_pin(num_pins_, 0.0);
    for (std::size_t i = 0; i < clock_pins.size(); ++i) {
      clk_r_by_pin[clock_pins[i]] = clock_rtran_[i];
      clk_f_by_pin[clock_pins[i]] = clock_ftran_[i];
    }
    if (recovery_rat_is_capture_deadline_) {
      recovery_deadline_r_ = flat_endpoint_rRAT_;
      recovery_deadline_f_ = flat_endpoint_fRAT_;
      recovery_keep_setup_.assign(num_pins_, {0, 0});
      for (std::size_t i = 0; i < setup_arcs_.size(); ++i) {
        const int p = static_cast<int>(setup_arcs_[i][1]);
        recovery_keep_setup_[p][0] |= setup_valid_[i][0];
        recovery_keep_setup_[p][1] |= setup_valid_[i][1];
      }
    }
    const auto check_valid = copyBoolPairs(config["check_valid"], check_arcs_.size(), "check_valid");
    std::vector<double> backend_ref_r(num_pins_, 0.0), backend_ref_f(num_pins_, 0.0);
    if (!recovery_uses_live_slew_ && !recovery_rat_is_capture_deadline_
        && !config["backend_rSlew"].is_none() && !config["backend_fSlew"].is_none()) {
      const auto br = copy1D<double>(config["backend_rSlew"], "backend_rSlew");
      const auto bf = copy1D<double>(config["backend_fSlew"], "backend_fSlew");
      for (std::size_t i = 0; i < ep_pins.size() && i < br.size(); ++i) {
        backend_ref_r[ep_pins[i]] = br[i];
        backend_ref_f[ep_pins[i]] = bf[i];
      }
    }
    for (std::size_t i = 0; i < check_arcs_.size(); ++i) {
      const auto& arc = check_arcs_[i];
      if (arc[6] == 1) has_setup[arc[1]] = 1;
      if (arc[6] != 3 || !(check_valid[i][0] || check_valid[i][1])) continue;
      has_recovery[arc[1]] = 1;
      const double cslew = arc[5] < 0 ? clk_f_by_pin[arc[0]] : clk_r_by_pin[arc[0]];
      recovery_clock_rtran_.push_back(cslew);
      recovery_clock_ftran_.push_back(cslew);
      recovery_ref_rtran_.push_back(backend_ref_r[arc[1]]);
      recovery_ref_ftran_.push_back(backend_ref_f[arc[1]]);
      recovery_check_rows_.push_back(static_cast<int>(i));
      recovery_valid_.push_back(check_valid[i]);
    }

    pin_to_flat_endpoint_.assign(num_pins_, -1);
    for (int64_t id : ep_pins) {
      const int p = static_cast<int>(id);
      if (scope == "setup_plus_recovery" || !has_recovery[p] || has_setup[p]) {
        if (pin_to_flat_endpoint_[p] < 0) {
          pin_to_flat_endpoint_[p] = static_cast<int>(endpoint_pins_.size());
          endpoint_pins_.push_back(p);
        }
      }
    }
    num_endpoints_ = static_cast<int>(endpoint_pins_.size());
    endpoint_setup_rows_.resize(num_endpoints_);
    endpoint_recovery_rows_.resize(num_endpoints_);
    state_endpoint_indices_.resize(num_states_);
    group_endpoint_indices_.resize(num_slew_groups_);
    for (int row = 0; row < static_cast<int>(setup_arcs_.size()); ++row) {
      const int ep = pin_to_flat_endpoint_[setup_arcs_[row][1]];
      if (ep >= 0) {
        endpoint_setup_rows_[ep].push_back(row);
        setup_arc_endpoints_[row].push_back(ep);
      }
    }
    for (int row = 0; row < static_cast<int>(recovery_check_rows_.size()); ++row) {
      const int check_row = recovery_check_rows_[row];
      const int ep = pin_to_flat_endpoint_[check_arcs_[check_row][1]];
      if (ep >= 0) {
        endpoint_recovery_rows_[ep].push_back(row);
        check_arc_endpoints_[check_row].push_back(ep);
      }
    }
    for (int e = 0; e < num_endpoints_; ++e) {
      const int p = endpoint_pins_[e];
      state_endpoint_indices_[p].push_back(e);
      group_endpoint_indices_[p].push_back(e);
    }
    state_source_type_.assign(num_states_, 1);
    state_role_.assign(num_states_, 0);
  }

  if (config.contains("net_driver_pin") && !config["net_driver_pin"].is_none()) {
    net_driver_pin_ = copy1D<int>(config["net_driver_pin"], "net_driver_pin");
    require(static_cast<int>(net_driver_pin_.size()) == num_nets_,
            "net_driver_pin length mismatch");
  } else {
    net_driver_pin_.assign(num_nets_, -1);
  }
  for (int s = 0; s < num_states_; ++s) {
    for (const auto& edge : state_in_edges_[s]) {
      if (edge.kind == IncomingEdge::Kind::kNet && edge.src_state >= 0) {
        const int drv_pin = state_pin_[edge.src_state];
        const int net = pin_net_[drv_pin];
        if (net >= 0 && net < num_nets_ && net_driver_pin_[net] < 0) {
          net_driver_pin_[net] = drv_pin;
        }
      }
    }
  }
  for (int net = 0; net < num_nets_; ++net) {
    if (net_driver_pin_[net] >= 0) continue;
    const int t = rc_forest_.tree_by_net[net];
    if (t >= 0 && t < static_cast<int>(rc_forest_.net_topo.size())
        && !rc_forest_.net_topo[t].empty()) {
      const int root_v = rc_forest_.net_topo[t][0];
      for (int pin : net_pins_[net]) {
        if (rc_forest_.pin_to_vertex[pin] == root_v) {
          net_driver_pin_[net] = pin;
          if (pin_values_[0][pin] == 0.0) break;
        }
      }
    }
  }

  auto dedup = [](std::vector<std::vector<int>>& lists) {
    for (auto& v : lists) {
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    }
  };
  dedup(state_out_states_);
  dedup(group_out_states_);
  dedup(cell_arc_dst_states_);
  dedup(setup_arc_endpoints_);
  dedup(check_arc_endpoints_);
  dedup(pin_net_sink_states_);
  dedup(state_endpoint_indices_);
  dedup(group_endpoint_indices_);

  constexpr int kLevelScale = 65536;
  for (int& lvl : state_level_) lvl *= kLevelScale;
  max_level_ *= kLevelScale;

  group_causal_.assign(num_slew_groups_, 1);
  for (int g = 0; g < num_slew_groups_; ++g) {
    int max_in = -1;
    for (int u : group_states_[g]) max_in = std::max(max_in, state_level_[u]);
    for (int v : group_out_states_[g]) {
      if (max_in >= state_level_[v]) {
        group_causal_[g] = 0;
        break;
      }
    }
  }

  dirty_state_mask_.assign(num_states_, 0);
  dirty_group_mask_.assign(num_slew_groups_, 0);
  touched_endpoint_mask_.assign(num_endpoints_, 0);
  propagateFull();
}

int IncrementalSizingEngine::mapLibertyArc(
    int source_cell, int target_cell, int old_arc_id) const
{
  if (source_cell == target_cell) return old_arc_id;
  require(source_cell >= 0 && source_cell < num_cells_
              && target_cell >= 0 && target_cell < num_cells_,
          "mapLibertyArc cell out of range");
  const int64_t src_begin = cell_arc_start_[source_cell];
  const int64_t src_end = cell_arc_start_[source_cell + 1];
  require(old_arc_id >= src_begin && old_arc_id < src_end,
          "runtime arc does not belong to source cell");
  const auto& sig = lib_arc_sigs_[old_arc_id];
  int rank = 0;
  for (int64_t a = src_begin; a < old_arc_id; ++a) {
    if (lib_arc_sigs_[a] == sig) ++rank;
  }
  const int64_t dst_begin = cell_arc_start_[target_cell];
  const int64_t dst_end = cell_arc_start_[target_cell + 1];
  int match_count = 0;
  int last_match = -1;
  for (int64_t a = dst_begin; a < dst_end; ++a) {
    if (lib_arc_sigs_[a] == sig) {
      last_match = static_cast<int>(a);
      if (match_count == rank) return last_match;
      ++match_count;
    }
  }
  require(last_match >= 0, "target Liberty cell is missing timing arc signature");
  return last_match;
}

void IncrementalSizingEngine::recomputeNetRc(
    int net_id, bool record_undo, std::vector<int>* changed_rc_pins)
{
  if (net_id < 0 || net_id >= num_nets_) return;
  const auto& pins = net_pins_[net_id];
  if (pins.empty()) return;
  const int tree_idx = rc_forest_.tree_by_net[net_id];

  auto write_pin_rc = [&](int pin, double lr, double lf, double dr, double df,
                          double ir, double jf) {
    const std::array<double, 6> prev{
        pin_load_r_[pin], pin_load_f_[pin], pin_delay_r_[pin],
        pin_delay_f_[pin], pin_impulse_r_[pin], pin_impulse_f_[pin]};
    const std::array<double, 6> next{lr, lf, dr, df, ir, jf};
    bool any_changed = false;
    for (int k = 0; k < 6; ++k) {
      if (bitsChanged(prev[k], next[k])) {
        any_changed = true;
        break;
      }
    }
    if (!any_changed) return;
    if (record_undo) undo_pin_rc_.emplace_back(pin, prev);
    pin_load_r_[pin] = lr;
    pin_load_f_[pin] = lf;
    pin_delay_r_[pin] = dr;
    pin_delay_f_[pin] = df;
    pin_impulse_r_[pin] = ir;
    pin_impulse_f_[pin] = jf;
    if (changed_rc_pins != nullptr) changed_rc_pins->push_back(pin);
  };

  if (tree_idx < 0 || tree_idx >= static_cast<int>(rc_forest_.net_topo.size())
      || rc_forest_.net_topo[tree_idx].empty()) {
    for (int pin : pins) {
      write_pin_rc(pin, pin_values_[1][pin], pin_values_[2][pin], 0.0, 0.0, 0.0, 0.0);
    }
    return;
  }

  const auto& topo = rc_forest_.net_topo[tree_idx];
  const int n_topo = static_cast<int>(topo.size());
  if (static_cast<int>(rc_forest_.scratch_cap.size()) < rc_forest_.vertex_count) {
    rc_forest_.scratch_cap.resize(rc_forest_.vertex_count, 0.0);
    rc_forest_.scratch_load.resize(rc_forest_.vertex_count, 0.0);
    rc_forest_.scratch_delay.resize(rc_forest_.vertex_count, 0.0);
    rc_forest_.scratch_ldelay.resize(rc_forest_.vertex_count, 0.0);
    rc_forest_.scratch_beta.resize(rc_forest_.vertex_count, 0.0);
  }
  auto& cap = rc_forest_.scratch_cap;
  auto& load = rc_forest_.scratch_load;
  auto& delay = rc_forest_.scratch_delay;
  auto& ldelay = rc_forest_.scratch_ldelay;
  auto& beta = rc_forest_.scratch_beta;
  const int root = topo[0];

  std::vector<std::array<double, 3>> rise_out(pins.size());
  for (int mode = 0; mode < 2; ++mode) {
    const auto& pin_cap = pin_values_[mode + 1];
    for (int k = 0; k < n_topo; ++k) {
      const int v = topo[k];
      cap[v] = rc_forest_.wire_cap[v];
    }
    for (int pin : pins) {
      const int v = rc_forest_.pin_to_vertex[pin];
      if (v >= 0) cap[v] += pin_cap[pin];
    }
    for (int k = n_topo - 1; k >= 0; --k) {
      const int v = topo[k];
      double s = cap[v];
      for (int ch : rc_forest_.adj_children[v]) s += load[ch];
      load[v] = s;
    }
    delay[root] = 0.0;
    for (int k = 1; k < n_topo; ++k) {
      const int v = topo[k];
      const int p = rc_forest_.parent[v];
      delay[v] = (p >= 0 && p != v ? delay[p] : 0.0)
                 + rc_forest_.incoming_resistance[v] * load[v];
    }
    for (int k = n_topo - 1; k >= 0; --k) {
      const int v = topo[k];
      double s = cap[v] * delay[v];
      for (int ch : rc_forest_.adj_children[v]) s += ldelay[ch];
      ldelay[v] = s;
    }
    beta[root] = 0.0;
    for (int k = 1; k < n_topo; ++k) {
      const int v = topo[k];
      const int p = rc_forest_.parent[v];
      beta[v] = (p >= 0 && p != v ? beta[p] : 0.0)
                + rc_forest_.incoming_resistance[v] * ldelay[v];
    }

    for (std::size_t idx = 0; idx < pins.size(); ++idx) {
      const int pin = pins[idx];
      const int v = rc_forest_.pin_to_vertex[pin];
      double l_val = pin_cap[pin];
      double d_val = 0.0;
      double i_val = 0.0;
      if (v >= 0) {
        l_val = load[v];
        d_val = delay[v];
        i_val = std::max(0.0, 2.0 * beta[v] - d_val * d_val);
      }
      if (mode == 0) {
        rise_out[idx] = {l_val, d_val, i_val};
      } else {
        write_pin_rc(pin, rise_out[idx][0], l_val, rise_out[idx][1], d_val,
                     rise_out[idx][2], i_val);
      }
    }
  }
}

std::pair<double, double> IncrementalSizingEngine::sourceSlewForEdge(
    int src_state, int dst_level) const
{
  const int g = state_slew_group_[src_state];
  if (group_causal_[g]) return {group_rs_[g], group_fs_[g]};
  double rs = 0.0, fs = 0.0;
  for (int u : group_states_[g]) {
    if (state_level_[u] < dst_level) {
      timing::hardMaximum(rs, state_local_rs_[u]);
      timing::hardMaximum(fs, state_local_fs_[u]);
    }
  }
  return {rs, fs};
}

bool IncrementalSizingEngine::recomputeState(
    int state_id, bool record_undo, bool* local_slew_changed)
{
  const double unreachable = -2e8;
  const int dst_level = state_level_[state_id];
  const int out_pin = state_pin_[state_id];
  const double init_ra = state_seed_[state_id][0] ? state_reference_[state_id][0] : unreachable;
  const double init_fa = state_seed_[state_id][1] ? state_reference_[state_id][1] : unreachable;
  double new_rs = state_seed_[state_id][0] ? state_reference_[state_id][2] : 0.0;
  double new_fs = state_seed_[state_id][1] ? state_reference_[state_id][3] : 0.0;

  double new_ra = init_ra;
  double new_fa = init_fa;
  std::vector<double> smooth_ra, smooth_fa;
  if (smooth_aggregation_) {
    smooth_ra.reserve(state_in_edges_[state_id].size());
    smooth_fa.reserve(state_in_edges_[state_id].size());
  }

  const auto rd = luts_[0].view(), fd = luts_[1].view();
  const auto rt = luts_[2].view(), ft = luts_[3].view();

  for (const auto& edge : state_in_edges_[state_id]) {
    double v[4] = {unreachable, unreachable, 0.0, 0.0};
    if (edge.kind == IncomingEdge::Kind::kClk2q) {
      clk2qSingleRow(cell_arcs_[edge.arc_row], pin_load_r_[out_pin], pin_load_f_[out_pin],
                     clock_rtran_, clock_ftran_, rd, fd, rt, ft,
                     pairwise_smooth_, pairwise_alpha_, v);
    } else if (edge.kind == IncomingEdge::Kind::kCell
               || edge.kind == IncomingEdge::Kind::kSlewOnlyCell) {
      const auto& arc = cell_arcs_[edge.arc_row];
      const auto [in_rs, in_fs] = sourceSlewForEdge(edge.src_state, dst_level);
      const double in_ra[1] = {state_ra_[edge.src_state]};
      const double in_fa[1] = {state_fa_[edge.src_state]};
      const double in_rs_arr[1] = {in_rs};
      const double in_fs_arr[1] = {in_fs};
      cellValueRow4AtLoad<double>(
          0, 0, arc[3], arc[4], 1, in_ra, in_fa, in_rs_arr, in_fs_arr,
          pin_load_r_[out_pin], pin_load_f_[out_pin], rd, fd, rt, ft,
          v, pairwise_smooth_, pairwise_alpha_, edge.mask);
      if (edge.kind == IncomingEdge::Kind::kCell) {
        const bool ok_r = state_valid_[state_id][0] && ((edge.mask & 5) != 0);
        const bool ok_f = state_valid_[state_id][1] && ((edge.mask & 10) != 0);
        if (!ok_r) v[0] = unreachable;
        if (!ok_f) v[1] = unreachable;
      }
    } else {
      const auto [in_rs, in_fs] = sourceSlewForEdge(edge.src_state, dst_level);
      const int64_t pair[2] = {0, 1};
      const double ra_pair[2] = {state_ra_[edge.src_state], 0.0};
      const double fa_pair[2] = {state_fa_[edge.src_state], 0.0};
      const double rs_pair[2] = {in_rs, 0.0};
      const double fs_pair[2] = {in_fs, 0.0};
      const double rd_pair[2] = {0.0, pin_delay_r_[out_pin]};
      const double fd_pair[2] = {0.0, pin_delay_f_[out_pin]};
      const double ri_pair[2] = {0.0, pin_impulse_r_[out_pin]};
      const double fi_pair[2] = {0.0, pin_impulse_f_[out_pin]};
      double raw[4], jac[6];
      netForwardRow<double>(0, pair, 2, 1, 2, ra_pair, fa_pair, rs_pair, fs_pair,
                            rd_pair, fd_pair, ri_pair, fi_pair, raw, jac, slew_epsilon_);
      v[0] = raw[0] + raw[2] * edge.net_adj[0];
      v[1] = raw[1] + raw[3] * edge.net_adj[1];
      v[2] = raw[2] * edge.net_adj[2];
      v[3] = raw[3] * edge.net_adj[3];
      const bool ok_r = state_valid_[state_id][0] && ((edge.mask & 1) != 0);
      const bool ok_f = state_valid_[state_id][1] && ((edge.mask & 8) != 0);
      if (!ok_r) v[0] = unreachable;
      if (!ok_f) v[1] = unreachable;
    }

    if (edge.kind != IncomingEdge::Kind::kSlewOnlyCell) {
      if (smooth_aggregation_) {
        smooth_ra.push_back(v[0]);
        smooth_fa.push_back(v[1]);
      } else {
        timing::hardMaximum(new_ra, v[0]);
        timing::hardMaximum(new_fa, v[1]);
      }
    }
    timing::hardMaximum(new_rs, v[2]);
    timing::hardMaximum(new_fs, v[3]);
  }

  if (smooth_aggregation_ && !smooth_ra.empty()) {
    new_ra = smoothMergeValues(init_ra, smooth_ra.data(), smooth_ra.size(), aggregation_tau_ps_);
    new_fa = smoothMergeValues(init_fa, smooth_fa.data(), smooth_fa.size(), aggregation_tau_ps_);
  }

  const bool arr_changed =
      bitsChanged(state_ra_[state_id], new_ra) || bitsChanged(state_fa_[state_id], new_fa);
  if (arr_changed) {
    if (record_undo) {
      undo_state_arr_.push_back({state_id, {state_ra_[state_id], state_fa_[state_id]}});
    }
    state_ra_[state_id] = new_ra;
    state_fa_[state_id] = new_fa;
  }

  const bool slew_changed = bitsChanged(state_local_rs_[state_id], new_rs)
                            || bitsChanged(state_local_fs_[state_id], new_fs);
  if (slew_changed) {
    if (record_undo) {
      undo_state_slew_.push_back(
          {state_id, {state_local_rs_[state_id], state_local_fs_[state_id]}});
    }
    state_local_rs_[state_id] = new_rs;
    state_local_fs_[state_id] = new_fs;
  }
  if (local_slew_changed != nullptr) *local_slew_changed = slew_changed;
  return arr_changed;
}

bool IncrementalSizingEngine::recomputeSlewGroup(
    int group_id, bool record_undo, std::vector<int>* changed_slew_pins)
{
  double new_rs = 0.0, new_fs = 0.0;
  for (int u : group_states_[group_id]) {
    timing::hardMaximum(new_rs, state_local_rs_[u]);
    timing::hardMaximum(new_fs, state_local_fs_[u]);
  }
  const bool changed =
      bitsChanged(group_rs_[group_id], new_rs) || bitsChanged(group_fs_[group_id], new_fs);
  if (!changed) return false;
  if (record_undo) {
    undo_group_slew_.push_back({group_id, {group_rs_[group_id], group_fs_[group_id]}});
  }
  group_rs_[group_id] = new_rs;
  group_fs_[group_id] = new_fs;

  // Update physical pin slew for all pins represented in this slew group.
  int last_pin = -1;
  for (int u : group_states_[group_id]) {
    const int pin = state_pin_[u];
    if (pin == last_pin) continue;
    last_pin = pin;
    double prs = 0.0, pfs = 0.0;
    for (int s : pin_states_[pin]) {
      timing::hardMaximum(prs, group_rs_[state_slew_group_[s]]);
      timing::hardMaximum(pfs, group_fs_[state_slew_group_[s]]);
    }
    if (bitsChanged(pin_rs_[pin], prs) || bitsChanged(pin_fs_[pin], pfs)) {
      if (record_undo) {
        undo_pin_slew_.push_back({pin, {pin_rs_[pin], pin_fs_[pin]}});
      }
      pin_rs_[pin] = prs;
      pin_fs_[pin] = pfs;
      if (changed_slew_pins != nullptr) changed_slew_pins->push_back(pin);
    }
  }
  return true;
}

double IncrementalSizingEngine::computeEndpointSlack(int endpoint_idx) const
{
  if (sparse_mode_) {
    const auto& rows = endpoint_sparse_checks_[endpoint_idx];
    const auto r_check = luts_[4].view();
    const auto f_check = luts_[5].view();
    if (!smooth_aggregation_) {
      double max_viol = -2e8;
      for (int id : rows) {
        const auto& c = sparse_checks_[id];
        double lut_val = 0.0;
        if (c.check_arc_row >= 0) {
          const int64_t lut_row = check_arcs_[c.check_arc_row][3];
          const int g = state_slew_group_[c.state];
          const double data_slew = c.transition == 0 ? group_rs_[g] : group_fs_[g];
          lut_val = timingLutQuery(
                        c.transition == 0 ? r_check : f_check,
                        lut_row, c.clock_slew, data_slew)
                        .value;
        }
        const double req = c.rule_offset + c.rule_sign * lut_val;
        const double arr = c.transition == 0 ? state_ra_[c.state] : state_fa_[c.state];
        const double slack = req - arr;
        timing::hardMaximum(max_viol, -slack);
      }
      return -max_viol;
    }
    double max_viol = -std::numeric_limits<double>::infinity();
    std::vector<double> neg_slacks(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto& c = sparse_checks_[rows[i]];
      double lut_val = 0.0;
      if (c.check_arc_row >= 0) {
        const int64_t lut_row = check_arcs_[c.check_arc_row][3];
        const int g = state_slew_group_[c.state];
        const double data_slew = c.transition == 0 ? group_rs_[g] : group_fs_[g];
        lut_val = timingLutQuery(
                      c.transition == 0 ? r_check : f_check,
                      lut_row, c.clock_slew, data_slew)
                      .value;
      }
      const double req = c.rule_offset + c.rule_sign * lut_val;
      const double arr = c.transition == 0 ? state_ra_[c.state] : state_fa_[c.state];
      neg_slacks[i] = -(req - arr);
      max_viol = std::max(max_viol, neg_slacks[i]);
    }
    if (!std::isfinite(max_viol)) return -max_viol;
    double total = 0.0;
    for (double ns : neg_slacks) {
      total += std::exp((ns - max_viol) / aggregation_tau_ps_);
    }
    return -(max_viol + aggregation_tau_ps_ * std::log(std::max(total, 1.0e-30)));
  }

  // Schema 2 flat endpoint reduction.
  const int pin = endpoint_pins_[endpoint_idx];
  double rr = flat_endpoint_rRAT_[pin];
  double fr = flat_endpoint_fRAT_[pin];
  const double seed_rr = rr, seed_fr = fr;
  const auto rd = luts_[0].view(), fd = luts_[1].view();
  for (int srow : endpoint_setup_rows_[endpoint_idx]) {
    const auto cand = timing::setupRatRow<double>(
        setup_arcs_[srow].data(), seed_rr, seed_fr,
        clock_rtran_.data(), clock_ftran_.data(),
        pin_rs_.data(), pin_fs_.data(), rd, fd, pairwise_smooth_, pairwise_alpha_);
    if (setup_valid_[srow][0]) rr = timing::hardMinimum(rr, cand.first);
    if (setup_valid_[srow][1]) fr = timing::hardMinimum(fr, cand.second);
  }
  const auto& rec_rows = endpoint_recovery_rows_[endpoint_idx];
  if (!rec_rows.empty()) {
    const auto rc_lut = luts_[4].view(), fc_lut = luts_[5].view();
    bool touched_r = false, touched_f = false;
    const double base_rr = rr, base_fr = fr;
    for (int rrow : rec_rows) {
      const int64_t lrow = check_arcs_[recovery_check_rows_[rrow]][3];
      const double ref_r = recovery_uses_live_slew_ ? pin_rs_[pin] : recovery_ref_rtran_[rrow];
      const double ref_f = recovery_uses_live_slew_ ? pin_fs_[pin] : recovery_ref_ftran_[rrow];
      const double cap_r = recovery_rat_is_capture_deadline_
                               ? 0.0
                               : timingLutQuery(rc_lut, lrow, recovery_clock_rtran_[rrow], ref_r).value;
      const double cap_f = recovery_rat_is_capture_deadline_
                               ? 0.0
                               : timingLutQuery(fc_lut, lrow, recovery_clock_ftran_[rrow], ref_f).value;
      const double anc_r = recovery_rat_is_capture_deadline_ ? recovery_deadline_r_[pin] : base_rr;
      const double anc_f = recovery_rat_is_capture_deadline_ ? recovery_deadline_f_[pin] : base_fr;
      const double live_r =
          timingLutQuery(rc_lut, lrow, recovery_clock_rtran_[rrow], pin_rs_[pin]).value;
      const double live_f =
          timingLutQuery(fc_lut, lrow, recovery_clock_ftran_[rrow], pin_fs_[pin]).value;
      if (recovery_valid_[rrow][0]) {
        const double cand = anc_r + cap_r - live_r;
        if (!touched_r && !(recovery_rat_is_capture_deadline_ && recovery_keep_setup_[pin][0])) {
          rr = cand;
          touched_r = true;
        } else {
          rr = timing::hardMinimum(rr, cand);
        }
      }
      if (recovery_valid_[rrow][1]) {
        const double cand = anc_f + cap_f - live_f;
        if (!touched_f && !(recovery_rat_is_capture_deadline_ && recovery_keep_setup_[pin][1])) {
          fr = cand;
          touched_f = true;
        } else {
          fr = timing::hardMinimum(fr, cand);
        }
      }
    }
  }
  const double inf = std::numeric_limits<double>::infinity();
  return timing::hardMinimum(
      flat_endpoint_valid_[pin][0] ? rr - state_ra_[pin] : inf,
      flat_endpoint_valid_[pin][1] ? fr - state_fa_[pin] : inf);
}

void IncrementalSizingEngine::propagateFull()
{
  state_ra_.assign(num_states_, -2e8);
  state_fa_.assign(num_states_, -2e8);
  state_local_rs_.assign(num_states_, 0.0);
  state_local_fs_.assign(num_states_, 0.0);
  group_rs_.assign(num_slew_groups_, 0.0);
  group_fs_.assign(num_slew_groups_, 0.0);
  pin_rs_.assign(num_pins_, 0.0);
  pin_fs_.assign(num_pins_, 0.0);

  std::vector<int> state_order(num_states_);
  for (int s = 0; s < num_states_; ++s) state_order[s] = s;
  std::stable_sort(state_order.begin(), state_order.end(), [&](int a, int b) {
    return state_level_[a] < state_level_[b];
  });
  std::vector<int> dirty_groups;
  std::size_t idx = 0;
  while (idx < state_order.size()) {
    const int lvl = state_level_[state_order[idx]];
    dirty_groups.clear();
    while (idx < state_order.size() && state_level_[state_order[idx]] == lvl) {
      const int s = state_order[idx++];
      bool slew_ch = false;
      recomputeState(s, false, &slew_ch);
      const int g = state_slew_group_[s];
      if (!dirty_group_mask_[g]) {
        dirty_group_mask_[g] = 1;
        dirty_groups.push_back(g);
      }
    }
    for (int g : dirty_groups) {
      dirty_group_mask_[g] = 0;
      recomputeSlewGroup(g, false, nullptr);
    }
  }

  endpoint_slacks_.resize(num_endpoints_);
  for (int e = 0; e < num_endpoints_; ++e) {
    endpoint_slacks_[e] = computeEndpointSlack(e);
  }
  clearUndo();
}

void IncrementalSizingEngine::propagateEvents(
    const std::vector<int>& seed_states,
    std::vector<int>* changed_slew_pins,
    std::vector<int>* touched_endpoints)
{
  if (seed_states.empty()) return;
  std::map<int, std::vector<int>> level_buckets;
  auto enqueue_state = [&](int s) {
    if (!dirty_state_mask_[s]) {
      dirty_state_mask_[s] = 1;
      level_buckets[state_level_[s]].push_back(s);
    }
  };
  auto mark_endpoint = [&](int ep) {
    if (touched_endpoints != nullptr && !touched_endpoint_mask_[ep]) {
      touched_endpoint_mask_[ep] = 1;
      touched_endpoints->push_back(ep);
    }
  };

  for (int s : seed_states) enqueue_state(s);

  std::vector<int> level_groups;
  for (auto it = level_buckets.begin(); it != level_buckets.end(); ++it) {
    level_groups.clear();
    const auto& states = it->second;
    for (int s : states) {
      dirty_state_mask_[s] = 0;
      bool local_slew_changed = false;
      const bool arr_changed = recomputeState(s, true, &local_slew_changed);
      if (arr_changed) {
        for (int dst : state_out_states_[s]) enqueue_state(dst);
        for (int ep : state_endpoint_indices_[s]) mark_endpoint(ep);
      }
      if (local_slew_changed) {
        const int g = state_slew_group_[s];
        if (!dirty_group_mask_[g]) {
          dirty_group_mask_[g] = 1;
          level_groups.push_back(g);
        }
      }
    }
    for (int g : level_groups) {
      dirty_group_mask_[g] = 0;
      if (recomputeSlewGroup(g, true, changed_slew_pins)) {
        for (int dst : group_out_states_[g]) enqueue_state(dst);
        for (int ep : group_endpoint_indices_[g]) mark_endpoint(ep);
      }
    }
  }
}

IncrementalSizingEngine::UndoCheckpoint IncrementalSizingEngine::captureUndo() const
{
  return {undo_cells_.size(),
          undo_cell_arcs_.size(),
          undo_setup_arcs_.size(),
          undo_check_arcs_.size(),
          undo_pin_vals_.size(),
          undo_pin_rc_.size(),
          undo_state_arr_.size(),
          undo_state_slew_.size(),
          undo_group_slew_.size(),
          undo_pin_slew_.size(),
          undo_endpoints_.size()};
}

void IncrementalSizingEngine::restoreUndo(const UndoCheckpoint& cp)
{
  while (undo_endpoints_.size() > cp.endpoint_pos) {
    const auto& [ep, val] = undo_endpoints_.back();
    endpoint_slacks_[ep] = val;
    undo_endpoints_.pop_back();
  }
  while (undo_pin_slew_.size() > cp.pin_slew_pos) {
    const auto& [p, val] = undo_pin_slew_.back();
    pin_rs_[p] = val[0];
    pin_fs_[p] = val[1];
    undo_pin_slew_.pop_back();
  }
  while (undo_group_slew_.size() > cp.group_slew_pos) {
    const auto& [g, val] = undo_group_slew_.back();
    group_rs_[g] = val[0];
    group_fs_[g] = val[1];
    undo_group_slew_.pop_back();
  }
  while (undo_state_slew_.size() > cp.state_slew_pos) {
    const auto& [s, val] = undo_state_slew_.back();
    state_local_rs_[s] = val[0];
    state_local_fs_[s] = val[1];
    undo_state_slew_.pop_back();
  }
  while (undo_state_arr_.size() > cp.state_arr_pos) {
    const auto& [s, val] = undo_state_arr_.back();
    state_ra_[s] = val[0];
    state_fa_[s] = val[1];
    undo_state_arr_.pop_back();
  }
  while (undo_pin_rc_.size() > cp.pin_rc_pos) {
    const auto& [p, val] = undo_pin_rc_.back();
    pin_load_r_[p] = val[0];
    pin_load_f_[p] = val[1];
    pin_delay_r_[p] = val[2];
    pin_delay_f_[p] = val[3];
    pin_impulse_r_[p] = val[4];
    pin_impulse_f_[p] = val[5];
    undo_pin_rc_.pop_back();
  }
  while (undo_pin_vals_.size() > cp.pin_val_pos) {
    const auto& [p, val] = undo_pin_vals_.back();
    for (int lane = 0; lane < 5; ++lane) pin_values_[lane][p] = val[lane];
    undo_pin_vals_.pop_back();
  }
  while (undo_check_arcs_.size() > cp.check_arc_pos) {
    const auto& [id, row] = undo_check_arcs_.back();
    check_arcs_[id] = row;
    undo_check_arcs_.pop_back();
  }
  while (undo_setup_arcs_.size() > cp.setup_arc_pos) {
    const auto& [id, row] = undo_setup_arcs_.back();
    setup_arcs_[id] = row;
    undo_setup_arcs_.pop_back();
  }
  while (undo_cell_arcs_.size() > cp.cell_arc_pos) {
    const auto& [id, row] = undo_cell_arcs_.back();
    cell_arcs_[id] = row;
    undo_cell_arcs_.pop_back();
  }
  while (undo_cells_.size() > cp.cell_pos) {
    const auto& [node, cell] = undo_cells_.back();
    inst_cell_id_[node] = cell;
    undo_cells_.pop_back();
  }
}

void IncrementalSizingEngine::clearUndo()
{
  undo_cells_.clear();
  undo_cell_arcs_.clear();
  undo_setup_arcs_.clear();
  undo_check_arcs_.clear();
  undo_pin_vals_.clear();
  undo_pin_rc_.clear();
  undo_state_arr_.clear();
  undo_state_slew_.clear();
  undo_group_slew_.clear();
  undo_pin_slew_.clear();
  undo_endpoints_.clear();
}

void IncrementalSizingEngine::remapNodeArcs(
    int node, int target, bool record_undo,
    std::vector<int>* seed_states,
    std::vector<int>* touched_endpoints)
{
  const int64_t cell_base = cell_arc_start_[target];
  auto mark_endpoint = [&](int ep) {
    if (touched_endpoints != nullptr && !touched_endpoint_mask_[ep]) {
      touched_endpoint_mask_[ep] = 1;
      touched_endpoints->push_back(ep);
    }
  };
  auto remap_row = [&](ArcRow& row, int offset_col) {
    const int lut = mapLibertyArc(static_cast<int>(row[2]), target, static_cast<int>(row[3]));
    row[2] = target;
    row[3] = lut;
    row[4] = lib_arc_sigs_[lut][2];
    row[5] = lib_arc_sigs_[lut][3];
    row[offset_col] = lut - cell_base;
  };

  for (int id : node_cell_arcs_[node]) {
    auto& row = cell_arcs_[id];
    const ArcRow prev = row;
    remap_row(row, 6);
    if (record_undo) undo_cell_arcs_.emplace_back(id, prev);
    if (seed_states != nullptr) {
      for (int s : cell_arc_dst_states_[id]) seed_states->push_back(s);
    }
  }
  for (int id : node_setup_arcs_[node]) {
    auto& row = setup_arcs_[id];
    const ArcRow prev = row;
    remap_row(row, 7);
    if (record_undo) undo_setup_arcs_.emplace_back(id, prev);
    for (int ep : setup_arc_endpoints_[id]) mark_endpoint(ep);
  }
  for (int id : node_check_arcs_[node]) {
    auto& row = check_arcs_[id];
    const ArcRow prev = row;
    remap_row(row, 7);
    if (record_undo) undo_check_arcs_.emplace_back(id, prev);
    for (int ep : check_arc_endpoints_[id]) mark_endpoint(ep);
  }
}

void IncrementalSizingEngine::syncState(
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& inst_cell_ids,
    const py::object& wire_cap,
    const py::object& incoming_resistance)
{
  require(inst_cell_ids.ndim() == 1
              && static_cast<int>(inst_cell_ids.shape(0)) == num_instances_,
          "syncState inst_cell_ids length mismatch");
  const int64_t* new_cells = inst_cell_ids.data();

  bool wire_rc_updated = false;
  if (!wire_cap.is_none() && !incoming_resistance.is_none()) {
    auto wc = copy1D<double>(wire_cap, "wire_cap");
    auto ir = copy1D<double>(incoming_resistance, "incoming_resistance");
    require(static_cast<int>(wc.size()) == rc_forest_.vertex_count
                && static_cast<int>(ir.size()) == rc_forest_.vertex_count,
            "syncState wire RC length mismatch");
    if (std::memcmp(wc.data(), rc_forest_.wire_cap.data(), wc.size() * sizeof(double)) != 0
        || std::memcmp(ir.data(), rc_forest_.incoming_resistance.data(),
                       ir.size() * sizeof(double))
               != 0) {
      rc_forest_.wire_cap = std::move(wc);
      rc_forest_.incoming_resistance = std::move(ir);
      wire_rc_updated = true;
    }
  }

  std::vector<int> seed_states;
  std::vector<int> touched_endpoints;
  std::vector<int> dirty_nets;

  for (int node = 0; node < num_instances_; ++node) {
    const int target = static_cast<int>(new_cells[node]);
    const int prev = inst_cell_id_[node];
    if (target == prev) continue;
    require(target >= 0 && target < num_cells_, "syncState target cell out of range");
    inst_cell_id_[node] = target;
    remapNodeArcs(node, target, false, &seed_states, &touched_endpoints);
    for (int pin : node_pins_[node]) {
      const int64_t lib_pin = cell_pin_start_[target] + pin_lib_offset_[pin];
      for (int lane = 0; lane < 5; ++lane) {
        const double val = lib_pin_values_[lane][lib_pin];
        if (lane < 3 && bitsChanged(pin_values_[lane][pin], val) && pin_net_[pin] >= 0) {
          dirty_nets.push_back(pin_net_[pin]);
        }
        pin_values_[lane][pin] = val;
      }
    }
  }

  if (wire_rc_updated) {
    for (std::size_t t = 0; t < rc_forest_.net_topo.size(); ++t) {
      const auto& topo = rc_forest_.net_topo[t];
      if (!topo.empty()) rc_forest_.incoming_wire_cap[topo[0]] = 0.0;
      for (int k = static_cast<int>(topo.size()) - 1; k >= 1; --k) {
        const int v = topo[k];
        double child_half = 0.0;
        for (int ch : rc_forest_.adj_children[v]) {
          child_half += 0.5 * rc_forest_.incoming_wire_cap[ch];
        }
        rc_forest_.incoming_wire_cap[v] =
            std::max(0.0, 2.0 * (rc_forest_.wire_cap[v] - child_half));
      }
    }
    for (int net = 0; net < num_nets_; ++net) {
      recomputeNetRc(net, false, nullptr);
    }
    for (int ep : touched_endpoints) touched_endpoint_mask_[ep] = 0;
    propagateFull();
    return;
  }

  std::sort(dirty_nets.begin(), dirty_nets.end());
  dirty_nets.erase(std::unique(dirty_nets.begin(), dirty_nets.end()), dirty_nets.end());
  std::vector<int> changed_rc_pins;
  for (int net : dirty_nets) {
    recomputeNetRc(net, false, &changed_rc_pins);
  }
  for (int pin : changed_rc_pins) {
    for (int s : pin_states_[pin]) seed_states.push_back(s);
  }
  propagateEvents(seed_states, nullptr, &touched_endpoints);
  for (int ep : touched_endpoints) {
    touched_endpoint_mask_[ep] = 0;
    endpoint_slacks_[ep] = computeEndpointSlack(ep);
  }
  clearUndo();
}

py::dict IncrementalSizingEngine::currentMetrics() const
{
  double wns = 0.0, tns = 0.0;
  for (double s : endpoint_slacks_) {
    const double neg = timing::hardMinimum(s, 0.0);
    wns = timing::hardMinimum(wns, neg);
    tns += neg;
  }
  py::dict out;
  out["wns_ps"] = wns;
  out["tns_ps"] = tns;
  out["endpoint_count"] = num_endpoints_;
  return out;
}

double IncrementalSizingEngine::RankingPolicy::movableNodeArea(int node, int cell) const
{
  if (node < 0 || node >= num_movable) return 0.0;
  double w = cell_width[cell] + 2.0 * padding_x;
  double h = cell_height[cell];
  if (inflated_mask[node]) {
    w = std::max(inflated_width[node], w);
    h = std::max(inflated_height[node], h);
  }
  return w * h;
}

double IncrementalSizingEngine::RankingPolicy::rawCellArea(int cell) const
{
  return (cell_width[cell] + 2.0 * padding_x) * cell_height[cell];
}

double IncrementalSizingEngine::RankingPolicy::computeLoss(
    double wns_ps, double tns_ps, double slew_ps, double cap_pf, double leak_mw) const
{
  const double timing_term = -(wns_coeff * wns_ps + tns_coeff * tns_ps) * timing_weight;
  const double slew_term = slew_ps * slew_weight;
  const double cap_term = (cap_pf * 1000.0) * cap_weight;
  const double leak_term = leak_mw * leakage_weight;
  return timing_term + slew_term + cap_term + leak_term;
}

IncrementalSizingEngine::RankingPolicy IncrementalSizingEngine::parseRankingPolicy(
    const py::dict& policy, bool /*allow_expanded_pins*/) const
{
  RankingPolicy pol;
  pol.wns_coeff = policy["wns_coeff"].cast<double>();
  pol.tns_coeff = policy["tns_coeff"].cast<double>();
  pol.timing_weight = policy["timing_weight"].cast<double>();
  pol.slew_weight = policy["slew_weight"].cast<double>();
  pol.cap_weight = policy["cap_weight"].cast<double>();
  pol.leakage_weight = policy["leakage_weight"].cast<double>();
  pol.loss_tolerance =
      policy.contains("loss_tolerance") ? policy["loss_tolerance"].cast<double>() : 0.0;

  pol.cell_width = copy1D<double>(policy["cell_width"], "cell_width");
  pol.cell_height = copy1D<double>(policy["cell_height"], "cell_height");
  pol.cell_leakage = copy1D<double>(policy["cell_leakage"], "cell_leakage");
  require(static_cast<int>(pol.cell_width.size()) == num_cells_
              && static_cast<int>(pol.cell_height.size()) == num_cells_
              && static_cast<int>(pol.cell_leakage.size()) == num_cells_,
          "cell physical table length mismatch");

  pol.padding_x = policy["padding_x"].cast<double>();
  pol.num_movable = policy["num_movable_nodes"].cast<int>();
  pol.virtual_area = policy["virtual_area"].cast<double>();
  pol.capacity = policy["capacity"].cast<double>();
  pol.inflated_mask = copy1D<uint8_t>(policy["inflated_mask"], "inflated_mask");
  pol.inflated_width = copy1D<double>(policy["inflated_width"], "inflated_width");
  pol.inflated_height = copy1D<double>(policy["inflated_height"], "inflated_height");
  require(static_cast<int>(pol.inflated_mask.size()) == pol.num_movable
              && static_cast<int>(pol.inflated_width.size()) == pol.num_movable
              && static_cast<int>(pol.inflated_height.size()) == pol.num_movable,
          "inflated geometry length mismatch");
  return pol;
}

double IncrementalSizingEngine::pinSlewViolationPs(
    const std::vector<uint8_t>& slew_mask, int pin) const
{
  if (!slew_mask[pin]) return 0.0;
  const double lim = pin_values_[3][pin];
  if (!std::isfinite(lim) || !(lim > 0.0)) return 0.0;
  return std::max(0.0, std::max(pin_rs_[pin], pin_fs_[pin]) - lim);
}

double IncrementalSizingEngine::pinCapViolationPf(
    const std::vector<uint8_t>& cap_mask, int pin) const
{
  if (!cap_mask[pin]) return 0.0;
  const double lim = pin_values_[4][pin];
  if (!std::isfinite(lim) || !(lim > 0.0)) return 0.0;
  return std::max(0.0, std::max(pin_load_r_[pin], pin_load_f_[pin]) - lim);
}

std::pair<double, double> IncrementalSizingEngine::computeWnsTns() const
{
  double wns = 0.0, tns = 0.0;
  for (double s : endpoint_slacks_) {
    const double neg = timing::hardMinimum(s, 0.0);
    wns = timing::hardMinimum(wns, neg);
    tns += neg;
  }
  return {wns, tns};
}

IncrementalSizingEngine::RankingEvalState IncrementalSizingEngine::initRankingEvalState(
    const RankingPolicy& pol,
    const py::dict& policy,
    bool fill_new_pin_masks,
    std::vector<double>* current_node_area) const
{
  RankingEvalState eval;
  eval.slew_mask = copy1D<uint8_t>(policy["slew_pin_mask"], "slew_pin_mask");
  eval.cap_mask = copy1D<uint8_t>(policy["cap_pin_mask"], "cap_pin_mask");
  if (fill_new_pin_masks) {
    require((static_cast<int>(eval.slew_mask.size()) == initial_num_pins_
             || static_cast<int>(eval.slew_mask.size()) == num_pins_)
                && eval.slew_mask.size() == eval.cap_mask.size(),
            "DRV pin mask length mismatch");
    if (static_cast<int>(eval.slew_mask.size()) < num_pins_) {
      const int old_sz = static_cast<int>(eval.slew_mask.size());
      eval.slew_mask.resize(num_pins_, 1);
      eval.cap_mask.resize(num_pins_, 0);
      for (int p = old_sz; p < num_pins_; ++p) {
        eval.cap_mask[p] = (pin_values_[0][p] == 0.0) ? 1 : 0;
      }
    }
  } else {
    require(static_cast<int>(eval.slew_mask.size()) == num_pins_
                && static_cast<int>(eval.cap_mask.size()) == num_pins_,
            "DRV pin mask length mismatch");
  }

  if (current_node_area != nullptr) current_node_area->assign(pol.num_movable, 0.0);
  eval.current_area = pol.virtual_area;
  for (int node = 0; node < pol.num_movable && node < initial_num_instances_; ++node) {
    const int cell = inst_cell_id_[node];
    if (cell >= 0 && cell < num_cells_) {
      const double a = pol.movableNodeArea(node, cell);
      if (current_node_area != nullptr) (*current_node_area)[node] = a;
      eval.current_area += a;
    }
  }
  for (int node = initial_num_instances_; node < num_instances_; ++node) {
    const int cell = inst_cell_id_[node];
    if (cell >= 0 && cell < num_cells_) {
      eval.current_area += pol.rawCellArea(cell);
    }
  }

  eval.current_leakage = 0.0;
  if (pol.leakage_weight != 0.0) {
    for (int node = 0; node < num_instances_; ++node) {
      const int cell = inst_cell_id_[node];
      if (cell >= 0 && cell < num_cells_) {
        eval.current_leakage += std::max(0.0, pol.cell_leakage[cell]);
      }
    }
  }

  eval.slew_viol.assign(num_pins_, 0.0);
  eval.cap_viol.assign(num_pins_, 0.0);
  eval.total_slew_ps = 0.0;
  eval.total_cap_pf = 0.0;
  for (int pin = 0; pin < num_pins_; ++pin) {
    eval.slew_viol[pin] = pinSlewViolationPs(eval.slew_mask, pin);
    eval.cap_viol[pin] = pinCapViolationPf(eval.cap_mask, pin);
    eval.total_slew_ps += eval.slew_viol[pin];
    eval.total_cap_pf += eval.cap_viol[pin];
  }

  const auto [wns, tns] = computeWnsTns();
  eval.cur_wns = wns;
  eval.cur_tns = tns;
  eval.current_loss = pol.computeLoss(
      wns, tns, eval.total_slew_ps, eval.total_cap_pf, eval.current_leakage);
  eval.initial_loss = eval.current_loss;
  eval.initial_wns = wns;
  eval.initial_tns = tns;
  eval.initial_slew_ps = eval.total_slew_ps;
  eval.initial_cap_pf = eval.total_cap_pf;
  eval.initial_area = eval.current_area;
  return eval;
}

bool IncrementalSizingEngine::evaluateTrialPropagationAndLoss(
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
    std::vector<std::tuple<int, double, double>>& drv_undo)
{
  std::sort(dirty_nets.begin(), dirty_nets.end());
  dirty_nets.erase(std::unique(dirty_nets.begin(), dirty_nets.end()), dirty_nets.end());
  for (int net : dirty_nets) {
    recomputeNetRc(net, true, &changed_rc_pins);
  }
  auto mark_drv_pin = [&](int p) {
    if (!drv_seen[p]) {
      drv_seen[p] = 1;
      drv_pins.push_back(p);
    }
  };
  for (int pin : changed_rc_pins) {
    for (int s : pin_states_[pin]) seed_states.push_back(s);
    mark_drv_pin(pin);
  }

  propagateEvents(seed_states, &changed_slew_pins, &touched_endpoints);
  for (int pin : changed_slew_pins) mark_drv_pin(pin);

  for (int ep : touched_endpoints) {
    touched_endpoint_mask_[ep] = 0;
    const double old_slack = endpoint_slacks_[ep];
    const double new_slack = computeEndpointSlack(ep);
    if (bitsChanged(old_slack, new_slack)) {
      undo_endpoints_.emplace_back(ep, old_slack);
      endpoint_slacks_[ep] = new_slack;
    }
  }

  double proposed_slew_ps = eval.total_slew_ps;
  double proposed_cap_pf = eval.total_cap_pf;
  for (int p : drv_pins) {
    drv_seen[p] = 0;
    const double old_sv = eval.slew_viol[p];
    const double old_cv = eval.cap_viol[p];
    const double new_sv = pinSlewViolationPs(eval.slew_mask, p);
    const double new_cv = pinCapViolationPf(eval.cap_mask, p);
    if (bitsChanged(old_sv, new_sv) || bitsChanged(old_cv, new_cv)) {
      drv_undo.emplace_back(p, old_sv, old_cv);
      eval.slew_viol[p] = new_sv;
      eval.cap_viol[p] = new_cv;
      proposed_slew_ps += (new_sv - old_sv);
      proposed_cap_pf += (new_cv - old_cv);
    }
  }
  proposed_slew_ps = std::max(0.0, proposed_slew_ps);
  proposed_cap_pf = std::max(0.0, proposed_cap_pf);
  const double proposed_leakage = eval.current_leakage + delta_leakage;

  const auto [prop_wns, prop_tns] = computeWnsTns();
  const double proposed_loss = pol.computeLoss(
      prop_wns, prop_tns, proposed_slew_ps, proposed_cap_pf, proposed_leakage);

  if (std::isfinite(proposed_loss) && proposed_loss < eval.current_loss - pol.loss_tolerance) {
    eval.current_loss = proposed_loss;
    eval.cur_wns = prop_wns;
    eval.cur_tns = prop_tns;
    eval.total_slew_ps = proposed_slew_ps;
    eval.total_cap_pf = proposed_cap_pf;
    eval.current_leakage = proposed_leakage;
    eval.current_area = proposed_area;
    return true;
  }
  return false;
}

void IncrementalSizingEngine::restoreDrvUndo(
    RankingEvalState& eval,
    const std::vector<std::tuple<int, double, double>>& drv_undo)
{
  for (auto it = drv_undo.rbegin(); it != drv_undo.rend(); ++it) {
    const auto& [p, sv, cv] = *it;
    if (p < static_cast<int>(eval.slew_viol.size())) {
      eval.slew_viol[p] = sv;
      eval.cap_viol[p] = cv;
    }
  }
}

void IncrementalSizingEngine::cleanupTrialMasks(
    const std::vector<int>& drv_pins,
    std::vector<uint8_t>& drv_seen,
    const std::vector<int>& touched_endpoints)
{
  for (int p : drv_pins) {
    if (p < static_cast<int>(drv_seen.size())) drv_seen[p] = 0;
  }
  for (int ep : touched_endpoints) touched_endpoint_mask_[ep] = 0;
  for (int s = 0; s < num_states_; ++s) dirty_state_mask_[s] = 0;
  for (int g = 0; g < num_slew_groups_; ++g) dirty_group_mask_[g] = 0;
}

py::dict IncrementalSizingEngine::makeEvalReport(
    const RankingPolicy& pol,
    const RankingEvalState& eval,
    int input_count,
    int evaluated_count,
    int accepted_count,
    int rejected_area_count,
    int rejected_loss_count,
    const std::chrono::steady_clock::time_point& started) const
{
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
          .count();
  py::dict result;
  result["status"] = "ok";
  result["input_count"] = input_count;
  result["evaluated_count"] = evaluated_count;
  result["accepted_count"] = accepted_count;
  result["rejected_area_count"] = rejected_area_count;
  result["rejected_loss_count"] = rejected_loss_count;
  result["initial_loss"] = eval.initial_loss;
  result["final_loss"] = eval.current_loss;
  result["initial_wns_ps"] = eval.initial_wns;
  result["final_wns_ps"] = eval.cur_wns;
  result["initial_tns_ps"] = eval.initial_tns;
  result["final_tns_ps"] = eval.cur_tns;
  result["initial_slew_violation_ps"] = eval.initial_slew_ps;
  result["final_slew_violation_ps"] = eval.total_slew_ps;
  result["initial_cap_violation_pf"] = eval.initial_cap_pf;
  result["final_cap_violation_pf"] = eval.total_cap_pf;
  result["initial_area"] = eval.initial_area;
  result["final_area"] = eval.current_area;
  result["capacity"] = pol.capacity;
  result["elapsed_ms"] = elapsed_ms;
  return result;
}

py::dict IncrementalSizingEngine::evaluateRankedCandidates(
    const std::vector<int>& instance_ids,
    const std::vector<int>& target_cell_ids,
    const py::dict& policy)
{
  const auto started = std::chrono::steady_clock::now();
  require(instance_ids.size() == target_cell_ids.size(),
          "instance_ids and target_cell_ids lengths must match");

  const RankingPolicy pol = parseRankingPolicy(policy, /*allow_expanded_pins=*/false);
  std::vector<double> current_node_area;
  RankingEvalState eval = initRankingEvalState(
      pol, policy, /*fill_new_pin_masks=*/false, &current_node_area);

  clearUndo();
  std::vector<int> accepted_instances;
  std::vector<int> accepted_cells;
  accepted_instances.reserve(instance_ids.size());
  accepted_cells.reserve(instance_ids.size());

  int evaluated_count = 0;
  int rejected_area_count = 0;
  int rejected_loss_count = 0;
  int skipped_noop_count = 0;

  std::vector<uint8_t> drv_seen(num_pins_, 0);
  std::vector<int> drv_pins;
  std::vector<std::tuple<int, double, double>> drv_undo;
  std::vector<int> dirty_nets;
  std::vector<int> changed_rc_pins;
  std::vector<int> changed_slew_pins;
  std::vector<int> seed_states;
  std::vector<int> touched_endpoints;

  for (std::size_t idx = 0; idx < instance_ids.size(); ++idx) {
    const int node = instance_ids[idx];
    const int target = target_cell_ids[idx];
    if (node < 0 || node >= num_instances_ || target < 0 || target >= num_cells_) {
      continue;
    }
    const int prev = inst_cell_id_[node];
    if (prev == target) {
      ++skipped_noop_count;
      continue;
    }

    const double old_node_area = node < pol.num_movable ? current_node_area[node] : 0.0;
    const double new_node_area = node < pol.num_movable ? pol.movableNodeArea(node, target) : 0.0;
    const double proposed_area = eval.current_area + (new_node_area - old_node_area);
    if (proposed_area > pol.capacity) {
      ++rejected_area_count;
      continue;
    }

    ++evaluated_count;
    const auto cp = captureUndo();
    dirty_nets.clear();
    changed_rc_pins.clear();
    changed_slew_pins.clear();
    seed_states.clear();
    touched_endpoints.clear();
    drv_pins.clear();
    drv_undo.clear();

    try {
      undo_cells_.emplace_back(node, prev);
      inst_cell_id_[node] = target;
      remapNodeArcs(node, target, true, &seed_states, &touched_endpoints);
      for (int pin : node_pins_[node]) {
        std::array<double, 5> old_v;
        for (int lane = 0; lane < 5; ++lane) old_v[lane] = pin_values_[lane][pin];
        undo_pin_vals_.emplace_back(pin, old_v);
        const int64_t lib_pin = cell_pin_start_[target] + pin_lib_offset_[pin];
        for (int lane = 0; lane < 5; ++lane) {
          const double val = lib_pin_values_[lane][lib_pin];
          if (lane < 3 && bitsChanged(old_v[lane], val) && pin_net_[pin] >= 0) {
            dirty_nets.push_back(pin_net_[pin]);
          }
          pin_values_[lane][pin] = val;
        }
        if (!drv_seen[pin]) {
          drv_seen[pin] = 1;
          drv_pins.push_back(pin);
        }
      }

      const double delta_leakage =
          (pol.leakage_weight != 0.0)
              ? std::max(0.0, pol.cell_leakage[target]) - std::max(0.0, pol.cell_leakage[prev])
              : 0.0;

      if (evaluateTrialPropagationAndLoss(pol, eval, proposed_area, delta_leakage,
                                          dirty_nets, changed_rc_pins, changed_slew_pins,
                                          seed_states, touched_endpoints, drv_pins, drv_seen,
                                          drv_undo)) {
        if (node < pol.num_movable) current_node_area[node] = new_node_area;
        accepted_instances.push_back(node);
        accepted_cells.push_back(target);
        clearUndo();
      } else {
        restoreDrvUndo(eval, drv_undo);
        restoreUndo(cp);
        ++rejected_loss_count;
      }
    } catch (...) {
      cleanupTrialMasks(drv_pins, drv_seen, touched_endpoints);
      restoreDrvUndo(eval, drv_undo);
      restoreUndo(cp);
      throw;
    }
  }

  clearUndo();
  py::dict result = makeEvalReport(
      pol, eval, static_cast<int>(instance_ids.size()), evaluated_count,
      static_cast<int>(accepted_instances.size()), rejected_area_count, rejected_loss_count,
      started);
  result["accepted_instance_ids"] = accepted_instances;
  result["accepted_cell_ids"] = accepted_cells;
  result["skipped_noop_count"] = skipped_noop_count;
  return result;
}

}  // namespace dreamplace
