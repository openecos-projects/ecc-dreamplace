// SPDX-License-Identifier: BSD-3-Clause
// Keep canonical index extents; populate timing/physical records only for one tree.
#include "rebuffer_timing_local.hh"

#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"

#include "timing_hard_math.h"


#include <algorithm>

namespace rebuffer {

void NativeTimingLocalPlanningSnapshot::buildViews(LocalPlanningView& out)
{
  const auto& m = *graph_->mutation_state_;
  const auto& base = inputs_->design;
  auto& d = out.projection.design;
  auto& t = out.timing;
  d.dbu = base.dbu; d.num_pins = pin_count_; d.num_nodes = node_count_; d.num_nets = net_count_;
  d.num_base_nodes = base.num_base_nodes;
  d.num_terminals = base.num_terminals;
  d.num_terminal_nis = base.num_terminal_nis;
  d.pin2node_map.resize(pin_count_, -1); d.pin2net_map.resize(pin_count_, -1);
  d.pin_offset_x.resize(pin_count_, 0.0); d.pin_offset_y.resize(pin_count_, 0.0);
  d.node_x.resize(node_count_, 0.0); d.node_y.resize(node_count_, 0.0);
  d.node_is_buffer.resize(node_count_, 0); d.inst_main_id.resize(node_count_, -1); d.inst_libcell_offset.resize(node_count_, 0);
  if (!base.node_is_fixed.empty()) d.node_is_fixed.resize(node_count_, 0);
  d.pin_2_libpin_offset.resize(pin_count_, -1);
  if (!out.static_design_initialized) {
    d.pin_cap_limit = base.pin_cap_limit; d.pin_cap = base.pin_cap;
    d.pin_slew_limit = base.pin_slew_limit; d.pin_drive_resistance = base.pin_drive_resistance;
    d.main_id_2_cell_id_start = base.main_id_2_cell_id_start;
    d.cell_id_2_libpin_id_start = base.cell_id_2_libpin_id_start;
    d.net2driver_pin_map = base.net2driver_pin_map;
  }
  if (!out.static_design_initialized
      || out.cached_topology_revision != identity_->topology_revision
      || static_cast<int>(d.net2driver_pin_map.size()) != net_count_) {
    d.net2driver_pin_map.resize(net_count_);
    for (int net = base.num_nets; net < net_count_; ++net) {
      const int driver = m.net_drivers.at(graphNet(net));
      d.net2driver_pin_map[net] = driver < 0 ? -1 : pinId(driver);
    }
  }
  out.cached_topology_revision = identity_->topology_revision;
  out.static_design_initialized = true;
  if (!shared_view_data_ready_) {
    endpoint_slacks_ = graph_->endpointSlacks();
    shared_view_data_ready_ = true;
  }
  if (!inputs_->static_pin_lookup_ready) {
    const int prog_pins = graph_->propagation_program_->pin_count;
    inputs_->metric_row_by_pin.assign(prog_pins, -1);
    for (std::size_t row = 0; row < graph_->propagation_program_->metric_pins.size(); ++row) {
      const int pin = static_cast<int>(graph_->propagation_program_->metric_pins[row]);
      if (pin >= 0 && pin < prog_pins) {
        inputs_->metric_row_by_pin[pin] = static_cast<int>(row);
      }
    }
    inputs_->clock_order_by_pin.assign(prog_pins, -1);
    inputs_->clock_pins_unique = true;
    for (std::size_t idx = 0; idx < graph_->timing_index_->clocks.size(); ++idx) {
      const int pin = graph_->timing_index_->clocks[idx];
      if (pin >= 0 && pin < prog_pins) {
        if (inputs_->clock_order_by_pin[pin] != -1) inputs_->clock_pins_unique = false;
        inputs_->clock_order_by_pin[pin] = static_cast<int>(idx);
      } else {
        inputs_->clock_pins_unique = false;
      }
    }
    inputs_->static_pin_lookup_ready = true;
  }
  d.flat_net2pin_start_map.resize(net_count_ + 1, 0);
  for (const int net : out.nets) {
    d.flat_net2pin_start_map[net] = static_cast<int>(d.flat_net2pin_map.size());
    for (const int pin : out.projection.topologies.at(net)) {
      if (pin < pin_count_) d.flat_net2pin_map.push_back(pin);
    }
    d.flat_net2pin_start_map[net + 1] = static_cast<int>(d.flat_net2pin_map.size());
  }
  std::vector<int> owners, read_pins;
  owners.reserve(out.pins.size());
  for (const int pin : out.pins) {
    const int native = pin < base.num_pins ? pin : identity_->canonical_to_pin.at(pin - base.num_pins);
    owners.push_back(m.pin_owner.at(native));
  }
  std::sort(owners.begin(), owners.end());
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  read_pins.reserve(owners.size() * 4);
  for (const int owner : owners) {
    const int node = nodeId(owner), cell = m.node_cells.at(owner);
    out.touched_nodes.push_back(node);
    const int main = cell < 0 ? -1 : m.cell_mains.at(cell);
    d.inst_main_id[node] = main;
    d.inst_libcell_offset[node] = main < 0 ? 0 : cell - base.main_id_2_cell_id_start.at(main);
    d.node_is_buffer[node] = owner < m.original_node_count
        ? base.node_is_buffer.at(owner) : graph_->queue_liberty_->cells.at(cell).is_buffer;
    if (!base.node_is_fixed.empty()) d.node_is_fixed[node] = owner < m.original_node_count ? base.node_is_fixed.at(owner) : 0;
    if (owner < m.original_node_count) {
      d.node_x[node] = inputs_->carrier.node_x.at(owner); d.node_y[node] = inputs_->carrier.node_y.at(owner);
    } else {
      const int buf_idx = identity_->buffer_index_by_node.at(owner - m.original_node_count);
      requireLocal(buf_idx >= 0, "local owner has no active native buffer");
      const auto& point = identity_->buffer_coordinates.at(buf_idx);
      d.node_x[node] = point[0]; d.node_y[node] = point[1];
    }
    for (const int pin : m.node_pins.at(owner)) if (m.pin_active.at(pin)) read_pins.push_back(pin);
  }
  std::sort(read_pins.begin(), read_pins.end());
  read_pins.erase(std::unique(read_pins.begin(), read_pins.end()), read_pins.end());
  const auto init = [&](std::vector<double>& values) { values.resize(pin_count_, 0.0); };
  for (auto* values : {&t.pin_cap, &t.pin_max_slew, &t.pin_max_cap, &t.pin_slew_rise, &t.pin_slew_fall,
       &t.pin_load, &t.pin_arrival_rise, &t.pin_arrival_fall, &t.pin_required_rise, &t.pin_required_fall, &t.pin_slack}) init(*values);
  const auto& s_rtran = graph_->state_.at("pin_rtran");
  const auto& s_ftran = graph_->state_.at("pin_ftran");
  const auto& s_load = graph_->state_.at("pin_net_cap_rise");
  const auto& s_raat = graph_->state_.at("pin_rAAT");
  const auto& s_faat = graph_->state_.at("pin_fAAT");
  const auto& s_rrat = graph_->state_.at("pin_rRAT");
  const auto& s_frat = graph_->state_.at("pin_fRAT");
  std::vector<std::pair<int, int>> clock_hits;
  for (const int native : read_pins) {
    const int pin = pinId(native), owner = m.pin_owner.at(native), cell = m.node_cells.at(owner);
    out.touched_pins.push_back(pin);
    d.pin2node_map[pin] = nodeId(owner);
    d.pin2net_map[pin] = m.pin_net.at(native) < 0 ? -1 : netId(m.pin_net[native]);
    if (native < base.num_pins) {
      d.pin_offset_x[pin] = inputs_->carrier.pin_x.at(native); d.pin_offset_y[pin] = inputs_->carrier.pin_y.at(native);
    }
    if (cell >= 0) {
      const auto begin = m.library_pin_names.begin() + m.cell_pin_start.at(cell);
      const auto end = m.library_pin_names.begin() + m.cell_pin_start.at(cell + 1);
      const auto found = std::find(begin, end, m.pin_ports.at(native));
      requireLocal(found != end, "local pin is absent from current Liberty cell");
      d.pin_2_libpin_offset[pin] = std::distance(begin, found);
    }
    t.pin_cap[pin] = m.pin_values[0][native]; t.pin_max_slew[pin] = m.pin_values[3][native]; t.pin_max_cap[pin] = m.pin_values[4][native];
    t.pin_slew_rise[pin] = s_rtran.at(native);
    t.pin_slew_fall[pin] = s_ftran.at(native);
    t.pin_load[pin] = s_load.at(native);
    t.pin_arrival_rise[pin] = s_raat.at(native);
    t.pin_arrival_fall[pin] = s_faat.at(native);
    t.pin_required_rise[pin] = s_rrat.at(native);
    t.pin_required_fall[pin] = s_frat.at(native);
    t.pin_slack[pin] = dreamplace::timing::hardMinimum(t.pin_required_rise[pin] - t.pin_arrival_rise[pin], t.pin_required_fall[pin] - t.pin_arrival_fall[pin]);
    if (native >= 0 && native < static_cast<int>(inputs_->metric_row_by_pin.size())) {
      const int row = inputs_->metric_row_by_pin[native];
      if (row >= 0 && static_cast<std::size_t>(row) < endpoint_slacks_.size()) {
        t.pin_slack[pin] = endpoint_slacks_[row];
      }
    }
    if (inputs_->clock_pins_unique
        && native >= 0 && native < static_cast<int>(inputs_->clock_order_by_pin.size())) {
      const int order = inputs_->clock_order_by_pin[native];
      if (order >= 0) clock_hits.emplace_back(order, pin);
    }
  }
  if (! inputs_->clock_pins_unique) {
    for (const int native : graph_->timing_index_->clocks) {
      if (std::binary_search(read_pins.begin(), read_pins.end(), native)) {
        t.clock_pins.push_back(pinId(native));
      }
    }
  } else if (!clock_hits.empty()) {
    std::sort(clock_hits.begin(), clock_hits.end());
    for (const auto& [_, pin] : clock_hits) t.clock_pins.push_back(pin);
  }
  for (const auto& item : queue_->items) {
    if (!out.nets.count(item.root_net_id)) continue;
    for (const int pin : item.path_pins) {
      if (pin >= 0 && pin < pin_count_) {
        const int net = d.pin2net_map[pin];
        if (net >= 0 && d.net2driver_pin_map[net] != pin && out.pins.count(pin)) {
          t.pin_slack[pin] = item.endpoint_slack_ps;
        }
      }
    }
  }
  filled_pins_ += read_pins.size();
}

}  // namespace rebuffer

// --- rebuffer_timing_local_geometry.cc ---
// Build only selected wire partitions, preserving the full projection's IDs.

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <cmath>

namespace rebuffer {

std::shared_ptr<LocalPlanningView> NativeTimingLocalPlanningSnapshot::view(int root)
{
  if (last_view_ != nullptr && last_view_root_ == root) return last_view_;
  requireCurrent();
  requireLocal(root >= 0 && root < net_count_, "local tree root is out of range");
  const auto existing_tree = trees_.find(root);
  const bool already_visited = existing_tree != trees_.end();
  const int prev_filled_vertices = filled_vertices_;
  const int prev_filled_pins = filled_pins_;
  std::shared_ptr<LocalPlanningView> result;
  if (last_view_ != nullptr && last_view_.use_count() == 1) {
    resetLocalPlanningView(*last_view_);
    result = std::move(last_view_);
    last_view_root_ = -1;
  } else if (!inputs_->local_view_pool.empty()) {
    result = std::move(inputs_->local_view_pool.back());
    inputs_->local_view_pool.pop_back();
  } else {
    result = std::make_shared<LocalPlanningView>();
  }
  const auto nets = reachable(root);
  result->nets.insert(nets.begin(), nets.end());
  auto started = std::chrono::steady_clock::now();
  buildTopology(*result);
  timings_["local_topology"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  started = std::chrono::steady_clock::now();
  buildViews(*result);
  timings_["local_views"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  if (already_visited) {
    filled_vertices_ = prev_filled_vertices;
    filled_pins_ = prev_filled_pins;
    result->tree = existing_tree->second;
  } else {
    started = std::chrono::steady_clock::now();
    TopologyView topology;
    const auto& p = result->projection;
    topology.steiner_node_base = pin_count_;
    topology.net_ids = nets;
    topology.net_flat_topo_sort_start.push_back(0);
    topology.pin_fa = std::move(result->projection.steiner.pin_fa);
    for (const int net : nets) {
      const auto& nodes = p.topologies.at(net);
      topology.net_flat_topo_sort.insert(topology.net_flat_topo_sort.end(), nodes.begin(), nodes.end());
      for (const int node : nodes) {
        topology.node_x.emplace(node, p.steiner.node_x[node]);
        topology.node_y.emplace(node, p.steiner.node_y[node]);
      }
      topology.net_flat_topo_sort_start.push_back(topology.net_flat_topo_sort.size());
    }
    result->tree = std::make_shared<CurrentDriverTree>();
    std::string reason;
    const bool stop_at_existing_buffers = config_.contains("native_tree_stop_at_existing_buffers")
        && config_["native_tree_stop_at_existing_buffers"].cast<bool>();
    std::vector<std::size_t> local_link_indices;
    for (const int net : result->nets) {
      if (net >= 0 && net < static_cast<int>(identity_->link_by_output.size())
          && identity_->link_by_output[net] >= 0) {
        local_link_indices.push_back(static_cast<std::size_t>(identity_->link_by_output[net]));
      }
      auto it = std::lower_bound(
          identity_->links.begin(), identity_->links.end(), net,
          [](const CurrentDriverBufferLink& link, int target) {
            return link.input_net_id < target;
          });
      for (; it != identity_->links.end() && it->input_net_id == net; ++it) {
        local_link_indices.push_back(static_cast<std::size_t>(it - identity_->links.begin()));
      }
    }
    std::sort(local_link_indices.begin(), local_link_indices.end());
    local_link_indices.erase(
        std::unique(local_link_indices.begin(), local_link_indices.end()),
        local_link_indices.end());
    std::vector<CurrentDriverBufferLink> local_links;
    local_links.reserve(local_link_indices.size());
    for (const std::size_t idx : local_link_indices) {
      local_links.push_back(identity_->links[idx]);
    }
    const bool imported = importCurrentDriverTree(
        p.design, topology, root, local_links, *result->tree, reason, stop_at_existing_buffers);
    result->projection.steiner.pin_fa = std::move(topology.pin_fa);
    requireLocal(imported, reason);
    timings_["tree_import"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    trees_.emplace(root, result->tree);
  }
  last_view_ = result;
  last_view_root_ = root;
  return result;
}

void NativeTimingLocalPlanningSnapshot::buildTopology(LocalPlanningView& out)
{
  const auto& m = *graph_->mutation_state_;
  const int base_nets = inputs_->design.num_nets;
  auto& p = out.projection;
  p.topologies.resize(net_count_);
  p.steiner.steiner_node_base = pin_count_;
  p.steiner.pin_fa.resize(vertex_count_, -1);
  p.steiner.node_x.resize(vertex_count_, 0.0);
  p.steiner.node_y.resize(vertex_count_, 0.0);
  if (static_cast<int>(p.actions.original_net_by_net.size()) < base_nets) {
    p.actions.original_net_by_net.resize(base_nets);
    std::iota(p.actions.original_net_by_net.begin(),
              p.actions.original_net_by_net.begin() + base_nets, 0);
  }
  if (out.cached_topology_revision != identity_->topology_revision
      || static_cast<int>(p.actions.original_net_by_net.size()) != net_count_) {
    p.actions.original_net_by_net.resize(net_count_);
    for (std::size_t i = 0; i < identity_->original_by_net.size(); ++i) {
      p.actions.original_net_by_net[base_nets + static_cast<int>(i)] = identity_->original_by_net[i];
    }
  }
  p.actions.state_node_by_vertex.resize(vertex_count_, -1);
  p.actions.state_net_by_vertex.resize(vertex_count_, -1);
  for (const int net : out.nets) {
    const int owner = originalNet(net), native_net = graphNet(net);
    const int driver = m.net_drivers.at(native_net);
    requireLocal(driver >= 0, "local partition has no driver");
    int source_root = driver;
    if (driver >= inputs_->design.num_pins) {
      const int canonical_pin = pinId(driver);
      const auto& source = identity_->source_by_pin.at(canonical_pin - inputs_->design.num_pins);
      requireLocal(source.first == owner, "local driver has a different native topology owner");
      source_root = source.second;
    }
    auto add = [&](int source, int parent, double x, double y) {
      const int id = vertex(owner, source);
      requireLocal(id >= 0 && id < vertex_count_, "local vertex identity is out of range");
      p.topologies[net].push_back(id);
      p.steiner.pin_fa[id] = parent < 0 ? -1 : vertex(owner, parent);
      p.steiner.node_x[id] = x; p.steiner.node_y[id] = y;
      p.actions.state_node_by_vertex[id] = source;
      p.actions.state_net_by_vertex[id] = owner;
      p.actions.state_provenance.push_back({{net, id}, {owner, source}});
      out.touched_vertices.push_back(id);
      if (id < pin_count_) out.pins.insert(id);
    };
    const auto current = graph_->net_states_.find(owner);
    if (current == graph_->net_states_.end()) {
      requireLocal(net == owner, "synthetic partition has no mutable native topology");
      const auto& s = inputs_->steiner;
      for (int pos = s.net_flat_topo_sort_start.at(owner); pos < s.net_flat_topo_sort_start.at(owner + 1); ++pos) {
        const int source = s.net_flat_topo_sort.at(pos);
        if (source != source_root) {
          const double resistance = s.edge_resistance.at(source);
          const double capacitance = resistance / inputs_->r_unit * inputs_->c_unit;
          requireLocal(std::isfinite(resistance) && resistance >= 0
                           && std::isfinite(capacitance) && capacitance >= 0,
                       "local wire edge has invalid RC value");
        }
        add(source, source == source_root ? -1 : s.pin_fa.at(source), s.node_x.at(source), s.node_y.at(source));
      }
    } else {
      const auto& state = *current->second;
      std::string reason;
      requireLocal(validateIncrementalTopology(state, reason), reason);
      std::set<std::pair<int, int>> cells;
      for (const auto& buffer : state.buffers) cells.emplace(buffer.input_node, buffer.output_node);
      std::map<int, std::vector<int>> children;
      for (const auto& edge : state.edges) {
        if (!cells.count({edge.parent, edge.child})) children[edge.parent].push_back(edge.child);
      }
      std::set<int> seen;
      std::function<void(int, int)> visit = [&](int source, int parent) {
        requireLocal(seen.insert(source).second, "local wire partition contains a cycle");
        const auto& point = state.coordinates.at(source);
        add(source, parent, point[0], point[1]);
        for (const int child : children[source]) visit(child, source);
      };
      visit(source_root, -1);
    }
    std::vector<int> actual, expected;
    for (const int id : p.topologies[net]) {
      if (id < pin_count_) actual.push_back(id);
    }
    for (const int pin : m.net_pins.at(native_net)) if (m.pin_active.at(pin)) expected.push_back(pinId(pin));
    std::sort(actual.begin(), actual.end()); std::sort(expected.begin(), expected.end());
    requireLocal(actual == expected, "local partition pins differ from native timing owner");
    requireLocal(std::find(actual.begin(), actual.end(), pinId(driver)) != actual.end(), "local driver is absent from its partition");
  }
  normalizePartitionProvenance(p.actions.state_provenance);
  filled_vertices_ += p.actions.state_provenance.size();
}

}  // namespace rebuffer

// --- rebuffer_timing_local_index.cc ---
// Update only changed physical edges and ranks; keep net-edge slots stable.

#include <algorithm>
#include <chrono>
#include <functional>
#include <queue>
#include <set>
#include <unordered_set>

namespace rebuffer {
namespace {

void eraseId(std::vector<int>& row, int id)
{
  row.erase(std::remove(row.begin(), row.end(), id), row.end());
}

void eraseArc(TimingGraphIndex& index, int id)
{
  const auto found = index.arc_ids.find(index.key(id));
  if (found != index.arc_ids.end() && found->second == id) index.arc_ids.erase(found);
}

void resizeRows(TimingMutationState& m, TimingGraphIndex& index, std::size_t size)
{
  index.rank.resize(size, 0); index.path_level.resize(size, 0); index.successors.resize(size);
  index.cell_out.resize(size); index.cell_in.resize(size);
  index.net_out.resize(size); index.net_in.resize(size);
  m.net_arc_by_pin.resize(size, -1);
  m.events.forward_inputs.resize(size); m.events.reverse_outputs.resize(size);
  m.events.forward_successors.resize(size); m.events.reverse_predecessors.resize(size);
}

}  // namespace

std::shared_ptr<TimingIndexUndo> captureTimingIndex(
    const TimingMutationState& m, const TimingGraphIndex& index)
{
  auto undo = std::make_shared<TimingIndexUndo>();
  undo->pins = index.pins.size(); undo->cells = index.cells.size(); undo->nets = index.nets.size();
  undo->canonical_schedule = m.canonical_schedule;
  return undo;
}

void captureTimingIndexPin(TimingIndexUndo& undo, const TimingMutationState& m,
                            const TimingGraphIndex& index, int pin)
{
  if (pin >= static_cast<int>(undo.pins) || undo.rows.count(pin)) return;
  undo.rows.emplace(pin, TimingIndexPinUndo{
      index.rank.at(pin), index.path_level.at(pin), m.net_arc_by_pin.at(pin), index.successors.at(pin),
      index.cell_out.at(pin), index.cell_in.at(pin), index.net_out.at(pin), index.net_in.at(pin),
      m.events.forward_inputs.at(pin), m.events.reverse_outputs.at(pin),
      m.events.forward_successors.at(pin), m.events.reverse_predecessors.at(pin)});
}

void initializeTimingLocalIndex(TimingMutationState& m, TimingGraphIndex& index,
                                 const TimingPropagationProgram& program)
{
  index.cell_active = m.cell_active;
  index.net_active.assign(index.nets.size(), 1);
  m.net_arc_by_pin.assign(index.pins.size(), -1);
  for (int id = 0; id < static_cast<int>(index.nets.size()); ++id) {
    const auto& edge = index.nets[id];
    const int net = m.pin_net.at(edge[1]);
    if (m.net_arc_by_pin.at(edge[1]) >= 0 || net < 0
        || m.net_drivers.at(net) != edge[0] || m.pin_net.at(edge[0]) != net) {
      throw pybind11::value_error("native mutation requires one physical net driver per load pin");
    }
    m.net_arc_by_pin[edge[1]] = id;
  }
  const auto started = std::chrono::steady_clock::now();
  auto schedule = makeTimingMutationSchedule(m, index, program);
  m.events = std::move(schedule->events);
  ++m.statistics.schedule_builds;
  m.statistics.schedule_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

void updateTimingLocalIndex(TimingMutationState& m, TimingGraphIndex& index,
    const TimingPropagationProgram& program, const TimingTopologyUndo& topology_undo)
{
  const auto started = std::chrono::steady_clock::now();
  auto& undo = *topology_undo.index_undo;
  resizeRows(m, index, index.pins.size());
  for (int pin = undo.pins; pin < static_cast<int>(index.pins.size()); ++pin) {
    if (!index.pin_ids.emplace(index.pins[pin], pin).second) {
      throw pybind11::value_error("duplicate timing pin");
    }
  }
  std::set<int> pins, cells, structural, rank_seeds, events;
  for (const auto& pin : topology_undo.previous_pins) pins.insert(pin.id);
  for (const auto& arc : topology_undo.previous_arcs) cells.insert(arc.id);
  for (int buf_id : m.buffers_by_source_net.at(topology_undo.source_net)) {
    const auto& buffer = m.buffers[buf_id];
    pins.insert(buffer.input_pin); pins.insert(buffer.output_pin);
    cells.insert(buffer.arcs.begin(), buffer.arcs.end());
  }
  events = pins;
  index.cells.resize(m.cells.size()); index.cell_active.resize(m.cells.size(), 0);
  for (int id : cells) {
    if (id < static_cast<int>(undo.cells)) {
      undo.cell_rows.push_back({id, index.cells.at(id), index.cell_active.at(id)});
    }
    if (index.cell_active[id]) eraseArc(index, id);
  }
  auto touch = [&](int pin) {
    captureTimingIndexPin(undo, m, index, pin);
    structural.insert(pin); events.insert(pin);
  };
  for (int id : cells) {
    const bool was_active = index.cell_active[id], active = m.cell_active[id];
    const int old_source = was_active ? index.source(id) : -1;
    const int old_sink = was_active ? index.cells[id][1] : -1;
    const int source = m.cells[id][0], sink = m.cells[id][1];
    const bool connectivity = was_active != active
        || (active && (old_source != source || old_sink != sink));
    if (connectivity && was_active) {
      touch(old_source); touch(old_sink);
      eraseId(index.cell_out[old_source], id); eraseId(index.cell_in[old_sink], id);
      rank_seeds.insert(old_sink);
    }
    std::copy(m.cells[id].begin(), m.cells[id].end(), index.cells[id].begin());
    index.cell_active[id] = active;
    if (connectivity && active) {
      touch(source); touch(sink);
      index.cell_out[source].push_back(id); index.cell_in[sink].push_back(id);
      rank_seeds.insert(sink);
    }
    if (active && !index.arc_ids.emplace(index.key(id), id).second) {
      throw pybind11::value_error("duplicate timing arc");
    }
  }
  for (int pin : pins) {
    const int net = m.pin_net.at(pin);
    const int driver = net < 0 ? -1 : m.net_drivers.at(net);
    const bool active = m.pin_active.at(pin) && driver >= 0 && driver != pin;
    int slot = m.net_arc_by_pin.at(pin);
    const bool was_active = slot >= 0 && index.net_active.at(slot);
    if (was_active == active && (!active || index.nets[slot][0] == driver)) continue;
    touch(pin);
    rank_seeds.insert(pin);
    if (slot >= 0 && slot < static_cast<int>(undo.nets)) {
      undo.net_rows.push_back({slot, index.nets[slot], index.net_active[slot]});
    }
    if (was_active) {
      const int previous = index.nets[slot][0];
      touch(previous); eraseId(index.net_out[previous], slot); eraseId(index.net_in[pin], slot);
    }
    if (active) {
      if (slot < 0) {
        slot = index.nets.size(); m.net_arc_by_pin[pin] = slot;
        index.nets.push_back({driver, pin}); index.net_active.push_back(0);
      }
      index.nets[slot] = {driver, pin}; index.net_active[slot] = 1;
      touch(driver); index.net_out[driver].push_back(slot); index.net_in[pin].push_back(slot);
    } else if (slot >= 0) index.net_active[slot] = 0;
  }
  for (int pin : structural) {
    auto& out = index.cell_out[pin]; auto& in = index.cell_in[pin];
    std::sort(out.begin(), out.end()); std::sort(in.begin(), in.end());
    auto& nets = index.net_out[pin];
    std::sort(nets.begin(), nets.end(), [&](int a, int b) { return index.nets[a][1] < index.nets[b][1]; });
    auto& successors = index.successors[pin]; successors.clear();
    for (int id : out) if (id >= index.clock_end) successors.push_back(index.cells[id][1]);
    for (int id : nets) successors.push_back(index.nets[id][1]);
  }
  // Source-forest partitioning only subdivides/removes edges in a qualified
  // combinational DAG. Recompute all current fanin and stop at unchanged ranks.
  using Event = std::pair<int, int>;
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> pending;
  std::unordered_set<int> queued;
  auto enqueue = [&](int pin) {
    if (queued.insert(pin).second) pending.emplace(index.rank[pin], pin);
  };
  for (int pin : rank_seeds) enqueue(pin);
  std::vector<int> changed_ranks;
  while (!pending.empty()) {
    const int pin = pending.top().second; pending.pop(); queued.erase(pin);
    ++m.statistics.rank_visits;
    int rank = 0, path_level = 0;
    for (int arc : index.cell_in[pin]) {
      if (arc >= index.clock_end) {
        const int source = index.source(arc);
        rank = std::max(rank, index.rank[source] + 1);
        path_level = std::max(path_level, index.path_level[source] + 1);
      }
    }
    for (int arc : index.net_in[pin]) {
      const int source = index.nets[arc][0];
      rank = std::max(rank, index.rank[source] + 1);
      path_level = std::max(path_level, index.path_level[source]);
    }
    if (rank == index.rank[pin] && path_level == index.path_level[pin]) continue;
    captureTimingIndexPin(undo, m, index, pin);
    if (rank != index.rank[pin]) changed_ranks.push_back(pin);
    index.rank[pin] = rank; index.path_level[pin] = path_level;
    for (int sink : index.successors[pin]) enqueue(sink);
  }
  for (int pin : changed_ranks) {
    events.insert(pin);
    const int net = m.pin_net[pin];
    if (net >= 0 && m.net_drivers[net] >= 0) events.insert(m.net_drivers[net]);
    for (int id : index.cell_out[pin]) {
      const int sink = index.cells[id][1]; events.insert(sink);
      for (int edge : index.net_out[sink]) events.insert(index.nets[edge][1]);
    }
  }
  const auto indexed = std::chrono::steady_clock::now();
  m.statistics.local_index_ms += std::chrono::duration<double, std::milli>(indexed - started).count();
  updateTimingEventRows(m, index, program, std::vector<int>(events.begin(), events.end()), undo);
  m.statistics.local_events_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - indexed).count();
  ++m.statistics.local_updates;
  m.canonical_schedule = true;
}

void restoreTimingLocalIndex(TimingMutationState& m, TimingGraphIndex& index, TimingIndexUndo& undo)
{
  for (const auto& row : undo.cell_rows) if (index.cell_active.at(row.id)) eraseArc(index, row.id);
  for (std::size_t id = undo.cells; id < index.cells.size(); ++id) {
    if (index.cell_active[id]) eraseArc(index, id);
  }
  for (const auto& row : undo.cell_rows) {
    index.cells[row.id] = row.row; index.cell_active[row.id] = row.active;
    if (row.active) index.arc_ids.emplace(index.key(row.id), row.id);
  }
  for (const auto& row : undo.net_rows) {
    index.nets[row.id] = row.row; index.net_active[row.id] = row.active;
  }
  for (auto& [pin, row] : undo.rows) {
    index.rank[pin] = row.rank; index.path_level[pin] = row.path_level;
    m.net_arc_by_pin[pin] = row.net_arc;
    index.successors[pin] = std::move(row.successors);
    index.cell_out[pin] = std::move(row.cell_out); index.cell_in[pin] = std::move(row.cell_in);
    index.net_out[pin] = std::move(row.net_out); index.net_in[pin] = std::move(row.net_in);
    m.events.forward_inputs[pin] = std::move(row.forward_inputs);
    m.events.reverse_outputs[pin] = std::move(row.reverse_outputs);
    m.events.forward_successors[pin] = std::move(row.forward_successors);
    m.events.reverse_predecessors[pin] = std::move(row.reverse_predecessors);
  }
  for (std::size_t pin = undo.pins; pin < index.pins.size(); ++pin) {
    const auto found = index.pin_ids.find(index.pins[pin]);
    if (found != index.pin_ids.end() && found->second == static_cast<int>(pin)) index.pin_ids.erase(found);
  }
  index.pins.resize(undo.pins); index.cells.resize(undo.cells); index.cell_active.resize(undo.cells);
  index.nets.resize(undo.nets); index.net_active.resize(undo.nets);
  resizeRows(m, index, undo.pins);
  m.canonical_schedule = undo.canonical_schedule;
}

}  // namespace rebuffer

// --- rebuffer_timing_local_events.cc ---
// Reconstruct affected event rows in the full levelized oracle's exact order.

#include <algorithm>
#include <map>
#include <tuple>

namespace rebuffer {
namespace {
struct OrderedOperation
{
  // Forward: level, cell-before-net, id. Reverse: endpoint-before-level,
  // descending level, cell-before-net, id.
  std::tuple<int, int, int, int> order;
  TimingEventOperation operation;
};

std::vector<TimingEventOperation> ordered(std::vector<OrderedOperation> rows)
{
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.order < b.order; });
  std::vector<TimingEventOperation> result;
  result.reserve(rows.size());
  for (const auto& row : rows) result.push_back(row.operation);
  return result;
}

void unique(std::vector<int>& values)
{
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}
}  // namespace

void updateTimingEventRows(TimingMutationState& m, const TimingGraphIndex& index,
    const TimingPropagationProgram& program, const std::vector<int>& pins, TimingIndexUndo& undo)
{
  std::map<int, std::vector<int>> forward_levels;
  auto levels = [&](int net) -> const std::vector<int>& {
    const auto found = forward_levels.find(net);
    if (found != forward_levels.end()) return found->second;
    auto& result = forward_levels[net];
    for (int pin : m.net_pins.at(net)) {
      if (pin < program.pin_count && program.start_row_by_pin[pin] >= 0) {
        result.push_back(0);
        break;
      }
    }
    const int driver = m.net_drivers.at(net);
    if (driver >= 0) {
      for (int arc : index.cell_in.at(driver)) {
        if (arc >= index.clock_end) result.push_back(index.rank[index.source(arc)] + 1);
      }
    }
    unique(result);
    return result;
  };
  auto cell_input = [&](int pin) {
    return std::any_of(index.cell_out[pin].begin(), index.cell_out[pin].end(),
                       [&](int arc) { return arc >= index.clock_end; });
  };
  for (int pin : pins) {
    captureTimingIndexPin(undo, m, index, pin);
    std::vector<OrderedOperation> forward, reverse;
    std::vector<int> successors, predecessors;
    for (int arc : index.cell_in[pin]) {
      const int source = index.source(arc);
      const bool clock = arc < index.clock_end;
      forward.push_back({{clock ? 0 : index.rank[source] + 1, 0, arc, 0},
          {clock ? TimingEventKind::clock : TimingEventKind::cell, source, pin, arc}});
      if (!clock && source != pin) predecessors.push_back(source);
    }
    for (int arc : index.net_in[pin]) {
      const int source = index.nets[arc][0], net = m.pin_net[pin];
      for (int level : levels(net)) {
        forward.push_back({{level, 1, net, pin}, {TimingEventKind::net, source, pin, -1}});
      }
    }
    for (int arc : index.cell_out[pin]) {
      if (arc < index.clock_end) continue;
      const int sink = index.cells[arc][1];
      reverse.push_back({{1, -index.rank[pin] - 1, 0, arc}, {TimingEventKind::cell, pin, sink, arc}});
      if (sink != pin) successors.push_back(sink);
    }
    const int net = m.pin_net[pin];
    if (net >= 0 && m.net_drivers[net] == pin) {
      const bool forward_net = !levels(net).empty();
      for (int sink : m.net_pins[net]) {
        const int endpoint = sink < program.pin_count ? program.endpoint_row_by_pin[sink] : -1;
        if (endpoint >= 0) {
          reverse.push_back({{0, 0, 0, endpoint}, {TimingEventKind::net, pin, sink, -1}});
        }
        if (cell_input(sink)) {
          reverse.push_back({{1, -index.rank[sink] - 1, 1, sink},
                              {TimingEventKind::net, pin, sink, -1}});
        }
        if (forward_net && sink != pin) successors.push_back(sink);
      }
    }
    const bool endpoint = pin < program.pin_count && program.endpoint_row_by_pin[pin] >= 0;
    if (net >= 0 && (endpoint || cell_input(pin))) {
      const int driver = m.net_drivers[net];
      if (driver >= 0 && driver != pin) predecessors.push_back(driver);
    }
    unique(successors); unique(predecessors);
    m.events.forward_inputs[pin] = ordered(std::move(forward));
    m.events.reverse_outputs[pin] = ordered(std::move(reverse));
    m.events.forward_successors[pin] = std::move(successors);
    m.events.reverse_predecessors[pin] = std::move(predecessors);
  }
  m.statistics.event_rows += pins.size();
}

}  // namespace rebuffer

// --- rebuffer_timing_index_oracle.cc ---
// Explicit diagnostic oracle: rebuild independently without altering the owner.

#include <algorithm>
#include <string>

#include "pybind11/stl.h"


namespace rebuffer {
namespace {
void require(bool condition, const std::string& field)
{
  if (!condition) throw pybind11::value_error("native timing index oracle mismatch: " + field);
}

std::vector<TimingNetArc> edges(const TimingGraphIndex& index, const std::vector<int>& ids)
{
  std::vector<TimingNetArc> result;
  for (int id : ids) {
    require(index.net_active.empty() || index.net_active.at(id), "inactive edge in adjacency");
    result.push_back(index.nets.at(id));
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<int> sorted(std::vector<int> values)
{
  std::sort(values.begin(), values.end());
  return values;
}

std::string operation(const std::vector<TimingEventOperation>& rows, std::size_t id)
{
  if (id >= rows.size()) return "end";
  const auto& row = rows[id];
  return std::to_string(static_cast<int>(row.kind)) + ":" + std::to_string(row.source)
      + ":" + std::to_string(row.sink) + ":" + std::to_string(row.arc);
}

void sameEvents(const std::vector<TimingEventOperation>& actual,
                 const std::vector<TimingEventOperation>& expected, const std::string& field)
{
  std::size_t id = 0;
  while (id < actual.size() && id < expected.size() && actual[id] == expected[id]) ++id;
  require(id == actual.size() && id == expected.size(), field + " row=" + std::to_string(id)
      + " actual=" + operation(actual, id) + " expected=" + operation(expected, id));
}
}  // namespace

pybind11::dict PersistentTimingGraph::mutationIndexOracle() const
{
  if (!mutation_state_ || !propagation_program_) {
    throw pybind11::value_error("mutation inputs are not initialized");
  }
  const auto& m = *mutation_state_;
  const auto& current = *timing_index_;
  TimingGraphIndex rebuilt;
  rebuilt.pins = current.pins; rebuilt.clocks = current.clocks; rebuilt.clock_end = current.clock_end;
  buildTimingMutationIndex(m, rebuilt);
  require(current.cells == rebuilt.cells && current.cell_active == rebuilt.cell_active, "cell slots");
  require(current.pin_ids == rebuilt.pin_ids, "pin identities");
  require(current.arc_ids == rebuilt.arc_ids, "arc identities");
  require(current.net_active.size() == current.nets.size(), "net active slots");
  auto schedule = makeTimingMutationSchedule(m, rebuilt, *propagation_program_);
  for (int pin = 0; pin < static_cast<int>(current.pins.size()); ++pin) {
    const auto label = " pin=" + std::to_string(pin) + " " + current.pins[pin];
    require(current.rank.at(pin) == rebuilt.rank.at(pin), "rank" + label);
    require(current.path_level.at(pin) == rebuilt.path_level.at(pin), "path level" + label);
    require(current.cell_out.at(pin) == rebuilt.cell_out.at(pin), "cell_out" + label);
    require(current.cell_in.at(pin) == rebuilt.cell_in.at(pin), "cell_in" + label);
    require(sorted(current.successors.at(pin)) == sorted(rebuilt.successors.at(pin)), "successors" + label);
    require(edges(current, current.net_out.at(pin)) == edges(rebuilt, rebuilt.net_out.at(pin)), "net_out" + label);
    require(edges(current, current.net_in.at(pin)) == edges(rebuilt, rebuilt.net_in.at(pin)), "net_in" + label);
    const int edge = m.net_arc_by_pin.at(pin);
    if (edge >= 0) require(current.nets.at(edge)[1] == pin, "net slot owner" + label);
    sameEvents(m.events.forward_inputs.at(pin), schedule->events.forward_inputs.at(pin), "forward inputs" + label);
    sameEvents(m.events.reverse_outputs.at(pin), schedule->events.reverse_outputs.at(pin), "reverse outputs" + label);
    require(m.events.forward_successors.at(pin) == schedule->events.forward_successors.at(pin), "event successors" + label);
    require(m.events.reverse_predecessors.at(pin) == schedule->events.reverse_predecessors.at(pin), "event predecessors" + label);
  }
  std::vector<TimingNetArc> active;
  for (std::size_t id = 0; id < current.nets.size(); ++id) if (current.net_active[id]) active.push_back(current.nets[id]);
  auto expected = rebuilt.nets;
  std::sort(active.begin(), active.end()); std::sort(expected.begin(), expected.end());
  require(active == expected, "active net edges");
  pybind11::dict result;
  result["status"] = "ok";
  result["pin_count"] = current.pins.size();
  result["cell_arc_count"] = current.cells.size();
  result["active_net_arc_count"] = active.size();
  return result;
}

}  // namespace rebuffer
