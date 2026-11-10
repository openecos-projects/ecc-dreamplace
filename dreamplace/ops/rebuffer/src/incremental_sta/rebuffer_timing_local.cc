// SPDX-License-Identifier: BSD-3-Clause
// Lazy current-tree views; canonical physical IDs remain identical to full projection.
#include "rebuffer_timing_local.hh"

#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"

#include "rebuffer_carrier.hh"


#include <algorithm>
#include <chrono>
#include <tuple>
#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {
namespace {
double elapsed(std::chrono::steady_clock::time_point started)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

void validateLocalActionsAgainstBuffers(
    const py::list& actions,
    const std::vector<TimingBufferRecord>& sorted_buffers,
    const TimingMutationState& m)
{
  const std::size_t n = py::len(actions);
  if (n == 0 && sorted_buffers.empty()) return;
  std::vector<std::pair<int, PyObject*>> records;
  records.reserve(n);
  bool sorted = true;
  int prev_id = -1;
  bool first = true;
  for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(n); ++i) {
    PyObject* item = PyList_GET_ITEM(actions.ptr(), i);
    if (PyDict_Check(item)) {
      PyObject* kind_obj = PyDict_GetItemString(item, "action_kind");
      if (kind_obj && PyUnicode_CheckExact(kind_obj)
          && PyUnicode_CompareWithASCIIString(kind_obj, "resize_cell") == 0) {
        continue;
      }
      PyObject* id_obj = PyDict_GetItemString(item, "session_buffer_id");
      PyObject* master_obj = PyDict_GetItemString(item, "buffer_master_name");
      if (id_obj && master_obj) {
        const int id = static_cast<int>(PyLong_AsLong(id_obj));
        if (PyErr_Occurred()) throw py::error_already_set();
        if (!first && id <= prev_id) sorted = false;
        first = false;
        prev_id = id;
        records.emplace_back(id, master_obj);
        continue;
      }
    }
    const auto row = py::reinterpret_borrow<py::object>(item).cast<py::dict>();
    if (row.contains("action_kind")
        && row["action_kind"].cast<std::string>() == "resize_cell") {
      continue;
    }
    const int id = row["session_buffer_id"].cast<int>();
    PyObject* master = row["buffer_master_name"].ptr();
    if (!first && id <= prev_id) sorted = false;
    first = false;
    prev_id = id;
    records.emplace_back(id, master);
  }
  if (!sorted) {
    std::sort(records.begin(), records.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (std::size_t i = 1; i < records.size(); ++i) {
      requireLocal(records[i].first != records[i - 1].first, "duplicate local action identity");
    }
  }
  requireLocal(records.size() == sorted_buffers.size(),
               "local actions do not match active native buffers");
  for (std::size_t i = 0; i < sorted_buffers.size(); ++i) {
    const auto& buffer = sorted_buffers[i];
    PyObject* master_obj = nullptr;
    if (records[i].first == buffer.action_id) {
      master_obj = records[i].second;
    } else {
      const auto it = std::lower_bound(
          records.begin(), records.end(), buffer.action_id,
          [](const auto& a, int b) { return a.first < b; });
      requireLocal(it != records.end() && it->first == buffer.action_id,
                   "native buffer has no action metadata");
      master_obj = it->second;
    }
    const std::string& expected = m.cell_names.at(m.node_cells.at(buffer.node));
    if (PyUnicode_CheckExact(master_obj)) {
      requireLocal(PyUnicode_CompareWithASCIIString(master_obj, expected.c_str()) == 0,
                   "local action master differs from native timing owner");
    } else {
      requireLocal(py::reinterpret_borrow<py::object>(master_obj).cast<std::string>() == expected,
                   "local action master differs from native timing owner");
    }
  }
}
}  // namespace

void resetLocalPlanningView(LocalPlanningView& v)
{
  auto& p = v.projection;
  for (const int net : v.nets) {
    if (net >= 0 && net < static_cast<int>(p.topologies.size())) {
      p.topologies[net].clear();
    }
    if (net >= 0 && net + 1 < static_cast<int>(p.design.flat_net2pin_start_map.size())) {
      p.design.flat_net2pin_start_map[net] = 0;
      p.design.flat_net2pin_start_map[net + 1] = 0;
    }
  }
  for (const int id : v.touched_vertices) {
    if (id >= 0 && id < static_cast<int>(p.steiner.pin_fa.size())) {
      p.steiner.pin_fa[id] = -1;
      p.steiner.node_x[id] = 0.0;
      p.steiner.node_y[id] = 0.0;
      p.actions.state_node_by_vertex[id] = -1;
      p.actions.state_net_by_vertex[id] = -1;
    }
  }
  v.touched_vertices.clear();
  p.actions.state_provenance.clear();
  p.net_names.clear();
  auto& d = p.design;
  for (const int node : v.touched_nodes) {
    if (node >= 0 && node < static_cast<int>(d.node_x.size())) {
      d.node_x[node] = 0.0;
      d.node_y[node] = 0.0;
      d.node_is_buffer[node] = 0;
      d.inst_main_id[node] = -1;
      d.inst_libcell_offset[node] = 0;
      if (!d.node_is_fixed.empty()) d.node_is_fixed[node] = 0;
    }
  }
  v.touched_nodes.clear();
  auto& t = v.timing;
  for (const int pin : v.touched_pins) {
    if (pin >= 0 && pin < static_cast<int>(d.pin2node_map.size())) {
      d.pin2node_map[pin] = -1;
      d.pin2net_map[pin] = -1;
      d.pin_offset_x[pin] = 0.0;
      d.pin_offset_y[pin] = 0.0;
      d.pin_2_libpin_offset[pin] = -1;
      t.pin_cap[pin] = 0.0;
      t.pin_max_slew[pin] = 0.0;
      t.pin_max_cap[pin] = 0.0;
      t.pin_slew_rise[pin] = 0.0;
      t.pin_slew_fall[pin] = 0.0;
      t.pin_load[pin] = 0.0;
      t.pin_arrival_rise[pin] = 0.0;
      t.pin_arrival_fall[pin] = 0.0;
      t.pin_required_rise[pin] = 0.0;
      t.pin_required_fall[pin] = 0.0;
      t.pin_slack[pin] = 0.0;
    }
  }
  v.touched_pins.clear();
  d.flat_net2pin_map.clear();
  t.clock_pins.clear();
  v.pins.clear();
  v.nets.clear();
  v.tree.reset();
}

std::shared_ptr<NativeTimingLocalPlanningSnapshot> PersistentTimingGraph::localPlanningSnapshot(
    const py::list& actions, const py::dict& config)
{
  requireLocal(planning_inputs_ != nullptr, "native planning inputs are not initialized");
  return std::shared_ptr<NativeTimingLocalPlanningSnapshot>(
      new NativeTimingLocalPlanningSnapshot(shared_from_this(), planning_inputs_, actions, config));
}

NativeTimingLocalPlanningSnapshot::NativeTimingLocalPlanningSnapshot(
    std::shared_ptr<PersistentTimingGraph> graph, std::shared_ptr<const TimingPlanningInputs> inputs,
    const py::list& actions, const py::dict& config)
    : graph_(std::move(graph)), checkpoint_(graph_->checkpoint()), inputs_(std::move(inputs)),
      config_(config.attr("copy")().cast<py::dict>())
{
  requireLocal(config_.contains("native_endpoint_pin")
              && config_.contains("native_tree_per_driver") && config_["native_tree_per_driver"].cast<bool>()
              && (!config_.contains("native_tree_branch_queue") || !config_["native_tree_branch_queue"].cast<bool>()),
          "local planning requires one requested endpoint and per-driver trees without branch focus");
  auto started = std::chrono::steady_clock::now();
  queue_metadata_ = graph_->timingQueueState(inputs_->steiner.net_ids,
      config_.contains("use_native_timing_path_context") && config_["use_native_timing_path_context"].cast<bool>(),
      config_["native_endpoint_pin"].cast<int>());
  queue_ = queue_metadata_["_native_queue_state"].cast<std::shared_ptr<NativeTimingQueueSnapshot>>();
  timings_["queue_state"] = elapsed(started);
  started = std::chrono::steady_clock::now();
  const auto& m = *graph_->mutation_state_;
  auto prev_cache = inputs_->cached_local_identity;
  bool cache_valid = prev_cache != nullptr
      && prev_cache->topology_revision == graph_->topology_revision_
      && prev_cache->net_states_revision == graph_->net_states_revision_
      && prev_cache->net_state_list.size() == graph_->net_states_.size();
  if (cache_valid) {
    for (std::size_t i = 0; i < prev_cache->net_state_list.size(); ++i) {
      if (prev_cache->net_state_list[i]->epoch != prev_cache->net_state_epochs[i]) {
        cache_valid = false;
        break;
      }
    }
  }
  if (cache_valid) {
    validateLocalActionsAgainstBuffers(actions, prev_cache->buffers, m);
    identity_ = prev_cache;
    pin_count_ = identity_->pin_count;
    node_count_ = identity_->node_count;
    net_count_ = identity_->net_count;
    vertex_count_ = identity_->vertex_count;
    for (auto& item : queue_->items) {
      item.endpoint_pin = pinId(item.endpoint_pin);
      item.driver_pin_id = pinId(item.driver_pin_id);
      item.root_net_id = netId(item.root_net_id);
      for (auto& pin : item.path_pins) pin = pinId(pin);
      item.tree_key =
          std::to_string(item.root_net_id) + ":" + std::to_string(item.driver_pin_id);
      item.focus_net_ids = reachable(item.root_net_id);
    }
  } else {
    auto id_cache = std::make_shared<CachedLocalPlanningIdentity>();
    id_cache->buffers.reserve(m.buffers.size());
    for (const auto& buffer : m.buffers) {
      if (buffer.active) id_cache->buffers.push_back(buffer);
    }
    std::sort(id_cache->buffers.begin(), id_cache->buffers.end(), [](const auto& a, const auto& b) {
      return a.action_id < b.action_id;
    });
    validateLocalActionsAgainstBuffers(actions, id_cache->buffers, m);
    const std::size_t num_buffers = id_cache->buffers.size();
    const int base_pins = inputs_->design.num_pins, base_nets = inputs_->design.num_nets;
    const int movable =
        m.original_node_count - inputs_->carrier.num_terminals - inputs_->carrier.num_terminal_nis;
    id_cache->buffer_coordinates.resize(num_buffers, {0.0, 0.0});
    id_cache->buffer_index_by_node.assign(
        std::max<std::size_t>(0, m.node_names.size() - m.original_node_count), -1);
    id_cache->pin_to_canonical.assign(
        std::max<std::size_t>(0, m.pin_owner.size() - base_pins), -1);
    id_cache->canonical_to_pin.assign(num_buffers * 2, -1);
    id_cache->node_to_canonical.assign(
        std::max<std::size_t>(0, m.node_names.size() - m.original_node_count), -1);
    id_cache->net_to_canonical.assign(
        std::max<std::size_t>(0, m.net_drivers.size() - base_nets), -1);
    id_cache->canonical_to_net.assign(num_buffers, -1);
    id_cache->original_by_net.assign(num_buffers, -1);
    id_cache->virtual_pins.reserve(num_buffers * 2);
    id_cache->source_by_pin.assign(num_buffers * 2, {-1, -1});
    for (std::size_t id = 0; id < num_buffers; ++id) {
      const auto& buffer = id_cache->buffers[id];
      if (inputs_->carrier.has_rebuffer_prefix) {
        const auto name = "__dreamplace_rebuffer_" + std::to_string(buffer.action_id);
        requireLocal(!inputs_->carrier.node_name_set.count(name)
                         && !inputs_->carrier.net_name_set.count(name + "_out"),
                     "overlay identity collides with the base design");
      }
      const auto& state = *graph_->net_states_.at(buffer.source_net);
      const auto physical = std::find_if(state.buffers.begin(), state.buffers.end(), [&](const auto& row) {
        return row.action_id == buffer.action_id;
      });
      requireLocal(physical != state.buffers.end(), "native buffer has no physical topology identity");
      id_cache->buffer_coordinates[id] = state.coordinates.at(physical->input_node);
      id_cache->buffer_index_by_node.at(buffer.node - m.original_node_count) = static_cast<int>(id);
      const int input = base_pins + 2 * static_cast<int>(id);
      const int output = input + 1;
      const int net = base_nets + static_cast<int>(id);
      const int node = movable + static_cast<int>(id);
      id_cache->pin_to_canonical.at(buffer.input_pin - base_pins) = input;
      id_cache->pin_to_canonical.at(buffer.output_pin - base_pins) = output;
      id_cache->canonical_to_pin[2 * id] = buffer.input_pin;
      id_cache->canonical_to_pin[2 * id + 1] = buffer.output_pin;
      id_cache->node_to_canonical.at(buffer.node - m.original_node_count) = node;
      id_cache->net_to_canonical.at(buffer.output_net - base_nets) = net;
      id_cache->canonical_to_net[id] = buffer.output_net;
      id_cache->original_by_net[id] = buffer.source_net;
      id_cache->virtual_pins.push_back({{buffer.source_net, physical->input_node}, input});
      id_cache->virtual_pins.push_back({{buffer.source_net, physical->output_node}, output});
      id_cache->source_by_pin[2 * id] = {buffer.source_net, physical->input_node};
      id_cache->source_by_pin[2 * id + 1] = {buffer.source_net, physical->output_node};
    }
    std::sort(id_cache->virtual_pins.begin(), id_cache->virtual_pins.end());
    id_cache->pin_count = base_pins + 2 * static_cast<int>(num_buffers);
    id_cache->node_count = m.original_node_count + static_cast<int>(num_buffers);
    id_cache->net_count = base_nets + static_cast<int>(num_buffers);
    id_cache->vertex_count =
        static_cast<int>(inputs_->steiner.node_x.size() + 2 * num_buffers);
    identity_ = id_cache;
    pin_count_ = identity_->pin_count;
    node_count_ = identity_->node_count;
    net_count_ = identity_->net_count;
    vertex_count_ = identity_->vertex_count;
    if (!inputs_->active_links || !inputs_->active_links_valid
        || inputs_->active_links_epoch != graph_->epoch_
        || inputs_->active_links_journal_size != graph_->journal_.size()
        || inputs_->active_links_journal_head
               != (graph_->journal_.empty() ? 0 : graph_->journal_.back().sequence)) {
      auto links = std::make_shared<std::vector<CurrentDriverBufferLink>>();
      links->reserve(m.buffers.size());
      for (const auto& buffer : m.buffers) {
        if (!buffer.active) continue;
        requireLocal(buffer.node >= 0
                         && buffer.input_pin >= 0
                         && buffer.output_pin >= 0
                         && buffer.input_pin < static_cast<int>(m.pin_net.size())
                         && buffer.output_pin < static_cast<int>(m.pin_net.size()),
                     "native active buffer link is outside timing mutation state");
        links->push_back({buffer.node,
                          buffer.input_pin,
                          buffer.output_pin,
                          m.pin_net[buffer.input_pin],
                          m.pin_net[buffer.output_pin]});
      }
      std::sort(links->begin(), links->end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.input_net_id, lhs.input_pin_id, lhs.output_net_id)
               < std::tie(rhs.input_net_id, rhs.input_pin_id, rhs.output_net_id);
      });
      inputs_->active_links = std::move(links);
      inputs_->active_links_epoch = graph_->epoch_;
      inputs_->active_links_journal_size = graph_->journal_.size();
      inputs_->active_links_journal_head =
          graph_->journal_.empty() ? 0 : graph_->journal_.back().sequence;
      inputs_->active_links_valid = true;
    }
    id_cache->links.reserve(inputs_->base_links->size() + inputs_->active_links->size());
    auto add_link = [&](int node, int input, int output) {
      id_cache->links.push_back({nodeId(node),
                                 pinId(input),
                                 pinId(output),
                                 netId(m.pin_net.at(input)),
                                 netId(m.pin_net.at(output))});
    };
    for (const auto& link : *inputs_->base_links) {
      add_link(link.instance_id, link.input_pin_id, link.output_pin_id);
    }
    for (const auto& link : *inputs_->active_links) {
      add_link(link.instance_id, link.input_pin_id, link.output_pin_id);
    }
    std::sort(id_cache->links.begin(), id_cache->links.end(), [](const auto& a, const auto& b) {
      return std::tie(a.input_net_id, a.input_pin_id, a.output_net_id)
             < std::tie(b.input_net_id, b.input_pin_id, b.output_net_id);
    });
    id_cache->link_by_output.assign(net_count_, -1);
    for (std::size_t idx = 0; idx < id_cache->links.size(); ++idx) {
      const auto& link = id_cache->links[idx];
      requireLocal(link.output_net_id >= 0 && link.output_net_id < net_count_
                       && id_cache->link_by_output[link.output_net_id] < 0,
                   "duplicate native buffer output net");
      id_cache->link_by_output[link.output_net_id] = static_cast<int>(idx);
    }
    for (auto& item : queue_->items) {
      item.endpoint_pin = pinId(item.endpoint_pin);
      item.driver_pin_id = pinId(item.driver_pin_id);
      item.root_net_id = netId(item.root_net_id);
      for (auto& pin : item.path_pins) pin = pinId(pin);
      item.tree_key =
          std::to_string(item.root_net_id) + ":" + std::to_string(item.driver_pin_id);
      item.focus_net_ids = reachable(item.root_net_id);
    }
    if (prev_cache != nullptr
        && prev_cache->net_states_revision == graph_->net_states_revision_
        && prev_cache->net_state_list.size() == graph_->net_states_.size()) {
      id_cache->net_state_list = prev_cache->net_state_list;
      id_cache->net_state_epochs = prev_cache->net_state_epochs;
      for (std::size_t i = 0; i < id_cache->net_state_list.size(); ++i) {
        const auto& state = *id_cache->net_state_list[i];
        if (state.epoch != id_cache->net_state_epochs[i]) {
          requireSynchronizedTimingTopology(*graph_->mutation_state_, state);
          id_cache->net_state_epochs[i] = state.epoch;
        }
      }
    } else {
      requireSynchronizedTimingTopologies(*graph_->mutation_state_, graph_->net_states_);
      id_cache->net_state_list.reserve(graph_->net_states_.size());
      id_cache->net_state_epochs.reserve(graph_->net_states_.size());
      for (const auto& [_, state] : graph_->net_states_) {
        id_cache->net_state_list.push_back(state);
        id_cache->net_state_epochs.push_back(state->epoch);
      }
    }
    id_cache->topology_revision = graph_->topology_revision_;
    id_cache->net_states_revision = graph_->net_states_revision_;
    inputs_->cached_local_identity = id_cache;
  }
  timings_["identity_links"] = elapsed(started);
  queue_metadata_["planner_payload_item_count"] = queue_->items.size();
  queue_metadata_["planner_payload_rows"] = queue_->items.size();
  queue_metadata_["timing_source"] = "native_graph_local_planning";
}

NativeTimingLocalPlanningSnapshot::~NativeTimingLocalPlanningSnapshot()
{
  if (!inputs_) {
    last_view_.reset();
    return;
  }
  if (last_view_ && last_view_.use_count() == 1) {
    resetLocalPlanningView(*last_view_);
    if (inputs_->local_view_pool.size() < 64) {
      inputs_->local_view_pool.push_back(std::move(last_view_));
    }
  }
  last_view_.reset();
  for (auto& [_, view_ptr] : views_) {
    if (!view_ptr || view_ptr.use_count() != 1) continue;
    resetLocalPlanningView(*view_ptr);
    if (inputs_->local_view_pool.size() < 64) {
      inputs_->local_view_pool.push_back(std::move(view_ptr));
    }
  }
}

int NativeTimingLocalPlanningSnapshot::pinId(int native) const
{
  const int base_pins = inputs_->design.num_pins;
  return native < base_pins ? native : identity_->pin_to_canonical.at(native - base_pins);
}
int NativeTimingLocalPlanningSnapshot::nodeId(int native) const
{
  const int base_nodes = inputs_->design.num_nodes;
  const int movable = base_nodes - inputs_->carrier.num_terminals - inputs_->carrier.num_terminal_nis;
  return native < base_nodes ? (native < movable ? native : native + static_cast<int>(identity_->buffers.size()))
                             : identity_->node_to_canonical.at(native - base_nodes);
}
int NativeTimingLocalPlanningSnapshot::netId(int native) const
{
  const int base_nets = inputs_->design.num_nets;
  return native < base_nets ? native : identity_->net_to_canonical.at(native - base_nets);
}
int NativeTimingLocalPlanningSnapshot::graphNet(int canonical) const
{
  const int base_nets = inputs_->design.num_nets;
  return canonical < base_nets ? canonical : identity_->canonical_to_net.at(canonical - base_nets);
}
int NativeTimingLocalPlanningSnapshot::originalNet(int canonical) const
{
  const int base_nets = inputs_->design.num_nets;
  return canonical < base_nets ? canonical : identity_->original_by_net.at(canonical - base_nets);
}
int NativeTimingLocalPlanningSnapshot::vertex(int owner, int source) const
{
  const std::pair<int, int> key{owner, source};
  const auto found = std::lower_bound(
      identity_->virtual_pins.begin(), identity_->virtual_pins.end(), key,
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs; });
  if (found != identity_->virtual_pins.end() && found->first == key) return found->second;
  requireLocal(source >= 0 && source < static_cast<int>(inputs_->steiner.node_x.size()), "local vertex is outside base Steiner domain");
  return source < inputs_->design.num_pins ? source : source + 2 * static_cast<int>(identity_->buffers.size());
}

void NativeTimingLocalPlanningSnapshot::requireCurrent() const
{
  graph_->validateCheckpoint(checkpoint_);
  requireLocal(graph_->epoch_ == checkpoint_->epoch && graph_->journal_.size() == checkpoint_->journal_size,
          "native local planning snapshot is stale");
  if (graph_->topology_revision_ == identity_->topology_revision
      && graph_->net_states_revision_ == identity_->net_states_revision
      && graph_->net_states_.size() == identity_->net_state_list.size()) {
    for (std::size_t i = 0; i < identity_->net_state_list.size(); ++i) {
      const auto& state = *identity_->net_state_list[i];
      if (state.epoch != identity_->net_state_epochs[i]) {
        requireSynchronizedTimingTopology(*graph_->mutation_state_, state);
        identity_->net_state_epochs[i] = state.epoch;
      }
    }
  } else {
    requireSynchronizedTimingTopologies(*graph_->mutation_state_, graph_->net_states_);
  }
}

std::vector<int> NativeTimingLocalPlanningSnapshot::reachable(int root) const
{
  std::set<int> seen{root}; std::vector<int> pending{root};
  while (!pending.empty()) {
    const int net = pending.back(); pending.pop_back();
    auto it = std::lower_bound(
        identity_->links.begin(), identity_->links.end(), net,
        [](const CurrentDriverBufferLink& link, int target) {
          return link.input_net_id < target;
        });
    for (; it != identity_->links.end() && it->input_net_id == net; ++it) {
      const int child = it->output_net_id;
      requireLocal(seen.insert(child).second, "native buffer net graph contains a cycle");
      pending.push_back(child);
    }
  }
  return {seen.begin(), seen.end()};
}

py::dict NativeTimingLocalPlanningSnapshot::queueState()
{
  auto result = queue_metadata_.attr("copy")().cast<py::dict>();
  result["_native_planning_state"] = shared_from_this();
  return result;
}

std::shared_ptr<CurrentDriverTree> NativeTimingLocalPlanningSnapshot::currentTree(int net)
{
  const auto it = trees_.find(net);
  if (it != trees_.end()) return it->second;
  return view(net)->tree;
}

py::dict NativeTimingLocalPlanningSnapshot::plan(const py::dict& request, const py::dict& config)
{
  requireCurrent();
  const int endpoint = request["endpoint_pin"].cast<int>(), root = request["root_net_id"].cast<int>();
  const auto item = std::find_if(queue_->items.begin(), queue_->items.end(), [&](const auto& row) {
    return row.endpoint_pin == endpoint && row.root_net_id == root;
  });
  requireLocal(item != queue_->items.end() && request["net_id"].cast<int>() == root, "local request is outside selected snapshot");
  last_view_ = view(root);
  NativeTreePlannerViews input{last_view_->projection.design, last_view_->timing, *graph_->queue_liberty_,
      inputs_->characterization, inputs_->config, last_view_->tree,
      graph_->mutation_state_->net_names.at(graphNet(root)), -1, 0, &timings_};
  const int base_pins = inputs_->design.num_pins;
  const int driver_pin = last_view_->tree->driver_pin_id;
  const int native_driver = driver_pin < base_pins
      ? driver_pin
      : identity_->canonical_to_pin.at(driver_pin - base_pins);
  std::vector<int> native_path;
  native_path.reserve(item->path_pins.size());
  for (const int pin : item->path_pins) {
    if (pin >= 0) {
      native_path.push_back(pin < base_pins ? pin : identity_->canonical_to_pin.at(pin - base_pins));
    }
  }
  const auto [resize_node, resize_cell] =
      graph_->findDriverResizeCandidate(native_driver, native_path, config);
  input.driver_resize_node_id = resize_node;
  input.driver_resize_cell_index = resize_cell;
  ++planned_;
  return planNativeCurrentTree(request, input, config);
}

std::vector<int> NativeTimingLocalPlanningSnapshot::sourceNets(const py::list& actions) const
{
  requireCurrent();
  const auto& m = *graph_->mutation_state_;
  std::set<int> sources;
  for (const auto& value : actions) {
    const auto row = value.cast<py::dict>();
    if (row.contains("action_kind")
        && row["action_kind"].cast<std::string>() == "resize_cell") {
      const int owner = row["node_id"].cast<int>();
      requireLocal(owner >= 0 && owner < m.original_node_count,
                   "resize node is outside original nodes");
      for (const int pin : m.node_pins[owner]) {
        const int net = m.pin_net[pin];
        if (net < 0) continue;
        const int source = sourceNetForMutationNet(m, net);
        requireLocal(source >= 0, "resize input has no source net provenance");
        sources.insert(source);
      }
      continue;
    }
    const int net = row["net_id"].cast<int>();
    requireLocal(last_view_ && last_view_->nets.count(net), "action is outside the local planning scope");
    int root = net; std::set<int> seen;
    while (root >= 0 && root < static_cast<int>(identity_->link_by_output.size())
           && identity_->link_by_output[root] >= 0) {
      requireLocal(seen.insert(root).second, "native buffer net graph contains a cycle");
      root = identity_->links[identity_->link_by_output[root]].input_net_id;
    }
    for (const int child : reachable(root)) sources.insert(originalNet(child));
  }
  return {sources.begin(), sources.end()};
}

py::list NativeTimingLocalPlanningSnapshot::projectActions(const py::list& actions) const
{
  requireCurrent();
  requireLocal(last_view_ != nullptr || py::len(actions) == 0, "local actions require a planned tree");
  if (py::len(actions) == 0) return py::list();
  const auto& m = *graph_->mutation_state_;
  py::list resize_projected;
  py::list buffer_actions;
  for (const auto& value : actions) {
    const auto row = value.cast<py::dict>();
    if (row.contains("action_kind")
        && row["action_kind"].cast<std::string>() == "resize_cell") {
      const int net = row["net_id"].cast<int>();
      requireLocal(last_view_->nets.count(net), "resize action is outside the local planning scope");
      const int owner = row["node_id"].cast<int>();
      const int target_cell = row["target_cell_index"].cast<int>();
      requireLocal(owner >= 0 && owner < m.original_node_count,
                   "resize node is outside original nodes");
      const int prev_cell = m.node_cells.at(owner);
      const int source_net = originalNet(net);
      py::dict projected = row.attr("copy")().cast<py::dict>();
      projected["action_kind"] = "resize_cell";
      projected["net_id"] = source_net;
      projected["net_name"] = m.net_names.at(source_net);
      projected["node_id"] = owner;
      projected["instance_name"] = m.node_names.at(owner);
      projected["previous_cell_index"] = prev_cell;
      projected["previous_master_name"] = m.cell_names.at(prev_cell);
      projected["target_cell_index"] = target_cell;
      projected["target_master_name"] = m.cell_names.at(target_cell);
      resize_projected.append(std::move(projected));
    } else {
      buffer_actions.append(row);
    }
  }
  if (py::len(buffer_actions) == 0) return resize_projected;
  auto& p = last_view_->projection;
  if (p.net_names.empty()) {
    p.net_names.resize(net_count_);
    for (const int net : last_view_->nets) {
      p.net_names[net] = graph_->mutation_state_->net_names.at(graphNet(net));
    }
  }
  for (const auto& value : buffer_actions) {
    const auto row = value.cast<py::dict>();
    const int net = row["net_id"].cast<int>();
    requireLocal(last_view_->nets.count(net), "action is outside the local planning scope");
    for (const char* field : {"source_parent_node_id", "source_child_node_id"}) {
      requireLocal(findPartitionProvenance(p.actions.state_provenance, {net, row[field].cast<int>()})
                  != p.actions.state_provenance.end(), "action source is outside local partition membership");
    }
    for (const int pin : row["downstream_pins"].cast<std::vector<int>>()) {
      requireLocal(last_view_->pins.count(pin), "action load is outside local tree membership");
    }
  }
  std::vector<std::shared_ptr<IncrementalTopologyState>> active_states;
  std::set<int> seen_owners;
  for (const int net : last_view_->nets) {
    const int owner = originalNet(net);
    if (!seen_owners.insert(owner).second) continue;
    const auto it = graph_->net_states_.find(owner);
    if (it != graph_->net_states_.end()) active_states.push_back(it->second);
  }
  py::list projected_buffers = projectPlanningActions(
      inputs_->carrier.identity, p, *graph_->queue_liberty_, active_states, buffer_actions);
  if (py::len(resize_projected) == 0) return projected_buffers;
  for (const auto& item : projected_buffers) resize_projected.append(item);
  return resize_projected;
}

py::dict NativeTimingLocalPlanningSnapshot::summary() const
{
  py::dict result;
  result["status"] = "ok"; result["epoch"] = checkpoint_->epoch;
  result["pin_count"] = pin_count_; result["net_count"] = net_count_;
  result["queue_item_count"] = queue_->items.size(); result["planner_payload_rows"] = queue_->items.size();
  result["planned_requests"] = planned_; result["imported_tree_count"] = trees_.size();
  result["local_filled_pin_count"] = filled_pins_; result["local_filled_vertex_count"] = filled_vertices_;
  result["phase_timings_ms"] = timings_;
  return result;
}

void registerTimingLocalBindings(py::module_& module)
{
  py::class_<NativeTimingLocalPlanningSnapshot, std::shared_ptr<NativeTimingLocalPlanningSnapshot>>(module, "NativeTimingLocalPlanningSnapshot")
      .def("queue_state", &NativeTimingLocalPlanningSnapshot::queueState)
      .def("plan", &NativeTimingLocalPlanningSnapshot::plan)
      .def("source_nets", &NativeTimingLocalPlanningSnapshot::sourceNets)
      .def("project_actions", &NativeTimingLocalPlanningSnapshot::projectActions)
      .def("current_tree", &NativeTimingLocalPlanningSnapshot::currentTree)
      .def("summary", &NativeTimingLocalPlanningSnapshot::summary);
}

}  // namespace rebuffer
