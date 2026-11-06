// SPDX-License-Identifier: BSD-3-Clause
// Lazy current-tree views; canonical physical IDs remain identical to full projection.
#include "rebuffer_timing_local.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_overlay_actions.hh"

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
}  // namespace

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
  std::map<int, py::dict> records;
  for (const auto& item : actions) {
    const auto row = item.cast<py::dict>();
    requireLocal(records.emplace(row["session_buffer_id"].cast<int>(), row).second, "duplicate local action identity");
  }
  for (const auto& buffer : m.buffers) if (buffer.active) buffers_.push_back(buffer);
  std::sort(buffers_.begin(), buffers_.end(), [](const auto& a, const auto& b) { return a.action_id < b.action_id; });
  requireLocal(records.size() == buffers_.size(), "local actions do not match active native buffers");
  const int base_pins = inputs_->design.num_pins, base_nets = inputs_->design.num_nets;
  const int movable = m.original_node_count - inputs_->carrier.num_terminals - inputs_->carrier.num_terminal_nis;
  for (std::size_t id = 0; id < buffers_.size(); ++id) {
    const auto& buffer = buffers_[id];
    requireLocal(records.count(buffer.action_id) != 0, "native buffer has no action metadata");
    requireLocal(records.at(buffer.action_id)["buffer_master_name"].cast<std::string>()
                     == m.cell_names.at(m.node_cells.at(buffer.node)),
                 "local action master differs from native timing owner");
    const auto name = "__dreamplace_rebuffer_" + std::to_string(buffer.action_id);
    requireLocal(!inputs_->carrier.node_name_set.count(name) && !inputs_->carrier.net_name_set.count(name + "_out"),
            "overlay identity collides with the base design");
    const auto& state = *graph_->net_states_.at(buffer.source_net);
    const auto physical = std::find_if(state.buffers.begin(), state.buffers.end(), [&](const auto& row) {
      return row.action_id == buffer.action_id;
    });
    requireLocal(physical != state.buffers.end(), "native buffer has no physical topology identity");
    const int input = base_pins + 2 * id, output = input + 1, net = base_nets + id, node = movable + id;
    for (const auto& pair : {std::make_pair(buffer.input_pin, input), std::make_pair(buffer.output_pin, output)}) {
      pin_to_canonical_.emplace(pair); canonical_to_pin_.emplace(pair.second, pair.first);
    }
    node_to_canonical_[buffer.node] = node; canonical_to_node_[node] = buffer.node;
    net_to_canonical_[buffer.output_net] = net; canonical_to_net_[net] = buffer.output_net;
    original_by_net_[net] = buffer.source_net;
    virtual_pins_[{buffer.source_net, physical->input_node}] = input;
    virtual_pins_[{buffer.source_net, physical->output_node}] = output;
    source_by_pin_[input] = {buffer.source_net, physical->input_node};
    source_by_pin_[output] = {buffer.source_net, physical->output_node};
  }
  pin_count_ = base_pins + 2 * buffers_.size();
  node_count_ = m.original_node_count + buffers_.size();
  net_count_ = base_nets + buffers_.size();
  vertex_count_ = inputs_->steiner.node_x.size() + 2 * buffers_.size();
  if (!inputs_->active_links || !inputs_->active_links_valid
      || inputs_->active_links_epoch != graph_->epoch_
      || inputs_->active_links_journal_size != graph_->journal_.size()
      || inputs_->active_links_journal_head
             != (graph_->journal_.empty() ? 0 : graph_->journal_.back().sequence)) {
    auto links = std::make_shared<std::vector<CurrentDriverBufferLink>>();
    const auto& m = *graph_->mutation_state_;
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
    inputs_->active_links_journal_head = graph_->journal_.empty()
        ? 0 : graph_->journal_.back().sequence;
    inputs_->active_links_valid = true;
  }
  auto add_link = [&](int node, int input, int output) {
    links_.push_back({nodeId(node), pinId(input), pinId(output), netId(m.pin_net.at(input)), netId(m.pin_net.at(output))});
  };
  for (const auto& link : *inputs_->base_links) add_link(link.instance_id, link.input_pin_id, link.output_pin_id);
  for (const auto& link : *inputs_->active_links) {
    add_link(link.instance_id, link.input_pin_id, link.output_pin_id);
  }
  std::sort(links_.begin(), links_.end(), [](const auto& a, const auto& b) {
    return std::tie(a.input_net_id, a.input_pin_id, a.output_net_id) < std::tie(b.input_net_id, b.input_pin_id, b.output_net_id);
  });
  for (const auto& link : links_) {
    requireLocal(input_by_output_.emplace(link.output_net_id, link.input_net_id).second, "duplicate native buffer output net");
    outputs_by_input_[link.input_net_id].push_back(link.output_net_id);
  }
  for (auto& item : queue_->items) {
    item.endpoint_pin = pinId(item.endpoint_pin); item.driver_pin_id = pinId(item.driver_pin_id);
    item.root_net_id = netId(item.root_net_id);
    for (auto& pin : item.path_pins) pin = pinId(pin);
    item.tree_key = std::to_string(item.root_net_id) + ":" + std::to_string(item.driver_pin_id);
    item.focus_net_ids = reachable(item.root_net_id);
  }
  timings_["identity_links"] = elapsed(started);
  queue_metadata_["planner_payload_item_count"] = queue_->items.size();
  queue_metadata_["planner_payload_rows"] = queue_->items.size();
  queue_metadata_["timing_source"] = "native_graph_local_planning";
}

int NativeTimingLocalPlanningSnapshot::pinId(int native) const
{
  return native < inputs_->design.num_pins ? native : pin_to_canonical_.at(native);
}
int NativeTimingLocalPlanningSnapshot::nodeId(int native) const
{
  const int movable = inputs_->design.num_nodes - inputs_->carrier.num_terminals - inputs_->carrier.num_terminal_nis;
  return native < inputs_->design.num_nodes ? (native < movable ? native : native + buffers_.size())
                                          : node_to_canonical_.at(native);
}
int NativeTimingLocalPlanningSnapshot::netId(int native) const
{
  return native < inputs_->design.num_nets ? native : net_to_canonical_.at(native);
}
int NativeTimingLocalPlanningSnapshot::graphNet(int canonical) const
{
  return canonical < inputs_->design.num_nets ? canonical : canonical_to_net_.at(canonical);
}
int NativeTimingLocalPlanningSnapshot::originalNet(int canonical) const
{
  return canonical < inputs_->design.num_nets ? canonical : original_by_net_.at(canonical);
}
int NativeTimingLocalPlanningSnapshot::vertex(int owner, int source) const
{
  const auto found = virtual_pins_.find({owner, source});
  if (found != virtual_pins_.end()) return found->second;
  requireLocal(source >= 0 && source < static_cast<int>(inputs_->steiner.node_x.size()), "local vertex is outside base Steiner domain");
  return source < inputs_->design.num_pins ? source : source + 2 * buffers_.size();
}

void NativeTimingLocalPlanningSnapshot::requireCurrent() const
{
  graph_->validateCheckpoint(checkpoint_);
  requireLocal(graph_->epoch_ == checkpoint_->epoch && graph_->journal_.size() == checkpoint_->journal_size,
          "native local planning snapshot is stale");
  for (const auto& [_, state] : graph_->net_states_) requireSynchronizedTimingTopology(*graph_->mutation_state_, *state);
}

std::vector<int> NativeTimingLocalPlanningSnapshot::reachable(int root) const
{
  std::set<int> seen{root}; std::vector<int> pending{root};
  while (!pending.empty()) {
    const int net = pending.back(); pending.pop_back();
    const auto found = outputs_by_input_.find(net);
    if (found == outputs_by_input_.end()) continue;
    for (const int child : found->second) {
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
      last_view_->projection.net_names.at(root), -1, 0, &timings_};
  ++planned_;
  return planNativeCurrentTree(request, input, config);
}

std::vector<int> NativeTimingLocalPlanningSnapshot::sourceNets(const py::list& actions) const
{
  requireCurrent();
  std::set<int> sources;
  for (const auto& value : actions) {
    const int net = value.cast<py::dict>()["net_id"].cast<int>();
    requireLocal(last_view_ && last_view_->nets.count(net), "action is outside the local planning scope");
    int root = net; std::set<int> seen;
    while (input_by_output_.count(root)) {
      requireLocal(seen.insert(root).second, "native buffer net graph contains a cycle");
      root = input_by_output_.at(root);
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
  const auto& p = last_view_->projection;
  for (const auto& value : actions) {
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
  std::vector<std::shared_ptr<IncrementalTopologyState>> states;
  for (const auto& [_, state] : graph_->net_states_) states.push_back(state);
  return projectPlanningActions(inputs_->carrier.identity, p, *graph_->queue_liberty_, states, actions);
}

py::dict NativeTimingLocalPlanningSnapshot::summary() const
{
  py::dict result;
  result["status"] = "ok"; result["epoch"] = checkpoint_->epoch;
  result["pin_count"] = pin_count_; result["net_count"] = net_count_;
  result["queue_item_count"] = queue_->items.size(); result["planner_payload_rows"] = queue_->items.size();
  result["planned_requests"] = planned_; result["imported_tree_count"] = views_.size();
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
