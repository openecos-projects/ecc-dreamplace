// SPDX-License-Identifier: BSD-3-Clause
// Build scheduler-ready native queue state directly from the timing owner.
#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"

#include "rebuffer_driver_tree.hh"

#include "rebuffer_repair.hh"

#include "rebuffer_queue_snapshot.hh"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>

#include "timing_legacy_paths.h"

#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {

py::dict PersistentTimingGraph::initializeQueue(const py::dict& values,
    const std::vector<std::uint8_t>& original_node_is_buffer)
{
  if (queue_liberty_) throw py::value_error("queue inputs are immutable");
  if (!mutation_state_) throw py::value_error("queue initialization requires mutation inputs");
  const auto& m = *mutation_state_;
  if (original_node_is_buffer.size() != static_cast<std::size_t>(m.original_node_count)
      || std::any_of(original_node_is_buffer.begin(), original_node_is_buffer.end(),
                     [](auto value) { return value > 1; })) {
    throw py::value_error("queue original node buffer flags do not align");
  }
  auto library = std::make_shared<LibertyView>(rebuffer_python::libertyFromDict(values));
  if (library->dbu != carrier_.attr("dbu").cast<int>()) {
    throw py::value_error("queue Liberty units do not match timing owner");
  }
  if (library->cells.size() != m.cell_names.size()) {
    throw py::value_error("queue Liberty cell count does not match timing owner");
  }
  for (std::size_t cell = 0; cell < library->cells.size(); ++cell) {
    if (library->cells[cell].name != m.cell_names[cell]) {
      throw py::value_error("queue Liberty cell identity does not match timing owner");
    }
  }
  // Publish only after every input is validated and owned.
  auto flags = original_node_is_buffer;
  queue_liberty_ = std::move(library);
  queue_original_buffer_nodes_ = std::move(flags);
  py::dict result;
  result["status"] = "ok";
  result["cell_count"] = queue_liberty_->cells.size();
  result["original_node_count"] = queue_original_buffer_nodes_.size();
  return result;
}

py::dict PersistentTimingGraph::timingQueueState(
    const std::vector<int>& original_net_by_net, bool use_path_context,
    std::optional<int> endpoint_pin) const
{
  const auto started = std::chrono::steady_clock::now();
  if (!queue_liberty_ || !mutation_state_) {
    throw py::value_error("timing queue requires initialized queue and mutation inputs");
  }
  const auto& m = *mutation_state_;
  const auto& index = *timing_index_;
  const auto& p = *propagation_program_;
  if (original_net_by_net.size() != m.base_net_pins.size()) {
    throw py::value_error("queue original net identities do not align");
  }
  const auto slacks = endpointSlacks();
  if (endpoint_pin && std::find(p.metric_pins.begin(), p.metric_pins.end(), *endpoint_pin)
                          == p.metric_pins.end()) {
    throw py::value_error("requested queue pin is not a timing endpoint");
  }
  std::vector<int64_t> active;
  for (std::size_t row = 0; row < p.metric_pins.size(); ++row) {
    if (endpoint_pin && p.metric_pins[row] != *endpoint_pin) continue;
    if (std::isfinite(slacks[row]) && slacks[row] < 0) active.push_back(p.metric_pins[row]);
  }
  const auto path_started = std::chrono::steady_clock::now();
  const auto paths = nativeSetupPaths(active, 0, 0, 1e-3, true);
  const double path_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - path_started).count();
  std::map<int64_t, const dreamplace::timing::SetupPathTrace*> selected;
  for (std::size_t row = 0; row < paths.batch.states.size(); ++row) {
    selected.emplace(paths.batch.states[row].pin, &paths.batch.traces[row]);
  }
  const auto& rise = state_.at("pin_rtran");
  const auto& fall = state_.at("pin_ftran");
  const auto& ra = state_.at("pin_rAAT");
  const auto& fa = state_.at("pin_fAAT");
  const auto& rr = state_.at("pin_rRAT");
  const auto& fr = state_.at("pin_fRAT");
  auto cell_for_node = [&](int node) {
    return node < 0 || node >= static_cast<int>(m.node_cells.size()) ? -1 : m.node_cells[node];
  };
  auto is_load = [&](int pin) {
    if (pin < 0 || pin >= static_cast<int>(m.pin_net.size())) return false;
    const int net = m.pin_net[pin];
    return net >= 0 && net < static_cast<int>(m.net_drivers.size()) && m.net_drivers[net] != pin;
  };
  const PathDriverAccess access{static_cast<int>(index.pins.size()), m.pin_net,
      m.net_drivers, m.pin_owner, rise, fall, state_.at("pin_net_cap_rise"), cell_for_node,
      [&](int pin) { return std::find(index.clocks.begin(), index.clocks.end(), pin) != index.clocks.end(); },
      [&](int driver, double& input_rise, double& input_fall) {
        const int node = m.pin_owner[driver], cell = cell_for_node(node);
        if (cell < 0) return false;
        return driverInputSlewFromPins(driver, index.pins.size(), queue_liberty_->cells[cell],
            m.node_pins[node], rise, fall, [&](const TimingArcView& arc, int pin) {
              const auto separator = index.pins[pin].find_last_of(":/");
              const auto port = separator == std::string::npos ? index.pins[pin]
                  : index.pins[pin].substr(separator + 1);
              if (!arc.source_name.empty() && !port.empty()) return arc.source_name == port;
              const auto begin = m.library_pin_names.begin() + m.cell_pin_start[cell];
              const auto end = m.library_pin_names.begin() + m.cell_pin_start[cell + 1];
              const auto found = std::find(begin, end, m.pin_ports[pin]);
              const int offset = found == end ? -1 : std::distance(begin, found);
              return arc.source == offset;
            }, is_load, input_rise, input_fall);
      }};
  auto net_map = original_net_by_net;
  net_map.resize(m.net_drivers.size(), -1);
  for (const auto& buffer : m.buffers) {
    net_map.at(buffer.output_net) = net_map.at(buffer.source_net);
  }
  const TimingPathPredecessors predecessors{index, m.canonical_schedule};
  std::vector<TimingQueueItem> items;
  std::size_t fallback_count = 0;
  std::size_t path_context_count = 0;
  double ranking_ms = 0.0;
  for (std::size_t row = 0; row < p.metric_pins.size(); ++row) {
    const int endpoint = p.metric_pins[row];
    if (endpoint_pin && endpoint != *endpoint_pin) continue;
    const double slack = slacks[row];
    if (!std::isfinite(slack) || slack >= 0) continue;
    std::vector<int> path;
    const auto found = selected.find(endpoint);
    if (found != selected.end()) {
      path.assign(found->second->pins.begin(), found->second->pins.end());
    }
    else {
      ++fallback_count;
      const auto legacy = dreamplace::timing::traceLegacySlackPath(endpoint, index.pins.size(),
          std::max<std::size_t>(1, index.pins.size()), 1e-3,
          [&](int64_t pin, const auto& visit) {
            predecessors(pin, [&](const auto& edge) { visit(edge.pin); });
          },
          [&](int64_t pin) { return pin < p.pin_count && p.start_row_by_pin[pin] >= 0; },
          [&](int64_t pin, bool rising) { return rising ? rr[pin] - ra[pin] : fr[pin] - fa[pin]; },
          [&](int64_t pin, bool rising) { return rising ? ra[pin] : fa[pin]; },
          [](int64_t, int64_t) { return int64_t{-1}; });
      path.assign(legacy.pins.begin(), legacy.pins.end());
    }
    if (path.empty()) continue;
    TimingView context;
    const bool has_context = use_path_context && found != selected.end()
        && found->second->valid != 0;
    if (has_context) {
      const auto& trace = *found->second;
      context.path_transitions.assign(trace.transitions.begin(), trace.transitions.end());
      for (std::size_t edge = 0; edge < trace.arc_ids.size(); ++edge) {
        const auto arc = trace.arc_ids[edge];
        context.path_arc_local_ids.push_back(arc >= 0 ? index.cells.at(arc)[6] : -1);
        if (arc < 0) {
          context.path_arc_delays.push_back(0.0);
          continue;
        }
        const bool input_rise = trace.transitions[edge] == 0;
        const bool output_rise = trace.transitions[edge + 1] == 0;
        const char* field = output_rise
            ? (input_rise ? "cell_arc_rr_delays" : "cell_arc_fr_delays")
            : (input_rise ? "cell_arc_rf_delays" : "cell_arc_ff_delays");
        context.path_arc_delays.push_back(state_.at(field).at(arc));
      }
      ++path_context_count;
    }
    // The opt-in context copies only this path's edge metadata. Pin timing
    // remains borrowed from the graph through PathDriverAccess.
    std::vector<double> driver_scores;
    const auto ranking_started = std::chrono::steady_clock::now();
    const auto drivers = rankPathDriverPins(access, *queue_liberty_, path,
        has_context ? &context : nullptr, has_context, &driver_scores);
    ranking_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ranking_started).count();
    std::vector<int> nets;
    auto add_net = [&](int net) {
      if (net >= 0 && net < static_cast<int>(m.net_drivers.size())
          && std::find(nets.begin(), nets.end(), net) == nets.end()) nets.push_back(net);
    };
    for (int driver : drivers) add_net(m.pin_net[driver]);
    if (nets.empty()) for (int pin : path) add_net(m.pin_net[pin]);
    for (int net : nets) {
      const int driver = m.net_drivers[net];
      if (driver < 0 || driver >= static_cast<int>(index.pins.size())) continue;
      TimingQueueItem item;
      item.snapshot_index = items.size();
      item.endpoint_pin = endpoint; item.endpoint_pin_name = index.pins[endpoint];
      item.endpoint_slack_ps = slack; item.path_pins = path;
      item.net_ids.push_back(net_map[net]);
      item.root_net_id = net; item.driver_pin_id = driver;
      const auto ranked = std::find(drivers.begin(), drivers.end(), driver);
      if (ranked != drivers.end()) {
        item.driver_score_ps = driver_scores[std::distance(drivers.begin(), ranked)];
        item.has_driver_score = std::isfinite(item.driver_score_ps);
      }
      item.tree_key = std::to_string(net) + ":" + std::to_string(driver);
      items.push_back(std::move(item));
    }
  }
  auto result = makeTimingQueueState(std::move(items), [&](int driver) {
    const int node = m.pin_owner[driver];
    if (node < static_cast<int>(queue_original_buffer_nodes_.size())) {
      return node >= 0 && queue_original_buffer_nodes_[node] != 0;
    }
    const int cell = cell_for_node(node);
    return cell >= 0 && queue_liberty_->cells[cell].is_buffer;
  });
  result["timing_source"] = "persistent_graph";
  result["graph_epoch"] = epoch_;
  py::dict statistics;
  statistics["active_endpoint_count"] = active.size();
  statistics["legacy_fallback_count"] = fallback_count;
  if (use_path_context) statistics["path_context_count"] = path_context_count;
  statistics["setup_path_state_count"] = paths.batch.failing_state_count;
  statistics["traced_endpoint_count"] = paths.batch.states.size();
  statistics["path_extraction_ms"] = path_ms;
  statistics["driver_ranking_ms"] = ranking_ms;
  if (endpoint_pin && !paths.batch.states.empty()) {
    statistics["endpoint_test_id"] = paths.batch.states.front().test_id;
    statistics["endpoint_transition"] = paths.batch.states.front().transition;
    statistics["path_valid"] = paths.batch.traces.front().valid != 0;
  }
  statistics["runtime_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  result["graph_queue_statistics"] = std::move(statistics);
  return result;
}

void PersistentTimingGraph::applyQueueGradientScores(py::dict& snapshot,
                                                    const py::dict& config) const
{
  if (!config.contains("gradient_ordering") || !config["gradient_ordering"].cast<bool>()) return;
  if (!config.contains("gradient_window_tiebreak")
      || !config["gradient_window_tiebreak"].cast<bool>()) {
    throw py::value_error("native gradient ordering requires the primary slack window");
  }
  if (!config.contains("gradient_score_epoch")
      || config["gradient_score_epoch"].cast<std::uint64_t>() != epoch_
      || !config.contains("gradient_score_by_driver_name")) {
    throw py::value_error("native gradient scores must match the current timing graph epoch");
  }
  if (!config.contains("gradient_slack_window_ps")) {
    throw py::value_error("native gradient ordering requires a bounded slack window");
  }
  const double slack_window_ps = config["gradient_slack_window_ps"].cast<double>();
  if (!std::isfinite(slack_window_ps) || slack_window_ps < 0) {
    throw py::value_error("native gradient slack window must be finite and nonnegative");
  }
  const auto scores = config["gradient_score_by_driver_name"].cast<std::map<std::string, double>>();
  for (const auto& [name, score] : scores) {
    if (!std::isfinite(score) || score < 0) {
      throw py::value_error("native gradient score is not finite and nonnegative: " + name);
    }
  }
  auto queue = snapshot["_native_queue_state"].cast<std::shared_ptr<NativeTimingQueueSnapshot>>();
  std::size_t matched = 0;
  for (auto& item : queue->items) {
    const auto found = scores.find(timing_index_->pins.at(item.driver_pin_id));
    if (found != scores.end()) {
      item.gradient_score = found->second;
      item.has_gradient_score = true;
      ++matched;
    }
  }
  snapshot["gradient_score_epoch"] = epoch_;
  snapshot["gradient_scored_item_count"] = matched;
  snapshot["gradient_slack_window_ps"] = slack_window_ps;
}

}  // namespace rebuffer

// --- rebuffer_timing_queue_projection.cc ---
// Project graph-owned queue identities onto one current physical carrier.

#include <algorithm>
#include <chrono>
#include <unordered_map>


namespace py = pybind11;
namespace rebuffer {
namespace {
std::unordered_map<std::string, int> nameMap(
    const std::vector<std::string>& names, const char* kind)
{
  std::unordered_map<std::string, int> result;
  result.reserve(names.size());
  for (std::size_t id = 0; id < names.size(); ++id) {
    if (!result.emplace(names[id], id).second) {
      throw py::value_error(std::string("queue carrier has duplicate ") + kind);
    }
  }
  return result;
}
}  // namespace

py::dict PersistentTimingGraph::timingQueueStateForCarrier(
    const std::vector<int>& original_net_by_net, const py::object& carrier,
    bool use_path_context) const
{
  if (!mutation_state_) {
    throw py::value_error("queue projection requires native mutation inputs");
  }
  const auto started = std::chrono::steady_clock::now();
  const auto& m = *mutation_state_;
  const auto pins = carrier.attr("pin_names").cast<std::vector<std::string>>();
  const auto nets = carrier.attr("net_names").cast<std::vector<std::string>>();
  const auto pin_net = carrier.attr("pin2net_map").cast<std::vector<int>>();
  const auto drivers = carrier.attr("net2driver_pin_map").cast<std::vector<int>>();
  const auto nodes = carrier.attr("node_names").cast<std::vector<std::string>>();
  const auto masters = carrier.attr("node_master_names").cast<std::vector<std::string>>();
  const auto pin_owner = carrier.attr("pin2node_map").cast<std::vector<int>>();
  if (pin_net.size() != pins.size() || pin_owner.size() != pins.size()
      || drivers.size() != nets.size() || masters.size() != nodes.size()) {
    throw py::value_error("queue carrier identity arrays do not align");
  }
  const auto pin_lookup = nameMap(pins, "pin names");
  const auto net_lookup = nameMap(nets, "net names");
  std::vector<int> pin_map(timing_index_->pins.size(), -1);
  std::vector<int> net_map(m.net_names.size(), -1);
  std::unordered_map<int, int> node_aliases;
  if (py::hasattr(carrier, "session_buffer_pin_ids")) {
    const auto aliases = carrier.attr("session_buffer_pin_ids")
        .cast<std::map<int, std::array<int, 2>>>();
    for (const auto& [action, target] : aliases) {
      const auto found = std::find_if(m.buffers.begin(), m.buffers.end(),
          [&](const auto& buffer) { return buffer.active && buffer.action_id == action; });
      if (found == m.buffers.end()
          || std::count_if(m.buffers.begin(), m.buffers.end(),
               [&](const auto& buffer) { return buffer.active && buffer.action_id == action; }) != 1
          || target[0] < 0 || target[1] < 0 || target[0] == target[1]
          || target[0] >= static_cast<int>(pins.size()) || target[1] >= static_cast<int>(pins.size())
          || pin_owner[target[0]] != pin_owner[target[1]]) {
        throw py::value_error("queue carrier has invalid session pin identities");
      }
      pin_map[found->input_pin] = target[0];
      pin_map[found->output_pin] = target[1];
      net_map[found->output_net] = pin_net[target[1]];
      node_aliases.emplace(found->node, pin_owner[target[0]]);
    }
  }
  std::vector<unsigned char> used_pins(pins.size(), 0), used_nets(nets.size(), 0);
  std::size_t live_pins = 0, live_nets = 0;
  std::size_t remapped_pins = 0, remapped_nets = 0;
  for (std::size_t id = 0; id < pin_map.size(); ++id) {
    if (!m.pin_active[id]) continue;
    ++live_pins;
    if (pin_map[id] < 0) {
      const auto found = pin_lookup.find(timing_index_->pins[id]);
      if (found == pin_lookup.end()) {
        throw py::value_error("queue carrier is missing an active timing pin");
      }
      pin_map[id] = found->second;
    }
    if (used_pins.at(pin_map[id])++) {
      throw py::value_error("queue carrier session pin identities overlap");
    }
    remapped_pins += pin_map[id] != static_cast<int>(id);
  }
  for (std::size_t id = 0; id < net_map.size(); ++id) {
    // Original dangling nets still exist in the physical carrier; retired
    // synthetic output nets have neither a driver nor any active pins.
    if (id >= m.base_net_pins.size() && m.net_drivers[id] < 0) continue;
    ++live_nets;
    if (net_map[id] < 0) {
      const auto found = net_lookup.find(m.net_names[id]);
      if (found == net_lookup.end()) {
        throw py::value_error("queue carrier is missing an active timing net");
      }
      net_map[id] = found->second;
    }
    if (net_map[id] >= static_cast<int>(nets.size()) || used_nets.at(net_map[id])++) {
      throw py::value_error("queue carrier session net identities overlap");
    }
    remapped_nets += net_map[id] != static_cast<int>(id);
  }
  if (live_pins != pins.size() || live_nets != nets.size()) {
    throw py::value_error("queue carrier has different active timing identities");
  }
  for (std::size_t pin = 0; pin < pin_map.size(); ++pin) {
    if (pin_map[pin] < 0) continue;
    const int target = pin_map[pin], source_net = m.pin_net[pin];
    if ((source_net < 0 ? -1 : net_map.at(source_net)) != pin_net[target]) {
      throw py::value_error("queue carrier has different pin net partitions");
    }
    const int source_node = m.pin_owner[pin], target_node = pin_owner[target];
    const auto alias = node_aliases.find(source_node);
    if (source_node < 0 || target_node < 0 || target_node >= static_cast<int>(nodes.size())
        || (alias != node_aliases.end() ? alias->second != target_node
                                      : nodes[target_node] != m.node_names.at(source_node))) {
      throw py::value_error("queue carrier has different pin owners");
    }
    const int cell = m.node_cells[source_node];
    if (cell >= 0 && masters[target_node] != m.cell_names.at(cell)) {
      throw py::value_error("queue carrier has different instance masters");
    }
  }
  for (std::size_t net = 0; net < net_map.size(); ++net) {
    if (net_map[net] < 0) continue;
    const int driver = m.net_drivers[net];
    if ((driver < 0 ? -1 : pin_map.at(driver)) != drivers[net_map[net]]) {
      throw py::value_error("queue carrier has different net drivers");
    }
  }
  const double projection_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  auto result = timingQueueState(original_net_by_net, use_path_context);
  auto snapshot = result["_native_queue_state"]
      .cast<std::shared_ptr<NativeTimingQueueSnapshot>>();
  projectTimingQueueItems(snapshot->items, pin_map, net_map);
  auto statistics = result["graph_queue_statistics"].cast<py::dict>();
  statistics["projection_ms"] = projection_ms;
  statistics["remapped_pin_count"] = remapped_pins;
  statistics["remapped_net_count"] = remapped_nets;
  result["timing_source"] = "persistent_graph_in_carrier_ids";
  return result;
}
}  // namespace rebuffer

// --- rebuffer_timing_paths.cc ---
// Read current graph-owned timing directly; no carrier or state materialization.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

#include "timing_hard_math.h"


namespace py = pybind11;
namespace rebuffer {

TimingSetupPaths PersistentTimingGraph::nativeSetupPaths(
    const std::optional<std::vector<int64_t>>& endpoint_pins,
    int64_t global_k, int64_t max_depth, double residual_tolerance_ps,
    bool worst_per_endpoint) const
{
  if (global_k < 0) throw py::value_error("global_k must be non-negative");
  if (!std::isfinite(residual_tolerance_ps) || residual_tolerance_ps <= 0) {
    throw py::value_error("residual_tolerance_ps must be finite and positive");
  }
  if (!timing_state_initialized_ || !propagation_program_) {
    throw py::value_error("setup paths require initialized timing state and propagation inputs");
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full || !seeds.forward.empty() || !seeds.reverse.empty()
      || !seeds.retired_arcs.empty()) {
    throw py::value_error("setup paths require current native propagation");
  }
  if (!net_states_.empty()) {
    if (!mutation_state_) throw py::value_error("setup paths require synchronized native topology");
    requireSynchronizedTimingTopologies(*mutation_state_, net_states_);
  }
  const auto& p = *propagation_program_;
  const auto& index = *timing_index_;
  std::set<int64_t> requested;
  const bool filtered = endpoint_pins.has_value();
  if (filtered) {
    for (auto pin : *endpoint_pins) {
      if (pin < 0 || pin >= static_cast<int64_t>(index.pins.size())) {
        throw py::value_error("requested setup endpoint pin is out of range");
      }
      requested.insert(pin);
    }
  }
  auto included = [&](int64_t pin) { return !filtered || requested.count(pin); };
  const auto& ra = state_.at("pin_rAAT");
  const auto& fa = state_.at("pin_fAAT");
  const auto& rs = state_.at("pin_rtran");
  const auto& fs = state_.at("pin_ftran");
  const auto& rr = state_.at("pin_rRAT");
  const auto& fr = state_.at("pin_fRAT");
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<dreamplace::timing::SetupPathState> states;
  std::set<int64_t> constraint_pins;
  std::size_t endpoint_records = 0;
  auto append = [&](int64_t pin, int64_t test, double rise, double fall) {
    if (!included(pin)) return;
    ++endpoint_records;
    dreamplace::timing::appendSetupPathStates(states, pin, test,
        p.endpoint_valid[pin * 2] ? rise : inf,
        p.endpoint_valid[pin * 2 + 1] ? fall : inf);
  };
  for (std::size_t row = 0; row < p.setup_arcs.size(); ++row) {
    const bool rise_valid = p.setup_valid[row * 2];
    const bool fall_valid = p.setup_valid[row * 2 + 1];
    if (!rise_valid && !fall_valid) continue;
    const auto& arc = p.setup_arcs[row];
    const auto pin = arc[1];
    if (!included(pin)) continue;
    if (p.timing_check_pins.empty()) constraint_pins.insert(pin);
    const int endpoint_row = p.endpoint_row_by_pin[pin];
    const auto candidate = dreamplace::timing::setupRatRow(arc.data(),
        endpoint_row < 0 ? 2e8 : p.endpoint_rRAT[endpoint_row],
        endpoint_row < 0 ? 2e8 : p.endpoint_fRAT[endpoint_row],
        p.clock_rtran.data(), p.clock_ftran.data(), rs.data(), fs.data(),
        p.luts[0].view(), p.luts[1].view(), p.pairwise_smooth, p.pairwise_alpha);
    append(pin, row, rise_valid ? candidate.first - ra[pin] : inf,
                     fall_valid ? candidate.second - fa[pin] : inf);
  }
  for (auto pin : p.endpoint_pins) {
    if (!included(pin)) continue;
    if (!p.endpoint_valid[pin * 2] && !p.endpoint_valid[pin * 2 + 1]) continue;
    // Match TP's actual no-setup-rows fallback, including recovery-only pins.
    if (!p.setup_arcs.empty()) {
      const bool check = p.timing_check_pins.empty() ? constraint_pins.count(pin)
          : std::binary_search(p.timing_check_pins.begin(), p.timing_check_pins.end(), pin);
      if (check) continue;
    }
    append(pin, -1, rr[pin] - ra[pin], fr[pin] - fa[pin]);
  }
  const dreamplace::timing::SetupPathTimingView view{
      ra.data(), fa.data(), state_.at("pin_net_delay_rise").data(),
      state_.at("pin_net_delay_fall").data(), state_.at("cell_arc_rr_delays").data(),
      state_.at("cell_arc_fr_delays").data(), state_.at("cell_arc_rf_delays").data(),
      state_.at("cell_arc_ff_delays").data()};
  const bool reordered = mutation_state_ && mutation_state_->canonical_schedule;
  const TimingPathPredecessors predecessors{index, reordered};
  auto extracted = dreamplace::timing::extractSetupPaths(std::move(states), view,
      index.pins.size(), global_k, max_depth, residual_tolerance_ps, predecessors,
      [&](int64_t pin) {
        return pin < p.pin_count && p.start_row_by_pin[pin] >= 0;
      }, worst_per_endpoint);
  return {std::move(extracted), endpoint_records};
}

py::dict PersistentTimingGraph::setupPathDetails(const py::object& endpoint_pins,
    int64_t global_k, int64_t max_depth, double residual_tolerance_ps) const
{
  const auto started = std::chrono::steady_clock::now();
  const auto result_paths = nativeSetupPaths(endpoint_pins.is_none() ? std::nullopt
      : std::make_optional(endpoint_pins.cast<std::vector<int64_t>>()),
      global_k, max_depth, residual_tolerance_ps);
  const auto& extracted = result_paths.batch;
  const auto& ra = state_.at("pin_rAAT");
  const auto& fa = state_.at("pin_fAAT");
  py::dict paths;
  std::size_t valid_count = 0, path_pins = 0;
  for (std::size_t id = 0; id < extracted.states.size(); ++id) {
    const auto& state = extracted.states[id];
    const auto& trace = extracted.traces[id];
    valid_count += trace.valid != 0;
    path_pins += trace.pins.size();
    const auto key = py::int_(state.pin);
    // States are sorted by slack/transition/pin/test; strict '<' in the
    // existing consumer retains the first state for this pin on equal slack.
    if (paths.contains(key)) continue;
    py::dict record;
    record["path"] = trace.pins;
    record["transitions"] = trace.transitions;
    record["arc_ids"] = trace.arc_ids;
    record["endpoint_transition"] = state.transition;
    record["endpoint_slack"] = state.slack;
    record["valid"] = trace.valid != 0;
    record["topology_epoch"] = epoch_;
    record["endpoint_test_id"] = state.test_id;
    record["invalid_reason"] = trace.invalid_reason;
    record["max_residual_ps"] = trace.max_residual_ps;
    std::vector<double> arrivals;
    arrivals.reserve(trace.pins.size());
    for (std::size_t pin = 0; pin < trace.pins.size(); ++pin) {
      arrivals.push_back((trace.transitions[pin] == 0 ? ra : fa)[trace.pins[pin]]);
    }
    record["path_arrivals"] = std::move(arrivals);
    paths[key] = std::move(record);
  }
  py::dict statistics;
  statistics["backend"] = "graph_owned_cpp_transition_aware";
  statistics["topology_epoch"] = epoch_;
  statistics["endpoint_record_count"] = result_paths.endpoint_record_count;
  statistics["failing_state_count"] = extracted.failing_state_count;
  statistics["selected_state_count"] = extracted.states.size();
  statistics["valid_path_count"] = valid_count;
  statistics["invalid_path_count"] = extracted.states.size() - valid_count;
  statistics["path_pin_count"] = path_pins;
  statistics["timing_state_values_copied"] = 0;
  statistics["extraction_runtime_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  py::dict result;
  result["paths"] = std::move(paths);
  result["statistics"] = std::move(statistics);
  return result;
}

}  // namespace rebuffer

// --- rebuffer_timing_endpoints.cc ---
// Copy only endpoint scalars; path selection remains inside the native scheduler.

#include <set>

namespace py = pybind11;
namespace rebuffer {

TimingEndpointState::TimingEndpointState(const std::vector<int>& pins,
    const std::vector<std::string>& names, const std::vector<double>& slacks)
{
  if (pins.size() != names.size() || pins.size() != slacks.size()) {
    throw py::value_error("endpoint state arrays do not align");
  }
  std::set<int> used_pins;
  std::set<std::string> used_names;
  for (std::size_t row = 0; row < pins.size(); ++row) {
    if (pins[row] < 0 || names[row].empty() || !used_pins.insert(pins[row]).second
        || !used_names.insert(names[row]).second) {
      throw py::value_error("endpoint state identities are not unique and valid");
    }
    endpoints.push_back({pins[row], names[row], slacks[row]});
  }
}

std::shared_ptr<TimingEndpointState> PersistentTimingGraph::timingEndpointState() const
{
  if (!timing_state_initialized_ || !propagation_program_) {
    throw py::value_error("endpoint state requires initialized native propagation");
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full || !seeds.forward.empty() || !seeds.reverse.empty()
      || !seeds.retired_arcs.empty()) {
    throw py::value_error("endpoint state requires current native propagation");
  }
  auto result = std::make_shared<TimingEndpointState>();
  const auto slacks = endpointSlacks();
  result->endpoints.reserve(slacks.size());
  for (std::size_t row = 0; row < slacks.size(); ++row) {
    const int pin = propagation_program_->metric_pins[row];
    result->endpoints.push_back({pin, timing_index_->pins.at(pin), slacks[row]});
  }
  return result;
}

void registerLegacyQueueBindings(py::module_& module)
{
  py::class_<TimingEndpointState, std::shared_ptr<TimingEndpointState>>(
      module, "TimingEndpointState")
      .def(py::init<const std::vector<int>&, const std::vector<std::string>&,
                    const std::vector<double>&>());
  module.def("run_repair_timing_legacy_queue_operator", &runRepairTimingLegacyQueueOperator,
      py::arg("endpoint_state"), py::arg("timing_snapshot"), py::arg("candidate_provider"),
      py::arg("checkpoint"), py::arg("apply"), py::arg("evaluate"), py::arg("restore"),
      py::arg("config"), py::arg("release_checkpoint") = py::none());
}

}  // namespace rebuffer
