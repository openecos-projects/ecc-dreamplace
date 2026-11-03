// SPDX-License-Identifier: BSD-3-Clause
// Build scheduler-ready native queue state directly from the timing owner.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_paths.hh"
#include "rebuffer_driver_input_slew.hh"
#include "rebuffer_path_policy.hh"
#include "rebuffer_python_views.hh"
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
    const std::vector<int>& original_net_by_net, bool use_path_context) const
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
  std::vector<int64_t> active;
  for (std::size_t row = 0; row < p.metric_pins.size(); ++row) {
    if (std::isfinite(slacks[row]) && slacks[row] < 0) active.push_back(p.metric_pins[row]);
  }
  const auto paths = nativeSetupPaths(active, 0, 0, 1e-3);
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
  for (std::size_t row = 0; row < p.metric_pins.size(); ++row) {
    const int endpoint = p.metric_pins[row];
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
    const auto drivers = rankPathDriverPins(access, *queue_liberty_, path,
        has_context ? &context : nullptr, has_context);
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
  statistics["setup_path_state_count"] = paths.batch.states.size();
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
