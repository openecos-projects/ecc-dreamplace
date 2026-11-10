// SPDX-License-Identifier: BSD-3-Clause
// Build owned native placement/timing views without a Python TP carrier.
#include "rebuffer_timing_planning.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_overlay_carrier.hh"
#include "rebuffer_python_views.hh"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <tuple>

#include "timing_hard_math.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {
void require(bool valid, const char* message)
{
  if (!valid) throw py::value_error(message);
}

}  // namespace

py::dict PersistentTimingGraph::initializePlanning(const py::dict& design,
    const py::dict& characterization, const py::dict& steiner,
    const py::dict& config, double r_unit, double c_unit)
{
  require(!planning_inputs_, "native planning inputs are immutable");
  require(mutation_state_ && queue_liberty_, "native planning requires mutation and queue inputs");
  require(mutation_state_->buffers.empty(), "native planning initializes before buffer mutations");
  auto inputs = std::make_shared<TimingPlanningInputs>();
  inputs->design = rebuffer_python::designFromDict(design);
  inputs->characterization = rebuffer_python::characterizationFromDict(characterization);
  inputs->steiner = rebuffer_python::steinerFromDict(steiner);
  inputs->config = config.attr("copy")().cast<py::dict>();
  inputs->r_unit = r_unit;
  inputs->c_unit = c_unit;
  require(std::isfinite(r_unit) && r_unit > 0 && std::isfinite(c_unit) && c_unit >= 0,
          "native planning requires valid wire RC units");
  const auto& base = inputs->design;
  const auto& m = *mutation_state_;
  require(base.num_pins == timing_index_->pins.size()
              && base.pin_names == timing_index_->pins
              && base.pin2node_map == m.pin_owner && base.pin2net_map == m.pin_net
              && base.net2driver_pin_map == m.net_drivers
              && base.num_nodes == m.original_node_count
              && base.num_nets == m.base_net_pins.size()
              && base.node_is_buffer == queue_original_buffer_nodes_
              && base.dbu == queue_liberty_->dbu,
          "native planning design does not match timing owner");
  require(inputs->steiner.steiner_node_base == base.num_pins
              && inputs->steiner.net_flat_topo_sort_start.size() == base.num_nets + 1,
          "native planning Steiner does not match base physical graph");
  require(inputs->steiner.net_ids.size() == base.num_nets,
          "native planning requires complete base Steiner net identities");
  for (int net = 0; net < base.num_nets; ++net) {
    require(inputs->steiner.net_ids[net] == net,
            "native planning requires ordered base Steiner net identities");
  }
  require(base.node_is_fixed.empty() || base.node_is_fixed.size() == base.num_nodes,
          "native planning physical fixed flags do not align");
  inputs->carrier = overlayCarrierInputs(carrier_);
  // Import immutable base buffer boundaries once.  Subsequent local
  // snapshots derive newly inserted boundaries from the native mutation
  // records instead of scanning every projected pin and instance.
  {
    auto links = std::make_shared<std::vector<CurrentDriverBufferLink>>();
    std::unordered_map<int, CurrentDriverBufferLink> by_input;
    std::string reason;
    if (!collectCurrentDriverBufferLinks(inputs->design, *links, by_input, reason)) {
      throw py::value_error(reason);
    }
    inputs->base_links = std::move(links);
  }
  planning_inputs_ = std::move(inputs);
  py::dict result;
  result["status"] = "ok";
  result["pin_count"] = base.num_pins;
  return result;
}

std::shared_ptr<NativeTimingPlanningSnapshot> PersistentTimingGraph::planningSnapshot(
    const py::list& actions, const py::dict& config)
{
  require(planning_inputs_ != nullptr, "native planning inputs are not initialized");
  return std::shared_ptr<NativeTimingPlanningSnapshot>(
      new NativeTimingPlanningSnapshot(shared_from_this(), planning_inputs_, actions, config));
}

std::shared_ptr<IncrementalTopologyState> PersistentTimingGraph::createNetTopology(
    int net_id) const
{
  require(planning_inputs_ != nullptr, "native planning inputs are not initialized");
  require(net_id >= 0 && net_id < planning_inputs_->design.num_nets,
          "net_id is out of range for native planning");
  return incrementalTopologyFromSteiner(
      planning_inputs_->steiner,
      net_id,
      planning_inputs_->design.net2driver_pin_map.at(net_id),
      planning_inputs_->r_unit,
      planning_inputs_->c_unit);
}

NativeTimingPlanningSnapshot::NativeTimingPlanningSnapshot(
    std::shared_ptr<PersistentTimingGraph> graph,
    std::shared_ptr<const TimingPlanningInputs> inputs,
    const py::list& actions, const py::dict& config)
    : graph_(std::move(graph)), checkpoint_(graph_->checkpoint()), inputs_(std::move(inputs)),
      liberty_(graph_->queue_liberty_), config_(config.attr("copy")().cast<py::dict>())
{
  // This validates current propagation/topology before copying any views.
  auto previous = std::chrono::steady_clock::now();
  auto finish_phase = [&](const char* name) {
    const auto now = std::chrono::steady_clock::now();
    phase_timings_ms_[name] = std::chrono::duration<double, std::milli>(now - previous).count();
    previous = now;
  };
  const bool path_context = config_.contains("use_native_timing_path_context")
      && config_["use_native_timing_path_context"].cast<bool>();
  const auto endpoint = config_.contains("native_endpoint_pin")
      ? std::make_optional(config_["native_endpoint_pin"].cast<int>()) : std::nullopt;
  queue_metadata_ = graph_->timingQueueState(inputs_->steiner.net_ids, path_context, endpoint);
  graph_->applyQueueGradientScores(queue_metadata_, config_);
  finish_phase("queue_state");
  std::vector<std::shared_ptr<IncrementalTopologyState>> states;
  for (const auto& [_, state] : graph_->net_states_) states.push_back(state);
  projection_ = buildOverlayCarrierProjection(inputs_->carrier, inputs_->steiner,
      states, actions, inputs_->r_unit, inputs_->c_unit);
  finish_phase("carrier_projection");
  buildIdentity();
  finish_phase("identity");
  buildViews();
  finish_phase("timing_views");
  buildBufferLinks();
  finish_phase("buffer_links");
  buildQueue();
  finish_phase("queue_projection");
}

void NativeTimingPlanningSnapshot::requireCurrent() const
{
  graph_->validateCheckpoint(checkpoint_);
  require(graph_->epoch_ == checkpoint_->epoch
              && graph_->journal_.size() == checkpoint_->journal_size,
          "native planning snapshot is stale");
  for (const auto& [net, _] : checkpoint_->topology_checkpoints) {
    require(graph_->net_states_.count(net) != 0, "native planning topology was detached");
  }
  for (const auto& [_, state] : graph_->net_states_) {
    requireSynchronizedTimingTopology(*graph_->mutation_state_, *state);
  }
}

void NativeTimingPlanningSnapshot::buildViews()
{
  const auto& m = *graph_->mutation_state_;
  const auto& b = inputs_->design;
  auto& d = projection_.design;
  d.pin_cap_limit = b.pin_cap_limit;
  d.pin_cap = b.pin_cap;
  d.pin_slew_limit = b.pin_slew_limit;
  d.pin_drive_resistance = b.pin_drive_resistance;
  d.main_id_2_cell_id_start = b.main_id_2_cell_id_start;
  d.cell_id_2_libpin_id_start = b.cell_id_2_libpin_id_start;
  for (int node = 0; node < d.num_nodes; ++node) {
    const int original = graph_node_by_node_.at(node);
    const int cell = m.node_cells.at(original);
    const int main = cell < 0 ? -1 : m.cell_mains.at(cell);
    d.inst_main_id.push_back(main);
    d.inst_libcell_offset.push_back(main < 0 ? 0 : cell - b.main_id_2_cell_id_start.at(main));
    d.node_is_buffer.push_back(original < m.original_node_count
        ? b.node_is_buffer.at(original) : liberty_->cells.at(cell).is_buffer);
    if (!b.node_is_fixed.empty()) {
      d.node_is_fixed.push_back(original < m.original_node_count ? b.node_is_fixed.at(original) : 0);
    }
    if (cell >= 0) projection_.node_master_names[node] = m.cell_names.at(cell);
  }
  for (int pin = 0; pin < d.num_pins; ++pin) {
    const int source = graph_pin_by_pin_[pin], owner = m.pin_owner[source];
    const int cell = m.node_cells[owner], net = m.pin_net[source];
    require((net < 0 ? -1 : net_by_graph_.at(net)) == d.pin2net_map[pin],
            "native planning pin partition differs from timing owner");
    int offset = -1;
    if (cell >= 0) {
      const auto begin = m.library_pin_names.begin() + m.cell_pin_start.at(cell);
      const auto end = m.library_pin_names.begin() + m.cell_pin_start.at(cell + 1);
      const auto found = std::find(begin, end, m.pin_ports[source]);
      require(found != end, "native planning pin is absent from current cell");
      offset = std::distance(begin, found);
    }
    d.pin_2_libpin_offset.push_back(offset);
    timing_.pin_cap.push_back(m.pin_values[0][source]);
    timing_.pin_max_slew.push_back(m.pin_values[3][source]);
    timing_.pin_max_cap.push_back(m.pin_values[4][source]);
  }
  for (std::size_t net = 0; net < m.net_drivers.size(); ++net) {
    if (net_by_graph_[net] < 0) continue;
    const int driver = m.net_drivers[net];
    require((driver < 0 ? -1 : pin_by_graph_.at(driver))
                == d.net2driver_pin_map.at(net_by_graph_[net]),
            "native planning driver differs from timing owner");
  }
  auto values = [&](const char* field) {
    const auto& source = graph_->state_.at(field);
    std::vector<double> result;
    result.reserve(d.num_pins);
    for (const int pin : graph_pin_by_pin_) result.push_back(source.at(pin));
    return result;
  };
  timing_.pin_slew_rise = values("pin_rtran");
  timing_.pin_slew_fall = values("pin_ftran");
  timing_.pin_load = values("pin_net_cap_rise");
  timing_.pin_arrival_rise = values("pin_rAAT");
  timing_.pin_arrival_fall = values("pin_fAAT");
  timing_.pin_required_rise = values("pin_rRAT");
  timing_.pin_required_fall = values("pin_fRAT");
  for (int pin = 0; pin < d.num_pins; ++pin) {
    timing_.pin_slack.push_back(dreamplace::timing::hardMinimum(
        timing_.pin_required_rise[pin] - timing_.pin_arrival_rise[pin],
        timing_.pin_required_fall[pin] - timing_.pin_arrival_fall[pin]));
  }
  const auto slacks = graph_->endpointSlacks();
  for (std::size_t row = 0; row < slacks.size(); ++row) {
    timing_.pin_slack.at(pin_by_graph_.at(graph_->propagation_program_->metric_pins[row])) = slacks[row];
  }
  for (const int pin : graph_->timing_index_->clocks) {
    timing_.clock_pins.push_back(pin_by_graph_.at(pin));
  }
}

py::dict NativeTimingPlanningSnapshot::timingView() const
{
  py::dict result;
  result["pin_slew_rise"] = timing_.pin_slew_rise;
  result["pin_slew_fall"] = timing_.pin_slew_fall;
  result["pin_load"] = timing_.pin_load;
  result["pin_cap"] = timing_.pin_cap;
  result["pin_max_slew"] = timing_.pin_max_slew;
  result["pin_max_cap"] = timing_.pin_max_cap;
  result["pin_slack"] = timing_.pin_slack;
  result["pin_arrival_rise"] = timing_.pin_arrival_rise;
  result["pin_arrival_fall"] = timing_.pin_arrival_fall;
  result["pin_required_rise"] = timing_.pin_required_rise;
  result["pin_required_fall"] = timing_.pin_required_fall;
  result["clock_pins"] = timing_.clock_pins;
  return result;
}

}  // namespace rebuffer
