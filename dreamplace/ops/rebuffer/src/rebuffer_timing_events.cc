// SPDX-License-Identifier: BSD-3-Clause
// Derive event seeds from native journal changes, including removed net edges.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_events.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_state.hh"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

TimingEventSeeds PersistentTimingGraph::timingEventSeeds() const
{
  TimingEventSeeds seeds;
  if (!propagation_mark_.ready) {
    seeds.require_full = true;
    return seeds;
  }
  for (const auto& entry : journal_) {
    if (entry.sequence <= propagation_mark_.sequence) continue;
    if (entry.resize_undo) {
      for (const auto& [id, previous] : entry.resize_undo->arcs) {
        seeds.forward.insert(timing_index_->cells.at(id)[1]);
        seeds.reverse.insert(previous[0]);
      }
    } else if (entry.topology_undo) {
      const auto& undo = *entry.topology_undo;
      for (const auto& net : undo.previous_nets) {
        if (net.driver >= 0) seeds.reverse.insert(net.driver);
        seeds.forward.insert(net.pins.begin(), net.pins.end());
      }
      auto add_net = [&](int net) {
        const auto& pins = mutation_state_->net_pins.at(net);
        seeds.forward.insert(pins.begin(), pins.end());
        const int driver = mutation_state_->net_drivers.at(net);
        if (driver >= 0) seeds.reverse.insert(driver);
      };
      add_net(undo.source_net);
      for (const auto& row : mutation_state_->buffers) {
        if (row.source_net != undo.source_net) continue;
        if (row.active) add_net(row.output_net);
        for (int arc : row.arcs) {
          if (!mutation_state_->cell_active.at(arc)) seeds.retired_arcs.insert(arc);
        }
      }
      for (const auto& arc : undo.previous_arcs) {
        if (!mutation_state_->cell_active.at(arc.id)) seeds.retired_arcs.insert(arc.id);
      }
    } else if (entry.state_undo) {
      for (const auto& [name, field] : entry.state_undo->fields) {
        if (field.previous.empty()) continue;
        if (name.rfind("pin_net_", 0) != 0) {
          seeds.require_full = true;
          return seeds;
        }
        for (const auto& old : field.previous) {
          const int pin = old.index;
          seeds.forward.insert(pin);
          seeds.reverse.insert(pin);
          for (int edge : timing_index_->net_in.at(pin)) {
            seeds.reverse.insert(timing_index_->nets[edge][0]);
          }
        }
      }
    } else if (entry.name.rfind("pin_net_", 0) == 0 && entry.existed) {
      const auto& current = state_.at(entry.name);
      for (int pin = 0; pin < static_cast<int>(current.size()); ++pin) {
        if (pin >= static_cast<int>(entry.previous.size())
            || std::isnan(current[pin]) || std::isnan(entry.previous[pin])
            || std::memcmp(&current[pin], &entry.previous[pin], sizeof(double))) {
          seeds.forward.insert(pin);
          seeds.reverse.insert(pin);
          for (int edge : timing_index_->net_in.at(pin)) {
            seeds.reverse.insert(timing_index_->nets[edge][0]);
          }
        }
      }
    } else if (entry.existed && state_.count(entry.name)
               && (entry.name.rfind("pin_", 0) == 0
                   || entry.name.rfind("cell_arc_", 0) == 0)) {
      const auto& current = state_.at(entry.name);
      // Buffer mutation expands the canonical arrays without changing old
      // timing values. New slots are already covered by topology seeds.
      if (current.size() < entry.previous.size()
          || (!entry.previous.empty() && std::memcmp(current.data(), entry.previous.data(),
                                                     entry.previous.size() * sizeof(double)))) {
        seeds.require_full = true;
        return seeds;
      }
    } else {
      // Canonical timing was replaced outside propagation, or generic
      // scratch state was edited. Re-establish a native full-seed baseline.
      seeds.require_full = true;
      return seeds;
    }
  }
  return seeds;
}

py::dict PersistentTimingGraph::propagateIncremental()
{
  if (!timing_state_initialized_ || !propagation_program_) return propagate();
  for (const auto& [_, topology] : net_states_) {
    if (!mutation_state_) {
      throw py::value_error("event propagation requires synchronized native topology");
    }
    requireSynchronizedTimingTopology(*mutation_state_, *topology);
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full) {
    auto result = propagate();
    result["fallback_reason"] = "native_full_seed_baseline_required";
    return result;
  }
  const auto checkpoint_size = journal_.size();
  const auto previous_slacks = endpointSlacks();
  try {
    // Register undo before the first in-place kernel write, including failures
    // that occur after forward propagation has already updated other pins.
    auto undo = beginStateUpdate();
    auto computed = propagateTimingEvents(*propagation_program_, *timing_index_,
                                          mutation_state_.get(), state_, seeds, *undo);
    py::dict coverage;
    for (const auto& [name, value] : computed.coverage) coverage[py::str(name)] = value;
    return finishSparsePropagation(undo, std::move(coverage), previous_slacks,
                                    std::move(computed.changed_pins));
  } catch (...) {
    restoreTo(checkpoint_size);
    throw;
  }
}

}  // namespace rebuffer
