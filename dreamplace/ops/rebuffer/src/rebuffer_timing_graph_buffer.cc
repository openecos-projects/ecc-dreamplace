// SPDX-License-Identifier: BSD-3-Clause
// Atomically connect native split/remove/resize to graph-owned timing state.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_local_index.hh"

#include <algorithm>
#include <exception>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

void PersistentTimingGraph::requireTimingMutationOwner() const
{
  if (!timing_state_initialized_ || propagation_program_ == nullptr || mutation_state_ == nullptr) {
    throw py::value_error("buffer timing mutation requires initialized graph state and inputs");
  }
}

py::dict PersistentTimingGraph::mutateTimingTopology(int net_id, const py::dict& action,
                                                    TimingTopologyOperation operation)
{
  requireTimingMutationOwner();
  if (net_id < 0 || net_id >= static_cast<int>(mutation_state_->base_net_pins.size())
      || mutation_state_->base_net_is_clock.at(net_id)) {
    throw py::value_error("buffer timing mutation requires an original signal source net");
  }
  const auto topology = topologyState(net_id);
  requireSynchronizedTimingTopology(*mutation_state_, *topology);
  // Only this source topology can be changed by one mutation.  The previous
  // guard called checkpoint(), which copied every attached net topology for
  // every candidate action.  That made apply grow with campaign history even
  // though the journal already contains sparse timing undo for the mutation.
  const auto topology_before = checkpointIncrementalTopology(*topology);
  const std::size_t journal_size_before = journal_.size();
  const std::uint64_t epoch_before = epoch_;
  const std::uint64_t topology_revision_before = topology_revision_;
  const auto propagation_mark_before = propagation_mark_;
  try {
    std::string reason;
    py::dict result;
    bool valid = false;
    if (operation == TimingTopologyOperation::insert) {
      valid = applyIncrementalBufferSplit(*topology, action, reason, result);
    } else if (operation == TimingTopologyOperation::remove) {
      const int id = action["action_id"].cast<int>();
      valid = !topology->general_removal && !topology->buffers.empty()
                      && topology->buffers.back().action_id == id
                  ? removeIncrementalBuffer(*topology, id, reason, result)
                  : removeIncrementalBufferAnywhere(*topology, id, reason, result);
    } else {
      valid = resizeIncrementalBuffer(*topology, action, reason, result);
    }
    if (!valid) throw py::value_error(reason);
    if (result["status"].cast<std::string>() == "noop") return result;
    auto undo = captureTimingTopology(*mutation_state_, *timing_index_, net_id);
    JournalEntry entry;
    entry.previous_epoch = epoch_;
    entry.previous_topology_revision = topology_revision_;
    entry.sequence = next_journal_sequence_++;
    entry.topology_undo = undo;
    journal_.push_back(std::move(entry));
    ++epoch_;
    synchronizeTimingBuffers(*mutation_state_, *timing_index_, *propagation_program_, *topology);
    updateTimingLocalIndex(*mutation_state_, *timing_index_, *propagation_program_, *undo);
    if (cached_active_counts_valid_) {
      for (const auto& pin : undo->previous_pins) {
        cached_active_pins_ += static_cast<int>(mutation_state_->pin_active[pin.id])
                             - static_cast<int>(pin.active);
      }
      for (std::size_t p = undo->pins; p < mutation_state_->pin_active.size(); ++p) {
        cached_active_pins_ += static_cast<std::size_t>(mutation_state_->pin_active[p]);
      }
      for (const auto& arc : undo->previous_arcs) {
        cached_active_cell_arcs_ += static_cast<int>(mutation_state_->cell_active[arc.id])
                                  - static_cast<int>(arc.active);
      }
      for (std::size_t a = undo->arcs; a < mutation_state_->cell_active.size(); ++a) {
        cached_active_cell_arcs_ += static_cast<std::size_t>(mutation_state_->cell_active[a]);
      }
      for (const auto& row : undo->index_undo->net_rows) {
        cached_active_net_arcs_ += static_cast<int>(timing_index_->net_active[row.id])
                                 - static_cast<int>(row.active);
      }
      for (std::size_t n = undo->index_undo->nets; n < timing_index_->net_active.size(); ++n) {
        cached_active_net_arcs_ += static_cast<std::size_t>(timing_index_->net_active[n]);
      }
    }
    topology_revision_ = ++next_topology_revision_;
    for (const auto& pin : undo->previous_pins) markElectricalDirtyPin(pin.id);
    for (const auto& net : undo->previous_nets) {
      for (const int pin : net.pins) markElectricalDirtyPin(pin);
    }
    for (const int pin : mutation_state_->net_pins.at(net_id)) markElectricalDirtyPin(pin);
    for (int buf_id : mutation_state_->buffers_by_source_net.at(net_id)) {
      const auto& row = mutation_state_->buffers[buf_id];
      markElectricalDirtyPin(row.input_pin);
      markElectricalDirtyPin(row.output_pin);
      if (row.output_net >= 0
          && row.output_net < static_cast<int>(mutation_state_->net_pins.size())) {
        for (const int pin : mutation_state_->net_pins[row.output_net]) {
          markElectricalDirtyPin(pin);
        }
      }
    }
    const auto rc = recomputeTimingNetRc(*topology, mutation_state_->pin_values);
    std::vector<int> inactive;
    for (const auto& pin : undo->previous_pins) {
      if (pin.active && !mutation_state_->pin_active.at(pin.id)) inactive.push_back(pin.id);
    }
    applyTimingRc({rc}, inactive);
    const int action_id = result["action_id"].cast<int>();
    for (int buf_id : mutation_state_->buffers_by_source_net.at(net_id)) {
      const auto& record = mutation_state_->buffers[buf_id];
      if (record.action_id == action_id) {
        result["timing_pin_ids"] = std::vector<int>{record.input_pin, record.output_pin};
        result["timing_arc_ids"] = record.arcs;
        result["timing_net_id"] = record.output_net;
        break;
      }
    }
    result["state"] = summary();
    return result;
  } catch (...) {
    const auto failure = std::current_exception();
    std::string restore_reason;
    cached_active_counts_valid_ = false;
    restoreTo(journal_size_before);
    epoch_ = epoch_before;
    topology_revision_ = topology_revision_before;
    propagation_mark_ = propagation_mark_before;
    if (!restoreIncrementalTopology(*topology, *topology_before, restore_reason)) {
      throw py::value_error("native topology mutation rollback failed: " + restore_reason);
    }
    std::rethrow_exception(failure);
  }
}

}  // namespace rebuffer
