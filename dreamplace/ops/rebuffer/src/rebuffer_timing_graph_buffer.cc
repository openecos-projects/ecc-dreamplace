// SPDX-License-Identifier: BSD-3-Clause
// Atomically connect native split/remove/resize to graph-owned timing state.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_local_index.hh"

#include <algorithm>

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
  const auto checkpoint_before = checkpoint();
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
    entry.sequence = next_journal_sequence_++;
    entry.topology_undo = undo;
    journal_.push_back(std::move(entry));
    ++epoch_;
    synchronizeTimingBuffers(*mutation_state_, *timing_index_, *propagation_program_, *topology);
    updateTimingLocalIndex(*mutation_state_, *timing_index_, *propagation_program_, *undo);
    const auto rc = recomputeTimingNetRc(*topology, mutation_state_->pin_values);
    std::vector<int> inactive;
    for (const auto& pin : undo->previous_pins) {
      if (pin.active && !mutation_state_->pin_active.at(pin.id)) inactive.push_back(pin.id);
    }
    applyTimingRc({rc}, inactive);
    const int action_id = result["action_id"].cast<int>();
    const auto found = std::find_if(mutation_state_->buffers.begin(), mutation_state_->buffers.end(),
        [&](const auto& record) { return record.source_net == net_id && record.action_id == action_id; });
    if (found != mutation_state_->buffers.end()) {
      result["timing_pin_ids"] = std::vector<int>{found->input_pin, found->output_pin};
      result["timing_arc_ids"] = found->arcs;
      result["timing_net_id"] = found->output_net;
    }
    result["state"] = summary();
    return result;
  } catch (...) {
    restore(checkpoint_before);
    throw;
  }
}

}  // namespace rebuffer
