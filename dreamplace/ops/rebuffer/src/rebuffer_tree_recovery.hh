// SPDX-License-Identifier: BSD-3-Clause
// Native placement-RC tree area recovery candidate generation.

#pragma once

#include <vector>

#include "rebuffer_driver_tree.hh"

namespace rebuffer {

struct TreeRecoveryCandidate
{
  DriverTreeMutation mutation;
  double area_saved = 0.0;
  int buffer_count_delta = -1;
  const char* recovery_kind = "remove_buffer_pass_through";
};

// Enumerate timing-preserving recovery candidates structurally.  Actual timing
// acceptance remains the Python TimingPropagation transaction callback; this
// function only generates valid one-buffer pass-through mutations and orders
// them by native cell area.
std::vector<TreeRecoveryCandidate> enumerateTreeAreaRecovery(
    const CurrentDriverTree& tree,
    const DesignView& design,
    const LibertyView& liberty,
    std::string& reason);

// Run the native recovery transaction. Python callbacks only own the physical
// placeio_ecc carrier mutation and Python TimingPropagation observation;
// candidate order,
// native tree lifecycle, envelope acceptance, restore and ledger stay native.
// physical_checkpoint receives (label, mutation) so the bridge can snapshot
// the stable physical instance identity before applying the mutation.
pybind11::dict runTreeAreaRecovery(
    const std::shared_ptr<CurrentDriverTreeTransaction>& transaction,
    const pybind11::dict& design,
    const pybind11::dict& liberty,
    const pybind11::object& physical_checkpoint,
    const pybind11::object& physical_apply,
    const pybind11::object& evaluate,
    const pybind11::object& physical_restore,
    const pybind11::dict& config);

}  // namespace rebuffer
