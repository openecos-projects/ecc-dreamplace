// SPDX-License-Identifier: BSD-3-Clause
// Export one session-owned timing graph patch without rebuilding a carrier.
#pragma once

#include <memory>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "rebuffer_incremental_topology.hh"

namespace rebuffer {

// Build the fixed one-buffer timing slot used by the Python
// TimingPropagation incremental evaluator.  Graph traversal, partition
// resolution, RC edge selection, and scheduled arc placement stay native;
// the caller only applies the returned arrays to an existing operator.
pybind11::dict buildIncrementalTimingPatch(
    const pybind11::object& base,
    const std::shared_ptr<IncrementalTopologyState>& state,
    const pybind11::dict& action,
    int base_vertex_count,
    int slot);

pybind11::dict evaluateIncrementalRc(
    const pybind11::dict& rc,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& generic_caps,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& rise_caps,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& fall_caps,
    const pybind11::dict& buffer_caps,
    int pin_count);

}  // namespace rebuffer
