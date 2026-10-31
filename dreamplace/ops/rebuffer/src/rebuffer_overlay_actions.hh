// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "rebuffer_incremental_topology.hh"
#include "rebuffer_repair.hh"

namespace rebuffer {
pybind11::dict resolveOverlaySplit(const IncrementalTopologyState& state, const pybind11::dict& action);
pybind11::list projectOverlayActions(
    const pybind11::object& base, const pybind11::dict& payload,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);
pybind11::list finalizeOverlayActions(
    const pybind11::object& base,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);
}  // namespace rebuffer
