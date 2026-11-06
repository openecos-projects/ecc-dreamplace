// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_incremental_topology.hh"
#include "rebuffer_repair.hh"
#include "rebuffer_carrier_projection.hh"
#include "rebuffer_overlay_inputs.hh"

namespace rebuffer {

OverlayCarrierProjection buildOverlayCarrierProjection(
    const OverlayCarrierInputs& base,
    const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions,
    double r_unit,
    double c_unit);

OverlayCarrierProjection buildOverlayCarrierProjection(
    const pybind11::object& base,
    const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions,
    double r_unit,
    double c_unit);

// Materialize the current electrical partitions without rebuilding Steiner.
pybind11::dict exportOverlayCarrier(
    const pybind11::object& base,
    const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions,
    double r_unit = 0.914,
    double c_unit = 1.89e-4);

}  // namespace rebuffer
