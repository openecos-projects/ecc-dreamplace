// SPDX-License-Identifier: BSD-3-Clause
// Endpoint observations and the LEGACY two-stage native repair scheduler.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"

namespace rebuffer {

struct TimingEndpoint
{
  int pin;
  std::string name;
  double slack;
};

struct TimingEndpointState
{
  TimingEndpointState() = default;
  TimingEndpointState(const std::vector<int>& pins,
      const std::vector<std::string>& names, const std::vector<double>& slacks);
  std::vector<TimingEndpoint> endpoints;
};

pybind11::dict runRepairTimingLegacyQueueOperator(
    const pybind11::object& endpoint_state,
    const pybind11::object& timing_snapshot,
    const pybind11::object& candidate_provider,
    const pybind11::object& checkpoint,
    const pybind11::object& apply,
    const pybind11::object& evaluate,
    const pybind11::object& restore,
    const pybind11::dict& config,
    const pybind11::object& release_checkpoint = pybind11::none());

void registerLegacyQueueBindings(pybind11::module_& module);

}  // namespace rebuffer
