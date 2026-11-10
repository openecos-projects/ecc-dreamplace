// SPDX-License-Identifier: BSD-3-Clause
// Copy only endpoint scalars; path selection remains inside the native scheduler.
#include "rebuffer_legacy_queue.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_program.hh"
#include "rebuffer_timing_events.hh"

#include <set>
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

TimingEndpointState::TimingEndpointState(const std::vector<int>& pins,
    const std::vector<std::string>& names, const std::vector<double>& slacks)
{
  if (pins.size() != names.size() || pins.size() != slacks.size()) {
    throw py::value_error("endpoint state arrays do not align");
  }
  std::set<int> used_pins;
  std::set<std::string> used_names;
  for (std::size_t row = 0; row < pins.size(); ++row) {
    if (pins[row] < 0 || names[row].empty() || !used_pins.insert(pins[row]).second
        || !used_names.insert(names[row]).second) {
      throw py::value_error("endpoint state identities are not unique and valid");
    }
    endpoints.push_back({pins[row], names[row], slacks[row]});
  }
}

std::shared_ptr<TimingEndpointState> PersistentTimingGraph::timingEndpointState() const
{
  if (!timing_state_initialized_ || !propagation_program_) {
    throw py::value_error("endpoint state requires initialized native propagation");
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full || !seeds.forward.empty() || !seeds.reverse.empty()
      || !seeds.retired_arcs.empty()) {
    throw py::value_error("endpoint state requires current native propagation");
  }
  auto result = std::make_shared<TimingEndpointState>();
  const auto slacks = endpointSlacks();
  result->endpoints.reserve(slacks.size());
  for (std::size_t row = 0; row < slacks.size(); ++row) {
    const int pin = propagation_program_->metric_pins[row];
    result->endpoints.push_back({pin, timing_index_->pins.at(pin), slacks[row]});
  }
  return result;
}

void registerLegacyQueueBindings(py::module_& module)
{
  py::class_<TimingEndpointState, std::shared_ptr<TimingEndpointState>>(
      module, "TimingEndpointState")
      .def(py::init<const std::vector<int>&, const std::vector<std::string>&,
                    const std::vector<double>&>());
  module.def("run_repair_timing_legacy_queue_operator", &runRepairTimingLegacyQueueOperator,
      py::arg("endpoint_state"), py::arg("timing_snapshot"), py::arg("candidate_provider"),
      py::arg("checkpoint"), py::arg("apply"), py::arg("evaluate"), py::arg("restore"),
      py::arg("config"), py::arg("release_checkpoint") = py::none());
}

}  // namespace rebuffer
