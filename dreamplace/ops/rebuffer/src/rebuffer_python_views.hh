// SPDX-License-Identifier: BSD-3-Clause
// Declarations for the rebuffer Python dict -> native view conversions.

#pragma once

#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "rebuffer_liberty.hh"
#include "rebuffer_repair.hh"
#include "rebuffer_types.hh"

namespace py = pybind11;

namespace rebuffer_python {

rebuffer::DesignView designFromDict(const py::dict& values);
rebuffer::TopologyView topologyFromDict(const py::dict& values);
rebuffer::LibertyView libertyFromDict(const py::dict& values);
py::dict characterizationResult(const rebuffer::CharacterizationResult& result);
rebuffer::TimingView timingFromDict(const py::dict& values);
rebuffer::SteinerView steinerFromDict(const py::dict& values);
rebuffer::RepairConfig configFromDict(const py::dict& values);
rebuffer::CharacterizationResult characterizationFromDict(const py::dict& values);
py::list netSummaries(const rebuffer::ForestResult& forest);
py::dict bufferActionDict(const rebuffer::BufferAction& action);

}  // namespace rebuffer_python
