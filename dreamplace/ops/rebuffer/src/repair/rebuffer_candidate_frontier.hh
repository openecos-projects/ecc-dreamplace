// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_queue_snapshot.hh"

namespace rebuffer {

struct TimingFrontierEvaluation
{
  pybind11::dict candidate;
  pybind11::dict observation;
  pybind11::list trials;
  int evaluation_calls = 0;
  int selected_rank = -1;
};

// All alternatives are complete tree bundles from one source snapshot. Each
// trial starts at the same checkpoint; return with the selected bundle active.
// Python callbacks supply mutations and TP observations, never the selection.
void evaluateTimingFrontier(
    const pybind11::dict& candidate,
    const pybind11::dict& current,
    const pybind11::dict& request,
    const TimingQueueItem& item,
    const pybind11::object& anchor,
    const pybind11::object& apply,
    const pybind11::object& evaluate,
    const pybind11::object& restore,
    const pybind11::dict& config,
    int evaluation_budget,
    TimingFrontierEvaluation& result);

}  // namespace rebuffer
