// SPDX-License-Identifier: BSD-3-Clause
// Snapshot identity and endpoint-window selection for the native timing queue.

#pragma once

#include <set>

#include <string>
#include <vector>

#include "pybind11/pybind11.h"

namespace rebuffer {

struct TimingQueueItem
{
  int snapshot_index = -1;
  int endpoint_pin = -1;
  std::string endpoint_pin_name;
  double endpoint_slack_ps = 0.0;
  int root_net_id = -1;
  int driver_pin_id = -1;
  std::string tree_key;
  std::vector<int> path_pins;
  std::vector<int> net_ids;
  // Current-tree queue units are active electrical partitions.  ``net_ids``
  // remains the logical/path identity exported by the snapshot, while this
  // list lets the queue visit the root and every reachable branch separately.
  std::vector<int> focus_net_ids;
};

std::vector<TimingQueueItem> parseTimingQueueSnapshot(
    const pybind11::dict& snapshot);
// ``retired`` endpoints (attempt budget spent) do not occupy window slots.
std::vector<int> timingQueueWindow(const std::vector<TimingQueueItem>& items,
                                 const pybind11::dict& config,
                                 const std::set<int>& retired = {});
// Return the original records so branch metadata written by the bridge is
// visible to the native scheduler in the same snapshot.  The queue publishes
// endpoint names whose attempt budget is spent as ``config["retired_endpoints"]``
// before each snapshot, so the bridge window matches the scheduler window.
pybind11::list selectTimingQueueItems(const pybind11::dict& snapshot,
                                    const pybind11::dict& config);
pybind11::dict bindTimingQueueObservation(const pybind11::dict& observation,
                                        const TimingQueueItem& item);

// Python supplies TP endpoint/path data; C++ ranks path drivers and emits
// one queue record per independently driven net.
pybind11::dict buildNativeTimingQueueSnapshot(
    const pybind11::dict& design_values,
    const pybind11::dict& timing_values,
    const pybind11::dict& liberty_values,
    const pybind11::list& endpoint_records,
    const std::vector<int>& original_net_by_net);

}  // namespace rebuffer
