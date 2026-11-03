// SPDX-License-Identifier: BSD-3-Clause
// Snapshot identity and endpoint-window selection for the native timing queue.

#pragma once

#include <memory>
#include <functional>
#include <set>

#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "rebuffer_liberty.hh"
#include "rebuffer_types.hh"

namespace rebuffer {

struct TimingQueueItem
{
  int snapshot_index = -1;
  int endpoint_pin = -1;
  std::string endpoint_pin_name;
  double endpoint_slack_ps = 0.0;
  int root_net_id = -1;
  int driver_pin_id = -1;
  double gradient_score = 0.0;
  bool has_gradient_score = false;
  std::string tree_key;
  std::vector<int> path_pins;
  std::vector<int> net_ids;
  // Current-tree queue units are active electrical partitions.  ``net_ids``
  // remains the logical/path identity exported by the snapshot, while this
  // list lets the queue visit the root and every reachable branch separately.
  std::vector<int> focus_net_ids;
};

// Design and Liberty are immutable during one placement-RC repair campaign.
// Keep their parsed native views with the queue state so endpoint-delta
// updates only parse the new Python TimingView and changed records.
struct NativeTimingQueueContext
{
  std::shared_ptr<DesignView> design;
  std::shared_ptr<LibertyView> liberty;
  pybind11::object design_source;
  pybind11::object liberty_source;
};

// A materialized queue snapshot keeps the parsed native items beside the
// Python records.  The bridge still exposes the records for payload
// construction, while the queue and endpoint-window selector reuse this
// parsed state instead of converting the same large list twice per epoch.
struct NativeTimingQueueSnapshot
{
  std::vector<TimingQueueItem> items;
  pybind11::list records;
  std::shared_ptr<NativeTimingQueueContext> context;
};

pybind11::dict timingQueueSnapshotStatistics(
    const std::vector<TimingQueueItem>& items,
    const std::function<bool(int)>& is_buffer_driver);
pybind11::dict makeTimingQueueState(std::vector<TimingQueueItem> items,
    const std::function<bool(int)>& is_buffer_driver,
    const std::shared_ptr<NativeTimingQueueContext>& context = nullptr);

std::vector<TimingQueueItem> parseTimingQueueSnapshot(
    const pybind11::dict& snapshot);
// ``retired`` endpoints (attempt budget spent) do not occupy window slots.
std::vector<int> timingQueueWindow(const std::vector<TimingQueueItem>& items,
                                 const pybind11::dict& config,
                                 const std::set<int>& retired = {});
std::vector<int> selectTimingQueueIndices(const std::vector<TimingQueueItem>& items,
                                         const pybind11::dict& config);
void projectTimingQueueItems(std::vector<TimingQueueItem>& items,
                             const std::vector<int>& pins, const std::vector<int>& nets);
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

pybind11::dict buildNativeTimingQueueState(
    const pybind11::dict& design_values,
    const pybind11::dict& timing_values,
    const pybind11::dict& liberty_values,
    const pybind11::list& endpoint_records,
    const std::vector<int>& original_net_by_net);

// Rebuild only the endpoint records whose path/slack/topology changed and
// retain the previous native queue items for the remaining active endpoints.
pybind11::dict updateNativeTimingQueueSnapshot(
    const pybind11::dict& design_values,
    const pybind11::dict& timing_values,
    const pybind11::dict& liberty_values,
    const pybind11::dict& previous_snapshot,
    const pybind11::list& changed_endpoint_records,
    const std::vector<int>& active_endpoint_pins,
    const std::vector<int>& changed_endpoint_pins,
    const std::vector<int>& original_net_by_net);

pybind11::dict updateNativeTimingQueueState(
    const pybind11::dict& design_values,
    const pybind11::dict& timing_values,
    const pybind11::dict& liberty_values,
    const pybind11::dict& previous_snapshot,
    const pybind11::list& changed_endpoint_records,
    const std::vector<int>& active_endpoint_pins,
    const std::vector<int>& changed_endpoint_pins,
    const std::vector<int>& original_net_by_net);

void updateNativeTimingQueueFocus(
    const pybind11::dict& snapshot,
    const pybind11::list& focus_updates);

}  // namespace rebuffer
