// SPDX-License-Identifier: BSD-3-Clause
// Queue-state publication shared by materialized and graph-owned consumers.
#include "rebuffer_queue_snapshot.hh"

#include <algorithm>
#include <set>

namespace py = pybind11;
namespace rebuffer {

void projectTimingQueueItems(std::vector<TimingQueueItem>& items,
                             const std::vector<int>& pins, const std::vector<int>& nets)
{
  for (auto& item : items) {
    item.endpoint_pin = pins.at(item.endpoint_pin);
    item.root_net_id = nets.at(item.root_net_id);
    item.driver_pin_id = pins.at(item.driver_pin_id);
    for (auto& pin : item.path_pins) pin = pins.at(pin);
    for (auto& net : item.focus_net_ids) net = nets.at(net);
    item.tree_key = std::to_string(item.root_net_id) + ":" + std::to_string(item.driver_pin_id);
  }
  std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    if (a.endpoint_slack_ps != b.endpoint_slack_ps) return a.endpoint_slack_ps < b.endpoint_slack_ps;
    return a.endpoint_pin < b.endpoint_pin;
  });
}

py::dict timingQueueSnapshotStatistics(const std::vector<TimingQueueItem>& items,
    const std::function<bool(int)>& is_buffer_driver)
{
  std::set<int> endpoints, roots, drivers, buffer_output_roots;
  for (const auto& item : items) {
    endpoints.insert(item.endpoint_pin);
    roots.insert(item.root_net_id);
    drivers.insert(item.driver_pin_id);
    if (is_buffer_driver(item.driver_pin_id)) buffer_output_roots.insert(item.root_net_id);
  }
  py::dict result;
  result["item_count"] = items.size();
  result["endpoint_count"] = endpoints.size();
  result["root_count"] = roots.size();
  result["driver_count"] = drivers.size();
  result["buffer_output_root_count"] = buffer_output_roots.size();
  result["path_driver_item_count"] = items.size();
  return result;
}

py::dict makeTimingQueueState(std::vector<TimingQueueItem> items,
    const std::function<bool(int)>& is_buffer_driver,
    const std::shared_ptr<NativeTimingQueueContext>& context)
{
  std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
    if (a.endpoint_slack_ps != b.endpoint_slack_ps) return a.endpoint_slack_ps < b.endpoint_slack_ps;
    return a.endpoint_pin < b.endpoint_pin;
  });
  auto state = std::make_shared<NativeTimingQueueSnapshot>();
  state->items = std::move(items);
  state->records = py::list();
  state->context = context;
  py::dict result;
  result["status"] = "ok";
  result["snapshot_statistics"] = timingQueueSnapshotStatistics(state->items, is_buffer_driver);
  result["queue_context_reused"] = false;
  result["_native_queue_state"] = std::move(state);
  return result;
}

}  // namespace rebuffer
