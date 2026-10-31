// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_queue_snapshot.hh"

#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_map>
#include <utility>

#include "pybind11/stl.h"
#include "rebuffer_path_policy.hh"
#include "rebuffer_python_views.hh"

namespace py = pybind11;

namespace rebuffer {

namespace {

std::string stringOr(const py::dict& values,
                     const char* key,
                     const char* fallback = "")
{
  if (!values.contains(key) || values[key].is_none()) {
    return fallback;
  }
  return py::cast<std::string>(values[key]);
}

}  // namespace

std::vector<TimingQueueItem> parseTimingQueueSnapshot(const py::dict& snapshot)
{
  if (snapshot.contains("status")
      && stringOr(snapshot, "status") != "ok") {
    return {};
  }
  if (!snapshot.contains("items")
      || !py::isinstance<py::list>(snapshot["items"])) {
    throw py::value_error("native queue timing snapshot requires list items");
  }
  std::vector<TimingQueueItem> items;
  int snapshot_index = 0;
  for (const auto& item_value : snapshot["items"].cast<py::list>()) {
    const py::dict values = item_value.cast<py::dict>();
    TimingQueueItem item;
    item.snapshot_index = snapshot_index++;
    if (!values.contains("endpoint_pin") || !values.contains("slack_ps")) {
      throw py::value_error(
          "native queue item requires endpoint_pin and slack_ps");
    }
    item.endpoint_pin = py::cast<int>(values["endpoint_pin"]);
    item.endpoint_pin_name = stringOr(values, "endpoint_pin_name");
    item.endpoint_slack_ps = py::cast<double>(values["slack_ps"]);
    if (values.contains("root_net_id")) {
      item.root_net_id = py::cast<int>(values["root_net_id"]);
    }
    if (values.contains("driver_pin_id")) {
      item.driver_pin_id = py::cast<int>(values["driver_pin_id"]);
    }
    item.tree_key = stringOr(values, "tree_key");
    if (values.contains("path_pins")) {
      item.path_pins = values["path_pins"].cast<std::vector<int>>();
    }
    if (values.contains("net_ids")) {
      item.net_ids = values["net_ids"].cast<std::vector<int>>();
    }
    if (values.contains("focus_net_ids")) {
      item.focus_net_ids
          = values["focus_net_ids"].cast<std::vector<int>>();
    } else if (values.contains("branch_net_ids")) {
      item.focus_net_ids
          = values["branch_net_ids"].cast<std::vector<int>>();
    }
    if (item.net_ids.empty() && values.contains("net_id")) {
      item.net_ids.push_back(py::cast<int>(values["net_id"]));
    }
    if (item.endpoint_pin < 0 || item.net_ids.empty()) {
      continue;
    }
    if (item.root_net_id < 0) {
      item.root_net_id = item.net_ids.front();
    }
    if (item.tree_key.empty()) {
      item.tree_key = std::to_string(item.root_net_id) + ":"
                      + std::to_string(item.driver_pin_id);
    }
    item.net_ids.erase(
        std::unique(item.net_ids.begin(), item.net_ids.end()), item.net_ids.end());
    item.focus_net_ids.erase(std::remove(item.focus_net_ids.begin(),
                                         item.focus_net_ids.end(),
                                         -1),
                             item.focus_net_ids.end());
    std::sort(item.focus_net_ids.begin(), item.focus_net_ids.end());
    item.focus_net_ids.erase(std::unique(item.focus_net_ids.begin(),
                                         item.focus_net_ids.end()),
                             item.focus_net_ids.end());
    items.push_back(std::move(item));
  }
  std::stable_sort(
      items.begin(),
      items.end(),
      [](const TimingQueueItem& left, const TimingQueueItem& right) {
        if (left.endpoint_slack_ps != right.endpoint_slack_ps) {
          return left.endpoint_slack_ps < right.endpoint_slack_ps;
        }
        return left.endpoint_pin < right.endpoint_pin;
      });
  return items;
}

std::vector<int> timingQueueWindow(const std::vector<TimingQueueItem>& items,
                                 const py::dict& config,
                                 const std::set<int>& retired)
{
  const bool round_robin = config.contains("native_tree_round_robin")
                           && !config["native_tree_round_robin"].is_none()
                           && py::cast<bool>(config["native_tree_round_robin"]);
  const int endpoint_window =
      round_robin
              && config.contains("native_tree_round_robin_endpoint_window")
              && !config["native_tree_round_robin_endpoint_window"].is_none()
          ? py::cast<int>(config["native_tree_round_robin_endpoint_window"])
          : 0;
  std::set<int> endpoints;
  std::vector<int> order;
  for (int index = 0; index < static_cast<int>(items.size()); ++index) {
    const int endpoint = items[index].endpoint_pin;
    if (retired.count(endpoint) != 0) {
      continue;
    }
    if (endpoint_window > 0 && endpoints.count(endpoint) == 0
        && endpoints.size() >= static_cast<std::size_t>(endpoint_window)) {
      continue;
    }
    endpoints.insert(endpoint);
    order.push_back(index);
  }
  return order;
}

py::list selectTimingQueueItems(const py::dict& snapshot,
                              const py::dict& config)
{
  const auto items = parseTimingQueueSnapshot(snapshot);
  std::set<std::string> retired_names;
  if (config.contains("retired_endpoints")) {
    for (const auto& name : config["retired_endpoints"].cast<py::list>()) {
      retired_names.insert(py::cast<std::string>(name));
    }
  }
  std::set<int> retired;
  for (const auto& item : items) {
    const std::string key = item.endpoint_pin_name.empty()
                                ? std::to_string(item.endpoint_pin)
                                : item.endpoint_pin_name;
    if (retired_names.count(key) != 0) retired.insert(item.endpoint_pin);
  }
  const auto order = timingQueueWindow(items, config, retired);
  py::list selected;
  if (order.empty()) {
    return selected;
  }
  const py::list records = snapshot["items"].cast<py::list>();
  for (const int index : order) {
    selected.append(records[items[index].snapshot_index]);
  }
  return selected;
}

py::dict bindTimingQueueObservation(const py::dict& observation,
                                   const TimingQueueItem& item)
{
  if (item.endpoint_pin_name.empty()) {
    return observation;
  }
  if (!observation.contains("endpoint_slacks_by_name")
      || !py::isinstance<py::dict>(observation["endpoint_slacks_by_name"])) {
    throw py::value_error("native queue observation requires endpoint_slacks_by_name");
  }
  const py::dict slacks = observation["endpoint_slacks_by_name"].cast<py::dict>();
  const py::str name(item.endpoint_pin_name);
  if (!slacks.contains(name)) {
    throw py::value_error("native queue target endpoint missing from observation: "
                          + item.endpoint_pin_name);
  }
  const double slack = py::cast<double>(slacks[name]);
  if (!std::isfinite(slack)) {
    throw py::value_error("native queue target endpoint slack is not finite: "
                          + item.endpoint_pin_name);
  }
  py::dict result = observation.attr("copy")().cast<py::dict>();
  result["endpoint_pin_name"] = name;
  result["endpoint_slack_ps"] = slack;
  return result;
}

namespace {

void applyPathTiming(TimingView& timing, const py::dict& values)
{
  if (values.contains("path_driver_input_pin")) {
    timing.path_driver_input_pin
        = py::cast<int>(values["path_driver_input_pin"]);
  }
  if (values.contains("path_driver_input_transition")) {
    timing.path_driver_input_transition
        = py::cast<int>(values["path_driver_input_transition"]);
  }
  if (values.contains("path_driver_output_transition")) {
    timing.path_driver_output_transition
        = py::cast<int>(values["path_driver_output_transition"]);
  }
  if (values.contains("path_driver_lib_arc_id")) {
    timing.path_driver_lib_arc_id
        = py::cast<int>(values["path_driver_lib_arc_id"]);
  }
  if (values.contains("path_arc_delays")) {
    timing.path_arc_delays
        = values["path_arc_delays"].cast<std::vector<double>>();
  }
  if (values.contains("path_arc_local_ids")) {
    timing.path_arc_local_ids
        = values["path_arc_local_ids"].cast<std::vector<int>>();
  }
  if (values.contains("path_transitions")) {
    timing.path_transitions
        = values["path_transitions"].cast<std::vector<int>>();
  }
  if (values.contains("path_driver_input_arrival")) {
    timing.path_driver_input_arrival
        = py::cast<double>(values["path_driver_input_arrival"]);
  }
  if (values.contains("path_driver_output_arrival")) {
    timing.path_driver_output_arrival
        = py::cast<double>(values["path_driver_output_arrival"]);
  }
  if (values.contains("path_driver_arrival_valid")) {
    timing.path_driver_arrival_valid
        = py::cast<bool>(values["path_driver_arrival_valid"]);
  }
}

}  // namespace

py::dict buildNativeTimingQueueSnapshot(
    const py::dict& design_values,
    const py::dict& timing_values,
    const py::dict& liberty_values,
    const py::list& endpoint_records,
    const std::vector<int>& original_net_by_net)
{
  const auto design = rebuffer_python::designFromDict(design_values);
  const auto base_timing = rebuffer_python::timingFromDict(timing_values);
  const auto liberty = rebuffer_python::libertyFromDict(liberty_values);
  std::unordered_map<int, int> net_by_driver;
  for (int net = 0; net < static_cast<int>(design.net2driver_pin_map.size());
       ++net) {
    net_by_driver.emplace(design.net2driver_pin_map[net], net);
  }
  const auto map_net = [&](int net_id) {
    if (net_id >= 0
        && net_id < static_cast<int>(original_net_by_net.size())) {
      return original_net_by_net[net_id];
    }
    return net_id;
  };

  std::set<int> snapshot_endpoints;
  std::set<int> snapshot_roots;
  std::set<int> snapshot_drivers;
  std::set<int> snapshot_buffer_output_roots;
  py::list items;
  for (const auto& value : endpoint_records) {
    const py::dict record = value.cast<py::dict>();
    if (!record.contains("endpoint_pin") || !record.contains("slack_ps")
        || !record.contains("path_pins")) {
      throw py::value_error(
          "native timing snapshot record requires endpoint_pin, slack_ps and path_pins");
    }
    const int endpoint = py::cast<int>(record["endpoint_pin"]);
    const double slack = py::cast<double>(record["slack_ps"]);
    if (!std::isfinite(slack) || slack >= 0.0) {
      continue;
    }
    const std::vector<int> path
        = record["path_pins"].cast<std::vector<int>>();
    if (path.empty()) {
      continue;
    }
    TimingView path_timing = base_timing;
    if (record.contains("path_timing")
        && py::isinstance<py::dict>(record["path_timing"])) {
      applyPathTiming(path_timing,
                      record["path_timing"].cast<py::dict>());
    }
    std::vector<int> active_nets;
    const auto ranked_drivers
        = rankPathDriverPins(design, path_timing, liberty, path);
    for (const int driver : ranked_drivers) {
      const auto found = net_by_driver.find(driver);
      if (found == net_by_driver.end()) {
        continue;
      }
      const int net = found->second;
      if (std::find(active_nets.begin(), active_nets.end(), net)
          == active_nets.end()) {
        active_nets.push_back(net);
      }
    }
    if (active_nets.empty()) {
      for (const int pin : path) {
        if (pin < 0 || pin >= static_cast<int>(design.pin2net_map.size())) {
          continue;
        }
        const int net = design.pin2net_map[pin];
        if (net < 0 || net >= design.num_nets) {
          continue;
        }
        if (std::find(active_nets.begin(), active_nets.end(), net)
            == active_nets.end()) {
          active_nets.push_back(net);
        }
      }
    }
    if (active_nets.empty()) {
      continue;
    }

    // A setup path can contain several independently driven nets.  The old
    // snapshot collapsed them to ``active_nets.front()`` and left the other
    // drivers only in the logical net list.  Strict current-tree mode then
    // replaced the path with one root tree, so those drivers could never
    // reach the native planner.  Emit one queue item per distinct path
    // driver net; the native queue can subsequently expand each item into
    // its reachable branch partitions.  Keep the current net id here: an
    // existing buffer output is a valid OpenROAD repair target and must not
    // be folded back into its upstream root before the tree is imported.
    std::set<int> emitted_driver_nets;
    for (const int active_net : active_nets) {
      if (active_net < 0
          || active_net >= static_cast<int>(design.net2driver_pin_map.size())
          || !emitted_driver_nets.insert(active_net).second) {
        continue;
      }
      const int driver_pin = design.net2driver_pin_map[active_net];
      if (driver_pin < 0 || driver_pin >= design.num_pins) {
        continue;
      }

      py::dict item;
      item["endpoint_pin"] = endpoint;
      if (record.contains("endpoint_pin_name")) {
        item["endpoint_pin_name"] = record["endpoint_pin_name"];
      }
      item["slack_ps"] = slack;
      item["path_pins"] = path;
      // Keep the path-local driver net as the legacy queue identity.  Native
      // current-tree mode selects ``root_net_id`` below; non-tree callers
      // therefore still visit every path driver independently.
      py::list net_ids;
      net_ids.append(map_net(active_net));
      item["net_ids"] = std::move(net_ids);
      item["root_net_id"] = active_net;
      item["driver_pin_id"] = driver_pin;
      item["tree_key"] = std::to_string(active_net) + ":"
                         + std::to_string(driver_pin);
      items.append(std::move(item));
      snapshot_endpoints.insert(endpoint);
      snapshot_roots.insert(active_net);
      snapshot_drivers.insert(driver_pin);
      if (driver_pin < static_cast<int>(design.pin2node_map.size())) {
        const int node = design.pin2node_map[driver_pin];
        if (node >= 0 && node < static_cast<int>(design.node_is_buffer.size())
            && design.node_is_buffer[node] != 0) {
          snapshot_buffer_output_roots.insert(active_net);
        }
      }
    }
  }

  py::dict result;
  result["status"] = "ok";
  result["items"] = std::move(items);
  py::dict statistics;
  statistics["item_count"] = py::len(result["items"]);
  statistics["endpoint_count"] = static_cast<int>(snapshot_endpoints.size());
  statistics["root_count"] = static_cast<int>(snapshot_roots.size());
  statistics["driver_count"] = static_cast<int>(snapshot_drivers.size());
  statistics["buffer_output_root_count"]
      = static_cast<int>(snapshot_buffer_output_roots.size());
  statistics["path_driver_item_count"] = py::len(result["items"]);
  result["snapshot_statistics"] = std::move(statistics);
  return result;
}

}  // namespace rebuffer
