// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_queue_snapshot.hh"


#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>

#include "pybind11/stl.h"

#include "rebuffer_repair.hh"


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

std::shared_ptr<NativeTimingQueueSnapshot> stateFromSnapshot(
    const py::dict& snapshot)
{
  if (!snapshot.contains("_native_queue_state")
      || snapshot["_native_queue_state"].is_none()) {
    return nullptr;
  }
  try {
    return snapshot["_native_queue_state"]
        .cast<std::shared_ptr<NativeTimingQueueSnapshot>>();
  } catch (const py::cast_error&) {
    return nullptr;
  }
}

std::shared_ptr<NativeTimingQueueContext> queueContextFromViews(
    DesignView design,
    LibertyView liberty,
    const py::object& design_source,
    const py::object& liberty_source)
{
  auto context = std::make_shared<NativeTimingQueueContext>();
  context->design = std::make_shared<DesignView>(std::move(design));
  context->liberty = std::make_shared<LibertyView>(std::move(liberty));
  context->design_source = design_source;
  context->liberty_source = liberty_source;
  return context;
}

std::shared_ptr<NativeTimingQueueContext> queueContextFromDicts(
    const py::dict& design_values,
    const py::dict& liberty_values)
{
  return queueContextFromViews(
      rebuffer_python::designFromDict(design_values),
      rebuffer_python::libertyFromDict(liberty_values),
      py::reinterpret_borrow<py::object>(design_values),
      py::reinterpret_borrow<py::object>(liberty_values));
}

bool sameQueueContextSources(
    const std::shared_ptr<NativeTimingQueueContext>& context,
    const py::dict& design_values,
    const py::dict& liberty_values)
{
  return context != nullptr
      && context->design_source.is(design_values)
      && context->liberty_source.is(liberty_values);
}

std::shared_ptr<NativeTimingQueueContext> queueContextForUpdate(
    const std::shared_ptr<NativeTimingQueueContext>& previous,
    const py::dict& design_values,
    const py::dict& liberty_values,
    bool* design_reused,
    bool* liberty_reused)
{
  const bool reuse_design = previous != nullptr
      && previous->design_source.is(design_values);
  const bool reuse_liberty = previous != nullptr
      && previous->liberty_source.is(liberty_values);
  if (design_reused != nullptr) *design_reused = reuse_design;
  if (liberty_reused != nullptr) *liberty_reused = reuse_liberty;
  if (reuse_design && reuse_liberty) return previous;

  auto context = std::make_shared<NativeTimingQueueContext>();
  context->design = reuse_design
      ? previous->design
      : std::make_shared<DesignView>(
            rebuffer_python::designFromDict(design_values));
  context->liberty = reuse_liberty
      ? previous->liberty
      : std::make_shared<LibertyView>(
            rebuffer_python::libertyFromDict(liberty_values));
  context->design_source = py::reinterpret_borrow<py::object>(design_values);
  context->liberty_source = py::reinterpret_borrow<py::object>(liberty_values);
  return context;
}

}  // namespace

namespace {
py::dict queueItemRecord(const TimingQueueItem& item);
}

std::vector<TimingQueueItem> parseTimingQueueSnapshot(const py::dict& snapshot)
{
  if (const auto state = stateFromSnapshot(snapshot); state != nullptr) {
    return state->items;
  }
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
    if (values.contains("gradient_score") && !values["gradient_score"].is_none()) {
      item.gradient_score = py::cast<double>(values["gradient_score"]);
      item.has_gradient_score = std::isfinite(item.gradient_score);
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

void reorderEndpointGroupsByGradient(
    const std::vector<TimingQueueItem>& items,
    std::vector<int>* order,
    double slack_window_ps = std::numeric_limits<double>::infinity())
{
  struct EndpointGroup
  {
    int first_position = -1;
    double score = 0.0;
    double slack = 0.0;
    int band = 0;
  };

  std::unordered_map<int, int> group_by_endpoint;
  std::vector<EndpointGroup> groups;
  std::vector<int> original_position(items.size(), -1);
  for (int position = 0; position < static_cast<int>(order->size());
       ++position) {
    const int item_index = (*order)[position];
    original_position[item_index] = position;
    const auto& item = items[item_index];
    const auto found = group_by_endpoint.find(item.endpoint_pin);
    if (found == group_by_endpoint.end()) {
      const double score = item.has_gradient_score
                               ? item.gradient_score
                               : 0.0;
      group_by_endpoint.emplace(item.endpoint_pin, groups.size());
      groups.push_back({position, score, item.endpoint_slack_ps});
      continue;
    }
    auto& group = groups[found->second];
    if (item.has_gradient_score) {
      group.score = std::max(group.score, item.gradient_score);
    }
    group.slack = std::min(group.slack, item.endpoint_slack_ps);
  }

  double band_anchor = 0.0;
  int band = -1;
  for (auto& group : groups) {
    if (band < 0 || group.slack > band_anchor + slack_window_ps) {
      ++band;
      band_anchor = group.slack;
    }
    group.band = band;
  }

  std::stable_sort(
      order->begin(),
      order->end(),
      [&](int left_index, int right_index) {
        const auto& left_item = items[left_index];
        const auto& right_item = items[right_index];
        const int left_group = group_by_endpoint.at(left_item.endpoint_pin);
        const int right_group = group_by_endpoint.at(right_item.endpoint_pin);
        if (left_group == right_group) {
          // Preserve rankPathDriverPins order inside one endpoint group.
          return original_position[left_index]
                 < original_position[right_index];
        }
        const auto& left = groups[left_group];
        const auto& right = groups[right_group];
        if (left.band != right.band) return left.band < right.band;
        if (left.score != right.score) return left.score > right.score;
        if (left.slack != right.slack) return left.slack < right.slack;
        return left.first_position < right.first_position;
      });
}

std::shared_ptr<NativeTimingQueueSnapshot> materializeQueueState(
    const py::dict& snapshot,
    const std::shared_ptr<NativeTimingQueueContext>& context = nullptr)
{
  auto state = std::make_shared<NativeTimingQueueSnapshot>();
  if (!snapshot.contains("items")
      || !py::isinstance<py::list>(snapshot["items"])) {
    throw py::value_error("native timing snapshot state requires list items");
  }
  state->records = snapshot["items"].cast<py::list>();
  state->items = parseTimingQueueSnapshot(snapshot);
  state->context = context;
  return state;
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
    order.push_back(index);
  }
  const bool gradient_ordering = config.contains("gradient_ordering")
      && !config["gradient_ordering"].is_none()
      && py::cast<bool>(config["gradient_ordering"]);
  const bool gradient_window_tiebreak = config.contains("gradient_window_tiebreak")
      && !config["gradient_window_tiebreak"].is_none()
      && py::cast<bool>(config["gradient_window_tiebreak"]);
  if (gradient_ordering && !gradient_window_tiebreak) {
    reorderEndpointGroupsByGradient(items, &order);
  }
  std::vector<int> window;
  for (const int index : order) {
    const int endpoint = items[index].endpoint_pin;
    if (endpoint_window > 0 && endpoints.count(endpoint) == 0
        && endpoints.size() >= static_cast<std::size_t>(endpoint_window)) {
      continue;
    }
    endpoints.insert(endpoint);
    window.push_back(index);
  }
  if (gradient_ordering && gradient_window_tiebreak) {
    const double slack_window_ps = config.contains("gradient_slack_window_ps")
        ? config["gradient_slack_window_ps"].cast<double>()
        : std::numeric_limits<double>::infinity();
    if (std::isnan(slack_window_ps) || slack_window_ps < 0) {
      throw py::value_error("gradient slack window must be nonnegative");
    }
    reorderEndpointGroupsByGradient(items, &window, slack_window_ps);
  }
  return window;
}

std::vector<int> selectTimingQueueIndices(const std::vector<TimingQueueItem>& items,
                                         const py::dict& config)
{
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
  return timingQueueWindow(items, config, retired);
}

py::list selectTimingQueueItems(const py::dict& snapshot,
                              const py::dict& config)
{
  const auto state = stateFromSnapshot(snapshot);
  std::vector<TimingQueueItem> parsed_items;
  if (state == nullptr) parsed_items = parseTimingQueueSnapshot(snapshot);
  const auto& items = state == nullptr ? parsed_items : state->items;
  const auto order = selectTimingQueueIndices(items, config);
  py::list selected;
  if (order.empty()) {
    return selected;
  }
  for (const int index : order) {
    if (state != nullptr && py::len(state->records) == 0) {
      selected.append(queueItemRecord(items[index]));
    } else {
      const py::list records = state != nullptr
                                   ? state->records
                                   : snapshot["items"].cast<py::list>();
      selected.append(records[items[index].snapshot_index]);
    }
  }
  return selected;
}

void updateNativeTimingQueueFocus(const py::dict& snapshot,
                                  const py::list& focus_updates)
{
  const auto state = stateFromSnapshot(snapshot);
  if (state == nullptr) {
    return;
  }
  for (const auto& value : focus_updates) {
    const py::dict update = value.cast<py::dict>();
    const std::string tree_key = stringOr(update, "tree_key");
    const int endpoint = update.contains("endpoint_pin")
                             ? py::cast<int>(update["endpoint_pin"])
                             : -1;
    const int root_net = update.contains("root_net_id")
                             ? py::cast<int>(update["root_net_id"])
                             : -1;
    const std::vector<int> focus = update.contains("focus_net_ids")
                                       ? update["focus_net_ids"].cast<std::vector<int>>()
                                       : std::vector<int>();
    for (auto& item : state->items) {
      const bool matches_tree = !tree_key.empty() && item.tree_key == tree_key;
      const bool matches_identity = item.endpoint_pin == endpoint
                                    && item.root_net_id == root_net;
      if (!matches_tree && !matches_identity) {
        continue;
      }
      item.focus_net_ids = focus;
      if (item.snapshot_index >= 0
          && item.snapshot_index < py::len(state->records)) {
        py::dict record = state->records[item.snapshot_index].cast<py::dict>();
        record["focus_net_ids"] = focus;
      }
    }
  }
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

namespace {

struct QueueItemBuild
{
  std::vector<TimingQueueItem> items;
};

QueueItemBuild buildNativeTimingQueueItemsView(
    const DesignView& design,
    const TimingView& base_timing,
    const LibertyView& liberty,
    const py::list& endpoint_records,
    const std::vector<int>& original_net_by_net)
{
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

  QueueItemBuild result;
  int record_index = 0;
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
      ++record_index;
      continue;
    }
    const std::vector<int> path
        = record["path_pins"].cast<std::vector<int>>();
    ++record_index;
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
      TimingQueueItem item;
      item.snapshot_index = record_index - 1;
      item.endpoint_pin = endpoint;
      if (record.contains("endpoint_pin_name")) {
        item.endpoint_pin_name
            = py::cast<std::string>(record["endpoint_pin_name"]);
      }
      item.endpoint_slack_ps = slack;
      item.path_pins = path;
      item.net_ids.push_back(map_net(active_net));
      item.root_net_id = active_net;
      item.driver_pin_id = driver_pin;
      if (record.contains("gradient_score_by_net")
          && py::isinstance<py::dict>(record["gradient_score_by_net"])) {
        const py::dict scores = record["gradient_score_by_net"].cast<py::dict>();
        const py::int_ key(active_net);
        if (scores.contains(key)) {
          item.gradient_score = py::cast<double>(scores[key]);
          item.has_gradient_score = std::isfinite(item.gradient_score);
        }
      }
      item.tree_key = std::to_string(active_net) + ":"
                      + std::to_string(driver_pin);
      result.items.push_back(std::move(item));
    }
  }
  std::stable_sort(
      result.items.begin(), result.items.end(),
      [](const TimingQueueItem& left, const TimingQueueItem& right) {
        if (left.endpoint_slack_ps != right.endpoint_slack_ps) {
          return left.endpoint_slack_ps < right.endpoint_slack_ps;
        }
        return left.endpoint_pin < right.endpoint_pin;
      });
  return result;
}

QueueItemBuild buildNativeTimingQueueItems(
    const py::dict& design_values,
    const py::dict& timing_values,
    const py::dict& liberty_values,
    const py::list& endpoint_records,
    const std::vector<int>& original_net_by_net)
{
  return buildNativeTimingQueueItemsView(
      rebuffer_python::designFromDict(design_values),
      rebuffer_python::timingFromDict(timing_values),
      rebuffer_python::libertyFromDict(liberty_values),
      endpoint_records,
      original_net_by_net);
}

py::dict queueItemRecord(const TimingQueueItem& item)
{
  py::dict record;
  record["endpoint_pin"] = item.endpoint_pin;
  if (!item.endpoint_pin_name.empty()) {
    record["endpoint_pin_name"] = item.endpoint_pin_name;
  }
  record["slack_ps"] = item.endpoint_slack_ps;
  record["path_pins"] = item.path_pins;
  record["net_ids"] = item.net_ids;
  record["root_net_id"] = item.root_net_id;
  record["driver_pin_id"] = item.driver_pin_id;
  if (item.has_gradient_score) {
    record["gradient_score"] = item.gradient_score;
  }
  record["tree_key"] = item.tree_key;
  if (!item.focus_net_ids.empty()) {
    record["focus_net_ids"] = item.focus_net_ids;
  }
  return record;
}

py::dict queueSnapshotStatistics(const std::vector<TimingQueueItem>& items,
                                 const DesignView& design)
{
  return timingQueueSnapshotStatistics(items, [&](int driver) {
    if (driver < 0 || driver >= static_cast<int>(design.pin2node_map.size())) return false;
    const int node = design.pin2node_map[driver];
    return node >= 0 && node < static_cast<int>(design.node_is_buffer.size())
        && design.node_is_buffer[node] != 0;
  });
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
      if (record.contains("gradient_score_by_net")
          && py::isinstance<py::dict>(record["gradient_score_by_net"])) {
        const py::dict scores = record["gradient_score_by_net"].cast<py::dict>();
        const py::int_ key(active_net);
        if (scores.contains(key)) {
          item["gradient_score"] = scores[key];
        }
      }
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
  result["queue_context_reused"] = false;
  result["_native_queue_state"] = materializeQueueState(
      result,
      queueContextFromViews(
          design,
          liberty,
          py::reinterpret_borrow<py::object>(design_values),
          py::reinterpret_borrow<py::object>(liberty_values)));
  return result;
}

py::dict buildNativeTimingQueueState(
    const py::dict& design_values,
    const py::dict& timing_values,
    const py::dict& liberty_values,
    const py::list& endpoint_records,
    const std::vector<int>& original_net_by_net)
{
  const auto context = queueContextFromDicts(design_values, liberty_values);
  QueueItemBuild built = buildNativeTimingQueueItemsView(
      *context->design,
      rebuffer_python::timingFromDict(timing_values),
      *context->liberty,
      endpoint_records,
      original_net_by_net);
  return makeTimingQueueState(std::move(built.items), [&](int driver) {
    const auto& design = *context->design;
    if (driver < 0 || driver >= static_cast<int>(design.pin2node_map.size())) return false;
    const int node = design.pin2node_map[driver];
    return node >= 0 && node < static_cast<int>(design.node_is_buffer.size())
        && design.node_is_buffer[node] != 0;
  }, context);
}

py::dict updateNativeTimingQueueSnapshot(
    const py::dict& design_values,
    const py::dict& timing_values,
    const py::dict& liberty_values,
    const py::dict& previous_snapshot,
    const py::list& changed_endpoint_records,
    const std::vector<int>& active_endpoint_pins,
    const std::vector<int>& changed_endpoint_pins,
    const std::vector<int>& original_net_by_net)
{
  if (changed_endpoint_records.empty() && changed_endpoint_pins.empty()) {
    py::dict result = previous_snapshot;
    result["status"] = "ok";
    result["queue_context_reused"] = sameQueueContextSources(
        stateFromSnapshot(previous_snapshot) == nullptr
            ? nullptr
            : stateFromSnapshot(previous_snapshot)->context,
        design_values,
        liberty_values);
    return result;
  }
  // The native builder still owns path-driver ranking and changed-record
  // construction. The immutable design/Liberty views stay with the previous
  // queue state; only the new TimingView and changed records are parsed.
  const auto previous_state = stateFromSnapshot(previous_snapshot);
  bool design_reused = false;
  bool liberty_reused = false;
  const auto context = queueContextForUpdate(
      previous_state == nullptr ? nullptr : previous_state->context,
      design_values,
      liberty_values,
      &design_reused,
      &liberty_reused);
  const bool context_reused = design_reused && liberty_reused;
  const auto base_timing = rebuffer_python::timingFromDict(timing_values);
  const QueueItemBuild delta_items = buildNativeTimingQueueItemsView(
      *context->design,
      base_timing,
      *context->liberty,
      changed_endpoint_records,
      original_net_by_net);
  const std::set<int> active(active_endpoint_pins.begin(), active_endpoint_pins.end());
  const std::set<int> changed(changed_endpoint_pins.begin(), changed_endpoint_pins.end());
  py::list merged;
  if (previous_snapshot.contains("items")
      && py::isinstance<py::list>(previous_snapshot["items"])) {
    for (const auto& value : previous_snapshot["items"].cast<py::list>()) {
      const py::dict item = value.cast<py::dict>();
      if (!item.contains("endpoint_pin")) continue;
      const int endpoint = py::cast<int>(item["endpoint_pin"]);
      if (active.count(endpoint) != 0 && changed.count(endpoint) == 0) {
        merged.append(item);
      }
    }
  }
  for (const auto& item : delta_items.items) {
    merged.append(queueItemRecord(item));
  }

  const auto& design = *context->design;
  std::set<int> endpoints;
  std::set<int> roots;
  std::set<int> drivers;
  std::set<int> buffer_output_roots;
  for (const auto& value : merged) {
    const py::dict item = value.cast<py::dict>();
    const int endpoint = item.contains("endpoint_pin")
                             ? py::cast<int>(item["endpoint_pin"])
                             : -1;
    const int root = item.contains("root_net_id")
                         ? py::cast<int>(item["root_net_id"])
                         : -1;
    const int driver = item.contains("driver_pin_id")
                           ? py::cast<int>(item["driver_pin_id"])
                           : -1;
    if (endpoint >= 0) endpoints.insert(endpoint);
    if (root >= 0) roots.insert(root);
    if (driver >= 0) drivers.insert(driver);
    if (driver >= 0 && driver < static_cast<int>(design.pin2node_map.size())) {
      const int node = design.pin2node_map[driver];
      if (node >= 0 && node < static_cast<int>(design.node_is_buffer.size())
          && design.node_is_buffer[node] != 0 && root >= 0) {
        buffer_output_roots.insert(root);
      }
    }
  }

  py::dict result;
  result["status"] = "ok";
  result["items"] = std::move(merged);
  py::dict statistics;
  statistics["item_count"] = py::len(result["items"]);
  statistics["endpoint_count"] = static_cast<int>(endpoints.size());
  statistics["root_count"] = static_cast<int>(roots.size());
  statistics["driver_count"] = static_cast<int>(drivers.size());
  statistics["buffer_output_root_count"]
      = static_cast<int>(buffer_output_roots.size());
  statistics["path_driver_item_count"] = py::len(result["items"]);
  result["snapshot_statistics"] = std::move(statistics);
  result["queue_context_reused"] = context_reused;
  result["queue_design_reused"] = design_reused;
  result["queue_liberty_reused"] = liberty_reused;
  result["_native_queue_state"] = materializeQueueState(result, context);
  return result;
}

py::dict updateNativeTimingQueueState(
    const py::dict& design_values,
    const py::dict& timing_values,
    const py::dict& liberty_values,
    const py::dict& previous_snapshot,
    const py::list& changed_endpoint_records,
    const std::vector<int>& active_endpoint_pins,
    const std::vector<int>& changed_endpoint_pins,
    const std::vector<int>& original_net_by_net)
{
  const auto previous = stateFromSnapshot(previous_snapshot);
  if (previous == nullptr) {
    throw py::value_error("native queue state update requires a materialized state");
  }
  if (changed_endpoint_records.empty() && changed_endpoint_pins.empty()) {
    return previous_snapshot;
  }

  bool design_reused = false;
  bool liberty_reused = false;
  const auto context = queueContextForUpdate(
      previous->context,
      design_values,
      liberty_values,
      &design_reused,
      &liberty_reused);
  const bool context_reused = design_reused && liberty_reused;
  const auto active = std::set<int>(
      active_endpoint_pins.begin(), active_endpoint_pins.end());
  const auto changed = std::set<int>(
      changed_endpoint_pins.begin(), changed_endpoint_pins.end());
  QueueItemBuild delta = buildNativeTimingQueueItemsView(
      *context->design,
      rebuffer_python::timingFromDict(timing_values),
      *context->liberty,
      changed_endpoint_records,
      original_net_by_net);
  auto state = std::make_shared<NativeTimingQueueSnapshot>();
  state->records = py::list();
  state->context = context;
  for (const auto& item : previous->items) {
    if (active.count(item.endpoint_pin) != 0
        && changed.count(item.endpoint_pin) == 0) {
      state->items.push_back(item);
    }
  }
  state->items.insert(
      state->items.end(), delta.items.begin(), delta.items.end());
  std::stable_sort(
      state->items.begin(), state->items.end(),
      [](const TimingQueueItem& left, const TimingQueueItem& right) {
        if (left.endpoint_slack_ps != right.endpoint_slack_ps) {
          return left.endpoint_slack_ps < right.endpoint_slack_ps;
        }
        return left.endpoint_pin < right.endpoint_pin;
      });

  py::dict result;
  result["status"] = "ok";
  result["snapshot_statistics"]
      = queueSnapshotStatistics(state->items, *context->design);
  result["queue_context_reused"] = context_reused;
  result["queue_design_reused"] = design_reused;
  result["queue_liberty_reused"] = liberty_reused;
  result["_native_queue_state"] = std::move(state);
  return result;
}

}  // namespace rebuffer

// --- rebuffer_queue_state.cc ---
// Queue-state publication shared by materialized and graph-owned consumers.

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
