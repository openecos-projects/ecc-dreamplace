// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_types.hh"

#include <algorithm>
#include <cstddef>

namespace rebuffer {

namespace {

bool valid_csr(const std::vector<int>& start,
               std::size_t element_count,
               std::string& reason,
               const char* name)
{
  if (start.size() < 2 || start.front() != 0
      || start.back() != static_cast<int>(element_count)) {
    reason = std::string(name) + "_invalid_csr_bounds";
    return false;
  }
  for (std::size_t index = 1; index < start.size(); ++index) {
    if (start[index] < start[index - 1]) {
      reason = std::string(name) + "_not_monotonic";
      return false;
    }
  }
  return true;
}

bool all_in_range(const std::vector<int>& values,
                  int upper_bound,
                  std::string& reason,
                  const char* name,
                  bool allow_negative_one = false)
{
  const int lower = allow_negative_one ? -1 : 0;
  for (const int value : values) {
    if (value < lower || value >= upper_bound) {
      reason = std::string(name) + "_out_of_range";
      return false;
    }
  }
  return true;
}

}  // namespace

bool validate_design(const DesignView& design, std::string& reason)
{
  if (design.dbu <= 0) {
    reason = "invalid_dbu";
    return false;
  }
  if (design.num_nets <= 0 || design.num_pins <= 0 || design.num_nodes <= 0) {
    reason = "empty_design";
    return false;
  }
  if (!valid_csr(
          design.flat_net2pin_start_map, design.flat_net2pin_map.size(),
          reason, "flat_net2pin_start_map")) {
    return false;
  }
  if (static_cast<int>(design.flat_net2pin_start_map.size()) - 1
      != design.num_nets) {
    reason = "net_count_mismatch";
    return false;
  }
  if (static_cast<int>(design.net2driver_pin_map.size()) != design.num_nets
      || static_cast<int>(design.pin2node_map.size()) != design.num_pins
      || static_cast<int>(design.pin2net_map.size()) != design.num_pins
      || static_cast<int>(design.pin_offset_x.size()) != design.num_pins
      || static_cast<int>(design.pin_offset_y.size()) != design.num_pins
      || static_cast<int>(design.node_x.size()) != design.num_nodes
      || static_cast<int>(design.node_y.size()) != design.num_nodes) {
    reason = "array_length_mismatch";
    return false;
  }
  if (!all_in_range(
          design.flat_net2pin_map, design.num_pins, reason, "flat_net2pin_map")
      || !all_in_range(
          design.pin2node_map, design.num_nodes, reason, "pin2node_map")
      || !all_in_range(design.pin2net_map, design.num_nets, reason, "pin2net_map")
      || !all_in_range(design.net2driver_pin_map,
                       design.num_pins,
                       reason,
                       "net2driver_pin_map",
                       true)) {
    return false;
  }
  for (int net_id = 0; net_id < design.num_nets; ++net_id) {
    const int begin = design.flat_net2pin_start_map[net_id];
    const int end = design.flat_net2pin_start_map[net_id + 1];
    const int driver = design.net2driver_pin_map[net_id];
    if (driver >= 0
        && std::find(design.flat_net2pin_map.begin() + begin,
                     design.flat_net2pin_map.begin() + end,
                     driver)
               == design.flat_net2pin_map.begin() + end) {
      reason = "net_driver_not_in_net";
      return false;
    }
  }
  return true;
}

}  // namespace rebuffer
