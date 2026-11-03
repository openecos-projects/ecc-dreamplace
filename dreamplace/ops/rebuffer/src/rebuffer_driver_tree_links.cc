// SPDX-License-Identifier: BSD-3-Clause
// Canonical buffer boundaries shared by tree import and graph planning.
#include "rebuffer_driver_tree_links.hh"
#include <algorithm>
#include <tuple>

namespace rebuffer {
bool collectCurrentDriverBufferLinks(const DesignView& design,
                        std::vector<CurrentDriverBufferLink>& links,
                        std::unordered_map<int, CurrentDriverBufferLink>& by_input_pin,
                        std::string& reason)
{
  std::vector<std::vector<int>> pins_by_instance(design.num_nodes);
  for (int pin = 0; pin < design.num_pins; ++pin) {
    const int instance = design.pin2node_map[pin];
    if (instance < 0 || instance >= design.num_nodes) {
      reason = "pin_instance_out_of_range";
      return false;
    }
    pins_by_instance[instance].push_back(pin);
  }
  for (int instance = 0; instance < design.num_nodes; ++instance) {
    if (design.node_is_buffer[instance] == 0) {
      continue;
    }
    const auto& pins = pins_by_instance[instance];
    if (pins.size() != 2) {
      reason = "buffer_instance_must_have_two_signal_pins";
      return false;
    }
    int input_pin = -1;
    int output_pin = -1;
    for (const int pin : pins) {
      const int net_id = design.pin2net_map[pin];
      if (net_id < 0 || net_id >= design.num_nets) {
        reason = "buffer_pin_net_out_of_range";
        return false;
      }
      if (design.net2driver_pin_map[net_id] == pin) {
        if (output_pin >= 0) {
          reason = "buffer_has_multiple_output_pins";
          return false;
        }
        output_pin = pin;
      } else {
        if (input_pin >= 0) {
          reason = "buffer_has_multiple_input_pins";
          return false;
        }
        input_pin = pin;
      }
    }
    if (input_pin < 0 || output_pin < 0) {
      reason = "buffer_input_output_pin_not_identified";
      return false;
    }
    const CurrentDriverBufferLink link{instance,
                          input_pin,
                          output_pin,
                          design.pin2net_map[input_pin],
                          design.pin2net_map[output_pin]};
    if (!by_input_pin.emplace(input_pin, link).second) {
      reason = "duplicate_buffer_input_pin";
      return false;
    }
    links.push_back(link);
  }
  std::sort(links.begin(), links.end(), [](const CurrentDriverBufferLink& lhs, const CurrentDriverBufferLink& rhs) {
    return std::tie(lhs.input_net_id, lhs.input_pin_id, lhs.output_net_id)
           < std::tie(rhs.input_net_id, rhs.input_pin_id, rhs.output_net_id);
  });
  return true;
}

}  // namespace rebuffer
