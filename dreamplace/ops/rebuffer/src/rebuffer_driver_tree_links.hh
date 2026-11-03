// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "rebuffer_types.hh"

namespace rebuffer {
struct CurrentDriverBufferLink
{
  int instance_id = -1;
  int input_pin_id = -1;
  int output_pin_id = -1;
  int input_net_id = -1;
  int output_net_id = -1;
};

bool collectCurrentDriverBufferLinks(const DesignView& design,
    std::vector<CurrentDriverBufferLink>& links,
    std::unordered_map<int, CurrentDriverBufferLink>& by_input_pin, std::string& reason);
}  // namespace rebuffer
