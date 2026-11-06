// SPDX-License-Identifier: BSD-3-Clause
// Immutable PyPlaceDB metadata imported once for a native repair campaign.
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "pybind11/pybind11.h"

namespace rebuffer {

struct OverlayActionInputs
{
  std::vector<std::string> cells, pins, nets;
  std::vector<int> pin2node, pin2net, drivers;
};

struct OverlayCarrierInputs
{
  struct Arc
  {
    std::string source, sink;
    int sense;
  };
  OverlayActionInputs identity;
  int dbu, num_terminals, num_terminal_nis;
  std::vector<std::string> nodes, masters, pin_direct;
  std::unordered_set<std::string> node_name_set, net_name_set;
  std::vector<int> cell_arc_start;
  std::vector<Arc> library_arcs;
  std::vector<double> pin_x, pin_y, node_x, node_y, width, height, lib_width, lib_height;
};

OverlayActionInputs overlayActionInputs(const pybind11::object& base);
OverlayCarrierInputs overlayCarrierInputs(const pybind11::object& base);

}  // namespace rebuffer
