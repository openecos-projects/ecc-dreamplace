// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>
#include <functional>

#include "rebuffer_liberty.hh"
#include "rebuffer_repair.hh"
#include "rebuffer_types.hh"

namespace rebuffer {

// Both materialized payloads and the persistent graph borrow the same ranking
// inputs. Accessors resolve their different node/cell and storage ownership.
struct PathDriverAccess
{
  int num_pins;
  const std::vector<int>& pin_net;
  const std::vector<int>& net_drivers;
  const std::vector<int>& pin_owner;
  const std::vector<double>& rise_slew;
  const std::vector<double>& fall_slew;
  const std::vector<double>& load;
  std::function<int(int)> cell_for_node;
  std::function<bool(int)> is_clock;
  std::function<bool(int, double&, double&)> input_slew;
};

std::vector<int> rankPathDriverPins(const PathDriverAccess& access,
    const LibertyView& liberty, const std::vector<int>& path,
    const TimingView* path_context = nullptr, bool openroad_intrinsic = false);

// Return driver pins on one timing path in descending load-dependent delay.
// This is the native counterpart of OpenROAD SetupLegacyBase::rankPathDrivers.
std::vector<int> rankPathDriverPins(const DesignView& design,
                                    const TimingView& timing,
                                    const LibertyView& liberty,
                                    const std::vector<int>& path,
                                    bool openroad_intrinsic = false);

}  // namespace rebuffer
