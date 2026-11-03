// SPDX-License-Identifier: BSD-3-Clause
// Immutable source-library attributes missing from the physical PyPlaceDB.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"

namespace rebuffer {

struct LibraryCellMetadata
{
  std::string name, library, footprint, function_class, site, buffer_signature;
  int source_order = -1;
  bool dont_use = false;
  bool has_master = false;
  bool master_is_core = false;
  bool has_leakage = false;
  double leakage = 0;
};

struct RepairLibraryMetadata
{
  std::vector<LibraryCellMetadata> cells;
  std::vector<std::string> liberty_paths, lef_paths;
  pybind11::dict summary() const;
  pybind11::list records() const;
};

std::shared_ptr<RepairLibraryMetadata> readRepairLibraryMetadata(
    const std::vector<std::string>& liberty_paths,
    const std::vector<std::string>& lef_paths,
    const std::vector<std::string>& cell_names);
void registerRepairLibraryMetadata(pybind11::module_& module);

}  // namespace rebuffer
