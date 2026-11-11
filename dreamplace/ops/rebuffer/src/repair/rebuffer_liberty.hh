// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Liberty timing tables and OpenROAD-compatible buffer characterization.


#include <array>
#include <memory>
#include <string>
#include <vector>

namespace rebuffer {

struct RepairLibraryMetadata;

struct LutTable
{
  int dim0 = 0;
  int dim1 = 0;
  std::vector<double> trans_axis;
  std::vector<double> cap_axis;
  std::vector<double> values;
};

struct TimingArcView
{
  int local_id = -1;
  int source = -1;
  int sink = -1;
  std::string source_name;
  std::string sink_name;
  int sense = 0;
  int trigger = 0;
  LutTable fall_delay;
  LutTable rise_delay;
  LutTable fall_transition;
  LutTable rise_transition;
};

struct LibertyCellView
{
  std::string name;
  int main_id = -1;
  int size = 0;
  int vt = 0;
  double area = 0.0;
  double leakage = 0.0;
  double input_cap = 0.0;
  double output_cap = 0.0;
  // Characterized once at the Liberty boundary; tree VGDP never reads LUTs.
  double intrinsic_delay = 0.0;
  double drive_resistance = 0.0;
  double max_cap = 0.0;
  double max_slew = 0.0;
  bool is_buffer = false;
  std::vector<TimingArcView> arcs;
};

struct LibertyView
{
  int dbu = 0;
  std::vector<LibertyCellView> cells;
  std::shared_ptr<const RepairLibraryMetadata> cell_metadata;
  // Ordered once at import, using the source-library order for exact ties.
  std::vector<int> buffer_equivalent_order;
};

struct CharacterizedBuffer
{
  int cell_index = -1;
  std::string name;
  int main_id = -1;
  double area = 0.0;
  double leakage = 0.0;
  double input_cap = 0.0;
  double max_cap = 0.0;
  double max_slew = 0.0;
  double drive_resistance = 0.0;
  double intrinsic_delay = 0.0;
  double output_cap = 0.0;
  double target_load = 0.0;
  double margined_max_cap = 0.0;
};

struct CharacterizationResult
{
  std::vector<CharacterizedBuffer> filtered_buffers;
  std::vector<CharacterizedBuffer> fast_buffers;
  std::array<double, 2> target_slew{0.0, 0.0};
  int lowest_drive_index = -1;
  double max_wire_length_um = 0.0;
};

double lutFindValue(const LutTable& table, double input_slew, double load_cap);

double cellDriveResistance(const LibertyCellView& cell);
double cellIntrinsicDelay(const LibertyCellView& cell);
double cellTargetLoad(const LibertyCellView& cell,
                      const std::array<double, 2>& target_slew);
double cellTargetSlewDelay(const LibertyCellView& cell,
                           const std::array<double, 2>& target_slew,
                           double load_cap);
double bufferWireDelay(const LibertyCellView& cell,
                       const std::array<double, 2>& target_slew,
                       double wire_length_um,
                       double r_unit,
                       double c_unit);
double findMaxWireLength(const LibertyView& liberty,
                         const CharacterizationResult& characterization,
                         double r_unit,
                         double c_unit);

bool characterizeBuffers(const LibertyView& liberty,
                         CharacterizationResult& result,
                         std::string& reason,
                         bool canonical_target_load = false,
                         double r_unit = 0.0,
                         double c_unit = 0.0);

}  // namespace rebuffer

// --- rebuffer_library_metadata.hh ---
// Immutable source-library attributes missing from the physical PyPlaceDB.

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

// --- rebuffer_cell_selection.hh ---

namespace pybind11 { class module_; }

namespace rebuffer {

void initializeBufferCellSelection(LibertyView& liberty);
std::vector<int> swappableBufferCells(const LibertyView& liberty,
                                      int source,
                                      double sizing_area_limit = 4.0,
                                      double sizing_leakage_limit = 4.0);
int findBufferTargetCell(const LibertyView& liberty,
                         const CharacterizationResult& characterization,
                         int source,
                         double load,
                         double sizing_area_limit = 4.0,
                         double sizing_leakage_limit = 4.0);
int findBufferSlewRepairCell(const LibertyView& liberty,
                             int source,
                             double input_slew_rise,
                             double input_slew_fall,
                             double load,
                             double slew_margin,
                             double sizing_area_limit = 4.0,
                             double sizing_leakage_limit = 4.0);
void registerCellSelectionBindings(pybind11::module_& module);

}  // namespace rebuffer
