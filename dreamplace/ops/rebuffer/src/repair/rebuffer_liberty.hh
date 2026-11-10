// SPDX-License-Identifier: BSD-3-Clause
// Liberty timing tables and OpenROAD-compatible buffer characterization.

#pragma once

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
};

double lutFindValue(const LutTable& table, double input_slew, double load_cap);

double cellDriveResistance(const LibertyCellView& cell);
double cellIntrinsicDelay(const LibertyCellView& cell);
double cellTargetLoad(const LibertyCellView& cell,
                      const std::array<double, 2>& target_slew);
double cellTargetSlewDelay(const LibertyCellView& cell,
                           const std::array<double, 2>& target_slew,
                           double load_cap);

bool characterizeBuffers(const LibertyView& liberty,
                         CharacterizationResult& result,
                         std::string& reason,
                         bool canonical_target_load = false);

}  // namespace rebuffer
