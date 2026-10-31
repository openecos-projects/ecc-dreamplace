// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_liberty.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace rebuffer {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kTargetSlewLoadCapFactor = 10.0;
constexpr double kCapRatioCutoff = 20.0;

std::size_t axisIndex(const std::vector<double>& axis, double value)
{
  const std::size_t size = axis.size();
  if (size <= 1 || value <= axis.front()) {
    return 0;
  }
  if (value >= axis.back()) {
    return size - 2;
  }
  std::size_t lower = 0;
  std::size_t upper = size - 1;
  while (upper - lower > 1) {
    const std::size_t middle = (upper + lower) / 2;
    if (value >= axis[middle]) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

double tableValue(const LutTable& table, int row, int column)
{
  return table.values[static_cast<std::size_t>(row) * table.dim1 + column];
}

double gateDelay(const TimingArcView& arc,
                 bool rise,
                 double input_slew,
                 double load_cap)
{
  return lutFindValue(rise ? arc.rise_delay : arc.fall_delay,
                      input_slew,
                      load_cap);
}

double gateSlew(const TimingArcView& arc,
                bool rise,
                double input_slew,
                double load_cap)
{
  return lutFindValue(rise ? arc.rise_transition : arc.fall_transition,
                      input_slew,
                      load_cap);
}

double maxCapSlew(const LutTable& transition, double input_slew)
{
  if (transition.cap_axis.empty()) {
    return 0.0;
  }
  const double cap = transition.cap_axis.back();
  const double slew = lutFindValue(transition, input_slew, cap);
  return slew / cap;
}

double driveResistance(const LibertyCellView& cell)
{
  double resistance = 0.0;
  for (const auto& arc : cell.arcs) {
    resistance = std::max(resistance, maxCapSlew(arc.rise_transition, 0.0));
    resistance = std::max(resistance, maxCapSlew(arc.fall_transition, 0.0));
  }
  return resistance;
}

CharacterizedBuffer characterizeCell(const LibertyView& liberty,
                                     int cell_index)
{
  const auto& cell = liberty.cells[cell_index];
  CharacterizedBuffer buffer;
  buffer.cell_index = cell_index;
  buffer.name = cell.name;
  buffer.main_id = cell.main_id;
  buffer.area = cell.area;
  buffer.leakage = cell.leakage;
  buffer.input_cap = cell.input_cap;
  buffer.max_cap = cell.max_cap;
  buffer.max_slew = cell.max_slew;
  buffer.drive_resistance = driveResistance(cell);
  buffer.intrinsic_delay = cell.intrinsic_delay;
  buffer.output_cap = cell.output_cap;
  return buffer;
}

double bufferDelay(const LibertyCellView& cell,
                   const std::array<double, 2>& target_slew,
                   double load_cap)
{
  double delay = 0.0;
  for (const auto& arc : cell.arcs) {
    delay = std::max(
        delay,
        gateDelay(arc, false, target_slew[1], load_cap));
    delay = std::max(
        delay,
        gateDelay(arc, true, target_slew[0], load_cap));
  }
  return delay;
}

double findTargetLoad(const LibertyCellView& cell,
                      const TimingArcView& arc,
                      bool rise,
                      double target_slew)
{
  const LutTable& transition
      = rise ? arc.rise_transition : arc.fall_transition;
  if (transition.cap_axis.empty()) {
    return 0.0;
  }
  auto objective = [&](double load_cap) {
    return lutFindValue(transition, target_slew, load_cap) - target_slew;
  };
  double cap1 = 0.0;
  double cap2 = 1.0e-12;
  const double tolerance = 0.01;
  double diff1 = objective(cap1);
  if (diff1 > 0.0) {
    return 0.0;
  }
  double diff2 = objective(cap2);
  while (std::abs(cap1 - cap2) > std::max(cap1, cap2) * tolerance) {
    if (diff2 < 0.0) {
      cap1 = cap2;
      cap2 *= 2.0;
      diff2 = objective(cap2);
    } else {
      const double cap3 = (cap1 + cap2) / 2.0;
      const double diff3 = objective(cap3);
      if (diff3 < 0.0) {
        cap1 = cap3;
      } else {
        cap2 = cap3;
        diff2 = diff3;
      }
    }
  }
  (void) cell;
  return cap1;
}

double findBufferLoadLimitImpliedByDriverSlew(const LibertyCellView& cell,
                                              double drive_resistance)
{
  if (drive_resistance <= 0.0 || cell.max_slew <= 0.0) {
    return kInf;
  }
  const double max_slew = cell.max_slew;
  const double input_slew = cell.max_slew;
  auto objective = [&](double load_cap) {
    double slew = -kInf;
    for (const auto& arc : cell.arcs) {
      slew = std::max(
          slew,
          gateSlew(arc, false, input_slew, load_cap));
      slew = std::max(
          slew,
          gateSlew(arc, true, input_slew, load_cap));
    }
    return slew - max_slew;
  };
  double cap1 = 0.0;
  double cap2 = max_slew / drive_resistance * 2.0;
  const double tolerance = 0.01;
  double diff1 = objective(cap2);
  while (std::abs(cap1 - cap2) > std::max(cap1, cap2) * tolerance) {
    if (diff1 < 0.0) {
      cap1 = cap2;
      cap2 *= 2.0;
      diff1 = objective(cap2);
    } else {
      const double cap3 = (cap1 + cap2) / 2.0;
      const double diff2 = objective(cap3);
      if (diff2 < 0.0) {
        cap1 = cap3;
      } else {
        cap2 = cap3;
        diff1 = diff2;
      }
    }
  }
  return cap1;
}

double findTargetLoad(const LibertyCellView& cell,
                      const std::array<double, 2>& target_slew)
{
  double sum = 0.0;
  int count = 0;
  for (const auto& arc : cell.arcs) {
    sum += findTargetLoad(cell, arc, false, target_slew[1]);
    sum += findTargetLoad(cell, arc, true, target_slew[0]);
    count += 2;
  }
  return count == 0 ? 0.0 : sum / count;
}

std::vector<double> bufferCapTestPoints(const LibertyCellView& cell)
{
  std::vector<double> points;
  for (const auto& arc : cell.arcs) {
    for (const auto* table : {&arc.fall_delay, &arc.rise_delay}) {
      points.insert(
          points.end(),
          table->cap_axis.begin(),
          table->cap_axis.end());
    }
  }
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  return points;
}

bool bufferSizeOutmatched(const LibertyView& liberty,
                          const CharacterizedBuffer& worse,
                          const CharacterizedBuffer& better,
                          double max_drive_resistance,
                          const std::array<double, 2>& target_slew)
{
  const auto& worse_cell = liberty.cells[worse.cell_index];
  const auto& better_cell = liberty.cells[better.cell_index];
  const double extra_input_cap
      = std::max(better.input_cap - worse.input_cap, 0.0);
  const double delay_penalty = max_drive_resistance * extra_input_cap;
  const double worse_limit = worse.max_cap;

  std::vector<double> test_points = bufferCapTestPoints(worse_cell);
  const auto better_points = bufferCapTestPoints(better_cell);
  test_points.insert(
      test_points.end(),
      better_points.begin(),
      better_points.end());
  if (test_points.empty()) {
    return false;
  }
  std::sort(test_points.begin(), test_points.end());
  test_points.erase(
      std::unique(test_points.begin(), test_points.end()),
      test_points.end());
  const double cap_limit
      = std::min(better.input_cap, worse.input_cap) * kCapRatioCutoff;
  for (const double point : test_points) {
    if ((worse_limit == 0.0 || point <= worse_limit)
        && point < cap_limit) {
      const double better_delay
          = bufferDelay(better_cell, target_slew, point);
      const double worse_delay
          = bufferDelay(worse_cell, target_slew, point);
      if (better_delay + delay_penalty > worse_delay) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

double lutFindValue(const LutTable& table, double input_slew, double load_cap)
{
  if (table.values.empty() || table.dim0 <= 0 || table.dim1 <= 0) {
    return 0.0;
  }
  if (table.dim0 == 1 && table.dim1 == 1) {
    return table.values.front();
  }
  if (table.dim0 == 1) {
    const std::size_t cap_index = axisIndex(table.cap_axis, load_cap);
    const double x0 = table.cap_axis[cap_index];
    const double x1 = table.cap_axis[cap_index + 1];
    const double dx = (load_cap - x0) / (x1 - x0);
    return table.values[cap_index] * (1.0 - dx)
        + table.values[cap_index + 1] * dx;
  }
  if (table.dim1 == 1) {
    const std::size_t slew_index = axisIndex(table.trans_axis, input_slew);
    const double x0 = table.trans_axis[slew_index];
    const double x1 = table.trans_axis[slew_index + 1];
    const double dx = (input_slew - x0) / (x1 - x0);
    return table.values[slew_index] * (1.0 - dx)
        + table.values[slew_index + 1] * dx;
  }

  const std::size_t slew_index = axisIndex(table.trans_axis, input_slew);
  const std::size_t cap_index = axisIndex(table.cap_axis, load_cap);
  const double slew0 = table.trans_axis[slew_index];
  const double slew1 = table.trans_axis[slew_index + 1];
  const double cap0 = table.cap_axis[cap_index];
  const double cap1 = table.cap_axis[cap_index + 1];
  const double ds = (input_slew - slew0) / (slew1 - slew0);
  const double dc = (load_cap - cap0) / (cap1 - cap0);
  const double v00 = tableValue(table, slew_index, cap_index);
  const double v01 = tableValue(table, slew_index, cap_index + 1);
  const double v10 = tableValue(table, slew_index + 1, cap_index);
  const double v11 = tableValue(table, slew_index + 1, cap_index + 1);
  return (1.0 - ds) * (1.0 - dc) * v00
      + (1.0 - ds) * dc * v01
      + ds * dc * v11
      + ds * (1.0 - dc) * v10;
}

double cellDriveResistance(const LibertyCellView& cell)
{
  return driveResistance(cell);
}

double cellIntrinsicDelay(const LibertyCellView& cell)
{
  // OpenSTA TimingArc::intrinsicDelay evaluates gateDelay at zero input
  // slew and zero load; LibertyPort takes the maximum positive arc delay.
  double delay = 0.0;
  for (const auto& arc : cell.arcs) {
    delay = std::max(delay, lutFindValue(arc.rise_delay, 0.0, 0.0));
    delay = std::max(delay, lutFindValue(arc.fall_delay, 0.0, 0.0));
  }
  return delay;
}

bool characterizeBuffers(const LibertyView& liberty,
                         CharacterizationResult& result,
                         std::string& reason)
{
  if (liberty.cells.empty()) {
    reason = "empty_liberty";
    return false;
  }

  std::vector<CharacterizedBuffer> candidates;
  for (int cell_index = 0;
       cell_index < static_cast<int>(liberty.cells.size());
       ++cell_index) {
    const auto& cell = liberty.cells[cell_index];
    if (!cell.is_buffer || cell.arcs.empty() || cell.input_cap <= 0.0
        || cell.max_cap <= 0.0) {
      continue;
    }
    auto buffer = characterizeCell(liberty, cell_index);
    if (buffer.drive_resistance <= 0.0) {
      continue;
    }
    candidates.push_back(std::move(buffer));
  }
  if (candidates.empty()) {
    reason = "no_characterizable_buffers";
    return false;
  }

  std::sort(
      candidates.begin(),
      candidates.end(),
      [](const CharacterizedBuffer& left, const CharacterizedBuffer& right) {
        if (left.drive_resistance != right.drive_resistance) {
          return left.drive_resistance < right.drive_resistance;
        }
        return left.name < right.name;
      });

  constexpr int kNumBuckets = 5;
  const int bucket_size = static_cast<int>(candidates.size()) / kNumBuckets;
  const int remainder = static_cast<int>(candidates.size()) % kNumBuckets;
  result.filtered_buffers.clear();
  for (int bucket = 0; bucket < kNumBuckets; ++bucket) {
    const int start = bucket * bucket_size + std::min(bucket, remainder);
    const int count = bucket_size + (bucket < remainder ? 1 : 0);
    std::vector<std::pair<double, int>> ranked;
    for (int offset = 0; offset < count; ++offset) {
      const int index = start + offset;
      ranked.emplace_back(
          candidates[index].drive_resistance * candidates[index].input_cap,
          index);
    }
    std::sort(
        ranked.begin(),
        ranked.end(),
        [&](const auto& left, const auto& right) {
          if (left.first != right.first) {
            return left.first < right.first;
          }
          return candidates[left.second].name
              < candidates[right.second].name;
        });
    const int selected = std::min(2, static_cast<int>(ranked.size()));
    for (int index = 0; index < selected; ++index) {
      result.filtered_buffers.push_back(
          candidates[ranked[index].second]);
    }
  }
  if (result.filtered_buffers.empty()) {
    reason = "no_filtered_buffers";
    return false;
  }
  result.lowest_drive_index
      = static_cast<int>(result.filtered_buffers.size()) - 1;

  std::array<double, 2> slew_sum{0.0, 0.0};
  std::array<int, 2> slew_count{0, 0};
  for (const auto& buffer : result.filtered_buffers) {
    const auto& cell = liberty.cells[buffer.cell_index];
    const double load_cap = buffer.input_cap * kTargetSlewLoadCapFactor;
    for (int rf = 0; rf < 2; ++rf) {
      const bool rise = rf == 0;
      double slew = 0.0;
      bool found = false;
      for (const auto& arc : cell.arcs) {
        const double first
            = gateSlew(arc, rise, 0.0, load_cap);
        const double second
            = gateSlew(arc, rise, first, load_cap);
        slew = std::max(slew, second);
        found = true;
      }
      if (found) {
        slew_sum[rf] += slew;
        ++slew_count[rf];
      }
    }
  }
  if (slew_count[0] == 0 || slew_count[1] == 0) {
    reason = "missing_buffer_target_slew";
    return false;
  }
  result.target_slew[0] = slew_sum[0] / slew_count[0];
  result.target_slew[1] = slew_sum[1] / slew_count[1];

  for (auto& buffer : result.filtered_buffers) {
    const auto& cell = liberty.cells[buffer.cell_index];
    buffer.target_load = findTargetLoad(cell, result.target_slew);
    const double implied
        = findBufferLoadLimitImpliedByDriverSlew(
            cell,
            buffer.drive_resistance);
    const double cap_margined
        = buffer.max_cap > 0.0 ? buffer.max_cap * 0.8 : kInf;
    buffer.margined_max_cap = std::min(cap_margined, implied);
  }

  std::vector<CharacterizedBuffer> by_input_cap
      = result.filtered_buffers;
  std::stable_sort(
      by_input_cap.begin(),
      by_input_cap.end(),
      [](const CharacterizedBuffer& left, const CharacterizedBuffer& right) {
        return left.input_cap < right.input_cap;
      });
  const double max_drive_resistance
      = result.filtered_buffers[result.lowest_drive_index].drive_resistance;
  result.fast_buffers.clear();
  for (const auto& buffer : by_input_cap) {
    if (result.fast_buffers.empty()) {
      result.fast_buffers.push_back(buffer);
      continue;
    }
    if (!bufferSizeOutmatched(
            liberty,
            buffer,
            result.fast_buffers.back(),
            max_drive_resistance,
            result.target_slew)) {
      while (!result.fast_buffers.empty()
             && bufferSizeOutmatched(
                 liberty,
                 result.fast_buffers.back(),
                 buffer,
                 max_drive_resistance,
                 result.target_slew)) {
        result.fast_buffers.pop_back();
      }
      result.fast_buffers.push_back(buffer);
    }
  }
  return true;
}

}  // namespace rebuffer
