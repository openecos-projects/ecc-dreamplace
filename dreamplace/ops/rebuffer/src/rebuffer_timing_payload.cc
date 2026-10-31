// SPDX-License-Identifier: BSD-3-Clause
// Gather exported metadata only; timing and RC evaluation remain existing ops.
#include "rebuffer_timing_payload.hh"
#include "rebuffer_carrier.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "pybind11/numpy.h"

namespace py = pybind11;
namespace rebuffer {
namespace {

template <typename T>
py::array_t<T> array(const std::vector<T>& values)
{
  // Explicit strides are required with the vendored pybind11 and NumPy 2.
  py::array_t<T> out({static_cast<py::ssize_t>(values.size())},
                     {static_cast<py::ssize_t>(sizeof(T))});
  std::copy(values.begin(), values.end(), out.mutable_data());
  return out;
}

}  // namespace

py::tuple pinPairArcMapping(const py::object& db)
{
  const auto starts = carrierArray<int>(db, "pin_pred_start");
  const auto sources = carrierArray<int>(db, "pin_pred_pin");
  const auto arcs = carrierArray<int>(db, "pin_pred_arc_id");
  std::map<std::pair<int, int>, int> positions;
  std::vector<std::array<int32_t, 2>> pairs;
  std::vector<std::vector<int32_t>> rows;
  for (int sink = 0; sink + 1 < static_cast<int>(starts.size()); ++sink) {
    for (int index = starts[sink]; index < starts[sink + 1]; ++index) {
      if (index < 0 || index >= static_cast<int>(sources.size()) || index >= static_cast<int>(arcs.size())) {
        throw py::value_error("timing predecessor CSR does not align");
      }
      if (sources[index] < 0 || arcs[index] < 0) continue;
      const auto inserted = positions.emplace(std::make_pair(sources[index], sink), rows.size());
      if (inserted.second) {
        pairs.push_back({sources[index], sink});
        rows.emplace_back();
      }
      rows[inserted.first->second].push_back(arcs[index]);
    }
  }
  if (pairs.empty()) return py::make_tuple(py::none(), py::none(), py::none());
  std::vector<int32_t> offsets{0}, indices;
  for (auto& row : rows) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    indices.insert(indices.end(), row.begin(), row.end());
    offsets.push_back(indices.size());
  }
  py::array_t<int32_t> keys({static_cast<py::ssize_t>(pairs.size()), py::ssize_t{2}},
                            {py::ssize_t{2 * sizeof(int32_t)}, py::ssize_t{sizeof(int32_t)}});
  for (int row = 0; row < static_cast<int>(pairs.size()); ++row) {
    std::copy(pairs[row].begin(), pairs[row].end(), keys.mutable_data() + 2 * row);
  }
  return py::make_tuple(keys, array(offsets), array(indices));
}

py::dict timingPinCapacitance(const py::object& db)
{
  const auto owner = carrierArray<int>(db, "pin2node_map");
  const auto mains = carrierArray<int>(db, "inst_main_id");
  const auto cell_offsets = carrierArray<int>(db, "inst_libcell_offset");
  const auto main_start = carrierArray<int>(db, "main_id_2_cell_id_start");
  const auto pin_start = carrierArray<int>(db, "cell_id_2_libpin_id_start");
  const auto pin_offsets = carrierArray<int>(db, "pin_2_libpin_offset");
  const auto endpoints = carrierArray<int>(db, "end_points");
  std::vector<int> library_pin(owner.size(), -1);
  for (int pin = 0; pin < static_cast<int>(owner.size()); ++pin) {
    const int node = owner[pin], main = mains.at(node);
    if (main < 0) continue;
    const int cell = main_start.at(main) + cell_offsets.at(node);
    library_pin[pin] = pin_start.at(cell) + pin_offsets.at(pin);
  }
  py::dict out;
  for (const auto& fields : std::array<std::array<const char*, 3>, 5>{{
           {"caps", "flat_lib_pin_cap", "outcaps"},
           {"rcaps", "flat_lib_pin_rcap", "out_rcaps"},
           {"fcaps", "flat_lib_pin_fcap", "out_fcaps"},
           {"max_cap", "flat_lib_pin_cap_limit", ""},
           {"max_slew", "flat_lib_pin_slew_limit", ""}}}) {
    const auto values = fields[2][0] || py::hasattr(db, fields[1])
        ? carrierArray<double>(db, fields[1]) : std::vector<double>{};
    std::vector<double> gathered(owner.size(), 0.0);
    for (int pin = 0; pin < static_cast<int>(owner.size()); ++pin) {
      const int lib = library_pin[pin];
      if (lib >= 0 && lib < static_cast<int>(values.size())) gathered[pin] = values[lib];
    }
    if (fields[2][0]) {
      const auto external = carrierArray<double>(db, py::hasattr(db, fields[2]) ? fields[2] : "outcaps");
      for (int row = 0; row < static_cast<int>(endpoints.size()); ++row) {
        const int pin = endpoints[row];
        if (mains.at(owner.at(pin)) < 0) gathered[pin] += external.at(row);
      }
    }
    out[fields[0]] = array(gathered);
  }
  return out;
}

}  // namespace rebuffer
