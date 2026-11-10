// SPDX-License-Identifier: BSD-3-Clause
// Read declaration metadata only; timing tables remain owned by Python TP.
#include "rebuffer_liberty.hh"


#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <zlib.h>
#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {
namespace {

std::string readFile(const std::string& path)
{
  gzFile file = gzopen(path.c_str(), "rb");
  if (!file) throw std::invalid_argument("cannot read repair library source: " + path);
  std::string result;
  char buffer[65536];
  int count;
  while ((count = gzread(file, buffer, sizeof(buffer))) > 0) result.append(buffer, count);
  int error;
  const std::string message = gzerror(file, &error);
  gzclose(file);
  if (count < 0) throw std::invalid_argument("repair library source: " + path + ": " + message);
  return result;
}

struct Token
{
  std::string_view value;
  bool quoted = false;
  bool eof() const { return value.empty() && !quoted; }
};

class Scanner
{
 public:
  Scanner(std::string text, std::string path) : text_(std::move(text)), path_(std::move(path)) {}
  Token next()
  {
    if (!has_saved_) return scan();
    has_saved_ = false;
    return saved_;
  }
  Token peek()
  {
    if (!has_saved_) { saved_ = scan(); has_saved_ = true; }
    return saved_;
  }
 private:
  Token scan()
  {
    while (position_ < text_.size()) {
      const char c = text_[position_];
      if (std::isspace(static_cast<unsigned char>(c))) { ++position_; continue; }
      if (c == '#' || (c == '/' && position_ + 1 < text_.size() && text_[position_ + 1] == '/')) {
        while (position_ < text_.size() && text_[position_] != '\n') ++position_;
        continue;
      }
      if (c == '/' && position_ + 1 < text_.size() && text_[position_ + 1] == '*') {
        const auto end = text_.find("*/", position_ + 2);
        if (end == std::string::npos) fail("unterminated comment");
        position_ = end + 2;
        continue;
      }
      if (c == '\\' && position_ + 1 < text_.size() && text_[position_ + 1] == '\n') {
        position_ += 2;
        continue;
      }
      const auto begin = position_++;
      if (c == '"') {
        while (position_ < text_.size()) {
          const char value = text_[position_++];
          if (value == '\\' && position_ < text_.size()) { ++position_; continue; }
          if (value == '"') return {std::string_view(text_).substr(begin + 1, position_ - begin - 2), true};
        }
        fail("unterminated quoted string");
      }
      if (std::string_view("(){}:;,=").find(c) != std::string_view::npos) {
        return {std::string_view(text_).substr(begin, 1), false};
      }
      while (position_ < text_.size()
             && !std::isspace(static_cast<unsigned char>(text_[position_]))
             && std::string_view("(){}:;,=\"").find(text_[position_]) == std::string_view::npos) ++position_;
      return {std::string_view(text_).substr(begin, position_ - begin), false};
    }
    return {};
  }
 public:
  [[noreturn]] void fail(const char* reason) const
  {
    const auto line = 1 + std::count(text_.begin(), text_.begin() + position_, '\n');
    throw std::invalid_argument("repair library declaration " + path_ + ":" + std::to_string(line) + ": " + reason);
  }
  void skipGroup()
  {
    int depth = 1;
    while (depth) {
      const auto token = next();
      if (token.eof()) fail("unterminated group");
      if (!token.quoted) {
        if (token.value == "{") ++depth;
        else if (token.value == "}") --depth;
      }
    }
  }
 private:
  std::string text_, path_;
  std::size_t position_ = 0;
  Token saved_;
  bool has_saved_ = false;
};

struct Member
{
  std::string name, argument, value;
  bool group = false, end = false, eof = false;
};

Member member(Scanner& scanner)
{
  const auto name = scanner.next();
  if (name.eof() || (!name.quoted && name.value == "}")) return {{}, {}, {}, false, true, name.eof()};
  Member result;
  result.name = name.value;
  const auto kind = scanner.next();
  if (kind.value == ":" || kind.value == "=") {
    for (;;) {
      const auto token = scanner.next();
      if (token.eof()) scanner.fail(("unterminated attribute " + result.name).c_str());
      if (!token.quoted && token.value == ";") return result;
      result.value += token.value;
    }
  }
  if (kind.value != "(") scanner.fail("expected attribute or group");
  int depth = 1;
  while (depth) {
    const auto token = scanner.next();
    if (token.eof()) scanner.fail("unterminated arguments");
    if (!token.quoted && token.value == "(") ++depth;
    else if (!token.quoted && token.value == ")") --depth;
    if (depth) result.argument += token.value;
  }
  const auto end = scanner.next();
  result.group = end.value == "{";
  if (!result.group && end.value != ";") scanner.fail("expected group body");
  return result;
}

struct Port
{
  std::string name, direction, function, tristate;
};

Port readPort(Scanner& scanner, std::string name)
{
  Port port;
  port.name = std::move(name);
  for (;;) {
    const auto entry = member(scanner);
    if (entry.end) {
      if (entry.eof) scanner.fail("unterminated pin group");
      return port;
    }
    if (entry.group) scanner.skipGroup();
    else if (entry.name == "direction") port.direction = entry.value;
    else if (entry.name == "function") port.function = entry.value;
    else if (entry.name == "three_state") port.tristate = entry.value;
  }
}

std::string portExpression(std::string value)
{
  value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); }), value.end());
  while (value.size() >= 2 && value.front() == '(' && value.back() == ')') {
    int depth = 0;
    bool outer = true;
    for (std::size_t i = 0; i + 1 < value.size(); ++i) {
      if (value[i] == '(') ++depth;
      else if (value[i] == ')') --depth;
      if (depth == 0) { outer = false; break; }
    }
    if (!outer) break;
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

double number(const std::string& value)
{
  std::size_t used;
  const double result = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(result)) throw std::invalid_argument("invalid library metadata number");
  return result;
}

void readCell(Scanner& scanner, LibraryCellMetadata& cell, double power_scale)
{
  std::vector<Port> ports;
  bool sequential = false;
  double state_leakage = 0;
  int leakage_states = 0;
  for (;;) {
    const auto entry = member(scanner);
    if (entry.end) {
      if (entry.eof) scanner.fail("unterminated cell group");
      break;
    }
    if (entry.group) {
      if (entry.name == "pin") ports.push_back(readPort(scanner, entry.argument));
      else if (entry.name == "leakage_power") {
        for (;;) {
          const auto attribute = member(scanner);
          if (attribute.end) {
            if (attribute.eof) scanner.fail("unterminated leakage group");
            break;
          }
          if (attribute.group) scanner.skipGroup();
          else if (attribute.name == "value") { state_leakage += number(attribute.value); ++leakage_states; }
        }
      } else {
        if (entry.name == "ff" || entry.name == "latch" || entry.name == "statetable") sequential = true;
        // Bus/bundle ports cannot belong to the two-scalar-port buffer domain.
        if (entry.name == "bus" || entry.name == "bundle") sequential = true;
        scanner.skipGroup();
      }
    } else if (entry.name == "dont_use") cell.dont_use = entry.value == "true";
    else if (entry.name == "cell_footprint") cell.footprint = entry.value;
    else if (entry.name == "user_function_class") cell.function_class = entry.value;
    else if (entry.name == "cell_leakage_power") { cell.has_leakage = true; cell.leakage = number(entry.value); }
  }
  if (!cell.has_leakage && leakage_states) {
    cell.has_leakage = true;
    cell.leakage = state_leakage / leakage_states;
  }
  cell.leakage *= power_scale;
  if (sequential || ports.size() != 2) return;
  std::sort(ports.begin(), ports.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
  const auto input = std::find_if(ports.begin(), ports.end(), [](const auto& p) { return p.direction == "input"; });
  const auto output = std::find_if(ports.begin(), ports.end(), [](const auto& p) { return p.direction == "output"; });
  if (input == ports.end() || output == ports.end() || portExpression(output->function) != input->name) return;
  // In this domain OpenSTA equivalence is the same port names/directions,
  // identity output function and matching optional tristate functions.
  for (const auto& port : ports) {
    cell.buffer_signature += port.name + ":" + port.direction + ":"
        + portExpression(port.function) + ":" + portExpression(port.tristate) + ";";
  }
}

double powerScale(const std::string& unit)
{
  static const std::map<std::string, double> scales{{"1W", 1}, {"1mW", 1e-3}, {"1uW", 1e-6}, {"1nW", 1e-9}, {"1pW", 1e-12}};
  const auto found = scales.find(unit);
  if (found == scales.end()) throw std::invalid_argument("unsupported library leakage_power_unit: " + unit);
  return found->second;
}

void readLiberty(const std::string& path, std::map<std::string, LibraryCellMetadata*>& cells, int& order)
{
  Scanner scanner(readFile(path), path);
  for (;;) {
    const auto library = member(scanner);
    if (library.end) break;
    if (!library.group) continue;
    if (library.name != "library") { scanner.skipGroup(); continue; }
    double power_scale = 1;
    for (;;) {
      const auto entry = member(scanner);
      if (entry.end) {
        if (entry.eof) scanner.fail("unterminated library group");
        break;
      }
      if (!entry.group) {
        if (entry.name == "leakage_power_unit") power_scale = powerScale(entry.value);
        continue;
      }
      const auto found = cells.find(entry.argument);
      if (entry.name != "cell") { scanner.skipGroup(); continue; }
      const int source_order = order++;
      if (found == cells.end()) { scanner.skipGroup(); continue; }
      auto& cell = *found->second;
      if (cell.source_order >= 0) throw std::invalid_argument("ambiguous linked Liberty cell metadata: " + cell.name);
      cell.library = library.argument;
      cell.source_order = source_order;
      readCell(scanner, cell, power_scale);
    }
  }
}

void readLef(const std::string& path, std::map<std::string, LibraryCellMetadata*>& cells)
{
  Scanner scanner(readFile(path), path);
  for (;;) {
    const auto token = scanner.next();
    if (token.eof()) break;
    if (token.quoted || token.value != "MACRO") continue;
    const std::string name(scanner.next().value);
    const auto found = cells.find(name);
    LibraryCellMetadata* cell = found == cells.end() ? nullptr : found->second;
    if (cell && cell->has_master) throw std::invalid_argument("duplicate physical master metadata: " + name);
    if (cell) cell->has_master = true;
    bool in_macro = true;
    while (in_macro) {
      const auto field = scanner.next();
      if (field.eof()) scanner.fail("unterminated LEF macro");
      if (field.quoted) continue;
      if (field.value == "PIN") {
        const std::string pin(scanner.next().value);
        for (;;) {
          const auto item = scanner.next();
          if (item.eof()) scanner.fail("unterminated LEF pin");
          if (!item.quoted && item.value == "END") {
            if (scanner.peek().value == pin) { scanner.next(); break; }
          }
        }
      } else if (field.value == "END") {
        if (scanner.peek().value == name) { scanner.next(); in_macro = false; }
      } else if (field.value == "CLASS") {
        const auto kind = scanner.next();
        if (cell) cell->master_is_core = kind.value == "CORE";
      } else if (field.value == "SITE") {
        const auto site = scanner.next();
        if (cell) cell->site = site.value;
      }
    }
  }
}

}  // namespace

std::shared_ptr<RepairLibraryMetadata> readRepairLibraryMetadata(
    const std::vector<std::string>& liberty_paths,
    const std::vector<std::string>& lef_paths,
    const std::vector<std::string>& cell_names)
{
  if (liberty_paths.empty() || lef_paths.empty()) throw std::invalid_argument("repair cell selection requires Liberty and LEF sources");
  auto metadata = std::make_shared<RepairLibraryMetadata>();
  metadata->liberty_paths = liberty_paths;
  metadata->lef_paths = lef_paths;
  metadata->cells.resize(cell_names.size());
  std::map<std::string, LibraryCellMetadata*> cells;
  for (std::size_t i = 0; i < cell_names.size(); ++i) {
    metadata->cells[i].name = cell_names[i];
    if (!cells.emplace(cell_names[i], &metadata->cells[i]).second) throw std::invalid_argument("duplicate PyDB library cell name");
  }
  int order = 0;
  for (const auto& path : liberty_paths) readLiberty(path, cells, order);
  for (const auto& path : lef_paths) readLef(path, cells);
  for (const auto& cell : metadata->cells) {
    if (cell.source_order < 0) throw std::invalid_argument("PyDB cell missing from Liberty sources: " + cell.name);
  }
  return metadata;
}

py::dict RepairLibraryMetadata::summary() const
{
  py::dict result;
  result["status"] = "ok";
  result["cell_count"] = cells.size();
  result["buffer_function_count"] = std::count_if(cells.begin(), cells.end(), [](const auto& c) { return !c.buffer_signature.empty(); });
  result["liberty_paths"] = liberty_paths;
  result["lef_paths"] = lef_paths;
  return result;
}

py::list RepairLibraryMetadata::records() const
{
  py::list result;
  for (const auto& cell : cells) {
    py::dict row;
    row["name"] = cell.name; row["library"] = cell.library;
    row["source_order"] = cell.source_order;
    row["dont_use"] = cell.dont_use; row["has_master"] = cell.has_master;
    row["master_is_core"] = cell.master_is_core; row["site"] = cell.site;
    row["footprint"] = cell.footprint; row["function_class"] = cell.function_class;
    row["buffer_signature"] = cell.buffer_signature;
    row["has_leakage"] = cell.has_leakage; row["leakage"] = cell.leakage;
    result.append(std::move(row));
  }
  return result;
}

void registerRepairLibraryMetadata(py::module_& module)
{
  py::class_<RepairLibraryMetadata, std::shared_ptr<RepairLibraryMetadata>>(module, "RepairLibraryMetadata")
      .def("summary", &RepairLibraryMetadata::summary)
      .def("records", &RepairLibraryMetadata::records);
  module.def("read_repair_library_metadata", &readRepairLibraryMetadata);
}

}  // namespace rebuffer

// --- rebuffer_cell_selection.cc ---
// OpenROAD target-load and driver-slew selection for scalar buffer cells.
#include "rebuffer_repair.hh"


#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>


namespace py = pybind11;
namespace rebuffer {

void initializeBufferCellSelection(LibertyView& liberty)
{
  if (!liberty.cell_metadata) return;
  const auto& records = liberty.cell_metadata->cells;
  std::map<std::string, std::vector<int>> equivalent;
  if (records.size() != liberty.cells.size()) throw std::invalid_argument("repair library metadata cell count differs from Liberty view");
  for (std::size_t i = 0; i < records.size(); ++i) {
    if (records[i].name != liberty.cells[i].name) throw std::invalid_argument("repair library metadata cell identity differs from Liberty view");
    if (!records[i].buffer_signature.empty() && !records[i].dont_use) {
      equivalent[records[i].buffer_signature].push_back(i);
    }
  }
  for (auto& [_, order] : equivalent) {
    std::sort(order.begin(), order.end(), [&](int a, int b) { return records[a].source_order < records[b].source_order; });
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return liberty.cells[a].drive_resistance > liberty.cells[b].drive_resistance;
    });
    liberty.buffer_equivalent_order.insert(liberty.buffer_equivalent_order.end(), order.begin(), order.end());
  }
}

std::vector<int> swappableBufferCells(const LibertyView& liberty,
                                      int source,
                                      double sizing_area_limit,
                                      double sizing_leakage_limit)
{
  if (!liberty.cell_metadata) throw std::invalid_argument("repair cell selection requires source Library/LEF metadata");
  const auto& metadata = liberty.cell_metadata->cells;
  const auto& current = metadata.at(source);
  if (current.buffer_signature.empty()) throw std::invalid_argument("target-cell source is outside the scalar buffer domain");
  if (!current.has_master || !current.master_is_core) return {};
  std::vector<int> equivalent;
  for (int candidate : liberty.buffer_equivalent_order) {
    if (metadata[candidate].buffer_signature == current.buffer_signature) equivalent.push_back(candidate);
  }
  // OpenSTA has no equivalence-set entry for a singleton or a dont-use
  // source. Resizer then returns the source without applying set filters.
  if (current.dont_use || equivalent.size() < 2) return {source};
  std::vector<int> result;
  const double source_area = liberty.cells[source].area;
  for (int candidate : equivalent) {
    const auto& target = metadata[candidate];
    if (!target.has_master) continue;
    if (sizing_area_limit > 0.0 && source_area != 0
        && liberty.cells[candidate].area / source_area > sizing_area_limit) {
      continue;
    }
    if (sizing_leakage_limit > 0.0 && current.has_leakage && target.has_leakage
        && target.leakage / current.leakage > sizing_leakage_limit) {
      continue;
    }
    if (!current.function_class.empty() && current.function_class != target.function_class) continue;
    result.push_back(candidate);
  }
  return result;
}

int findBufferTargetCell(const LibertyView& liberty,
                         const CharacterizationResult& characterization,
                         int source,
                         double load,
                         double sizing_area_limit,
                         double sizing_leakage_limit)
{
  if (!std::isfinite(load) || load < 0) throw std::invalid_argument("invalid target-cell load");
  const auto& cells = liberty.cells;
  int best = source;
  double distance = std::abs(load - cellTargetLoad(cells.at(source), characterization.target_slew));
  double delay = cellTargetSlewDelay(cells[source], characterization.target_slew, load);
  for (int candidate : swappableBufferCells(
           liberty, source, sizing_area_limit, sizing_leakage_limit)) {
    const double candidate_distance = std::abs(load - cellTargetLoad(cells[candidate], characterization.target_slew));
    const double candidate_delay = cellTargetSlewDelay(cells[candidate], characterization.target_slew, load);
    if ((candidate_delay < delay && candidate_distance < distance * 1.1)
        || (candidate_distance < distance && candidate_delay < delay * 1.1)) {
      best = candidate;
      distance = candidate_distance;
      delay = candidate_delay;
    }
  }
  return best;
}

int findBufferSlewRepairCell(const LibertyView& liberty,
                             int source,
                             double input_slew_rise,
                             double input_slew_fall,
                             double load,
                             double slew_margin,
                             double sizing_area_limit,
                             double sizing_leakage_limit)
{
  std::vector<std::pair<double, int>> sizes;
  for (int candidate : swappableBufferCells(
           liberty, source, sizing_area_limit, sizing_leakage_limit)) {
    const auto& cell = liberty.cells[candidate];
    const double slew = rebufferCellOutputSlew(
        cell, input_slew_rise, input_slew_fall, load);
    const double violation = cell.max_slew > 0.0
        ? std::max(0.0, slew - cell.max_slew * (1.0 - slew_margin))
        : 0.0;
    sizes.emplace_back(violation, candidate);
  }
  // RepairDesign sorts the ordered swappable sequence by violation, or by
  // area when both cells meet slew. Equal scores have no extra id tie-break.
  std::sort(sizes.begin(), sizes.end(), [&](const auto& a, const auto& b) {
    if (a.first == 0.0 && b.first == 0.0) {
      return liberty.cells[a.second].area < liberty.cells[b.second].area;
    }
    return a.first < b.first;
  });
  return sizes.empty() ? source : sizes.front().second;
}

void registerCellSelectionBindings(py::module_& module)
{
  module.def("validate_buffer_resize_pairs", [](const py::dict& payload,
                                                const std::vector<int>& sources,
                                                const std::vector<int>& targets) {
    if (sources.size() != targets.size()) {
      throw std::invalid_argument("buffer resize source/target counts differ");
    }
    const auto liberty = rebuffer_python::libertyFromDict(payload);
    for (std::size_t i = 0; i < sources.size(); ++i) {
      if (sources[i] < 0 || sources[i] >= static_cast<int>(liberty.cells.size())
          || targets[i] < 0 || targets[i] >= static_cast<int>(liberty.cells.size())) {
        throw std::invalid_argument("buffer resize cell index is outside Liberty");
      }
      const auto candidates = swappableBufferCells(liberty, sources[i], 0.0, 0.0);
      if (std::find(candidates.begin(), candidates.end(), targets[i]) == candidates.end()) {
        throw std::invalid_argument("target is not a swappable buffer: "
            + liberty.cells[sources[i]].name + " -> " + liberty.cells[targets[i]].name);
      }
    }
  });
  module.def("repair_buffer_cell_candidates", [](const py::dict& payload, int source) {
    const auto liberty = rebuffer_python::libertyFromDict(payload);
    return swappableBufferCells(liberty, source);
  });
  module.def("repair_buffer_target_cell", [](const py::dict& payload, const py::dict& characterization,
                                              int source, double load) {
    const auto liberty = rebuffer_python::libertyFromDict(payload);
    return findBufferTargetCell(liberty, rebuffer_python::characterizationFromDict(characterization), source, load);
  });
}

}  // namespace rebuffer
