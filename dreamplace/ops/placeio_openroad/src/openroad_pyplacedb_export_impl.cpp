#include "openroad_place_io_impl_bridge.h"
#include "openroad_sizing_cell_classification.h"
#include <boost/polygon/polygon.hpp>

namespace dreamplace {
namespace placeio_openroad {
namespace impl {

py::list toPyList(const std::vector<int>& values)
{
  py::list result;
  for (const int value : values) {
    result.append(value);
  }
  return result;
}

void appendUniqueInstGroup(py::list& groups, const std::vector<int>& inst_ids)
{
  std::vector<int> clean;
  clean.reserve(inst_ids.size());
  for (const int inst_id : inst_ids) {
    if (inst_id >= 0) {
      clean.push_back(inst_id);
    }
  }
  std::sort(clean.begin(), clean.end());
  clean.erase(std::unique(clean.begin(), clean.end()), clean.end());
  if (clean.size() < 2) {
    return;
  }
  groups.append(toPyList(clean));
}

void appendBox(py::list& boxes, int xl, int yl, int xh, int yh)
{
  py::list box;
  box.append(xl);
  box.append(yl);
  box.append(xh);
  box.append(yh);
  boxes.append(box);
}

double finiteOrDefault(float value, double fallback = 0.0)
{
  return std::isfinite(value) ? static_cast<double>(value) : fallback;
}

double nonNegativeOrDefault(float value, double fallback = 0.0)
{
  if (!std::isfinite(value) || value < 0.0f) {
    return fallback;
  }
  return static_cast<double>(value);
}

void appendTimingArc(py::list& flat_arcs, int driver_pin_id, int sink_pin_id)
{
  py::list arc;
  arc.append(driver_pin_id);
  arc.append(sink_pin_id);
  flat_arcs.append(arc);
}

void appendConstraintArc(py::list& arcs,
                         int from_pin_id,
                         int to_pin_id,
                         int lib_cell_id,
                         int lib_arc_id,
                         int timing_sense)
{
  py::list arc;
  arc.append(from_pin_id);
  arc.append(to_pin_id);
  arc.append(lib_cell_id);
  arc.append(lib_arc_id);
  arc.append(timing_sense);
  arcs.append(arc);
}

void appendTimingCheckArc(py::list& arcs,
                          int from_pin_id,
                          int to_pin_id,
                          int lib_cell_id,
                          int lib_arc_id,
                          int timing_sense,
                          int timing_type,
                          int check_class,
                          int lib_arc_offset)
{
  py::list arc;
  arc.append(from_pin_id);
  arc.append(to_pin_id);
  arc.append(lib_cell_id);
  arc.append(lib_arc_id);
  arc.append(timing_sense);
  arc.append(timing_type);
  arc.append(check_class);
  arc.append(lib_arc_offset);
  arcs.append(arc);
}

bool isPythonSetupConstraintRole(const sta::TimingRole* role)
{
  if (role == nullptr) {
    return false;
  }
  return role == sta::TimingRole::setup()
         || role == sta::TimingRole::latchSetup();
}

int timingCheckClassToInt(const sta::TimingRole* role)
{
  if (role == nullptr) {
    return 0;
  }
  if (role == sta::TimingRole::setup() || role == sta::TimingRole::latchSetup()) {
    return 1;
  }
  if (role == sta::TimingRole::hold()) {
    return 2;
  }
  if (role == sta::TimingRole::recovery()) {
    return 3;
  }
  if (role == sta::TimingRole::removal()) {
    return 4;
  }
  return 0;
}

void appendInstArc(py::list& arcs,
                   int from_pin_id,
                   int to_pin_id,
                   int lib_cell_id,
                   int lib_arc_id,
                   int timing_sense,
                   int timing_type,
                   int lib_arc_offset,
                   int inst_id)
{
  py::list arc;
  arc.append(from_pin_id);
  arc.append(to_pin_id);
  arc.append(lib_cell_id);
  arc.append(lib_arc_id);
  arc.append(timing_sense);
  arc.append(timing_type);
  arc.append(lib_arc_offset);
  arc.append(inst_id);
  arcs.append(arc);
}

int timingSenseToInt(sta::TimingSense sense)
{
  switch (sense) {
    case sta::TimingSense::positive_unate:
      return 1;
    case sta::TimingSense::negative_unate:
      return -1;
    case sta::TimingSense::non_unate:
      return 0;
    default:
      return 1;
  }
}

int timingTypeToInt(sta::TimingType type);

struct TempTimingArcSet
{
  int lib_arc_idx;
  int lib_arc_offset;
  sta::TimingArcSet* arc_set;
  sta::TimingArc* rise_arc;
  sta::TimingArc* fall_arc;
  sta::TimingArc* representative_arc;

  int senseToInt() const
  {
    return timingSenseToInt(
        arc_set == nullptr ? sta::TimingSense::positive_unate : arc_set->sense());
  }

  int typeToInt() const
  {
    if (arc_set == nullptr) {
      return 0;
    }
    const auto* edge = arc_set->isRisingFallingEdge();
    if (edge == sta::RiseFall::rise()) {
      return 1;
    }
    if (edge == sta::RiseFall::fall()) {
      return -1;
    }
    return 0;
  }
};

TempTimingArcSet buildTempTimingArcSet(int lib_arc_idx,
                                       int lib_arc_offset,
                                       sta::TimingArcSet* arc_set)
{
  sta::TimingArc* rise_arc = arc_set == nullptr ? nullptr : arc_set->arcTo(sta::RiseFall::rise());
  sta::TimingArc* fall_arc = arc_set == nullptr ? nullptr : arc_set->arcTo(sta::RiseFall::fall());
  sta::TimingArc* representative_arc = rise_arc != nullptr ? rise_arc : fall_arc;
  return TempTimingArcSet{
      lib_arc_idx,
      lib_arc_offset,
      arc_set,
      rise_arc,
      fall_arc,
      representative_arc,
  };
}

struct InstArcRecord
{
  int from_pin_id;
  int to_pin_id;
  int lib_cell_id;
  int lib_arc_id;
  int timing_sense;
  int timing_type;
  int lib_arc_offset;
  int inst_id;
  bool source_is_clock{false};
};

struct TimingInstRecord
{
  odb::dbInst* inst{nullptr};
  int node_id{-1};
  bool is_sequential_timing{false};
};

std::vector<TimingInstRecord> buildTimingInstView(
    OpenRoadPlaceIOBridgeImpl& raw_db,
    const std::vector<odb::dbInst*>& movable_insts,
    const std::vector<odb::dbInst*>& fixed_insts)
{
  std::vector<TimingInstRecord> timing_insts;
  if (!raw_db.hasTimingInputs() || raw_db.design() == nullptr) {
    return timing_insts;
  }

  ord::Timing timing(raw_db.design());
  auto* sta = timing.getSta();
  auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
  if (network == nullptr) {
    return timing_insts;
  }

  timing_insts.reserve(movable_insts.size() + fixed_insts.size());
  auto append_timing_inst = [&](odb::dbInst* inst, int node_id) {
    auto* master = inst == nullptr ? nullptr : inst->getMaster();
    auto* cell = master == nullptr ? nullptr : network->dbToSta(master);
    auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
    // Physical-only fixed cells (filler/tap/decap) have no data timing arcs.
    // Keep them in the physical node view, but out of the timing graph.
    if (liberty_cell == nullptr || liberty_cell->timingArcSets().empty()) {
      return;
    }
    timing_insts.push_back({
        inst,
        node_id,
        liberty_cell->hasSequentials() || liberty_cell->hasInferedRegTimingArcs(),
    });
  };

  for (int node_id = 0; node_id < static_cast<int>(movable_insts.size()); ++node_id) {
    append_timing_inst(movable_insts[node_id], node_id);
  }
  const int fixed_node_offset = static_cast<int>(movable_insts.size());
  for (int fixed_id = 0; fixed_id < static_cast<int>(fixed_insts.size()); ++fixed_id) {
    append_timing_inst(fixed_insts[fixed_id], fixed_node_offset + fixed_id);
  }
  return timing_insts;
}

struct InstMTermKey
{
  odb::dbInst* inst{nullptr};
  odb::dbMTerm* mterm{nullptr};

  bool operator==(const InstMTermKey& other) const
  {
    return inst == other.inst && mterm == other.mterm;
  }
};

struct InstMTermKeyHash
{
  std::size_t operator()(const InstMTermKey& key) const
  {
    const auto inst_hash = std::hash<odb::dbInst*>{}(key.inst);
    const auto mterm_hash = std::hash<odb::dbMTerm*>{}(key.mterm);
    return inst_hash ^ (mterm_hash + 0x9e3779b97f4a7c15ULL + (inst_hash << 6) + (inst_hash >> 2));
  }
};

void appendLevelizedInstArc(PyPlaceDB& pydb, const InstArcRecord& record)
{
  appendInstArc(pydb.flat_inst_arcs_by_level,
                record.from_pin_id,
                record.to_pin_id,
                record.lib_cell_id,
                record.lib_arc_id,
                record.timing_sense,
                record.timing_type,
                record.lib_arc_offset,
                record.inst_id);
  pydb.arc_src_pin.append(record.from_pin_id);
  pydb.arc_dst_pin.append(record.to_pin_id);
  pydb.arc_inst_id.append(record.inst_id);
  pydb.arc_libcell_id.append(record.lib_cell_id);
  pydb.arc_libarc_id.append(record.lib_arc_id);
  pydb.arc_sense.append(record.timing_sense);
  pydb.arc_type.append(record.timing_type);
  pydb.arc_offset.append(record.lib_arc_offset);
}

std::vector<std::vector<InstArcRecord>> levelizeInstArcsForPython(
    const std::vector<InstArcRecord>& clk2q_arc_records,
    const std::vector<InstArcRecord>& comb_arc_records,
    const std::vector<NetRecord>& nets,
    int num_pins)
{
  std::vector<std::vector<InstArcRecord>> arc_records_by_level;
  arc_records_by_level.push_back(clk2q_arc_records);
  if (comb_arc_records.empty()) {
    return arc_records_by_level;
  }

  std::vector<std::vector<int>> driver_to_net_sinks(static_cast<std::size_t>(std::max(num_pins, 0)));
  for (const auto& net : nets) {
    if (net.driver_pin_id < 0 || net.driver_pin_id >= num_pins) {
      continue;
    }
    auto& sinks = driver_to_net_sinks[static_cast<std::size_t>(net.driver_pin_id)];
    for (const int pin_id : net.pin_ids) {
      if (pin_id >= 0 && pin_id < num_pins && pin_id != net.driver_pin_id) {
        sinks.push_back(pin_id);
      }
    }
  }

  std::unordered_map<int, int> to_pin_to_group;
  std::vector<std::vector<int>> group_arc_ids;
  std::vector<int> group_output_pin;
  group_arc_ids.reserve(comb_arc_records.size());
  for (int arc_id = 0; arc_id < static_cast<int>(comb_arc_records.size()); ++arc_id) {
    const int to_pin = comb_arc_records[arc_id].to_pin_id;
    auto group_it = to_pin_to_group.find(to_pin);
    if (group_it == to_pin_to_group.end()) {
      const int group_id = static_cast<int>(group_arc_ids.size());
      group_it = to_pin_to_group.emplace(to_pin, group_id).first;
      group_arc_ids.push_back({});
      group_output_pin.push_back(to_pin);
    }
    group_arc_ids[group_it->second].push_back(arc_id);
  }

  std::vector<std::vector<int>> from_pin_to_groups(static_cast<std::size_t>(std::max(num_pins, 0)));
  std::unordered_set<long long> from_pin_group_seen;
  auto encode_pair = [](int lhs, int rhs) -> long long {
    return (static_cast<long long>(lhs) << 32) ^ static_cast<unsigned int>(rhs);
  };
  for (int group_id = 0; group_id < static_cast<int>(group_arc_ids.size()); ++group_id) {
    for (const int arc_id : group_arc_ids[group_id]) {
      const int from_pin = comb_arc_records[arc_id].from_pin_id;
      if (from_pin < 0 || from_pin >= num_pins) {
        continue;
      }
      const long long key = encode_pair(from_pin, group_id);
      if (from_pin_group_seen.insert(key).second) {
        from_pin_to_groups[static_cast<std::size_t>(from_pin)].push_back(group_id);
      }
    }
  }

  std::vector<std::vector<int>> group_succs(group_arc_ids.size());
  std::vector<int> group_indegree(group_arc_ids.size(), 0);
  std::unordered_set<long long> dependency_seen;
  for (int group_id = 0; group_id < static_cast<int>(group_arc_ids.size()); ++group_id) {
    const int output_pin = group_output_pin[group_id];
    if (output_pin < 0 || output_pin >= num_pins) {
      continue;
    }
    for (const int sink_pin : driver_to_net_sinks[static_cast<std::size_t>(output_pin)]) {
      if (sink_pin < 0 || sink_pin >= num_pins) {
        continue;
      }
      for (const int succ_group : from_pin_to_groups[static_cast<std::size_t>(sink_pin)]) {
        if (succ_group == group_id) {
          continue;
        }
        const long long key = encode_pair(group_id, succ_group);
        if (dependency_seen.insert(key).second) {
          group_succs[static_cast<std::size_t>(group_id)].push_back(succ_group);
          group_indegree[static_cast<std::size_t>(succ_group)] += 1;
        }
      }
    }
  }

  std::vector<int> current_groups;
  current_groups.reserve(group_arc_ids.size());
  for (int group_id = 0; group_id < static_cast<int>(group_indegree.size()); ++group_id) {
    if (group_indegree[group_id] == 0) {
      current_groups.push_back(group_id);
    }
  }

  std::vector<char> visited(group_arc_ids.size(), 0);
  int visited_count = 0;
  while (!current_groups.empty()) {
    std::vector<InstArcRecord> level_records;
    std::vector<int> next_groups;
    for (const int group_id : current_groups) {
      if (group_id < 0 || group_id >= static_cast<int>(group_arc_ids.size()) || visited[group_id]) {
        continue;
      }
      visited[group_id] = 1;
      visited_count += 1;
      for (const int arc_id : group_arc_ids[static_cast<std::size_t>(group_id)]) {
        level_records.push_back(comb_arc_records[arc_id]);
      }
      for (const int succ_group : group_succs[static_cast<std::size_t>(group_id)]) {
        if (succ_group < 0 || succ_group >= static_cast<int>(group_indegree.size())) {
          continue;
        }
        group_indegree[static_cast<std::size_t>(succ_group)] -= 1;
        if (group_indegree[static_cast<std::size_t>(succ_group)] == 0) {
          next_groups.push_back(succ_group);
        }
      }
    }
    if (!level_records.empty()) {
      arc_records_by_level.push_back(std::move(level_records));
    }
    std::sort(next_groups.begin(), next_groups.end());
    next_groups.erase(std::unique(next_groups.begin(), next_groups.end()), next_groups.end());
    current_groups = std::move(next_groups);
  }

  if (visited_count < static_cast<int>(group_arc_ids.size())) {
    std::vector<InstArcRecord> cyclic_or_unreachable_records;
    for (int group_id = 0; group_id < static_cast<int>(group_arc_ids.size()); ++group_id) {
      if (visited[group_id]) {
        continue;
      }
      for (const int arc_id : group_arc_ids[static_cast<std::size_t>(group_id)]) {
        cyclic_or_unreachable_records.push_back(comb_arc_records[arc_id]);
      }
    }
    if (!cyclic_or_unreachable_records.empty()) {
      arc_records_by_level.push_back(std::move(cyclic_or_unreachable_records));
    }
  }

  return arc_records_by_level;
}

double staToUser(const sta::Unit* unit, double value)
{
  return unit == nullptr ? value : const_cast<sta::Unit*>(unit)->staToUser(value);
}

bool staScalarIsUsable(double value)
{
  return std::isfinite(value) && std::abs(value) < static_cast<double>(sta::INF) * 0.5;
}

double staToUserOrDefault(const sta::Unit* unit, double value, double fallback = 0.0)
{
  return staScalarIsUsable(value) ? staToUser(unit, value) : fallback;
}

double timeUnitToPsScale(const sta::Unit* unit)
{
  if (unit == nullptr || !std::isfinite(unit->scale()) || unit->scale() <= 0.0f) {
    return 1.0;
  }
  return static_cast<double>(unit->scale()) / 1.0e-12;
}

double staTimeToPs(const sta::Unit* unit, double value)
{
  return staToUser(unit, value) * timeUnitToPsScale(unit);
}

double staTimeToPsOrDefault(const sta::Unit* unit, double value, double fallback = 0.0)
{
  return staScalarIsUsable(value) ? staTimeToPs(unit, value) : fallback;
}

double psToStaTimeOrDefault(const sta::Unit* unit, double value_ps, double fallback = 0.0)
{
  if (!std::isfinite(value_ps)) {
    return fallback;
  }
  const double user_value = value_ps / timeUnitToPsScale(unit);
  return unit == nullptr ? user_value : const_cast<sta::Unit*>(unit)->userToSta(user_value);
}

double staCapToUserOrDefault(const sta::Unit* unit, double value, double fallback = 0.0)
{
  return staToUserOrDefault(unit, value, fallback);
}

double capUnitToPfScale(const sta::Unit* unit)
{
  if (unit == nullptr || !std::isfinite(unit->scale()) || unit->scale() <= 0.0f) {
    return 1.0;
  }
  return static_cast<double>(unit->scale()) / 1.0e-12;
}

double staCapToPfOrDefault(const sta::Unit* unit, double value, double fallback = 0.0)
{
  return staScalarIsUsable(value) ? staToUser(unit, value) * capUnitToPfScale(unit) : fallback;
}

double pfToStaCapOrDefault(const sta::Unit* unit, double value_pf, double fallback = 0.0)
{
  if (!std::isfinite(value_pf) || value_pf <= 0.0) {
    return fallback;
  }
  const double user_value = value_pf / capUnitToPfScale(unit);
  return unit == nullptr ? user_value : const_cast<sta::Unit*>(unit)->userToSta(user_value);
}

double resolveLibPinSlewLimitForPythonPs(const sta::Unit* unit, sta::LibertyPort* liberty_port)
{
  if (liberty_port == nullptr) {
    return 0.0;
  }
  float limit = 0.0f;
  bool exists = false;
  liberty_port->slewLimit(sta::MinMax::max(), limit, exists);
  if (!exists && liberty_port->libertyLibrary() != nullptr) {
    liberty_port->libertyLibrary()->defaultMaxSlew(limit, exists);
  }
  return exists ? staTimeToPsOrDefault(unit, limit) : 0.0;
}

double resistanceUnitToOhmScale(const sta::Unit* unit)
{
  if (unit == nullptr || !std::isfinite(unit->scale()) || unit->scale() <= 0.0f) {
    return 1.0;
  }
  return static_cast<double>(unit->scale());
}

double staResistanceToOhmOrDefault(const sta::Unit* unit, double value, double fallback = 0.0)
{
  return staScalarIsUsable(value) ? staToUser(unit, value) * resistanceUnitToOhmScale(unit) : fallback;
}

double exportLibcellLeakageForPython(sta::LibertyCell* liberty_cell)
{
  if (liberty_cell == nullptr) {
    return 0.0;
  }

  const sta::Units* units = liberty_cell->libertyLibrary() == nullptr ? nullptr : liberty_cell->libertyLibrary()->units();
  const sta::Unit* power_unit = units == nullptr ? nullptr : units->powerUnit();

  float scalar_leakage = 0.0f;
  bool scalar_exists = false;
  liberty_cell->leakagePower(scalar_leakage, scalar_exists);
  if (scalar_exists && std::isfinite(scalar_leakage) && scalar_leakage > 0.0f) {
    return staToUser(power_unit, scalar_leakage);
  }

  auto* leakage_powers = liberty_cell->leakagePowers();
  if (leakage_powers == nullptr || leakage_powers->empty()) {
    return scalar_exists && std::isfinite(scalar_leakage) ? staToUser(power_unit, scalar_leakage) : 0.0;
  }

  double positive_sum = 0.0;
  int positive_count = 0;
  for (auto* leakage_power : *leakage_powers) {
    if (leakage_power == nullptr) {
      continue;
    }
    const float value = leakage_power->power();
    if (std::isfinite(value) && value > 0.0f) {
      positive_sum += static_cast<double>(value);
      positive_count += 1;
    }
  }
  if (positive_count > 0) {
    return staToUser(power_unit, positive_sum / static_cast<double>(positive_count));
  }
  return scalar_exists && std::isfinite(scalar_leakage) ? staToUser(power_unit, scalar_leakage) : 0.0;
}

double clockPeriodForPython(sta::dbSta* sta, const sta::Unit* time_unit, double fallback = 9.0e7)
{
  if (sta == nullptr || sta->sdc() == nullptr || sta->sdc()->clocks() == nullptr) {
    return fallback;
  }

  double min_period = fallback;
  bool found_period = false;
  for (auto* clock : *sta->sdc()->clocks()) {
    if (clock == nullptr) {
      continue;
    }
    const double period = staTimeToPsOrDefault(time_unit, clock->period(), fallback);
    if (!std::isfinite(period) || period <= 0.0 || period >= fallback) {
      continue;
    }
    min_period = found_period ? std::min(min_period, period) : period;
    found_period = true;
  }
  return found_period ? min_period : fallback;
}

template <typename PortDelaySet>
double portDelayForPython(PortDelaySet* port_delays,
                          const sta::RiseFall* rf,
                          const sta::Unit* time_unit,
                          double fallback = 0.0)
{
  if (port_delays == nullptr || rf == nullptr) {
    return fallback;
  }

  double max_delay = fallback;
  bool found_delay = false;
  for (auto* port_delay : *port_delays) {
    if (port_delay == nullptr || port_delay->delays() == nullptr) {
      continue;
    }
    float delay = 0.0f;
    bool exists = false;
    port_delay->delays()->value(rf, sta::MinMax::max(), delay, exists);
    if (!exists || !staScalarIsUsable(delay)) {
      continue;
    }
    const double delay_ps = staTimeToPs(time_unit, delay);
    max_delay = found_delay ? std::max(max_delay, delay_ps) : delay_ps;
    found_delay = true;
  }
  return found_delay ? max_delay : fallback;
}

double inputDelayForPython(sta::dbSta* sta,
                           const sta::Pin* pin,
                           const sta::RiseFall* rf,
                           const sta::Unit* time_unit,
                           double fallback = 0.0)
{
  if (sta == nullptr || sta->sdc() == nullptr || pin == nullptr) {
    return fallback;
  }
  return portDelayForPython(sta->sdc()->inputDelaysLeafPin(pin), rf, time_unit, fallback);
}

double outputDelayForPython(sta::dbSta* sta,
                            const sta::Pin* pin,
                            const sta::RiseFall* rf,
                            const sta::Unit* time_unit,
                            double fallback = 0.0)
{
  if (sta == nullptr || sta->sdc() == nullptr || pin == nullptr) {
    return fallback;
  }
  return portDelayForPython(sta->sdc()->outputDelaysLeafPin(pin), rf, time_unit, fallback);
}

double inputArrivalForPython(const sta::Unit* time_unit, double arrival, double fallback = 0.0)
{
  return staTimeToPsOrDefault(time_unit, arrival, fallback);
}

double invalidEndpointRequiredSentinelForPython()
{
  return 9.0e7;
}

void appendTopInputStartPointForPython(PyPlaceDB& pydb,
                                       int pin_id,
                                       sta::dbSta* sta,
                                       const sta::Pin* sta_pin,
                                       const sta::Unit* time_unit,
                                       double rise_slew_ps,
                                       double fall_slew_ps)
{
  pydb.start_points.append(pin_id);
  pydb.inrdelays.append(inputDelayForPython(sta, sta_pin, sta::RiseFall::rise(), time_unit));
  pydb.infdelays.append(inputDelayForPython(sta, sta_pin, sta::RiseFall::fall(), time_unit));
  pydb.inrtrans.append(rise_slew_ps);
  pydb.inftrans.append(fall_slew_ps);
}

double outputRequiredTimeForPython(sta::dbSta* sta,
                                   const sta::Pin* pin,
                                   const sta::RiseFall* rf,
                                   const sta::Unit* time_unit)
{
  const double invalid_required = invalidEndpointRequiredSentinelForPython();
  if (sta == nullptr || sta->sdc() == nullptr || pin == nullptr
      || !sta->sdc()->hasOutputDelay(pin)) {
    return invalid_required;
  }
  const double period = clockPeriodForPython(sta, time_unit, invalid_required);
  if (period >= invalid_required) {
    return invalid_required;
  }
  return period - outputDelayForPython(sta, pin, rf, time_unit, 0.0);
}

double registerRequiredTimeForPython(sta::dbSta* sta, const sta::Unit* time_unit)
{
  return clockPeriodForPython(sta, time_unit, 9.0e7);
}

double pinRequiredTimeForPython(const sta::Unit* time_unit,
                                double arrival,
                                double slack,
                                double fallback)
{
  if (!staScalarIsUsable(arrival) || !staScalarIsUsable(slack)) {
    return fallback;
  }
  return staTimeToPsOrDefault(time_unit, arrival + slack, fallback);
}

std::unordered_map<int, int> endpointIndexByPinId(const PyPlaceDB& pydb)
{
  std::unordered_map<int, int> endpoint_index_by_pin_id;
  for (int endpoint_index = 0; endpoint_index < static_cast<int>(py::len(pydb.end_points)); ++endpoint_index) {
    const int pin_id = pydb.end_points[endpoint_index].cast<int>();
    endpoint_index_by_pin_id[pin_id] = endpoint_index;
  }
  return endpoint_index_by_pin_id;
}

void markPythonSetupEndpointBaseRat(PyPlaceDB& pydb,
                                    const std::unordered_map<int, int>& endpoint_index_by_pin_id,
                                    int endpoint_pin_id,
                                    double base_required)
{
  const auto endpoint_it = endpoint_index_by_pin_id.find(endpoint_pin_id);
  if (endpoint_it == endpoint_index_by_pin_id.end()) {
    return;
  }
  const int endpoint_index = endpoint_it->second;
  pydb.endpoints_rRAT[py::int_(endpoint_index)] = base_required;
  pydb.endpoints_fRAT[py::int_(endpoint_index)] = base_required;
}

bool axisIsTransition(sta::TableAxisVariable variable)
{
  switch (variable) {
    case sta::TableAxisVariable::input_net_transition:
    case sta::TableAxisVariable::input_transition_time:
    case sta::TableAxisVariable::related_pin_transition:
    case sta::TableAxisVariable::constrained_pin_transition:
    case sta::TableAxisVariable::output_pin_transition:
    case sta::TableAxisVariable::connect_delay:
    case sta::TableAxisVariable::time:
      return true;
    default:
      return false;
  }
}

bool axisIsRelatedPinTransition(sta::TableAxisVariable variable)
{
  return variable == sta::TableAxisVariable::related_pin_transition;
}

bool axisIsConstrainedPinTransition(sta::TableAxisVariable variable)
{
  return variable == sta::TableAxisVariable::constrained_pin_transition;
}

bool axisIsCapacitance(sta::TableAxisVariable variable)
{
  switch (variable) {
    case sta::TableAxisVariable::total_output_net_capacitance:
    case sta::TableAxisVariable::equal_or_opposite_output_net_capacitance:
    case sta::TableAxisVariable::related_out_total_output_net_capacitance:
      return true;
    default:
      return false;
  }
}

odb::dbMaster* libertyCellMaster(sta::dbNetwork* network,
                                 odb::dbBlock* block,
                                 const sta::LibertyCell* liberty_cell)
{
  if (liberty_cell == nullptr) {
    return nullptr;
  }
  if (network != nullptr) {
    if (auto* master = network->staToDb(liberty_cell)) {
      return master;
    }
  }
  auto* db = block == nullptr ? nullptr : block->getDataBase();
  return db == nullptr ? nullptr : db->findMaster(liberty_cell->name());
}

std::string libertyPortLookupName(const sta::LibertyCell* liberty_cell,
                                  std::string_view physical_port_name)
{
  if (liberty_cell == nullptr || physical_port_name.empty()) {
    return {};
  }
  const std::size_t left_bracket = physical_port_name.rfind('[');
  if (left_bracket != std::string_view::npos
      && physical_port_name.back() == ']'
      && left_bracket > 0) {
    // LEF macro pins are commonly bit-blasted (for example, addr_in[0]),
    // while the Liberty macro declares one bus port (addr_in).  Some
    // OpenSTA versions also expose a Liberty bus member by the bit name, so
    // the aggregate lookup must take precedence for this metadata map.
    // The physical ITerm and its net remain distinct.
    const std::string base_name(physical_port_name.substr(0, left_bracket));
    if (liberty_cell->findLibertyPort(base_name.c_str()) != nullptr) {
      return base_name;
    }
  }

  const std::string physical_name(physical_port_name);
  return liberty_cell->findLibertyPort(physical_name.c_str()) == nullptr
             ? std::string()
             : physical_name;
}

odb::Rect libPortBBox(sta::dbNetwork* network,
                      odb::dbMaster* master,
                      sta::LibertyPort* liberty_port,
                      bool* exists)
{
  if (exists != nullptr) {
    *exists = false;
  }
  odb::Rect bbox;
  odb::dbMTerm* mterm = nullptr;
  if (network != nullptr && liberty_port != nullptr) {
    mterm = network->staToDb(liberty_port);
  }
  if (mterm == nullptr && master != nullptr && liberty_port != nullptr) {
    mterm = master->findMTerm(liberty_port->name());
  }
  if (mterm == nullptr) {
    return bbox;
  }
  bbox = mterm->getBBox();
  if (exists != nullptr) {
    *exists = true;
  }
  return bbox;
}

odb::dbMTerm* resolveLibertyPortMTerm(sta::dbNetwork* network,
                                      odb::dbMaster* master,
                                      sta::LibertyPort* liberty_port)
{
  odb::dbMTerm* mterm = nullptr;
  if (network != nullptr && liberty_port != nullptr) {
    mterm = network->staToDb(liberty_port);
  }
  if (mterm == nullptr && master != nullptr && liberty_port != nullptr) {
    mterm = master->findMTerm(liberty_port->name());
  }
  return mterm;
}

int resolveInstMTermPinId(
    const std::unordered_map<InstMTermKey, int, InstMTermKeyHash>& inst_mterm_pin_ids,
    odb::dbInst* inst,
    odb::dbMTerm* mterm)
{
  if (inst == nullptr || mterm == nullptr) {
    return -1;
  }
  const auto it = inst_mterm_pin_ids.find({inst, mterm});
  return it == inst_mterm_pin_ids.end() ? -1 : it->second;
}

void appendLibArcInfo(py::list& rows,
                      const sta::TimingArc* arc,
                      int lib_cell_id,
                      int lib_arc_offset,
                      int timing_sense,
                      int timing_type)
{
  py::list row;
  row.append(arc != nullptr && arc->from() != nullptr ? arc->from()->name() : std::string());
  row.append(arc != nullptr && arc->to() != nullptr ? arc->to()->name() : std::string());
  row.append(lib_cell_id);
  row.append(lib_arc_offset);
  row.append(timing_sense);
  row.append(timing_type);
  rows.append(row);
}

void appendPinGraphCsr(py::list& flat_edges,
                       py::list& starts,
                       const std::vector<std::vector<int>>& edges)
{
  std::size_t edge_count = 0;
  for (const auto& pin_edges : edges) {
    edge_count += pin_edges.size();
  }
  py::list flat(static_cast<py::ssize_t>(edge_count));
  py::list offsets(static_cast<py::ssize_t>(edges.size() + 1));
  PyList_SET_ITEM(offsets.ptr(), 0, PyLong_FromLong(0));
  Py_ssize_t edge_offset = 0;
  Py_ssize_t pin_offset = 1;
  for (const auto& pin_edges : edges) {
    for (const int edge : pin_edges) {
      PyList_SET_ITEM(flat.ptr(), edge_offset, PyLong_FromLong(edge));
      edge_offset += 1;
    }
    PyList_SET_ITEM(offsets.ptr(), pin_offset, PyLong_FromSsize_t(edge_offset));
    pin_offset += 1;
  }
  flat_edges = std::move(flat);
  starts = std::move(offsets);
}

void appendPinArcCsr(py::list& start,
                     py::list& pin_values,
                     py::list& arc_values,
                     const std::vector<std::vector<int>>& pins,
                     const std::vector<std::vector<int>>& arcs)
{
  std::size_t edge_count = 0;
  for (const auto& pin_list : pins) {
    edge_count += pin_list.size();
  }
  py::list starts(static_cast<py::ssize_t>(pins.size() + 1));
  py::list flat_pins(static_cast<py::ssize_t>(edge_count));
  py::list flat_arcs(static_cast<py::ssize_t>(edge_count));
  PyList_SET_ITEM(starts.ptr(), 0, PyLong_FromLong(0));
  Py_ssize_t edge_offset = 0;
  Py_ssize_t pin_offset = 1;
  for (std::size_t pin_id = 0; pin_id < pins.size(); ++pin_id) {
    const auto& pin_list = pins.at(pin_id);
    const auto& arc_list = arcs.at(pin_id);
    for (std::size_t edge_id = 0; edge_id < pin_list.size(); ++edge_id) {
      PyList_SET_ITEM(flat_pins.ptr(), edge_offset, PyLong_FromLong(pin_list[edge_id]));
      PyList_SET_ITEM(flat_arcs.ptr(),
                      edge_offset,
                      PyLong_FromLong(edge_id < arc_list.size() ? arc_list[edge_id] : -1));
      edge_offset += 1;
    }
    PyList_SET_ITEM(starts.ptr(), pin_offset, PyLong_FromSsize_t(edge_offset));
    pin_offset += 1;
  }
  start = std::move(starts);
  pin_values = std::move(flat_pins);
  arc_values = std::move(flat_arcs);
}

void appendEmptyLut(py::list& values,
                    py::list& trans_table,
                    py::list& cap_table,
                    py::list& dims)
{
  values.append(py::list());
  trans_table.append(py::list());
  cap_table.append(py::list());
  py::list lut_dim;
  lut_dim.append(0);
  lut_dim.append(0);
  dims.append(lut_dim);
}

void fillAxisValues(py::list& target,
                    const sta::TableAxis* axis,
                    bool transition_axis,
                    bool constraint_table,
                    const sta::Units* units)
{
  if (axis == nullptr) {
    return;
  }
  const sta::Unit* unit = nullptr;
  if (transition_axis || constraint_table) {
    unit = units == nullptr ? nullptr : units->timeUnit();
  } else {
    unit = units == nullptr ? nullptr : units->capacitanceUnit();
  }
  auto* values = axis->values();
  if (values == nullptr) {
    return;
  }
  for (const float axis_value : *values) {
    if (transition_axis || constraint_table) {
      target.append(staTimeToPs(unit, axis_value));
    } else {
      target.append(staCapToPfOrDefault(unit, axis_value));
    }
  }
}

float tableAxisValueOrZero(const sta::TableAxis* axis, std::size_t index)
{
  if (axis == nullptr || axis->size() == 0) {
    return 0.0f;
  }
  const std::size_t clamped = std::min(index, axis->size() - 1);
  return axis->axisValue(clamped);
}

float scaledTableValue(const sta::TableModel* model,
                       const sta::LibertyCell* liberty_cell,
                       const sta::Pvt* pvt,
                       std::size_t index1,
                       std::size_t index2,
                       std::size_t index3)
{
  if (model == nullptr || liberty_cell == nullptr) {
    return 0.0f;
  }
  const float axis_value1 = tableAxisValueOrZero(model->axis1(), index1);
  const float axis_value2 = tableAxisValueOrZero(model->axis2(), index2);
  const float axis_value3 = tableAxisValueOrZero(model->axis3(), index3);
  return model->findValue(liberty_cell, pvt, axis_value1, axis_value2, axis_value3);
}

void appendTableLut(py::list& values,
                    py::list& trans_table,
                    py::list& cap_table,
                    py::list& dims,
                    const sta::TableModel* model,
                    const sta::LibertyCell* liberty_cell,
                    const sta::Pvt* pvt,
                    bool constraint_table)
{
  if (model == nullptr || liberty_cell == nullptr) {
    appendEmptyLut(values, trans_table, cap_table, dims);
    return;
  }

  const sta::Units* units = liberty_cell->libertyLibrary() == nullptr ? nullptr : liberty_cell->libertyLibrary()->units();
  const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
  const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
  const sta::TableAxis* axis1 = model->axis1();
  const sta::TableAxis* axis2 = model->axis2();
  const sta::TableAxis* axis3 = model->axis3();
  const int order = model->order();

  py::list lut_values;
  py::list lut_trans;
  py::list lut_cap;
  py::list lut_dim;

  if (order <= 0) {
    lut_values.append(staTimeToPs(time_unit, scaledTableValue(model, liberty_cell, pvt, 0, 0, 0)));
    lut_dim.append(1);
    lut_dim.append(1);
    values.append(lut_values);
    trans_table.append(lut_trans);
    cap_table.append(lut_cap);
    dims.append(lut_dim);
    return;
  }

  const bool axis1_is_trans = axis1 != nullptr && axisIsTransition(axis1->variable());
  const bool axis1_is_cap = axis1 != nullptr && axisIsCapacitance(axis1->variable());
  const bool axis2_is_trans = axis2 != nullptr && axisIsTransition(axis2->variable());
  const bool axis2_is_cap = axis2 != nullptr && axisIsCapacitance(axis2->variable());

  const sta::TableAxis* trans_axis = nullptr;
  const sta::TableAxis* cap_axis = nullptr;
  int trans_axis_index = -1;
  int cap_axis_index = -1;

  if (constraint_table) {
    const sta::TableAxis* axes[] = {axis1, axis2, axis3};
    for (int axis_index = 0; axis_index < 3; ++axis_index) {
      const auto* axis = axes[axis_index];
      if (axis == nullptr) {
        continue;
      }
      const auto variable = axis->variable();
      if (trans_axis == nullptr && axisIsRelatedPinTransition(variable)) {
        trans_axis = axis;
        trans_axis_index = axis_index;
      }
      if (cap_axis == nullptr && axisIsConstrainedPinTransition(variable)) {
        cap_axis = axis;
        cap_axis_index = axis_index;
      }
    }
  }

  if (trans_axis == nullptr) {
    if (axis1_is_trans) {
      trans_axis = axis1;
      trans_axis_index = 0;
    } else if (axis2_is_trans) {
      trans_axis = axis2;
      trans_axis_index = 1;
    } else if (axis1 != nullptr) {
      trans_axis = axis1;
      trans_axis_index = 0;
    }
  }

  if (cap_axis == nullptr) {
    if (axis1_is_cap) {
      cap_axis = axis1;
      cap_axis_index = 0;
    } else if (axis2_is_cap) {
      cap_axis = axis2;
      cap_axis_index = 1;
    } else if (axis2 != nullptr && axis2 != trans_axis) {
      cap_axis = axis2;
      cap_axis_index = 1;
    } else if (axis3 != nullptr) {
      cap_axis = axis3;
      cap_axis_index = 2;
    }
  }

  fillAxisValues(lut_trans, trans_axis, true, constraint_table, units);
  fillAxisValues(lut_cap, cap_axis, false, constraint_table, units);

  const std::size_t dim1 = axis1 == nullptr ? 1 : axis1->size();
  const std::size_t dim2 = axis2 == nullptr ? 1 : axis2->size();
  const std::size_t dim3 = axis3 == nullptr ? 1 : axis3->size();
  if (trans_axis != nullptr && cap_axis != nullptr
      && trans_axis_index >= 0 && cap_axis_index >= 0 && trans_axis_index != cap_axis_index) {
    const std::size_t trans_dim = trans_axis->size();
    const std::size_t cap_dim = cap_axis->size();
    for (std::size_t trans_index = 0; trans_index < trans_dim; ++trans_index) {
      for (std::size_t cap_index = 0; cap_index < cap_dim; ++cap_index) {
        std::size_t original_indices[] = {0, 0, 0};
        original_indices[trans_axis_index] = trans_index;
        original_indices[cap_axis_index] = cap_index;
        lut_values.append(staTimeToPs(
            time_unit,
            scaledTableValue(
                model,
                liberty_cell,
                pvt,
                original_indices[0],
                original_indices[1],
                original_indices[2])));
      }
    }
  } else {
    for (std::size_t i = 0; i < dim1; ++i) {
      for (std::size_t j = 0; j < dim2; ++j) {
        for (std::size_t k = 0; k < dim3; ++k) {
          lut_values.append(staTimeToPs(
              time_unit,
              scaledTableValue(model, liberty_cell, pvt, i, j, k)));
        }
      }
    }
  }

  const int trans_dim = static_cast<int>(trans_axis == nullptr ? 0 : trans_axis->size());
  int cap_dim = static_cast<int>(cap_axis == nullptr ? 0 : cap_axis->size());
  if (order == 1 && trans_axis != nullptr && cap_axis == nullptr) {
    cap_dim = 0;
  }
  if (order == 1 && cap_axis != nullptr && trans_axis == cap_axis) {
    lut_trans = py::list();
    lut_cap = py::list();
    fillAxisValues(lut_cap, cap_axis, false, constraint_table, units);
    lut_dim.append(0);
    lut_dim.append(static_cast<int>(cap_axis->size()));
  } else {
    lut_dim.append(trans_dim);
    lut_dim.append(cap_dim);
  }

  if (py::len(lut_dim) == 0) {
    lut_dim.append(static_cast<int>(dim1));
    lut_dim.append(static_cast<int>(dim2 * dim3));
  }
  values.append(lut_values);
  trans_table.append(lut_trans);
  cap_table.append(lut_cap);
  dims.append(lut_dim);
}

int timingTypeToInt(sta::TimingType type)
{
  switch (type) {
    case sta::TimingType::rising_edge:
    case sta::TimingType::combinational_rise:
    case sta::TimingType::setup_rising:
    case sta::TimingType::hold_rising:
    case sta::TimingType::recovery_rising:
    case sta::TimingType::removal_rising:
    case sta::TimingType::non_seq_setup_rising:
    case sta::TimingType::non_seq_hold_rising:
      return 1;
    case sta::TimingType::falling_edge:
    case sta::TimingType::combinational_fall:
    case sta::TimingType::setup_falling:
    case sta::TimingType::hold_falling:
    case sta::TimingType::recovery_falling:
    case sta::TimingType::removal_falling:
    case sta::TimingType::non_seq_setup_falling:
    case sta::TimingType::non_seq_hold_falling:
      return -1;
    default:
      return 0;
  }
}

sta::TimingType timingTypeFromCheckArcSet(const sta::TimingArcSet* arc_set, int check_class)
{
  const sta::RiseFall* edge = arc_set == nullptr ? nullptr : arc_set->isRisingFallingEdge();
  const bool is_falling = edge == sta::RiseFall::fall();
  switch (check_class) {
    case 1:
      return is_falling ? sta::TimingType::setup_falling : sta::TimingType::setup_rising;
    case 2:
      return is_falling ? sta::TimingType::hold_falling : sta::TimingType::hold_rising;
    case 3:
      return is_falling ? sta::TimingType::recovery_falling : sta::TimingType::recovery_rising;
    case 4:
      return is_falling ? sta::TimingType::removal_falling : sta::TimingType::removal_rising;
    default:
      return sta::TimingType::unknown;
  }
}

bool isBufferMaster(OpenRoadPlaceIOBridgeImpl& raw_db, odb::dbMaster* master)
{
  if (master == nullptr || !raw_db.hasTimingInputs()) {
    return false;
  }
  auto* design = raw_db.design();
  if (design == nullptr) {
    return false;
  }
  return design->isBuffer(master);
}

void buildBasicTiming(OpenRoadPlaceIOBridgeImpl& raw_db,
                      const std::vector<PinRecord>& pins,
                      const std::vector<NetRecord>& nets,
                      const std::unordered_map<odb::dbITerm*, int>& iterm_pin_ids,
                      const std::unordered_map<odb::dbBTerm*, int>& bterm_pin_ids,
                      const std::vector<TimingInstRecord>& timing_insts,
                      PyPlaceDB& pydb)
{
  if (!raw_db.hasTimingInputs()) {
    return;
  }

  auto stage_begin = std::chrono::steady_clock::now();
  auto record_stage = [&](const char* name) {
    pydb.export_profile[py::str(name)] = elapsedMs(stage_begin);
    stage_begin = std::chrono::steady_clock::now();
  };

  auto* design = raw_db.design();
  auto* block = raw_db.block();
  if (design == nullptr || block == nullptr) {
    return;
  }

  auto* sta_for_export = design->getTech()->getSta();
  if (sta_for_export == nullptr) {
    return;
  }
  sta_for_export->updateTiming(false);
  record_stage("setup_rawdb_pydb_export_basic_timing_update_timing_ms");
  ord::Timing timing(design);
  auto* corner = timing.cmdCorner();
  auto* sta = timing.getSta();
  auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
  const sta::Units* units = sta == nullptr ? nullptr : sta->units();
  const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
  const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
  auto isSequentialClockPort = [&](odb::dbITerm* iterm) -> bool {
    if (network == nullptr || iterm == nullptr) {
      return false;
    }
    auto* master = iterm->getInst() == nullptr ? nullptr : iterm->getInst()->getMaster();
    if (master == nullptr) {
      return false;
    }
    auto* cell = network->dbToSta(master);
    auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
    auto* mterm = iterm->getMTerm();
    if (liberty_cell == nullptr || mterm == nullptr) {
      return false;
    }
    const std::string liberty_port_name
        = libertyPortLookupName(liberty_cell, mterm->getName());
    auto* liberty_port = liberty_port_name.empty()
                             ? nullptr
                             : liberty_cell->findLibertyPort(liberty_port_name.c_str());
    if (liberty_port == nullptr) {
      return false;
    }
    return liberty_port->isClock() || liberty_port->isRegClk() || liberty_port->isCheckClk();
  };

  pydb.flat_cells_by_level_start.append(0);
  pydb.flat_cells_by_reverse_level_start.append(0);

  py::list flat_level;
  py::list flat_reverse_level;
  for (const auto& timing_inst : timing_insts) {
    auto* inst = timing_inst.inst;
    const int node_id = timing_inst.node_id;
    if (inst == nullptr || inst->getMaster() == nullptr) {
      continue;
    }
    if (timing_inst.is_sequential_timing) {
      pydb.FF_ids.append(node_id);
      continue;
    }
    flat_level.append(node_id);
    flat_reverse_level.append(node_id);
  }
  for (auto item : flat_level) {
    pydb.flat_cells_by_level.append(item);
  }
  for (auto item : flat_reverse_level) {
    pydb.flat_cells_by_reverse_level.append(item);
  }
  pydb.flat_cells_by_level_start.append(py::len(pydb.flat_cells_by_level));
  pydb.flat_cells_by_reverse_level_start.append(py::len(pydb.flat_cells_by_reverse_level));
  record_stage("setup_rawdb_pydb_export_basic_timing_level_lists_ms");

  std::unordered_map<std::string, int> clock_pin_name_to_id;
  std::vector<double> clk_pin_raat_values;
  std::vector<double> clk_pin_faat_values;
  std::vector<double> clk_pin_rtran_values;
  std::vector<double> clk_pin_ftran_values;
  std::vector<std::string> clk_pin_name_values;
  auto get_clock_pin_id = [&](odb::dbInst* inst, odb::dbITerm* iterm) -> int {
    if (inst == nullptr || iterm == nullptr || iterm->getMTerm() == nullptr) {
      return -1;
    }
    const std::string key = clockPinKey(inst->getName(), iterm->getMTerm()->getName());
    auto it = clock_pin_name_to_id.find(key);
    if (it != clock_pin_name_to_id.end()) {
      return it->second;
    }
    const int clock_pin_id = static_cast<int>(clock_pin_name_to_id.size());
    clock_pin_name_to_id.emplace(key, clock_pin_id);
    clk_pin_raat_values.push_back(0.0);
    clk_pin_faat_values.push_back(0.0);
    clk_pin_rtran_values.push_back(0.0);
    clk_pin_ftran_values.push_back(0.0);
    clk_pin_name_values.push_back(clockPinKey(inst->getName(), iterm->getMTerm()->getName()));
    return clock_pin_id;
  };

  for (const auto& timing_inst : timing_insts) {
    auto* inst = timing_inst.inst;
    if (inst == nullptr || inst->getMaster() == nullptr || !timing_inst.is_sequential_timing) {
      continue;
    }
    for (auto* iterm : inst->getITerms()) {
      auto* sta_pin = network == nullptr ? nullptr : network->dbToSta(iterm);
      auto* mterm = iterm == nullptr ? nullptr : iterm->getMTerm();
      if (mterm == nullptr) {
        continue;
      }
      if (sta_pin != nullptr && (network->isCheckClk(sta_pin) || network->isRegClkPin(sta_pin))) {
        get_clock_pin_id(inst, iterm);
        continue;
      }
      if (isSequentialClockPort(iterm)) {
        get_clock_pin_id(inst, iterm);
        continue;
      }
      if (mterm->getSigType() == odb::dbSigType::CLOCK) {
        get_clock_pin_id(inst, iterm);
      }
    }
  }
  record_stage("setup_rawdb_pydb_export_basic_timing_clock_pin_discovery_ms");

  pydb.net2driver_pin_map = py::list();
  for (const auto& timing_inst : timing_insts) {
    auto* inst = timing_inst.inst;
    if (inst == nullptr || inst->getMaster() == nullptr || !timing_inst.is_sequential_timing) {
      continue;
    }
    for (auto* iterm : inst->getITerms()) {
      auto pin_it = iterm_pin_ids.find(iterm);
      if (pin_it == iterm_pin_ids.end()) {
        continue;
      }
      auto* sta_pin = network == nullptr ? nullptr : network->dbToSta(iterm);
      const bool is_clock_pin
          = (sta_pin != nullptr && (network->isCheckClk(sta_pin) || network->isRegClkPin(sta_pin)))
            || isSequentialClockPort(iterm)
            || design->isInClock(iterm);
      if (is_clock_pin) {
        const int clock_pin_id = get_clock_pin_id(inst, iterm);
        if (clock_pin_id >= 0) {
          pydb.clock_pins.append(clock_pin_id);
          clk_pin_raat_values[clock_pin_id] = staTimeToPsOrDefault(time_unit, timing.getPinArrival(iterm, ord::Timing::Rise));
          clk_pin_faat_values[clock_pin_id] = staTimeToPsOrDefault(time_unit, timing.getPinArrival(iterm, ord::Timing::Fall));
          clk_pin_rtran_values[clock_pin_id] = staTimeToPsOrDefault(time_unit, timing.getPinSlew(iterm, ord::Timing::Rise));
          clk_pin_ftran_values[clock_pin_id] = staTimeToPsOrDefault(time_unit, timing.getPinSlew(iterm, ord::Timing::Fall));
        }
      }
      const std::string dir = ioTypeString(iterm->getIoType());
      if (dir == "OUTPUT") {
        pydb.start_points.append(pin_it->second);
        pydb.inrdelays.append(inputArrivalForPython(time_unit, timing.getPinArrival(iterm, ord::Timing::Rise)));
        pydb.infdelays.append(inputArrivalForPython(time_unit, timing.getPinArrival(iterm, ord::Timing::Fall)));
        pydb.inrtrans.append(staTimeToPsOrDefault(time_unit, timing.getPinSlew(iterm, ord::Timing::Rise)));
        pydb.inftrans.append(staTimeToPsOrDefault(time_unit, timing.getPinSlew(iterm, ord::Timing::Fall)));
      }
      if (timing.isEndpoint(iterm)) {
        pydb.end_points.append(pin_it->second);
        pydb.outcaps.append(0.0);
        const double invalid_required = invalidEndpointRequiredSentinelForPython();
        const double rise_arrival = timing.getPinArrival(iterm, ord::Timing::Rise, ord::Timing::Max);
        const double fall_arrival = timing.getPinArrival(iterm, ord::Timing::Fall, ord::Timing::Max);
        const double rise_slack = timing.getPinSlack(iterm, ord::Timing::Rise, ord::Timing::Max);
        const double fall_slack = timing.getPinSlack(iterm, ord::Timing::Fall, ord::Timing::Max);
        const double rise_min_arrival = timing.getPinArrival(iterm, ord::Timing::Rise, ord::Timing::Min);
        const double fall_min_arrival = timing.getPinArrival(iterm, ord::Timing::Fall, ord::Timing::Min);
        const double rise_min_slack = timing.getPinSlack(iterm, ord::Timing::Rise, ord::Timing::Min);
        const double fall_min_slack = timing.getPinSlack(iterm, ord::Timing::Fall, ord::Timing::Min);
        const double rise_required = pinRequiredTimeForPython(
            time_unit,
            rise_arrival,
            rise_slack,
            invalid_required);
        const double fall_required = pinRequiredTimeForPython(
            time_unit,
            fall_arrival,
            fall_slack,
            invalid_required);
        const double rise_min_required = pinRequiredTimeForPython(
            time_unit,
            rise_min_arrival,
            rise_min_slack,
            invalid_required);
        const double fall_min_required = pinRequiredTimeForPython(
            time_unit,
            fall_min_arrival,
            fall_min_slack,
            invalid_required);
        pydb.endpoints_rRAT.append(rise_required);
        pydb.endpoints_fRAT.append(fall_required);
        pydb.backend_endpoint_rAAT.append(inputArrivalForPython(time_unit, rise_arrival, -9.0e7));
        pydb.backend_endpoint_fAAT.append(inputArrivalForPython(time_unit, fall_arrival, -9.0e7));
        pydb.backend_endpoint_rRAT.append(rise_required);
        pydb.backend_endpoint_fRAT.append(fall_required);
        pydb.backend_endpoint_min_rAAT.append(inputArrivalForPython(time_unit, rise_min_arrival, -9.0e7));
        pydb.backend_endpoint_min_fAAT.append(inputArrivalForPython(time_unit, fall_min_arrival, -9.0e7));
        pydb.backend_endpoint_min_rRAT.append(rise_min_required);
        pydb.backend_endpoint_min_fRAT.append(fall_min_required);
      }
    }
  }
  record_stage("setup_rawdb_pydb_export_basic_timing_inst_pins_ms");

  for (std::size_t clock_pin_id = 0; clock_pin_id < clk_pin_name_values.size(); ++clock_pin_id) {
    pydb.clk_pin_names.append(clk_pin_name_values[clock_pin_id]);
    pydb.clk_pin_r_aat.append(clk_pin_raat_values[clock_pin_id]);
    pydb.clk_pin_f_aat.append(clk_pin_faat_values[clock_pin_id]);
    pydb.clk_pin_rtran.append(clk_pin_rtran_values[clock_pin_id]);
    pydb.clk_pin_ftran.append(clk_pin_ftran_values[clock_pin_id]);
  }
  record_stage("setup_rawdb_pydb_export_basic_timing_clock_pin_append_ms");

  for (auto* bterm : block->getBTerms()) {
    if (bterm == nullptr || bterm->getNet() == nullptr || !isSignalNet(bterm->getNet())) {
      continue;
    }
    auto pin_it = bterm_pin_ids.find(bterm);
    if (pin_it == bterm_pin_ids.end()) {
      continue;
    }
    const std::string dir = ioTypeString(bterm->getIoType());
    if (dir == "INPUT") {
      auto* sta_pin = network == nullptr ? nullptr : network->dbToSta(bterm);
      appendTopInputStartPointForPython(
          pydb,
          pin_it->second,
          sta,
          sta_pin,
          time_unit,
          staTimeToPsOrDefault(time_unit, timing.getPinSlew(bterm, ord::Timing::Rise)),
          staTimeToPsOrDefault(time_unit, timing.getPinSlew(bterm, ord::Timing::Fall)));
    }
    if (timing.isEndpoint(bterm)) {
      auto* sta_pin = network == nullptr ? nullptr : network->dbToSta(bterm);
      pydb.end_points.append(pin_it->second);
      pydb.outcaps.append(0.0);
      const double rise_required = outputRequiredTimeForPython(sta, sta_pin, sta::RiseFall::rise(), time_unit);
      const double fall_required = outputRequiredTimeForPython(sta, sta_pin, sta::RiseFall::fall(), time_unit);
      const double invalid_required = invalidEndpointRequiredSentinelForPython();
      const double rise_min_arrival = timing.getPinArrival(bterm, ord::Timing::Rise, ord::Timing::Min);
      const double fall_min_arrival = timing.getPinArrival(bterm, ord::Timing::Fall, ord::Timing::Min);
      const double rise_min_slack = timing.getPinSlack(bterm, ord::Timing::Rise, ord::Timing::Min);
      const double fall_min_slack = timing.getPinSlack(bterm, ord::Timing::Fall, ord::Timing::Min);
      const double rise_min_required = pinRequiredTimeForPython(time_unit, rise_min_arrival, rise_min_slack, invalid_required);
      const double fall_min_required = pinRequiredTimeForPython(time_unit, fall_min_arrival, fall_min_slack, invalid_required);
      pydb.endpoints_rRAT.append(rise_required);
      pydb.endpoints_fRAT.append(fall_required);
      pydb.backend_endpoint_rAAT.append(inputArrivalForPython(time_unit, timing.getPinArrival(bterm, ord::Timing::Rise, ord::Timing::Max), -9.0e7));
      pydb.backend_endpoint_fAAT.append(inputArrivalForPython(time_unit, timing.getPinArrival(bterm, ord::Timing::Fall, ord::Timing::Max), -9.0e7));
      pydb.backend_endpoint_rRAT.append(rise_required);
      pydb.backend_endpoint_fRAT.append(fall_required);
      pydb.backend_endpoint_min_rAAT.append(inputArrivalForPython(time_unit, rise_min_arrival, -9.0e7));
      pydb.backend_endpoint_min_fAAT.append(inputArrivalForPython(time_unit, fall_min_arrival, -9.0e7));
      pydb.backend_endpoint_min_rRAT.append(rise_min_required);
      pydb.backend_endpoint_min_fRAT.append(fall_min_required);
    }
  }
  record_stage("setup_rawdb_pydb_export_basic_timing_bterms_ms");

  pydb.net_flat_arcs_start.append(0);
  int flat_arc_offset = 0;
  for (const auto& net : nets) {
    pydb.net2driver_pin_map.append(net.driver_pin_id);
    for (const int pin_id : net.pin_ids) {
      if (pin_id == net.driver_pin_id) {
        continue;
      }
      appendTimingArc(pydb.net_flat_arcs, net.driver_pin_id, pin_id);
      flat_arc_offset += 1;
    }
    pydb.net_flat_arcs_start.append(flat_arc_offset);
  }
  record_stage("setup_rawdb_pydb_export_basic_timing_net_arcs_ms");
}

void appendDiffGuidedBatchConflictGroups(
    PyPlaceDB& pydb,
    const std::vector<PinRecord>& pins,
    const std::vector<NetRecord>& nets,
    const std::vector<std::vector<InstArcRecord>>& arc_records_by_level)
{
  for (const auto& endpoint_pin_obj : pydb.endpoint_pin_ids) {
    const int endpoint_pin_id = endpoint_pin_obj.cast<int>();
    if (endpoint_pin_id >= 0 && endpoint_pin_id < static_cast<int>(pins.size())) {
      const int inst_id = pins[endpoint_pin_id].node_id;
      if (inst_id >= 0) {
        pydb.diff_guided_batch_endpoint_groups.append(toPyList({inst_id}));
      }
    }
  }

  for (const auto& level_records : arc_records_by_level) {
    std::vector<int> level_inst_ids;
    level_inst_ids.reserve(level_records.size());
    for (const auto& record : level_records) {
      if (record.inst_id >= 0) {
        level_inst_ids.push_back(record.inst_id);
      }
    }
    appendUniqueInstGroup(pydb.diff_guided_batch_top_path_groups, level_inst_ids);
  }

  for (const auto& net : nets) {
    std::vector<int> net_inst_ids;
    net_inst_ids.reserve(net.pin_ids.size());
    for (const int pin_id : net.pin_ids) {
      if (pin_id >= 0 && pin_id < static_cast<int>(pins.size())) {
        net_inst_ids.push_back(pins[pin_id].node_id);
      }
    }
    appendUniqueInstGroup(pydb.diff_guided_batch_net_neighborhood_groups, net_inst_ids);
  }

  for (const auto& level_records : arc_records_by_level) {
    for (const auto& record : level_records) {
      std::vector<int> edge_inst_ids;
      if (record.from_pin_id >= 0 && record.from_pin_id < static_cast<int>(pins.size())) {
        edge_inst_ids.push_back(pins[record.from_pin_id].node_id);
      }
      if (record.to_pin_id >= 0 && record.to_pin_id < static_cast<int>(pins.size())) {
        edge_inst_ids.push_back(pins[record.to_pin_id].node_id);
      }
      if (record.inst_id >= 0) {
        edge_inst_ids.push_back(record.inst_id);
      }
      appendUniqueInstGroup(pydb.diff_guided_batch_fanin_fanout_groups, edge_inst_ids);
    }
  }
}

PyPlaceDB buildPyDB(OpenRoadPlaceIOBridgeImpl& raw_db)
{
  const auto export_begin = std::chrono::steady_clock::now();
  auto stage_begin = export_begin;
  PyPlaceDB pydb;
  pydb.export_profile[py::str("openroad_vt_suffixes")] = raw_db.vtSuffixes();
  auto* block = raw_db.block();
  if (block == nullptr) {
    throw std::runtime_error("placeio_openroad raw database has no block");
  }
  auto record_stage = [&](const char* name) {
    pydb.export_profile[py::str(name)] = elapsedMs(stage_begin);
    stage_begin = std::chrono::steady_clock::now();
  };

  std::vector<odb::dbInst*> movable_insts;
  std::vector<odb::dbInst*> fixed_insts;
  for (auto* inst : block->getInsts()) {
    if (inst->isFixed()) {
      fixed_insts.push_back(inst);
    } else {
      movable_insts.push_back(inst);
    }
  }

  std::unordered_map<std::string, int> node_id_by_name;
  auto add_inst = [&](odb::dbInst* inst, bool fixed) {
    const int node_id = static_cast<int>(node_id_by_name.size());
    const std::string name = inst->getName();
    int loc_x = 0;
    int loc_y = 0;
    inst->getLocation(loc_x, loc_y);
    auto* master = inst->getMaster();

    node_id_by_name.emplace(name, node_id);
    pydb.node_name2id_map[py::str(name)] = node_id;
    pydb.node_names.append(name);
    pydb.node_master_names.append(master == nullptr ? std::string() : master->getName());
    pydb.node_is_buffer.append(isBufferMaster(raw_db, master));
    // Native macro masks expected by macroPlaceDB.initialize_from_rawdb.  The
    // meaning mirrors the iEDA backend: a hard macro is an instance whose LEF
    // class is BLOCK, and a writeback candidate is a hard macro the placer owns
    // (no explicit placement in the DEF).
    const bool is_hard_macro = master != nullptr && master->getType().isBlock();
    const auto placement_status = inst->getPlacementStatus();
    pydb.node_is_hard_macro.append(is_hard_macro);
    pydb.macro_writeback_candidate.append(
        is_hard_macro
        && (placement_status == odb::dbPlacementStatus::NONE
            || placement_status == odb::dbPlacementStatus::UNPLACED));
    pydb.node_x.append(loc_x);
    pydb.node_y.append(loc_y);
    pydb.node_orient.append(inst->getOrient().getString());
    pydb.node_size_x.append(master == nullptr ? 0 : static_cast<int>(master->getWidth()));
    pydb.node_size_y.append(master == nullptr ? 0 : static_cast<int>(master->getHeight()));
    pydb.node2orig_node_map.append(node_id);
    if (fixed) {
      pydb.num_terminals += 1;
    }
  };

  for (auto* inst : movable_insts) {
    add_inst(inst, false);
  }
  for (auto* inst : fixed_insts) {
    add_inst(inst, true);
  }
  const auto timing_insts = buildTimingInstView(raw_db, movable_insts, fixed_insts);

  std::vector<int> io_node_ids;
  for (auto* bterm : block->getBTerms()) {
    if (bterm == nullptr || bterm->getNet() == nullptr || !isSignalNet(bterm->getNet())) {
      continue;
    }
    int x = 0;
    int y = 0;
    bterm->getFirstPinLocation(x, y);
    const int node_id = static_cast<int>(node_id_by_name.size());
    const std::string name = bterm->getName();
    node_id_by_name.emplace(name, node_id);
    io_node_ids.push_back(node_id);
    pydb.node_name2id_map[py::str(name)] = node_id;
    pydb.node_names.append(name);
    pydb.node_master_names.append("");
    pydb.node_is_buffer.append(false);
    pydb.node_is_hard_macro.append(false);
    pydb.macro_writeback_candidate.append(false);
    pydb.node_x.append(x);
    pydb.node_y.append(y);
    pydb.node_orient.append("R0");
    pydb.node_size_x.append(0);
    pydb.node_size_y.append(0);
    pydb.node2orig_node_map.append(-1);
    pydb.num_terminal_NIs += 1;
  }

  std::unordered_map<std::string, int> io_node_by_name;
  std::unordered_map<odb::dbITerm*, int> iterm_pin_ids;
  std::unordered_map<odb::dbBTerm*, int> bterm_pin_ids;
  std::unordered_map<InstMTermKey, int, InstMTermKeyHash> inst_mterm_pin_ids;
  int io_index = 0;
  for (auto* bterm : block->getBTerms()) {
    if (bterm == nullptr || bterm->getNet() == nullptr || !isSignalNet(bterm->getNet())) {
      continue;
    }
    io_node_by_name.emplace(bterm->getName(), io_node_ids.at(io_index++));
  }
  record_stage("setup_rawdb_pydb_export_nodes_io_ms");

  std::vector<PinRecord> pins;
  std::vector<NetRecord> nets;
  nets.reserve(block->getNets().size());
  for (auto* net : block->getNets()) {
    if (!isSignalNet(net)) {
      continue;
    }

    NetRecord net_record;
    net_record.name = net->getName();
    std::vector<PendingPinRecord> net_pins;

    for (auto* iterm : net->getITerms()) {
      auto* mterm = iterm == nullptr ? nullptr : iterm->getMTerm();
      auto* inst = iterm == nullptr ? nullptr : iterm->getInst();
      if (inst == nullptr || mterm == nullptr) {
        continue;
      }
      int pin_x = 0;
      int pin_y = 0;
      if (!iterm->getAvgXY(&pin_x, &pin_y)) {
        int loc_x = 0;
        int loc_y = 0;
        inst->getLocation(loc_x, loc_y);
        pin_x = loc_x;
        pin_y = loc_y;
      }
      int node_x = 0;
      int node_y = 0;
      inst->getLocation(node_x, node_y);

      PinRecord pin;
      pin.name = inst->getName() + ":" + mterm->getName();
      pin.node_id = node_id_by_name.at(inst->getName());
      pin.net_id = -1;
      pin.offset_x = pin_x - node_x;
      pin.offset_y = pin_y - node_y;
      pin.direction = ioTypeString(iterm->getIoType());
      pin.is_io = false;
      net_pins.push_back({pin, iterm, nullptr});
    }

    for (auto* bterm : net->getBTerms()) {
      if (bterm == nullptr || !io_node_by_name.count(bterm->getName())) {
        continue;
      }
      int pin_x = 0;
      int pin_y = 0;
      bterm->getFirstPinLocation(pin_x, pin_y);

      PinRecord pin;
      pin.name = bterm->getName();
      pin.node_id = io_node_by_name.at(bterm->getName());
      pin.net_id = -1;
      pin.offset_x = 0;
      pin.offset_y = 0;
      pin.direction = ioTypeString(bterm->getIoType());
      pin.is_io = true;
      net_pins.push_back({pin, nullptr, bterm});
    }

    const int local_driver_pin_id = findTimingDriverPinId(net_pins);
    if (local_driver_pin_id < 0 || net_pins.size() < 2) {
      continue;
    }
    moveTimingDriverPinToFront(net_pins, local_driver_pin_id);
    const int load_count = static_cast<int>(net_pins.size()) - 1;
    if (load_count <= 0) {
      continue;
    }

    const int net_id = static_cast<int>(nets.size());
    pydb.net_name2id_map[py::str(net_record.name)] = net_id;
    pydb.net_names.append(net_record.name);
    pydb.net_weights.append(1.0);
    pydb.net_weight_deltas.append(0.0);
    pydb.net_criticality.append(0.0);
    pydb.net_criticality_deltas.append(0.0);

    for (int local_pin_id = 0; local_pin_id < static_cast<int>(net_pins.size()); ++local_pin_id) {
      const auto& pending_pin = net_pins.at(local_pin_id);
      const auto& pin = pending_pin.pin;
      const int pin_id = static_cast<int>(pins.size());
      PinRecord committed_pin = pin;
      committed_pin.net_id = net_id;
      pins.push_back(committed_pin);
      if (pending_pin.iterm != nullptr) {
        iterm_pin_ids.emplace(pending_pin.iterm, pin_id);
        inst_mterm_pin_ids.emplace(
            InstMTermKey{pending_pin.iterm->getInst(), pending_pin.iterm->getMTerm()},
            pin_id);
      }
      if (pending_pin.bterm != nullptr) {
        bterm_pin_ids.emplace(pending_pin.bterm, pin_id);
      }
      pydb.pin_name2id_map[py::str(committed_pin.name)] = pin_id;
      net_record.pin_ids.push_back(pin_id);
      if (local_pin_id == 0) {
        net_record.driver_pin_id = pin_id;
      }
    }

    nets.push_back(std::move(net_record));
  }
  record_stage("setup_rawdb_pydb_export_nets_pins_collect_ms");

  std::vector<std::vector<int>> node_to_pins(node_id_by_name.size());
  std::vector<int> node_to_region(node_id_by_name.size(), std::numeric_limits<int>::max());
  for (int pin_id = 0; pin_id < static_cast<int>(pins.size()); ++pin_id) {
    const auto& pin = pins.at(pin_id);
    pydb.pin_names.append(pin.name);
    pydb.pin_direct.append(pin.direction);
    pydb.pin_offset_x.append(pin.offset_x);
    pydb.pin_offset_y.append(pin.offset_y);
    pydb.pin2node_map.append(pin.node_id);
    pydb.pin2net_map.append(pin.net_id);
    node_to_pins.at(pin.node_id).push_back(pin_id);
    if (pin.node_id < static_cast<int>(movable_insts.size())) {
      pydb.num_movable_pins += 1;
    }
  }

  int net_pin_offset = 0;
  pydb.flat_net2pin_start_map.append(0);
  for (const auto& net : nets) {
    pydb.net2pin_map.append(toPyList(net.pin_ids));
    for (const int pin_id : net.pin_ids) {
      pydb.flat_net2pin_map.append(pin_id);
      net_pin_offset += 1;
    }
    pydb.flat_net2pin_start_map.append(net_pin_offset);
    if (!raw_db.hasTimingInputs()) {
      pydb.net2driver_pin_map.append(net.driver_pin_id);
    }
  }

  int node_pin_offset = 0;
  pydb.flat_node2pin_start_map.append(0);
  for (const auto& node_pins : node_to_pins) {
    pydb.node2pin_map.append(toPyList(node_pins));
    for (const int pin_id : node_pins) {
      pydb.flat_node2pin_map.append(pin_id);
      node_pin_offset += 1;
    }
    pydb.flat_node2pin_start_map.append(node_pin_offset);
  }
  record_stage("setup_rawdb_pydb_export_connectivity_csr_ms");

  int region_box_offset = 0;
  pydb.flat_region_boxes_start.append(0);
  for (auto* region : block->getRegions()) {
    if (region == nullptr) {
      continue;
    }
    py::list region_boxes;
    for (auto* box : region->getBoundaries()) {
      if (box == nullptr) {
        continue;
      }
      appendBox(region_boxes, box->xMin(), box->yMin(), box->xMax(), box->yMax());
      appendBox(pydb.flat_region_boxes, box->xMin(), box->yMin(), box->xMax(), box->yMax());
      region_box_offset += 1;
    }
    pydb.regions.append(region_boxes);
    pydb.flat_region_boxes_start.append(region_box_offset);

    if (region->getRegionType() == odb::dbRegionType::EXCLUSIVE) {
      const int region_id = static_cast<int>(py::len(pydb.regions)) - 1;
      for (auto* inst : region->getRegionInsts()) {
        if (inst == nullptr || inst->isFixed()) {
          continue;
        }
        auto it = node_id_by_name.find(inst->getName());
        if (it != node_id_by_name.end()) {
          node_to_region[it->second] = region_id;
        }
      }
      for (auto* group : region->getGroups()) {
        if (group == nullptr) {
          continue;
        }
        for (auto* inst : group->getInsts()) {
          if (inst == nullptr || inst->isFixed()) {
            continue;
          }
          auto it = node_id_by_name.find(inst->getName());
          if (it != node_id_by_name.end()) {
            node_to_region[it->second] = region_id;
          }
        }
      }
    }
  }
  for (const int region_id : node_to_region) {
    pydb.node2fence_region_map.append(region_id);
  }

  using namespace boost::polygon::operators;
  boost::polygon::polygon_90_set_data<int> fixed_geometry;
  boost::polygon::polygon_90_set_data<int> row_geometry;
  for (auto* inst : fixed_insts) {
    const auto box = inst->getBBox()->getBox();
    fixed_geometry.insert(boost::polygon::rectangle_data<int>(
        box.xMin(), box.yMin(), box.xMax(), box.yMax()));
  }
  auto rows = block->getRows();
  for (auto* row : rows) {
    const auto bbox = row->getBBox();
    appendBox(pydb.rows, bbox.xMin(), bbox.yMin(), bbox.xMax(), bbox.yMax());
    row_geometry.insert(boost::polygon::rectangle_data<int>(
        bbox.xMin(), bbox.yMin(), bbox.xMax(), bbox.yMax()));
  }

  pydb.num_nodes = static_cast<unsigned int>(node_id_by_name.size());
  pydb.dbu = block->getDbUnitsPerMicron();

  const auto die = block->getDieArea();
  const auto core = block->getCoreArea();
  const auto bounds = core.area() > 0 ? core : die;
  pydb.xl = bounds.xMin();
  pydb.yl = bounds.yMin();
  pydb.xh = bounds.xMax();
  pydb.yh = bounds.yMax();
  pydb.routing_grid_xl = pydb.xl;
  pydb.routing_grid_yl = pydb.yl;
  pydb.routing_grid_xh = pydb.xh;
  pydb.routing_grid_yh = pydb.yh;

  if (!rows.empty()) {
    auto* first_row = *rows.begin();
    pydb.row_height = first_row->getBBox().dy();
    auto* site = first_row->getSite();
    if (site != nullptr) {
      pydb.site_width = site->getWidth();
    }
  } else {
    pydb.row_height = 1;
    pydb.site_width = 1;
    pydb.total_space_area = static_cast<double>(bounds.dx()) * static_cast<double>(bounds.dy());
    row_geometry.insert(boost::polygon::rectangle_data<int>(
        bounds.xMin(), bounds.yMin(), bounds.xMax(), bounds.yMax()));
  }
  pydb.total_fixed_node_area = boost::polygon::area(fixed_geometry);
  row_geometry -= fixed_geometry;
  pydb.total_space_area = boost::polygon::area(row_geometry);

  if (auto* gcell_grid = block->getGCellGrid()) {
    std::vector<int> grid_x;
    std::vector<int> grid_y;
    gcell_grid->getGridX(grid_x);
    gcell_grid->getGridY(grid_y);
    pydb.num_routing_grids_x = grid_x.size() > 1 ? static_cast<unsigned int>(grid_x.size() - 1) : 0;
    pydb.num_routing_grids_y = grid_y.size() > 1 ? static_cast<unsigned int>(grid_y.size() - 1) : 0;
    if (!grid_x.empty()) {
      pydb.routing_grid_xl = grid_x.front();
      pydb.routing_grid_xh = grid_x.back();
    }
    if (!grid_y.empty()) {
      pydb.routing_grid_yl = grid_y.front();
      pydb.routing_grid_yh = grid_y.back();
    }

    auto* tech = block->getTech();
    if (tech != nullptr && pydb.num_routing_grids_x > 0 && pydb.num_routing_grids_y > 0) {
      const std::size_t cell_count = static_cast<std::size_t>(pydb.num_routing_grids_x) * pydb.num_routing_grids_y;
      for (auto* layer : tech->getLayers()) {
        if (layer == nullptr || layer->getRoutingLevel() <= 0) {
          continue;
        }
        const bool is_horizontal = std::string(layer->getDirection().getString()) == "HORIZONTAL";
        double total_capacity = 0.0;
        for (unsigned int x = 0; x < pydb.num_routing_grids_x; ++x) {
          for (unsigned int y = 0; y < pydb.num_routing_grids_y; ++y) {
            const float capacity = gcell_grid->getCapacity(layer, x, y);
            const float usage = gcell_grid->getUsage(layer, x, y);
            pydb.initial_horizontal_demand_map.append(is_horizontal ? usage : 0.0f);
            pydb.initial_vertical_demand_map.append(is_horizontal ? 0.0f : usage);
            total_capacity += capacity;
          }
        }
        const double avg_capacity = cell_count == 0 ? 0.0 : total_capacity / static_cast<double>(cell_count);
        pydb.unit_horizontal_capacities.append(is_horizontal ? avg_capacity : 0.0);
        pydb.unit_vertical_capacities.append(is_horizontal ? 0.0 : avg_capacity);
      }
    }
  }
  record_stage("setup_rawdb_pydb_export_physical_rows_regions_grid_ms");

  buildBasicTiming(raw_db, pins, nets, iterm_pin_ids, bterm_pin_ids, timing_insts, pydb);
  record_stage("setup_rawdb_pydb_export_basic_timing_ms");

  if (auto* design = raw_db.design()) {
    ord::Timing timing(design);
    auto* sta = timing.getSta();
    auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
    auto* corner = timing.cmdCorner();
    const sta::Units* units = sta == nullptr ? nullptr : sta->units();
    const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
    const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
    const sta::Unit* resistance_unit = units == nullptr ? nullptr : units->resistanceUnit();

    auto* openroad = design->getOpenRoad();
    auto* estimate_parasitics = openroad == nullptr ? nullptr : openroad->getEstimateParasitics();
    if (raw_db.hasTimingInputs() && corner != nullptr && estimate_parasitics != nullptr) {
      double wire_res_per_meter = 0.0;
      double wire_cap_per_meter = 0.0;
      estimate_parasitics->wireSignalRC(corner, wire_res_per_meter, wire_cap_per_meter);
      pydb.r_unit = staResistanceToOhmOrDefault(resistance_unit, wire_res_per_meter / 1.0e6, 0.0);
      pydb.c_unit = staCapToPfOrDefault(cap_unit, wire_cap_per_meter / 1.0e6, 0.0);
    }
    record_stage("setup_rawdb_pydb_export_rc_units_ms");

    std::unordered_map<std::string, int> clock_pin_name_to_id;
    for (int clock_pin_id = 0; clock_pin_id < py::len(pydb.clk_pin_names); ++clock_pin_id) {
      clock_pin_name_to_id.emplace(pydb.clk_pin_names[clock_pin_id].cast<std::string>(), clock_pin_id);
    }

    std::unordered_map<std::string, int> cell_name_to_cell_id;
    std::unordered_map<std::string, int> cell_name_to_main_id;
    std::unordered_map<std::string, int> cell_name_to_inst_size;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> cell_pin_offset;
    std::vector<sta::LibertyCell*> ordered_cells;
    std::vector<int> ordered_cells_timing_coordinate;
    std::vector<int> ordered_cells_vt;
    std::vector<int> cell_libpin_start;
    std::vector<int> main_id_2_cell_id_start_values;
    std::vector<int> main_id_size_limits;
    std::vector<int> main_id_vt_limits;
    std::vector<int> buffer_main_type_candidates;
    int lib_pin_offset = 0;

    std::vector<sta::LibertyCell*> used_cells;
    std::unordered_set<std::string> used_cell_names;
    for (auto* inst : block->getInsts()) {
      auto* master = inst == nullptr ? nullptr : inst->getMaster();
      if (master == nullptr || network == nullptr) {
        continue;
      }
      auto* cell = network->dbToSta(master);
      auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
      if (liberty_cell == nullptr || !used_cell_names.emplace(liberty_cell->name()).second) {
        continue;
      }
      used_cells.push_back(liberty_cell);
    }

    sta::LibertyLibrarySeq equiv_libs;
    if (network != nullptr) {
      std::unique_ptr<sta::LibertyLibraryIterator> lib_iter(network->libertyLibraryIterator());
      while (lib_iter != nullptr && lib_iter->hasNext()) {
        equiv_libs.push_back(lib_iter->next());
      }
    }
    if (sta != nullptr && !equiv_libs.empty()) {
      sta->makeEquivCells(&equiv_libs, nullptr);
    }
    record_stage("setup_rawdb_pydb_export_used_lib_cells_ms");

    std::unordered_set<std::string> assigned_cells;
    for (auto* liberty_cell : used_cells) {
      if (liberty_cell == nullptr || assigned_cells.count(liberty_cell->name())) {
        continue;
      }

      std::vector<sta::LibertyCell*> family;
      family.push_back(liberty_cell);
      if (sta != nullptr) {
        if (auto* equiv = sta->equivCells(liberty_cell)) {
          for (auto* equiv_cell : *equiv) {
            if (equiv_cell != nullptr) {
              family.push_back(equiv_cell);
            }
          }
        }
      }
      std::sort(family.begin(), family.end(), [](sta::LibertyCell* lhs, sta::LibertyCell* rhs) {
        const double lhs_leakage = exportLibcellLeakageForPython(lhs);
        const double rhs_leakage = exportLibcellLeakageForPython(rhs);
        const bool lhs_usable = std::isfinite(lhs_leakage) && lhs_leakage > 0.0;
        const bool rhs_usable = std::isfinite(rhs_leakage) && rhs_leakage > 0.0;
        if (lhs_usable != rhs_usable) {
          return lhs_usable;
        }
        if (lhs_usable && lhs_leakage != rhs_leakage) {
          return lhs_leakage < rhs_leakage;
        }
        const std::string lhs_name = lhs == nullptr ? std::string() : lhs->name();
        const std::string rhs_name = rhs == nullptr ? std::string() : rhs->name();
        return lhs_name < rhs_name;
      });
      family.erase(std::unique(family.begin(), family.end(), [](sta::LibertyCell* lhs, sta::LibertyCell* rhs) {
                     if (lhs == rhs) {
                       return true;
                     }
                     if (lhs == nullptr || rhs == nullptr) {
                       return false;
                     }
                     return lhs->name() == rhs->name();
                   }),
                   family.end());

      const int main_id = static_cast<int>(main_id_2_cell_id_start_values.size());
      main_id_2_cell_id_start_values.push_back(static_cast<int>(ordered_cells.size()));
      std::unordered_map<std::string, std::vector<sta::LibertyCell*>> drive_groups;
      std::unordered_map<std::string, std::vector<sta::LibertyCell*>> vt_groups;
      for (sta::LibertyCell* family_cell : family) {
        if (family_cell == nullptr || assigned_cells.count(family_cell->name())) {
          continue;
        }
        const sizing_metadata::CellClass cell_class =
            sizing_metadata::classifyCell(family_cell->name(), raw_db.vtSuffixes());
        drive_groups[cell_class.drive_key].push_back(family_cell);
        vt_groups[cell_class.vt_key].push_back(family_cell);
      }
      const auto drive_order = sizing_metadata::orderedAxis(drive_groups);
      const auto vt_order = sizing_metadata::orderedAxis(vt_groups);
      main_id_vt_limits.push_back(std::max(static_cast<int>(vt_order.size()), 1));
      int inst_size = 0;
      bool family_has_buffer_master = false;
      for (auto* family_cell : family) {
        if (family_cell == nullptr || assigned_cells.count(family_cell->name())) {
          continue;
        }
        const std::string cell_name = family_cell->name();
        auto* family_master = libertyCellMaster(network, block, family_cell);
        if (isBufferMaster(raw_db, family_master)) {
          family_has_buffer_master = true;
        }
        assigned_cells.insert(cell_name);
        cell_name_to_cell_id.emplace(cell_name, static_cast<int>(ordered_cells.size()));
        cell_name_to_main_id.emplace(cell_name, main_id);
        cell_name_to_inst_size.emplace(cell_name, inst_size++);
        const sizing_metadata::CellClass cell_class =
            sizing_metadata::classifyCell(cell_name, raw_db.vtSuffixes());
        ordered_cells_timing_coordinate.push_back(
            drive_order.at(cell_class.drive_key) + 1);
        ordered_cells_vt.push_back(vt_order.at(cell_class.vt_key));
        ordered_cells.push_back(family_cell);
      }
      main_id_size_limits.push_back(
          std::max(static_cast<int>(drive_order.size()), 1));
      if (family_has_buffer_master) {
        buffer_main_type_candidates.push_back(main_id);
      }
    }

    for (const int main_id : buffer_main_type_candidates) {
      pydb.buffer_main_type_candidate_indices.append(main_id);
    }
    if (buffer_main_type_candidates.size() == 1) {
      pydb.buffer_main_type_index = buffer_main_type_candidates.front();
      pydb.buffer_main_type_status = "ok";
    } else if (buffer_main_type_candidates.empty()) {
      pydb.buffer_main_type_index = -1;
      pydb.buffer_main_type_status = "unsupported_no_buffer_family";
    } else {
      pydb.buffer_main_type_index = -1;
      pydb.buffer_main_type_status = "unsupported_multiple_buffer_families";
    }
    record_stage("setup_rawdb_pydb_export_lib_families_ms");

    for (int cell_id = 0; cell_id < static_cast<int>(ordered_cells.size()); ++cell_id) {
      auto* liberty_cell = ordered_cells[cell_id];
      const std::string cell_name = liberty_cell->name();
      auto* master = libertyCellMaster(network, block, liberty_cell);
      const int main_id = cell_name_to_main_id[cell_name];

      py::list cell_info;
      cell_info.append(cell_name);
      cell_info.append(main_id);
      cell_info.append(static_cast<double>(ordered_cells_timing_coordinate[cell_id]));
      cell_info.append(ordered_cells_vt[cell_id]);
      pydb.flat_libcell_names.append(cell_name);
      pydb.flat_libcell_info.append(cell_info);
      pydb.flat_libcell_width.append(master == nullptr ? 0 : static_cast<int>(master->getWidth()));
      pydb.flat_libcell_height.append(master == nullptr ? 0 : static_cast<int>(master->getHeight()));
      pydb.flat_libcell_leakage.append(exportLibcellLeakageForPython(liberty_cell));

      cell_libpin_start.push_back(lib_pin_offset);
      const sta::Units* lib_units = liberty_cell->libertyLibrary() == nullptr ? units : liberty_cell->libertyLibrary()->units();
      const sta::Unit* lib_time_unit = lib_units == nullptr ? nullptr : lib_units->timeUnit();
      const sta::Unit* lib_cap_unit = lib_units == nullptr ? nullptr : lib_units->capacitanceUnit();

      int pin_offset = 0;
      sta::LibertyCellPortIterator port_iter(liberty_cell);
      while (port_iter.hasNext()) {
        auto* liberty_port = port_iter.next();
        if (liberty_port == nullptr || liberty_port->isPwrGnd()) {
          continue;
        }
        const std::string port_name = liberty_port->name();
        cell_pin_offset[cell_name][port_name] = pin_offset++;
        bool has_bbox = false;
        const odb::Rect pin_bbox = libPortBBox(network, master, liberty_port, &has_bbox);
        pydb.flat_lib_pin_offset_x.append(has_bbox ? (pin_bbox.xMin() + pin_bbox.xMax()) / 2 : 0);
        pydb.flat_lib_pin_offset_y.append(has_bbox ? (pin_bbox.yMin() + pin_bbox.yMax()) / 2 : 0);
        const auto* port_direction = liberty_port->direction();
        const bool is_input_port = port_direction != nullptr && port_direction->isAnyInput();
        if (is_input_port) {
          pydb.flat_lib_pin_cap.append(staCapToPfOrDefault(lib_cap_unit, liberty_port->capacitance()));
          pydb.flat_lib_pin_rcap.append(staCapToPfOrDefault(lib_cap_unit, liberty_port->capacitance(sta::RiseFall::rise(), sta::MinMax::max())));
          pydb.flat_lib_pin_fcap.append(staCapToPfOrDefault(lib_cap_unit, liberty_port->capacitance(sta::RiseFall::fall(), sta::MinMax::max())));
        } else {
          pydb.flat_lib_pin_cap.append(0.0);
          pydb.flat_lib_pin_rcap.append(0.0);
          pydb.flat_lib_pin_fcap.append(0.0);
        }
        float limit = 0.0f;
        bool exists = false;
        liberty_port->capacitanceLimit(sta::MinMax::max(), limit, exists);
        pydb.flat_lib_pin_cap_limit.append(exists ? staCapToPfOrDefault(lib_cap_unit, limit) : 0.0);
        pydb.flat_lib_pin_slew_limit.append(resolveLibPinSlewLimitForPythonPs(lib_time_unit, liberty_port));
        lib_pin_offset += 1;
      }
    }
    record_stage("setup_rawdb_pydb_export_lib_pin_metadata_ms");

    for (const int cell_start : main_id_2_cell_id_start_values) {
      pydb.main_id_2_cell_id_start.append(cell_start);
    }
    pydb.main_id_2_cell_id_start.append(static_cast<int>(ordered_cells.size()));
    for (int main_id = 0; main_id < static_cast<int>(main_id_2_cell_id_start_values.size()); ++main_id) {
      py::list limit_row;
      const int cell_begin = main_id_2_cell_id_start_values[main_id];
      const std::string main_type = cell_begin < static_cast<int>(ordered_cells.size())
                                        ? ordered_cells[cell_begin]->name()
                                        : std::string();
      limit_row.append(main_type);
      limit_row.append(main_id < static_cast<int>(main_id_size_limits.size()) ? main_id_size_limits[main_id] : 1);
      limit_row.append(main_id < static_cast<int>(main_id_vt_limits.size()) ? main_id_vt_limits[main_id] : 1);
      pydb.flat_libcell_main_id2size_vt_limit.append(limit_row);
      const bool has_size_choices = main_id < static_cast<int>(main_id_size_limits.size())
                                    && main_id_size_limits[main_id] > 1;
      const bool has_vt_choices = main_id < static_cast<int>(main_id_vt_limits.size())
                                  && main_id_vt_limits[main_id] > 1;
      pydb.main_id_is_sizeable.append(has_size_choices || has_vt_choices);
    }
    for (int cell_id = 0; cell_id < static_cast<int>(ordered_cells.size()); ++cell_id) {
      pydb.cell_id_2_libpin_id_start.append(cell_libpin_start[cell_id]);
    }
    pydb.cell_id_2_libpin_id_start.append(lib_pin_offset);
    record_stage("setup_rawdb_pydb_export_size_metadata_ms");

    for (auto* inst : movable_insts) {
      auto* master = inst == nullptr ? nullptr : inst->getMaster();
      if (master == nullptr || network == nullptr) {
        pydb.inst_main_id.append(-1);
        pydb.inst_libcell_offset.append(0);
        pydb.inst_size.append(0);
        continue;
      }
      auto* cell = network->dbToSta(master);
      auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
      if (liberty_cell == nullptr) {
        pydb.inst_main_id.append(-1);
        pydb.inst_libcell_offset.append(0);
        pydb.inst_size.append(0);
        continue;
      }
      const std::string cell_name = liberty_cell->name();
      const int inst_offset = cell_name_to_inst_size[cell_name];
      pydb.inst_main_id.append(cell_name_to_main_id[cell_name]);
      pydb.inst_libcell_offset.append(inst_offset);
      pydb.inst_size.append(inst_offset);
    }
    for (auto* inst : fixed_insts) {
      auto* master = inst == nullptr ? nullptr : inst->getMaster();
      if (master == nullptr || network == nullptr) {
        pydb.inst_main_id.append(-1);
        pydb.inst_libcell_offset.append(0);
        pydb.inst_size.append(0);
        continue;
      }
      auto* cell = network->dbToSta(master);
      auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
      if (liberty_cell == nullptr) {
        pydb.inst_main_id.append(-1);
        pydb.inst_libcell_offset.append(0);
        pydb.inst_size.append(0);
        continue;
      }
      const std::string cell_name = liberty_cell->name();
      const int inst_offset = cell_name_to_inst_size[cell_name];
      pydb.inst_main_id.append(cell_name_to_main_id[cell_name]);
      pydb.inst_libcell_offset.append(inst_offset);
      pydb.inst_size.append(inst_offset);
    }
    for (std::size_t i = 0; i < io_node_ids.size(); ++i) {
      pydb.inst_main_id.append(-1);
      pydb.inst_libcell_offset.append(0);
      pydb.inst_size.append(0);
    }

    for (const auto& pin : pins) {
      if (pin.is_io) {
        pydb.pin_2_libpin_offset.append(-1);
        continue;
      }
      const auto inst_name_end = pin.name.find(':');
      const std::string inst_name = pin.name.substr(0, inst_name_end);
      const std::string port_name = pin.name.substr(inst_name_end + 1);
      auto node_it = node_id_by_name.find(inst_name);
      if (node_it == node_id_by_name.end() || node_it->second >= static_cast<int>(movable_insts.size() + fixed_insts.size())) {
        pydb.pin_2_libpin_offset.append(-1);
        continue;
      }
      odb::dbInst* inst = node_it->second < static_cast<int>(movable_insts.size())
                              ? movable_insts[node_it->second]
                              : fixed_insts[node_it->second - movable_insts.size()];
      auto* master = inst == nullptr ? nullptr : inst->getMaster();
      auto* cell = master == nullptr ? nullptr : network->dbToSta(master);
      auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
      if (liberty_cell == nullptr) {
        pydb.pin_2_libpin_offset.append(-1);
        continue;
      }
      const std::string cell_name = liberty_cell->name();
      const std::string liberty_port_name
          = libertyPortLookupName(liberty_cell, port_name);
      auto offset_it = cell_pin_offset[cell_name].find(liberty_port_name);
      pydb.pin_2_libpin_offset.append(offset_it == cell_pin_offset[cell_name].end() ? -1 : offset_it->second);
    }
    record_stage("setup_rawdb_pydb_export_inst_size_pin_maps_ms");

    auto* max_dcalc_ap = corner == nullptr ? nullptr : corner->findDcalcAnalysisPt(sta::MinMax::max());
    const sta::Pvt* max_pvt = max_dcalc_ap == nullptr ? nullptr : max_dcalc_ap->operatingConditions();
    std::vector<std::vector<TempTimingArcSet>> cell_lib_arcs;
    std::vector<std::vector<TempTimingArcSet>> cell_timing_check_arcs;
    cell_lib_arcs.reserve(ordered_cells.size());
    cell_timing_check_arcs.reserve(ordered_cells.size());
    int arc_offset = 0;
    for (int lib_cell_id = 0; lib_cell_id < static_cast<int>(ordered_cells.size()); ++lib_cell_id) {
      auto* liberty_cell = ordered_cells[lib_cell_id];
      std::vector<TempTimingArcSet> lib_arcs;
      std::vector<TempTimingArcSet> timing_check_arcs;
      pydb.cell_id_2_arc_id_start.append(arc_offset);
      int lib_arc_offset = 0;
      for (auto* arc_set : liberty_cell->timingArcSets()) {
        if (arc_set == nullptr || arc_set->from() == nullptr || arc_set->to() == nullptr) {
          continue;
        }
        TempTimingArcSet temp_arc = buildTempTimingArcSet(arc_offset, lib_arc_offset, arc_set);
        auto* representative_arc = temp_arc.representative_arc;
        if (representative_arc == nullptr || representative_arc->from() == nullptr || representative_arc->to() == nullptr) {
          continue;
        }
        const auto* role = arc_set->role();
        if (role == nullptr) {
          continue;
        }

        const bool is_setup_constraint = isPythonSetupConstraintRole(role);
        if (role->isTimingCheck() || role->isAsyncTimingCheck() || role->isNonSeqTimingCheck() || role->isDataCheck()) {
          TempTimingArcSet timing_check_arc = temp_arc;
          if (is_setup_constraint) {
            const auto* rise_check_model = max_dcalc_ap == nullptr || temp_arc.rise_arc == nullptr
                                               ? nullptr
                                               : dynamic_cast<const sta::CheckTableModel*>(
                                                   temp_arc.rise_arc->checkModel(max_dcalc_ap));
            const auto* fall_check_model = max_dcalc_ap == nullptr || temp_arc.fall_arc == nullptr
                                               ? nullptr
                                               : dynamic_cast<const sta::CheckTableModel*>(
                                                   temp_arc.fall_arc->checkModel(max_dcalc_ap));
            const auto* rise_constraint_model = rise_check_model == nullptr ? nullptr : rise_check_model->model();
            const auto* fall_constraint_model = fall_check_model == nullptr ? nullptr : fall_check_model->model();
            appendTableLut(pydb.f_delay_flat_luts_values,
                           pydb.f_delay_flat_luts_trans_table,
                           pydb.f_delay_flat_luts_cap_table,
                           pydb.f_delay_flat_luts_dim,
                           fall_constraint_model,
                           liberty_cell,
                           max_pvt,
                           true);
            appendTableLut(pydb.r_delay_flat_luts_values,
                           pydb.r_delay_flat_luts_trans_table,
                           pydb.r_delay_flat_luts_cap_table,
                           pydb.r_delay_flat_luts_dim,
                           rise_constraint_model,
                           liberty_cell,
                           max_pvt,
                           true);
            appendEmptyLut(pydb.r_trans_flat_luts_values,
                           pydb.r_trans_flat_luts_trans_table,
                           pydb.r_trans_flat_luts_cap_table,
                           pydb.r_trans_flat_luts_dim);
            appendEmptyLut(pydb.f_trans_flat_luts_values,
                           pydb.f_trans_flat_luts_trans_table,
                           pydb.f_trans_flat_luts_cap_table,
                           pydb.f_trans_flat_luts_dim);
            appendLibArcInfo(pydb.flat_libarc_info,
                             representative_arc,
                             lib_cell_id,
                             lib_arc_offset,
                             temp_arc.senseToInt(),
                             temp_arc.typeToInt());
            lib_arcs.push_back(temp_arc);
            timing_check_arc = temp_arc;
            arc_offset += 1;
            lib_arc_offset += 1;
          } else {
            timing_check_arc.lib_arc_idx = -1;
            timing_check_arc.lib_arc_offset = -1;
          }
          timing_check_arcs.push_back(timing_check_arc);
          continue;
        }

        const auto* rise_gate_model = max_dcalc_ap == nullptr || temp_arc.rise_arc == nullptr
                                           ? nullptr
                                           : temp_arc.rise_arc->gateTableModel(max_dcalc_ap);
        const auto* fall_gate_model = max_dcalc_ap == nullptr || temp_arc.fall_arc == nullptr
                                           ? nullptr
                                           : temp_arc.fall_arc->gateTableModel(max_dcalc_ap);
        appendTableLut(pydb.f_delay_flat_luts_values,
                       pydb.f_delay_flat_luts_trans_table,
                       pydb.f_delay_flat_luts_cap_table,
                       pydb.f_delay_flat_luts_dim,
                       fall_gate_model == nullptr ? nullptr : fall_gate_model->delayModel(),
                       liberty_cell,
                       max_pvt,
                       false);
        appendTableLut(pydb.r_delay_flat_luts_values,
                       pydb.r_delay_flat_luts_trans_table,
                       pydb.r_delay_flat_luts_cap_table,
                       pydb.r_delay_flat_luts_dim,
                       rise_gate_model == nullptr ? nullptr : rise_gate_model->delayModel(),
                       liberty_cell,
                       max_pvt,
                       false);
        appendTableLut(pydb.f_trans_flat_luts_values,
                       pydb.f_trans_flat_luts_trans_table,
                       pydb.f_trans_flat_luts_cap_table,
                       pydb.f_trans_flat_luts_dim,
                       fall_gate_model == nullptr ? nullptr : fall_gate_model->slewModel(),
                       liberty_cell,
                       max_pvt,
                       false);
        appendTableLut(pydb.r_trans_flat_luts_values,
                       pydb.r_trans_flat_luts_trans_table,
                       pydb.r_trans_flat_luts_cap_table,
                       pydb.r_trans_flat_luts_dim,
                       rise_gate_model == nullptr ? nullptr : rise_gate_model->slewModel(),
                       liberty_cell,
                       max_pvt,
                       false);
        appendLibArcInfo(pydb.flat_libarc_info,
                         representative_arc,
                         lib_cell_id,
                         lib_arc_offset,
                         temp_arc.senseToInt(),
                         temp_arc.typeToInt());
        lib_arcs.push_back(temp_arc);
          arc_offset += 1;
          lib_arc_offset += 1;
      }
      cell_lib_arcs.push_back(std::move(lib_arcs));
      cell_timing_check_arcs.push_back(std::move(timing_check_arcs));
    }
    pydb.cell_id_2_arc_id_start.append(arc_offset);
    record_stage("setup_rawdb_pydb_export_lib_luts_arcs_ms");

    std::vector<std::vector<int>> inst_to_arc_indices(pydb.num_nodes);
    std::vector<std::vector<int>> graph(static_cast<std::size_t>(py::len(pydb.pin_names)));
    std::vector<std::vector<int>> reverse_graph(static_cast<std::size_t>(py::len(pydb.pin_names)));
    std::vector<std::vector<int>> succ_pins(static_cast<std::size_t>(py::len(pydb.pin_names)));
    std::vector<std::vector<int>> succ_arc_ids(static_cast<std::size_t>(py::len(pydb.pin_names)));
    std::vector<std::vector<int>> pred_pins(static_cast<std::size_t>(py::len(pydb.pin_names)));
    std::vector<std::vector<int>> pred_arc_ids(static_cast<std::size_t>(py::len(pydb.pin_names)));
    std::map<std::pair<int, int>, std::vector<int>> pin_pair_to_arc_indices;
    std::vector<InstArcRecord> clk2q_arc_records;
    std::vector<InstArcRecord> comb_arc_records;
    std::unordered_set<int> startpoint_pin_ids;
    for (const auto& start_pin : pydb.start_points) {
      startpoint_pin_ids.insert(start_pin.cast<int>());
    }
    const auto endpoint_index_by_pin_id = endpointIndexByPinId(pydb);
    const double base_required = registerRequiredTimeForPython(sta, time_unit);

    auto add_graph_edge = [&](int from_pin, int to_pin, int arc_idx) {
      if (from_pin < 0 || to_pin < 0) {
        return;
      }
      const auto from = static_cast<std::size_t>(from_pin);
      const auto to = static_cast<std::size_t>(to_pin);
      if (from >= graph.size() || to >= reverse_graph.size()) {
        return;
      }
      graph[from].push_back(to_pin);
      reverse_graph[to].push_back(from_pin);
      succ_pins[from].push_back(to_pin);
      succ_arc_ids[from].push_back(arc_idx);
      pred_pins[to].push_back(from_pin);
      pred_arc_ids[to].push_back(arc_idx);
    };

    for (const auto& timing_inst : timing_insts) {
      auto* inst = timing_inst.inst;
      const int node_id = timing_inst.node_id;
      auto* master = inst == nullptr ? nullptr : inst->getMaster();
      auto* cell = master == nullptr || network == nullptr ? nullptr : network->dbToSta(master);
      auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
      if (liberty_cell == nullptr) {
        continue;
      }
      const std::string cell_name = liberty_cell->name();
      const int lib_cell_id = cell_name_to_cell_id[cell_name];
      for (const auto& lib_arc : cell_timing_check_arcs[lib_cell_id]) {
        auto* representative_arc = lib_arc.representative_arc;
        if (representative_arc == nullptr || representative_arc->from() == nullptr || representative_arc->to() == nullptr) {
          continue;
        }
        const auto* role = lib_arc.arc_set == nullptr ? nullptr : lib_arc.arc_set->role();
        const int check_class = timingCheckClassToInt(role);
        if (check_class == 0) {
          continue;
        }
        int from_pin_id = resolveInstMTermPinId(
            inst_mterm_pin_ids,
            inst,
            resolveLibertyPortMTerm(network, master, representative_arc->from()));
        const int to_pin_id = resolveInstMTermPinId(
            inst_mterm_pin_ids,
            inst,
            resolveLibertyPortMTerm(network, master, representative_arc->to()));
        auto clk_id_it = clock_pin_name_to_id.find(clockPinKey(inst->getName(), representative_arc->from()->name()));
        if (clk_id_it != clock_pin_name_to_id.end()) {
          from_pin_id = clk_id_it->second;
        }
        if (from_pin_id < 0 || to_pin_id < 0) {
          continue;
        }
        appendTimingCheckArc(pydb.endpoints_timing_check_arcs,
                             from_pin_id,
                             to_pin_id,
                             lib_cell_id,
                             lib_arc.lib_arc_idx,
                             lib_arc.senseToInt(),
                             timingTypeToInt(
                                 timingTypeFromCheckArcSet(lib_arc.arc_set, check_class)),
                             check_class,
                             lib_arc.lib_arc_offset);
      }
      for (const auto& lib_arc : cell_lib_arcs[lib_cell_id]) {
        auto* representative_arc = lib_arc.representative_arc;
        if (representative_arc == nullptr || representative_arc->from() == nullptr || representative_arc->to() == nullptr) {
          continue;
        }
        const auto* role = lib_arc.arc_set == nullptr ? nullptr : lib_arc.arc_set->role();
        const auto* generic_role = role == nullptr ? nullptr : role->genericRole();
        const bool is_timing_check = role != nullptr
                                     && (role->isTimingCheck() || role->isAsyncTimingCheck()
                                         || role->isNonSeqTimingCheck() || role->isDataCheck());
        const bool is_setup_constraint = isPythonSetupConstraintRole(role);
        const bool is_clk2q_arc = generic_role == sta::TimingRole::regClkToQ()
                                  || generic_role == sta::TimingRole::latchEnToQ();
        const int from_pin_id = resolveInstMTermPinId(
            inst_mterm_pin_ids,
            inst,
            resolveLibertyPortMTerm(network, master, representative_arc->from()));
        const int to_pin_id = resolveInstMTermPinId(
            inst_mterm_pin_ids,
            inst,
            resolveLibertyPortMTerm(network, master, representative_arc->to()));
        if (is_timing_check) {
          if (is_setup_constraint
              && (representative_arc->from()->isClock() || representative_arc->from()->isRegClk() || representative_arc->from()->isCheckClk())) {
            auto clk_id_it = clock_pin_name_to_id.find(clockPinKey(inst->getName(), representative_arc->from()->name()));
            if (clk_id_it != clock_pin_name_to_id.end() && to_pin_id >= 0) {
              markPythonSetupEndpointBaseRat(pydb, endpoint_index_by_pin_id, to_pin_id, base_required);
              appendConstraintArc(pydb.endpoints_constraint_arcs,
                                  clk_id_it->second,
                                  to_pin_id,
                                  lib_cell_id,
                                  lib_arc.lib_arc_idx,
                                  lib_arc.senseToInt());
            }
          }
          continue;
        }

        if (is_clk2q_arc) {
          auto clk_id_it = clock_pin_name_to_id.find(clockPinKey(inst->getName(), representative_arc->from()->name()));
          if (clk_id_it == clock_pin_name_to_id.end() || to_pin_id < 0) {
            continue;
          }
          clk2q_arc_records.push_back({
              clk_id_it->second,
              to_pin_id,
              lib_cell_id,
              lib_arc.lib_arc_idx,
              lib_arc.senseToInt(),
              lib_arc.typeToInt(),
              lib_arc.lib_arc_offset,
              node_id,
              true,
          });
          continue;
        }

        if (from_pin_id < 0 || to_pin_id < 0) {
          continue;
        }
        const bool drives_startpoint_pin = startpoint_pin_ids.count(to_pin_id) > 0;
        if (drives_startpoint_pin) {
          continue;
        }
        comb_arc_records.push_back({
            from_pin_id,
            to_pin_id,
            lib_cell_id,
            lib_arc.lib_arc_idx,
            lib_arc.senseToInt(),
            lib_arc.typeToInt(),
            lib_arc.lib_arc_offset,
            node_id,
            false,
        });
      }
    }

    std::vector<InstArcRecord> inst_arc_records;
    const auto arc_records_by_level = levelizeInstArcsForPython(
        clk2q_arc_records,
        comb_arc_records,
        nets,
        static_cast<int>(py::len(pydb.pin_names)));
    record_stage("setup_rawdb_pydb_export_inst_arc_collect_levelize_ms");

    std::vector<int> succ_degree(graph.size(), 0);
    std::vector<int> pred_degree(graph.size(), 0);
    auto count_graph_edge = [&](int from_pin, int to_pin) {
      if (from_pin < 0 || to_pin < 0) {
        return;
      }
      const auto from = static_cast<std::size_t>(from_pin);
      const auto to = static_cast<std::size_t>(to_pin);
      if (from >= graph.size() || to >= pred_degree.size()) {
        return;
      }
      succ_degree[from] += 1;
      pred_degree[to] += 1;
    };
    for (const auto& level_records : arc_records_by_level) {
      for (const auto& record : level_records) {
        if (!record.source_is_clock) {
          count_graph_edge(record.from_pin_id, record.to_pin_id);
        }
      }
    }
    for (const auto& net : nets) {
      const int driver_pin_id = net.driver_pin_id;
      for (const int pin_id : net.pin_ids) {
        if (pin_id != driver_pin_id) {
          count_graph_edge(driver_pin_id, pin_id);
        }
      }
    }
    for (std::size_t pin_id = 0; pin_id < graph.size(); ++pin_id) {
      graph[pin_id].reserve(succ_degree[pin_id]);
      succ_pins[pin_id].reserve(succ_degree[pin_id]);
      succ_arc_ids[pin_id].reserve(succ_degree[pin_id]);
      reverse_graph[pin_id].reserve(pred_degree[pin_id]);
      pred_pins[pin_id].reserve(pred_degree[pin_id]);
      pred_arc_ids[pin_id].reserve(pred_degree[pin_id]);
    }

    for (const auto& level_records : arc_records_by_level) {
      pydb.flat_inst_arcs_by_level_start.append(py::len(pydb.flat_inst_arcs_by_level));
      pydb.arc_level_start.append(py::len(pydb.flat_inst_arcs_by_level));
      pydb.inst_topo_start.append(py::len(pydb.inst_topo_ids));
      std::unordered_set<int> level_seen_insts;
      for (const auto& record : level_records) {
        const int arc_idx = static_cast<int>(inst_arc_records.size());
        inst_arc_records.push_back(record);
        appendLevelizedInstArc(pydb, record);
        if (record.inst_id >= 0 && record.inst_id < static_cast<int>(inst_to_arc_indices.size())) {
          inst_to_arc_indices[record.inst_id].push_back(arc_idx);
        }
        if (!record.source_is_clock) {
          add_graph_edge(record.from_pin_id, record.to_pin_id, arc_idx);
          pin_pair_to_arc_indices[{record.from_pin_id, record.to_pin_id}].push_back(arc_idx);
        }
        if (record.inst_id >= 0 && level_seen_insts.insert(record.inst_id).second) {
          pydb.inst_topo_ids.append(record.inst_id);
        }
      }
    }
    pydb.flat_inst_arcs_by_level_start.append(py::len(pydb.flat_inst_arcs_by_level));
    pydb.arc_level_start.append(py::len(pydb.flat_inst_arcs_by_level));
    pydb.inst_topo_start.append(py::len(pydb.inst_topo_ids));

    for (int node_id = 0; node_id < static_cast<int>(pydb.num_nodes); ++node_id) {
      pydb.inst_flat_arcs_start.append(py::len(pydb.inst_flat_arcs));
      if (node_id < static_cast<int>(inst_to_arc_indices.size())) {
        for (const int arc_idx : inst_to_arc_indices[node_id]) {
          pydb.inst_flat_arcs.append(arc_idx);
        }
      }
    }
    pydb.inst_flat_arcs_start.append(py::len(pydb.inst_flat_arcs));

    for (const auto& net : nets) {
      const int driver_pin_id = net.driver_pin_id;
      for (const int pin_id : net.pin_ids) {
        if (pin_id != driver_pin_id) {
          add_graph_edge(driver_pin_id, pin_id, -1);
        }
      }
    }

    appendPinGraphCsr(pydb.flat_pin_to_graph, pydb.flat_pin_to_graph_start, graph);
    appendPinGraphCsr(pydb.flat_pin_to_graph_reverse, pydb.flat_pin_to_graph_start_reverse, reverse_graph);
    appendPinArcCsr(pydb.pin_succ_start, pydb.pin_succ_pin, pydb.pin_succ_arc_id, succ_pins, succ_arc_ids);
    appendPinArcCsr(pydb.pin_pred_start, pydb.pin_pred_pin, pydb.pin_pred_arc_id, pred_pins, pred_arc_ids);

    int pair_arc_offset = 0;
    pydb.flat_pin_pair_arc_start.append(0);
    for (const auto& entry : pin_pair_to_arc_indices) {
      py::list key;
      key.append(entry.first.first);
      key.append(entry.first.second);
      pydb.pin_pair_arc_keys.append(key);
      for (const int arc_idx : entry.second) {
        pydb.flat_pin_pair_arc_indices.append(arc_idx);
        pair_arc_offset += 1;
      }
      pydb.flat_pin_pair_arc_start.append(pair_arc_offset);
    }
    if (pin_pair_to_arc_indices.empty()) {
      pydb.flat_pin_pair_arc_start = py::list();
      pydb.flat_pin_pair_arc_start.append(0);
    }

    for (const auto& start_pin : pydb.start_points) {
      pydb.start_pin_ids.append(start_pin);
    }
    for (const auto& endpoint_pin : pydb.end_points) {
      pydb.endpoint_pin_ids.append(endpoint_pin);
    }
    for (const auto& node_id : pydb.pin2node_map) {
      pydb.pin_to_node_id.append(node_id);
      pydb.pin_to_inst_id.append(node_id);
    }
    record_stage("setup_rawdb_pydb_export_graph_csr_ms");

    appendDiffGuidedBatchConflictGroups(pydb, pins, nets, arc_records_by_level);
    record_stage("setup_rawdb_pydb_export_conflict_groups_ms");
  }

  assert(py::len(pydb.node_names) == pydb.num_nodes);
  assert(py::len(pydb.node_master_names) == pydb.num_nodes);
  assert(py::len(pydb.node_is_buffer) == pydb.num_nodes);
  assert(py::len(pydb.node_is_hard_macro) == pydb.num_nodes);
  assert(py::len(pydb.macro_writeback_candidate) == pydb.num_nodes);
  assert(py::len(pydb.node2pin_map) == pydb.num_nodes);
  assert(py::len(pydb.flat_node2pin_start_map) == pydb.num_nodes + 1);
  assert(py::len(pydb.pin2node_map) == py::len(pydb.pin2net_map));
  assert(py::len(pydb.pin2node_map) == py::len(pydb.pin_names));
  assert(py::len(pydb.net_names) == py::len(pydb.net2pin_map));
  assert(py::len(pydb.net_names) == py::len(pydb.flat_net2pin_start_map) - 1);
  record_stage("setup_rawdb_pydb_export_asserts_ms");
  pydb.export_profile["setup_rawdb_pydb_export_total_cpp_ms"] = elapsedMs(export_begin);

  return pydb;
}

PyPlaceDB OpenRoadPlaceIOBridgeImpl::exportPyDBView()
{
  return buildPyDB(*this);
}

PyPlaceDB OpenRoadPlaceIOBridgeImpl::syncFromOpenRoad()
{
  return exportPyDBView();
}

}  // namespace impl
}  // namespace placeio_openroad
}  // namespace dreamplace
