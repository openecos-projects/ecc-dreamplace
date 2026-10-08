#pragma once

#include "openroad_place_io_impl_shared.h"

namespace dreamplace {
namespace placeio_openroad {
namespace impl {

class OpenRoadPlaceIOBridgeImpl
{
 public:
  OpenRoadPlaceIOBridgeImpl(const std::vector<std::string>& lef_files,
                            const std::string& def_file,
                            const std::vector<std::string>& liberty_files,
                            const std::string& sdc_file,
                            const std::vector<std::string>& vt_suffixes,
                            int thread_count)
      : runtime_(OpenRoadRuntime::instance()),
        tech_(std::make_unique<ord::Tech>()),
        design_(std::make_unique<ord::Design>(tech_.get())),
        vt_suffixes_(vt_suffixes)
  {
    if (lef_files.empty()) {
      throw std::runtime_error("placeio_openroad requires at least one LEF file");
    }
    if (def_file.empty()) {
      throw std::runtime_error("placeio_openroad requires a DEF file");
    }
    if (tech_ == nullptr || design_ == nullptr) {
      throw std::runtime_error("OpenROAD failed to create Tech/Design session");
    }

    logger_ = design_->getLogger();
    if (logger_ == nullptr) {
      throw std::runtime_error("OpenROAD bridge did not receive a logger");
    }
    auto* openroad = design_->getOpenRoad();
    if (openroad == nullptr) {
      throw std::runtime_error("OpenROAD bridge has no OpenROAD session");
    }
    thread_count = std::max(1, thread_count);
    omp_set_num_threads(thread_count);
    openroad->setThreadCount(thread_count, false);
    has_timing_inputs_ = !liberty_files.empty();

    for (const auto& lef : lef_files) {
      tech_->readLef(lef);
    }

    for (const auto& liberty : liberty_files) {
      tech_->readLiberty(liberty);
    }

    design_->readDef(def_file);
    if (!sdc_file.empty()) {
      design_->evalTclString("read_sdc " + tclQuote(sdc_file));
    }

    block_ = design_->getBlock();
    if (block_ == nullptr) {
      throw std::runtime_error("OpenROAD failed to create a dbBlock from the input DEF");
    }
    rebuildNodeInstIndex();
  }

  odb::dbBlock* block() const { return block_; }
  ord::Design* design() const { return design_.get(); }
  int lefUnit() const { return block_->getDbUnitsPerMicron(); }
  int defUnit() const { return block_->getDefUnits(); }
  bool hasTimingInputs() const { return has_timing_inputs_; }
  const std::vector<std::string>& vtSuffixes() const { return vt_suffixes_; }
  PyPlaceDB exportPyDBView();
  PyPlaceDB syncFromOpenRoad();
  std::string evalTclString(const std::string& cmd) { return design_->evalTclString(cmd); }

  py::dict refreshTiming()
  {
    py::dict result;
    result["artifact"] = "openroad_timing_refresh";
    result["artifact_version"] = 1;
    result["method"] = "sta::dbSta::updateTiming(false)";
    result["has_timing_inputs"] = has_timing_inputs_;
    if (!has_timing_inputs_) {
      result["status"] = "no_timing_inputs";
      result["elapsed_ms"] = 0.0;
      return result;
    }

    const auto begin = std::chrono::steady_clock::now();
    try {
      auto* sta = tech_->getSta();
      if (sta == nullptr) {
        result["status"] = "missing_sta";
        result["elapsed_ms"] = elapsedMs(begin);
        return result;
      }
      sta->updateTiming(false);
      result["status"] = "ok";
    } catch (const std::exception& exc) {
      result["status"] = "failed";
      result["error"] = exc.what();
    }
    result["elapsed_ms"] = elapsedMs(begin);
    return result;
  }

  void refreshTimingOrThrow()
  {
    const py::dict result = refreshTiming();
    const std::string status = pyStringOrDefault(result, "status", "failed");
    if (status != "ok" && status != "no_timing_inputs") {
      throw std::runtime_error(
          "OpenSTA timing refresh failed: "
          + pyStringOrDefault(result, "error", status));
    }
  }

  py::dict summarizeLibertyTableAxis(const sta::TableAxis* axis,
                                      const sta::Units* units,
                                      bool force_time_unit)
  {
    py::dict result;
    result["present"] = axis != nullptr;
    if (axis == nullptr) {
      return result;
    }
    result["variable"] = sta::tableVariableString(axis->variable());
    result["size"] = static_cast<int>(axis->size());
    const sta::Unit* unit = force_time_unit ? (units == nullptr ? nullptr : units->timeUnit())
                                            : sta::tableVariableUnit(axis->variable(), units);
    py::list raw_values;
    py::list user_values;
    auto* values = axis->values();
    if (values != nullptr) {
      for (const float value : *values) {
        raw_values.append(value);
        user_values.append(unit == nullptr ? static_cast<double>(value)
                                           : const_cast<sta::Unit*>(unit)->staToUser(value));
      }
    }
    result["raw_values"] = raw_values;
    result["user_values"] = user_values;
    return result;
  }

  py::dict summarizeLibertyTableModel(const sta::TableModel* model,
                                      const sta::Units* units)
  {
    py::dict result;
    result["present"] = model != nullptr;
    if (model == nullptr) {
      return result;
    }
    result["order"] = model->order();
    result["axis1"] = summarizeLibertyTableAxis(model->axis1(), units, false);
    result["axis2"] = summarizeLibertyTableAxis(model->axis2(), units, false);
    result["axis3"] = summarizeLibertyTableAxis(model->axis3(), units, false);
    py::list corner_values;
    if (model->order() > 0) {
      const std::size_t dim1 = model->axis1() == nullptr ? 1 : model->axis1()->size();
      const std::size_t dim2 = model->axis2() == nullptr ? 1 : model->axis2()->size();
      const std::size_t dim3 = model->axis3() == nullptr ? 1 : model->axis3()->size();
      const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
      const std::size_t index1_values[] = {0, dim1 == 0 ? 0 : dim1 - 1};
      const std::size_t index2_values[] = {0, dim2 == 0 ? 0 : dim2 - 1};
      const std::size_t index3_values[] = {0, dim3 == 0 ? 0 : dim3 - 1};
      for (const std::size_t i : index1_values) {
        for (const std::size_t j : index2_values) {
          for (const std::size_t k : index3_values) {
            py::dict row;
            row["i"] = static_cast<int>(i);
            row["j"] = static_cast<int>(j);
            row["k"] = static_cast<int>(k);
            const float raw_value = model->value(i, j, k);
            row["raw_value"] = raw_value;
            row["user_value"] =
                time_unit == nullptr
                    ? static_cast<double>(raw_value)
                    : const_cast<sta::Unit*>(time_unit)->staToUser(raw_value);
            corner_values.append(row);
          }
        }
      }
    }
    result["corner_values"] = corner_values;
    return result;
  }

  std::string runBufferInsertion(const std::string& cmd)
  {
    if (cmd.empty()) {
      throw std::runtime_error("buffer insertion command must not be empty");
    }
    auto result = design_->evalTclString(cmd);
    rebuildNodeInstIndex();
    return result;
  }

  py::dict runOneNetBuffer(const std::string& net_name, const py::dict& config)
  {
    py::dict summary;
    summary["artifact"] = "buffer_insertion_one_net_result";
    summary["artifact_version"] = 1;
    summary["net_name"] = net_name;
    summary["realization_backend"] = "openroad_rsz_repair_net_cmd";
    summary["command"] = "";
    summary["has_timing_inputs"] = has_timing_inputs_;
    summary["requires_sync_back"] = false;
    summary["requires_runtimedb_rebuild"] = false;
    summary["inserted_buffer_count_delta"] = 0;
    summary["instance_count_delta"] = 0;
    summary["net_count_delta"] = 0;
    summary["actual_delta_wns"] = 0.0;
    summary["actual_delta_tns"] = 0.0;
    summary["setup_violation_count_delta"] = 0;
    summary["tcl_output"] = "";
    summary["error"] = "";

    const auto trim = [](const std::string& value) {
      const auto begin = value.find_first_not_of(" \t\n\r");
      if (begin == std::string::npos) {
        return std::string();
      }
      const auto end = value.find_last_not_of(" \t\n\r");
      return value.substr(begin, end - begin + 1);
    };
    const std::string trimmed_net_name = trim(net_name);
    if (trimmed_net_name.empty()) {
      summary["status"] = "failed";
      summary["error"] = "one-net buffer insertion requires a net name";
      return summary;
    }
    if (design_ == nullptr || block_ == nullptr) {
      summary["status"] = "failed";
      summary["error"] = "OpenROAD bridge is not initialized";
      return summary;
    }

    const auto instance_count = [&]() {
      return block_ == nullptr ? 0 : static_cast<int>(block_->getInsts().size());
    };
    const auto net_count = [&]() {
      return block_ == nullptr ? 0 : static_cast<int>(block_->getNets().size());
    };
    const auto buffer_instance_count = [&]() {
      int count = 0;
      if (design_ == nullptr || block_ == nullptr) {
        return count;
      }
      for (auto* inst : block_->getInsts()) {
        auto* master = inst == nullptr ? nullptr : inst->getMaster();
        if (master != nullptr && design_->isBuffer(master)) {
          count += 1;
        }
      }
      return count;
    };

    const double max_wire_length = pyDoubleOrDefault(config, "max_wire_length", 0.0);
    const double slew_margin = pyDoubleOrDefault(config, "slew_margin", 0.0);
    const double cap_margin = pyDoubleOrDefault(config, "cap_margin", 0.0);
    std::ostringstream command_stream;
    command_stream << "rsz::repair_net_cmd [get_net " << tclQuote(trimmed_net_name)
                   << "] " << max_wire_length << " " << slew_margin << " "
                   << cap_margin;
    const std::string command = command_stream.str();
    summary["command"] = command;

    try {
      const py::dict before_metrics = queryDiffGuidedBatchTimingMetrics();
      const int before_instance_count = instance_count();
      const int before_net_count = net_count();
      const int before_buffer_count = buffer_instance_count();

      const auto mutation_begin = std::chrono::steady_clock::now();
      const std::string tcl_output = design_->evalTclString(command);
      summary["mutation_ms"] = elapsedMs(mutation_begin);
      summary["tcl_output"] = tcl_output;

      rebuildNodeInstIndex();

      const py::dict after_metrics = queryDiffGuidedBatchTimingMetrics();
      const int after_instance_count = instance_count();
      const int after_net_count = net_count();
      const int after_buffer_count = buffer_instance_count();
      const int instance_delta = after_instance_count - before_instance_count;
      const int net_delta = after_net_count - before_net_count;
      const int buffer_delta = after_buffer_count - before_buffer_count;

      const double before_wns = pyDoubleOrDefault(before_metrics, "wns", 0.0);
      const double before_tns = pyDoubleOrDefault(before_metrics, "tns", 0.0);
      const int before_setup_vio =
          pyIntOrDefault(before_metrics, "setup_violation_count", 0);
      const double after_wns = pyDoubleOrDefault(after_metrics, "wns", before_wns);
      const double after_tns = pyDoubleOrDefault(after_metrics, "tns", before_tns);
      const int after_setup_vio =
          pyIntOrDefault(after_metrics, "setup_violation_count", before_setup_vio);

      summary["status"] = "ok";
      summary["before_metrics"] = before_metrics;
      summary["after_metrics"] = after_metrics;
      summary["actual_delta_wns"] = after_wns - before_wns;
      summary["actual_delta_tns"] = after_tns - before_tns;
      summary["setup_violation_count_delta"] = after_setup_vio - before_setup_vio;
      summary["inserted_buffer_count_delta"] = buffer_delta;
      summary["instance_count_delta"] = instance_delta;
      summary["net_count_delta"] = net_delta;
      const bool requires_rebuild =
          instance_delta != 0 || net_delta != 0 || buffer_delta != 0;
      summary["requires_sync_back"] = requires_rebuild;
      summary["requires_runtimedb_rebuild"] = requires_rebuild;
    } catch (const std::exception& exc) {
      summary["status"] = "failed";
      summary["error"] = exc.what();
      summary["requires_sync_back"] = false;
      summary["requires_runtimedb_rebuild"] = false;
    }
    return summary;
  }

  py::dict runCoordinateBufferInsert(const py::dict& action, const py::dict& config)
  {
    py::dict summary;
    summary["artifact"] = "buffer_insertion_coordinate_result";
    summary["artifact_version"] = 1;
    summary["status"] = "unsupported";
    summary["realization_backend"] = "openroad_coordinate_buffer_insert";
    summary["implementation_scope"] = "flat_singleton_load_partition";
    summary["net_name"] = pyStringOrDefault(action, "net_name", "");
    summary["buffer_master_name"] = pyStringOrDefault(action, "buffer_master_name", "");
    summary["buffer_main_type_index"] = pyIntOrDefault(action, "buffer_main_type_index", -1);
    summary["bsu"] = pyIntOrDefault(action, "bsu", -1);
    summary["driver_pin_name"] = pyStringOrDefault(action, "driver_pin_name", "");
    summary["load_pin_name"] = pyStringOrDefault(action, "load_pin_name", "");
    summary["load_partition_mode"] = pyStringOrDefault(config, "load_partition_mode", "singleton");
    summary["moved_load_count"] = 0;
    summary["requested_downstream_pin_count"] = 0;
    summary["moved_load_pin_names"] = py::list();
    summary["inserted_buffer_name"] = "";
    summary["downstream_net_name"] = "";
    summary["hard_commit_diagnostic"] = py::dict();
    summary["hard_commit_diagnostic_status"] = "missing";
    summary["hard_commit_topology_delta"] = py::dict();
    summary["hard_commit_load_partition"] = py::dict();
    summary["hard_commit_cell_arc_delay_samples"] = py::dict();
    summary["has_timing_inputs"] = has_timing_inputs_;
    summary["requires_sync_back"] = false;
    summary["requires_runtimedb_rebuild"] = false;
    summary["inserted_buffer_count_delta"] = 0;
    summary["instance_count_delta"] = 0;
    summary["net_count_delta"] = 0;
    summary["actual_delta_wns"] = 0.0;
    summary["actual_delta_tns"] = 0.0;
    summary["setup_violation_count_delta"] = 0;
    summary["slew_violation_count_delta"] = 0.0;
    summary["cap_violation_count_delta"] = 0.0;
    summary["unsupported_reasons"] = py::list();
    summary["error"] = "";

    auto unsupported = [&](const std::string& reason) {
      py::list reasons;
      reasons.append(reason);
      summary["status"] = "unsupported";
      summary["unsupported_reasons"] = reasons;
      summary["error"] = reason;
      return summary;
    };
    auto failed = [&](const std::string& reason) {
      summary["status"] = "failed";
      summary["error"] = reason;
      return summary;
    };

    const std::string action_kind = pyStringOrDefault(action, "action_kind", "buffer_insert");
    if (action_kind != "buffer_insert") {
      return unsupported("only_buffer_insert_action_supported");
    }
    const std::string load_partition_mode =
        pyStringOrDefault(config, "load_partition_mode", "singleton");
    const bool defer_timing_update = pyBoolOrDefault(config, "defer_timing_update", false);
    summary["defer_timing_update"] = defer_timing_update;
    if (load_partition_mode == "singleton"
        && !pyBoolOrDefault(config, "allow_singleton_load_partition", false)) {
      return unsupported("singleton_load_partition_requires_explicit_allow");
    }
    if (load_partition_mode != "singleton" && load_partition_mode != "all_loads"
        && load_partition_mode != "branch_downstream") {
      return unsupported("unsupported_load_partition_mode");
    }
    if (load_partition_mode == "branch_downstream") {
      summary["implementation_scope"] = "flat_explicit_downstream_load_partition";
    }
    if (design_ == nullptr || block_ == nullptr) {
      return failed("OpenROAD bridge is not initialized");
    }

    const std::string net_name = pyStringOrDefault(action, "net_name", "");
    const std::string buffer_master_name =
        pyStringOrDefault(action, "buffer_master_name", "");
    const std::string load_pin_name = pyStringOrDefault(action, "load_pin_name", "");
    if (net_name.empty()) {
      return failed("coordinate buffer insertion requires net_name");
    }
    if (buffer_master_name.empty()) {
      return failed("coordinate buffer insertion requires buffer_master_name");
    }
    if (load_pin_name.empty()) {
      return failed("coordinate buffer insertion requires load_pin_name");
    }

    odb::dbNet* source_net = block_->findNet(net_name.c_str());
    if (source_net == nullptr) {
      return failed("net not found: " + net_name);
    }
    auto* db = block_->getDataBase();
    if (db == nullptr) {
      return failed("OpenROAD bridge has no dbDatabase");
    }
    odb::dbMaster* buffer_master = db->findMaster(buffer_master_name.c_str());
    if (buffer_master == nullptr) {
      return failed("buffer master not found: " + buffer_master_name);
    }

    auto find_pin_by_name = [&](const std::string& pin_name)
        -> std::pair<odb::dbITerm*, odb::dbBTerm*> {
      if (pin_name.empty()) {
        return {nullptr, nullptr};
      }
      const auto colon = pin_name.rfind(':');
      const auto slash = pin_name.rfind('/');
      const auto separator = colon == std::string::npos ? slash : colon;
      if (separator != std::string::npos) {
        const std::string inst_name = pin_name.substr(0, separator);
        const std::string port_name = pin_name.substr(separator + 1);
        auto* inst = block_->findInst(inst_name.c_str());
        auto* iterm = inst == nullptr ? nullptr : inst->findITerm(port_name.c_str());
        return {iterm, nullptr};
      }
      return {nullptr, block_->findBTerm(pin_name.c_str())};
    };

    auto downstream_pin_names_from_action = [&]() {
      std::vector<std::string> names;
      if (!action.contains("downstream_pin_names")
          || action["downstream_pin_names"].is_none()) {
        return names;
      }
      try {
        for (auto item : action["downstream_pin_names"]) {
          std::string name = py::cast<std::string>(item);
          if (!name.empty()) {
            names.push_back(name);
          }
        }
      } catch (const std::exception&) {
        names.clear();
      }
      return names;
    };

    auto load_pin = find_pin_by_name(load_pin_name);
    odb::dbITerm* load_iterm = load_pin.first;
    odb::dbBTerm* load_bterm = load_pin.second;
    if (load_iterm == nullptr && load_bterm == nullptr) {
      return failed("load pin not found: " + load_pin_name);
    }
    odb::dbNet* load_net =
        load_iterm != nullptr ? load_iterm->getNet() : load_bterm->getNet();
    if (load_net == nullptr) {
      return failed("load pin is disconnected: " + load_pin_name);
    }
    odb::dbNet* net = load_net;
    const bool target_net_remapped = net != source_net;
    summary["source_net_name"] = source_net->getName();
    summary["resolved_target_net_name"] = net->getName();
    summary["target_net_remapped"] = target_net_remapped;

    const std::string driver_pin_name = pyStringOrDefault(action, "driver_pin_name", "");
    odb::dbITerm* driver_iterm = nullptr;
    odb::dbBTerm* driver_bterm = nullptr;
    std::string resolved_driver_pin_name = driver_pin_name;
    auto resolve_driver_on_net = [&](odb::dbNet* target_net) {
      driver_iterm = nullptr;
      driver_bterm = nullptr;
      for (auto* iterm : target_net->getITerms()) {
        if (iterm != nullptr && iterm->getIoType() == odb::dbIoType::OUTPUT) {
          driver_iterm = iterm;
          if (iterm->getInst() != nullptr && iterm->getMTerm() != nullptr) {
            resolved_driver_pin_name = iterm->getInst()->getName()
                                       + std::string(":")
                                       + iterm->getMTerm()->getName();
          }
          return true;
        }
      }
      for (auto* bterm : target_net->getBTerms()) {
        if (bterm != nullptr && bterm->getIoType() == odb::dbIoType::INPUT) {
          driver_bterm = bterm;
          resolved_driver_pin_name = bterm->getName();
          return true;
        }
      }
      return false;
    };
    if (!driver_pin_name.empty()) {
      auto driver_pin = find_pin_by_name(driver_pin_name);
      driver_iterm = driver_pin.first;
      driver_bterm = driver_pin.second;
      odb::dbNet* driver_net =
          driver_iterm != nullptr ? driver_iterm->getNet()
                                  : (driver_bterm == nullptr ? nullptr : driver_bterm->getNet());
      if (driver_net != net) {
        if (!target_net_remapped || !resolve_driver_on_net(net)) {
          return unsupported("driver_pin_not_connected_to_resolved_target_net");
        }
      }
    } else if (!resolve_driver_on_net(net)) {
      return unsupported("resolved_target_net_missing_driver_pin");
    }
    summary["resolved_driver_pin_name"] = resolved_driver_pin_name;
    summary["driver_pin_remapped"] = resolved_driver_pin_name != driver_pin_name;

    std::vector<odb::dbITerm*> load_iterms;
    std::vector<odb::dbBTerm*> load_bterms;
    std::vector<std::string> moved_load_pin_names;
    if (load_partition_mode == "all_loads") {
      for (auto* iterm : net->getITerms()) {
        if (iterm == nullptr || iterm == driver_iterm) {
          continue;
        }
        const auto io_type = iterm->getIoType();
        if (io_type == odb::dbIoType::INPUT || io_type == odb::dbIoType::INOUT) {
          load_iterms.push_back(iterm);
          moved_load_pin_names.push_back(
              iterm->getInst()->getName() + std::string("/") + iterm->getMTerm()->getName());
        }
      }
      for (auto* bterm : net->getBTerms()) {
        if (bterm == nullptr || bterm == driver_bterm) {
          continue;
        }
        const auto io_type = bterm->getIoType();
        if (io_type == odb::dbIoType::OUTPUT || io_type == odb::dbIoType::INOUT) {
          load_bterms.push_back(bterm);
          moved_load_pin_names.push_back(bterm->getName());
        }
      }
    } else if (load_partition_mode == "branch_downstream") {
      const auto downstream_pin_names = downstream_pin_names_from_action();
      summary["requested_downstream_pin_count"] =
          static_cast<int>(downstream_pin_names.size());
      if (downstream_pin_names.empty()) {
        return unsupported("branch_downstream_requires_explicit_downstream_pin_names");
      }
      std::unordered_set<std::string> seen_pin_names;
      for (const auto& downstream_pin_name : downstream_pin_names) {
        if (!seen_pin_names.insert(downstream_pin_name).second) {
          continue;
        }
        auto downstream_pin = find_pin_by_name(downstream_pin_name);
        odb::dbITerm* downstream_iterm = downstream_pin.first;
        odb::dbBTerm* downstream_bterm = downstream_pin.second;
        if (downstream_iterm == nullptr && downstream_bterm == nullptr) {
          return failed("downstream pin not found: " + downstream_pin_name);
        }
        if ((downstream_iterm != nullptr && downstream_iterm == driver_iterm)
            || (downstream_bterm != nullptr && downstream_bterm == driver_bterm)) {
          return unsupported("branch_downstream_partition_includes_driver_pin");
        }
        odb::dbNet* downstream_net =
            downstream_iterm != nullptr ? downstream_iterm->getNet()
                                        : downstream_bterm->getNet();
        if (downstream_net != net) {
          return unsupported("downstream_pin_not_connected_to_target_net");
        }
        if (downstream_iterm != nullptr) {
          const auto io_type = downstream_iterm->getIoType();
          if (io_type != odb::dbIoType::INPUT && io_type != odb::dbIoType::INOUT) {
            return unsupported("downstream_iterm_is_not_load_pin");
          }
          load_iterms.push_back(downstream_iterm);
        } else {
          const auto io_type = downstream_bterm->getIoType();
          if (io_type != odb::dbIoType::OUTPUT && io_type != odb::dbIoType::INOUT) {
            return unsupported("downstream_bterm_is_not_load_pin");
          }
          load_bterms.push_back(downstream_bterm);
        }
        moved_load_pin_names.push_back(downstream_pin_name);
      }
    } else if (load_iterm != nullptr) {
      load_iterms.push_back(load_iterm);
      moved_load_pin_names.push_back(load_pin_name);
    } else {
      load_bterms.push_back(load_bterm);
      moved_load_pin_names.push_back(load_pin_name);
    }
    if (load_iterms.empty() && load_bterms.empty()) {
      return unsupported("empty_downstream_load_partition");
    }

    odb::dbITerm* buffer_input = nullptr;
    odb::dbITerm* buffer_output = nullptr;
    auto choose_buffer_pins = [&](odb::dbInst* inst) {
      for (auto* iterm : inst->getITerms()) {
        if (iterm == nullptr) {
          continue;
        }
        const auto io_type = iterm->getIoType();
        if (buffer_input == nullptr && io_type == odb::dbIoType::INPUT) {
          buffer_input = iterm;
        } else if (buffer_output == nullptr && io_type == odb::dbIoType::OUTPUT) {
          buffer_output = iterm;
        }
      }
    };

    auto clean_name = [](const std::string& raw) {
      std::string cleaned;
      cleaned.reserve(raw.size());
      for (char ch : raw) {
        const bool keep = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
                          || (ch >= '0' && ch <= '9') || ch == '_';
        cleaned.push_back(keep ? ch : '_');
      }
      return cleaned.empty() ? std::string("net") : cleaned;
    };
    auto make_unique_inst_name = [&](const std::string& base) {
      std::string name = base;
      int suffix = 0;
      while (block_->findInst(name.c_str()) != nullptr) {
        ++suffix;
        name = base + "_" + std::to_string(suffix);
      }
      return name;
    };
    auto make_unique_net_name = [&](const std::string& base) {
      std::string name = base;
      int suffix = 0;
      while (block_->findNet(name.c_str()) != nullptr) {
        ++suffix;
        name = base + "_" + std::to_string(suffix);
      }
      return name;
    };
    const int action_id = pyIntOrDefault(action, "action_id", 0);
    const std::string name_seed =
        std::to_string(action_id) + "_" + clean_name(net_name);
    const std::string buffer_inst_name =
        make_unique_inst_name("buffer_coord_buf_" + name_seed);
    const std::string downstream_net_name =
        make_unique_net_name(net_name + "_buffer_coord_" + std::to_string(action_id));

    const auto instance_count = [&]() {
      return block_ == nullptr ? 0 : static_cast<int>(block_->getInsts().size());
    };
    const auto net_count = [&]() {
      return block_ == nullptr ? 0 : static_cast<int>(block_->getNets().size());
    };
    const auto buffer_instance_count = [&]() {
      int count = 0;
      if (design_ == nullptr || block_ == nullptr) {
        return count;
      }
      for (auto* inst : block_->getInsts()) {
        auto* master = inst == nullptr ? nullptr : inst->getMaster();
        if (master != nullptr && design_->isBuffer(master)) {
          count += 1;
        }
      }
      return count;
    };
    const auto net_iterm_count = [](odb::dbNet* target_net) {
      return target_net == nullptr ? 0 : static_cast<int>(target_net->getITerms().size());
    };
    const auto net_bterm_count = [](odb::dbNet* target_net) {
      return target_net == nullptr ? 0 : static_cast<int>(target_net->getBTerms().size());
    };
    const auto net_load_pin_count = [&](odb::dbNet* target_net) {
      int count = 0;
      if (target_net == nullptr) {
        return count;
      }
      for (auto* iterm : target_net->getITerms()) {
        if (iterm == nullptr || iterm == driver_iterm) {
          continue;
        }
        const auto io_type = iterm->getIoType();
        if (io_type == odb::dbIoType::INPUT || io_type == odb::dbIoType::INOUT) {
          count += 1;
        }
      }
      for (auto* bterm : target_net->getBTerms()) {
        if (bterm == nullptr || bterm == driver_bterm) {
          continue;
        }
        const auto io_type = bterm->getIoType();
        if (io_type == odb::dbIoType::OUTPUT || io_type == odb::dbIoType::INOUT) {
          count += 1;
        }
      }
      return count;
    };
    const auto iterm_name = [](odb::dbITerm* iterm) {
      if (iterm == nullptr || iterm->getInst() == nullptr || iterm->getMTerm() == nullptr) {
        return std::string();
      }
      return iterm->getInst()->getName() + std::string("/") + iterm->getMTerm()->getName();
    };
    const auto append_unique_pin_name =
        [](std::vector<std::string>& pins,
           std::unordered_set<std::string>& seen,
           const std::string& pin_name) {
          if (pin_name.empty() || !seen.insert(pin_name).second) {
            return;
          }
          pins.push_back(pin_name);
        };
    auto build_pin_sample_names = [&]() {
      std::vector<std::string> names;
      std::unordered_set<std::string> seen;
      append_unique_pin_name(names, seen, driver_pin_name);
      append_unique_pin_name(names, seen, load_pin_name);
      for (const auto& moved_name : moved_load_pin_names) {
        append_unique_pin_name(names, seen, moved_name);
      }
      return names;
    };
    auto build_pin_timing_samples = [&](const std::vector<std::string>& pin_names) {
      py::dict samples;
      samples["schema_name"] = "buffering_hard_commit_pin_timing_samples";
      samples["schema_version"] = 1;
      samples["status"] = "ok";
      samples["has_timing_inputs"] = has_timing_inputs_;
      py::list pin_samples;
      if (!has_timing_inputs_) {
        samples["status"] = "no_timing_inputs";
        samples["pin_samples"] = pin_samples;
        samples["pin_sample_count"] = 0;
        return samples;
      }
      try {
        refreshTimingOrThrow();
        ord::Timing timing(design_.get());
        auto* sta = timing.getSta();
        if (sta == nullptr) {
          samples["status"] = "missing_sta";
          samples["pin_samples"] = pin_samples;
          samples["pin_sample_count"] = 0;
          return samples;
        }
        const sta::Units* units = sta->units();
        const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
        const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
        sta::Corner* corner = timing.cmdCorner();
        for (const auto& pin_name : pin_names) {
          pin_samples.append(
              buildDiffGuidedBatchPinSample(timing, time_unit, cap_unit, corner, pin_name));
        }
        samples["pin_samples"] = pin_samples;
        samples["pin_sample_count"] = py::len(pin_samples);
      } catch (const std::exception& exc) {
        samples["status"] = "failed";
        samples["error"] = exc.what();
        samples["pin_samples"] = pin_samples;
        samples["pin_sample_count"] = py::len(pin_samples);
      }
      return samples;
    };
    auto build_pin_sample_map = [](const py::dict& samples) {
      std::unordered_map<std::string, py::dict> result;
      if (!samples.contains("pin_samples") || samples["pin_samples"].is_none()) {
        return result;
      }
      try {
        for (auto item : samples["pin_samples"]) {
          py::dict sample = py::cast<py::dict>(item);
          const std::string name = pyStringOrDefault(sample, "name", "");
          if (!name.empty()) {
            result[name] = sample;
          }
        }
      } catch (const std::exception&) {
        result.clear();
      }
      return result;
    };
    auto sample_numeric_value = [](const py::dict& sample,
                                   const std::string& field,
                                   double& value) {
      if (!sample.contains(field.c_str()) || sample[field.c_str()].is_none()) {
        return false;
      }
      try {
        value = py::cast<double>(sample[field.c_str()]);
      } catch (const std::exception&) {
        return false;
      }
      return std::isfinite(value);
    };
    auto build_pin_timing_sample_delta = [&](const py::dict& before_samples,
                                             const py::dict& after_samples) {
      py::dict delta;
      delta["schema_name"] = "buffering_hard_commit_pin_timing_sample_delta";
      delta["schema_version"] = 1;
      py::list rows;
      py::list added_after_pin_names;
      py::list missing_after_pin_names;
      const auto before_by_name = build_pin_sample_map(before_samples);
      const auto after_by_name = build_pin_sample_map(after_samples);
      int common_pin_count = 0;
      int numeric_delta_count = 0;
      double max_abs_cap_delta = 0.0;
      double max_abs_slew_delta = 0.0;
      for (const auto& [name, before_sample] : before_by_name) {
        auto after_iter = after_by_name.find(name);
        if (after_iter == after_by_name.end()) {
          missing_after_pin_names.append(name);
          continue;
        }
        common_pin_count += 1;
        const py::dict& after_sample = after_iter->second;
        py::dict row;
        row["name"] = name;
        row["mapped_before"] = pyBoolOrDefault(before_sample, "mapped", false);
        row["mapped_after"] = pyBoolOrDefault(after_sample, "mapped", false);
        for (const std::string field :
             {"opensta_slew", "opensta_arrival", "opensta_slack", "opensta_cap"}) {
          double before_value = 0.0;
          double after_value = 0.0;
          const bool has_before = sample_numeric_value(before_sample, field, before_value);
          const bool has_after = sample_numeric_value(after_sample, field, after_value);
          row[(field + "_before").c_str()] = has_before ? py::cast(before_value) : py::none();
          row[(field + "_after").c_str()] = has_after ? py::cast(after_value) : py::none();
          if (has_before && has_after) {
            const double value_delta = after_value - before_value;
            row[(field + "_delta").c_str()] = value_delta;
            numeric_delta_count += 1;
            if (field == "opensta_cap") {
              max_abs_cap_delta = std::max(max_abs_cap_delta, std::abs(value_delta));
            } else if (field == "opensta_slew") {
              max_abs_slew_delta = std::max(max_abs_slew_delta, std::abs(value_delta));
            }
          } else {
            row[(field + "_delta").c_str()] = py::none();
          }
        }
        rows.append(row);
      }
      for (const auto& [name, after_sample] : after_by_name) {
        if (before_by_name.find(name) == before_by_name.end()) {
          added_after_pin_names.append(name);
        }
      }
      delta["status"] = "ok";
      delta["common_pin_count"] = common_pin_count;
      delta["numeric_delta_count"] = numeric_delta_count;
      delta["max_abs_opensta_cap_delta"] = max_abs_cap_delta;
      delta["max_abs_opensta_slew_delta"] = max_abs_slew_delta;
      delta["missing_after_pin_names"] = missing_after_pin_names;
      delta["added_after_pin_names"] = added_after_pin_names;
      delta["rows"] = rows;
      return delta;
    };
    auto build_cell_arc_delay_samples = [&](odb::dbInst* buffer_inst,
                                            odb::dbITerm* input_iterm,
                                            odb::dbITerm* output_iterm,
                                            odb::dbNet* output_net) {
      py::dict samples;
      samples["schema_name"] = "buffering_hard_commit_cell_arc_delay_samples";
      samples["schema_version"] = 1;
      samples["status"] = "ok";
      samples["source"] = "opensta_pin_arrival_and_liberty_gate_table";
      samples["has_timing_inputs"] = has_timing_inputs_;
      samples["buffer_master_name"] = buffer_master_name;
      samples["buffer_input_pin_name"] = iterm_name(input_iterm);
      samples["buffer_output_pin_name"] = iterm_name(output_iterm);
      samples["downstream_net_name"] =
          output_net == nullptr ? std::string() : output_net->getName();
      py::list arc_samples;
      samples["liberty_arc_samples"] = arc_samples;
      samples["liberty_arc_sample_count"] = 0;
      auto set_optional_double = [](py::dict& target,
                                    const char* key,
                                    bool has_value,
                                    double value) {
        if (has_value) {
          target[key] = value;
        } else {
          target[key] = py::none();
        }
      };
      if (!has_timing_inputs_) {
        samples["status"] = "no_timing_inputs";
        return samples;
      }
      if (buffer_inst == nullptr || input_iterm == nullptr || output_iterm == nullptr) {
        samples["status"] = "missing_buffer_pin";
        return samples;
      }
      try {
        refreshTimingOrThrow();
        ord::Timing timing(design_.get());
        auto* sta = timing.getSta();
        auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
        if (sta == nullptr || network == nullptr) {
          samples["status"] = "missing_sta_network";
          return samples;
        }
        const sta::Units* units = sta->units();
        const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
        const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
        sta::Corner* corner = timing.cmdCorner();
        const double input_r_slew =
            timing.getPinSlew(input_iterm, ord::Timing::Rise, ord::Timing::Max);
        const double input_f_slew =
            timing.getPinSlew(input_iterm, ord::Timing::Fall, ord::Timing::Max);
        const double output_r_slew =
            timing.getPinSlew(output_iterm, ord::Timing::Rise, ord::Timing::Max);
        const double output_f_slew =
            timing.getPinSlew(output_iterm, ord::Timing::Fall, ord::Timing::Max);
        const double input_r_arrival =
            timing.getPinArrival(input_iterm, ord::Timing::Rise, ord::Timing::Max);
        const double input_f_arrival =
            timing.getPinArrival(input_iterm, ord::Timing::Fall, ord::Timing::Max);
        const double output_r_arrival =
            timing.getPinArrival(output_iterm, ord::Timing::Rise, ord::Timing::Max);
        const double output_f_arrival =
            timing.getPinArrival(output_iterm, ord::Timing::Fall, ord::Timing::Max);
        const bool has_input_r_slew = staScalarIsUsable(input_r_slew);
        const bool has_input_f_slew = staScalarIsUsable(input_f_slew);
        const bool has_output_r_slew = staScalarIsUsable(output_r_slew);
        const bool has_output_f_slew = staScalarIsUsable(output_f_slew);
        const bool has_input_r_arrival = staScalarIsUsable(input_r_arrival);
        const bool has_input_f_arrival = staScalarIsUsable(input_f_arrival);
        const bool has_output_r_arrival = staScalarIsUsable(output_r_arrival);
        const bool has_output_f_arrival = staScalarIsUsable(output_f_arrival);
        const double input_slew =
            std::max(has_input_r_slew ? input_r_slew : 0.0,
                     has_input_f_slew ? input_f_slew : 0.0);
        const bool has_input_slew = has_input_r_slew || has_input_f_slew;
        const bool has_output_net_cap = corner != nullptr && output_net != nullptr;
        const double output_net_cap_pf = has_output_net_cap
                                             ? staCapToPfOrDefault(
                                                   cap_unit,
                                                   timing.getNetCap(
                                                       output_net, corner, ord::Timing::Max),
                                                   0.0)
                                             : 0.0;

        py::dict pin_arrival_delta;
        pin_arrival_delta["schema_name"] = "buffering_hard_commit_cell_arc_pin_arrival_delta";
        pin_arrival_delta["schema_version"] = 1;
        set_optional_double(pin_arrival_delta,
                            "input_r_slew",
                            has_input_r_slew,
                            staTimeToPsOrDefault(time_unit, input_r_slew, 0.0));
        set_optional_double(pin_arrival_delta,
                            "input_f_slew",
                            has_input_f_slew,
                            staTimeToPsOrDefault(time_unit, input_f_slew, 0.0));
        set_optional_double(pin_arrival_delta,
                            "output_r_slew",
                            has_output_r_slew,
                            staTimeToPsOrDefault(time_unit, output_r_slew, 0.0));
        set_optional_double(pin_arrival_delta,
                            "output_f_slew",
                            has_output_f_slew,
                            staTimeToPsOrDefault(time_unit, output_f_slew, 0.0));
        set_optional_double(pin_arrival_delta,
                            "input_r_arrival",
                            has_input_r_arrival,
                            staTimeToPsOrDefault(time_unit, input_r_arrival, 0.0));
        set_optional_double(pin_arrival_delta,
                            "input_f_arrival",
                            has_input_f_arrival,
                            staTimeToPsOrDefault(time_unit, input_f_arrival, 0.0));
        set_optional_double(pin_arrival_delta,
                            "output_r_arrival",
                            has_output_r_arrival,
                            staTimeToPsOrDefault(time_unit, output_r_arrival, 0.0));
        set_optional_double(pin_arrival_delta,
                            "output_f_arrival",
                            has_output_f_arrival,
                            staTimeToPsOrDefault(time_unit, output_f_arrival, 0.0));
        set_optional_double(pin_arrival_delta,
                            "r_arrival_delta",
                            has_input_r_arrival && has_output_r_arrival,
                            staTimeToPsOrDefault(
                                time_unit, output_r_arrival - input_r_arrival, 0.0));
        set_optional_double(pin_arrival_delta,
                            "f_arrival_delta",
                            has_input_f_arrival && has_output_f_arrival,
                            staTimeToPsOrDefault(
                                time_unit, output_f_arrival - input_f_arrival, 0.0));
        samples["pin_arrival_delta"] = pin_arrival_delta;
        samples["output_net_cap_pf"] =
            has_output_net_cap ? py::cast(output_net_cap_pf) : py::none();

        py::dict upstream_split_transfer;
        upstream_split_transfer["schema_name"] =
            "buffering_hard_commit_upstream_split_transfer";
        upstream_split_transfer["schema_version"] = 1;
        upstream_split_transfer["status"] = "ok";
        upstream_split_transfer["buffer_input_pin_name"] = iterm_name(input_iterm);
        odb::dbNet* upstream_net = input_iterm->getNet();
        upstream_split_transfer["upstream_net_name"] =
            upstream_net == nullptr ? std::string() : upstream_net->getName();
        upstream_split_transfer["upstream_net_iterm_count"] = net_iterm_count(upstream_net);
        upstream_split_transfer["upstream_net_bterm_count"] = net_bterm_count(upstream_net);
        upstream_split_transfer["upstream_net_load_pin_count"] =
            net_load_pin_count(upstream_net);
        const bool has_upstream_net_cap = corner != nullptr && upstream_net != nullptr;
        const double upstream_net_cap_pf =
            has_upstream_net_cap
                ? staCapToPfOrDefault(
                      cap_unit, timing.getNetCap(upstream_net, corner, ord::Timing::Max), 0.0)
                : 0.0;
        upstream_split_transfer["upstream_net_cap_pf"] =
            has_upstream_net_cap ? py::cast(upstream_net_cap_pf) : py::none();
        odb::dbITerm* upstream_driver_iterm =
            upstream_net == nullptr ? nullptr : diffGuidedBatchFindNetDriverITerm(upstream_net);
        upstream_split_transfer["upstream_driver_pin_name"] =
            iterm_name(upstream_driver_iterm);
        if (upstream_net == nullptr) {
          upstream_split_transfer["status"] = "missing_upstream_net";
        } else if (upstream_driver_iterm == nullptr) {
          upstream_split_transfer["status"] = "missing_upstream_driver_iterm";
        } else {
          const double driver_r_slew =
              timing.getPinSlew(upstream_driver_iterm, ord::Timing::Rise, ord::Timing::Max);
          const double driver_f_slew =
              timing.getPinSlew(upstream_driver_iterm, ord::Timing::Fall, ord::Timing::Max);
          const double driver_r_arrival =
              timing.getPinArrival(upstream_driver_iterm, ord::Timing::Rise, ord::Timing::Max);
          const double driver_f_arrival =
              timing.getPinArrival(upstream_driver_iterm, ord::Timing::Fall, ord::Timing::Max);
          const bool has_driver_r_slew = staScalarIsUsable(driver_r_slew);
          const bool has_driver_f_slew = staScalarIsUsable(driver_f_slew);
          const bool has_driver_r_arrival = staScalarIsUsable(driver_r_arrival);
          const bool has_driver_f_arrival = staScalarIsUsable(driver_f_arrival);
          set_optional_double(upstream_split_transfer,
                              "driver_r_slew",
                              has_driver_r_slew,
                              staTimeToPsOrDefault(time_unit, driver_r_slew, 0.0));
          set_optional_double(upstream_split_transfer,
                              "driver_f_slew",
                              has_driver_f_slew,
                              staTimeToPsOrDefault(time_unit, driver_f_slew, 0.0));
          set_optional_double(upstream_split_transfer,
                              "driver_r_arrival",
                              has_driver_r_arrival,
                              staTimeToPsOrDefault(time_unit, driver_r_arrival, 0.0));
          set_optional_double(upstream_split_transfer,
                              "driver_f_arrival",
                              has_driver_f_arrival,
                              staTimeToPsOrDefault(time_unit, driver_f_arrival, 0.0));
          set_optional_double(upstream_split_transfer,
                              "buffer_input_r_slew",
                              has_input_r_slew,
                              staTimeToPsOrDefault(time_unit, input_r_slew, 0.0));
          set_optional_double(upstream_split_transfer,
                              "buffer_input_f_slew",
                              has_input_f_slew,
                              staTimeToPsOrDefault(time_unit, input_f_slew, 0.0));
          set_optional_double(upstream_split_transfer,
                              "buffer_input_r_arrival",
                              has_input_r_arrival,
                              staTimeToPsOrDefault(time_unit, input_r_arrival, 0.0));
          set_optional_double(upstream_split_transfer,
                              "buffer_input_f_arrival",
                              has_input_f_arrival,
                              staTimeToPsOrDefault(time_unit, input_f_arrival, 0.0));
          set_optional_double(
              upstream_split_transfer,
              "r_arrival_delta",
              has_driver_r_arrival && has_input_r_arrival,
              staTimeToPsOrDefault(time_unit, input_r_arrival - driver_r_arrival, 0.0));
          set_optional_double(
              upstream_split_transfer,
              "f_arrival_delta",
              has_driver_f_arrival && has_input_f_arrival,
              staTimeToPsOrDefault(time_unit, input_f_arrival - driver_f_arrival, 0.0));
          set_optional_double(
              upstream_split_transfer,
              "r_slew_delta",
              has_driver_r_slew && has_input_r_slew,
              staTimeToPsOrDefault(time_unit, input_r_slew - driver_r_slew, 0.0));
          set_optional_double(
              upstream_split_transfer,
              "f_slew_delta",
              has_driver_f_slew && has_input_f_slew,
              staTimeToPsOrDefault(time_unit, input_f_slew - driver_f_slew, 0.0));
        }
        samples["upstream_split_transfer"] = upstream_split_transfer;

        auto* master = buffer_inst->getMaster();
        auto* cell = master == nullptr ? nullptr : network->dbToSta(master);
        auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
        auto* input_mterm = input_iterm->getMTerm();
        auto* output_mterm = output_iterm->getMTerm();
        sta::LibertyPort* input_port =
            liberty_cell == nullptr || input_mterm == nullptr
                ? nullptr
                : liberty_cell->findLibertyPort(input_mterm->getName().c_str());
        sta::LibertyPort* output_port =
            liberty_cell == nullptr || output_mterm == nullptr
                ? nullptr
                : liberty_cell->findLibertyPort(output_mterm->getName().c_str());
        auto* dcalc_ap = corner == nullptr ? nullptr : corner->findDcalcAnalysisPt(sta::MinMax::max());
        const sta::Pvt* pvt = dcalc_ap == nullptr ? nullptr : dcalc_ap->operatingConditions();
        if (liberty_cell == nullptr || input_port == nullptr || output_port == nullptr
            || dcalc_ap == nullptr) {
          samples["status"] = "partial_missing_liberty_arc_context";
          samples["liberty_arc_samples"] = arc_samples;
          samples["liberty_arc_sample_count"] = 0;
          return samples;
        }
        const float input_slew_sta =
            static_cast<float>(has_input_slew ? input_slew : 0.0);
        const float output_load_cap_sta = static_cast<float>(
            pfToStaCapOrDefault(cap_unit, std::max(0.0, output_net_cap_pf), 0.0));
        py::dict gate_delay_query_units;
        gate_delay_query_units["time_unit_scale"] =
            time_unit == nullptr ? py::none() : py::cast(time_unit->scale());
        gate_delay_query_units["cap_unit_scale"] =
            cap_unit == nullptr ? py::none() : py::cast(cap_unit->scale());
        gate_delay_query_units["cap_unit_to_pf_scale"] =
            cap_unit == nullptr ? py::none() : py::cast(
                (!std::isfinite(cap_unit->scale()) || cap_unit->scale() <= 0.0f)
                    ? 1.0
                    : static_cast<double>(cap_unit->scale()) / 1.0e-12);
        gate_delay_query_units["input_slew_internal"] = input_slew;
        gate_delay_query_units["input_slew_user"] =
            has_input_slew && time_unit != nullptr
                ? py::cast(
                      const_cast<sta::Unit*>(time_unit)->staToUser(input_slew))
                : py::none();
        gate_delay_query_units["input_slew_ps"] =
            has_input_slew
                ? py::cast(staTimeToPsOrDefault(time_unit, input_slew, 0.0))
                : py::none();
        gate_delay_query_units["canonical_time_unit"] = "ps";
        gate_delay_query_units["input_slew_sta_for_gate_delay"] = input_slew_sta;
        gate_delay_query_units["output_load_cap_pf"] =
            has_output_net_cap ? py::cast(output_net_cap_pf) : py::none();
        gate_delay_query_units["output_load_cap_sta_for_gate_delay"] = output_load_cap_sta;
        samples["gate_delay_query_units"] = gate_delay_query_units;
        py::dict selected_max_delay_sample;
        bool selected_max_delay = false;
        double selected_delay_abs = 0.0;
        int arc_index = 0;
        for (sta::TimingArcSet* arc_set : liberty_cell->timingArcSets(input_port, output_port)) {
          if (arc_set == nullptr || arc_set->role() == nullptr
              || arc_set->role()->isTimingCheck()) {
            continue;
          }
          for (sta::TimingArc* arc : arc_set->arcs()) {
            if (arc == nullptr) {
              continue;
            }
            sta::GateTableModel* model = arc->gateTableModel(dcalc_ap);
            if (model == nullptr) {
              continue;
            }
            sta::ArcDelay gate_delay = 0.0;
            sta::Slew gate_slew = 0.0;
            model->gateDelay(
                pvt, input_slew_sta, output_load_cap_sta, false, gate_delay, gate_slew);
            const sta::TableModel* delay_model = model->delayModel();
            const sta::TableModel* slew_model = model->slewModel();
            const float delay_find_value =
                delay_model == nullptr
                    ? 0.0f
                    : delay_model->findValue(
                          liberty_cell, pvt, input_slew_sta, output_load_cap_sta, 0.0f);
            const float slew_find_value =
                slew_model == nullptr
                    ? 0.0f
                    : slew_model->findValue(
                          liberty_cell, pvt, input_slew_sta, output_load_cap_sta, 0.0f);
            const double gate_delay_user =
                staTimeToPsOrDefault(time_unit, gate_delay, 0.0);
            const double gate_slew_user =
                staTimeToPsOrDefault(time_unit, gate_slew, 0.0);
            const double delay_find_value_user =
                staTimeToPsOrDefault(time_unit, delay_find_value, 0.0);
            const double slew_find_value_user =
                staTimeToPsOrDefault(time_unit, slew_find_value, 0.0);
            py::dict row;
            row["arc_index"] = arc_index++;
            row["from_port_name"] =
                arc->from() == nullptr ? std::string() : arc->from()->name();
            row["to_port_name"] =
                arc->to() == nullptr ? std::string() : arc->to()->name();
            row["input_slew"] = has_input_slew
                                    ? py::cast(staTimeToPsOrDefault(
                                          time_unit, input_slew, 0.0))
                                    : py::none();
            row["output_load_cap_pf"] =
                has_output_net_cap ? py::cast(output_net_cap_pf) : py::none();
            row["input_slew_internal"] = input_slew;
            row["input_slew_sta_for_gate_delay"] = input_slew_sta;
            row["output_load_cap_sta_for_gate_delay"] = output_load_cap_sta;
            row["gate_delay_internal"] = gate_delay;
            row["gate_slew_internal"] = gate_slew;
            row["liberty_gate_delay"] = gate_delay_user;
            row["liberty_output_slew"] = gate_slew_user;
            row["delay_model_find_value_internal"] = delay_find_value;
            row["slew_model_find_value_internal"] = slew_find_value;
            row["delay_model_find_value"] = delay_find_value_user;
            row["slew_model_find_value"] = slew_find_value_user;
            row["gate_delay_over_find_value"] =
                std::abs(delay_find_value_user) > 0.0
                    ? py::cast(gate_delay_user / delay_find_value_user)
                    : py::none();
            row["gate_slew_over_find_value"] =
                std::abs(slew_find_value_user) > 0.0
                    ? py::cast(gate_slew_user / slew_find_value_user)
                    : py::none();
            row["delay_model_diagnostics"] =
                summarizeLibertyTableModel(delay_model, units);
            row["slew_model_diagnostics"] =
                summarizeLibertyTableModel(slew_model, units);
            arc_samples.append(row);
            if (!selected_max_delay || std::abs(gate_delay_user) > selected_delay_abs) {
              selected_max_delay = true;
              selected_delay_abs = std::abs(gate_delay_user);
              selected_max_delay_sample = row;
            }
          }
        }
        samples["liberty_arc_samples"] = arc_samples;
        samples["liberty_arc_sample_count"] = py::len(arc_samples);
        if (selected_max_delay) {
          samples["selected_max_delay_sample"] = selected_max_delay_sample;
        } else {
          samples["selected_max_delay_sample"] = py::none();
        }
        if (py::len(arc_samples) == 0) {
          samples["status"] = "partial_no_liberty_arc_samples";
        }
      } catch (const std::exception& exc) {
        samples["status"] = "failed";
        samples["error"] = exc.what();
      }
      return samples;
    };

    try {
      const py::dict before_metrics =
          defer_timing_update ? py::dict() : queryDiffGuidedBatchTimingMetrics();
      const int before_instance_count = instance_count();
      const int before_net_count = net_count();
      const int before_buffer_count = buffer_instance_count();
      const int before_original_net_iterm_count = net_iterm_count(net);
      const int before_original_net_bterm_count = net_bterm_count(net);
      const int before_original_net_load_pin_count = net_load_pin_count(net);
      const std::vector<std::string> before_pin_sample_names = build_pin_sample_names();
      const py::dict before_pin_timing_samples =
          defer_timing_update ? py::dict() : build_pin_timing_samples(before_pin_sample_names);

      const auto mutation_begin = std::chrono::steady_clock::now();
      odb::dbInst* buffer_inst =
          odb::dbInst::create(block_, buffer_master, buffer_inst_name.c_str());
      if (buffer_inst == nullptr) {
        return failed("failed to create buffer instance");
      }
      buffer_inst->setPlacementStatus(odb::dbPlacementStatus::PLACED);
      buffer_inst->setLocation(
          static_cast<int>(pyDoubleOrDefault(action, "candidate_location_x_dbu", 0.0)),
          static_cast<int>(pyDoubleOrDefault(action, "candidate_location_y_dbu", 0.0)));
      choose_buffer_pins(buffer_inst);
      if (buffer_input == nullptr || buffer_output == nullptr) {
        return failed("buffer master missing input/output pins: " + buffer_master_name);
      }

      odb::dbNet* downstream_net =
          odb::dbNet::create(block_, downstream_net_name.c_str());
      if (downstream_net == nullptr) {
        return failed("failed to create downstream net");
      }
      downstream_net->setSigType(net->getSigType());
      buffer_input->connect(net);
      buffer_output->connect(downstream_net);
      for (auto* moved_iterm : load_iterms) {
        moved_iterm->disconnect();
        moved_iterm->connect(downstream_net);
      }
      for (auto* moved_bterm : load_bterms) {
        moved_bterm->connect(downstream_net);
      }
      summary["mutation_ms"] = elapsedMs(mutation_begin);
      summary["inserted_buffer_name"] = buffer_inst_name;
      summary["downstream_net_name"] = downstream_net_name;
      summary["moved_load_count"] =
          static_cast<int>(load_iterms.size() + load_bterms.size());
      py::list moved_names;
      for (const auto& moved_name : moved_load_pin_names) {
        moved_names.append(moved_name);
      }
      summary["moved_load_pin_names"] = moved_names;
      summary["legalization_status"] = "placed_at_candidate_requires_later_legalization";

      rebuildNodeInstIndex();
      if (!defer_timing_update) {
        try {
          design_->evalTclString("estimate_parasitics -placement");
        } catch (const std::exception&) {
        }
      }

      const py::dict after_metrics =
          defer_timing_update ? py::dict() : queryDiffGuidedBatchTimingMetrics();
      const int after_instance_count = instance_count();
      const int after_net_count = net_count();
      const int after_buffer_count = buffer_instance_count();
      const int after_original_net_iterm_count = net_iterm_count(net);
      const int after_original_net_bterm_count = net_bterm_count(net);
      const int after_original_net_load_pin_count = net_load_pin_count(net);
      const int after_downstream_net_iterm_count = net_iterm_count(downstream_net);
      const int after_downstream_net_bterm_count = net_bterm_count(downstream_net);
      const int after_downstream_net_load_pin_count = net_load_pin_count(downstream_net);
      std::vector<std::string> after_pin_sample_names = before_pin_sample_names;
      std::unordered_set<std::string> after_pin_sample_name_set(
          after_pin_sample_names.begin(), after_pin_sample_names.end());
      append_unique_pin_name(after_pin_sample_names, after_pin_sample_name_set, iterm_name(buffer_input));
      append_unique_pin_name(after_pin_sample_names, after_pin_sample_name_set, iterm_name(buffer_output));
      const py::dict after_pin_timing_samples =
          defer_timing_update ? py::dict() : build_pin_timing_samples(after_pin_sample_names);
      const py::dict pin_timing_sample_delta =
          defer_timing_update
              ? py::dict()
              : build_pin_timing_sample_delta(before_pin_timing_samples, after_pin_timing_samples);
      const int instance_delta = after_instance_count - before_instance_count;
      const int net_delta = after_net_count - before_net_count;
      const int buffer_delta = after_buffer_count - before_buffer_count;

      const double before_wns = pyDoubleOrDefault(before_metrics, "wns", 0.0);
      const double before_tns = pyDoubleOrDefault(before_metrics, "tns", 0.0);
      const int before_setup_vio =
          pyIntOrDefault(before_metrics, "setup_violation_count", 0);
      const double before_slew_vio = pyDoubleOrDefault(before_metrics, "slew_vio_count", 0.0);
      const double before_cap_vio = pyDoubleOrDefault(before_metrics, "cap_vio_count", 0.0);
      const double after_wns = pyDoubleOrDefault(after_metrics, "wns", before_wns);
      const double after_tns = pyDoubleOrDefault(after_metrics, "tns", before_tns);
      const int after_setup_vio =
          pyIntOrDefault(after_metrics, "setup_violation_count", before_setup_vio);
      const double after_slew_vio =
          pyDoubleOrDefault(after_metrics, "slew_vio_count", before_slew_vio);
      const double after_cap_vio =
          pyDoubleOrDefault(after_metrics, "cap_vio_count", before_cap_vio);

      summary["status"] = "ok";
      summary["before_metrics"] = before_metrics;
      summary["after_metrics"] = after_metrics;
      summary["actual_delta_wns"] = after_wns - before_wns;
      summary["actual_delta_tns"] = after_tns - before_tns;
      summary["setup_violation_count_delta"] = after_setup_vio - before_setup_vio;
      summary["slew_violation_count_delta"] = after_slew_vio - before_slew_vio;
      summary["cap_violation_count_delta"] = after_cap_vio - before_cap_vio;
      summary["inserted_buffer_count_delta"] = buffer_delta;
      summary["instance_count_delta"] = instance_delta;
      summary["net_count_delta"] = net_delta;
      const bool requires_rebuild =
          instance_delta != 0 || net_delta != 0 || buffer_delta != 0;
      summary["requires_sync_back"] = requires_rebuild;
      summary["requires_runtimedb_rebuild"] = requires_rebuild;
      summary["unsupported_reasons"] = py::list();

      py::dict topology_before;
      topology_before["instance_count"] = before_instance_count;
      topology_before["net_count"] = before_net_count;
      topology_before["buffer_count"] = before_buffer_count;
      topology_before["original_net_name"] = net_name;
      topology_before["original_net_iterm_count"] = before_original_net_iterm_count;
      topology_before["original_net_bterm_count"] = before_original_net_bterm_count;
      topology_before["original_net_load_pin_count"] = before_original_net_load_pin_count;

      py::dict topology_after;
      topology_after["instance_count"] = after_instance_count;
      topology_after["net_count"] = after_net_count;
      topology_after["buffer_count"] = after_buffer_count;
      topology_after["original_net_name"] = net_name;
      topology_after["original_net_iterm_count"] = after_original_net_iterm_count;
      topology_after["original_net_bterm_count"] = after_original_net_bterm_count;
      topology_after["original_net_load_pin_count"] = after_original_net_load_pin_count;
      topology_after["downstream_net_name"] = downstream_net_name;
      topology_after["downstream_net_iterm_count"] = after_downstream_net_iterm_count;
      topology_after["downstream_net_bterm_count"] = after_downstream_net_bterm_count;
      topology_after["downstream_net_load_pin_count"] = after_downstream_net_load_pin_count;

      py::dict topology_delta;
      topology_delta["instance_count_delta"] = instance_delta;
      topology_delta["net_count_delta"] = net_delta;
      topology_delta["buffer_count_delta"] = buffer_delta;
      topology_delta["original_net_iterm_count_delta"] =
          after_original_net_iterm_count - before_original_net_iterm_count;
      topology_delta["original_net_bterm_count_delta"] =
          after_original_net_bterm_count - before_original_net_bterm_count;
      topology_delta["original_net_load_pin_count_delta"] =
          after_original_net_load_pin_count - before_original_net_load_pin_count;
      topology_delta["downstream_net_iterm_count_after"] =
          after_downstream_net_iterm_count;
      topology_delta["downstream_net_bterm_count_after"] =
          after_downstream_net_bterm_count;
      topology_delta["downstream_net_load_pin_count_after"] =
          after_downstream_net_load_pin_count;

      py::dict load_partition;
      load_partition["mode"] = load_partition_mode;
      load_partition["requested_downstream_pin_count"] =
          pyIntOrDefault(summary, "requested_downstream_pin_count", 0);
      load_partition["moved_load_count"] =
          static_cast<int>(load_iterms.size() + load_bterms.size());
      load_partition["moved_iterm_count"] = static_cast<int>(load_iterms.size());
      load_partition["moved_bterm_count"] = static_cast<int>(load_bterms.size());
      py::list hard_moved_names;
      for (const auto& moved_name : moved_load_pin_names) {
        hard_moved_names.append(moved_name);
      }
      load_partition["moved_load_pin_names"] = hard_moved_names;

      py::dict connectivity;
      connectivity["original_net_name"] = net_name;
      connectivity["downstream_net_name"] = downstream_net_name;
      connectivity["inserted_buffer_name"] = buffer_inst_name;
      connectivity["buffer_master_name"] = buffer_master_name;
      connectivity["buffer_input_pin_name"] = iterm_name(buffer_input);
      connectivity["buffer_output_pin_name"] = iterm_name(buffer_output);
      connectivity["buffer_input_net_name"] =
          buffer_input->getNet() == nullptr ? "" : buffer_input->getNet()->getName();
      connectivity["buffer_output_net_name"] =
          buffer_output->getNet() == nullptr ? "" : buffer_output->getNet()->getName();
      connectivity["driver_pin_name"] = driver_pin_name;
      connectivity["load_pin_name"] = load_pin_name;

      py::dict pin_timing_samples;
      pin_timing_samples["status"] = defer_timing_update ? "deferred_batch_timing" : "ok";
      pin_timing_samples["before"] = before_pin_timing_samples;
      pin_timing_samples["after"] = after_pin_timing_samples;
      pin_timing_samples["delta"] = pin_timing_sample_delta;

      const py::dict cell_arc_delay_samples = build_cell_arc_delay_samples(
          buffer_inst, buffer_input, buffer_output, downstream_net);

      py::dict hard_commit_diagnostic;
      hard_commit_diagnostic["schema_name"] = "buffering_hard_commit_diagnostic";
      hard_commit_diagnostic["schema_version"] = 1;
      hard_commit_diagnostic["status"] = "ok";
      hard_commit_diagnostic["source"] = "openroad_coordinate_buffer_insert";
      hard_commit_diagnostic["topology_before"] = topology_before;
      hard_commit_diagnostic["topology_after"] = topology_after;
      hard_commit_diagnostic["topology_delta"] = topology_delta;
      hard_commit_diagnostic["load_partition"] = load_partition;
      hard_commit_diagnostic["connectivity"] = connectivity;
      hard_commit_diagnostic["pin_timing_samples"] = pin_timing_samples;
      hard_commit_diagnostic["cell_arc_delay_samples"] = cell_arc_delay_samples;
      summary["hard_commit_diagnostic"] = hard_commit_diagnostic;
      summary["hard_commit_diagnostic_status"] = "ok";
      summary["hard_commit_topology_delta"] = topology_delta;
      summary["hard_commit_load_partition"] = load_partition;
      summary["hard_commit_cell_arc_delay_samples"] = cell_arc_delay_samples;
    } catch (const std::exception& exc) {
      return failed(exc.what());
    }
    return summary;
  }

  void setNodeOrient(int node_id, int orient_value)
  {
    if (node_id < 0 || node_id >= static_cast<int>(node_insts_.size())) {
      return;
    }
    auto* inst = node_insts_[node_id];
    if (inst == nullptr) {
      return;
    }
    inst->setOrient(odb::dbOrientType(static_cast<odb::dbOrientType::Value>(orient_value)));
  }

  odb::dbInst* nodeInst(int node_id) const
  {
    if (node_id < 0 || node_id >= static_cast<int>(node_insts_.size())) {
      return nullptr;
    }
    return node_insts_[node_id];
  }

  void invalidateDiffGuidedBatchNPathSnapshot()
  {
    diff_guided_batch_npath_snapshot_ = DiffGuidedBatchNPathSnapshot{};
  }

  const DiffGuidedBatchNPathSnapshot& ensureDiffGuidedBatchNPathSnapshot(double weight_cap)
  {
    if (diff_guided_batch_npath_snapshot_.valid
        && diff_guided_batch_npath_snapshot_.node_count
               == static_cast<int>(node_insts_.size())
        && diff_guided_batch_npath_snapshot_.weight_cap == weight_cap) {
      return diff_guided_batch_npath_snapshot_;
    }

    const auto build_begin = std::chrono::steady_clock::now();
    DiffGuidedBatchNPathSnapshot snapshot;
    snapshot.valid = true;
    snapshot.weight_cap = weight_cap;
    snapshot.node_count = static_cast<int>(node_insts_.size());
    snapshot.source = "openroad_bridge_cached_topology_npath";
    snapshot.inst_npath_weight_by_id.assign(node_insts_.size(), 1.0);
    snapshot.n_from_by_id.assign(node_insts_.size(), 0.0);
    snapshot.n_to_by_id.assign(node_insts_.size(), 0.0);
    snapshot.inst_id_by_db_inst.reserve(node_insts_.size());
    for (int inst_id = 0; inst_id < static_cast<int>(node_insts_.size()); ++inst_id) {
      if (node_insts_[inst_id] != nullptr) {
        snapshot.inst_id_by_db_inst[node_insts_[inst_id]] = inst_id;
      }
    }

    std::vector<std::vector<int>> fanout_edges(node_insts_.size());
    std::vector<int> indegree(node_insts_.size(), 0);
    std::vector<uint8_t> source_boundary(node_insts_.size(), 0);
    std::vector<uint8_t> endpoint_boundary(node_insts_.size(), 0);
    auto inst_id_for = [&](odb::dbInst* inst) -> int {
      const auto it = snapshot.inst_id_by_db_inst.find(inst);
      return it == snapshot.inst_id_by_db_inst.end() ? -1 : it->second;
    };
    auto is_timing_boundary = [&](odb::dbInst* inst) -> bool {
      if (inst == nullptr || inst->getMaster() == nullptr) {
        return true;
      }
      auto* master = inst->getMaster();
      const bool is_boundary =
          master->isBlock() || (design_ != nullptr && design_->isSequential(master));
      if (is_boundary) {
        snapshot.sequential_or_macro_boundary_count += 1;
      }
      return is_boundary;
    };
    auto saturating_add = [&](double lhs, double rhs) -> double {
      const double cap = weight_cap * weight_cap;
      if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return cap;
      }
      return std::min(cap, lhs + rhs);
    };

    for (auto* net : block_->getNets()) {
      if (!isSignalNet(net)) {
        continue;
      }
      odb::dbITerm* driver_iterm = diffGuidedBatchFindNetDriverITerm(net);
      odb::dbInst* driver_inst = driver_iterm == nullptr ? nullptr : driver_iterm->getInst();
      const int driver_id = inst_id_for(driver_inst);
      bool source_bterm_driver = false;
      bool endpoint_bterm_sink = false;
      for (auto* bterm : net->getBTerms()) {
        if (bterm == nullptr) {
          continue;
        }
        const auto io_type = bterm->getIoType();
        if (io_type == odb::dbIoType::INPUT || io_type == odb::dbIoType::INOUT) {
          source_bterm_driver = true;
        }
        if (io_type == odb::dbIoType::OUTPUT || io_type == odb::dbIoType::INOUT) {
          endpoint_bterm_sink = true;
        }
      }

      if (driver_id >= 0 && endpoint_bterm_sink) {
        endpoint_boundary[driver_id] = 1;
        snapshot.endpoint_boundary_count += 1;
      }
      const bool driver_is_boundary = driver_id >= 0 && is_timing_boundary(driver_inst);
      if (driver_id >= 0 && driver_is_boundary) {
        source_boundary[driver_id] = 1;
      }

      for (auto* sink_iterm : net->getITerms()) {
        if (sink_iterm == nullptr || sink_iterm == driver_iterm) {
          continue;
        }
        const auto sink_io_type = sink_iterm->getIoType();
        if (sink_io_type != odb::dbIoType::INPUT && sink_io_type != odb::dbIoType::INOUT) {
          continue;
        }
        odb::dbInst* sink_inst = sink_iterm->getInst();
        const int sink_id = inst_id_for(sink_inst);
        if (sink_id < 0) {
          continue;
        }
        const bool sink_is_boundary = is_timing_boundary(sink_inst);
        if (source_bterm_driver || driver_is_boundary) {
          source_boundary[sink_id] = 1;
          snapshot.source_boundary_count += 1;
        }
        if (driver_id < 0) {
          continue;
        }
        if (sink_is_boundary) {
          endpoint_boundary[driver_id] = 1;
          snapshot.endpoint_boundary_count += 1;
          continue;
        }
        if (driver_is_boundary) {
          continue;
        }
        fanout_edges[driver_id].push_back(sink_id);
        indegree[sink_id] += 1;
        snapshot.edge_count += 1;
      }
    }

    std::deque<int> queue;
    for (int inst_id = 0; inst_id < snapshot.node_count; ++inst_id) {
      if (source_boundary[inst_id]) {
        snapshot.n_from_by_id[inst_id] = 1.0;
      }
      if (endpoint_boundary[inst_id] || fanout_edges[inst_id].empty()) {
        snapshot.n_to_by_id[inst_id] = 1.0;
      }
      if (indegree[inst_id] == 0) {
        queue.push_back(inst_id);
      }
    }

    std::vector<int> topo_order;
    topo_order.reserve(node_insts_.size());
    std::vector<int> remaining_indegree = indegree;
    while (!queue.empty()) {
      const int inst_id = queue.front();
      queue.pop_front();
      topo_order.push_back(inst_id);
      if (snapshot.n_from_by_id[inst_id] <= 0.0) {
        snapshot.n_from_by_id[inst_id] = 1.0;
      }
      for (const int sink_id : fanout_edges[inst_id]) {
        snapshot.n_from_by_id[sink_id] =
            saturating_add(snapshot.n_from_by_id[sink_id], snapshot.n_from_by_id[inst_id]);
        remaining_indegree[sink_id] -= 1;
        if (remaining_indegree[sink_id] == 0) {
          queue.push_back(sink_id);
        }
      }
    }

    std::vector<uint8_t> resolved(node_insts_.size(), 0);
    for (const int inst_id : topo_order) {
      resolved[inst_id] = 1;
    }
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
      const int inst_id = *it;
      if (snapshot.n_to_by_id[inst_id] <= 0.0) {
        snapshot.n_to_by_id[inst_id] = 1.0;
      }
      for (const int sink_id : fanout_edges[inst_id]) {
        snapshot.n_to_by_id[inst_id] =
            saturating_add(snapshot.n_to_by_id[inst_id], snapshot.n_to_by_id[sink_id]);
      }
    }

    for (int inst_id = 0; inst_id < snapshot.node_count; ++inst_id) {
      if (!resolved[inst_id]) {
        snapshot.cycle_or_unresolved_count += 1;
        snapshot.fallback_weight_count += 1;
        snapshot.inst_npath_weight_by_id[inst_id] = 1.0;
        continue;
      }
      const double n_from = snapshot.n_from_by_id[inst_id];
      const double n_to = snapshot.n_to_by_id[inst_id];
      const double weight =
          std::min(weight_cap,
                   std::sqrt(std::max(1.0, n_from)) * std::sqrt(std::max(1.0, n_to))
                       + 1.0);
      snapshot.inst_npath_weight_by_id[inst_id] = std::isfinite(weight) ? weight : 1.0;
      if (snapshot.inst_npath_weight_by_id[inst_id] <= 1.0) {
        snapshot.fallback_weight_count += 1;
      }
    }

    snapshot.build_ms = elapsedMs(build_begin);
    diff_guided_batch_npath_snapshot_ = std::move(snapshot);
    diff_guided_batch_npath_snapshot_build_count_ += 1;
    return diff_guided_batch_npath_snapshot_;
  }

  void syncToOpenRoad(const py::object& node_x, const py::object& node_y)
  {
    const auto xs = node_x.cast<std::vector<double>>();
    const auto ys = node_y.cast<std::vector<double>>();
    const int movable_count = static_cast<int>(std::min(xs.size(), ys.size()));
    for (int node_id = 0; node_id < movable_count; ++node_id) {
      auto* inst = nodeInst(node_id);
      if (inst == nullptr || inst->isFixed()) {
        continue;
      }
      inst->setLocation(static_cast<int>(xs[node_id]), static_cast<int>(ys[node_id]));
    }
  }
  py::dict applySizing(const py::object& inst_cell_ids, const py::object& cell_master_names)
  {
    const auto cell_ids = inst_cell_ids.cast<std::vector<int>>();
    const auto master_names = cell_master_names.cast<std::vector<std::string>>();
    auto* db = block_ == nullptr ? nullptr : block_->getDataBase();
    if (db == nullptr) {
      throw std::runtime_error("OpenROAD bridge has no dbDatabase for sizing apply");
    }

    int applied = 0;
    int skipped = 0;
    int unchanged_same_master = 0;
    int missing_masters = 0;
    int invalid_cell_ids = 0;
    const int count = static_cast<int>(std::min(cell_ids.size(), node_insts_.size()));
    for (int node_id = 0; node_id < count; ++node_id) {
      auto* inst = nodeInst(node_id);
      if (inst == nullptr || inst->isFixed()) {
        skipped += 1;
        continue;
      }

      const int cell_id = cell_ids[node_id];
      if (cell_id < 0 || cell_id >= static_cast<int>(master_names.size())) {
        invalid_cell_ids += 1;
        skipped += 1;
        continue;
      }

      odb::dbMaster* master = db->findMaster(master_names[cell_id].c_str());
      if (master == nullptr) {
        missing_masters += 1;
        skipped += 1;
        continue;
      }
      if (inst->getMaster() == master) {
        unchanged_same_master += 1;
        skipped += 1;
        continue;
      }
      inst->swapMaster(master);
      applied += 1;
    }

    rebuildNodeInstIndex(false);
    py::dict summary;
    summary["instances_seen"] = count;
    summary["applied"] = applied;
    summary["skipped"] = skipped;
    summary["unchanged_same_master"] = unchanged_same_master;
    summary["missing_masters"] = missing_masters;
    summary["invalid_cell_ids"] = invalid_cell_ids;
    return summary;
  }

  double queryDiffGuidedBatchDrvViolationCount(const std::string& tcl_command)
  {
    if (design_ == nullptr || tcl_command.empty()) {
      return 0.0;
    }
    const std::string result = design_->evalTclString(tcl_command);
    try {
      const double value = std::stod(result);
      return std::isfinite(value) ? value : 0.0;
    } catch (const std::exception&) {
      return 0.0;
    }
  }

  double sumDiffGuidedBatchCurrentLeakage()
  {
    if (design_ == nullptr || block_ == nullptr) {
      return 0.0;
    }
    ord::Timing timing(design_.get());
    auto* sta = timing.getSta();
    auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
    if (network == nullptr) {
      return 0.0;
    }
    double total_leakage = 0.0;
    for (auto* inst : block_->getInsts()) {
      auto* master = inst == nullptr ? nullptr : inst->getMaster();
      if (master == nullptr) {
        continue;
      }
      auto* cell = network->dbToSta(master);
      auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
      total_leakage += exportLibcellLeakageForPython(liberty_cell);
    }
    return total_leakage;
  }

  std::pair<odb::dbITerm*, odb::dbBTerm*> findDiffGuidedBatchDbPinByName(
      const std::string& pin_name)
  {
    if (block_ == nullptr || pin_name.empty()) {
      return {nullptr, nullptr};
    }
    const auto colon = pin_name.rfind(':');
    const auto slash = pin_name.rfind('/');
    const auto separator = colon == std::string::npos ? slash : colon;
    if (separator != std::string::npos) {
      const std::string inst_name = pin_name.substr(0, separator);
      const std::string port_name = pin_name.substr(separator + 1);
      auto* inst = block_->findInst(inst_name.c_str());
      auto* iterm = inst == nullptr ? nullptr : inst->findITerm(port_name.c_str());
      return {iterm, nullptr};
    }
    return {nullptr, block_->findBTerm(pin_name.c_str())};
  }

  py::dict buildDiffGuidedBatchEndpointSample(
      ord::Timing& timing,
      const sta::Unit* time_unit,
      const std::string& pin_name)
  {
    py::dict sample;
    sample["name"] = pin_name;
    sample["mapped"] = false;
    const auto [iterm, bterm] = findDiffGuidedBatchDbPinByName(pin_name);
    if (iterm == nullptr && bterm == nullptr) {
      sample["missing_mapping"] = true;
      return sample;
    }
    sample["mapped"] = true;
    const double rise_slack = iterm != nullptr
                                  ? timing.getPinSlack(iterm, ord::Timing::Rise, ord::Timing::Max)
                                  : timing.getPinSlack(bterm, ord::Timing::Rise, ord::Timing::Max);
    const double fall_slack = iterm != nullptr
                                  ? timing.getPinSlack(iterm, ord::Timing::Fall, ord::Timing::Max)
                                  : timing.getPinSlack(bterm, ord::Timing::Fall, ord::Timing::Max);
    const double rise_arrival = iterm != nullptr
                                    ? timing.getPinArrival(iterm, ord::Timing::Rise, ord::Timing::Max)
                                    : timing.getPinArrival(bterm, ord::Timing::Rise, ord::Timing::Max);
    const double fall_arrival = iterm != nullptr
                                    ? timing.getPinArrival(iterm, ord::Timing::Fall, ord::Timing::Max)
                                    : timing.getPinArrival(bterm, ord::Timing::Fall, ord::Timing::Max);
    sample["opensta_slack"] = staTimeToPsOrDefault(time_unit, std::min(rise_slack, fall_slack), 0.0);
    sample["opensta_arrival"] = staTimeToPsOrDefault(time_unit, std::max(rise_arrival, fall_arrival), 0.0);
    sample["opensta_r_slack"] = staScalarIsUsable(rise_slack)
                                     ? py::cast(staTimeToPsOrDefault(time_unit, rise_slack, 0.0))
                                     : py::none();
    sample["opensta_f_slack"] = staScalarIsUsable(fall_slack)
                                     ? py::cast(staTimeToPsOrDefault(time_unit, fall_slack, 0.0))
                                     : py::none();
    sample["opensta_r_arrival"] = staScalarIsUsable(rise_arrival)
                                       ? py::cast(staTimeToPsOrDefault(time_unit, rise_arrival, 0.0))
                                       : py::none();
    sample["opensta_f_arrival"] = staScalarIsUsable(fall_arrival)
                                       ? py::cast(staTimeToPsOrDefault(time_unit, fall_arrival, 0.0))
                                       : py::none();
    sample["opensta_r_required"] = staScalarIsUsable(rise_arrival) && staScalarIsUsable(rise_slack)
                                        ? py::cast(staTimeToPsOrDefault(time_unit, rise_arrival + rise_slack, 0.0))
                                        : py::none();
    sample["opensta_f_required"] = staScalarIsUsable(fall_arrival) && staScalarIsUsable(fall_slack)
                                        ? py::cast(staTimeToPsOrDefault(time_unit, fall_arrival + fall_slack, 0.0))
                                        : py::none();
    return sample;
  }

  py::dict buildDiffGuidedBatchPinSample(
      ord::Timing& timing,
      const sta::Unit* time_unit,
      const sta::Unit* cap_unit,
      sta::Corner* corner,
      const std::string& pin_name)
  {
    return buildDiffGuidedBatchPinTimingContextSample(timing, time_unit, cap_unit, corner, pin_name);
  }

  DiffGuidedBatchCurrentLocalTimingContext queryDiffGuidedBatchCurrentLocalTimingContext(
      ord::Timing& timing,
      const sta::Unit* time_unit,
      const sta::Unit* cap_unit,
      sta::Corner* corner,
      odb::dbInst* inst)
  {
    DiffGuidedBatchCurrentLocalTimingContext context;
    if (inst == nullptr) {
      context.fallback_reason = "missing_instance";
      return context;
    }

    for (auto* iterm : inst->getITerms()) {
      if (iterm == nullptr || iterm->getMTerm() == nullptr) {
        continue;
      }
      if (iterm->getIoType() == odb::dbIoType::INPUT
          || iterm->getIoType() == odb::dbIoType::INOUT) {
        const double rise_slew = timing.getPinSlew(iterm, ord::Timing::Rise, ord::Timing::Max);
        const double fall_slew = timing.getPinSlew(iterm, ord::Timing::Fall, ord::Timing::Max);
        if (staScalarIsUsable(rise_slew) || staScalarIsUsable(fall_slew)) {
          const double input_slew = staTimeToPsOrDefault(
              time_unit,
              std::max(staScalarIsUsable(rise_slew) ? rise_slew : 0.0,
                       staScalarIsUsable(fall_slew) ? fall_slew : 0.0),
              0.0);
          context.input_slew_ps = std::max(context.input_slew_ps, input_slew);
          context.input_slew_pin_count += 1;
        }
      }
      if (iterm->getIoType() == odb::dbIoType::OUTPUT
          || iterm->getIoType() == odb::dbIoType::INOUT) {
        odb::dbNet* net = iterm->getNet();
        const double net_cap = staCapToPfOrDefault(
            cap_unit,
            (net == nullptr || corner == nullptr) ? 0.0 : timing.getNetCap(net, corner, ord::Timing::Max),
            0.0);
        if (net_cap > 0.0) {
          context.load_cap_pf = std::max(context.load_cap_pf, net_cap);
          context.load_cap_pin_count += 1;
        }
        const double rise_slew = timing.getPinSlew(iterm, ord::Timing::Rise, ord::Timing::Max);
        const double fall_slew = timing.getPinSlew(iterm, ord::Timing::Fall, ord::Timing::Max);
        if (staScalarIsUsable(rise_slew) || staScalarIsUsable(fall_slew)) {
          const double output_slew = staTimeToPsOrDefault(
              time_unit,
              std::max(staScalarIsUsable(rise_slew) ? rise_slew : 0.0,
                       staScalarIsUsable(fall_slew) ? fall_slew : 0.0),
              0.0);
          context.output_slew_ps = std::max(context.output_slew_ps, output_slew);
          context.output_slew_pin_count += 1;
        }
        auto* master = inst->getMaster();
        auto* sta = timing.getSta();
        auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
        auto* cell = (network == nullptr || master == nullptr) ? nullptr : network->dbToSta(master);
        auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
        auto* mterm = iterm->getMTerm();
        auto* liberty_port = (liberty_cell == nullptr || mterm == nullptr)
                                 ? nullptr
                                 : liberty_cell->findLibertyPort(mterm->getName().c_str());
        if (liberty_port != nullptr) {
          const double slew_limit =
              resolveLibPinSlewLimitForPythonPs(time_unit, liberty_port);
          if (slew_limit > 0.0) {
            const double slew_violation =
                std::max(0.0, context.output_slew_ps - slew_limit);
            if (slew_violation > context.output_slew_violation_ps
                || context.output_slew_limit_pin_count == 0) {
              context.output_slew_for_violation_ps = context.output_slew_ps;
              context.output_slew_limit_ps = slew_limit;
              context.output_slew_violation_ps = slew_violation;
              context.output_slew_limit_port_name = mterm->getName();
            }
            context.output_slew_limit_pin_count += 1;
            context.output_slew_limit_source = "liberty_output_slew_limit";
          }
          const double cap_limit =
              diffGuidedBatchResolveLibPinCapLimitPf(cap_unit, liberty_port);
          if (cap_limit > 0.0) {
            const double cap_violation = std::max(0.0, context.load_cap_pf - cap_limit);
            if (cap_violation > context.output_cap_violation_pf
                || context.output_cap_limit_pin_count == 0) {
              context.output_cap_for_violation_pf = context.load_cap_pf;
              context.output_cap_limit_pf = cap_limit;
              context.output_cap_violation_pf = cap_violation;
              context.output_cap_limit_port_name = mterm->getName();
            }
            context.output_cap_limit_pin_count += 1;
            context.output_cap_limit_source = "liberty_output_cap_limit";
          }
        }
        const double rise_slack = timing.getPinSlack(iterm, ord::Timing::Rise, ord::Timing::Max);
        const double fall_slack = timing.getPinSlack(iterm, ord::Timing::Fall, ord::Timing::Max);
        const double rise_arrival = timing.getPinArrival(iterm, ord::Timing::Rise, ord::Timing::Max);
        const double fall_arrival = timing.getPinArrival(iterm, ord::Timing::Fall, ord::Timing::Max);
        const bool rise_usable = staScalarIsUsable(rise_slack) && staScalarIsUsable(rise_arrival);
        const bool fall_usable = staScalarIsUsable(fall_slack) && staScalarIsUsable(fall_arrival);
        if (rise_usable || fall_usable) {
          const bool use_fall = !rise_usable || (fall_usable && fall_slack < rise_slack);
          const double slack = use_fall ? fall_slack : rise_slack;
          const double arrival = use_fall ? fall_arrival : rise_arrival;
          const double slack_ps = staTimeToPsOrDefault(time_unit, slack, 0.0);
          const double arrival_ps = staTimeToPsOrDefault(time_unit, arrival, 0.0);
          const double required_ps =
              staTimeToPsOrDefault(time_unit, arrival + slack, arrival_ps + slack_ps);
          const double violation_ps = std::max(0.0, -slack_ps);
          if (!context.output_slack_used || slack_ps < context.output_slack_ps) {
            context.output_slack_used = true;
            context.output_slack_ps = slack_ps;
            context.output_arrival_ps = arrival_ps;
            context.output_required_ps = required_ps;
            context.output_slack_violation_ps = violation_ps;
            context.slack_source = use_fall ? "opensta_output_fall_slack"
                                            : "opensta_output_rise_slack";
          }
          context.output_slack_pin_count += 1;
        }
      }
    }

    if (context.input_slew_pin_count > 0 || context.load_cap_pin_count > 0
        || context.output_slew_pin_count > 0 || context.output_slack_pin_count > 0) {
      context.used = true;
      context.local_context_source = "opensta_current_state";
      context.input_slew_source = context.input_slew_pin_count > 0
                                      ? "opensta_pin_slew"
                                      : "proposal_fallback";
      context.load_cap_source = context.load_cap_pin_count > 0
                                    ? "opensta_load_cap"
                                    : "proposal_fallback";
      context.output_slew_source = context.output_slew_pin_count > 0
                                      ? "opensta_output_pin_slew"
                                      : "proposal_fallback";
      if (!context.output_slack_used) {
        context.slack_source = "proposal_fallback";
      }
      context.fallback_reason.clear();
    } else {
      context.fallback_reason = "missing_opensta_pin_slew_and_load_cap";
    }
    return context;
  }

  py::dict buildDiffGuidedBatchPinTimingContextSample(
      ord::Timing& timing,
      const sta::Unit* time_unit,
      const sta::Unit* cap_unit,
      sta::Corner* corner,
      const std::string& pin_name)
  {
    py::dict sample;
    sample["name"] = pin_name;
    sample["mapped"] = false;
    const auto [iterm, bterm] = findDiffGuidedBatchDbPinByName(pin_name);
    if (iterm == nullptr && bterm == nullptr) {
      sample["missing_mapping"] = true;
      return sample;
    }
    sample["mapped"] = true;
    const double rise_slew = iterm != nullptr
                                 ? timing.getPinSlew(iterm, ord::Timing::Rise, ord::Timing::Max)
                                 : timing.getPinSlew(bterm, ord::Timing::Rise, ord::Timing::Max);
    const double fall_slew = iterm != nullptr
                                 ? timing.getPinSlew(iterm, ord::Timing::Fall, ord::Timing::Max)
                                 : timing.getPinSlew(bterm, ord::Timing::Fall, ord::Timing::Max);
    const double rise_arrival = iterm != nullptr
                                    ? timing.getPinArrival(iterm, ord::Timing::Rise, ord::Timing::Max)
                                    : timing.getPinArrival(bterm, ord::Timing::Rise, ord::Timing::Max);
    const double fall_arrival = iterm != nullptr
                                    ? timing.getPinArrival(iterm, ord::Timing::Fall, ord::Timing::Max)
                                    : timing.getPinArrival(bterm, ord::Timing::Fall, ord::Timing::Max);
    const double rise_slack = iterm != nullptr
                                  ? timing.getPinSlack(iterm, ord::Timing::Rise, ord::Timing::Max)
                                  : timing.getPinSlack(bterm, ord::Timing::Rise, ord::Timing::Max);
    const double fall_slack = iterm != nullptr
                                  ? timing.getPinSlack(iterm, ord::Timing::Fall, ord::Timing::Max)
                                  : timing.getPinSlack(bterm, ord::Timing::Fall, ord::Timing::Max);
    sample["opensta_slew"] = staTimeToPsOrDefault(time_unit, std::max(rise_slew, fall_slew), 0.0);
    sample["opensta_arrival"] = staTimeToPsOrDefault(time_unit, std::max(rise_arrival, fall_arrival), 0.0);
    sample["opensta_slack"] = staTimeToPsOrDefault(time_unit, std::min(rise_slack, fall_slack), 0.0);
    sample["opensta_r_slew"] = staScalarIsUsable(rise_slew)
                                    ? py::cast(staTimeToPsOrDefault(time_unit, rise_slew, 0.0))
                                    : py::none();
    sample["opensta_f_slew"] = staScalarIsUsable(fall_slew)
                                    ? py::cast(staTimeToPsOrDefault(time_unit, fall_slew, 0.0))
                                    : py::none();
    sample["opensta_r_arrival"] = staScalarIsUsable(rise_arrival)
                                       ? py::cast(staTimeToPsOrDefault(time_unit, rise_arrival, 0.0))
                                       : py::none();
    sample["opensta_f_arrival"] = staScalarIsUsable(fall_arrival)
                                       ? py::cast(staTimeToPsOrDefault(time_unit, fall_arrival, 0.0))
                                       : py::none();
    sample["opensta_r_slack"] = staScalarIsUsable(rise_slack)
                                     ? py::cast(staTimeToPsOrDefault(time_unit, rise_slack, 0.0))
                                     : py::none();
    sample["opensta_f_slack"] = staScalarIsUsable(fall_slack)
                                     ? py::cast(staTimeToPsOrDefault(time_unit, fall_slack, 0.0))
                                     : py::none();

    odb::dbNet* net = nullptr;
    if (iterm != nullptr) {
      if (iterm->getIoType() == odb::dbIoType::OUTPUT) {
        net = iterm->getNet();
      } else {
        const double input_cap = staCapToPfOrDefault(
            cap_unit,
            corner == nullptr ? 0.0 : timing.getPortCap(iterm, corner, ord::Timing::Max),
            0.0);
        sample["opensta_cap"] = input_cap;
        sample["opensta_r_cap"] = input_cap;
        sample["opensta_f_cap"] = input_cap;
        return sample;
      }
    } else if (bterm != nullptr) {
      net = bterm->getNet();
    }
    const double net_cap = staCapToPfOrDefault(
        cap_unit,
        (net == nullptr || corner == nullptr) ? 0.0 : timing.getNetCap(net, corner, ord::Timing::Max),
        0.0);
    sample["opensta_cap"] = net_cap;
    sample["opensta_r_cap"] = net_cap;
    sample["opensta_f_cap"] = net_cap;
    return sample;
  }

  py::dict queryDiffGuidedBatchTimingSamples(const py::dict& sample_request)
  {
    py::dict samples;
    samples["artifact"] = "diff_guided_batch_opensta_timing_samples";
    samples["artifact_version"] = 1;
    samples["status"] = "ok";
    samples["has_timing_inputs"] = has_timing_inputs_;
    py::list endpoint_samples;
    py::list pin_samples;
    if (!has_timing_inputs_) {
      samples["status"] = "no_timing_inputs";
      samples["endpoint_samples"] = endpoint_samples;
      samples["pin_samples"] = pin_samples;
      return samples;
    }

    const auto begin = std::chrono::steady_clock::now();
    const py::dict timing_refresh = refreshTiming();
    samples["timing_refresh"] = timing_refresh;
    const std::string timing_refresh_status =
        pyStringOrDefault(timing_refresh, "status", "failed");
    if (timing_refresh_status != "ok") {
      samples["status"] = timing_refresh_status;
      samples["error"] = pyStringOrDefault(
          timing_refresh, "error", timing_refresh_status);
      samples["endpoint_samples"] = endpoint_samples;
      samples["pin_samples"] = pin_samples;
      return samples;
    }
    ord::Timing timing(design_.get());
    auto* sta = timing.getSta();
    if (sta == nullptr) {
      samples["status"] = "missing_sta";
      samples["endpoint_samples"] = endpoint_samples;
      samples["pin_samples"] = pin_samples;
      return samples;
    }
    const sta::Units* units = sta->units();
    const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
    const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
    sta::Corner* corner = timing.cmdCorner();

    for (const auto& pin_name : pyStringListOrEmpty(sample_request, "endpoint_pin_names")) {
      endpoint_samples.append(buildDiffGuidedBatchEndpointSample(timing, time_unit, pin_name));
    }
    for (const auto& pin_name : pyStringListOrEmpty(sample_request, "pin_names")) {
      pin_samples.append(buildDiffGuidedBatchPinSample(timing, time_unit, cap_unit, corner, pin_name));
    }
    samples["endpoint_samples"] = endpoint_samples;
    samples["pin_samples"] = pin_samples;
    samples["endpoint_sample_count"] = py::len(endpoint_samples);
    samples["pin_sample_count"] = py::len(pin_samples);
    samples["query_ms"] = elapsedMs(begin);
    return samples;
  }

  py::dict queryDiffGuidedBatchTimingMetrics()
  {
    py::dict metrics;
    metrics["artifact"] = "diff_guided_batch_opensta_timing_metrics";
    metrics["metric_source"] = "in_process_opensta";
    metrics["status"] = "ok";
    metrics["has_timing_inputs"] = has_timing_inputs_;
    metrics["wns"] = 0.0;
    metrics["tns"] = 0.0;
    metrics["setup_violation_count"] = 0;
    metrics["slew_vio"] = -1.0;
    metrics["cap_vio"] = -1.0;
    metrics["slew_vio_count"] = -1.0;
    metrics["cap_vio_count"] = -1.0;
    metrics["slew_vio_semantics"] = "count";
    metrics["cap_vio_semantics"] = "count";
    metrics["leakage"] = -1.0;
    if (!has_timing_inputs_) {
      metrics["status"] = "no_timing_inputs";
      return metrics;
    }

    const auto begin = std::chrono::steady_clock::now();
    const py::dict timing_refresh = refreshTiming();
    metrics["timing_refresh"] = timing_refresh;
    const std::string timing_refresh_status =
        pyStringOrDefault(timing_refresh, "status", "failed");
    if (timing_refresh_status != "ok") {
      metrics["status"] = timing_refresh_status;
      metrics["error"] = pyStringOrDefault(
          timing_refresh, "error", timing_refresh_status);
      return metrics;
    }
    ord::Timing timing(design_.get());
    auto* sta = timing.getSta();
    if (sta == nullptr) {
      metrics["status"] = "missing_sta";
      return metrics;
    }
    const sta::Units* units = sta->units();
    const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
    metrics["wns"] = staTimeToPsOrDefault(
        time_unit, sta->worstSlack(sta::MinMax::max()), 0.0);
    metrics["tns"] = staTimeToPsOrDefault(
        time_unit, sta->totalNegativeSlack(sta::MinMax::max()), 0.0);
    metrics["setup_violation_count"] = sta->endpointViolationCount(sta::MinMax::max());
    const double slew_vio_count = queryDiffGuidedBatchDrvViolationCount(
        "sta::max_slew_violation_count");
    const double cap_vio_count = queryDiffGuidedBatchDrvViolationCount(
        "sta::max_capacitance_violation_count");
    metrics["slew_vio"] = slew_vio_count;
    metrics["cap_vio"] = cap_vio_count;
    metrics["slew_vio_count"] = slew_vio_count;
    metrics["cap_vio_count"] = cap_vio_count;
    metrics["leakage"] = sumDiffGuidedBatchCurrentLeakage();
    metrics["query_ms"] = elapsedMs(begin);
    return metrics;
  }

  py::dict queryDiffGuidedBatchDynamicConflictSignature(
      const py::dict& config = py::dict())
  {
    py::dict signature;
    signature["artifact"] = "diff_guided_batch_dynamic_conflict_signature";
    signature["artifact_version"] = 1;
    signature["status"] = "ok";
    signature["has_timing_inputs"] = has_timing_inputs_;
    signature["dynamic_top_path_signature_source"] = "opensta_findPathEnds";
    signature["dynamic_endpoint_signature_source"] = "opensta_findPathEnds_endpoint";
    signature["dynamic_path_count"] = 0;
    signature["dynamic_inst_hit_count"] = 0;
    signature["dynamic_unique_inst_count"] = 0;
    signature["component_id_by_top_path_dynamic"] = std::vector<int>(node_insts_.size(), -1);
    signature["component_id_by_endpoint_dynamic"] = std::vector<int>(node_insts_.size(), -1);
    signature["critical_path_slacks"] = std::vector<double>();
    signature["path_instance_groups"] = std::vector<std::vector<int>>();
    signature["endpoint_instance_groups"] = std::vector<std::vector<int>>();
    if (!has_timing_inputs_) {
      signature["status"] = "no_timing_inputs";
      return signature;
    }

    const auto begin = std::chrono::steady_clock::now();
    const int group_path_count = std::max(
        1,
        pyIntOrDefault(
            config,
            "dynamic_conflict_group_path_count",
            pyIntOrDefault(config, "diff_guided_batch_dynamic_group_path_count", 32)));
    const int endpoint_path_count = std::max(
        1,
        pyIntOrDefault(
            config,
            "dynamic_conflict_endpoint_path_count",
            pyIntOrDefault(config, "diff_guided_batch_dynamic_endpoint_path_count", 1)));

    refreshTimingOrThrow();
    ord::Timing timing(design_.get());
    auto* sta = timing.getSta();
    auto* network = sta == nullptr ? nullptr : sta->getDbNetwork();
    if (sta == nullptr || network == nullptr || sta->search() == nullptr) {
      signature["status"] = "missing_sta";
      signature["query_ms"] = elapsedMs(begin);
      return signature;
    }

    std::unordered_map<odb::dbInst*, int> node_id_by_inst;
    node_id_by_inst.reserve(node_insts_.size());
    for (int node_id = 0; node_id < static_cast<int>(node_insts_.size()); ++node_id) {
      if (node_insts_[node_id] != nullptr) {
        node_id_by_inst[node_insts_[node_id]] = node_id;
      }
    }

    std::vector<int> component_id_by_top_path_dynamic(node_insts_.size(), -1);
    std::vector<int> component_id_by_endpoint_dynamic(node_insts_.size(), -1);
    std::vector<double> critical_path_slacks;
    std::vector<double> top_path_residual_budgets;
    std::vector<std::vector<int>> path_instance_groups;
    std::vector<std::vector<int>> endpoint_instance_groups;
    std::unordered_map<const sta::Pin*, int> endpoint_component_by_pin;
    std::unordered_set<int> unique_inst_ids;
    int dynamic_inst_hit_count = 0;

    sta::PathEndSeq path_ends = sta->search()->findPathEnds(
        nullptr,
        nullptr,
        nullptr,
        false,
        sta->cmdCorner(),
        sta::MinMaxAll::max(),
        static_cast<size_t>(group_path_count),
        static_cast<size_t>(endpoint_path_count),
        true,
        true,
        -sta::INF,
        sta::INF,
        true,
        nullptr,
        true,
        false,
        false,
        false,
        false,
        false);

    const sta::Units* units = sta->units();
    const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
    int path_component_id = 0;
    for (auto* path_end : path_ends) {
      if (path_end == nullptr || path_end->path() == nullptr) {
        continue;
      }
      sta::PathExpanded expanded(path_end->path(), sta);
      const sta::Path* endpoint_path = expanded.endPath();
      const sta::Pin* endpoint_pin = endpoint_path == nullptr ? nullptr : endpoint_path->pin(sta);
      int endpoint_component_id = -1;
      if (endpoint_pin != nullptr) {
        auto endpoint_it = endpoint_component_by_pin.find(endpoint_pin);
        if (endpoint_it == endpoint_component_by_pin.end()) {
          endpoint_component_id = static_cast<int>(endpoint_component_by_pin.size());
          endpoint_component_by_pin[endpoint_pin] = endpoint_component_id;
          endpoint_instance_groups.emplace_back();
        } else {
          endpoint_component_id = endpoint_it->second;
        }
      }
      std::vector<int> path_inst_ids;
      std::unordered_set<int> path_seen_inst_ids;
      for (size_t path_index = 0; path_index < expanded.size(); ++path_index) {
        const sta::Path* path_ref = expanded.path(path_index);
        if (path_ref == nullptr) {
          continue;
        }
        sta::Pin* pin = path_ref->pin(sta);
        const sta::Instance* sta_inst = pin == nullptr ? nullptr : network->instance(pin);
        odb::dbInst* db_inst = sta_inst == nullptr ? nullptr : network->staToDb(sta_inst);
        auto node_it = db_inst == nullptr ? node_id_by_inst.end() : node_id_by_inst.find(db_inst);
        if (node_it == node_id_by_inst.end()) {
          continue;
        }
        const int node_id = node_it->second;
        component_id_by_top_path_dynamic[node_id] = path_component_id;
        if (endpoint_component_id >= 0) {
          component_id_by_endpoint_dynamic[node_id] = endpoint_component_id;
        }
        dynamic_inst_hit_count += 1;
        unique_inst_ids.insert(node_id);
        if (path_seen_inst_ids.insert(node_id).second) {
          path_inst_ids.push_back(node_id);
          if (endpoint_component_id >= 0
              && endpoint_component_id < static_cast<int>(endpoint_instance_groups.size())) {
            endpoint_instance_groups[endpoint_component_id].push_back(node_id);
          }
        }
      }
      if (!path_inst_ids.empty()) {
        path_instance_groups.push_back(path_inst_ids);
        const double path_slack = staTimeToPsOrDefault(
            time_unit, path_end->slack(sta), 0.0);
        critical_path_slacks.push_back(path_slack);
        top_path_residual_budgets.push_back(std::max(0.0, -path_slack));
        path_component_id += 1;
      }
    }

    std::vector<double> endpoint_residual_budgets(endpoint_instance_groups.size(), 0.0);
    for (int node_id = 0; node_id < static_cast<int>(component_id_by_top_path_dynamic.size()); ++node_id) {
      const int top_path_component = component_id_by_top_path_dynamic[node_id];
      const int endpoint_component = component_id_by_endpoint_dynamic[node_id];
      if (top_path_component < 0
          || top_path_component >= static_cast<int>(top_path_residual_budgets.size())
          || endpoint_component < 0
          || endpoint_component >= static_cast<int>(endpoint_residual_budgets.size())) {
        continue;
      }
      endpoint_residual_budgets[endpoint_component] = std::max(
          endpoint_residual_budgets[endpoint_component],
          top_path_residual_budgets[top_path_component]);
    }

    signature["status"] = "ok";
    signature["group_path_count"] = group_path_count;
    signature["endpoint_path_count"] = endpoint_path_count;
    signature["dynamic_path_count"] = static_cast<int>(path_instance_groups.size());
    signature["dynamic_inst_hit_count"] = dynamic_inst_hit_count;
    signature["dynamic_unique_inst_count"] = static_cast<int>(unique_inst_ids.size());
    signature["component_id_by_top_path_dynamic"] = component_id_by_top_path_dynamic;
    signature["component_id_by_endpoint_dynamic"] = component_id_by_endpoint_dynamic;
    signature["critical_path_slacks"] = critical_path_slacks;
    signature["top_path_residual_budgets"] = top_path_residual_budgets;
    signature["endpoint_residual_budgets"] = endpoint_residual_budgets;
    signature["local_residual_budget_by_top_path_component"] = top_path_residual_budgets;
    signature["local_residual_budget_by_endpoint_component"] = endpoint_residual_budgets;
    signature["local_residual_snapshot_source"] = "opensta_findPathEnds_slack_violation";
    signature["path_instance_groups"] = path_instance_groups;
    signature["endpoint_instance_groups"] = endpoint_instance_groups;
    signature["dynamic_endpoint_count"] = static_cast<int>(endpoint_instance_groups.size());
    signature["query_ms"] = elapsedMs(begin);
    return signature;
  }

  py::dict refreshDiffGuidedBatchDynamicConflictSignature(
      const py::dict& config,
      int loop_id,
      const py::dict& transaction_decision)
  {
    py::dict signature = queryDiffGuidedBatchDynamicConflictSignature(config);
    signature["refresh_loop_id"] = loop_id;
    signature["refresh_after_transaction_status"] =
        pyStringOrDefault(transaction_decision, "status", "");
    signature["refresh_after_transaction_confirmed"] =
        pyBoolOrDefault(transaction_decision, "confirmed", false);
    signature["refresh_after_transaction_rolled_back"] =
        pyBoolOrDefault(transaction_decision, "rolled_back", false);
    return signature;
  }

  CompactBridgeEvaluationSummary evaluateDiffGuidedBatchActionProposalsTyped(
      const std::vector<CompactBridgeActionProposal>& compact_actions,
      const py::dict& config = py::dict())
  {
    CompactBridgeEvaluationSummary typed_summary;
    py::dict evaluator_summary;
    evaluator_summary["artifact"] = "diff_guided_batch_opensta_precise_evaluator";
    evaluator_summary["status"] = "ok";
    evaluator_summary["opensta_status"] = has_timing_inputs_ ? "integrated" : "no_timing_inputs";
    evaluator_summary["trial_parallel_mode"] = "parallel_objective_delta_prescreen_serial_opensta_verify";
    evaluator_summary["opensta_trial_mutation_mode"] = "serial_mutation";
    evaluator_summary["requested_trial_worker_count"] = diffGuidedBatchTrialWorkerCount(config);
    evaluator_summary["trial_worker_count"] = diffGuidedBatchTrialWorkerCount(config);
    const std::string trial_mode = diffGuidedBatchTrialMode(config);
    const int trial_mini_batch_size = diffGuidedBatchTrialMiniBatchSize(config);
    evaluator_summary["trial_mode"] = trial_mode;
    evaluator_summary["trial_mini_batch_size"] = trial_mini_batch_size;
    evaluator_summary["input_action_count"] = static_cast<int>(compact_actions.size());
    evaluator_summary["proposal_input_format"] = "compact_action_proposals";
    evaluator_summary["compact_action_proposal_count"] = static_cast<int>(compact_actions.size());
    evaluator_summary["typed_result_count"] = 0;
    const CompactBridgeObjectiveDeltaPrescreenResult prescreen_result =
        evaluateDiffGuidedBatchObjectiveDeltaPrescreen(compact_actions, config);
    evaluator_summary["parallel_prescreen_summary"] = prescreen_result.summary;
    evaluator_summary["parallel_prescreen_worker_count"] =
        prescreen_result.summary["parallel_prescreen_worker_count"];
    evaluator_summary["parallel_prescreen_candidate_count"] =
        prescreen_result.summary["parallel_prescreen_candidate_count"];
    evaluator_summary["parallel_prescreen_ms"] = prescreen_result.summary["parallel_prescreen_ms"];
    py::list results;

    auto* db = block_ == nullptr ? nullptr : block_->getDataBase();
    if (db == nullptr) {
      throw std::runtime_error("OpenROAD bridge has no dbDatabase for diff-guided batch trial");
    }
    ord::Timing local_estimator_timing(design_.get());
    auto* local_estimator_sta = local_estimator_timing.getSta();
    auto* local_estimator_network =
        local_estimator_sta == nullptr ? nullptr : local_estimator_sta->getDbNetwork();
    auto* local_estimator_corner =
        local_estimator_sta == nullptr ? nullptr : local_estimator_sta->cmdCorner();

    int accepted_count = 0;
    int rejected_count = 0;
    int unsupported_count = 0;
    int trial_count = 0;
    int trial_query_count = 0;
    int trial_batch_query_count = 0;
    int trial_fallback_count = 0;
    double trial_mutation_ms = 0.0;
    double trial_query_ms = 0.0;
    int snapshot_query_count = 0;
    double snapshot_query_ms = 0.0;
    int full_metric_query_count = 0;
    int full_sta_trial_mutation_count = 0;
    int local_context_query_count = 0;
    int local_dcalc_query_count = 0;
    int local_cap_query_count = 0;
    int local_leakage_query_count = 0;
    int batch_verify_query_count = 0;
    const bool use_local_candidate_delta = trial_mode == "local_candidate_delta";
    const bool use_no_query_local_evaluator =
        trial_mode == "local_residual" || use_local_candidate_delta;
    py::dict baseline_metrics;
    if (!use_no_query_local_evaluator) {
      const auto baseline_query_begin = std::chrono::steady_clock::now();
      baseline_metrics = queryDiffGuidedBatchTimingMetrics();
      snapshot_query_ms += elapsedMs(baseline_query_begin);
      snapshot_query_count += 1;
    } else {
      baseline_metrics = makeDiffGuidedBatchLocalResidualPlaceholderMetrics(
          use_local_candidate_delta ? "local_candidate_delta" : "local_residual");
      if (use_local_candidate_delta) {
        evaluator_summary["metric_snapshot_mode"] = "local_context_no_full_metric";
      } else {
        evaluator_summary["metric_snapshot_mode"] = "none_local_residual";
      }
    }
    const double baseline_tns = pyDoubleOrDefault(baseline_metrics, "tns", 0.0);
    const double baseline_wns = pyDoubleOrDefault(baseline_metrics, "wns", 0.0);
    const double baseline_leakage = pyDoubleOrDefault(baseline_metrics, "leakage", 0.0);
    const std::string accept_mode = diffGuidedBatchAcceptMode(config);
    const double leakage_weight = diffGuidedBatchTnsPowerLeakageWeight(config);
      const bool local_objective_align_global =
          diffGuidedBatchLocalObjectiveAlignGlobal(config);
      const bool noop_baseline_enabled = diffGuidedBatchNoopBaseline(config);
      const bool reject_nonpositive_trial =
          diffGuidedBatchRejectNonpositiveTrial(config);
      const bool force_accept_selected_actions =
          diffGuidedBatchForceAcceptSelectedActions(config);
      const double nonpositive_trial_eps =
          diffGuidedBatchNonpositiveTrialEps(config);
      const DiffGuidedBatchEffectiveLocalWeight local_delta_slew_weight_info =
          diffGuidedBatchEffectiveLocalDeltaSlewPenaltyWeight(config);
      const DiffGuidedBatchEffectiveLocalWeight local_delta_cap_weight_info =
          diffGuidedBatchEffectiveLocalDeltaCapPenaltyWeight(config);
      const double local_delta_slew_penalty_weight =
          local_delta_slew_weight_info.value;
      const double local_delta_cap_penalty_weight =
          local_delta_cap_weight_info.value;
      const double local_residual_feedback_scale =
          diffGuidedBatchLocalResidualFeedbackScale(config);
      const std::string local_residual_feedback_policy =
          diffGuidedBatchLocalResidualFeedbackPolicy(config);
      const double local_residual_feedback_decay =
          diffGuidedBatchLocalResidualFeedbackDecay(config);
      const double local_residual_max_gain_per_action =
          diffGuidedBatchLocalResidualMaxGainPerAction(config);
	      const double local_residual_budget_discount =
	          diffGuidedBatchLocalResidualBudgetDiscount(config);
	      const double fanin_penalty_ps_per_pf =
	          diffGuidedBatchFaninPenaltySensitivityPsPerPf(config);
	      const std::string slack_delta_evaluator_mode =
	          diffGuidedBatchSlackDeltaEvaluatorMode(config);
	      const std::string slack_delta_weight_mode =
	          diffGuidedBatchSlackDeltaWeightMode(config);
	      const double npath_weight_cap =
	          diffGuidedBatchNPathWeightCap(config);
	      const DiffGuidedBatchNPathSnapshot* npath_snapshot =
	          &ensureDiffGuidedBatchNPathSnapshot(npath_weight_cap);
	      const bool use_tns_power_tradeoff = accept_mode == "tns_power_tradeoff";
	      const bool use_local_objective = local_objective_align_global || use_tns_power_tradeoff;
      evaluator_summary["accept_mode"] = accept_mode;
      evaluator_summary["tns_power_leakage_weight"] = leakage_weight;
      evaluator_summary["local_objective_align_global"] =
          local_objective_align_global;
      evaluator_summary["noop_baseline_enabled"] = noop_baseline_enabled;
      evaluator_summary["reject_nonpositive_trial_enabled"] =
          reject_nonpositive_trial;
      evaluator_summary["force_accept_selected_actions"] =
          force_accept_selected_actions;
      evaluator_summary["nonpositive_trial_eps"] = nonpositive_trial_eps;
      evaluator_summary["timing_objective_lane"] =
          pyStringOrDefault(config, "timing_objective_lane", "timing_only");
      evaluator_summary["local_delta_slew_penalty_weight"] =
          local_delta_slew_penalty_weight;
      evaluator_summary["local_delta_cap_penalty_weight"] =
          local_delta_cap_penalty_weight;
      evaluator_summary["local_delta_slew_penalty_weight_source"] =
          local_delta_slew_weight_info.source;
      evaluator_summary["local_delta_cap_penalty_weight_source"] =
          local_delta_cap_weight_info.source;
      evaluator_summary["local_residual_feedback_scale"] =
          local_residual_feedback_scale;
      evaluator_summary["local_residual_feedback_policy"] =
          local_residual_feedback_policy;
      evaluator_summary["local_residual_feedback_decay"] =
          local_residual_feedback_decay;
      evaluator_summary["local_residual_max_gain_per_action"] =
          local_residual_max_gain_per_action;
      evaluator_summary["local_residual_budget_discount"] =
          local_residual_budget_discount;
	      evaluator_summary["fanin_penalty_ps_per_pf"] =
	          fanin_penalty_ps_per_pf;
	      evaluator_summary["slack_delta_evaluator_mode"] =
	          slack_delta_evaluator_mode;
	      evaluator_summary["slack_delta_weight_mode"] =
	          slack_delta_weight_mode;
	      evaluator_summary["npath_weight_cap"] = npath_weight_cap;
	      evaluator_summary["npath_snapshot_build_count"] =
	          diff_guided_batch_npath_snapshot_build_count_;
	      evaluator_summary["npath_snapshot_build_ms"] =
	          npath_snapshot == nullptr ? 0.0 : npath_snapshot->build_ms;
	      evaluator_summary["npath_snapshot_edge_count"] =
	          npath_snapshot == nullptr ? 0 : npath_snapshot->edge_count;
	      evaluator_summary["npath_snapshot_fallback_weight_count"] =
	          npath_snapshot == nullptr ? 0 : npath_snapshot->fallback_weight_count;

    if (use_no_query_local_evaluator) {
      if (use_local_candidate_delta) {
        evaluator_summary["trial_parallel_mode"] =
          "local_candidate_delta_no_opensta_trial_query";
        evaluator_summary["opensta_trial_mutation_mode"] = "none_per_candidate_mutation";
      } else {
        evaluator_summary["trial_parallel_mode"] =
            "local_delay_slack_residual_no_opensta_trial_query";
        evaluator_summary["opensta_trial_mutation_mode"] = "none_local_residual";
      }
      const std::vector<double> top_path_residual_budgets =
          pyDoubleArrayOrListOrEmpty(config, "local_residual_budget_by_top_path_component");
      const std::vector<double> endpoint_residual_budgets =
          pyDoubleArrayOrListOrEmpty(config, "local_residual_budget_by_endpoint_component");
      const std::string local_residual_snapshot_source = pyStringOrDefault(
          config, "local_residual_snapshot_source", "missing_snapshot_fallback");
      std::unordered_map<std::string, double> residual_budget_by_component;
      double local_residual_accepted_gain_sum = 0.0;
      int local_residual_trial_count = 0;
      int local_residual_budget_update_count = 0;
      int local_residual_snapshot_budget_count = 0;
      int local_candidate_delta_trial_count = 0;
      int cpp_hot_path_input_cap_estimator_count = 0;
      int cpp_hot_path_input_cap_estimator_positive_count = 0;
      int cpp_hot_path_delay_estimator_count = 0;
      int cpp_hot_path_slew_estimator_count = 0;
      int cpp_hot_path_delay_estimator_fallback_count = 0;
      int cpp_hot_path_slew_estimator_fallback_count = 0;
      int cpp_hot_path_leakage_delta_estimator_count = 0;
      int cpp_hot_path_leakage_delta_positive_count = 0;
	      int cpp_hot_path_leakage_delta_fallback_count = 0;
	      int slack_delta_eval_count = 0;
	      double slack_delta_eval_ms = 0.0;
	      int slack_delta_dcalc_query_count = 0;
	      int slack_delta_fanin_net_count = 0;
	      int slack_delta_affected_sink_count = 0;
	      int slack_delta_fallback_count = 0;
	      int noop_selected_count = 0;
      int nonpositive_trial_reject_count = 0;
      int best_trial_noop_count = 0;
      int best_trial_nonzero_count = 0;
      std::vector<double> fanin_raw_delta_by_inst_id(node_insts_.size(), 0.0);
      std::vector<double> fanin_signed_delta_by_inst_id(node_insts_.size(), 0.0);
      std::vector<int> fanin_seen_epoch_by_inst_id(node_insts_.size(), -1);
      std::vector<int> fanin_touched_inst_ids;
      int fanin_scratch_epoch = 0;
      evaluator_summary["local_residual_snapshot_source"] = local_residual_snapshot_source;
      evaluator_summary["local_residual_top_path_budget_count"] =
          static_cast<int>(top_path_residual_budgets.size());
      evaluator_summary["local_residual_endpoint_budget_count"] =
          static_cast<int>(endpoint_residual_budgets.size());

      for (int action_pos = 0; action_pos < static_cast<int>(compact_actions.size()); ++action_pos) {
        const auto& action = compact_actions[action_pos];
        const ActionKind kind = action.kind;
        py::dict result;
        const int64_t action_index = action.action_index;
        result["action_id"] = action_index;
        result["action_index"] = action_index;
        result["action_kind"] = diffGuidedBatchActionKindName(kind);
        result["supported"] = false;
        result["accepted"] = false;
        result["status"] = "rejected";
        result["reject_reason"] = "";
        result["actual_delta_tns"] = 0.0;
        result["actual_delta_wns"] = 0.0;
        result["actual_delta_tns_semantics"] = "estimated";
        result["local_predicted_delta_tns_like"] = 0.0;
        result["local_predicted_delta_obj"] = 0.0;
        result["local_predicted_effective_timing_gain"] = 0.0;
        result["prescreen_predicted_delta_obj"] =
            vectorValueOrDefault<double>(prescreen_result.prescreen_delta_objs, action_pos, 0.0);
        result["prescreen_predicted_improvement"] =
            vectorValueOrDefault<double>(prescreen_result.predicted_improvements, action_pos, 0.0);

        if (kind != ActionKind::kSizing) {
          result["status"] = "unsupported";
          result["reject_reason"] = diffGuidedBatchActionKindName(kind) + " is reserved and disabled in v1";
          result["affected_net_id"] = action.affected_net_id;
          result["driver_pin_id"] = action.driver_pin_id;
          result["load_pin_id"] = action.load_pin_id;
          result["buffer_master_id"] = action.buffer_master_id;
          result["buffer_master_name"] = action.buffer_master_name;
          result["candidate_location_x"] = action.candidate_location_x;
          result["candidate_location_y"] = action.candidate_location_y;
          unsupported_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        const int inst_id = action.inst_id;
        result["inst_id"] = inst_id;
        result["endpoint_component"] = action.endpoint_component;
        result["top_path_component"] = action.top_path_component;
        result["current_size_idx"] = action.current_size_idx;
        result["target_size_idx"] = action.target_size_idx;
        result["old_timing_coordinate"] = action.old_timing_coordinate;
        result["new_timing_coordinate"] = action.new_timing_coordinate;
        result["local_delta_source"] = diffGuidedBatchLocalResidualTrialDeltaSource(action);
        result["estimator_source"] = action.estimator_source.empty()
                                         ? diffGuidedBatchLocalResidualTrialDeltaSource(action)
                                         : action.estimator_source;
        result["local_delta_delay_ps"] = action.local_delta_delay_ps;
        result["local_delta_slew_ps"] = action.local_delta_slew_ps;
        result["local_delta_cap"] = action.local_delta_cap;
        result["local_delta_delay_estimator_source"] = "python_or_proposal_local_delta_delay";
        result["local_delta_slew_estimator_source"] = "python_or_proposal_local_delta_slew";
        result["local_delta_delay_cpp_fallback_reason"] =
            "cpp_hot_path_delay_estimator_unavailable";
        result["local_delta_slew_cpp_fallback_reason"] =
            "cpp_hot_path_slew_estimator_unavailable";
        result["local_delta_delay_cpp"] = action.local_delta_delay_ps;
        result["local_delta_slew_cpp"] = action.local_delta_slew_ps;
        auto* inst = nodeInst(inst_id);
        if (inst == nullptr || inst->isFixed()) {
          result["reject_reason"] = "invalid_or_fixed_instance";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        odb::dbMaster* old_master = inst->getMaster();
        if (old_master == nullptr) {
          result["reject_reason"] = "missing_current_master";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }
        const auto sizing_trials = buildDiffGuidedBatchSizingTrials(action, old_master, config);
        if (sizing_trials.empty()) {
          result["reject_reason"] = "missing_target_master";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        const auto residual_key = diffGuidedBatchLocalResidualKey(action);
        result["local_residual_key"] = residual_key.key;
        result["local_residual_component_id"] = residual_key.component_id;
        result["local_residual_key_source"] = residual_key.source;
        const auto budget_it = residual_budget_by_component.find(residual_key.key);
        const DiffGuidedBatchLocalResidualSnapshotBudget snapshot_budget =
            diffGuidedBatchLocalResidualSnapshotBudget(
                action,
                top_path_residual_budgets,
                endpoint_residual_budgets);
        const double residual_budget = budget_it == residual_budget_by_component.end()
                                           ? snapshot_budget.budget
                                           : budget_it->second;
        const double residual_budget_before_discount = residual_budget;
        const double discounted_residual_budget =
            residual_budget_before_discount * local_residual_budget_discount;
        result["local_residual_snapshot_budget"] = snapshot_budget.budget;
        result["local_residual_budget_source"] = budget_it == residual_budget_by_component.end()
                                                    ? snapshot_budget.source
                                                    : "updated_component_residual_state";
        result["local_residual_snapshot_source"] = local_residual_snapshot_source;
        if (budget_it == residual_budget_by_component.end() && snapshot_budget.from_snapshot) {
          local_residual_snapshot_budget_count += 1;
        }
        bool has_valid_trial = noop_baseline_enabled;
        bool best_trial_is_noop = noop_baseline_enabled;
        std::string best_master_name;
        int best_trial_step = 0;
        double best_delta_tns =
            noop_baseline_enabled ? 0.0 : -std::numeric_limits<double>::infinity();
        double best_delta_wns = 0.0;
        double best_leakage_delta = 0.0;
        double best_local_delta_slew_penalty = 0.0;
        double best_local_delta_cap_penalty = 0.0;
        double best_local_delta_slew_violation_improvement = 0.0;
        double best_local_delta_cap_violation_improvement = 0.0;
        double best_local_delta_delay_cpp = action.local_delta_delay_ps;
        double best_local_delta_slew_cpp = action.local_delta_slew_ps;
        double best_local_predicted_downstream_delay_gain = action.local_delta_delay_ps;
        double best_local_predicted_downstream_slew_gain = 0.0;
	        double best_local_predicted_fanin_neighborhood_penalty = 0.0;
	        double best_local_predicted_downstream_slack_delta = 0.0;
	        double best_local_predicted_downstream_npath_weight = 1.0;
	        double best_local_predicted_downstream_weighted_slack_delta = 0.0;
	        double best_local_predicted_downstream_old_slack_ps = 0.0;
	        double best_local_predicted_downstream_new_slack_ps = 0.0;
	        double best_local_predicted_downstream_delta_delay_ps = 0.0;
	        std::string best_local_predicted_downstream_dcalc_source = "unavailable";
	        double best_local_predicted_fanin_slack_delta = 0.0;
	        double best_local_predicted_fanin_weighted_slack_delta = 0.0;
	        double best_local_predicted_net_slack_delta = 0.0;
	        double best_local_predicted_weighted_net_tns_delta = 0.0;
	        int best_local_predicted_downstream_point_count = 0;
	        int best_local_predicted_fanin_point_count = 0;
	        int best_local_predicted_dcalc_query_count = 0;
	        int best_local_predicted_fallback_point_count = 0;
	        std::string best_local_predicted_slack_delta_source = "not_evaluated";
	        std::string best_local_predicted_slack_delta_weight_mode =
	            slack_delta_weight_mode;
	        py::list best_local_predicted_slack_delta_points;
	        double best_local_predicted_raw_timing_gain = action.local_delta_delay_ps;
        double best_local_predicted_effective_timing_gain = 0.0;
        std::string best_local_predicted_downstream_delay_source = "python_or_proposal_local_delta_delay";
        std::string best_local_predicted_fanin_penalty_source =
            "cpp_hot_path_fanin_neighborhood_penalty_not_evaluated";
        int best_local_predicted_fanin_net_count = 0;
        int best_local_predicted_affected_fo_cell_count = 0;
        int best_local_predicted_positive_input_cap_delta_count = 0;
        double best_local_predicted_fanin_penalty_ps_per_pf = fanin_penalty_ps_per_pf;
        std::string best_local_delta_delay_estimator_source = "python_or_proposal_local_delta_delay";
        std::string best_local_delta_slew_estimator_source = "python_or_proposal_local_delta_slew";
        std::string best_local_delta_delay_cpp_fallback_reason =
            "cpp_hot_path_delay_estimator_unavailable";
        std::string best_local_delta_slew_cpp_fallback_reason =
            "cpp_hot_path_slew_estimator_unavailable";
        int best_local_delta_delay_cpp_arc_count = 0;
        int best_local_delta_slew_cpp_arc_count = 0;
        double best_local_delta_cap = action.local_delta_cap;
        std::string best_local_delta_cap_estimator_source = "python_or_proposal_local_delta_cap";
        int best_local_delta_cap_compared_input_port_count = 0;
        int best_local_delta_cap_positive_input_port_count = 0;
        std::string best_candidate_delta_source = "proposal_local_delta_fallback";
        std::string best_local_context_source = "unavailable";
        std::string best_input_slew_source = "proposal_fallback";
        std::string best_load_cap_source = "proposal_fallback";
        std::string best_local_context_fallback_reason = "not_evaluated";
        bool best_candidate_context_stale_protected = false;
        int best_local_context_input_slew_pin_count = 0;
        int best_local_context_load_cap_pin_count = 0;
        int best_local_context_output_slack_pin_count = 0;
        bool best_local_slack_budget_used = false;
        double best_local_slack_budget_ps = 0.0;
        double best_component_slack_budget_ps = 0.0;
        double best_effective_candidate_budget_ps = 0.0;
        double best_local_slack_ps = 0.0;
        double best_local_arrival_ps = 0.0;
        double best_local_required_ps = 0.0;
        std::string best_local_slack_clamp_source = "component_residual_budget";
        double best_current_leakage = 0.0;
        double best_target_leakage = 0.0;
        double best_uncapped_delta_tns = 0.0;
        std::string best_leakage_delta_estimator_source =
            "cpp_hot_path_liberty_leakage_delta_estimator_unavailable";
        std::string best_leakage_delta_cpp_fallback_reason =
            "missing_liberty_cell_or_leakage";
        double best_actual_delta_obj =
            noop_baseline_enabled ? 0.0 : -std::numeric_limits<double>::infinity();
        CompactBridgeSizingTrial best_sizing_trial;
        py::list local_candidate_trial_debug_rows;

        for (const auto& trial : sizing_trials) {
          const std::string& trial_master_name = trial.master_name;
          odb::dbMaster* target_master = db->findMaster(trial_master_name.c_str());
          if (target_master == nullptr || target_master == old_master) {
            continue;
          }
          trial_count += 1;
          local_residual_trial_count += 1;
          DiffGuidedBatchCurrentLocalTimingContext current_context;
          if (use_local_candidate_delta) {
            current_context = queryDiffGuidedBatchCurrentLocalTimingContext(
                local_estimator_timing,
                local_estimator_sta == nullptr || local_estimator_sta->units() == nullptr
                    ? nullptr
                    : local_estimator_sta->units()->timeUnit(),
                local_estimator_sta == nullptr || local_estimator_sta->units() == nullptr
                    ? nullptr
                    : local_estimator_sta->units()->capacitanceUnit(),
                local_estimator_corner,
                inst);
            local_context_query_count += 1;
            if (!current_context.used) {
              trial_fallback_count += 1;
            }
          }
          DiffGuidedBatchCppLocalCapEstimatorResult cpp_local_cap =
              diffGuidedBatchCppLocalInputCapEstimator(
                  local_estimator_network, old_master, target_master);
          local_cap_query_count += 1;
          const double local_delta_cap_for_trial =
              cpp_local_cap.used ? cpp_local_cap.local_delta_cap : action.local_delta_cap;
          const std::string local_delta_cap_estimator_source =
              cpp_local_cap.used ? cpp_local_cap.estimator_source
                                 : "python_or_proposal_local_delta_cap";
          if (cpp_local_cap.used) {
            cpp_hot_path_input_cap_estimator_count += 1;
            if (cpp_local_cap.local_delta_cap > 0.0) {
              cpp_hot_path_input_cap_estimator_positive_count += 1;
            }
          }
          DiffGuidedBatchCppLocalDelaySlewEstimatorResult cpp_local_delay_slew =
              diffGuidedBatchCppLocalDelaySlewEstimator(local_estimator_sta,
                                                        local_estimator_network,
                                                        local_estimator_corner,
                                                        old_master,
                                                        target_master,
                                                        current_context,
                                                        action,
                                                        local_delta_cap_for_trial);
          local_dcalc_query_count += 1;
          CompactBridgeActionProposal trial_action = action;
          if (cpp_local_delay_slew.used) {
            cpp_hot_path_delay_estimator_count += 1;
            cpp_hot_path_slew_estimator_count += 1;
            trial_action.local_delta_delay_ps = cpp_local_delay_slew.local_delta_delay_ps;
            trial_action.local_delta_slew_ps = cpp_local_delay_slew.local_delta_slew_ps;
          } else {
            cpp_hot_path_delay_estimator_fallback_count += 1;
            cpp_hot_path_slew_estimator_fallback_count += 1;
          }
	          const auto slack_delta_begin = std::chrono::steady_clock::now();
	          const DiffGuidedBatchCellReplaceSlackDeltaResult slack_delta_result =
	              diffGuidedBatchCellReplaceSlackDeltaEvaluator(
	                  local_estimator_sta,
	                  local_estimator_timing,
	                  local_estimator_sta == nullptr || local_estimator_sta->units() == nullptr
	                      ? nullptr
	                      : local_estimator_sta->units()->timeUnit(),
	                  local_estimator_sta == nullptr || local_estimator_sta->units() == nullptr
	                      ? nullptr
	                      : local_estimator_sta->units()->capacitanceUnit(),
	                  local_estimator_network,
	                  local_estimator_corner,
	                  inst,
	                  old_master,
	                  target_master,
	                  current_context,
	                  cpp_local_delay_slew,
	                  fanin_penalty_ps_per_pf,
	                  slack_delta_evaluator_mode,
	                  slack_delta_weight_mode,
	                  npath_snapshot,
	                  fanin_raw_delta_by_inst_id,
	                  fanin_signed_delta_by_inst_id,
	                  fanin_seen_epoch_by_inst_id,
	                  fanin_touched_inst_ids,
	                  ++fanin_scratch_epoch);
	          slack_delta_eval_count += 1;
	          slack_delta_eval_ms += elapsedMs(slack_delta_begin);
	          slack_delta_dcalc_query_count += slack_delta_result.dcalc_query_count;
	          slack_delta_fanin_net_count += slack_delta_result.fanin_net_count;
	          slack_delta_affected_sink_count += slack_delta_result.affected_fo_cell_count;
	          slack_delta_fallback_count += slack_delta_result.fallback_point_count;
	          const double local_predicted_downstream_delay_gain =
	              cpp_local_delay_slew.used ? cpp_local_delay_slew.signed_delay_delta
	                                        : trial_action.local_delta_delay_ps;
	          const double local_predicted_downstream_slew_gain =
	              cpp_local_delay_slew.used ? cpp_local_delay_slew.signed_slew_delta
	                                        : -std::max(0.0, trial_action.local_delta_slew_ps);
	          const double local_predicted_fanin_neighborhood_penalty =
	              slack_delta_result.fanin_slack_penalty_ps;
	          const double local_predicted_raw_timing_gain =
	              slack_delta_result.used
	                  ? (slack_delta_weight_mode == "npath_weighted_tns"
	                         ? slack_delta_result.weighted_net_tns_delta_ps
	                         : slack_delta_result.net_slack_obj_delta_ps)
	                  : local_predicted_downstream_delay_gain - local_predicted_fanin_neighborhood_penalty;
          const DiffGuidedBatchEffectiveCandidateBudget effective_candidate_budget =
              use_local_candidate_delta
                  ? diffGuidedBatchEffectiveLocalCandidateBudget(
                        discounted_residual_budget, current_context)
                  : DiffGuidedBatchEffectiveCandidateBudget{discounted_residual_budget,
                                                            discounted_residual_budget,
                                                            0.0,
                                                            false,
                                                            "component_residual_budget"};
	          const double local_predicted_effective_timing_gain =
	              local_predicted_raw_timing_gain >= 0.0
	                  ? std::min(local_predicted_raw_timing_gain,
	                             std::max(0.0, effective_candidate_budget.budget))
	                  : local_predicted_raw_timing_gain;
          const double residual_budget_clamped_gain =
              use_local_candidate_delta
                  ? local_predicted_effective_timing_gain
                  : diffGuidedBatchCandidateDeltaForTrial(
                        trial_action, effective_candidate_budget.budget);
          const double synthetic_actual_delta_tns =
              diffGuidedBatchLocalResidualTrialDelta(
                  trial_action, trial.step, discounted_residual_budget);
          if (use_local_candidate_delta) {
            local_candidate_delta_trial_count += 1;
          }
          const double uncapped_actual_delta_tns =
              use_local_candidate_delta ? residual_budget_clamped_gain
                                        : synthetic_actual_delta_tns;
          const double gain_capped_delta_tns =
              !use_local_candidate_delta && local_residual_max_gain_per_action > 0.0
                  ? std::min(uncapped_actual_delta_tns, local_residual_max_gain_per_action)
                  : uncapped_actual_delta_tns;
          const double actual_delta_tns = gain_capped_delta_tns;
          const double actual_delta_wns = actual_delta_tns > 0.0 ? actual_delta_tns : 0.0;
          const DiffGuidedBatchCppLeakageDeltaEstimatorResult cpp_leakage_delta =
              diffGuidedBatchCppLeakageDeltaEstimator(
                  local_estimator_network, old_master, target_master);
          local_leakage_query_count += 1;
          const double leakage_delta = cpp_leakage_delta.used
                                           ? cpp_leakage_delta.leakage_delta
                                           : 0.0;
          if (cpp_leakage_delta.used) {
            cpp_hot_path_leakage_delta_estimator_count += 1;
            if (cpp_leakage_delta.leakage_delta > 0.0) {
              cpp_hot_path_leakage_delta_positive_count += 1;
            }
          } else {
            cpp_hot_path_leakage_delta_fallback_count += 1;
          }
          const double local_delta_slew_penalty =
              use_local_candidate_delta
                  ? diffGuidedBatchCandidateLocalDeltaSlewPenalty(
                        trial_action, local_delta_slew_penalty_weight)
                  : diffGuidedBatchLocalDeltaSlewPenalty(
                        trial_action, trial.step, local_delta_slew_penalty_weight);
          const double local_delta_cap_penalty =
              use_local_candidate_delta
                  ? diffGuidedBatchCandidateLocalDeltaCapPenalty(
                        local_delta_cap_for_trial, local_delta_cap_penalty_weight)
                  : local_delta_cap_penalty_weight
                        * std::max(0.0, local_delta_cap_for_trial)
                        * diffGuidedBatchTrialStepScale(action, trial.step);
          const double estimated_output_slew_after = std::max(
              0.0, current_context.output_slew_for_violation_ps - trial_action.local_delta_slew_ps);
          sta::LibertyPort* target_slew_limit_port =
              diffGuidedBatchFindLibertyPort(local_estimator_network,
                                             target_master,
                                             current_context.output_slew_limit_port_name);
          const double target_slew_limit_ps =
              target_slew_limit_port == nullptr
                  ? current_context.output_slew_limit_ps
                  : resolveLibPinSlewLimitForPythonPs(
                        local_estimator_sta == nullptr || local_estimator_sta->units() == nullptr
                            ? nullptr
                            : local_estimator_sta->units()->timeUnit(),
                        target_slew_limit_port);
          const double local_delta_slew_violation_improvement =
              diffGuidedBatchLocalSlewCapViolationImprovement(
                  current_context.output_slew_for_violation_ps,
                  estimated_output_slew_after,
                  current_context.output_slew_limit_ps,
                  target_slew_limit_ps,
                  local_delta_slew_penalty_weight);
          sta::LibertyPort* target_cap_limit_port =
              diffGuidedBatchFindLibertyPort(local_estimator_network,
                                             target_master,
                                             current_context.output_cap_limit_port_name);
          const double target_cap_limit_pf =
              target_cap_limit_port == nullptr
                  ? current_context.output_cap_limit_pf
                  : diffGuidedBatchResolveLibPinCapLimitPf(
                        local_estimator_sta == nullptr || local_estimator_sta->units() == nullptr
                            ? nullptr
                            : local_estimator_sta->units()->capacitanceUnit(),
                        target_cap_limit_port);
          const double local_delta_cap_violation_improvement =
              diffGuidedBatchLocalSlewCapViolationImprovement(
                  current_context.output_cap_for_violation_pf,
                  current_context.output_cap_for_violation_pf,
                  current_context.output_cap_limit_pf,
                  target_cap_limit_pf,
                  local_delta_cap_penalty_weight);
          const double actual_delta_obj = use_local_objective
                                              ? diffGuidedBatchLocalResidualActualDeltaObj(
                                                    actual_delta_tns,
                                                    leakage_delta,
                                                    leakage_weight,
                                                    local_delta_slew_penalty,
                                                    local_delta_cap_penalty,
                                                    local_delta_slew_violation_improvement,
                                                    local_delta_cap_violation_improvement)
                                              : actual_delta_tns;
          const bool trial_would_beat_current_best =
              diffGuidedBatchCandidateBetterThanBest(has_valid_trial,
                                                     actual_delta_obj,
                                                     trial.step,
                                                     best_actual_delta_obj,
                                                     best_trial_step);
          py::dict trial_debug;
          trial_debug["trial_master_name"] = trial_master_name;
          trial_debug["trial_step"] = trial.step;
          trial_debug["trial_python_seed_local_delta_delay_ps"] =
              action.local_delta_delay_ps;
          trial_debug["trial_python_seed_local_delta_slew_ps"] =
              action.local_delta_slew_ps;
          trial_debug["trial_python_seed_local_delta_cap"] =
              action.local_delta_cap;
          trial_debug["trial_predicted_seed_delta_obj"] =
              action.predicted_delta_obj;
          trial_debug["trial_sensitivity_score"] = action.sensitivity_score;
          trial_debug["trial_cpp_dcalc_used"] = cpp_local_delay_slew.used;
          trial_debug["trial_cpp_signed_delay_delta_ps"] =
              cpp_local_delay_slew.signed_delay_delta;
          trial_debug["trial_cpp_signed_slew_delta_ps"] =
              cpp_local_delay_slew.signed_slew_delta;
          trial_debug["trial_cpp_local_delta_delay_ps"] =
              trial_action.local_delta_delay_ps;
          trial_debug["trial_cpp_local_delta_slew_ps"] =
              trial_action.local_delta_slew_ps;
          trial_debug["trial_cpp_current_delay_ps"] =
              cpp_local_delay_slew.selected_current_delay_ps;
          trial_debug["trial_cpp_target_delay_ps"] =
              cpp_local_delay_slew.selected_target_delay_ps;
          trial_debug["trial_cpp_current_slew_ps"] =
              cpp_local_delay_slew.selected_current_slew_ps;
          trial_debug["trial_cpp_target_slew_ps"] =
              cpp_local_delay_slew.selected_target_slew_ps;
          trial_debug["trial_cpp_dcalc_arc_count"] =
              cpp_local_delay_slew.compared_arc_count;
          trial_debug["trial_cpp_dcalc_positive_delay_arc_count"] =
              cpp_local_delay_slew.positive_delay_arc_count;
          trial_debug["trial_cpp_dcalc_positive_slew_arc_count"] =
              cpp_local_delay_slew.positive_slew_arc_count;
          trial_debug["trial_cpp_dcalc_fallback_reason"] =
              cpp_local_delay_slew.used ? "" : cpp_local_delay_slew.fallback_reason;
          trial_debug["trial_output_slack_used"] =
              current_context.output_slack_used;
          trial_debug["trial_output_slack_ps"] =
              current_context.output_slack_ps;
          trial_debug["trial_output_arrival_ps"] =
              current_context.output_arrival_ps;
          trial_debug["trial_output_required_ps"] =
              current_context.output_required_ps;
          trial_debug["trial_local_context_source"] =
              current_context.local_context_source;
          trial_debug["trial_local_context_fallback_reason"] =
              current_context.fallback_reason;
          trial_debug["trial_input_slew_ps"] = current_context.input_slew_ps;
          trial_debug["trial_load_cap_pf"] = current_context.load_cap_pf;
          trial_debug["trial_slack_delta_used"] = slack_delta_result.used;
          trial_debug["trial_downstream_slack_delta"] =
              slack_delta_result.downstream_slack_improvement_ps;
          trial_debug["trial_downstream_npath_weight"] =
              slack_delta_result.downstream_npath_weight;
          trial_debug["trial_downstream_weighted_slack_delta"] =
              slack_delta_result.downstream_weighted_slack_improvement_ps;
          trial_debug["trial_downstream_old_slack_ps"] =
              slack_delta_result.downstream_old_slack_ps;
          trial_debug["trial_downstream_new_slack_ps"] =
              slack_delta_result.downstream_new_slack_ps;
          trial_debug["trial_downstream_delta_delay_ps"] =
              slack_delta_result.downstream_delta_delay_ps;
          trial_debug["trial_downstream_dcalc_source"] =
              slack_delta_result.downstream_dcalc_source;
          trial_debug["trial_fanin_slack_delta"] =
              slack_delta_result.fanin_slack_delta_ps;
          trial_debug["trial_fanin_weighted_slack_delta"] =
              slack_delta_result.fanin_weighted_slack_delta_ps;
          trial_debug["trial_fanin_slack_penalty_ps"] =
              slack_delta_result.fanin_slack_penalty_ps;
          trial_debug["trial_net_slack_delta"] =
              slack_delta_result.net_slack_obj_delta_ps;
          trial_debug["trial_weighted_net_tns_delta"] =
              slack_delta_result.weighted_net_tns_delta_ps;
          trial_debug["trial_slack_delta_weight_mode"] =
              slack_delta_result.slack_delta_weight_mode;
          trial_debug["trial_slack_delta_point_count"] =
              slack_delta_result.downstream_point_count
              + slack_delta_result.fanin_point_count;
          trial_debug["trial_slack_delta_dcalc_query_count"] =
              slack_delta_result.dcalc_query_count;
          trial_debug["trial_slack_delta_fallback_point_count"] =
              slack_delta_result.fallback_point_count;
          trial_debug["trial_slack_delta_source"] =
              slack_delta_result.estimator_source;
          trial_debug["trial_fanin_source"] = slack_delta_result.fanin_source;
          trial_debug["trial_local_predicted_raw_timing_gain"] =
              local_predicted_raw_timing_gain;
          trial_debug["trial_local_predicted_effective_timing_gain"] =
              local_predicted_effective_timing_gain;
          trial_debug["trial_residual_budget_clamped_gain"] =
              residual_budget_clamped_gain;
          trial_debug["trial_uncapped_actual_delta_tns"] =
              uncapped_actual_delta_tns;
          trial_debug["trial_local_delta_cap_violation_improvement"] =
              local_delta_cap_violation_improvement;
          trial_debug["trial_local_delta_slew_violation_improvement"] =
              local_delta_slew_violation_improvement;
          trial_debug["trial_local_delta_cap_penalty"] =
              local_delta_cap_penalty;
          trial_debug["trial_local_delta_slew_penalty"] =
              local_delta_slew_penalty;
          trial_debug["trial_leakage_delta"] = leakage_delta;
          trial_debug["trial_current_leakage"] =
              cpp_leakage_delta.current_leakage;
          trial_debug["trial_target_leakage"] =
              cpp_leakage_delta.target_leakage;
          trial_debug["trial_leakage_delta_used"] = cpp_leakage_delta.used;
          trial_debug["trial_actual_delta_tns"] = actual_delta_tns;
          trial_debug["trial_actual_delta_obj"] = actual_delta_obj;
          trial_debug["trial_would_beat_current_best"] =
              trial_would_beat_current_best;
          local_candidate_trial_debug_rows.append(trial_debug);
          if (trial_would_beat_current_best) {
            has_valid_trial = true;
            best_trial_is_noop = false;
            best_master_name = target_master->getName();
            best_trial_step = trial.step;
            best_sizing_trial = trial;
            best_delta_tns = actual_delta_tns;
            best_delta_wns = actual_delta_wns;
            best_leakage_delta = leakage_delta;
            best_uncapped_delta_tns = uncapped_actual_delta_tns;
            best_local_delta_slew_penalty = local_delta_slew_penalty;
            best_local_delta_cap_penalty = local_delta_cap_penalty;
            best_local_delta_slew_violation_improvement =
                local_delta_slew_violation_improvement;
            best_local_delta_cap_violation_improvement =
                local_delta_cap_violation_improvement;
            best_local_delta_delay_cpp = trial_action.local_delta_delay_ps;
            best_local_delta_slew_cpp = trial_action.local_delta_slew_ps;
            best_local_predicted_downstream_delay_gain =
                local_predicted_downstream_delay_gain;
            best_local_predicted_downstream_slew_gain =
                local_predicted_downstream_slew_gain;
	            best_local_predicted_fanin_neighborhood_penalty =
	                local_predicted_fanin_neighborhood_penalty;
	            best_local_predicted_downstream_slack_delta =
	                slack_delta_result.downstream_slack_improvement_ps;
	            best_local_predicted_downstream_npath_weight =
	                slack_delta_result.downstream_npath_weight;
	            best_local_predicted_downstream_weighted_slack_delta =
	                slack_delta_result.downstream_weighted_slack_improvement_ps;
	            best_local_predicted_downstream_old_slack_ps =
	                slack_delta_result.downstream_old_slack_ps;
	            best_local_predicted_downstream_new_slack_ps =
	                slack_delta_result.downstream_new_slack_ps;
	            best_local_predicted_downstream_delta_delay_ps =
	                slack_delta_result.downstream_delta_delay_ps;
	            best_local_predicted_downstream_dcalc_source =
	                slack_delta_result.downstream_dcalc_source;
	            best_local_predicted_fanin_slack_delta =
	                slack_delta_result.fanin_slack_delta_ps;
	            best_local_predicted_fanin_weighted_slack_delta =
	                slack_delta_result.fanin_weighted_slack_delta_ps;
	            best_local_predicted_net_slack_delta =
	                slack_delta_result.net_slack_obj_delta_ps;
	            best_local_predicted_weighted_net_tns_delta =
	                slack_delta_result.weighted_net_tns_delta_ps;
	            best_local_predicted_downstream_point_count =
	                slack_delta_result.downstream_point_count;
	            best_local_predicted_fanin_point_count =
	                slack_delta_result.fanin_point_count;
	            best_local_predicted_dcalc_query_count =
	                slack_delta_result.dcalc_query_count;
	            best_local_predicted_fallback_point_count =
	                slack_delta_result.fallback_point_count;
	            best_local_predicted_slack_delta_source =
	                slack_delta_result.estimator_source;
	            best_local_predicted_slack_delta_weight_mode =
	                slack_delta_result.slack_delta_weight_mode;
	            best_local_predicted_slack_delta_points =
	                diffGuidedBatchSlackDeltaPointsToPyList(slack_delta_result.points);
	            best_local_predicted_raw_timing_gain =
	                local_predicted_raw_timing_gain;
            best_local_predicted_effective_timing_gain =
                local_predicted_effective_timing_gain;
            best_local_predicted_downstream_delay_source =
                cpp_local_delay_slew.used ? cpp_local_delay_slew.estimator_source
                                          : "python_or_proposal_local_delta_delay";
	            best_local_predicted_fanin_penalty_source =
	                slack_delta_result.fanin_source;
	            best_local_predicted_fanin_net_count =
	                slack_delta_result.fanin_net_count;
	            best_local_predicted_affected_fo_cell_count =
	                slack_delta_result.affected_fo_cell_count;
	            best_local_predicted_positive_input_cap_delta_count =
	                slack_delta_result.positive_input_cap_delta_count;
	            best_local_predicted_fanin_penalty_ps_per_pf =
	                fanin_penalty_ps_per_pf;
            best_local_delta_delay_estimator_source =
                cpp_local_delay_slew.used ? cpp_local_delay_slew.estimator_source
                                          : "python_or_proposal_local_delta_delay";
            best_local_delta_slew_estimator_source =
                cpp_local_delay_slew.used ? cpp_local_delay_slew.estimator_source
                                          : "python_or_proposal_local_delta_slew";
            best_local_delta_delay_cpp_fallback_reason =
                cpp_local_delay_slew.used ? "" : cpp_local_delay_slew.fallback_reason;
            best_local_delta_slew_cpp_fallback_reason =
                cpp_local_delay_slew.used ? "" : cpp_local_delay_slew.fallback_reason;
            best_local_delta_delay_cpp_arc_count =
                cpp_local_delay_slew.compared_arc_count;
            best_local_delta_slew_cpp_arc_count =
                cpp_local_delay_slew.compared_arc_count;
            best_local_delta_cap = local_delta_cap_for_trial;
            best_local_delta_cap_estimator_source = local_delta_cap_estimator_source;
            best_local_delta_cap_compared_input_port_count =
                cpp_local_cap.compared_input_port_count;
            best_local_delta_cap_positive_input_port_count =
                cpp_local_cap.positive_input_port_count;
            best_candidate_delta_source =
                current_context.used && cpp_local_delay_slew.used
                    ? "opensta_current_context_dcalc"
                    : (cpp_local_delay_slew.used ? "mixed" : "proposal_local_delta_fallback");
            best_local_context_source = current_context.local_context_source;
            best_input_slew_source = current_context.input_slew_source;
            best_load_cap_source = current_context.load_cap_source;
            best_local_context_fallback_reason = current_context.fallback_reason;
            best_candidate_context_stale_protected =
                current_context.used && cpp_local_delay_slew.used;
            best_local_context_input_slew_pin_count =
                current_context.input_slew_pin_count;
            best_local_context_load_cap_pin_count =
                current_context.load_cap_pin_count;
            best_local_context_output_slack_pin_count =
                current_context.output_slack_pin_count;
            best_local_slack_budget_used = effective_candidate_budget.pin_slack_used;
            best_local_slack_budget_ps = effective_candidate_budget.pin_slack_budget;
            best_component_slack_budget_ps = effective_candidate_budget.component_budget;
            best_effective_candidate_budget_ps = effective_candidate_budget.budget;
            best_local_slack_ps = current_context.output_slack_ps;
            best_local_arrival_ps = current_context.output_arrival_ps;
            best_local_required_ps = current_context.output_required_ps;
            best_local_slack_clamp_source = effective_candidate_budget.source;
            best_current_leakage = cpp_leakage_delta.current_leakage;
            best_target_leakage = cpp_leakage_delta.target_leakage;
            best_leakage_delta_estimator_source =
                cpp_leakage_delta.used
                    ? cpp_leakage_delta.estimator_source
                    : "cpp_hot_path_liberty_leakage_delta_estimator_unavailable";
            best_leakage_delta_cpp_fallback_reason =
                cpp_leakage_delta.used ? "" : cpp_leakage_delta.fallback_reason;
            best_actual_delta_obj = actual_delta_obj;
          }
        }

        result["local_candidate_trial_debug_rows"] =
            local_candidate_trial_debug_rows;

        if (!has_valid_trial) {
          result["status"] = "unchanged";
          result["reject_reason"] = "no_valid_trial_master";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        if (best_trial_is_noop) {
          result["status"] = "rejected";
          result["accepted"] = false;
          result["reject_reason"] = "noop_baseline_selected";
          result["actual_delta_tns"] = 0.0;
          result["actual_delta_obj"] = 0.0;
          result["best_trial_step"] = 0;
          result["trial_mode"] = trial_mode;
          result["noop_baseline_enabled"] = noop_baseline_enabled;
          result["reject_nonpositive_trial_enabled"] = reject_nonpositive_trial;
          result["nonpositive_trial_eps"] = nonpositive_trial_eps;
          result["local_objective_align_global"] = local_objective_align_global;
          result["local_delta_slew_penalty_weight"] = local_delta_slew_penalty_weight;
          result["local_delta_cap_penalty_weight"] = local_delta_cap_penalty_weight;
          result["local_delta_slew_penalty_weight_source"] =
              local_delta_slew_weight_info.source;
          result["local_delta_cap_penalty_weight_source"] =
              local_delta_cap_weight_info.source;
          result["timing_objective_lane"] =
              pyStringOrDefault(config, "timing_objective_lane", "timing_only");
          noop_selected_count += 1;
          best_trial_noop_count += 1;
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        if (reject_nonpositive_trial && best_actual_delta_obj <= nonpositive_trial_eps) {
          result["status"] = "rejected";
          result["accepted"] = false;
          result["reject_reason"] = "nonpositive_trial_over_noop";
          result["actual_delta_tns"] = best_delta_tns;
          result["actual_delta_obj"] = best_actual_delta_obj;
          result["actual_delta_tns_semantics"] = "estimated";
          result["local_predicted_delta_tns_like"] = best_delta_tns;
          result["local_predicted_delta_obj"] = best_actual_delta_obj;
          result["local_predicted_downstream_delay_gain"] =
              best_local_predicted_downstream_delay_gain;
          result["local_predicted_downstream_slew_gain"] =
              best_local_predicted_downstream_slew_gain;
          result["local_predicted_downstream_delay_source"] =
              best_local_predicted_downstream_delay_source;
          result["local_predicted_fanin_neighborhood_penalty"] =
              best_local_predicted_fanin_neighborhood_penalty;
          result["local_predicted_fanin_penalty_source"] =
              best_local_predicted_fanin_penalty_source;
          result["local_predicted_fanin_net_count"] =
              best_local_predicted_fanin_net_count;
          result["local_predicted_affected_fo_cell_count"] =
              best_local_predicted_affected_fo_cell_count;
          result["local_predicted_positive_input_cap_delta_count"] =
              best_local_predicted_positive_input_cap_delta_count;
	          result["local_predicted_fanin_penalty_ps_per_pf"] =
	              best_local_predicted_fanin_penalty_ps_per_pf;
	          result["local_predicted_slack_delta_source"] =
	              best_local_predicted_slack_delta_source;
	          result["local_predicted_slack_delta_weight_mode"] =
	              best_local_predicted_slack_delta_weight_mode;
	          result["local_predicted_downstream_slack_delta"] =
	              best_local_predicted_downstream_slack_delta;
	          result["local_predicted_downstream_npath_weight"] =
	              best_local_predicted_downstream_npath_weight;
	          result["local_predicted_downstream_weighted_slack_delta"] =
	              best_local_predicted_downstream_weighted_slack_delta;
	          result["local_predicted_downstream_old_slack_ps"] =
	              best_local_predicted_downstream_old_slack_ps;
	          result["local_predicted_downstream_new_slack_ps"] =
	              best_local_predicted_downstream_new_slack_ps;
	          result["local_predicted_downstream_delta_delay_ps"] =
	              best_local_predicted_downstream_delta_delay_ps;
	          result["local_predicted_downstream_dcalc_source"] =
	              best_local_predicted_downstream_dcalc_source;
	          result["local_predicted_fanin_slack_delta"] =
	              best_local_predicted_fanin_slack_delta;
	          result["local_predicted_fanin_weighted_slack_delta"] =
	              best_local_predicted_fanin_weighted_slack_delta;
	          result["local_predicted_net_slack_delta"] =
	              best_local_predicted_net_slack_delta;
	          result["local_predicted_weighted_net_tns_delta"] =
	              best_local_predicted_weighted_net_tns_delta;
	          result["local_predicted_downstream_point_count"] =
	              best_local_predicted_downstream_point_count;
	          result["local_predicted_fanin_point_count"] =
	              best_local_predicted_fanin_point_count;
	          result["local_predicted_dcalc_query_count"] =
	              best_local_predicted_dcalc_query_count;
	          result["local_predicted_fallback_point_count"] =
	              best_local_predicted_fallback_point_count;
	          result["local_predicted_slack_delta_points"] =
	              best_local_predicted_slack_delta_points;
	          result["local_predicted_raw_timing_gain"] =
	              best_local_predicted_raw_timing_gain;
          result["local_predicted_effective_timing_gain"] =
              best_local_predicted_effective_timing_gain;
          result["best_trial_step"] = best_trial_step;
          result["trial_mode"] = trial_mode;
          result["noop_baseline_enabled"] = noop_baseline_enabled;
          result["reject_nonpositive_trial_enabled"] = reject_nonpositive_trial;
          result["nonpositive_trial_eps"] = nonpositive_trial_eps;
          result["local_objective_align_global"] = local_objective_align_global;
          result["local_delta_slew_penalty_weight"] = local_delta_slew_penalty_weight;
          result["local_delta_cap_penalty_weight"] = local_delta_cap_penalty_weight;
          result["local_delta_slew_penalty_weight_source"] =
              local_delta_slew_weight_info.source;
          result["local_delta_cap_penalty_weight_source"] =
              local_delta_cap_weight_info.source;
          result["timing_objective_lane"] =
              pyStringOrDefault(config, "timing_objective_lane", "timing_only");
          nonpositive_trial_reject_count += 1;
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        best_trial_nonzero_count += 1;

        result["supported"] = true;
        result["baseline_metrics"] = baseline_metrics;
        result["trial_metrics"] = baseline_metrics;
        result["old_master"] = old_master->getName();
        result["new_master"] = best_master_name;
        writeDiffGuidedBatchFinalSizingMetadata(result, action, best_sizing_trial);
        result["actual_delta_tns"] = best_delta_tns;
        if (trial_mode == "local_candidate_delta") {
          result["predicted_candidate_delta_obj"] = best_actual_delta_obj;
          result["predicted_candidate_timing_gain"] = best_local_delta_delay_cpp;
          result["residual_budget_clamped_gain"] = best_delta_tns;
          result["synthetic_actual_delta_tns"] = best_uncapped_delta_tns;
        }
        result["uncapped_actual_delta_tns"] = best_uncapped_delta_tns;
        result["gain_capped_delta_tns"] = best_delta_tns;
        result["actual_delta_wns"] = best_delta_wns;
        result["actual_delta_tns_semantics"] = "estimated";
        result["actual_delta_obj"] = best_actual_delta_obj;
        result["local_predicted_delta_tns_like"] = best_delta_tns;
        result["local_predicted_delta_obj"] = best_actual_delta_obj;
        result["local_predicted_downstream_delay_gain"] =
            best_local_predicted_downstream_delay_gain;
        result["local_predicted_downstream_slew_gain"] =
            best_local_predicted_downstream_slew_gain;
        result["local_predicted_downstream_delay_source"] =
            best_local_predicted_downstream_delay_source;
        result["local_predicted_fanin_neighborhood_penalty"] =
            best_local_predicted_fanin_neighborhood_penalty;
        result["local_predicted_fanin_penalty_source"] =
            best_local_predicted_fanin_penalty_source;
        result["local_predicted_fanin_net_count"] =
            best_local_predicted_fanin_net_count;
        result["local_predicted_affected_fo_cell_count"] =
            best_local_predicted_affected_fo_cell_count;
        result["local_predicted_positive_input_cap_delta_count"] =
            best_local_predicted_positive_input_cap_delta_count;
	        result["local_predicted_fanin_penalty_ps_per_pf"] =
	            best_local_predicted_fanin_penalty_ps_per_pf;
	        result["local_predicted_slack_delta_source"] =
	            best_local_predicted_slack_delta_source;
	        result["local_predicted_slack_delta_weight_mode"] =
	            best_local_predicted_slack_delta_weight_mode;
	        result["local_predicted_downstream_slack_delta"] =
	            best_local_predicted_downstream_slack_delta;
	        result["local_predicted_downstream_npath_weight"] =
	            best_local_predicted_downstream_npath_weight;
	        result["local_predicted_downstream_weighted_slack_delta"] =
	            best_local_predicted_downstream_weighted_slack_delta;
	        result["local_predicted_downstream_old_slack_ps"] =
	            best_local_predicted_downstream_old_slack_ps;
	        result["local_predicted_downstream_new_slack_ps"] =
	            best_local_predicted_downstream_new_slack_ps;
	        result["local_predicted_downstream_delta_delay_ps"] =
	            best_local_predicted_downstream_delta_delay_ps;
	        result["local_predicted_downstream_dcalc_source"] =
	            best_local_predicted_downstream_dcalc_source;
	        result["local_predicted_fanin_slack_delta"] =
	            best_local_predicted_fanin_slack_delta;
	        result["local_predicted_fanin_weighted_slack_delta"] =
	            best_local_predicted_fanin_weighted_slack_delta;
	        result["local_predicted_net_slack_delta"] =
	            best_local_predicted_net_slack_delta;
	        result["local_predicted_weighted_net_tns_delta"] =
	            best_local_predicted_weighted_net_tns_delta;
	        result["local_predicted_downstream_point_count"] =
	            best_local_predicted_downstream_point_count;
	        result["local_predicted_fanin_point_count"] =
	            best_local_predicted_fanin_point_count;
	        result["local_predicted_dcalc_query_count"] =
	            best_local_predicted_dcalc_query_count;
	        result["local_predicted_fallback_point_count"] =
	            best_local_predicted_fallback_point_count;
	        result["local_predicted_slack_delta_points"] =
	            best_local_predicted_slack_delta_points;
	        result["local_predicted_raw_timing_gain"] =
	            best_local_predicted_raw_timing_gain;
        result["local_predicted_effective_timing_gain"] =
            best_local_predicted_effective_timing_gain;
        result["actual_leakage_delta"] = best_leakage_delta;
        result["leakage_delta"] = best_leakage_delta;
        result["leakage_delta_cpp"] = best_leakage_delta;
        result["leakage_delta_estimator_source"] = best_leakage_delta_estimator_source;
        result["leakage_delta_cpp_fallback_reason"] =
            best_leakage_delta_cpp_fallback_reason;
        result["current_leakage"] = best_current_leakage;
        result["target_leakage"] = best_target_leakage;
        result["local_delta_delay_cpp"] = best_local_delta_delay_cpp;
        result["local_delta_slew_cpp"] = best_local_delta_slew_cpp;
        result["local_delta_delay_estimator_source"] =
            best_local_delta_delay_estimator_source;
        result["local_delta_slew_estimator_source"] =
            best_local_delta_slew_estimator_source;
        result["local_delta_delay_cpp_fallback_reason"] =
            best_local_delta_delay_cpp_fallback_reason;
        result["local_delta_slew_cpp_fallback_reason"] =
            best_local_delta_slew_cpp_fallback_reason;
        result["local_delta_delay_cpp_arc_count"] =
            best_local_delta_delay_cpp_arc_count;
        result["local_delta_slew_cpp_arc_count"] =
            best_local_delta_slew_cpp_arc_count;
        result["local_delta_slew_penalty"] = best_local_delta_slew_penalty;
        result["local_delta_cap_penalty"] = best_local_delta_cap_penalty;
        result["local_delta_slew_violation_improvement"] =
            best_local_delta_slew_violation_improvement;
        result["local_delta_cap_violation_improvement"] =
            best_local_delta_cap_violation_improvement;
        result["local_objective_matches_timing_slew_cap_terms"] = true;
        result["local_delta_slew_penalty_weight"] = local_delta_slew_penalty_weight;
        result["local_delta_cap_penalty_weight"] = local_delta_cap_penalty_weight;
        result["local_delta_cap_estimator_source"] = best_local_delta_cap_estimator_source;
        result["local_delta_cap_cpp"] = best_local_delta_cap;
        result["local_delta_cap_cpp_compared_input_port_count"] =
            best_local_delta_cap_compared_input_port_count;
        result["local_delta_cap_cpp_positive_input_port_count"] =
            best_local_delta_cap_positive_input_port_count;
        result["candidate_delta_source"] = best_candidate_delta_source;
        result["local_context_source"] = best_local_context_source;
        result["input_slew_source"] = best_input_slew_source;
        result["load_cap_source"] = best_load_cap_source;
        result["local_context_fallback_reason"] = best_local_context_fallback_reason;
        result["candidate_context_stale_protected"] =
            best_candidate_context_stale_protected;
        result["local_context_input_slew_pin_count"] =
            best_local_context_input_slew_pin_count;
        result["local_context_load_cap_pin_count"] =
            best_local_context_load_cap_pin_count;
        result["local_context_output_slack_pin_count"] =
            best_local_context_output_slack_pin_count;
        result["local_slack_budget_used"] = best_local_slack_budget_used;
        result["local_slack_budget_ps"] = best_local_slack_budget_ps;
        result["component_slack_budget_ps"] = best_component_slack_budget_ps;
        result["effective_candidate_budget_ps"] = best_effective_candidate_budget_ps;
        result["local_slack_ps"] = best_local_slack_ps;
        result["local_arrival_ps"] = best_local_arrival_ps;
        result["local_required_ps"] = best_local_required_ps;
        result["local_slack_clamp_source"] = best_local_slack_clamp_source;
        result["accept_mode"] = accept_mode;
        result["tns_power_leakage_weight"] = leakage_weight;
        result["local_objective_align_global"] = local_objective_align_global;
        result["noop_baseline_enabled"] = noop_baseline_enabled;
        result["reject_nonpositive_trial_enabled"] = reject_nonpositive_trial;
        result["force_accept_selected_actions"] = force_accept_selected_actions;
        result["nonpositive_trial_eps"] = nonpositive_trial_eps;
        result["timing_objective_lane"] =
            pyStringOrDefault(config, "timing_objective_lane", "timing_only");
        result["local_delta_slew_penalty_weight_source"] =
            local_delta_slew_weight_info.source;
        result["local_delta_cap_penalty_weight_source"] =
            local_delta_cap_weight_info.source;
        result["best_trial_step"] = best_trial_step;
        result["evaluated_trial_count"] = static_cast<int>(sizing_trials.size());
        result["trial_mode"] = trial_mode;
        if (trial_mode == "local_candidate_delta") {
          result["trial_evaluator"] = "local_candidate_delta";
        } else {
          result["trial_evaluator"] = "local_delay_slack_residual";
        }
        result["local_residual_feedback_scale"] = local_residual_feedback_scale;
        result["local_residual_max_gain_per_action"] = local_residual_max_gain_per_action;
        result["local_residual_budget_discount"] = local_residual_budget_discount;
        result["local_residual_budget_before_discount"] = residual_budget_before_discount;
        result["local_residual_budget_after_discount"] = discounted_residual_budget;
        result["local_residual_budget_before"] = residual_budget;
        result["local_residual_budget_after"] = std::max(0.0, residual_budget - best_delta_tns);
        result["trial_query_count"] = 0;
        result["trial_query_count_per_trial"] = 0.0;
        const double accept_score = use_local_objective ? best_actual_delta_obj : best_delta_tns;
        const double feedback_scaled_accept_score =
            accept_score * local_residual_feedback_scale;
        const double feedback_scaled_delta_tns =
            best_delta_tns * local_residual_feedback_scale;
        const double feedback_scaled_delta_obj =
            best_actual_delta_obj * local_residual_feedback_scale;
        result["raw_accept_score"] = accept_score;
        result["feedback_scaled_accept_score"] = feedback_scaled_accept_score;
        result["unscaled_actual_delta_tns"] = best_delta_tns;
        result["unscaled_actual_delta_obj"] = best_actual_delta_obj;
        result["feedback_scaled_delta_tns"] = feedback_scaled_delta_tns;
        result["feedback_scaled_delta_obj"] = feedback_scaled_delta_obj;
        const double min_action_tns_gain =
            pyDoubleOrDefault(config, "min_action_tns_gain", 0.0);
        const bool local_accept_gate_passed =
            !reject_nonpositive_trial
            || (feedback_scaled_accept_score > nonpositive_trial_eps
                && feedback_scaled_accept_score >= min_action_tns_gain);
        result["local_accept_gate_passed"] = local_accept_gate_passed;
        result["force_accept_selected_actions"] = force_accept_selected_actions;
        result["local_accept_gate_bypassed_by_allow_nonpositive"] =
            !reject_nonpositive_trial;
        result["min_action_tns_gain"] = min_action_tns_gain;
        if (force_accept_selected_actions || local_accept_gate_passed) {
          result["actual_delta_tns"] = feedback_scaled_delta_tns;
          result["actual_delta_obj"] = feedback_scaled_delta_obj;
          result["accepted"] = true;
          result["status"] =
              force_accept_selected_actions && !local_accept_gate_passed
                  ? "force_accepted_for_batch_verify"
                  : "accepted";
          result["force_accept_reason"] =
              force_accept_selected_actions && !local_accept_gate_passed
                  ? "debug_bypass_local_accept_score"
                  : "";
          residual_budget_by_component[residual_key.key] =
              std::max(0.0, residual_budget - std::max(0.0, feedback_scaled_delta_tns));
          local_residual_budget_update_count += 1;
          local_residual_accepted_gain_sum += std::max(0.0, feedback_scaled_delta_tns);
          accepted_count += 1;
        } else {
          result["status"] = "rejected";
          result["reject_reason"] = use_local_objective
                                        ? "local_objective_rejected"
                                        : (local_residual_feedback_scale <= 0.0
                                               ? "local_residual_feedback_suppressed"
                                               : "non_improving_local_residual_tns");
          rejected_count += 1;
        }
        CompactBridgeActionResult typed_result = dgb::parseActionResult(result);
        if (typed_result.accepted) {
          typed_result.actual_delta_tns = feedback_scaled_delta_tns;
          typed_result.actual_delta_obj = feedback_scaled_delta_obj;
        }
        typed_summary.evaluator_result_count += 1;
        if (typed_result.accepted) {
          typed_summary.accepted_delta_tns_sum += typed_result.actual_delta_tns;
          typed_summary.accepted_action_results.push_back(typed_result);
        }
        results.append(result);
      }

      evaluator_summary["local_residual_trial_count"] = local_residual_trial_count;
      evaluator_summary["local_residual_accepted_gain_sum"] = local_residual_accepted_gain_sum;
      evaluator_summary["local_residual_budget_update_count"] = local_residual_budget_update_count;
      evaluator_summary["local_residual_component_budget_count"] =
          static_cast<int>(residual_budget_by_component.size());
      evaluator_summary["local_residual_snapshot_budget_count"] =
          local_residual_snapshot_budget_count;
      evaluator_summary["local_candidate_delta_trial_count"] =
          local_candidate_delta_trial_count;
      evaluator_summary["noop_selected_count"] = noop_selected_count;
      evaluator_summary["nonpositive_trial_reject_count"] =
          nonpositive_trial_reject_count;
      evaluator_summary["best_trial_noop_count"] = best_trial_noop_count;
      evaluator_summary["best_trial_nonzero_count"] = best_trial_nonzero_count;
      evaluator_summary["cpp_hot_path_input_cap_estimator_count"] =
          cpp_hot_path_input_cap_estimator_count;
      evaluator_summary["cpp_hot_path_input_cap_estimator_positive_count"] =
          cpp_hot_path_input_cap_estimator_positive_count;
      evaluator_summary["cpp_hot_path_delay_estimator_count"] =
          cpp_hot_path_delay_estimator_count;
      evaluator_summary["cpp_hot_path_slew_estimator_count"] =
          cpp_hot_path_slew_estimator_count;
      evaluator_summary["cpp_hot_path_delay_estimator_fallback_count"] =
          cpp_hot_path_delay_estimator_fallback_count;
      evaluator_summary["cpp_hot_path_slew_estimator_fallback_count"] =
          cpp_hot_path_slew_estimator_fallback_count;
      evaluator_summary["cpp_hot_path_leakage_delta_estimator_count"] =
          cpp_hot_path_leakage_delta_estimator_count;
      evaluator_summary["cpp_hot_path_leakage_delta_positive_count"] =
          cpp_hot_path_leakage_delta_positive_count;
	      evaluator_summary["cpp_hot_path_leakage_delta_fallback_count"] =
	          cpp_hot_path_leakage_delta_fallback_count;
	      evaluator_summary["slack_delta_eval_count"] = slack_delta_eval_count;
	      evaluator_summary["slack_delta_eval_ms"] = slack_delta_eval_ms;
	      evaluator_summary["slack_delta_dcalc_query_count"] =
	          slack_delta_dcalc_query_count;
	      evaluator_summary["slack_delta_fanin_net_count"] =
	          slack_delta_fanin_net_count;
	      evaluator_summary["slack_delta_affected_sink_count"] =
	          slack_delta_affected_sink_count;
	      evaluator_summary["slack_delta_fallback_count"] =
	          slack_delta_fallback_count;
	      evaluator_summary["local_context_query_count"] = local_context_query_count;
      evaluator_summary["local_dcalc_query_count"] = local_dcalc_query_count;
      evaluator_summary["local_cap_query_count"] = local_cap_query_count;
      evaluator_summary["local_leakage_query_count"] = local_leakage_query_count;
    } else if (trial_mode == "mini_batch") {
      struct MiniBatchTrial
      {
        CompactBridgeActionProposal action;
        CompactBridgeSizingTrial sizing_trial;
        odb::dbInst* inst{nullptr};
        odb::dbMaster* old_master{nullptr};
        odb::dbMaster* target_master{nullptr};
        py::dict result;
      };
      std::vector<MiniBatchTrial> pending_trials;

      for (int action_pos = 0; action_pos < static_cast<int>(compact_actions.size()); ++action_pos) {
        const auto& action = compact_actions[action_pos];
        const ActionKind kind = action.kind;
        py::dict result;
        const int64_t action_index = action.action_index;
        result["action_id"] = action_index;
        result["action_index"] = action_index;
        result["action_kind"] = diffGuidedBatchActionKindName(kind);
        result["supported"] = false;
        result["accepted"] = false;
        result["status"] = "rejected";
        result["reject_reason"] = "";
        result["actual_delta_tns"] = 0.0;
        result["actual_delta_wns"] = 0.0;
        result["prescreen_predicted_delta_obj"] =
            vectorValueOrDefault<double>(prescreen_result.prescreen_delta_objs, action_pos, 0.0);
        result["prescreen_predicted_improvement"] =
            vectorValueOrDefault<double>(prescreen_result.predicted_improvements, action_pos, 0.0);

        if (kind != ActionKind::kSizing) {
          result["status"] = "unsupported";
          result["reject_reason"] = diffGuidedBatchActionKindName(kind) + " is reserved and disabled in v1";
          result["affected_net_id"] = action.affected_net_id;
          result["driver_pin_id"] = action.driver_pin_id;
          result["load_pin_id"] = action.load_pin_id;
          result["buffer_master_id"] = action.buffer_master_id;
          result["buffer_master_name"] = action.buffer_master_name;
          result["candidate_location_x"] = action.candidate_location_x;
          result["candidate_location_y"] = action.candidate_location_y;
          unsupported_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        const int inst_id = action.inst_id;
        result["inst_id"] = inst_id;
        result["current_size_idx"] = action.current_size_idx;
        result["target_size_idx"] = action.target_size_idx;
        result["old_timing_coordinate"] = action.old_timing_coordinate;
        result["new_timing_coordinate"] = action.new_timing_coordinate;
        auto* inst = nodeInst(inst_id);
        if (inst == nullptr || inst->isFixed()) {
          result["reject_reason"] = "invalid_or_fixed_instance";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }

        odb::dbMaster* old_master = inst->getMaster();
        if (old_master == nullptr) {
          result["reject_reason"] = "missing_current_master";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
          continue;
        }
        const auto sizing_trials = buildDiffGuidedBatchSizingTrials(action, old_master, config);
        bool found_trial = false;
        for (const auto& trial : sizing_trials) {
          odb::dbMaster* target_master = db->findMaster(trial.master_name.c_str());
          if (target_master == nullptr || target_master == old_master) {
            continue;
          }
          found_trial = true;
          pending_trials.push_back({
              action,
              trial,
              inst,
              old_master,
              target_master,
              result,
          });
          break;
        }
        if (!found_trial) {
          result["reject_reason"] = sizing_trials.empty() ? "missing_target_master" : "no_valid_trial_master";
          rejected_count += 1;
          typed_summary.evaluator_result_count += 1;
          results.append(result);
        }
      }

      for (std::size_t begin = 0; begin < pending_trials.size();
           begin += static_cast<std::size_t>(trial_mini_batch_size)) {
        const std::size_t end = std::min(
            pending_trials.size(),
            begin + static_cast<std::size_t>(trial_mini_batch_size));
        const std::size_t batch_size = end - begin;
        if (batch_size == 0) {
          continue;
        }

        for (std::size_t index = begin; index < end; ++index) {
          const auto mutation_begin = std::chrono::steady_clock::now();
          pending_trials[index].inst->swapMaster(pending_trials[index].target_master);
          trial_mutation_ms += elapsedMs(mutation_begin);
          trial_count += 1;
        }

      const auto query_begin = std::chrono::steady_clock::now();
      const py::dict trial_metrics = queryDiffGuidedBatchTimingMetrics();
      trial_query_ms += elapsedMs(query_begin);
      trial_query_count += 1;
      trial_batch_query_count += 1;
      full_metric_query_count += 1;

        const double trial_tns = pyDoubleOrDefault(trial_metrics, "tns", baseline_tns);
        const double trial_wns = pyDoubleOrDefault(trial_metrics, "wns", baseline_wns);
        const double trial_leakage = pyDoubleOrDefault(trial_metrics, "leakage", baseline_leakage);
        const double batch_delta_tns = trial_tns - baseline_tns;
        const double batch_delta_wns = trial_wns - baseline_wns;
        const double batch_leakage_delta = trial_leakage - baseline_leakage;
        const double share = 1.0 / static_cast<double>(batch_size);

        for (std::size_t index = begin; index < end; ++index) {
          auto& pending = pending_trials[index];
          py::dict result = pending.result;
          const double actual_delta_tns = batch_delta_tns * share;
          const double actual_delta_wns = batch_delta_wns * share;
          const double leakage_delta = batch_leakage_delta * share;
          const double actual_delta_obj = use_tns_power_tradeoff
                                              ? diffGuidedBatchActualDeltaObj(
                                                    actual_delta_tns,
                                                    leakage_delta,
                                                    leakage_weight)
                                              : actual_delta_tns;
          result["supported"] = true;
          result["baseline_metrics"] = baseline_metrics;
          result["trial_metrics"] = trial_metrics;
          result["old_master"] = pending.old_master->getName();
          result["new_master"] = pending.target_master->getName();
          writeDiffGuidedBatchFinalSizingMetadata(result, pending.action, pending.sizing_trial);
          result["actual_delta_tns"] = actual_delta_tns;
          result["actual_delta_wns"] = actual_delta_wns;
          result["actual_delta_obj"] = actual_delta_obj;
          result["actual_leakage_delta"] = leakage_delta;
          result["leakage_delta"] = leakage_delta;
          result["accept_mode"] = accept_mode;
          result["tns_power_leakage_weight"] = leakage_weight;
          result["evaluated_trial_count"] = 1;
          result["trial_mode"] = trial_mode;
          result["trial_query_count"] = 1;
          result["trial_query_count_per_trial"] = 1.0 / static_cast<double>(batch_size);
          const double accept_score = use_tns_power_tradeoff ? actual_delta_obj : actual_delta_tns;
          if (accept_score >= pyDoubleOrDefault(config, "min_action_tns_gain", 0.0)) {
            result["accepted"] = true;
            result["status"] = "accepted";
            accepted_count += 1;
          } else {
            result["status"] = "rejected";
            result["reject_reason"] = use_tns_power_tradeoff
                                          ? "tns_power_tradeoff_rejected"
                                          : "non_improving_tns";
            rejected_count += 1;
          }
          CompactBridgeActionResult typed_result = dgb::parseActionResult(result);
          typed_summary.evaluator_result_count += 1;
          if (typed_result.accepted) {
            typed_summary.accepted_delta_tns_sum += typed_result.actual_delta_tns;
            typed_summary.accepted_action_results.push_back(typed_result);
          }
          results.append(result);
        }

        for (std::size_t index = begin; index < end; ++index) {
          const auto rollback_begin = std::chrono::steady_clock::now();
          pending_trials[index].inst->swapMaster(pending_trials[index].old_master);
          trial_mutation_ms += elapsedMs(rollback_begin);
        }
      }
    } else {
    for (const auto& action : compact_actions) {
      const ActionKind kind = action.kind;
      py::dict result;
      const int64_t action_index = action.action_index;
      result["action_id"] = action_index;
      result["action_index"] = action_index;
      result["action_kind"] = diffGuidedBatchActionKindName(kind);
      result["supported"] = false;
      result["accepted"] = false;
      result["status"] = "rejected";
      result["reject_reason"] = "";
      result["actual_delta_tns"] = 0.0;
      result["actual_delta_wns"] = 0.0;
      result["prescreen_predicted_delta_obj"] =
          vectorValueOrDefault<double>(prescreen_result.prescreen_delta_objs, typed_summary.evaluator_result_count, 0.0);
      result["prescreen_predicted_improvement"] =
          vectorValueOrDefault<double>(prescreen_result.predicted_improvements, typed_summary.evaluator_result_count, 0.0);

      if (kind != ActionKind::kSizing) {
        result["status"] = "unsupported";
        result["reject_reason"] = diffGuidedBatchActionKindName(kind) + " is reserved and disabled in v1";
        result["affected_net_id"] = action.affected_net_id;
        result["driver_pin_id"] = action.driver_pin_id;
        result["load_pin_id"] = action.load_pin_id;
        result["buffer_master_id"] = action.buffer_master_id;
        result["buffer_master_name"] = action.buffer_master_name;
        result["candidate_location_x"] = action.candidate_location_x;
        result["candidate_location_y"] = action.candidate_location_y;
        unsupported_count += 1;
        typed_summary.evaluator_result_count += 1;
        results.append(result);
        continue;
      }

      const int inst_id = action.inst_id;
      result["inst_id"] = inst_id;
      result["current_size_idx"] = action.current_size_idx;
      result["target_size_idx"] = action.target_size_idx;
      result["old_timing_coordinate"] = action.old_timing_coordinate;
      result["new_timing_coordinate"] = action.new_timing_coordinate;
      auto* inst = nodeInst(inst_id);
      if (inst == nullptr || inst->isFixed()) {
        result["reject_reason"] = "invalid_or_fixed_instance";
        rejected_count += 1;
        typed_summary.evaluator_result_count += 1;
        results.append(result);
        continue;
      }

      odb::dbMaster* old_master = inst->getMaster();
      if (old_master == nullptr) {
        result["reject_reason"] = "missing_current_master";
        rejected_count += 1;
        typed_summary.evaluator_result_count += 1;
        results.append(result);
        continue;
      }
      const auto sizing_trials = buildDiffGuidedBatchSizingTrials(action, old_master, config);
      if (sizing_trials.empty()) {
        result["reject_reason"] = "missing_target_master";
        rejected_count += 1;
        typed_summary.evaluator_result_count += 1;
        results.append(result);
        continue;
      }

      bool has_valid_trial = false;
      std::string best_master_name;
      int best_trial_step = 0;
      double best_delta_tns = -std::numeric_limits<double>::infinity();
      double best_delta_wns = 0.0;
      double best_leakage_delta = 0.0;
      double best_actual_delta_obj = -std::numeric_limits<double>::infinity();
      py::dict best_trial_metrics;
      CompactBridgeSizingTrial best_sizing_trial;

      for (const auto& trial : sizing_trials) {
        const std::string& trial_master_name = trial.master_name;
        odb::dbMaster* target_master = db->findMaster(trial_master_name.c_str());
        if (target_master == nullptr || target_master == old_master) {
          continue;
        }
        const auto mutation_begin = std::chrono::steady_clock::now();
        inst->swapMaster(target_master);
        trial_mutation_ms += elapsedMs(mutation_begin);
        full_sta_trial_mutation_count += 1;

        const auto query_begin = std::chrono::steady_clock::now();
        const py::dict trial_metrics = queryDiffGuidedBatchTimingMetrics();
        trial_query_ms += elapsedMs(query_begin);
        trial_count += 1;
        trial_query_count += 1;
        full_metric_query_count += 1;

        const double trial_tns = pyDoubleOrDefault(trial_metrics, "tns", baseline_tns);
        const double trial_wns = pyDoubleOrDefault(trial_metrics, "wns", baseline_wns);
        const double trial_leakage = pyDoubleOrDefault(trial_metrics, "leakage", baseline_leakage);
        const double actual_delta_tns = trial_tns - baseline_tns;
        const double actual_delta_wns = trial_wns - baseline_wns;
        const double leakage_delta = trial_leakage - baseline_leakage;
        const double actual_delta_obj = use_tns_power_tradeoff
                                            ? diffGuidedBatchActualDeltaObj(
                                                  actual_delta_tns,
                                                  leakage_delta,
                                                  leakage_weight)
                                            : actual_delta_tns;
        if (!has_valid_trial || actual_delta_obj > best_actual_delta_obj) {
          has_valid_trial = true;
          best_master_name = target_master->getName();
          best_trial_step = trial.step;
          best_sizing_trial = trial;
          best_delta_tns = actual_delta_tns;
          best_delta_wns = actual_delta_wns;
          best_leakage_delta = leakage_delta;
          best_actual_delta_obj = actual_delta_obj;
          best_trial_metrics = trial_metrics;
        }

        const auto rollback_begin = std::chrono::steady_clock::now();
        inst->swapMaster(old_master);
        trial_mutation_ms += elapsedMs(rollback_begin);
        full_sta_trial_mutation_count += 1;
      }

      if (!has_valid_trial) {
        result["status"] = "unchanged";
        result["reject_reason"] = "no_valid_trial_master";
        rejected_count += 1;
        typed_summary.evaluator_result_count += 1;
        results.append(result);
        continue;
      }

      result["supported"] = true;
      result["baseline_metrics"] = baseline_metrics;
      result["trial_metrics"] = best_trial_metrics;
      result["old_master"] = old_master->getName();
      result["new_master"] = best_master_name;
      writeDiffGuidedBatchFinalSizingMetadata(result, action, best_sizing_trial);
      result["actual_delta_tns"] = best_delta_tns;
      result["actual_delta_wns"] = best_delta_wns;
      result["actual_delta_obj"] = best_actual_delta_obj;
      result["actual_leakage_delta"] = best_leakage_delta;
      result["leakage_delta"] = best_leakage_delta;
      result["accept_mode"] = accept_mode;
      result["tns_power_leakage_weight"] = leakage_weight;
      result["best_trial_step"] = best_trial_step;
      result["evaluated_trial_count"] = static_cast<int>(sizing_trials.size());
      const double accept_score = use_tns_power_tradeoff ? best_actual_delta_obj : best_delta_tns;
      if (accept_score >= pyDoubleOrDefault(config, "min_action_tns_gain", 0.0)) {
        result["accepted"] = true;
        result["status"] = "accepted";
        accepted_count += 1;
      } else {
        result["status"] = "rejected";
        result["reject_reason"] = use_tns_power_tradeoff
                                      ? "tns_power_tradeoff_rejected"
                                      : "non_improving_tns";
        rejected_count += 1;
      }
      CompactBridgeActionResult typed_result = dgb::parseActionResult(result);
      typed_summary.evaluator_result_count += 1;
      if (typed_result.accepted) {
        typed_summary.accepted_delta_tns_sum += typed_result.actual_delta_tns;
        typed_summary.accepted_action_results.push_back(typed_result);
      }
      results.append(result);
    }
    }

    py::dict final_metrics;
    if (trial_mode != "local_residual" && trial_mode != "local_candidate_delta") {
      const auto final_query_begin = std::chrono::steady_clock::now();
      final_metrics = queryDiffGuidedBatchTimingMetrics();
      const double final_query_ms = elapsedMs(final_query_begin);
      trial_query_ms += final_query_ms;
      snapshot_query_ms += final_query_ms;
      trial_query_count += 1;
      snapshot_query_count += 1;
      full_metric_query_count += 1;
      evaluator_summary["metric_snapshot_mode"] = "opensta_full";
    } else {
      final_metrics = baseline_metrics;
      trial_query_count = 0;
      trial_query_ms = 0.0;
    }
    evaluator_summary["results"] = results;
    evaluator_summary["accepted_action_count"] = accepted_count;
    evaluator_summary["rejected_action_count"] = rejected_count;
    evaluator_summary["unsupported_action_count"] = unsupported_count;
    evaluator_summary["trial_count"] = trial_count;
    evaluator_summary["trial_query_count"] = trial_query_count;
    evaluator_summary["trial_query_count_per_trial"] =
        trial_count > 0 ? static_cast<double>(trial_query_count) / static_cast<double>(trial_count) : 0.0;
    evaluator_summary["trial_batch_query_count"] = trial_batch_query_count;
    evaluator_summary["trial_fallback_count"] = trial_fallback_count;
    evaluator_summary["full_metric_query_count"] = full_metric_query_count;
    evaluator_summary["full_sta_trial_mutation_count"] = full_sta_trial_mutation_count;
    evaluator_summary["local_context_query_count"] = local_context_query_count;
    evaluator_summary["local_dcalc_query_count"] = local_dcalc_query_count;
    evaluator_summary["local_cap_query_count"] = local_cap_query_count;
    evaluator_summary["local_leakage_query_count"] = local_leakage_query_count;
    evaluator_summary["batch_verify_query_count"] = batch_verify_query_count;
    evaluator_summary["trial_mutation_ms"] = trial_mutation_ms;
    evaluator_summary["trial_query_ms"] = trial_query_ms;
    evaluator_summary["snapshot_query_count"] = snapshot_query_count;
    evaluator_summary["snapshot_query_ms"] = snapshot_query_ms;
    evaluator_summary["baseline_metrics"] = baseline_metrics;
    evaluator_summary["final_metrics_after_rollback"] = final_metrics;
    evaluator_summary["typed_result_count"] = typed_summary.evaluator_result_count;
    typed_summary.summary = evaluator_summary;
    return typed_summary;
  }

  py::dict evaluateDiffGuidedBatchActionProposals(
      const std::vector<CompactBridgeActionProposal>& compact_actions,
      const py::dict& config = py::dict())
  {
    return evaluateDiffGuidedBatchActionProposalsTyped(compact_actions, config).summary;
  }

  py::dict evaluateDiffGuidedBatchActions(const std::vector<py::dict>& actions,
                                          const py::dict& config = py::dict())
  {
    const std::vector<CompactBridgeActionProposal> compact_actions =
        dgb::parseActionProposals(actions);
    return evaluateDiffGuidedBatchActionProposals(compact_actions, config);
  }

  py::dict applyDiffGuidedBatchTransactionResults(
      const std::vector<CompactBridgeActionResult>& compact_results,
      const py::dict& config = py::dict())
  {
    py::dict summary;
    summary["artifact"] = "diff_guided_batch_opensta_transaction";
    summary["status"] = "ok";
    summary["confirmed"] = false;
    summary["rolled_back"] = false;
    summary["transaction_mutation_mode"] = "single_mutator";
    summary["accepted_action_count"] = 0;
    summary["unsupported_action_count"] = 0;
    summary["apply_ms"] = 0.0;
    summary["batch_verify_ms"] = 0.0;
    summary["rollback_ms"] = 0.0;
    summary["batch_verify_query_count"] = 0;
    summary["rollback_verify_query_count"] = 0;
    summary["transaction_input_format"] = "compact_action_results";
    summary["compact_action_result_count"] = static_cast<int>(compact_results.size());
    py::list transaction_log;

    auto* db = block_ == nullptr ? nullptr : block_->getDataBase();
    if (db == nullptr) {
      throw std::runtime_error("OpenROAD bridge has no dbDatabase for diff-guided batch transaction");
    }

    struct AppliedSizingAction
    {
      int64_t action_index;
      int inst_id;
      odb::dbInst* inst;
      odb::dbMaster* old_master;
      odb::dbMaster* new_master;
      CompactBridgeActionResult action_result;
    };
    std::vector<AppliedSizingAction> applied_actions;
    ord::Timing transaction_timing(design_.get());
    auto* transaction_sta = transaction_timing.getSta();
    auto* transaction_network =
        transaction_sta == nullptr ? nullptr : transaction_sta->getDbNetwork();
    const py::dict baseline_metrics = queryDiffGuidedBatchTimingMetrics();
    const double baseline_tns = pyDoubleOrDefault(baseline_metrics, "tns", 0.0);
    summary["transaction_apply_primitive"] = "opensta_replace_cell";

    for (const auto& action_result : compact_results) {
      if (!action_result.accepted) {
        continue;
      }
      const ActionKind kind = action_result.kind;
      if (kind != ActionKind::kSizing) {
        summary["unsupported_action_count"] = py::cast<int>(summary["unsupported_action_count"]) + 1;
        continue;
      }

      const int inst_id = action_result.inst_id;
      auto* inst = nodeInst(inst_id);
      const std::string target_master_name =
          action_result.new_master.empty() ? action_result.target_master : action_result.new_master;
      odb::dbMaster* target_master = target_master_name.empty() ? nullptr : db->findMaster(target_master_name.c_str());
      if (inst == nullptr || target_master == nullptr || inst->isFixed()) {
        continue;
      }
      odb::dbMaster* old_master = inst->getMaster();
      if (old_master == target_master) {
        continue;
      }
      const auto apply_begin = std::chrono::steady_clock::now();
      const DiffGuidedBatchStaAwareReplaceCellResult replace_result =
          diffGuidedBatchStaAwareReplaceCell(
              transaction_sta, transaction_network, inst, target_master);
      summary["apply_ms"] = py::cast<double>(summary["apply_ms"]) + elapsedMs(apply_begin);
      if (!replace_result.replaced) {
        py::dict row;
        row["action_kind"] = "sizing";
        row["action_index"] = action_result.action_index;
        row["inst_id"] = inst_id;
        row["old_master"] = old_master == nullptr ? std::string() : old_master->getName();
        row["new_master"] = target_master->getName();
        row["status"] = "replace_failed";
        row["transaction_apply_primitive"] = replace_result.transaction_apply_primitive;
        row["replace_fallback_reason"] = replace_result.fallback_reason;
        transaction_log.append(row);
        continue;
      }
      applied_actions.push_back({
          action_result.action_index,
          inst_id,
          inst,
          old_master,
          target_master,
          action_result,
      });

      py::dict row;
      row["action_kind"] = "sizing";
      row["action_index"] = applied_actions.back().action_index;
      row["inst_id"] = inst_id;
      row["old_master"] = old_master == nullptr ? std::string() : old_master->getName();
      row["new_master"] = target_master->getName();
      row["old_master_id"] = action_result.old_master_id;
      row["new_master_id"] = action_result.new_master_id;
      row["old_size_idx"] = action_result.current_size_idx;
      row["new_size_idx"] = action_result.target_size_idx;
      row["old_timing_coordinate"] = action_result.old_timing_coordinate;
      row["new_timing_coordinate"] = action_result.new_timing_coordinate;
      row["actual_delta_tns"] = action_result.actual_delta_tns;
      row["actual_delta_obj"] = action_result.actual_delta_obj;
      row["status"] = "applied";
      row["transaction_apply_primitive"] = replace_result.transaction_apply_primitive;
      transaction_log.append(row);
    }

    summary["accepted_action_count"] = static_cast<int>(applied_actions.size());
    if (applied_actions.empty()) {
      py::list failed_reasons;
      failed_reasons.append("no_applied_actions");
      summary["status"] = "no_effect_rejected";
      summary["reject_reason"] = "no_applied_actions";
      summary["guardrail_failed_reasons"] = failed_reasons;
      summary["transaction_log"] = transaction_log;
      summary["baseline_metrics"] = baseline_metrics;
      summary["verified_metrics"] = baseline_metrics;
      summary["pending_metrics_before_rollback"] = baseline_metrics;
      summary["final_metrics_after_rollback"] = baseline_metrics;
      summary["python_sync_required"] = false;
      return summary;
    }

    const bool single_action_audit_enabled = diffGuidedBatchSingleActionAudit(config);
    const int single_action_audit_limit = diffGuidedBatchSingleActionAuditLimit(config);
    summary["single_action_audit_enabled"] = single_action_audit_enabled;
    summary["single_action_audit_limit"] = single_action_audit_limit;
    if (single_action_audit_enabled && single_action_audit_limit > 0) {
      const auto audit_begin = std::chrono::steady_clock::now();
      py::list single_action_audit_results;
      py::dict single_action_audit_summary;
      int audit_count = 0;
      int positive_tns_count = 0;
      int zero_tns_count = 0;
      int negative_tns_count = 0;
      int positive_global_obj_count = 0;
      double sum_tns_delta = 0.0;
      double sum_wns_delta = 0.0;
      double sum_slew_vio_delta = 0.0;
      double sum_cap_vio_delta = 0.0;
      double sum_leakage_delta = 0.0;
      const double baseline_wns = pyDoubleOrDefault(baseline_metrics, "wns", 0.0);
      const double baseline_slew_vio = pyDoubleOrDefault(baseline_metrics, "slew_vio", 0.0);
      const double baseline_cap_vio = pyDoubleOrDefault(baseline_metrics, "cap_vio", 0.0);
      const double baseline_leakage = pyDoubleOrDefault(baseline_metrics, "leakage", 0.0);
      const double global_slew_weight =
          diffGuidedBatchEffectiveLocalDeltaSlewPenaltyWeight(config).value;
      const double global_cap_weight =
          diffGuidedBatchEffectiveLocalDeltaCapPenaltyWeight(config).value;
      const double leakage_weight = diffGuidedBatchTnsPowerLeakageWeight(config);

      for (auto it = applied_actions.rbegin(); it != applied_actions.rend(); ++it) {
        if (it->inst != nullptr && it->old_master != nullptr) {
          (void) diffGuidedBatchStaAwareReplaceCell(
              transaction_sta, transaction_network, it->inst, it->old_master);
        }
      }

      for (const AppliedSizingAction& applied : applied_actions) {
        if (audit_count >= single_action_audit_limit) {
          break;
        }
        if (applied.inst == nullptr || applied.new_master == nullptr || applied.old_master == nullptr) {
          continue;
        }
        const DiffGuidedBatchStaAwareReplaceCellResult audit_apply =
            diffGuidedBatchStaAwareReplaceCell(
                transaction_sta, transaction_network, applied.inst, applied.new_master);
        if (!audit_apply.replaced) {
          continue;
        }
        py::dict single_metrics = queryDiffGuidedBatchTimingMetrics();
        const double single_tns_delta =
            pyDoubleOrDefault(single_metrics, "tns", baseline_tns) - baseline_tns;
        const double single_wns_delta =
            pyDoubleOrDefault(single_metrics, "wns", baseline_wns) - baseline_wns;
        const double single_slew_vio_delta =
            pyDoubleOrDefault(single_metrics, "slew_vio", baseline_slew_vio) - baseline_slew_vio;
        const double single_cap_vio_delta =
            pyDoubleOrDefault(single_metrics, "cap_vio", baseline_cap_vio) - baseline_cap_vio;
        const double single_leakage_delta =
            pyDoubleOrDefault(single_metrics, "leakage", baseline_leakage) - baseline_leakage;
        const double single_action_global_obj_delta =
            diffGuidedBatchMetricDeltaObj(single_tns_delta,
                                          single_slew_vio_delta,
                                          single_cap_vio_delta,
                                          single_leakage_delta,
                                          global_slew_weight,
                                          global_cap_weight,
                                          leakage_weight);
        py::dict row;
        row["action_index"] = applied.action_index;
        row["inst_id"] = applied.inst_id;
        row["old_master"] = applied.old_master->getName();
        row["new_master"] = applied.new_master->getName();
        row["old_size_idx"] = applied.action_result.current_size_idx;
        row["new_size_idx"] = applied.action_result.target_size_idx;
        row["best_trial_step"] = applied.action_result.best_trial_step;
        row["actual_delta_tns_estimated"] = applied.action_result.actual_delta_tns;
        row["actual_delta_slew_cap_obj_estimated"] = applied.action_result.actual_delta_obj;
        row["predicted_net_slack_delta"] =
            applied.action_result.local_predicted_net_slack_delta;
        row["single_action_tns_delta"] = single_tns_delta;
        row["single_action_wns_delta"] = single_wns_delta;
        row["single_action_slew_vio_delta"] = single_slew_vio_delta;
        row["single_action_cap_vio_delta"] = single_cap_vio_delta;
        row["single_action_leakage_delta"] = single_leakage_delta;
        row["single_action_global_obj_delta"] = single_action_global_obj_delta;
        row["prediction_error_obj_delta"] =
            single_action_global_obj_delta - applied.action_result.local_predicted_net_slack_delta;
        row["prediction_sign_match"] =
            (single_action_global_obj_delta >= 0.0)
            == (applied.action_result.local_predicted_net_slack_delta >= 0.0);
        row["prediction_source"] =
            applied.action_result.local_predicted_slack_delta_source;
        row["single_action_metrics"] = single_metrics;
        single_action_audit_results.append(row);
        audit_count += 1;
        sum_tns_delta += single_tns_delta;
        sum_wns_delta += single_wns_delta;
        sum_slew_vio_delta += single_slew_vio_delta;
        sum_cap_vio_delta += single_cap_vio_delta;
        sum_leakage_delta += single_leakage_delta;
        if (single_tns_delta > 0.0) {
          positive_tns_count += 1;
        } else if (single_tns_delta < 0.0) {
          negative_tns_count += 1;
        } else {
          zero_tns_count += 1;
        }
        if (single_action_global_obj_delta > 0.0) {
          positive_global_obj_count += 1;
        }
        (void) diffGuidedBatchStaAwareReplaceCell(
            transaction_sta, transaction_network, applied.inst, applied.old_master);
      }

      single_action_audit_summary["single_action_audit_enabled"] = true;
      single_action_audit_summary["single_action_audit_sample_count"] = audit_count;
      single_action_audit_summary["single_action_positive_tns_delta_count"] =
          positive_tns_count;
      single_action_audit_summary["single_action_zero_tns_delta_count"] = zero_tns_count;
      single_action_audit_summary["single_action_negative_tns_delta_count"] =
          negative_tns_count;
      single_action_audit_summary["single_action_positive_global_obj_delta_count"] =
          positive_global_obj_count;
      single_action_audit_summary["sum_single_action_tns_delta"] = sum_tns_delta;
      single_action_audit_summary["mean_single_action_tns_delta"] =
          audit_count > 0 ? sum_tns_delta / static_cast<double>(audit_count) : 0.0;
      single_action_audit_summary["sum_single_action_wns_delta"] = sum_wns_delta;
      single_action_audit_summary["sum_single_action_slew_vio_delta"] =
          sum_slew_vio_delta;
      single_action_audit_summary["sum_single_action_cap_vio_delta"] = sum_cap_vio_delta;
      single_action_audit_summary["sum_single_action_leakage_delta"] = sum_leakage_delta;
      single_action_audit_summary["single_action_audit_ms"] = elapsedMs(audit_begin);
      summary["single_action_audit_results"] = single_action_audit_results;
      summary["single_action_audit_summary"] = single_action_audit_summary;

      for (const AppliedSizingAction& applied : applied_actions) {
        if (applied.inst != nullptr && applied.new_master != nullptr) {
          (void) diffGuidedBatchStaAwareReplaceCell(
              transaction_sta, transaction_network, applied.inst, applied.new_master);
        }
      }
    }

    const auto verify_begin = std::chrono::steady_clock::now();
    py::dict verified_metrics = queryDiffGuidedBatchTimingMetrics();
    summary["pending_metrics_before_rollback"] = verified_metrics;
    summary["batch_verify_ms"] = elapsedMs(verify_begin);
    summary["batch_verify_query_count"] = 1;
    py::dict guardrail_summary = checkDiffGuidedBatchGuardrails(
        baseline_metrics,
        verified_metrics,
        config);
    const double batch_tns_delta = pyDoubleOrDefault(guardrail_summary, "tns_delta", 0.0);
    if (summary.contains("single_action_audit_summary")) {
      py::dict audit_summary = summary["single_action_audit_summary"];
      audit_summary["batch_tns_delta"] = batch_tns_delta;
      audit_summary["batch_vs_single_tns_delta_gap"] =
          batch_tns_delta - pyDoubleOrDefault(audit_summary, "sum_single_action_tns_delta", 0.0);
      audit_summary["batch_objective_delta"] =
          pyDoubleOrDefault(guardrail_summary, "batch_objective_delta", batch_tns_delta);
      summary["single_action_audit_summary"] = audit_summary;
    }
    const double batch_objective_delta =
        pyDoubleOrDefault(guardrail_summary, "batch_objective_delta", batch_tns_delta);
    const double tns_delta = batch_tns_delta;
    if (tns_delta <= 0.0) {
      py::list failed_reasons = guardrail_summary["guardrail_failed_reasons"];
      failed_reasons.append("zero_batch_tns_gain");
      guardrail_summary["guardrail_failed_reasons"] = failed_reasons;
      guardrail_summary["passed"] = false;
    }
    if (batch_objective_delta <= 0.0) {
      py::list failed_reasons = guardrail_summary["guardrail_failed_reasons"];
      failed_reasons.append("zero_or_negative_batch_objective_gain");
      guardrail_summary["guardrail_failed_reasons"] = failed_reasons;
      guardrail_summary["passed"] = false;
    }
    if (guardrail_summary["passed"].cast<bool>()) {
      summary["confirmed"] = true;
      summary["status"] = "confirmed";
    } else {
      const auto rollback_begin = std::chrono::steady_clock::now();
      for (auto it = applied_actions.rbegin(); it != applied_actions.rend(); ++it) {
        if (it->inst != nullptr && it->old_master != nullptr) {
          (void) diffGuidedBatchStaAwareReplaceCell(
              transaction_sta, transaction_network, it->inst, it->old_master);
        }
      }
      summary["rollback_ms"] = elapsedMs(rollback_begin);
      summary["rolled_back"] = true;
      summary["status"] = "rolled_back";
      std::vector<std::string> failed_reasons;
      for (const auto reason : guardrail_summary["guardrail_failed_reasons"]) {
        failed_reasons.push_back(reason.cast<std::string>());
      }
      std::string reject_reason;
      for (std::size_t idx = 0; idx < failed_reasons.size(); ++idx) {
        if (idx > 0) {
          reject_reason += ",";
        }
        reject_reason += failed_reasons[idx];
      }
      summary["reject_reason"] = reject_reason;
      verified_metrics = queryDiffGuidedBatchTimingMetrics();
      summary["rollback_verify_query_count"] = 1;
      summary["final_metrics_after_rollback"] = verified_metrics;
    }

    summary["baseline_metrics"] = baseline_metrics;
    summary["verified_metrics"] = verified_metrics;
    summary["guardrail_summary"] = guardrail_summary;
    summary["guardrail_failed_reasons"] = guardrail_summary["guardrail_failed_reasons"];
    summary["transaction_log"] = transaction_log;
    summary["python_sync_required"] = py::cast<bool>(summary["confirmed"]);
    return summary;
  }

  py::dict applyDiffGuidedBatchTransaction(const std::vector<py::dict>& action_results,
                                           const py::dict& config = py::dict())
  {
    const std::vector<CompactBridgeActionResult> compact_results =
        dgb::parseActionResults(action_results);
    return applyDiffGuidedBatchTransactionResults(compact_results, config);
  }

  py::dict runDiffGuidedBatchLoop(const std::vector<py::dict>& seed_queue,
                                  const py::dict& conflict_precompute = py::dict(),
                                  const py::dict& config = py::dict())
  {
    const std::vector<CompactBridgeActionProposal> typed_proposal_pool =
        dgb::parseActionProposals(seed_queue);
    return runDiffGuidedBatchLoopTyped(
        typed_proposal_pool,
        conflict_precompute,
        config,
        "typed_proposal_pool",
        static_cast<int>(seed_queue.size()));
  }

  py::dict runDiffGuidedBatchLoopCompact(const py::dict& action_buffers,
                                         const py::dict& conflict_precompute = py::dict(),
                                         const py::dict& config = py::dict())
  {
    const std::vector<CompactBridgeActionProposal> typed_proposal_pool =
        dgb::parseActionProposalsFromBuffers(action_buffers);
    return runDiffGuidedBatchLoopTyped(
        typed_proposal_pool,
        conflict_precompute,
        config,
        "compact_action_proposal_buffers",
        static_cast<int>(typed_proposal_pool.size()));
  }

  py::dict runDiffGuidedBatchLoopTyped(
      const std::vector<CompactBridgeActionProposal>& typed_proposal_pool,
      const py::dict& conflict_precompute,
      const py::dict& config,
      const std::string& loop_input_format,
      int proposal_seed_count)
  {
    py::dict summary;
    summary["artifact"] = "diff_guided_batch_loop_summary";
    summary["artifact_version"] = 1;
    summary["mode"] = "diff_guided_batch";
    summary["status"] = "completed";
    summary["opensta_status"] = has_timing_inputs_ ? "integrated" : "no_timing_inputs";
    summary["cxx_batch_loop_status"] = "opensta_bridge_batch_loop";
    summary["batch_loop_core"] = "cxx_opensta_bridge_batch_level";
    summary["python_inner_loop"] = false;
    summary["selector_backend"] = "cxx_compact_buffer";
    summary["selector_input_format"] = "compact_buffers";
    summary["evaluator_backend"] = "cxx_opensta_in_process";
    summary["transaction_backend"] = "cxx_opensta_in_process";
    summary["selector_parallel_strategy"] = "serial_greedy";
    summary["proposal_seed_count"] = proposal_seed_count;
    summary["selected_batch_count"] = 0;
    summary["accepted_action_count"] = 0;
    summary["rejected_action_count"] = 0;
    summary["confirmed_batch_count"] = 0;
    summary["confirmed_action_count"] = 0;
    summary["rolled_back_batch_count"] = 0;
    summary["rolled_back"] = false;
    summary["python_sync_required"] = false;

    py::dict dynamic_conflict_signature = queryDiffGuidedBatchDynamicConflictSignature(config);
    summary["dynamic_conflict_signature_summary"] = dynamic_conflict_signature;
    summary["dynamic_top_path_signature_source"] = pyStringOrDefault(
        dynamic_conflict_signature, "dynamic_top_path_signature_source", "disabled");

    const int max_loop_batches = std::max(1, pyIntOrDefault(
        config,
        "max_loop_batches",
        pyIntOrDefault(config, "diff_guided_batch_max_loop_batches", 1)));
    const int max_batch_size = std::max(1, pyIntOrDefault(
        config,
        "max_batch_size",
        pyIntOrDefault(config, "diff_guided_batch_max_batch_size", 64)));
    const double min_recent_gain = pyDoubleOrDefault(
        config,
        "min_recent_tns_gain",
        pyDoubleOrDefault(config, "diff_guided_batch_min_recent_tns_gain", 0.0));
    const int max_consecutive_rollback_batches = std::max(
        0,
        pyIntOrDefault(
            config,
            "max_consecutive_rollback_batches",
            pyIntOrDefault(config, "diff_guided_batch_max_consecutive_rollback_batches", 3)));
    const int selector_worker_count = diffGuidedBatchSelectorWorkerCount(config);
    const int trial_worker_count = diffGuidedBatchTrialWorkerCount(config);
    const std::string trial_mode = diffGuidedBatchTrialMode(config);
    const int trial_mini_batch_size = diffGuidedBatchTrialMiniBatchSize(config);
    const std::string local_residual_feedback_policy =
        diffGuidedBatchLocalResidualFeedbackPolicy(config);
    const double local_residual_feedback_decay =
        diffGuidedBatchLocalResidualFeedbackDecay(config);
    const std::string selector_residual_budget_filter_mode = pyStringOrDefault(
        config,
        "selector_residual_budget_filter_mode",
        pyStringOrDefault(
            config,
            "diff_guided_batch_selector_residual_budget_filter_mode",
            "off"));
    const std::string selector_residual_budget_score_mode = pyStringOrDefault(
        config,
        "selector_residual_budget_score_mode",
        pyStringOrDefault(
            config,
            "diff_guided_batch_selector_residual_budget_score_mode",
            "off"));
    const DiffGuidedBatchConflictRuleMask conflict_rule_mask =
        dgb::conflictRuleMask(config);

    std::vector<int> remaining_action_indices =
        dgb::makeSequentialActionIndices(typed_proposal_pool.size());
    summary["loop_input_format"] = loop_input_format;
    summary["typed_proposal_pool_count"] = static_cast<int>(typed_proposal_pool.size());
    summary["requested_selector_worker_count"] = selector_worker_count;
    summary["requested_trial_worker_count"] = trial_worker_count;
    summary["selector_parallel_worker_count"] = selector_worker_count;
    summary["trial_worker_count"] = trial_worker_count;
    summary["trial_mode"] = trial_mode;
    summary["trial_mini_batch_size"] = trial_mini_batch_size;
    if (trial_mode == "local_residual" || trial_mode == "local_candidate_delta") {
      if (trial_mode == "local_candidate_delta") {
        summary["metric_snapshot_mode"] = "local_context_no_full_metric";
        summary["trial_parallel_mode"] =
            "local_candidate_delta_no_opensta_trial_query";
        summary["opensta_trial_mutation_mode"] = "none_per_candidate_mutation";
      } else {
        summary["metric_snapshot_mode"] = "none_local_residual";
        summary["trial_parallel_mode"] =
            "local_delay_slack_residual_no_opensta_trial_query";
        summary["opensta_trial_mutation_mode"] = "none_local_residual";
      }
    } else {
      summary["metric_snapshot_mode"] = "opensta_full";
      summary["trial_parallel_mode"] = "parallel_objective_delta_prescreen_serial_opensta_verify";
      summary["opensta_trial_mutation_mode"] = "serial_mutation";
    }
    summary["transaction_mutation_mode"] = "single_mutator";
    summary["enabled_same_instance_conflict"] = conflict_rule_mask.same_instance;
    summary["enabled_endpoint_conflict"] = conflict_rule_mask.same_endpoint;
    summary["enabled_top_path_conflict"] = conflict_rule_mask.same_top_path;
    summary["enabled_static_component_conflict"] = conflict_rule_mask.static_component;
    summary["max_consecutive_rollback_batches"] = max_consecutive_rollback_batches;
    summary["consecutive_rollback_batch_count"] = 0;
    summary["early_stop_reason"] = "";
    summary["skipped_remaining_loop_batches"] = 0;
    py::list selector_summaries;
    py::list evaluator_summaries;
    py::list transaction_decisions;
    py::list transaction_log;
    py::dict final_metrics;
    int loop_trial_count = 0;
    int loop_trial_query_count = 0;
    int loop_trial_batch_query_count = 0;
    int loop_trial_fallback_count = 0;
    int loop_local_residual_trial_count = 0;
    int loop_full_metric_query_count = 0;
    int loop_full_sta_trial_mutation_count = 0;
    int loop_local_context_query_count = 0;
    int loop_local_dcalc_query_count = 0;
    int loop_local_cap_query_count = 0;
    int loop_local_leakage_query_count = 0;
    int loop_batch_verify_query_count = 0;
    int loop_rollback_verify_query_count = 0;
    double loop_trial_query_ms = 0.0;
    double loop_trial_mutation_ms = 0.0;
    double loop_batch_verify_ms = 0.0;
    int dynamic_conflict_signature_refresh_count = 0;
    int consecutive_rollback_batch_count = 0;
    std::string early_stop_reason;
    int skipped_remaining_loop_batches = 0;
    double local_residual_feedback_scale = 1.0;
    int local_residual_feedback_update_count = 0;
    summary["local_residual_feedback_scale"] = local_residual_feedback_scale;
    summary["local_residual_feedback_update_count"] = local_residual_feedback_update_count;
    summary["local_residual_feedback_policy"] = local_residual_feedback_policy;
    summary["local_residual_feedback_decay"] = local_residual_feedback_decay;
    summary["selector_residual_budget_filter_mode"] =
        selector_residual_budget_filter_mode;
    summary["selector_residual_budget_score_mode"] =
        selector_residual_budget_score_mode;

    for (int loop_id = 0;
         loop_id < max_loop_batches && !remaining_action_indices.empty();
         ++loop_id) {
      const auto selector_begin = std::chrono::steady_clock::now();
      std::vector<int> selected_action_indices;
      std::vector<int> next_action_indices;
      py::list selected_action_ids;
      py::list remaining_action_ids;

      const CompactBridgeConflictPrecompute typed_conflict_precompute =
          dgb::parseConflictPrecompute(conflict_precompute, dynamic_conflict_signature);
      const std::vector<CompactBridgeAction> compact_actions =
          dgb::buildCompactActions(
              typed_proposal_pool,
              remaining_action_indices,
              typed_conflict_precompute);
      const CompactBridgeSelection compact_selection =
          dgb::selectCompact(
              compact_actions,
              max_batch_size,
              conflict_rule_mask,
              typed_conflict_precompute,
              selector_residual_budget_filter_mode,
              selector_residual_budget_score_mode);

      for (const int queue_index : compact_selection.selected_queue_indices) {
        if (queue_index >= 0 && queue_index < static_cast<int>(typed_proposal_pool.size())) {
          selected_action_indices.push_back(queue_index);
        }
      }
      for (const int queue_index : compact_selection.remaining_queue_indices) {
        if (queue_index >= 0 && queue_index < static_cast<int>(typed_proposal_pool.size())) {
          next_action_indices.push_back(queue_index);
        }
      }
      for (const int64_t action_id : compact_selection.selected_action_ids) {
        selected_action_ids.append(action_id);
      }
      for (const int64_t action_id : compact_selection.remaining_action_ids) {
        remaining_action_ids.append(action_id);
      }

      py::dict selector_summary;
      selector_summary["proposal_seed_count"] = static_cast<int>(remaining_action_indices.size());
      selector_summary["selected_batch_size"] = static_cast<int>(selected_action_indices.size());
      selector_summary["remaining_seed_count"] = static_cast<int>(next_action_indices.size());
      selector_summary["selected_action_ids"] = selected_action_ids;
      selector_summary["remaining_action_ids"] = remaining_action_ids;
      selector_summary["selector_queue_format"] = "typed_action_indices";
      selector_summary["conflict_reject_count"] = compact_selection.conflict_reject_count;
      selector_summary["component_conflict_count"] = compact_selection.conflict_reject_count;
      selector_summary["residual_budget_reject_count"] =
          compact_selection.residual_budget_reject_count;
      selector_summary["residual_budget_score_reorder_count"] =
          compact_selection.residual_budget_score_reorder_count;
      selector_summary["selector_residual_budget_filter_mode"] =
          selector_residual_budget_filter_mode;
      selector_summary["selector_residual_budget_score_mode"] =
          selector_residual_budget_score_mode;
      selector_summary["residual_budget_source"] = compact_selection.residual_budget_source;
      selector_summary["residual_budget_score_source"] =
          compact_selection.residual_budget_score_source;
      selector_summary["residual_snapshot_source"] =
          typed_conflict_precompute.local_residual_snapshot_source;
      selector_summary["requested_selector_worker_count"] = selector_worker_count;
      selector_summary["parallel_worker_count"] = selector_worker_count;
      selector_summary["parallel_strategy"] = "serial_greedy";
      selector_summary["selector_backend"] = "cxx_compact_buffer";
      selector_summary["selector_input_format"] = "compact_buffers";
      selector_summary["enabled_same_instance_conflict"] = conflict_rule_mask.same_instance;
      selector_summary["enabled_endpoint_conflict"] = conflict_rule_mask.same_endpoint;
      selector_summary["enabled_top_path_conflict"] = conflict_rule_mask.same_top_path;
      selector_summary["enabled_static_component_conflict"] = conflict_rule_mask.static_component;
      selector_summary["compact_action_count"] = static_cast<int>(compact_actions.size());
      selector_summary["dynamic_top_path_signature_source"] = pyStringOrDefault(
          dynamic_conflict_signature, "dynamic_top_path_signature_source", "disabled");
      selector_summary["dynamic_conflict_signature_refresh_index"] =
          dynamic_conflict_signature_refresh_count;
      selector_summary["dynamic_endpoint_signature_source"] = pyStringOrDefault(
          dynamic_conflict_signature, "dynamic_endpoint_signature_source", "disabled");
      selector_summary["dynamic_path_count"] = pyIntOrDefault(
          dynamic_conflict_signature, "dynamic_path_count", 0);
      selector_summary["dynamic_endpoint_count"] = pyIntOrDefault(
          dynamic_conflict_signature, "dynamic_endpoint_count", 0);
      selector_summary["dynamic_unique_inst_count"] = pyIntOrDefault(
          dynamic_conflict_signature, "dynamic_unique_inst_count", 0);
      selector_summary["endpoint_component_source"] =
          pyIntOrDefault(dynamic_conflict_signature, "dynamic_endpoint_count", 0) > 0
              ? "dynamic_opensta_findPathEnds_endpoint_with_static_fallback"
              : "static_conflict_precompute";
      selector_summary["top_path_component_source"] =
          pyIntOrDefault(dynamic_conflict_signature, "dynamic_unique_inst_count", 0) > 0
              ? "dynamic_opensta_findPathEnds_with_static_fallback"
              : "static_conflict_precompute";
      selector_summary["component_count"] = pyIntOrDefault(conflict_precompute, "component_count", 0);
      selector_summary["max_component_size"] = pyIntOrDefault(conflict_precompute, "max_component_size", 0);
      selector_summary["csr_edge_count"] = pyIntOrDefault(conflict_precompute, "csr_edge_count", 0);
      selector_summary["bitset_block_count"] = pyIntOrDefault(conflict_precompute, "bitset_block_count", 0);
      selector_summary["conflict_precompute_source"] = pyStringOrDefault(
          conflict_precompute, "conflict_precompute_source", pyStringOrDefault(conflict_precompute, "source", ""));
      selector_summary["conflict_precompute_version"] = pyIntOrDefault(
          conflict_precompute, "conflict_precompute_version", pyIntOrDefault(conflict_precompute, "version", 1));
      selector_summary["conflict_precompute_input_hash"] = pyStringOrDefault(
          conflict_precompute, "conflict_precompute_input_hash", pyStringOrDefault(conflict_precompute, "input_fingerprint", ""));
      selector_summary["cpp_selector_ms"] = elapsedMs(selector_begin);
      selector_summaries.append(selector_summary);

      if (selected_action_indices.empty()) {
        break;
      }
      summary["selected_batch_count"] = py::cast<int>(summary["selected_batch_count"]) + 1;

      const std::vector<CompactBridgeActionProposal> selected_proposals =
          dgb::collectActionProposals(
              typed_proposal_pool,
              selected_action_indices,
              typed_conflict_precompute);
      py::dict evaluator_config = copyDiffGuidedBatchConfig(config);
      evaluator_config["local_residual_budget_by_top_path_component"] =
          dynamic_conflict_signature["local_residual_budget_by_top_path_component"];
      evaluator_config["local_residual_budget_by_endpoint_component"] =
          dynamic_conflict_signature["local_residual_budget_by_endpoint_component"];
      evaluator_config["local_residual_snapshot_source"] =
          dynamic_conflict_signature["local_residual_snapshot_source"];
      evaluator_config["local_residual_feedback_scale"] = local_residual_feedback_scale;
      evaluator_config["local_residual_feedback_policy"] = local_residual_feedback_policy;
      evaluator_config["local_residual_feedback_decay"] = local_residual_feedback_decay;
      const CompactBridgeEvaluationSummary evaluator_typed_summary =
          evaluateDiffGuidedBatchActionProposalsTyped(selected_proposals, evaluator_config);
      const py::dict batch_evaluator_summary = evaluator_typed_summary.summary;
      evaluator_summaries.append(batch_evaluator_summary);
      loop_trial_count += pyIntOrDefault(batch_evaluator_summary, "trial_count", 0);
      loop_trial_query_count += pyIntOrDefault(batch_evaluator_summary, "trial_query_count", 0);
      loop_trial_batch_query_count += pyIntOrDefault(batch_evaluator_summary, "trial_batch_query_count", 0);
      loop_trial_fallback_count += pyIntOrDefault(batch_evaluator_summary, "trial_fallback_count", 0);
      loop_local_residual_trial_count += pyIntOrDefault(batch_evaluator_summary, "local_residual_trial_count", 0);
      loop_full_metric_query_count += pyIntOrDefault(batch_evaluator_summary, "full_metric_query_count", 0);
      loop_full_sta_trial_mutation_count += pyIntOrDefault(
          batch_evaluator_summary, "full_sta_trial_mutation_count", 0);
      loop_local_context_query_count += pyIntOrDefault(batch_evaluator_summary, "local_context_query_count", 0);
      loop_local_dcalc_query_count += pyIntOrDefault(batch_evaluator_summary, "local_dcalc_query_count", 0);
      loop_local_cap_query_count += pyIntOrDefault(batch_evaluator_summary, "local_cap_query_count", 0);
      loop_local_leakage_query_count += pyIntOrDefault(batch_evaluator_summary, "local_leakage_query_count", 0);
      loop_trial_query_ms += pyDoubleOrDefault(batch_evaluator_summary, "trial_query_ms", 0.0);
      loop_trial_mutation_ms += pyDoubleOrDefault(batch_evaluator_summary, "trial_mutation_ms", 0.0);
      summary["accepted_action_count"] =
          py::cast<int>(summary["accepted_action_count"]) +
          static_cast<int>(evaluator_typed_summary.accepted_action_results.size());
      summary["rejected_action_count"] =
          py::cast<int>(summary["rejected_action_count"]) +
          std::max(0,
                   evaluator_typed_summary.evaluator_result_count
                       - static_cast<int>(evaluator_typed_summary.accepted_action_results.size()));

      const py::dict transaction_decision =
          applyDiffGuidedBatchTransactionResults(
              evaluator_typed_summary.accepted_action_results,
              config);
      const double predicted_local_residual_gain =
          pyDoubleOrDefault(batch_evaluator_summary, "local_residual_accepted_gain_sum", 0.0);
      transaction_decision["predicted_local_residual_gain"] = predicted_local_residual_gain;
      transaction_decisions.append(transaction_decision);
      loop_batch_verify_ms += pyDoubleOrDefault(transaction_decision, "batch_verify_ms", 0.0);
      loop_batch_verify_query_count += pyIntOrDefault(transaction_decision, "batch_verify_query_count", 0);
      loop_rollback_verify_query_count += pyIntOrDefault(transaction_decision, "rollback_verify_query_count", 0);
      if (transaction_decision.contains("confirmed") && transaction_decision["confirmed"].cast<bool>()) {
        summary["confirmed_batch_count"] = py::cast<int>(summary["confirmed_batch_count"]) + 1;
        consecutive_rollback_batch_count = 0;
        if (transaction_decision.contains("transaction_log") && !transaction_decision["transaction_log"].is_none()) {
          for (const auto& item : transaction_decision["transaction_log"]) {
            transaction_log.append(item);
          }
        }
        if (transaction_decision.contains("verified_metrics") && !transaction_decision["verified_metrics"].is_none()) {
          final_metrics = py::cast<py::dict>(transaction_decision["verified_metrics"]);
        }
      }
      if (transaction_decision.contains("rolled_back") && transaction_decision["rolled_back"].cast<bool>()) {
        summary["rolled_back_batch_count"] = py::cast<int>(summary["rolled_back_batch_count"]) + 1;
        summary["rolled_back"] = true;
        consecutive_rollback_batch_count += 1;
        const py::dict guardrail_summary =
            transaction_decision.contains("guardrail_summary") && !transaction_decision["guardrail_summary"].is_none()
                ? py::cast<py::dict>(transaction_decision["guardrail_summary"])
                : py::dict();
        const double verified_tns_delta =
            pyDoubleOrDefault(guardrail_summary, "tns_delta", 0.0);
        if ((trial_mode == "local_residual" || trial_mode == "local_candidate_delta")
            && predicted_local_residual_gain > 0.0
            && verified_tns_delta <= 0.0) {
          if (local_residual_feedback_policy == "soft_decay") {
            local_residual_feedback_scale = std::max(0.0, local_residual_feedback_scale * local_residual_feedback_decay);
          }
          if (local_residual_feedback_policy == "hard_zero") {
            local_residual_feedback_scale = std::max(0.0, 0.0);
          }
          local_residual_feedback_update_count += 1;
        }
      }
      summary["consecutive_rollback_batch_count"] = consecutive_rollback_batch_count;
      summary["local_residual_feedback_scale"] = local_residual_feedback_scale;
      summary["local_residual_feedback_update_count"] = local_residual_feedback_update_count;
      summary["local_residual_feedback_policy"] = local_residual_feedback_policy;
      summary["local_residual_feedback_decay"] = local_residual_feedback_decay;

      dynamic_conflict_signature = refreshDiffGuidedBatchDynamicConflictSignature(
          config, loop_id, transaction_decision);
      dynamic_conflict_signature_refresh_count += 1;

      remaining_action_indices = std::move(next_action_indices);
      if ((trial_mode == "local_residual" || trial_mode == "local_candidate_delta")
          && local_residual_feedback_scale <= 0.0) {
        early_stop_reason = "local_residual_feedback_suppressed";
        summary["status"] = "early_stopped";
        summary["early_stop_reason"] = "local_residual_feedback_suppressed";
        skipped_remaining_loop_batches = std::max(0, max_loop_batches - loop_id - 1);
        summary["skipped_remaining_loop_batches"] = skipped_remaining_loop_batches;
        break;
      }
      if (max_consecutive_rollback_batches > 0
          && consecutive_rollback_batch_count >= max_consecutive_rollback_batches) {
        early_stop_reason = "consecutive_rollback_batches";
        summary["status"] = "early_stopped";
        summary["early_stop_reason"] = "consecutive_rollback_batches";
        skipped_remaining_loop_batches = std::max(0, max_loop_batches - loop_id - 1);
        summary["skipped_remaining_loop_batches"] = skipped_remaining_loop_batches;
        break;
      }
      if (min_recent_gain > 0.0) {
        if (evaluator_typed_summary.accepted_delta_tns_sum < min_recent_gain) {
          break;
        }
      }
    }

    summary["loop_count"] = py::len(selector_summaries);
    summary["selector_summaries"] = selector_summaries;
    summary["evaluator_summaries"] = evaluator_summaries;
    summary["transaction_decisions"] = transaction_decisions;
    summary["confirmed_action_transaction_log"] = transaction_log;
    summary["confirmed_action_count"] = py::len(transaction_log);
    summary["final_opensta_metrics"] = final_metrics;
    summary["dynamic_conflict_signature_refresh_count"] =
        dynamic_conflict_signature_refresh_count;
    summary["dynamic_conflict_signature_final_summary"] = dynamic_conflict_signature;
    summary["trial_count"] = loop_trial_count;
    summary["trial_query_count"] = loop_trial_query_count;
    summary["trial_query_count_per_trial"] =
        loop_trial_count > 0 ? static_cast<double>(loop_trial_query_count) / static_cast<double>(loop_trial_count) : 0.0;
    summary["trial_batch_query_count"] = loop_trial_batch_query_count;
    summary["trial_fallback_count"] = loop_trial_fallback_count;
    summary["local_residual_trial_count"] = loop_local_residual_trial_count;
    summary["full_metric_query_count"] = loop_full_metric_query_count;
    summary["full_sta_trial_mutation_count"] = loop_full_sta_trial_mutation_count;
    summary["local_context_query_count"] = loop_local_context_query_count;
    summary["local_dcalc_query_count"] = loop_local_dcalc_query_count;
    summary["local_cap_query_count"] = loop_local_cap_query_count;
    summary["local_leakage_query_count"] = loop_local_leakage_query_count;
    summary["batch_verify_query_count"] = loop_batch_verify_query_count;
    summary["rollback_verify_query_count"] = loop_rollback_verify_query_count;
    summary["trial_query_ms_total"] = loop_trial_query_ms;
    summary["trial_mutation_ms_total"] = loop_trial_mutation_ms;
    summary["batch_verify_ms_total"] = loop_batch_verify_ms;
    summary["early_stop_reason"] = early_stop_reason;
    summary["skipped_remaining_loop_batches"] = skipped_remaining_loop_batches;
    summary["max_consecutive_rollback_batches"] = max_consecutive_rollback_batches;
    summary["consecutive_rollback_batch_count"] = consecutive_rollback_batch_count;
    summary["local_residual_feedback_scale"] = local_residual_feedback_scale;
    summary["local_residual_feedback_update_count"] = local_residual_feedback_update_count;
    summary["local_residual_feedback_policy"] = local_residual_feedback_policy;
    summary["local_residual_feedback_decay"] = local_residual_feedback_decay;
    summary["python_sync_required"] = py::len(transaction_log) > 0;
    return summary;
  }

  void writeDef(const std::string& filename) const
  {
    design_->writeDef(filename);
  }

 private:
  void rebuildNodeInstIndex(bool invalidate_npath_snapshot = true)
  {
    node_insts_.clear();
    if (block_ == nullptr) {
      if (invalidate_npath_snapshot) {
        invalidateDiffGuidedBatchNPathSnapshot();
      }
      return;
    }
    for (auto* inst : block_->getInsts()) {
      if (!inst->isFixed()) {
        node_insts_.push_back(inst);
      }
    }
    for (auto* inst : block_->getInsts()) {
      if (inst->isFixed()) {
        node_insts_.push_back(inst);
      }
    }
    if (invalidate_npath_snapshot) {
      invalidateDiffGuidedBatchNPathSnapshot();
    }
  }

  OpenRoadRuntime& runtime_;
  utl::Logger* logger_{nullptr};
  std::unique_ptr<ord::Tech> tech_;
  std::unique_ptr<ord::Design> design_;
  odb::dbBlock* block_{nullptr};
  std::vector<odb::dbInst*> node_insts_;
  std::vector<std::string> vt_suffixes_;
  bool has_timing_inputs_{false};
  DiffGuidedBatchNPathSnapshot diff_guided_batch_npath_snapshot_;
  int diff_guided_batch_npath_snapshot_build_count_{0};
};

}  // namespace impl
}  // namespace placeio_openroad
}  // namespace dreamplace
