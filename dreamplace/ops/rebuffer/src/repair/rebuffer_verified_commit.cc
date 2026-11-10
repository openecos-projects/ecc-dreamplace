// SPDX-License-Identifier: BSD-3-Clause
// Native ownership of physical commit acceptance, rollback and bundle retries.
#include "rebuffer_verified_commit.hh"
#include "rebuffer_electrical_observation.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {

using Group = std::vector<py::dict>;
using Groups = std::vector<Group>;

// Match the fresh-TP restoration tolerance, independently of QoR margins.
constexpr double kTimingTolerancePs = 1.0e-6;

int integer(const py::dict& values, const char* key)
{
  return values.contains(key) ? py::cast<int>(values[key]) : 0;
}

std::string status(const py::dict& values)
{
  return values.contains("status") ? py::cast<std::string>(values["status"]) : "";
}

py::dict copy(const py::dict& value)
{
  return value.attr("copy")().cast<py::dict>();
}

// Either a shared physical source net or a shared native tree bundle makes
// actions inseparable. Transitive closure also preserves chains built over
// several accepted tree epochs. Close each dependency's interval in the
// accepted stream so grouping never moves a later epoch ahead of an earlier one.
Groups actionGroups(const py::list& actions)
{
  std::vector<int> parents(actions.size());
  std::iota(parents.begin(), parents.end(), 0);
  std::function<int(int)> root = [&](int i) {
    return parents[i] == i ? i : parents[i] = root(parents[i]);
  };
  std::map<std::pair<std::string, std::string>, int> owners;
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
    const auto action = actions[i].cast<py::dict>();
    std::string kind = "buffer_insert";
    if (action.contains("action_kind")) {
      kind = py::cast<std::string>(action["action_kind"]);
      if (kind != "buffer_insert" && kind != "resize_cell") {
        throw py::value_error("verified insertion requires buffer_insert or resize_cell actions");
      }
    }
    for (const char* key : {"net_name", "tree_bundle_id", "instance_name"}) {
      if (!action.contains(key) || action[key].is_none()) continue;
      const std::string value = py::cast<std::string>(action[key]);
      if (value.empty()) continue;
      const auto found = owners.emplace(std::make_pair(key, value), i);
      if (!found.second) parents[root(i)] = root(found.first->second);
    }
    if (kind == "resize_cell") {
      if (!action.contains("instance_name")) throw py::value_error("resize_cell action requires instance_name");
    } else if (!action.contains("net_name")) {
      throw py::value_error("buffer action requires net_name");
    }
  }
  std::map<int, int> last_action;
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
    last_action[root(i)] = i;
  }
  Groups groups;
  int end = -1;
  for (int i = 0; i < static_cast<int>(actions.size()); ++i) {
    if (i > end) groups.emplace_back();
    end = std::max(end, last_action.at(root(i)));
    groups.back().push_back(copy(actions[i].cast<py::dict>()));
  }
  return groups;
}

struct Observation {
  std::string state;
  py::dict timing;
};

class VerifiedCommit
{
 public:
  VerifiedCommit(const py::object& apply, const py::object& refresh,
                 const py::object& evaluate, const py::object& counts,
                 const py::object& digest, double wns_margin, double tns_margin)
      : apply_(apply), refresh_(refresh), evaluate_(evaluate), counts_(counts),
        digest_(digest), wns_margin_(wns_margin), tns_margin_(tns_margin)
  {
    result_["status"] = "skipped";
    result_["core"] = "cpp";
    for (const char* key : {"chunk_count", "kept_count", "reverted_count", "native_commit_count",
                           "commit_attempt_count", "commit_failed_count", "refresh_failed_count",
                           "timing_failed_count", "revert_failed_count"}) result_[key] = 0;
    result_["kept_actions"] = py::list();
    result_["chunks"] = py::list();
    result_["action_results"] = py::list();
  }

  py::dict run(const py::list& actions, const py::object& prepare, int chunk_size)
  {
    const auto groups = actionGroups(actions);
    result_["chunk_size"] = chunk_size;
    result_["action_digest"] = digest_(actions);
    if (actions.empty()) {
      result_["reason"] = "no_actions";
      return result_;
    }
    try {
      prepare();
    } catch (const std::exception& error) {
      result_["status"] = "unsupported_timing_schema";
      result_["failure_reason"] = error.what();
      return result_;
    }
    try {
      current_ = timing();
    } catch (const std::exception& error) {
      result_["status"] = "timing_failed";
      result_["failure_reason"] = error.what();
      return result_;
    }
    result_["timing_before"] = copy(current_);
    result_["status"] = "ok";
    timing_valid_ = true;
    Groups chunk;
    int count = 0;
    for (const auto& group : groups) {
      if (chunk_size > 0 && !chunk.empty() && count + static_cast<int>(group.size()) > chunk_size) {
        verify(chunk, 0);
        chunk.clear();
        count = 0;
        if (status(result_) != "ok") break;
      }
      chunk.push_back(group);
      count += group.size();
    }
    if (status(result_) == "ok" && !chunk.empty()) verify(chunk, 0);
    if (timing_valid_) result_["timing_after"] = copy(current_);
    result_["timing_valid"] = timing_valid_;
    result_["chunk_count"] = py::len(result_["chunks"]);
    result_["native_commit_count"] = integer(result_, "kept_count") + integer(result_, "reverted_count");
    result_["native_counts_after"] = timing_valid_ ? counts_() : py::none();
    return result_;
  }

 private:
  py::dict timing()
  {
    const auto values = evaluate_().cast<py::dict>();
    for (const char* field : {"wns_ps", "tns_ps"}) {
      if (!values.contains(field) || values[field].is_none()
          || !std::isfinite(py::cast<double>(values[field]))) {
        throw py::value_error(std::string("invalid committed timing: ") + field);
      }
    }
    return values;
  }

  void add(const char* key, int value) { result_[key] = integer(result_, key) + value; }

  void decisions(const py::list& actions, const char* state)
  {
    for (const auto& action : actions) {
      py::dict row;
      row["action_digest"] = digest_(py::make_tuple(action));
      row["verified_status"] = state;
      result_["action_results"].cast<py::list>().append(std::move(row));
    }
  }

  Observation refresh(py::dict& record, const char* field)
  {
    timing_valid_ = false;
    try {
      const auto summary = refresh_().cast<py::dict>();
      record[field] = summary;
      if (status(summary) != "ok") throw py::value_error("physical timing refresh failed");
    } catch (const std::exception& error) {
      py::dict failure;
      failure["status"] = "failed";
      failure["message"] = error.what();
      record[field] = failure;
      return {"refresh_failed", {}};
    }
    try {
      auto values = timing();
      timing_valid_ = true;
      return {"ok", values};
    } catch (const std::exception& error) {
      record[(std::string(field) + "_timing_error").c_str()] = error.what();
      return {"timing_failed", {}};
    }
  }

  bool undo(py::dict& record, const py::dict& before)
  {
    const auto names = record["committed_buffer_instance_names"].cast<py::list>();
    const auto committed_resizes = record.contains("committed_resizes")
        ? record["committed_resizes"].cast<py::list>() : py::list();
    if (names.empty() && committed_resizes.empty()) {
      record["revert_status"] = "missing_committed_instance_names";
      return false;
    }
    py::list removals;
    for (const auto& name : names) {
      py::dict action;
      action["action_kind"] = "buffer_remove";
      action["instance_name"] = name;
      removals.append(std::move(action));
    }
    for (const auto& item : committed_resizes) {
      const auto r = item.cast<py::dict>();
      py::dict action;
      action["action_kind"] = "resize_revert";
      action["node_id"] = r["node_id"];
      action["instance_name"] = r["instance_name"];
      action["target_cell_index"] = r["previous_cell_index"];
      action["target_master_name"] = r["previous_master_name"];
      removals.append(std::move(action));
    }
    record["revert_digest"] = digest_(removals);
    py::dict reverted;
    try {
      reverted = apply_(removals).cast<py::dict>();
    } catch (const std::exception& error) {
      record["revert_status"] = "native_remove_failed";
      record["revert_message"] = error.what();
      timing_valid_ = false;
      return false;
    }
    record["revert_status"] = status(reverted);
    record["removed_count"] = integer(reverted, "removed_count");
    if ((status(reverted) != "accepted" && status(reverted) != "ok")
        || integer(reverted, "removed_count") != py::len(names) + py::len(committed_resizes)) {
      timing_valid_ = false;
      return false;
    }
    const auto restored = refresh(record, "revert_refresh");
    if (restored.state != "ok") {
      record["revert_status"] = restored.state;
      return false;
    }
    current_ = restored.timing;
    record["timing_after_revert"] = copy(current_);
    for (const char* field : {"wns_ps", "tns_ps"}) {
      if (std::abs(py::cast<double>(current_[field]) - py::cast<double>(before[field])) > kTimingTolerancePs) {
        record["revert_status"] = "timing_not_restored";
        return false;
      }
    }
    if (!electricalNonRegressing(before, current_) || !electricalNonRegressing(current_, before)) {
      record["revert_status"] = "electrical_not_restored";
      return false;
    }
    return true;
  }

  void verify(const Groups& groups, int depth)
  {
    if (status(result_) != "ok") return;
    py::list batch;
    int expected_buffers = 0;
    int expected_resizes = 0;
    for (const auto& group : groups) {
      for (const auto& action : group) {
        batch.append(action);
        if (action.contains("action_kind")
            && py::cast<std::string>(action["action_kind"]) == "resize_cell") {
          ++expected_resizes;
        } else {
          ++expected_buffers;
        }
      }
    }
    const auto before = copy(current_);
    py::dict record;
    record["index"] = py::len(result_["chunks"]);
    record["depth"] = depth;
    record["action_count"] = py::len(batch);
    record["bundle_count"] = groups.size();
    record["input_action_digest"] = digest_(batch);
    record["timing_before"] = before;
    record["wns_before_ps"] = before["wns_ps"];
    record["tns_before_ps"] = before["tns_ps"];
    result_["chunks"].cast<py::list>().append(record);
    add("commit_attempt_count", py::len(batch));
    py::dict commit;
    try {
      commit = apply_(batch).cast<py::dict>();
    } catch (const std::exception& error) {
      record["status"] = "commit_failed";
      result_["status"] = "commit_failed";
      record["message"] = error.what();
      add("commit_failed_count", py::len(batch));
      decisions(batch, "commit_failed");
      const auto actual = refresh(record, "failed_commit_refresh");
      if (actual.state == "ok") current_ = actual.timing;
      return;
    }
    record["commit_status"] = status(commit);
    record["commit_digest"] = digest_(batch);
    const auto names = commit.contains("committed_buffer_instance_names")
        ? commit["committed_buffer_instance_names"].cast<py::list>() : py::list();
    const auto committed_resizes = commit.contains("committed_resizes")
        ? commit["committed_resizes"].cast<py::list>() : py::list();
    record["committed_buffer_instance_names"] = names;
    if (!committed_resizes.empty()) {
      record["committed_resizes"] = committed_resizes;
    }
    const int accepted = integer(commit, "accepted_count");
    const bool complete = (status(commit) == "accepted" || status(commit) == "ok")
        && accepted == py::len(batch)
        && static_cast<int>(py::len(names)) == expected_buffers
        && (expected_resizes == 0 || static_cast<int>(py::len(committed_resizes)) > 0)
        && integer(commit, "rejected_count") == 0 && integer(commit, "failed_count") == 0;
    if (!complete) {
      record["status"] = "commit_failed";
      result_["status"] = "commit_failed";
      if (commit.contains("message")) record["message"] = commit["message"];
      add("commit_failed_count", py::len(batch));
      decisions(batch, "commit_failed");
      if (accepted || !names.empty() || !committed_resizes.empty()) {
        timing_valid_ = false;
        if (!undo(record, before)) {
          result_["status"] = "revert_failed";
          add("revert_failed_count", py::len(batch));
        }
      } else if (integer(commit, "failed_count")) {
        const auto actual = refresh(record, "failed_commit_refresh");
        if (actual.state == "ok") current_ = actual.timing;
      }
      return;
    }
    const auto after = refresh(record, "refresh");
    if (after.state == "ok") {
      current_ = after.timing;
      record["timing_after"] = copy(current_);
      record["wns_after_ps"] = current_["wns_ps"];
      record["tns_after_ps"] = current_["tns_ps"];
      record["electrical_check"] = compareElectricalObservations(before, current_);
      if (py::cast<double>(current_["wns_ps"]) >= py::cast<double>(before["wns_ps"]) - wns_margin_ - kTimingTolerancePs
          && py::cast<double>(current_["tns_ps"]) >= py::cast<double>(before["tns_ps"]) - tns_margin_ - kTimingTolerancePs
          && electricalNonRegressing(before, current_)) {
        record["status"] = "kept";
        add("kept_count", py::len(batch));
        for (const auto& action : batch) result_["kept_actions"].cast<py::list>().append(action);
        decisions(batch, "kept");
        return;
      }
    } else {
      result_["status"] = after.state;
      add((after.state + "_count").c_str(), py::len(batch));
    }
    if (!undo(record, before)) {
      record["status"] = "revert_failed";
      result_["status"] = "revert_failed";
      add("revert_failed_count", py::len(batch));
      decisions(batch, "revert_failed");
      return;
    }
    if (after.state != "ok") {
      record["status"] = after.state;
      decisions(batch, after.state.c_str());
      return;
    }
    if (groups.size() == 1) {
      record["status"] = "reverted";
      add("reverted_count", py::len(batch));
      decisions(batch, "reverted");
      return;
    }
    record["status"] = "split";
    const auto middle = groups.begin() + groups.size() / 2;
    verify(Groups(groups.begin(), middle), depth + 1);
    verify(Groups(middle, groups.end()), depth + 1);
  }

  py::object apply_, refresh_, evaluate_, counts_, digest_;
  double wns_margin_, tns_margin_;
  py::dict result_, current_;
  bool timing_valid_ = false;
};

}  // namespace

void registerVerifiedCommitBindings(py::module_& module)
{
  module.def("run_verified_buffer_commit", [](const py::list& actions,
      const py::object& prepare, const py::object& apply, const py::object& refresh,
      const py::object& evaluate, const py::object& counts, const py::object& digest,
      int chunk_size, double wns_margin, double tns_margin) {
    if (!std::isfinite(wns_margin) || !std::isfinite(tns_margin)
        || wns_margin < 0.0 || tns_margin < 0.0) {
      throw py::value_error("verification margins must be finite and nonnegative");
    }
    return VerifiedCommit(apply, refresh, evaluate, counts, digest, wns_margin, tns_margin)
        .run(actions, prepare, chunk_size);
  }, py::arg("actions"), py::arg("prepare"), py::arg("apply"), py::arg("refresh"),
     py::arg("evaluate"), py::arg("counts"), py::arg("digest"),
     py::arg("chunk_size") = 0, py::arg("wns_margin_ps") = 0.0,
     py::arg("tns_margin_ps") = 0.0);
}

}  // namespace rebuffer
