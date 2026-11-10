// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_driver_tree.hh"


#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <tuple>

#include "pybind11/pybind11.h"

#include "pybind11/stl.h"


namespace rebuffer_python {
rebuffer::DesignView designFromDict(const pybind11::dict& values);
rebuffer::LibertyView libertyFromDict(const pybind11::dict& values);
}  // namespace rebuffer_python

namespace rebuffer {

namespace {

bool callable(const pybind11::object& callback)
{
  return callback && !callback.is_none()
         && PyCallable_Check(callback.ptr());
}

pybind11::dict callDict(const pybind11::object& callback,
                        const pybind11::args& args,
                        const char* name)
{
  if (!callable(callback)) {
    throw pybind11::type_error(std::string("tree recovery requires ") + name
                               + " callback");
  }
  pybind11::object value = callback(*args);
  if (!pybind11::isinstance<pybind11::dict>(value)) {
    throw pybind11::type_error(std::string("tree recovery ") + name
                               + " callback must return a dict");
  }
  return value.cast<pybind11::dict>();
}

pybind11::dict copyDict(const pybind11::dict& values)
{
  return values.attr("copy")().cast<pybind11::dict>();
}

double configDouble(const pybind11::dict& values,
                    const char* key,
                    double fallback)
{
  if (!values.contains(key) || values[key].is_none()) {
    return fallback;
  }
  return pybind11::cast<double>(values[key]);
}

int configInt(const pybind11::dict& values, const char* key, int fallback)
{
  if (!values.contains(key) || values[key].is_none()) {
    return fallback;
  }
  return pybind11::cast<int>(values[key]);
}

pybind11::dict physicalCheckpoint(const pybind11::object& callback,
                                  const std::string& label,
                                  const pybind11::dict& mutation)
{
  pybind11::tuple args(2);
  args[0] = label;
  args[1] = mutation;
  return callDict(callback, args, "physical_checkpoint");
}

pybind11::object checkpointToken(const pybind11::dict& checkpoint)
{
  if (checkpoint.contains("checkpoint_id")) {
    return checkpoint["checkpoint_id"];
  }
  if (checkpoint.contains("token")) {
    return checkpoint["token"];
  }
  return pybind11::reinterpret_borrow<pybind11::object>(checkpoint);
}

void restorePhysical(const pybind11::object& callback,
                     const pybind11::object& token)
{
  pybind11::tuple args(1);
  args[0] = token;
  callDict(callback, args, "physical_restore");
}

bool keepsUpperBound(const pybind11::dict& current,
                     const pybind11::dict& candidate,
                     const char* key,
                     double margin)
{
  if (!current.contains(key) || current[key].is_none()) {
    return true;
  }
  if (!candidate.contains(key) || candidate[key].is_none()) {
    return false;
  }
  const double current_value = pybind11::cast<double>(current[key]);
  const double candidate_value = pybind11::cast<double>(candidate[key]);
  return std::isfinite(current_value) && std::isfinite(candidate_value)
         && candidate_value <= current_value + margin;
}

bool keepsLowerBound(const pybind11::dict& current,
                     const pybind11::dict& candidate,
                     const char* key,
                     double margin)
{
  if (!current.contains(key) || current[key].is_none()
      || !candidate.contains(key) || candidate[key].is_none()) {
    return false;
  }
  const double current_value = pybind11::cast<double>(current[key]);
  const double candidate_value = pybind11::cast<double>(candidate[key]);
  return std::isfinite(current_value) && std::isfinite(candidate_value)
         && candidate_value >= current_value - margin;
}

bool keepsTiming(const pybind11::dict& current,
                 const pybind11::dict& candidate,
                 const pybind11::dict& config)
{
  if (candidate.contains("status")
      && pybind11::cast<std::string>(candidate["status"]) != "evaluated") {
    return false;
  }
  const double wns_margin = configDouble(config, "recovery_wns_margin_ps", 0.0);
  const double tns_margin = configDouble(config, "recovery_tns_margin_ps", 0.0);
  const double slew_margin = configDouble(config, "recovery_slew_margin", 0.0);
  const double cap_margin = configDouble(config, "recovery_cap_margin", 0.0);
  const bool timing_ok = keepsLowerBound(
                             current, candidate, "wns_ps", wns_margin)
                         && keepsLowerBound(
                             current, candidate, "tns_ps", tns_margin);
  return timing_ok
         && keepsUpperBound(
             current, candidate, "max_slew_violation", slew_margin)
         && keepsUpperBound(
             current, candidate, "max_cap_violation", cap_margin);
}

}  // namespace

std::vector<TreeRecoveryCandidate> enumerateTreeAreaRecovery(
    const CurrentDriverTree& tree,
    const DesignView& design,
    const LibertyView& liberty,
    std::string& reason)
{
  std::vector<TreeRecoveryCandidate> result;
  if (!validateCurrentDriverTree(tree, reason)) {
    return result;
  }
  for (const auto& node : tree.nodes) {
    if (node.kind != static_cast<int>(DriverTreeNodeKind::buffer)
        || node.fixed) {
      continue;
    }
    std::vector<int> parents;
    std::vector<int> children;
    for (const auto& edge : tree.edges) {
      if (edge.child_node_id == node.id) {
        parents.push_back(edge.parent_node_id);
      }
      if (edge.parent_node_id == node.id) {
        children.push_back(edge.child_node_id);
      }
    }
    if (node.source_instance_id < 0
        || node.source_instance_id >= static_cast<int>(design.inst_main_id.size())) {
      reason = "recovery_buffer_instance_identity_missing";
      return {};
    }
    if (node.cell_id < 0 || node.cell_id >= static_cast<int>(liberty.cells.size())) {
      reason = "recovery_buffer_cell_identity_missing";
      return {};
    }
    const bool pass_through = parents.size() == 1 && children.size() == 1;
    if (pass_through) {
      TreeRecoveryCandidate candidate;
      candidate.mutation.kind = static_cast<int>(DriverTreeMutationKind::remove);
      candidate.mutation.node_id = node.id;
      candidate.mutation.source_instance_id = node.source_instance_id;
      candidate.mutation.parent_node_id = parents.front();
      candidate.mutation.child_node_id = children.front();
      candidate.area_saved = liberty.cells[node.cell_id].area;
      candidate.recovery_kind = "remove_buffer_pass_through";
      result.push_back(candidate);
    }
    const int main_id = design.inst_main_id[node.source_instance_id];
    for (int target_cell = 0;
         target_cell < static_cast<int>(liberty.cells.size());
         ++target_cell) {
      const auto& target = liberty.cells[target_cell];
      if (!target.is_buffer || target.main_id != main_id
          || target.area >= liberty.cells[node.cell_id].area) {
        continue;
      }
      TreeRecoveryCandidate candidate;
      candidate.mutation.kind = static_cast<int>(DriverTreeMutationKind::rewrite);
      candidate.mutation.node_id = node.id;
      candidate.mutation.source_instance_id = node.source_instance_id;
      candidate.mutation.parent_node_id = pass_through ? parents.front() : -1;
      candidate.mutation.child_node_id = pass_through ? children.front() : -1;
      candidate.mutation.target_cell_index = target_cell;
      candidate.mutation.target_main_id = target.main_id;
      candidate.area_saved = liberty.cells[node.cell_id].area - target.area;
      candidate.buffer_count_delta = 0;
      candidate.recovery_kind = "rewrite_buffer_smaller_master";
      result.push_back(candidate);
    }
  }
  std::sort(
      result.begin(),
      result.end(),
      [](const TreeRecoveryCandidate& left, const TreeRecoveryCandidate& right) {
        if (left.area_saved != right.area_saved) {
          return left.area_saved > right.area_saved;
        }
        if (left.mutation.node_id != right.mutation.node_id) {
          return left.mutation.node_id < right.mutation.node_id;
        }
        if (left.mutation.kind != right.mutation.kind) {
          return left.mutation.kind < right.mutation.kind;
        }
        return left.mutation.target_cell_index
               < right.mutation.target_cell_index;
      });
  return result;
}

pybind11::dict runTreeAreaRecovery(
    const std::shared_ptr<CurrentDriverTreeTransaction>& transaction,
    const pybind11::dict& design_values,
    const pybind11::dict& liberty_values,
    const pybind11::object& physical_checkpoint,
    const pybind11::object& physical_apply,
    const pybind11::object& evaluate,
    const pybind11::object& physical_restore,
    const pybind11::dict& config)
{
  if (transaction == nullptr) {
    throw pybind11::value_error("tree recovery transaction is null");
  }
  // Validate and materialize the native views before changing the transaction.
  const auto design = rebuffer_python::designFromDict(design_values);
  const auto liberty = rebuffer_python::libertyFromDict(liberty_values);
  std::string reason;
  const auto initial_candidates = enumerateTreeAreaRecovery(
      transaction->activeTreeRef(), design, liberty, reason);
  if (!reason.empty()) {
    throw pybind11::value_error(reason);
  }

  pybind11::tuple initial_args(2);
  initial_args[0] = pybind11::none();
  initial_args[1] = pybind11::none();
  pybind11::dict current = callDict(evaluate, initial_args, "evaluate");
  if (current.contains("status")
      && pybind11::cast<std::string>(current["status"]) != "evaluated") {
    throw pybind11::value_error("tree recovery baseline evaluation failed");
  }
  const pybind11::dict initial = copyDict(current);
  pybind11::list ledger;
  pybind11::list accepted;
  double area_saved = 0.0;
  int accepted_count = 0;
  int attempt = 0;
  const int default_attempt_limit
      = static_cast<int>(initial_candidates.size());
  const int configured_attempt_limit = configInt(
      config, "recovery_max_attempts", default_attempt_limit);
  if (configured_attempt_limit < 0) {
    throw pybind11::value_error("recovery_max_attempts must be nonnegative");
  }
  std::set<std::tuple<int, int, int>> attempted_candidates;

  while (attempt < configured_attempt_limit) {
    std::string iteration_reason;
    const auto candidates = enumerateTreeAreaRecovery(
        transaction->activeTreeRef(), design, liberty, iteration_reason);
    if (!iteration_reason.empty()) {
      throw pybind11::value_error(iteration_reason);
    }
    const auto candidate_it = std::find_if(
        candidates.begin(),
        candidates.end(),
        [&attempted_candidates](const TreeRecoveryCandidate& value) {
          return attempted_candidates.find(
                     {value.mutation.source_instance_id,
                      value.mutation.kind,
                      value.mutation.target_cell_index})
                 == attempted_candidates.end();
        });
    if (candidate_it == candidates.end()) {
      break;
    }
    const auto& candidate = *candidate_it;
    attempted_candidates.insert(
        {candidate.mutation.source_instance_id,
         candidate.mutation.kind,
         candidate.mutation.target_cell_index});
    const pybind11::dict mutation
        = driverTreeMutationToDict(candidate.mutation);
    mutation["area_saved"] = candidate.area_saved;
    mutation["buffer_count_delta"] = candidate.buffer_count_delta;
    mutation["recovery_kind"] = candidate.recovery_kind;
    const auto native_checkpoint = transaction->checkpoint();
    const pybind11::dict physical_values = physicalCheckpoint(
        physical_checkpoint,
        "tree_area_recovery_" + std::to_string(attempt),
        mutation);
    const pybind11::object physical_token = checkpointToken(physical_values);
    pybind11::dict row;
    row["attempt"] = attempt++;
    row["node_id"] = candidate.mutation.node_id;
    row["source_instance_id"] = candidate.mutation.source_instance_id;
    row["area_saved"] = candidate.area_saved;
    row["buffer_count_delta"] = candidate.buffer_count_delta;
    row["recovery_kind"] = candidate.recovery_kind;
    try {
      const pybind11::dict tree_result
          = transaction->applyMutation(mutation);
      pybind11::tuple apply_args(2);
      apply_args[0] = mutation;
      apply_args[1] = tree_result;
      callDict(physical_apply, apply_args, "physical_apply");
      pybind11::tuple evaluate_args(2);
      evaluate_args[0] = mutation;
      evaluate_args[1] = tree_result;
      pybind11::dict observation
          = callDict(evaluate, evaluate_args, "evaluate");
      observation["status"] = "evaluated";
      observation["timing_epoch"] = attempt;
      observation["source_tree_digest"] = tree_result["source_tree_digest"];
      observation["candidate_tree_digest"]
          = tree_result["candidate_tree_digest"];
      const bool keep = keepsTiming(current, observation, config);
      if (keep) {
        transaction->evaluateCandidate(observation);
        transaction->acceptCandidate();
        current = observation;
        accepted.append(mutation);
        area_saved += candidate.area_saved;
        ++accepted_count;
        row["status"] = "accepted";
      } else {
        restorePhysical(physical_restore, physical_token);
        transaction->restore(native_checkpoint);
        row["status"] = "rejected";
      }
    } catch (const pybind11::error_already_set& error) {
      try {
        restorePhysical(physical_restore, physical_token);
        transaction->restore(native_checkpoint);
      } catch (const pybind11::error_already_set&) {
        throw;
      }
      row["status"] = "failed_restored";
      row["reason"] = error.what();
    }
    ledger.append(std::move(row));
  }

  pybind11::dict result;
  result["status"] = "ok";
  result["operator"] = "rebuffer_native_tree_area_recovery";
  result["core"] = "cpp";
  result["baseline"] = initial;
  result["final"] = current;
  result["ledger"] = std::move(ledger);
  result["accepted_candidates"] = std::move(accepted);
  result["accepted_count"] = accepted_count;
  result["area_saved"] = area_saved;
  return result;
}

}  // namespace rebuffer
