// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_tree_bindings.hh"
#include "pybind11/stl.h"
#include "rebuffer_python_views.hh"
#include "rebuffer_tree_search.hh"

namespace py = pybind11;
using namespace rebuffer_python;

namespace rebuffer {
void registerTreeSearchBindings(py::module_& module)
{
  auto plan = [](bool optimize, const py::dict& design_values,
         const py::dict& timing_values,
         const py::dict& liberty_values,
         const py::dict& characterization_values,
         const py::dict& config_values,
         const std::shared_ptr<rebuffer::CurrentDriverTree>& tree) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        rebuffer::TreeVgdpResult result;
        std::string reason;
        try {
          const auto design = designFromDict(design_values);
          const auto timing = timingFromDict(timing_values);
          const auto liberty = libertyFromDict(liberty_values);
          const auto characterization
              = characterizationFromDict(characterization_values);
          const auto config = configFromDict(config_values);
          const bool planned = optimize
              ? rebuffer::optimizeCurrentDriverTree(design, timing, liberty,
                    characterization, config, *tree, result, reason)
              : rebuffer::planCurrentDriverTree(design, timing, liberty,
                    characterization, config, *tree, result, reason);
          if (!planned) {
            py::dict output;
            output["status"] = "unsupported";
            py::list reasons;
            reasons.append(reason);
            output["unsupported_reasons"] = std::move(reasons);
            return output;
          }
        } catch (const std::exception& error) {
          py::dict output;
          output["status"] = "unsupported";
          py::list reasons;
          reasons.append(error.what());
          output["unsupported_reasons"] = std::move(reasons);
          return output;
        }
        py::dict output;
        output["status"] = "ok";
        output["generated_option_count"] = result.generated_option_count;
        output["surviving_option_count"] = result.surviving_option_count;
        output["pruned_option_count"] = result.pruned_option_count;
        output["tree_node_count"] = result.tree_node_count;
        output["tree_edge_count"] = result.tree_edge_count;
        output["tree_branch_count"] = result.tree_branch_count;
        output["tree_max_depth"] = result.tree_max_depth;
        output["driver_resistance"] = result.driver_resistance;
        output["current_driver_load"] = result.current_driver_load;
        output["long_wire_edge_count"] = result.long_wire_edge_count;
        output["wire_site_count"] = result.wire_site_count;
        output["electrical_rejected_count"] = result.electrical_rejected_count;
        output["timing_round_count"] = result.timing_round_count;
        output["recovery_attempt_count"] = result.recovery_attempt_count;
        output["recovered_buffer_count"] = result.recovered_buffer_count;
        output["round_source_tree_digests"] = result.round_source_tree_digests;
        output["round_source_buffer_counts"] = result.round_source_buffer_counts;
        py::list candidates;
        for (const auto& candidate : result.candidates) {
          py::dict row;
          row["cap"] = candidate.cap;
          row["slack"] = candidate.slack;
          row["slew_rise"] = candidate.slew_rise;
          row["slew_fall"] = candidate.slew_fall;
          row["area"] = candidate.area;
          row["buffer_count"] = candidate.buffer_count;
          py::list actions;
          for (const auto& action : candidate.actions) {
            actions.append(bufferActionDict(action));
          }
          row["actions"] = std::move(actions);
          candidates.append(std::move(row));
        }
        output["candidates"] = std::move(candidates);
        py::list trace;
        for (const auto& item : result.trace) {
          py::dict row;
          row["trace_id"] = item.trace_id;
          row["node_id"] = item.node_id;
          row["operation"] = item.operation;
          row["cap"] = item.cap;
          row["slack"] = item.slack;
          row["slew_rise"] = item.slew_rise;
          row["slew_fall"] = item.slew_fall;
          row["area"] = item.area;
          row["buffer_count"] = item.buffer_count;
          row["survivor"] = item.survivor;
          row["reason"] = item.reason;
          trace.append(std::move(row));
        }
        output["trace"] = std::move(trace);
        return output;
      };
  for (const bool optimize : {false, true}) {
    module.def(optimize ? "optimize_current_driver_tree" : "plan_current_driver_tree",
        [plan, optimize](const py::dict& design, const py::dict& timing,
                         const py::dict& liberty, const py::dict& characterization,
                         const py::dict& config,
                         const std::shared_ptr<CurrentDriverTree>& tree) {
          return plan(optimize, design, timing, liberty, characterization, config, tree);
        }, py::arg("design"), py::arg("timing"), py::arg("liberty"),
        py::arg("characterization"), py::arg("config"), py::arg("tree"));
  }
}
}  // namespace rebuffer
