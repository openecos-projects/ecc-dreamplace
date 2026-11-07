// SPDX-License-Identifier: BSD-3-Clause
// Python module registration for the native rebuffer implementation.

#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "rebuffer_liberty.hh"
#include "rebuffer_library_metadata.hh"
#include "rebuffer_cell_selection.hh"
#include "rebuffer_incremental_topology.hh"
#include "rebuffer_incremental_timing.hh"
#include "rebuffer_incremental_carrier.hh"
#include "rebuffer_incremental_patch.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_planning.hh"
#include "rebuffer_timing_payload.hh"
#include "rebuffer_python_views.hh"
#include "rebuffer_path_policy.hh"
#include "rebuffer_native_queue.hh"
#include "rebuffer_legacy_queue.hh"
#include "rebuffer_timing_local.hh"
#include "rebuffer_queue_snapshot.hh"
#include "rebuffer_native_planner.hh"
#include "rebuffer_verified_commit.hh"
#include "rebuffer_electrical_observation.hh"
#include "rebuffer_repair.hh"
#include "rebuffer_timing_session.hh"
#include "rebuffer_types.hh"
#include "rebuffer_carrier.hh"
#include "rebuffer_driver_tree.hh"
#include "rebuffer_tree_dp.hh"
#include "rebuffer_tree_bindings.hh"
#include "rebuffer_tree_recovery.hh"
#include "rebuffer_overlay_carrier.hh"
#include "rebuffer_overlay_actions.hh"

namespace py = pybind11;

using namespace rebuffer_python;

PYBIND11_MODULE(rebuffer_cpp, module)
{
  module.doc() = "Native rebuffer design and Steiner forest skeleton";
  rebuffer::registerVerifiedCommitBindings(module);
  rebuffer::registerElectricalObservationBindings(module);
  rebuffer::registerRepairLibraryMetadata(module);
  rebuffer::registerCellSelectionBindings(module);
  py::class_<rebuffer::TimingGraphIndex, std::shared_ptr<rebuffer::TimingGraphIndex>>(module, "TimingGraphIndex")
      .def("update_cells", [](rebuffer::TimingGraphIndex& self, const std::vector<int>& indices,
                              const std::vector<rebuffer::TimingCellArc>& rows) {
        for (std::size_t i = 0; i < indices.size() && i < rows.size(); ++i) {
          const int id = indices[i];
          if (id >= 0 && id < static_cast<int>(self.cells.size())) {
            self.arc_ids.erase(self.key(id));
            self.cells[id] = rows[i];
            self.arc_ids.emplace(self.key(id), id);
          }
        }
      });
  py::class_<rebuffer::PersistentTimingGraphCheckpoint,
             std::shared_ptr<rebuffer::PersistentTimingGraphCheckpoint>>(
      module, "PersistentTimingGraphCheckpoint");
  py::class_<rebuffer::PersistentTimingGraph,
             std::shared_ptr<rebuffer::PersistentTimingGraph>>(
      module, "PersistentTimingGraph")
      .def("summary", &rebuffer::PersistentTimingGraph::summary)
      .def_property_readonly("epoch", &rebuffer::PersistentTimingGraph::epoch)
      .def("state_view", &rebuffer::PersistentTimingGraph::stateView,
           py::arg("copy") = true)
      .def("state_view_statistics", &rebuffer::PersistentTimingGraph::stateViewStatistics)
      .def("set_state", &rebuffer::PersistentTimingGraph::setState)
      .def("timing_state_schema", &rebuffer::PersistentTimingGraph::timingStateSchema)
      .def("initialize_propagation", &rebuffer::PersistentTimingGraph::initializePropagation,
           py::arg("values"))
      .def("propagate", &rebuffer::PersistentTimingGraph::propagate)
      .def("propagate_incremental", &rebuffer::PersistentTimingGraph::propagateIncremental)
      .def("setup_path_details", &rebuffer::PersistentTimingGraph::setupPathDetails,
           py::arg("endpoint_pins") = py::none(), py::arg("global_k") = 0,
           py::arg("max_depth") = 0, py::arg("residual_tolerance_ps") = 1e-3)
      .def("initialize_queue", &rebuffer::PersistentTimingGraph::initializeQueue,
           py::arg("liberty"), py::arg("original_node_is_buffer"))
      .def("timing_queue_state", &rebuffer::PersistentTimingGraph::timingQueueState,
           py::arg("original_net_by_net"), py::arg("use_path_context") = false,
           py::arg("endpoint_pin") = std::nullopt)
      .def("timing_endpoint_state", &rebuffer::PersistentTimingGraph::timingEndpointState)
      .def("timing_queue_state_for_carrier",
           &rebuffer::PersistentTimingGraph::timingQueueStateForCarrier,
           py::arg("original_net_by_net"), py::arg("carrier"),
           py::arg("use_path_context") = false)
      .def("evaluation_observation", &rebuffer::PersistentTimingGraph::evaluationObservation,
           py::arg("max_fanout") = 32, py::arg("cap_margin") = 0.0, py::arg("slew_margin") = 0.0)
      .def("initialize_planning", &rebuffer::PersistentTimingGraph::initializePlanning,
           py::arg("design"), py::arg("characterization"), py::arg("steiner"),
           py::arg("config"), py::arg("r_unit"), py::arg("c_unit"))
      .def("planning_snapshot", &rebuffer::PersistentTimingGraph::planningSnapshot,
           py::arg("actions"), py::arg("config"))
      .def("local_planning_snapshot", &rebuffer::PersistentTimingGraph::localPlanningSnapshot,
           py::arg("actions"), py::arg("config"))
      .def("initialize_mutations", &rebuffer::PersistentTimingGraph::initializeMutations)
      .def("mutation_view", &rebuffer::PersistentTimingGraph::mutationView)
      .def("mutation_statistics", &rebuffer::PersistentTimingGraph::mutationStatistics)
      .def("mutation_index_oracle", &rebuffer::PersistentTimingGraph::mutationIndexOracle)
      .def("compact", &rebuffer::PersistentTimingGraph::compact)
      .def("resize_cell", &rebuffer::PersistentTimingGraph::resizeCell,
           py::arg("instance_name"), py::arg("master_name"))
      .def("initialize_timing_state",
           &rebuffer::PersistentTimingGraph::initializeTimingState,
           py::arg("state"))
      .def("attach_net_topology",
           &rebuffer::PersistentTimingGraph::attachNetTopology,
           py::arg("net_id"), py::arg("driver_pin_id"),
           py::arg("next_node_id"), py::arg("topology_values"))
      .def("attach_net_topology_state",
           &rebuffer::PersistentTimingGraph::attachNetTopologyState,
           py::arg("net_id"), py::arg("state"))
      .def("clear_net_topologies",
           &rebuffer::PersistentTimingGraph::clearNetTopologies)
      .def("apply_buffer_split",
           &rebuffer::PersistentTimingGraph::applyBufferSplit,
           py::arg("net_id"), py::arg("action"))
      .def("remove_buffer", &rebuffer::PersistentTimingGraph::removeBuffer,
           py::arg("net_id"), py::arg("action_id"))
      .def("resize_buffer", &rebuffer::PersistentTimingGraph::resizeBuffer,
           py::arg("net_id"), py::arg("action"))
      .def("topology_view", &rebuffer::PersistentTimingGraph::topologyView,
           py::arg("net_id"))
      .def("relevelize_net", &rebuffer::PersistentTimingGraph::relevelizeNet,
           py::arg("net_id"))
      .def("compact_net_topology",
           &rebuffer::PersistentTimingGraph::compactNetTopology,
           py::arg("net_id"))
      .def("critical_path_subset",
           &rebuffer::PersistentTimingGraph::criticalPathSubset,
           py::arg("path_details"), py::arg("endpoint_pins"))
      .def("finalize_actions", &rebuffer::PersistentTimingGraph::finalizeActions,
           py::arg("actions"))
      .def("build_net_rc", &rebuffer::PersistentTimingGraph::buildNetRc,
           py::arg("net_id"))
      .def("build_timing_patch",
           &rebuffer::PersistentTimingGraph::buildTimingPatch,
           py::arg("net_id"), py::arg("action"),
           py::arg("base_vertex_count"), py::arg("slot"))
      .def("export_carrier", &rebuffer::PersistentTimingGraph::exportCarrier,
           py::arg("steiner"), py::arg("actions"),
           py::arg("r_unit"), py::arg("c_unit"))
      .def("evaluate_net_rc",
           &rebuffer::PersistentTimingGraph::evaluateNetRc,
           py::arg("net_id"), py::arg("rc"), py::arg("generic_caps"),
           py::arg("rise_caps"), py::arg("fall_caps"),
           py::arg("buffer_caps"), py::arg("pin_count"))
      .def("checkpoint", &rebuffer::PersistentTimingGraph::checkpoint)
      .def("restore", &rebuffer::PersistentTimingGraph::restore)
      .def("commit", &rebuffer::PersistentTimingGraph::commit);
  module.def("create_persistent_timing_graph",
             &rebuffer::createPersistentTimingGraph,
             py::arg("carrier"));
  module.def("persistent_timing_state_schema",
             &rebuffer::persistentTimingStateSchema,
             py::arg("pin_count"), py::arg("cell_arc_count"));
  py::class_<rebuffer::NativeTimingQueueSnapshot,
             std::shared_ptr<rebuffer::NativeTimingQueueSnapshot>>(
      module, "NativeTimingQueueSnapshot");
  module.def("affected_timing_schedule", &rebuffer::affectedTimingSchedule,
             py::arg("previous"), py::arg("current"), py::arg("previous_rc"),
             py::arg("current_rc"), py::arg("changed_pins") = py::none());
  module.def("pin_pair_arc_mapping", &rebuffer::pinPairArcMapping);
  module.def("run_repair_timing_queue_operator",
             &rebuffer::runRepairTimingQueueOperator,
             py::arg("timing_snapshot"),
             py::arg("candidate_provider"),
             py::arg("checkpoint"),
             py::arg("apply"),
             py::arg("evaluate"),
             py::arg("restore"),
             py::arg("config"),
             py::arg("release_checkpoint") = py::none());
  module.def("build_native_timing_queue_snapshot",
             &rebuffer::buildNativeTimingQueueSnapshot,
             py::arg("design"),
             py::arg("timing"),
             py::arg("liberty"),
             py::arg("endpoint_records"),
             py::arg("original_net_by_net"));
  module.def("build_native_timing_queue_state",
             &rebuffer::buildNativeTimingQueueState,
             py::arg("design"),
             py::arg("timing"),
             py::arg("liberty"),
             py::arg("endpoint_records"),
             py::arg("original_net_by_net"));
  module.def("update_native_timing_queue_snapshot",
             &rebuffer::updateNativeTimingQueueSnapshot,
             py::arg("design"),
             py::arg("timing"),
             py::arg("liberty"),
             py::arg("previous_snapshot"),
             py::arg("changed_endpoint_records"),
             py::arg("active_endpoint_pins"),
             py::arg("changed_endpoint_pins"),
             py::arg("original_net_by_net"));
  module.def("update_native_timing_queue_state",
             &rebuffer::updateNativeTimingQueueState,
             py::arg("design"),
             py::arg("timing"),
             py::arg("liberty"),
             py::arg("previous_snapshot"),
             py::arg("changed_endpoint_records"),
             py::arg("active_endpoint_pins"),
             py::arg("changed_endpoint_pins"),
             py::arg("original_net_by_net"));
  module.def("select_timing_queue_items",
             &rebuffer::selectTimingQueueItems,
             py::arg("snapshot"),
             py::arg("config"));
  module.def("update_native_timing_queue_focus",
             &rebuffer::updateNativeTimingQueueFocus,
             py::arg("snapshot"),
             py::arg("focus_updates"));
  module.def("run_repair_timing_native_snapshot_operator",
             &rebuffer::runRepairTimingNativeSnapshotOperator,
             py::arg("timing_snapshot"),
             py::arg("checkpoint"),
             py::arg("apply"),
             py::arg("evaluate"),
             py::arg("restore"),
             py::arg("config"), py::arg("endpoint_state") = py::none(),
             py::arg("release_checkpoint") = py::none());
  module.def(
      "rank_path_driver_pins",
      [](const py::dict& design_values,
         const py::dict& timing_values,
         const py::dict& liberty_values,
         const std::vector<int>& path,
         bool openroad_intrinsic) {
        return rebuffer::rankPathDriverPins(
            designFromDict(design_values),
            timingFromDict(timing_values),
            libertyFromDict(liberty_values),
            path, openroad_intrinsic);
      },
      py::arg("design"),
      py::arg("timing"),
      py::arg("liberty"),
      py::arg("path"), py::arg("openroad_intrinsic") = false);
  module.def("timing_pin_capacitance", &rebuffer::timingPinCapacitance);
  module.def("repair_design_electrical_payload", &rebuffer::repairDesignElectricalPayload,
             py::arg("rise_load"), py::arg("fall_load"),
             py::arg("rise_cap"), py::arg("fall_cap"));
  module.def("rebuild_timing_carrier_incremental", &rebuffer::rebuildTimingCarrierIncremental,
             py::arg("base_pydb"), py::arg("previous_pydb"), py::arg("physical_pydb"));
  module.def("build_incremental_timing_patch", &rebuffer::buildIncrementalTimingPatch,
             py::arg("base_pydb"), py::arg("topology_state"), py::arg("action"),
             py::arg("base_vertex_count"), py::arg("slot"));
  module.def("evaluate_incremental_rc", &rebuffer::evaluateIncrementalRc,
             py::arg("rc"), py::arg("generic_caps"), py::arg("rise_caps"),
             py::arg("fall_caps"), py::arg("buffer_caps"), py::arg("pin_count"));
  module.def("resolve_overlay_split", &rebuffer::resolveOverlaySplit);
  module.def("project_overlay_actions", &rebuffer::projectOverlayActions);
  module.def("finalize_overlay_actions", &rebuffer::finalizeOverlayActions);
  module.def("rebuild_timing_carrier", &rebuffer::rebuildTimingCarrier,
             py::arg("base_pydb"), py::arg("physical_pydb"));
  module.def("export_overlay_carrier",
             [](const py::object& base, const py::dict& steiner,
                const std::vector<std::shared_ptr<rebuffer::IncrementalTopologyState>>& states,
                const py::list& actions, double r_unit, double c_unit) {
               return rebuffer::exportOverlayCarrier(
                   base, steinerFromDict(steiner), states, actions, r_unit, c_unit);
             }, py::arg("base_pydb"), py::arg("steiner"), py::arg("states"),
             py::arg("actions"), py::arg("r_unit") = 0.914,
             py::arg("c_unit") = 1.89e-4);
  module.def(
      "incremental_topology_from_steiner",
      [](const py::dict& steiner, int net_id, int driver_pin_id,
         double r_unit, double c_unit) {
        return rebuffer::incrementalTopologyFromSteiner(
            steinerFromDict(steiner), net_id, driver_pin_id, r_unit, c_unit);
      },
      py::arg("steiner"), py::arg("net_id"), py::arg("driver_pin_id"),
      py::arg("r_unit"), py::arg("c_unit"));
  module.def(
      "remap_incremental_topology",
      &rebuffer::remapIncrementalTopology,
      py::arg("state"), py::arg("original_pin_count"), py::arg("node_shift"),
      py::arg("target_net_id") = -1,
      py::arg("synthetic_node_base") = -1,
      py::arg("source_node_mapping") = py::dict());
  py::class_<rebuffer::IncrementalTopologyState,
             std::shared_ptr<rebuffer::IncrementalTopologyState>>(
      module, "IncrementalTopologyState")
      .def_property_readonly("epoch", [](const rebuffer::IncrementalTopologyState& state) {
        return state.epoch;
      })
      .def_property_readonly("net_id", [](const rebuffer::IncrementalTopologyState& state) {
        return state.net_id;
      });
  py::class_<rebuffer::IncrementalTopologyCheckpoint,
             std::shared_ptr<rebuffer::IncrementalTopologyCheckpoint>>(
      module, "IncrementalTopologyCheckpoint");
  py::class_<rebuffer::RepairTimingSessionCheckpoint,
             std::shared_ptr<rebuffer::RepairTimingSessionCheckpoint>>(
      module, "RepairTimingSessionCheckpoint");
  py::class_<rebuffer::RepairTimingSession,
             std::shared_ptr<rebuffer::RepairTimingSession>>(
      module, "RepairTimingSession")
      .def("apply_candidate", &rebuffer::RepairTimingSession::applyCandidate)
      .def("remove_candidate", &rebuffer::RepairTimingSession::removeCandidate)
      .def("checkpoint", &rebuffer::RepairTimingSession::checkpoint)
      .def("restore", &rebuffer::RepairTimingSession::restore)
      .def("summary", &rebuffer::RepairTimingSession::summary)
      .def_property_readonly("topology",
                             &rebuffer::RepairTimingSession::topology);
  module.def("create_repair_timing_session",
             &rebuffer::createRepairTimingSession,
             py::arg("topology"));
  py::class_<rebuffer::RepairTimingCampaignCheckpoint,
             std::shared_ptr<rebuffer::RepairTimingCampaignCheckpoint>>(
      module, "RepairTimingCampaignCheckpoint");
  py::class_<rebuffer::RepairTimingCampaignSession,
             std::shared_ptr<rebuffer::RepairTimingCampaignSession>>(
      module, "RepairTimingCampaignSession")
      .def("attach_topology",
           &rebuffer::RepairTimingCampaignSession::attachTopology)
      .def("apply_candidate",
           &rebuffer::RepairTimingCampaignSession::applyCandidate)
      .def("remove_candidate",
           &rebuffer::RepairTimingCampaignSession::removeCandidate)
      .def("replace_candidate_tree",
           &rebuffer::RepairTimingCampaignSession::replaceCandidateTree)
      .def("record_candidate",
           &rebuffer::RepairTimingCampaignSession::recordCandidate)
      .def("record_action_state",
           &rebuffer::RepairTimingCampaignSession::recordActionState)
      .def("checkpoint", &rebuffer::RepairTimingCampaignSession::checkpoint)
      .def("restore_state", &rebuffer::RepairTimingCampaignSession::restoreState)
      .def("restore", &rebuffer::RepairTimingCampaignSession::restore)
      .def("summary", &rebuffer::RepairTimingCampaignSession::summary)
      .def_property_readonly("timing_graph", &rebuffer::RepairTimingCampaignSession::timingGraph);
  module.def("create_repair_timing_campaign_session",
             &rebuffer::createRepairTimingCampaignSession,
             py::arg("timing_graph") = nullptr);
  module.def(
      "create_incremental_topology",
      [](const int net_id,
         const int driver_pin_id,
         const int next_node_id,
         const py::dict& topology) {
        std::string reason;
        const auto state = rebuffer::createIncrementalTopology(
            net_id, driver_pin_id, next_node_id, topology, reason);
        if (state == nullptr) {
          throw py::value_error(reason);
        }
        return state;
      },
      py::arg("net_id"),
      py::arg("driver_pin_id"),
      py::arg("next_node_id"),
      py::arg("topology"));
  module.def(
      "apply_incremental_buffer_split",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state,
         const py::dict& action) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        std::string reason;
        py::dict result;
        if (!rebuffer::applyIncrementalBufferSplit(
                *state, action, reason, result)) {
          throw py::value_error(reason);
        }
        return result;
      },
      py::arg("state"),
      py::arg("action"));
  module.def(
      "remove_incremental_buffer",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state,
         const int action_id) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        std::string reason;
        py::dict result;
        if (!rebuffer::removeIncrementalBuffer(
                *state, action_id, reason, result)) {
          throw py::value_error(reason);
        }
        return result;
      },
      py::arg("state"),
      py::arg("action_id"));
  module.def(
      "resize_incremental_buffer",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state,
         const py::dict& action) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        std::string reason;
        py::dict result;
        if (!rebuffer::resizeIncrementalBuffer(
                *state, action, reason, result)) {
          throw py::value_error(reason);
        }
        return result;
      },
      py::arg("state"),
      py::arg("action"));
  module.def(
      "checkpoint_incremental_topology",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        return rebuffer::checkpointIncrementalTopology(*state);
      },
      py::arg("state"));
  module.def(
      "restore_incremental_topology",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state,
         const std::shared_ptr<rebuffer::IncrementalTopologyCheckpoint>& checkpoint) {
        if (state == nullptr || checkpoint == nullptr) {
          throw py::value_error("topology state and checkpoint are required");
        }
        std::string reason;
        if (!rebuffer::restoreIncrementalTopology(*state, *checkpoint, reason)) {
          throw py::value_error(reason);
        }
        return rebuffer::incrementalTopologyView(*state);
      },
      py::arg("state"),
      py::arg("checkpoint"));
  module.def(
      "validate_incremental_topology",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        std::string reason;
        const bool valid = rebuffer::validateIncrementalTopology(*state, reason);
        py::dict result;
        result["status"] = valid ? "ok" : "invalid";
        result["valid"] = valid;
        result["reason"] = reason;
        result["epoch"] = state->epoch;
        return result;
      },
      py::arg("state"));
  module.def(
      "incremental_topology_view",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        return rebuffer::incrementalTopologyView(*state);
      },
      py::arg("state"));
  module.def(
      "clone_incremental_topology",
      [](const std::shared_ptr<rebuffer::IncrementalTopologyState>& state) {
        if (state == nullptr) {
          throw py::value_error("topology state is null");
        }
        return std::make_shared<rebuffer::IncrementalTopologyState>(*state);
      },
      py::arg("state"));
  module.def(
      "characterize_buffers",
      [](const py::dict& liberty_values, bool canonical_target_load) {
        rebuffer::CharacterizationResult result;
        std::string reason;
        try {
          const auto liberty = libertyFromDict(liberty_values);
          if (!rebuffer::characterizeBuffers(liberty, result, reason,
                                             canonical_target_load)) {
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
        return characterizationResult(result);
      },
      py::arg("liberty"), py::arg("canonical_target_load") = false);
  module.def(
      "build_forest",
      [](const py::dict& design_values, const py::dict& topology_values) {
        rebuffer::ForestResult forest;
        std::string reason;
        try {
          const auto design = designFromDict(design_values);
          const auto topology = topologyFromDict(topology_values);
          if (!rebuffer::build_forest(design, topology, forest, reason)) {
            py::dict result;
            result["status"] = "unsupported";
            py::list reasons;
            reasons.append(reason);
            result["unsupported_reasons"] = std::move(reasons);
            return result;
          }
        } catch (const std::exception& error) {
          py::dict result;
          result["status"] = "unsupported";
          py::list reasons;
          reasons.append(error.what());
          result["unsupported_reasons"] = std::move(reasons);
          return result;
        }

        py::dict result;
        result["status"] = "ok";
        result["num_nets"] = forest.num_nets;
        result["num_nodes"] = forest.num_nodes;
        result["num_pin_nodes"] = forest.num_pin_nodes;
        result["num_steiner_nodes"] = forest.num_steiner_nodes;
        result["num_edges"] = forest.num_edges;
        result["max_depth"] = forest.max_depth;
        result["max_fanout"] = forest.max_fanout;
        result["net_node_start"] = forest.net_node_start;
        result["net_node_ids"] = forest.net_node_ids;
        result["node_parent"] = forest.node_parent;
        result["node_depth"] = forest.node_depth;
        result["node_fanout"] = forest.node_fanout;
        result["net_summaries"] = netSummaries(forest);
        return result;
      },
      py::arg("design"),
      py::arg("topology"));
  py::class_<rebuffer::CurrentDriverTree,
             std::shared_ptr<rebuffer::CurrentDriverTree>>(
      module, "CurrentDriverTree")
      .def_property_readonly("root_net_id",
                             [](const rebuffer::CurrentDriverTree& tree) {
                               return tree.root_net_id;
                             })
      .def_property_readonly("driver_pin_id",
                             [](const rebuffer::CurrentDriverTree& tree) {
                               return tree.driver_pin_id;
                             })
      .def_property_readonly("topology_epoch",
                             [](const rebuffer::CurrentDriverTree& tree) {
                               return tree.topology_epoch;
                             })
      .def_property_readonly("tree_digest",
                             [](const rebuffer::CurrentDriverTree& tree) {
                               return tree.tree_digest;
                             });
  module.def(
      "import_current_driver_tree",
      [](const py::dict& design_values,
         const py::dict& topology_values,
         const int root_net_id,
         const bool stop_at_existing_buffers) {
        const auto design = designFromDict(design_values);
        const auto topology = topologyFromDict(topology_values);
        auto tree = std::make_shared<rebuffer::CurrentDriverTree>();
        std::string reason;
        if (!rebuffer::importCurrentDriverTree(
                design, topology, root_net_id, *tree, reason,
                stop_at_existing_buffers)) {
          throw py::value_error(reason);
        }
        return tree;
      },
      py::arg("design"),
      py::arg("topology"),
      py::arg("root_net_id"),
      py::arg("stop_at_existing_buffers") = false);
  module.def(
      "export_current_driver_tree",
      [](const std::shared_ptr<rebuffer::CurrentDriverTree>& tree) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        return rebuffer::currentDriverTreeToDict(*tree);
      },
      py::arg("tree"));
  module.def(
      "validate_current_driver_tree_export",
      [](const py::dict& values) {
        rebuffer::CurrentDriverTree tree;
        std::string reason;
        py::dict result;
        if (!rebuffer::currentDriverTreeFromDict(values, tree, reason)) {
          result["status"] = "unsupported";
          result["valid"] = false;
          result["reason"] = reason;
          return result;
        }
        result["status"] = "ok";
        result["valid"] = true;
        result["reason"] = "";
        result["tree_digest"] = tree.tree_digest;
        result["node_count"] = static_cast<int>(tree.nodes.size());
        result["edge_count"] = static_cast<int>(tree.edges.size());
        return result;
      },
      py::arg("values"));
  module.def(
      "enumerate_current_driver_tree_mutations",
      [](const py::dict& design_values,
         const py::dict& liberty_values,
         const std::shared_ptr<rebuffer::CurrentDriverTree>& tree) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        const auto design = designFromDict(design_values);
        const auto liberty = libertyFromDict(liberty_values);
        std::string reason;
        const auto mutations = rebuffer::enumerateCurrentDriverTreeMutations(
            *tree, design, liberty, reason);
        if (!reason.empty()) {
          throw py::value_error(reason);
        }
        py::list output;
        for (const auto& mutation : mutations) {
          output.append(rebuffer::driverTreeMutationToDict(mutation));
        }
        return output;
      },
      py::arg("design"),
      py::arg("liberty"),
      py::arg("tree"));
  module.def(
      "apply_current_driver_tree_mutation",
      [](const std::shared_ptr<rebuffer::CurrentDriverTree>& tree,
         const py::dict& mutation_values) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        const auto mutation
            = rebuffer::driverTreeMutationFromDict(mutation_values);
        auto target = std::make_shared<rebuffer::CurrentDriverTree>();
        std::string reason;
        if (!rebuffer::applyCurrentDriverTreeMutation(
                *tree, mutation, *target, reason)) {
          throw py::value_error(reason);
        }
        return rebuffer::currentDriverTreeToDict(*target);
      },
      py::arg("tree"),
      py::arg("mutation"));
  module.def(
      "enumerate_tree_area_recovery",
      [](const py::dict& design_values,
         const py::dict& liberty_values,
         const std::shared_ptr<rebuffer::CurrentDriverTree>& tree) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        const auto design = designFromDict(design_values);
        const auto liberty = libertyFromDict(liberty_values);
        std::string reason;
        const auto candidates = rebuffer::enumerateTreeAreaRecovery(
            *tree, design, liberty, reason);
        if (!reason.empty()) {
          throw py::value_error(reason);
        }
        py::list output;
        for (const auto& candidate : candidates) {
          py::dict value = rebuffer::driverTreeMutationToDict(candidate.mutation);
          value["area_saved"] = candidate.area_saved;
          value["buffer_count_delta"] = candidate.buffer_count_delta;
          value["recovery_kind"] = candidate.recovery_kind;
          output.append(std::move(value));
        }
        return output;
      },
      py::arg("design"),
      py::arg("liberty"),
      py::arg("tree"));
  module.def(
      "run_tree_area_recovery",
      &rebuffer::runTreeAreaRecovery,
      py::arg("transaction"),
      py::arg("design"),
      py::arg("liberty"),
      py::arg("physical_checkpoint"),
      py::arg("physical_apply"),
      py::arg("evaluate"),
      py::arg("physical_restore"),
      py::arg("config"));
  module.def(
      "apply_current_driver_tree_buffer_actions",
      [](const std::shared_ptr<rebuffer::CurrentDriverTree>& tree,
         const py::dict& liberty_values,
         const py::list& action_values) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        const auto liberty = libertyFromDict(liberty_values);
        std::vector<rebuffer::BufferAction> actions;
        for (const auto& item : action_values) {
          const py::dict values = item.cast<py::dict>();
          rebuffer::BufferAction action;
          action.net_id = values["net_id"].cast<int>();
          action.driver_pin = values["driver_pin"].cast<int>();
          action.buffer_cell_index = values["buffer_cell_index"].cast<int>();
          action.parent_node_id = values["parent_node_id"].cast<int>();
          action.child_node_id = values["child_node_id"].cast<int>();
          if (values.contains("source_parent_node_id")
              || values.contains("source_child_node_id")) {
            if (!values.contains("source_parent_node_id")
                || !values.contains("source_child_node_id")) {
              throw py::value_error(
                  "buffer action physical source identity is incomplete");
            }
            action.source_parent_node_id
                = values["source_parent_node_id"].cast<int>();
            action.source_child_node_id
                = values["source_child_node_id"].cast<int>();
          }
          action.x_dbu = values.contains("x_dbu") ? values["x_dbu"].cast<int>() : 0;
          action.y_dbu = values.contains("y_dbu") ? values["y_dbu"].cast<int>() : 0;
          if (values.contains("downstream_pins")) {
            action.downstream_pins = values["downstream_pins"].cast<std::vector<int>>();
          }
          actions.push_back(std::move(action));
        }
        auto target = std::make_shared<rebuffer::CurrentDriverTree>();
        std::string reason;
        if (!rebuffer::applyCurrentDriverTreeBufferActions(
                *tree, liberty, actions, *target, reason)) {
          throw py::value_error(reason);
        }
        return target;
      },
      py::arg("tree"),
      py::arg("liberty"),
      py::arg("actions"));
  py::class_<rebuffer::CurrentDriverTreeCheckpoint,
             std::shared_ptr<rebuffer::CurrentDriverTreeCheckpoint>>(
      module, "CurrentDriverTreeCheckpoint");
  py::class_<rebuffer::CurrentDriverTreeTransaction,
             std::shared_ptr<rebuffer::CurrentDriverTreeTransaction>>(
      module, "CurrentDriverTreeTransaction")
      .def("apply_mutation",
           &rebuffer::CurrentDriverTreeTransaction::applyMutation)
      .def("area_recovery_candidates",
           &rebuffer::CurrentDriverTreeTransaction::areaRecoveryCandidates,
           py::arg("design"),
           py::arg("liberty"))
      .def("apply_candidate_tree",
           &rebuffer::CurrentDriverTreeTransaction::applyCandidateTree,
           py::arg("candidate"),
           py::arg("source_tree_digest"))
      .def("active_tree",
           &rebuffer::CurrentDriverTreeTransaction::activeTree)
      .def("evaluate_candidate",
           &rebuffer::CurrentDriverTreeTransaction::evaluateCandidate)
      .def("accept_candidate",
           &rebuffer::CurrentDriverTreeTransaction::acceptCandidate)
      .def("checkpoint",
           &rebuffer::CurrentDriverTreeTransaction::checkpoint)
      .def("restore",
           &rebuffer::CurrentDriverTreeTransaction::restore)
      .def("summary",
           &rebuffer::CurrentDriverTreeTransaction::summary);
  module.def("create_current_driver_tree_transaction",
             &rebuffer::createCurrentDriverTreeTransaction,
             py::arg("tree"));
  rebuffer::registerTreeSearchBindings(module);
  rebuffer::registerTimingPlanningBindings(module);
  rebuffer::registerTimingLocalBindings(module);
  rebuffer::registerLegacyQueueBindings(module);
  module.def(
      "plan_repair_design",
      [](const py::dict& design_values,
         const py::dict& timing_values,
         const py::dict& steiner_values,
         const py::dict& liberty_values,
         const py::dict& characterization_values,
         const py::dict& config_values) {
        rebuffer::RepairDesignResult result;
        std::string reason;
        try {
          const auto design = designFromDict(design_values);
          const auto timing = timingFromDict(timing_values);
          const auto steiner = steinerFromDict(steiner_values);
          const auto liberty = libertyFromDict(liberty_values);
          const auto characterization
              = characterizationFromDict(characterization_values);
          const auto config = configFromDict(config_values);
          if (!rebuffer::planRepairDesign(design,
                                          timing,
                                          steiner,
                                          liberty,
                                          characterization,
                                          config,
                                          result,
                                          reason)) {
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
        py::list actions;
        for (const auto& action : result.actions) {
          actions.append(bufferActionDict(action));
        }
        output["actions"] = std::move(actions);
        py::list resizes;
        for (const auto& resize : result.resizes) {
          py::dict row;
          row["node_id"] = resize.node_id;
          row["target_cell_index"] = resize.target_cell_index;
          resizes.append(std::move(row));
        }
        output["resizes"] = std::move(resizes);
        output["driver_slew_violations"] = result.driver_slew_violations;
        output["driver_cap_violations"] = result.driver_cap_violations;
        output["load_slew_violations"] = result.load_slew_violations;
        output["load_cap_violations"] = result.load_cap_violations;
        output["repaired_nets"] = result.repaired_nets;
        output["inserted_count"] = static_cast<int>(result.actions.size());
        output["resized_count"] = static_cast<int>(result.resizes.size());
        return output;
      },
      py::arg("design"),
      py::arg("timing"),
      py::arg("steiner"),
      py::arg("liberty"),
      py::arg("characterization"),
      py::arg("config"));

}
