set(OP_NAME rebuffer)
find_package(ZLIB REQUIRED)

add_pytorch_extension(${OP_NAME}_cpp
  src/rebuffer_design.cc
  src/rebuffer_forest.cc
  src/rebuffer_incremental_topology.cc
  src/rebuffer_topology_partitions.cc
  src/rebuffer_topology_history.cc
  src/rebuffer_topology_remove.cc
  src/rebuffer_incremental_steiner.cc
  src/rebuffer_incremental_timing.cc
  src/rebuffer_incremental_carrier.cc
  src/rebuffer_incremental_patch.cc
  src/rebuffer_incremental_rc.cc
  src/rebuffer_incremental_rc_python.cc
  src/rebuffer_timing_graph.cc
  src/rebuffer_timing_state.cc
  src/rebuffer_timing_state_view.cc
  src/rebuffer_timing_state_undo.cc
  src/rebuffer_timing_program.cc
  src/rebuffer_timing_graph_propagate.cc
  src/rebuffer_timing_observation.cc
  src/rebuffer_timing_electrical.cc
  src/rebuffer_timing_planning.cc
  src/rebuffer_timing_planning_tree.cc
  src/rebuffer_timing_planning_queue.cc
  src/rebuffer_timing_paths.cc
  src/rebuffer_timing_queue.cc
  src/rebuffer_timing_queue_projection.cc
  src/rebuffer_timing_event_schedule.cc
  src/rebuffer_timing_events.cc
  src/rebuffer_timing_event_math.cc
  src/rebuffer_timing_mutation.cc
  src/rebuffer_timing_schedule.cc
  src/rebuffer_timing_local_index.cc
  src/rebuffer_timing_local_events.cc
  src/rebuffer_timing_index_oracle.cc
  src/rebuffer_timing_buffer_state.cc
  src/rebuffer_timing_buffer_rc.cc
  src/rebuffer_timing_graph_buffer.cc
  src/rebuffer_timing_compact.cc
  src/rebuffer_timing_session.cc
  src/rebuffer_driver_tree.cc
  src/rebuffer_driver_tree_links.cc
  src/rebuffer_tree_annotation.cc
  src/rebuffer_tree_electrical.cc
  src/rebuffer_wire_candidates.cc
  src/rebuffer_tree_search.cc
  src/rebuffer_tree_actions.cc
  src/rebuffer_tree_bindings.cc
  src/rebuffer_timing_policy.cc
  src/rebuffer_candidate_frontier.cc
  src/rebuffer_verified_commit.cc
  src/rebuffer_electrical_observation.cc
  src/rebuffer_tree_dp.cc
  src/rebuffer_tree_recovery.cc
  src/rebuffer_timing_payload.cc
  src/rebuffer_carrier.cc
  src/rebuffer_carrier_graph.cc
  src/rebuffer_overlay_carrier.cc
  src/rebuffer_overlay_actions.cc
  src/rebuffer_liberty.cc
  src/rebuffer_library_metadata.cc
  src/rebuffer_cell_selection.cc
  src/rebuffer_repair.cc
  src/rebuffer_repair_buffer.cc
  src/rebuffer_net_repair.cc
  src/rebuffer_path_policy.cc
  src/rebuffer_native_queue.cc
  src/rebuffer_queue_snapshot.cc
  src/rebuffer_queue_state.cc
  src/rebuffer_native_planner.cc
  src/rebuffer_python_views.cc
  src/rebuffer_bindings.cc
  EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../timing_propagation/src
)
target_link_libraries(${OP_NAME}_cpp PRIVATE ZLIB::ZLIB)

add_pytorch_extension(${OP_NAME}_steiner_cpp
  src/openroad_pd.cc
  src/openroad_flute.cc
  src/openroad_check_tree.cc
  src/openroad_steiner_bindings.cc
  EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(${OP_NAME}_steiner_cpp PRIVATE cxx_std_20)

install(TARGETS ${OP_NAME}_steiner_cpp DESTINATION dreamplace/ops/${OP_NAME})
install(TARGETS ${OP_NAME}_cpp DESTINATION dreamplace/ops/${OP_NAME})
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
  DESTINATION dreamplace/ops/${OP_NAME}
)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
