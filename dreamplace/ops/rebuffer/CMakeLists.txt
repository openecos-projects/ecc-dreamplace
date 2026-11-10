set(OP_NAME rebuffer)
find_package(ZLIB REQUIRED)

add_pytorch_extension(${OP_NAME}_cpp
  # topology & carrier
  src/topology/rebuffer_forest.cc
  src/topology/rebuffer_incremental_topology.cc
  src/topology/rebuffer_topology_partitions.cc
  src/topology/rebuffer_topology_history.cc
  src/topology/rebuffer_topology_remove.cc
  src/topology/rebuffer_incremental_steiner.cc
  src/topology/rebuffer_incremental_carrier.cc
  src/topology/rebuffer_incremental_patch.cc
  src/topology/rebuffer_incremental_rc.cc
  src/topology/rebuffer_incremental_rc_python.cc
  src/topology/rebuffer_carrier.cc
  src/topology/rebuffer_carrier_graph.cc
  src/topology/rebuffer_overlay_carrier.cc
  src/topology/rebuffer_carrier_projection.cc
  src/topology/rebuffer_overlay_inputs.cc
  src/topology/rebuffer_overlay_actions.cc
  # incremental STA
  src/incremental_sta/rebuffer_incremental_timing.cc
  src/incremental_sta/rebuffer_timing_graph.cc
  src/incremental_sta/rebuffer_timing_state.cc
  src/incremental_sta/rebuffer_timing_state_view.cc
  src/incremental_sta/rebuffer_timing_state_undo.cc
  src/incremental_sta/rebuffer_timing_program.cc
  src/incremental_sta/rebuffer_timing_graph_propagate.cc
  src/incremental_sta/rebuffer_timing_observation.cc
  src/incremental_sta/rebuffer_timing_electrical.cc
  src/incremental_sta/rebuffer_timing_local.cc
  src/incremental_sta/rebuffer_timing_local_geometry.cc
  src/incremental_sta/rebuffer_timing_local_views.cc
  src/incremental_sta/rebuffer_timing_paths.cc
  src/incremental_sta/rebuffer_timing_queue.cc
  src/incremental_sta/rebuffer_timing_endpoints.cc
  src/incremental_sta/rebuffer_timing_queue_projection.cc
  src/incremental_sta/rebuffer_timing_event_schedule.cc
  src/incremental_sta/rebuffer_timing_events.cc
  src/incremental_sta/rebuffer_timing_event_math.cc
  src/incremental_sta/rebuffer_timing_mutation.cc
  src/incremental_sta/rebuffer_timing_schedule.cc
  src/incremental_sta/rebuffer_timing_local_index.cc
  src/incremental_sta/rebuffer_timing_local_events.cc
  src/incremental_sta/rebuffer_timing_index_oracle.cc
  src/incremental_sta/rebuffer_timing_buffer_state.cc
  src/incremental_sta/rebuffer_timing_buffer_rc.cc
  src/incremental_sta/rebuffer_timing_graph_buffer.cc
  src/incremental_sta/rebuffer_timing_compact.cc
  src/incremental_sta/rebuffer_timing_session.cc
  src/incremental_sta/rebuffer_electrical_observation.cc
  src/incremental_sta/rebuffer_timing_payload.cc
  # tree DP & VGDP search
  src/tree_dp/rebuffer_driver_tree.cc
  src/tree_dp/rebuffer_driver_tree_links.cc
  src/tree_dp/rebuffer_tree_annotation.cc
  src/tree_dp/rebuffer_tree_electrical.cc
  src/tree_dp/rebuffer_wire_candidates.cc
  src/tree_dp/rebuffer_tree_search.cc
  src/tree_dp/rebuffer_tree_actions.cc
  src/tree_dp/rebuffer_tree_bindings.cc
  src/tree_dp/rebuffer_tree_dp.cc
  src/tree_dp/rebuffer_tree_recovery.cc
  # repair planner & bindings
  src/repair/rebuffer_design.cc
  src/repair/rebuffer_timing_planning.cc
  src/repair/rebuffer_timing_planning_identity.cc
  src/repair/rebuffer_timing_planning_tree.cc
  src/repair/rebuffer_timing_planning_queue.cc
  src/repair/rebuffer_legacy_queue.cc
  src/repair/rebuffer_timing_driver_resize.cc
  src/repair/rebuffer_timing_policy.cc
  src/repair/rebuffer_candidate_frontier.cc
  src/repair/rebuffer_verified_commit.cc
  src/repair/rebuffer_liberty.cc
  src/repair/rebuffer_library_metadata.cc
  src/repair/rebuffer_cell_selection.cc
  src/repair/rebuffer_repair.cc
  src/repair/rebuffer_repair_buffer.cc
  src/repair/rebuffer_net_repair.cc
  src/repair/rebuffer_path_policy.cc
  src/repair/rebuffer_native_queue.cc
  src/repair/rebuffer_queue_snapshot.cc
  src/repair/rebuffer_queue_state.cc
  src/repair/rebuffer_native_planner.cc
  src/repair/rebuffer_python_views.cc
  src/repair/rebuffer_bindings.cc
  EXTRA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/incremental_sta
    ${CMAKE_CURRENT_SOURCE_DIR}/src/topology
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tree_dp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/repair
    ${CMAKE_CURRENT_SOURCE_DIR}/../timing_propagation/src
)
target_link_libraries(${OP_NAME}_cpp PRIVATE ZLIB::ZLIB)

add_pytorch_extension(${OP_NAME}_steiner_cpp
  src/topology/openroad_pd.cc
  src/topology/openroad_flute.cc
  src/topology/openroad_check_tree.cc
  src/topology/openroad_steiner_bindings.cc
  EXTRA_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/topology
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
