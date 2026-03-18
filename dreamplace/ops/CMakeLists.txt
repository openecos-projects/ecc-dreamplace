add_subdirectory(utility)
add_subdirectory(place_io)
add_subdirectory(dct)
add_subdirectory(pin_pos)
add_subdirectory(move_boundary)
add_subdirectory(draw_place)
add_subdirectory(density_map)
add_subdirectory(density_overflow)
add_subdirectory(density_potential)
add_subdirectory(electric_potential)
add_subdirectory(hpwl)
add_subdirectory(rmst_wl)
add_subdirectory(weighted_average_wirelength)
add_subdirectory(logsumexp_wirelength)
add_subdirectory(fence_region)
# legalization operators
add_subdirectory(legality_check)
add_subdirectory(macro_legalize)
add_subdirectory(greedy_legalize)
add_subdirectory(abacus_legalize)
# detailed placement operators
add_subdirectory(global_swap)
add_subdirectory(independent_set_matching)
add_subdirectory(k_reorder)
# routing congestion operator
add_subdirectory(pin_utilization)
add_subdirectory(rudy)
add_subdirectory(pinrudy)
add_subdirectory(nctugr_binary)
add_subdirectory(irt_egr)
add_subdirectory(adjust_node_area)
add_subdirectory(timing_propagation)
add_subdirectory(rc_timing)
add_subdirectory(steiner_topo)

file(GLOB INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
install(
    FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops
    )

file(GLOB MACRO_OVERLAP_INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/macro_overlap/*.py")
install(
    FILES ${MACRO_OVERLAP_INSTALL_SRCS} DESTINATION dreamplace/ops/macro_overlap
    )

file(GLOB MACRO_REFINEMENT_INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/macro_refinement/*.py")
install(
    FILES ${MACRO_REFINEMENT_INSTALL_SRCS} DESTINATION dreamplace/ops/macro_refinement
    )
