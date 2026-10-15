add_subdirectory(utility)
add_subdirectory(placeio_common)
add_subdirectory(placeio_ecc)
add_subdirectory(placeio_ieda)
if(DREAMPLACE_USE_EXTERNAL_OPENROAD)
  add_subdirectory(placeio_openroad)
endif()
add_subdirectory(openroad_handoff)
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
add_subdirectory(m2_soft_legalize)
# detailed placement operators
add_subdirectory(global_swap)
add_subdirectory(independent_set_matching)
add_subdirectory(k_reorder)
add_subdirectory(m2_pa_refine)
# routing congestion operator
add_subdirectory(pin_utilization)
add_subdirectory(rudy)
add_subdirectory(pinrudy)
add_subdirectory(irt_egr)
add_subdirectory(adjust_node_area)
add_subdirectory(pin_weight_sum)
add_subdirectory(timing_net_weighting)
add_subdirectory(pin2pin_attraction)
#diff tdp 
add_subdirectory(timing_propagation)
add_subdirectory(rc_timing)
add_subdirectory(gr_parasitics)
add_subdirectory(steiner_topo)
add_subdirectory(net_subgraph_timing)
add_subdirectory(cell_modeling)
add_subdirectory(diff_guided_batch)
add_subdirectory(size_interpolated_pin)
add_subdirectory(routability)

# GPUGR wrapper and its built-in Xplace backend (the native extensions are
# built from thirdparty/xplace by thirdparty/build_xplace_gpugr.sh).
file(GLOB GPUGR_INSTALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/gpugr/*.py")
install(
    FILES ${GPUGR_INSTALL_SRCS} DESTINATION dreamplace/ops/gpugr
    )

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

foreach(PURE_PY_OP buffer_insertion discrete_gradient_topk gate_projection physical_action m2_legalize)
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${PURE_PY_OP}
        DESTINATION dreamplace/ops
        FILES_MATCHING PATTERN "*.py"
        )
endforeach()
