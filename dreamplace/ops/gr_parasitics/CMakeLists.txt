set(OP_NAME gr_parasitics)
add_pytorch_extension(${OP_NAME}_cpp src/route_tree.cpp)
# The vendored legacy pybind11 is 2.11 and produces invalid strides with
# NumPy 2. Use the pinned Torch package's NumPy-2-compatible headers here.
target_include_directories(${OP_NAME}_cpp BEFORE PRIVATE ${TORCH_INSTALL_PREFIX}/include)
install(TARGETS ${OP_NAME}_cpp DESTINATION dreamplace/ops/${OP_NAME})
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
