set(OP_NAME net_subgraph_timing)
set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_pytorch_extension(${TARGET_NAME}_cpp
  src/net_subgraph_timing.cpp
  src/segment_count_packer.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES utility_cxx OpenMP::OpenMP_CXX
  )

install(TARGETS
  ${TARGET_NAME}_cpp
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)
add_pytorch_extension(${TARGET_NAME}_cuda
  src/segment_transfer_cuda.cpp
  src/segment_transfer_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES utility_cxx
  )

install(TARGETS
  ${TARGET_NAME}_cuda
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})

# Keep the pure-Python segment-transfer package, including its nested data and
# model modules, in the installed wheel alongside the native timing operator.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/segment_transfer
  DESTINATION dreamplace/ops/${OP_NAME}
  FILES_MATCHING PATTERN "*.py"
)
