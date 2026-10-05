set(OP_NAME routability)

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set(LINK_LIBS
  utility_cxx
  OpenMP::OpenMP_CXX
)

add_pytorch_extension(same_net_topo_scoring_cpp
  src/same_net_topo_scoring.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
)

add_pytorch_extension(segment_compaction_cpp
  src/segment_compaction.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
)

add_pytorch_extension(adaptive_padding_cpp
  src/adaptive_padding.cpp
)

if(TORCH_ENABLE_CUDA)
add_pytorch_extension(segment_compaction_cuda
  src/segment_compaction_cuda.cpp
  src/segment_compaction_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
)
endif(TORCH_ENABLE_CUDA)

install(TARGETS
  same_net_topo_scoring_cpp
  segment_compaction_cpp
  adaptive_padding_cpp
  DESTINATION dreamplace/ops/${OP_NAME}
)

if(TORCH_ENABLE_CUDA)
install(TARGETS
  segment_compaction_cuda
  DESTINATION dreamplace/ops/${OP_NAME}
)
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})

file(GLOB INSTALL_MISC ${CMAKE_CURRENT_SOURCE_DIR}/*.lef)
install(FILES ${INSTALL_MISC} DESTINATION dreamplace/ops/${OP_NAME})
