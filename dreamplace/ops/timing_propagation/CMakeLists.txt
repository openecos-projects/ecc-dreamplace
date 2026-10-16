set(OP_NAME timing_propagation)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${FLUTE_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  )

set(LINK_LIBS utility_cxx
  OpenMP::OpenMP_CXX
  flute
  stdc++fs
  )

file(GLOB CPP_SOURCES
  src/timing_propagation_paths.cpp
  src/timing_propagation_pybind.cpp
  )
add_pytorch_extension(${TARGET_NAME}_cpp 
  ${CPP_SOURCES}
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

set_source_files_properties(
  src/cell_aat_cpu.cpp
  src/net_aat_cpu.cpp
  src/timing_lanes_cpu.cpp
  src/lut_entry_indexed_cpu.cpp
  src/buffer_interpolate_cpu.cpp
  src/timing_scatter_cpu.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

add_pytorch_extension(lut_entry_2d_cpp
  src/lut_entry_2d.cpp
  src/cell_aat_cpu.cpp
  src/net_aat_cpu.cpp
  src/timing_lanes_cpu.cpp
  src/lut_entry_indexed_cpu.cpp
  src/buffer_interpolate_cpu.cpp
  src/timing_scatter_cpu.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  lut_entry_2d_cpp
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)
# Match Torch's separate square/add and multiply/add operations.
set_source_files_properties(src/net_aat_cuda_kernel.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(src/buffer_interpolate_cuda_kernel.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
set_source_files_properties(src/timing_lanes_cuda_kernel.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
add_pytorch_extension(lut_entry_2d_cuda
  src/lut_entry_2d_cuda.cpp
  src/lut_entry_2d_cuda_kernel.cu
  src/lut_entry_indexed_cuda.cpp
  src/lut_entry_indexed_cuda_kernel.cu
  src/smooth_scatter_cuda.cpp
  src/smooth_scatter_cuda_kernel.cu
  src/hard_scatter_cuda.cpp
  src/hard_scatter_cuda_kernel.cu
  src/cell_aat_cuda.cpp
  src/cell_aat_cuda_kernel.cu
  src/net_aat_cuda.cpp
  src/net_aat_cuda_kernel.cu
  src/timing_lanes_cuda.cpp
  src/timing_lanes_cuda_kernel.cu
  src/buffer_interpolate_cuda.cpp
  src/buffer_interpolate_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS
  lut_entry_2d_cuda
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
