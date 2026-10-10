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

add_pytorch_extension(lut_entry_2d_cpp
  src/lut_entry_2d.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS 
  ${TARGET_NAME}_cpp 
  lut_entry_2d_cpp
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)
add_pytorch_extension(lut_entry_2d_cuda
  src/lut_entry_2d_cuda.cpp
  src/lut_entry_2d_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(TARGETS
  lut_entry_2d_cuda
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
