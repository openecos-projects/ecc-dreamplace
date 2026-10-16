set(OP_NAME size_interpolated_pin)

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_pytorch_extension(${OP_NAME}_cpp
  src/size_interpolated_pin.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES utility_cxx
  )

install(TARGETS
  ${OP_NAME}_cpp
  DESTINATION dreamplace/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)
add_pytorch_extension(${OP_NAME}_cuda
  src/size_interpolated_pin_cuda.cpp
  src/size_interpolated_pin_cuda_kernel.cu
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES utility_cxx
  )

install(TARGETS
  ${OP_NAME}_cuda
  DESTINATION dreamplace/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
