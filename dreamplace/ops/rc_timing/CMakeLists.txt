set(OP_NAME rc_timing)

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${FLUTE_INCLUDE_DIRS}
  ${LIMBO_SOURCE_DIR}
)

set(LINK_LIBS 
  utility_cxx
  ${FLUTE_LINK_DIRS}/libflute.a
  OpenMP::OpenMP_CXX
)

add_pytorch_extension(${TARGET_NAME}_cpp 
  src/beta_op.cpp
  src/delay_op.cpp
  src/ldelay_op.cpp
  src/load_op.cpp
  src/binds.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS}
)

install(TARGETS 
  ${TARGET_NAME}_cpp 
  DESTINATION dreamplace/ops/${OP_NAME}
)

# if(TORCH_ENABLE_CUDA)
#   # 第一个 CUDA 扩展目标
#   add_pytorch_extension(${TARGET_NAME}_cuda 
#     src/steiner_topo_cuda.cpp
#     src/steiner_topo_cuda_kernel.cu
#     EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
#     EXTRA_LINK_LIBRARIES 
#       ${LINK_LIBS}
#       CUDA::cudart  # 显式链接 CUDA 运行时
#   )

#   # 第二个 CUDA 扩展目标（segment 实现）
#   add_pytorch_extension(${TARGET_NAME}_cuda_segment
#     src/steiner_topo_cuda_segment.cpp
#     src/steiner_topo_cuda_segment_kernel.cu
#     EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
#     EXTRA_LINK_LIBRARIES 
#       ${LINK_LIBS}
#       CUDA::cudart  # 显式链接 CUDA 运行时
#   )

#   # 安装两个 CUDA 目标
#   install(TARGETS 
#     ${TARGET_NAME}_cuda 
#     ${TARGET_NAME}_cuda_segment
#     DESTINATION dreamplace/ops/${OP_NAME}
#   )
# endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})