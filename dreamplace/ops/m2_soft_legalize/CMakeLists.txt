set(OP_NAME m2_soft_legalize)

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

add_pytorch_extension(${OP_NAME}_cpp
  src/m2_soft_legalize.cpp
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES utility_cxx OpenMP::OpenMP_CXX)

install(TARGETS ${OP_NAME}_cpp DESTINATION dreamplace/ops/${OP_NAME})

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
