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

install(TARGETS
  same_net_topo_scoring_cpp
  DESTINATION dreamplace/ops/${OP_NAME}
)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})

file(GLOB INSTALL_MISC ${CMAKE_CURRENT_SOURCE_DIR}/*.lef)
install(FILES ${INSTALL_MISC} DESTINATION dreamplace/ops/${OP_NAME})
