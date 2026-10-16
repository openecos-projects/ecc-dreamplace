set(OP_NAME diff_guided_batch)
set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  )

pybind11_add_module(${TARGET_NAME}_cpp MODULE
  cpp/bindings.cpp
  cpp/conflict_precompute.cpp
  cpp/loop_controller.cpp
  cpp/opensta_trial.cpp
  cpp/selector.cpp
  cpp/transaction.cpp
)

target_include_directories(${TARGET_NAME}_cpp PRIVATE ${INCLUDE_DIRS})
target_compile_definitions(${TARGET_NAME}_cpp PRIVATE
  TORCH_EXTENSION_NAME=${TARGET_NAME}_cpp
  ENABLE_CUDA=0)

install(TARGETS ${TARGET_NAME}_cpp DESTINATION dreamplace/ops/${OP_NAME})
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
