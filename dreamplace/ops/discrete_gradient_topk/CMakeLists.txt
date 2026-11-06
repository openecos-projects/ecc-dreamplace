set(TARGET_NAME quad_gradient_topk)

add_pytorch_extension(${TARGET_NAME}_cpp
  src/quad_gradient_topk.cpp)

install(TARGETS ${TARGET_NAME}_cpp
  DESTINATION dreamplace/ops/discrete_gradient_topk)
