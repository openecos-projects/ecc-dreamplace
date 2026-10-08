#project(thirdparty)

add_subdirectory(flute)

add_subdirectory(Limbo)
add_subdirectory(munkres-cpp)
if (DREAMPLACE_ENABLE_OPENTIMER_TIMING)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenTimer/CMakeLists.txt")
    add_subdirectory(OpenTimer)
  else()
    message(FATAL_ERROR "DREAMPLACE_ENABLE_OPENTIMER_TIMING=ON requires thirdparty/OpenTimer")
  endif()
endif()
