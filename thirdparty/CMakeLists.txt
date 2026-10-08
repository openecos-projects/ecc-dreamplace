#project(thirdparty)

add_subdirectory(flute)

set(OPENBLAS OFF CACHE BOOL "Turn off OPENBLAS for Limbo")
set(INSTALL_LIMBO OFF CACHE BOOL "Turn off installation for Limbo")
set(GENERATE_DOCS OFF CACHE BOOL "Turn off documentation for Limbo")
set(ENABLE_TEST OFF CACHE BOOL "Turn off test for Limbo")
set(ALGORITHMS OFF CACHE BOOL "Turn off Limbo algorithms; not part of the vendored subset")
add_subdirectory(Limbo)
add_subdirectory(munkres-cpp)
if (DREAMPLACE_ENABLE_OPENTIMER_TIMING)
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenTimer/CMakeLists.txt")
    add_subdirectory(OpenTimer)
  else()
    message(FATAL_ERROR "DREAMPLACE_ENABLE_OPENTIMER_TIMING=ON requires thirdparty/OpenTimer")
  endif()
endif()

install(
    DIRECTORY NCTUgr.ICCAD2012 DESTINATION thirdparty
    )
