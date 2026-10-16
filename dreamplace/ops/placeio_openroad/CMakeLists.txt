set(OP_NAME placeio_openroad)
set(TARGET_NAME ${OP_NAME})
set(PLACEIO_OPENROAD_CXX_ABI "1" CACHE STRING
  "Target-specific _GLIBCXX_USE_CXX11_ABI value for placeio_openroad_cpp")

if (DREAMPLACE_USE_EXTERNAL_OPENROAD)
  set(OPENROAD_EXTERNAL_SOURCE_DIR "" CACHE PATH
    "OpenROAD source tree used to derive include directories in external-link mode")
  set(OPENROAD_EXTERNAL_BUILD_DIR "" CACHE PATH
    "OpenROAD standalone build tree used to derive library paths in external-link mode")
  set(OPENROAD_EXTERNAL_INSTALL_PREFIX "" CACHE PATH
    "OpenROAD standalone install prefix used to derive headers/libraries in external-link mode")
  set(OPENROAD_EXTERNAL_INCLUDE_DIRS "" CACHE STRING
    "Explicit semicolon-separated include directories for an externally built OpenROAD")
  set(OPENROAD_EXTERNAL_LINK_LIBS "" CACHE STRING
    "Explicit semicolon-separated libraries/targets for an externally built OpenROAD")

  if (OPENROAD_EXTERNAL_INSTALL_PREFIX STREQUAL ""
      AND NOT OPENROAD_EXTERNAL_SOURCE_DIR STREQUAL ""
      AND EXISTS ${OPENROAD_EXTERNAL_SOURCE_DIR}/install-shared-abi1)
    set(OPENROAD_EXTERNAL_INSTALL_PREFIX ${OPENROAD_EXTERNAL_SOURCE_DIR}/install-shared-abi1)
  endif()

  if (OPENROAD_EXTERNAL_BUILD_DIR STREQUAL ""
      AND NOT OPENROAD_EXTERNAL_SOURCE_DIR STREQUAL ""
      AND EXISTS ${OPENROAD_EXTERNAL_SOURCE_DIR}/build-shared-abi1)
    set(OPENROAD_EXTERNAL_BUILD_DIR ${OPENROAD_EXTERNAL_SOURCE_DIR}/build-shared-abi1)
  endif()

  if (OPENROAD_EXTERNAL_INCLUDE_DIRS STREQUAL "")
    if (NOT OPENROAD_EXTERNAL_INSTALL_PREFIX STREQUAL "")
      set(OPENROAD_INCLUDE_DIRS
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/include
      )
    elseif (NOT OPENROAD_EXTERNAL_SOURCE_DIR STREQUAL "")
      set(OPENROAD_INCLUDE_DIRS
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/utl/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/dbSta/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/est/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/grt/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/ifp/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/odb/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/sta/include
        ${OPENROAD_EXTERNAL_SOURCE_DIR}/src/stt/include
      )
      if (NOT OPENROAD_EXTERNAL_BUILD_DIR STREQUAL "")
        list(APPEND OPENROAD_INCLUDE_DIRS
          ${OPENROAD_EXTERNAL_BUILD_DIR}/src
          ${OPENROAD_EXTERNAL_BUILD_DIR}/src/sta/include
          ${OPENROAD_EXTERNAL_BUILD_DIR}/src/sta/include/sta
        )
      endif()
    endif()
  else()
    set(OPENROAD_INCLUDE_DIRS ${OPENROAD_EXTERNAL_INCLUDE_DIRS})
  endif()

  if (OPENROAD_EXTERNAL_LINK_LIBS STREQUAL "")
    if (NOT OPENROAD_EXTERNAL_INSTALL_PREFIX STREQUAL "")
      if (EXISTS ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libopenroad.so)
        set(OPENROAD_LINK_LIBS
          ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libopenroad.so
        )
      else()
        set(OPENROAD_LINK_LIBS
        -Wl,--start-group
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libopenroad.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/ord.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/ifp.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/pad.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/gpl.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libgpl_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/dpl.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdpl_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/exa.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libexa_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/fin.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/ram.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/rsz.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/librsz_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/ppl.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/stt.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libstt_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/dbSta.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdbSta_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libOpenSTA.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdb.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdef.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdefzlib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdefin.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdefout.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/liblef.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/liblefzlib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/liblefin.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libzutil.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/rcx.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/librcx_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/rmp.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/librmp_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/cgt.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/cts.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libcts_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libcut.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/grt.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libgrt_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libFastRoute4.1.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libCUGR.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/tap.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/gui.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/drt.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdrt_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/dst.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libdst_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/psm.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/ant.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libant_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libabc.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/upf.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libupf_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/utl.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libutl_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/pdn.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/dft.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/mpl.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libmpl_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/par.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libpar_lib.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/est.a
        ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib/libest_lib.a
        -Wl,--end-group
        )
      endif()
    elseif (NOT OPENROAD_EXTERNAL_BUILD_DIR STREQUAL "")
      if (EXISTS ${OPENROAD_EXTERNAL_BUILD_DIR}/src/libopenroad.so)
        set(OPENROAD_LINK_LIBS
          ${OPENROAD_EXTERNAL_BUILD_DIR}/src/libopenroad.so
        )
      else()
        set(OPENROAD_LINK_LIBS
        -Wl,--start-group
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/libopenroad.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/ord.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/ifp/src/ifp.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/pad/pad.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/gpl/gpl.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/gpl/libgpl_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dpl/dpl.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dpl/libdpl_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/exa/exa.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/exa/libexa_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/fin/fin.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/ram/ram.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/rsz/src/rsz.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/rsz/src/librsz_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/ppl/ppl.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/stt/stt.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/stt/libstt_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dbSta/src/dbSta.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dbSta/src/libdbSta_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/sta/libOpenSTA.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/db/libdb.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/def/libdef.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/def/libdefzlib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/defin/libdefin.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/defout/libdefout.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/lef/liblef.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/lef/liblefzlib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/lefin/liblefin.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/odb/src/zutil/libzutil.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/rcx/src/rcx.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/rcx/src/librcx_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/rmp/src/rmp.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/rmp/src/librmp_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/cgt/src/cgt.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/cts/src/cts.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/cts/src/libcts_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/cut/src/libcut.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/grt/grt.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/grt/libgrt_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/grt/src/fastroute/libFastRoute4.1.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/grt/src/cugr/libCUGR.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/tap/src/tap.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/gui/gui.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/drt/drt.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/drt/libdrt_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dst/dst.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dst/libdst_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/psm/src/psm.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/ant/src/ant.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/ant/src/libant_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/third-party/abc/libabc.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/upf/src/upf.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/upf/src/libupf_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/utl/utl.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/utl/libutl_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/pdn/src/pdn.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/dft/dft.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/mpl/mpl.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/mpl/libmpl_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/par/par.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/par/libpar_lib.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/est/src/est.a
        ${OPENROAD_EXTERNAL_BUILD_DIR}/src/est/src/libest_lib.a
        -Wl,--end-group
        )
      endif()
    else()
      message(FATAL_ERROR
        "DREAMPLACE_USE_EXTERNAL_OPENROAD=ON requires either OPENROAD_EXTERNAL_LINK_LIBS "
        "or OPENROAD_EXTERNAL_BUILD_DIR or OPENROAD_EXTERNAL_INSTALL_PREFIX")
    endif()
  else()
    set(OPENROAD_LINK_LIBS ${OPENROAD_EXTERNAL_LINK_LIBS})
  endif()

  find_package(Boost REQUIRED COMPONENTS iostreams serialization thread)
  find_package(ZLIB REQUIRED)
  find_package(spdlog REQUIRED)
  find_package(Threads REQUIRED)
  find_package(Qt5 QUIET COMPONENTS Core Widgets Charts)
  find_package(TCL)
  find_package(ortools REQUIRED)
  find_library(TCL_READLINE_LIBRARY NAMES tclreadline tclreadline-2.3.8
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
  set(OPENROAD_EXTERNAL_CUDD_LIBRARY "" CACHE FILEPATH
    "Explicit CUDD library for OpenROAD external-link mode")
  find_library(CUDD_LIBRARY NAMES cudd
    PATHS
      /usr/local/lib
      /usr/lib
      /usr/lib64)
  if (NOT OPENROAD_EXTERNAL_CUDD_LIBRARY STREQUAL "")
    set(CUDD_LIBRARY ${OPENROAD_EXTERNAL_CUDD_LIBRARY})
  endif()
  list(APPEND OPENROAD_LINK_LIBS
    Boost::iostreams
    Boost::serialization
    Boost::thread
    ZLIB::ZLIB
    spdlog::spdlog
    Threads::Threads
  )
  if (CUDD_LIBRARY)
    list(APPEND OPENROAD_LINK_LIBS ${CUDD_LIBRARY})
  endif()
  if (TARGET fmt::fmt)
    list(APPEND OPENROAD_LINK_LIBS fmt::fmt)
  elseif (TARGET fmt::fmt-header-only)
    list(APPEND OPENROAD_LINK_LIBS fmt::fmt-header-only)
  endif()
  if (TARGET Qt5::Core)
    list(APPEND OPENROAD_LINK_LIBS Qt5::Core)
  endif()
  if (TARGET Qt5::Widgets)
    list(APPEND OPENROAD_LINK_LIBS Qt5::Widgets)
  endif()
  if (TARGET Qt5::Charts)
    list(APPEND OPENROAD_LINK_LIBS Qt5::Charts)
  endif()
  if (TCL_READLINE_LIBRARY)
    list(APPEND OPENROAD_LINK_LIBS ${TCL_READLINE_LIBRARY})
  endif()
  if (TCL_LIBRARY)
    list(APPEND OPENROAD_LINK_LIBS ${TCL_LIBRARY})
  endif()
  if (TARGET ortools::ortools)
    list(APPEND OPENROAD_LINK_LIBS ortools::ortools)
  endif()

  if (OPENROAD_INCLUDE_DIRS STREQUAL "")
    message(FATAL_ERROR
      "DREAMPLACE_USE_EXTERNAL_OPENROAD=ON requires either OPENROAD_EXTERNAL_INCLUDE_DIRS "
      "or OPENROAD_EXTERNAL_SOURCE_DIR/OPENROAD_EXTERNAL_INSTALL_PREFIX")
  endif()
endif()

list(REMOVE_DUPLICATES OPENROAD_INCLUDE_DIRS)
list(REMOVE_DUPLICATES OPENROAD_LINK_LIBS)

pybind11_add_module(${TARGET_NAME}_cpp MODULE
  src/openroad_place_io.cpp
  src/openroad_diff_guided_batch_bridge.cpp
  src/openroad_place_io_impl.cpp
  src/openroad_pyplacedb_export_impl.cpp
  src/openroad_place_io_bridge.cpp
  src/openroad_runtime.cpp
  ../diff_guided_batch/cpp/bridge_action_io.cpp
  ../diff_guided_batch/cpp/config.cpp
  ../diff_guided_batch/cpp/objective.cpp
)
set_property(SOURCE src/openroad_pyplacedb_export_impl.cpp APPEND PROPERTY OBJECT_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/openroad_sizing_cell_classification.h)
target_include_directories(${TARGET_NAME}_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${OPENROAD_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME}_cpp PRIVATE ${OPENROAD_LINK_LIBS})
target_compile_definitions(${TARGET_NAME}_cpp PRIVATE
  TORCH_EXTENSION_NAME=${TARGET_NAME}_cpp
  ENABLE_CUDA=0)
target_compile_options(${TARGET_NAME}_cpp PRIVATE
  -U_GLIBCXX_USE_CXX11_ABI
  -D_GLIBCXX_USE_CXX11_ABI=${PLACEIO_OPENROAD_CXX_ABI})

set(PLACEIO_OPENROAD_RPATH_DIRS "")
if (DREAMPLACE_USE_EXTERNAL_OPENROAD)
  if (NOT OPENROAD_EXTERNAL_BUILD_DIR STREQUAL "")
    list(APPEND PLACEIO_OPENROAD_RPATH_DIRS
      ${OPENROAD_EXTERNAL_BUILD_DIR}/src)
  endif()
  if (NOT OPENROAD_EXTERNAL_INSTALL_PREFIX STREQUAL "")
    list(APPEND PLACEIO_OPENROAD_RPATH_DIRS
      ${OPENROAD_EXTERNAL_INSTALL_PREFIX}/lib)
  endif()
endif()
list(REMOVE_DUPLICATES PLACEIO_OPENROAD_RPATH_DIRS)

if (PLACEIO_OPENROAD_RPATH_DIRS)
  set_target_properties(${TARGET_NAME}_cpp PROPERTIES
    BUILD_RPATH "${PLACEIO_OPENROAD_RPATH_DIRS}"
    INSTALL_RPATH "${PLACEIO_OPENROAD_RPATH_DIRS}"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

install(TARGETS
  ${TARGET_NAME}_cpp
  DESTINATION dreamplace/ops/${OP_NAME})
file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION dreamplace/ops/${OP_NAME})
