cmake_minimum_required(VERSION 3.7)

project(Limbo LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(INSTALL_LIMBO OFF)

find_package(BISON 3.3 REQUIRED)
find_package(FLEX REQUIRED)

# Solvers.h still inherits Limbo's LP parser database. Keep this small parser
# until the model definitions are separated from file parsing.
add_subdirectory(limbo/parsers/lp/bison)

set(LEMON_ENABLE_GLPK OFF CACHE BOOL "Disable unused GLPK backend" FORCE)
set(LEMON_ENABLE_ILOG OFF CACHE BOOL "Disable unused ILOG backend" FORCE)
set(LEMON_ENABLE_COIN OFF CACHE BOOL "Disable unused COIN backend" FORCE)
set(LEMON_ENABLE_SOPLEX OFF CACHE BOOL "Disable unused SoPlex backend" FORCE)
add_subdirectory(limbo/thirdparty/lemon)
