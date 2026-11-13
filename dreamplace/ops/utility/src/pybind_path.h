/**
 * @file   pybind_path.h
 * @brief  pybind11 caster for filesystem path arguments.
 *
 * Filesystem paths cross the Python/C++ boundary as
 * std::filesystem::path. The caster in the vendored pybind11
 * stl/filesystem.h loads arguments through PyOS_FSPath, so plain str
 * (and bytes) keep working while pathlib.Path and any custom
 * os.PathLike object are accepted as well. None and non-path values
 * are still rejected with TypeError.
 *
 * The caster is not pulled in by pybind11/stl.h, so translation units that bind path
 * arguments must include this header.
 */
#ifndef DREAMPLACE_UTILITY_PYBIND_PATH_H
#define DREAMPLACE_UTILITY_PYBIND_PATH_H

#include <filesystem>
#include <string>

#include <pybind11/stl/filesystem.h>

#endif
