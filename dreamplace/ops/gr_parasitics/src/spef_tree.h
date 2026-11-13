#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>

namespace gr_parasitics {

using IndexArray = pybind11::array_t<int32_t, pybind11::array::c_style>;

pybind11::dict prepareSpefTree(const std::string& spefPath,
                               const pybind11::dict& pinNames,
                               const pybind11::dict& netNames,
                               IndexArray pyPinNetArray,
                               IndexArray driverArray,
                               IndexArray eligibleArray,
                               double couplingCapScale);

}  // namespace gr_parasitics
