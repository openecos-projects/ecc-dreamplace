#pragma once

#include "utility/src/torch.h"

std::vector<at::Tensor> lut_indexed_cpu(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> buffer_lut_indexed_cpu(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor);

std::vector<at::Tensor> smooth_scatter_cpu(at::Tensor, at::Tensor, at::Tensor, double, bool,
                                        std::vector<at::Tensor>);
std::vector<at::Tensor> hard_scatter_forward_cpu(at::Tensor, at::Tensor, at::Tensor, bool);
std::vector<at::Tensor> hard_scatter_backward_cpu(at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool);
std::vector<at::Tensor> cell_aat_forward_cpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>, bool, double);
std::vector<at::Tensor> cell_aat_backward_cpu(at::Tensor, at::Tensor, at::Tensor, int64_t,
                                           std::vector<at::Tensor>);
std::vector<at::Tensor> net_aat_forward_cpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, double);
std::vector<at::Tensor> net_aat_backward_cpu(at::Tensor, at::Tensor, at::Tensor, int64_t,
    std::vector<at::Tensor>, bool, bool);
std::vector<at::Tensor> buffer_interpolate_forward_cpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> buffer_interpolate_backward_cpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> timing_lanes_forward_cpu(at::Tensor, std::vector<at::Tensor>,
    std::vector<at::Tensor>, bool, double, std::vector<at::Tensor>);
std::vector<at::Tensor> timing_lanes_backward_cpu(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    std::vector<at::Tensor>, bool);


inline void registerTimingCpu(pybind11::module_& m) {
  m.def("timing_lanes_forward", &timing_lanes_forward_cpu, "Grouped arrival/slew lanes (CPU)");
  m.def("timing_lanes_backward", &timing_lanes_backward_cpu, "Arrival/slew lane adjoints (CPU)");
  m.def("buffer_interpolate_forward", &buffer_interpolate_forward_cpu, "Buffer drive-strength interpolation (CPU)");
  m.def("buffer_interpolate_backward", &buffer_interpolate_backward_cpu, "Buffer interpolation derivatives (CPU)");
  m.def("net_aat_forward", &net_aat_forward_cpu, "Net arrival/slew updates (CPU)");
  m.def("net_aat_backward", &net_aat_backward_cpu, "Deterministic net pin derivatives (CPU)");
  m.def("hard_scatter_forward", &hard_scatter_forward_cpu, "Slew max and Torch tie counts (CPU)");
  m.def("hard_scatter_backward", &hard_scatter_backward_cpu, "Slew max first-order adjoints (CPU)");
  m.def("buffer_indexed_forward", &buffer_lut_indexed_cpu, "Indexed buffer LUT query with 1D extrapolation (CPU)");
  m.def("cell_aat_forward", &cell_aat_forward_cpu, "Fixed-master cell AAT forward (CPU)");
  m.def("cell_aat_backward", &cell_aat_backward_cpu, "Fixed-master cell AAT backward (CPU)",
        pybind11::arg("arcs"), pybind11::arg("jac"), pybind11::arg("gradient"), pybind11::arg("pins"),
        pybind11::arg("groups") = std::vector<at::Tensor>{});
  m.def("smooth_scatter_forward", &smooth_scatter_cpu, "Timing scatter log-sum-exp with first derivatives (CPU)",
        pybind11::arg("dest"), pybind11::arg("index"), pybind11::arg("src"), pybind11::arg("tau"),
        pybind11::arg("include_self"), pybind11::arg("groups") = std::vector<at::Tensor>{});
  m.def("indexed_forward", &lut_indexed_cpu, "Indexed scalar/1D/2D LUT query with input derivatives (CPU)");
}
