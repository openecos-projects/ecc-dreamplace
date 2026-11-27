/**
 * @file   abacus_legalize.cpp
 * @author Yibo Lin
 * @date   Jun 2018
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
// database dependency
#include "utility/src/legalization_db.h"
#include "utility/src/make_placedb.h"
// local dependency
#include "abacus_legalize/src/abacus_legalize_cpu.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int abacusLegalizationLauncher(LegalizationDB<T> db, bool use_node_weights);

at::Tensor abacus_legalization_forward(
    at::Tensor init_pos, at::Tensor pos, at::Tensor node_size_x,
    at::Tensor node_size_y, at::Tensor node_weights,
    at::Tensor flat_region_boxes, at::Tensor flat_region_boxes_start,
    at::Tensor node2fence_region_map, double xl, double yl, double xh,
    double yh, double site_width, double row_height, int num_bins_x,
    int num_bins_y, int num_movable_nodes, int num_terminal_NIs,
    int num_filler_nodes, bool use_node_weights) {
  CHECK_FLAT_CPU(init_pos);
  CHECK_EVEN(init_pos);
  CHECK_CONTIGUOUS(init_pos);

  auto pos_copy = pos.clone();

  CPUTimer::hr_clock_rep timer_start, timer_stop;
  timer_start = CPUTimer::getGlobaltime();
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "abacusLegalizationLauncher", [&] {
        auto db = make_placedb<scalar_t>(
            init_pos, pos_copy, node_size_x, node_size_y, node_weights,
            flat_region_boxes, flat_region_boxes_start, node2fence_region_map,
            xl, yl, xh, yh, site_width, row_height, num_bins_x, num_bins_y,
            num_movable_nodes, num_terminal_NIs, num_filler_nodes);
        abacusLegalizationLauncher<scalar_t>(db, use_node_weights);
      });
  timer_stop = CPUTimer::getGlobaltime();
  dreamplacePrint(kINFO, "Abacus legalization takes %g ms\n",
                  (timer_stop - timer_start) * CPUTimer::getTimerPeriod());

  return pos_copy;
}

template <typename T>
int abacusLegalizationLauncher(LegalizationDB<T> db, bool use_node_weights) {
  abacusLegalizationCPU(db.init_x, db.init_y, db.node_size_x, db.node_size_y,
                        use_node_weights ? db.node_weights : (const T*)nullptr,
                        db.x, db.y, db.xl, db.yl, db.xh, db.yh,
                        db.site_width, db.row_height, 1, db.num_bins_y,
                        db.num_nodes, db.num_movable_nodes);

  return 0;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::abacus_legalization_forward,
        "Abacus legalization forward",
        py::arg("init_pos"), py::arg("pos"), py::arg("node_size_x"),
        py::arg("node_size_y"), py::arg("node_weights"),
        py::arg("flat_region_boxes"), py::arg("flat_region_boxes_start"),
        py::arg("node2fence_region_map"), py::arg("xl"), py::arg("yl"),
        py::arg("xh"), py::arg("yh"), py::arg("site_width"),
        py::arg("row_height"), py::arg("num_bins_x"), py::arg("num_bins_y"),
        py::arg("num_movable_nodes"), py::arg("num_terminal_NIs"),
        py::arg("num_filler_nodes"),
        py::arg("use_node_weights") = false);
}
