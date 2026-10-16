#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace dreamplace {

pybind11::dict pack_segment_count_topology(
    const at::Tensor& flat_net2pin,
    const at::Tensor& flat_net2pin_start,
    const at::Tensor& net2driver,
    const at::Tensor& pin2node,
    const at::Tensor& net_flat_topo_sort,
    const at::Tensor& net_flat_topo_sort_start,
    const at::Tensor& pin_fa,
    const at::Tensor& node_x,
    const at::Tensor& node_y,
    const at::Tensor& node_x_dbu,
    const at::Tensor& node_y_dbu,
    const at::Tensor& pin_capacitance,
    int64_t num_movable_nodes,
    int64_t num_terminals,
    double dbu,
    double scale_factor,
    double r_unit,
    double c_unit,
    int64_t max_repeater_count,
    int64_t num_threads);

pybind11::dict pack_candidate_topology(
    const at::Tensor& flat_net2pin,
    const at::Tensor& flat_net2pin_start,
    const at::Tensor& net2driver,
    const at::Tensor& pin2node,
    const at::Tensor& net_flat_topo_sort,
    const at::Tensor& net_flat_topo_sort_start,
    const at::Tensor& pin_fa,
    const at::Tensor& node_x,
    const at::Tensor& node_y,
    const at::Tensor& node_x_dbu,
    const at::Tensor& node_y_dbu,
    const at::Tensor& candidate_net_id,
    const at::Tensor& candidate_parent_node_id,
    const at::Tensor& candidate_child_node_id,
    const at::Tensor& candidate_tree_node_id,
    const at::Tensor& candidate_synthetic_node_id,
    const at::Tensor& candidate_is_segment,
    const at::Tensor& candidate_split_ratio,
    const at::Tensor& pin_capacitance,
    int64_t num_movable_nodes,
    int64_t num_terminals,
    double dbu,
    double scale_factor,
    double r_unit,
    double c_unit,
    int64_t num_threads);

pybind11::dict select_packed_segment_count_inputs(
    const pybind11::dict& prepared_timing_inputs,
    const pybind11::dict& packed_segment_geometry,
    const at::Tensor& active_net_ids,
    int64_t num_threads);

pybind11::dict select_packed_candidate_inputs(
    const pybind11::dict& prepared_timing_inputs,
    const at::Tensor& active_net_ids,
    int64_t num_threads);

}  // namespace dreamplace
