#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace dreamplace {

pybind11::dict compute_candidate_dirty_nets(
    const at::Tensor& changed_instance_ids,
    const at::Tensor& moved_instance_ids,
    const at::Tensor& accepted_net_ids,
    const at::Tensor& candidate_net_ids,
    const at::Tensor& flat_node2pin_map,
    const at::Tensor& flat_node2pin_start_map,
    const at::Tensor& pin2net_map);

pybind11::dict refresh_candidate_geometry(
    const at::Tensor& dirty_net_ids,
    const at::Tensor& candidate_net_id,
    const at::Tensor& candidate_parent_node_id,
    const at::Tensor& candidate_child_node_id,
    const at::Tensor& candidate_tree_node_id,
    const at::Tensor& candidate_is_segment,
    const at::Tensor& candidate_split_ratio,
    const at::Tensor& node_x_dbu,
    const at::Tensor& node_y_dbu,
    bool all_candidates);

}  // namespace dreamplace
