/**
 * @file   steiner_topo_standalone.cpp
 * @author Chaoyu Xing
 * @date   Mar 2025
 * @brief  Standalone CPU-only Steiner tree topology generation
 */

#include "directional_ufs.h"
#include "flute.hpp"
#include "utility/src/torch.h"
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeSteinerTopoLauncher(
    T *x, T *y, int *flat_netpin, int *netpin_start, int num_nets, int num_pins,
    int ignore_net_degree, std::vector<int> &wl, std::vector<int> &newx,
    std::vector<int> &newy, std::vector<int> &pin_relate_x,
    std::vector<int> &pin_relate_y, std::vector<int> &branch_u,
    std::vector<int> &branch_v, std::vector<int> &local2global_index,
    int *netbranch_start, int *netvertex_start, int *netsteiner_start) {
  // ATTENTION: global id might changed due to the insertion of steiner points

  // Load FLUTE lookup tables for Steiner tree generation
  flute::readLUT("thirdparty/flute/lut.ICCAD2015/POWV9.dat",
                 "thirdparty/flute/lut.ICCAD2015/POST9.dat");

  // Define scaling factor for integer coordinates
  constexpr int scale = 1000;

  // Initialize start indices for the first net
  netbranch_start[0] = 0;
  netvertex_start[0] = 0;

  // Process each net in sequence
  for (int i = 0; i < num_nets; ++i) {
    int degree = netpin_start[i + 1] - netpin_start[i];
    int num_former_vertices = netvertex_start[i];
    bool duplicate_pin = false;

    // Print progress periodically
    if (i % 50000 == 0 || i + 1 == num_nets) {
      std::cout << "Processing net " << i << " / " << num_nets - 1 << std::endl;
    }

    // --- Collect unique pin coordinates and map local indices ---
    std::vector<int> vx, vy;
    local2global_index.resize(local2global_index.size() + degree);
    std::map<Point<int>, std::vector<int>> pos2local_map;

    for (int current_index = 0; current_index < degree; ++current_index) {
      assert(current_index == newx.size() - num_former_vertices);
      int pin = flat_netpin[current_index + netpin_start[i]];
      Point<int> point(static_cast<int>(x[pin] * scale),
                       static_cast<int>(y[pin] * scale));

      // Map global pin index to local index
      local2global_index[netpin_start[i] + current_index] = pin;
      pos2local_map[point].push_back(current_index);
      newx.emplace_back(point.x());
      newy.emplace_back(point.y());

      // Check for duplicate pins at same location
      if (pos2local_map[point].size() > 1) {
        duplicate_pin = true;
      } else if (pos2local_map[point].size() == 1) {
        vx.emplace_back(point.x());
        vy.emplace_back(point.y());
      }
    }

    int num_valid_pins = pos2local_map.size();

    // --- Handle trivial case: net with only one unique pin location ---
    if (num_valid_pins == 1) {
      pin_relate_x.emplace_back(newx.size() - 1); // indexing to newx
      pin_relate_y.emplace_back(newx.size() - 1); // indexing to newy
      wl[i] = 0;

      // Handle duplicate pins at same location
      if (degree != 1) {
        pin_relate_x.resize(pin_relate_x.size() + pos2local_map.size());
        pin_relate_y.resize(pin_relate_y.size() + pos2local_map.size());
        for (const auto &[pos, indices] : pos2local_map) {
          for (const auto &idx : indices) {
            assert(idx < static_cast<int>(pin_relate_x.size()));
            pin_relate_x[num_former_vertices + idx] = num_former_vertices + idx;
            pin_relate_y[num_former_vertices + idx] = num_former_vertices + idx;
            if (idx != indices[0]) {
              branch_u.emplace_back(idx);
              branch_v.emplace_back(indices[0]);
            }
          }
        }
      }
      netbranch_start[i + 1] = branch_u.size();
      netvertex_start[i + 1] = newx.size();
      netsteiner_start[i] = netvertex_start[i] + degree;
      continue;
    }

    // --- Run FLUTE for nets with >= 2 unique pin locations ---
    flute::Tree ftree =
        flute::flute(num_valid_pins, vx.data(), vy.data(), ACCURACY);

    // Map to store Steiner point coordinates and their new local indices
    std::map<Point<int>, int> pos2steiner_map;

    // --- Identify Steiner points ---
    for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
      flute::Branch &b = ftree.branch[bid];
      Point<int> p(b.x, b.y);

      // Unmapped nodes are Steiner points
      if (pos2local_map.find(p) == pos2local_map.end()) {
        int current_index = newx.size() - num_former_vertices;
        pos2steiner_map[p] = current_index;
        pos2local_map[p].push_back(current_index);
        newx.emplace_back(b.x);
        newy.emplace_back(b.y);
      }
    }

    // Total number of vertices (pins + Steiner points) for current net
    const int total_vertex_local = newx.size() - num_former_vertices;
    UnifiedUFS<int> ufs(total_vertex_local);

    if (duplicate_pin) {
      std::cout << "Net " << i << " / " << num_nets - 1 << " with " << degree
                << " pins" << std::endl;
      std::cout << "Total vertex: " << total_vertex_local << std::endl;
    }

    // Map to store adjacency for Steiner diagonal connections
    std::map<int, std::vector<int>> steiner_adj_vertices_map;

    // --- Build branches, calculate WL, populate UFS ---
    for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
      flute::Branch &b1 = ftree.branch[bid];
      flute::Branch &b2 = ftree.branch[b1.n];

      Point<int> p1(b1.x, b1.y);
      Point<int> p2(b2.x, b2.y);

      if (p1 == p2)
        continue;

      // Get node indices
      int u = pos2local_map[p1][0];
      int v = pos2local_map[p2][0];

      // Record branch
      branch_u.emplace_back(u);
      branch_v.emplace_back(v);
      assert(u < total_vertex_local);
      assert(v < total_vertex_local);

      bool is_pin_u = pos2steiner_map.count(p1) > 0 ? false : true;
      bool is_pin_v = pos2steiner_map.count(p2) > 0 ? false : true;

      // Calculate wirelength (Manhattan distance)
      wl[i] += std::abs(newx[u] - newx[v]) + std::abs(newy[u] - newy[v]);

      // Skip if both endpoints are pins
      if (is_pin_u && is_pin_v)
        continue;

      // Handle Steiner point connections
      if (!is_pin_v) { // v is steiner
        if (p1.x() != p2.x() && p1.y() != p2.y()) {
          // Diagonal branch
          steiner_adj_vertices_map[v].emplace_back(u);
        } else {
          // Manhattan branch
          ufs.unite(u, v, p1, p2);
        }
      }
      if (!is_pin_u) { // u is steiner
        if (p1.x() != p2.x() && p1.y() != p2.y()) {
          // Diagonal branch
          steiner_adj_vertices_map[u].emplace_back(v);
        } else {
          // Manhattan branch
          ufs.unite(v, u, p2, p1);
        }
      }
    }

    // --- Calculate pin relations ---
    pin_relate_x.resize(pin_relate_x.size() + total_vertex_local);
    pin_relate_y.resize(pin_relate_y.size() + total_vertex_local);

    // Set pin relations for original pins
    for (const auto &[pos, indices] : pos2local_map) {
      for (const auto &idx : indices) {
        if (idx >= degree)
          continue;
        pin_relate_x[num_former_vertices + idx] = num_former_vertices + idx;
        pin_relate_y[num_former_vertices + idx] = num_former_vertices + idx;
        if (idx != indices[0]) {
          branch_u.emplace_back(idx);
          branch_v.emplace_back(indices[0]);
        }
      }
    }

    // Set pin relations for Steiner points using UFS
    for (const auto &[pos, idx] : pos2steiner_map) {
      assert(idx < static_cast<int>(pin_relate_x.size()));
      auto [x_pin, y_pin] = ufs.getRelateVertex(
          idx, degree, steiner_adj_vertices_map, newx, newy);
      pin_relate_x[num_former_vertices + idx] = num_former_vertices + x_pin;
      pin_relate_y[num_former_vertices + idx] = num_former_vertices + y_pin;
    }

    // output for duplicate pin positions
    if (duplicate_pin) {
      for (auto &[pos, indices] : pos2local_map) {
        if (indices.size() < 2)
          continue;
        std::cout << "Position (" << pos.x() << ", " << pos.y()
                  << ") has indices: ";
        for (const auto &idx : indices) {
          std::cout << idx << " ";
        }
        std::cout << std::endl;
      }
    }

    // Free memory allocated by FLUTE
    free(ftree.branch);

    // Update net start indices
    netbranch_start[i + 1] = branch_u.size();
    netvertex_start[i + 1] = newx.size();
    netsteiner_start[i] = netvertex_start[i] + degree - pos2steiner_map.size();
  }
  return 0;
}

/**
 * @brief PyTorch wrapper for the Steiner tree topology generation launcher
 */
std::vector<at::Tensor> steiner_topo_forward(at::Tensor pos,
                                             at::Tensor flat_netpin,
                                             at::Tensor netpin_start,
                                             int ignore_net_degree) {
  // Input tensor validation
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);

  // Parse parameters
  const int num_nets = netpin_start.numel() - 1;
  const int num_pins = pos.numel() / 2;

  // Initialize output vectors
  std::vector<int> wl_vec(num_nets);
  std::vector<int> newx_vec;
  std::vector<int> newy_vec;
  std::vector<int> pin_relate_x_vec;
  std::vector<int> pin_relate_y_vec;
  std::vector<int> branch_u_vec;
  std::vector<int> branch_v_vec;
  std::vector<int> local2global_index_vec;

  // Initialize output tensors for net start indices
  auto net_branch_start =
      at::zeros({static_cast<int>(num_nets) + 1}, torch::kInt32);
  auto net_vertex_start =
      at::zeros({static_cast<int>(num_nets) + 1}, torch::kInt32);
  auto net_steiner_start =
      at::zeros({static_cast<int>(num_nets) + 1}, torch::kInt32);

  // Dispatch to C++ kernel with appropriate data types
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeSteinerTopoLauncher", [&] {
    computeSteinerTopoLauncher<scalar_t>(
        // Input data
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins, // y coordinates
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int), num_nets, num_pins,
        ignore_net_degree,
        // Output data
        wl_vec, newx_vec, newy_vec, pin_relate_x_vec, pin_relate_y_vec,
        branch_u_vec, branch_v_vec, local2global_index_vec,
        DREAMPLACE_TENSOR_DATA_PTR(net_branch_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_vertex_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_steiner_start, int));
  });

  // Convert resulting C++ vectors to PyTorch tensors
  auto newx = at::zeros({static_cast<int>(newx_vec.size())}, torch::kInt32);
  auto newy = at::zeros({static_cast<int>(newx_vec.size())}, torch::kInt32);
  auto pin_relate_x =
      at::zeros({static_cast<int>(pin_relate_x_vec.size())}, torch::kInt32);
  auto pin_relate_y =
      at::zeros({static_cast<int>(pin_relate_x_vec.size())}, torch::kInt32);
  auto branch_u =
      at::zeros({static_cast<int>(branch_u_vec.size())}, torch::kInt32);
  auto branch_v =
      at::zeros({static_cast<int>(branch_u_vec.size())}, torch::kInt32);
  auto local2global_index = at::zeros(
      {static_cast<int>(local2global_index_vec.size())}, torch::kInt32);

  // Copy data from vectors to tensors
  for (int i = 0; i < newx_vec.size(); ++i) {
    newx[i] = newx_vec[i];
    newy[i] = newy_vec[i];
  }
  for (int i = 0; i < pin_relate_x_vec.size(); ++i) {
    pin_relate_x[i] = pin_relate_x_vec[i];
    pin_relate_y[i] = pin_relate_y_vec[i];
  }
  for (int i = 0; i < branch_u_vec.size(); ++i) {
    branch_u[i] = branch_u_vec[i];
    branch_v[i] = branch_v_vec[i];
  }
  for (int i = 0; i < local2global_index_vec.size(); ++i) {
    local2global_index[i] = local2global_index_vec[i];
  }

  // Return all computed tensors
  return {newx,     newy,     pin_relate_x,     pin_relate_y,
          branch_u, branch_v, net_branch_start, local2global_index};
}

DREAMPLACE_END_NAMESPACE

// Register the C++ function with pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::steiner_topo_forward,
        "SteinerTopo forward", py::arg("pos"), py::arg("flat_netpin"),
        py::arg("netpin_start"), py::arg("ignore_net_degree"));
}