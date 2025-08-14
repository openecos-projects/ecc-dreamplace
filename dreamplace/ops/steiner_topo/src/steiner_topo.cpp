/**
 * @file   steiner_topo.cpp
 * @author Chaoyu Xing
 * @date   Mar 2025
 * @brief  CPU-only Steiner tree topology generation
 */

#include "directional_ufs.h"
#include "flute.hpp"
#include "utility/src/torch.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>
#include <queue>
#include <utility>
#include <vector>
#include <filesystem>

DREAMPLACE_BEGIN_NAMESPACE

struct NetResult {
  int num_steiner = 0;
  int netid = 0;
  std::vector<int> newx;
  std::vector<int> newy;
  std::vector<int> vtx_relate_x;
  std::vector<int> vtx_relate_y;
  std::vector<int> vtx_fa;
  std::vector<int> net_flat_topo_idx;
  std::vector<int> local2global_idx;
};

template <typename T>
int computeSteinerTreeLauncher(
    T *x, T *y, int *flat_netpin, int *netpin_start, int num_nets, int num_pins,
    int ignore_net_degree, int *wl, std::vector<T> &newx,
    std::vector<T> &newy, std::vector<int> &vtx_relate_x,
    std::vector<int> &vtx_relate_y, int *netsteiner_start,
    std::vector<int> &vtx_fa, std::vector<int> &flat_vtx_to,
    std::vector<int> &flat_vtx_from, std::vector<int> &net_flat_topo_idx,
    std::vector<int> &flat_vtx_to_start, int *net_flat_topo_idx_start) {

  static bool is_lut_loaded = false;
  if (!is_lut_loaded) {
    is_lut_loaded = true;
    std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path project_root = source_dir.parent_path().parent_path().parent_path().parent_path();
    std::string powv9_path = (project_root / "thirdparty/flute/lut.ICCAD2015/POWV9.dat").string();
    std::string post9_path = (project_root / "thirdparty/flute/lut.ICCAD2015/POST9.dat").string();
    flute::readLUT(powv9_path.c_str(), post9_path.c_str());
  }

  constexpr int scale = 1000;
  int total_steiner = 0;
  std::vector<NetResult> net_result(num_nets);

#pragma omp parallel for reduction(+ : total_steiner)
  for (int netid = 0; netid < num_nets; ++netid) {
    int degree = netpin_start[netid + 1] - netpin_start[netid];
    bool duplicate_pin = false;

    // --- Collect unique pin coordinates and map local indices ---
    std::map<Point<int>, std::vector<int>> pos2local_map;
    std::vector<int> vx, vy;
    vx.reserve(degree);
    vy.reserve(degree);
    net_result[netid].local2global_idx.resize(degree);
    for (int cur_local_idx = 0; cur_local_idx < degree; ++cur_local_idx) {
      int pin_global_idx = flat_netpin[netpin_start[netid] + cur_local_idx];
      Point<int> point(static_cast<int>(x[pin_global_idx] * scale),
                       static_cast<int>(y[pin_global_idx] * scale));
      net_result[netid].local2global_idx[cur_local_idx] = pin_global_idx;
      pos2local_map[point].push_back(cur_local_idx);
      net_result[netid].newx.push_back(point.x());
      net_result[netid].newy.push_back(point.y());

      // Check for duplicate pins at same location
      if (pos2local_map[point].size() > 1) {
        duplicate_pin = true;
      } else if (pos2local_map[point].size() == 1) {
        vx.push_back(point.x());
        vy.push_back(point.y());
      }
    }

    int num_valid_pins = pos2local_map.size();
    std::vector<std::vector<int>> edge(degree);
    auto add_edge = [&edge](int u, int v) {
      edge[u].push_back(v);
      edge[v].push_back(u);
    };

    if (num_valid_pins == 1) {
      // --- net with only one unique pin location ---
      wl[netid] = 0;
      net_result[netid].vtx_fa.resize(degree);
      net_result[netid].net_flat_topo_idx.resize(degree);
      net_result[netid].vtx_relate_x.resize(degree);
      net_result[netid].vtx_relate_y.resize(degree);
      for (const auto &[pos, indices] : pos2local_map) {
        int first_local_idx = indices[0];
        for (const auto &local_idx : indices) {
          net_result[netid].vtx_relate_x[local_idx] = local_idx;
          net_result[netid].vtx_relate_y[local_idx] = local_idx;
          if (local_idx != first_local_idx) {
            add_edge(local_idx, first_local_idx);
          }
        }
      }
    } else {
      // --- nets with >= 2 unique pin locations ---
      flute::Tree ftree =
          flute::flute(num_valid_pins, vx.data(), vy.data(), ACCURACY);
      std::map<Point<int>, int> pos2steiner_map;
      int num_steiner_points = 0;

      for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
        flute::Branch &b = ftree.branch[bid];
        Point<int> p(b.x, b.y);
        auto it = pos2local_map.find(p);
        bool is_original_pin_loc =
            (it != pos2local_map.end() && !it->second.empty());
        if (!is_original_pin_loc &&
            pos2steiner_map.find(p) == pos2steiner_map.end()) {
          // It's a new Steiner point location
          int steiner_local_idx = degree + num_steiner_points++;
          pos2steiner_map[p] = steiner_local_idx;
          pos2local_map[p].push_back(steiner_local_idx);
          net_result[netid].newx.push_back(b.x);
          net_result[netid].newy.push_back(b.y);
        }
      }
      net_result[netid].num_steiner = num_steiner_points;
      total_steiner += net_result[netid].num_steiner;

      const int total_vertex_local = degree + net_result[netid].num_steiner;
      UnifiedUFS<int> ufs(total_vertex_local);

      edge.resize(total_vertex_local);
      net_result[netid].vtx_relate_x.resize(total_vertex_local);
      net_result[netid].vtx_relate_y.resize(total_vertex_local);
      net_result[netid].vtx_fa.resize(total_vertex_local);
      net_result[netid].net_flat_topo_idx.resize(total_vertex_local);

      // store adjacency for Steiner diagonal connections
      std::map<int, std::vector<int>> steiner_adj_vertices_map;
      int cur_wl = 0;

      // --- construct relate ---
      for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
        flute::Branch &b1 = ftree.branch[bid];
        flute::Branch &b2 = ftree.branch[b1.n];

        Point<int> p1(b1.x, b1.y);
        Point<int> p2(b2.x, b2.y);

        if (p1 == p2)
          continue;

        int u_local = pos2local_map[p1][0];
        int v_local = pos2local_map[p2][0];
        add_edge(u_local, v_local);

        cur_wl += std::abs(net_result[netid].newx[u_local] -
                           net_result[netid].newx[v_local]) +
                  std::abs(net_result[netid].newy[u_local] -
                           net_result[netid].newy[v_local]);

        bool is_steiner_u = (u_local >= degree);
        bool is_steiner_v = (v_local >= degree);
        if (!is_steiner_u && !is_steiner_v)
          continue;
        if (is_steiner_v) {
          if (p1.x() != p2.x() && p1.y() != p2.y()) {
            // Diagonal branch
            steiner_adj_vertices_map[v_local].emplace_back(u_local);
          } else {
            // Manhattan branch
            ufs.unite(u_local, v_local, p1, p2);
          }
        }
        if (is_steiner_u) {
          if (p1.x() != p2.x() && p1.y() != p2.y()) {
            // Diagonal branch
            steiner_adj_vertices_map[u_local].emplace_back(v_local);
          } else {
            // Manhattan branch
            ufs.unite(v_local, u_local, p2, p1);
          }
        }
      }
      wl[netid] = cur_wl;
      for (const auto &[pos, indices] : pos2local_map) {
        int first_local_idx = indices[0];
        if (first_local_idx >= degree) {
          auto [x_pin_local, y_pin_local] = ufs.getRelateVertex(
              first_local_idx, degree, steiner_adj_vertices_map,
              net_result[netid].newx, net_result[netid].newy);
          net_result[netid].vtx_relate_x[first_local_idx] = x_pin_local;
          net_result[netid].vtx_relate_y[first_local_idx] = y_pin_local;
        } else {
          for (const auto &local_idx : indices) {
            if (local_idx >= degree) {
              net_result[netid].vtx_relate_x[local_idx] = first_local_idx;
              net_result[netid].vtx_relate_y[local_idx] = first_local_idx;
            } else {
              net_result[netid].vtx_relate_x[local_idx] = local_idx;
              net_result[netid].vtx_relate_y[local_idx] = local_idx;
            }
            if (local_idx != first_local_idx) {
              add_edge(local_idx, first_local_idx);
            }
          }
        }
      }

      // output for duplicate pin pos
      if (duplicate_pin) {
        for (auto &[pos, indices] : pos2local_map) {
          if (indices.size() < 2)
            continue;
          std::cout << "Position (" << pos.x() << ", " << pos.y()
                    << ") has local indices: ";
          for (const auto &idx : indices) {
            std::cout << idx << "(" << net_result[netid].local2global_idx[idx]
                      << ") ";
          }
          std::cout << std::endl;
        }
      }

      free(ftree.branch);
    }

    // --- graph stucture ---
    auto topo_sort = [&net_result, &edge, &degree](int netid) {
      int vtx_degree = net_result[netid].num_steiner + degree;
      std::vector<bool> visit(vtx_degree, false);
      int root = 0;
      int topo_ptr = 0;
      visit[root] = true;
      net_result[netid].vtx_fa[root] = -1; // root has no parent
      net_result[netid].net_flat_topo_idx[topo_ptr++] = root;
      std::queue<int> q;
      q.push(root);
      while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int &v : edge[u]) {
          if (visit[v]) {
            v = -1;
            continue;
          }
          visit[v] = true;
          q.push(v);
          net_result[netid].net_flat_topo_idx[topo_ptr++] = v;
          net_result[netid].vtx_fa[v] = u;
        }
      }
    };
    topo_sort(netid);
  }

  // --- merge net results ---
  int total_vtx = num_pins + total_steiner;
  std::vector<std::vector<int>> global_adj(total_vtx);
  newx.resize(total_vtx);
  newy.resize(total_vtx);
  vtx_relate_x.resize(total_vtx);
  vtx_relate_y.resize(total_vtx);
  vtx_fa.resize(total_vtx);
  net_flat_topo_idx.resize(total_vtx);
  net_flat_topo_idx_start[0] = 0;
  netsteiner_start[0] = num_pins;
  
  for (int netid = 0; netid < num_nets; ++netid) {
    int degree = netpin_start[netid + 1] - netpin_start[netid];
    int degree_steiner = net_result[netid].num_steiner;

    net_flat_topo_idx_start[netid + 1] =
        net_flat_topo_idx_start[netid] + net_result[netid].newx.size();
    netsteiner_start[netid + 1] = netsteiner_start[netid] + degree_steiner;

    auto local2global = [&degree, &net_result, &netid,
                         &netsteiner_start](int idx) {
      if (idx == -1) {
        return idx; // -1 for root
      } else if (idx < degree) {
        return net_result[netid].local2global_idx[idx];
      } else {
        return netsteiner_start[netid] + idx - degree;
      }
    };
    for (int local_id = 0; local_id < degree + degree_steiner; ++local_id) {
      int global_idx = local2global(local_id);
      newx[global_idx] = net_result[netid].newx[local_id] / scale;
      newy[global_idx] = net_result[netid].newy[local_id] / scale;
      vtx_fa[global_idx] = local2global(net_result[netid].vtx_fa[local_id]);
      vtx_relate_x[global_idx] = local2global(net_result[netid].vtx_relate_x[local_id]);
      vtx_relate_y[global_idx] = local2global(net_result[netid].vtx_relate_y[local_id]);
      net_flat_topo_idx[net_flat_topo_idx_start[netid] + local_id] = 
          local2global(net_result[netid].net_flat_topo_idx[local_id]);
      if (vtx_fa[global_idx] != -1) {
        global_adj[vtx_fa[global_idx]].push_back(global_idx);
      }
    }
  }
  
  flat_vtx_to.reserve(total_vtx);
  flat_vtx_from.reserve(total_vtx);
  flat_vtx_to_start.resize(total_vtx + 1);
  flat_vtx_to_start[0] = 0;

  for (int i = 0; i < total_vtx; ++i) {
      for (int neighbor : global_adj[i]) {
          flat_vtx_to.push_back(neighbor);
          flat_vtx_from.push_back(i);
      }
      flat_vtx_to_start[i + 1] = flat_vtx_to.size();
  }
  
  // DEBUG
  std::cout << "build tree done" << std::endl;
  return 0;
}

template <typename T>
int computeSteinerPosLauncher(const T *pin_pos_x, const T *pin_pos_y,
                              const std::vector<int> &vtx_relate_x,
                              const std::vector<int> &vtx_relate_y,
                              const int num_vertices,
                              std::vector<T> &updated_newx, std::vector<T> &updated_newy) {
  updated_newx.resize(num_vertices);
  updated_newy.resize(num_vertices);

#pragma omp parallel for
  for (int vtx_id = 0; vtx_id < num_vertices; ++vtx_id) {
    updated_newx[vtx_id] = pin_pos_x[vtx_relate_x[vtx_id]];
    updated_newy[vtx_id] = pin_pos_y[vtx_relate_y[vtx_id]];
  }
  return 0;
}

template <typename T>
int computeSteinerTopoGradLauncher(T *grad_vertices_x, T *grad_vertices_y,
                                   const int *vtx_relate_x,
                                   const int *vtx_relate_y,
                                   const int num_vertices, T *grad_pin_x,
                                   T *grad_pin_y) {
  for (int vtx_id = 0; vtx_id < num_vertices; ++vtx_id) {
    grad_pin_x[vtx_relate_x[vtx_id]] += grad_vertices_x[vtx_id];
    grad_pin_y[vtx_relate_y[vtx_id]] += grad_vertices_y[vtx_id];
  }
  return 0;
}

template <typename T>
at::Tensor convertVecToTens(const std::vector<T>& vec, const at::TensorOptions& options) {
    return at::from_blob(const_cast<T*>(vec.data()), {static_cast<long>(vec.size())}, options).clone();
}

std::vector<at::Tensor> build_tree(at::Tensor pos, at::Tensor flat_netpin,
                                   at::Tensor netpin_start,
                                   int ignore_net_degree) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);

  const int num_nets = netpin_start.numel() - 1;
  const int num_pins = pos.numel() / 2;
  std::vector<int> vtx_relate_x_vec;
  std::vector<int> vtx_relate_y_vec;
  std::vector<int> vtx_fa_vec;
  std::vector<int> flat_vtx_to_vec;
  std::vector<int> flat_vtx_from_vec;
  std::vector<int> net_flat_topo_idx_vec;
  std::vector<int> flat_vtx_to_start_vec;

  auto options_float = pos.options();
  auto options_int = flat_netpin.options();

  auto net_vertex_start = at::zeros({num_nets + 1}, options_int);
  auto wl = at::zeros({num_nets + 1}, options_int);
  auto net_steiner_start = at::zeros({num_nets + 1}, options_int);
  auto net_flat_topo_idx_start_tensor = at::zeros({num_nets + 1}, options_int);
  std::vector<at::Tensor> result;
  
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeSteinerTreeLauncher", [&] {
    std::vector<scalar_t> newx_vec;
    std::vector<scalar_t> newy_vec;

    computeSteinerTreeLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int), num_nets, num_pins,
        ignore_net_degree, DREAMPLACE_TENSOR_DATA_PTR(wl, int), 
        newx_vec, newy_vec, vtx_relate_x_vec, vtx_relate_y_vec,
        DREAMPLACE_TENSOR_DATA_PTR(net_steiner_start, int), vtx_fa_vec,
        flat_vtx_to_vec, flat_vtx_from_vec, net_flat_topo_idx_vec,
        flat_vtx_to_start_vec,
        DREAMPLACE_TENSOR_DATA_PTR(net_flat_topo_idx_start_tensor, int));

    auto newx                     = convertVecToTens(newx_vec, options_float);
    auto newy                     = convertVecToTens(newy_vec, options_float);
    auto vtx_relate_x             = convertVecToTens(vtx_relate_x_vec, options_int);
    auto vtx_relate_y             = convertVecToTens(vtx_relate_y_vec, options_int);
    auto vtx_fa                   = convertVecToTens(vtx_fa_vec, options_int);
    auto flat_vtx_to              = convertVecToTens(flat_vtx_to_vec, options_int);
    auto flat_vtx_from            = convertVecToTens(flat_vtx_from_vec, options_int);
    auto net_flat_topo_idx        = convertVecToTens(net_flat_topo_idx_vec, options_int);
    auto flat_vtx_to_start_tensor = convertVecToTens(flat_vtx_to_start_vec, options_int);

    result = {newx,
              newy,
              vtx_relate_x,
              vtx_relate_y,
              net_vertex_start,
              net_steiner_start,
              vtx_fa,
              flat_vtx_to,
              flat_vtx_from,
              flat_vtx_to_start_tensor,
              net_flat_topo_idx,
              net_flat_topo_idx_start_tensor};
  });

  return result;
}

std::vector<at::Tensor> steiner_topo_forward(at::Tensor pin_pos,
                                             at::Tensor cached_vtx_relate_x,
                                             at::Tensor cached_vtx_relate_y,
                                             int num_vertices) {
  CHECK_FLAT_CPU(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);
  CHECK_FLAT_CPU(cached_vtx_relate_x);
  CHECK_CONTIGUOUS(cached_vtx_relate_x);
  CHECK_FLAT_CPU(cached_vtx_relate_y);
  CHECK_CONTIGUOUS(cached_vtx_relate_y);

  const int num_pins = pin_pos.numel() / 2;
  auto options_float = pin_pos.options();

  std::vector<at::Tensor> result;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "computeSteinerPosLauncher", [&] {
    std::vector<int> vtx_relate_x_vec(
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_x, int),
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_x, int) + num_vertices);
    std::vector<int> vtx_relate_y_vec(
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_y, int),
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_y, int) + num_vertices);
    std::vector<scalar_t> updated_newx_vec;
    std::vector<scalar_t> updated_newy_vec;

    computeSteinerPosLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
        vtx_relate_x_vec, vtx_relate_y_vec, num_vertices, 
        updated_newx_vec,
        updated_newy_vec);

    auto updated_newx = convertVecToTens(updated_newx_vec, options_float);
    auto updated_newy = convertVecToTens(updated_newy_vec, options_float);

    result = {updated_newx, updated_newy};
  });

  return result;
}

at::Tensor steiner_topo_backward(at::Tensor grad_newx, at::Tensor grad_newy,
                                 at::Tensor pos, at::Tensor vtx_relate_x,
                                 at::Tensor vtx_relate_y) {

  CHECK_FLAT_CPU(grad_newx);
  CHECK_CONTIGUOUS(grad_newx);
  CHECK_FLAT_CPU(grad_newy);
  CHECK_CONTIGUOUS(grad_newy);
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(vtx_relate_x);
  CHECK_CONTIGUOUS(vtx_relate_x);
  CHECK_FLAT_CPU(vtx_relate_y);
  CHECK_CONTIGUOUS(vtx_relate_y);

  auto grad_pin = at::zeros(pos.numel());

  int num_vertices = grad_newx.numel();
  int num_pins = pos.numel() / 2;

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeSteinerTopoGradLauncher", [&] {
        computeSteinerTopoGradLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad_newx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_newy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_x, int),
            DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_x, int), num_vertices,
            DREAMPLACE_TENSOR_DATA_PTR(grad_pin, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pin, scalar_t) + num_pins);
      });

  return grad_pin;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::steiner_topo_forward,
        "SteinerTopo forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::steiner_topo_backward,
        "SteinerTopo backward");
  m.def("build_tree", &DREAMPLACE_NAMESPACE::build_tree, "Build Tree");
}