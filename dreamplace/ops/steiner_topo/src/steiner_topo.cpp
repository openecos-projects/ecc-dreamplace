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

DREAMPLACE_BEGIN_NAMESPACE

struct NetResult {
  int num_steiner = 0;
  int netid = 0;
  std::vector<int> newx;
  std::vector<int> newy;
  std::vector<int> vtx_relate_x;
  std::vector<int> vtx_relate_y;
  std::vector<int> vtx_fa;
  std::vector<int> flat_vtx_to;
  std::vector<int> flat_vtx_to_start;
  std::vector<int> net_flat_topo_idx;
  std::vector<int> local2global_idx;
};

template <typename T>
int computeSteinerTreeLauncher(
    T *x, T *y, int *flat_netpin, int *netpin_start, int num_nets, int num_pins,
    int ignore_net_degree, int *wl, std::vector<int> &newx,
    std::vector<int> &newy, std::vector<int> &vtx_relate_x,
    std::vector<int> &vtx_relate_y, int *netsteiner_start,
    std::vector<int> &vtx_fa, std::vector<int> &flat_vtx_to,
    std::vector<int> &flat_vtx_from, std::vector<int> &net_flat_topo_idx,
    std::vector<int> &flat_vtx_to_start, int *net_flat_topo_idx_start) {

  static bool is_lut_loaded = false;
  if (!is_lut_loaded) {
    is_lut_loaded = true;
    flute::readLUT("thirdparty/flute/lut.ICCAD2015/POWV9.dat",
                   "thirdparty/flute/lut.ICCAD2015/POST9.dat");
  }

  // Define scaling factor for integer coordinates
  constexpr int scale = 1000;

  int total_steiner = 0;

  std::vector<NetResult> net_result(num_nets);

#pragma omp parallel for reduction(+ : total_steiner)
  for (int netid = 0; netid < num_nets; ++netid) {
    int degree = netpin_start[netid + 1] - netpin_start[netid];
    bool duplicate_pin = false;

    net_result[netid].local2global_idx.resize(degree);

    // --- Collect unique pin coordinates and map local indices ---
    std::vector<int> vx, vy;
    vx.reserve(degree);
    vy.reserve(degree);

    // Maps scaled pos -> local pin index (0 to degree-1)
    std::map<Point<int>, std::vector<int>> pos2local_map;

    for (int cur_local_idx = 0; cur_local_idx < degree; ++cur_local_idx) {

      int pin_global_idx = flat_netpin[netpin_start[netid] + cur_local_idx];
      Point<int> point(static_cast<int>(x[pin_global_idx] * scale),
                       static_cast<int>(y[pin_global_idx] * scale));

      net_result[netid].local2global_idx.at(cur_local_idx) = pin_global_idx;

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

    // --- Handle trivial case: net with only one unique pin location ---
    if (num_valid_pins == 1) {
      wl[netid] = 0;
      net_result[netid].vtx_fa.resize(degree);
      net_result[netid].flat_vtx_to_start.resize(degree + 1);
      net_result[netid].net_flat_topo_idx.resize(degree);
      net_result[netid].vtx_relate_x.resize(degree);
      net_result[netid].vtx_relate_y.resize(degree);
      // Handle duplicate pins at same location
      for (const auto &[pos, indices] : pos2local_map) {
        int first_local_idx = indices[0]; // First local index at this position
        for (const auto &local_idx : indices) { // local_idx is 0 to degree-1
          net_result[netid].vtx_relate_x.at(local_idx) = local_idx;
          net_result[netid].vtx_relate_y.at(local_idx) = local_idx;

          if (local_idx != first_local_idx) {
            // Add branches between duplicate pins using LOCAL indices
            add_edge(local_idx, first_local_idx); // FIX
          }
        }
      }
    } else if (degree == 0) {
      wl[netid] = 0;
    } else {

      // --- Run FLUTE for nets with >= 2 unique pin locations ---
      flute::Tree ftree =
          flute::flute(num_valid_pins, vx.data(), vy.data(), ACCURACY);

      // Map to store Steiner point coordinates and their new local indices
      // (relative to net start, >= degree)
      std::map<Point<int>, int> pos2steiner_map;
      int num_steiner_points = 0;

      // --- Identify Steiner points ---
      for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
        flute::Branch &b = ftree.branch[bid];
        Point<int> p(b.x, b.y);

        // Use map.find() for efficient lookup - O(log N)
        auto it = pos2local_map.find(p);

        // Check if p exists in the map AND corresponds to an original pin
        bool is_original_pin_loc =
            (it != pos2local_map.end() && !it->second.empty());

        if (!is_original_pin_loc &&
            pos2steiner_map.find(p) == pos2steiner_map.end()) {
          // It's a new Steiner point location
          int steiner_local_idx = degree + num_steiner_points++;
          pos2steiner_map[p] = steiner_local_idx;
          pos2local_map[p].push_back(steiner_local_idx); // Add to general map

          net_result[netid].newx.push_back(b.x);
          net_result[netid].newy.push_back(b.y);
        }
      }
      net_result[netid].num_steiner = num_steiner_points;
      // Total number of vertices (pins + Steiner points) for cur net
      const int total_vertex_local = degree + num_steiner_points;
      UnifiedUFS<int> ufs(total_vertex_local);

      // Resize pin_relate vectors for the vertices added by this net
      edge.resize(total_vertex_local);
      net_result[netid].vtx_relate_x.resize(total_vertex_local);
      net_result[netid].vtx_relate_y.resize(total_vertex_local);

      if (duplicate_pin) {
        std::cout << "Net " << netid << " / " << num_nets - 1 << " with "
                  << degree << " pins" << std::endl;
        std::cout << "Total vertex: " << total_vertex_local << std::endl;
      }

      // Map to store adjacency for Steiner diagonal connections
      std::map<int, std::vector<int>> steiner_adj_vertices_map;
      int cur_wl = 0;

      // --- Build branches, calculate WL, populate UFS ---
      for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
        flute::Branch &b1 = ftree.branch[bid];
        flute::Branch &b2 = ftree.branch[b1.n];

        Point<int> p1(b1.x, b1.y);
        Point<int> p2(b2.x, b2.y);

        if (p1 == p2)
          continue;

        int u_local = pos2local_map[p1][0];
        int v_local = pos2local_map[p2][0];

        // Record branch using LOCAL indices
        add_edge(u_local, v_local);

        // Calculate wirelength (Manhattan distance)
        cur_wl += std::abs(net_result[netid].newx.at(u_local) -
                           net_result[netid].newx.at(v_local)) +
                  std::abs(net_result[netid].newy.at(u_local) -
                           net_result[netid].newy.at(v_local));

        // Check if endpoints are Steiner points using LOCAL indices
        bool is_steiner_u = (u_local >= degree);
        bool is_steiner_v = (v_local >= degree);

        // Skip if both endpoints are original pins (not Steiner)
        // Note: Original code skipped if both were *pins* (is_pin_u &&
        // is_pin_v), which relied on pos2steiner_map.count. Let's stick to
        // Steiner check.
        if (!is_steiner_u && !is_steiner_v)
          continue;

        // Handle Steiner point connections using LOCAL indices for UFS
        if (is_steiner_v) {                           // v is steiner
          if (p1.x() != p2.x() && p1.y() != p2.y()) { // Diagonal branch
            steiner_adj_vertices_map[v_local].emplace_back(u_local);
          } else { // Manhattan branch
            ufs.unite(u_local, v_local, p1, p2);
          }
        }
        if (is_steiner_u) {                           // u is steiner
          if (p1.x() != p2.x() && p1.y() != p2.y()) { // Diagonal branch
            steiner_adj_vertices_map[u_local].emplace_back(v_local);
          } else { // Manhattan branch
            ufs.unite(v_local, u_local, p2, p1);
          }
        }
      }
      wl[netid] = cur_wl; // Scaled wirelength for the net

      // --- Calculate pin relations ---

      // Set pin relations for original pins
      for (const auto &[pos, indices] : pos2local_map) {
        int first_local_idx = indices[0];
        if (first_local_idx >= degree)
          continue; // Skip if the first index belongs to a Steiner point

        for (const auto &local_idx : indices) {
          if (local_idx >= degree) {
            net_result[netid].vtx_relate_x.at(local_idx) = first_local_idx;
            net_result[netid].vtx_relate_y.at(local_idx) = first_local_idx;
          } else {
            net_result[netid].vtx_relate_x.at(local_idx) = local_idx;
            net_result[netid].vtx_relate_y.at(local_idx) = local_idx;
          }

          if (local_idx != first_local_idx) {
            // Add branches for duplicate pins using LOCAL indices
            add_edge(local_idx, first_local_idx);
          }
        }
      }

      // Set pin relations for Steiner points NOT coinciding with original pins,
      // using UFS
      for (const auto &[pos, steiner_local_idx] : pos2steiner_map) {
        // Check if this location contained an original pin (already handled
        // above) Use the simplified check based on the first element
        bool handled_above = false;
        auto it = pos2local_map.find(pos);
        // Check if key exists, vector is not empty, AND the first element is an
        // original pin index
        if (it != pos2local_map.end() && !it->second.empty() &&
            it->second[0] < degree) {
          handled_above = true;
        }

        // Only process if not handled in the loop above (i.e., location has no
        // original pins)
        if (!handled_above) {

          // Now pass these named variables to the function
          auto [x_pin_local, y_pin_local] = ufs.getRelateVertex(
              steiner_local_idx, degree, steiner_adj_vertices_map,
              net_result[netid].newx, net_result[netid].newy);

          net_result[netid].vtx_relate_x.at(steiner_local_idx) = x_pin_local;
          net_result[netid].vtx_relate_y.at(steiner_local_idx) = y_pin_local;
        }
      }

      // output for duplicate pin positions (using local indices)
      if (duplicate_pin) {
        for (auto &[pos, indices] : pos2local_map) {
          if (indices.size() < 2)
            continue;
          std::cout << "Position (" << pos.x() << ", " << pos.y()
                    << ") has local indices: ";
          for (const auto &idx : indices) {
            std::cout << idx << " ";
          }
          std::cout << std::endl;
        }
      }
      net_result[netid].vtx_fa.resize(total_vertex_local);
      net_result[netid].flat_vtx_to_start.resize(total_vertex_local + 1);
      net_result[netid].net_flat_topo_idx.resize(total_vertex_local);
      total_steiner += num_steiner_points;
      free(ftree.branch);
    }
    auto topo_sort = [&net_result, &edge, &degree](int netid) {
      int vtx_degree = net_result[netid].num_steiner + degree;
      std::vector<bool> visit(vtx_degree, false);
      int root = 0;
      int topo_ptr = 0;
      visit[root] = true;
      net_result[netid].net_flat_topo_idx.at(root) = topo_ptr++;
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
          net_result[netid].net_flat_topo_idx.at(v) = topo_ptr++;
          net_result[netid].vtx_fa.at(v) = u;
        }
      }
      net_result[netid].flat_vtx_to_start.at(0) = 0;
      for (int local_idx = 0; local_idx < vtx_degree; ++local_idx) {
        int num_son = 0;
        for (auto v : edge[local_idx]) {
          if (v == -1)
            continue;
          ++num_son;
          net_result[netid].flat_vtx_to.push_back(v);
          if (netid == 255) {
            std::cout << "edge: " << local_idx << "("
                      << net_result[netid].newx[local_idx] << ", "
                      << net_result[netid].newy[local_idx] << ")" << " to " << v
                      << "(" << net_result[netid].newx[v] << ", "
                      << net_result[netid].newy[v] << ")" << std::endl;
          }
        }
        net_result[netid].flat_vtx_to_start.at(local_idx + 1) =
            net_result[netid].flat_vtx_to_start.at(local_idx) + num_son;
      }
    };
    topo_sort(netid);
  }
  newx.resize(num_pins + total_steiner);
  newy.resize(num_pins + total_steiner);
  vtx_relate_x.resize(num_pins + total_steiner);
  vtx_relate_y.resize(num_pins + total_steiner);
  vtx_fa.resize(num_pins + total_steiner);
  flat_vtx_from.resize(num_pins + total_steiner);
  flat_vtx_to.resize(num_pins + total_steiner);
  flat_vtx_to_start.resize(num_pins + total_steiner + 1);
  net_flat_topo_idx.resize(num_pins + total_steiner);
  netsteiner_start[0] = num_pins;
  for (int netid = 0, prefix_steiner = 0; netid < num_nets; ++netid) {
    int degree = netpin_start[netid + 1] - netpin_start[netid];
    int degree_steiner = net_result[netid].num_steiner;

    net_flat_topo_idx_start[netid + 1] =
        net_flat_topo_idx_start[netid] + net_result[netid].newx.size();
    netsteiner_start[netid + 1] = netsteiner_start[netid] + degree_steiner;

    auto local2global = [&degree, &net_result, &netid,
                         &netsteiner_start](int idx) {
      if (idx < degree) {
        return net_result[netid].local2global_idx.at(idx);
      } else {
        return netsteiner_start[netid] + idx - degree;
      }
    };
    auto merge_result = [&net_result, &newx, &newy, &vtx_relate_x,
                         &vtx_relate_y, &vtx_fa, &flat_vtx_to,
                         &flat_vtx_to_start, &net_flat_topo_idx, &netpin_start,
                         &prefix_steiner, &degree, &netsteiner_start,
                         &local2global](int netid, int from, int to) {
      newx[to] = net_result[netid].newx.at(from);
      newy[to] = net_result[netid].newy.at(from);
      vtx_relate_x.at(to) =
          local2global(net_result[netid].vtx_relate_x.at(from));
      vtx_relate_y.at(to) =
          local2global(net_result[netid].vtx_relate_y.at(from));
      vtx_fa[to] = local2global(net_result[netid].vtx_fa.at(from));
      flat_vtx_to_start[to] = net_result[netid].flat_vtx_to_start.at(from);
      for (int i = net_result[netid].flat_vtx_to_start.at(from);
           i < net_result[netid].flat_vtx_to_start.at(from + 1); ++i) {
        int prefix_vertices = netpin_start[netid];
        flat_vtx_to[prefix_vertices + i] =
            local2global(net_result[netid].flat_vtx_to.at(i));
      }
      net_flat_topo_idx[to] = net_result[netid].net_flat_topo_idx.at(from);
    };
    for (int pin = 0; pin < degree; ++pin) {
      int global_idx = net_result[netid].local2global_idx.at(pin);
      merge_result(netid, pin, global_idx);
    }
    for (int steiner = 0; steiner < degree_steiner; ++steiner) {
      int global_idx = num_pins + prefix_steiner + steiner;
      merge_result(netid, degree + steiner, global_idx);
    }
    prefix_steiner += degree_steiner;
    for (int i = 0; i < net_result[netid].newx.size(); ++i) {
      int begin_idx = net_result[netid].flat_vtx_to_start.at(i);
      int end_idx = net_result[netid].flat_vtx_to_start.at(i + 1);
      int from_vtx = local2global(i);
      std::fill(flat_vtx_from.begin() + begin_idx,
                flat_vtx_from.begin() + end_idx, from_vtx);
    }
  }
  std::cout << "Steiner tree generation completed." << std::endl;
  return 0;
}

template <typename T>
int computeSteinerPosLauncher(
    const T *cur_pos_x,
    const T *cur_pos_y, // cur pin positions (float/double)
    const std::vector<int>
        &vtx_relate_x, // Cached: LOCAL pin index this vertex relates to for X
    const std::vector<int>
        &vtx_relate_y, // Cached: LOCAL pin index this vertex relates to for Y
    const std::vector<int>
        &local2global_idx,   // Cached: Maps flat_pin_idx -> global_pin_idx
    const int *netpin_start, // Cached: Start index in flat_netpin for each net
    const int *netvertex_start, // Cached: Start index in newx/newy for each net
    const int num_nets,         // Number of nets
    const int num_pins,         // Total number of pins
    const int num_total_vertices, // Total vertices (pins + steiners) from cache
    const int num_threads,        // Number of threads
    std::vector<int> &updated_newx, // Output: Updated X coordinates (int)
    std::vector<int> &updated_newy) // Output: Updated Y coordinates (int)
{
  constexpr int scale = 1000;
  updated_newx.resize(
      num_total_vertices); // Ensure output vectors have correct size
  updated_newy.resize(num_total_vertices);

#pragma omp parallel for num_threads(num_threads)
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    int v_bgn = netvertex_start[net_id];
    int v_end = netvertex_start[net_id + 1];
    int pin_bgn = netpin_start[net_id]; // Base flat pin index for this net
    int degree = netpin_start[net_id + 1] - pin_bgn;

    // Temporary storage for cur pin coordinates for this net (scaled)
    // This avoids repeated lookups inside the vertex loop.
    std::vector<int> cur_pin_coords_x_scaled(degree);
    std::vector<int> cur_pin_coords_y_scaled(degree);
    for (int local_pin_idx = 0; local_pin_idx < degree; ++local_pin_idx) {
      int flat_pin_idx = pin_bgn + local_pin_idx;
      int global_pin_idx = local2global_idx[flat_pin_idx];
      cur_pin_coords_x_scaled[local_pin_idx] =
          static_cast<int>(cur_pos_x[global_pin_idx] * scale);
      cur_pin_coords_y_scaled[local_pin_idx] =
          static_cast<int>(cur_pos_y[global_pin_idx] * scale);
    }

    // Iterate through all vertices (pins + steiner) belonging to this net
    for (int k = v_bgn; k < v_end; ++k) { // k is the GLOBAL vertex index
      int local_vertex_idx =
          k - v_bgn; // Local index within this net (0 to total_vertex_local-1)

      if (local_vertex_idx < degree) {
        // --- This is an original pin ---
        // Its coordinate comes directly from the cur scaled pin coordinates
        updated_newx[k] = cur_pin_coords_x_scaled[local_vertex_idx];
        updated_newy[k] = cur_pin_coords_y_scaled[local_vertex_idx];
      } else {
        // --- This is a Steiner point ---
        // Its coordinate is determined by the pin it relates to.
        // vtx_relate_x.at(k) gives the LOCAL index (0 to degree-1) of the
        // related pin.
        int related_local_pin_idx_x = vtx_relate_x.at(k);
        int related_local_pin_idx_y = vtx_relate_y.at(k);

        assert(related_local_pin_idx_x >= 0 &&
               related_local_pin_idx_x < degree);
        assert(related_local_pin_idx_y >= 0 &&
               related_local_pin_idx_y < degree);

        // Get the cur scaled coordinate of the related pin
        updated_newx[k] = cur_pin_coords_x_scaled[related_local_pin_idx_x];
        updated_newy[k] = cur_pin_coords_y_scaled[related_local_pin_idx_y];
      }
    }
  }

  return 0;
}

template <typename T>
int computeSteinerTopoGradLauncher(
    T *grad_vertices_x, T *grad_vertices_y, const int num_nets,
    const int num_pins, const std::vector<int> &vtx_relate_x,
    const std::vector<int> &vtx_relate_y, const int *netpin_start,
    const int *netvertex_start, const std::vector<int> &local2global_idx,
    const int num_threads, T *grad_pos_x, T *grad_pos_y) {
#pragma omp parallel for num_threads(num_threads)
  for (int net_id = 0; net_id < num_nets; ++net_id) {
    int v_bgn = netvertex_start[net_id];
    int v_end = netvertex_start[net_id + 1];
    int pin_bgn = netpin_start[net_id];

    // Iterate through all vertices (pins + steiner) belonging to this net
    for (int k = v_bgn; k < v_end; ++k) {
      T &g_vertex_x = grad_vertices_x[k];
      T &g_vertex_y = grad_vertices_y[k];

      // --- Handle X gradient ---
      int local2global_abs_idx_x = pin_bgn + vtx_relate_x.at(k);
      int global_pin_idx_x = local2global_idx[local2global_abs_idx_x];
      assert(global_pin_idx_x < num_pins);
#pragma omp atomic update
      grad_pos_x[global_pin_idx_x] += g_vertex_x;

      // --- Handle Y gradient ---
      int local2global_abs_idx_y = pin_bgn + vtx_relate_y.at(k);
      int global_pin_idx_y = local2global_idx[local2global_abs_idx_y];
      assert(global_pin_idx_y < num_pins);
#pragma omp atomic update
      grad_pos_y[global_pin_idx_y] += g_vertex_y;
    }
  }

  return 0;
}

std::vector<at::Tensor> build_tree(at::Tensor pos, at::Tensor flat_netpin,
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
  const int num_pins = pos.numel() / 2; // Total number of pins in the design

  // DEBUG
  std::cout << "pos.size / 2 = " << pos.numel() / 2 << std::endl;
  std::cout << "flat_netpin.size = " << flat_netpin.numel() << std::endl;

  // Initialize output vectors
  std::vector<int> newx_vec; // Will contain scaled coordinates
  std::vector<int> newy_vec;
  std::vector<int> vtx_relate_x_vec;
  std::vector<int> vtx_relate_y_vec;
  std::vector<int>(flat_netpin.numel());
  std::vector<int> vtx_fa_vec;
  std::vector<int> flat_vtx_to_vec;
  std::vector<int> flat_vtx_from_vec;
  std::vector<int> net_flat_topo_idx_vec;
  std::vector<int> flat_vtx_to_start_vec;

  // Initialize output tensors for net start indices
  auto options_int = torch::dtype(torch::kInt32).device(torch::kCPU);
  auto net_vertex_start = at::zeros({num_nets + 1}, options_int);
  auto wl = at::zeros({num_nets + 1}, options_int);
  auto net_steiner_start = at::zeros({num_nets + 1}, options_int);
  auto net_flat_topo_idx_start_tensor = at::zeros({num_nets + 1}, options_int);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeSteinerTreeLauncher", [&] {
    computeSteinerTreeLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int), num_nets, num_pins,
        ignore_net_degree, DREAMPLACE_TENSOR_DATA_PTR(wl, int), newx_vec,
        newy_vec, vtx_relate_x_vec, vtx_relate_y_vec,
        DREAMPLACE_TENSOR_DATA_PTR(net_steiner_start, int), vtx_fa_vec,
        flat_vtx_to_vec, flat_vtx_from_vec, net_flat_topo_idx_vec,
        flat_vtx_to_start_vec,
        DREAMPLACE_TENSOR_DATA_PTR(net_flat_topo_idx_start_tensor, int));
  });

  int num_total_vertices = newx_vec.size();
  int num_flat_vtx_to = flat_vtx_to_vec.size();
  int num_flat_topo_sort = net_flat_topo_idx_vec.size();

  auto newx =
      at::from_blob(newx_vec.data(), {num_total_vertices}, options_int).clone();
  auto newy =
      at::from_blob(newy_vec.data(), {num_total_vertices}, options_int).clone();
  auto vtx_relate_x =
      at::from_blob(vtx_relate_x_vec.data(), {num_total_vertices}, options_int)
          .clone();
  auto vtx_relate_y =
      at::from_blob(vtx_relate_y_vec.data(), {num_total_vertices}, options_int)
          .clone();
  auto vtx_fa =
      at::from_blob(vtx_fa_vec.data(), {num_total_vertices}, options_int)
          .clone();
  auto flat_vtx_to =
      at::from_blob(flat_vtx_to_vec.data(), {num_flat_vtx_to}, options_int)
          .clone();
  auto flat_vtx_from =
      at::from_blob(flat_vtx_from_vec.data(), {num_flat_vtx_to}, options_int)
          .clone();
  auto net_flat_topo_idx = at::from_blob(net_flat_topo_idx_vec.data(),
                                         {num_flat_topo_sort}, options_int)
                               .clone();
  // Convert the populated vector to a tensor
  auto flat_vtx_to_start_tensor =
      at::from_blob(flat_vtx_to_start_vec.data(),
                    {(long)flat_vtx_to_start_vec.size()}, options_int)
          .clone();

  // DEBUG
  std::cout << "return" << std::endl;
  return {newx,
          newy,
          vtx_relate_x,
          vtx_relate_y,
          local2global_idx,
          net_vertex_start,
          net_steiner_start,
          vtx_fa,
          flat_vtx_to,
          flat_vtx_from,
          flat_vtx_to_start_tensor,
          net_flat_topo_idx,
          net_flat_topo_idx_start_tensor};
}

std::vector<at::Tensor> steiner_topo_forward(at::Tensor cur_pos,
                                             at::Tensor cached_vtx_relate_x,
                                             at::Tensor cached_vtx_relate_y,
                                             at::Tensor cached_local2global_idx,
                                             at::Tensor cached_netpin_start,
                                             at::Tensor cached_netvertex_start,
                                             int num_total_vertices) {
  CHECK_FLAT_CPU(cur_pos);
  CHECK_EVEN(cur_pos);
  CHECK_CONTIGUOUS(cur_pos);
  CHECK_FLAT_CPU(cached_vtx_relate_x);
  CHECK_CONTIGUOUS(cached_vtx_relate_x);
  CHECK_FLAT_CPU(cached_vtx_relate_y);
  CHECK_CONTIGUOUS(cached_vtx_relate_y);
  CHECK_FLAT_CPU(cached_local2global_idx);
  CHECK_CONTIGUOUS(cached_local2global_idx);
  CHECK_FLAT_CPU(cached_netpin_start);
  CHECK_CONTIGUOUS(cached_netpin_start);
  CHECK_FLAT_CPU(cached_netvertex_start);
  CHECK_CONTIGUOUS(cached_netvertex_start);

  const int num_pins = cur_pos.numel() / 2;
  const int num_nets = cached_netpin_start.numel() - 1;

  std::vector<int> updated_newx_vec;
  std::vector<int> updated_newy_vec;

  std::vector<int> vtx_relate_x_vec(
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_x, int),
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_x, int) +
          num_total_vertices);
  std::vector<int> vtx_relate_y_vec(
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_y, int),
      DREAMPLACE_TENSOR_DATA_PTR(cached_vtx_relate_y, int) +
          num_total_vertices);
  std::vector<int> local2global_idx_vec(
      DREAMPLACE_TENSOR_DATA_PTR(cached_local2global_idx, int),
      DREAMPLACE_TENSOR_DATA_PTR(cached_local2global_idx, int) +
          cached_local2global_idx.numel());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(cur_pos, "computeSteinerPosLauncher", [&] {
    computeSteinerPosLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(cur_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(cur_pos, scalar_t) + num_pins,
        vtx_relate_x_vec, vtx_relate_y_vec, local2global_idx_vec,
        DREAMPLACE_TENSOR_DATA_PTR(cached_netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(cached_netvertex_start, int), num_nets,
        num_pins, num_total_vertices, at::get_num_threads(), updated_newx_vec,
        updated_newy_vec);
  });

  auto options_int = torch::dtype(torch::kInt32).device(torch::kCPU);
  auto updated_newx =
      torch::from_blob(updated_newx_vec.data(),
                       {static_cast<long>(updated_newx_vec.size())},
                       options_int)
          .clone();
  auto updated_newy =
      torch::from_blob(updated_newy_vec.data(),
                       {static_cast<long>(updated_newy_vec.size())},
                       options_int)
          .clone();

  return {updated_newx, updated_newy};
}

at::Tensor steiner_topo_backward(at::Tensor grad_vertices, at::Tensor pos,
                                 at::Tensor vtx_relate_x,
                                 at::Tensor vtx_relate_y,
                                 at::Tensor netpin_start,
                                 at::Tensor netvertex_start,
                                 at::Tensor local2global_idx) {

  // Input validation (same as before)
  CHECK_FLAT_CPU(grad_vertices);
  CHECK_EVEN(grad_vertices);
  CHECK_CONTIGUOUS(grad_vertices);
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(vtx_relate_x);
  CHECK_CONTIGUOUS(vtx_relate_x);
  CHECK_FLAT_CPU(vtx_relate_y);
  CHECK_CONTIGUOUS(vtx_relate_y);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(netvertex_start);
  CHECK_CONTIGUOUS(netvertex_start);
  CHECK_FLAT_CPU(local2global_idx);
  CHECK_CONTIGUOUS(local2global_idx);

  // Allocate output tensor for gradients w.r.t. original positions
  auto grad_pos = at::zeros_like(pos);

  int num_total_vertices = grad_vertices.numel() / 2;
  int num_pins = pos.numel() / 2;
  int num_nets = netpin_start.numel() - 1;

  std::vector<int> vtx_relate_x_vec(
      DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_x, int),
      DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_x, int) + num_total_vertices);
  std::vector<int> vtx_relate_y_vec(
      DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_y, int),
      DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_y, int) + num_total_vertices);
  std::vector<int> local2global_idx_vec(
      DREAMPLACE_TENSOR_DATA_PTR(local2global_idx, int),
      DREAMPLACE_TENSOR_DATA_PTR(local2global_idx, int) +
          local2global_idx.numel());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeSteinerTopoGradLauncher", [&] {
        computeSteinerTopoGradLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad_vertices, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_vertices, scalar_t) +
                num_total_vertices,
            num_nets, num_pins, vtx_relate_x_vec, vtx_relate_y_vec,
            DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
            DREAMPLACE_TENSOR_DATA_PTR(netvertex_start, int),
            local2global_idx_vec, at::get_num_threads(),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pos, scalar_t) + num_pins);
      });

  return grad_pos;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::steiner_topo_forward,
        "SteinerTopo forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::steiner_topo_backward,
        "SteinerTopo backward");
  m.def("build_tree", &DREAMPLACE_NAMESPACE::build_tree, "Build Tree");
}