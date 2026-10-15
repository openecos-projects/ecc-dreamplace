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
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <omp.h>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <filesystem>
#include "frozen_net_topology.h"

void bindLDirectionMatcher(pybind11::module_& m);

DREAMPLACE_BEGIN_NAMESPACE

namespace {

void loadFluteLut(const std::string &powv_file,
                  const std::string &post_file) {
  TORCH_CHECK(std::filesystem::is_regular_file(powv_file),
              "Flute POWV LUT file does not exist: ", powv_file);
  TORCH_CHECK(std::filesystem::is_regular_file(post_file),
              "Flute POST LUT file does not exist: ", post_file);
  static std::once_flag load_lut_once;
  std::call_once(load_lut_once, [&] {
    flute::readLUT(powv_file.c_str(), post_file.c_str());
  });
}

}  // namespace

void checkFlatCpuContiguous(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), name, " must reside on CPU");
  TORCH_CHECK(tensor.dim() == 1, name, " must be a flat tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void checkFloatingPosTensor(const at::Tensor &tensor, const char *name) {
  checkFlatCpuContiguous(tensor, name);
  TORCH_CHECK(tensor.numel() % 2 == 0, name,
              " must have an even number of elements");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat ||
                  tensor.scalar_type() == at::kDouble,
              name, " must have dtype torch.float32 or torch.float64");
}

void checkRelationTensor(const at::Tensor &relation, const char *name,
                         int64_t expected_size, int num_pins) {
  checkFlatCpuContiguous(relation, name);
  TORCH_CHECK(relation.scalar_type() == at::kInt, name,
              " must have dtype torch.int32");
  TORCH_CHECK(relation.numel() == expected_size, name, " length ",
              relation.numel(), " does not match expected vertex count ",
              expected_size);

  const int *relation_ptr = relation.data_ptr<int>();
  for (int64_t vertex_id = 0; vertex_id < expected_size; ++vertex_id) {
    const int pin_id = relation_ptr[vertex_id];
    TORCH_CHECK(pin_id >= 0 && pin_id < num_pins, name, "[", vertex_id,
                "] = ", pin_id, " is outside [0, ", num_pins, ")");
  }
}

void checkBuildInputs(const at::Tensor &pos, const at::Tensor &flat_netpin,
                      const at::Tensor &netpin_start) {
  checkFloatingPosTensor(pos, "pos");
  checkFlatCpuContiguous(flat_netpin, "flat_netpin");
  checkFlatCpuContiguous(netpin_start, "netpin_start");
  TORCH_CHECK(flat_netpin.scalar_type() == at::kInt,
              "flat_netpin must have dtype torch.int32");
  TORCH_CHECK(netpin_start.scalar_type() == at::kInt,
              "netpin_start must have dtype torch.int32");
  TORCH_CHECK(netpin_start.numel() >= 2,
              "netpin_start must contain at least one net interval");

  const int64_t num_pins = pos.numel() / 2;
  TORCH_CHECK(num_pins <= std::numeric_limits<int>::max(),
              "pos has too many pins for the int32 native topology API");
  const int *flat_netpin_ptr = flat_netpin.data_ptr<int>();
  const int *netpin_start_ptr = netpin_start.data_ptr<int>();
  TORCH_CHECK(netpin_start_ptr[0] == 0,
              "netpin_start must begin at zero");
  TORCH_CHECK(netpin_start_ptr[netpin_start.numel() - 1] == flat_netpin.numel(),
              "netpin_start must end at flat_netpin length ", flat_netpin.numel());
  std::vector<int> global_pin_owner(num_pins, -1);
  for (int64_t net_id = 0; net_id + 1 < netpin_start.numel(); ++net_id) {
    const int begin = netpin_start_ptr[net_id];
    const int end = netpin_start_ptr[net_id + 1];
    TORCH_CHECK(begin >= 0 && begin < end && end <= flat_netpin.numel(),
                "netpin_start has invalid interval [", begin, ", ", end,
                ") for net ", net_id);
    std::unordered_set<int> net_pin_ids;
    for (int local_idx = begin; local_idx < end; ++local_idx) {
      const int pin_id = flat_netpin_ptr[local_idx];
      TORCH_CHECK(pin_id >= 0 && pin_id < num_pins, "flat_netpin[",
                  local_idx, "] = ", pin_id, " is outside [0, ", num_pins,
                  ")");
      TORCH_CHECK(net_pin_ids.insert(pin_id).second, "flat_netpin repeats "
                  "global pin ", pin_id, " within net ", net_id);
      TORCH_CHECK(global_pin_owner[pin_id] == -1, "flat_netpin global pin ",
                  pin_id, " appears in multiple nets: ",
                  global_pin_owner[pin_id], " and ", net_id);
      global_pin_owner[pin_id] = static_cast<int>(net_id);
    }
  }
}

template <typename T>
int computeSteinerTreeLauncher(
    T *x, T *y, int *flat_netpin, int *netpin_start, int num_nets, int num_pins,
    int ignore_net_degree, int *wl, std::vector<T> &newx,
    std::vector<T> &newy, std::vector<int> &vtx_relate_x,
    std::vector<int> &vtx_relate_y, int *netsteiner_start,
    std::vector<int> &vtx_fa, std::vector<int> &flat_vtx_to,
    std::vector<int> &flat_vtx_from, std::vector<int> &net_flat_topo_idx,
    std::vector<int> &flat_vtx_to_start, int *net_flat_topo_idx_start,
    bool deterministic_flag, std::vector<NetResult> net_result) {

  constexpr int scale = 1000;
  int total_steiner = 0;

#pragma omp parallel for reduction(+ : total_steiner) if(!deterministic_flag)
  for (int netid = 0; netid < num_nets; ++netid) {
    NetResult &result = net_result[netid];
    if (!result.net_flat_topo_idx.empty()) {
      total_steiner += result.num_steiner;
      continue;  // Buffered nets retain their entire tree, not just active edges.
    }
    int degree = netpin_start[netid + 1] - netpin_start[netid];
    if (degree <= 0) {
      result.error = "SteinerTopo build requires at least one real pin for net " +
                     std::to_string(netid);
      continue;
    }

    // Keep physical sites separate from synthesized Steiner locations.  The
    // former defines stable terminal ownership; the latter never does.
    std::map<Point<int>, std::vector<int>> pin_sites;
    std::vector<int> vx, vy;
    vx.reserve(degree);
    vy.reserve(degree);
    result.local2global_idx.resize(degree);
    const auto quantize_lattice_coordinate = [](double scaled_coordinate,
                                                int *quantized_coordinate) {
      if (!std::isfinite(scaled_coordinate) ||
          scaled_coordinate < std::numeric_limits<int>::min() - 0.5 ||
          scaled_coordinate > std::numeric_limits<int>::max() + 0.5) {
        return false;
      }
      const long long rounded_coordinate = std::llround(scaled_coordinate);
      if (rounded_coordinate < std::numeric_limits<int>::min() ||
          rounded_coordinate > std::numeric_limits<int>::max()) {
        return false;
      }
      *quantized_coordinate = static_cast<int>(rounded_coordinate);
      return true;
    };
    for (int cur_local_idx = 0; cur_local_idx < degree; ++cur_local_idx) {
      int pin_global_idx = flat_netpin[netpin_start[netid] + cur_local_idx];
      const T raw_x = x[pin_global_idx];
      const T raw_y = y[pin_global_idx];
      const double scaled_x = static_cast<double>(raw_x) * scale;
      const double scaled_y = static_cast<double>(raw_y) * scale;
      int quantized_x = 0;
      int quantized_y = 0;
      if (!quantize_lattice_coordinate(scaled_x, &quantized_x) ||
          !quantize_lattice_coordinate(scaled_y, &quantized_y)) {
        std::ostringstream message;
        message << "SteinerTopo build received a non-finite or out-of-range "
                << "coordinate: net=" << netid << ", local_pin="
                << cur_local_idx << ", global_pin=" << pin_global_idx
                << ", raw=(" << raw_x << ", " << raw_y << ")"
                << ", scaled=(" << scaled_x << ", " << scaled_y << ")";
        result.error = message.str();
        break;
      }

      // The integer FLUTE tree is a frozen quantized approximation.  Raw
      // continuous positions may remain elsewhere in the same lattice cell;
      // forward/backward gather those raw values through direct pin witnesses.
      Point<int> point(quantized_x, quantized_y);
      result.local2global_idx[cur_local_idx] = pin_global_idx;
      std::vector<int> &site_members = pin_sites[point];
      if (site_members.empty()) {
        vx.push_back(point.x());
        vy.push_back(point.y());
      }
      site_members.push_back(cur_local_idx);
      result.newx.push_back(point.x());
      result.newy.push_back(point.y());
    }
    if (!result.error.empty()) {
      continue;
    }

    std::vector<int> canonical_for_local(degree, -1);
    for (auto &[point, site_members] : pin_sites) {
      std::sort(site_members.begin(), site_members.end(),
                [&result](int lhs, int rhs) {
                  return result.local2global_idx[lhs] <
                         result.local2global_idx[rhs];
                });
      const auto driver_it =
          std::find(site_members.begin(), site_members.end(), 0);
      const int canonical_local =
          driver_it == site_members.end() ? site_members.front() : *driver_it;
      for (const int local_idx : site_members) {
        canonical_for_local[local_idx] = canonical_local;
      }
    }

    int num_valid_pins = pin_sites.size();
    std::vector<std::vector<int>> edge(degree);
    auto add_edge = [&edge](int u, int v) {
      edge[u].push_back(v);
      edge[v].push_back(u);
    };

    auto publish_real_pin_expansion = [&]() {
      for (const auto &[point, site_members] : pin_sites) {
        const int canonical_local = canonical_for_local[site_members.front()];
        for (const int local_idx : site_members) {
          result.vtx_relate_x[local_idx] = local_idx;
          result.vtx_relate_y[local_idx] = local_idx;
          if (local_idx != canonical_local) {
            add_edge(local_idx, canonical_local);
          }
        }
      }
    };

    if (num_valid_pins == 1) {
      // --- net with only one unique pin location ---
      wl[netid] = 0;
      result.vtx_fa.resize(degree);
      result.net_flat_topo_idx.resize(degree);
      result.vtx_relate_x.resize(degree);
      result.vtx_relate_y.resize(degree);
      publish_real_pin_expansion();
    } else {
      // --- nets with >= 2 unique pin locations ---
      flute::Tree ftree =
          flute::flute(num_valid_pins, vx.data(), vy.data(), ACCURACY);
      std::map<Point<int>, int> pos2steiner_map;
      int num_steiner_points = 0;

      for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
        flute::Branch &b = ftree.branch[bid];
        Point<int> p(b.x, b.y);
        auto it = pin_sites.find(p);
        bool is_original_pin_loc =
            (it != pin_sites.end() && !it->second.empty());
        if (!is_original_pin_loc &&
            pos2steiner_map.find(p) == pos2steiner_map.end()) {
          // It's a new Steiner point location
          int steiner_local_idx = degree + num_steiner_points++;
          pos2steiner_map[p] = steiner_local_idx;
          result.newx.push_back(b.x);
          result.newy.push_back(b.y);
        }
      }
      result.num_steiner = num_steiner_points;
      total_steiner += result.num_steiner;

      const int total_vertex_local = degree + result.num_steiner;
      UnifiedUFS<int> ufs(total_vertex_local);

      edge.resize(total_vertex_local);
      result.vtx_relate_x.resize(total_vertex_local);
      result.vtx_relate_y.resize(total_vertex_local);
      result.vtx_fa.resize(total_vertex_local);
      result.net_flat_topo_idx.resize(total_vertex_local);

      // store adjacency for Steiner diagonal connections
      std::map<int, std::vector<int>> steiner_adj_vertices_map;
      int cur_wl = 0;

      auto local_for_point = [&pin_sites, &pos2steiner_map,
                              &canonical_for_local](const Point<int> &point) {
        const auto pin_it = pin_sites.find(point);
        if (pin_it != pin_sites.end()) {
          return canonical_for_local[pin_it->second.front()];
        }
        const auto steiner_it = pos2steiner_map.find(point);
        return steiner_it == pos2steiner_map.end() ? -1 : steiner_it->second;
      };

      // --- construct relate ---
      for (int bid = 0; bid < 2 * ftree.deg - 2; ++bid) {
        flute::Branch &b1 = ftree.branch[bid];
        flute::Branch &b2 = ftree.branch[b1.n];

        Point<int> p1(b1.x, b1.y);
        Point<int> p2(b2.x, b2.y);

        if (p1 == p2)
          continue;

        int u_local = local_for_point(p1);
        int v_local = local_for_point(p2);
        if (u_local < 0 || v_local < 0) {
          std::ostringstream message;
          message << "SteinerTopo could not resolve a FLUTE branch endpoint: "
                  << "net=" << netid << ", branch=" << bid << ", p1=("
                  << p1.x() << ", " << p1.y() << "), p2=(" << p2.x()
                  << ", " << p2.y() << ")";
          result.error = message.str();
          break;
        }
        add_edge(u_local, v_local);

        cur_wl += std::abs(result.newx[u_local] - result.newx[v_local]) +
                  std::abs(result.newy[u_local] - result.newy[v_local]);

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
      free(ftree.branch);
      if (!result.error.empty()) {
        continue;
      }
      wl[netid] = cur_wl;

      publish_real_pin_expansion();
      std::vector<int> raw_relate_x(total_vertex_local, -1);
      std::vector<int> raw_relate_y(total_vertex_local, -1);
      for (int local_idx = 0; local_idx < degree; ++local_idx) {
        raw_relate_x[local_idx] = local_idx;
        raw_relate_y[local_idx] = local_idx;
      }
      for (const auto &[point, steiner_local] : pos2steiner_map) {
        const auto [x_candidate, y_candidate] = ufs.getRelateVertex(
            steiner_local, degree, steiner_adj_vertices_map, result.newx,
            result.newy);
        raw_relate_x[steiner_local] = x_candidate;
        raw_relate_y[steiner_local] = y_candidate;
      }

      std::map<int, std::vector<int>> x_witnesses;
      std::map<int, std::vector<int>> y_witnesses;
      for (const auto &[point, site_members] : pin_sites) {
        const int canonical_local = canonical_for_local[site_members.front()];
        x_witnesses[point.x()].push_back(canonical_local);
        y_witnesses[point.y()].push_back(canonical_local);
      }
      auto sort_witnesses = [&result](std::map<int, std::vector<int>> &table) {
        for (auto &[coordinate, candidates] : table) {
          std::sort(candidates.begin(), candidates.end(), [&result](int lhs,
                                                                    int rhs) {
            return result.local2global_idx[lhs] <
                   result.local2global_idx[rhs];
          });
          candidates.erase(std::unique(candidates.begin(), candidates.end()),
                           candidates.end());
        }
      };
      sort_witnesses(x_witnesses);
      sort_witnesses(y_witnesses);

      auto select_direct_witness =
          [&](int steiner_local, int raw_candidate, bool x_axis) {
            const int cached_coordinate =
                x_axis ? result.newx[steiner_local] : result.newy[steiner_local];
            const int raw_coordinate =
                raw_candidate >= 0 && raw_candidate < degree
                    ? (x_axis ? result.newx[raw_candidate]
                              : result.newy[raw_candidate])
                    : std::numeric_limits<int>::min();
            if (raw_candidate >= 0 && raw_candidate < degree &&
                raw_coordinate == cached_coordinate) {
              return canonical_for_local[raw_candidate];
            }

            const auto &witnesses = x_axis ? x_witnesses : y_witnesses;
            const auto witness_it = witnesses.find(cached_coordinate);
            if (witness_it == witnesses.end() || witness_it->second.empty()) {
              std::ostringstream message;
              message << "SteinerTopo direct witness unavailable: net=" << netid
                      << ", local_vertex=" << steiner_local << ", axis="
                      << (x_axis ? "x" : "y") << ", cached_coordinate="
                      << cached_coordinate << ", raw_ufs_candidate="
                      << raw_candidate << ", candidate_count=0";
              result.error = message.str();
              return -1;
            }

            const std::vector<int> &candidates = witness_it->second;
            const auto best_candidate = std::min_element(
                candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
                  const int lhs_other =
                      x_axis ? result.newy[lhs] : result.newx[lhs];
                  const int rhs_other =
                      x_axis ? result.newy[rhs] : result.newx[rhs];
                  const int steiner_other =
                      x_axis ? result.newy[steiner_local]
                             : result.newx[steiner_local];
                  const long long lhs_delta =
                      static_cast<long long>(lhs_other) - steiner_other;
                  const long long rhs_delta =
                      static_cast<long long>(rhs_other) - steiner_other;
                  const long long lhs_distance =
                      lhs_delta < 0 ? -lhs_delta : lhs_delta;
                  const long long rhs_distance =
                      rhs_delta < 0 ? -rhs_delta : rhs_delta;
                  if (lhs_distance != rhs_distance) {
                    return lhs_distance < rhs_distance;
                  }
                  return result.local2global_idx[lhs] <
                         result.local2global_idx[rhs];
                });
            return *best_candidate;
          };

      for (const auto &[point, steiner_local] : pos2steiner_map) {
        result.vtx_relate_x[steiner_local] =
            select_direct_witness(steiner_local, raw_relate_x[steiner_local],
                                  true);
        if (!result.error.empty()) {
          break;
        }
        result.vtx_relate_y[steiner_local] =
            select_direct_witness(steiner_local, raw_relate_y[steiner_local],
                                  false);
        if (!result.error.empty()) {
          break;
        }
      }
    }

    if (!result.error.empty()) {
      continue;
    }

    const int total_vertex_local = degree + result.num_steiner;
    if (static_cast<int>(result.vtx_relate_x.size()) != total_vertex_local ||
        static_cast<int>(result.vtx_relate_y.size()) != total_vertex_local) {
      std::ostringstream message;
      message << "SteinerTopo relation length mismatch before cache publish: net="
              << netid << ", vertices=" << total_vertex_local
              << ", relation_x=" << result.vtx_relate_x.size()
              << ", relation_y=" << result.vtx_relate_y.size();
      result.error = message.str();
      continue;
    }
    for (int local_vertex = 0; local_vertex < total_vertex_local;
         ++local_vertex) {
      const auto validate_axis = [&](int relation_local, bool x_axis) {
        const int cached_coordinate =
            x_axis ? result.newx[local_vertex] : result.newy[local_vertex];
        if (relation_local < 0 || relation_local >= degree) {
          std::ostringstream message;
          message << "SteinerTopo published a non-real relation: net=" << netid
                  << ", local_vertex=" << local_vertex << ", axis="
                  << (x_axis ? "x" : "y") << ", cached_coordinate="
                  << cached_coordinate << ", relation=" << relation_local
                  << ", real_pin_count=" << degree;
          result.error = message.str();
          return;
        }
        const int relation_coordinate =
            x_axis ? result.newx[relation_local] : result.newy[relation_local];
        if (relation_coordinate != cached_coordinate) {
          std::ostringstream message;
          message << "SteinerTopo published an axis-mismatched relation: net="
                  << netid << ", local_vertex=" << local_vertex << ", axis="
                  << (x_axis ? "x" : "y") << ", cached_coordinate="
                  << cached_coordinate << ", relation=" << relation_local
                  << ", relation_coordinate=" << relation_coordinate;
          result.error = message.str();
          return;
        }
      };
      validate_axis(result.vtx_relate_x[local_vertex], true);
      if (!result.error.empty()) {
        break;
      }
      validate_axis(result.vtx_relate_y[local_vertex], false);
      if (!result.error.empty()) {
        break;
      }
    }
    if (!result.error.empty()) {
      continue;
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

  for (int netid = 0; netid < num_nets; ++netid) {
    TORCH_CHECK(net_result[netid].error.empty(), net_result[netid].error);
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
      newx[global_idx] = static_cast<T>(net_result[netid].newx[local_id]) / static_cast<T>(scale);
      newy[global_idx] = static_cast<T>(net_result[netid].newy[local_id]) / static_cast<T>(scale);
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
  
  return 0;
}

template <typename T>
int computeSteinerPosLauncher(const T *pin_pos_x, const T *pin_pos_y,
                              const std::vector<int> &vtx_relate_x,
                              const std::vector<int> &vtx_relate_y,
                              const int num_vertices,
                              std::vector<T> &updated_newx, std::vector<T> &updated_newy,
                              bool deterministic_flag) {
  updated_newx.resize(num_vertices);
  updated_newy.resize(num_vertices);

#pragma omp parallel for if(!deterministic_flag)
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
                                   int ignore_net_degree,
                                   const std::string &powv_file,
                                   const std::string &post_file,
                                   bool deterministic_flag,
                                   const std::vector<int>& frozen_net_ids,
                                   const std::vector<at::Tensor>& previous_cache) {
  checkBuildInputs(pos, flat_netpin, netpin_start);
  TORCH_CHECK(netpin_start.numel() - 1 <= std::numeric_limits<int>::max(),
              "netpin_start has too many nets for the int32 native topology API");
  loadFluteLut(powv_file, post_file);

  const int num_nets = static_cast<int>(netpin_start.numel() - 1);
  const int num_pins = static_cast<int>(pos.numel() / 2);
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
    auto retained = restoreFrozenNets(
        frozen_net_ids, previous_cache, pos.data_ptr<scalar_t>(),
        pos.data_ptr<scalar_t>() + num_pins, flat_netpin.data_ptr<int>(),
        netpin_start.data_ptr<int>(), num_nets, num_pins);

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
        DREAMPLACE_TENSOR_DATA_PTR(net_flat_topo_idx_start_tensor, int),
        deterministic_flag, std::move(retained));

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

  if (!frozen_net_ids.empty())
    result.push_back(frozenVertexMap(frozen_net_ids, previous_cache, result, num_pins));

  return result;
}

std::vector<at::Tensor> steiner_topo_forward(at::Tensor pin_pos,
                                             at::Tensor cached_vtx_relate_x,
                                             at::Tensor cached_vtx_relate_y,
                                             int num_vertices,
                                             bool deterministic_flag) {
  checkFloatingPosTensor(pin_pos, "pin_pos");
  TORCH_CHECK(num_vertices >= 0, "num_vertices must be non-negative");
  TORCH_CHECK(pin_pos.numel() / 2 <= std::numeric_limits<int>::max(),
              "pin_pos has too many pins for the int32 native topology API");

  const int num_pins = static_cast<int>(pin_pos.numel() / 2);
  checkRelationTensor(cached_vtx_relate_x, "cached_vtx_relate_x",
                      num_vertices, num_pins);
  checkRelationTensor(cached_vtx_relate_y, "cached_vtx_relate_y",
                      num_vertices, num_pins);
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
        updated_newy_vec,
        deterministic_flag);

    auto updated_newx = convertVecToTens(updated_newx_vec, options_float);
    auto updated_newy = convertVecToTens(updated_newy_vec, options_float);

    result = {updated_newx, updated_newy};
  });

  return result;
}

at::Tensor steiner_topo_backward(at::Tensor grad_newx, at::Tensor grad_newy,
                                 at::Tensor pos, at::Tensor vtx_relate_x,
                                 at::Tensor vtx_relate_y) {
  checkFlatCpuContiguous(grad_newx, "grad_newx");
  checkFlatCpuContiguous(grad_newy, "grad_newy");
  checkFloatingPosTensor(pos, "pos");
  TORCH_CHECK(grad_newx.scalar_type() == pos.scalar_type(),
              "grad_newx dtype must match pos dtype");
  TORCH_CHECK(grad_newy.scalar_type() == pos.scalar_type(),
              "grad_newy dtype must match pos dtype");
  TORCH_CHECK(grad_newx.numel() == grad_newy.numel(),
              "grad_newx and grad_newy lengths must match");
  TORCH_CHECK(pos.numel() / 2 <= std::numeric_limits<int>::max(),
              "pos has too many pins for the int32 native topology API");

  auto grad_pin = at::zeros_like(pos);

  const int64_t num_vertices = grad_newx.numel();
  const int num_pins = static_cast<int>(pos.numel() / 2);
  checkRelationTensor(vtx_relate_x, "vtx_relate_x", num_vertices, num_pins);
  checkRelationTensor(vtx_relate_y, "vtx_relate_y", num_vertices, num_pins);
  TORCH_CHECK(num_vertices <= std::numeric_limits<int>::max(),
              "gradient has too many vertices for the int32 native topology API");

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pos, "computeSteinerTopoGradLauncher", [&] {
        computeSteinerTopoGradLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(grad_newx, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_newy, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_x, int),
            DREAMPLACE_TENSOR_DATA_PTR(vtx_relate_y, int),
            static_cast<int>(num_vertices),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pin, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_pin, scalar_t) + num_pins);
      });

  return grad_pin;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bindLDirectionMatcher(m);
  m.def("forward", &DREAMPLACE_NAMESPACE::steiner_topo_forward,
        "SteinerTopo forward",
        pybind11::arg("pin_pos"),
        pybind11::arg("cached_vtx_relate_x"),
        pybind11::arg("cached_vtx_relate_y"),
        pybind11::arg("num_vertices"),
        pybind11::arg("deterministic_flag") = false);
  m.def("backward", &DREAMPLACE_NAMESPACE::steiner_topo_backward,
        "SteinerTopo backward");
  m.def("build_tree", &DREAMPLACE_NAMESPACE::build_tree,
        "Build Tree",
        pybind11::arg("pos"),
        pybind11::arg("flat_netpin"),
        pybind11::arg("netpin_start"),
        pybind11::arg("ignore_net_degree"),
        pybind11::arg("powv_file"),
        pybind11::arg("post_file"),
        pybind11::arg("deterministic_flag") = false,
        pybind11::arg("frozen_net_ids") = std::vector<int>{},
        pybind11::arg("previous_cache") = std::vector<at::Tensor>{});
}
