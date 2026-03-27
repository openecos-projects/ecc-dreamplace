#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using scalar_t = float;

constexpr scalar_t kEps = 1e-12f;
constexpr scalar_t kTieTol = 1e-6f;
constexpr scalar_t kSignTol = 1e-9f;

struct Piece {
  scalar_t x1;
  scalar_t y1;
  scalar_t x2;
  scalar_t y2;
};

struct LocalStats {
  long edges_with_topology = 0;
  long edges_with_observed_intervals = 0;
  scalar_t gap_sum = 0.0f;
  long tie_count = 0;
  long zero_zero_edges = 0;
  long observed_zero_zero_edges = 0;
  long exact_equal_edges = 0;
  long hv_path_observed_edges = 0;
  long vh_path_observed_edges = 0;
  long both_paths_observed_edges = 0;
};

inline scalar_t cross_value(scalar_t dx, scalar_t dy, scalar_t x1, scalar_t y1, scalar_t qx, scalar_t qy) {
  return dx * (qy - y1) - dy * (qx - x1);
}

inline int sign_with_tol(scalar_t value, scalar_t tol = kSignTol) {
  if (value > tol) {
    return 1;
  }
  if (value < -tol) {
    return -1;
  }
  return 0;
}

inline scalar_t segment_length(scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2) {
  return std::max(std::abs(x2 - x1) + std::abs(y2 - y1), 1e-9f);
}

inline scalar_t interval_overlap_len(scalar_t lo1, scalar_t hi1, scalar_t lo2, scalar_t hi2) {
  if (lo1 > hi1) {
    std::swap(lo1, hi1);
  }
  if (lo2 > hi2) {
    std::swap(lo2, hi2);
  }
  return std::max(0.0f, std::min(hi1, hi2) - std::max(lo1, lo2));
}

inline scalar_t interval_gap(scalar_t lo1, scalar_t hi1, scalar_t lo2, scalar_t hi2) {
  if (lo1 > hi1) {
    std::swap(lo1, hi1);
  }
  if (lo2 > hi2) {
    std::swap(lo2, hi2);
  }
  if (interval_overlap_len(lo1, hi1, lo2, hi2) > 0.0f) {
    return 0.0f;
  }
  if (hi1 < lo2) {
    return lo2 - hi1;
  }
  if (hi2 < lo1) {
    return lo1 - hi2;
  }
  return 0.0f;
}

inline int split_horizontal_segment_by_line(
    scalar_t sx1,
    scalar_t sy,
    scalar_t sx2,
    scalar_t x1,
    scalar_t y1,
    scalar_t dx,
    scalar_t dy,
    Piece out_pieces[2]) {
  scalar_t sx_lo = std::min(sx1, sx2);
  scalar_t sx_hi = std::max(sx1, sx2);
  scalar_t cross1 = cross_value(dx, dy, x1, y1, sx_lo, sy);
  scalar_t cross2 = cross_value(dx, dy, x1, y1, sx_hi, sy);
  int sign1 = sign_with_tol(cross1);
  int sign2 = sign_with_tol(cross2);

  if ((sign1 == 0 && sign2 == 0) || sign1 == sign2 || sign1 == 0 || sign2 == 0 ||
      std::abs(dy) <= kEps) {
    out_pieces[0] = {sx_lo, sy, sx_hi, sy};
    return 1;
  }

  scalar_t x_int = x1 + dx * ((sy - y1) / dy);
  x_int = std::min(std::max(x_int, sx_lo), sx_hi);
  if (x_int <= sx_lo + kEps || x_int >= sx_hi - kEps) {
    out_pieces[0] = {sx_lo, sy, sx_hi, sy};
    return 1;
  }

  out_pieces[0] = {sx_lo, sy, x_int, sy};
  out_pieces[1] = {x_int, sy, sx_hi, sy};
  return 2;
}

inline int split_vertical_segment_by_line(
    scalar_t sx,
    scalar_t sy1,
    scalar_t sy2,
    scalar_t x1,
    scalar_t y1,
    scalar_t dx,
    scalar_t dy,
    Piece out_pieces[2]) {
  scalar_t sy_lo = std::min(sy1, sy2);
  scalar_t sy_hi = std::max(sy1, sy2);
  scalar_t cross1 = cross_value(dx, dy, x1, y1, sx, sy_lo);
  scalar_t cross2 = cross_value(dx, dy, x1, y1, sx, sy_hi);
  int sign1 = sign_with_tol(cross1);
  int sign2 = sign_with_tol(cross2);

  if ((sign1 == 0 && sign2 == 0) || sign1 == sign2 || sign1 == 0 || sign2 == 0 ||
      std::abs(dx) <= kEps) {
    out_pieces[0] = {sx, sy_lo, sx, sy_hi};
    return 1;
  }

  scalar_t y_int = y1 + dy * ((sx - x1) / dx);
  y_int = std::min(std::max(y_int, sy_lo), sy_hi);
  if (y_int <= sy_lo + kEps || y_int >= sy_hi - kEps) {
    out_pieces[0] = {sx, sy_lo, sx, sy_hi};
    return 1;
  }

  out_pieces[0] = {sx, sy_lo, sx, y_int};
  out_pieces[1] = {sx, y_int, sx, sy_hi};
  return 2;
}

inline std::pair<scalar_t, scalar_t> piece_side_affinity(
    const Piece& piece,
    scalar_t x1,
    scalar_t y1,
    scalar_t dx,
    scalar_t dy,
    int sign_h) {
  scalar_t mid_x = 0.5f * (piece.x1 + piece.x2);
  scalar_t mid_y = 0.5f * (piece.y1 + piece.y2);
  int piece_sign = sign_with_tol(cross_value(dx, dy, x1, y1, mid_x, mid_y));
  if (piece_sign == 0) {
    return {0.5f, 0.5f};
  }
  if (piece_sign == sign_h) {
    return {1.0f, 0.0f};
  }
  return {0.0f, 1.0f};
}

inline scalar_t horizontal_leg_affinity(
    const Piece& piece,
    scalar_t leg_y,
    scalar_t leg_x1,
    scalar_t leg_x2,
    scalar_t sigma,
    scalar_t max_distance) {
  scalar_t piece_lo = std::min(piece.x1, piece.x2);
  scalar_t piece_hi = std::max(piece.x1, piece.x2);
  scalar_t leg_lo = std::min(leg_x1, leg_x2);
  scalar_t leg_hi = std::max(leg_x1, leg_x2);
  scalar_t raw_dist = std::abs(piece.y1 - leg_y) + interval_gap(piece_lo, piece_hi, leg_lo, leg_hi);
  if (max_distance > 0.0f && raw_dist > max_distance) {
    return 0.0f;
  }
  scalar_t piece_len = std::max(piece_hi - piece_lo, 1e-9f);
  scalar_t dist_norm = raw_dist / piece_len;
  return std::exp(-dist_norm / std::max(sigma, 1e-9f));
}

inline scalar_t vertical_leg_affinity(
    const Piece& piece,
    scalar_t leg_x,
    scalar_t leg_y1,
    scalar_t leg_y2,
    scalar_t sigma,
    scalar_t max_distance) {
  scalar_t piece_lo = std::min(piece.y1, piece.y2);
  scalar_t piece_hi = std::max(piece.y1, piece.y2);
  scalar_t leg_lo = std::min(leg_y1, leg_y2);
  scalar_t leg_hi = std::max(leg_y1, leg_y2);
  scalar_t raw_dist = std::abs(piece.x1 - leg_x) + interval_gap(piece_lo, piece_hi, leg_lo, leg_hi);
  if (max_distance > 0.0f && raw_dist > max_distance) {
    return 0.0f;
  }
  scalar_t piece_len = std::max(piece_hi - piece_lo, 1e-9f);
  scalar_t dist_norm = raw_dist / piece_len;
  return std::exp(-dist_norm / std::max(sigma, 1e-9f));
}

inline scalar_t quantile_from_sorted(const std::vector<scalar_t>& values, scalar_t q) {
  if (values.empty()) {
    return 0.0f;
  }
  if (values.size() == 1) {
    return values.front();
  }
  scalar_t pos = q * static_cast<scalar_t>(values.size() - 1);
  std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  scalar_t frac = pos - static_cast<scalar_t>(lo);
  return values[lo] * (1.0f - frac) + values[hi] * frac;
}

py::tuple forward(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> net_ids,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> h_seg_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> v_seg_offsets,
    py::array_t<float, py::array::c_style | py::array::forcecast> h_x1,
    py::array_t<float, py::array::c_style | py::array::forcecast> h_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> h_x2,
    py::array_t<float, py::array::c_style | py::array::forcecast> v_x,
    py::array_t<float, py::array::c_style | py::array::forcecast> v_y1,
    py::array_t<float, py::array::c_style | py::array::forcecast> v_y2,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_net_ids,
    py::array_t<float, py::array::c_style | py::array::forcecast> edge_x1,
    py::array_t<float, py::array::c_style | py::array::forcecast> edge_y1,
    py::array_t<float, py::array::c_style | py::array::forcecast> edge_x2,
    py::array_t<float, py::array::c_style | py::array::forcecast> edge_y2,
    float sigma,
    float min_support,
    float max_distance) {
  auto net_ids_v = net_ids.unchecked<1>();
  auto h_seg_offsets_v = h_seg_offsets.unchecked<1>();
  auto v_seg_offsets_v = v_seg_offsets.unchecked<1>();
  auto h_x1_v = h_x1.unchecked<1>();
  auto h_y_v = h_y.unchecked<1>();
  auto h_x2_v = h_x2.unchecked<1>();
  auto v_x_v = v_x.unchecked<1>();
  auto v_y1_v = v_y1.unchecked<1>();
  auto v_y2_v = v_y2.unchecked<1>();
  auto edge_net_ids_v = edge_net_ids.unchecked<1>();
  auto edge_x1_v = edge_x1.unchecked<1>();
  auto edge_y1_v = edge_y1.unchecked<1>();
  auto edge_x2_v = edge_x2.unchecked<1>();
  auto edge_y2_v = edge_y2.unchecked<1>();

  std::size_t num_nets = static_cast<std::size_t>(net_ids_v.shape(0));
  std::size_t num_edges = static_cast<std::size_t>(edge_net_ids_v.shape(0));
  if (static_cast<std::size_t>(h_seg_offsets_v.shape(0)) != num_nets + 1 ||
      static_cast<std::size_t>(v_seg_offsets_v.shape(0)) != num_nets + 1) {
    throw std::runtime_error("same_net_topo_scoring_cpp: offset array length must be num_nets + 1");
  }

  std::unordered_map<int32_t, int32_t> net_index_by_id;
  net_index_by_id.reserve(num_nets * 2 + 1);
  for (std::size_t idx = 0; idx < num_nets; ++idx) {
    net_index_by_id[net_ids_v(idx)] = static_cast<int32_t>(idx);
  }

  py::array_t<float> topo_cost_h(num_edges);
  py::array_t<float> topo_cost_v(num_edges);
  py::array_t<uint8_t> topo_observed_mask(num_edges);
  auto topo_cost_h_v = topo_cost_h.mutable_unchecked<1>();
  auto topo_cost_v_v = topo_cost_v.mutable_unchecked<1>();
  auto topo_observed_mask_v = topo_observed_mask.mutable_unchecked<1>();
  for (std::size_t i = 0; i < num_edges; ++i) {
    topo_cost_h_v(i) = 0.0f;
    topo_cost_v_v(i) = 0.0f;
    topo_observed_mask_v(i) = static_cast<uint8_t>(0);
  }

  std::vector<float> gap_values(num_edges, 0.0f);
  int num_threads = 1;
#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#endif
  std::vector<LocalStats> thread_stats(static_cast<std::size_t>(std::max(num_threads, 1)));

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    LocalStats& local = thread_stats[static_cast<std::size_t>(tid)];

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (std::size_t edge_id = 0; edge_id < num_edges; ++edge_id) {
      auto it = net_index_by_id.find(edge_net_ids_v(edge_id));
      if (it == net_index_by_id.end()) {
        continue;
      }
      local.edges_with_topology += 1;

      float ex1 = edge_x1_v(edge_id);
      float ey1 = edge_y1_v(edge_id);
      float ex2 = edge_x2_v(edge_id);
      float ey2 = edge_y2_v(edge_id);
      float dx = ex2 - ex1;
      float dy = ey2 - ey1;
      if (std::abs(dx) <= kEps || std::abs(dy) <= kEps) {
        continue;
      }

      int sign_h = sign_with_tol(cross_value(dx, dy, ex1, ey1, ex2, ey1));
      if (sign_h == 0) {
        sign_h = 1;
      }

      float score_h = 0.0f;
      float score_v = 0.0f;
      int observed_piece_count = 0;
      int32_t net_index = it->second;

      int32_t h_begin = h_seg_offsets_v(net_index);
      int32_t h_end = h_seg_offsets_v(net_index + 1);
      for (int32_t seg_idx = h_begin; seg_idx < h_end; ++seg_idx) {
        Piece pieces[2];
        int piece_count = split_horizontal_segment_by_line(
            h_x1_v(seg_idx), h_y_v(seg_idx), h_x2_v(seg_idx), ex1, ey1, dx, dy, pieces);
        for (int piece_idx = 0; piece_idx < piece_count; ++piece_idx) {
          const Piece& piece = pieces[piece_idx];
          float piece_len = segment_length(piece.x1, piece.y1, piece.x2, piece.y2);
          auto [alpha_h, alpha_v] = piece_side_affinity(piece, ex1, ey1, dx, dy, sign_h);
          float aff_h = horizontal_leg_affinity(piece, ey1, ex1, ex2, sigma, max_distance);
          float aff_v = horizontal_leg_affinity(piece, ey2, ex1, ex2, sigma, max_distance);
          float support_h_piece = piece_len * alpha_h * aff_h;
          float support_v_piece = piece_len * alpha_v * aff_v;
          if (support_h_piece > 0.0 || support_v_piece > 0.0) {
            observed_piece_count += 1;
          }
          score_h += support_h_piece;
          score_v += support_v_piece;
        }
      }

      int32_t v_begin = v_seg_offsets_v(net_index);
      int32_t v_end = v_seg_offsets_v(net_index + 1);
      for (int32_t seg_idx = v_begin; seg_idx < v_end; ++seg_idx) {
        Piece pieces[2];
        int piece_count = split_vertical_segment_by_line(
            v_x_v(seg_idx), v_y1_v(seg_idx), v_y2_v(seg_idx), ex1, ey1, dx, dy, pieces);
        for (int piece_idx = 0; piece_idx < piece_count; ++piece_idx) {
          const Piece& piece = pieces[piece_idx];
          float piece_len = segment_length(piece.x1, piece.y1, piece.x2, piece.y2);
          auto [alpha_h, alpha_v] = piece_side_affinity(piece, ex1, ey1, dx, dy, sign_h);
          float aff_h = vertical_leg_affinity(piece, ex2, ey1, ey2, sigma, max_distance);
          float aff_v = vertical_leg_affinity(piece, ex1, ey1, ey2, sigma, max_distance);
          float support_h_piece = piece_len * alpha_h * aff_h;
          float support_v_piece = piece_len * alpha_v * aff_v;
          if (support_h_piece > 0.0 || support_v_piece > 0.0) {
            observed_piece_count += 1;
          }
          score_h += support_h_piece;
          score_v += support_v_piece;
        }
      }

      if (score_h > min_support) {
        local.hv_path_observed_edges += 1;
      }
      if (score_v > min_support) {
        local.vh_path_observed_edges += 1;
      }
      if (score_h > min_support && score_v > min_support) {
        local.both_paths_observed_edges += 1;
      }

      float total_support = score_h + score_v;
      if (total_support > min_support) {
        float weight_h = score_h / total_support;
        float weight_v = score_v / total_support;
        topo_cost_h_v(edge_id) = -std::log(weight_h + 1e-12f);
        topo_cost_v_v(edge_id) = -std::log(weight_v + 1e-12f);
        topo_observed_mask_v(edge_id) = static_cast<uint8_t>(1);
        local.edges_with_observed_intervals += 1;
      }

      float cost_h = topo_cost_h_v(edge_id);
      float cost_v = topo_cost_v_v(edge_id);
      float gap = std::abs(cost_h - cost_v);
      gap_values[edge_id] = gap;
      local.gap_sum += gap;
      if (gap <= kTieTol) {
        local.tie_count += 1;
      }
      if (gap <= kEps) {
        local.exact_equal_edges += 1;
      }
      if (std::abs(cost_h) <= kEps && std::abs(cost_v) <= kEps) {
        local.zero_zero_edges += 1;
        if (observed_piece_count > 0) {
          local.observed_zero_zero_edges += 1;
        }
      }
    }
  }

  LocalStats stats_acc;
  for (const auto& local : thread_stats) {
    stats_acc.edges_with_topology += local.edges_with_topology;
    stats_acc.edges_with_observed_intervals += local.edges_with_observed_intervals;
    stats_acc.gap_sum += local.gap_sum;
    stats_acc.tie_count += local.tie_count;
    stats_acc.zero_zero_edges += local.zero_zero_edges;
    stats_acc.observed_zero_zero_edges += local.observed_zero_zero_edges;
    stats_acc.exact_equal_edges += local.exact_equal_edges;
    stats_acc.hv_path_observed_edges += local.hv_path_observed_edges;
    stats_acc.vh_path_observed_edges += local.vh_path_observed_edges;
    stats_acc.both_paths_observed_edges += local.both_paths_observed_edges;
  }

  std::vector<float> sorted_gaps = gap_values;
  std::sort(sorted_gaps.begin(), sorted_gaps.end());
  std::vector<float> nonzero_gaps;
  nonzero_gaps.reserve(sorted_gaps.size());
  for (float gap : sorted_gaps) {
    if (gap > kEps) {
      nonzero_gaps.push_back(gap);
    }
  }

  py::dict stats;
  stats["diag_edges"] = py::int_(static_cast<long>(num_edges));
  stats["edges_with_topology"] = py::int_(stats_acc.edges_with_topology);
  stats["edges_with_observed_intervals"] = py::int_(stats_acc.edges_with_observed_intervals);
  stats["mean_gap"] = py::float_(num_edges > 0 ? stats_acc.gap_sum / static_cast<float>(num_edges) : 0.0f);
  stats["tie_ratio"] = py::float_(num_edges > 0 ? static_cast<float>(stats_acc.tie_count) / static_cast<float>(num_edges) : 0.0f);
  stats["sigma"] = py::float_(sigma);
  stats["min_support"] = py::float_(min_support);
  stats["max_distance"] = py::float_(max_distance);
  stats["zero_zero_edges"] = py::int_(stats_acc.zero_zero_edges);
  stats["observed_zero_zero_edges"] = py::int_(stats_acc.observed_zero_zero_edges);
  stats["unobserved_zero_zero_edges"] = py::int_(stats_acc.zero_zero_edges - stats_acc.observed_zero_zero_edges);
  stats["exact_equal_edges"] = py::int_(stats_acc.exact_equal_edges);
  stats["hv_path_observed_edges"] = py::int_(stats_acc.hv_path_observed_edges);
  stats["vh_path_observed_edges"] = py::int_(stats_acc.vh_path_observed_edges);
  stats["both_paths_observed_edges"] = py::int_(stats_acc.both_paths_observed_edges);
  stats["gap_p50"] = py::float_(quantile_from_sorted(sorted_gaps, 0.5));
  stats["gap_p90"] = py::float_(quantile_from_sorted(sorted_gaps, 0.9));
  stats["nonzero_gap_edges"] = py::int_(static_cast<long>(nonzero_gaps.size()));
  stats["nonzero_gap_p50"] = py::float_(quantile_from_sorted(nonzero_gaps, 0.5));

  return py::make_tuple(topo_cost_h, topo_cost_v, topo_observed_mask, stats);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Same-net diagonal split topology scoring (CPU)");
}
