#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <limits>
#include <string>
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

struct IntInterval {
  int lo = 0;
  int hi = 0;
};

struct TopologyRecord {
  int net_id = -1;
  std::string net_name;
  std::map<int, std::vector<IntInterval>> horizontal_by_row;
  std::map<int, std::vector<IntInterval>> vertical_by_col;
  int bbox_x_lo = 0;
  int bbox_y_lo = 0;
  int bbox_x_hi = 0;
  int bbox_y_hi = 0;
  bool bbox_initialized = false;
  int num_horizontal_intervals = 0;
  int num_vertical_intervals = 0;
  int total_horizontal_length = 0;
  int total_vertical_length = 0;
  int raw_wire_count = 0;
  int invalid_wire_count = 0;
  bool route_failed = false;
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

inline py::object dict_get_or_none(const py::dict& dict, const char* key) {
  py::str py_key(key);
  if (dict.contains(py_key)) {
    return py::reinterpret_borrow<py::object>(dict[py_key]);
  }
  return py::none();
}

inline std::string object_to_string(py::handle obj) {
  if (obj.is_none()) {
    return "";
  }
  if (py::isinstance<py::bytes>(obj)) {
    return py::reinterpret_borrow<py::bytes>(obj).cast<std::string>();
  }
  return py::str(obj).cast<std::string>();
}

inline std::string dict_string_or(const py::dict& dict, const char* key, const std::string& default_value = "") {
  py::object value = dict_get_or_none(dict, key);
  if (value.is_none()) {
    return default_value;
  }
  return object_to_string(value);
}

inline int dict_int_or(const py::dict& dict, const char* key, int default_value = 0) {
  py::object value = dict_get_or_none(dict, key);
  if (value.is_none()) {
    return default_value;
  }
  return value.cast<int>();
}

inline bool dict_bool_or(const py::dict& dict, const char* key, bool default_value = false) {
  py::object value = dict_get_or_none(dict, key);
  if (value.is_none()) {
    return default_value;
  }
  return value.cast<bool>();
}

inline void merge_normalized_int_intervals(std::vector<IntInterval>& intervals, int max_gap) {
  if (intervals.size() <= 1) {
    return;
  }
  std::sort(intervals.begin(), intervals.end(), [](const IntInterval& a, const IntInterval& b) {
    if (a.lo != b.lo) {
      return a.lo < b.lo;
    }
    return a.hi < b.hi;
  });

  std::vector<IntInterval> merged;
  merged.reserve(intervals.size());
  IntInterval current = intervals.front();
  for (std::size_t i = 1; i < intervals.size(); ++i) {
    const IntInterval& next = intervals[i];
    if (next.lo <= current.hi + max_gap) {
      if (next.hi > current.hi) {
        current.hi = next.hi;
      }
    } else {
      merged.push_back(current);
      current = next;
    }
  }
  merged.push_back(current);
  intervals.swap(merged);
}

inline int interval_total_length(const std::vector<IntInterval>& intervals) {
  int total = 0;
  for (const IntInterval& interval : intervals) {
    total += interval.hi - interval.lo;
  }
  return total;
}

template <typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& values) {
  py::array_t<T> result(values.size());
  auto result_v = result.template mutable_unchecked<1>();
  for (std::size_t i = 0; i < values.size(); ++i) {
    result_v(i) = values[i];
  }
  return result;
}

inline py::dict intervals_by_axis_to_python(const std::map<int, std::vector<IntInterval>>& intervals_by_axis) {
  py::dict result;
  for (const auto& [axis_idx, intervals] : intervals_by_axis) {
    py::list py_intervals;
    for (const IntInterval& interval : intervals) {
      py_intervals.append(py::make_tuple(interval.lo, interval.hi));
    }
    result[py::int_(axis_idx)] = py_intervals;
  }
  return result;
}

inline py::dict topology_record_to_python(const TopologyRecord& record) {
  py::dict result;
  result["net_id"] = py::int_(record.net_id);
  result["net_name"] = py::str(record.net_name);
  result["horizontal_by_row"] = intervals_by_axis_to_python(record.horizontal_by_row);
  result["vertical_by_col"] = intervals_by_axis_to_python(record.vertical_by_col);
  result["bbox"] = py::make_tuple(record.bbox_x_lo, record.bbox_y_lo, record.bbox_x_hi, record.bbox_y_hi);
  result["num_horizontal_intervals"] = py::int_(record.num_horizontal_intervals);
  result["num_vertical_intervals"] = py::int_(record.num_vertical_intervals);
  result["total_horizontal_length"] = py::int_(record.total_horizontal_length);
  result["total_vertical_length"] = py::int_(record.total_vertical_length);
  result["raw_wire_count"] = py::int_(record.raw_wire_count);
  result["invalid_wire_count"] = py::int_(record.invalid_wire_count);
  result["route_failed"] = py::bool_(record.route_failed);
  return result;
}

inline bool finalize_topology_record(TopologyRecord& record, int max_gap) {
  for (auto it = record.horizontal_by_row.begin(); it != record.horizontal_by_row.end();) {
    merge_normalized_int_intervals(it->second, max_gap);
    if (it->second.empty()) {
      it = record.horizontal_by_row.erase(it);
    } else {
      record.num_horizontal_intervals += static_cast<int>(it->second.size());
      record.total_horizontal_length += interval_total_length(it->second);
      ++it;
    }
  }
  for (auto it = record.vertical_by_col.begin(); it != record.vertical_by_col.end();) {
    merge_normalized_int_intervals(it->second, max_gap);
    if (it->second.empty()) {
      it = record.vertical_by_col.erase(it);
    } else {
      record.num_vertical_intervals += static_cast<int>(it->second.size());
      record.total_vertical_length += interval_total_length(it->second);
      ++it;
    }
  }
  return record.num_horizontal_intervals > 0 || record.num_vertical_intervals > 0;
}

inline float grid_center(int coord_idx, float origin, float step) {
  return origin + (static_cast<float>(coord_idx) + 0.5f) * step;
}

py::dict pack_topology_records(
    const std::map<int, TopologyRecord>& records,
    double xl,
    double yl,
    double route_bin_size_x,
    double route_bin_size_y) {
  std::vector<int32_t> net_ids;
  std::vector<int32_t> h_seg_offsets;
  std::vector<int32_t> v_seg_offsets;
  std::vector<float> h_x1;
  std::vector<float> h_y;
  std::vector<float> h_x2;
  std::vector<float> v_x;
  std::vector<float> v_y1;
  std::vector<float> v_y2;

  net_ids.reserve(records.size());
  h_seg_offsets.reserve(records.size() + 1);
  v_seg_offsets.reserve(records.size() + 1);
  h_seg_offsets.push_back(0);
  v_seg_offsets.push_back(0);

  float xl_f = static_cast<float>(xl);
  float yl_f = static_cast<float>(yl);
  float route_bin_size_x_f = static_cast<float>(route_bin_size_x);
  float route_bin_size_y_f = static_cast<float>(route_bin_size_y);
  int max_net_id = -1;

  for (const auto& [net_id, record] : records) {
    net_ids.push_back(static_cast<int32_t>(net_id));
    max_net_id = std::max(max_net_id, net_id);

    int32_t h_count = 0;
    for (const auto& [row_idx, intervals] : record.horizontal_by_row) {
      float seg_y = grid_center(row_idx, yl_f, route_bin_size_y_f);
      for (const IntInterval& interval : intervals) {
        h_x1.push_back(grid_center(interval.lo, xl_f, route_bin_size_x_f));
        h_y.push_back(seg_y);
        h_x2.push_back(grid_center(interval.hi, xl_f, route_bin_size_x_f));
        ++h_count;
      }
    }
    h_seg_offsets.push_back(h_seg_offsets.back() + h_count);

    int32_t v_count = 0;
    for (const auto& [col_idx, intervals] : record.vertical_by_col) {
      float seg_x = grid_center(col_idx, xl_f, route_bin_size_x_f);
      for (const IntInterval& interval : intervals) {
        v_x.push_back(seg_x);
        v_y1.push_back(grid_center(interval.lo, yl_f, route_bin_size_y_f));
        v_y2.push_back(grid_center(interval.hi, yl_f, route_bin_size_y_f));
        ++v_count;
      }
    }
    v_seg_offsets.push_back(v_seg_offsets.back() + v_count);
  }

  std::vector<int32_t> net_index_by_id;
  if (max_net_id >= 0) {
    net_index_by_id.assign(static_cast<std::size_t>(max_net_id) + 1, static_cast<int32_t>(-1));
    for (std::size_t index = 0; index < net_ids.size(); ++index) {
      int32_t net_id = net_ids[index];
      if (net_id >= 0) {
        net_index_by_id[static_cast<std::size_t>(net_id)] = static_cast<int32_t>(index);
      }
    }
  }

  py::dict packed;
  packed["net_ids"] = vector_to_numpy(net_ids);
  packed["net_index_by_id"] = vector_to_numpy(net_index_by_id);
  packed["h_seg_offsets"] = vector_to_numpy(h_seg_offsets);
  packed["v_seg_offsets"] = vector_to_numpy(v_seg_offsets);
  packed["h_x1"] = vector_to_numpy(h_x1);
  packed["h_y"] = vector_to_numpy(h_y);
  packed["h_x2"] = vector_to_numpy(h_x2);
  packed["v_x"] = vector_to_numpy(v_x);
  packed["v_y1"] = vector_to_numpy(v_y1);
  packed["v_y2"] = vector_to_numpy(v_y2);
  return packed;
}

py::tuple build_topology_cache(
    py::iterable route_entries,
    py::dict net_name_to_id,
    py::tuple route_grid_shape,
    int max_gap,
    double xl,
    double yl,
    double route_bin_size_x,
    double route_bin_size_y,
    bool build_packed,
    bool build_python_cache) {
  std::map<int, TopologyRecord> native_topologies;
  py::dict py_topologies;
  py::dict route_entry_meta;

  int num_route_entry_nets = 0;
  int num_route_failed_nets = 0;
  int unknown_name_count = 0;
  int total_segments_h = 0;
  int total_segments_v = 0;
  int total_length_h = 0;
  int total_length_v = 0;
  int max_intervals_per_net = 0;
  int invalid_wire_count = 0;

  for (py::handle net_obj : route_entries) {
    ++num_route_entry_nets;
    py::dict net_route = py::reinterpret_borrow<py::dict>(net_obj);
    bool route_failed = dict_bool_or(net_route, "route_failed", false);
    if (route_failed) {
      ++num_route_failed_nets;
    }

    std::string net_name = dict_string_or(net_route, "net_name", "");
    int net_id = -1;
    py::str net_name_key(net_name);
    if (net_name_to_id.contains(net_name_key)) {
      net_id = net_name_to_id[net_name_key].cast<int>();
    } else {
      net_id = dict_int_or(net_route, "net_id", -1);
      if (net_id < 0) {
        ++unknown_name_count;
        continue;
      }
    }

    py::object entries_obj = dict_get_or_none(net_route, "entries");
    bool has_entries = !entries_obj.is_none();
    std::size_t entry_count = has_entries ? static_cast<std::size_t>(py::len(entries_obj)) : 0;

    TopologyRecord record;
    record.net_id = net_id;
    record.net_name = net_name.empty() ? ("net_" + std::to_string(net_id)) : net_name;
    record.route_failed = route_failed;

    if (has_entries) {
      for (py::handle entry_obj : entries_obj) {
        py::dict entry = py::reinterpret_borrow<py::dict>(entry_obj);
        if (dict_string_or(entry, "type", "") != "wire") {
          continue;
        }
        ++record.raw_wire_count;

        int x1 = dict_int_or(entry, "grid_x1", 0);
        int y1 = dict_int_or(entry, "grid_y1", 0);
        int x2 = dict_int_or(entry, "grid_x2", 0);
        int y2 = dict_int_or(entry, "grid_y2", 0);
        std::string orientation = dict_string_or(entry, "orientation", "");
        bool is_horizontal = orientation == "H" || y1 == y2;
        bool is_vertical = orientation == "V" || x1 == x2;

        if (!record.bbox_initialized) {
          record.bbox_x_lo = x1;
          record.bbox_y_lo = y1;
          record.bbox_x_hi = x1;
          record.bbox_y_hi = y1;
          record.bbox_initialized = true;
        } else {
          record.bbox_x_lo = std::min(record.bbox_x_lo, std::min(x1, x2));
          record.bbox_y_lo = std::min(record.bbox_y_lo, std::min(y1, y2));
          record.bbox_x_hi = std::max(record.bbox_x_hi, std::max(x1, x2));
          record.bbox_y_hi = std::max(record.bbox_y_hi, std::max(y1, y2));
        }

        if (is_horizontal && y1 == y2) {
          int lo = std::min(x1, x2);
          int hi = std::max(x1, x2);
          record.horizontal_by_row[y1].push_back({lo, hi});
          continue;
        }
        if (is_vertical && x1 == x2) {
          int lo = std::min(y1, y2);
          int hi = std::max(y1, y2);
          record.vertical_by_col[x1].push_back({lo, hi});
          continue;
        }
        ++record.invalid_wire_count;
      }
    }

    if (build_python_cache) {
      py::dict meta;
      meta["net_id"] = py::int_(net_id);
      meta["net_name"] = py::str(record.net_name);
      meta["route_failed"] = py::bool_(route_failed);
      meta["entry_count"] = py::int_(static_cast<int>(entry_count));
      meta["wire_entry_count"] = py::int_(record.raw_wire_count);
      route_entry_meta[py::int_(net_id)] = meta;
    }

    if (!finalize_topology_record(record, max_gap)) {
      continue;
    }

    native_topologies[net_id] = record;
    if (build_python_cache) {
      py_topologies[py::int_(net_id)] = topology_record_to_python(record);
    }
    total_segments_h += record.num_horizontal_intervals;
    total_segments_v += record.num_vertical_intervals;
    total_length_h += record.total_horizontal_length;
    total_length_v += record.total_vertical_length;
    invalid_wire_count += record.invalid_wire_count;
    max_intervals_per_net = std::max(
        max_intervals_per_net,
        record.num_horizontal_intervals + record.num_vertical_intervals);
  }

  int route_num_bins_x = route_grid_shape.size() > 0 ? route_grid_shape[0].cast<int>() : 0;
  int route_num_bins_y = route_grid_shape.size() > 1 ? route_grid_shape[1].cast<int>() : 0;
  py::tuple py_route_grid_shape = py::make_tuple(route_num_bins_x, route_num_bins_y);

  py::dict stats;
  stats["route_grid_shape"] = py_route_grid_shape;
  stats["num_route_entry_nets"] = py::int_(num_route_entry_nets);
  stats["num_nets_with_topology"] = py::int_(static_cast<int>(native_topologies.size()));
  stats["num_route_failed_nets"] = py::int_(num_route_failed_nets);
  stats["unknown_name_count"] = py::int_(unknown_name_count);
  stats["num_segments_h"] = py::int_(total_segments_h);
  stats["num_segments_v"] = py::int_(total_segments_v);
  stats["total_horizontal_length"] = py::int_(total_length_h);
  stats["total_vertical_length"] = py::int_(total_length_v);
  stats["invalid_wire_count"] = py::int_(invalid_wire_count);
  stats["max_intervals_per_net"] = py::int_(max_intervals_per_net);
  stats["backend"] = py::str("cpp");

  py::dict cache;
  cache["route_grid_shape"] = py_route_grid_shape;
  cache["net_topologies"] = py_topologies;
  cache["route_entry_meta"] = route_entry_meta;
  cache["num_nets_with_topology"] = py::int_(static_cast<int>(native_topologies.size()));
  cache["num_segments_h"] = py::int_(total_segments_h);
  cache["num_segments_v"] = py::int_(total_segments_v);
  cache["compact_topology_cache"] = py::bool_(!build_python_cache);
  if (build_packed && !native_topologies.empty()) {
    cache["_same_net_topo_cpp_pack_key"] = py::make_tuple(xl, yl, route_bin_size_x, route_bin_size_y);
    cache["_same_net_topo_cpp_packed"] = pack_topology_records(
        native_topologies,
        xl,
        yl,
        route_bin_size_x,
        route_bin_size_y);
  }

  return py::make_tuple(cache, stats);
}

py::tuple forward(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> net_ids,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> net_index_by_id,
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
    float max_distance,
    bool collect_stats = true) {
  auto net_ids_v = net_ids.unchecked<1>();
  auto net_index_by_id_v = net_index_by_id.unchecked<1>();
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

  std::size_t net_index_by_id_size = static_cast<std::size_t>(net_index_by_id_v.shape(0));

  py::array_t<float> topo_cost_h(num_edges);
  py::array_t<float> topo_cost_v(num_edges);
  py::array_t<uint8_t> topo_observed_mask(num_edges);
  std::fill(topo_cost_h.mutable_data(), topo_cost_h.mutable_data() + num_edges, 0.0f);
  std::fill(topo_cost_v.mutable_data(), topo_cost_v.mutable_data() + num_edges, 0.0f);
  std::fill(
      topo_observed_mask.mutable_data(),
      topo_observed_mask.mutable_data() + num_edges,
      static_cast<uint8_t>(0));
  auto topo_cost_h_v = topo_cost_h.mutable_unchecked<1>();
  auto topo_cost_v_v = topo_cost_v.mutable_unchecked<1>();
  auto topo_observed_mask_v = topo_observed_mask.mutable_unchecked<1>();

  std::vector<float> gap_values;
  if (collect_stats) {
    gap_values.assign(num_edges, 0.0f);
  }
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
      int32_t edge_net_id = edge_net_ids_v(edge_id);
      if (edge_net_id < 0 || static_cast<std::size_t>(edge_net_id) >= net_index_by_id_size) {
        continue;
      }
      int32_t net_index = net_index_by_id_v(static_cast<std::size_t>(edge_net_id));
      if (net_index < 0 || static_cast<std::size_t>(net_index) >= num_nets) {
        continue;
      }
      if (collect_stats) {
        local.edges_with_topology += 1;
      }

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
          if (collect_stats && (support_h_piece > 0.0 || support_v_piece > 0.0)) {
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
          if (collect_stats && (support_h_piece > 0.0 || support_v_piece > 0.0)) {
            observed_piece_count += 1;
          }
          score_h += support_h_piece;
          score_v += support_v_piece;
        }
      }

      if (collect_stats) {
        if (score_h > min_support) {
          local.hv_path_observed_edges += 1;
        }
        if (score_v > min_support) {
          local.vh_path_observed_edges += 1;
        }
        if (score_h > min_support && score_v > min_support) {
          local.both_paths_observed_edges += 1;
        }
      }

      float total_support = score_h + score_v;
      if (total_support > min_support) {
        float weight_h = score_h / total_support;
        float weight_v = score_v / total_support;
        topo_cost_h_v(edge_id) = -std::log(weight_h + 1e-12f);
        topo_cost_v_v(edge_id) = -std::log(weight_v + 1e-12f);
        topo_observed_mask_v(edge_id) = static_cast<uint8_t>(1);
        if (collect_stats) {
          local.edges_with_observed_intervals += 1;
        }
      }

      if (collect_stats) {
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

  py::dict stats;
  stats["diag_edges"] = py::int_(static_cast<long>(num_edges));
  stats["edges_with_topology"] = py::int_(stats_acc.edges_with_topology);
  stats["edges_with_observed_intervals"] = py::int_(stats_acc.edges_with_observed_intervals);
  stats["sigma"] = py::float_(sigma);
  stats["min_support"] = py::float_(min_support);
  stats["max_distance"] = py::float_(max_distance);
  if (collect_stats) {
    std::vector<float> sorted_gaps = gap_values;
    std::sort(sorted_gaps.begin(), sorted_gaps.end());
    std::vector<float> nonzero_gaps;
    nonzero_gaps.reserve(sorted_gaps.size());
    for (float gap : sorted_gaps) {
      if (gap > kEps) {
        nonzero_gaps.push_back(gap);
      }
    }

    stats["mean_gap"] = py::float_(num_edges > 0 ? stats_acc.gap_sum / static_cast<float>(num_edges) : 0.0f);
    stats["tie_ratio"] = py::float_(num_edges > 0 ? static_cast<float>(stats_acc.tie_count) / static_cast<float>(num_edges) : 0.0f);
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
  } else {
    stats["mean_gap"] = py::float_(0.0f);
    stats["tie_ratio"] = py::float_(0.0f);
  }

  return py::make_tuple(topo_cost_h, topo_cost_v, topo_observed_mask, stats);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Same-net diagonal split topology scoring (CPU)");
  m.def(
      "build_topology_cache",
      &build_topology_cache,
      "Build same-net topology cache from route entries (CPU)",
      py::arg("route_entries"),
      py::arg("net_name_to_id"),
      py::arg("route_grid_shape"),
      py::arg("max_gap") = 1,
      py::arg("xl") = 0.0,
      py::arg("yl") = 0.0,
      py::arg("route_bin_size_x") = 1.0,
      py::arg("route_bin_size_y") = 1.0,
      py::arg("build_packed") = false,
      py::arg("build_python_cache") = true);
}
