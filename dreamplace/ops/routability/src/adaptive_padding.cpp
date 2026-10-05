#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Mask = py::array_t<bool, py::array::c_style | py::array::forcecast>;
using Interval = std::pair<double, double>;

std::vector<Interval> merge_clipped(const std::vector<Interval>& intervals, double lower, double upper) {
  std::vector<Interval> clipped;
  for (const auto& [lo, hi] : intervals) {
    if (std::min(hi, upper) > std::max(lo, lower)) {
      clipped.emplace_back(std::max(lo, lower), std::min(hi, upper));
    }
  }
  std::sort(clipped.begin(), clipped.end());
  std::vector<Interval> merged;
  for (const auto& [lo, hi] : clipped) {
    if (merged.empty() || lo > merged.back().second) {
      merged.emplace_back(lo, hi);
    } else {
      merged.back().second = std::max(merged.back().second, hi);
    }
  }
  return merged;
}

std::vector<Interval> free_segments(const std::vector<Interval>& fixed, double lower, double upper) {
  std::vector<Interval> segments;
  double cursor = lower;
  for (const auto& [lo, hi] : merge_clipped(fixed, lower, upper)) {
    if (lo > cursor) {
      segments.emplace_back(cursor, lo);
    }
    cursor = std::max(cursor, hi);
  }
  if (cursor < upper) {
    segments.emplace_back(cursor, upper);
  }
  return segments;
}

py::dict plan_padding(const Array& horizontal, const Array& vertical,
                      const Array& grid_x, const Array& grid_y,
                      const Array& physical_pos, const Array& logical_pos,
                      const Array& physical_size_x, const Array& logical_size_x,
                      const Array& size_y, const Array& rows, const Mask& eligible_mask,
                      int num_nodes, int movable, int physical_nodes,
                      double xl, double yl, double xh, double yh,
                      double site_width, double row_height, double hot_ratio,
                      double hot_cell_ratio) {
  if (horizontal.ndim() != 2 || vertical.ndim() != 2 ||
      horizontal.shape(0) != vertical.shape(0) || horizontal.shape(1) != vertical.shape(1) ||
      horizontal.shape(0) == 0 || horizontal.shape(1) == 0 || rows.ndim() != 2 || rows.shape(1) != 4 ||
      grid_x.ndim() != 1 || grid_x.shape(0) != horizontal.shape(0) + 1 ||
      grid_y.ndim() != 1 || grid_y.shape(0) != horizontal.shape(1) + 1 ||
      physical_pos.ndim() != 1 || logical_pos.ndim() != 1 ||
      physical_pos.shape(0) < 2 * num_nodes || logical_pos.shape(0) < 2 * num_nodes ||
      physical_size_x.ndim() != 1 || logical_size_x.ndim() != 1 || size_y.ndim() != 1 ||
      physical_size_x.shape(0) < num_nodes || logical_size_x.shape(0) < num_nodes ||
      size_y.shape(0) < num_nodes || eligible_mask.ndim() != 1 ||
      eligible_mask.shape(0) != movable || movable < 0 || num_nodes < movable ||
      physical_nodes < movable ||
      site_width <= 0 || row_height <= 0 || xh <= xl || yh <= yl || hot_ratio <= 0 || hot_ratio > 1 ||
      hot_cell_ratio < 0 || hot_cell_ratio > 1) {
    throw py::value_error("invalid adaptive padding geometry or array shape");
  }
  const int nx = static_cast<int>(horizontal.shape(0));
  const int ny = static_cast<int>(horizontal.shape(1));
  const auto* x_edges = grid_x.data();
  const auto* y_edges = grid_y.data();
  for (int i = 0; i < nx; ++i) {
    if (!(x_edges[i + 1] > x_edges[i])) {
      throw py::value_error("routing grid x edges must be increasing");
    }
  }
  for (int i = 0; i < ny; ++i) {
    if (!(y_edges[i + 1] > y_edges[i])) {
      throw py::value_error("routing grid y edges must be increasing");
    }
  }
  const auto* h = horizontal.data();
  const auto* v = vertical.data();
  std::vector<double> overflow(nx * ny, 0.0);
  std::vector<int> ranked_bins;
  for (int i = 0; i < nx * ny; ++i) {
    overflow[i] = std::max({h[i], v[i], 0.0});
    if (overflow[i] > 0) {
      ranked_bins.push_back(i);
    }
  }
  std::stable_sort(ranked_bins.begin(), ranked_bins.end(), [&](int a, int b) {
    return overflow[a] > overflow[b];
  });
  const int hot_bins = static_cast<int>(std::ceil(ranked_bins.size() * hot_ratio));
  std::vector<float> selected(nx * ny, 0.0f);
  for (int i = 0; i < hot_bins; ++i) {
    selected[ranked_bins[i]] = static_cast<float>(overflow[ranked_bins[i]]);
  }

  py::array_t<float> scores({movable}, {sizeof(float)});
  py::array_t<std::int32_t> padding({movable}, {sizeof(std::int32_t)});
  auto* score = scores.mutable_data();
  auto* pad = padding.mutable_data();
  std::fill(score, score + movable, 0.0f);
  std::fill(pad, pad + movable, 0);
  if (!hot_bins) {
    py::dict result;
    result["padding_sites"] = std::move(padding);
    result["scores"] = std::move(scores);
    result["hot_bins_count"] = 0;
    result["positive_score_count"] = 0;
    result["allocated_count"] = 0;
    result["total_free_sites"] = 0;
    return result;
  }

  const auto* pos = physical_pos.data();
  const auto* width = physical_size_x.data();
  const auto* height = size_y.data();
  for (int node = 0; node < movable; ++node) {
    if (width[node] <= 0 || height[node] <= 0) {
      continue;
    }
    if (pos[node] + width[node] <= x_edges[0] || pos[node] >= x_edges[nx] ||
        pos[num_nodes + node] + height[node] <= y_edges[0] || pos[num_nodes + node] >= y_edges[ny]) {
      continue;
    }
    const int left = std::clamp(static_cast<int>(std::upper_bound(x_edges, x_edges + nx + 1, pos[node]) - x_edges - 1), 0, nx - 1);
    const int right = std::clamp(static_cast<int>(std::lower_bound(x_edges, x_edges + nx + 1, pos[node] + width[node]) - x_edges - 1), 0, nx - 1);
    const int bottom = std::clamp(static_cast<int>(std::upper_bound(y_edges, y_edges + ny + 1, pos[num_nodes + node]) - y_edges - 1), 0, ny - 1);
    const int top = std::clamp(static_cast<int>(std::lower_bound(y_edges, y_edges + ny + 1, pos[num_nodes + node] + height[node]) - y_edges - 1), 0, ny - 1);
    for (int x = left; x <= right; ++x) {
      for (int y = bottom; y <= top; ++y) {
        score[node] = std::max(score[node], selected[x * ny + y]);
      }
    }
  }

  const int num_rows = std::max(1, static_cast<int>(std::ceil((yh - yl) / row_height)));
  std::vector<std::vector<Interval>> fixed(num_rows), row_bounds(num_rows), segments(num_rows);
  auto covered_rows = [&](double y, double height_value) {
    return std::pair<int, int>{std::max(0, static_cast<int>(std::floor((y - yl) / row_height))),
                               std::min(num_rows - 1, static_cast<int>(std::ceil((y + height_value - yl) / row_height) - 1))};
  };
  const auto* lpos = logical_pos.data();
  const auto* lwidth = logical_size_x.data();
  for (int node = movable; node < std::min(physical_nodes, num_nodes); ++node) {
    if (lwidth[node] <= 0 || height[node] <= 0) {
      continue;
    }
    const auto [first, last] = covered_rows(lpos[num_nodes + node], height[node]);
    for (int row = first; row <= last; ++row) {
      fixed[row].emplace_back(lpos[node], lpos[node] + lwidth[node]);
    }
  }
  const auto row_boxes = rows.unchecked<2>();
  for (py::ssize_t i = 0; i < rows.shape(0); ++i) {
    const auto [first, last] = covered_rows(row_boxes(i, 1), row_boxes(i, 3) - row_boxes(i, 1));
    for (int row = first; row <= last; ++row) {
      row_bounds[row].emplace_back(row_boxes(i, 0), row_boxes(i, 2));
    }
  }
  for (int row = 0; row < num_rows; ++row) {
    for (const auto& [lo, hi] : merge_clipped(row_bounds[row], xl, xh)) {
      const auto free = free_segments(fixed[row], lo, hi);
      segments[row].insert(segments[row].end(), free.begin(), free.end());
    }
  }

  std::vector<std::vector<int>> used(num_rows), budget(num_rows);
  for (int row = 0; row < num_rows; ++row) {
    used[row].resize(segments[row].size(), 0);
  }
  std::vector<std::pair<int, int>> node_segment(movable, {-1, -1});
  std::vector<bool> eligible(movable, false);
  const auto* mask = eligible_mask.data();
  for (int node = 0; node < movable; ++node) {
    eligible[node] = mask[node] && lwidth[node] > 0 && height[node] > 0 && height[node] <= row_height * 1.01;
    if (lwidth[node] <= 0 || height[node] <= 0) {
      continue;
    }
    const double center = lpos[node] + lwidth[node] * 0.5;
    const int sites = static_cast<int>(std::ceil(lwidth[node] / site_width - 1e-9));
    const auto [first, last] = covered_rows(lpos[num_nodes + node], height[node]);
    for (int row = first; row <= last; ++row) {
      for (std::size_t segment = 0; segment < segments[row].size(); ++segment) {
        if (segments[row][segment].first <= center && center <= segments[row][segment].second) {
          if (node_segment[node].first < 0) {
            node_segment[node] = {row, static_cast<int>(segment)};
          }
          used[row][segment] += sites;
          break;
        }
      }
    }
    if (node_segment[node].first < 0) {
      eligible[node] = false;
    }
  }

  int total_free = 0;
  for (int row = 0; row < num_rows; ++row) {
    budget[row].resize(segments[row].size(), 0);
    for (std::size_t i = 0; i < segments[row].size(); ++i) {
      const int capacity = static_cast<int>(std::floor(
          (segments[row][i].second - segments[row][i].first) / site_width + 1e-9));
      budget[row][i] = std::max(0, capacity - used[row][i]);
      total_free += budget[row][i];
    }
  }
  std::vector<int> candidates;
  for (int node = 0; node < movable; ++node) {
    if (eligible[node] && score[node] > 0) {
      candidates.push_back(node);
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) {
    return score[a] > score[b];
  });
  const int eligible_count = static_cast<int>(std::count(eligible.begin(), eligible.end(), true));
  const int hot_cell_count = std::min(static_cast<int>(candidates.size()),
                                      static_cast<int>(std::ceil(eligible_count * hot_cell_ratio)));
  int allocated = 0;
  for (int i = 0; i < hot_cell_count; ++i) {
    const int node = candidates[i];
    const auto [row, segment] = node_segment[node];
    if (budget[row][segment] >= 2) {
      pad[node] = 1;
      budget[row][segment] -= 2;
      ++allocated;
    }
  }

  py::dict result;
  result["padding_sites"] = std::move(padding);
  result["scores"] = std::move(scores);
  result["hot_bins_count"] = hot_bins;
  result["positive_score_count"] = static_cast<int>(candidates.size());
  result["allocated_count"] = allocated;
  result["total_free_sites"] = total_free;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan_padding", &plan_padding, "Route-aware one-site padding on real row segments");
}
