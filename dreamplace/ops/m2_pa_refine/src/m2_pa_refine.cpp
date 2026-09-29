#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
struct RowItem {
  int node_id;
  T lx;
  T hx;
  bool movable;
};

template <typename T>
struct MoveCandidate {
  bool valid = false;
  T displacement = std::numeric_limits<T>::max();
  std::vector<std::pair<int, T>> targets;
};

template <typename T>
inline T floorToSite(T value, T xl, T site_width) {
  const T tolerance = site_width * static_cast<T>(1e-4);
  return std::floor((value - xl + tolerance) / site_width) * site_width + xl;
}

template <typename T>
inline T ceilToSite(T value, T xl, T site_width) {
  const T tolerance = site_width * static_cast<T>(1e-4);
  return std::ceil((value - xl - tolerance) / site_width) * site_width + xl;
}

template <typename T>
inline T quantizedWidth(const RowItem<T>& item, T site_width) {
  return std::ceil((item.hx - item.lx) / site_width - static_cast<T>(1e-4)) *
         site_width;
}

template <typename T>
MoveCandidate<T> buildLeftCandidate(const std::vector<RowItem<T>>& row,
                                    int cell_index, T rail_lx, T xl,
                                    T site_width, int max_neighbors,
                                    T max_displacement,
                                    const T* initial_x) {
  MoveCandidate<T> best;
  const T tolerance = site_width * static_cast<T>(1e-4);
  const int min_start = std::max(0, cell_index - max_neighbors + 1);

  for (int start = cell_index; start >= min_start; --start) {
    bool all_movable = true;
    for (int index = start; index <= cell_index; ++index) {
      all_movable = all_movable && row[index].movable;
    }
    if (!all_movable) {
      break;
    }

    std::vector<std::pair<int, T>> targets;
    T cursor = floorToSite(rail_lx, xl, site_width);
    for (int index = cell_index; index >= start; --index) {
      const T current_x = floorToSite(row[index].lx, xl, site_width);
      const T target_x =
          std::min(current_x, cursor - quantizedWidth(row[index], site_width));
      targets.emplace_back(index, target_x);
      cursor = target_x;
    }
    std::reverse(targets.begin(), targets.end());

    const T lower_bound =
        start == 0 ? xl : ceilToSite(row[start - 1].hx, xl, site_width);
    if (targets.front().second + tolerance < lower_bound) {
      continue;
    }

    T displacement = 0;
    for (const auto& target : targets) {
      displacement +=
          std::abs(initial_x[row[target.first].node_id] - target.second);
    }
    if (displacement > max_displacement + tolerance ||
        displacement >= best.displacement) {
      continue;
    }
    best.valid = true;
    best.displacement = displacement;
    best.targets = std::move(targets);
  }
  return best;
}

template <typename T>
MoveCandidate<T> buildRightCandidate(const std::vector<RowItem<T>>& row,
                                     int cell_index, T rail_hx, T xl, T xh,
                                     T site_width, int max_neighbors,
                                     T max_displacement,
                                     const T* initial_x) {
  MoveCandidate<T> best;
  const T tolerance = site_width * static_cast<T>(1e-4);
  int overlap_end = cell_index;
  while (overlap_end + 1 < static_cast<int>(row.size()) &&
         row[overlap_end + 1].lx < rail_hx - tolerance) {
    ++overlap_end;
    if (!row[overlap_end].movable) {
      return best;
    }
  }

  const int max_end = std::min(
      static_cast<int>(row.size()) - 1, overlap_end + max_neighbors - 1);
  for (int end = overlap_end; end <= max_end; ++end) {
    bool all_movable = true;
    for (int index = cell_index; index <= end; ++index) {
      all_movable = all_movable && row[index].movable;
    }
    if (!all_movable) {
      break;
    }

    std::vector<std::pair<int, T>> targets;
    T cursor = ceilToSite(rail_hx, xl, site_width);
    for (int index = cell_index; index <= end; ++index) {
      const T current_x = ceilToSite(row[index].lx, xl, site_width);
      const T target_x = std::max(current_x, cursor);
      targets.emplace_back(index, target_x);
      cursor = target_x + quantizedWidth(row[index], site_width);
    }

    const T upper_bound =
        end + 1 == static_cast<int>(row.size())
            ? xh
            : floorToSite(row[end + 1].lx, xl, site_width);
    if (targets.back().second + quantizedWidth(row[end], site_width) >
        upper_bound + tolerance) {
      continue;
    }

    T displacement = 0;
    for (const auto& target : targets) {
      displacement +=
          std::abs(initial_x[row[target.first].node_id] - target.second);
    }
    if (displacement > max_displacement + tolerance ||
        displacement >= best.displacement) {
      continue;
    }
    best.valid = true;
    best.displacement = displacement;
    best.targets = std::move(targets);
  }
  return best;
}

template <typename T>
std::pair<double, double> computeRailOverlap(
    const T* x, const T* y, const T* node_size_x, const T* node_size_y,
    const T* rail_boxes, int num_rails, int num_movable_nodes, T row_height) {
  double count = 0;
  double area = 0;
  const T tolerance = row_height * static_cast<T>(1e-4);
  for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
    if (node_size_y[node_id] > row_height + tolerance ||
        node_size_x[node_id] <= 0 || node_size_y[node_id] <= 0) {
      continue;
    }
    const T node_hx = x[node_id] + node_size_x[node_id];
    const T node_hy = y[node_id] + node_size_y[node_id];
    for (int rail_id = 0; rail_id < num_rails; ++rail_id) {
      const T* rail = rail_boxes + rail_id * 4;
      const T overlap_x =
          std::min(node_hx, rail[2]) - std::max(x[node_id], rail[0]);
      const T overlap_y =
          std::min(node_hy, rail[3]) - std::max(y[node_id], rail[1]);
      if (overlap_x > 0 && overlap_y > 0) {
        count += 1;
        area += static_cast<double>(overlap_x * overlap_y);
      }
    }
  }
  return {count, area};
}

template <typename T>
void applyCandidate(std::vector<RowItem<T>>& row,
                    const MoveCandidate<T>& candidate, T* x) {
  for (const auto& target : candidate.targets) {
    RowItem<T>& item = row[target.first];
    const T width = item.hx - item.lx;
    item.lx = target.second;
    item.hx = target.second + width;
    x[item.node_id] = target.second;
  }
}

template <typename T>
void runM2PARefine(T* x, const T* y, const T* node_size_x,
                   const T* node_size_y, const T* rail_boxes, int num_nodes,
                   int num_rails, T xl, T yl, T xh, T yh, T site_width,
                   T row_height, int num_movable_nodes, int num_terminals,
                   int max_neighbors, T max_displacement_sites,
                   double* stats) {
  const std::vector<T> initial_x(x, x + num_nodes);
  const auto before = computeRailOverlap(
      x, y, node_size_x, node_size_y, rail_boxes, num_rails,
      num_movable_nodes, row_height);
  const int num_rows =
      std::max(1, static_cast<int>(std::ceil((yh - yl) / row_height)));
  std::vector<std::vector<RowItem<T>>> rows(num_rows);
  const T tolerance = row_height * static_cast<T>(1e-4);
  const int physical_end =
      std::min(num_nodes, num_movable_nodes + num_terminals);

  for (int node_id = 0; node_id < physical_end; ++node_id) {
    const T width = node_size_x[node_id];
    const T height = node_size_y[node_id];
    if (width <= 0 || height <= 0) {
      continue;
    }
    const bool movable =
        node_id < num_movable_nodes && height <= row_height + tolerance;
    if (movable) {
      int row_id = static_cast<int>(
          std::floor((y[node_id] + height / 2 - yl) / row_height));
      row_id = std::max(0, std::min(num_rows - 1, row_id));
      rows[row_id].push_back(
          {node_id, x[node_id], x[node_id] + width, true});
    } else {
      int row_begin =
          static_cast<int>(std::floor((y[node_id] - yl) / row_height));
      int row_end = static_cast<int>(
          std::ceil((y[node_id] + height - yl) / row_height) - 1);
      row_begin = std::max(0, row_begin);
      row_end = std::min(num_rows - 1, row_end);
      for (int row_id = row_begin; row_id <= row_end; ++row_id) {
        rows[row_id].push_back(
            {node_id, x[node_id], x[node_id] + width, false});
      }
    }
  }

  std::vector<std::vector<std::pair<T, T>>> row_rails(num_rows);
  for (int rail_id = 0; rail_id < num_rails; ++rail_id) {
    const T* rail = rail_boxes + rail_id * 4;
    int row_begin =
        static_cast<int>(std::floor((rail[1] - yl) / row_height));
    int row_end =
        static_cast<int>(std::ceil((rail[3] - yl) / row_height) - 1);
    row_begin = std::max(0, row_begin);
    row_end = std::min(num_rows - 1, row_end);
    const T rail_lx = std::max(xl, rail[0]);
    const T rail_hx = std::min(xh, rail[2]);
    if (rail_lx >= rail_hx) {
      continue;
    }
    for (int row_id = row_begin; row_id <= row_end; ++row_id) {
      row_rails[row_id].emplace_back(rail_lx, rail_hx);
    }
  }

  int move_events = 0;
  int skipped_illegal_rows = 0;
  const T max_displacement = max_displacement_sites * site_width;
  for (int row_id = 0; row_id < num_rows; ++row_id) {
    auto& row = rows[row_id];
    auto& rails = row_rails[row_id];
    if (row.empty() || rails.empty()) {
      continue;
    }
    std::sort(row.begin(), row.end(), [](const RowItem<T>& lhs,
                                         const RowItem<T>& rhs) {
      if (lhs.lx != rhs.lx) return lhs.lx < rhs.lx;
      if (lhs.hx != rhs.hx) return lhs.hx > rhs.hx;
      return lhs.node_id < rhs.node_id;
    });
    std::sort(rails.begin(), rails.end());

    std::vector<RowItem<T>> normalized;
    bool row_illegal = false;
    for (const auto& item : row) {
      if (normalized.empty() ||
          normalized.back().hx <= item.lx + tolerance) {
        normalized.push_back(item);
      } else if (!normalized.back().movable && !item.movable) {
        normalized.back().lx = std::min(normalized.back().lx, item.lx);
        normalized.back().hx = std::max(normalized.back().hx, item.hx);
        normalized.back().node_id = -1;
      } else {
        row_illegal = true;
        break;
      }
    }
    if (row_illegal) {
      ++skipped_illegal_rows;
      continue;
    }
    row.swap(normalized);

    for (const auto& rail : rails) {
      for (int cell_index = 0;
           cell_index < static_cast<int>(row.size()); ++cell_index) {
        RowItem<T>& item = row[cell_index];
        if (item.hx <= rail.first + tolerance) {
          continue;
        }
        if (item.lx >= rail.second - tolerance) {
          break;
        }
        if (!item.movable) {
          continue;
        }

        MoveCandidate<T> left = buildLeftCandidate(
            row, cell_index, rail.first, xl, site_width, max_neighbors,
            max_displacement, initial_x.data());
        MoveCandidate<T> right = buildRightCandidate(
            row, cell_index, rail.second, xl, xh, site_width, max_neighbors,
            max_displacement, initial_x.data());
        const MoveCandidate<T>* chosen = nullptr;
        if (left.valid && (!right.valid || left.displacement <= right.displacement)) {
          chosen = &left;
        } else if (right.valid) {
          chosen = &right;
        }
        if (chosen != nullptr) {
          applyCandidate(row, *chosen, x);
          ++move_events;
        }
      }
    }
  }

  const auto after = computeRailOverlap(
      x, y, node_size_x, node_size_y, rail_boxes, num_rails,
      num_movable_nodes, row_height);
  int moved_count = 0;
  double total_displacement = 0;
  double max_final_displacement = 0;
  for (int node_id = 0; node_id < num_movable_nodes; ++node_id) {
    const double displacement =
        std::abs(static_cast<double>(x[node_id] - initial_x[node_id]));
    if (displacement > static_cast<double>(site_width) * 1e-4) {
      ++moved_count;
      total_displacement += displacement;
      max_final_displacement = std::max(max_final_displacement, displacement);
    }
  }

  stats[0] = before.first;
  stats[1] = before.second;
  stats[2] = after.first;
  stats[3] = after.second;
  stats[4] = moved_count;
  stats[5] = total_displacement;
  stats[6] = max_final_displacement;
  stats[7] = move_events;
  stats[8] = skipped_illegal_rows;
}

std::vector<at::Tensor> m2_pa_refine_forward(
    at::Tensor pos, at::Tensor node_size_x, at::Tensor node_size_y,
    at::Tensor rail_boxes, double xl, double yl, double xh, double yh,
    double site_width, double row_height, int num_movable_nodes,
    int num_terminals, int max_neighbors, double max_displacement_sites) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_CPU(node_size_x);
  CHECK_CPU(node_size_y);
  CHECK_CPU(rail_boxes);
  CHECK_CONTIGUOUS(node_size_x);
  CHECK_CONTIGUOUS(node_size_y);
  CHECK_CONTIGUOUS(rail_boxes);
  AT_ASSERTM(rail_boxes.ndimension() == 2 && rail_boxes.size(1) == 4,
             "rail_boxes must have shape [N, 4]");
  AT_ASSERTM(pos.scalar_type() == node_size_x.scalar_type() &&
                 pos.scalar_type() == node_size_y.scalar_type() &&
                 pos.scalar_type() == rail_boxes.scalar_type(),
             "all input tensors must have the same floating-point dtype");
  const int num_nodes = pos.numel() / 2;
  AT_ASSERTM(node_size_x.numel() == num_nodes &&
                 node_size_y.numel() == num_nodes,
             "node-size tensors must match the position tensor");
  AT_ASSERTM(num_movable_nodes >= 0 && num_terminals >= 0 &&
                 num_movable_nodes + num_terminals <= num_nodes,
             "invalid movable/fixed node counts");
  AT_ASSERTM(site_width > 0 && row_height > 0 && xh > xl && yh > yl,
             "invalid placement geometry");
  AT_ASSERTM(max_neighbors > 0 && max_displacement_sites >= 0,
             "invalid PA-Refine search limits");

  at::Tensor output = pos.clone();
  at::Tensor stats = at::zeros(
      {9}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
  DREAMPLACE_DISPATCH_FLOATING_TYPES(output, "runM2PARefine", [&] {
    runM2PARefine<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(output, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(output, scalar_t) + num_nodes,
        DREAMPLACE_TENSOR_DATA_PTR(node_size_x, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(node_size_y, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(rail_boxes, scalar_t), num_nodes,
        rail_boxes.size(0), static_cast<scalar_t>(xl),
        static_cast<scalar_t>(yl), static_cast<scalar_t>(xh),
        static_cast<scalar_t>(yh), static_cast<scalar_t>(site_width),
        static_cast<scalar_t>(row_height), num_movable_nodes, num_terminals,
        max_neighbors, static_cast<scalar_t>(max_displacement_sites),
        DREAMPLACE_TENSOR_DATA_PTR(stats, double));
  });
  return {output, stats};
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::m2_pa_refine_forward,
        "M2 PA-Refine forward");
}
