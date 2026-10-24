#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace {

constexpr int kActions = 4;
constexpr int kSizeUp = 0;
constexpr int kSizeDown = 1;
constexpr int kVtSlowdown = 2;
constexpr int kVtSpeedup = 3;

int64_t select_count(int64_t candidates, double percent) {
  if (candidates <= 0 || percent <= 0.0) {
    return 0;
  }
  const auto count = static_cast<int64_t>(std::ceil(
      static_cast<double>(candidates) * percent / 100.0));
  return std::min<int64_t>(candidates, std::max<int64_t>(1, count));
}

template <typename Scalar>
void select_one_direction(
    const Scalar* scores,
    const bool* valid,
    int64_t rows,
    int action,
    double percent,
    bool* selected) {
  std::vector<std::pair<Scalar, int64_t>> candidates;
  candidates.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    const auto score = scores[row * kActions + action];
    if (valid[row * kActions + action] && std::isfinite(score) && score < 0) {
      candidates.emplace_back(score, row);
    }
  }
  const auto count = select_count(static_cast<int64_t>(candidates.size()), percent);
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (lhs.first != rhs.first) {
                       return lhs.first < rhs.first;
                     }
                     return lhs.second < rhs.second;
                   });
  for (int64_t index = 0; index < count; ++index) {
    selected[candidates[static_cast<size_t>(index)].second * kActions + action] =
        true;
  }
}

template <typename Scalar>
void select_vt(
    const Scalar* scores,
    const bool* valid,
    int64_t rows,
    double percent,
    bool* selected) {
  std::vector<std::tuple<Scalar, int64_t, int>> candidates;
  candidates.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    int best_action = -1;
    Scalar best_score = std::numeric_limits<Scalar>::infinity();
    for (int action = kVtSlowdown; action <= kVtSpeedup; ++action) {
      const auto score = scores[row * kActions + action];
      if (!valid[row * kActions + action] || !std::isfinite(score) || score >= 0) {
        continue;
      }
      if (best_action < 0 || score < best_score ||
          (score == best_score && action < best_action)) {
        best_score = score;
        best_action = action;
      }
    }
    if (best_action >= 0) {
      candidates.emplace_back(best_score, row, best_action);
    }
  }
  const auto count = select_count(static_cast<int64_t>(candidates.size()), percent);
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (std::get<0>(lhs) != std::get<0>(rhs)) {
                       return std::get<0>(lhs) < std::get<0>(rhs);
                     }
                     return std::get<1>(lhs) < std::get<1>(rhs);
                   });
  for (int64_t index = 0; index < count; ++index) {
    const auto& item = candidates[static_cast<size_t>(index)];
    selected[std::get<1>(item) * kActions + std::get<2>(item)] = true;
  }
}

template <typename Scalar>
void select_shared(
    const Scalar* scores,
    const bool* valid,
    const bool* blocked,
    int64_t rows,
    int64_t budget,
    bool apply_blocked,
    bool* selected) {
  std::vector<std::tuple<Scalar, int64_t, int>> candidates;
  candidates.reserve(static_cast<size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    if (apply_blocked && blocked[row]) {
      continue;
    }
    int best_action = -1;
    Scalar best_score = std::numeric_limits<Scalar>::infinity();
    for (int action = 0; action < kActions; ++action) {
      const auto score = scores[row * kActions + action];
      if (!valid[row * kActions + action] || !std::isfinite(score) || score >= 0) {
        continue;
      }
      if (best_action < 0 || score < best_score ||
          (score == best_score && action < best_action)) {
        best_score = score;
        best_action = action;
      }
    }
    if (best_action >= 0) {
      candidates.emplace_back(best_score, row, best_action);
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& lhs, const auto& rhs) {
                     if (std::get<0>(lhs) != std::get<0>(rhs)) {
                       return std::get<0>(lhs) < std::get<0>(rhs);
                     }
                     return std::get<1>(lhs) < std::get<1>(rhs);
                   });
  const auto count = std::min<int64_t>(budget, candidates.size());
  for (int64_t index = 0; index < count; ++index) {
    const auto& item = candidates[static_cast<size_t>(index)];
    selected[std::get<1>(item) * kActions + std::get<2>(item)] = true;
  }
}

template <typename Scalar>
std::vector<at::Tensor> select_quad_actions_impl(
    const at::Tensor& scores,
    const at::Tensor& valid,
    const at::Tensor& sizeable,
    const at::Tensor& blocked,
    double shared_budget_percent,
    double size_up_percent,
    double size_down_percent,
    double vt_percent) {
  const auto rows = scores.size(0);
  const auto score_ptr = scores.data_ptr<Scalar>();
  const auto valid_ptr = valid.data_ptr<bool>();
  const auto sizeable_ptr = sizeable.data_ptr<bool>();
  const bool has_blocked = blocked.numel() != 0;
  const auto blocked_ptr = has_blocked ? blocked.data_ptr<bool>() : nullptr;

  auto selected = at::zeros({rows, kActions}, valid.options());
  auto baseline = at::zeros({rows, kActions}, valid.options());
  auto* selected_ptr = selected.data_ptr<bool>();
  auto* baseline_ptr = baseline.data_ptr<bool>();

  int64_t budget = -1;
  if (shared_budget_percent >= 0.0) {
    TORCH_CHECK(shared_budget_percent <= 100.0,
                "shared budget percentage must be in [0, 100]");
    int64_t sizeable_count = 0;
    for (int64_t row = 0; row < rows; ++row) {
      sizeable_count += sizeable_ptr[row] ? 1 : 0;
    }
    budget = select_count(sizeable_count, shared_budget_percent);
    select_shared(score_ptr, valid_ptr, nullptr, rows, budget, false,
                  baseline_ptr);
    if (has_blocked) {
      select_shared(score_ptr, valid_ptr, blocked_ptr, rows, budget, true,
                    selected_ptr);
    } else {
      select_shared(score_ptr, valid_ptr, nullptr, rows, budget, false,
                    selected_ptr);
    }
  } else {
    select_one_direction(score_ptr, valid_ptr, rows, kSizeUp, size_up_percent,
                         selected_ptr);
    select_one_direction(score_ptr, valid_ptr, rows, kSizeDown,
                         size_down_percent, selected_ptr);
    select_vt(score_ptr, valid_ptr, rows, vt_percent, selected_ptr);
  }

  auto budget_tensor = at::tensor({budget}, at::TensorOptions().dtype(at::kLong));
  return {selected, baseline, budget_tensor};
}

}  // namespace

std::vector<at::Tensor> select_quad_actions(
    at::Tensor scores,
    at::Tensor valid,
    at::Tensor sizeable,
    at::Tensor blocked,
    double shared_budget_percent,
    double size_up_percent,
    double size_down_percent,
    double vt_percent) {
  TORCH_CHECK(scores.device().is_cpu(), "quad selection C++ op expects CPU tensors");
  TORCH_CHECK(scores.dim() == 2 && scores.size(1) == kActions,
              "scores must have shape [N, 4]");
  TORCH_CHECK(valid.device().is_cpu() && valid.scalar_type() == at::kBool,
              "valid must be a CPU bool tensor");
  TORCH_CHECK(sizeable.device().is_cpu() && sizeable.scalar_type() == at::kBool,
              "sizeable must be a CPU bool tensor");
  TORCH_CHECK(valid.sizes() == scores.sizes(), "valid must match scores");
  TORCH_CHECK(sizeable.dim() == 1 && sizeable.size(0) == scores.size(0),
              "sizeable must have shape [N]");
  TORCH_CHECK(blocked.numel() == 0 ||
                  (blocked.device().is_cpu() && blocked.scalar_type() == at::kBool &&
                   blocked.dim() == 1 && blocked.size(0) == scores.size(0)),
              "blocked must be empty or a CPU bool tensor with shape [N]");
  TORCH_CHECK(scores.scalar_type() == at::kFloat ||
                  scores.scalar_type() == at::kDouble,
              "scores must be float32 or float64");

  scores = scores.contiguous();
  valid = valid.contiguous();
  sizeable = sizeable.contiguous();
  blocked = blocked.contiguous();
  if (scores.scalar_type() == at::kDouble) {
    return select_quad_actions_impl<double>(
        scores, valid, sizeable, blocked, shared_budget_percent,
        size_up_percent, size_down_percent, vt_percent);
  }
  return select_quad_actions_impl<float>(
      scores, valid, sizeable, blocked, shared_budget_percent, size_up_percent,
      size_down_percent, vt_percent);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_quad_actions", &select_quad_actions,
        "Select legal four-action candidates");
}
