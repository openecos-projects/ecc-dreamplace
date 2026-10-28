#include "utility/src/cpu_tensor.h"
#include "utility/src/torch.h"
#include "timing_scatter_groups.h"
#include "timing_math.h"
#include <ATen/Parallel.h>
#include <omp.h>

namespace {
void checkInputs(const at::Tensor& dest, const at::Tensor& index, const at::Tensor& src) {
  for (const auto& value : {dest, index, src}) {
    TORCH_CHECK(value.device().is_cpu() && value.is_contiguous() && value.dim() == 1,
                "timing scatter expects contiguous CPU vectors");
  }
  TORCH_CHECK(index.scalar_type() == at::kLong && index.sizes() == src.sizes() &&
              dest.scalar_type() == src.scalar_type(), "timing scatter source/index mismatch");
  const auto* ids = index.data_ptr<int64_t>();
  for (int64_t i = 0; i < index.numel(); ++i) {
    TORCH_CHECK(ids[i] >= 0 && ids[i] < dest.numel(), "timing scatter index out of range");
  }
}
}

std::vector<at::Tensor> hard_scatter_forward_cpu(
    at::Tensor dest, at::Tensor index, at::Tensor src, bool include_self) {
  checkInputs(dest, index, src);
  auto output = dest.clone();
  output.scatter_reduce_(0, index, src, "amax", include_self);
  auto counts = fullyWrittenCpuTensor(dest.sizes(), dest.options()), touched = fullyWrittenCpuTensor(dest.sizes(), dest.options().dtype(at::kInt));
  const auto groups = timingScatterGroups(index, dest.numel());
  const auto* order = groups.order.data_ptr<int64_t>();
  const auto* offsets = groups.offsets.data_ptr<int64_t>();
  auto* seen = touched.data_ptr<int32_t>();
  const auto compute = [&](auto sample) {
    using scalar_t = decltype(sample);
    const auto* d = dest.data_ptr<scalar_t>(); const auto* s = src.data_ptr<scalar_t>();
    const auto* out = output.data_ptr<scalar_t>(); auto* n = counts.data_ptr<scalar_t>();
    #pragma omp parallel for if(dest.numel() >= 2048) num_threads(at::get_num_threads()) schedule(static)
    for (int64_t pin = 0; pin < dest.numel(); ++pin) {
      scalar_t count = scalar_t(d[pin] == out[pin]);
      for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) {
        count += scalar_t(s[order[row]] == out[pin]);
      }
      n[pin] = count; seen[pin] = offsets[pin] != offsets[pin + 1];
    }
  };
  AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "hardScatterCpuForward", [&] { compute(scalar_t(0)); });
  return {output, counts, touched};
}

std::vector<at::Tensor> hard_scatter_backward_cpu(at::Tensor dest, at::Tensor index, at::Tensor src,
    at::Tensor output, at::Tensor counts, at::Tensor touched, at::Tensor gradient, bool include_self) {
  checkInputs(dest, index, src);
  for (const auto& value : {output, counts, touched, gradient}) {
    TORCH_CHECK(value.device().is_cpu() && value.is_contiguous() && value.sizes() == dest.sizes(),
                "timing scatter adjoint shape mismatch");
  }
  TORCH_CHECK(touched.scalar_type() == at::kInt && output.scalar_type() == dest.scalar_type() &&
              counts.scalar_type() == dest.scalar_type() && gradient.scalar_type() == dest.scalar_type(),
              "timing scatter adjoint dtype mismatch");
  auto gd = fullyWrittenCpuTensor(dest.sizes(), dest.options()), gs = fullyWrittenCpuTensor(src.sizes(), src.options());
  const auto* ids = index.data_ptr<int64_t>(); const auto* seen = touched.data_ptr<int32_t>();
  const auto compute = [&](auto sample) {
    using scalar_t = decltype(sample);
    const auto* d = dest.data_ptr<scalar_t>(); const auto* s = src.data_ptr<scalar_t>();
    const auto* out = output.data_ptr<scalar_t>(); const auto* n = counts.data_ptr<scalar_t>();
    const auto* g = gradient.data_ptr<scalar_t>(); auto* dp = gd.data_ptr<scalar_t>();
    auto* sp = gs.data_ptr<scalar_t>();
    #pragma omp parallel for if(dest.numel() >= 2048) num_threads(at::get_num_threads()) schedule(static)
    for (int64_t pin = 0; pin < dest.numel(); ++pin) {
      dp[pin] = !include_self && seen[pin] ? scalar_t(0) : (g[pin] / n[pin]) * scalar_t(d[pin] == out[pin]);
    }
    #pragma omp parallel for if(src.numel() >= 2048) num_threads(at::get_num_threads()) schedule(static)
    for (int64_t row = 0; row < src.numel(); ++row) {
      const auto pin = ids[row];
      sp[row] = (g[pin] / n[pin]) * scalar_t(s[row] == out[pin]);
    }
  };
  AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "hardScatterCpuBackward", [&] { compute(scalar_t(0)); });
  return {gd, gs};
}

std::vector<at::Tensor> smooth_scatter_cpu(at::Tensor dest, at::Tensor index, at::Tensor src,
    double tau, bool include_self, std::vector<at::Tensor> prepared_groups) {
  checkInputs(dest, index, src);
  TORCH_CHECK(tau > 0, "timing scatter temperature must be positive");
  const auto groups = timingScatterGroups(index, dest.numel(), prepared_groups);
  const auto* order = groups.order.data_ptr<int64_t>(); const auto* offsets = groups.offsets.data_ptr<int64_t>();
  auto output = fullyWrittenCpuTensor(dest.sizes(), dest.options()), dw = fullyWrittenCpuTensor(dest.sizes(), dest.options()), sw = fullyWrittenCpuTensor(src.sizes(), src.options());
  const auto compute = [&](auto sample) {
    using scalar_t = decltype(sample);
    const auto* d = dest.data_ptr<scalar_t>(); const auto* s = src.data_ptr<scalar_t>();
    auto* out = output.data_ptr<scalar_t>(); auto* df = dw.data_ptr<scalar_t>(); auto* sf = sw.data_ptr<scalar_t>();
    const scalar_t temperature = scalar_t(tau);
    auto contribution = [temperature](scalar_t v, scalar_t m) {
      return isfinite(m) ? exp((v - m) / temperature) : scalar_t(v == m);
    };
    #pragma omp parallel for if(dest.numel() >= 2048) num_threads(at::get_num_threads()) schedule(static)
    for (int64_t pin = 0; pin < dest.numel(); ++pin) {
      scalar_t maximum = include_self ? d[pin] : scalar_t(-INFINITY);
      for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) {
        const scalar_t value = s[order[row]];
        maximum = isnan(maximum) ? maximum : (isnan(value) ? value : max(maximum, value));
      }
      const scalar_t tie = isinf(maximum) && maximum < 0 ? scalar_t(1) : scalar_t(0);
      scalar_t sum = tie;
      for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) sum += contribution(s[order[row]], maximum);
      if (include_self) sum += contribution(d[pin], maximum);
      if (!include_self && sum == tie && !isnan(maximum)) { out[pin] = d[pin]; df[pin] = 1; }
      else {
        out[pin] = isfinite(maximum) ? maximum + temperature * log(max(sum, scalar_t(1e-30))) : maximum;
        df[pin] = include_self ? contribution(d[pin], maximum) / sum : scalar_t(0);
      }
      for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) {
        const auto arc = order[row]; sf[arc] = contribution(s[arc], maximum) / sum;
      }
    }
  };
  AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "smoothScatterCpu", [&] { compute(scalar_t(0)); });
  return {output, dw, sw};
}
