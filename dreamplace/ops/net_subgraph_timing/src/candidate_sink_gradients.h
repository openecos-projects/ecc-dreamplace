#pragma once

#include "utility/src/torch.h"
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
void candidateSinkGradientsLauncher(
    const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
    const scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, cudaStream_t);

// Each forest net owns its nodes; stable sink order also handles repeated sinks.
inline void seedCandidateSinkGradients(
    const at::Tensor& topology, const at::Tensor& starts,
    const at::Tensor& sinks, const at::Tensor& sink_nets,
    const at::Tensor& slew_out,
    const at::Tensor& grad_sink_arrival, const at::Tensor& grad_sink_slew,
    const at::Tensor& grad_sink_load, const at::Tensor& grad_sink_delay,
    const at::Tensor& grad_sink_impulse,
    at::Tensor& grad_lin, at::Tensor& grad_arrival_out, at::Tensor& grad_slew_out) {
  if (sinks.numel() == 0) return;
  const int nets = starts.numel() - 1;
  const auto order = at::argsort(sink_nets, /*stable=*/true, 0, false);
  const auto counts = at::bincount(sink_nets, {}, nets);
  const auto offsets = at::cat({at::zeros({1}, sink_nets.options()), counts.cumsum(0)});
  const auto stream = c10::cuda::getCurrentCUDAStream(sinks.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(slew_out.scalar_type(), "candidateSinkGradients", [&] {
    candidateSinkGradientsLauncher<scalar_t>(topology.data_ptr<int64_t>(), starts.data_ptr<int64_t>(),
        sinks.data_ptr<int64_t>(), order.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        slew_out.data_ptr<scalar_t>(), grad_sink_arrival.data_ptr<scalar_t>(),
        grad_sink_slew.data_ptr<scalar_t>(), grad_sink_load.data_ptr<scalar_t>(),
        grad_sink_delay.data_ptr<scalar_t>(), grad_sink_impulse.data_ptr<scalar_t>(),
        grad_lin.data_ptr<scalar_t>(), grad_arrival_out.data_ptr<scalar_t>(),
        grad_slew_out.data_ptr<scalar_t>(), nets, stream);
  });
}
