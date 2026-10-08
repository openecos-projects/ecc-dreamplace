/**
 * @brief Compute deterministic weighted Pin2Pin attraction and gradient.
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void pin2pinAttractionCudaForwardLauncher(
    const T *, const T *, const int *, const T *, int, T *, T *, cudaStream_t);

template <typename T>
void pin2pinAttractionCudaBackwardLauncher(
    const T *, const T *, const int *, const T *, int, const int *,
    const int64_t *, const T *, T *, T *, cudaStream_t);

static void checkInputs(const at::Tensor &pin_pos, const at::Tensor &pairs,
                        const at::Tensor &weights) {
  CHECK_FLAT_CUDA(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);
  CHECK_FLAT_CUDA(pairs);
  CHECK_EVEN(pairs);
  CHECK_CONTIGUOUS(pairs);
  CHECK_FLAT_CUDA(weights);
  CHECK_CONTIGUOUS(weights);
  TORCH_CHECK(pairs.scalar_type() == at::kInt, "pairs must be int32");
  TORCH_CHECK(weights.scalar_type() == pin_pos.scalar_type(),
              "weights and pin_pos must have the same dtype");
  TORCH_CHECK(pairs.device() == pin_pos.device() && weights.device() == pin_pos.device(),
              "inputs must be on the same CUDA device");
  TORCH_CHECK(weights.numel() == pairs.numel() / 2, "one weight is required per pair");
}

std::vector<at::Tensor> pin2pin_attraction_forward(
    at::Tensor pin_pos, at::Tensor pairs, at::Tensor weights) {
  checkInputs(pin_pos, pairs, weights);
  const c10::cuda::CUDAGuard guard(pin_pos.device());
  int num_pins = pin_pos.numel() / 2;
  int num_pairs = pairs.numel() / 2;
  at::Tensor total = at::zeros({}, pin_pos.options());
  if (!num_pairs) return {total};
  at::Tensor partials = at::empty({(num_pairs + 255) / 256}, pin_pos.options());
  auto stream = c10::cuda::getCurrentCUDAStream();
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "pin2pinAttractionForward", [&] {
    auto pos = DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t);
    pin2pinAttractionCudaForwardLauncher<scalar_t>(
        pos, pos + num_pins, DREAMPLACE_TENSOR_DATA_PTR(pairs, int),
        DREAMPLACE_TENSOR_DATA_PTR(weights, scalar_t), num_pairs,
        DREAMPLACE_TENSOR_DATA_PTR(partials, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(total, scalar_t), stream);
  });
  return {total};
}

at::Tensor pin2pin_attraction_backward(
    at::Tensor grad, at::Tensor pin_pos, at::Tensor pairs, at::Tensor weights) {
  checkInputs(pin_pos, pairs, weights);
  CHECK_CONTIGUOUS(grad);
  TORCH_CHECK(grad.numel() == 1 && grad.device() == pin_pos.device() &&
              grad.scalar_type() == pin_pos.scalar_type(),
              "grad must be a scalar matching pin_pos dtype and device");
  const c10::cuda::CUDAGuard guard(pin_pos.device());
  int num_pins = pin_pos.numel() / 2;
  at::Tensor grad_out = at::zeros_like(pin_pos);
  if (!pairs.numel()) return grad_out;
  // Sort endpoint IDs stably, retaining their original pair/direction indices.
  // Rebuild each call because the caller can update pairs in place.
  auto sorted = at::sort(pairs, /*stable=*/true, /*dim=*/0, /*descending=*/false);
  auto stream = c10::cuda::getCurrentCUDAStream();
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "pin2pinAttractionBackward", [&] {
    auto pos = DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t);
    auto out = DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t);
    pin2pinAttractionCudaBackwardLauncher<scalar_t>(
        pos, pos + num_pins, DREAMPLACE_TENSOR_DATA_PTR(pairs, int),
        DREAMPLACE_TENSOR_DATA_PTR(weights, scalar_t), pairs.numel(),
        DREAMPLACE_TENSOR_DATA_PTR(std::get<0>(sorted), int),
        DREAMPLACE_TENSOR_DATA_PTR(std::get<1>(sorted), int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad, scalar_t), out, out + num_pins, stream);
  });
  return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin2pin_attraction_forward,
        "Pin2PinAttraction forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::pin2pin_attraction_backward,
        "Pin2PinAttraction backward (CUDA)");
}
