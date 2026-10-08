/**
 * @file   pin2pin_attraction.cpp (modified weighted_average_wirelength.cpp)
 * @author Xi Lin (Yibo Lin)
 * @date   Dec 2023 (Jun 2018)
 * @brief  Compute Pin2PinAttraction and gradient
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "pin2pin_attraction/src/functional.h"
#include <cassert>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
T pin2pinAttractionLauncher(
  const T *pin_pos_x, const T *pin_pos_y,
  pybind11::dict& pin2pin_net_weight,
  int num_pins,
  int num_threads,
  const T *grad_tensor,
  T *grad_x_tensor, T *grad_y_tensor
);


std::vector<at::Tensor> pin2pin_attraction_forward(
    at::Tensor pin_pos,
    pybind11::dict& pin2pin_net_weight
) {
  CHECK_FLAT_CPU(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  int num_pins = pin_pos.numel() / 2;

  at::Tensor total_distance = at::zeros({}, pin_pos.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pin_pos, "pin2pinAttractionLauncher", [&] {
        // template <typename T>
        total_distance = at::scalar_tensor(pin2pinAttractionLauncher<scalar_t>(
            // const T *pin_pos_x, *pin_pos_y
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
            // dict pin2pin_net_weight
            pin2pin_net_weight,
            // int num_pins
            num_pins,
            // int num_threads
            at::get_num_threads(),
            // const T *grad_tensor
            nullptr,
            // T *grad_x_tensor, *grad_y_tensor
            nullptr,
            nullptr
        ), pin_pos.options());
      });
  return {total_distance};
}

at::Tensor pin2pin_attraction_backward(
    at::Tensor grad,
    at::Tensor pin_pos,
    pybind11::dict& pin2pin_net_weight
) {
  CHECK_FLAT_CPU(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  int num_pins = pin_pos.numel() / 2; 

  at::Tensor grad_out = at::zeros_like(pin_pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      pin_pos, "pin2pinAttractionLauncher", [&] {
        // template <typename T>
        pin2pinAttractionLauncher<scalar_t>(
            // const T *pin_pos_x, *pin_pos_y
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
            // dict pin2pin_net_weight
            pin2pin_net_weight,
            // int num_pins
            num_pins,
            // int num_threads
            at::get_num_threads(),
            // T *grad_tensor
            DREAMPLACE_TENSOR_DATA_PTR(grad, scalar_t),
            // T *grad_x_tensor, *grad_y_tensor
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(grad_out, scalar_t) + num_pins
          );
      });
  return grad_out;
}

namespace py = pybind11;

template <typename T>
T pin2pinAttractionLauncher(
    const T *pin_pos_x, const T *pin_pos_y,
    py::dict& pin2pin_net_weight,
    int num_pins, int num_threads,
    const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor
) {
    if (grad_tensor) {
        // backward
        // #pragma omp parallel for num_threads(num_threads)
        for (auto item : pin2pin_net_weight) {
            auto key = item.first.cast<py::tuple>();
            int pin_id1 = key[0].cast<int>();
            int pin_id2 = key[1].cast<int>();
            T weight = item.second.cast<float>();

            T dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
            T dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];

            T grad_x = 2 * (*grad_tensor) * weight * dx;
            T grad_y = 2 * (*grad_tensor) * weight * dy;

            // #pragma omp atomic
            grad_x_tensor[pin_id1] += grad_x;
            // #pragma omp atomic
            grad_y_tensor[pin_id1] += grad_y;

            // #pragma omp atomic
            grad_x_tensor[pin_id2] -= grad_x;
            // #pragma omp atomic
            grad_y_tensor[pin_id2] -= grad_y;
        }
    } else {
        // forward
        T total_distance = 0;
        // #pragma omp parallel for reduction(+:total_distance) num_threads(num_threads)
        for (auto item : pin2pin_net_weight) {
            auto key = item.first.cast<py::tuple>();
            int pin_id1 = key[0].cast<int>();
            int pin_id2 = key[1].cast<int>();
            T weight = item.second.cast<float>();
            T dx = pin_pos_x[pin_id1] - pin_pos_x[pin_id2];
            T dy = pin_pos_y[pin_id1] - pin_pos_y[pin_id2];
            total_distance += weight * (dx * dx + dy * dy);
        }
        return total_distance;
    }
    return T(0);
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::pin2pin_attraction_forward,
        "Pin2PinAttraction forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::pin2pin_attraction_backward,
        "Pin2PinAttraction backward");
}
