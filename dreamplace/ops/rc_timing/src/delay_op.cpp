
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <pybind11/detail/common.h>
#include <queue>
#include <torch/torch.h>
#include "rc_timing.hh"

// Forward declaration
template <typename scalar_t>
void delayForwardLauncher(
    const scalar_t *resistance_ptr, const scalar_t *load_ptr,
    const int32_t *pin_fa_ptr, // Parent index for each node (-1 for roots)
    const int32_t
        *topo_sort_ptr, // Assumed post-order, used forward for top-down
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *delay_ptr // Output
);

/**
 * @brief Computes Elmore delay (forward pass) using a templated launcher.
 * Delay(u) = Delay(fa(u)) + Res(fa(u)->u) * Load(u)
 * Calculation is top-down.
 *
 * @param resistance_tensor Resistance tensor (1D, float/double).
 * resistance_tensor[u] is assumed to be r_{fa(u)->u}.
 * @param load_tensor Load tensor from Load operator (1D, float/double).
 * @param pin_fa_tensor Parent pin index tensor (1D, int64). Root nodes should
 * have parent index < 0 (e.g., -1).
 * @param net_driver_pin_tensor Driver pin tensor (1D, int64). Unused if pin_fa
 * identifies roots.
 * @param net_flat_topo Flattened topological sort (assumed post-order, traverse
 * FORWARD for top-down) (1D, int64).
 * @param net_flat_topo_start Start indices for nets in topo sort (1D, int64,
 * size=num_nets+1).
 * @return at::Tensor The calculated delay tensor (1D, same dtype as inputs).
 */
at::Tensor
delay_forward_cpp(at::Tensor resistance_tensor, at::Tensor load_tensor,
                  at::Tensor pin_fa_tensor,
                  at::Tensor net_flat_topo, at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(resistance_tensor);
  CHECK_FLAT(resistance_tensor);
  CHECK_CONTIGUOUS(resistance_tensor);
  CHECK_CPU(load_tensor);
  CHECK_FLAT(load_tensor);
  CHECK_CONTIGUOUS(load_tensor);
  CHECK_CPU(pin_fa_tensor);
  CHECK_FLAT(pin_fa_tensor);
  CHECK_CONTIGUOUS(pin_fa_tensor);
  CHECK_CPU(net_flat_topo);
  CHECK_FLAT(net_flat_topo);
  CHECK_CONTIGUOUS(net_flat_topo);
  CHECK_CPU(net_flat_topo_start);
  CHECK_FLAT(net_flat_topo_start);
  CHECK_CONTIGUOUS(net_flat_topo_start);

  TORCH_CHECK(resistance_tensor.scalar_type() == load_tensor.scalar_type(),
              "Input dtypes mismatch");
  TORCH_CHECK(pin_fa_tensor.scalar_type() == at::kInt,
              "pin_fa_tensor must be int64 (Long)");
  TORCH_CHECK(net_flat_topo.scalar_type() == at::kInt,
              "net_flat_topo must be int64 (Long)");
  TORCH_CHECK(net_flat_topo_start.scalar_type() == at::kInt,
              "net_flat_topo_start must be int64 (Long)");

  int32_t num_nodes = load_tensor.numel();
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(resistance_tensor.numel() == num_nodes,
              "resistance_tensor size mismatch");
  TORCH_CHECK(pin_fa_tensor.numel() == num_nodes,
              "pin_fa_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensor ---
  // Delay is accumulated, initialize with zeros. Roots have delay 0.
  at::Tensor delay_tensor = at::zeros({num_nodes}, resistance_tensor.options());

  // --- Get Pointers ---
  const int32_t *pin_fa_ptr = pin_fa_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch ---
  AT_DISPATCH_FLOATING_TYPES(
      resistance_tensor.scalar_type(), "delayForwardLauncher", [&] {
        const scalar_t *resistance_ptr = resistance_tensor.data_ptr<scalar_t>();
        const scalar_t *load_ptr = load_tensor.data_ptr<scalar_t>();
        scalar_t *delay_ptr =
            delay_tensor.data_ptr<scalar_t>(); // Non-const for output

        delayForwardLauncher<scalar_t>(resistance_ptr, load_ptr, pin_fa_ptr,
                                       topo_sort_ptr, topo_sort_start_ptr,
                                       num_nodes, num_nets, delay_ptr);
      });

  return delay_tensor;
}

// --- Launcher Implementation ---
template <typename scalar_t>
void delayForwardLauncher(const scalar_t *resistance_ptr,
                          const scalar_t *load_ptr, const int32_t *pin_fa_ptr,
                          const int32_t *topo_sort_ptr,
                          const int32_t *topo_sort_start_ptr, int32_t num_nodes,
                          int32_t num_nets,
                          scalar_t *delay_ptr // Output
) {
  // Iterate through each net
  for (int32_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    int32_t start_topo_idx = topo_sort_start_ptr[net_idx];
    int32_t end_topo_idx = topo_sort_start_ptr[net_idx + 1];

    // Iterate nodes in FORWARD topological order (top-down)
    // Assumes net_flat_topo processes parents before children.
    for (int32_t i = start_topo_idx; i < end_topo_idx; ++i) {
      int32_t u = topo_sort_ptr[i]; // Current node index

      if (u < 0 || u >= num_nodes) {
        continue;
      } // Bounds check

      int32_t parent_idx = pin_fa_ptr[u]; // Get parent index

      // If parent_idx is valid (>= 0) and within bounds, it's not a root.
      if (parent_idx >= 0 && parent_idx < num_nodes) {
        // Delay(u) = Delay(parent) + Res(parent->u) * Load(u)
        // Assuming resistance_ptr[u] holds Res(parent->u)
        delay_ptr[u] = delay_ptr[parent_idx] + resistance_ptr[u] * load_ptr[u];
      } else {
        // Node u is a root (or parent info is invalid)
        // Delay is initialized to 0, so nothing to do unless a different
        // initial delay is required for roots.
        // delay_ptr[u] = 0.0; // Already initialized
      }
    } // End loop over nodes in net
  } // End loop over nets
}

// Forward declaration
template <typename scalar_t>
void delayBackwardLauncher(
    const scalar_t *grad_output_delay_ptr, // dF/dDelay (initial from next op)
    const scalar_t *resistance_ptr,        // Res(fa(u)->u) at index u
    const scalar_t *load_ptr,              // Load(u) at index u
    const int32_t *pin_fa_ptr, // Parent index for node u (<0 for root)
    const int32_t
        *topo_sort_ptr, // Parent-first order, use backward for bottom-up
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *accum_grad_delay_ptr, // Intermediate buffer (In/Out)
    scalar_t *grad_input_res_ptr,   // Output: dF/dRes
    scalar_t
        *grad_input_load_ptr // Output: dF/dLoad (contribution from this op)
);

/**
 * @brief Computes gradients for Elmore delay w.r.t. resistance and load.
 * Implements the backward pass logic described in the image.
 * Returns gradients for inputs: resistance_tensor, load_tensor.
 *
 * @param grad_output_delay Gradient w.r.t. the output delay (dF/dDelay).
 * @param resistance_tensor Resistance tensor used in forward.
 * @param load_tensor Load tensor used in forward.
 * @param pin_fa_tensor Parent pin index tensor used in forward.
 * @param net_driver_pin_tensor Unused.
 * @param net_flat_topo Flattened topological sort (parent-first, used BACKWARD
 * for bottom-up accumulation).
 * @param net_flat_topo_start Start indices for nets in topo sort.
 * @return std::vector<at::Tensor> Gradients [grad_input_res, grad_input_load].
 */
std::vector<at::Tensor>
delay_backward_cpp(at::Tensor grad_output_delay,
                   at::Tensor resistance_tensor, // From forward pass context
                   at::Tensor load_tensor,       // From forward pass context
                   at::Tensor pin_fa_tensor,     // From forward pass context
                   at::Tensor net_flat_topo, at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(grad_output_delay);
  CHECK_FLAT(grad_output_delay);
  CHECK_CONTIGUOUS(grad_output_delay);
  CHECK_CPU(resistance_tensor);
  CHECK_FLAT(resistance_tensor);
  CHECK_CONTIGUOUS(resistance_tensor);
  CHECK_CPU(load_tensor);
  CHECK_FLAT(load_tensor);
  CHECK_CONTIGUOUS(load_tensor);
  CHECK_CPU(pin_fa_tensor);
  CHECK_FLAT(pin_fa_tensor);
  CHECK_CONTIGUOUS(pin_fa_tensor);
  CHECK_CPU(net_flat_topo);
  CHECK_FLAT(net_flat_topo);
  CHECK_CONTIGUOUS(net_flat_topo);
  CHECK_CPU(net_flat_topo_start);
  CHECK_FLAT(net_flat_topo_start);
  CHECK_CONTIGUOUS(net_flat_topo_start);

  TORCH_CHECK(grad_output_delay.scalar_type() ==
                      resistance_tensor.scalar_type() &&
                  grad_output_delay.scalar_type() == load_tensor.scalar_type(),
              "Input floating point dtypes mismatch");
  TORCH_CHECK(pin_fa_tensor.scalar_type() == at::kInt,
              "pin_fa_tensor must be int64 (Long)");
  // ... other dtype checks

  int32_t num_nodes = load_tensor.numel();
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(grad_output_delay.numel() == num_nodes,
              "grad_output_delay size mismatch");
  TORCH_CHECK(resistance_tensor.numel() == num_nodes,
              "resistance_tensor size mismatch");
  TORCH_CHECK(pin_fa_tensor.numel() == num_nodes,
              "pin_fa_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensors (Initialize to Zero) ---
  at::Tensor grad_input_res =
      at::zeros_like(resistance_tensor, resistance_tensor.options());
  // grad_input_load is the gradient w.r.t. the Load *input* of this operator.
  at::Tensor grad_input_load =
      at::zeros_like(load_tensor, load_tensor.options());

  // --- Intermediate Tensor ---
  // For accumulating dF/dDelay bottom-up. Initialize with input gradient.
  at::Tensor accum_grad_delay = grad_output_delay.clone();

  // --- Get Pointers ---
  const int32_t *pin_fa_ptr = pin_fa_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch ---
  AT_DISPATCH_FLOATING_TYPES(
      grad_output_delay.scalar_type(), "delayBackwardLauncher", [&] {
        const scalar_t *grad_output_delay_ptr =
            grad_output_delay
                .data_ptr<scalar_t>(); // May not be needed if cloned
        const scalar_t *resistance_ptr = resistance_tensor.data_ptr<scalar_t>();
        const scalar_t *load_ptr = load_tensor.data_ptr<scalar_t>();

        scalar_t *accum_grad_delay_ptr =
            accum_grad_delay.data_ptr<scalar_t>(); // In/Out
        scalar_t *grad_input_res_ptr =
            grad_input_res.data_ptr<scalar_t>(); // Out
        scalar_t *grad_input_load_ptr =
            grad_input_load.data_ptr<scalar_t>(); // Out

        delayBackwardLauncher<scalar_t>(
            grad_output_delay_ptr, resistance_ptr, load_ptr, pin_fa_ptr,
            topo_sort_ptr, topo_sort_start_ptr, num_nodes, num_nets,
            accum_grad_delay_ptr, // In/Out buffer
            grad_input_res_ptr,   // Out
            grad_input_load_ptr   // Out
        );
      });

  // Return gradients corresponding to inputs (resistance_tensor, load_tensor)
  // Order matters for autograd.
  // Inputs to forward were: resistance_tensor, load_tensor, pin_fa_tensor, ...
  // Return grads for:      resistance_tensor, load_tensor
  // Gradients for index tensors (pin_fa, topo) are None.
  // We return only the computed gradients. PyTorch C++ extension bindings
  // handle matching them to the inputs that require gradients.
  return {grad_input_res, grad_input_load};
}

// --- Launcher Implementation ---
template <typename scalar_t>
void delayBackwardLauncher(
    const scalar_t *grad_output_delay_ptr, // dF/dDelay (initial)
    const scalar_t *resistance_ptr,        // Res(fa(u)->u) at index u
    const scalar_t *load_ptr,              // Load(u) at index u
    const int32_t *pin_fa_ptr, // Parent index for node u (<0 for root)
    const int32_t
        *topo_sort_ptr, // Parent-first order, use backward for bottom-up
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *accum_grad_delay_ptr, // In: cloned dF/dDelay, Out: accumulated
                                    // dF/dDelay
    scalar_t *grad_input_res_ptr,   // Out: dF/dRes
    scalar_t *grad_input_load_ptr   // Out: dF/dLoad (contribution from this op)
) {
  // --- Step 1: Accumulate dF/dDelay bottom-up ---
  // accum_grad_delay_ptr is pre-initialized with grad_output_delay values.
#pragma omp parallel for num_threads(8)
  for (int32_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    int32_t start_topo_idx = topo_sort_start_ptr[net_idx];
    int32_t end_topo_idx = topo_sort_start_ptr[net_idx + 1];

    // Iterate nodes in REVERSE topological order (bottom-up)
    // Assumes iterating topo_sort_ptr backward gives child-first order.
    for (int32_t i = end_topo_idx - 1; i >= start_topo_idx; --i) {
      int32_t u = topo_sort_ptr[i]; // Current node index

      if (u < 0 || u >= num_nodes) {
        continue;
      } // Bounds check

      int32_t parent_idx = pin_fa_ptr[u]; // Get parent index

      // If parent_idx is valid, propagate gradient upwards.
      if (parent_idx >= 0 && parent_idx < num_nodes) {
        // AccumGrad(parent) += AccumGrad(u)
        accum_grad_delay_ptr[parent_idx] += accum_grad_delay_ptr[u];
      }
    } // End bottom-up accumulation for net
  } // End loop over nets

  // --- Step 2: Calculate gradients dF/dRes and dF/dLoad ---
  // This step uses the *final* accumulated gradients `accum_grad_delay_ptr`.
  // Iterate over all nodes (this part is embarrassingly parallel).
#pragma omp parallel for num_threads(8) // Adjust as needed
  for (int32_t u = 0; u < num_nodes; ++u) {
    int32_t parent_idx = pin_fa_ptr[u]; // Get parent index

    // Gradients are computed only for non-root nodes based on the forward
    // formula.
    if (parent_idx >= 0 && parent_idx < num_nodes) {
      // Get the final accumulated gradient for node u
      scalar_t accumulated_grad_u = accum_grad_delay_ptr[u];

      // dF/dRes(fa->u) = AccumGrad(u) * Load(u)
      // Store gradient at index u, corresponding to resistance_ptr[u].
      grad_input_res_ptr[u] = accumulated_grad_u * load_ptr[u];

      // dF/dLoad(u) = AccumGrad(u) * Res(fa->u)
      // This is the gradient contribution for Load(u) from this Delay op.
      grad_input_load_ptr[u] = accumulated_grad_u * resistance_ptr[u];
    }
    // else: node u is a root.
    // grad_input_res[u] and grad_input_load[u] remain 0 (initialized state).
  }
}
