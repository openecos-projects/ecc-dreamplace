#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <pybind11/detail/common.h>
#include <queue>
#include <torch/torch.h>
#include "rc_timing.hh"

// Forward declaration
template <typename scalar_t>
void ldelayForwardLauncher(
    const scalar_t *cap_ptr, const scalar_t *delay_ptr,
    const int32_t *pin_start_ptr, const int32_t *pin_to_ptr,
    const int32_t *
        topo_sort_ptr, // Assumed parent-first order, use backward for bottom-up
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *ldelay_ptr // Output
);

/**
 * @brief Computes LDelay = C*D + sum(LDelay_children) (forward pass).
 * Calculation is bottom-up, similar to Load operator.
 *
 * @param cap_tensor Capacitance tensor (1D, float/double).
 * @param delay_tensor Delay tensor from Delay operator (1D, float/double).
 * @param pin_start_tensor Start indices for edges (1D, int64,
 * size=num_nodes+1).
 * @param pin_to_tensor Destination pins for edges (1D, int64).
 * @param net_flat_topo Flattened topological sort (parent-first order assumed,
 * traverse BACKWARD for bottom-up) (1D, int64).
 * @param net_flat_topo_start Start indices for nets in topo sort (1D, int64,
 * size=num_nets+1).
 * @return at::Tensor The calculated LDelay tensor (1D, same dtype as inputs).
 */
at::Tensor ldelay_forward_cpp(at::Tensor cap_tensor, at::Tensor delay_tensor,
                              at::Tensor pin_start_tensor,
                              at::Tensor pin_to_tensor,
                              at::Tensor net_flat_topo,
                              at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(cap_tensor);
  CHECK_FLAT(cap_tensor);
  CHECK_CONTIGUOUS(cap_tensor);
  CHECK_CPU(delay_tensor);
  CHECK_FLAT(delay_tensor);
  CHECK_CONTIGUOUS(delay_tensor);
  CHECK_CPU(pin_start_tensor);
  CHECK_FLAT(pin_start_tensor);
  CHECK_CONTIGUOUS(pin_start_tensor);
  CHECK_CPU(pin_to_tensor);
  CHECK_FLAT(pin_to_tensor);
  CHECK_CONTIGUOUS(pin_to_tensor);
  CHECK_CPU(net_flat_topo);
  CHECK_FLAT(net_flat_topo);
  CHECK_CONTIGUOUS(net_flat_topo);
  CHECK_CPU(net_flat_topo_start);
  CHECK_FLAT(net_flat_topo_start);
  CHECK_CONTIGUOUS(net_flat_topo_start);

  TORCH_CHECK(cap_tensor.scalar_type() == delay_tensor.scalar_type(),
              "Input float dtypes mismatch");
  TORCH_CHECK(pin_start_tensor.scalar_type() == at::kInt,
              "pin_start_tensor must be int64 (Long)");
  TORCH_CHECK(pin_to_tensor.scalar_type() == at::kInt,
              "pin_to_tensor must be int64 (Long)");
  TORCH_CHECK(net_flat_topo.scalar_type() == at::kInt,
              "net_flat_topo must be int64 (Long)");
  TORCH_CHECK(net_flat_topo_start.scalar_type() == at::kInt,
              "net_flat_topo_start must be int64 (Long)");

  int32_t num_nodes = cap_tensor.numel();
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(delay_tensor.numel() == num_nodes, "delay_tensor size mismatch");
  TORCH_CHECK(pin_start_tensor.numel() == num_nodes + 1,
              "pin_start_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensor ---
  // Initialize with zeros, calculation is accumulative bottom-up.
  at::Tensor ldelay_tensor = at::zeros({num_nodes}, cap_tensor.options());

  // --- Get Pointers ---
  const int32_t *pin_start_ptr = pin_start_tensor.data_ptr<int32_t>();
  const int32_t *pin_to_ptr = pin_to_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch ---
  AT_DISPATCH_FLOATING_TYPES(
      cap_tensor.scalar_type(), "ldelayForwardLauncher", [&] {
        const scalar_t *cap_ptr = cap_tensor.data_ptr<scalar_t>();
        const scalar_t *delay_ptr = delay_tensor.data_ptr<scalar_t>();
        scalar_t *ldelay_ptr = ldelay_tensor.data_ptr<scalar_t>(); // Output

        ldelayForwardLauncher<scalar_t>(
            cap_ptr, delay_ptr, pin_start_ptr, pin_to_ptr, topo_sort_ptr,
            topo_sort_start_ptr, num_nodes, num_nets, ldelay_ptr);
      });

  return ldelay_tensor;
}

// --- Launcher Implementation ---
template <typename scalar_t>
void ldelayForwardLauncher(
    const scalar_t *cap_ptr, const scalar_t *delay_ptr,
    const int32_t *pin_start_ptr, const int32_t *pin_to_ptr,
    const int32_t
        *topo_sort_ptr, // Parent-first order, use backward for bottom-up
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *ldelay_ptr // Output
) {
  // Iterate through each net
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

      // 1. Calculate local term: Cap(u) * Delay(u)
      scalar_t local_term = cap_ptr[u] * delay_ptr[u];

      // 2. Sum LDelay(v) for children v
      scalar_t sum_children_ldelay = 0.0;
      int32_t edge_start_idx = pin_start_ptr[u];
      int32_t edge_end_idx = pin_start_ptr[u + 1];

      for (int32_t edge_idx = edge_start_idx; edge_idx < edge_end_idx;
           ++edge_idx) {
        int32_t v = pin_to_ptr[edge_idx]; // Get child node index
        if (v >= 0 && v < num_nodes) {    // Check bounds for child
          sum_children_ldelay += ldelay_ptr[v];
        }
      }

      // 3. Combine: LDelay(u) = local_term + sum_children_ldelay
      ldelay_ptr[u] = local_term + sum_children_ldelay;

    } // End loop over nodes in net (bottom-up)
  } // End loop over nets
}

// Forward declaration
template <typename scalar_t>
void ldelayBackwardLauncher(
    const scalar_t *grad_output_ldelay_ptr, // dF/dLDelay (initial)
    const scalar_t *cap_ptr,                // Cap(u) needed for dF/dDelay
    const scalar_t *delay_ptr,              // Delay(u) needed for dF/dCap
    const int32_t
        *pin_start_ptr,        // Children structure for top-down accumulation
    const int32_t *pin_to_ptr, // Children structure
    const int32_t
        *topo_sort_ptr, // Parent-first order, use forward for top-down
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *accum_grad_ldelay_ptr, // Intermediate buffer (In/Out)
    scalar_t *grad_input_cap_ptr,    // Output: dF/dCap
    scalar_t
        *grad_input_delay_ptr // Output: dF/dDelay (contribution from this op)
);

/**
 * @brief Computes gradients for LDelay operator w.r.t. Cap and Delay.
 * Backward pass mirrors the Load operator's backward pass structure.
 * Returns gradients for inputs: cap_tensor, delay_tensor.
 *
 * @param grad_output_ldelay Gradient w.r.t. the output LDelay (dF/dLDelay).
 * @param cap_tensor Capacitance tensor used in forward.
 * @param delay_tensor Delay tensor used in forward.
 * @param pin_start_tensor Start indices for edges used in forward.
 * @param pin_to_tensor Destination pins for edges used in forward.
 * @param net_flat_topo Flattened topological sort (parent-first, used FORWARD
 * for top-down accumulation).
 * @param net_flat_topo_start Start indices for nets in topo sort.
 * @return std::vector<at::Tensor> Gradients [grad_input_cap, grad_input_delay].
 */
std::vector<at::Tensor>
ldelay_backward_cpp(at::Tensor grad_output_ldelay,
                    at::Tensor cap_tensor,       // From forward pass context
                    at::Tensor delay_tensor,     // From forward pass context
                    at::Tensor pin_start_tensor, // For structure
                    at::Tensor pin_to_tensor,    // For structure
                    at::Tensor net_flat_topo, at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(grad_output_ldelay);
  CHECK_FLAT(grad_output_ldelay);
  CHECK_CONTIGUOUS(grad_output_ldelay);
  CHECK_CPU(cap_tensor);
  CHECK_FLAT(cap_tensor);
  CHECK_CONTIGUOUS(cap_tensor);
  CHECK_CPU(delay_tensor);
  CHECK_FLAT(delay_tensor);
  CHECK_CONTIGUOUS(delay_tensor);
  CHECK_CPU(pin_start_tensor);
  CHECK_FLAT(pin_start_tensor);
  CHECK_CONTIGUOUS(pin_start_tensor);
  // ... check other tensors ...

  TORCH_CHECK(grad_output_ldelay.scalar_type() == cap_tensor.scalar_type() &&
                  grad_output_ldelay.scalar_type() ==
                      delay_tensor.scalar_type(),
              "Input floating point dtypes mismatch");
  TORCH_CHECK(pin_start_tensor.scalar_type() == at::kInt,
              "pin_start_tensor must be int64 (Long)");
  // ... other dtype checks ...

  int32_t num_nodes = cap_tensor.numel();
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(grad_output_ldelay.numel() == num_nodes,
              "grad_output_ldelay size mismatch");
  TORCH_CHECK(delay_tensor.numel() == num_nodes, "delay_tensor size mismatch");
  TORCH_CHECK(pin_start_tensor.numel() == num_nodes + 1,
              "pin_start_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensors (Initialize to Zero) ---
  at::Tensor grad_input_cap = at::zeros_like(cap_tensor, cap_tensor.options());
  // grad_input_delay is the gradient w.r.t. the Delay *input* of this operator.
  at::Tensor grad_input_delay =
      at::zeros_like(delay_tensor, delay_tensor.options());

  // --- Intermediate Tensor ---
  // For accumulating dF/dLDelay top-down. Initialize with input gradient.
  at::Tensor accum_grad_ldelay = grad_output_ldelay.clone();

  // --- Get Pointers ---
  const int32_t *pin_start_ptr = pin_start_tensor.data_ptr<int32_t>();
  const int32_t *pin_to_ptr = pin_to_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch ---
  AT_DISPATCH_FLOATING_TYPES(
      grad_output_ldelay.scalar_type(), "ldelayBackwardLauncher", [&] {
        const scalar_t *grad_output_ldelay_ptr =
            grad_output_ldelay.data_ptr<scalar_t>(); // Might not be needed
        const scalar_t *cap_ptr = cap_tensor.data_ptr<scalar_t>();
        const scalar_t *delay_ptr = delay_tensor.data_ptr<scalar_t>();

        scalar_t *accum_grad_ldelay_ptr =
            accum_grad_ldelay.data_ptr<scalar_t>(); // In/Out
        scalar_t *grad_input_cap_ptr =
            grad_input_cap.data_ptr<scalar_t>(); // Out
        scalar_t *grad_input_delay_ptr =
            grad_input_delay.data_ptr<scalar_t>(); // Out

        ldelayBackwardLauncher<scalar_t>(
            grad_output_ldelay_ptr, cap_ptr, delay_ptr, pin_start_ptr,
            pin_to_ptr, topo_sort_ptr, topo_sort_start_ptr, num_nodes, num_nets,
            accum_grad_ldelay_ptr, // In/Out buffer
            grad_input_cap_ptr,    // Out
            grad_input_delay_ptr   // Out
        );
      });

  // Return gradients corresponding to inputs (cap_tensor, delay_tensor)
  // Order matters for autograd.
  return {grad_input_cap, grad_input_delay};
}

// --- Launcher Implementation ---
template <typename scalar_t>
void ldelayBackwardLauncher(
    const scalar_t *grad_output_ldelay_ptr, // dF/dLDelay (initial)
    const scalar_t *cap_ptr,                // Cap(u) needed for dF/dDelay
    const scalar_t *delay_ptr,              // Delay(u) needed for dF/dCap
    const int32_t
        *pin_start_ptr,        // Children structure for top-down accumulation
    const int32_t *pin_to_ptr, // Children structure
    const int32_t
        *topo_sort_ptr, // Parent-first order, use forward for top-down
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *accum_grad_ldelay_ptr, // In: cloned dF/dLDelay, Out: accumulated
                                     // dF/dLDelay
    scalar_t *grad_input_cap_ptr,    // Out: dF/dCap
    scalar_t *grad_input_delay_ptr // Out: dF/dDelay (contribution from this op)
) {
  // --- Step 1: Accumulate dF/dLDelay top-down ---
  // accum_grad_ldelay_ptr is pre-initialized with grad_output_ldelay values.
  // This propagation is identical to the Load backward pass's accumulation.
  for (int32_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    int32_t start_topo_idx = topo_sort_start_ptr[net_idx];
    int32_t end_topo_idx = topo_sort_start_ptr[net_idx + 1];

    // Iterate nodes in FORWARD topological order (top-down)
    for (int32_t i = start_topo_idx; i < end_topo_idx; ++i) {
      int32_t u =
          topo_sort_ptr[i]; // Current node index u (parent in this context)

      if (u < 0 || u >= num_nodes) {
        continue;
      } // Bounds check

      // Get the accumulated gradient at parent u
      scalar_t parent_accum_grad = accum_grad_ldelay_ptr[u];

      // Find children v of u and propagate gradient
      int32_t edge_start_idx = pin_start_ptr[u];
      int32_t edge_end_idx = pin_start_ptr[u + 1];

      for (int32_t edge_idx = edge_start_idx; edge_idx < edge_end_idx;
           ++edge_idx) {
        int32_t v = pin_to_ptr[edge_idx]; // Get child node index v
        if (v >= 0 && v < num_nodes) {    // Check bounds
          // Propagate gradient: AccumGrad(v) += AccumGrad(u)
          accum_grad_ldelay_ptr[v] += parent_accum_grad;
        }
      }
    } // End top-down accumulation for net
  } // End loop over nets

  // --- Step 2: Calculate gradients dF/dCap and dF/dDelay ---
  // This step uses the *final* accumulated gradients `accum_grad_ldelay_ptr`.
  // Iterate over all nodes (this part is embarrassingly parallel).
  for (int32_t u = 0; u < num_nodes; ++u) {
    // Get the final accumulated gradient for node u
    scalar_t accumulated_grad_u = accum_grad_ldelay_ptr[u];

    // dF/dCap(u) = AccumGradLd(u) * (dLDelay(u) / dCap(u))
    // dF/dCap(u) = AccumGradLd(u) * Delay(u)
    grad_input_cap_ptr[u] = accumulated_grad_u * delay_ptr[u];

    // dF/dDelay(u) = AccumGradLd(u) * (dLDelay(u) / dDelay(u))
    // dF/dDelay(u) = AccumGradLd(u) * Cap(u)
    // This is the gradient contribution for Delay(u) from this LDelay op.
    grad_input_delay_ptr[u] = accumulated_grad_u * cap_ptr[u];
  }
}

