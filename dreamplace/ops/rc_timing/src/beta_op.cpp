#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <pybind11/detail/common.h>
#include <queue>
#include <torch/torch.h>
#include "rc_timing.hh"

// Forward declaration of the launcher
template <typename scalar_t>
void betaForwardLauncher(
    const scalar_t *resistance_ptr,
    const scalar_t *ldelay_ptr,   // Input: LDelay tensor
    const int32_t *pin_fa_ptr,    // Parent index for each node (-1 for roots)
    const int32_t *topo_sort_ptr, // Assumed parent-first order
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *beta_ptr // Output: Beta tensor
);

/**
 * @brief Computes Beta = Beta(fa) + Res(fa->u) * LDelay(u) (forward pass).
 * Calculation is top-down, identical structure to Delay forward pass.
 *
 * @param resistance_tensor Resistance tensor (1D, float/double).
 * resistance_tensor[u] is r_{fa(u)->u}.
 * @param ldelay_tensor LDelay tensor from LDelay operator (1D, float/double).
 * @param pin_fa_tensor Parent pin index tensor (1D, int64). Root nodes should
 * have parent index < 0.
 * @param net_driver_pin_tensor Driver pin tensor (1D, int64). Unused if pin_fa
 * identifies roots.
 * @param net_flat_topo Flattened topological sort (assumed parent-first,
 * traverse FORWARD for top-down) (1D, int64).
 * @param net_flat_topo_start Start indices for nets in topo sort (1D, int64,
 * size=num_nets+1).
 * @return at::Tensor The calculated Beta tensor (1D, same dtype as inputs).
 */
at::Tensor
beta_forward_cpp(at::Tensor resistance_tensor,
                 at::Tensor ldelay_tensor, // Input changed from load_tensor
                 at::Tensor pin_fa_tensor,
                 at::Tensor net_flat_topo, at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(resistance_tensor);
  CHECK_FLAT(resistance_tensor);
  CHECK_CONTIGUOUS(resistance_tensor);
  CHECK_CPU(ldelay_tensor);
  CHECK_FLAT(ldelay_tensor);
  CHECK_CONTIGUOUS(ldelay_tensor); // Changed
  CHECK_CPU(pin_fa_tensor);
  CHECK_FLAT(pin_fa_tensor);
  CHECK_CONTIGUOUS(pin_fa_tensor);
  CHECK_CPU(net_flat_topo);
  CHECK_FLAT(net_flat_topo);
  CHECK_CONTIGUOUS(net_flat_topo);
  CHECK_CPU(net_flat_topo_start);
  CHECK_FLAT(net_flat_topo_start);
  CHECK_CONTIGUOUS(net_flat_topo_start);

  TORCH_CHECK(resistance_tensor.scalar_type() == ldelay_tensor.scalar_type(),
              "Input float dtypes mismatch"); // Changed
  TORCH_CHECK(pin_fa_tensor.scalar_type() == at::kInt,
              "pin_fa_tensor must be int64 (Long)");
  TORCH_CHECK(net_flat_topo.scalar_type() == at::kInt,
              "net_flat_topo must be int64 (Long)");
  TORCH_CHECK(net_flat_topo_start.scalar_type() == at::kInt,
              "net_flat_topo_start must be int64 (Long)");

  int32_t num_nodes = ldelay_tensor.numel(); // Base size check on ldelay_tensor
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(resistance_tensor.numel() == num_nodes,
              "resistance_tensor size mismatch");
  TORCH_CHECK(pin_fa_tensor.numel() == num_nodes,
              "pin_fa_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensor ---
  // Initialize Beta with zeros. Roots have Beta = 0.
  at::Tensor beta_tensor = at::zeros(
      {num_nodes}, resistance_tensor.options()); // Output tensor named beta

  // --- Get Pointers ---
  const int32_t *pin_fa_ptr = pin_fa_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch ---
  AT_DISPATCH_FLOATING_TYPES(
      resistance_tensor.scalar_type(), "betaForwardLauncher",
      [&] { // Dispatch based on float type
        const scalar_t *resistance_ptr = resistance_tensor.data_ptr<scalar_t>();
        const scalar_t *ldelay_ptr =
            ldelay_tensor.data_ptr<scalar_t>(); // Use LDelay pointer
        scalar_t *beta_ptr =
            beta_tensor.data_ptr<scalar_t>(); // Output Beta pointer

        betaForwardLauncher<scalar_t>( // Call Beta launcher
            resistance_ptr,
            ldelay_ptr, // Pass LDelay pointer
            pin_fa_ptr, topo_sort_ptr, topo_sort_start_ptr, num_nodes, num_nets,
            beta_ptr); // Pass Beta output pointer
      });

  return beta_tensor; // Return Beta tensor
}

// --- Launcher Implementation ---
template <typename scalar_t>
void betaForwardLauncher( // Renamed launcher
    const scalar_t *resistance_ptr,
    const scalar_t *ldelay_ptr, // Renamed input
    const int32_t *pin_fa_ptr, const int32_t *topo_sort_ptr,
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *beta_ptr // Renamed output
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
        // Beta(u) = Beta(parent) + Res(parent->u) * LDelay(u)
        beta_ptr[u] = beta_ptr[parent_idx] +
                      resistance_ptr[u] * ldelay_ptr[u]; // Use LDelay
      } else {
        // Node u is a root
        beta_ptr[u] = 0.0; // Initialize root Beta to 0
      }
    } // End loop over nodes in net
  } // End loop over nets
}

// Forward declaration
template <typename scalar_t>
void betaBackwardLauncher(                // Renamed launcher
    const scalar_t *grad_output_beta_ptr, // Input: dF/dBeta
    const scalar_t *resistance_ptr,       // Res(fa(u)->u) at index u
    const scalar_t *ldelay_ptr,           // LDelay(u) at index u
    const int32_t *pin_fa_ptr,            // Parent index for node u
    const int32_t
        *topo_sort_ptr, // Parent-first order, use backward for bottom-up
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *accum_grad_beta_ptr,  // Intermediate buffer (In/Out)
    scalar_t *grad_input_res_ptr,   // Output: dF/dRes
    scalar_t *grad_input_ldelay_ptr // Output: dF/dLDelay
);

/**
 * @brief Computes gradients for Beta operator w.r.t. resistance and LDelay.
 * Structure is identical to Delay backward pass.
 * Returns gradients for inputs: resistance_tensor, ldelay_tensor.
 *
 * @param grad_output_beta Gradient w.r.t. the output Beta (dF/dBeta).
 * @param resistance_tensor Resistance tensor used in forward.
 * @param ldelay_tensor LDelay tensor used in forward.
 * @param pin_fa_tensor Parent pin index tensor used in forward.
 * @param net_driver_pin_tensor Unused.
 * @param net_flat_topo Flattened topological sort (parent-first, used BACKWARD
 * for bottom-up accumulation).
 * @param net_flat_topo_start Start indices for nets in topo sort.
 * @return std::vector<at::Tensor> Gradients [grad_input_res,
 * grad_input_ldelay].
 */
std::vector<at::Tensor> beta_backward_cpp( // Renamed function
    at::Tensor grad_output_beta,           // Renamed input gradient
    at::Tensor resistance_tensor,          // From forward pass context
    at::Tensor ldelay_tensor,              // Input LDelay from forward context
    at::Tensor pin_fa_tensor,              // From forward pass context
    at::Tensor net_flat_topo, at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(grad_output_beta);
  CHECK_FLAT(grad_output_beta);
  CHECK_CONTIGUOUS(grad_output_beta); // Renamed
  CHECK_CPU(resistance_tensor);
  CHECK_FLAT(resistance_tensor);
  CHECK_CONTIGUOUS(resistance_tensor);
  CHECK_CPU(ldelay_tensor);
  CHECK_FLAT(ldelay_tensor);
  CHECK_CONTIGUOUS(ldelay_tensor); // Changed
  CHECK_CPU(pin_fa_tensor);
  CHECK_FLAT(pin_fa_tensor);
  CHECK_CONTIGUOUS(pin_fa_tensor);
  CHECK_CPU(net_flat_topo);
  CHECK_FLAT(net_flat_topo);
  CHECK_CONTIGUOUS(net_flat_topo);
  CHECK_CPU(net_flat_topo_start);
  CHECK_FLAT(net_flat_topo_start);
  CHECK_CONTIGUOUS(net_flat_topo_start);

  TORCH_CHECK(grad_output_beta.scalar_type() ==
                      resistance_tensor.scalar_type() && // Renamed
                  grad_output_beta.scalar_type() ==
                      ldelay_tensor.scalar_type(), // Renamed & Changed
              "Input floating point dtypes mismatch");
  TORCH_CHECK(pin_fa_tensor.scalar_type() == at::kInt,
              "pin_fa_tensor must be int64 (Long)");
  // ... other dtype checks

  int32_t num_nodes = ldelay_tensor.numel(); // Base size check on ldelay_tensor
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(grad_output_beta.numel() == num_nodes,
              "grad_output_beta size mismatch"); // Renamed
  TORCH_CHECK(resistance_tensor.numel() == num_nodes,
              "resistance_tensor size mismatch");
  TORCH_CHECK(pin_fa_tensor.numel() == num_nodes,
              "pin_fa_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensors (Initialize to Zero) ---
  at::Tensor grad_input_res =
      at::zeros_like(resistance_tensor, resistance_tensor.options());
  // Gradient w.r.t. LDelay *input* of this operator.
  at::Tensor grad_input_ldelay =
      at::zeros_like(ldelay_tensor, ldelay_tensor.options()); // Renamed output

  // --- Intermediate Tensor ---
  // For accumulating dF/dBeta bottom-up
  at::Tensor accum_grad_beta = grad_output_beta.clone(); // Renamed accumulator

  // --- Get Pointers ---
  const int32_t *pin_fa_ptr = pin_fa_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch ---
  AT_DISPATCH_FLOATING_TYPES(
      grad_output_beta.scalar_type(), "betaBackwardLauncher",
      [&] { // Renamed dispatch
        const scalar_t *grad_output_beta_ptr =
            grad_output_beta.data_ptr<scalar_t>(); // Renamed
        const scalar_t *resistance_ptr = resistance_tensor.data_ptr<scalar_t>();
        const scalar_t *ldelay_ptr =
            ldelay_tensor.data_ptr<scalar_t>(); // Changed

        scalar_t *accum_grad_beta_ptr =
            accum_grad_beta.data_ptr<scalar_t>(); // Renamed, In/Out
        scalar_t *grad_input_res_ptr =
            grad_input_res.data_ptr<scalar_t>(); // Out
        scalar_t *grad_input_ldelay_ptr =
            grad_input_ldelay.data_ptr<scalar_t>(); // Renamed, Out

        betaBackwardLauncher<scalar_t>( // Renamed launcher
            grad_output_beta_ptr,       // Renamed
            resistance_ptr,
            ldelay_ptr, // Changed
            pin_fa_ptr, topo_sort_ptr, topo_sort_start_ptr, num_nodes, num_nets,
            accum_grad_beta_ptr,  // Renamed, In/Out buffer
            grad_input_res_ptr,   // Out
            grad_input_ldelay_ptr // Renamed, Out
        );
      });

  // Return gradients corresponding to inputs (resistance_tensor, ldelay_tensor)
  return {grad_input_res, grad_input_ldelay}; // Return grad w.r.t LDelay
}

// --- Launcher Implementation ---
template <typename scalar_t>
void betaBackwardLauncher(                // Renamed launcher
    const scalar_t *grad_output_beta_ptr, // Renamed: dF/dBeta (initial)
    const scalar_t *resistance_ptr,       // Res(fa(u)->u) at index u
    const scalar_t *ldelay_ptr,           // LDelay(u) at index u
    const int32_t *pin_fa_ptr,            // Parent index for node u
    const int32_t
        *topo_sort_ptr, // Parent-first order, use backward for bottom-up
    const int32_t *topo_sort_start_ptr, int32_t num_nodes, int32_t num_nets,
    scalar_t *
        accum_grad_beta_ptr, // Renamed: In/Out buffer for dF/dBeta accumulation
    scalar_t *grad_input_res_ptr,   // Output: dF/dRes
    scalar_t *grad_input_ldelay_ptr // Renamed Output: dF/dLDelay
) {
  // --- Step 1: Accumulate dF/dBeta bottom-up ---
  // accum_grad_beta_ptr is pre-initialized with grad_output_beta values.
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
        accum_grad_beta_ptr[parent_idx] +=
            accum_grad_beta_ptr[u]; // Accumulate into parent
      }
    } // End bottom-up accumulation for net
  } // End loop over nets

  // --- Step 2: Calculate gradients dF/dRes and dF/dLDelay ---
  // This step uses the *final* accumulated gradients `accum_grad_beta_ptr`.
  // Iterate over all nodes.
  for (int32_t u = 0; u < num_nodes; ++u) {
    int32_t parent_idx = pin_fa_ptr[u]; // Get parent index

    // Gradients are computed only for non-root nodes.
    if (parent_idx >= 0 && parent_idx < num_nodes) {
      // Get the final accumulated gradient for node u
      scalar_t accumulated_grad_u =
          accum_grad_beta_ptr[u]; // Use accumulated beta gradient

      // dF/dRes(fa->u) = AccumGradBeta(u) * LDelay(u)
      grad_input_res_ptr[u] =
          accumulated_grad_u * ldelay_ptr[u]; // Use LDelay value

      // dF/dLDelay(u) = AccumGradBeta(u) * Res(fa->u)
      grad_input_ldelay_ptr[u] =
          accumulated_grad_u * resistance_ptr[u]; // Calculate grad w.r.t LDelay
    }
    // else: node u is a root. Gradients remain 0.
  }
}
