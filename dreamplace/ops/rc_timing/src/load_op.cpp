#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <pybind11/detail/common.h>
#include <queue>
#include <torch/torch.h>
#include "rc_timing.hh"

template <typename scalar_t>
void loadForwardLauncher( // Renamed
    const int32_t *pin_start_ptr, const int32_t *pin_to_ptr,
    const int32_t *topo_sort_ptr, const int32_t *topo_sort_start_ptr,
    int32_t num_nodes, int32_t num_nets,
    scalar_t *load_ptr // Input/Output pointer (initialized with cap)
);

// --- Main Function using clone for initialization ---
at::Tensor load_forward_cpp(at::Tensor cap_tensor, at::Tensor pin_start_tensor,
                            at::Tensor pin_to_tensor, at::Tensor net_flat_topo,
                            at::Tensor net_flat_topo_start) {
  // --- Input Checks (Same as before) ---
  CHECK_CPU(cap_tensor);
  CHECK_FLAT(cap_tensor);
  CHECK_CONTIGUOUS(cap_tensor);
  CHECK_CPU(pin_start_tensor);
  CHECK_FLAT(pin_start_tensor);
  CHECK_CONTIGUOUS(pin_start_tensor); // etc.
  TORCH_CHECK(pin_start_tensor.scalar_type() == at::kInt,
              "pin_start_tensor must be int64 (Long)"); // etc.

  int32_t num_nodes = cap_tensor.numel();
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(pin_start_tensor.numel() == num_nodes + 1,
              "pin_start_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");

  // --- Output Tensor - Initialize by cloning cap_tensor ---
  // This ensures Load(u) starts as Cap(u) for ALL nodes, even disconnected
  // ones.
  at::Tensor load_tensor = cap_tensor.clone();

  // --- Get Pointers for Index Tensors (int32_t) ---
  const int32_t *pin_start_ptr = pin_start_tensor.data_ptr<int32_t>();
  const int32_t *pin_to_ptr = pin_to_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch to Templated Launcher ---
  AT_DISPATCH_FLOATING_TYPES(
      load_tensor.scalar_type(), "loadForwardLaunche", [&] {
        // Get pointer for the output tensor (which is already initialized)
        scalar_t *load_ptr = load_tensor.data_ptr<scalar_t>();

        // Call the launcher implementation (needs only load_ptr for
        // modification) We don't need cap_ptr inside the launcher anymore if
        // initialized via clone.
        loadForwardLauncher<scalar_t>( // Note:  launcher name
            pin_start_ptr, pin_to_ptr, topo_sort_ptr, topo_sort_start_ptr,
            num_nodes, num_nets,
            load_ptr // Output (initialized with cap)
        );
      });

  return load_tensor;
}

template <typename scalar_t>
void loadForwardLauncher( // Renamed
    const int32_t *pin_start_ptr, const int32_t *pin_to_ptr,
    const int32_t *topo_sort_ptr, const int32_t *topo_sort_start_ptr,
    int32_t num_nodes, int32_t num_nets,
    scalar_t *load_ptr // Input/Output pointer (initialized with cap)
) {
  // Iterate through each net
  for (int32_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    int32_t start_topo_idx = topo_sort_start_ptr[net_idx];
    int32_t end_topo_idx = topo_sort_start_ptr[net_idx + 1];

    // Iterate nodes in REVERSE topological order (bottom-up)
    for (int32_t i = end_topo_idx - 1; i >= start_topo_idx; --i) {
      int32_t u = topo_sort_ptr[i];
      if (u < 0 || u >= num_nodes) {
        continue;
      }

      // Load(u) is already initialized with Cap(u).
      // Just add children's loads.
      int32_t edge_start_idx = pin_start_ptr[u];
      int32_t edge_end_idx = pin_start_ptr[u + 1];

      for (int32_t edge_idx = edge_start_idx; edge_idx < edge_end_idx;
           ++edge_idx) {
        int32_t v = pin_to_ptr[edge_idx];
        if (v < 0 || v >= num_nodes) {
          continue;
        }

        // Add the fully computed Load(v) to Load(u)
        load_ptr[u] += load_ptr[v];
      }
    }
  }
}

// Forward declaration
template <typename scalar_t>
void loadBackwardLauncher(
    const int32_t *pin_start_ptr, const int32_t *pin_to_ptr,
    const int32_t *topo_sort_ptr, const int32_t *topo_sort_start_ptr,
    int32_t num_nodes, int32_t num_nets,
    scalar_t *grad_input_cap_ptr // Input is grad_output, Output is grad_cap
                                 // (accumulated)
);

/**
 * @brief Computes gradient w.r.t. capacitance for Elmore load (backward pass).
 * dF/dCap(u) = AccumGrad(u)
 * AccumGrad(u) = dF/dLoad(u) + sum_{p=parents(u)} AccumGrad(p)
 * Calculated via top-down propagation: AccumGrad(v) += AccumGrad(u) for v in
 * children(u)
 *
 * @param grad_output Gradient from the next op (dF/dLoad), (1D, float or
 * double).
 * @param pin_fa_tensor Parent pin tensor (1D, int64). Unused in current impl.
 * @param net_driver_pin_tensor Driver pin tensor (1D, int64). Unused in current
 * impl.
 * @param pin_start_tensor Start indices for edges (1D, int64,
 * size=num_nodes+1).
 * @param pin_to_tensor Destination pins for edges (1D, int64).
 * @param net_flat_topo Flattened topological sort (post-order assumed, used
 * forward for top-down) (1D, int64).
 * @param net_flat_topo_start Start indices for nets in topo sort (1D, int64,
 * size=num_nets+1).
 * @return torch::Tensor Gradient w.r.t input capacitance (dF/dCap), (1D, same
 * dtype as grad_output).
 */
at::Tensor load_backward_cpp(at::Tensor grad_output,
                             at::Tensor pin_start_tensor,
                             at::Tensor pin_to_tensor, at::Tensor net_flat_topo,
                             at::Tensor net_flat_topo_start) {
  // --- Input Checks ---
  CHECK_CPU(grad_output);
  CHECK_FLAT(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  // Add checks for other tensors (CPU, FLAT, CONTIGUOUS, DTYPE)
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

  TORCH_CHECK(pin_start_tensor.scalar_type() == at::kInt,
              "pin_start_tensor must be int64 (Long)");
  TORCH_CHECK(pin_to_tensor.scalar_type() == at::kInt,
              "pin_to_tensor must be int64 (Long)");
  TORCH_CHECK(net_flat_topo.scalar_type() == at::kInt,
              "net_flat_topo must be int64 (Long)");
  TORCH_CHECK(net_flat_topo_start.scalar_type() == at::kInt,
              "net_flat_topo_start must be int64 (Long)");

  int32_t num_nodes = grad_output.numel();
  int32_t num_nets =
      net_flat_topo_start.numel() ? (net_flat_topo_start.numel() - 1) : 0;

  TORCH_CHECK(pin_start_tensor.numel() == num_nodes + 1,
              "pin_start_tensor size mismatch");
  TORCH_CHECK(net_flat_topo_start.numel() == num_nets + 1 || num_nets == 0,
              "net_flat_topo_start size mismatch");
  // Check grad_output size
  TORCH_CHECK(grad_output.numel() == num_nodes,
              "grad_output size must match num_nodes");

  // --- Output/Accumulation Tensor ---
  // Initialize the output gradient (dF/dCap) by cloning the input gradient
  // (dF/dLoad). This tensor will be used to accumulate gradients top-down.
  // AccumGrad(u) starts as dF/dLoad(u).
  at::Tensor grad_input_cap = grad_output.clone();

  // --- Get Pointers for Index Tensors (int32_t) ---
  const int32_t *pin_start_ptr = pin_start_tensor.data_ptr<int32_t>();
  const int32_t *pin_to_ptr = pin_to_tensor.data_ptr<int32_t>();
  const int32_t *topo_sort_ptr = net_flat_topo.data_ptr<int32_t>();
  const int32_t *topo_sort_start_ptr = net_flat_topo_start.data_ptr<int32_t>();

  // --- Dispatch to Templated Launcher ---
  AT_DISPATCH_FLOATING_TYPES(
      grad_input_cap.scalar_type(), "loadBackwardLauncher", [&] {
        // Get pointer for the float/double tensor (used for accumulation)
        scalar_t *grad_input_cap_ptr = grad_input_cap.data_ptr<scalar_t>();

        // Call the launcher implementation
        loadBackwardLauncher<scalar_t>(
            pin_start_ptr, pin_to_ptr, topo_sort_ptr, topo_sort_start_ptr,
            num_nodes, num_nets,
            grad_input_cap_ptr // Input/Output accumulator
        );
      });

  return grad_input_cap;
}

// --- Launcher Implementation ---
template <typename scalar_t>
void loadBackwardLauncher(
    const int32_t *pin_start_ptr, const int32_t *pin_to_ptr,
    const int32_t *topo_sort_ptr, const int32_t *topo_sort_start_ptr,
    int32_t num_nodes, int32_t num_nets,
    scalar_t *grad_input_cap_ptr // Input: Initialized with dF/dLoad. Output:
                                 // Accumulated dF/dCap.
) {
  // Iterate through each net defined by the topological sort segments
  for (int32_t net_idx = 0; net_idx < num_nets; ++net_idx) {
    int32_t start_topo_idx = topo_sort_start_ptr[net_idx];
    int32_t end_topo_idx = topo_sort_start_ptr[net_idx + 1];

    // Iterate through nodes of the CURRENT net in FORWARD topological order
    // (top-down) Assumes the order in net_flat_topo[start:end] processes
    // parents before children.
    for (int32_t i = start_topo_idx; i < end_topo_idx; ++i) {
      int32_t u = topo_sort_ptr[i]; // Get the parent node index u

      // Basic bounds check
      if (u < 0 || u >= num_nodes) {
        continue;
      }

      // Get the accumulated gradient at parent u
      scalar_t parent_grad = grad_input_cap_ptr[u];

      // If parent's accumulated gradient is zero, it contributes nothing down
      // This check might provide minor speedup if gradients are sparse.
      // if (parent_grad == static_cast<scalar_t>(0.0)) {
      //     continue;
      // }

      // Find children v of u and propagate gradient
      int32_t edge_start_idx = pin_start_ptr[u];
      int32_t edge_end_idx = pin_start_ptr[u + 1];

      for (int32_t edge_idx = edge_start_idx; edge_idx < edge_end_idx;
           ++edge_idx) {
        int32_t v = pin_to_ptr[edge_idx]; // Get child node index v

        // Basic bounds check
        if (v < 0 || v >= num_nodes) {
          continue;
        }

        // Propagate gradient: AccumGrad(v) += AccumGrad(u)
        // grad_input_cap_ptr[v] represents AccumGrad(v)
        grad_input_cap_ptr[v] += parent_grad;
      }
    } // End loop over nodes in the current net (top-down)
  } // End loop over all nets

  // After the loops complete, grad_input_cap_ptr contains the final dF/dCap
  // values.
}

