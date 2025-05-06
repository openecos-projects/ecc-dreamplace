#ifndef RC_TIMING_OPS_H
#define RC_TIMING_OPS_H

#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <pybind11/detail/common.h>
#include <queue>
#include <torch/torch.h>


// Declare all functions that will be implemented in separate .cpp files
// and bound in bindings.cpp

//=================================
// Load Operator Declarations
//=================================

/**
 * @brief Computes Elmore load (forward pass, approach).
 * Inputs match rc_timing_cpp.load_forward_cpp signature.
 */
at::Tensor load_forward_cpp(
    at::Tensor cap_tensor,
    at::Tensor pin_start_tensor,
    at::Tensor pin_to_tensor,
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

/**
 * @brief Computes gradient w.r.t. capacitance for Elmore load (backward pass).
 * Inputs match rc_timing_cpp.load_backward_cpp signature.
 * Returns gradient w.r.t cap_tensor.
 */
at::Tensor load_backward_cpp(
    at::Tensor grad_output, // grad_load
    at::Tensor pin_start_tensor,
    at::Tensor pin_to_tensor,
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

//=================================
// Delay Operator Declarations
//=================================

/**
 * @brief Computes Elmore delay (forward pass).
 * Inputs match rc_timing_cpp.delay_forward_cpp signature.
 */
at::Tensor delay_forward_cpp(
    at::Tensor resistance_tensor,
    at::Tensor load_tensor,
    at::Tensor pin_fa_tensor,
    at::Tensor net_driver_pin_tensor, // [[maybe_unused]]
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

/**
 * @brief Computes gradients for Elmore delay w.r.t. resistance and load (backward pass).
 * Inputs match rc_timing_cpp.delay_backward_cpp signature.
 * Returns a vector containing [grad_input_res, grad_input_load].
 */
std::vector<at::Tensor> delay_backward_cpp(
    at::Tensor grad_output_delay,
    at::Tensor resistance_tensor, // From forward pass context
    at::Tensor load_tensor,       // From forward pass context
    at::Tensor pin_fa_tensor,     // From forward pass context
    at::Tensor net_driver_pin_tensor, // [[maybe_unused]]
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

//=================================
// LDelay Operator Declarations
//=================================

/**
 * @brief Computes LDelay = C*D + sum(LDelay_children) (forward pass).
 * Inputs match rc_timing_cpp.ldelay_forward_cpp signature.
 */
at::Tensor ldelay_forward_cpp(
    at::Tensor cap_tensor,
    at::Tensor delay_tensor,
    at::Tensor pin_start_tensor,
    at::Tensor pin_to_tensor,
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

/**
 * @brief Computes gradients for LDelay operator w.r.t. Cap and Delay (backward pass).
 * Inputs match rc_timing_cpp.ldelay_backward_cpp signature.
 * Returns a vector containing [grad_input_cap, grad_input_delay].
 */
std::vector<at::Tensor> ldelay_backward_cpp(
    at::Tensor grad_output_ldelay,
    at::Tensor cap_tensor,       // From forward pass context
    at::Tensor delay_tensor,     // From forward pass context
    at::Tensor pin_start_tensor, // For structure
    at::Tensor pin_to_tensor,    // For structure
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

//=================================
// Beta Operator Declarations
//=================================

/**
 * @brief Computes Beta = Beta(fa) + Res(fa->u) * LDelay(u) (forward pass).
 * Inputs match rc_timing_cpp.beta_forward_cpp signature.
 */
at::Tensor beta_forward_cpp(
    at::Tensor resistance_tensor,
    at::Tensor ldelay_tensor,
    at::Tensor pin_fa_tensor,
    at::Tensor net_driver_pin_tensor, // [[maybe_unused]]
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);

/**
 * @brief Computes gradients for Beta operator w.r.t. resistance and LDelay (backward pass).
 * Inputs match rc_timing_cpp.beta_backward_cpp signature.
 * Returns a vector containing [grad_input_res, grad_input_ldelay].
 */
std::vector<at::Tensor> beta_backward_cpp(
    at::Tensor grad_output_beta,
    at::Tensor resistance_tensor, // From forward pass context
    at::Tensor ldelay_tensor,     // From forward pass context
    at::Tensor pin_fa_tensor,     // From forward pass context
    at::Tensor net_driver_pin_tensor, // [[maybe_unused]]
    at::Tensor net_flat_topo,
    at::Tensor net_flat_topo_start);


#endif // RC_TIMING_OPS_H