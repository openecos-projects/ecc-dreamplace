#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <chrono>
#include <initializer_list>
#include <utility>

template <typename scalar_t>
void segmentTransferForwardCudaLauncher(
    const scalar_t* input_arrival,
    const scalar_t* input_slew,
    const scalar_t* downstream_load,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const int64_t* repeater_count,
    const scalar_t* split_fractions,
    const scalar_t* bsu_index,
    const scalar_t* upstream_retained_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    scalar_t* upstream_visible_load,
    scalar_t* segment_delay,
    scalar_t* output_arrival,
    scalar_t* output_slew,
    scalar_t* first_buffer_input_slew,
    scalar_t* first_buffer_output_load,
    scalar_t* first_buffer_delay,
    scalar_t* first_buffer_output_slew,
    int32_t sample_count,
    int32_t nmax,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    cudaStream_t stream);

template <typename scalar_t>
void segmentCountTransferForwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const scalar_t* node_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    const scalar_t* load_input_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* segment_sub_resistance_fraction,
    const int64_t* sink_node_compact_id,
    scalar_t* effective_node_cap,
    scalar_t* node_load,
    scalar_t* node_arrival,
    scalar_t* node_slew,
    scalar_t* segment_delay,
    scalar_t* segment_output_slew,
    scalar_t* segment_upstream_cap,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    int32_t net_count,
    int32_t node_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t sink_count,
    int32_t max_count,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    cudaStream_t stream);

template <typename scalar_t>
void segmentCountCapForwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_capacitance,
    const scalar_t* node_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* z_value,
    const scalar_t* load_input_cap,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    scalar_t* effective_node_cap,
    scalar_t* node_load,
    int32_t net_count,
    int32_t node_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t max_count,
    cudaStream_t stream);

template <typename scalar_t>
void segmentCountTransferBackwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* driver_slew,
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    const scalar_t* load_input_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* segment_sub_resistance_fraction,
    const int64_t* sink_node_compact_id,
    const scalar_t* node_load,
    const scalar_t* node_slew,
    const scalar_t* effective_node_cap,
    const scalar_t* grad_segment_delay,
    const scalar_t* grad_segment_output_slew,
    const scalar_t* grad_segment_upstream_cap,
    const scalar_t* grad_sink_arrival,
    const scalar_t* grad_sink_slew,
    const scalar_t* grad_sink_load,
    scalar_t* grad_load,
    scalar_t* grad_arrival,
    scalar_t* grad_slew,
    scalar_t* grad_eff_cap,
    scalar_t* grad_z,
    scalar_t* grad_bsu,
    scalar_t* grad_load_input_cap,
    scalar_t* grad_driver_arrival,
    scalar_t* grad_driver_slew,
    scalar_t* grad_edge_resistance,
    scalar_t* grad_edge_capacitance,
    int32_t net_count,
    int32_t node_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t sink_count,
    int32_t max_count,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    cudaStream_t stream);

template <typename scalar_t>
void segmentCountCapBackwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* z_value,
    const scalar_t* load_input_cap,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* node_load,
    const scalar_t* grad_driver_net_cap,
    scalar_t* grad_load,
    scalar_t* grad_effective_node_cap,
    const scalar_t* grad_segment_upstream_cap,
    scalar_t* grad_z,
    scalar_t* grad_load_input_cap,
    int32_t net_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t max_count,
    cudaStream_t stream);

template <typename scalar_t>
void candidateNetSubgraphForwardCudaLauncher(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const scalar_t* node_capacitance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_node_id,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    int64_t* candidate_index_by_node,
    scalar_t* effective_node_cap,
    scalar_t* lout,
    scalar_t* lin,
    scalar_t* arrival_in,
    scalar_t* arrival_out,
    scalar_t* slew_in,
    scalar_t* slew_out,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    scalar_t* sink_cap,
    scalar_t* sink_net_delay,
    scalar_t* sink_net_impulse,
    int32_t net_count,
    int32_t node_count,
    int32_t candidate_count,
    int32_t sink_count,
    cudaStream_t stream);

template <typename scalar_t>
void candidateNetSubgraphFixedBsuForwardCudaLauncher(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const scalar_t* node_capacitance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_node_id,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* probe_buffer_delay,
    const scalar_t* probe_buffer_output_slew,
    const scalar_t* upstream_retained_cap,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    int32_t fixed_bsu_index,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    int64_t* candidate_index_by_node,
    scalar_t* effective_node_cap,
    scalar_t* lout,
    scalar_t* lin,
    scalar_t* probe_arrival_in,
    scalar_t* probe_arrival_out,
    scalar_t* probe_slew_in,
    scalar_t* probe_slew_out,
    scalar_t* candidate_input_slew,
    scalar_t* candidate_output_load,
    scalar_t* buffer_delay,
    scalar_t* buffer_output_slew,
    scalar_t* buffer_delay_grad_slew,
    scalar_t* buffer_delay_grad_load,
    scalar_t* buffer_output_slew_grad_slew,
    scalar_t* buffer_output_slew_grad_load,
    scalar_t* arrival_in,
    scalar_t* arrival_out,
    scalar_t* slew_in,
    scalar_t* slew_out,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    scalar_t* sink_cap,
    scalar_t* sink_net_delay,
    scalar_t* sink_net_impulse,
    int32_t net_count,
    int32_t node_count,
    int32_t candidate_count,
    int32_t sink_count,
    cudaStream_t stream);

template <typename scalar_t>
void candidateNetSubgraphBackwardCudaLauncher(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const int64_t* candidate_index_by_node,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    const scalar_t* lin,
    const scalar_t* lout,
    const scalar_t* slew_in,
    const scalar_t* slew_out,
    const scalar_t* grad_sink_arrival,
    const scalar_t* grad_sink_slew,
    const scalar_t* grad_sink_load,
    const scalar_t* grad_sink_net_delay,
    const scalar_t* grad_sink_net_impulse,
    scalar_t* grad_lout,
    scalar_t* grad_lin,
    scalar_t* grad_arrival_in,
    scalar_t* grad_arrival_out,
    scalar_t* grad_slew_in,
    scalar_t* grad_slew_out,
    scalar_t* grad_driver_arrival,
    scalar_t* grad_driver_slew,
    scalar_t* grad_candidate_bu,
    scalar_t* grad_buffer_input_cap,
    scalar_t* grad_buffer_delay,
    scalar_t* grad_buffer_output_slew,
    int32_t net_count,
    int32_t sink_count,
    cudaStream_t stream);

namespace {

constexpr int64_t kMaxBackwardRepeaterCount = 16;

void check_inputs(
    const at::Tensor& input_arrival,
    const at::Tensor& input_slew,
    const at::Tensor& downstream_load,
    const at::Tensor& edge_resistance,
    const at::Tensor& edge_capacitance,
    const at::Tensor& repeater_count,
    const at::Tensor& split_fractions,
    const at::Tensor& bsu_index,
    const at::Tensor& upstream_retained_cap,
    const at::Tensor& buffer_input_cap_by_size,
    const at::Tensor& buffer_slew_axis,
    const at::Tensor& buffer_load_axis,
    const at::Tensor& buffer_delay_lut,
    const at::Tensor& buffer_output_slew_lut) {
  CHECK_FLAT_CUDA(input_arrival);
  CHECK_FLAT_CUDA(input_slew);
  CHECK_FLAT_CUDA(downstream_load);
  CHECK_FLAT_CUDA(edge_resistance);
  CHECK_FLAT_CUDA(edge_capacitance);
  CHECK_FLAT_CUDA(repeater_count);
  CHECK_CUDA(split_fractions);
  CHECK_FLAT_CUDA(bsu_index);
  CHECK_FLAT_CUDA(upstream_retained_cap);
  CHECK_FLAT_CUDA(buffer_input_cap_by_size);
  CHECK_FLAT_CUDA(buffer_slew_axis);
  CHECK_FLAT_CUDA(buffer_load_axis);
  CHECK_CUDA(buffer_delay_lut);
  CHECK_CUDA(buffer_output_slew_lut);
  CHECK_CONTIGUOUS(input_arrival);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_CONTIGUOUS(downstream_load);
  CHECK_CONTIGUOUS(edge_resistance);
  CHECK_CONTIGUOUS(edge_capacitance);
  CHECK_CONTIGUOUS(repeater_count);
  CHECK_CONTIGUOUS(split_fractions);
  CHECK_CONTIGUOUS(bsu_index);
  CHECK_CONTIGUOUS(upstream_retained_cap);
  CHECK_CONTIGUOUS(buffer_input_cap_by_size);
  CHECK_CONTIGUOUS(buffer_slew_axis);
  CHECK_CONTIGUOUS(buffer_load_axis);
  CHECK_CONTIGUOUS(buffer_delay_lut);
  CHECK_CONTIGUOUS(buffer_output_slew_lut);
  TORCH_CHECK(repeater_count.scalar_type() == at::kLong, "repeater_count must be int64");
  TORCH_CHECK(split_fractions.dim() == 2, "split_fractions must be 2-D");
  TORCH_CHECK(buffer_delay_lut.dim() == 3, "buffer_delay_lut must be 3-D");
  TORCH_CHECK(buffer_output_slew_lut.dim() == 3, "buffer_output_slew_lut must be 3-D");
  TORCH_CHECK(input_arrival.scalar_type() == input_slew.scalar_type(), "input_slew dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == downstream_load.scalar_type(), "downstream_load dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == edge_resistance.scalar_type(), "edge_resistance dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == edge_capacitance.scalar_type(), "edge_capacitance dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == split_fractions.scalar_type(), "split_fractions dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == bsu_index.scalar_type(), "bsu_index dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == upstream_retained_cap.scalar_type(), "upstream_retained_cap dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == buffer_input_cap_by_size.scalar_type(), "buffer_input_cap_by_size dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == buffer_slew_axis.scalar_type(), "buffer_slew_axis dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == buffer_load_axis.scalar_type(), "buffer_load_axis dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == buffer_delay_lut.scalar_type(), "buffer_delay_lut dtype mismatch");
  TORCH_CHECK(input_arrival.scalar_type() == buffer_output_slew_lut.scalar_type(), "buffer_output_slew_lut dtype mismatch");
  const int64_t sample_count = input_arrival.numel();
  TORCH_CHECK(input_slew.numel() == sample_count, "input_slew size mismatch");
  TORCH_CHECK(downstream_load.numel() == sample_count, "downstream_load size mismatch");
  TORCH_CHECK(edge_resistance.numel() == sample_count, "edge_resistance size mismatch");
  TORCH_CHECK(edge_capacitance.numel() == sample_count, "edge_capacitance size mismatch");
  TORCH_CHECK(repeater_count.numel() == sample_count, "repeater_count size mismatch");
  TORCH_CHECK(split_fractions.size(0) == sample_count, "split_fractions batch size mismatch");
  TORCH_CHECK(bsu_index.numel() == sample_count, "bsu_index size mismatch");
  TORCH_CHECK(upstream_retained_cap.numel() == sample_count, "upstream_retained_cap size mismatch");
  TORCH_CHECK(split_fractions.size(1) >= 1, "split_fractions must have at least one column");
  TORCH_CHECK(buffer_input_cap_by_size.numel() >= 1, "buffer_input_cap_by_size must be non-empty");
  TORCH_CHECK(buffer_slew_axis.numel() >= 1, "buffer_slew_axis must be non-empty");
  TORCH_CHECK(buffer_load_axis.numel() >= 1, "buffer_load_axis must be non-empty");
  TORCH_CHECK(
      buffer_delay_lut.size(0) == buffer_input_cap_by_size.numel() &&
          buffer_delay_lut.size(1) == buffer_slew_axis.numel() &&
          buffer_delay_lut.size(2) == buffer_load_axis.numel(),
      "buffer_delay_lut shape mismatch");
  TORCH_CHECK(
      buffer_output_slew_lut.sizes() == buffer_delay_lut.sizes(),
      "buffer_output_slew_lut shape mismatch");
}

void check_int64_cuda_1d(const at::Tensor& tensor, const char* name) {
  CHECK_FLAT_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  TORCH_CHECK(tensor.scalar_type() == at::kLong, name, " must be int64");
}

void check_float_cuda_1d_like(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalar_type) {
  CHECK_FLAT_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " dtype mismatch");
}

void check_float_cuda_2d_like(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalar_type) {
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D");
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " dtype mismatch");
}

void check_float_cuda_3d_like(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType scalar_type) {
  CHECK_CUDA(tensor);
  CHECK_CONTIGUOUS(tensor);
  TORCH_CHECK(tensor.dim() == 3, name, " must be 3-D");
  TORCH_CHECK(tensor.scalar_type() == scalar_type, name, " dtype mismatch");
}

void check_same_cuda_device(
    const at::Tensor& reference,
    std::initializer_list<std::pair<const at::Tensor*, const char*>> tensors) {
  for (const auto& entry : tensors) {
    TORCH_CHECK(
        entry.first->device() == reference.device(),
        entry.second,
        " must be on ",
        reference.device(),
        ", got ",
        entry.first->device());
  }
}

}  // namespace

std::vector<at::Tensor> segment_transfer_forward_cuda(
    at::Tensor input_arrival,
    at::Tensor input_slew,
    at::Tensor downstream_load,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor repeater_count,
    at::Tensor split_fractions,
    at::Tensor bsu_index,
    at::Tensor upstream_retained_cap,
    at::Tensor buffer_input_cap_by_size,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut) {
  check_inputs(
      input_arrival,
      input_slew,
      downstream_load,
      edge_resistance,
      edge_capacitance,
      repeater_count,
      split_fractions,
      bsu_index,
      upstream_retained_cap,
      buffer_input_cap_by_size,
      buffer_slew_axis,
      buffer_load_axis,
      buffer_delay_lut,
      buffer_output_slew_lut);
  check_same_cuda_device(
      input_arrival,
      {{&input_slew, "input_slew"},
       {&downstream_load, "downstream_load"},
       {&edge_resistance, "edge_resistance"},
       {&edge_capacitance, "edge_capacitance"},
       {&repeater_count, "repeater_count"},
       {&split_fractions, "split_fractions"},
       {&bsu_index, "bsu_index"},
       {&upstream_retained_cap, "upstream_retained_cap"},
       {&buffer_input_cap_by_size, "buffer_input_cap_by_size"},
       {&buffer_slew_axis, "buffer_slew_axis"},
       {&buffer_load_axis, "buffer_load_axis"},
       {&buffer_delay_lut, "buffer_delay_lut"},
       {&buffer_output_slew_lut, "buffer_output_slew_lut"}});
  const c10::cuda::CUDAGuard device_guard(input_arrival.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(input_arrival.get_device()).stream();

  auto upstream_visible_load = at::empty_like(input_arrival);
  auto segment_delay = at::empty_like(input_arrival);
  auto output_arrival = at::empty_like(input_arrival);
  auto output_slew = at::empty_like(input_arrival);
  auto first_buffer_input_slew = at::zeros_like(input_arrival);
  auto first_buffer_output_load = at::zeros_like(input_arrival);
  auto first_buffer_delay = at::zeros_like(input_arrival);
  auto first_buffer_output_slew = at::zeros_like(input_arrival);

  const int32_t sample_count = static_cast<int32_t>(input_arrival.numel());
  const int32_t nmax = static_cast<int32_t>(split_fractions.size(1) - 1);
  const int32_t size_count = static_cast<int32_t>(buffer_input_cap_by_size.numel());
  const int32_t slew_count = static_cast<int32_t>(buffer_slew_axis.numel());
  const int32_t load_count = static_cast<int32_t>(buffer_load_axis.numel());

  AT_DISPATCH_FLOATING_TYPES(input_arrival.scalar_type(), "segmentTransferForwardCuda", [&] {
    segmentTransferForwardCudaLauncher<scalar_t>(
        input_arrival.data_ptr<scalar_t>(),
        input_slew.data_ptr<scalar_t>(),
        downstream_load.data_ptr<scalar_t>(),
        edge_resistance.data_ptr<scalar_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        repeater_count.data_ptr<int64_t>(),
        split_fractions.data_ptr<scalar_t>(),
        bsu_index.data_ptr<scalar_t>(),
        upstream_retained_cap.data_ptr<scalar_t>(),
        buffer_input_cap_by_size.data_ptr<scalar_t>(),
        buffer_slew_axis.data_ptr<scalar_t>(),
        buffer_load_axis.data_ptr<scalar_t>(),
        buffer_delay_lut.data_ptr<scalar_t>(),
        buffer_output_slew_lut.data_ptr<scalar_t>(),
        upstream_visible_load.data_ptr<scalar_t>(),
        segment_delay.data_ptr<scalar_t>(),
        output_arrival.data_ptr<scalar_t>(),
        output_slew.data_ptr<scalar_t>(),
        first_buffer_input_slew.data_ptr<scalar_t>(),
        first_buffer_output_load.data_ptr<scalar_t>(),
        first_buffer_delay.data_ptr<scalar_t>(),
        first_buffer_output_slew.data_ptr<scalar_t>(),
        sample_count,
        nmax,
        size_count,
        slew_count,
        load_count,
        stream);
  });

  return {
      upstream_visible_load,
      segment_delay,
      output_arrival,
      output_slew,
      first_buffer_input_slew,
      first_buffer_output_load,
      first_buffer_delay,
      first_buffer_output_slew,
  };
}

std::vector<at::Tensor> segment_count_transfer_forward_cuda(
    at::Tensor net_topo_start,
    at::Tensor flat_topo_node_id,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor node_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor z_value,
    at::Tensor bsu_index,
    at::Tensor load_input_cap,
    at::Tensor buffer_input_cap_by_size,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor segment_sub_resistance_fraction,
    at::Tensor segment_retained_upstream_cap,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index,
    at::Tensor sink_node_compact_id) {
  (void)flat_topo_node_id;
  (void)segment_retained_upstream_cap;
  (void)sink_node_id;
  (void)sink_net_index;
  const auto scalar_type = node_capacitance.scalar_type();
  check_int64_cuda_1d(net_topo_start, "net_topo_start");
  check_int64_cuda_1d(edge_start, "edge_start");
  check_int64_cuda_1d(edge_parent_compact_id, "edge_parent_compact_id");
  check_int64_cuda_1d(edge_child_compact_id, "edge_child_compact_id");
  check_int64_cuda_1d(edge_to_segment_id, "edge_to_segment_id");
  check_int64_cuda_1d(sink_node_compact_id, "sink_node_compact_id");
  check_float_cuda_1d_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_cuda_1d_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_cuda_1d_like(driver_arrival, "driver_arrival", scalar_type);
  check_float_cuda_1d_like(driver_slew, "driver_slew", scalar_type);
  check_float_cuda_1d_like(z_value, "z_value", scalar_type);
  check_float_cuda_1d_like(bsu_index, "bsu_index", scalar_type);
  check_float_cuda_1d_like(load_input_cap, "load_input_cap", scalar_type);
  check_float_cuda_1d_like(buffer_input_cap_by_size, "buffer_input_cap_by_size", scalar_type);
  check_float_cuda_1d_like(buffer_slew_axis, "buffer_slew_axis", scalar_type);
  check_float_cuda_1d_like(buffer_load_axis, "buffer_load_axis", scalar_type);
  check_float_cuda_1d_like(node_capacitance, "node_capacitance", scalar_type);
  check_float_cuda_2d_like(parent_cap_fraction, "parent_cap_fraction", scalar_type);
  check_float_cuda_2d_like(child_cap_fraction, "child_cap_fraction", scalar_type);
  check_float_cuda_3d_like(
      segment_sub_resistance_fraction,
      "segment_sub_resistance_fraction",
      scalar_type);
  check_float_cuda_3d_like(buffer_delay_lut, "buffer_delay_lut", scalar_type);
  check_float_cuda_3d_like(buffer_output_slew_lut, "buffer_output_slew_lut", scalar_type);
  TORCH_CHECK(buffer_input_cap_by_size.numel() >= 1, "buffer_input_cap_by_size must be non-empty");
  TORCH_CHECK(buffer_slew_axis.numel() >= 1, "buffer_slew_axis must be non-empty");
  TORCH_CHECK(buffer_load_axis.numel() >= 1, "buffer_load_axis must be non-empty");
  TORCH_CHECK(
      buffer_delay_lut.size(0) == buffer_input_cap_by_size.numel() &&
          buffer_delay_lut.size(1) == buffer_slew_axis.numel() &&
          buffer_delay_lut.size(2) == buffer_load_axis.numel(),
      "buffer_delay_lut shape mismatch");
  TORCH_CHECK(buffer_output_slew_lut.sizes() == buffer_delay_lut.sizes(),
              "buffer_output_slew_lut shape mismatch");
  TORCH_CHECK(parent_cap_fraction.size(0) == z_value.numel(), "parent_cap_fraction segment size mismatch");
  TORCH_CHECK(child_cap_fraction.sizes() == parent_cap_fraction.sizes(), "child_cap_fraction shape mismatch");
  TORCH_CHECK(segment_sub_resistance_fraction.size(0) == z_value.numel(),
              "segment_sub_resistance_fraction segment size mismatch");
  check_same_cuda_device(
      node_capacitance,
      {{&net_topo_start, "net_topo_start"},
       {&edge_start, "edge_start"},
       {&edge_parent_compact_id, "edge_parent_compact_id"},
       {&edge_child_compact_id, "edge_child_compact_id"},
       {&edge_resistance, "edge_resistance"},
       {&edge_capacitance, "edge_capacitance"},
       {&edge_to_segment_id, "edge_to_segment_id"},
       {&driver_arrival, "driver_arrival"},
       {&driver_slew, "driver_slew"},
       {&z_value, "z_value"},
       {&bsu_index, "bsu_index"},
       {&load_input_cap, "load_input_cap"},
       {&buffer_input_cap_by_size, "buffer_input_cap_by_size"},
       {&buffer_slew_axis, "buffer_slew_axis"},
       {&buffer_load_axis, "buffer_load_axis"},
       {&buffer_delay_lut, "buffer_delay_lut"},
       {&buffer_output_slew_lut, "buffer_output_slew_lut"},
       {&parent_cap_fraction, "parent_cap_fraction"},
       {&child_cap_fraction, "child_cap_fraction"},
       {&segment_sub_resistance_fraction, "segment_sub_resistance_fraction"},
       {&sink_node_compact_id, "sink_node_compact_id"}});
  const c10::cuda::CUDAGuard device_guard(node_capacitance.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(node_capacitance.get_device()).stream();
  TORCH_CHECK(segment_sub_resistance_fraction.size(1) == parent_cap_fraction.size(1),
              "segment_sub_resistance_fraction count size mismatch");
  TORCH_CHECK(segment_sub_resistance_fraction.size(2) == parent_cap_fraction.size(1),
              "segment_sub_resistance_fraction sub-count size mismatch");

  auto effective_node_cap = at::empty_like(node_capacitance);
  auto node_load = at::empty_like(node_capacitance);
  auto node_arrival = at::empty_like(node_capacitance);
  auto node_slew = at::empty_like(node_capacitance);
  auto segment_delay = at::empty_like(z_value);
  auto segment_output_slew = at::empty_like(z_value);
  auto segment_upstream_cap = at::empty_like(z_value);
  auto sink_arrival = at::empty(
      {sink_node_compact_id.numel()},
      node_capacitance.options());
  auto sink_slew = at::empty(
      {sink_node_compact_id.numel()},
      node_capacitance.options());
  auto sink_load = at::empty(
      {sink_node_compact_id.numel()},
      node_capacitance.options());

  const int32_t net_count = static_cast<int32_t>(net_topo_start.numel() - 1);
  const int32_t node_count = static_cast<int32_t>(node_capacitance.numel());
  const int32_t edge_count = static_cast<int32_t>(edge_resistance.numel());
  const int32_t segment_count = static_cast<int32_t>(z_value.numel());
  const int32_t sink_count = static_cast<int32_t>(sink_node_compact_id.numel());
  const int32_t max_count = static_cast<int32_t>(parent_cap_fraction.size(1) - 1);
  const int32_t size_count = static_cast<int32_t>(buffer_input_cap_by_size.numel());
  const int32_t slew_count = static_cast<int32_t>(buffer_slew_axis.numel());
  const int32_t load_count = static_cast<int32_t>(buffer_load_axis.numel());

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segmentCountTransferForwardCuda", [&] {
    segmentCountTransferForwardCudaLauncher<scalar_t>(
        net_topo_start.data_ptr<int64_t>(),
        edge_start.data_ptr<int64_t>(),
        edge_parent_compact_id.data_ptr<int64_t>(),
        edge_child_compact_id.data_ptr<int64_t>(),
        edge_resistance.data_ptr<scalar_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        node_capacitance.data_ptr<scalar_t>(),
        edge_to_segment_id.data_ptr<int64_t>(),
        driver_arrival.data_ptr<scalar_t>(),
        driver_slew.data_ptr<scalar_t>(),
        z_value.data_ptr<scalar_t>(),
        bsu_index.data_ptr<scalar_t>(),
        load_input_cap.data_ptr<scalar_t>(),
        buffer_input_cap_by_size.data_ptr<scalar_t>(),
        buffer_slew_axis.data_ptr<scalar_t>(),
        buffer_load_axis.data_ptr<scalar_t>(),
        buffer_delay_lut.data_ptr<scalar_t>(),
        buffer_output_slew_lut.data_ptr<scalar_t>(),
        parent_cap_fraction.data_ptr<scalar_t>(),
        child_cap_fraction.data_ptr<scalar_t>(),
        segment_sub_resistance_fraction.data_ptr<scalar_t>(),
        sink_node_compact_id.data_ptr<int64_t>(),
        effective_node_cap.data_ptr<scalar_t>(),
        node_load.data_ptr<scalar_t>(),
        node_arrival.data_ptr<scalar_t>(),
        node_slew.data_ptr<scalar_t>(),
        segment_delay.data_ptr<scalar_t>(),
        segment_output_slew.data_ptr<scalar_t>(),
        segment_upstream_cap.data_ptr<scalar_t>(),
        sink_arrival.data_ptr<scalar_t>(),
        sink_slew.data_ptr<scalar_t>(),
        sink_load.data_ptr<scalar_t>(),
        net_count,
        node_count,
        edge_count,
        segment_count,
        sink_count,
        max_count,
        size_count,
        slew_count,
        load_count,
        stream);
  });

  return {
      segment_delay,
      segment_output_slew,
      segment_upstream_cap,
      node_load,
      node_arrival,
      node_slew,
      sink_arrival,
      sink_slew,
      sink_load,
      effective_node_cap,
  };
}

std::vector<at::Tensor> segment_count_cap_forward_cuda(
    at::Tensor net_topo_start,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_capacitance,
    at::Tensor node_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor z_value,
    at::Tensor load_input_cap,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction) {
  const auto scalar_type = node_capacitance.scalar_type();
  check_int64_cuda_1d(net_topo_start, "net_topo_start");
  check_int64_cuda_1d(edge_start, "edge_start");
  check_int64_cuda_1d(edge_parent_compact_id, "edge_parent_compact_id");
  check_int64_cuda_1d(edge_child_compact_id, "edge_child_compact_id");
  check_int64_cuda_1d(edge_to_segment_id, "edge_to_segment_id");
  check_float_cuda_1d_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_cuda_1d_like(node_capacitance, "node_capacitance", scalar_type);
  check_float_cuda_1d_like(z_value, "z_value", scalar_type);
  check_float_cuda_1d_like(load_input_cap, "load_input_cap", scalar_type);
  check_float_cuda_2d_like(parent_cap_fraction, "parent_cap_fraction", scalar_type);
  check_float_cuda_2d_like(child_cap_fraction, "child_cap_fraction", scalar_type);
  TORCH_CHECK(parent_cap_fraction.size(0) == z_value.numel(),
              "parent_cap_fraction segment size mismatch");
  TORCH_CHECK(child_cap_fraction.sizes() == parent_cap_fraction.sizes(),
              "child_cap_fraction shape mismatch");
  TORCH_CHECK(load_input_cap.numel() == z_value.numel(),
              "load_input_cap segment size mismatch");
  check_same_cuda_device(
      node_capacitance,
      {{&net_topo_start, "net_topo_start"},
       {&edge_start, "edge_start"},
       {&edge_parent_compact_id, "edge_parent_compact_id"},
       {&edge_child_compact_id, "edge_child_compact_id"},
       {&edge_capacitance, "edge_capacitance"},
       {&edge_to_segment_id, "edge_to_segment_id"},
       {&z_value, "z_value"},
       {&load_input_cap, "load_input_cap"},
       {&parent_cap_fraction, "parent_cap_fraction"},
       {&child_cap_fraction, "child_cap_fraction"}});
  const c10::cuda::CUDAGuard device_guard(node_capacitance.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(node_capacitance.get_device()).stream();
  const int32_t net_count = static_cast<int32_t>(net_topo_start.numel() - 1);
  const int32_t node_count = static_cast<int32_t>(node_capacitance.numel());
  const int32_t edge_count = static_cast<int32_t>(edge_capacitance.numel());
  const int32_t segment_count = static_cast<int32_t>(z_value.numel());
  const int32_t max_count = static_cast<int32_t>(parent_cap_fraction.size(1) - 1);
  auto effective_node_cap = at::empty_like(node_capacitance);
  auto node_load = at::empty_like(node_capacitance);

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segmentCountCapForwardCuda", [&] {
    segmentCountCapForwardCudaLauncher<scalar_t>(
        net_topo_start.data_ptr<int64_t>(),
        edge_start.data_ptr<int64_t>(),
        edge_parent_compact_id.data_ptr<int64_t>(),
        edge_child_compact_id.data_ptr<int64_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        node_capacitance.data_ptr<scalar_t>(),
        edge_to_segment_id.data_ptr<int64_t>(),
        z_value.data_ptr<scalar_t>(),
        load_input_cap.data_ptr<scalar_t>(),
        parent_cap_fraction.data_ptr<scalar_t>(),
        child_cap_fraction.data_ptr<scalar_t>(),
        effective_node_cap.data_ptr<scalar_t>(),
        node_load.data_ptr<scalar_t>(),
        net_count,
        node_count,
        edge_count,
        segment_count,
        max_count,
        stream);
  });
  return {node_load, effective_node_cap};
}

py::dict segment_count_transfer_backward_cuda(
    at::Tensor net_topo_start,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_resistance,
    at::Tensor edge_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor z_value,
    at::Tensor bsu_index,
    at::Tensor load_input_cap,
    at::Tensor buffer_input_cap_by_size,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor segment_sub_resistance_fraction,
    at::Tensor sink_node_compact_id,
    at::Tensor node_load,
    at::Tensor node_slew,
    at::Tensor effective_node_cap,
    at::Tensor grad_segment_delay,
    at::Tensor grad_segment_output_slew,
    at::Tensor grad_segment_upstream_cap,
    at::Tensor grad_sink_arrival,
    at::Tensor grad_sink_slew,
    at::Tensor grad_sink_load) {
  (void)driver_arrival;
  (void)effective_node_cap;
  const auto scalar_type = node_load.scalar_type();
  check_int64_cuda_1d(net_topo_start, "net_topo_start");
  check_int64_cuda_1d(edge_start, "edge_start");
  check_int64_cuda_1d(edge_parent_compact_id, "edge_parent_compact_id");
  check_int64_cuda_1d(edge_child_compact_id, "edge_child_compact_id");
  check_int64_cuda_1d(edge_to_segment_id, "edge_to_segment_id");
  check_int64_cuda_1d(sink_node_compact_id, "sink_node_compact_id");
  check_float_cuda_1d_like(edge_resistance, "edge_resistance", scalar_type);
  check_float_cuda_1d_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_cuda_1d_like(driver_slew, "driver_slew", scalar_type);
  check_float_cuda_1d_like(z_value, "z_value", scalar_type);
  check_float_cuda_1d_like(bsu_index, "bsu_index", scalar_type);
  check_float_cuda_1d_like(load_input_cap, "load_input_cap", scalar_type);
  check_float_cuda_1d_like(buffer_input_cap_by_size, "buffer_input_cap_by_size", scalar_type);
  check_float_cuda_1d_like(buffer_slew_axis, "buffer_slew_axis", scalar_type);
  check_float_cuda_1d_like(buffer_load_axis, "buffer_load_axis", scalar_type);
  check_float_cuda_1d_like(node_load, "node_load", scalar_type);
  check_float_cuda_1d_like(node_slew, "node_slew", scalar_type);
  check_float_cuda_1d_like(grad_segment_delay, "grad_segment_delay", scalar_type);
  check_float_cuda_1d_like(
      grad_segment_output_slew,
      "grad_segment_output_slew",
      scalar_type);
  check_float_cuda_1d_like(
      grad_segment_upstream_cap,
      "grad_segment_upstream_cap",
      scalar_type);
  check_float_cuda_1d_like(grad_sink_arrival, "grad_sink_arrival", scalar_type);
  check_float_cuda_1d_like(grad_sink_slew, "grad_sink_slew", scalar_type);
  check_float_cuda_1d_like(grad_sink_load, "grad_sink_load", scalar_type);
  check_float_cuda_2d_like(parent_cap_fraction, "parent_cap_fraction", scalar_type);
  check_float_cuda_2d_like(child_cap_fraction, "child_cap_fraction", scalar_type);
  check_float_cuda_3d_like(
      segment_sub_resistance_fraction,
      "segment_sub_resistance_fraction",
      scalar_type);
  check_float_cuda_3d_like(buffer_delay_lut, "buffer_delay_lut", scalar_type);
  check_float_cuda_3d_like(buffer_output_slew_lut, "buffer_output_slew_lut", scalar_type);

  const int32_t net_count = static_cast<int32_t>(net_topo_start.numel() - 1);
  const int32_t node_count = static_cast<int32_t>(node_load.numel());
  const int32_t edge_count = static_cast<int32_t>(edge_resistance.numel());
  const int32_t segment_count = static_cast<int32_t>(z_value.numel());
  const int32_t sink_count = static_cast<int32_t>(sink_node_compact_id.numel());
  const int32_t max_count = static_cast<int32_t>(parent_cap_fraction.size(1) - 1);
  const int32_t size_count = static_cast<int32_t>(buffer_input_cap_by_size.numel());
  const int32_t slew_count = static_cast<int32_t>(buffer_slew_axis.numel());
  const int32_t load_count = static_cast<int32_t>(buffer_load_axis.numel());
  TORCH_CHECK(
      max_count <= kMaxBackwardRepeaterCount,
      "CUDA segment-count backward supports max_repeater_count <= ",
      kMaxBackwardRepeaterCount);
  check_same_cuda_device(
      node_load,
      {{&net_topo_start, "net_topo_start"},
       {&edge_start, "edge_start"},
       {&edge_parent_compact_id, "edge_parent_compact_id"},
       {&edge_child_compact_id, "edge_child_compact_id"},
       {&edge_resistance, "edge_resistance"},
       {&edge_capacitance, "edge_capacitance"},
       {&edge_to_segment_id, "edge_to_segment_id"},
       {&driver_slew, "driver_slew"},
       {&z_value, "z_value"},
       {&bsu_index, "bsu_index"},
       {&load_input_cap, "load_input_cap"},
       {&buffer_input_cap_by_size, "buffer_input_cap_by_size"},
       {&buffer_slew_axis, "buffer_slew_axis"},
       {&buffer_load_axis, "buffer_load_axis"},
       {&buffer_delay_lut, "buffer_delay_lut"},
       {&buffer_output_slew_lut, "buffer_output_slew_lut"},
       {&parent_cap_fraction, "parent_cap_fraction"},
       {&child_cap_fraction, "child_cap_fraction"},
       {&segment_sub_resistance_fraction, "segment_sub_resistance_fraction"},
       {&sink_node_compact_id, "sink_node_compact_id"},
       {&node_slew, "node_slew"},
       {&effective_node_cap, "effective_node_cap"},
       {&grad_segment_delay, "grad_segment_delay"},
       {&grad_segment_output_slew, "grad_segment_output_slew"},
       {&grad_segment_upstream_cap, "grad_segment_upstream_cap"},
       {&grad_sink_arrival, "grad_sink_arrival"},
       {&grad_sink_slew, "grad_sink_slew"},
       {&grad_sink_load, "grad_sink_load"}});
  const c10::cuda::CUDAGuard device_guard(node_load.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(node_load.get_device()).stream();

  const auto allocation_started_at = std::chrono::steady_clock::now();
  auto grad_z = at::zeros_like(z_value);
  auto grad_bsu = at::zeros_like(z_value);
  auto grad_load_input_cap = at::zeros_like(z_value);
  auto grad_driver_arrival = at::zeros_like(driver_slew);
  auto grad_driver_slew = at::zeros_like(driver_slew);
  auto grad_load = at::zeros_like(node_load);
  auto grad_arrival = at::zeros_like(node_load);
  auto grad_slew = at::zeros_like(node_load);
  auto grad_eff_cap = at::zeros_like(node_load);
  auto grad_edge_resistance = at::zeros_like(edge_resistance);
  auto grad_edge_capacitance = at::zeros_like(edge_capacitance);
  const auto allocation_finished_at = std::chrono::steady_clock::now();
  const int64_t temporary_allocation_bytes =
      grad_z.nbytes() + grad_bsu.nbytes() + grad_load_input_cap.nbytes() +
      grad_driver_arrival.nbytes() + grad_driver_slew.nbytes() +
      grad_load.nbytes() + grad_arrival.nbytes() + grad_slew.nbytes() +
      grad_eff_cap.nbytes() + grad_edge_resistance.nbytes() +
      grad_edge_capacitance.nbytes();
  const double temporary_allocation_cpu_ms =
      std::chrono::duration<double, std::milli>(
          allocation_finished_at - allocation_started_at)
          .count();

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segmentCountTransferBackwardCuda", [&] {
    segmentCountTransferBackwardCudaLauncher<scalar_t>(
        net_topo_start.data_ptr<int64_t>(),
        edge_start.data_ptr<int64_t>(),
        edge_parent_compact_id.data_ptr<int64_t>(),
        edge_child_compact_id.data_ptr<int64_t>(),
        edge_resistance.data_ptr<scalar_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        edge_to_segment_id.data_ptr<int64_t>(),
        driver_slew.data_ptr<scalar_t>(),
        z_value.data_ptr<scalar_t>(),
        bsu_index.data_ptr<scalar_t>(),
        load_input_cap.data_ptr<scalar_t>(),
        buffer_input_cap_by_size.data_ptr<scalar_t>(),
        buffer_slew_axis.data_ptr<scalar_t>(),
        buffer_load_axis.data_ptr<scalar_t>(),
        buffer_delay_lut.data_ptr<scalar_t>(),
        buffer_output_slew_lut.data_ptr<scalar_t>(),
        parent_cap_fraction.data_ptr<scalar_t>(),
        child_cap_fraction.data_ptr<scalar_t>(),
        segment_sub_resistance_fraction.data_ptr<scalar_t>(),
        sink_node_compact_id.data_ptr<int64_t>(),
        node_load.data_ptr<scalar_t>(),
        node_slew.data_ptr<scalar_t>(),
        effective_node_cap.data_ptr<scalar_t>(),
        grad_segment_delay.data_ptr<scalar_t>(),
        grad_segment_output_slew.data_ptr<scalar_t>(),
        grad_segment_upstream_cap.data_ptr<scalar_t>(),
        grad_sink_arrival.data_ptr<scalar_t>(),
        grad_sink_slew.data_ptr<scalar_t>(),
        grad_sink_load.data_ptr<scalar_t>(),
        grad_load.data_ptr<scalar_t>(),
        grad_arrival.data_ptr<scalar_t>(),
        grad_slew.data_ptr<scalar_t>(),
        grad_eff_cap.data_ptr<scalar_t>(),
        grad_z.data_ptr<scalar_t>(),
        grad_bsu.data_ptr<scalar_t>(),
        grad_load_input_cap.data_ptr<scalar_t>(),
        grad_driver_arrival.data_ptr<scalar_t>(),
        grad_driver_slew.data_ptr<scalar_t>(),
        grad_edge_resistance.data_ptr<scalar_t>(),
        grad_edge_capacitance.data_ptr<scalar_t>(),
        net_count,
        node_count,
        edge_count,
        segment_count,
        sink_count,
        max_count,
        size_count,
        slew_count,
        load_count,
        stream);
  });

  py::dict result;
  result["grad_z"] = grad_z;
  result["grad_bsu"] = grad_bsu;
  result["grad_load_input_cap"] = grad_load_input_cap;
  result["grad_driver_arrival"] = grad_driver_arrival;
  result["grad_driver_slew"] = grad_driver_slew;
  result["grad_edge_resistance"] = grad_edge_resistance;
  result["grad_edge_capacitance"] = grad_edge_capacitance;
  result["temporary_allocation_bytes"] = temporary_allocation_bytes;
  result["temporary_allocation_cpu_ms"] = temporary_allocation_cpu_ms;
  return result;
}

py::dict segment_count_cap_backward_cuda(
    at::Tensor net_topo_start,
    at::Tensor edge_start,
    at::Tensor edge_parent_compact_id,
    at::Tensor edge_child_compact_id,
    at::Tensor edge_capacitance,
    at::Tensor edge_to_segment_id,
    at::Tensor z_value,
    at::Tensor load_input_cap,
    at::Tensor parent_cap_fraction,
    at::Tensor child_cap_fraction,
    at::Tensor node_load,
    at::Tensor grad_driver_net_cap) {
  const auto scalar_type = node_load.scalar_type();
  check_int64_cuda_1d(net_topo_start, "net_topo_start");
  check_int64_cuda_1d(edge_start, "edge_start");
  check_int64_cuda_1d(edge_parent_compact_id, "edge_parent_compact_id");
  check_int64_cuda_1d(edge_child_compact_id, "edge_child_compact_id");
  check_int64_cuda_1d(edge_to_segment_id, "edge_to_segment_id");
  check_float_cuda_1d_like(edge_capacitance, "edge_capacitance", scalar_type);
  check_float_cuda_1d_like(z_value, "z_value", scalar_type);
  check_float_cuda_1d_like(load_input_cap, "load_input_cap", scalar_type);
  check_float_cuda_1d_like(node_load, "node_load", scalar_type);
  check_float_cuda_1d_like(
      grad_driver_net_cap,
      "grad_driver_net_cap",
      scalar_type);
  check_float_cuda_2d_like(parent_cap_fraction, "parent_cap_fraction", scalar_type);
  check_float_cuda_2d_like(child_cap_fraction, "child_cap_fraction", scalar_type);
  TORCH_CHECK(parent_cap_fraction.size(0) == z_value.numel(),
              "parent_cap_fraction segment size mismatch");
  TORCH_CHECK(child_cap_fraction.sizes() == parent_cap_fraction.sizes(),
              "child_cap_fraction shape mismatch");
  TORCH_CHECK(load_input_cap.numel() == z_value.numel(),
              "load_input_cap segment size mismatch");
  TORCH_CHECK(grad_driver_net_cap.numel() == net_topo_start.numel() - 1,
              "grad_driver_net_cap net size mismatch");
  check_same_cuda_device(
      node_load,
      {{&net_topo_start, "net_topo_start"},
       {&edge_start, "edge_start"},
       {&edge_parent_compact_id, "edge_parent_compact_id"},
       {&edge_child_compact_id, "edge_child_compact_id"},
       {&edge_capacitance, "edge_capacitance"},
       {&edge_to_segment_id, "edge_to_segment_id"},
       {&z_value, "z_value"},
       {&load_input_cap, "load_input_cap"},
       {&parent_cap_fraction, "parent_cap_fraction"},
       {&child_cap_fraction, "child_cap_fraction"},
       {&grad_driver_net_cap, "grad_driver_net_cap"}});
  const c10::cuda::CUDAGuard device_guard(node_load.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(node_load.get_device()).stream();
  const int32_t net_count = static_cast<int32_t>(net_topo_start.numel() - 1);
  const int32_t edge_count = static_cast<int32_t>(edge_capacitance.numel());
  const int32_t segment_count = static_cast<int32_t>(z_value.numel());
  const int32_t max_count = static_cast<int32_t>(parent_cap_fraction.size(1) - 1);
  const auto allocation_started_at = std::chrono::steady_clock::now();
  auto grad_load = at::zeros_like(node_load);
  auto grad_effective_node_cap = at::zeros_like(node_load);
  auto grad_segment_upstream_cap = at::zeros_like(z_value);
  auto grad_z = at::zeros_like(z_value);
  auto grad_load_input_cap = at::zeros_like(z_value);
  const auto allocation_finished_at = std::chrono::steady_clock::now();
  const int64_t temporary_allocation_bytes =
      grad_load.nbytes() + grad_effective_node_cap.nbytes() +
      grad_segment_upstream_cap.nbytes() + grad_z.nbytes() +
      grad_load_input_cap.nbytes();
  const double temporary_allocation_cpu_ms =
      std::chrono::duration<double, std::milli>(
          allocation_finished_at - allocation_started_at)
          .count();

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "segmentCountCapBackwardCuda", [&] {
    segmentCountCapBackwardCudaLauncher<scalar_t>(
        net_topo_start.data_ptr<int64_t>(),
        edge_start.data_ptr<int64_t>(),
        edge_parent_compact_id.data_ptr<int64_t>(),
        edge_child_compact_id.data_ptr<int64_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        edge_to_segment_id.data_ptr<int64_t>(),
        z_value.data_ptr<scalar_t>(),
        load_input_cap.data_ptr<scalar_t>(),
        parent_cap_fraction.data_ptr<scalar_t>(),
        child_cap_fraction.data_ptr<scalar_t>(),
        node_load.data_ptr<scalar_t>(),
        grad_driver_net_cap.data_ptr<scalar_t>(),
        grad_load.data_ptr<scalar_t>(),
        grad_effective_node_cap.data_ptr<scalar_t>(),
        grad_segment_upstream_cap.data_ptr<scalar_t>(),
        grad_z.data_ptr<scalar_t>(),
        grad_load_input_cap.data_ptr<scalar_t>(),
        net_count,
        edge_count,
        segment_count,
        max_count,
        stream);
  });
  py::dict result;
  result["grad_z"] = grad_z;
  result["grad_load_input_cap"] = grad_load_input_cap;
  result["temporary_allocation_bytes"] = temporary_allocation_bytes;
  result["temporary_allocation_cpu_ms"] = temporary_allocation_cpu_ms;
  return result;
}

py::dict candidate_net_subgraph_forward_cuda(
    at::Tensor net_flat_topo_sort,
    at::Tensor net_flat_topo_sort_start,
    at::Tensor pin_fa,
    at::Tensor flat_pin_to_start,
    at::Tensor flat_pin_to,
    at::Tensor edge_resistance,
    at::Tensor node_capacitance,
    at::Tensor edge_capacitance,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor candidate_node_id,
    at::Tensor candidate_bu,
    at::Tensor buffer_input_cap,
    at::Tensor buffer_delay,
    at::Tensor buffer_output_slew,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index) {
  const auto scalar_type = node_capacitance.scalar_type();
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&net_flat_topo_sort, "net_flat_topo_sort"},
           {&net_flat_topo_sort_start, "net_flat_topo_sort_start"},
           {&pin_fa, "pin_fa"},
           {&flat_pin_to_start, "flat_pin_to_start"},
           {&flat_pin_to, "flat_pin_to"},
           {&candidate_node_id, "candidate_node_id"},
           {&sink_node_id, "sink_node_id"},
           {&sink_net_index, "sink_net_index"}}) {
    check_int64_cuda_1d(*entry.first, entry.second);
  }
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&edge_resistance, "edge_resistance"},
           {&node_capacitance, "node_capacitance"},
           {&edge_capacitance, "edge_capacitance"},
           {&driver_arrival, "driver_arrival"},
           {&driver_slew, "driver_slew"},
           {&candidate_bu, "candidate_bu"},
           {&buffer_input_cap, "buffer_input_cap"},
           {&buffer_delay, "buffer_delay"},
           {&buffer_output_slew, "buffer_output_slew"}}) {
    check_float_cuda_1d_like(*entry.first, entry.second, scalar_type);
  }
  const int64_t node_count = node_capacitance.numel();
  const int64_t net_count = net_flat_topo_sort_start.numel() - 1;
  const int64_t candidate_count = candidate_node_id.numel();
  const int64_t sink_count = sink_node_id.numel();
  TORCH_CHECK(pin_fa.numel() == node_count, "pin_fa size mismatch");
  TORCH_CHECK(flat_pin_to_start.numel() == node_count + 1, "flat_pin_to_start size mismatch");
  TORCH_CHECK(edge_resistance.numel() == node_count, "edge_resistance size mismatch");
  TORCH_CHECK(edge_capacitance.numel() == node_count, "edge_capacitance size mismatch");
  TORCH_CHECK(driver_arrival.numel() == net_count, "driver_arrival size mismatch");
  TORCH_CHECK(driver_slew.numel() == net_count, "driver_slew size mismatch");
  TORCH_CHECK(candidate_bu.numel() == candidate_count, "candidate_bu size mismatch");
  TORCH_CHECK(buffer_input_cap.numel() == candidate_count, "buffer_input_cap size mismatch");
  TORCH_CHECK(buffer_delay.numel() == candidate_count, "buffer_delay size mismatch");
  TORCH_CHECK(buffer_output_slew.numel() == candidate_count, "buffer_output_slew size mismatch");
  TORCH_CHECK(sink_net_index.numel() == sink_count, "sink_net_index size mismatch");
  check_same_cuda_device(
      node_capacitance,
      {{&net_flat_topo_sort, "net_flat_topo_sort"},
       {&net_flat_topo_sort_start, "net_flat_topo_sort_start"},
       {&pin_fa, "pin_fa"},
       {&flat_pin_to_start, "flat_pin_to_start"},
       {&flat_pin_to, "flat_pin_to"},
       {&edge_resistance, "edge_resistance"},
       {&edge_capacitance, "edge_capacitance"},
       {&driver_arrival, "driver_arrival"},
       {&driver_slew, "driver_slew"},
       {&candidate_node_id, "candidate_node_id"},
       {&candidate_bu, "candidate_bu"},
       {&buffer_input_cap, "buffer_input_cap"},
       {&buffer_delay, "buffer_delay"},
       {&buffer_output_slew, "buffer_output_slew"},
       {&sink_node_id, "sink_node_id"},
       {&sink_net_index, "sink_net_index"}});
  const c10::cuda::CUDAGuard device_guard(node_capacitance.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(node_capacitance.get_device()).stream();

  auto index_options = node_capacitance.options().dtype(at::kLong);
  auto candidate_index_by_node = at::full({node_count}, -1, index_options);
  auto effective_node_cap = at::zeros_like(node_capacitance);
  auto lout = at::zeros_like(node_capacitance);
  auto lin = at::zeros_like(node_capacitance);
  auto arrival_in = at::zeros_like(node_capacitance);
  auto arrival_out = at::zeros_like(node_capacitance);
  auto slew_in = at::zeros_like(node_capacitance);
  auto slew_out = at::zeros_like(node_capacitance);
  auto sink_arrival = at::zeros({sink_count}, node_capacitance.options());
  auto sink_slew = at::zeros({sink_count}, node_capacitance.options());
  auto sink_load = at::zeros({sink_count}, node_capacitance.options());
  auto sink_cap = at::zeros({sink_count}, node_capacitance.options());
  auto sink_net_delay = at::zeros({sink_count}, node_capacitance.options());
  auto sink_net_impulse = at::zeros({sink_count}, node_capacitance.options());

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "candidateNetSubgraphForwardCuda", [&] {
    candidateNetSubgraphForwardCudaLauncher<scalar_t>(
        net_flat_topo_sort.data_ptr<int64_t>(),
        net_flat_topo_sort_start.data_ptr<int64_t>(),
        pin_fa.data_ptr<int64_t>(),
        flat_pin_to_start.data_ptr<int64_t>(),
        flat_pin_to.data_ptr<int64_t>(),
        edge_resistance.data_ptr<scalar_t>(),
        node_capacitance.data_ptr<scalar_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        driver_arrival.data_ptr<scalar_t>(),
        driver_slew.data_ptr<scalar_t>(),
        candidate_node_id.data_ptr<int64_t>(),
        candidate_bu.data_ptr<scalar_t>(),
        buffer_input_cap.data_ptr<scalar_t>(),
        buffer_delay.data_ptr<scalar_t>(),
        buffer_output_slew.data_ptr<scalar_t>(),
        sink_node_id.data_ptr<int64_t>(),
        sink_net_index.data_ptr<int64_t>(),
        candidate_index_by_node.data_ptr<int64_t>(),
        effective_node_cap.data_ptr<scalar_t>(),
        lout.data_ptr<scalar_t>(),
        lin.data_ptr<scalar_t>(),
        arrival_in.data_ptr<scalar_t>(),
        arrival_out.data_ptr<scalar_t>(),
        slew_in.data_ptr<scalar_t>(),
        slew_out.data_ptr<scalar_t>(),
        sink_arrival.data_ptr<scalar_t>(),
        sink_slew.data_ptr<scalar_t>(),
        sink_load.data_ptr<scalar_t>(),
        sink_cap.data_ptr<scalar_t>(),
        sink_net_delay.data_ptr<scalar_t>(),
        sink_net_impulse.data_ptr<scalar_t>(),
        static_cast<int32_t>(net_count),
        static_cast<int32_t>(node_count),
        static_cast<int32_t>(candidate_count),
        static_cast<int32_t>(sink_count),
        stream);
  });
  py::dict result;
  result["candidate_index_by_node"] = candidate_index_by_node;
  result["effective_node_cap"] = effective_node_cap;
  result["lout"] = lout;
  result["lin"] = lin;
  result["arrival_in"] = arrival_in;
  result["arrival_out"] = arrival_out;
  result["slew_in"] = slew_in;
  result["slew_out"] = slew_out;
  result["sink_arrival"] = sink_arrival;
  result["sink_slew"] = sink_slew;
  result["sink_load"] = sink_load;
  result["sink_cap"] = sink_cap;
  result["sink_net_delay"] = sink_net_delay;
  result["sink_net_impulse"] = sink_net_impulse;
  return result;
}

py::dict candidate_net_subgraph_fixed_bsu_forward_cuda(
    at::Tensor net_flat_topo_sort,
    at::Tensor net_flat_topo_sort_start,
    at::Tensor pin_fa,
    at::Tensor flat_pin_to_start,
    at::Tensor flat_pin_to,
    at::Tensor edge_resistance,
    at::Tensor node_capacitance,
    at::Tensor edge_capacitance,
    at::Tensor driver_arrival,
    at::Tensor driver_slew,
    at::Tensor candidate_node_id,
    at::Tensor candidate_bu,
    at::Tensor buffer_input_cap,
    at::Tensor probe_buffer_delay,
    at::Tensor probe_buffer_output_slew,
    at::Tensor upstream_retained_cap,
    at::Tensor buffer_slew_axis,
    at::Tensor buffer_load_axis,
    at::Tensor buffer_delay_lut,
    at::Tensor buffer_output_slew_lut,
    int64_t fixed_bsu_index,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index) {
  const auto scalar_type = node_capacitance.scalar_type();
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&net_flat_topo_sort, "net_flat_topo_sort"},
           {&net_flat_topo_sort_start, "net_flat_topo_sort_start"},
           {&pin_fa, "pin_fa"},
           {&flat_pin_to_start, "flat_pin_to_start"},
           {&flat_pin_to, "flat_pin_to"},
           {&candidate_node_id, "candidate_node_id"},
           {&sink_node_id, "sink_node_id"},
           {&sink_net_index, "sink_net_index"}}) {
    check_int64_cuda_1d(*entry.first, entry.second);
  }
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&edge_resistance, "edge_resistance"},
           {&node_capacitance, "node_capacitance"},
           {&edge_capacitance, "edge_capacitance"},
           {&driver_arrival, "driver_arrival"},
           {&driver_slew, "driver_slew"},
           {&candidate_bu, "candidate_bu"},
           {&buffer_input_cap, "buffer_input_cap"},
           {&probe_buffer_delay, "probe_buffer_delay"},
           {&probe_buffer_output_slew, "probe_buffer_output_slew"},
           {&upstream_retained_cap, "upstream_retained_cap"},
           {&buffer_slew_axis, "buffer_slew_axis"},
           {&buffer_load_axis, "buffer_load_axis"}}) {
    check_float_cuda_1d_like(*entry.first, entry.second, scalar_type);
  }
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&buffer_delay_lut, "buffer_delay_lut"},
           {&buffer_output_slew_lut, "buffer_output_slew_lut"}}) {
    TORCH_CHECK(entry.first->is_cuda(), entry.second, " must be a CUDA tensor");
    TORCH_CHECK(entry.first->is_contiguous(), entry.second, " must be contiguous");
    TORCH_CHECK(entry.first->dim() == 3, entry.second, " must have shape [B, Ts, Tl]");
    TORCH_CHECK(entry.first->scalar_type() == scalar_type, entry.second, " dtype mismatch");
  }
  const int64_t node_count = node_capacitance.numel();
  const int64_t net_count = net_flat_topo_sort_start.numel() - 1;
  const int64_t candidate_count = candidate_node_id.numel();
  const int64_t sink_count = sink_node_id.numel();
  const int64_t size_count = buffer_delay_lut.size(0);
  const int64_t slew_count = buffer_delay_lut.size(1);
  const int64_t load_count = buffer_delay_lut.size(2);
  TORCH_CHECK(size_count > 0 && slew_count > 0 && load_count > 0, "LUT axes must be non-empty");
  TORCH_CHECK(
      buffer_output_slew_lut.sizes() == buffer_delay_lut.sizes(),
      "buffer output-slew LUT shape mismatch");
  TORCH_CHECK(buffer_slew_axis.numel() == slew_count, "buffer slew axis size mismatch");
  TORCH_CHECK(buffer_load_axis.numel() == load_count, "buffer load axis size mismatch");
  TORCH_CHECK(
      fixed_bsu_index >= 0 && fixed_bsu_index < size_count,
      "fixed_bsu_index is out of range");
  TORCH_CHECK(pin_fa.numel() == node_count, "pin_fa size mismatch");
  TORCH_CHECK(flat_pin_to_start.numel() == node_count + 1, "flat_pin_to_start size mismatch");
  TORCH_CHECK(edge_resistance.numel() == node_count, "edge_resistance size mismatch");
  TORCH_CHECK(edge_capacitance.numel() == node_count, "edge_capacitance size mismatch");
  TORCH_CHECK(driver_arrival.numel() == net_count, "driver_arrival size mismatch");
  TORCH_CHECK(driver_slew.numel() == net_count, "driver_slew size mismatch");
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&candidate_bu, "candidate_bu"},
           {&buffer_input_cap, "buffer_input_cap"},
           {&probe_buffer_delay, "probe_buffer_delay"},
           {&probe_buffer_output_slew, "probe_buffer_output_slew"},
           {&upstream_retained_cap, "upstream_retained_cap"}}) {
    TORCH_CHECK(entry.first->numel() == candidate_count, entry.second, " size mismatch");
  }
  TORCH_CHECK(sink_net_index.numel() == sink_count, "sink_net_index size mismatch");
  check_same_cuda_device(
      node_capacitance,
      {{&net_flat_topo_sort, "net_flat_topo_sort"},
       {&net_flat_topo_sort_start, "net_flat_topo_sort_start"},
       {&pin_fa, "pin_fa"},
       {&flat_pin_to_start, "flat_pin_to_start"},
       {&flat_pin_to, "flat_pin_to"},
       {&edge_resistance, "edge_resistance"},
       {&edge_capacitance, "edge_capacitance"},
       {&driver_arrival, "driver_arrival"},
       {&driver_slew, "driver_slew"},
       {&candidate_node_id, "candidate_node_id"},
       {&candidate_bu, "candidate_bu"},
       {&buffer_input_cap, "buffer_input_cap"},
       {&probe_buffer_delay, "probe_buffer_delay"},
       {&probe_buffer_output_slew, "probe_buffer_output_slew"},
       {&upstream_retained_cap, "upstream_retained_cap"},
       {&buffer_slew_axis, "buffer_slew_axis"},
       {&buffer_load_axis, "buffer_load_axis"},
       {&buffer_delay_lut, "buffer_delay_lut"},
       {&buffer_output_slew_lut, "buffer_output_slew_lut"},
       {&sink_node_id, "sink_node_id"},
       {&sink_net_index, "sink_net_index"}});
  const c10::cuda::CUDAGuard device_guard(node_capacitance.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(node_capacitance.get_device()).stream();

  auto index_options = node_capacitance.options().dtype(at::kLong);
  auto candidate_index_by_node = at::full({node_count}, -1, index_options);
  auto effective_node_cap = at::zeros_like(node_capacitance);
  auto lout = at::zeros_like(node_capacitance);
  auto lin = at::zeros_like(node_capacitance);
  auto probe_arrival_in = at::zeros_like(node_capacitance);
  auto probe_arrival_out = at::zeros_like(node_capacitance);
  auto probe_slew_in = at::zeros_like(node_capacitance);
  auto probe_slew_out = at::zeros_like(node_capacitance);
  auto candidate_input_slew = at::zeros_like(candidate_bu);
  auto candidate_output_load = at::zeros_like(candidate_bu);
  auto buffer_delay = at::zeros_like(candidate_bu);
  auto buffer_output_slew = at::zeros_like(candidate_bu);
  auto buffer_delay_grad_slew = at::zeros_like(candidate_bu);
  auto buffer_delay_grad_load = at::zeros_like(candidate_bu);
  auto buffer_output_slew_grad_slew = at::zeros_like(candidate_bu);
  auto buffer_output_slew_grad_load = at::zeros_like(candidate_bu);
  auto arrival_in = at::zeros_like(node_capacitance);
  auto arrival_out = at::zeros_like(node_capacitance);
  auto slew_in = at::zeros_like(node_capacitance);
  auto slew_out = at::zeros_like(node_capacitance);
  auto sink_arrival = at::zeros({sink_count}, node_capacitance.options());
  auto sink_slew = at::zeros({sink_count}, node_capacitance.options());
  auto sink_load = at::zeros({sink_count}, node_capacitance.options());
  auto sink_cap = at::zeros({sink_count}, node_capacitance.options());
  auto sink_net_delay = at::zeros({sink_count}, node_capacitance.options());
  auto sink_net_impulse = at::zeros({sink_count}, node_capacitance.options());

  AT_DISPATCH_FLOATING_TYPES(scalar_type, "candidateNetSubgraphFixedBsuForwardCuda", [&] {
    candidateNetSubgraphFixedBsuForwardCudaLauncher<scalar_t>(
        net_flat_topo_sort.data_ptr<int64_t>(),
        net_flat_topo_sort_start.data_ptr<int64_t>(),
        pin_fa.data_ptr<int64_t>(),
        flat_pin_to_start.data_ptr<int64_t>(),
        flat_pin_to.data_ptr<int64_t>(),
        edge_resistance.data_ptr<scalar_t>(),
        node_capacitance.data_ptr<scalar_t>(),
        edge_capacitance.data_ptr<scalar_t>(),
        driver_arrival.data_ptr<scalar_t>(),
        driver_slew.data_ptr<scalar_t>(),
        candidate_node_id.data_ptr<int64_t>(),
        candidate_bu.data_ptr<scalar_t>(),
        buffer_input_cap.data_ptr<scalar_t>(),
        probe_buffer_delay.data_ptr<scalar_t>(),
        probe_buffer_output_slew.data_ptr<scalar_t>(),
        upstream_retained_cap.data_ptr<scalar_t>(),
        buffer_slew_axis.data_ptr<scalar_t>(),
        buffer_load_axis.data_ptr<scalar_t>(),
        buffer_delay_lut.data_ptr<scalar_t>(),
        buffer_output_slew_lut.data_ptr<scalar_t>(),
        static_cast<int32_t>(fixed_bsu_index),
        static_cast<int32_t>(size_count),
        static_cast<int32_t>(slew_count),
        static_cast<int32_t>(load_count),
        sink_node_id.data_ptr<int64_t>(),
        sink_net_index.data_ptr<int64_t>(),
        candidate_index_by_node.data_ptr<int64_t>(),
        effective_node_cap.data_ptr<scalar_t>(),
        lout.data_ptr<scalar_t>(),
        lin.data_ptr<scalar_t>(),
        probe_arrival_in.data_ptr<scalar_t>(),
        probe_arrival_out.data_ptr<scalar_t>(),
        probe_slew_in.data_ptr<scalar_t>(),
        probe_slew_out.data_ptr<scalar_t>(),
        candidate_input_slew.data_ptr<scalar_t>(),
        candidate_output_load.data_ptr<scalar_t>(),
        buffer_delay.data_ptr<scalar_t>(),
        buffer_output_slew.data_ptr<scalar_t>(),
        buffer_delay_grad_slew.data_ptr<scalar_t>(),
        buffer_delay_grad_load.data_ptr<scalar_t>(),
        buffer_output_slew_grad_slew.data_ptr<scalar_t>(),
        buffer_output_slew_grad_load.data_ptr<scalar_t>(),
        arrival_in.data_ptr<scalar_t>(),
        arrival_out.data_ptr<scalar_t>(),
        slew_in.data_ptr<scalar_t>(),
        slew_out.data_ptr<scalar_t>(),
        sink_arrival.data_ptr<scalar_t>(),
        sink_slew.data_ptr<scalar_t>(),
        sink_load.data_ptr<scalar_t>(),
        sink_cap.data_ptr<scalar_t>(),
        sink_net_delay.data_ptr<scalar_t>(),
        sink_net_impulse.data_ptr<scalar_t>(),
        static_cast<int32_t>(net_count),
        static_cast<int32_t>(node_count),
        static_cast<int32_t>(candidate_count),
        static_cast<int32_t>(sink_count),
        stream);
  });
  py::dict result;
  result["candidate_index_by_node"] = candidate_index_by_node;
  result["effective_node_cap"] = effective_node_cap;
  result["lout"] = lout;
  result["lin"] = lin;
  result["probe_arrival_in"] = probe_arrival_in;
  result["probe_arrival_out"] = probe_arrival_out;
  result["probe_slew_in"] = probe_slew_in;
  result["probe_slew_out"] = probe_slew_out;
  result["candidate_input_slew"] = candidate_input_slew;
  result["candidate_output_load"] = candidate_output_load;
  result["buffer_delay"] = buffer_delay;
  result["buffer_output_slew"] = buffer_output_slew;
  result["buffer_delay_grad_slew"] = buffer_delay_grad_slew;
  result["buffer_delay_grad_load"] = buffer_delay_grad_load;
  result["buffer_output_slew_grad_slew"] = buffer_output_slew_grad_slew;
  result["buffer_output_slew_grad_load"] = buffer_output_slew_grad_load;
  result["arrival_in"] = arrival_in;
  result["arrival_out"] = arrival_out;
  result["slew_in"] = slew_in;
  result["slew_out"] = slew_out;
  result["sink_arrival"] = sink_arrival;
  result["sink_slew"] = sink_slew;
  result["sink_load"] = sink_load;
  result["sink_cap"] = sink_cap;
  result["sink_net_delay"] = sink_net_delay;
  result["sink_net_impulse"] = sink_net_impulse;
  return result;
}

py::dict candidate_net_subgraph_backward_cuda(
    at::Tensor net_flat_topo_sort,
    at::Tensor net_flat_topo_sort_start,
    at::Tensor pin_fa,
    at::Tensor flat_pin_to_start,
    at::Tensor flat_pin_to,
    at::Tensor edge_resistance,
    at::Tensor candidate_index_by_node,
    at::Tensor candidate_bu,
    at::Tensor buffer_input_cap,
    at::Tensor buffer_delay,
    at::Tensor buffer_output_slew,
    at::Tensor sink_node_id,
    at::Tensor sink_net_index,
    at::Tensor lin,
    at::Tensor lout,
    at::Tensor slew_in,
    at::Tensor slew_out,
    at::Tensor grad_lout_input,
    at::Tensor grad_lin_input,
    at::Tensor grad_arrival_in_input,
    at::Tensor grad_arrival_out_input,
    at::Tensor grad_slew_in_input,
    at::Tensor grad_slew_out_input,
    at::Tensor grad_sink_arrival,
    at::Tensor grad_sink_slew,
    at::Tensor grad_sink_load,
    at::Tensor grad_sink_net_delay,
    at::Tensor grad_sink_net_impulse) {
  const auto scalar_type = lin.scalar_type();
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&net_flat_topo_sort, "net_flat_topo_sort"},
           {&net_flat_topo_sort_start, "net_flat_topo_sort_start"},
           {&pin_fa, "pin_fa"},
           {&flat_pin_to_start, "flat_pin_to_start"},
           {&flat_pin_to, "flat_pin_to"},
           {&candidate_index_by_node, "candidate_index_by_node"},
           {&sink_node_id, "sink_node_id"},
           {&sink_net_index, "sink_net_index"}}) {
    check_int64_cuda_1d(*entry.first, entry.second);
  }
  for (const auto& entry : {
           std::pair<const at::Tensor*, const char*>{&edge_resistance, "edge_resistance"},
           {&candidate_bu, "candidate_bu"},
           {&buffer_input_cap, "buffer_input_cap"},
           {&buffer_delay, "buffer_delay"},
           {&buffer_output_slew, "buffer_output_slew"},
           {&lin, "lin"},
           {&lout, "lout"},
           {&slew_in, "slew_in"},
           {&slew_out, "slew_out"},
           {&grad_lout_input, "grad_lout"},
           {&grad_lin_input, "grad_lin"},
           {&grad_arrival_in_input, "grad_arrival_in"},
           {&grad_arrival_out_input, "grad_arrival_out"},
           {&grad_slew_in_input, "grad_slew_in"},
           {&grad_slew_out_input, "grad_slew_out"},
           {&grad_sink_arrival, "grad_sink_arrival"},
           {&grad_sink_slew, "grad_sink_slew"},
           {&grad_sink_load, "grad_sink_load"},
           {&grad_sink_net_delay, "grad_sink_net_delay"},
           {&grad_sink_net_impulse, "grad_sink_net_impulse"}}) {
    check_float_cuda_1d_like(*entry.first, entry.second, scalar_type);
  }
  check_same_cuda_device(
      lin,
      {{&net_flat_topo_sort, "net_flat_topo_sort"},
       {&net_flat_topo_sort_start, "net_flat_topo_sort_start"},
       {&pin_fa, "pin_fa"},
       {&flat_pin_to_start, "flat_pin_to_start"},
       {&flat_pin_to, "flat_pin_to"},
       {&edge_resistance, "edge_resistance"},
       {&candidate_index_by_node, "candidate_index_by_node"},
       {&candidate_bu, "candidate_bu"},
       {&buffer_input_cap, "buffer_input_cap"},
       {&buffer_delay, "buffer_delay"},
       {&buffer_output_slew, "buffer_output_slew"},
       {&sink_node_id, "sink_node_id"},
       {&sink_net_index, "sink_net_index"},
       {&lout, "lout"},
       {&slew_in, "slew_in"},
       {&slew_out, "slew_out"},
       {&grad_lout_input, "grad_lout"},
       {&grad_lin_input, "grad_lin"},
       {&grad_arrival_in_input, "grad_arrival_in"},
       {&grad_arrival_out_input, "grad_arrival_out"},
       {&grad_slew_in_input, "grad_slew_in"},
       {&grad_slew_out_input, "grad_slew_out"},
       {&grad_sink_arrival, "grad_sink_arrival"},
       {&grad_sink_slew, "grad_sink_slew"},
       {&grad_sink_load, "grad_sink_load"},
       {&grad_sink_net_delay, "grad_sink_net_delay"},
       {&grad_sink_net_impulse, "grad_sink_net_impulse"}});
  const c10::cuda::CUDAGuard device_guard(lin.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(lin.get_device()).stream();
  auto grad_lout = grad_lout_input.clone();
  auto grad_lin = grad_lin_input.clone();
  auto grad_arrival_in = grad_arrival_in_input.clone();
  auto grad_arrival_out = grad_arrival_out_input.clone();
  auto grad_slew_in = grad_slew_in_input.clone();
  auto grad_slew_out = grad_slew_out_input.clone();
  auto grad_driver_arrival = at::zeros(
      {net_flat_topo_sort_start.numel() - 1}, lin.options());
  auto grad_driver_slew = at::zeros_like(grad_driver_arrival);
  auto grad_candidate_bu = at::zeros_like(candidate_bu);
  auto grad_buffer_input_cap = at::zeros_like(buffer_input_cap);
  auto grad_buffer_delay = at::zeros_like(buffer_delay);
  auto grad_buffer_output_slew = at::zeros_like(buffer_output_slew);
  const int32_t net_count = static_cast<int32_t>(net_flat_topo_sort_start.numel() - 1);
  const int32_t sink_count = static_cast<int32_t>(sink_node_id.numel());
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "candidateNetSubgraphBackwardCuda", [&] {
    candidateNetSubgraphBackwardCudaLauncher<scalar_t>(
        net_flat_topo_sort.data_ptr<int64_t>(),
        net_flat_topo_sort_start.data_ptr<int64_t>(),
        pin_fa.data_ptr<int64_t>(),
        flat_pin_to_start.data_ptr<int64_t>(),
        flat_pin_to.data_ptr<int64_t>(),
        edge_resistance.data_ptr<scalar_t>(),
        candidate_index_by_node.data_ptr<int64_t>(),
        candidate_bu.data_ptr<scalar_t>(),
        buffer_input_cap.data_ptr<scalar_t>(),
        buffer_delay.data_ptr<scalar_t>(),
        buffer_output_slew.data_ptr<scalar_t>(),
        sink_node_id.data_ptr<int64_t>(),
        sink_net_index.data_ptr<int64_t>(),
        lin.data_ptr<scalar_t>(),
        lout.data_ptr<scalar_t>(),
        slew_in.data_ptr<scalar_t>(),
        slew_out.data_ptr<scalar_t>(),
        grad_sink_arrival.data_ptr<scalar_t>(),
        grad_sink_slew.data_ptr<scalar_t>(),
        grad_sink_load.data_ptr<scalar_t>(),
        grad_sink_net_delay.data_ptr<scalar_t>(),
        grad_sink_net_impulse.data_ptr<scalar_t>(),
        grad_lout.data_ptr<scalar_t>(),
        grad_lin.data_ptr<scalar_t>(),
        grad_arrival_in.data_ptr<scalar_t>(),
        grad_arrival_out.data_ptr<scalar_t>(),
        grad_slew_in.data_ptr<scalar_t>(),
        grad_slew_out.data_ptr<scalar_t>(),
        grad_driver_arrival.data_ptr<scalar_t>(),
        grad_driver_slew.data_ptr<scalar_t>(),
        grad_candidate_bu.data_ptr<scalar_t>(),
        grad_buffer_input_cap.data_ptr<scalar_t>(),
        grad_buffer_delay.data_ptr<scalar_t>(),
        grad_buffer_output_slew.data_ptr<scalar_t>(),
        net_count,
        sink_count,
        stream);
  });
  py::dict result;
  result["grad_driver_arrival"] = grad_driver_arrival;
  result["grad_driver_slew"] = grad_driver_slew;
  result["grad_candidate_bu"] = grad_candidate_bu;
  result["grad_buffer_input_cap"] = grad_buffer_input_cap;
  result["grad_buffer_delay"] = grad_buffer_delay;
  result["grad_buffer_output_slew"] = grad_buffer_output_slew;
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA segment transfer forward operators";
  m.def("segment_transfer_forward", &segment_transfer_forward_cuda, "Batched segment transfer forward (CUDA)");
  m.def(
      "segment_count_transfer_forward",
      &segment_count_transfer_forward_cuda,
      "Segment-count relaxed timing forward with LUT segment transfer (CUDA)");
  m.def(
      "segment_count_cap_forward",
      &segment_count_cap_forward_cuda,
      "Segment-count root-load forward without timing transfer (CUDA)");
  m.def(
      "segment_count_transfer_backward",
      &segment_count_transfer_backward_cuda,
      "Segment-count relaxed timing backward with LUT segment transfer (CUDA)");
  m.def(
      "segment_count_cap_backward",
      &segment_count_cap_backward_cuda,
      "Segment-count root-load backward without timing transfer (CUDA)");
  m.def(
      "candidate_net_subgraph_forward",
      &candidate_net_subgraph_forward_cuda,
      "Candidate buffer-aware net subgraph forward (CUDA)");
  m.def(
      "candidate_net_subgraph_fixed_bsu_forward",
      &candidate_net_subgraph_fixed_bsu_forward_cuda,
      "Candidate fixed-BSu Liberty-aware fused net subgraph forward (CUDA)");
  m.def(
      "candidate_net_subgraph_backward",
      &candidate_net_subgraph_backward_cuda,
      "Candidate buffer-aware net subgraph backward (CUDA)");
}
