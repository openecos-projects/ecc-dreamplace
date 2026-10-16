#include "utility/src/torch.h"
#include "candidate_backward_math.h"
#include <ATen/Parallel.h>
#include <omp.h>

namespace py = pybind11;
namespace {
void checkIndex(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.dim() == 1 && t.is_contiguous() && t.scalar_type() == at::kLong,
              name, " must be a contiguous CPU int64 vector");
}
void checkFloat(const at::Tensor& t, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(t.device().is_cpu() && t.dim() == 1 && t.is_contiguous() && t.scalar_type() == dtype,
              name, " must be a contiguous CPU vector of the timing dtype");
}
}

py::dict candidate_net_subgraph_backward_cpu(
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
    checkIndex(*entry.first, entry.second);
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
    checkFloat(*entry.first, entry.second, scalar_type);
  }
  const auto nodes = lin.numel(), candidates = candidate_bu.numel();
  TORCH_CHECK(net_flat_topo_sort_start.numel() >= 1 && pin_fa.numel() == nodes &&
              flat_pin_to_start.numel() == nodes + 1 && candidate_index_by_node.numel() == nodes,
              "candidate backward forest shape mismatch");
  for (const auto& value : {edge_resistance, lout, slew_in, slew_out, grad_lout_input,
                            grad_lin_input, grad_arrival_in_input, grad_arrival_out_input,
                            grad_slew_in_input, grad_slew_out_input}) {
    TORCH_CHECK(value.numel() == nodes, "candidate backward node shape mismatch");
  }
  for (const auto& value : {buffer_input_cap, buffer_delay, buffer_output_slew}) {
    TORCH_CHECK(value.numel() == candidates, "candidate backward buffer shape mismatch");
  }
  for (const auto& value : {sink_net_index, grad_sink_arrival, grad_sink_slew, grad_sink_load,
                            grad_sink_net_delay, grad_sink_net_impulse}) {
    TORCH_CHECK(value.numel() == sink_node_id.numel(), "candidate backward sink shape mismatch");
  }
  const auto nets = net_flat_topo_sort_start.numel() - 1;
  const auto* starts = net_flat_topo_sort_start.data_ptr<int64_t>();
  const auto* topo = net_flat_topo_sort.data_ptr<int64_t>();
  const auto* parents = pin_fa.data_ptr<int64_t>();
  const auto* child_starts = flat_pin_to_start.data_ptr<int64_t>();
  const auto* children = flat_pin_to.data_ptr<int64_t>();
  TORCH_CHECK(starts[0] == 0 && starts[nets] == net_flat_topo_sort.numel() &&
              child_starts[0] == 0 && child_starts[nodes] == flat_pin_to.numel(),
              "candidate backward CSR range mismatch");
  std::vector<int64_t> owner(nodes, -1);
  for (int64_t net = 0; net < nets; ++net) {
    TORCH_CHECK(starts[net] <= starts[net + 1] && starts[net + 1] <= net_flat_topo_sort.numel(), "candidate backward net offsets not ordered");
    for (int64_t row = starts[net]; row < starts[net + 1]; ++row) {
      const auto node = topo[row];
      TORCH_CHECK(node >= 0 && node < nodes && owner[node] == -1,
                  "candidate backward needs disjoint net trees");
      owner[node] = net;
    }
  }
  for (int64_t node = 0; node < nodes; ++node) {
    const auto candidate = candidate_index_by_node.data_ptr<int64_t>()[node];
    TORCH_CHECK(candidate >= -1 && candidate < candidates && parents[node] >= -1 && parents[node] < nodes,
                "candidate backward node index out of range");
    TORCH_CHECK(child_starts[node] <= child_starts[node + 1] && child_starts[node + 1] <= flat_pin_to.numel(), "candidate backward child offsets not ordered");
    if (owner[node] < 0) continue;
    TORCH_CHECK(parents[node] < 0 || owner[parents[node]] == owner[node],
                "candidate backward parent crosses net tree");
    for (int64_t row = child_starts[node]; row < child_starts[node + 1]; ++row) {
      const auto child = children[row];
      TORCH_CHECK(child >= 0 && child < nodes && owner[child] == owner[node],
                  "candidate backward child crosses net tree");
    }
  }
  for (int64_t row = 0; row < sink_node_id.numel(); ++row) {
    const auto node = sink_node_id.data_ptr<int64_t>()[row];
    TORCH_CHECK(node >= 0 && node < nodes, "candidate backward sink node out of range");
  }
  auto grad_lout = grad_lout_input.clone();
  auto grad_lin = grad_lin_input.clone();
  auto grad_arrival_in = grad_arrival_in_input.clone();
  auto grad_arrival_out = grad_arrival_out_input.clone();
  auto grad_slew_in = grad_slew_in_input.clone();
  auto grad_slew_out = grad_slew_out_input.clone();
  // Keep independent tensors; only their clearing is combined by the launcher.
  auto grad_driver_arrival = at::zeros(
      {net_flat_topo_sort_start.numel() - 1}, lin.options());
  auto grad_driver_slew = at::zeros(grad_driver_arrival.sizes(), grad_driver_arrival.options());
  auto grad_candidate_bu = at::zeros(candidate_bu.sizes(), candidate_bu.options());
  auto grad_buffer_input_cap = at::zeros(buffer_input_cap.sizes(), buffer_input_cap.options());
  auto grad_buffer_delay = at::zeros(buffer_delay.sizes(), buffer_delay.options());
  auto grad_buffer_output_slew = at::zeros(buffer_output_slew.sizes(), buffer_output_slew.options());
  const int32_t net_count = static_cast<int32_t>(net_flat_topo_sort_start.numel() - 1);
  const auto compute = [&](auto sample) {
    using scalar_t = decltype(sample);
    const auto* sinks = sink_node_id.data_ptr<int64_t>();
    const auto* nets = sink_net_index.data_ptr<int64_t>();
    const auto* starts = net_flat_topo_sort_start.data_ptr<int64_t>();
    const auto* topo = net_flat_topo_sort.data_ptr<int64_t>();
    auto* ga = grad_arrival_out.data_ptr<scalar_t>(); auto* gs = grad_slew_out.data_ptr<scalar_t>();
    auto* gl = grad_lin.data_ptr<scalar_t>(); const auto* slew = slew_out.data_ptr<scalar_t>();
    for (int64_t i = 0; i < sink_node_id.numel(); ++i) {
      const auto node = sinks[i], net = nets[i];
      ga[node] += grad_sink_arrival.data_ptr<scalar_t>()[i];
      gs[node] += grad_sink_slew.data_ptr<scalar_t>()[i];
      gl[node] += grad_sink_load.data_ptr<scalar_t>()[i];
      const auto delay = grad_sink_net_delay.data_ptr<scalar_t>()[i];
      const auto impulse = grad_sink_net_impulse.data_ptr<scalar_t>()[i];
      ga[node] += delay;
      gs[node] += scalar_t(2) * slew[node] * impulse;
      if (net >= 0 && net < net_count && owner[node] >= 0 && starts[net] != starts[net + 1]) {
        const auto root = topo[starts[net]];
        if (candidate_index_by_node.data_ptr<int64_t>()[root] >= 0) {
          // CPU sink-relative outputs subtract the input driver timing, before
          // any candidate at the root. Preserve that boundary in its adjoint.
          grad_driver_arrival.data_ptr<scalar_t>()[net] -= delay;
          grad_driver_slew.data_ptr<scalar_t>()[net] -=
              scalar_t(2) * slew_in.data_ptr<scalar_t>()[root] * impulse;
        } else {
          ga[root] -= delay; gs[root] -= scalar_t(2) * slew[root] * impulse;
        }
      }
    }
    #pragma omp parallel for if(net_count >= 64) num_threads(at::get_num_threads()) schedule(static)
    for (int32_t net = 0; net < net_count; ++net) {
    candidateBackwardNet<scalar_t>(net,
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
        lin.data_ptr<scalar_t>(),
        lout.data_ptr<scalar_t>(),
        slew_in.data_ptr<scalar_t>(),
        slew_out.data_ptr<scalar_t>(),
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
        net_count);
    }
  };
  AT_DISPATCH_FLOATING_TYPES(scalar_type, "candidateNetSubgraphBackwardCpu", [&] { compute(scalar_t(0)); });
  py::dict result;
  result["grad_driver_arrival"] = grad_driver_arrival;
  result["grad_driver_slew"] = grad_driver_slew;
  result["grad_candidate_bu"] = grad_candidate_bu;
  result["grad_buffer_input_cap"] = grad_buffer_input_cap;
  result["grad_buffer_delay"] = grad_buffer_delay;
  result["grad_buffer_output_slew"] = grad_buffer_output_slew;
  result["grad_lout"] = grad_lout;
  result["grad_arrival_in"] = grad_arrival_in;
  result["grad_slew_in"] = grad_slew_in;
  result["grad_lin"] = grad_lin;
  result["grad_arrival_out"] = grad_arrival_out;
  result["grad_slew_out"] = grad_slew_out;
  return result;
}
