#pragma once

#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

DREAMPLACE_BEGIN_NAMESPACE

// Local records are shared by the FLUTE builder and retained buffered trees.
struct NetResult {
  int num_steiner = 0;
  int netid = 0;
  std::vector<int> newx, newy;
  std::vector<int> vtx_relate_x, vtx_relate_y, vtx_fa;
  std::vector<int> net_flat_topo_idx, local2global_idx;
  std::string error;
};

inline void checkFrozenCache(const std::vector<at::Tensor>& cache,
                            int num_nets, int num_pins) {
  TORCH_CHECK(cache.size() == 12, "frozen topology requires the previous tree cache");
  const int64_t vertices = cache[0].numel();
  TORCH_CHECK(vertices >= num_pins, "frozen cache has fewer vertices than pins");
  for (const int field : {2, 3, 5, 6, 10, 11}) {
    const auto& tensor = cache[field];
    TORCH_CHECK(tensor.device().is_cpu() && tensor.dim() == 1 &&
                    tensor.is_contiguous() && tensor.scalar_type() == at::kInt,
                "frozen cache indices must be flat contiguous CPU int32 tensors");
    TORCH_CHECK(tensor.numel() == ((field == 5 || field == 11) ? num_nets + 1 : vertices),
                "frozen cache index length does not match its domain");
  }
  const int* starts = cache[5].data_ptr<int>();
  const int* topo_starts = cache[11].data_ptr<int>();
  TORCH_CHECK(starts[0] == num_pins && starts[num_nets] == vertices &&
                  topo_starts[0] == 0 && topo_starts[num_nets] == vertices,
              "frozen cache has inconsistent vertex ranges");
  for (int net = 0; net < num_nets; ++net) {
    TORCH_CHECK(starts[net] <= starts[net + 1] &&
                    topo_starts[net] <= topo_starts[net + 1],
                "frozen cache ranges must be monotonic");
  }
}

template <typename T>
std::vector<NetResult> restoreFrozenNets(
    const std::vector<int>& frozen_ids, const std::vector<at::Tensor>& cache,
    const T* x, const T* y, const int* pins, const int* pin_starts,
    int num_nets, int num_pins) {
  std::vector<NetResult> records(num_nets);
  if (frozen_ids.empty()) return records;
  checkFrozenCache(cache, num_nets, num_pins);
  const int* steiner_starts = cache[5].data_ptr<int>();
  const int* parents = cache[6].data_ptr<int>();
  const int* topo = cache[10].data_ptr<int>();
  const int* topo_starts = cache[11].data_ptr<int>();
  const int* relate_x = cache[2].data_ptr<int>();
  const int* relate_y = cache[3].data_ptr<int>();
  std::vector<bool> seen(num_nets, false);
  for (const int net : frozen_ids) {
    TORCH_CHECK(net >= 0 && net < num_nets && !seen[net],
                "frozen net IDs must be unique and within the current net domain");
    seen[net] = true;
    auto& record = records[net];
    const int degree = pin_starts[net + 1] - pin_starts[net];
    record.netid = net;
    record.num_steiner = steiner_starts[net + 1] - steiner_starts[net];
    const int vertices = degree + record.num_steiner;
    TORCH_CHECK(topo_starts[net + 1] - topo_starts[net] == vertices,
                "frozen net pin/Steiner ranges do not match its topology");
    std::unordered_map<int, int> local;
    record.local2global_idx.assign(pins + pin_starts[net], pins + pin_starts[net + 1]);
    for (int i = 0; i < degree; ++i) local.emplace(record.local2global_idx[i], i);
    for (int i = 0; i < record.num_steiner; ++i)
      local.emplace(steiner_starts[net] + i, degree + i);
    auto local_id = [&](int global) {
      const auto found = local.find(global);
      TORCH_CHECK(found != local.end(), "frozen tree references a vertex from another net");
      return found->second;
    };
    record.newx.resize(vertices);
    record.newy.resize(vertices);
    record.vtx_relate_x.resize(vertices);
    record.vtx_relate_y.resize(vertices);
    record.vtx_fa.resize(vertices);
    record.net_flat_topo_idx.resize(vertices);
    std::vector<bool> visited(vertices, false);
    for (int order = 0; order < vertices; ++order) {
      const int global = topo[topo_starts[net] + order];
      const int index = local_id(global);
      TORCH_CHECK(!visited[index], "frozen tree repeats a vertex");
      const int parent = parents[global];
      const int parent_local = parent == -1 ? -1 : local_id(parent);
      TORCH_CHECK(order == 0 ? (parent_local == -1 && global == record.local2global_idx[0])
                              : (parent_local >= 0 && visited[parent_local]),
                  "frozen tree must be rooted at its driver and parent ordered");
      visited[index] = true;
      record.vtx_fa[index] = parent_local;
      record.net_flat_topo_idx[order] = index;
      for (int axis = 0; axis < 2; ++axis) {
        const int witness = axis == 0 ? relate_x[global] : relate_y[global];
        const int witness_local = local_id(witness);
        TORCH_CHECK(witness_local < degree, "frozen coordinate witness must be a real pin");
        const double coordinate = double(axis == 0 ? x[witness] : y[witness]) * 1000;
        TORCH_CHECK(std::isfinite(coordinate) &&
                        coordinate >= std::numeric_limits<int>::min() &&
                        coordinate <= std::numeric_limits<int>::max(),
                    "frozen tree received a non-finite or out-of-range coordinate");
        (axis == 0 ? record.newx : record.newy)[index] = int(std::llround(coordinate));
        (axis == 0 ? record.vtx_relate_x : record.vtx_relate_y)[index] = witness_local;
      }
    }
  }
  return records;
}

inline at::Tensor frozenVertexMap(const std::vector<int>& frozen_ids,
                                 const std::vector<at::Tensor>& old_cache,
                                 const std::vector<at::Tensor>& new_cache,
                                 int num_pins) {
  auto mapping = at::full({old_cache[0].numel()}, -1, old_cache[5].options());
  int* out = mapping.data_ptr<int>();
  for (int pin = 0; pin < num_pins; ++pin) out[pin] = pin;
  const int* before = old_cache[5].data_ptr<int>();
  const int* after = new_cache[5].data_ptr<int>();
  for (const int net : frozen_ids)
    for (int vertex = before[net]; vertex < before[net + 1]; ++vertex)
      out[vertex] = after[net] + vertex - before[net];
  return mapping;
}

DREAMPLACE_END_NAMESPACE
