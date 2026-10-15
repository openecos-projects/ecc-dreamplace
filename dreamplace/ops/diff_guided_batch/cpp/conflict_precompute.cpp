#include "conflict_precompute.h"

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace dreamplace {
namespace diff_guided_batch {

ConflictPrecompute buildConflictPrecomputeFromComponentIds(
    const std::vector<int32_t>& component_ids,
    std::string input_fingerprint) {
  const auto begin = std::chrono::steady_clock::now();
  ConflictPrecompute result;
  result.component_ids = component_ids;
  result.input_fingerprint = std::move(input_fingerprint);

  std::unordered_map<int32_t, int32_t> component_sizes;
  for (int32_t component_id : component_ids) {
    if (component_id < 0) {
      continue;
    }
    component_sizes[component_id] += 1;
  }
  result.component_count = static_cast<int32_t>(component_sizes.size());
  for (const auto& entry : component_sizes) {
    result.max_component_size = std::max(result.max_component_size, entry.second);
  }

  result.csr_indptr.assign(component_ids.size() + 1, 0);
  result.csr_indices.clear();
  result.csr_edge_count = 0;
  result.bitset_block_count = 0;

  const auto end = std::chrono::steady_clock::now();
  result.build_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
