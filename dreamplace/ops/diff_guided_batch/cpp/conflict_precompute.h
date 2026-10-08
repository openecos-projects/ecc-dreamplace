#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dreamplace {
namespace diff_guided_batch {

struct ConflictPrecompute {
  int32_t component_count = 0;
  int32_t max_component_size = 0;
  int64_t csr_edge_count = 0;
  int32_t bitset_block_count = 0;
  double build_ms = 0.0;
  std::string input_fingerprint;
  std::string source;
  int32_t version = 1;
  std::vector<int32_t> component_ids;
  std::vector<int32_t> csr_indptr;
  std::vector<int32_t> csr_indices;
  std::vector<double> local_residual_budget_by_top_path_component;
  std::vector<double> local_residual_budget_by_endpoint_component;
  std::string local_residual_snapshot_source;
};

ConflictPrecompute buildConflictPrecomputeFromComponentIds(
    const std::vector<int32_t>& component_ids,
    std::string input_fingerprint = "");

}  // namespace diff_guided_batch
}  // namespace dreamplace
