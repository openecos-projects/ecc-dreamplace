// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_carrier_projection.hh"

#include <algorithm>
#include <stdexcept>

namespace rebuffer {

void normalizePartitionProvenance(PartitionProvenance& rows)
{
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  for (std::size_t index = 1; index < rows.size(); ++index) {
    if (rows[index - 1].first == rows[index].first
        && rows[index - 1].second != rows[index].second) {
      throw std::invalid_argument("duplicate partition vertex provenance");
    }
  }
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

PartitionProvenance::const_iterator findPartitionProvenance(
    const PartitionProvenance& rows, const std::pair<int, int>& key)
{
  const auto found = std::lower_bound(rows.begin(), rows.end(), key,
      [](const auto& row, const auto& value) { return row.first < value; });
  return found != rows.end() && found->first == key ? found : rows.end();
}

}  // namespace rebuffer
