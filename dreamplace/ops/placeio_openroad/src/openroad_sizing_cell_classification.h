#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openroad_place_io_impl_shared.h"
#include "pdk_cell_classification.h"

namespace dreamplace {
namespace placeio_openroad {
namespace impl {
namespace sizing_metadata {

inline double groupLeakageRank(const std::vector<sta::LibertyCell*>& cells)
{
  double rank = std::numeric_limits<double>::infinity();
  for (sta::LibertyCell* cell : cells) {
    const double leakage = exportLibcellLeakageForPython(cell);
    if (std::isfinite(leakage) && leakage > 0.0) {
      rank = std::min(rank, leakage);
    }
  }
  return rank;
}

inline std::unordered_map<std::string, int> orderedAxis(
    const std::unordered_map<std::string, std::vector<sta::LibertyCell*>>& groups)
{
  std::vector<std::pair<std::string, double>> ranked;
  ranked.reserve(groups.size());
  for (const auto& [key, cells] : groups) {
    ranked.emplace_back(key, groupLeakageRank(cells));
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second < rhs.second;
    }
    return lhs.first < rhs.first;
  });

  std::unordered_map<std::string, int> order;
  for (int index = 0; index < static_cast<int>(ranked.size()); ++index) {
    order.emplace(ranked[index].first, index);
  }
  return order;
}

}  // namespace sizing_metadata
}  // namespace impl
}  // namespace placeio_openroad
}  // namespace dreamplace
