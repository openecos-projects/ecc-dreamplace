#pragma once

#include <string>
#include <vector>

namespace dreamplace::placeio_openroad::impl::sizing_metadata {
struct CellClass
{
  std::string drive_key;
  std::string vt_key;
};

inline bool endsWith(const std::string& value, const std::string& suffix)
{
  return value.size() >= suffix.size()
         && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline CellClass classifyCell(const std::string& cell_name,
                              const std::vector<std::string>& configured_vt_suffixes)
{
  for (const std::string& suffix : configured_vt_suffixes) {
    if (!suffix.empty() && endsWith(cell_name, suffix)) {
      return {cell_name.substr(0, cell_name.size() - suffix.size()), suffix};
    }
  }

  return {cell_name, "default"};
}

}  // namespace dreamplace::placeio_openroad::impl::sizing_metadata
