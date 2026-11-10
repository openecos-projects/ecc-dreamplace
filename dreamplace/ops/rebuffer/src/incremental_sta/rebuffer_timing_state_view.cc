// SPDX-License-Identifier: BSD-3-Clause
// Read-only zero-copy snapshots retain their storage across graph mutations.
#include "rebuffer_timing_graph.hh"

#include <utility>

namespace py = pybind11;
namespace rebuffer {

struct TimingStateViewStorage
{
  // The graph owns current storage until the next write. On detach, retained
  // takes that exact allocation and the graph reference is released.
  std::shared_ptr<const PersistentTimingGraph> graph;
  std::vector<double> retained;
  double empty = 0;
};

py::dict PersistentTimingGraph::stateView(bool copy) const
{
  py::dict result;
  result["status"] = "ok";
  result["epoch"] = epoch_;
  for (const auto& [name, value] : state_) {
    if (copy) {
      result[py::str(name)] = writeState(value);
      continue;
    }
    auto& cached = state_views_[name];
    auto storage = cached.lock();
    if (!storage) {
      storage = std::make_shared<TimingStateViewStorage>();
      storage->graph = shared_from_this();
      cached = storage;
    }
    auto* holder = new std::shared_ptr<TimingStateViewStorage>(storage);
    py::capsule base(holder, [](void* pointer) {
      delete static_cast<std::shared_ptr<TimingStateViewStorage>*>(pointer);
    });
    const double* data = value.empty() ? &storage->empty : value.data();
    py::array_t<double> array({static_cast<py::ssize_t>(value.size())},
                              {static_cast<py::ssize_t>(sizeof(double))}, data, base);
    array.attr("setflags")(py::arg("write") = false);
    result[py::str(name)] = std::move(array);
    ++state_view_counters_.exports;
  }
  return result;
}

void PersistentTimingGraph::detachStateViews()
{
  cached_endpoint_slacks_valid_ = false;
  if (state_views_.empty()) return;
  // Ownership transfer below may release references from every view holder.
  // Keep this graph alive until the complete transfer has finished.
  const auto owner = shared_from_this();
  for (auto it = state_views_.begin(); it != state_views_.end();) {
    if (auto storage = it->second.lock()) {
      auto& current = state_.at(it->first);
      auto replacement = current;  // Allocate before changing either owner.
      storage->retained = std::move(current);
      current = std::move(replacement);
      ++state_view_counters_.detached_fields;
      state_view_counters_.detached_values += storage->retained.size();
      storage->graph.reset();
    }
    it = state_views_.erase(it);
  }
}

py::dict PersistentTimingGraph::stateViewStatistics() const
{
  py::dict result;
  result["exports"] = state_view_counters_.exports;
  result["detached_fields"] = state_view_counters_.detached_fields;
  result["detached_values"] = state_view_counters_.detached_values;
  std::size_t current = 0;
  for (const auto& [_, storage] : state_views_) if (!storage.expired()) ++current;
  result["current_view_fields"] = current;
  return result;
}

}  // namespace rebuffer
