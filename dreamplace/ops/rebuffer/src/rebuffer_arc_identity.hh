// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <vector>

#include "pybind11/pybind11.h"

namespace rebuffer {

struct TimingLibraryArc
{
  std::string source, sink;
  int sense, trigger;
};

inline bool sameTimingSignature(const TimingLibraryArc& a, const TimingLibraryArc& b)
{
  return a.source == b.source && a.sink == b.sink
         && a.sense == b.sense && a.trigger == b.trigger;
}

// Repeated state-dependent arcs retain their occurrence rank across masters.
// Lookup may read an imported library or the carrier's existing lazy cache.
template <typename Lookup>
int equivalentTimingArc(int begin, int end, int old_lut, int old_offset, Lookup lookup)
{
  if (old_lut >= begin && old_lut < end) return old_lut;
  const auto previous = lookup(old_lut);
  int rank = 0;
  if (old_offset >= 0) {
    for (int lut = old_lut - old_offset; lut < old_lut; ++lut) {
      if (sameTimingSignature(lookup(lut), previous)) ++rank;
    }
  }
  std::vector<int> matches;
  for (int lut = begin; lut < end; ++lut) {
    if (sameTimingSignature(lookup(lut), previous)) matches.push_back(lut);
  }
  if (matches.empty()) {
    throw pybind11::value_error("current master does not contain the timing arc");
  }
  if (matches.size() > 1 && old_offset < 0) {
    throw pybind11::value_error("ambiguous equivalent master timing arc");
  }
  if (rank >= static_cast<int>(matches.size())) {
    throw pybind11::value_error("current master has fewer repeated timing arcs");
  }
  return matches[rank];
}

}  // namespace rebuffer
