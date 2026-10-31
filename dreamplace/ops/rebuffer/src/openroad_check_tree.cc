// SPDX-License-Identifier: BSD-3-Clause
// Derived from OpenROAD src/stt/src/SteinerTreeBuilder.cpp
// Copyright (c) 2019-2025, The OpenROAD Authors.

#include <algorithm>
#include <vector>

#include "openroad_steiner.hh"

namespace rebuffer::openroad {

namespace {

struct Rect
{
  int x_min;
  int y_min;
  int x_max;
  int y_max;

  Rect(int x1, int y1, int x2, int y2)
      : x_min(std::min(x1, x2)),
        y_min(std::min(y1, y2)),
        x_max(std::max(x1, x2)),
        y_max(std::max(y1, y2))
  {
  }

  bool areaZero() const { return x_min == x_max && y_min == y_max; }

  bool intersects(const Rect& other) const
  {
    return x_min <= other.x_max && other.x_min <= x_max && y_min <= other.y_max
           && other.y_min <= y_max;
  }

  bool containsPoint(int x, int y) const
  {
    return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
  }
};

bool isCorner(const Rect& rect, int x, int y)
{
  return (rect.x_min == x && rect.y_min == y)
         || (rect.x_min == x && rect.y_max == y)
         || (rect.x_max == x && rect.y_min == y)
         || (rect.x_max == x && rect.y_max == y);
}

bool shareCorner(const Rect& lhs, const Rect& rhs)
{
  return isCorner(lhs, rhs.x_min, rhs.y_min)
         || isCorner(lhs, rhs.x_min, rhs.y_max)
         || isCorner(lhs, rhs.x_max, rhs.y_min)
         || isCorner(lhs, rhs.x_max, rhs.y_max);
}

}  // namespace

bool checkTree(const Tree& tree)
{
  // Such high fanout nets are going to get buffered so we do not need to
  // worry about them.
  if (tree.deg > 100) {
    return true;
  }

  std::vector<Rect> rects;
  rects.reserve(tree.branchCount());
  for (int i = 0; i < tree.branchCount(); ++i) {
    const Branch& branch = tree.branch[i];
    const Branch& neighbor = tree.branch[branch.n];
    rects.emplace_back(branch.x, branch.y, neighbor.x, neighbor.y);
  }

  for (size_t i = 0; i < rects.size(); ++i) {
    for (size_t j = i + 1; j < rects.size(); ++j) {
      if (!rects[i].areaZero() && !rects[j].areaZero()
          && rects[i].intersects(rects[j])
          && !shareCorner(rects[i], rects[j])) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace rebuffer::openroad
