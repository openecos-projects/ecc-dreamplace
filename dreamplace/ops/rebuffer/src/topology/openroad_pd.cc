// SPDX-License-Identifier: BSD-3-Clause
// Derived from OpenROAD src/stt/src/pdr/src/pd.cpp
// Copyright (c) 2022-2025, The OpenROAD Authors.
//
// The algorithm is unchanged.  lemon::ListGraph is replaced by the small
// graph in openroad_graph.hh so the provider can be built by ECC DreamPlace
// without an OpenROAD/ODB runtime dependency.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "boost/heap/d_ary_heap.hpp"

#include "openroad_steiner.hh"


namespace rebuffer::openroad::pdr {

using rebuffer::openroad::graph::ListGraph;
using std::vector;

struct Point
{
  int x = 0;
  int y = 0;

  int getX() const { return x; }
  int getY() const { return y; }
  void setX(int value) { x = value; }
  void setY(int value) { y = value; }

  bool operator==(const Point& other) const
  {
    return x == other.x && y == other.y;
  }

  static int manhattanDistance(const Point& lhs, const Point& rhs)
  {
    return std::abs(lhs.x - rhs.x) + std::abs(lhs.y - rhs.y);
  }
};

/////////// Nearest Neighbors

// This is the method in "Prim-Dijkstra Revisited" section 4.
// The key idea is: "We say that vi is a neighbor of vj if the smallest
// bounding box containing vi and vj contains no other nodes."
//
// This has nothing to do with the Guibas & Stolfi method despite its
// mention in the paper and the comments of the original code.  GS is
// not a good choice as it excludes edges that may be optimal for high
// alpha values.

using Neighbors = std::vector<int>;
static vector<Neighbors> get_nearest_neighbors(const vector<Point>& pts)
{
  thread_local static vector<int> data;
  data.clear();

  const size_t pt_count = pts.size();
  vector<Neighbors> neighbors(pt_count);

  // These keep track of the closest node in X seen so far in the
  // respective quadrant of each node (the index).  Any node beyond
  // this coordinate would have another node in its bbox and is
  // therefore not a nearest neighbor.  This depends on processing the
  // nodes in order of increasing y distance.
  data.reserve(pt_count * 5);
  data.resize(pt_count * 2, std::numeric_limits<int>::max());
  data.resize(pt_count * 4, std::numeric_limits<int>::min());
  data.resize(pt_count * 5);
  int* const ur = &data[0];  // NOLINT
  int* const lr = &data[pt_count];
  int* const ul = &data[pt_count * 2];
  int* const ll = &data[pt_count * 3];
  int* const sorted = &data[pt_count * 4];

  // sort in y-axis
  std::iota(sorted, sorted + pt_count, 0);
  std::stable_sort(sorted, sorted + pt_count, [&pts](int i, int j) {
    return std::make_pair(pts[i].getY(), pts[i].getX())
           < std::make_pair(pts[j].getY(), pts[j].getX());
  });

  // Compute neighbors going from bottom to top in Y
  for (int idx = 0; idx < pt_count; ++idx) {
    const int pt_idx = sorted[idx];
    const int pt_x = pts[pt_idx].getX();
    // Update upper neighbors of all pts below pt (below.y <= pt.y)
    for (int i = 0; i < idx; ++i) {
      const int below_idx = sorted[i];
      const int below_x = pts[below_idx].getX();
      if (below_x <= pt_x && pt_x < ur[below_idx]) {  // pt in ur
        neighbors[below_idx].push_back(pt_idx);
        ur[below_idx] = pt_x;
      } else if (ul[below_idx] < pt_x && pt_x < below_x) {  // pt in ul
        neighbors[below_idx].push_back(pt_idx);
        ul[below_idx] = pt_x;
      }
    }

    // Set all lower neighbors for 'pt' (below.y <= pt.y)
    for (int i = idx - 1; i >= 0; --i) {
      const int below_idx = sorted[i];
      const int below_x = pts[below_idx].getX();
      if (pt_x <= below_x && below_x < lr[pt_idx]) {  // below in lr
        neighbors[pt_idx].push_back(below_idx);
        lr[pt_idx] = below_x;
      } else if (ll[pt_idx] < below_x && below_x < pt_x) {  // below in ll
        neighbors[pt_idx].push_back(below_idx);
        ll[pt_idx] = below_x;
      }
    }
  }

  return neighbors;
}

/////////// Minimum Spanning Tree per PD costing

// A potential edge in the minimum spanning tree (MST).  These are stored
// in the heap during the PD search.
struct SearchEdge
{
  float weight;            // the heap key holding the PD cost of this edge
  int path_length;         // distance from the driver to node
  ListGraph::Node parent;  // may be updated during the search
  const ListGraph::Node node;
};

// Boost provides a max heap but we need a min heap so this reverses
// the comparison to achieve that.
struct CmpEdge
{
  bool operator()(const SearchEdge& lhs, const SearchEdge& rhs) const
  {
    return std::tie(lhs.weight, lhs.parent, lhs.node)
           > std::tie(rhs.weight, rhs.parent, rhs.node);
  }
};

static void buildSpanningTree(const vector<Point>& node_point,
                              const ListGraph::Node& driver_node,
                              const float alpha,
                              const vector<Neighbors>& nn,
                              ListGraph& graph)
{
  using Heap = boost::heap::d_ary_heap<SearchEdge,
                                       boost::heap::arity<2>,
                                       boost::heap::mutable_<true>,
                                       boost::heap::compare<CmpEdge>>;

  const int num_nodes = graph.maxNodeId() + 1;
  int num_visited = 0;
  vector<bool> visited(num_nodes, false);

  // Handles allow you to reference a heap entry for the node even
  // as the heap is updated.
  vector<Heap::handle_type> handles(num_nodes);

  Heap heap;
  const ListGraph::Node invalid_node{-1};
  handles[ListGraph::id(driver_node)]
      = heap.push({0, 0, invalid_node, driver_node});

  while (!heap.empty()) {
    const SearchEdge edge = heap.top();
    heap.pop();

    visited[ListGraph::id(edge.node)] = true;
    ++num_visited;
    if (edge.parent.id != -1) {  // skip root node
      graph.addEdge(edge.parent, edge.node);
    }

    if (num_visited == num_nodes) {
      break;  // all nodes are in the graph so we are done
    }

    // Add the nearest neighbors to the heap
    for (const int neighbor : nn[ListGraph::id(edge.node)]) {
      auto neighbor_node = ListGraph::nodeFromId(neighbor);
      if (visited[neighbor]) {
        continue;  // already in the graph
      }

      const int edge_length = Point::manhattanDistance(
          node_point[neighbor], node_point[ListGraph::id(edge.node)]);
      const int neighbor_path_length = edge_length + edge.path_length;
      const float neighbor_weight = edge_length + alpha * edge.path_length;

      auto& handle = handles[neighbor];
      if (handle == Heap::handle_type()) {  // not in the heap
        handles[neighbor] = heap.push(
            {neighbor_weight, neighbor_path_length, edge.node, neighbor_node});
      } else if (neighbor_weight <= (*handle).weight) {
        (*handle).weight = neighbor_weight;
        (*handle).path_length = neighbor_path_length;
        (*handle).parent = edge.node;
        heap.increase(handle);
      }
    }
  }
}

/////////// Steinerize the Spanning Tree

// A possible Steiner node
struct CandidateSteiner
{
  int gain;  // wire length improvement by adding this node
  ListGraph::Node node;
  ListGraph::Edge edge1;
  ListGraph::Edge edge2;
  Point steiner_point;
};

struct CmpCandidate
{
  bool operator()(const CandidateSteiner& lhs,
                  const CandidateSteiner& rhs) const
  {
    // Gain is key, the steiner_point is just an arbitrary tie breaker.
    return std::make_tuple(
               lhs.gain, lhs.steiner_point.getX(), lhs.steiner_point.getY())
           < std::make_tuple(
               rhs.gain, rhs.steiner_point.getX(), rhs.steiner_point.getY());
  }
};

// Compute the best pair of edges connected to a node that can be
// improved by adding a Steiner node.
static CandidateSteiner best_steiner_for_node(
    const ListGraph& graph,
    const ListGraph::Node& node,
    const vector<Point>& node_point)
{
  const Point pt_node = node_point[ListGraph::id(node)];

  CandidateSteiner best;
  best.gain = 0;

  // Loop through all edge pairs.  N^2 but N is small.
  for (auto edge1 = graph.incidentBegin(node);
       edge1 != graph.incidentEnd();
       ++edge1) {
    auto n1 = graph.runningNode(edge1);
    const Point pt1 = node_point[n1];
    auto edge2 = edge1;
    for (++edge2; edge2 != graph.incidentEnd(); ++edge2) {
      auto n2 = graph.runningNode(edge2);
      const Point pt2 = node_point[n2];

      Point pt_steiner = pt_node;
      if (std::min(pt1.getX(), pt2.getX()) > pt_node.getX()) {
        // both right of node
        pt_steiner.setX(std::min(pt1.getX(), pt2.getX()));
      } else if (std::max(pt1.getX(), pt2.getX()) < pt_node.getX()) {
        // both left of node
        pt_steiner.setX(std::max(pt1.getX(), pt2.getX()));
      }

      if (std::min(pt1.getY(), pt2.getY()) > pt_node.getY()) {
        // both above node
        pt_steiner.setY(std::min(pt1.getY(), pt2.getY()));
      } else if (std::max(pt1.getY(), pt2.getY()) < pt_node.getY()) {
        // both below node
        pt_steiner.setY(std::max(pt1.getY(), pt2.getY()));
      }

      const int gain = Point::manhattanDistance(pt_steiner, pt_node);
      if (gain > best.gain) {
        best = {gain, node, *edge1, *edge2, pt_steiner};
      }
    }
  }
  return best;
}

// Steinerize by looking at all adjacent edge pairs and finding the one
// with maximum improvement by adding a Steiner point.  Repeat until
// no further improvement can be found.  This idea comes from footnote
// 1 in "Prim-Dijkstra tradeoffs for improved performance-driven
// routing tree design".
static void steinerize(ListGraph& graph, vector<Point>& node_point)
{
  using Heap = boost::heap::d_ary_heap<CandidateSteiner,
                                       boost::heap::arity<2>,
                                       boost::heap::mutable_<true>,
                                       boost::heap::compare<CmpCandidate>>;
  Heap heap;
  std::map<ListGraph::Node, Heap::handle_type> handles;

  // Setup the heap with all the candidates
  for (auto node = graph.nodesBegin(); node != graph.nodesEnd(); ++node) {
    handles[*node] = heap.push(best_steiner_for_node(graph, *node, node_point));
  }

  while (!heap.empty()) {
    const CandidateSteiner best = heap.top();
    heap.pop();

    if (best.gain == 0) {
      break;
    }

    auto opp1 = graph.oppositeNode(best.node, best.edge1);
    auto opp2 = graph.oppositeNode(best.node, best.edge2);

    bool new_node = false;
    ListGraph::Node steiner_node;
    if (best.steiner_point == node_point[ListGraph::id(opp1)]) {
      steiner_node = opp1;
    } else if (best.steiner_point == node_point[ListGraph::id(opp2)]) {
      steiner_node = opp2;
    } else {
      steiner_node = graph.addNode();
      node_point.push_back(best.steiner_point);
      new_node = true;
    }

    if (steiner_node != opp1) {
      if (graph.u(best.edge1) == best.node) {
        graph.changeU(best.edge1, steiner_node);
      } else {
        graph.changeV(best.edge1, steiner_node);
      }
    }

    if (steiner_node != opp2) {
      if (graph.u(best.edge2) == best.node) {
        graph.changeU(best.edge2, steiner_node);
      } else {
        graph.changeV(best.edge2, steiner_node);
      }
    }

    if (new_node) {
      // Add to the heap
      graph.addEdge(steiner_node, best.node);
      handles[steiner_node]
          = heap.push(best_steiner_for_node(graph, steiner_node, node_point));
    }

    // Find the new best candidate for the updated nodes
    handles[best.node]
        = heap.push(best_steiner_for_node(graph, best.node, node_point));

    for (const auto& node : {opp1, opp2}) {
      *handles[node] = best_steiner_for_node(graph, node, node_point);
      heap.update(handles[node]);
    }
  }
}

/////////// Convert to Flute-style Tree

// Flute only returns nodes with maximum degree of three.  This mirrors
// OpenROAD's enforcement of the same invariant.
static void splitDegree4Nodes(ListGraph& graph, vector<Point>& node_point)
{
  std::deque<ListGraph::Node> to_process;
  for (auto node = graph.nodesBegin(); node != graph.nodesEnd(); ++node) {
    to_process.push_back(*node);
  }

  while (!to_process.empty()) {
    ListGraph::Node node = to_process.front();
    to_process.pop_front();
    int edge_cnt = 0;
    for (auto edge = graph.incidentBegin(node);
         edge != graph.incidentEnd();
         ++edge) {
      ++edge_cnt;
    }
    if (edge_cnt <= 3) {
      continue;
    }

    auto new_node = graph.addNode();
    node_point.push_back(node_point[ListGraph::id(node)]);

    edge_cnt = 0;
    auto edge = graph.incidentBegin(node);
    while (edge != graph.incidentEnd()) {
      if (++edge_cnt >= 3) {
        auto working_edge = edge;
        ++edge;
        if (graph.u(*working_edge) == node) {
          graph.changeU(*working_edge, new_node);
        } else {
          graph.changeV(*working_edge, new_node);
        }
      } else {
        ++edge;
      }
    }
    graph.addEdge(node, new_node);
    to_process.push_back(new_node);
  }
}

static void makeTreeRecursive(const ListGraph& graph,
                              const ListGraph::Node& node,
                              const ListGraph::Node& parent,
                              const vector<Point>& node_point,
                              Tree& tree)
{
  const int parent_id = ListGraph::id(parent);
  const int parent_x = node_point[parent_id].getX();
  const int parent_y = node_point[parent_id].getY();

  const int n = ListGraph::id(node);
  const int x = node_point[n].getX();
  const int y = node_point[n].getY();
  tree.branch[n] = {x, y, parent_id};
  tree.length += std::abs(x - parent_x) + std::abs(y - parent_y);

  for (auto edge = graph.incidentBegin(node);
       edge != graph.incidentEnd();
       ++edge) {
    auto child_node = graph.runningNode(edge);
    if (child_node != parent_id) {
      makeTreeRecursive(
          graph, ListGraph::nodeFromId(child_node), node, node_point, tree);
    }
  }
}

static Tree makeTree(ListGraph& graph,
                     const int num_terminals,
                     const ListGraph::Node& driver_node,
                     const vector<Point>& node_point)
{
  Tree tree;
  tree.deg = num_terminals;
  tree.length = 0;
  // Flute-style needs a self edge on the driver node
  tree.branch.resize(graph.maxNodeId() + 1);

  makeTreeRecursive(graph, driver_node, driver_node, node_point, tree);

  return tree;
}

/////////// Entry Point

Tree primDijkstra(const vector<int>& x,
                  const vector<int>& y,
                  const int driver_index,
                  const float alpha)
{
  if (x.size() != y.size()) {
    throw std::invalid_argument("x size != y size");
  }
  if (x.empty()) {
    throw std::invalid_argument("invalid request for an empty Steiner tree");
  }
  if (driver_index < 0 || driver_index >= static_cast<int>(x.size())) {
    throw std::out_of_range("driver index is outside the point list");
  }

  ListGraph graph;
  vector<Point> node_point;  // Node -> location

  const int num_terminals = x.size();
  vector<Point> pts(num_terminals);
  node_point.reserve(num_terminals);
  for (int i = 0; i < num_terminals; ++i) {
    pts[i] = {x[i], y[i]};
    graph.addNode();
    node_point.push_back(pts[i]);
  }

  const auto nn = get_nearest_neighbors(pts);
  auto driver_node = ListGraph::nodeFromId(driver_index);
  buildSpanningTree(node_point, driver_node, alpha, nn, graph);
  steinerize(graph, node_point);
  splitDegree4Nodes(graph, node_point);
  return makeTree(graph, num_terminals, driver_node, node_point);
}

}  // namespace rebuffer::openroad::pdr

// --- openroad_check_tree.cc ---
// Derived from OpenROAD src/stt/src/SteinerTreeBuilder.cpp
// Copyright (c) 2019-2025, The OpenROAD Authors.

#include <algorithm>
#include <vector>


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

// --- openroad_steiner_bindings.cc ---
// Python boundary for the OpenROAD-compatible placement Steiner provider.

#include <stdexcept>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

#include "pybind11/pybind11.h"

#include "pybind11/stl.h"


namespace py = pybind11;

namespace {

std::mutex& fluteMutex()
{
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<rebuffer::openroad::flt::Flute>& cachedFlute()
{
  static std::unique_ptr<rebuffer::openroad::flt::Flute> flute;
  return flute;
}

std::string& cachedPostLut()
{
  static std::string path;
  return path;
}

std::string& cachedPowvLut()
{
  static std::string path;
  return path;
}

rebuffer::openroad::Tree fluteTree(const std::vector<int>& x,
                                   const std::vector<int>& y,
                                   int accuracy,
                                   const std::string& post_lut,
                                   const std::string& powv_lut)
{
  std::lock_guard<std::mutex> lock(fluteMutex());
  auto& flute = cachedFlute();
  if (!flute || cachedPostLut() != post_lut || cachedPowvLut() != powv_lut) {
    flute = std::make_unique<rebuffer::openroad::flt::Flute>();
    flute->setLutPaths(post_lut, powv_lut);
    cachedPostLut() = post_lut;
    cachedPowvLut() = powv_lut;
  }
  return flute->flute(x, y, accuracy);
}

py::dict treeToDict(const rebuffer::openroad::Tree& tree,
                    const std::string& provider)
{
  py::list branch_x;
  py::list branch_y;
  py::list branch_n;
  py::list edges;
  for (int index = 0; index < tree.branchCount(); ++index) {
    const auto& branch = tree.branch[index];
    branch_x.append(branch.x);
    branch_y.append(branch.y);
    branch_n.append(branch.n);
    if (branch.n >= 0 && branch.n < tree.branchCount()) {
      const auto& neighbor = tree.branch[branch.n];
      if (branch.x != neighbor.x || branch.y != neighbor.y) {
        edges.append(py::make_tuple(branch.x, branch.y, neighbor.x, neighbor.y));
      }
    }
  }

  py::dict result;
  result["status"] = "ok";
  result["provider"] = provider;
  result["deg"] = tree.deg;
  result["wirelength"] = tree.length;
  result["branch_x"] = std::move(branch_x);
  result["branch_y"] = std::move(branch_y);
  result["branch_n"] = std::move(branch_n);
  result["edges"] = std::move(edges);
  result["branch_count"] = tree.branchCount();
  return result;
}

rebuffer::openroad::Tree makeSttTree(const std::vector<int>& x,
                                     const std::vector<int>& y,
                                     int driver_index,
                                     float alpha,
                                     const std::string& post_lut,
                                     const std::string& powv_lut,
                                     bool& used_pd)
{
  used_pd = false;
  if (alpha > 0.0f) {
    auto tree
        = rebuffer::openroad::pdr::primDijkstra(x, y, driver_index, alpha);
    if (rebuffer::openroad::checkTree(tree)) {
      used_pd = true;
      return tree;
    }
  }

  return fluteTree(x, y, 3, post_lut, powv_lut);
}

}  // namespace

PYBIND11_MODULE(rebuffer_steiner_cpp, module)
{
  module.doc()
      = "OpenROAD-compatible Prim-Dijkstra and FLUTE placement Steiner provider";

  module.def(
      "prim_dijkstra",
      [](const std::vector<int>& x,
         const std::vector<int>& y,
         int driver_index,
         float alpha) {
        return treeToDict(
            rebuffer::openroad::pdr::primDijkstra(
                x, y, driver_index, alpha),
            "openroad_prim_dijkstra");
      },
      py::arg("x"),
      py::arg("y"),
      py::arg("driver_index"),
      py::arg("alpha") = 0.3f);

  module.def(
      "flute",
      [](const std::vector<int>& x,
         const std::vector<int>& y,
         int accuracy,
         const std::string& post_lut,
         const std::string& powv_lut) {
        return treeToDict(
            fluteTree(x, y, accuracy, post_lut, powv_lut), "openroad_flute");
      },
      py::arg("x"),
      py::arg("y"),
      py::arg("accuracy") = 3,
      py::arg("post_lut"),
      py::arg("powv_lut"));

  module.def(
      "stt",
      [](const std::vector<int>& x,
         const std::vector<int>& y,
         int driver_index,
         float alpha,
         const std::string& post_lut,
         const std::string& powv_lut) {
        bool used_pd = false;
        auto tree = makeSttTree(
            x, y, driver_index, alpha, post_lut, powv_lut, used_pd);
        return treeToDict(
            tree, used_pd ? "openroad_prim_dijkstra" : "openroad_flute");
      },
      py::arg("x"),
      py::arg("y"),
      py::arg("driver_index"),
      py::arg("alpha") = 0.3f,
      py::arg("post_lut"),
      py::arg("powv_lut"));
}
