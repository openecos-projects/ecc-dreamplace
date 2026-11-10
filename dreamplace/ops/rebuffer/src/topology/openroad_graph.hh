// SPDX-License-Identifier: BSD-3-Clause
// A small undirected graph with the exact incidence-list behaviour used by
// OpenROAD's pdr::primDijkstra.  Node and edge ids are stable and allocated
// in insertion order, matching lemon::ListGraph.

#pragma once

#include <cstddef>
#include <vector>

namespace rebuffer::openroad::graph {

class ListGraph
{
 public:
  struct Node
  {
    int id = -1;

    bool operator==(const Node& other) const { return id == other.id; }
    bool operator!=(const Node& other) const { return id != other.id; }
    bool operator<(const Node& other) const { return id < other.id; }
  };

  struct Edge
  {
    int id = -1;

    bool operator==(const Edge& other) const { return id == other.id; }
    bool operator!=(const Edge& other) const { return id != other.id; }
    bool operator<(const Edge& other) const { return id < other.id; }
  };

  class NodeIt
  {
   public:
    NodeIt() = default;
    NodeIt(const ListGraph* graph, int node)
        : graph_(graph), node_(node)
    {
    }

    bool operator!=(const NodeIt& other) const { return node_ != other.node_; }
    bool operator==(const NodeIt& other) const { return node_ == other.node_; }
    NodeIt& operator++()
    {
      node_ = graph_->nodes_[node_].next;
      return *this;
    }
    Node operator*() const { return Node{node_}; }
    int id() const { return node_; }

   private:
    const ListGraph* graph_ = nullptr;
    int node_ = -1;
  };

  class IncEdgeIt
  {
   public:
    IncEdgeIt() = default;
    IncEdgeIt(const ListGraph* graph, int arc)
        : graph_(graph), arc_(arc)
    {
    }

    bool operator!=(const IncEdgeIt& other) const { return arc_ != other.arc_; }
    bool operator==(const IncEdgeIt& other) const { return arc_ == other.arc_; }
    IncEdgeIt& operator++()
    {
      arc_ = graph_->arcs_[arc_].next_out;
      return *this;
    }
    int id() const { return arc_ / 2; }
    int arcId() const { return arc_; }
    Edge operator*() const { return Edge{arc_ / 2}; }

   private:
    const ListGraph* graph_ = nullptr;
    int arc_ = -1;
  };

  ListGraph() = default;

  Node addNode()
  {
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back(NodeT{});
    nodes_[id].next = first_node_;
    if (first_node_ != -1) {
      nodes_[first_node_].prev = id;
    }
    first_node_ = id;
    nodes_[id].prev = -1;
    nodes_[id].first_out = -1;
    return Node{id};
  }

  Edge addEdge(Node u, Node v)
  {
    const int n = static_cast<int>(arcs_.size());
    arcs_.push_back(ArcT{});
    arcs_.push_back(ArcT{});

    arcs_[n].target = u.id;
    arcs_[n | 1].target = v.id;

    arcs_[n].next_out = nodes_[v.id].first_out;
    if (nodes_[v.id].first_out != -1) {
      arcs_[nodes_[v.id].first_out].prev_out = n;
    }
    arcs_[n].prev_out = -1;
    nodes_[v.id].first_out = n;

    arcs_[n | 1].next_out = nodes_[u.id].first_out;
    if (nodes_[u.id].first_out != -1) {
      arcs_[nodes_[u.id].first_out].prev_out = (n | 1);
    }
    arcs_[n | 1].prev_out = -1;
    nodes_[u.id].first_out = (n | 1);

    return Edge{n / 2};
  }

  int maxNodeId() const { return static_cast<int>(nodes_.size()) - 1; }
  int maxEdgeId() const { return static_cast<int>(arcs_.size()) / 2 - 1; }

  static int id(Node node) { return node.id; }
  static int id(Edge edge) { return edge.id; }
  static Node nodeFromId(int id) { return Node{id}; }
  static Edge edgeFromId(int id) { return Edge{id}; }

  Node u(Edge edge) const { return Node{arcs_[2 * edge.id].target}; }
  Node v(Edge edge) const { return Node{arcs_[2 * edge.id + 1].target}; }

  NodeIt nodesBegin() const { return NodeIt(this, first_node_); }
  NodeIt nodesEnd() const { return NodeIt(this, -1); }

  IncEdgeIt incidentBegin(Node node) const
  {
    return IncEdgeIt(this, nodes_[node.id].first_out);
  }
  IncEdgeIt incidentEnd() const { return IncEdgeIt(this, -1); }

  int runningNode(IncEdgeIt edge) const
  {
    return arcs_[edge.arcId()].target;
  }

  Node oppositeNode(Node node, Edge edge) const
  {
    const Node first = u(edge);
    const Node second = v(edge);
    return first == node ? second : first;
  }

  void changeU(Edge edge, Node node)
  {
    const int arc = 2 * edge.id + 1;
    if (arcs_[arc].next_out != -1) {
      arcs_[arcs_[arc].next_out].prev_out = arcs_[arc].prev_out;
    }
    if (arcs_[arc].prev_out != -1) {
      arcs_[arcs_[arc].prev_out].next_out = arcs_[arc].next_out;
    } else {
      nodes_[arcs_[2 * edge.id].target].first_out = arcs_[arc].next_out;
    }

    if (nodes_[node.id].first_out != -1) {
      arcs_[nodes_[node.id].first_out].prev_out = arc;
    }
    arcs_[2 * edge.id].target = node.id;
    arcs_[arc].prev_out = -1;
    arcs_[arc].next_out = nodes_[node.id].first_out;
    nodes_[node.id].first_out = arc;
  }

  void changeV(Edge edge, Node node)
  {
    const int arc = 2 * edge.id;
    if (arcs_[arc].next_out != -1) {
      arcs_[arcs_[arc].next_out].prev_out = arcs_[arc].prev_out;
    }
    if (arcs_[arc].prev_out != -1) {
      arcs_[arcs_[arc].prev_out].next_out = arcs_[arc].next_out;
    } else {
      nodes_[arcs_[arc | 1].target].first_out = arcs_[arc].next_out;
    }

    if (nodes_[node.id].first_out != -1) {
      arcs_[nodes_[node.id].first_out].prev_out = arc;
    }
    arcs_[arc | 1].target = node.id;
    arcs_[arc].prev_out = -1;
    arcs_[arc].next_out = nodes_[node.id].first_out;
    nodes_[node.id].first_out = arc;
  }

 private:
  struct NodeT
  {
    int first_out = -1;
    int prev = -1;
    int next = -1;
  };

  struct ArcT
  {
    int target = -1;
    int prev_out = -1;
    int next_out = -1;
  };

  std::vector<NodeT> nodes_;
  std::vector<ArcT> arcs_;
  int first_node_ = -1;
};

}  // namespace rebuffer::openroad::graph
