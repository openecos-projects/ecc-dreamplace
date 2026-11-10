// SPDX-License-Identifier: BSD-3-Clause
// Resolve active wire segments and export final physical commit partitions.
#include "rebuffer_overlay_actions.hh"
#include "rebuffer_carrier.hh"
#include "rebuffer_tree_actions.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <limits>
#include <set>
#include <sstream>

namespace py = pybind11;
namespace rebuffer {
namespace {

// The physical bridge stores integer DBU coordinates, so an endpoint
// insertion has zero wire length after commit.  The incremental graph still
// requires a strict interior split; keep the modelling epsilon below the
// endpoint quantization error so it cannot accumulate into overlay/commit TP
// drift across a campaign.
constexpr double kEndpointProjectionEpsilon = 1.0e-9;

std::vector<int> terminalLoads(const IncrementalTopologyState& state, int root)
{
  std::map<int, std::vector<int>> children;
  for (const auto& edge : state.edges) children[edge.parent].push_back(edge.child);
  std::vector<int> stack{root}, loads;
  while (!stack.empty()) {
    const int node = stack.back(); stack.pop_back();
    const auto found = state.node_loads.find(node);
    if (found != state.node_loads.end()) loads.insert(loads.end(), found->second.begin(), found->second.end());
    for (const int child : children[node]) stack.push_back(child);
  }
  std::sort(loads.begin(), loads.end());
  return loads;
}

bool cellEdge(const IncrementalTopologyState& state, const IncrementalEdge& edge)
{
  return std::any_of(state.buffers.begin(), state.buffers.end(), [&](const auto& buffer) {
    return edge.parent == buffer.input_node && edge.child == buffer.output_node;
  });
}

bool containsLoads(const std::vector<int>& closure,
                   const std::set<int>& selected)
{
  return std::all_of(selected.begin(), selected.end(), [&](const int pin) {
    return std::binary_search(closure.begin(), closure.end(), pin);
  });
}

std::vector<int> sourceChildClosure(const IncrementalTopologyState& state,
                                    const std::vector<int>& source_nodes,
                                    const std::set<int>& selected)
{
  // A current-tree source child can be a branch head in the active carrier
  // after a different Steiner realization or an accepted ancestor split.  In
  // that case the candidate's raw downstream vertex is only a source hint;
  // the physical split must carry the complete terminal closure of the
  // active incoming wire edge.
  for (const int source_node : source_nodes) {
    for (const auto& edge : state.edges) {
      if (edge.child != source_node || cellEdge(state, edge)) {
        continue;
      }
      const auto closure = terminalLoads(state, edge.child);
      if (!closure.empty() && containsLoads(closure, selected)) {
        return closure;
      }
    }
  }
  return {};
}

std::string displayName(std::string name)
{
  std::replace(name.begin(), name.end(), ':', '/');
  return name;
}

void physicalNames(py::dict& action, const std::vector<std::string>& pins,
                   const std::vector<std::string>& nets,
                   const IncrementalTopologyState& state, const std::vector<int>& loads)
{
  std::vector<std::string> names;
  for (const int pin : loads) names.push_back(displayName(pins.at(pin)));
  action["action_kind"] = "buffer_insert";
  action["net_id"] = state.net_id;
  action["net_name"] = nets.at(state.net_id);
  action["driver_pin"] = state.driver_pin_id;
  action["driver_pin_name"] = displayName(pins.at(state.driver_pin_id));
  action["downstream_pins"] = loads;
  action["downstream_pin_names"] = names;
}

}  // namespace

py::dict resolveOverlaySplit(const IncrementalTopologyState& state, const py::dict& action)
{
  auto selected = action["downstream_pin_ids"].cast<std::vector<int>>();
  std::sort(selected.begin(), selected.end());
  if (selected.empty() || std::adjacent_find(selected.begin(), selected.end()) != selected.end()) {
    throw py::value_error("downstream partition must be nonempty and unique");
  }
  const double x = py::cast<double>(action["x_dbu"]), y = py::cast<double>(action["y_dbu"]);
  const int hint_parent = action.contains("parent_node_id") ? py::cast<int>(action["parent_node_id"]) : -1;
  const int hint_child = action.contains("child_node_id") ? py::cast<int>(action["child_node_id"]) : -1;
  auto edges = state.edges;
  std::stable_sort(edges.begin(), edges.end(), [&](const auto& a, const auto& b) {
    const bool ah = a.parent == hint_parent && a.child == hint_child;
    const bool bh = b.parent == hint_parent && b.child == hint_child;
    return ah > bh;
  });
  for (const auto& edge : edges) {
    if (cellEdge(state, edge) || terminalLoads(state, edge.child) != selected) continue;
    const auto& from = state.coordinates.at(edge.parent);
    const auto& to = state.coordinates.at(edge.child);
    const double dx = to[0] - from[0], dy = to[1] - from[1];
    const double length2 = dx * dx + dy * dy;
    // Coincident pins still admit distinct buffer input/output vertices.
    double ratio = length2 == 0.0 ? 0.5
        : ((x - from[0]) * dx + (y - from[1]) * dy) / length2;
    const bool hinted = edge.parent == hint_parent && edge.child == hint_child;
    if (hinted && action.contains("split_ratio")) ratio = py::cast<double>(action["split_ratio"]);
    if (ratio < -1e-9 || ratio > 1.0 + 1e-9) continue;
    // Junction producers do not carry a ratio. Keep their integer DBU
    // location while introducing distinct logical input/output vertices.
    if (!action.contains("split_ratio") || !hinted) {
      const double epsilon = std::min(
          kEndpointProjectionEpsilon,
          0.25 / std::max({std::abs(dx), std::abs(dy), 1.0}));
      ratio = std::clamp(ratio, epsilon, 1.0 - epsilon);
    }
    if (!(ratio > 0 && ratio < 1) || !std::isfinite(ratio)
        || std::abs(x - from[0] - ratio * dx) > 1.0
        || std::abs(y - from[1] - ratio * dy) > 1.0) continue;
    py::dict resolved = action.attr("copy")().cast<py::dict>();
    resolved["parent_node_id"] = edge.parent;
    resolved["child_node_id"] = edge.child;
    resolved["split_ratio"] = ratio;
    resolved["downstream_pin_ids"] = selected;
    return resolved;
  }
  // A repair_design producer can be placed at a Steiner junction.  The
  // physical coordinate is then exactly an endpoint of the active edge,
  // while the buffer cell still needs a strictly interior split ratio for
  // the immutable edge representation.  Accept only an endpoint match with
  // the complete subtree partition; never relax the load-set check.
  for (const auto& edge : edges) {
    if (cellEdge(state, edge) || terminalLoads(state, edge.child) != selected) continue;
    const auto& from = state.coordinates.at(edge.parent);
    const auto& to = state.coordinates.at(edge.child);
    const double from_distance = std::hypot(x - from[0], y - from[1]);
    const double to_distance = std::hypot(x - to[0], y - to[1]);
    const double endpoint_tolerance = 1.0;
    if (std::min(from_distance, to_distance) > endpoint_tolerance) continue;
    const double dx = to[0] - from[0], dy = to[1] - from[1];
    const double length = std::hypot(dx, dy);
    if (!(length > 0.0)) continue;
    const double epsilon = std::min(kEndpointProjectionEpsilon, 0.25 / length);
    const double ratio = to_distance <= from_distance ? 1.0 - epsilon : epsilon;
    py::dict resolved = action.attr("copy")().cast<py::dict>();
    resolved["parent_node_id"] = edge.parent;
    resolved["child_node_id"] = edge.child;
    resolved["split_ratio"] = ratio;
    resolved["downstream_pin_ids"] = selected;
    return resolved;
  }
  // FLUTE/native placement snapshots may give the same electrical subtree
  // with a different physical junction coordinate.  Preserve the exact load
  // partition and project the requested point onto its unique active edge;
  // callers receive an explicit provenance marker and the adjusted point.
  double best_distance = std::numeric_limits<double>::infinity();
  const IncrementalEdge* best_edge = nullptr;
  double best_ratio = 0.0;
  for (const auto& edge : edges) {
    if (cellEdge(state, edge) || terminalLoads(state, edge.child) != selected) continue;
    const auto& from = state.coordinates.at(edge.parent);
    const auto& to = state.coordinates.at(edge.child);
    const double dx = to[0] - from[0], dy = to[1] - from[1];
    const double length2 = dx * dx + dy * dy;
    if (!(length2 > 0.0)) continue;
    const double projected = std::clamp(
        ((x - from[0]) * dx + (y - from[1]) * dy) / length2,
        0.0, 1.0);
    const double px = from[0] + projected * dx, py = from[1] + projected * dy;
    const double distance = std::hypot(x - px, y - py);
    if (distance < best_distance) {
      best_distance = distance;
      best_edge = &edge;
      best_ratio = projected;
    }
  }
  if (best_edge != nullptr) {
    const auto& from = state.coordinates.at(best_edge->parent);
    const auto& to = state.coordinates.at(best_edge->child);
    const double length = std::hypot(to[0] - from[0], to[1] - from[1]);
    const double epsilon = std::min(
        kEndpointProjectionEpsilon,
        0.25 / std::max(length, 1.0));
    best_ratio = std::clamp(best_ratio, epsilon, 1.0 - epsilon);
    py::dict resolved = action.attr("copy")().cast<py::dict>();
    resolved["parent_node_id"] = best_edge->parent;
    resolved["child_node_id"] = best_edge->child;
    resolved["split_ratio"] = best_ratio;
    resolved["x_dbu"] = from[0] + best_ratio * (to[0] - from[0]);
    resolved["y_dbu"] = from[1] + best_ratio * (to[1] - from[1]);
    resolved["downstream_pin_ids"] = selected;
    resolved["resolution_source"] = "native_active_edge_projection";
    resolved["projection_distance_dbu"] = best_distance;
    return resolved;
  }
  std::ostringstream diagnostic;
  diagnostic << "no active wire segment matches the requested physical load partition and location"
             << "; state_net=" << state.net_id
             << "; x_dbu=" << x
             << "; y_dbu=" << y
             << "; hint_parent=" << hint_parent
             << "; hint_child=" << hint_child
             << "; requested_loads=";
  for (std::size_t index = 0; index < selected.size(); ++index) {
    if (index != 0) diagnostic << ',';
    diagnostic << selected[index];
  }
  if (action.contains("projection_selected_loads")) {
    diagnostic << "; projection_selected_loads=";
    for (const int pin : action["projection_selected_loads"].cast<std::vector<int>>()) {
      diagnostic << pin << ',';
    }
  }
  if (action.contains("projection_source_child")) {
    diagnostic << "; projection_source_child="
               << py::cast<int>(action["projection_source_child"]);
  }
  if (action.contains("projection_source_child_identity")) {
    diagnostic << "; projection_source_child_identity="
               << py::cast<int>(action["projection_source_child_identity"]);
  }
  if (action.contains("projection_source_closure_expanded")) {
    diagnostic << "; projection_source_closure_expanded="
               << py::cast<bool>(action["projection_source_closure_expanded"]);
  }
  diagnostic << "; edges=";
  for (const auto& edge : state.edges) {
    if (cellEdge(state, edge)) continue;
    diagnostic << '[' << edge.parent << ',' << edge.child << ':';
    const auto loads = terminalLoads(state, edge.child);
    for (std::size_t index = 0; index < loads.size(); ++index) {
      if (index != 0) diagnostic << ',';
      diagnostic << loads[index];
    }
    diagnostic << ']';
  }
  throw py::value_error(diagnostic.str());
}

py::list projectOverlayActions(
    const py::object& base, const py::dict& payload,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions)
{
  OverlayActionProjection projection;
  projection.original_net_by_net = payload["original_net_by_net"].cast<std::vector<int>>();
  projection.state_node_by_vertex = payload["state_node_by_vertex"].cast<std::vector<int>>();
  projection.state_net_by_vertex = payload["state_net_by_vertex"].cast<std::vector<int>>();
  projection.has_partition_provenance = payload.contains("state_provenance");
  if (projection.has_partition_provenance) {
    for (const auto& item : payload["state_provenance"].cast<py::sequence>()) {
      const auto row = item.cast<py::dict>();
      const int partition_id = py::cast<int>(row["partition_id"]);
      const int vertex_id = py::cast<int>(row["vertex_id"]);
      const int state_net_id = py::cast<int>(row["state_net_id"]);
      const int state_node_id = py::cast<int>(row["state_node_id"]);
      projection.state_provenance.push_back(
          {{partition_id, vertex_id}, {state_net_id, state_node_id}});
    }
    normalizePartitionProvenance(projection.state_provenance);
  }
  return projectOverlayActionsFromProjection(base, projection, states, actions);
}

py::list projectOverlayActionsFromProjection(
    const OverlayActionInputs& base, const OverlayActionProjection& projection,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions)
{
  std::map<int, std::shared_ptr<IncrementalTopologyState>> working_by_net;
  std::set<int> copied_states;
  for (const auto& state : states) {
    if (state == nullptr) throw py::value_error("null overlay topology state");
    working_by_net[state->net_id] = state;
  }
  const auto& original_net = projection.original_net_by_net;
  const auto& state_node = projection.state_node_by_vertex;
  const auto& state_net = projection.state_net_by_vertex;
  const auto& partition_provenance = projection.state_provenance;
  const bool has_partition_provenance = projection.has_partition_provenance;
  const auto& cells = base.cells;
  const auto& pins = base.pins;
  const auto& nets = base.nets;
  const auto& pin2node = base.pin2node;
  const auto& pin2net = base.pin2net;
  const auto& drivers = base.drivers;
  const auto ordered = orderTreeBufferActions(actions);
  py::list output;
  int simulation_action_id = 1000000000;
  for (const auto& ordered_action : ordered) {
    py::dict action = ordered_action.attr("copy")().cast<py::dict>();
    const int partition = py::cast<int>(action["net_id"]);
    const int planning_net = original_net.at(partition);
    IncrementalTopologyState unchanged;
    const int graph_parent = py::cast<int>(action["parent_node_id"]);
    const int graph_child = py::cast<int>(action["child_node_id"]);
    const bool has_source_parent = action.contains("source_parent_node_id");
    const bool has_source_child = action.contains("source_child_node_id");
    if (has_source_parent != has_source_child) {
      throw py::value_error("source physical identity is incomplete");
    }

    // A complete current-tree candidate carries two identity domains.  The
    // candidate graph ids describe the native DP tree and are not placement
    // vertices.  The source ids identify the physical carrier/Steiner edge
    // that must be projected into one of the mutable per-net states.  Prefer
    // that provenance whenever it is present; otherwise retain the legacy
    // partition-vertex mapping for single-net plans.
    const int source_parent = has_source_parent
                                  ? py::cast<int>(action["source_parent_node_id"])
                                  : graph_parent;
    const int source_child = has_source_child
                                 ? py::cast<int>(action["source_child_node_id"])
                                 : graph_child;
    auto activeStateVertex = [&](int preferred_net, int vertex) {
      auto matches = [&](const IncrementalTopologyState& candidate) {
        if (candidate.node_loads.count(vertex) != 0) return true;
        return std::any_of(candidate.edges.begin(), candidate.edges.end(),
                           [&](const auto& edge) {
                             return edge.parent == vertex || edge.child == vertex;
                           });
      };
      const auto preferred = working_by_net.find(preferred_net);
      if (preferred != working_by_net.end() && matches(*preferred->second)) {
        return std::pair<int, int>{preferred->second->net_id, vertex};
      }
      // A current-tree partition id can refer to a synthetic output net while
      // the physical vertex is owned by the original root state.  Search the
      // remaining active states only when the vertex has a unique owner;
      // silently choosing one of several states would hide a topology bug.
      int owner_net = -1;
      for (const auto& item : working_by_net) {
        if (item.first == preferred_net || !matches(*item.second)) {
          continue;
        }
        if (owner_net >= 0 && owner_net != item.second->net_id) {
          return std::pair<int, int>{-1, -1};
        }
        owner_net = item.second->net_id;
      }
      return owner_net >= 0 ? std::pair<int, int>{owner_net, vertex}
                            : std::pair<int, int>{-1, -1};
    };
    auto sourceVertex = [&](int partition_id, int vertex, const char* role) {
      if (has_partition_provenance) {
        const auto found = findPartitionProvenance(partition_provenance, {partition_id, vertex});
        if (found != partition_provenance.end()) {
          return found->second;
        }
      }
      // A descendant load has carrier provenance in its own partition.
      // Resolve it before consulting mutable nodes, whose ids may alias it.
      if (vertex < 0 || vertex >= static_cast<int>(state_node.size())) {
        return activeStateVertex(planning_net, vertex);
      }
      if (state_node.at(vertex) < 0 || state_net.at(vertex) < 0) {
        return activeStateVertex(planning_net, vertex);
      }
      return std::pair<int, int>{state_net.at(vertex), state_node.at(vertex)};
    };
    // Unique input pin of the buffer driving ``downstream_net``, or -1.
    auto bufferInputPin = [&](int downstream_net) {
      if (downstream_net < 0 || downstream_net >= static_cast<int>(drivers.size())) {
        return -1;
      }
      if (base.buffer_input_pin_by_net.size() == drivers.size()) {
        return base.buffer_input_pin_by_net[downstream_net];
      }
      const int downstream_driver = drivers.at(downstream_net);
      if (downstream_driver < 0
          || downstream_driver >= static_cast<int>(pin2node.size())) {
        return -1;
      }
      const int buffer_node = pin2node.at(downstream_driver);
      int input_pin = -1;
      for (int pin = 0; pin < static_cast<int>(pin2node.size()); ++pin) {
        const int pin_net = pin2net.at(pin);
        if (pin2node.at(pin) != buffer_node || pin_net < 0
            || drivers.at(pin_net) == pin) {
          continue;
        }
        if (input_pin >= 0) {
          return -1;
        }
        input_pin = pin;
      }
      return input_pin;
    };
    // Walk the existing buffer chain from ``downstream_net`` up to the
    // buffer whose input sits on ``upstream_net``; a current tree may hold
    // several buffer levels below the planning partition.
    auto upstreamBoundaryPin = [&](int upstream_net, int downstream_net) {
      if (upstream_net < 0 || upstream_net >= static_cast<int>(drivers.size())) {
        return -1;
      }
      int current = downstream_net;
      for (std::size_t depth = 0; depth < drivers.size(); ++depth) {
        const int input_pin = bufferInputPin(current);
        if (input_pin < 0) {
          return -1;
        }
        const int input_net = pin2net.at(input_pin);
        if (input_net == upstream_net) {
          return input_pin;
        }
        current = input_net;
      }
      return -1;
    };
    const auto source_parent_identity =
        sourceVertex(partition, source_parent, "parent");
    const auto source_child_identity =
        sourceVertex(partition, source_child, "child");
    const bool source_identity_valid =
        source_parent_identity.first >= 0 && source_child_identity.first >= 0
        && source_parent_identity.first == source_child_identity.first
        && (!has_partition_provenance
            || (source_parent_identity.first == planning_net
                && findPartitionProvenance(partition_provenance, {partition, source_parent})
                    != partition_provenance.end()
                && findPartitionProvenance(partition_provenance, {partition, source_child})
                    != partition_provenance.end()));
    const int net = source_identity_valid ? source_parent_identity.first : planning_net;
    const auto state_it = working_by_net.find(net);
    const bool has_active_state = state_it != working_by_net.end();
    // Only this net's simulated ancestor splits can modify its scratch state.
    // Other nets remain borrowed read-only for physical provenance lookup.
    if (has_active_state && copied_states.insert(net).second) {
      state_it->second = std::make_shared<IncrementalTopologyState>(*state_it->second);
    }
    unchanged.net_id = net;
    unchanged.driver_pin_id = drivers.at(net);
    auto* state = has_active_state ? state_it->second.get() : &unchanged;
    auto downstreamVertex = [&](int vertex) {
      auto identity = sourceVertex(partition, vertex, "downstream vertex");
      if ((!has_partition_provenance || identity.first == net)
          && identity.first >= 0) {
        return identity;
      }
      std::pair<int, int> owner{-1, -1};
      if (has_partition_provenance) {
        for (const auto& item : partition_provenance) {
          if (item.first.second != vertex || item.second.first != net) {
            continue;
          }
          if (owner.first >= 0 && owner != item.second) {
            return std::pair<int, int>{-1, -1};
          }
          owner = item.second;
        }
      }
      return owner.first >= 0 ? owner : identity;
    };
    std::set<int> selected;
    bool cross_partition_boundary = false;
    for (const int pin : action["downstream_pins"].cast<std::vector<int>>()) {
      auto load_identity = downstreamVertex(pin);
      int selected_pin = pin;
      if (load_identity.first >= 0 && load_identity.first != net) {
        // A current-tree action may be planned on the input net of an
        // existing buffer while its DP load set contains terminals on the
        // buffer's output net.  Collapse that descendant partition to the
        // unique physical buffer input pin.  This preserves the active
        // partition boundary and gives resolveOverlaySplit the load identity
        // owned by the state being projected.
        selected_pin = upstreamBoundaryPin(net, load_identity.first);
        if (selected_pin >= 0) {
          load_identity = downstreamVertex(selected_pin);
          cross_partition_boundary = true;
        }
      }
      if (load_identity.first < 0 || load_identity.first != net) {
        std::ostringstream diagnostic;
        diagnostic << "downstream vertex has no unique planning-partition provenance"
                   << "; partition=" << partition
                   << "; planning_net=" << planning_net
                   << "; resolved_net=" << net
                   << "; vertex=" << pin
                   << "; source_parent=" << source_parent
                   << "; source_child=" << source_child
                   << "; source_child_state_net=" << source_child_identity.first
                   << "; source_child_state_node=" << source_child_identity.second
                   << "; load_state_net=" << load_identity.first
                   << "; load_state_node=" << load_identity.second
                   << "; boundary_pin=" << selected_pin;
        const auto provenance = findPartitionProvenance(partition_provenance, {partition, pin});
        if (provenance != partition_provenance.end()) {
          diagnostic << "; exact_partition_provenance="
                     << provenance->second.first << ':'
                     << provenance->second.second;
        } else {
          diagnostic << "; exact_partition_provenance=missing";
        }
        diagnostic << "; active_state_matches=";
        bool first_state = true;
        for (const auto& item : working_by_net) {
          const auto& candidate = *item.second;
          const bool node_load = candidate.node_loads.count(pin) != 0;
          const bool edge_endpoint = std::any_of(
              candidate.edges.begin(), candidate.edges.end(), [&](const auto& edge) {
                return edge.parent == pin || edge.child == pin;
              });
          if (!node_load && !edge_endpoint) {
            continue;
          }
          if (!first_state) {
            diagnostic << ',';
          }
          first_state = false;
          diagnostic << candidate.net_id << ':'
                     << (node_load ? "load" : "edge");
        }
        if (first_state) {
          diagnostic << "none";
        }
        throw py::value_error(diagnostic.str());
      }
      const int node = load_identity.second;
      bool buffer_input = false;
      for (const auto& buffer : state->buffers) {
        if (buffer.input_node != node) continue;
        const auto loads = terminalLoads(*state, buffer.output_node);
            selected.insert(loads.begin(), loads.end());
        buffer_input = true;
      }
      if (!buffer_input) {
        const bool has_wire_child = std::any_of(
            state->edges.begin(), state->edges.end(), [&](const auto& edge) {
              return !cellEdge(*state, edge) && edge.parent == node;
            });
        if (has_wire_child) {
          // A current-tree action may carry the source vertex at the head of
          // a downstream branch.  It is not itself a physical leaf load once
          // an earlier ancestor mutation has materialized the branch.  Use
          // the exact terminal-load closure so the projection remains tied
          // to one active wire partition.
          const auto loads = terminalLoads(*state, node);
          if (!loads.empty()) {
            selected.erase(pin);
            selected.insert(loads.begin(), loads.end());
          } else {
            selected.insert(selected_pin);
          }
        } else {
          selected.insert(selected_pin);
        }
      }
    }
    bool source_closure_expanded = false;
    if (has_active_state) {
      // Carrier source hints are mutable-state nodes only after provenance
      // resolution. Another Steiner realization can reuse their integer ids.
      std::vector<int> source_nodes;
      if (!has_partition_provenance) source_nodes.push_back(source_child);
      if (source_identity_valid) {
        source_nodes.push_back(source_child_identity.second);
      }
      const auto source_loads = sourceChildClosure(*state,
                                                  source_nodes,
                                                  selected);
      if (!source_loads.empty()
          && std::set<int>(source_loads.begin(), source_loads.end())
                 != selected) {
        selected.clear();
        selected.insert(source_loads.begin(), source_loads.end());
        source_closure_expanded = true;
      }
    }
    if (has_active_state && selected.size() == 1
        && selected.count(source_child) != 0) {
      // A tree action may identify the head of an existing branch as its
      // single downstream vertex.  When that vertex is the physical source
      // child, the incoming active edge owns the complete terminal closure,
      // including any load recorded at the branch head.  Resolve that exact
      // closure before asking the geometric projector to find the edge.
      const auto source_child_loads = terminalLoads(*state, source_child);
      if (!source_child_loads.empty()) {
        selected.clear();
        selected.insert(source_child_loads.begin(), source_child_loads.end());
      }
    }
    const std::vector<int> loads(selected.begin(), selected.end());
    int parent = -1;
    int child = -1;
    bool resolved_from_source = false;
    if (source_identity_valid && source_parent_identity.first == net) {
      parent = source_parent_identity.second;
      child = source_child_identity.second;
      if (!has_active_state) {
        // The first action on a net has no mutable native state yet.  The
        // partition provenance still maps the carrier ids to the base state
        // ids; active-edge validation happens when native apply creates that
        // state.
        resolved_from_source = true;
      } else {
        const auto edge_it = std::find_if(
            state->edges.begin(), state->edges.end(), [&](const auto& edge) {
              return edge.parent == parent && edge.child == child;
            });
        resolved_from_source = edge_it != state->edges.end()
                               && !cellEdge(*state, *edge_it)
                               && terminalLoads(*state, child) == loads;
      }
    } else if (!has_active_state) {
      // Legacy single-net actions may carry candidate vertices without the
      // complete-tree source identity.  Resolve those through the planning
      // partition while preserving the no-active-state contract above.
      const auto graph_parent_identity =
          sourceVertex(partition, graph_parent, "graph parent");
      const auto graph_child_identity =
          sourceVertex(partition, graph_child, "graph child");
      if (graph_parent_identity.first != net
          || graph_child_identity.first != net) {
        throw py::value_error(
            "planned graph edge has no unique planning-partition provenance");
      }
      parent = graph_parent_identity.second;
      child = graph_child_identity.second;
      resolved_from_source = true;
    }
    if (!resolved_from_source) {
      if (!has_active_state) {
        throw py::value_error(
            "source physical edge cannot be projected without an active native state");
      }
      py::dict probe;
      probe["downstream_pin_ids"] = loads;
      probe["x_dbu"] = action["x_dbu"];
      probe["y_dbu"] = action["y_dbu"];
      probe["projection_selected_loads"] = loads;
      probe["projection_source_child"] = source_child;
      probe["projection_source_child_identity"] = source_child_identity.second;
      probe["projection_source_closure_expanded"] = source_closure_expanded;
      const auto projected = resolveOverlaySplit(*state, probe);
      parent = py::cast<int>(projected["parent_node_id"]);
      child = py::cast<int>(projected["child_node_id"]);
      action["x_dbu"] = projected["x_dbu"];
      action["y_dbu"] = projected["y_dbu"];
      if (projected.contains("split_ratio")) {
        action["split_ratio"] = projected["split_ratio"];
      }
      if (projected.contains("resolution_source")) {
        action["resolution_source"] = projected["resolution_source"];
      }
      if (projected.contains("projection_distance_dbu")) {
        action["projection_distance_dbu"]
            = projected["projection_distance_dbu"];
      }
    }
    if (has_active_state) {
      // Resolve against the current working state.  The first ancestor split
      // may have changed the active edge that a descendant action needs.
      py::dict probe;
      probe["downstream_pin_ids"] = loads;
      probe["x_dbu"] = action["x_dbu"];
      probe["y_dbu"] = action["y_dbu"];
      probe["projection_selected_loads"] = loads;
      probe["projection_source_child"] = source_child;
      probe["projection_source_child_identity"] = source_child_identity.second;
      probe["projection_source_closure_expanded"] = source_closure_expanded;
      if (parent >= 0 && child >= 0) {
        probe["parent_node_id"] = parent;
        probe["child_node_id"] = child;
      }
      const auto projected = resolveOverlaySplit(*state, probe);
      parent = py::cast<int>(projected["parent_node_id"]);
      child = py::cast<int>(projected["child_node_id"]);
      action["x_dbu"] = projected["x_dbu"];
      action["y_dbu"] = projected["y_dbu"];
      if (projected.contains("split_ratio")) {
        action["split_ratio"] = projected["split_ratio"];
      }
      if (projected.contains("resolution_source")) {
        action["resolution_source"] = projected["resolution_source"];
      }
      if (projected.contains("projection_distance_dbu")) {
        action["projection_distance_dbu"]
            = projected["projection_distance_dbu"];
      }
    }
    physicalNames(action, pins, nets, *state, loads);
    action["planning_partition_id"] = partition;
    action["planning_original_net_id"] = planning_net;
    action["resolved_state_net_id"] = net;
    action["projection_had_active_state"] = has_active_state;
    action["projection_simulated_bundle_state"] = has_active_state;
    action["source_parent_node_id"] = source_parent;
    action["source_child_node_id"] = source_child;
    action["resolved_parent_node_id"] = parent;
    action["resolved_child_node_id"] = child;
    action["source_closure_expanded"] = source_closure_expanded;
    action["projection_cross_partition_boundary"] = cross_partition_boundary;
    if (!action.contains("resolution_source")) {
      action["resolution_source"] = resolved_from_source
                                         ? "native_source_provenance"
                                         : "native_partition_load_projection";
    }
    action["parent_node_id"] = parent; action["child_node_id"] = child;
    action["buffer_master_name"] = cells.at(py::cast<int>(action["buffer_cell_index"]));
    action["candidate_location_x_dbu"] = action["x_dbu"];
    action["candidate_location_y_dbu"] = action["y_dbu"];
    output.append(action);

    if (has_active_state) {
      py::dict simulation;
      simulation["action_id"] = simulation_action_id++;
      simulation["buffer_master_name"] = action["buffer_master_name"];
      simulation["parent_node_id"] = parent;
      simulation["child_node_id"] = child;
      simulation["split_ratio"] = action["split_ratio"];
      simulation["downstream_pin_ids"] = loads;
      simulation["x_dbu"] = action["x_dbu"];
      simulation["y_dbu"] = action["y_dbu"];
      std::string reason;
      py::dict simulation_result;
      if (!applyIncrementalBufferSplit(
              *state, simulation, reason, simulation_result)) {
        throw py::value_error(
            "current tree bundle simulation failed: " + reason);
      }
    }
  }
  return output;
}

py::list finalizeOverlayActions(
    const py::object& base,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions)
{
  struct Record {
    const IncrementalTopologyState* state;
    const IncrementalBuffer* buffer;
    int depth;
    int parent_action_id;
  };
  std::map<int, Record> records;
  for (const auto& state : states) {
    std::string reason;
    if (!state || !validateIncrementalTopology(*state, reason)) throw py::value_error(reason);
    std::map<int, std::vector<int>> children;
    std::map<int, const IncrementalBuffer*> inputs;
    std::map<int, int> parent_node;
    std::map<int, int> action_by_output_node;
    for (const auto& edge : state->edges) children[edge.parent].push_back(edge.child);
    for (const auto& edge : state->edges) parent_node[edge.child] = edge.parent;
    for (const auto& buffer : state->buffers) {
      inputs[buffer.input_node] = &buffer;
      action_by_output_node[buffer.output_node] = buffer.action_id;
    }
    std::function<void(int, int)> visit = [&](int node, int depth) {
      if (inputs.count(node)) {
        const auto* buffer = inputs.at(node);
        int parent_action_id = -1;
        // Use the live input node. A later upstream split can replace the
        // original parent edge while preserving the buffer's historical
        // original_parent field; the current input node follows the active
        // topology and therefore captures the real action ancestry.
        int ancestor = buffer->input_node;
        std::set<int> visited_ancestors;
        while (ancestor >= 0 && visited_ancestors.insert(ancestor).second) {
          const auto action = action_by_output_node.find(ancestor);
          if (action != action_by_output_node.end()
              && action->second != buffer->action_id) {
            parent_action_id = action->second;
            break;
          }
          const auto parent = parent_node.find(ancestor);
          if (parent == parent_node.end()) break;
          ancestor = parent->second;
        }
        records.emplace(buffer->action_id,
                        Record{state.get(), buffer, depth, parent_action_id});
      }
      for (const int child : children[node]) {
        const bool cell = inputs.count(node) && inputs.at(node)->output_node == child;
        visit(child, depth + (cell ? 1 : 0));
      }
    };
    visit(state->driver_pin_id, 0);
  }
  if (records.size() != actions.size()) throw py::value_error("commit ledger does not cover current native topology");
  const auto pins = carrierArray<std::string>(base, "pin_names");
  const auto nets = carrierArray<std::string>(base, "net_names");
  py::list output;
  for (const auto& item : actions) {
    py::dict action = item.cast<py::dict>().attr("copy")().cast<py::dict>();
    const auto& record = records.at(py::cast<int>(action["session_buffer_id"]));
    const auto loads = terminalLoads(*record.state, record.buffer->output_node);
    physicalNames(action, pins, nets, *record.state, loads);
    action["segment_tree_depth"] = record.depth;
    action["parent_action_id"] = record.parent_action_id;
    output.append(action);
  }
  return output;
}

}  // namespace rebuffer
