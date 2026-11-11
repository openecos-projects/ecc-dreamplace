// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Rebuild timing graph indices from physical PyPlaceDB without running STA.

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"

#include "pybind11/stl.h"


namespace rebuffer {

struct TimingCarrierView
{
  pybind11::dict values;
  std::vector<std::string> node_names;
  std::vector<std::string> pin_names;
  std::vector<std::string> net_names;
  std::vector<int> old_node_ids;
  std::vector<int> old_pin_ids;
  std::vector<int> pin2node;
  std::vector<int> cell_by_node;
  std::vector<int> cell_arc_start;
  std::vector<std::vector<int>> net_pins;
  std::unordered_map<std::string, int> pin_ids;
};

template <typename T>
std::vector<T> carrierArray(const pybind11::object& source, const char* name)
{
  return source.attr(name).cast<std::vector<T>>();
}

inline int carrierIndex(int index, const std::vector<int>& mapping, const char* field)
{
  if (index < 0 || index >= static_cast<int>(mapping.size()) || mapping[index] < 0) {
    throw pybind11::value_error(std::string("unmapped timing identity: ") + field);
  }
  return mapping[index];
}

void pruneBidirectionalPadNetArcs(
    int pin_count,
    const std::vector<char>& has_cell_arc,
    const std::vector<int>& net_arc_count,
    std::vector<std::array<int, 2>>& net_arcs,
    std::vector<int>& net_starts);
void compileCarrierGraph(const pybind11::object& base, TimingCarrierView& carrier);
pybind11::dict rebuildTimingCarrier(const pybind11::object& base,
                                   const pybind11::object& physical);

}  // namespace rebuffer

// --- rebuffer_carrier_projection.hh ---
// Owned placement/tree projection shared by Python export and native planning.

#include <string>
#include <utility>
#include <vector>

#include "rebuffer_repair.hh"


namespace rebuffer {

// Sorted contiguous rows preserve the complete (partition, vertex) domain
// without allocating one tree-map node per Steiner vertex on every snapshot.
using PartitionProvenance = std::vector<
    std::pair<std::pair<int, int>, std::pair<int, int>>>;

void normalizePartitionProvenance(PartitionProvenance& rows);
PartitionProvenance::const_iterator findPartitionProvenance(
    const PartitionProvenance& rows, const std::pair<int, int>& key);

struct OverlayActionProjection
{
  std::vector<int> original_net_by_net, state_node_by_vertex, state_net_by_vertex;
  PartitionProvenance state_provenance;
  bool has_partition_provenance = true;
};

struct ProjectedBufferIdentity
{
  int action_id, cell, node, input_pin, output_pin, output_net;
};

struct OverlayCarrierProjection
{
  DesignView design;
  SteinerView steiner;
  std::vector<std::string> node_names, node_master_names, net_names, pin_direct;
  std::vector<double> node_size_x, node_size_y;
  std::vector<std::vector<int>> net_pins, topologies, children;
  std::vector<ProjectedBufferIdentity> buffers;
  OverlayActionProjection actions;
};

}  // namespace rebuffer

// --- rebuffer_incremental_carrier.hh ---

namespace rebuffer {

// Rebind one mutable overlay carrier from the preceding carrier snapshot.
// This does not run STA or rebuild Steiner; it only transports graph arrays.
pybind11::dict rebuildTimingCarrierIncremental(
    const pybind11::object& base,
    const pybind11::object& previous,
    const pybind11::object& physical);

}  // namespace rebuffer

// --- rebuffer_overlay_inputs.hh ---
// Immutable PyPlaceDB metadata imported once for a native repair campaign.

#include <string>
#include <unordered_set>
#include <vector>


namespace rebuffer {

struct OverlayActionInputs
{
  std::vector<std::string> cells, pins, nets;
  std::vector<int> pin2node, pin2net, drivers;
  std::vector<int> buffer_input_pin_by_net;
};

struct OverlayCarrierInputs
{
  struct Arc
  {
    std::string source, sink;
    int sense;
  };
  OverlayActionInputs identity;
  int dbu, num_terminals, num_terminal_nis;
  bool has_rebuffer_prefix = false;
  std::vector<std::string> nodes, masters, pin_direct;
  std::unordered_set<std::string> node_name_set, net_name_set;
  std::vector<int> cell_arc_start;
  std::vector<Arc> library_arcs;
  std::vector<double> pin_x, pin_y, node_x, node_y, width, height, lib_width, lib_height;
};

OverlayActionInputs overlayActionInputs(const pybind11::object& base);
OverlayCarrierInputs overlayCarrierInputs(const pybind11::object& base);

}  // namespace rebuffer

// --- rebuffer_overlay_carrier.hh ---
#include "rebuffer_incremental_topology.hh"


namespace rebuffer {

OverlayCarrierProjection buildOverlayCarrierProjection(
    const OverlayCarrierInputs& base,
    const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions,
    double r_unit,
    double c_unit);

OverlayCarrierProjection buildOverlayCarrierProjection(
    const pybind11::object& base,
    const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions,
    double r_unit,
    double c_unit);

// Materialize the current electrical partitions without rebuilding Steiner.
pybind11::dict exportOverlayCarrier(
    const pybind11::object& base,
    const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions,
    double r_unit = 0.914,
    double c_unit = 1.89e-4);

}  // namespace rebuffer

// --- rebuffer_overlay_actions.hh ---

namespace rebuffer {
pybind11::list projectOverlayActionsFromProjection(
    const OverlayActionInputs& base, const OverlayActionProjection& projection,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);

pybind11::dict resolveOverlaySplit(const IncrementalTopologyState& state, const pybind11::dict& action);
pybind11::list projectOverlayActions(
    const pybind11::object& base, const pybind11::dict& payload,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);
pybind11::list projectOverlayActionsFromProjection(
    const pybind11::object& base, const OverlayActionProjection& projection,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);
pybind11::list finalizeOverlayActions(
    const pybind11::object& base,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);
}  // namespace rebuffer
