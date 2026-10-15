"""Retained buffered trees coexist with fresh FLUTE trees and live gradients."""

import pytest
import torch

from dreamplace.ops.buffer_insertion.segment_count_state import build_segment_count_state
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import build_segment_count_timing_inputs
from dreamplace.ops.buffer_insertion.virtual_cell_density import build_segment_endpoint_mapping
from dreamplace.ops.steiner_topo.steiner_topo import SteinerTopo


def make_topology():
    topo = SteinerTopo(
        torch.arange(6, dtype=torch.int32),
        torch.tensor([0, 3, 6], dtype=torch.int32),
        deterministic_flag=True,
    )
    pos = torch.tensor(
        [0., 1., 2., 10., 11., 12., 0., 0., 0., 0., 2., 0.],
        dtype=torch.float64,
    )
    topo.rebuild_tree(pos)
    return topo, pos


def test_frozen_tree_rebases_while_unbuffered_tree_refreshes_and_geometry_moves():
    topo, pos = make_topology()
    before_parents = topo.pin_fa.clone()
    before_order = topo.net_flat_topo_sort.clone()
    before_starts = topo.net_steiner_start.clone()
    before_relations = (topo.pin_relate_x.clone(), topo.pin_relate_y.clone())
    assert (before_starts[1:] - before_starts[:-1]).tolist() == [0, 1]
    topo.freeze_nets([1])
    vertices = before_order[3:].tolist()
    children = {vertex: [] for vertex in vertices}
    edges = {}
    for vertex in vertices[1:]:
        parent = int(before_parents[vertex])
        children[parent].append(vertex)
        edges[(parent, vertex)] = {'r': 1., 'c': .01}
    net = {'net_id': 1, 'driver_pin_id': 3,
           'coordinates': {vertex: (float(topo.newx[vertex]), float(topo.newy[vertex]))
                           for vertex in vertices},
           'rc_tree': {'root_node_id': 3, 'children_by_node': children, 'edge_rc': edges,
                       'sink_nodes': [4, 5], 'node_cap': {4: .02, 5: .02}}}
    state = build_segment_count_state([net], buffer_main_type_index=0, legal_buffer_count=2,
                                      max_repeater_count=3, initial_z=1., fixed_bsu_index=1)
    state.prepared_timing_inputs = build_segment_count_timing_inputs(
        [net], state, dtype=torch.float64, device='cpu',
    )
    old_segment_parents = state.parent_node_id.clone()
    old_segment_children = state.child_node_id.clone()
    topo.frozen_nets.segment_state = state
    # Net 0 gains a Steiner vertex; net 1 would lose its vertex if re-extracted.
    pos[7] = 3.
    pos[10] = 0.
    pos[4] += 0.25
    fresh, _ = make_topology()
    fresh.rebuild_tree(pos)
    assert (fresh.net_steiner_start[1:] - fresh.net_steiner_start[:-1]).tolist() == [1, 0]

    topo.rebuild_tree(pos)
    assert (topo.net_steiner_start[1:] - topo.net_steiner_start[:-1]).tolist() == [1, 1]
    assert topo.frozen_nets.net_ids == (1,)
    # The entire frozen tree, including currently unbuffered branches, is retained.
    old_vertices = before_order[3:]
    mapped = topo.frozen_nets.remap(old_vertices)
    assert torch.equal(topo.net_flat_topo_sort[4:], mapped)
    old_parents = before_parents[old_vertices.long()]
    expected_parents = old_parents.clone()
    nonroot = old_parents >= 0
    expected_parents[nonroot] = topo.frozen_nets.remap(old_parents[nonroot])
    assert torch.equal(topo.pin_fa[mapped.long()], expected_parents)
    assert torch.equal(topo.net_flat_topo_sort[:4], fresh.net_flat_topo_sort[:4])
    assert topo.frozen_nets.vertex_map.tolist() == [0, 1, 2, 3, 4, 5, 7]
    assert torch.equal(state.parent_node_id, topo.frozen_nets.remap(old_segment_parents))
    assert torch.equal(state.child_node_id, topo.frozen_nets.remap(old_segment_children))
    assert state.z_param.tolist() == [1., 1., 1.]
    density_parents, density_children = build_segment_endpoint_mapping(
        state.prepared_timing_inputs, state.segment_ids,
    )
    assert torch.equal(density_parents, state.parent_node_id)
    assert torch.equal(density_children, state.child_node_id)

    pos.requires_grad_(True)
    x, y = topo(pos)
    for axis, coordinates in enumerate((x, y)):
        assert torch.equal(
            coordinates[mapped.long()],
            pos[axis * 6 + before_relations[axis][old_vertices.long()].long()],
        )
    # A virtual midpoint force reaches physical pin witnesses after rebasing.
    child = mapped[-1].long()
    parent = topo.pin_fa[child].long()
    energy = (0.5 * (x[parent] + x[child])).square()
    gradient = torch.autograd.grad(energy, pos)[0]
    expected = torch.zeros_like(pos)
    expected[topo.pin_relate_x[parent].long()] += 0.5 * (x[parent] + x[child])
    expected[topo.pin_relate_x[child].long()] += 0.5 * (x[parent] + x[child])
    torch.testing.assert_close(gradient, expected)
    # Refresh again: mapping applies to the immediately previous domain.
    topo.rebuild_tree(pos.detach())
    assert torch.equal(topo.frozen_nets.remap(mapped), mapped)


def test_invalid_retained_tree_is_rejected_before_cache_publication():
    topo, pos = make_topology()
    topo.freeze_nets([1])
    generation = topo.topology_generation
    parents = topo.pin_fa
    parents[4] = 0  # Cross-net reference cannot be assigned a retained identity.
    with pytest.raises(RuntimeError, match="another net"):
        topo.rebuild_tree(pos)
    assert topo.topology_generation == generation
    assert topo.pin_fa is parents
