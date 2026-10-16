from types import SimpleNamespace

import pytest
import torch

from dreamplace.ops.buffer_insertion.real_design_adapter import (
    build_buffering_nets_from_pydb,
)
from dreamplace.ops.buffer_insertion.buffering_state_builder import (
    _native_selected_net_record_resolver,
    build_native_segment_count_packing_for_model,
)
from dreamplace.ops.buffer_insertion.segment_count_projection import (
    project_segment_count_state_to_candidates,
)
from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_packed_segment_count_state,
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing.segment_count_native_packer import (
    pack_segment_count_topology,
    select_packed_segment_count_inputs,
    select_packed_segment_count_inputs_native,
)
from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import (
    SegmentCountDynamicNetProvider,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.steiner_topo.pydb_topology import (
    build_edge_rc_by_net_from_rc_timing_topology,
)


def _fixture():
    flat_net2pin = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    flat_net2pin_start = torch.tensor([0, 1, 4], dtype=torch.int32)
    net2driver = torch.tensor([-1, 2], dtype=torch.int32)
    pin2node = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    topo = torch.tensor([0, 2, 4, 1, 3], dtype=torch.int32)
    topo_start = torch.tensor([0, 1, 5], dtype=torch.int32)
    pin_fa = torch.tensor([-1, 4, -1, 4, 2], dtype=torch.int32)
    node_x = torch.tensor([0.0, 200.0, 0.0, 100.0, 100.0], dtype=torch.float64)
    node_y = torch.tensor([0.0, 0.0, 0.0, 100.0, 0.0], dtype=torch.float64)
    node_x_dbu = torch.tensor([0.0, 2000.0, 0.0, 1000.0, 1000.0], dtype=torch.float64)
    node_y_dbu = torch.tensor([0.0, 0.0, 0.0, 1000.0, 0.0], dtype=torch.float64)
    pin_cap = torch.tensor([0.0, 1.1, 0.0, 1.3], dtype=torch.float64)
    return {
        "flat_net2pin": flat_net2pin,
        "flat_net2pin_start": flat_net2pin_start,
        "net2driver": net2driver,
        "pin2node": pin2node,
        "net_flat_topo_sort": topo,
        "net_flat_topo_sort_start": topo_start,
        "pin_fa": pin_fa,
        "node_x": node_x,
        "node_y": node_y,
        "node_x_dbu": node_x_dbu,
        "node_y_dbu": node_y_dbu,
        "pin_capacitance": pin_cap,
        "num_movable_nodes": 4,
        "num_terminals": 0,
        "dbu": 100.0,
        "scale_factor": 1.0,
        "r_unit": 2.0,
        "c_unit": 0.25,
        "max_repeater_count": 3,
    }


def _multi_valid_net_fixture():
    values = _fixture()
    values.update(
        {
            "flat_net2pin": torch.tensor([0, 1, 2, 3, 5, 6], dtype=torch.int32),
            "flat_net2pin_start": torch.tensor([0, 1, 4, 6], dtype=torch.int32),
            "net2driver": torch.tensor([-1, 2, 5], dtype=torch.int32),
            "pin2node": torch.arange(7, dtype=torch.int32),
            "net_flat_topo_sort": torch.tensor(
                [0, 2, 4, 1, 3, 5, 6],
                dtype=torch.int32,
            ),
            "net_flat_topo_sort_start": torch.tensor(
                [0, 1, 5, 7],
                dtype=torch.int32,
            ),
            "pin_fa": torch.tensor(
                [-1, 4, -1, 4, 2, -1, 5],
                dtype=torch.int32,
            ),
            "node_x": torch.tensor(
                [0.0, 200.0, 0.0, 100.0, 100.0, 300.0, 500.0],
                dtype=torch.float64,
            ),
            "node_y": torch.tensor(
                [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
                dtype=torch.float64,
            ),
            "node_x_dbu": torch.tensor(
                [0.0, 2000.0, 0.0, 1000.0, 1000.0, 3000.0, 5000.0],
                dtype=torch.float64,
            ),
            "node_y_dbu": torch.tensor(
                [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0],
                dtype=torch.float64,
            ),
            "pin_capacitance": torch.tensor(
                [0.0, 1.1, 0.0, 1.3, 0.0, 0.0, 1.6],
                dtype=torch.float64,
            ),
            "num_movable_nodes": 7,
        }
    )
    return values


def _python_reference(values):
    pin_count = int(values["pin2node"].numel())
    net_count = int(values["flat_net2pin_start"].numel()) - 1
    pin_names = ["n0", "sink1", "driver", "sink3"] + [
        f"pin{index}" for index in range(4, pin_count)
    ]
    net_names = ["single", "branch"] + [
        f"net{index}" for index in range(2, net_count)
    ]
    pydb = SimpleNamespace(
        flat_net2pin_map=values["flat_net2pin"].tolist(),
        flat_net2pin_start_map=values["flat_net2pin_start"].tolist(),
        net2driver_pin_map=values["net2driver"].tolist(),
        net_names=net_names,
        pin_names=pin_names,
        pin2node_map=values["pin2node"].tolist(),
        node_x=[0.0] * values["num_movable_nodes"],
        node_y=[0.0] * values["num_movable_nodes"],
        pin_offset_x=[0.0] * pin_count,
        pin_offset_y=[0.0] * pin_count,
        num_movable_nodes=values["num_movable_nodes"],
        num_terminals=0,
        num_terminal_NIs=0,
        dbu=100,
    )
    topology = {
        "topology_source": "live_timing_topology",
        "net_flat_topo_sort": values["net_flat_topo_sort"],
        "net_flat_topo_sort_start": values["net_flat_topo_sort_start"],
        "pin_fa": values["pin_fa"],
        "node_x": values["node_x"],
        "node_y": values["node_y"],
        "node_x_dbu": values["node_x_dbu"],
        "node_y_dbu": values["node_y_dbu"],
    }
    edge_rc, _ = build_edge_rc_by_net_from_rc_timing_topology(
        topology,
        dbu=values["dbu"],
        scale_factor=values["scale_factor"],
        r_unit=values["r_unit"],
        c_unit=values["c_unit"],
    )
    nets, _ = build_buffering_nets_from_pydb(
        pydb,
        topology=topology,
        edge_rc_by_net=edge_rc,
        pin_cap_by_pin_id={
            index: float(cap)
            for index, cap in enumerate(values["pin_capacitance"].tolist())
        },
    )
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
    )
    prepared = build_segment_count_timing_inputs(
        nets,
        state,
        dtype=values["pin_capacitance"].dtype,
    )
    return nets, state, prepared


def test_native_packer_matches_python_reference():
    values = _fixture()
    _reference_nets, reference_state, reference_prepared = _python_reference(values)

    result = pack_segment_count_topology(**values, num_threads=1)
    prepared = result["prepared_timing_inputs"]
    geometry = result["packed_segment_geometry"]

    for key, reference in reference_prepared.items():
        if key == "metadata":
            continue
        torch.testing.assert_close(prepared[key], reference)
    torch.testing.assert_close(geometry["segment_ids"], reference_state.segment_ids)
    torch.testing.assert_close(geometry["segment_net_id"], reference_state.segment_net_id)
    torch.testing.assert_close(geometry["segment_net_index"], reference_state.segment_net_index)
    torch.testing.assert_close(geometry["net_ids"], reference_state.net_ids)
    torch.testing.assert_close(geometry["parent_node_id"], reference_state.parent_node_id)
    torch.testing.assert_close(geometry["child_node_id"], reference_state.child_node_id)
    assert result["metadata"]["valid_net_count"] == 1
    assert result["metadata"]["affected_net_count"] == 1
    assert result["metadata"]["skipped_reasons"] == {"single_pin_net": 1}
    assert result["metadata"]["native_input_export_ms"] >= 0.0
    assert result["metadata"]["native_return_attach_ms"] >= 0.0


def test_native_packer_uses_zero_cap_for_io_sink():
    values = _fixture()
    values["pin2node"] = values["pin2node"].clone()
    values["pin2node"][1] = values["num_movable_nodes"] + values["num_terminals"]
    values["pin_capacitance"] = values["pin_capacitance"].clone()
    values["pin_capacitance"][1] = float("nan")

    result = pack_segment_count_topology(**values, num_threads=1)
    prepared = result["prepared_timing_inputs"]
    sink_index = int(
        torch.nonzero(
            prepared["flat_topo_node_id"] == 1,
            as_tuple=False,
        ).flatten().item()
    )

    assert result["metadata"]["valid_net_count"] == 1
    assert result["metadata"]["skipped_reasons"] == {"single_pin_net": 1}
    assert prepared["node_capacitance"][sink_index].item() == 0.0


def test_native_packer_keeps_zero_length_timing_edge_without_segment():
    values = _fixture()
    for coordinate_name in ("node_x", "node_y", "node_x_dbu", "node_y_dbu"):
        values[coordinate_name] = values[coordinate_name].clone()
        values[coordinate_name][1] = values[coordinate_name][4]

    result = pack_segment_count_topology(**values, num_threads=1)
    prepared = result["prepared_timing_inputs"]
    geometry = result["packed_segment_geometry"]
    edge_index = int(
        torch.nonzero(
            (prepared["edge_parent_node_id"] == 4)
            & (prepared["edge_child_node_id"] == 1),
            as_tuple=False,
        ).flatten().item()
    )

    assert result["metadata"]["edge_count"] == 3
    assert result["metadata"]["segment_count"] == 2
    assert prepared["edge_to_segment_id"][edge_index].item() == -1
    assert prepared["edge_resistance"][edge_index].item() == 0.0
    assert prepared["edge_capacitance"][edge_index].item() == 0.0
    torch.testing.assert_close(geometry["segment_ids"], torch.tensor([0, 1]))


def test_native_packer_quantized_zero_length_segment_has_finite_fractions():
    values = _fixture()
    for coordinate_name in ("node_x_dbu", "node_y_dbu"):
        values[coordinate_name] = values[coordinate_name].clone()
        values[coordinate_name][1] = values[coordinate_name][4]

    result = pack_segment_count_topology(**values, num_threads=1)
    prepared = result["prepared_timing_inputs"]
    edge_index = int(
        torch.nonzero(
            (prepared["edge_parent_node_id"] == 4)
            & (prepared["edge_child_node_id"] == 1),
            as_tuple=False,
        ).flatten().item()
    )
    segment_id = int(prepared["edge_to_segment_id"][edge_index].item())
    expected = torch.tensor([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=torch.float64)

    assert segment_id >= 0
    assert torch.isfinite(prepared["parent_cap_fraction"]).all()
    assert torch.isfinite(prepared["child_cap_fraction"]).all()
    assert torch.isfinite(prepared["segment_sub_resistance_fraction"]).all()
    torch.testing.assert_close(prepared["parent_cap_fraction"][segment_id], expected)
    torch.testing.assert_close(prepared["child_cap_fraction"][segment_id], expected)


def test_native_packer_rejects_invalid_topology_range_with_canonical_reason():
    values = _fixture()
    values["net_flat_topo_sort_start"] = values[
        "net_flat_topo_sort_start"
    ].clone()
    values["net_flat_topo_sort_start"][-1] += 1

    result = pack_segment_count_topology(**values, num_threads=1)

    assert result["metadata"]["valid_net_count"] == 0
    assert result["metadata"]["skipped_reasons"] == {
        "single_pin_net": 1,
        "invalid_topology_range": 1,
    }
    assert result["prepared_timing_inputs"]["net_ids"].numel() == 0


def test_native_packer_rejects_nonfinite_coordinate_with_canonical_reason():
    values = _fixture()
    values["node_x"] = values["node_x"].clone()
    values["node_x"][4] = float("nan")

    result = pack_segment_count_topology(**values, num_threads=1)

    assert result["metadata"]["valid_net_count"] == 0
    assert result["metadata"]["skipped_reasons"] == {
        "single_pin_net": 1,
        "missing_coordinate": 1,
    }
    assert result["prepared_timing_inputs"]["net_ids"].numel() == 0


def test_native_packer_is_thread_count_deterministic():
    values = _fixture()
    single = pack_segment_count_topology(**values, num_threads=1)
    parallel = pack_segment_count_topology(**values, num_threads=4)

    for section in ("prepared_timing_inputs", "packed_segment_geometry"):
        for key, value in single[section].items():
            torch.testing.assert_close(value, parallel[section][key])


def _model_input_fixture(values):
    placedb = SimpleNamespace(
        flat_net2pin_map=values["flat_net2pin"].numpy(),
        flat_net2pin_start_map=values["flat_net2pin_start"].numpy(),
        net2driver_pin_map=values["net2driver"].numpy(),
        pin2node_map=values["pin2node"].numpy(),
        num_movable_nodes=values["num_movable_nodes"],
        num_terminals=values["num_terminals"],
        dbu=values["dbu"],
        r_unit=values["r_unit"],
        c_unit=values["c_unit"],
    )
    data_collections = SimpleNamespace(
        buffering_timing_topology={
            "topology_source": "live_timing_topology",
            "net_flat_topo_sort": values["net_flat_topo_sort"],
            "net_flat_topo_sort_start": values["net_flat_topo_sort_start"],
            "pin_fa": values["pin_fa"],
            "node_x": values["node_x"],
            "node_y": values["node_y"],
            "node_x_dbu": values["node_x_dbu"],
            "node_y_dbu": values["node_y_dbu"],
        },
        buffering_timing_topology_dbu=values["dbu"],
        buffering_timing_topology_scale_factor=values["scale_factor"],
        buffering_timing_topology_r_unit=values["r_unit"],
        buffering_timing_topology_c_unit=values["c_unit"],
        inst_libcell_offset=torch.zeros(values["num_movable_nodes"], dtype=torch.long),
    )

    def pin_caps_op(_offset, _data_collections):
        pin_cap = values["pin_capacitance"]
        return pin_cap, pin_cap, pin_cap

    return SimpleNamespace(
        placedb=placedb,
        data_collections=data_collections,
        pin_caps_op=pin_caps_op,
    )


def test_model_input_adapter_uses_live_flat_tensor_owners():
    values = _fixture()
    model = _model_input_fixture(values)
    result = build_native_segment_count_packing_for_model(
        model,
        SimpleNamespace(scale_factor=values["scale_factor"]),
        max_repeater_count=values["max_repeater_count"],
        num_threads=1,
    )

    assert result["status"] == "ok"
    assert result["metadata"]["flat_input_owner"] == "SimpleNamespace"
    assert result["metadata"]["pin_cap_source"] == "current_pin_caps_op_rise_base"
    assert result["metadata"]["native_total_ms"] >= 0.0


@pytest.mark.parametrize("identity_source", ["native_net_type", "sta_clock_pin"])
def test_model_buffering_pack_excludes_clock_without_changing_timing_topology(identity_source):
    values = _multi_valid_net_fixture()
    model = _model_input_fixture(values)
    if identity_source == "native_net_type":
        model.placedb.net_names = [b"single", b"clk", b"data"]
        model.placedb.clock_net_names = ["clk"]
    else:
        model.data_collections.clock_pins = torch.tensor([1, 3])
        model.data_collections.pin2net_map = torch.tensor([0, 1, 1, 1, -1, 2, 2])
    before = {name: value.clone() for name, value in values.items() if torch.is_tensor(value)}
    result = build_native_segment_count_packing_for_model(
        model, SimpleNamespace(scale_factor=values["scale_factor"]),
        max_repeater_count=values["max_repeater_count"], num_threads=1,
    )
    assert result["packed"]["prepared_timing_inputs"]["net_ids"].tolist() == [2]
    assert result["packed"]["packed_segment_geometry"]["net_ids"].tolist() == [2]
    assert result["metadata"]["excluded_clock_net_ids"] == [1]
    for name, original in before.items():
        torch.testing.assert_close(values[name], original)


def test_packed_state_and_full_prepared_view_match_python_contract():
    values = _fixture()
    _reference_nets, reference_state, reference_prepared = _python_reference(values)
    packed = pack_segment_count_topology(**values, num_threads=1)
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
    )

    assert state.is_tensor_backed
    assert state.segment_rows is None
    assert state.segment_indices_for_nets([1]) == tuple(range(state.num_segments))
    record = state.segment_record(0)
    assert record["net_id"] == reference_state.segment_rows[0]["net_id"]
    assert record["parent_node_id"] == reference_state.segment_rows[0]["parent_node_id"]
    assert record["child_node_id"] == reference_state.segment_rows[0]["child_node_id"]
    assert state.summary["selected_row_materialization_count"] == 1

    selected = select_packed_segment_count_inputs(
        state.prepared_timing_inputs,
        state.packed_segment_geometry,
        active_net_ids={1},
        row_indices=state.segment_indices_for_nets([1]),
        dtype=state.z_param.dtype,
        device=state.z_param.device,
    )
    for key, reference in reference_prepared.items():
        if key == "metadata":
            continue
        torch.testing.assert_close(selected[key], reference)
    torch.testing.assert_close(
        selected["source_prepared_edge_index"],
        torch.arange(reference_prepared["edge_resistance"].numel()),
    )


def test_native_selected_net_record_preserves_segment_subtree_load_seed():
    values = _fixture()
    reference_nets, reference_state, _reference_prepared = _python_reference(values)
    packed = pack_segment_count_topology(**values, num_threads=1)
    placedb = SimpleNamespace(
        flat_net2pin_map=values["flat_net2pin"].tolist(),
        flat_net2pin_start_map=values["flat_net2pin_start"].tolist(),
        net_names=["single", "branch"],
        pin_names=["n0", "sink1", "driver", "sink3"],
    )
    resolver = _native_selected_net_record_resolver(
        SimpleNamespace(placedb=placedb),
        packed,
    )
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
        net_record_resolver=resolver,
    )
    target_index = int(
        torch.nonzero(state.child_node_id == 3, as_tuple=False).flatten().item()
    )
    reference_target_index = int(
        torch.nonzero(
            reference_state.child_node_id == 3,
            as_tuple=False,
        ).flatten().item()
    )
    with torch.no_grad():
        state.z_param[target_index] = 1.0
        reference_state.z_param[reference_target_index] = 1.0

    native_nets = state.net_records([1])
    assert "load_pin_id" not in native_nets[0]
    assert "load_pin_name" not in native_nets[0]
    native_projection = project_segment_count_state_to_candidates(native_nets, state)
    reference_projection = project_segment_count_state_to_candidates(
        reference_nets,
        reference_state,
    )

    assert native_projection["projected_candidate_count"] == 1
    assert native_projection["candidates"][0]["load_pin_id"] == 3
    assert native_projection["candidates"][0]["load_pin_name"] == "sink3"
    assert native_projection["candidates"][0]["downstream_pin_ids"] == [3]
    assert native_projection["candidates"] == reference_projection["candidates"]


def test_packed_active_net_view_matches_single_net_python_reference():
    values = _multi_valid_net_fixture()
    reference_nets, _reference_state, _reference_prepared = _python_reference(values)
    target_nets = [net for net in reference_nets if int(net["net_id"]) == 2]
    target_state = build_segment_count_state(
        target_nets,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
    )
    target_prepared = build_segment_count_timing_inputs(
        target_nets,
        target_state,
        dtype=values["pin_capacitance"].dtype,
    )

    packed = pack_segment_count_topology(**values, num_threads=2)
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
    )
    row_indices = state.segment_indices_for_nets([2])
    selected = select_packed_segment_count_inputs(
        state.prepared_timing_inputs,
        state.packed_segment_geometry,
        active_net_ids={2},
        row_indices=row_indices,
        dtype=state.z_param.dtype,
        device=state.z_param.device,
    )

    assert packed["metadata"]["valid_net_count"] == 2
    selected_reference = select_packed_segment_count_inputs(
        _reference_prepared, None, active_net_ids={2}, row_indices=row_indices,
        dtype=state.z_param.dtype, device=state.z_param.device,
    )
    assert selected_reference["metadata"] == selected["metadata"]
    torch.testing.assert_close(
        {key: value for key, value in selected_reference.items() if key != "metadata"},
        {key: value for key, value in selected.items() if key != "metadata"},
    )
    assert selected["metadata"]["topology_source"] == "native_packed_active_view"
    assert row_indices == (3,)
    for key, reference in target_prepared.items():
        if key == "metadata":
            continue
        torch.testing.assert_close(selected[key], reference)
    source_edges = selected["source_prepared_edge_index"]
    torch.testing.assert_close(
        state.prepared_timing_inputs["edge_parent_node_id"].index_select(
            0,
            source_edges,
        ),
        target_prepared["edge_parent_node_id"],
    )
    torch.testing.assert_close(
        state.prepared_timing_inputs["edge_child_node_id"].index_select(
            0,
            source_edges,
        ),
        target_prepared["edge_child_node_id"],
    )


def test_native_active_view_matches_python_selector_and_preserves_source_maps():
    values = _multi_valid_net_fixture()
    packed = pack_segment_count_topology(**values, num_threads=2)
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
    )
    expected_rows = state.segment_indices_for_nets([1, 2])
    reference = select_packed_segment_count_inputs(
        state.prepared_timing_inputs,
        state.packed_segment_geometry,
        active_net_ids={2, 1},
        row_indices=expected_rows,
        dtype=state.z_param.dtype,
        device=state.z_param.device,
    )
    selected = select_packed_segment_count_inputs_native(
        state.prepared_timing_inputs,
        state.packed_segment_geometry,
        active_net_ids=torch.tensor([2, 99, 1, 2], dtype=torch.long),
        dtype=state.z_param.dtype,
        device=state.z_param.device,
        num_threads=2,
    )

    for key, expected in reference.items():
        if key == "metadata":
            continue
        torch.testing.assert_close(selected[key], expected)
    torch.testing.assert_close(
        selected["source_prepared_net_positions"], torch.tensor([0, 1])
    )
    torch.testing.assert_close(
        selected["source_prepared_edge_index"],
        torch.arange(state.prepared_timing_inputs["edge_resistance"].numel()),
    )
    torch.testing.assert_close(
        selected["source_segment_row_index"], torch.tensor(expected_rows)
    )
    assert selected["metadata"]["topology_source"] == (
        "native_packed_active_view_selector"
    )


def test_native_active_view_ignores_valid_unaffected_and_unknown_net_ids():
    values = _multi_valid_net_fixture()
    packed = pack_segment_count_topology(**values, num_threads=1)
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=16,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=7.0,
        dtype=values["pin_capacitance"].dtype,
    )
    selected = select_packed_segment_count_inputs_native(
        state.prepared_timing_inputs,
        state.packed_segment_geometry,
        active_net_ids=torch.tensor([-1, 0, 999999], dtype=torch.long),
        dtype=state.z_param.dtype,
        device=state.z_param.device,
    )

    assert selected["net_ids"].numel() == 0
    assert selected["net_topo_start"].tolist() == [0]
    assert selected["edge_start"].tolist() == [0]
    assert selected["source_prepared_net_positions"].numel() == 0
    assert selected["source_prepared_edge_index"].numel() == 0
    assert selected["source_segment_row_index"].numel() == 0


def test_dynamic_provider_consumes_packed_prepared_state_without_rows():
    values = _fixture()
    packed = pack_segment_count_topology(**values, num_threads=1)
    state = build_packed_segment_count_state(
        packed,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        max_repeater_count=values["max_repeater_count"],
        initial_z=0.0,
        initial_bsu_index=1.0,
        dtype=values["pin_capacitance"].dtype,
    )
    table = torch.tensor([[0.1, 0.2]], dtype=state.z_param.dtype)
    provider = SegmentCountDynamicNetProvider(
        nets=(),
        segment_state=state,
        per_size_input_cap=table,
        per_size_delay=table,
        per_size_output_slew=table,
        backend="cpp_cuda_segment_transfer_explicit_autograd",
        profile_enabled=False,
        prepared_timing_inputs=state.prepared_timing_inputs,
    )

    row_indices, active_nets = provider._active_view({1})
    active_state = provider._active_segment_state(row_indices)
    selected = provider._prepared_inputs(
        {1},
        row_indices,
        active_nets,
        active_state,
    )

    assert provider.metadata["prepared_input_source"] == "native_packed"
    assert active_nets == [1]
    assert row_indices == list(range(state.num_segments))
    assert active_state.segment_rows == ()
    torch.testing.assert_close(selected["net_ids"], torch.tensor([1]))
    torch.testing.assert_close(
        selected["edge_to_segment_id"],
        state.prepared_timing_inputs["edge_to_segment_id"],
    )
