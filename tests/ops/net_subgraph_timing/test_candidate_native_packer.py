import torch

from dreamplace.ops.net_subgraph_timing.candidate_native_packer import (
    pack_candidate_topology,
    select_packed_candidate_inputs_native,
)
from dreamplace.ops.net_subgraph_timing.dynamic_provider import (
    RelaxedBufferDynamicNetProvider,
)
from dreamplace.ops.net_subgraph_timing.tensor_builder import (
    build_net_subgraph_timing_inputs,
)
from dreamplace.ops.net_subgraph_timing.timing_adapter import (
    build_relaxed_buffer_timing_payload,
)
from dreamplace.ops.buffer_insertion.buffering_state_builder import (
    build_buffering_state_for_model,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_discrete_candidate_state,
)

from test_segment_count_native_packer import (
    _fixture,
    _multi_valid_net_fixture,
    _python_reference,
)


def _candidate_values(values):
    return {
        "flat_net2pin": values["flat_net2pin"],
        "flat_net2pin_start": values["flat_net2pin_start"],
        "net2driver": values["net2driver"],
        "pin2node": values["pin2node"],
        "net_flat_topo_sort": values["net_flat_topo_sort"],
        "net_flat_topo_sort_start": values["net_flat_topo_sort_start"],
        "pin_fa": values["pin_fa"],
        "node_x": values["node_x"],
        "node_y": values["node_y"],
        "node_x_dbu": values["node_x_dbu"],
        "node_y_dbu": values["node_y_dbu"],
        "pin_capacitance": values["pin_capacitance"],
        "num_movable_nodes": values["num_movable_nodes"],
        "num_terminals": values["num_terminals"],
        "dbu": values["dbu"],
        "scale_factor": values["scale_factor"],
        "r_unit": values["r_unit"],
        "c_unit": values["c_unit"],
        "num_threads": 1,
    }


def test_candidate_native_packer_matches_python_expanded_topology():
    values = _fixture()
    reference_nets, _reference_state, _reference_prepared = _python_reference(values)
    candidates = [
        {
            "candidate_id": 7,
            "net_id": 1,
            "parent_node_id": 2,
            "child_node_ids": [4],
            "x_dbu": 25,
            "y_dbu": 25,
            "segment_split_ratio": 0.25,
            "synthetic_node_id": -1,
        },
        {
            "candidate_id": 8,
            "net_id": 1,
            "parent_node_id": 4,
            "child_node_ids": [1],
            "x_dbu": 150,
            "y_dbu": 0,
            "segment_split_ratio": 0.5,
            "synthetic_node_id": -2,
        },
    ]
    native = pack_candidate_topology(
        **_candidate_values(values),
        candidates=candidates,
    )
    reference = build_net_subgraph_timing_inputs(
        reference_nets,
        candidates=candidates,
        dtype=values["pin_capacitance"].dtype,
    )
    prepared = native["prepared_timing_inputs"]
    assert prepared["net_ids"].tolist() == reference["metadata"]["net_ids"]
    for native_key, reference_key in {
        "net_topo_start": "net_flat_topo_sort_start",
        "pin_fa": "pin_fa",
        "flat_pin_to_start": "flat_pin_to_start",
        "flat_pin_to": "flat_pin_to",
        "sink_node_id": "sink_node_id",
        "sink_net_index": "sink_net_index",
        "sink_pin_id": "sink_pin_id",
        "candidate_node_id": "candidate_node_id",
        "candidate_net_id": "candidate_net_id",
    }.items():
        assert prepared[native_key].tolist() == reference[reference_key].tolist(), native_key
    assert prepared["flat_topo_node_id"].tolist() == reference["metadata"][
        "compact_node_to_original"
    ]
    for key in (
        "edge_resistance",
        "edge_capacitance",
        "node_capacitance",
    ):
        assert prepared[key].tolist() == reference[key].tolist(), key


def test_candidate_native_packer_preserves_multiple_candidates_on_one_edge():
    values = _fixture()
    candidates = [
        {
            "candidate_id": 0,
            "net_id": 1,
            "parent_node_id": 4,
            "child_node_ids": [1],
            "x_dbu": 125,
            "y_dbu": 0,
            "segment_split_ratio": 0.25,
            "synthetic_node_id": -10,
        },
        {
            "candidate_id": 1,
            "net_id": 1,
            "parent_node_id": 4,
            "child_node_ids": [1],
            "x_dbu": 175,
            "y_dbu": 0,
            "segment_split_ratio": 0.75,
            "synthetic_node_id": -11,
        },
    ]
    native = pack_candidate_topology(
        **_candidate_values(values),
        candidates=candidates,
    )
    prepared = native["prepared_timing_inputs"]
    assert prepared["candidate_node_id"].numel() == 2
    assert prepared["candidate_node_id"].tolist()[0] < prepared["candidate_node_id"].tolist()[1]
    assert prepared["candidate_net_id"].tolist() == [1, 1]


def _two_net_candidate_payload():
    values = _multi_valid_net_fixture()
    reference_nets, _reference_state, _reference_prepared = _python_reference(values)
    candidates = [
        {
            "candidate_id": 7,
            "net_id": 1,
            "node_id": 4,
            "candidate_node_id": 4,
            "buffer_main_type_index": 7,
            "x_dbu": 1000,
            "y_dbu": 0,
        },
        {
            "candidate_id": 8,
            "net_id": 2,
            "node_id": 6,
            "candidate_node_id": 6,
            "buffer_main_type_index": 7,
            "x_dbu": 5000,
            "y_dbu": 0,
        },
    ]
    packed = pack_candidate_topology(
        **_candidate_values(values),
        candidates=candidates,
    )
    state = build_discrete_candidate_state(
        candidates,
        buffer_main_type_index=7,
        legal_buffer_count=2,
        fixed_bsu_index=1,
        dtype=values["pin_capacitance"].dtype,
    )
    state.candidate_node_id = packed["prepared_timing_inputs"][
        "candidate_node_id"
    ].to(dtype=state.candidate_node_id.dtype)
    payload = build_relaxed_buffer_timing_payload(
        reference_nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]], dtype=values["pin_capacitance"].dtype
        ),
        per_size_delay=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=values["pin_capacitance"].dtype
        ),
        per_size_output_slew=torch.tensor(
            [[0.5, 0.6], [0.7, 0.8]], dtype=values["pin_capacitance"].dtype
        ),
        prebuilt_inputs=packed["prepared_timing_inputs"],
    )
    return payload, state


def test_candidate_native_active_view_matches_python_reference():
    payload, state = _two_net_candidate_payload()
    native_provider = RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
    )
    reference_payload = dict(payload)
    reference_payload["metadata"] = dict(payload["metadata"])
    reference_payload["metadata"]["topology_source"] = "python_reference_test"
    reference_provider = RelaxedBufferDynamicNetProvider(
        static_payload=reference_payload,
        buffer_state=state,
    )

    native_view = native_provider._active_view(torch.tensor([2, 99, 2]))
    reference_view = reference_provider._active_view({2})

    for name in (
        "net_indices_cpu",
        "candidate_indices_cpu",
        "sink_indices_cpu",
        "candidate_node_id_cpu",
        "sink_pin_id_cpu",
        "sink_net_id_cpu",
    ):
        torch.testing.assert_close(native_view[name], reference_view[name])
    for name, expected in reference_view["static"].items():
        torch.testing.assert_close(native_view["static"][name], expected)
    assert native_provider.metadata["active_view_selector_used"] == "native"
    assert native_provider.metadata["native_active_view_selector_count"] == 1


def test_candidate_native_active_view_preserves_global_candidate_gradient_row():
    payload, state = _two_net_candidate_payload()
    provider = RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
    )

    result = provider._run_lane(
        driver_arrival=torch.tensor([0.0], dtype=state.activation_param.dtype),
        driver_slew=torch.tensor([0.1], dtype=state.activation_param.dtype),
        active_net_ids={2},
    )
    grad, = torch.autograd.grad(result["sink_arrival"].sum(), state.activation_param)

    assert float(grad[0].abs()) == 0.0
    assert float(grad[1].abs()) > 0.0


def test_candidate_native_selector_ignores_unaffected_and_unknown_net_ids():
    payload, _state = _two_net_candidate_payload()
    selected = select_packed_candidate_inputs_native(
        payload,
        active_net_ids=torch.tensor([-1, 0, 99], dtype=torch.long),
    )

    assert selected["net_ids"].numel() == 0
    assert selected["net_indices_cpu"].numel() == 0
    assert selected["candidate_indices_cpu"].numel() == 0
    assert selected["sink_indices_cpu"].numel() == 0
    assert selected["static"]["net_flat_topo_sort_start"].tolist() == [0]
    assert selected["static"]["flat_pin_to_start"].tolist() == [0]


def test_candidate_state_builder_uses_native_topology_when_flat_inputs_exist():
    values = _fixture()
    reference_nets, _reference_state, _reference_prepared = _python_reference(values)
    candidates = [
        {
            "candidate_id": 7,
            "net_id": 1,
            "parent_node_id": 2,
            "child_node_ids": [4],
            "x_dbu": 25,
            "y_dbu": 25,
            "segment_split_ratio": 0.25,
            "synthetic_node_id": -1,
            "buffer_main_type_index": 7,
        },
        {
            "candidate_id": 8,
            "net_id": 1,
            "parent_node_id": 4,
            "child_node_ids": [1],
            "x_dbu": 150,
            "y_dbu": 0,
            "segment_split_ratio": 0.5,
            "synthetic_node_id": -2,
            "buffer_main_type_index": 7,
        },
    ]
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
    placedb = type(
        "PlaceDBFixture",
        (),
        {
            "flat_net2pin_map": values["flat_net2pin"],
            "flat_net2pin_start_map": values["flat_net2pin_start"],
            "net2driver_pin_map": values["net2driver"],
            "pin2node_map": values["pin2node"],
            "net_names": ["single", "branch"],
            "pin_names": ["n0", "sink1", "driver", "sink3"],
            "num_movable_nodes": values["num_movable_nodes"],
            "num_terminals": values["num_terminals"],
            "dbu": values["dbu"],
            "r_unit": values["r_unit"],
            "c_unit": values["c_unit"],
        },
    )()
    data = type(
        "DataFixture",
        (),
        {
            "buffering_nets": reference_nets,
            "buffering_timing_topology": topology,
            "buffering_timing_topology_dbu": values["dbu"],
            "buffering_timing_topology_scale_factor": values["scale_factor"],
            "buffering_timing_topology_r_unit": values["r_unit"],
            "buffering_timing_topology_c_unit": values["c_unit"],
            "buffering_candidate_rows": candidates,
            "buffering_buffer_library": {
                0: {"input_cap": 0.2, "delay": 0.1, "output_slew": 0.1},
                1: {"input_cap": 0.4, "delay": 0.2, "output_slew": 0.2},
            },
            "buffer_main_type_index": 7,
            "inst_libcell_offset": values["pin_capacitance"].new_zeros(4),
            "pos": [values["node_x"].new_zeros(4)],
        },
    )()

    def pin_caps_op(_offset, _data):
        pin_cap = values["pin_capacitance"]
        return pin_cap, pin_cap, pin_cap

    model = type(
        "ModelFixture",
        (),
        {
            "placedb": placedb,
            "data_collections": data,
            "pin_caps_op": staticmethod(pin_caps_op),
            "op_collections": type(
                "OpsFixture",
                (),
                {"timing_propagation_op": type("TimingFixture", (), {"device": "cpu"})()},
            )(),
        },
    )()
    config = type(
        "ConfigFixture",
        (),
        {
            "mode": "candidate",
            "candidate_strategy": "discrete_net_gradient",
            "fixed_bsu_index": 1,
        },
    )()
    params = type("ParamsFixture", (), {"buffering_mode": "candidate"})()

    result = build_buffering_state_for_model(model, params, config)

    assert result.status == "ok"
    assert result.metadata["candidate_topology_builder_backend_used"] == "native_cpp"
    assert result.buffer_optimization_state.candidate_node_id.tolist() == [1, 3]
    assert result.buffer_relaxed_timing_payload["net_flat_topo_sort"].numel() == 6
    assert result.buffer_relaxed_timing_payload["metadata"]["net_ids"] == [1]
