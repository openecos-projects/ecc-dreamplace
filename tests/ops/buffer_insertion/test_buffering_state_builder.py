from types import SimpleNamespace

import torch


def _fixture():
    nets = [
        {
            "net_id": 10,
            "driver_pin_id": 0,
            "net_pin_ids": [0, 1],
            "undirected_edges": [(0, 1)],
            "coordinates": {0: (0, 0), 1: (2, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 1.0, "c": 0.0}},
                "node_cap": {0: 0.0, 1: 1.0},
                "sink_nodes": [1],
            },
        }
    ]
    candidates = [
        {
            "candidate_id": 100,
            "net_id": 10,
            "candidate_node_id": 1,
            "x_dbu": 100,
            "y_dbu": 0,
            "buffer_main_type_index": 7,
            "bu": 0.05,
            "bsu": 0,
        }
    ]
    buffer_library = {
        0: {
            "master_name": "BUFX2H7H",
            "input_cap": 0.4,
            "delay": 2.0,
            "output_slew": 0.3,
        },
        1: {
            "master_name": "BUFX4H7H",
            "input_cap": 0.8,
            "delay": 3.0,
            "output_slew": 0.2,
        },
    }
    return nets, candidates, buffer_library


def test_state_builder_prefers_live_timing_device_over_stale_position_device():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        _buffer_timing_device,
    )

    model = SimpleNamespace(
        op_collections=SimpleNamespace(
            timing_propagation_op=SimpleNamespace(device=torch.device("cuda:1"))
        )
    )
    data_collections = SimpleNamespace(
        pos=[torch.nn.Parameter(torch.tensor([0.0]))]
    )

    assert _buffer_timing_device(model, data_collections) == torch.device("cuda:1")


def test_state_builder_builds_payload_without_backward_dependency():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_candidate_rows=candidates,
            buffering_buffer_library=buffer_library,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    params = SimpleNamespace(buffering_mode="candidate")
    config = SimpleNamespace(mode="candidate")

    result = build_buffering_state_for_model(model, params, config)

    assert result.status == "ok"
    assert result.candidate_count == 1
    assert result.mode == "candidate"
    assert result.buffer_optimization_state.bu_logits.grad is None
    assert result.buffer_relaxed_timing_payload["metadata"]["candidate_count"] == 1
    assert result.metadata["state_source"] == "data_collections_buffering_inputs"
    assert result.buffer_relaxed_timing_payload["metadata"]["net_ids"] == [10]
    assert result.buffer_relaxed_timing_payload["metadata"][
        "compact_node_to_original"
    ]
    assert "synthetic_segment_splits" not in result.metadata
    assert "original_node_to_compact" not in result.metadata
    assert result.metadata["omitted_runtime_metadata"] == {}


def test_state_builder_candidate_mode_reports_state_kind():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_candidate_rows=candidates,
            buffering_buffer_library=buffer_library,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(buffering_mode="candidate"),
        SimpleNamespace(mode="candidate"),
    )

    assert result.status == "ok"
    assert result.metadata["state_kind"] == "candidate"
    assert result.candidate_count > 0
    assert hasattr(result.buffer_optimization_state, "bu_logits")
    assert hasattr(result.buffer_optimization_state, "bsu_index_param")


def test_state_builder_candidate_discrete_strategy_builds_exact_zero_state():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_candidate_rows=candidates,
            buffering_buffer_library=buffer_library,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    config = SimpleNamespace(
        mode="candidate",
        candidate_strategy="discrete_net_gradient",
        fixed_bsu_index=1,
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(buffering_mode="candidate"),
        config,
    )

    assert result.status == "ok"
    assert result.metadata["candidate_strategy"] == "discrete_net_gradient"
    assert result.metadata["fixed_bsu_index"] == 1
    state = result.buffer_optimization_state
    assert not hasattr(state, "bu_logits")
    assert state.activation_param.tolist() == [0.0]
    assert state.bsu_index().tolist() == [1.0]


def test_state_builder_resolves_auto_x4_to_legal_buffer_master():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_candidate_rows=candidates,
            buffering_buffer_library=buffer_library,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    config = SimpleNamespace(
        mode="candidate",
        candidate_strategy="discrete_net_gradient",
        fixed_buffer_master="auto:x4",
        fixed_bsu_index=None,
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(
            buffering_mode="candidate",
            buffering_fixed_buffer_master="auto:x4",
        ),
        config,
    )

    assert result.status == "ok"
    assert result.metadata["fixed_buffer_master"] == "BUFX4H7H"
    assert result.metadata["fixed_bsu_index"] == 1
    assert result.metadata["fixed_buffer_selection_source"] == "auto_x4"
    assert result.buffer_optimization_state.bsu_index().tolist() == [1.0]


def test_state_builder_explicit_bsu_index_overrides_default_master_selector():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_candidate_rows=candidates,
            buffering_buffer_library=buffer_library,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    params = SimpleNamespace(
        buffering_mode="candidate",
        buffering_fixed_buffer_master="auto:x4",
        buffering_fixed_bsu_index=0,
        _buffering_fixed_bsu_index_explicit=True,
    )
    config = SimpleNamespace(
        mode="candidate",
        candidate_strategy="discrete_net_gradient",
        fixed_buffer_master="auto:x4",
        fixed_bsu_index=0,
    )

    result = build_buffering_state_for_model(model, params, config)

    assert result.status == "ok"
    assert result.metadata["fixed_buffer_master"] == "BUFX2H7H"
    assert result.metadata["fixed_bsu_index"] == 0
    assert result.metadata["fixed_buffer_selection_source"] == "bsu_index"


def test_state_builder_segment_mode_builds_segment_count_state():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_buffer_library=buffer_library,
            buffer_main_type_index=7,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(
            buffering_mode="segment",
            buffering_max_repeaters_per_segment=3,
            buffering_segment_count_z_init=0.25,
            buffering_initial_bsu_index=1.0,
            buffering_fixed_buffer_master="BUFX4H7H",
        ),
        SimpleNamespace(
            mode="segment",
            fixed_buffer_master="BUFX4H7H",
            fixed_bsu_index=None,
        ),
    )

    assert result.status == "ok"
    assert result.mode == "segment"
    assert result.metadata["state_kind"] == "segment_count"
    assert result.buffer_optimization_state.z_param is not None
    assert result.buffer_optimization_state.bsu_index_param is not None
    assert result.metadata["segment_count"] == 1
    assert result.metadata["full_design_scope"] is True
    assert result.metadata["fixed_buffer_master"] == "BUFX4H7H"
    assert result.metadata["fixed_bsu_index"] == 1
    assert result.metadata["fixed_buffer_selection_source"] == "exact_master"
    assert result.buffer_optimization_state.fixed_bsu_index == 1
    assert result.buffer_optimization_state.bsu_index_param.tolist() == [1.0]
    assert result.buffer_relaxed_timing_payload["metadata"]["state_kind"] == "segment_count"


def test_segment_state_builder_runtime_profile_accounts_common_and_state_work():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_buffer_library=buffer_library,
            buffer_main_type_index=7,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(
            buffering_mode="segment",
            buffering_max_repeaters_per_segment=3,
            buffering_segment_count_z_init=0.0,
            buffering_initial_bsu_index=1.0,
            buffering_runtime_profile=True,
        ),
        SimpleNamespace(mode="segment"),
    )

    profile = result.metadata["state_build_runtime_profile"]
    assert result.status == "ok"
    assert profile["common_inputs_build_ms"] >= profile["common_inputs_accounted_ms"]
    assert profile["common_inputs_unattributed_ms"] >= 0.0
    assert profile["segment_state_build_ms"] >= 0.0
    assert profile["per_segment_size_table_build_ms"] >= 0.0
    assert profile["segment_relaxed_timing_payload_build_ms"] >= 0.0
    assert profile["segment_size_table_elements"] == 6
    assert profile["segment_size_table_bytes"] == 24
    assert profile["segment_rc_tree_node_count"] == 2
    assert profile["segment_rc_tree_edge_count"] == 1
    assert (
        result.buffer_relaxed_timing_payload["metadata"][
            "state_build_runtime_profile"
        ]
        is profile
    )


def test_cuda_segment_state_builder_uses_one_shared_library_size_row():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, buffer_library = _fixture()
    second = dict(nets[0])
    second["net_id"] = 11
    second["driver_pin_id"] = 2
    second["coordinates"] = {2: (0, 0), 3: (2, 0)}
    second["rc_tree"] = {
        "root_node_id": 2,
        "children_by_node": {2: [3], 3: []},
        "edge_rc": {(2, 3): {"r": 1.0, "c": 0.0}},
        "node_cap": {2: 0.0, 3: 1.0},
        "sink_nodes": [3],
    }
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=[nets[0], second],
            buffering_buffer_library=buffer_library,
            buffer_main_type_index=7,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(
            buffering_mode="segment",
            buffering_max_repeaters_per_segment=3,
            buffering_segment_count_timing_backend=(
                "cpp_cuda_segment_transfer_explicit_autograd"
            ),
        ),
        SimpleNamespace(mode="segment"),
    )

    payload = result.buffer_relaxed_timing_payload
    assert result.metadata["segment_size_table_layout"] == "shared_library_row"
    assert result.buffer_optimization_state.z_param.numel() == 2
    assert payload["per_size_input_cap"].shape == (1, 2)
    assert payload["per_size_delay"].shape == (1, 2)
    assert payload["per_size_output_slew"].shape == (1, 2)


def test_segment_state_exposes_compact_device_net_index_for_scheduler():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_buffer_library=buffer_library,
            buffer_main_type_index=7,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(buffering_mode="segment", buffering_fixed_bsu_index=1),
        SimpleNamespace(mode="segment"),
    )

    state = result.buffer_optimization_state
    assert state.net_ids.tolist() == [10]
    assert state.segment_net_index.tolist() == [0]
    assert state.segment_net_index.device == state.z_param.device


def test_state_builder_segment_z_param_is_free_count_not_probability():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, buffer_library = _fixture()
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_buffer_library=buffer_library,
            buffer_main_type_index=7,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(
            buffering_mode="segment",
            buffering_max_repeaters_per_segment=8,
            buffering_segment_count_z_init=2.5,
            buffering_initial_bsu_index=1.0,
        ),
        SimpleNamespace(mode="segment"),
    )

    state = result.buffer_optimization_state
    assert result.status == "ok"
    assert isinstance(state.z_param, torch.nn.Parameter)
    assert torch.allclose(state.z_param.detach(), torch.full_like(state.z_param, 2.5))
    assert torch.all(state.z_value() >= 0.0)
    assert torch.all(state.z_value() <= 8.0)
    assert not hasattr(state, "z_logits")
    assert not hasattr(state, "relaxed_z_probability")


def test_state_builder_skips_when_inputs_are_missing():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    model = SimpleNamespace(data_collections=SimpleNamespace())
    params = SimpleNamespace(buffering_mode="candidate")
    config = SimpleNamespace(mode="candidate")

    result = build_buffering_state_for_model(model, params, config)

    assert result.status == "skipped"
    assert result.reason == "missing_buffering_inputs"
    assert result.buffer_optimization_state is None
    assert result.buffer_relaxed_timing_payload is None
    diagnostics = result.metadata["input_diagnostics"]
    assert diagnostics["missing_inputs"] == [
        "nets",
        "candidates",
        "buffer_library",
        "buffer_main_type_index",
    ]
    assert diagnostics["net_count"] == 0
    assert diagnostics["candidate_count"] == 0
    assert diagnostics["buffer_library_size"] == 0


def test_state_builder_skip_reports_rejected_buffer_family_contract():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, _buffer_library = _fixture()
    metadata = SimpleNamespace(
        buffer_main_type_index=-1,
        buffer_main_type_status="unsupported_multiple_buffer_families",
    )
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_metadata=metadata,
            buffer_main_type_index=-1,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(buffering_mode="candidate"),
        SimpleNamespace(mode="candidate"),
    )

    assert result.status == "skipped"
    diagnostics = result.metadata["input_diagnostics"]
    assert diagnostics["missing_inputs"] == [
        "candidates",
        "buffer_library",
        "buffer_main_type_index",
    ]
    assert diagnostics["buffer_library_reason"] == "buffer_family_contract_not_ok"
    assert diagnostics["buffer_family_contract_status"] == "unsupported"
    assert diagnostics["buffer_family_contract_reasons"] == [
        "unsupported_multiple_buffer_families"
    ]


def test_state_builder_generates_candidates_and_library_from_nets_and_metadata():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, _buffer_library = _fixture()
    metadata = SimpleNamespace(
        buffer_main_type_index=7,
        buffer_main_type_status="ok",
        flat_libcell_info=[
            [0, 7, 1.0, 0],
            [1, 7, 2.0, 0],
        ],
        flat_libcell_names=["BUF_X1", "BUF_X2"],
        flat_libcell_width=[10, 20],
        flat_libcell_height=[20, 20],
        flat_libcell_leakage=[0.1, 0.2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        flat_lib_pin_cap=[0.4, 0.0, 0.8, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2],
        flat_libarc_info=[[0, 1], [0, 1]],
        f_delay_flat_luts_values=[[2.0], [3.0]],
        r_delay_flat_luts_values=[[2.0], [3.0]],
        f_trans_flat_luts_values=[0.3, 0.2],
        r_trans_flat_luts_values=[0.3, 0.2],
    )
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            buffering_nets=nets,
            buffering_metadata=metadata,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    params = SimpleNamespace(
        buffering_mode="candidate",
        buffering_max_repeaters_per_segment=3,
    )
    config = SimpleNamespace(mode="candidate", max_repeaters_per_segment=3)

    result = build_buffering_state_for_model(model, params, config)

    assert result.status == "ok"
    assert result.candidate_count == 1
    assert result.metadata["candidate_source"] == "generated_from_nets"
    assert result.metadata["candidate_generation_mode"] == "equal_count"
    assert result.metadata["max_candidates_per_segment"] == 3
    assert result.metadata["buffer_library_source"] == "contract_metadata_proxy"
    assert result.buffer_optimization_state.candidate_records[0]["parent_node_id"] == 0
    assert result.buffer_optimization_state.candidate_records[0]["child_node_ids"] == [1]


def test_state_builder_can_build_nets_from_pydb_metadata():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    pydb = SimpleNamespace(
        dbu=1000,
        buffer_main_type_index=7,
        buffer_main_type_status="ok",
        flat_libcell_info=[
            [0, 7, 1.0, 0],
            [1, 7, 2.0, 0],
        ],
        flat_libcell_names=["BUF_X1", "BUF_X2"],
        flat_libcell_width=[10, 20],
        flat_libcell_height=[20, 20],
        flat_libcell_leakage=[0.1, 0.2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        flat_lib_pin_cap=[0.4, 0.0, 0.8, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2],
        flat_libarc_info=[[0, 1], [0, 1]],
        f_delay_flat_luts_values=[[2.0], [3.0]],
        r_delay_flat_luts_values=[[2.0], [3.0]],
        f_trans_flat_luts_values=[0.3, 0.2],
        r_trans_flat_luts_values=[0.3, 0.2],
        flat_net2pin_map=[0, 1],
        flat_net2pin_start_map=[0, 2],
        net2driver_pin_map=[0],
        pin2net_map=[0, 0],
        pin2node_map=[0, 1],
        node_x=[0, 2],
        node_y=[0, 0],
        pin_offset_x=[0, 0],
        pin_offset_y=[0, 0],
        net_names=["net0"],
        pin_names=["drv0:Y", "u0:A"],
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        inst_main_id=[0, 1],
        inst_libcell_offset=[0, 0],
        main_id_2_cell_id_start=[0, 1],
        pin_2_libpin_offset=[0, 0],
    )
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            pydb=pydb,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
        )
    )
    params = SimpleNamespace(
        buffering_mode="candidate",
        buffering_max_repeaters_per_segment=3,
    )
    config = SimpleNamespace(mode="candidate", max_repeaters_per_segment=3)

    result = build_buffering_state_for_model(model, params, config)

    assert result.status == "ok"
    assert result.net_count == 1
    assert result.candidate_count == 1
    assert result.metadata["net_source"] == "pydb"


def test_candidate_strategies_share_the_same_generated_candidate_domain():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    nets, _candidates, buffer_library = _fixture()
    nets[0]["coordinates"] = {0: (0, 0), 1: (8, 0)}

    def build(strategy):
        model = SimpleNamespace(
            data_collections=SimpleNamespace(
                buffering_nets=nets,
                buffering_buffer_library=buffer_library,
                buffer_main_type_index=7,
                dbu=1000,
                pos=[torch.nn.Parameter(torch.tensor([0.0]))],
            )
        )
        params = SimpleNamespace(
            buffering_mode="candidate",
            buffering_max_repeaters_per_segment=3,
        )
        config = SimpleNamespace(
            mode="candidate",
            candidate_strategy=strategy,
            fixed_bsu_index=0,
            max_repeaters_per_segment=3,
        )
        return build_buffering_state_for_model(model, params, config)

    continuous = build("continuous")
    discrete = build("discrete_net_gradient")

    assert continuous.status == "ok"
    assert discrete.status == "ok"
    continuous_domain = [
        (row["candidate_id"], row["x_dbu"], row["y_dbu"])
        for row in continuous.buffer_optimization_state.candidate_records
    ]
    discrete_domain = [
        (row["candidate_id"], row["x_dbu"], row["y_dbu"])
        for row in discrete.buffer_optimization_state.candidate_records
    ]
    assert continuous_domain == discrete_domain
    assert continuous_domain == [(0, 2, 0), (1, 4, 0), (2, 6, 0)]


def test_state_builder_prefers_live_timing_topology_for_pydb_nets():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        build_buffering_state_for_model,
    )

    pydb = SimpleNamespace(
        dbu=1000,
        r_unit=2.0,
        c_unit=0.25,
        buffer_main_type_index=7,
        buffer_main_type_status="ok",
        flat_libcell_info=[
            [0, 7, 1.0, 0],
            [1, 7, 2.0, 0],
        ],
        flat_libcell_names=["BUF_X1", "BUF_X2"],
        flat_libcell_width=[10, 20],
        flat_libcell_height=[20, 20],
        flat_libcell_leakage=[0.1, 0.2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        flat_lib_pin_cap=[0.4, 0.0, 0.8, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2],
        flat_libarc_info=[[0, 1], [0, 1]],
        f_delay_flat_luts_values=[[2.0], [3.0]],
        r_delay_flat_luts_values=[[2.0], [3.0]],
        f_trans_flat_luts_values=[0.3, 0.2],
        r_trans_flat_luts_values=[0.3, 0.2],
        flat_net2pin_map=[0, 1],
        flat_net2pin_start_map=[0, 2],
        net2driver_pin_map=[0],
        pin2net_map=[0, 0],
        pin2node_map=[0, 1],
        node_x=[0, 2000],
        node_y=[0, 0],
        pin_offset_x=[0, 0],
        pin_offset_y=[0, 0],
        net_names=["net0"],
        pin_names=["drv0:Y", "u0:A"],
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        inst_main_id=[0, 1],
        inst_libcell_offset=[0, 0],
        main_id_2_cell_id_start=[0, 1],
        pin_2_libpin_offset=[0, 0],
    )
    model = SimpleNamespace(
        data_collections=SimpleNamespace(
            pydb=pydb,
            pos=[torch.nn.Parameter(torch.tensor([0.0]))],
            buffering_timing_topology={
                "topology_source": "live_timing_topology",
                "net_flat_topo_sort": torch.tensor([0, 1], dtype=torch.int32),
                "net_flat_topo_sort_start": torch.tensor([0, 2], dtype=torch.int32),
                "pin_fa": torch.tensor([-1, 0], dtype=torch.int32),
                "flat_pin_to": torch.tensor([1], dtype=torch.int32),
                "flat_pin_from": torch.tensor([0], dtype=torch.int32),
                "node_x": torch.tensor([0.0, 4000.0]),
                "node_y": torch.tensor([0.0, 0.0]),
            },
            buffering_timing_topology_dbu=1000,
            buffering_timing_topology_r_unit=2.0,
            buffering_timing_topology_c_unit=0.25,
            buffering_timing_topology_scale_factor=1.0,
        )
    )

    result = build_buffering_state_for_model(
        model,
        SimpleNamespace(
            buffering_mode="segment",
            buffering_max_repeaters_per_segment=3,
            buffering_segment_count_z_init=0.0,
            buffering_initial_bsu_index=1.0,
            scale_factor=1.0,
        ),
        SimpleNamespace(mode="segment"),
    )

    assert result.status == "ok"
    assert result.metadata["net_source"] == "live_timing_topology"
    assert result.metadata["net_adapter_rc_source"] == "external_edge_rc"
    assert result.metadata["edge_rc_source"] == "rc_timing_topology_formula"
    assert result.metadata["edge_rc_map_net_count"] == 1
    net = result.buffer_relaxed_timing_payload["nets"][0]
    assert net["rc_tree"]["edge_rc"][(0, 1)] == {
        "r": 8.0,
        "c": 1.0,
        "length_um": 4.0,
    }


def test_candidate_common_inputs_use_live_timing_topology_and_real_edge_rc():
    from dreamplace.ops.buffer_insertion.buffering_state_builder import (
        _resolve_common_inputs,
    )

    pydb = SimpleNamespace(
        dbu=1000,
        r_unit=2.0,
        c_unit=0.25,
        flat_net2pin_map=[0, 1],
        flat_net2pin_start_map=[0, 2],
        net2driver_pin_map=[0],
        pin2node_map=[0, 1],
        node_x=[0, 2_000],
        node_y=[0, 0],
        pin_offset_x=[0, 0],
        pin_offset_y=[0, 0],
        net_names=["net0"],
        pin_names=["drv0:Y", "u0:A"],
        num_movable_nodes=2,
        num_terminals=0,
        num_terminal_NIs=0,
        inst_main_id=[0, 1],
        inst_libcell_offset=[0, 0],
        main_id_2_cell_id_start=[0, 1, 2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        pin_2_libpin_offset=[0, 0],
        flat_lib_pin_cap=[0.0, 0.0, 0.4, 0.0],
    )
    data_collections = SimpleNamespace(
        pydb=pydb,
        buffering_buffer_library={
            0: {"input_cap": 0.2, "delay": 2.0, "output_slew": 0.3},
            1: {"input_cap": 0.4, "delay": 3.0, "output_slew": 0.2},
        },
        buffer_main_type_index=7,
        buffering_timing_topology={
            "topology_source": "live_timing_topology",
            "net_flat_topo_sort": torch.tensor([0, 1], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 2], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1], dtype=torch.int32),
            "flat_pin_from": torch.tensor([0], dtype=torch.int32),
            "node_x": torch.tensor([0.0, 2.0]),
            "node_y": torch.tensor([0.0, 0.0]),
            "node_x_dbu": torch.tensor([0.0, 2000.0]),
            "node_y_dbu": torch.tensor([0.0, 0.0]),
        },
        buffering_timing_topology_dbu=1000,
        buffering_timing_topology_r_unit=2.0,
        buffering_timing_topology_c_unit=0.25,
        buffering_timing_topology_scale_factor=1.0,
    )

    common = _resolve_common_inputs(
        data_collections,
        SimpleNamespace(scale_factor=1.0),
        mode="candidate",
    )

    assert common["net_source"] == "live_timing_topology"
    assert common["edge_rc_summary"]["status"] == "ok"
    assert common["adapter_summary"]["rc_source"] == "external_edge_rc"
    assert len(common["nets"]) == 1
    assert common["nets"][0]["coordinates"] == {0: (0, 0), 1: (2, 0)}
    assert common["nets"][0]["coordinates_dbu"] == {0: (0, 0), 1: (2000, 0)}
    assert common["nets"][0]["rc_tree"]["edge_rc"][(0, 1)] == {
        "r": 0.004,
        "c": 0.0005,
        "length_um": 0.002,
    }
