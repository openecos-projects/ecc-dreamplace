import tempfile
from types import SimpleNamespace

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_buffer_optimization_state,
)
from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing import build_relaxed_buffer_timing_payload


def _install_ieda_sta_stub():
    return


def _buffer_lut_metadata():
    return SimpleNamespace(
        buffer_main_type_index=4,
        buffer_main_type_status="ok",
        flat_libcell_info=[
            [0, 4, 1.0, 0],
            [1, 4, 2.0, 0],
            [2, 9, 1.0, 0],
        ],
        flat_libcell_names=["BUF_X1", "BUF_X2", "NAND_X1"],
        flat_libcell_width=[1.0, 2.0, 1.0],
        flat_libcell_height=[1.0, 1.0, 1.0],
        flat_libcell_leakage=[0.1, 0.2, 0.3],
        cell_id_2_libpin_id_start=[0, 2, 4, 6],
        flat_lib_pin_cap=[0.02, 0.0, 0.04, 0.0, 0.08, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2, 3],
        flat_libarc_info=[[0, 1], [0, 1], [0, 1]],
        f_delay_flat_luts_values=[
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        f_delay_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        f_delay_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
        f_delay_flat_luts_dim=[[2, 2], [2, 2], [2, 2]],
        r_delay_flat_luts_values=[
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 5.0, 7.0, 9.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        r_delay_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        r_delay_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
        r_delay_flat_luts_dim=[[2, 2], [2, 2], [2, 2]],
        f_trans_flat_luts_values=[
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.0, 0.0, 0.0, 0.0],
        ],
        f_trans_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        f_trans_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
        f_trans_flat_luts_dim=[[2, 2], [2, 2], [2, 2]],
        r_trans_flat_luts_values=[
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9],
            [0.0, 0.0, 0.0, 0.0],
        ],
        r_trans_flat_luts_trans_table=[[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        r_trans_flat_luts_cap_table=[[0.0, 20.0], [0.0, 20.0], [0.0, 20.0]],
        r_trans_flat_luts_dim=[[2, 2], [2, 2], [2, 2]],
    )


def test_placeobj_does_not_export_pin_violation_csv_by_default(monkeypatch):
    _install_ieda_sta_stub()
    import dreamplace.PlaceObj as place_obj_module
    from dreamplace.PlaceObj import PlaceObj

    def fail_if_csv_is_written(*args, **kwargs):
        raise AssertionError("pin violation CSV should be opt-in")

    monkeypatch.setattr(
        place_obj_module,
        "write_pin_violation_detail_csv",
        fail_if_csv_is_written,
    )

    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(
        result_dir=tempfile.mkdtemp(),
        design_name=lambda: "unit",
    )
    place_obj.placedb = SimpleNamespace(pin_names=["a", "b"])
    place_obj.invoke_timing_count = 0
    place_obj.inst_pins_mask = torch.ones(2, dtype=torch.bool)
    place_obj.output_pin_mask = torch.ones(2, dtype=torch.bool)

    timing_op = SimpleNamespace(
        pin_rtran=torch.zeros(2, dtype=torch.float32),
        pin_ftran=torch.zeros(2, dtype=torch.float32),
        pin_net_cap_rise=torch.zeros(2, dtype=torch.float32),
        pin_net_cap_fall=torch.zeros(2, dtype=torch.float32),
    )
    limits = torch.ones(2, dtype=torch.float32)

    place_obj._log_violation_details(timing_op, limits, limits)


def test_timing_profile_copies_dynamic_provider_snapshot():
    _install_ieda_sta_stub()
    from dreamplace.PlaceObj import PlaceObj

    dynamic_provider_profile = {
        "gpu_static_runtime_build_count": 1,
        "driver_cap_overlay_path": "cuda_cap_only",
        "nested": {"warm": False},
    }
    timing_op = SimpleNamespace(
        last_critical_endpoint_pruning_stats={},
        last_traversal_pruning_stats={},
        last_profile_payload={
            "stage_runtime_ms": {"timing_propagation_ms": 1.25},
            "cell_aat_levels_detail": {"level_count": 3},
            "dynamic_provider_profile": dynamic_provider_profile,
        },
    )

    profile_fields = PlaceObj.__new__(PlaceObj)._timing_pruning_profile_fields(
        timing_op
    )
    dynamic_provider_profile["nested"]["warm"] = True

    assert profile_fields["dynamic_provider_profile"] == {
        "gpu_static_runtime_build_count": 1,
        "driver_cap_overlay_path": "cuda_cap_only",
        "nested": {"warm": False},
    }


def test_placeobj_exports_pin_violation_csv_when_explicitly_enabled(monkeypatch):
    _install_ieda_sta_stub()
    import dreamplace.PlaceObj as place_obj_module
    from dreamplace.PlaceObj import PlaceObj

    written_paths = []

    def record_csv_write(path, rows):
        written_paths.append(path)

    monkeypatch.setattr(
        place_obj_module,
        "write_pin_violation_detail_csv",
        record_csv_write,
    )
    monkeypatch.setattr(PlaceObj, "write_cell_arc_py_report", lambda self: None)
    monkeypatch.setattr(PlaceObj, "write_cell_arc_py_semantic_summary", lambda self: None)
    monkeypatch.setattr(PlaceObj, "write_cell_arc_lut_fingerprint", lambda self: None)
    monkeypatch.setattr(PlaceObj, "write_python_endpoint_slack", lambda self: None)
    monkeypatch.setattr(PlaceObj, "write_endpoint_constraint_debug", lambda self: None)

    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(
        export_pin_violation_csv=True,
        result_dir=tempfile.mkdtemp(),
        design_name=lambda: "unit",
    )
    place_obj.placedb = SimpleNamespace(pin_names=["a", "b"])
    place_obj.invoke_timing_count = 0
    place_obj.inst_pins_mask = torch.ones(2, dtype=torch.bool)
    place_obj.output_pin_mask = torch.ones(2, dtype=torch.bool)

    timing_op = SimpleNamespace(
        pin_rtran=torch.zeros(2, dtype=torch.float32),
        pin_ftran=torch.zeros(2, dtype=torch.float32),
        pin_net_cap_rise=torch.zeros(2, dtype=torch.float32),
        pin_net_cap_fall=torch.zeros(2, dtype=torch.float32),
    )
    limits = torch.ones(2, dtype=torch.float32)

    place_obj._log_violation_details(timing_op, limits, limits)

    assert len(written_paths) == 4
    assert any(path.endswith("_slew_pin_detail.csv") for path in written_paths)
    assert any(path.endswith("_cap_pin_detail.csv") for path in written_paths)


def test_placeobj_builds_relaxed_buffer_dynamic_net_arc_inputs_from_data_collections():
    _install_ieda_sta_stub()
    from dreamplace.PlaceObj import PlaceObj

    state = build_buffer_optimization_state(
        [
            {
                "candidate_id": 3,
                "net_id": 7,
                "candidate_node_id": 1,
                "buffer_main_type_index": 4,
                "x_dbu": 10,
                "y_dbu": 20,
            }
        ],
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    payload = {
        "per_size_input_cap": torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        "per_size_delay": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        "per_size_output_slew": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "sink_pin_id": torch.tensor([2], dtype=torch.int32),
        "fill_value": 0.0,
        "net_flat_topo_sort": torch.tensor([0, 1], dtype=torch.int32),
        "net_flat_topo_sort_start": torch.tensor([0, 2], dtype=torch.int32),
        "pin_fa": torch.tensor([-1, 0], dtype=torch.int32),
        "flat_pin_to_start": torch.tensor([0, 1, 1], dtype=torch.int32),
        "flat_pin_to": torch.tensor([1], dtype=torch.int32),
        "edge_resistance": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "node_capacitance": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "driver_arrival": torch.tensor([70.0], dtype=torch.float32),
        "driver_slew": torch.tensor([0.1], dtype=torch.float32),
        "sink_node_id": torch.tensor([1], dtype=torch.int32),
        "sink_net_index": torch.tensor([0], dtype=torch.int32),
        "coordinate_source": "fixed_smoke",
    }
    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(enable_relaxed_buffer_timing=True)
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )

    dynamic = place_obj._build_relaxed_buffer_dynamic_net_arc_inputs(
        {"rise": torch.zeros(3, dtype=torch.float32), "fall": torch.zeros(3, dtype=torch.float32)}
    )
    loss = dynamic["pin_net_delays"]["rise"][2]
    loss.backward()

    assert dynamic["source"] == "net_subgraph_timing_relaxed_buffer"
    assert dynamic["gradient_chain_level"] == "toy_global_timing_autograd"
    assert dynamic["buffer_coordinate_source"] == "fixed_smoke"
    assert dynamic["metadata"]["net_subgraph_invocation_timing"] == (
        "pre_timing_propagation_overlay"
    )
    assert dynamic["metadata"]["uses_current_propagated_driver_slew"] is False
    assert torch.isfinite(state.bu_logits.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.bu_logits.grad.abs().max()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max()) > 0.0


def test_timing_obj_passes_relaxed_buffer_dynamic_inputs_to_timing_op(monkeypatch):
    _install_ieda_sta_stub()
    import torch.nn as nn
    import dreamplace.PlaceObj as place_obj_module
    from dreamplace.PlaceObj import PlaceObj

    monkeypatch.setattr(
        place_obj_module,
        "compute_size_interpolated_pin_properties",
        lambda data_collections, tables, pin_mask=None: [
            torch.zeros(3, dtype=torch.float32) for _ in tables
        ],
    )

    state = build_buffer_optimization_state(
        [
            {
                "candidate_id": 3,
                "net_id": 7,
                "candidate_node_id": 1,
                "buffer_main_type_index": 4,
                "x_dbu": 10,
                "y_dbu": 20,
            }
        ],
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    payload = {
        "per_size_input_cap": torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        "per_size_delay": torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        "per_size_output_slew": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "sink_pin_id": torch.tensor([2], dtype=torch.int32),
        "fill_value": 0.0,
        "net_flat_topo_sort": torch.tensor([0, 1], dtype=torch.int32),
        "net_flat_topo_sort_start": torch.tensor([0, 2], dtype=torch.int32),
        "pin_fa": torch.tensor([-1, 0], dtype=torch.int32),
        "flat_pin_to_start": torch.tensor([0, 1, 1], dtype=torch.int32),
        "flat_pin_to": torch.tensor([1], dtype=torch.int32),
        "edge_resistance": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "node_capacitance": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "driver_arrival": torch.tensor([70.0], dtype=torch.float32),
        "driver_slew": torch.tensor([0.1], dtype=torch.float32),
        "sink_node_id": torch.tensor([1], dtype=torch.int32),
        "sink_net_index": torch.tensor([0], dtype=torch.int32),
        "coordinate_source": "fixed_smoke",
    }

    class FakeTimingOp:
        def __init__(self):
            self.log_top_violations = True
            self.log_stage_progress = True
            self.log_violation_shape_mismatch = True
            self.captured_dynamic_net_arc_inputs = None

        def __call__(
            self,
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            *,
            surrogate_mode=None,
            dynamic_net_arc_inputs=None,
        ):
            self.captured_dynamic_net_arc_inputs = dynamic_net_arc_inputs
            tns = -dynamic_net_arc_inputs["pin_net_delays"]["rise"][2]
            return tns, tns, torch.zeros((), dtype=torch.float32), torch.zeros((), dtype=torch.float32)

        def get_total_slew_violation_tensor(self, *args, **kwargs):
            return torch.zeros((), dtype=torch.float32)

        def get_total_cap_violation_tensor(self, *args, **kwargs):
            return torch.zeros((), dtype=torch.float32)

    fake_timing_op = FakeTimingOp()
    place_obj = PlaceObj.__new__(PlaceObj)
    nn.Module.__init__(place_obj)
    place_obj.placedb = SimpleNamespace(gr_sizing=None)
    place_obj.params = SimpleNamespace(enable_relaxed_buffer_timing=True)
    place_obj.invoke_timing_count = 0
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        inst_libcell_offset=None,
        net_flat_topo_sort=torch.empty(0, dtype=torch.int32),
        net_flat_topo_sort_start=torch.empty(0, dtype=torch.int32),
        pin_fa=torch.empty(0, dtype=torch.int32),
        flat_pin_to_start=torch.empty(0, dtype=torch.int32),
        flat_pin_to=torch.empty(0, dtype=torch.int32),
        flat_pin_from=torch.empty(0, dtype=torch.int32),
        flat_lib_pin_slew_limit=torch.zeros(3, dtype=torch.float32),
        flat_lib_pin_cap_limit=torch.zeros(3, dtype=torch.float32),
        inst_leakage_init=None,
    )
    place_obj.op_collections = SimpleNamespace(
        pin_pos_op=lambda pos: pos,
        steiner_topo_op=lambda pin_pos: (
            torch.zeros(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        ),
        elmore_delay_op=lambda *args, **kwargs: (
            torch.zeros(3, dtype=torch.float32),
            {
                "rise": torch.zeros(3, dtype=torch.float32),
                "fall": torch.zeros(3, dtype=torch.float32),
            },
            {
                "rise": torch.zeros(3, dtype=torch.float32),
                "fall": torch.zeros(3, dtype=torch.float32),
            },
            None,
            None,
            {
                "rise": torch.zeros(3, dtype=torch.float32),
                "fall": torch.zeros(3, dtype=torch.float32),
            },
        ),
        timing_propagation_op=fake_timing_op,
    )
    place_obj.pin_caps_op = lambda inst_offset, data_collections: (
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
    )
    place_obj._should_write_timing_obj_profile = lambda iteration: False
    place_obj._should_write_timing_artifacts = lambda iteration: False
    place_obj._should_write_pin_violation_detail = lambda iteration: False
    place_obj._production_fast_loop_enabled = lambda: True
    place_obj._timing_obj_profile_interval = lambda: 0
    place_obj._timing_rc_cap_input_tensor = lambda tensor: tensor
    place_obj._should_copy_pin_slack_after_timing = lambda timing_artifact_enabled: False
    place_obj._should_use_active_cone_aux_pin_masks = lambda: False
    place_obj._should_reuse_zero_slew_aux = lambda timing_op: False
    place_obj._should_compute_timing_aux_tensor = lambda name: False
    place_obj.inst_pins_mask = torch.ones(3, dtype=torch.bool)
    place_obj.output_pin_mask = torch.ones(3, dtype=torch.bool)
    place_obj.pin2libpin_flat_ids = torch.arange(3, dtype=torch.long)

    _, tns, _, _ = place_obj.timing_obj(torch.zeros(3, dtype=torch.float32))
    loss = -tns
    loss.backward()

    dynamic = fake_timing_op.captured_dynamic_net_arc_inputs
    assert dynamic is not None
    assert dynamic["source"] == "net_subgraph_timing_relaxed_buffer"
    assert torch.isfinite(state.bu_logits.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.bu_logits.grad.abs().max()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max()) > 0.0
def test_timing_obj_can_install_relaxed_buffer_dynamic_net_provider(monkeypatch):
    _install_ieda_sta_stub()
    import torch.nn as nn
    import dreamplace.PlaceObj as place_obj_module
    from dreamplace.PlaceObj import PlaceObj

    monkeypatch.setattr(
        place_obj_module,
        "compute_size_interpolated_pin_properties",
        lambda data_collections, tables, pin_mask=None: [
            torch.zeros(3, dtype=torch.float32) for _ in tables
        ],
    )

    candidates = [
        {
            "candidate_id": 3,
            "net_id": 7,
            "node_id": 2,
            "candidate_node_id": 2,
            "buffer_main_type_index": 4,
            "x_dbu": 10,
            "y_dbu": 20,
        }
    ]
    state = build_buffer_optimization_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    payload = build_relaxed_buffer_timing_payload(
        [
            {
                "net_id": 7,
                "driver_pin_id": 1,
                "rc_tree": {
                    "root_node_id": 1,
                    "children_by_node": {1: [2], 2: []},
                    "edge_rc": {(1, 2): {"r": 0.0, "c": 0.0}},
                    "node_cap": {1: 0.0, 2: 1.0},
                    "sink_nodes": [2],
                },
            }
        ],
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        driver_arrival_by_net={7: -999.0},
        driver_slew_by_net={7: 99.0},
        coordinate_source="fixed_smoke",
        num_pins=3,
    )

    class FakeTimingOp:
        def __init__(self):
            self.log_top_violations = True
            self.log_stage_progress = True
            self.log_violation_shape_mismatch = True
            self.captured_dynamic_net_arc_inputs = None
            self.captured_dynamic_net_provider = None

        def __call__(
            self,
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            *,
            surrogate_mode=None,
            dynamic_net_arc_inputs=None,
            dynamic_net_provider=None,
        ):
            self.captured_dynamic_net_arc_inputs = dynamic_net_arc_inputs
            self.captured_dynamic_net_provider = dynamic_net_provider
            assert dynamic_net_arc_inputs is None
            assert dynamic_net_provider is not None
            pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
            pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
            pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
            pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)

            def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
                return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = dynamic_net_provider.propagate_net_aat_level(
                net_ids=torch.tensor([7], dtype=torch.long),
                pin_rAAT=pin_rAAT,
                pin_fAAT=pin_fAAT,
                pin_rtran=pin_rtran,
                pin_ftran=pin_ftran,
                base_pin_net_delay_rise=pin_net_delays["rise"],
                base_pin_net_delay_fall=pin_net_delays["fall"],
                base_pin_net_impulse_rise=pin_net_impulses["rise"],
                base_pin_net_impulse_fall=pin_net_impulses["fall"],
                static_calculate_net_aat_level=static_calculate_net_aat_level,
            )
            tns = -(pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2])
            return tns, tns, torch.zeros((), dtype=torch.float32), torch.zeros((), dtype=torch.float32)

        def get_total_slew_violation_tensor(self, *args, **kwargs):
            return torch.zeros((), dtype=torch.float32)

        def get_total_cap_violation_tensor(self, *args, **kwargs):
            return torch.zeros((), dtype=torch.float32)

    fake_timing_op = FakeTimingOp()
    place_obj = PlaceObj.__new__(PlaceObj)
    nn.Module.__init__(place_obj)
    place_obj.placedb = SimpleNamespace(gr_sizing=None)
    place_obj.params = SimpleNamespace(
        enable_relaxed_buffer_timing=True,
        relaxed_buffer_timing_integration_mode="dynamic_net_provider",
    )
    place_obj.invoke_timing_count = 0
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        inst_libcell_offset=None,
        net_flat_topo_sort=torch.empty(0, dtype=torch.int32),
        net_flat_topo_sort_start=torch.empty(0, dtype=torch.int32),
        pin_fa=torch.empty(0, dtype=torch.int32),
        flat_pin_to_start=torch.empty(0, dtype=torch.int32),
        flat_pin_to=torch.empty(0, dtype=torch.int32),
        flat_pin_from=torch.empty(0, dtype=torch.int32),
        flat_lib_pin_slew_limit=torch.zeros(3, dtype=torch.float32),
        flat_lib_pin_cap_limit=torch.zeros(3, dtype=torch.float32),
        inst_leakage_init=None,
    )
    place_obj.op_collections = SimpleNamespace(
        pin_pos_op=lambda pos: pos,
        steiner_topo_op=lambda pin_pos: (
            torch.zeros(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        ),
        elmore_delay_op=lambda *args, **kwargs: (
            torch.zeros(3, dtype=torch.float32),
            {
                "rise": torch.zeros(3, dtype=torch.float32),
                "fall": torch.zeros(3, dtype=torch.float32),
            },
            {
                "rise": torch.zeros(3, dtype=torch.float32),
                "fall": torch.zeros(3, dtype=torch.float32),
            },
            None,
            None,
            {
                "rise": torch.zeros(3, dtype=torch.float32),
                "fall": torch.zeros(3, dtype=torch.float32),
            },
        ),
        timing_propagation_op=fake_timing_op,
    )
    place_obj.pin_caps_op = lambda inst_offset, data_collections: (
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
    )
    place_obj._should_write_timing_obj_profile = lambda iteration: False
    place_obj._should_write_timing_artifacts = lambda iteration: False
    place_obj._should_write_pin_violation_detail = lambda iteration: False
    place_obj._production_fast_loop_enabled = lambda: True
    place_obj._timing_obj_profile_interval = lambda: 0
    place_obj._timing_rc_cap_input_tensor = lambda tensor: tensor
    place_obj._should_copy_pin_slack_after_timing = lambda timing_artifact_enabled: False
    place_obj._should_use_active_cone_aux_pin_masks = lambda: False
    place_obj._should_reuse_zero_slew_aux = lambda timing_op: False
    place_obj._should_compute_timing_aux_tensor = lambda name: False
    place_obj.inst_pins_mask = torch.ones(3, dtype=torch.bool)
    place_obj.output_pin_mask = torch.ones(3, dtype=torch.bool)
    place_obj.pin2libpin_flat_ids = torch.arange(3, dtype=torch.long)

    _, tns, _, _ = place_obj.timing_obj(torch.zeros(3, dtype=torch.float32))
    loss = -tns
    loss.backward()

    provider = fake_timing_op.captured_dynamic_net_provider
    assert fake_timing_op.captured_dynamic_net_arc_inputs is None
    assert provider is not None
    assert provider.metadata["uses_current_propagated_driver_slew"] is True
    assert provider.metadata["dynamic_provider_affected_net_call_count"] == 1
    assert torch.isfinite(state.bu_logits.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.bu_logits.grad.abs().max()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max()) > 0.0
    assert place_obj._relaxed_buffer_dynamic_net_provider is provider
    assert provider.metadata["net_subgraph_forward_backend"] == (
        "native_explicit_autograd"
    )
    reused_provider = place_obj._build_relaxed_buffer_dynamic_net_provider()
    assert reused_provider is provider
    assert reused_provider.metadata["provider_reused"] is True


def test_placeobj_builds_segment_count_dynamic_net_provider():
    _install_ieda_sta_stub()
    from dreamplace.PlaceObj import PlaceObj
    from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import (
        SegmentCountDynamicNetProvider,
    )

    nets = [
        {
            "net_id": 7,
            "driver_pin_id": 1,
            "coordinates": {1: (0, 0), 2: (100, 0)},
            "rc_tree": {
                "root_node_id": 1,
                "children_by_node": {1: [2], 2: []},
                "edge_rc": {(1, 2): {"r": 1.0, "c": 0.0}},
                "node_cap": {1: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        }
    ]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=4,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.0,
        initial_bsu_index=1.0,
    )
    payload = {
        "metadata": {"state_kind": "segment_count"},
        "nets": nets,
        "per_size_input_cap": torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32),
        "per_size_delay": torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
        "per_size_output_slew": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
    }
    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(
        enable_relaxed_buffer_timing=True,
        relaxed_buffer_timing_integration_mode="dynamic_net_provider",
        buffering_segment_count_timing_backend="prepared_python",
    )
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        buffer_segment_count_state=state,
        buffer_segment_count_nets=nets,
        buffer_segment_count_per_size_input_cap=payload["per_size_input_cap"],
        buffer_segment_count_per_size_delay=payload["per_size_delay"],
        buffer_segment_count_per_size_output_slew=payload["per_size_output_slew"],
        buffering_metadata=_buffer_lut_metadata(),
        buffer_segment_timing_topology_epoch=0,
        buffer_segment_timing_topology_epoch_reason="unit_initial",
    )

    provider = place_obj._build_relaxed_buffer_dynamic_net_provider()

    assert isinstance(provider, SegmentCountDynamicNetProvider)
    assert provider.segment_state is state
    assert provider.metadata["timing_integration_mode"] == "segment_count_dynamic_net_provider"
    assert provider.metadata["segment_count_timing_backend_used"] == "prepared_python"
    assert provider.metadata["segment_transfer_device_lut_source"] == "metadata_contract_lut"
    assert (
        payload["metadata"]["segment_transfer_device_lut"]["status"]
        == "ok"
    )
    with torch.no_grad():
        state.z_param.add_(0.25)
    reused_provider = place_obj._build_relaxed_buffer_dynamic_net_provider()
    assert reused_provider is provider
    assert reused_provider.metadata["provider_reused"] is True

    place_obj.data_collections.buffer_segment_timing_topology_epoch = 1
    place_obj.data_collections.buffer_segment_timing_topology_epoch_reason = "unit_rebuild"
    rebuilt_provider = place_obj._build_relaxed_buffer_dynamic_net_provider()
    assert rebuilt_provider is not provider
    assert rebuilt_provider.metadata["provider_reused"] is False
    assert rebuilt_provider.metadata["topology_epoch"] == 1


def test_placeobj_segment_count_provider_defaults_to_cpp_explicit_autograd():
    _install_ieda_sta_stub()
    from dreamplace.PlaceObj import PlaceObj

    nets = [
        {
            "net_id": 7,
            "driver_pin_id": 1,
            "coordinates": {1: (0, 0), 2: (100, 0)},
            "rc_tree": {
                "root_node_id": 1,
                "children_by_node": {1: [2], 2: []},
                "edge_rc": {(1, 2): {"r": 1.0, "c": 0.0}},
                "node_cap": {1: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        }
    ]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=4,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.0,
        initial_bsu_index=1.0,
    )
    payload = {
        "metadata": {"state_kind": "segment_count"},
        "nets": nets,
        "per_size_input_cap": torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32),
        "per_size_delay": torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
        "per_size_output_slew": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
    }
    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(
        enable_relaxed_buffer_timing=True,
        relaxed_buffer_timing_integration_mode="dynamic_net_provider",
    )
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        buffer_segment_count_state=state,
        buffer_segment_count_nets=nets,
        buffer_segment_count_per_size_input_cap=payload["per_size_input_cap"],
        buffer_segment_count_per_size_delay=payload["per_size_delay"],
        buffer_segment_count_per_size_output_slew=payload["per_size_output_slew"],
    )

    provider = place_obj._build_relaxed_buffer_dynamic_net_provider()

    assert provider.metadata["segment_count_timing_backend_used"] == (
        "cpp_cpu_explicit_autograd"
    )


def test_timing_obj_can_install_segment_count_dynamic_net_provider(monkeypatch):
    _install_ieda_sta_stub()
    import torch.nn as nn
    import dreamplace.PlaceObj as place_obj_module
    from dreamplace.PlaceObj import PlaceObj

    monkeypatch.setattr(
        place_obj_module,
        "compute_size_interpolated_pin_properties",
        lambda data_collections, tables, pin_mask=None: [
            torch.zeros(3, dtype=torch.float32) for _ in tables
        ],
    )

    nets = [
        {
            "net_id": 7,
            "driver_pin_id": 1,
            "coordinates": {1: (0, 0), 2: (100, 0)},
            "rc_tree": {
                "root_node_id": 1,
                "children_by_node": {1: [2], 2: []},
                "edge_rc": {(1, 2): {"r": 1.0, "c": 0.0}},
                "node_cap": {1: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        }
    ]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=4,
        legal_buffer_count=3,
        max_repeater_count=2,
        initial_z=1.25,
        initial_bsu_index=1.0,
    )
    payload = {
        "metadata": {"state_kind": "segment_count"},
        "nets": nets,
        "per_size_input_cap": torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32),
        "per_size_delay": torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
        "per_size_output_slew": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
    }

    class FakeTimingOp:
        def __init__(self):
            self.log_top_violations = True
            self.log_stage_progress = True
            self.log_violation_shape_mismatch = True
            self.captured_dynamic_net_arc_inputs = None
            self.captured_dynamic_net_provider = None

        def __call__(
            self,
            pin_net_delays,
            pin_net_impulses,
            pin_net_caps,
            *,
            surrogate_mode=None,
            dynamic_net_arc_inputs=None,
            dynamic_net_provider=None,
        ):
            self.captured_dynamic_net_arc_inputs = dynamic_net_arc_inputs
            self.captured_dynamic_net_provider = dynamic_net_provider
            assert dynamic_net_arc_inputs is None
            assert dynamic_net_provider is not None
            pin_rAAT = torch.tensor([0.0, 40.0, -1.0], dtype=torch.float32)
            pin_fAAT = torch.tensor([0.0, 50.0, -1.0], dtype=torch.float32)
            pin_rtran = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
            pin_ftran = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float32)

            def static_calculate_net_aat_level(curnets, pin_rAAT, pin_fAAT, pin_rtran, pin_ftran, *args):
                return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

            pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = dynamic_net_provider.propagate_net_aat_level(
                net_ids=torch.tensor([7], dtype=torch.long),
                pin_rAAT=pin_rAAT,
                pin_fAAT=pin_fAAT,
                pin_rtran=pin_rtran,
                pin_ftran=pin_ftran,
                base_pin_net_delay_rise=pin_net_delays["rise"],
                base_pin_net_delay_fall=pin_net_delays["fall"],
                base_pin_net_impulse_rise=pin_net_impulses["rise"],
                base_pin_net_impulse_fall=pin_net_impulses["fall"],
                static_calculate_net_aat_level=static_calculate_net_aat_level,
            )
            tns = -(pin_rAAT[2] + pin_fAAT[2] + pin_rtran[2] + pin_ftran[2])
            return tns, tns, torch.zeros((), dtype=torch.float32), torch.zeros((), dtype=torch.float32)

        def get_total_slew_violation_tensor(self, *args, **kwargs):
            return torch.zeros((), dtype=torch.float32)

        def get_total_cap_violation_tensor(self, *args, **kwargs):
            return torch.zeros((), dtype=torch.float32)

    fake_timing_op = FakeTimingOp()
    place_obj = PlaceObj.__new__(PlaceObj)
    nn.Module.__init__(place_obj)
    place_obj.placedb = SimpleNamespace(gr_sizing=None)
    place_obj.params = SimpleNamespace(
        enable_relaxed_buffer_timing=True,
        relaxed_buffer_timing_integration_mode="dynamic_net_provider",
        buffering_segment_count_timing_backend="prepared_python",
        buffering_segment_live_geometry=True,
    )
    place_obj.invoke_timing_count = 0
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        buffer_segment_count_state=state,
        buffer_segment_count_nets=nets,
        buffer_segment_count_per_size_input_cap=payload["per_size_input_cap"],
        buffer_segment_count_per_size_delay=payload["per_size_delay"],
        buffer_segment_count_per_size_output_slew=payload["per_size_output_slew"],
        inst_libcell_offset=None,
        net_flat_topo_sort=torch.empty(0, dtype=torch.int32),
        net_flat_topo_sort_start=torch.empty(0, dtype=torch.int32),
        pin_fa=torch.empty(0, dtype=torch.int32),
        flat_pin_to_start=torch.empty(0, dtype=torch.int32),
        flat_pin_to=torch.empty(0, dtype=torch.int32),
        flat_pin_from=torch.empty(0, dtype=torch.int32),
        flat_lib_pin_slew_limit=torch.zeros(3, dtype=torch.float32),
        flat_lib_pin_cap_limit=torch.zeros(3, dtype=torch.float32),
        inst_leakage_init=None,
        buffer_segment_timing_topology_epoch=0,
    )

    class FakeElmoreOp:
        r_unit = 0.01
        c_unit = 0.001
        scale_factor = 1.0
        dbu = 1.0

        def __call__(self, *args, **kwargs):
            return (
                torch.zeros(3, dtype=torch.float32),
                {
                    "rise": torch.zeros(3, dtype=torch.float32),
                    "fall": torch.zeros(3, dtype=torch.float32),
                },
                {
                    "rise": torch.zeros(3, dtype=torch.float32),
                    "fall": torch.zeros(3, dtype=torch.float32),
                },
                None,
                None,
                {
                    "rise": torch.zeros(3, dtype=torch.float32),
                    "fall": torch.zeros(3, dtype=torch.float32),
                },
            )

    place_obj.op_collections = SimpleNamespace(
        pin_pos_op=lambda pos: pos,
        steiner_topo_op=lambda pin_pos: (
            pin_pos,
            torch.zeros_like(pin_pos),
        ),
        elmore_delay_op=FakeElmoreOp(),
        timing_propagation_op=fake_timing_op,
    )
    place_obj.pin_caps_op = lambda inst_offset, data_collections: (
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
    )
    place_obj._should_write_timing_obj_profile = lambda iteration: False
    place_obj._should_write_timing_artifacts = lambda iteration: False
    place_obj._should_write_pin_violation_detail = lambda iteration: False
    place_obj._production_fast_loop_enabled = lambda: True
    place_obj._timing_obj_profile_interval = lambda: 0
    place_obj._timing_rc_cap_input_tensor = lambda tensor: tensor
    place_obj._should_copy_pin_slack_after_timing = lambda timing_artifact_enabled: False
    place_obj._should_use_active_cone_aux_pin_masks = lambda: False
    place_obj._should_reuse_zero_slew_aux = lambda timing_op: False
    place_obj._should_compute_timing_aux_tensor = lambda name: False
    place_obj.inst_pins_mask = torch.ones(3, dtype=torch.bool)
    place_obj.output_pin_mask = torch.ones(3, dtype=torch.bool)
    place_obj.pin2libpin_flat_ids = torch.arange(3, dtype=torch.long)

    pos = torch.tensor(
        [0.0, 0.0, 100.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    _, tns, _, _ = place_obj.timing_obj(pos)
    loss = -tns
    loss.backward()

    provider = fake_timing_op.captured_dynamic_net_provider
    assert fake_timing_op.captured_dynamic_net_arc_inputs is None
    assert provider is not None
    assert provider.metadata["timing_integration_mode"] == "segment_count_dynamic_net_provider"
    assert provider.metadata["uses_current_propagated_driver_slew"] is True
    assert provider.metadata["dynamic_provider_affected_net_call_count"] == 1
    assert state.z_param.grad is not None
    assert state.bsu_index_param.grad is not None
    assert torch.isfinite(state.z_param.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.z_param.grad.abs().max()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max()) > 0.0
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert float(pos.grad.abs().max()) > 0.0
    assert provider.metadata["live_geometry_bind_count"] == 1
    assert provider.metadata["live_geometry_release_count"] == 1
    # Rise/fall each evaluate the current and z=0 residual-reference states.
    assert provider.metadata["live_rc_local_build_count"] == 4
    assert provider.metadata["live_geometry_gradient_hook_count"] == 4 * 3
    gradient_stats = provider.metadata["live_geometry_gradient_stats"]
    assert set(gradient_stats) == {"length", "edge_resistance", "edge_capacitance"}
    assert gradient_stats["length"]["finite_count"] > 0
    assert gradient_stats["length"]["nonzero_count"] > 0
    assert gradient_stats["edge_resistance"]["norm"] > 0.0
    assert gradient_stats["edge_capacitance"]["norm"] > 0.0
    assert "1" in gradient_stats["length"]["by_integer_z"]
    assert provider._live_geometry_context is None


def test_relaxed_buffer_payload_maps_original_candidate_nodes_to_compact_nodes():
    _install_ieda_sta_stub()
    from dreamplace.PlaceObj import PlaceObj

    nets = [
        {
            "net_id": 10,
            "driver_pin_id": 100,
            "rc_tree": {
                "root_node_id": 100,
                "children_by_node": {100: [300, 200], 300: [], 200: []},
                "edge_rc": {
                    (100, 300): {"r": 0.0, "c": 0.0},
                    (100, 200): {"r": 0.0, "c": 0.0},
                },
                "node_cap": {100: 0.0, 300: 3.0, 200: 2.0},
                "sink_nodes": [300, 200],
            },
        },
    ]
    candidates = [
        {
            "candidate_id": 31,
            "net_id": 10,
            "node_id": 300,
            "candidate_node_id": 300,
            "buffer_main_type_index": 4,
            "x_dbu": 30,
            "y_dbu": 40,
        }
    ]
    state = build_buffer_optimization_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    payload = build_relaxed_buffer_timing_payload(
        nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        driver_arrival_by_net={10: 70.0},
        driver_slew_by_net={10: 0.1},
        coordinate_source="fixed_smoke",
        num_pins=400,
    )
    assert torch.equal(payload["candidate_node_id"], torch.tensor([1], dtype=torch.int32))
    assert torch.equal(payload["sink_pin_id"], torch.tensor([300, 200], dtype=torch.int32))

    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(enable_relaxed_buffer_timing=True)
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )
    dynamic = place_obj._build_relaxed_buffer_dynamic_net_arc_inputs(
        {"rise": torch.zeros(400, dtype=torch.float32), "fall": torch.zeros(400, dtype=torch.float32)}
    )
    loss = dynamic["pin_net_delays"]["rise"][300]
    loss.backward()

    assert dynamic["source"] == "net_subgraph_timing_relaxed_buffer"
    assert torch.isfinite(state.bu_logits.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.bu_logits.grad.abs().max()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max()) > 0.0


def test_placeobj_consumes_exported_relaxed_buffer_context():
    _install_ieda_sta_stub()
    from dreamplace.PlaceObj import PlaceObj

    nets = [
        {
            "net_id": 10,
            "driver_pin_id": 100,
            "driver_pin_name": "drv:Y",
            "net_pin_ids": [100, 300, 200],
            "rc_tree": {
                "root_node_id": 100,
                "children_by_node": {100: [300, 200], 300: [], 200: []},
                "edge_rc": {
                    (100, 300): {"r": 0.0, "c": 0.0},
                    (100, 200): {"r": 0.0, "c": 0.0},
                },
                "node_cap": {100: 0.0, 300: 3.0, 200: 2.0},
                "sink_nodes": [300, 200],
            },
        },
    ]
    candidates = [
        {
            "candidate_id": 31,
            "net_id": 10,
            "node_id": 300,
            "candidate_node_id": 300,
            "buffer_main_type_index": 4,
            "x_dbu": 30_000,
            "y_dbu": 0,
        },
    ]
    buffer_library = {
        0: {"input_cap": 0.2, "delay": 10.0, "output_slew": 0.1},
        1: {"input_cap": 0.4, "delay": 20.0, "output_slew": 0.2},
    }
    state = build_buffer_optimization_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
    )
    state.bu_logits.grad = None
    state.bsu_index_param.grad = None
    payload = build_relaxed_buffer_timing_payload(
        nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float32),
        per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        per_size_output_slew=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        driver_arrival_by_net={10: 0.0},
        driver_slew_by_net={10: 0.1},
        coordinate_source="fixed_smoke",
        num_pins=400,
    )

    place_obj = PlaceObj.__new__(PlaceObj)
    place_obj.params = SimpleNamespace(enable_relaxed_buffer_timing=True)
    place_obj.data_collections = SimpleNamespace(
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
    )
    dynamic = place_obj._build_relaxed_buffer_dynamic_net_arc_inputs(
        {
            "rise": torch.zeros(400, dtype=torch.float32),
            "fall": torch.zeros(400, dtype=torch.float32),
        }
    )
    loss = dynamic["pin_net_delays"]["rise"][300]
    loss.backward()

    assert dynamic["source"] == "net_subgraph_timing_relaxed_buffer"
    assert dynamic["buffer_coordinate_source"] == "fixed_smoke"
    assert torch.isfinite(state.bu_logits.grad).all()
    assert torch.isfinite(state.bsu_index_param.grad).all()
    assert float(state.bu_logits.grad.abs().max()) > 0.0
    assert float(state.bsu_index_param.grad.abs().max()) > 0.0
