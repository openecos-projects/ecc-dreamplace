import pytest
import torch

from sparse_fixtures import (
    inputs,
    tables,
    schedule,
    production_timing,
    shared_driver_timing,
    shared_driver_net,
    run_production,
    rise_fall,
)


def test_production_forward_uses_sparse_objective_and_rc_gradients(native_device):
    timing = production_timing(native_device)
    values = inputs(native_device)
    lanes = schedule(native_device, cell=True).propagate(
        tables(native_device), values[:2], values[2:4], values[4:]
    )
    expected = schedule(native_device, cell=True).reduce_endpoints(
        lanes, values[0].new_zeros(4)
    )
    wns, tns, _, _ = timing(
        rise_fall(values[2:4]),
        rise_fall(values[4:]),
        rise_fall(values[:2]),
        surrogate_mode="lut_only",
    )
    torch.testing.assert_close((wns, tns), (expected.wns, expected.tns))
    actual_gradients = torch.autograd.grad(tns, values, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.tns, values)
    torch.testing.assert_close(actual_gradients, expected_gradients)
    assert timing.last_profile_payload["mode"] == "sparse"


def test_production_sparse_uses_existing_size_vt_model(native_device):
    timing = production_timing(native_device)
    size = torch.ones(4, device=native_device, dtype=torch.float64, requires_grad=True)
    vt_logits = torch.zeros(
        (4, 2), device=native_device, dtype=torch.float64, requires_grad=True
    )
    timing.pin2node_map = torch.arange(4, device=native_device)
    timing.inst_main_id = torch.zeros(4, device=native_device, dtype=torch.long)
    timing.size_var_getter = lambda: size
    timing.vt_var_getter = lambda: torch.softmax(vt_logits, dim=1)
    timing.cell_modeling_op = lambda main, offset, vt, size, slew, cap, kind: (
        1 / (size + 1) + 0.1 * vt + 0.01 * slew + 0.02 * cap
    )
    values = inputs(native_device)
    _, tns, _, _ = timing(
        rise_fall(values[2:4]),
        rise_fall(values[4:]),
        rise_fall(values[:2]),
        surrogate_mode="surrogate_only",
    )
    gs, gv = torch.autograd.grad(tns, (size, vt_logits))
    assert gs[3].abs() > 0
    assert gv[3].abs().sum() > 0
    torch.testing.assert_close(gs[:3], torch.zeros_like(gs[:3]))
    torch.testing.assert_close(gv[:3], torch.zeros_like(gv[:3]))


def test_master_projection_rebinds_liberty_rows_without_rebuilding_states(
    native_device,
):
    from dreamplace.ops.timing_propagation.timing_propagation import LUTS_INFO

    timing = production_timing(native_device)
    for name in ("r_delay_luts", "f_delay_luts", "r_trans_luts", "f_trans_luts"):
        info = getattr(timing.arcs_info, name)
        setattr(
            timing.arcs_info,
            name,
            LUTS_INFO(
                torch.cat((info.flat_luts_values, info.flat_luts_values * 3)),
                info.flat_luts_trans_table.repeat(2, 1),
                info.flat_luts_cap_table.repeat(2, 1),
                info.flat_luts_dim.repeat(2, 1),
            ),
        )
    values = inputs(native_device)
    args = (rise_fall(values[2:4]), rise_fall(values[4:]), rise_fall(values[:2]))
    _, before, _, _ = timing(*args, surrogate_mode="lut_only")
    schedule = timing._sparse_runtime.schedule
    timing.flat_inst_arcs_by_level[:, 3] = 1
    _, after, _, _ = timing(*args, surrogate_mode="lut_only")
    assert after < before
    assert timing._sparse_runtime.schedule is schedule


def test_candidate_provider_runs_once_for_multiple_launch_states():
    from dreamplace.ops.buffer_insertion.optimization_state import (
        build_buffer_optimization_state,
    )
    from dreamplace.ops.net_subgraph_timing.dynamic_provider import (
        RelaxedBufferDynamicNetProvider,
    )
    from dreamplace.ops.net_subgraph_timing.timing_adapter import (
        build_relaxed_buffer_timing_payload,
    )

    timing = shared_driver_timing()
    candidates = [
        {
            "candidate_id": 11,
            "net_id": 2,
            "node_id": 2,
            "candidate_node_id": 2,
            "buffer_main_type_index": 4,
        }
    ]
    state = build_buffer_optimization_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        initial_bu_logit=0.0,
        initial_bsu_index=0.5,
        dtype=torch.float64,
    )
    nets = [
        {
            "net_id": 2,
            "driver_pin_id": 0,
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [2], 2: []},
                "edge_rc": {(0, 2): {"r": 0.0, "c": 0.0}},
                "node_cap": {0: 0.0, 2: 1.0},
                "sink_nodes": [2],
            },
        }
    ]
    payload = build_relaxed_buffer_timing_payload(
        nets,
        candidates,
        buffer_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float64),
        per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.3]], dtype=torch.float64),
        driver_arrival_by_net={2: 0.0},
        driver_slew_by_net={2: 0.3},
        num_pins=4,
    )
    provider = RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
        affected_net_ids=[2],
        forward_backend="python",
    )
    values = inputs("cpu")
    _, tns, _, _ = timing(
        rise_fall(values[2:4]),
        rise_fall(values[4:]),
        rise_fall(values[:2]),
        surrogate_mode="lut_only",
        dynamic_net_provider=provider,
    )
    lanes = timing._sparse_runtime.last_result.lanes
    torch.testing.assert_close(lanes[0][3] - lanes[0][2], lanes[0].new_tensor(10.0))
    assert provider.affected_call_count == 1
    gradients = torch.autograd.grad(tns, (state.bu_logits, state.bsu_index_param))
    assert all(
        torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in gradients
    )


@pytest.mark.parametrize("initial_z", [0.0, 1.0])
def test_segment_provider_preserves_static_calibration_and_live_geometry(initial_z):
    from dreamplace.ops.buffer_insertion.segment_count_state import (
        build_segment_count_state,
    )
    from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import (
        SegmentCountDynamicNetProvider,
    )

    timing = shared_driver_timing()
    values = inputs("cpu")
    _, baseline, _, _ = run_production(timing, values)
    nets = [shared_driver_net()]
    state = build_segment_count_state(
        nets,
        buffer_main_type_index=4,
        legal_buffer_count=2,
        max_repeater_count=2,
        initial_z=initial_z,
        initial_bsu_index=0.5,
        dtype=torch.float64,
    )
    provider = SegmentCountDynamicNetProvider(
        nets=nets,
        segment_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.4]], dtype=torch.float64),
        per_size_delay=torch.tensor([[10.0, 20.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.3]], dtype=torch.float64),
    )
    if initial_z:
        x = torch.tensor(
            [0.0, 0.0, 100.0, 100.0], dtype=torch.float64, requires_grad=True
        )
        y = torch.zeros_like(x, requires_grad=True)
        provider.bind_live_geometry(
            new_x=x,
            new_y=y,
            r_unit=0.1,
            c_unit=0.005,
            scale_factor=1.0,
            dbu=1.0,
            topology_epoch=0,
            forward_id="sparse",
        )
    _, actual, _, _ = timing(
        rise_fall(values[2:4]),
        rise_fall(values[4:]),
        rise_fall(values[:2]),
        surrogate_mode="lut_only",
        dynamic_net_provider=provider,
    )
    lanes = timing._sparse_runtime.last_result.lanes
    torch.testing.assert_close(lanes[0][3] - lanes[0][2], actual.new_tensor(10.0))
    if not initial_z:
        torch.testing.assert_close(actual, baseline)
    else:
        gradients = torch.autograd.grad(
            actual, (state.z_param, state.bsu_index_param, x, y)
        )
        assert all(torch.isfinite(value).all() for value in gradients)
        assert all(value.abs().sum() > 0 for value in gradients[:3])
        provider.release_live_geometry(forward_id="sparse")


def test_sparse_uses_real_piecewise_cell_model_and_repeated_size_updates():
    from types import SimpleNamespace
    from dreamplace.ops.cell_modeling.cell_modeling import CellModeling
    from dreamplace.ops.cell_modeling import cell_modeling_op
    from dreamplace.ops.timing_propagation.timing_propagation import LUTS_INFO

    timing = production_timing("cpu")
    timing.dtype = torch.float32
    for name in ("r_delay_luts", "f_delay_luts", "r_trans_luts", "f_trans_luts"):
        info = getattr(timing.arcs_info, name)
        setattr(
            timing.arcs_info,
            name,
            LUTS_INFO(
                torch.cat(
                    tuple(
                        info.flat_luts_values * scale for scale in (2.0, 1.0, 1.8, 0.8)
                    )
                ).float(),
                info.flat_luts_trans_table.repeat(4, 1).float(),
                info.flat_luts_cap_table.repeat(4, 1).float(),
                info.flat_luts_dim.repeat(4, 1),
            ),
        )
    data = SimpleNamespace(
        device=timing.device,
        arcs_info=timing.arcs_info,
        flat_libcell_info=torch.tensor(
            [[0, 0, 1.0, 0], [1, 0, 2.0, 0], [2, 0, 1.0, 1], [3, 0, 2.0, 1]]
        ),
        flat_libarc_info=torch.tensor([[0, 1, row, 0] for row in range(4)]),
    )
    timing.cell_modeling_op = CellModeling(data)
    size = torch.nn.Parameter(torch.full((4,), 1.5, dtype=torch.float32))
    vt = torch.nn.Parameter(torch.zeros((4, 2), dtype=torch.float32))
    timing.pin2node_map = torch.arange(4)
    timing.inst_main_id = torch.zeros(4, dtype=torch.long)
    timing.size_var_getter = lambda: size
    timing.vt_var_getter = lambda: torch.softmax(vt, dim=1)
    values = tuple(value.float() for value in inputs("cpu"))
    losses = []
    cell_modeling_op.reset_piecewise_size_backend_profile()
    for _ in range(3):
        _, tns, _, _ = timing(
            rise_fall(values[2:4]),
            rise_fall(values[4:]),
            rise_fall(values[:2]),
            surrogate_mode="surrogate_only",
        )
        losses.append(float(-tns.detach()))
        gs, gv = torch.autograd.grad(-tns, (size, vt))
        assert gs[3].abs() > 0 and gv[3].abs().sum() > 0
        with torch.no_grad():
            size.sub_(0.01 * gs)
            vt.sub_(0.01 * gv)
    assert losses[-1] < losses[0]
    profile = cell_modeling_op.get_piecewise_size_backend_profile()
    assert profile["piecewise_size_forward_backend"] == "cpp"
    assert profile["piecewise_size_backward_backend"] == "cpp"


def test_sparse_typed_discrete_candidates_reuse_existing_four_type_gradient_probe():
    from dreamplace.ops.buffer_insertion.drive_type_table import (
        DriveTypeEntry,
        DriveTypeTable,
    )
    from dreamplace.ops.buffer_insertion.optimization_state import (
        build_typed_discrete_candidate_state,
    )
    from dreamplace.ops.net_subgraph_timing.dynamic_provider import (
        RelaxedBufferDynamicNetProvider,
    )
    from dreamplace.ops.net_subgraph_timing.timing_adapter import (
        build_relaxed_buffer_timing_payload,
    )

    timing = shared_driver_timing()
    candidate = {
        "candidate_id": 11,
        "net_id": 2,
        "node_id": 2,
        "candidate_node_id": 2,
        "buffer_main_type_index": 4,
        "x_dbu": 100.0,
        "y_dbu": 0.0,
    }
    table = DriveTypeTable(
        tuple(DriveTypeEntry(i, f"BUF{i}", i + 1.0, i, i) for i in range(4)), "fixture"
    )
    state = build_typed_discrete_candidate_state(
        [candidate],
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
        dtype=torch.float64,
    )
    payload = build_relaxed_buffer_timing_payload(
        [shared_driver_net()],
        [candidate],
        buffer_state=state,
        per_size_input_cap=torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float64),
        per_size_delay=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64),
        per_size_output_slew=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64),
        driver_arrival_by_net={2: 0.0},
        driver_slew_by_net={2: 0.3},
        num_pins=4,
    )
    provider = RelaxedBufferDynamicNetProvider(
        static_payload=payload,
        buffer_state=state,
        forward_backend="native_explicit_autograd",
    )
    values = inputs("cpu")
    for drive in range(4):
        state.set_gradient_drive_type(drive)
        _, tns, _, _ = timing(
            rise_fall(values[2:4]),
            rise_fall(values[4:]),
            rise_fall(values[:2]),
            surrogate_mode="lut_only",
            dynamic_net_provider=provider,
        )
        (gradient,) = torch.autograd.grad(-tns, state.activation_param)
        assert torch.isfinite(gradient).all()
        assert gradient[0, drive].abs() > 0
    state.clear_gradient_drive_type()


def test_sparse_reuses_production_cpu_cuda_parity_check():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    timing = production_timing("cpu")
    timing.timing_propagation_parity_check = True
    with torch.no_grad():
        run_production(timing, inputs("cpu"))
    assert timing.last_parity_payload["checked"]
    assert timing.last_parity_payload["passed"]
