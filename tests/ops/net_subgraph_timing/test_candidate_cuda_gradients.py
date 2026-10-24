"""CUDA candidate reverse pass must preserve every RC and buffer derivative."""

from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
    _typed_candidate_objective_gradient,
)
from dreamplace.ops.buffer_insertion.drive_type_table import (
    EXPECTED_DRIVE_TYPE_MASTERS,
    write_case_drive_type_table,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_typed_discrete_candidate_state,
)
from dreamplace.ops.net_subgraph_timing.net_subgraph_timing import (
    net_subgraph_forward,
    net_subgraph_forward_cuda,
    net_subgraph_forward_cuda_explicit_autograd,
    net_subgraph_forward_relaxed,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import lookup_buffer_device_per_size
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.schema import BufferDeviceLut


@pytest.fixture(autouse=True, params=[False, True])
def deterministic_candidate_mode(request):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(request.param)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


VARIABLES = (
    "edge_resistance",
    "node_capacitance",
    "edge_capacitance",
    "driver_arrival",
    "driver_slew",
    "candidate_bu",
    "buffer_input_cap",
    "buffer_delay",
    "buffer_output_slew",
)
OUTPUTS = (
    "lout",
    "lin",
    "effective_node_cap",
    "arrival_in",
    "arrival_out",
    "slew_in",
    "slew_out",
    "sink_arrival",
    "sink_slew",
    "sink_load",
    "sink_cap",
    "sink_net_delay",
    "sink_net_impulse",
)


def _inputs(device, activation, dtype=torch.float64):
    topology = {
        "net_flat_topo_sort": [0, 1, 2, 4, 3, 5, 6, 7, 8],
        "net_flat_topo_sort_start": [0, 5, 9],
        "pin_fa": [-1, 0, 1, 1, 2, -1, 5, 6, 5],
        "flat_pin_to_start": [0, 1, 3, 4, 4, 4, 6, 7, 7, 7],
        "flat_pin_to": [1, 2, 3, 4, 6, 8, 7],
        "candidate_node_id": [1, 2, 6, 8],
        "sink_node_id": [3, 4, 7, 8],
        "sink_net_index": [0, 0, 1, 1],
    }
    values = {
        "edge_resistance": [0, 2, 3, 1, 0.5, 0, 1, 2, 0.4],
        "node_capacitance": [0, 0.4, 0.2, 1, 0.6, 0, 0.2, 0.8, 0.5],
        "edge_capacitance": [0, 0.5, 0.7, 0.2, 0.1, 0, 0.4, 0.2, 0.1],
        "driver_arrival": [0.2, 0.3],
        "driver_slew": [0.1, 0.2],
        "candidate_bu": [activation] * 4,
        "buffer_input_cap": [0.2, 0.3, 0.4, 0.5],
        "buffer_delay": [0.1, 0.2, 0.3, 0.4],
        "buffer_output_slew": [0.2, 0.3, 0.4, 0.5],
    }
    return {
        **{k: torch.tensor(v, dtype=torch.long, device=device) for k, v in topology.items()},
        **{k: torch.tensor(v, dtype=dtype, device=device) for k, v in values.items()},
    }


def test_shared_root_and_repeated_sink_gradients_repeat_exactly(
    native_device,
    native_candidate_forward,
):
    if not torch.are_deterministic_algorithms_enabled():
        pytest.skip("requires deterministic algorithms")
    inputs = _inputs(native_device, 1.0, torch.float32)
    inputs["sink_node_id"] = inputs["sink_node_id"].repeat(256)
    inputs["sink_net_index"] = inputs["sink_net_index"].repeat(256)
    generator = torch.Generator(device=native_device).manual_seed(3000)
    order = torch.randperm(
        inputs["sink_node_id"].numel(), device=native_device, generator=generator
    )
    for name in ("sink_node_id", "sink_net_index"):
        inputs[name] = inputs[name][order]
    variables = tuple(inputs[name].requires_grad_() for name in VARIABLES)
    keys = ("sink_arrival", "sink_slew", "sink_load", "sink_net_delay", "sink_net_impulse")
    weights = {
        key: torch.rand(order.shape, device=native_device, generator=generator) for key in keys
    }

    def evaluate():
        output = native_candidate_forward(**inputs)
        loss = sum((output[key] * weights[key]).sum() for key in keys)
        return torch.autograd.grad(loss, variables)

    reference = evaluate()
    for _ in range(20):
        assert all(torch.equal(a, b) for a, b in zip(evaluate(), reference, strict=True))


@pytest.mark.parametrize("activation", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_candidate_outputs_and_all_input_gradients_match_reference(
    native_device, native_candidate_forward, activation, output, dtype
):
    results = []
    for device, forward in (
        ("cpu", net_subgraph_forward),
        (native_device, native_candidate_forward),
    ):
        inputs = _inputs(device, activation, dtype)
        variables = tuple(inputs[name].requires_grad_() for name in VARIABLES)
        values = forward(**inputs)[output]
        weights = torch.arange(1, values.numel() + 1, device=device, dtype=values.dtype)
        gradients = torch.autograd.grad((values * weights).sum(), variables, allow_unused=True)
        results.append(
            (
                values.detach().cpu(),
                tuple(
                    torch.zeros_like(var).cpu() if grad is None else grad.detach().cpu()
                    for var, grad in zip(variables, gradients, strict=True)
                ),
            )
        )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(results[1][0], results[0][0], rtol=tolerance, atol=tolerance)
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fill_uninitialized", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_candidate_initializes_unvisited_nodes(dtype, fill_uninitialized):
    from torch.utils import deterministic

    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_fill = deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    deterministic.fill_uninitialized_memory = fill_uninitialized
    try:
        inputs = _inputs("cuda", 1.0, dtype)
        reference = net_subgraph_forward_cuda(**inputs)
        for name in ("pin_fa", "edge_resistance", "node_capacitance", "edge_capacitance"):
            value = -1 if name == "pin_fa" else 7
            inputs[name] = torch.cat((inputs[name], inputs[name].new_full((2,), value)))
        inputs["flat_pin_to_start"] = torch.cat(
            (
                inputs["flat_pin_to_start"],
                inputs["flat_pin_to_start"][-1:].expand(2),
            )
        )
        result = net_subgraph_forward_cuda(**inputs)
        torch.testing.assert_close(
            {key: result[key][: reference[key].numel()] for key in OUTPUTS},
            {key: reference[key] for key in OUTPUTS},
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            {key: result[key][-2:] for key in OUTPUTS[:7]},
            {key: inputs["node_capacitance"].new_zeros(2) for key in OUTPUTS[:7]},
        )
        assert result["candidate_index_by_node"][-2:].tolist() == [-1, -1]
    finally:
        deterministic.fill_uninitialized_memory = original_fill
        torch.use_deterministic_algorithms(original_deterministic)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("net_count", [0, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_candidate_empty_graph_initializes_driver_adjoints(dtype, net_count):
    inputs = _inputs("cuda", 0.0, dtype)
    inputs = {key: value[:0] for key, value in inputs.items()}
    inputs["net_flat_topo_sort_start"] = torch.zeros(net_count + 1, device="cuda", dtype=torch.long)
    inputs["flat_pin_to_start"] = torch.zeros(1, device="cuda", dtype=torch.long)
    for name in ("driver_arrival", "driver_slew"):
        inputs[name] = torch.zeros(net_count, device="cuda", dtype=dtype)
    variables = tuple(inputs[name].requires_grad_() for name in VARIABLES)
    result = net_subgraph_forward_cuda_explicit_autograd(**inputs)
    gradients = torch.autograd.grad(
        sum(result[key].sum() for key in OUTPUTS),
        variables,
        allow_unused=True,
    )
    # The public wrapper replaces an empty optional wire-cap vector. Its
    # disconnected derivative is zero, as in the reference comparisons above.
    gradients = tuple(
        torch.zeros_like(value) if grad is None else grad
        for value, grad in zip(variables, gradients, strict=True)
    )
    torch.testing.assert_close(gradients, tuple(torch.zeros_like(value) for value in variables))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_candidate_outputs_have_independent_storage_and_version_counters():
    result = net_subgraph_forward_cuda(**_inputs("cuda", 0.4))
    before = {key: result[key].clone() for key in OUTPUTS}
    versions = {key: result[key]._version for key in OUTPUTS}
    assert len({result[key].untyped_storage().data_ptr() for key in OUTPUTS}) == len(OUTPUTS)
    result["lout"].add_(1)
    torch.testing.assert_close(
        {key: result[key] for key in OUTPUTS if key != "lout"},
        {key: before[key] for key in OUTPUTS if key != "lout"},
    )
    assert {key: result[key]._version for key in OUTPUTS} == {
        key: version + int(key == "lout") for key, version in versions.items()
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("output", OUTPUTS[:7])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_candidate_unvisited_output_derivatives_match_finite_difference(dtype, output):
    inputs = _inputs("cuda", 0.4, dtype)
    for name in ("pin_fa", "edge_resistance", "node_capacitance", "edge_capacitance"):
        value = -1 if name == "pin_fa" else 7
        inputs[name] = torch.cat((inputs[name], inputs[name].new_full((2,), value)))
    inputs["flat_pin_to_start"] = torch.cat(
        (
            inputs["flat_pin_to_start"],
            inputs["flat_pin_to_start"][-1:].expand(2),
        )
    )
    variables = tuple(inputs[name].requires_grad_() for name in VARIABLES)
    result = net_subgraph_forward_cuda_explicit_autograd(**inputs)[output][-2:]
    loss = result.sum()
    gradients = torch.autograd.grad(loss, variables)
    torch.testing.assert_close(gradients, tuple(torch.zeros_like(value) for value in variables))
    reference = {name: value.detach().clone() for name, value in inputs.items()}
    for name in VARIABLES:
        for index in range(reference[name].numel()):
            value = reference[name][index].item()
            reference[name][index] = value + 1e-4
            plus = net_subgraph_forward_cuda(**reference)[output][-2:].sum()
            reference[name][index] = value - 1e-4
            minus = net_subgraph_forward_cuda(**reference)[output][-2:].sum()
            reference[name][index] = value
            assert float((plus - minus).abs()) == 0.0


@pytest.mark.parametrize("aggregation", ["hard", "smooth"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("use_device_lut", [False, True])
@pytest.mark.parametrize("use_state_view", [False, True])
def test_four_drive_candidate_gradients_match_cpu_reference_and_finite_difference(
    native_device,
    native_candidate_forward,
    tmp_path,
    aggregation,
    active,
    use_device_lut,
    use_state_view,
):
    library = {
        i: {
            "cell_id": 100 + i,
            "master_name": name,
            "input_cap": 0.1 + i,
            "delay": 1.0 + i,
            "output_slew": 0.5 + i,
        }
        for i, name in enumerate(EXPECTED_DRIVE_TYPE_MASTERS)
    }
    table = write_case_drive_type_table(tmp_path / "drive.json", buffer_library=library)
    base = default_fake_buffer_device()
    buffer_device = BufferDeviceLut(
        input_cap_by_size=torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
        input_slew_axis=base.input_slew_axis,
        output_load_axis=base.output_load_axis,
        delay_lut=torch.stack([base.delay_lut[0] + 0.1 * i for i in range(4)]),
        output_slew_lut=torch.stack([base.output_slew_lut[0] + 0.05 * i for i in range(4)]),
        source="four_drive_gradient_fixture",
    )

    def coordinate_callback(**kwargs):
        return lookup_buffer_device_per_size(
            buffer_device,
            input_slew=kwargs["candidate_input_slew"],
            output_load=kwargs["candidate_output_cap"],
        )

    candidates = [
        {
            "candidate_id": i,
            "net_id": i // 2,
            "candidate_node_id": node,
            "buffer_main_type_index": 4,
            "x_dbu": i,
            "y_dbu": 0,
        }
        for i, node in enumerate([1, 2, 6, 8])
    ]
    lanes = []
    for device, backend in (
        ("cpu", "python"),
        (
            native_device,
            "cuda_explicit_autograd" if native_device == "cuda" else "native_explicit_autograd",
        ),
    ):
        state = build_typed_discrete_candidate_state(
            candidates,
            buffer_main_type_index=4,
            legal_buffer_count=4,
            drive_type_table=table,
            dtype=torch.float64,
            device=device,
        )
        if active:
            with torch.no_grad():
                state.activation_param[0, 1] = 1.0
                state.activation_param[2, 3] = 1.0
        inputs = _inputs(device, 0.0)
        for name in (
            "candidate_node_id",
            "candidate_bu",
            "buffer_input_cap",
            "buffer_delay",
            "buffer_output_slew",
        ):
            del inputs[name]
        sizes = {
            "per_size_input_cap": [[0.1, 0.2, 0.3, 0.4]] * 4,
            "per_size_delay": [[0.4, 0.3, 0.2, 0.1]] * 4,
            "per_size_output_slew": [[0.5, 0.4, 0.3, 0.2]] * 4,
        }
        sizes = {k: torch.tensor(v, device=device, dtype=torch.float64) for k, v in sizes.items()}

        def objective(_model, *, state=state, backend=backend, inputs=inputs, sizes=sizes):
            current_state, current_sizes = state, sizes
            if use_state_view:
                rows = torch.arange(4, device=state.activation_param.device).flip(0)
                current_state = state.slice_candidates(
                    rows,
                    candidate_node_id=state.candidate_node_id[rows],
                    device=state.activation_param.device,
                )
                current_sizes = {
                    name: values.index_select(0, rows) for name, values in sizes.items()
                }
            result = net_subgraph_forward_relaxed(
                buffer_state=current_state,
                forward_backend=backend,
                buffer_coordinate_callback=coordinate_callback if use_device_lut else None,
                **inputs,
                **current_sizes,
            )
            arrival = result["sink_arrival"]
            worst = (
                arrival.max() if aggregation == "hard" else 2.0 * torch.logsumexp(arrival / 2, 0)
            )
            loss = 500.0 * worst + 5.0 * arrival.sum()
            loss = loss + result["sink_slew"].sum() + result["lin"][[0, 5]].sum()
            return loss, {"wns": -worst, "tns": -arrival.sum()}

        lanes.append(SimpleNamespace(state=state, objective=objective))
    gradients = [
        _typed_candidate_objective_gradient(lane, SimpleNamespace(), lane.state, snapshot_id="toy")[
            1
        ]
        for lane in lanes
    ]
    torch.testing.assert_close(gradients[1].cpu(), gradients[0], rtol=1e-9, atol=1e-9)
    gpu_lane = lanes[1]
    gpu_lane.shared_typed_candidate_gradients = True
    _, shared_gradient, shared_metadata = _typed_candidate_objective_gradient(
        gpu_lane,
        SimpleNamespace(),
        gpu_lane.state,
        snapshot_id="toy_shared",
    )
    torch.testing.assert_close(shared_gradient.cpu(), gradients[0], rtol=1e-9, atol=1e-9)
    assert shared_metadata["gradient_pass_count"] == 1
    assert not gpu_lane.state._shared_gradient_probe
    reference = lanes[0]
    state = reference.state
    for drive in range(4):
        state.set_gradient_drive_type(drive)
        for row in range(4):
            with torch.no_grad():
                value = state.activation_param[row, drive].item()
                state.activation_param[row, drive] = value + 1e-6
                plus = reference.objective(None)[0].item()
                state.activation_param[row, drive] = value - 1e-6
                minus = reference.objective(None)[0].item()
                state.activation_param[row, drive] = value
            assert gradients[0][row, drive].item() == pytest.approx(
                (plus - minus) / 2e-6,
                rel=1e-6,
                abs=1e-6,
            )
    state.clear_gradient_drive_type()
    assert torch.equal(
        torch.argsort(gradients[1].cpu().flatten()), torch.argsort(gradients[0].flatten())
    )
