"""Fixed-state scalar-reference and shared candidate gradient probes."""

import importlib
import time

import torch

from .candidate_objective import objective_snapshot as _objective_snapshot


def typed_candidate_objective_gradient(lane, model, state, *, snapshot_id):
    """Probe all typed masters through the existing candidate gradient path."""

    if not getattr(state, "is_typed_candidate", False):
        raise ValueError("typed gradient probing requires typed candidate state")
    shared = bool(getattr(lane, "shared_typed_candidate_gradients", False))
    if shared and not state.activation_param.is_cuda:
        native = importlib.import_module(
            "dreamplace.ops.net_subgraph_timing.net_subgraph_timing_cpp"
        )
        shared = hasattr(native, "candidate_net_subgraph_backward")
    provider = getattr(model, "_relaxed_buffer_dynamic_net_provider", None)
    if provider is not None:
        shared = shared and provider.forward_backend in (
            "native_explicit_autograd",
            "cuda_explicit_autograd",
        )
    raw_grad = torch.zeros_like(state.activation_param)
    snapshots = []
    phase_times = []
    started_at = time.perf_counter()
    # Buffering probes optimize only candidate activations.  GP coordinates and
    # cell state are held fixed for the discrete action, but they used to keep
    # their ``requires_grad`` flags from the surrounding electrical window.
    # That made every probe retain and reverse the full placement/size graph.
    data = getattr(model, "data_collections", None)
    probe_pos = None
    restored_requires_grad = []
    if data is not None:
        positions = getattr(data, "pos", None)
        if isinstance(positions, (list, tuple)) and positions:
            probe_pos = positions[0].detach()
        for name in ("real_size", "size_logits", "vt_logits"):
            value = getattr(data, name, None)
            if value is not None and getattr(value, "requires_grad", False):
                restored_requires_grad.append((value, True))
                value.requires_grad_(requires_grad=False)
    try:
        drive_types = [None] if shared else range(int(state.drive_type_count))
        for drive_type_id in drive_types:
            if shared:
                state.begin_shared_gradient_probe()
            else:
                state.set_gradient_drive_type(drive_type_id)
            if state.activation_param.is_cuda:
                torch.cuda.synchronize(state.activation_param.device)
            forward_started_at = time.perf_counter()
            if probe_pos is None:
                loss, metrics = lane.objective(model)
            else:
                loss, metrics = lane.objective(model, pos=probe_pos)
            if state.activation_param.is_cuda:
                torch.cuda.synchronize(state.activation_param.device)
            forward_ms = (time.perf_counter() - forward_started_at) * 1000.0
            backward_started_at = time.perf_counter()
            (gradient,) = torch.autograd.grad(
                loss,
                state.activation_param,
                retain_graph=False,
                create_graph=False,
            )
            if state.activation_param.is_cuda:
                torch.cuda.synchronize(state.activation_param.device)
            phase_times.append(
                {
                    "drive_type_id": drive_type_id,
                    "forward_ms": forward_ms,
                    "backward_ms": (time.perf_counter() - backward_started_at) * 1000.0,
                }
            )
            if shared:
                raw_grad.copy_(gradient.detach())
            else:
                raw_grad[:, drive_type_id] = gradient[:, drive_type_id].detach()
            snapshots.append(_objective_snapshot(model, loss, metrics))
    finally:
        state.clear_gradient_drive_type()
        for value, requires_grad in restored_requires_grad:
            value.requires_grad_(requires_grad)
    if not snapshots:
        raise ValueError("typed candidate gradient probing produced no passes")
    return (
        snapshots[0],
        raw_grad,
        {
            "snapshot_id": str(snapshot_id),
            "snapshot_state": "frozen_per_round",
            "drive_type_order": list(range(int(state.drive_type_count))),
            "refresh_between_types": False,
            "drive_type_count": int(state.drive_type_count),
            "gradient_pass_count": len(snapshots),
            "shared_timing_backward": shared,
            "gradient_total_ms": float((time.perf_counter() - started_at) * 1000.0),
            "phase_times_ms": phase_times,
            "typed_snapshots": snapshots,
        },
    )
