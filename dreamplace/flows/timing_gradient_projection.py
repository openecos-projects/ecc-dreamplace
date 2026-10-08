"""ECC diff-TDP gradient-to-net-weight projection transaction."""

import logging
import time
from dataclasses import dataclass

import torch


@dataclass
class TimingGradientProjectionContext:
    """Explicit owners for one gradient-carrier update."""

    params: object
    placedb: object
    model: object
    pos: torch.Tensor
    data_collections: object
    op_collections: object
    iteration: int
    gate_status: dict
    timing_carrier: str
    record_timing_metrics: object
    write_artifact: object
    topology_prepared: bool = False


def project_timing_gradient_to_net_weights(context):
    params = context.params
    placedb = context.placedb
    model = context.model
    pos = context.pos
    op_collections = context.op_collections
    data_collections = context.data_collections
    if getattr(op_collections, "timing_propagation_op", None) is None:
        payload = {
            "status": "skipped",
            "reason": "timing_propagation_op_missing",
            "carrier": context.timing_carrier,
            "iteration": int(context.iteration),
            "gate_status": dict(context.gate_status),
        }
        context.write_artifact(payload)
        return payload, False

    old_pos_grad = None if pos.grad is None else pos.grad.detach().clone()
    old_use_timing_obj = bool(getattr(model, "use_timing_obj", False))
    if pos.grad is not None:
        pos.grad.zero_()
    try:
        started = time.time()
        if not context.topology_prepared:
            pin_pos = op_collections.pin_pos_op(pos)
            pin_pos_for_topology = pin_pos.detach().cpu().contiguous()
            (
                data_collections.net_flat_topo_sort,
                data_collections.net_flat_topo_sort_start,
                data_collections.pin_fa,
                data_collections.flat_pin_to,
                data_collections.flat_pin_to_start,
                data_collections.flat_pin_from,
            ) = op_collections.steiner_topo_op.rebuild_tree(pin_pos_for_topology)
        steiner_update_ms = (time.time() - started) * 1000.0

        started = time.time()
        model.use_timing_obj = False
        wns, tns, ws, ts = model.timing_obj(pos)
        timing_loss = model._timing_loss(wns, tns, ws, ts)
        timing_forward_ms = (time.time() - started) * 1000.0

        started = time.time()
        timing_loss.backward()
        timing_backward_ms = (time.time() - started) * 1000.0

        if pos.grad is None:
            raise RuntimeError("timing gradient projection produced no pos.grad")
        pos_grad = pos.grad.detach()
        pos_grad_float = pos_grad.float()
        pos_grad_finite_mask = torch.isfinite(pos_grad_float)
        pos_grad_nonfinite_count = int(
            torch.count_nonzero(~pos_grad_finite_mask).detach().cpu().item()
        )
        pos_grad_finite_count = int(
            torch.count_nonzero(pos_grad_finite_mask).detach().cpu().item()
        )
        if pos_grad_nonfinite_count:
            logging.warning(
                "Diff TDP gradient projection sanitized %d non-finite position-gradient entries",
                pos_grad_nonfinite_count,
            )
            pos_grad_float = torch.nan_to_num(
                pos_grad_float,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        half = int(pos_grad_float.numel() // 2)
        node_score = torch.sqrt(
            pos_grad_float[:half].pow(2) + pos_grad_float[half : half * 2].pow(2)
        )

        pin2node = data_collections.pin2node_map.long().to(node_score.device)
        valid_pin_mask = (pin2node >= 0) & (pin2node < node_score.numel())
        pin_score = torch.zeros(
            pin2node.numel(), dtype=node_score.dtype, device=node_score.device
        )
        if torch.any(valid_pin_mask):
            pin_score[valid_pin_mask] = node_score[pin2node[valid_pin_mask]]

        pin2net = data_collections.pin2net_map.long().to(pin_score.device)
        num_nets = int(
            getattr(placedb, "num_nets", data_collections.net_weights.numel())
        )
        net_score = torch.zeros(
            num_nets, dtype=pin_score.dtype, device=pin_score.device
        )
        valid_net_mask = (pin2net >= 0) & (pin2net < num_nets)
        if torch.any(valid_net_mask):
            if hasattr(net_score, "scatter_reduce_"):
                net_score.scatter_reduce_(
                    0,
                    pin2net[valid_net_mask],
                    pin_score[valid_net_mask],
                    reduce="amax",
                    include_self=True,
                )
            else:
                valid_net_ids = pin2net[valid_net_mask]
                valid_pin_scores = pin_score[valid_net_mask]
                for idx in range(valid_net_ids.numel()):
                    net_id = int(valid_net_ids[idx].item())
                    net_score[net_id] = torch.maximum(
                        net_score[net_id], valid_pin_scores[idx]
                    )

        degree_start = data_collections.flat_net2pin_start_map.long().to(
            net_score.device
        )
        if degree_start.numel() >= num_nets + 1:
            net_degree = degree_start[1 : num_nets + 1] - degree_start[:num_nets]
            net_mask = (net_degree >= 2) & (
                net_degree < int(getattr(params, "ignore_net_degree", 1000000000))
            )
        else:
            net_mask = torch.ones(
                num_nets, dtype=torch.bool, device=net_score.device
            )

        finite_mask = net_mask & torch.isfinite(net_score) & (net_score > 0)
        scale = float(getattr(params, "timing_gradient_net_weight_scale", 0.4))
        max_weight = float(getattr(params, "timing_gradient_net_weight_max", 2.0))
        if torch.any(finite_mask):
            normalizer = torch.quantile(net_score[finite_mask], 0.95).clamp_min(1e-12)
            new_weights = torch.ones(
                num_nets, dtype=net_score.dtype, device=net_score.device
            )
            new_weights[net_mask] = torch.clamp(
                1.0 + scale * net_score[net_mask] / normalizer,
                min=1.0,
                max=max_weight,
            )
        else:
            normalizer = torch.tensor(
                0.0, dtype=net_score.dtype, device=net_score.device
            )
            new_weights = torch.ones(
                num_nets, dtype=net_score.dtype, device=net_score.device
            )

        target_weights = new_weights.to(
            dtype=data_collections.net_weights.dtype,
            device=data_collections.net_weights.device,
        )
        data_collections.net_weights[: target_weights.numel()].copy_(target_weights)
        placedb.net_weights[: target_weights.numel()] = target_weights.detach().cpu().numpy().astype(
            placedb.net_weights.dtype, copy=False
        )
        projected = bool(
            torch.any(target_weights > 1.0 + 1e-6).detach().cpu().item()
        )

        top_k = min(10, num_nets)
        top_values, top_indices = torch.topk(net_score.detach(), k=top_k)
        net_names = getattr(placedb, "net_names", [])
        top_nets = []
        for rank, (net_id_tensor, score_tensor) in enumerate(
            zip(top_indices, top_values)
        ):
            net_id = int(net_id_tensor.detach().cpu().item())
            net_name = ""
            if 0 <= net_id < len(net_names):
                net_name = net_names[net_id]
                if isinstance(net_name, bytes):
                    net_name = net_name.decode("utf-8", errors="ignore")
                else:
                    net_name = str(net_name)
            top_nets.append(
                {
                    "rank": int(rank),
                    "net_id": net_id,
                    "net_name": net_name,
                    "score": float(score_tensor.detach().cpu().item()),
                    "weight": float(target_weights[net_id].detach().cpu().item()),
                }
            )

        payload = {
            "status": "updated",
            "carrier": context.timing_carrier,
            "iteration": int(context.iteration),
            "gate_status": dict(context.gate_status),
            "scale": scale,
            "max_weight": max_weight,
            "steiner_update_ms": float(steiner_update_ms),
            "timing_forward_ms": float(timing_forward_ms),
            "timing_backward_ms": float(timing_backward_ms),
            "wns": float(wns.detach().cpu().item()) if torch.is_tensor(wns) else float(wns),
            "tns": float(tns.detach().cpu().item()) if torch.is_tensor(tns) else float(tns),
            "timing_loss": float(timing_loss.detach().cpu().item()),
            "pos_grad_norm": float(pos_grad_float.norm().detach().cpu().item()),
            "pos_grad_finite_count": pos_grad_finite_count,
            "pos_grad_nonfinite_count": pos_grad_nonfinite_count,
            "net_score_max": float(net_score.max().detach().cpu().item()) if num_nets else 0.0,
            "net_score_p95": (
                float(torch.quantile(net_score[finite_mask], 0.95).detach().cpu().item())
                if torch.any(finite_mask)
                else 0.0
            ),
            "normalizer": float(normalizer.detach().cpu().item()),
            "non_unit_net_weight_count": int(
                torch.count_nonzero(target_weights > 1.0 + 1e-6).detach().cpu().item()
            ),
            "max_net_weight": float(target_weights.max().detach().cpu().item())
            if target_weights.numel()
            else 1.0,
            "top_nets": top_nets,
        }
        context.record_timing_metrics(wns, tns, ws, ts)
        context.write_artifact(payload)
        logging.info(
            "Diff TDP gradient net-weight update at iteration=%d "
            "overflow=%.6f non_unit_nets=%d max_weight=%.3f "
            "timing_loss=%.6f timing_forward_ms=%.3f timing_backward_ms=%.3f",
            int(context.iteration),
            float(context.gate_status.get("overflow") or 0.0),
            payload["non_unit_net_weight_count"],
            payload["max_net_weight"],
            payload["timing_loss"],
            payload["timing_forward_ms"],
            payload["timing_backward_ms"],
        )
        return payload, projected
    finally:
        model.use_timing_obj = old_use_timing_obj
        if old_pos_grad is None:
            pos.grad = None
        elif pos.grad is None:
            pos.grad = old_pos_grad
        else:
            pos.grad.copy_(old_pos_grad)
