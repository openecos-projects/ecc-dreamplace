"""Combine the native L-shape component with the placement gradient.

The route component shares the base precondition step and owns its adaptive
weight and telemetry. Timing backward remains in PlaceObj.
"""

import logging

import torch

from .profile_timing import l_shape_log_verbose, profile_scope


def apply_l_shape_gradient(model, pos, obj):
    # 保存 wirelength + density 的 raw 梯度，后面与 L-shape 梯度共享 precondition
    current_iteration = model._l_shape_outer_iteration
    debug_hash_op = (
        model.l_shape_routability_op
        if hasattr(model.l_shape_routability_op, "log_debug_hash")
        else None
    )
    base_grad_raw = pos.grad.data.clone()
    if debug_hash_op is not None:
        debug_hash_op.log_debug_hash(
            "place_obj.base_grad_raw",
            base_grad_raw,
            norm="%.9e" % float(base_grad_raw.norm(p=2).item()),
        )

    pos.grad.zero_()

    # 计算L形routability cost
    with profile_scope(
        model.params,
        "place_obj.l_shape_cost_forward",
        tensor=pos,
        logger=logging,
        iteration=current_iteration,
    ):
        l_shape_cost = model.l_shape_routability_obj(
            pos,
            use_l_direction=True,
            update_capacity_al_lambda=bool(
                getattr(model.params, "l_shape_capacity_al_enable", False)
            ),
            placement_iteration_id=current_iteration,
        )
    with profile_scope(
        model.params,
        "place_obj.l_shape_cost_backward",
        tensor=pos,
        logger=logging,
        iteration=current_iteration,
    ):
        l_shape_cost.backward()
    density_op = getattr(model.l_shape_routability_op, "density_op", None)
    model.l_shape_gradient_mode = model.l_shape_routability_op.gradient_mode
    model.l_shape_fast_mode = bool(
        getattr(density_op, "fast_mode", getattr(model.params, "l_shape_fast_mode", 0))
    )
    model.l_shape_energy_valid = bool(
        getattr(density_op, "energy_valid", not model.l_shape_fast_mode)
    )

    # 获取原始 L-shape 梯度范数
    l_shape_grad_raw = pos.grad.data.clone()
    l_shape_grad_raw_norm = l_shape_grad_raw.norm(p=2)
    l_shape_grad_raw_norm_value = float(l_shape_grad_raw_norm.item())
    if debug_hash_op is not None:
        debug_hash_op.log_debug_hash(
            "place_obj.l_shape_grad_raw",
            l_shape_grad_raw,
            norm="%.9e" % l_shape_grad_raw_norm_value,
        )
    with profile_scope(
        model.params,
        "place_obj.shared_precondition",
        tensor=pos.grad,
        logger=logging,
        iteration=current_iteration,
    ):
        base_grad, l_shape_grad = model.op_collections.precondition_op.apply_components(
            [base_grad_raw, l_shape_grad_raw],
            model.density_weight,
            model.update_mask,
            model.fix_nodes_mask,
        )
    base_grad_norm = base_grad.norm(p=2)
    l_shape_grad_norm = l_shape_grad.norm(p=2)
    base_grad_norm_value = float(base_grad_norm.item())
    l_shape_grad_norm_value = float(l_shape_grad_norm.item())
    if debug_hash_op is not None:
        debug_hash_op.log_debug_hash(
            "place_obj.base_grad_preconditioned",
            base_grad,
            norm="%.9e" % base_grad_norm_value,
        )
        debug_hash_op.log_debug_hash(
            "place_obj.l_shape_grad_preconditioned",
            l_shape_grad,
            norm="%.9e" % l_shape_grad_norm_value,
        )
    target_weight_value = None
    weight_candidate_value = None
    cap_active = False
    pos.grad.data.copy_(l_shape_grad)

    current_weight = float(model.l_shape_routability_weight.item())

    # 自适应调整权重
    # 目标: l_shape_grad_norm * weight ≈ target_ratio * base_grad_norm
    if l_shape_grad_norm_value > 1e-10 and base_grad_norm_value > 1e-10:
        target_weight = model._compute_l_shape_target_weight(
            base_grad_norm_value,
            l_shape_grad_norm_value,
            model.l_shape_grad_target_ratio,
        )
        target_weight_value = float(target_weight)
        old_weight = current_weight

        if not model._l_shape_weight_initialized:
            new_weight = target_weight
            model._l_shape_weight_initialized = True
            if l_shape_log_verbose(model.params) >= 1:
                logging.info(
                    f"L-shape weight auto-initialized: {new_weight:.4e} "
                    f"(base_grad={base_grad_norm_value:.4e}, l_shape_grad={l_shape_grad_norm_value:.4e})"
                )
        else:
            new_weight = (
                1 - model.l_shape_weight_momentum
            ) * old_weight + model.l_shape_weight_momentum * target_weight

        weight_candidate_value = float(new_weight)
        if new_weight > target_weight:
            new_weight = target_weight
            cap_active = True
        new_weight = max(model.l_shape_weight_min, min(model.l_shape_weight_max, new_weight))
        current_weight = float(new_weight)
        model.l_shape_routability_weight.data.fill_(current_weight)
        pos.grad.data.mul_(current_weight)

        cost_label = (
            "N/A(fast_mode)" if not model.l_shape_energy_valid else f"{l_shape_cost.item():.4e}"
        )
        logging.debug(
            f"L-shape: cost={cost_label}, "
            f"grad_norm={l_shape_grad_norm_value:.4e}, "
            f"base_grad_norm={base_grad_norm_value:.4e}, "
            f"weight={old_weight:.4e}->{new_weight:.4e}"
        )
    else:
        pos.grad.data.mul_(current_weight)

    l_shape_weighted = l_shape_cost * model.l_shape_routability_weight.item()
    current_weight = float(model.l_shape_routability_weight.item())
    grad_ratio_value = None
    if l_shape_grad_norm_value > 1e-10 and base_grad_norm_value > 1e-10:
        grad_ratio_value = current_weight * l_shape_grad_norm_value / (base_grad_norm_value + 1e-12)
    obj = obj + l_shape_weighted
    # Filler reverse force: push fillers toward congested areas
    # using the L-shape Poisson field (bilinear interpolation on field_map)
    if model.l_shape_filler_reverse_force > 0:
        from dreamplace.ops.routability.l_shape_electric_potential import (
            SegmentElectricPotentialFunction,
        )

        field_map_x = SegmentElectricPotentialFunction.last_field_map_x
        field_map_y = SegmentElectricPotentialFunction.last_field_map_y
        if field_map_x is not None and field_map_y is not None:
            density_op = model.l_shape_routability_op.density_op
            num_physical = model.placedb.num_physical_nodes
            num_nodes = model.placedb.num_nodes
            # filler positions
            filler_x = pos.data[num_physical:num_nodes]
            filler_y = pos.data[num_nodes + num_physical : 2 * num_nodes]
            # map filler positions to bin coordinates (continuous)
            bin_x = (filler_x - density_op.xl) / density_op.bin_size_x - 0.5
            bin_y = (filler_y - density_op.yl) / density_op.bin_size_y - 0.5
            bin_x = bin_x.clamp(0, density_op.num_bins_x - 1.001)
            bin_y = bin_y.clamp(0, density_op.num_bins_y - 1.001)
            # bilinear interpolation indices
            ix0 = bin_x.long()
            iy0 = bin_y.long()
            ix1 = (ix0 + 1).clamp(max=density_op.num_bins_x - 1)
            iy1 = (iy0 + 1).clamp(max=density_op.num_bins_y - 1)
            wx = bin_x - ix0.float()
            wy = bin_y - iy0.float()
            # interpolate field_map_x (force in x direction)
            fx = (
                field_map_x[ix0, iy0] * (1 - wx) * (1 - wy)
                + field_map_x[ix1, iy0] * wx * (1 - wy)
                + field_map_x[ix0, iy1] * (1 - wx) * wy
                + field_map_x[ix1, iy1] * wx * wy
            )
            # interpolate field_map_y (force in y direction)
            fy = (
                field_map_y[ix0, iy0] * (1 - wx) * (1 - wy)
                + field_map_y[ix1, iy0] * wx * (1 - wy)
                + field_map_y[ix0, iy1] * (1 - wx) * wy
                + field_map_y[ix1, iy1] * wx * wy
            )
            # Adaptive scaling: l_shape_filler_reverse_force is the target ratio
            # of filler reverse force norm to base_grad filler norm
            base_filler_x = base_grad[num_physical:num_nodes]
            base_filler_y = base_grad[num_nodes + num_physical : 2 * num_nodes]
            base_filler_norm = (base_filler_x.norm(p=2) ** 2 + base_filler_y.norm(p=2) ** 2).sqrt()
            raw_rev_norm = (fx.norm(p=2) ** 2 + fy.norm(p=2) ** 2).sqrt()
            if raw_rev_norm > 1e-12 and base_filler_norm > 1e-12:
                filler_force_scale = (
                    model.l_shape_filler_reverse_force * base_filler_norm / raw_rev_norm
                )
            else:
                filler_force_scale = 0.0
            filler_force_x = filler_force_scale * fx
            filler_force_y = filler_force_scale * fy
            pos.grad.data[num_physical:num_nodes] += filler_force_x
            pos.grad.data[num_nodes + num_physical : 2 * num_nodes] += filler_force_y

            # Diagnostic log
            filler_rev_norm = (filler_force_x.norm(p=2) ** 2 + filler_force_y.norm(p=2) ** 2).sqrt()
            if l_shape_log_verbose(model.params) >= 2:
                logging.info(
                    f"FillerRevForce: rev_norm={filler_rev_norm:.4e}, "
                    f"base_filler_norm={base_filler_norm:.4e}, "
                    f"ratio={filler_rev_norm / (base_filler_norm + 1e-12):.4f}, "
                    f"scale={filler_force_scale:.4e}, "
                    f"num_fillers={num_nodes - num_physical}"
                )

    # Filler pseudo wire force: pull random fillers to most congested point.
    if model.l_shape_filler_pseudo_wire_ratio > 0:
        from dreamplace.ops.routability.l_shape_electric_potential import (
            SegmentElectricPotentialFunction,
        )
        import torchvision

        overflow_map = SegmentElectricPotentialFunction.last_overflow_map
        if overflow_map is not None:
            num_physical = model.placedb.num_physical_nodes
            num_nodes = model.placedb.num_nodes
            num_fillers = num_nodes - num_physical
            density_op = model.l_shape_routability_op.density_op

            # 1. Gaussian blur + average pooling to find local most congested point
            blurrer = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=2)
            overflow_blurred = blurrer(overflow_map.unsqueeze(0)).squeeze(0)
            mean_kernel = 11
            overflow_mean = torch.nn.functional.avg_pool2d(
                overflow_map.unsqueeze(0), mean_kernel, 1, padding=mean_kernel // 2
            ).squeeze(0)

            # 2. Find most congested bin
            max_idx = overflow_mean.view(-1).argmax()
            max_bin_x = max_idx // overflow_mean.shape[1]
            max_bin_y = max_idx % overflow_mean.shape[1]

            # 3. Convert to physical coordinates (bin center)
            target_x = density_op.xl + (max_bin_x.float() + 0.5) * density_op.bin_size_x
            target_y = density_op.yl + (max_bin_y.float() + 0.5) * density_op.bin_size_y

            # 4. Randomly select fillers
            num_selected = max(1, int(num_fillers * model.l_shape_filler_pseudo_wire_ratio))
            selected_indices = torch.randperm(num_fillers, device=pos.device)[:num_selected]

            # 5. Get selected filler positions (as leaf tensors with grad)
            filler_idx_x = num_physical + selected_indices
            filler_idx_y = num_nodes + num_physical + selected_indices
            filler_pos_x = pos[filler_idx_x].clone().requires_grad_(True)
            filler_pos_y = pos[filler_idx_y].clone().requires_grad_(True)
            filler_pos = torch.stack([filler_pos_x, filler_pos_y], dim=1)  # [N, 2]

            # 6. Create virtual pin positions with noise.
            # Each filler has slightly different target to avoid clustering
            target_pos = torch.tensor([[target_x, target_y]], device=pos.device, dtype=pos.dtype)
            target_pos = target_pos.repeat(num_selected, 1)  # [N, 2]
            # Add noise: scale * 5 * randn, where scale is roughly bin_size
            noise_scale = max(density_op.bin_size_x, density_op.bin_size_y) * 5
            target_pos.add_(torch.randn_like(target_pos) * noise_scale)

            # 7. Compute WA wirelength and gradient using autograd
            # Each filler-target pair is a 2-pin net
            gamma = 4.0  # WA gamma parameter
            dist = torch.norm(filler_pos - target_pos, dim=1)  # [N]

            # WA = sum(dist * exp(gamma * dist)) / sum(exp(gamma * dist))
            # Clamp to prevent overflow
            exp_gamma_dist = torch.exp((gamma * dist).clamp(max=80))
            wa_wirelength = (dist * exp_gamma_dist).sum() / exp_gamma_dist.sum()

            # Backward to get gradient w.r.t. filler positions
            wa_wirelength.backward()

            # Extract gradients
            force_x = filler_pos_x.grad
            force_y = filler_pos_y.grad

            if force_x is not None and force_y is not None:
                # Apply to pos.grad (negative because WA is minimized)
                pos.grad.data[filler_idx_x] -= force_x
                pos.grad.data[filler_idx_y] -= force_y

                if l_shape_log_verbose(model.params) >= 2:
                    logging.info(
                        f"FillerPseudoWire: selected={num_selected}, target=({target_x:.1f},{target_y:.1f}), "
                        f"wa_wirelength={wa_wirelength.item():.4e}, "
                        f"force_norm={(force_x.norm() ** 2 + force_y.norm() ** 2).sqrt():.4e}"
                    )

    with profile_scope(
        model.params,
        "place_obj.l_shape_apply_grad",
        tensor=pos.grad,
        logger=logging,
        iteration=current_iteration,
    ):
        if debug_hash_op is not None:
            debug_hash_op.log_debug_hash(
                "place_obj.l_shape_grad_weighted",
                pos.grad.data,
                weight="%.9e" % float(model.l_shape_routability_weight.item()),
            )
        pos.grad.data.add_(base_grad)
        model._apply_gradient_masks_only(pos.grad.data)
        if debug_hash_op is not None:
            debug_hash_op.log_debug_hash("place_obj.final_grad", pos.grad.data)
    model.l_shape_last_cost = float(l_shape_cost.item()) if model.l_shape_energy_valid else None
    model.l_shape_last_weighted_cost = (
        float(l_shape_weighted.item()) if model.l_shape_energy_valid else None
    )
    model.l_shape_last_weight = current_weight
    model.l_shape_last_target_weight = target_weight_value
    model.l_shape_last_weight_candidate = weight_candidate_value
    model.l_shape_last_cap_active = bool(cap_active)
    model.l_shape_last_base_grad_norm = base_grad_norm_value
    model.l_shape_last_grad_raw_norm = l_shape_grad_raw_norm_value
    model.l_shape_last_grad_norm = l_shape_grad_norm_value
    model.l_shape_last_grad_ratio = grad_ratio_value
    al_stats = getattr(density_op, "last_al_stats", None)
    if isinstance(al_stats, dict):
        model.l_shape_capacity_al_last_summary = {
            key: model._telemetry_scalar(value)
            for key, value in al_stats.items()
            if key != "reset_reason"
        }
        reset_reason = al_stats.get("reset_reason")
        if reset_reason is not None:
            model.l_shape_capacity_al_last_summary["reset_reason"] = str(reset_reason)
        if al_stats.get("enabled") and l_shape_log_verbose(model.params) >= 2:
            logging.info(
                "L-shape capacity AL telemetry: "
                "base_grad_norm=%.4e l_shape_grad_norm=%.4e "
                "l_shape_grad_raw_norm=%.4e "
                "l_shape_routability_weight=%.4e weighted_grad_ratio=%s "
                "E_total=%s q_h_max=%s q_v_max=%s lambda_h_max=%s lambda_v_max=%s",
                base_grad_norm_value,
                l_shape_grad_norm_value,
                l_shape_grad_raw_norm_value,
                current_weight,
                str(grad_ratio_value),
                str(model.l_shape_capacity_al_last_summary.get("E_cap_smooth_total")),
                str(model.l_shape_capacity_al_last_summary.get("q_h_max")),
                str(model.l_shape_capacity_al_last_summary.get("q_v_max")),
                str(model.l_shape_capacity_al_last_summary.get("lambda_h_max")),
                str(model.l_shape_capacity_al_last_summary.get("lambda_v_max")),
            )
    macro_stats = getattr(density_op, "last_macro_exclusion_stats", None)
    if isinstance(macro_stats, dict):
        model.l_shape_macro_exclusion_last_summary = {
            key: model._telemetry_scalar(value)
            for key, value in macro_stats.items()
            if key != "sources"
        }
        if macro_stats.get("macro_exclusion_enabled") and l_shape_log_verbose(model.params) >= 2:
            logging.info(
                "L-shape macro exclusion telemetry: "
                "macro_count=%s macro_source_active_bins=%s "
                "macro_body_bins=%s macro_halo_bins=%s "
                "macro_source_max=%s macro_source_sum=%s "
                "macro_usage_bins=%s macro_usage_max=%s macro_usage_sum=%s "
                "macro_source_grid_shape=%s macro_source_coordinate_system=%s "
                "macro_set_source=%s",
                str(model.l_shape_macro_exclusion_last_summary.get("macro_count")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_source_active_bins")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_body_bins")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_halo_bins")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_source_max")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_source_sum")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_usage_active_bins")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_usage_max")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_usage_sum")),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_source_grid_shape")),
                str(
                    model.l_shape_macro_exclusion_last_summary.get("macro_source_coordinate_system")
                ),
                str(model.l_shape_macro_exclusion_last_summary.get("macro_set_source")),
            )
    soft_debug = getattr(model.l_shape_routability_op, "cached_soft_debug", None)
    if isinstance(soft_debug, dict):
        model.soft_l_last_summary = {
            key: model._telemetry_scalar(soft_debug.get(key))
            for key in (
                "diag_edge_count",
                "mean_cost_gap",
                "raw_cost_gap_p50",
                "biased_cost_gap_p50",
                "tau_source_gap",
                "mean_max_prob",
                "mean_entropy",
                "near_tie_ratio",
                "tau",
                "effective_hotspot_weight",
                "resolver_agreement_ratio",
                "target_demand_supply_ratio",
                "same_net_topo_cache_present",
                "same_net_topo_nets",
                "same_net_topo_segments_h",
                "same_net_topo_segments_v",
                "same_net_topo_diag_edges",
                "same_net_topo_edges_with_topology",
                "same_net_topo_edges_with_observed_intervals",
                "same_net_topo_mean_gap",
                "same_net_topo_tie_ratio",
                "same_net_topo_zero_zero_edges",
                "same_net_topo_exact_equal_edges",
            )
        }
        model.soft_l_last_summary["current_demand_supply_ratio"] = model._telemetry_scalar(
            getattr(density_op, "last_demand_supply_ratio", None)
        )
    return obj
