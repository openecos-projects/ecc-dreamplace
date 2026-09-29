"""Project L-shape objective diagnostics into placement metrics."""

from dreamplace.ops.routability.profile_timing import l_shape_log_verbose


def collect_l_shape_telemetry(model, metric):
    """Copy objective-owned L-shape state into an evaluation metric."""
    l_shape_metric_active = (
        model.use_l_shape_routability
        or model.l_shape_last_cost is not None
        or model.l_shape_last_grad_norm is not None
    )
    if not l_shape_metric_active:
        return

    field_map = {
        "l_shape_fast_mode": "l_shape_fast_mode",
        "l_shape_energy_valid": "l_shape_energy_valid",
        "l_shape_cost": "l_shape_last_cost",
        "l_shape_weighted_cost": "l_shape_last_weighted_cost",
        "l_shape_weight": "l_shape_last_weight",
        "l_shape_target_weight": "l_shape_last_target_weight",
        "l_shape_weight_candidate": "l_shape_last_weight_candidate",
        "l_shape_cap_active": "l_shape_last_cap_active",
        "l_shape_base_grad_norm": "l_shape_last_base_grad_norm",
        "l_shape_grad_raw_norm": "l_shape_last_grad_raw_norm",
        "l_shape_grad_norm": "l_shape_last_grad_norm",
        "l_shape_grad_ratio": "l_shape_last_grad_ratio",
    }
    for metric_field, model_field in field_map.items():
        value = getattr(model, model_field, None)
        if value is not None:
            setattr(metric, metric_field, value)

    metric.l_shape_log_verbose = l_shape_log_verbose(model.params)
    metric.l_shape_target_ratio = float(model.l_shape_grad_target_ratio)
    overflow_ema = getattr(model, "_l_shape_overflow_ema", None)
    if overflow_ema is not None:
        metric.l_shape_overflow_ema = float(overflow_ema)

    summary_maps = (
        (
            "l_shape_capacity_al_last_summary",
            {
                "l_shape_capacity_al_enabled": "enabled",
                "l_shape_capacity_al_updated": "updated",
                "l_shape_capacity_al_g_h_max": "g_h_max",
                "l_shape_capacity_al_g_v_max": "g_v_max",
                "l_shape_capacity_al_g_h_sum": "g_h_sum",
                "l_shape_capacity_al_g_v_sum": "g_v_sum",
                "l_shape_capacity_al_g_h_pos_ratio": "g_h_pos_ratio",
                "l_shape_capacity_al_g_v_pos_ratio": "g_v_pos_ratio",
                "l_shape_capacity_al_q_h_max": "q_h_max",
                "l_shape_capacity_al_q_v_max": "q_v_max",
                "l_shape_capacity_al_q_h_sum": "q_h_sum",
                "l_shape_capacity_al_q_v_sum": "q_v_sum",
                "l_shape_capacity_al_lambda_h_max": "lambda_h_max",
                "l_shape_capacity_al_lambda_v_max": "lambda_v_max",
                "l_shape_capacity_al_lambda_h_sum": "lambda_h_sum",
                "l_shape_capacity_al_lambda_v_sum": "lambda_v_sum",
                "l_shape_capacity_al_energy_h": "E_cap_smooth_h",
                "l_shape_capacity_al_energy_v": "E_cap_smooth_v",
                "l_shape_capacity_al_energy_total": "E_cap_smooth_total",
                "l_shape_capacity_al_pq_h_min": "Pq_h_min",
                "l_shape_capacity_al_pq_h_max": "Pq_h_max",
                "l_shape_capacity_al_pq_h_sum": "Pq_h_sum",
                "l_shape_capacity_al_pq_v_min": "Pq_v_min",
                "l_shape_capacity_al_pq_v_max": "Pq_v_max",
                "l_shape_capacity_al_pq_v_sum": "Pq_v_sum",
                "l_shape_capacity_al_active_memory_bins_h": "active_memory_bins_h",
                "l_shape_capacity_al_active_memory_bins_v": "active_memory_bins_v",
            },
        ),
        (
            "l_shape_macro_exclusion_last_summary",
            {
                "l_shape_macro_exclusion_enabled": "macro_exclusion_enabled",
                "l_shape_macro_exclusion_macro_count": "macro_count",
                "l_shape_macro_exclusion_body_bins": "macro_body_bins",
                "l_shape_macro_exclusion_halo_bins": "macro_halo_bins",
                "l_shape_macro_exclusion_active_bins": "macro_source_active_bins",
                "l_shape_macro_exclusion_source_max": "macro_source_max",
                "l_shape_macro_exclusion_source_sum": "macro_source_sum",
                "l_shape_macro_exclusion_body_source_max": "macro_body_source_max",
                "l_shape_macro_exclusion_body_source_sum": "macro_body_source_sum",
                "l_shape_macro_exclusion_halo_source_max": "macro_halo_source_max",
                "l_shape_macro_exclusion_halo_source_sum": "macro_halo_source_sum",
                "l_shape_macro_exclusion_usage_max": "macro_usage_max",
                "l_shape_macro_exclusion_usage_sum": "macro_usage_sum",
                "l_shape_macro_exclusion_usage_bins": "macro_usage_active_bins",
                "l_shape_macro_exclusion_dominates_bins": "macro_dominates_bins",
                "l_shape_macro_exclusion_routing_dominates_bins": "routing_dominates_macro_bins",
            },
        ),
        (
            "soft_l_last_summary",
            {
                "soft_l_diag_count": "diag_edge_count",
                "soft_l_mean_cost_gap": "mean_cost_gap",
                "soft_l_raw_cost_gap_p50": "raw_cost_gap_p50",
                "soft_l_biased_cost_gap_p50": "biased_cost_gap_p50",
                "soft_l_tau_source_gap": "tau_source_gap",
                "soft_l_mean_max_prob": "mean_max_prob",
                "soft_l_mean_entropy": "mean_entropy",
                "soft_l_near_tie_ratio": "near_tie_ratio",
                "soft_l_tau": "tau",
                "soft_l_effective_hotspot_weight": "effective_hotspot_weight",
                "soft_l_resolver_agreement_ratio": "resolver_agreement_ratio",
                "soft_l_target_demand_supply_ratio": "target_demand_supply_ratio",
                "soft_l_current_demand_supply_ratio": "current_demand_supply_ratio",
                "soft_l_same_net_topo_nets": "same_net_topo_nets",
                "soft_l_same_net_topo_segments_h": "same_net_topo_segments_h",
                "soft_l_same_net_topo_segments_v": "same_net_topo_segments_v",
                "soft_l_same_net_topo_diag_edges": "same_net_topo_diag_edges",
                "soft_l_same_net_topo_edges_with_topology": "same_net_topo_edges_with_topology",
                "soft_l_same_net_topo_edges_with_observed_intervals": "same_net_topo_edges_with_observed_intervals",
                "soft_l_same_net_topo_mean_gap": "same_net_topo_mean_gap",
                "soft_l_same_net_topo_tie_ratio": "same_net_topo_tie_ratio",
            },
        ),
    )
    for summary_name, field_map in summary_maps:
        summary = getattr(model, summary_name, None) or {}
        for metric_field, summary_field in field_map.items():
            value = summary.get(summary_field)
            if value is not None or (
                summary_name == "l_shape_capacity_al_last_summary"
                and summary_field in summary
            ):
                setattr(metric, metric_field, value)
