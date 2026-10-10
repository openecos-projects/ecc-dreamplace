from enum import Enum


class FlowKind(str, Enum):
    PLACEMENT = "placement"
    STA = "sta"
    SIZING = "sizing"
    BUFFERING = "buffering"
    JOINT = "joint"
    PHYSICAL_ECO = "physical_eco"


PLACEMENT_DEFAULTS = (
    ("placement_sizing_mode", "place_only"),
    ("cell_model_schema", "main_id_arc_offset_piecewise_linear"),
    ("piecewise_gradient_mode", "native_piecewise"),
    ("sizing_parameterization", "logits"),
    ("real_size_learning_rate", 0.1),
    ("real_size_execution_mode", "continuous_only"),
    ("real_size_warmup_steps", 0),
    ("continuous_size_dynamics_mode", "none"),
    ("buffering_continuous_relaxed_optimization", False),
    ("buffering_segment_count_tns_gradient", False),
    ("diff_timing_driven_placement", 0),
    ("differentiable_timing_obj", 0),
    ("with_sta", 0),
)


PLACEMENT_RECIPE_DEFAULTS = (
    ("global_place_stages", [
        {
            "num_bins_x": 256,
            "num_bins_y": 256,
            "iteration": 3000,
            "learning_rate": 0.01,
            "wirelength": "weighted_average",
            "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1,
            "learning_rate_decay": 1.0,
        }
    ]),
    ("num_bins_x", 256),
    ("num_bins_y", 256),
    ("auto_adjust_bins", True),
    ("target_density", 0.8),
    ("density_weight", 8.0e-5),
    ("stop_overflow", 0.1),
    ("random_seed", 3000),
    ("random_center_init_flag", 1),
    ("gp_noise_ratio", 0.0),
    ("enable_fillers", 0),
    ("routability_opt_flag", 0),
    ("get_congestion_map", 0),
    ("deterministic_flag", 1),
)


TIMING_OBJECTIVE_DEFAULTS = (
    ("diff_timing_driven_placement", 1),
    ("differentiable_timing_obj", 1),
    ("with_sta", 1),
    ("timing_eval_flag", 1),
    ("timing_surrogate_mode", "lut_only"),
    ("timing_objective_lane", "timing_only"),
    ("critical_endpoint_pruning_mode", "off"),
    ("production_fast_loop", True),
    ("timing_lut_2d_native_op", "auto"),
    ("size_interpolated_pin_native_op", "auto"),
    ("timing_topology_refresh_interval", 15),
    ("timing_topology_enable_overflow_threshold", 0.35),
    ("timing_wns_coeff", 0.01),
    ("timing_tns_coeff", 0.0001),
    ("timing_slew_weight", 1.0),
    ("timing_cap_weight", 1.0),
    ("timing_placement_carrier", "direct_loss"),
    ("timing_gradient_net_weight_scale", 0.4),
    ("timing_gradient_net_weight_max", 2.0),
    ("enable_net_weighting", 1),
    ("net_weighting_scheme", "lilith"),
    ("net_weighting_npaths", 0.0),
    ("net_weighting_nendpoints", 0),
    ("net_weighting_update_interval", 15),
    ("pin2pin_weight", 0.0005),
    ("pin2pin_min_weight", 10.0),
    ("pin2pin_max_weight", 50.0),
    ("pin2pin_accumulate_weight", 0.2),
    ("pin2pin_net_weighting", 0),
)


TIMING_DRIVEN_PLACEMENT_DEFAULTS = (
    PLACEMENT_RECIPE_DEFAULTS + TIMING_OBJECTIVE_DEFAULTS
)


INFLATION_S5B1_DEFAULTS = PLACEMENT_RECIPE_DEFAULTS + TIMING_OBJECTIVE_DEFAULTS + (
    ("placement_sizing_mode", "place_only"),
    ("sizing_parameterization", "real_size"),
    ("continuous_size_dynamics_mode", "none"),
    ("routability_opt_flag", 1),
    ("l_shape_routability_flag", 1),
    ("l_shape_use_ggr_topology", 0),
    ("l_shape_capacity_al_enable", 0),
    ("l_direction_use_gpugr", 1),
    ("adjust_gpugr_area_flag", 1),
    ("adjust_rudy_area_flag", 0),
    ("adjust_nctugr_area_flag", 0),
    ("enhanced_inflation_replay_best_round_flag", 0),
    ("l_shape_overflow_threshold", 0.30),
    ("node_area_adjust_overflow", 0.15),
    ("enable_net_weighting", 0),
    ("pin2pin_net_weighting", 0),
    ("discrete_gradient_topk_shared_budget_percent", 1.0),
    ("buffering_mode", "segment"),
    ("buffering_segment_strategy", "discrete_net_gradient"),
    ("buffering_segment_count_z_init", 0.0),
    ("buffering_fixed_buffer_master", "BUFX1H7L"),
    ("buffering_fixed_bsu_index", None),
    ("buffering_max_repeaters_per_segment", 3),
    ("buffering_route_b_selection_fraction", 0.001),
    ("buffering_segment_count_timing_backend", "cpp_cpu_segment_transfer_explicit_autograd"),
    ("buffering_segment_transfer_backend", "segment_transfer_native"),
    ("buffering_segment_live_geometry", 1),
    ("buffering_segment_integer_projection_interval", 0),
    ("buffering_segment_integer_projection_start_step", 0),
    ("buffering_segment_integer_projection_project_bsu", 0),
    ("buffering_discrete_count_proximal_lambda", 0.0),
    ("relaxed_buffer_timing_integration_mode", "dynamic_net_provider"),
    ("joint_segment_virtual_density_enabled", 1),
)


PIN2PIN_TIMING_PLACEMENT_DEFAULTS = (
    PLACEMENT_RECIPE_DEFAULTS
    + (
        ("timing_placement_carrier", "pin2pin"),
        ("diff_timing_driven_placement", 0),
        ("differentiable_timing_obj", 0),
        ("with_sta", 1),
        ("timing_eval_flag", 1),
        ("timing_surrogate_mode", "lut_only"),
        ("timing_topology_refresh_interval", 15),
        ("timing_topology_enable_overflow_threshold", 0.35),
        ("enable_net_weighting", 1),
        ("net_weighting_scheme", "pin2pin"),
        ("net_weighting_nendpoints", 0),
        ("net_weighting_update_interval", 15),
        ("pin2pin_net_weighting", 1),
        ("pin2pin_weight", 0.0005),
        ("pin2pin_min_weight", 10.0),
        ("pin2pin_max_weight", 50.0),
        ("pin2pin_accumulate_weight", 0.2),
    )
)


SIZING_DEFAULTS = (
    ("placement_sizing_mode", "size_only"),
    ("sizing_parameterization", "logits"),
    ("real_size_learning_rate", 0.1),
    ("real_size_execution_mode", "continuous_only"),
    ("real_size_warmup_steps", 0),
    ("real_size_transition_backend", "vectorized"),
    ("real_size_transition_artifact_policy", "summary_only"),
    ("real_size_transition_profile", 0),
    ("continuous_size_dynamics_mode", "discrete_gradient_topk"),
    ("timing_surrogate_mode", "surrogate_only"),
    ("diff_timing_driven_placement", 0),
    ("timing_wns_coeff", 0.01),
    ("timing_tns_coeff", 0.0001),
    ("timing_slew_weight", 1.0),
    ("timing_cap_weight", 1.0),
    ("timing_grad_balance_target_ratio", 0.0),
    ("projection_resolver", "nearest_size_round"),
    ("cell_model_schema", "main_id_arc_offset_piecewise_linear"),
    ("piecewise_gradient_mode", "native_piecewise"),
    ("critical_endpoint_pruning_mode", "off"),
    ("production_fast_loop", True),
    ("timing_lut_2d_native_op", "auto"),
    ("size_interpolated_pin_native_op", "auto"),
    ("discrete_gradient_topk_ranking_mode", "real_size_taylor"),
    ("discrete_gradient_topk_up_percent", 30.0),
    ("discrete_gradient_topk_down_percent", 0.0),
    ("discrete_gradient_topk_vt_percent", 10.0),
    ("discrete_gradient_topk_shared_budget", 1),
    ("discrete_gradient_topk_shared_budget_percent", 1.0),
    ("early_stop_restore_best", True),
)


JOINT_DEFAULTS = (
    ("placement_sizing_mode", "joint"),
    ("sizing_parameterization", "logits"),
    ("real_size_learning_rate", 0.1),
    ("real_size_execution_mode", "continuous_only"),
    ("real_size_warmup_steps", 0),
    ("real_size_transition_backend", "reference"),
    ("real_size_transition_artifact_policy", "summary_only"),
    ("real_size_transition_profile", 0),
    ("continuous_size_dynamics_mode", "discrete_gradient_topk"),
    ("projection_resolver", "nearest_size_round"),
    ("cell_model_schema", "main_id_arc_offset_piecewise_linear"),
    ("piecewise_gradient_mode", "native_piecewise"),
    ("critical_endpoint_pruning_mode", "off"),
    ("production_fast_loop", True),
    ("timing_lut_2d_native_op", "auto"),
    ("size_interpolated_pin_native_op", "auto"),
    ("discrete_gradient_topk_ranking_mode", "real_size_taylor"),
    ("discrete_gradient_topk_up_percent", 30.0),
    ("discrete_gradient_topk_down_percent", 0.0),
    ("discrete_gradient_topk_vt_percent", 10.0),
    ("early_stop_restore_best", True),
    *TIMING_OBJECTIVE_DEFAULTS,
    ("timing_surrogate_mode", "mixed"),
    ("joint_buffer_commit_period", 100),
    ("joint_buffer_outer_iterations", 1),
    ("joint_buffer_max_per_segment", 3),
    ("joint_buffer_overflow_gate", 0.30),
    ("joint_post_commit_continue", True),
    ("joint_post_commit_openroad_eco", "none"),
    ("joint_sizing_activation_overflow", 0.30),
    ("joint_buffering_activation_overflow", 0.30),
)


PROXIMAL_ALTERNATING_JOINT_DEFAULTS = (
    ("joint_proximal_enabled", 1),
    ("joint_proximal_lambda_x", 1.0e-8),
    ("joint_proximal_lambda_s", 1.0e-4),
    ("joint_proximal_lambda_z", 1.0e-4),
    ("joint_proximal_lambda_b", 1.0e-4),
    ("joint_proximal_buffer_mu", 1.0e-4),
    ("timing_placement_carrier", "gradient_net_weight"),
    ("buffering_segment_projection_min_z_to_insert", 0.15),
    ("buffering_segment_projection_require_setup_criticality", 1),
    ("buffering_commit_enabled", 0),
)


STAGED_SMOKE_JOINT_DEFAULTS = (
    ("joint_staged_smoke_outer_iterations", 1),
    ("buffering_commit_enabled", 0),
    ("random_center_init_flag", 0),
    ("enable_fillers", 0),
    ("gp_noise_ratio", 0.0),
    ("auto_adjust_bins", False),
)


SEGMENT_COUNT_DIRECT_JOINT_DEFAULTS = (
    ("placement_sizing_mode", "joint"),
    ("continuous_size_dynamics_mode", "none"),
    ("timing_placement_carrier", "direct_loss"),
    ("timing_topology_enable_overflow_threshold", 0.35),
    ("timing_topology_refresh_interval", 15),
    ("enable_net_weighting", 1),
    ("net_weighting_scheme", "pin2pin"),
    ("pin2pin_net_weighting", 1),
    ("pin2pin_weight", 0.0005),
    ("pin2pin_min_weight", 10.0),
    ("pin2pin_max_weight", 50.0),
    ("pin2pin_accumulate_weight", 0.2),
    ("net_weighting_npaths", 0.0),
    ("net_weighting_nendpoints", 0),
    ("net_weighting_update_interval", 15),
    ("buffering_mode", "segment"),
    ("buffering_segment_strategy", "discrete_net_gradient"),
    ("buffering_candidate_strategy", "continuous"),
    ("buffering_segment_count_z_init", 0.0),
    ("buffering_fixed_buffer_master", "BUFX1H7L"),
    ("buffering_fixed_bsu_index", None),
    ("buffering_max_repeaters_per_segment", 3),
    ("buffering_route_b_selection_fraction", 0.001),
    ("buffering_segment_count_timing_backend", "cpp_cuda_segment_transfer_explicit_autograd"),
    ("buffering_segment_transfer_backend", "segment_transfer_native"),
    ("buffering_segment_live_geometry", 1),
    ("buffering_segment_capacity_enabled", 0),
    ("buffering_segment_integer_projection_interval", 0),
    ("buffering_segment_integer_projection_start_step", 0),
    ("buffering_segment_integer_projection_project_bsu", 0),
    ("buffering_segment_projection_min_z_to_insert", 0.5),
    ("joint_proximal_enabled", 1),
    ("joint_proximal_lambda_x", 1.0e-8),
    ("joint_proximal_lambda_s", 0.0),
    ("joint_proximal_lambda_z", 1.0e-4),
    ("joint_proximal_lambda_b", 0.0),
    ("joint_proximal_buffer_mu", 0.0),
    ("joint_buffer_commit_period", 0),
    ("joint_buffer_max_per_segment", 3),
    ("joint_buffer_overflow_gate", 0.30),
    ("joint_segment_sizing_enabled", 1),
    ("joint_segment_virtual_density_enabled", 1),
    ("joint_segment_route_b_enabled", 1),
    ("joint_segment_virtual_window_steps", 0),
    ("joint_segment_virtual_warmup_steps", 0),
    ("joint_segment_milestones", (0.30, 0.25, 0.20, 0.15, 0.10)),
    ("joint_segment_sizing_rounds", 5),
    ("joint_segment_sizing_up_percent", 10.0),
    ("joint_segment_buffering_rounds", 1),
    ("joint_segment_buffering_selection_fraction", 0.001),
    ("joint_segment_pin2pin_rebootstrap_enabled", 1),
    ("buffering_commit_enabled", 1),
    ("joint_post_commit_continue", False),
)


STA_DEFAULTS = (
    ("placement_sizing_mode", "size_only"),
    ("with_sta", 1),
    ("timing_surrogate_mode", "lut_only"),
    ("cell_model_schema", "main_id_arc_offset_piecewise_linear"),
    ("piecewise_gradient_mode", "native_piecewise"),
    ("critical_endpoint_pruning_mode", "off"),
    ("production_fast_loop", True),
    ("timing_lut_2d_native_op", "auto"),
    ("size_interpolated_pin_native_op", "auto"),
)


STAGED_JOINT_TDP_STAGE_DEFAULTS = TIMING_OBJECTIVE_DEFAULTS + (
    ("timing_topology_enable_overflow_threshold", 0.35),
    ("timing_placement_carrier", "gradient_net_weight"),
    ("enable_fillers", 0),
    ("auto_adjust_bins", False),
)

STAGED_JOINT_SIZING_STAGE_DEFAULTS = SIZING_DEFAULTS

STAGED_JOINT_TDP_ITERATIONS = 3000
STAGED_JOINT_SIZING_ITERATIONS = 50

SEGMENT_JOINT_OVERFLOW_MILESTONE_SCHEDULE = "overflow_milestones"
SEGMENT_JOINT_FIXED_WINDOW_SCHEDULE = "fixed_window_compat"
