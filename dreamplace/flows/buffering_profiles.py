BUFFERING_DEFAULTS = (
    ("buffering_mode", "segment"),
    ("buffering_continuous_relaxed_optimization", 1),
    ("buffering_candidate_policy", "segment_only"),
    ("buffering_max_repeaters_per_segment", 3),
    ("buffering_fixed_buffer_master", "BUFX1H7L"),
    ("buffering_fixed_bsu_index", None),
    ("buffering_route_b_selection_fraction", 0.001),
    ("buffering_include_tree_node_candidates", 0),
    ("buffering_relaxed_timing_integration_mode", "dynamic_net_provider"),
    ("buffering_continuous_initial_relaxed_bu", 0.05),
    ("buffering_continuous_steps", 120),
    ("buffering_continuous_lr", 0.01),
    ("buffering_segment_strategy", "continuous"),
    ("buffering_candidate_strategy", "continuous"),
    ("buffering_segment_count_z_init", 0.1),
    ("buffering_segment_count_timing_backend", "cpp_cuda_segment_transfer_explicit_autograd"),
    ("buffering_segment_transfer_backend", "segment_transfer_native"),
    ("buffering_segment_integer_projection_interval", 0),
    ("buffering_segment_integer_projection_start_step", 0),
    ("buffering_segment_integer_projection_project_bsu", 0),
    ("buffering_segment_integer_projection_reset_optimizer_state", 1),
    ("buffering_segment_projection_min_z_to_insert", 0.5),
    ("buffering_commit_enabled", 1),
    ("timing_surrogate_mode", "lut_only"),
    ("timing_lut_2d_native_op", "auto"),
    ("size_interpolated_pin_native_op", "auto"),
)

SEGMENT_BUFFERING_DEFAULTS = (
    ("buffering_segment_count_tns_gradient", 1),
)

CANDIDATE_BUFFERING_DEFAULTS = (
    ("buffering_segment_count_tns_gradient", 0),
)

PHYSICAL_ECO_DEFAULTS = (
)

SEGMENT_PHYSICAL_ECO_DEFAULTS = (
)

CANDIDATE_PHYSICAL_ECO_DEFAULTS = (
)


def buffering_default_dict():
    defaults = dict(BUFFERING_DEFAULTS)
    defaults.update(SEGMENT_BUFFERING_DEFAULTS)
    return defaults


def physical_eco_default_dict():
    defaults = buffering_default_dict()
    defaults.update(PHYSICAL_ECO_DEFAULTS)
    defaults.update(SEGMENT_BUFFERING_DEFAULTS)
    defaults.update(SEGMENT_PHYSICAL_ECO_DEFAULTS)
    return defaults
