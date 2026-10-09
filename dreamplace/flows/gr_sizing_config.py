"""Validate the first GR timing mode before native database initialization."""

from dreamplace.ops.gpugr.xplace_backend import normalize_gpugr_backend


def configure_gr_sizing(params):
    mode = str(getattr(params, "timing_rc_mode", "placement"))
    if mode not in {"placement", "gr"}:
        raise ValueError("timing_rc_mode must be placement or gr")
    params.timing_rc_mode = mode
    if mode == "placement":
        return
    if getattr(params, "place_io_engine", "ieda") != "ecc" or params.gpu:
        raise ValueError("GR timing requires the native ECC backend and gpu=0")
    backend = normalize_gpugr_backend(getattr(params, "gpugr_backend", "auto"))
    if backend == "cuda":
        raise ValueError("GR timing requires a CPU routing backend: cpu_pr, cpu_pr_mt, or cpu_pr_maze")
    # This production lane is CPU-qualified even when the host has CUDA.
    params.gpugr_backend = "cpu_pr_mt" if backend == "auto" else backend
    rrr_iters = int(getattr(params, "gr_sizing_rrr_iters", 0))
    if rrr_iters < 0:
        raise ValueError("gr_sizing_rrr_iters must be non-negative")
    if rrr_iters > 0 and params.gpugr_backend != "cpu_pr_maze":
        raise ValueError("positive gr_sizing_rrr_iters requires gpugr_backend=cpu_pr_maze")
    if params.flow_kind not in {"sizing", "sta"}:
        raise ValueError("GR timing supports standalone sizing or read-only STA only")
    if getattr(params, "placement_sizing_mode", "place_only") == "joint":
        raise ValueError("GR timing does not support joint placement/sizing")
    unsupported = (
        "timing_opt_enabled",
        "routability_opt_flag",
        "l_shape_routability_flag",
        "adjust_gpugr_area_flag",
        "gpugr_final_eval_flag",
        "enable_relaxed_buffer_timing",
        "buffering_continuous_relaxed_optimization",
        "buffering_segment_count_tns_gradient",
        "joint_segment_virtual_density_enabled",
        "enable_net_weighting",
        "pin2pin_net_weighting",
    )
    active = [name for name in unsupported if getattr(params, name, False)]
    if active:
        raise ValueError("GR timing does not support: " + ", ".join(active))
    params.with_sta = 1
    params.diff_timing_driven_placement = 0
    params.cell_model_schema = "main_id_arc_offset_piecewise_linear"
    params.piecewise_gradient_mode = "native_piecewise"
    params.timing_propagation_device = "cpu"
    params.enable_fillers = 0
    params.gp_noise_ratio = 0.0
    params.random_center_init_flag = 0
    params.detailed_place_flag = 0
    params.detailed_place_engine = ""
    if params.flow_kind == "sta":
        params.differentiable_timing_obj = 0
        params.placement_sizing_mode = "place_only"
        params.sizing_parameterization = "logits"
        params.real_size_execution_mode = "continuous_only"
        params.real_size_warmup_steps = 0
        params.continuous_size_dynamics_mode = "none"
        params.global_place_flag = 0
        params.legalize_flag = 0
        params.timing_surrogate_mode = "lut_only"
        return
    params.placement_sizing_mode = "size_only"
    # The fixed S50 profile needs clock-to-Q size/VT gradients. "mixed"
    # evaluates those FF arcs with the exact current-master LUT instead.
    params.timing_surrogate_mode = "surrogate_only"
    params.sizing_parameterization = "real_size"
    params.real_size_execution_mode = "warmup_to_discrete"
    params.real_size_warmup_steps = 1
    params.continuous_size_dynamics_mode = "none"
    params.discrete_gradient_topk_shared_budget = 1
    params.discrete_gradient_topk_shared_budget_percent = 1.0
    params.discrete_gradient_topk_up_percent = 1.0
    params.discrete_gradient_topk_down_percent = 1.0
    params.discrete_gradient_topk_vt_percent = 1.0
    params.discrete_gradient_topk_preserve_vt = False
    params.timing_objective_lane = "timing_slew_cap"
    params.differentiable_timing_obj = 1
    params.early_stop_patience = 0
    params.early_stop_restore_best = True
    params.global_place_flag = 1
    params.legalize_flag = 1
    stage = dict(params.global_place_stages[0])
    stage.update(
        iteration=50, optimizer="adam", Llambda_density_weight_iteration=1, Lsub_iteration=1
    )
    params.global_place_stages = [stage]
