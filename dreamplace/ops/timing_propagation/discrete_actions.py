"""Opt-in four-action gradient context for fixed-master LUT timing."""

import copy
from dataclasses import dataclass, field

import torch


@dataclass
class DiscreteActionContext:
    """State needed to attach four zero-valued discrete action probes."""

    beta: torch.Tensor
    valid_instances: torch.Tensor
    target_cell_ids: torch.Tensor
    target_arc_ids: torch.Tensor
    target_senses: torch.Tensor
    valid_arcs: torch.Tensor
    inst_ids: torch.Tensor
    action_names: tuple[str, ...]
    topology_key: tuple
    mapping_summary: dict = field(default_factory=dict)
    current_cell_ids: torch.Tensor | None = None
    check_target_arc_ids: torch.Tensor | None = None
    check_inst_ids: torch.Tensor | None = None
    constraint_target_arc_ids: torch.Tensor | None = None
    constraint_inst_ids: torch.Tensor | None = None
    evaluate_targets: bool = False
    output_rcap_delta: torch.Tensor | None = None
    output_fcap_delta: torch.Tensor | None = None
    drive_load_rcap: torch.Tensor | None = None
    drive_load_fcap: torch.Tensor | None = None
    drive_load_node: torch.Tensor | None = None
    drive_load_pin_net: torch.Tensor | None = None
    drive_load_net_count: int = 0
    drive_load_driver_mask: torch.Tensor | None = None
    # PlaceObj feeds action pin caps through Elmore before timing. The
    # resulting load, delay and impulse already depend on beta, so timing
    # must not add the same cap delta again.
    rc_includes_action_cap: bool = False
    native_values_used: bool = False
    native_action_values_used: bool = False
    native_action_backward_used: bool = False
    action_cell_aat_calls: int = 0

    def validate(self, *, device, dtype):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self.beta.ndim != 2 or int(self.beta.shape[1]) != len(self.action_names):
            raise ValueError("discrete action beta must have shape [N, 4]")
        if self.beta.device != device or self.beta.dtype != dtype:
            raise ValueError(
                "discrete action beta device/dtype mismatch: "
                f"beta={self.beta.device}/{self.beta.dtype}, "
                f"timing={device}/{dtype}"
            )
        def check_tensor(name, *, expected_dtype=None, required=False):
            value = getattr(self, name)
            if value is None:
                if required:
                    raise ValueError(f"discrete action {name} is required")
                return
            if not torch.is_tensor(value):
                raise ValueError(f"discrete action {name} must be a tensor")
            if value.device != device:
                raise ValueError(
                    f"discrete action {name} device mismatch: {value.device} != {device}"
                )
            if expected_dtype is not None and value.dtype != expected_dtype:
                raise ValueError(
                    f"discrete action {name} dtype mismatch: "
                    f"{value.dtype} != {expected_dtype}"
                )

        for name, expected_dtype in (
            ("valid_instances", torch.bool),
            ("target_cell_ids", torch.long),
            ("target_arc_ids", torch.long),
            ("valid_arcs", torch.bool),
            ("target_senses", torch.long),
            ("inst_ids", torch.long),
        ):
            check_tensor(name, expected_dtype=expected_dtype, required=True)
        for name, expected_dtype in (
            ("check_target_arc_ids", torch.long),
            ("check_inst_ids", torch.long),
            ("constraint_target_arc_ids", torch.long),
            ("constraint_inst_ids", torch.long),
            ("current_cell_ids", torch.long),
            ("drive_load_node", torch.long),
            ("drive_load_pin_net", torch.long),
            ("drive_load_driver_mask", torch.bool),
        ):
            check_tensor(name, expected_dtype=expected_dtype)
        for name in (
            "output_rcap_delta",
            "output_fcap_delta",
            "drive_load_rcap",
            "drive_load_fcap",
        ):
            check_tensor(name, expected_dtype=dtype)
        if self.valid_instances.shape != self.beta.shape:
            raise ValueError("valid_instances shape must match beta")
        if self.target_arc_ids.shape != self.valid_arcs.shape:
            raise ValueError("target_arc_ids shape must match valid_arcs")
        if self.target_arc_ids.shape[0] != self.inst_ids.numel():
            raise ValueError("inst_ids must have one entry per runtime arc")

    def action_scores(self, gradient):
        """Return a finite ``dL/dbeta`` tensor for the top-k consumer."""

        if gradient is None:
            return torch.zeros_like(self.beta)
        scores = gradient.detach().to(
            device=self.beta.device,
            dtype=self.beta.dtype,
        )
        scores = torch.where(
            self.valid_instances,
            scores,
            torch.zeros_like(scores),
        )
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("lut_action produced a non-finite valid action gradient")
        return scores

    def apply_drive_load(self, cap_rise, cap_fall):
        """Make the fanin driver load depend on ``beta`` (paper Eq. 7 edge).

        Resizing a cell changes its input pin caps, which changes the load of
        the cells driving those nets.  ``d(load)/dbeta`` is this returned edge;
        the existing cell LUT load Jacobian and the timing adjoint then turn it
        into ``dC/dbeta`` without any post-hoc correction term.
        """

        if self.drive_load_rcap is None:
            return cap_rise, cap_fall
        weights = self.beta.index_select(0, self.drive_load_node)
        rise = (self.drive_load_rcap * weights).sum(dim=1)
        fall = (self.drive_load_fcap * weights).sum(dim=1)
        net_rise = rise.new_zeros(self.drive_load_net_count)
        net_fall = fall.new_zeros(self.drive_load_net_count)
        net_rise.index_add_(0, self.drive_load_pin_net, rise)
        net_fall.index_add_(0, self.drive_load_pin_net, fall)
        pin_rise = net_rise.index_select(0, self.drive_load_pin_net)
        pin_fall = net_fall.index_select(0, self.drive_load_pin_net)
        pin_rise = torch.where(self.drive_load_driver_mask, pin_rise, torch.zeros_like(pin_rise))
        pin_fall = torch.where(self.drive_load_driver_mask, pin_fall, torch.zeros_like(pin_fall))
        num_pins = int(cap_rise.shape[0])
        if int(pin_rise.shape[0]) < num_pins:
            # The timing lane appends Steiner nodes after the instance pins.
            pad = pin_rise.new_zeros(num_pins - int(pin_rise.shape[0]))
            pin_rise = torch.cat((pin_rise, pad))
            pin_fall = torch.cat((pin_fall, pad.to(pin_fall.dtype)))
        elif int(pin_rise.shape[0]) > num_pins:
            pin_rise = pin_rise[:num_pins]
            pin_fall = pin_fall[:num_pins]
        return cap_rise + pin_rise.to(cap_rise.dtype), cap_fall + pin_fall.to(cap_fall.dtype)

    def maybe_apply_drive_load(self, cap_rise, cap_fall):
        """Add the driver-load edge unless Elmore already consumed this beta."""

        if self.rc_includes_action_cap:
            return cap_rise, cap_fall
        return self.apply_drive_load(cap_rise, cap_fall)


def _drive_load_tables(data, table, *, device, dtype):
    """Per input pin cap delta for each of the four actions.

    Physical pin ownership supplies both the master and its Liberty offset.
    Runtime arc sources can instead be local clock indices, and multiple arcs
    can share an input. Neither is a valid domain for assigning pin caps.
    """

    nodes = data.pin2node_map.to(device=device, dtype=torch.long)
    current_cells = data.inst_cell_id.to(device=device, dtype=torch.long)
    safe_inst = nodes.clamp(min=0, max=current_cells.numel() - 1)
    targets = table.target_cell_ids.to(device=device)[safe_inst].long()
    starts = data.cell_id_2_libpin_id_start.to(device=device, dtype=torch.long)
    offsets = data.pin_2_libpin_offset.to(device=device, dtype=torch.long)
    cells = current_cells[safe_inst]
    safe_cells = cells.clamp(min=0, max=starts.numel() - 2)
    safe_targets = targets.clamp(min=0, max=starts.numel() - 2)
    target_pins = starts[safe_targets] + offsets.unsqueeze(1)
    current_pins = (starts[safe_cells] + offsets).unsqueeze(1)
    valid_pin = (nodes >= 0) & (nodes < current_cells.numel()) & (offsets >= 0)
    valid_pin &= (cells >= 0) & (cells + 1 < starts.numel())
    valid_pin &= current_pins[:, 0] < starts[safe_cells + 1]
    valid = table.valid_instances.to(device=device)[safe_inst] & valid_pin.unsqueeze(1)
    valid &= (targets >= 0) & (targets + 1 < starts.numel())
    valid &= target_pins < starts[safe_targets + 1]

    def one(pin_table):
        max_pin = max(int(pin_table.numel()) - 1, 0)
        in_bounds = (
            (target_pins >= 0)
            & (target_pins <= max_pin)
            & (current_pins >= 0)
            & (current_pins <= max_pin)
        )
        target = pin_table[target_pins.clamp(min=0, max=max_pin)]
        current = pin_table[current_pins.clamp(min=0, max=max_pin)]
        return torch.where(valid & in_bounds, target - current, torch.zeros_like(target))

    rcap = one(data.flat_lib_pin_rcap.to(device=device, dtype=dtype))
    fcap = one(data.flat_lib_pin_fcap.to(device=device, dtype=dtype))
    return rcap, fcap


def build_discrete_action_context(
    data,
    candidate_table,
    *,
    action_names,
    pin_names=None,
    device=None,
    dtype=None,
    requires_grad=True,
    load_delta=False,
    drive_load_feedback=False,
    cache=None,
):
    """Build a beta probe and all-arcs mapping for the active runtime table.

    ``cache`` stores only detached mapping data.  The beta tensor and the two
    validity masks are always per-context objects because the timing forward
    and native action backward are allowed to mutate those masks.  The cache
    key includes all runtime/library tensor versions that can change the
    action mapping, so a committed master or a topology/library update cannot
    reuse stale rows.
    """

    from dreamplace.ops.discrete_gradient_topk.action_arcs import (
        build_action_arc_table,
    )

    if device is None:
        device = data.inst_cell_id.device
    if dtype is None:
        dtype = (
            data.flat_libcell_info.dtype
            if torch.is_floating_point(data.flat_libcell_info)
            else torch.float32
        )
    def tensor_key(name):
        value = getattr(data, name, None)
        if value is None:
            return None
        return (
            int(value.data_ptr()),
            int(value._version),
            tuple(int(item) for item in value.shape),
        )

    pin_names_key = None if pin_names is None else hash(tuple(pin_names))
    endpoint_key = tuple(
        (kind, tensor_key(attribute))
        for kind, attribute in (
            ("check", "endpoints_timing_check_arcs"),
            ("constraint", "endpoints_constraint_arcs"),
        )
    )
    cache_key = (
        tensor_key("flat_inst_arcs_by_level"),
        tensor_key("inst_cell_id"),
        tensor_key("inst_is_sizeable"),
        tensor_key("inst_vt_mask"),
        tensor_key("flat_lib_pin_rcap"),
        tensor_key("flat_lib_pin_fcap"),
        tensor_key("cell_id_2_arc_id_start"),
        tensor_key("flat_libarc_info"),
        tensor_key("flat_libarc_check_types"),
        tensor_key("cell_id_2_libpin_id_start"),
        tensor_key("pin_2_libpin_offset"),
        tensor_key("pin2node_map"),
        tensor_key("pin2net_map"),
        tensor_key("net2driver_pin_map"),
        int(candidate_table.target_cell_ids.data_ptr()),
        int(candidate_table.target_cell_ids._version),
        tuple(action_names),
        pin_names_key,
        str(device),
        str(dtype),
        bool(drive_load_feedback),
        bool(load_delta),
        endpoint_key,
    )
    cache_hit = False
    cached = None if cache is None else cache.get("entry")
    output_cache = None if cache is None else cache.setdefault("output_tables", {})
    if cached is not None and cached.get("key") == cache_key:
        cache_hit = True
        table = cached["table"]
        endpoint_tables = cached["endpoint_tables"]
        drive_cache = cached.get("drive")
    else:
        table = build_action_arc_table(
            data,
            candidate_table,
            action_names=action_names,
            pin_names=pin_names,
            output_cache=output_cache,
            device=device,
        )
        endpoint_tables = {}
        for kind, attribute in (
            ("check", "endpoints_timing_check_arcs"),
            ("constraint", "endpoints_constraint_arcs"),
        ):
            endpoint_arcs = getattr(data, attribute, None)
            if endpoint_arcs is None or not endpoint_arcs.numel():
                continue
            endpoint_table = build_action_arc_table(
                data,
                candidate_table,
                action_names=action_names,
                pin_names=pin_names,
                runtime_arcs=endpoint_arcs,
                output_cache=output_cache,
                device=device,
            )
            has_checks = torch.zeros(
                table.valid_instances.shape[0], dtype=torch.bool, device=device
            )
            owners = endpoint_table.inst_ids
            has_checks[owners[owners >= 0]] = True
            table.valid_instances.logical_and_(
                ~has_checks[:, None] | endpoint_table.valid_instances
            )
            table.summary[kind + "_arc_count"] = int(endpoint_arcs.shape[0])
            endpoint_tables[kind] = endpoint_table
        drive_cache = None
        if drive_load_feedback:
            drive_cache = {}
        if cache is not None:
            cache["entry"] = {
                "key": cache_key,
                "table": table,
                "endpoint_tables": endpoint_tables,
                "drive": drive_cache,
            }
    endpoint_timing_ms = {
        kind: float(
            (endpoint.summary.get("timing_ms", {}) or {}).get("total", 0.0) or 0.0
        )
        for kind, endpoint in endpoint_tables.items()
    }
    table_timing_ms = float(
        (table.summary.get("timing_ms", {}) or {}).get("total", 0.0) or 0.0
    )
    table.summary["endpoint_timing_ms"] = endpoint_timing_ms
    table.summary["action_table_total_ms"] = table_timing_ms + sum(
        endpoint_timing_ms.values()
    )
    beta = torch.zeros(
        (int(data.inst_cell_id.numel()), len(action_names)),
        device=device,
        dtype=dtype,
        requires_grad=bool(requires_grad),
    )
    arcs = data.flat_inst_arcs_by_level
    topology_key = (
        int(arcs.data_ptr()),
        int(arcs._version),
        int(data.inst_cell_id.data_ptr()),
        int(data.inst_cell_id._version),
        int(candidate_table.target_cell_ids.data_ptr()),
        int(data.flat_libarc_info.data_ptr()),
    )
    drive_rcap = drive_fcap = None
    drive_node = drive_pin_net = None
    drive_net_count = 0
    if drive_load_feedback:
        if drive_cache is not None and "rcap" in drive_cache:
            drive_rcap = drive_cache["rcap"]
            drive_fcap = drive_cache["fcap"]
            drive_node = drive_cache["node"]
            drive_pin_net = drive_cache["pin_net"]
            drive_net_count = drive_cache["net_count"]
            drive_driver_mask = drive_cache["driver_mask"]
        else:
            drive_rcap, drive_fcap = _drive_load_tables(
                data, table, device=device, dtype=dtype
            )
            drive_node = data.pin2node_map.to(device=device, dtype=torch.long)
            drive_pin_net = data.pin2net_map.to(
                device=device, dtype=torch.long
            ).clamp(min=0)
            drive_net_count = max(int(drive_pin_net.max().item()) + 1, 1)
            net2driver_pin_map = data.net2driver_pin_map.to(
                device=device, dtype=torch.long
            )
            driver_of_pin = net2driver_pin_map[drive_pin_net]
            drive_driver_mask = driver_of_pin == torch.arange(
                int(drive_pin_net.numel()), device=driver_of_pin.device
            )
            if drive_cache is not None:
                drive_cache.update(
                    {
                        "rcap": drive_rcap,
                        "fcap": drive_fcap,
                        "node": drive_node,
                        "pin_net": drive_pin_net,
                        "net_count": drive_net_count,
                        "driver_mask": drive_driver_mask,
                    }
                )
    else:
        drive_driver_mask = None
    if load_delta:
        raise ValueError("lut_action target queries must use the base-point load")
    context = DiscreteActionContext(
        beta=beta,
        # These masks are intentionally private to this beta graph.  The
        # native timing backward can invalidate rows in-place.
        valid_instances=table.valid_instances.clone(),
        target_cell_ids=table.target_cell_ids,
        target_arc_ids=table.target_arc_ids,
        target_senses=table.target_senses,
        valid_arcs=table.valid_arcs.clone(),
        inst_ids=table.inst_ids,
        action_names=tuple(action_names),
        topology_key=topology_key,
        mapping_summary=copy.deepcopy(table.summary),
        current_cell_ids=data.inst_cell_id.detach().to(
            device=device, dtype=torch.long
        ).clone(),
        drive_load_rcap=drive_rcap,
        drive_load_fcap=drive_fcap,
        drive_load_node=drive_node,
        drive_load_pin_net=drive_pin_net,
        drive_load_net_count=drive_net_count,
        drive_load_driver_mask=drive_driver_mask,
    )
    for kind, endpoint_table in endpoint_tables.items():
        setattr(context, kind + "_target_arc_ids", endpoint_table.target_arc_ids)
        setattr(context, kind + "_inst_ids", endpoint_table.inst_ids)
    context.validate(device=device, dtype=dtype)
    context.mapping_summary["cache_mode"] = "hit" if cache_hit else "miss"
    context.mapping_summary["cache_hit"] = cache_hit
    return context


__all__ = ["DiscreteActionContext", "build_discrete_action_context"]
