import time

import torch

try:
    from dreamplace.ops.size_interpolated_pin.size_interpolated_pin_op import (
        size_interpolated_pin,
    )
except Exception:
    size_interpolated_pin = None


def _resolve_size_interpolated_pin_native_op_mode(data_collections):
    mode = getattr(data_collections, "size_interpolated_pin_native_op", "auto")
    mode = str(mode or "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "enable", "enabled"):
        mode = "on"
    elif mode in ("0", "false", "no", "off", "disable", "disabled"):
        mode = "off"
    if mode not in ("auto", "on", "off"):
        raise ValueError(
            "size_interpolated_pin_native_op must be one of: auto, on, off"
        )
    return mode


def _size_interpolated_pin_native_unavailable(mode, reason):
    if mode == "on":
        raise RuntimeError(
            "size_interpolated_pin_native_op=on requires native "
            f"size_interpolated_pin support: {reason}"
        )
    return None


def _interpolate_1d_piecewise_clamped(x, x_table, y_table, actual_dims):
    batch_size = x.shape[0]
    device = x.device

    if batch_size == 0:
        return x.new_zeros((0,), dtype=y_table.dtype)

    x_table = x_table.contiguous()
    y_table = y_table.contiguous()
    max_idx_actual = (actual_dims - 1).clamp(min=0)
    batch_indices = torch.arange(batch_size, device=device)
    x_min = x_table[:, 0]
    x_max = x_table[batch_indices, max_idx_actual]
    x_clamped = torch.minimum(torch.maximum(x, x_min), x_max)
    idx_padded = torch.searchsorted(x_table, x_clamped.unsqueeze(1), right=True).squeeze(1)
    idx_high = idx_padded.clamp(min=1).clamp(max=max_idx_actual)
    idx_low = (idx_high - 1).clamp(min=0)

    x0 = x_table[batch_indices, idx_low]
    x1 = x_table[batch_indices, idx_high]
    y0 = y_table[batch_indices, idx_low]
    y1 = y_table[batch_indices, idx_high]
    denom = x1 - x0
    safe_denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    factor = (x_clamped - x0) / safe_denom
    return torch.where(denom.abs() < 1e-12, y0, torch.lerp(y0, y1, factor))


def compute_current_pin2libpin_flat_ids(data_collections):
    required_names = (
        "pin2node_map",
        "inst_main_id",
        "inst_libcell_offset",
        "main_id_2_cell_id_start",
        "cell_id_2_libpin_id_start",
        "pin_2_libpin_offset",
    )
    if any(getattr(data_collections, name, None) is None for name in required_names):
        return None, None, None

    pin_nodes = data_collections.pin2node_map.long()
    pin_device = pin_nodes.device
    num_pins = pin_nodes.numel()
    inst_main_id = data_collections.inst_main_id[pin_nodes]
    inst_pins_mask = inst_main_id >= 0
    pin2libpin_flat_ids = torch.zeros(num_pins, device=pin_device, dtype=torch.long)

    if torch.any(inst_pins_mask):
        cell_ids = (
            data_collections.main_id_2_cell_id_start[inst_main_id[inst_pins_mask]].long()
            + data_collections.inst_libcell_offset[pin_nodes[inst_pins_mask]].long()
        )
        inst_pin2libpin_flat_ids = (
            data_collections.cell_id_2_libpin_id_start[cell_ids].long()
            + data_collections.pin_2_libpin_offset[inst_pins_mask].long()
        )
        pin2libpin_flat_ids[inst_pins_mask] = inst_pin2libpin_flat_ids

    return pin2libpin_flat_ids, inst_pins_mask, pin_nodes


def _vt_probability_rows(vt_var, node_ids, num_vt_classes, device, dtype):
    if vt_var is None:
        return None
    if vt_var.dim() == 2:
        rows = vt_var[node_ids].to(device=device, dtype=dtype)
        if rows.size(1) < num_vt_classes:
            padded = torch.zeros(
                (rows.size(0), num_vt_classes),
                device=device,
                dtype=dtype,
            )
            padded[:, : rows.size(1)] = rows
            rows = padded
        elif rows.size(1) > num_vt_classes:
            rows = rows[:, :num_vt_classes]
        return rows
    if vt_var.dim() == 1:
        rows = torch.zeros((node_ids.numel(), num_vt_classes), device=device, dtype=dtype)
        vt_index = vt_var[node_ids].long().to(device=device)
        valid = (vt_index >= 0) & (vt_index < num_vt_classes)
        if torch.any(valid):
            rows[torch.nonzero(valid, as_tuple=False).flatten(), vt_index[valid]] = 1.0
        return rows
    return None


def _size_interpolated_pin_property_cache_key(data_collections):
    names = (
        "pin2node_map",
        "inst_main_id",
        "inst_libcell_offset",
        "inst_is_sizeable",
        "pin_2_libpin_offset",
        "main_id_2_cell_id_start",
        "cell_id_2_libpin_id_start",
        "flat_libcell_info",
    )
    key = []
    for name in names:
        value = getattr(data_collections, name, None)
        if not isinstance(value, torch.Tensor):
            return None
        key.append(
            (
                name,
                tuple(value.shape),
                value.device.type,
                value.device.index,
                value.data_ptr(),
                getattr(value, "_version", None),
            )
        )
    return tuple(key)


def _get_size_interpolated_pin_property_cache(
    data_collections,
    pin2libpin_flat_ids,
    inst_pins_mask,
    pin_nodes,
):
    cache_key = _size_interpolated_pin_property_cache_key(data_collections)
    if cache_key is None:
        return None
    existing = getattr(data_collections, "_size_interpolated_pin_property_cache", None)
    if isinstance(existing, dict) and existing.get("key") == cache_key:
        return existing

    pin_device = pin2libpin_flat_ids.device
    cell_info = data_collections.flat_libcell_info
    cell_sizes = cell_info[:, 2].float().to(device=pin_device)
    cell_vts = cell_info[:, 3].long().to(device=pin_device)
    num_vt_classes = int(cell_vts.max().item()) + 1 if cell_vts.numel() else 0

    pin_offsets = data_collections.pin_2_libpin_offset.long()
    pin_main_ids = data_collections.inst_main_id[pin_nodes].long()
    sizeable_mask = (
        (pin_main_ids >= 0)
        & data_collections.inst_is_sizeable[pin_nodes]
        & (pin_offsets >= 0)
    )

    sizeable_pin_ids = torch.nonzero(sizeable_mask, as_tuple=False).flatten()
    sizeable_main_ids = pin_main_ids[sizeable_mask]
    sizeable_pin_offsets = pin_offsets[sizeable_mask]
    sizeable_pin_nodes = pin_nodes[sizeable_mask]
    main_groups = []

    if torch.any(sizeable_mask) and num_vt_classes > 0:
        for main_id in torch.unique(sizeable_main_ids):
            local_mask = sizeable_main_ids == main_id
            if not torch.any(local_mask):
                continue

            local_pin_ids = sizeable_pin_ids[local_mask]
            local_offsets = sizeable_pin_offsets[local_mask]
            local_pin_nodes = sizeable_pin_nodes[local_mask]

            cell_start = data_collections.main_id_2_cell_id_start[main_id].long()
            cell_end = data_collections.main_id_2_cell_id_start[main_id + 1].long()
            candidate_cell_ids = torch.arange(
                cell_start,
                cell_end,
                device=pin_device,
                dtype=torch.long,
            )
            if candidate_cell_ids.numel() == 0:
                continue

            candidate_sizes = cell_sizes[candidate_cell_ids]
            candidate_vts = cell_vts[candidate_cell_ids]
            vt_groups = []
            for vt_class in torch.unique(candidate_vts):
                vt_value = int(vt_class.item())
                vt_mask = candidate_vts == vt_class
                vt_cell_ids = candidate_cell_ids[vt_mask]
                if vt_cell_ids.numel() == 0:
                    continue
                vt_sizes = candidate_sizes[vt_mask]
                order = torch.argsort(vt_sizes)
                vt_cell_ids = vt_cell_ids[order]
                vt_sizes = vt_sizes[order]
                candidate_libpin_ids = (
                    data_collections.cell_id_2_libpin_id_start[vt_cell_ids].long().unsqueeze(0)
                    + local_offsets.unsqueeze(1)
                )
                actual_dims = torch.full(
                    (local_pin_nodes.numel(),),
                    vt_cell_ids.numel(),
                    device=pin_device,
                    dtype=torch.long,
                )
                vt_groups.append(
                    {
                        "vt_value": vt_value,
                        "vt_sizes": vt_sizes,
                        "candidate_libpin_ids": candidate_libpin_ids,
                        "actual_dims": actual_dims,
                    }
                )

            main_groups.append(
                {
                    "local_pin_ids": local_pin_ids,
                    "local_pin_nodes": local_pin_nodes,
                    "vt_groups": vt_groups,
                }
            )

    cache = {
        "key": cache_key,
        "pin2libpin_flat_ids": pin2libpin_flat_ids,
        "inst_pins_mask": inst_pins_mask,
        "has_inst_pins": bool(torch.any(inst_pins_mask).detach().cpu().item()),
        "pin_nodes": pin_nodes,
        "sizeable_mask": sizeable_mask,
        "has_sizeable": bool(torch.any(sizeable_mask).detach().cpu().item()),
        "sizeable_pin_nodes": sizeable_pin_nodes,
        "num_vt_classes": num_vt_classes,
        "main_groups": main_groups,
        "num_main_groups": len(main_groups),
    }
    try:
        setattr(data_collections, "_size_interpolated_pin_property_cache", cache)
    except Exception:
        pass
    return cache


def _get_size_interpolated_pin_native_cache(
    data_collections,
    pin2libpin_flat_ids,
    inst_pins_mask,
    pin_nodes,
):
    cache_key = _size_interpolated_pin_property_cache_key(data_collections)
    if cache_key is None:
        return None
    existing = getattr(data_collections, "_size_interpolated_pin_native_cache", None)
    if isinstance(existing, dict) and existing.get("key") == cache_key:
        return existing

    pin_device = pin2libpin_flat_ids.device
    cell_info = data_collections.flat_libcell_info
    cell_sizes = cell_info[:, 2].float().to(device=pin_device)
    cell_vts = cell_info[:, 3].long().to(device=pin_device)
    num_vt_classes = int(cell_vts.max().item()) + 1 if cell_vts.numel() else 0

    pin_offsets = data_collections.pin_2_libpin_offset.long()
    pin_main_ids = data_collections.inst_main_id[pin_nodes].long()
    sizeable_mask = (
        (pin_main_ids >= 0)
        & data_collections.inst_is_sizeable[pin_nodes]
        & (pin_offsets >= 0)
    )
    has_sizeable = bool(torch.any(sizeable_mask).detach().cpu().item())
    local_pin_ids = torch.nonzero(sizeable_mask, as_tuple=False).flatten()
    local_pin_nodes = pin_nodes[sizeable_mask]
    local_offsets = pin_offsets[sizeable_mask]
    local_main_ids = pin_main_ids[sizeable_mask]

    cache = {
        "key": cache_key,
        "pin2libpin_flat_ids": pin2libpin_flat_ids,
        "inst_pins_mask": inst_pins_mask,
        "has_inst_pins": bool(torch.any(inst_pins_mask).detach().cpu().item()),
        "pin_nodes": pin_nodes,
        "sizeable_mask": sizeable_mask,
        "has_sizeable": has_sizeable,
        "local_pin_ids": local_pin_ids,
        "local_pin_nodes": local_pin_nodes,
        "num_vt_classes": num_vt_classes,
    }
    if not has_sizeable or num_vt_classes <= 0:
        try:
            setattr(data_collections, "_size_interpolated_pin_native_cache", cache)
        except Exception:
            pass
        return cache

    group_specs = []
    kmax = 0
    for main_id in torch.unique(local_main_ids):
        main_value = int(main_id.item())
        cell_start = data_collections.main_id_2_cell_id_start[main_value].long()
        cell_end = data_collections.main_id_2_cell_id_start[main_value + 1].long()
        candidate_cell_ids = torch.arange(
            cell_start,
            cell_end,
            device=pin_device,
            dtype=torch.long,
        )
        if candidate_cell_ids.numel() == 0:
            continue
        candidate_sizes = cell_sizes[candidate_cell_ids]
        candidate_vts = cell_vts[candidate_cell_ids]
        for vt_class in torch.unique(candidate_vts):
            vt_value = int(vt_class.item())
            vt_mask = candidate_vts == vt_class
            vt_cell_ids = candidate_cell_ids[vt_mask]
            if vt_cell_ids.numel() == 0:
                continue
            vt_sizes = candidate_sizes[vt_mask]
            order = torch.argsort(vt_sizes)
            vt_cell_ids = vt_cell_ids[order]
            vt_sizes = vt_sizes[order]
            dim = int(vt_cell_ids.numel())
            kmax = max(kmax, dim)
            group_specs.append((main_value, vt_value, vt_sizes, vt_cell_ids, dim))

    if kmax <= 0:
        try:
            setattr(data_collections, "_size_interpolated_pin_native_cache", cache)
        except Exception:
            pass
        return cache

    total_pins = int(local_pin_ids.numel())
    candidate_sizes = torch.zeros(
        (total_pins, num_vt_classes, kmax),
        device=pin_device,
        dtype=torch.float32,
    )
    candidate_libpin_ids = torch.full(
        (total_pins, num_vt_classes, kmax),
        -1,
        device=pin_device,
        dtype=torch.long,
    )
    actual_dims = torch.zeros(
        (total_pins, num_vt_classes),
        device=pin_device,
        dtype=torch.int32,
    )
    for main_value, vt_value, vt_sizes, vt_cell_ids, dim in group_specs:
        if vt_value < 0 or vt_value >= num_vt_classes:
            continue
        row_mask = local_main_ids == main_value
        if not torch.any(row_mask):
            continue
        rows = torch.nonzero(row_mask, as_tuple=False).flatten()
        offsets = local_offsets[row_mask]
        candidate_sizes[rows, vt_value, :dim] = vt_sizes.to(
            device=pin_device,
            dtype=torch.float32,
        ).unsqueeze(0)
        candidate_libpin_ids[rows, vt_value, :dim] = (
            data_collections.cell_id_2_libpin_id_start[vt_cell_ids].long().unsqueeze(0)
            + offsets.unsqueeze(1)
        )
        actual_dims[rows, vt_value] = dim

    cache.update(
        {
            "candidate_sizes": candidate_sizes,
            "candidate_libpin_ids": candidate_libpin_ids,
            "actual_dims": actual_dims,
        }
    )
    try:
        setattr(data_collections, "_size_interpolated_pin_native_cache", cache)
    except Exception:
        pass
    return cache


def _get_size_interpolated_pin_native_payload(data_collections, cache):
    if cache is None or not cache.get("has_sizeable", False):
        return None
    existing = cache.get("native_payload")
    if isinstance(existing, dict):
        return existing

    direct_payload_names = (
        "local_pin_ids",
        "local_pin_nodes",
        "candidate_sizes",
        "candidate_libpin_ids",
        "actual_dims",
    )
    if all(name in cache for name in direct_payload_names):
        payload = {name: cache[name] for name in direct_payload_names}
        cache["native_payload"] = payload
        return payload

    if "main_groups" not in cache:
        return None

    pin_device = cache["pin2libpin_flat_ids"].device
    sizeable_pin_ids = []
    sizeable_pin_nodes = []
    candidate_dims = []
    for group in cache["main_groups"]:
        for vt_group in group["vt_groups"]:
            candidate_dims.append(int(vt_group["vt_sizes"].numel()))
    if not candidate_dims:
        return None
    kmax = max(candidate_dims)
    num_vt_classes = int(cache["num_vt_classes"])
    if kmax <= 0 or num_vt_classes <= 0:
        return None

    total_pins = sum(int(group["local_pin_ids"].numel()) for group in cache["main_groups"])
    if total_pins <= 0:
        return None

    candidate_sizes = torch.zeros(
        (total_pins, num_vt_classes, kmax),
        device=pin_device,
        dtype=torch.float32,
    )
    candidate_libpin_ids = torch.full(
        (total_pins, num_vt_classes, kmax),
        -1,
        device=pin_device,
        dtype=torch.long,
    )
    actual_dims = torch.zeros(
        (total_pins, num_vt_classes),
        device=pin_device,
        dtype=torch.int32,
    )

    offset = 0
    for group in cache["main_groups"]:
        local_pin_ids = group["local_pin_ids"]
        local_pin_nodes = group["local_pin_nodes"]
        local_count = int(local_pin_ids.numel())
        if local_count <= 0:
            continue
        sizeable_pin_ids.append(local_pin_ids)
        sizeable_pin_nodes.append(local_pin_nodes)
        row_slice = slice(offset, offset + local_count)
        for vt_group in group["vt_groups"]:
            vt_value = int(vt_group["vt_value"])
            if vt_value < 0 or vt_value >= num_vt_classes:
                continue
            dim = int(vt_group["vt_sizes"].numel())
            if dim <= 0:
                continue
            candidate_sizes[row_slice, vt_value, :dim] = vt_group["vt_sizes"].to(
                device=pin_device,
                dtype=torch.float32,
            ).unsqueeze(0)
            candidate_libpin_ids[row_slice, vt_value, :dim] = vt_group[
                "candidate_libpin_ids"
            ].to(device=pin_device, dtype=torch.long)
            actual_dims[row_slice, vt_value] = dim
        offset += local_count

    payload = {
        "local_pin_ids": torch.cat(sizeable_pin_ids, dim=0),
        "local_pin_nodes": torch.cat(sizeable_pin_nodes, dim=0),
        "candidate_sizes": candidate_sizes,
        "candidate_libpin_ids": candidate_libpin_ids,
        "actual_dims": actual_dims,
    }
    cache["native_payload"] = payload
    return payload


def _native_current_values_cache_key(flat_pin_values_list):
    key = []
    for flat_pin_values in flat_pin_values_list:
        key.append(
            (
                tuple(flat_pin_values.shape),
                flat_pin_values.device.type,
                flat_pin_values.device.index,
                flat_pin_values.dtype,
                flat_pin_values.data_ptr(),
            )
        )
    return tuple(key)


def _get_size_interpolated_pin_native_current_values(
    cache,
    flat_pin_values_list,
    property_dtype,
    pin_device,
):
    value_key = _native_current_values_cache_key(flat_pin_values_list)
    existing = cache.get("native_current_values")
    if isinstance(existing, dict) and existing.get("key") == value_key:
        return existing["values"]

    pin2libpin_flat_ids = cache["pin2libpin_flat_ids"]
    inst_pins_mask = cache["inst_pins_mask"]
    current_rows = []
    for flat_pin_values in flat_pin_values_list:
        current_values = torch.zeros(
            pin2libpin_flat_ids.numel(),
            device=pin_device,
            dtype=property_dtype,
        )
        if cache["has_inst_pins"]:
            current_values[inst_pins_mask] = flat_pin_values[
                pin2libpin_flat_ids[inst_pins_mask]
            ].to(device=pin_device, dtype=property_dtype)
        current_rows.append(current_values)
    current_values_tensor = torch.stack(current_rows, dim=0)
    cache["native_current_values"] = {
        "key": value_key,
        "values": current_values_tensor,
    }
    return current_values_tensor


def _compute_size_interpolated_pin_properties_native(
    data_collections,
    flat_pin_values_list,
    pin_mask=None,
):
    profile_enabled = bool(getattr(data_collections, "_size_interpolated_pin_profile_enabled", False))
    profile = {}

    def mark(name, started_at):
        if profile_enabled:
            profile[name] = profile.get(name, 0.0) + (time.perf_counter() - started_at) * 1000.0

    started = time.perf_counter()
    native_mode = _resolve_size_interpolated_pin_native_op_mode(data_collections)
    if native_mode == "off":
        return None
    if size_interpolated_pin is None:
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "extension is not available",
        )
    if not flat_pin_values_list:
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "flat_pin_values_list is empty",
        )
    if any(not torch.is_floating_point(v) for v in flat_pin_values_list):
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "flat pin values must be floating-point tensors",
        )
    property_dtype = flat_pin_values_list[0].dtype
    if any(v.dtype != property_dtype for v in flat_pin_values_list):
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "flat pin value tensors must share one dtype",
        )

    pin2libpin_flat_ids, inst_pins_mask, pin_nodes = compute_current_pin2libpin_flat_ids(
        data_collections
    )
    mark("pin2libpin_ms", started)
    if pin2libpin_flat_ids is None:
        if native_mode == "on":
            raise RuntimeError(
                "size_interpolated_pin_native_op=on requires pin-to-libpin "
                "mapping tensors"
            )
        return [None for _ in flat_pin_values_list]
    started = time.perf_counter()
    cache = _get_size_interpolated_pin_native_cache(
        data_collections,
        pin2libpin_flat_ids,
        inst_pins_mask,
        pin_nodes,
    )
    mark("cache_ms", started)
    if cache is None:
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "failed to build native cache",
        )
    pin2libpin_flat_ids = cache["pin2libpin_flat_ids"]
    pin_device = pin2libpin_flat_ids.device

    started = time.perf_counter()
    current_values_tensor = _get_size_interpolated_pin_native_current_values(
        cache,
        flat_pin_values_list,
        property_dtype,
        pin_device,
    )
    mark("current_values_ms", started)
    current_values_list = [current_values_tensor[idx] for idx in range(current_values_tensor.size(0))]

    size_var_getter = getattr(data_collections, "get_size_var", None)
    vt_var_getter = getattr(data_collections, "get_vt_var", None)
    size_var = None if size_var_getter is None else size_var_getter()
    vt_var = None if vt_var_getter is None else vt_var_getter()
    if (
        size_var is None
        or vt_var is None
        or getattr(data_collections, "flat_libcell_info", None) is None
        or getattr(data_collections, "inst_is_sizeable", None) is None
        or not cache["has_sizeable"]
        or cache["num_vt_classes"] <= 0
    ):
        return current_values_list

    started = time.perf_counter()
    payload = _get_size_interpolated_pin_native_payload(data_collections, cache)
    mark("payload_ms", started)
    if payload is None:
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "failed to build native payload",
        )
    local_pin_ids = payload["local_pin_ids"]
    local_pin_nodes = payload["local_pin_nodes"]
    if pin_mask is not None:
        active_pin_mask = torch.as_tensor(pin_mask, device=pin_device, dtype=torch.bool)
        if active_pin_mask.numel() != pin2libpin_flat_ids.numel():
            raise ValueError(
                "pin_mask length must match the design pin count: "
                f"mask={active_pin_mask.numel()} pins={pin2libpin_flat_ids.numel()}"
            )
        active_local_mask = active_pin_mask[local_pin_ids]
        if not torch.any(active_local_mask):
            return [current_values_tensor[idx].clone() for idx in range(current_values_tensor.size(0))]
        local_pin_ids = local_pin_ids[active_local_mask]
        local_pin_nodes = local_pin_nodes[active_local_mask]
        candidate_sizes = payload["candidate_sizes"][active_local_mask]
        candidate_libpin_ids = payload["candidate_libpin_ids"][active_local_mask]
        actual_dims = payload["actual_dims"][active_local_mask]
    else:
        candidate_sizes = payload["candidate_sizes"]
        candidate_libpin_ids = payload["candidate_libpin_ids"]
        actual_dims = payload["actual_dims"]

    started = time.perf_counter()
    vt_probs = _vt_probability_rows(
        vt_var,
        local_pin_nodes,
        cache["num_vt_classes"],
        pin_device,
        property_dtype,
    )
    mark("vt_probs_ms", started)
    if vt_probs is None:
        if native_mode == "on":
            raise RuntimeError(
                "size_interpolated_pin_native_op=on requires size/VT tensors "
                "with a supported shape"
            )
        return current_values_list

    started = time.perf_counter()
    local_sizes = size_var[local_pin_nodes].to(device=pin_device, dtype=property_dtype)
    candidate_sizes = candidate_sizes.to(dtype=property_dtype)
    flat_pin_values = torch.stack(
        [v.to(device=pin_device, dtype=property_dtype) for v in flat_pin_values_list],
        dim=0,
    )
    current_values_active = torch.stack(
        [current_values_tensor[idx, local_pin_ids] for idx in range(current_values_tensor.size(0))],
        dim=0,
    )
    mark("pack_ms", started)
    started = time.perf_counter()
    active_values = size_interpolated_pin(
        local_sizes,
        vt_probs,
        candidate_sizes,
        candidate_libpin_ids,
        actual_dims,
        flat_pin_values,
        current_values_active,
    )
    mark("op_ms", started)
    started = time.perf_counter()
    result_list = [current_values_tensor[idx].clone() for idx in range(current_values_tensor.size(0))]
    for prop_idx, result in enumerate(result_list):
        result[local_pin_ids] = active_values[prop_idx]
    mark("scatter_ms", started)
    if profile_enabled:
        try:
            setattr(data_collections, "_size_interpolated_pin_last_profile", profile)
        except Exception:
            pass
    return result_list


def _compute_size_interpolated_pin_properties_legacy(data_collections, flat_pin_values_list, pin_mask=None):
    if not flat_pin_values_list:
        return []
    pin2libpin_flat_ids, inst_pins_mask, pin_nodes = compute_current_pin2libpin_flat_ids(data_collections)
    if pin2libpin_flat_ids is None:
        return [None for _ in flat_pin_values_list]
    cache = _get_size_interpolated_pin_property_cache(
        data_collections,
        pin2libpin_flat_ids,
        inst_pins_mask,
        pin_nodes,
    )
    if cache is not None:
        pin2libpin_flat_ids = cache["pin2libpin_flat_ids"]
        inst_pins_mask = cache["inst_pins_mask"]
        pin_nodes = cache["pin_nodes"]
    has_inst_pins = (
        bool(torch.any(inst_pins_mask).detach().cpu().item())
        if cache is None
        else cache["has_inst_pins"]
    )

    pin_device = pin2libpin_flat_ids.device
    active_pin_mask = None
    if pin_mask is not None:
        active_pin_mask = torch.as_tensor(pin_mask, device=pin_device, dtype=torch.bool)
        if active_pin_mask.numel() != pin2libpin_flat_ids.numel():
            raise ValueError(
                "pin_mask length must match the design pin count: "
                f"mask={active_pin_mask.numel()} pins={pin2libpin_flat_ids.numel()}"
            )
    property_dtypes = [
        flat_pin_values.dtype
        if torch.is_floating_point(flat_pin_values)
        else torch.float32
        for flat_pin_values in flat_pin_values_list
    ]
    current_values_list = []
    for flat_pin_values, property_dtype in zip(flat_pin_values_list, property_dtypes):
        current_values = torch.zeros(
            pin2libpin_flat_ids.numel(),
            device=pin_device,
            dtype=property_dtype,
        )
        if has_inst_pins:
            current_values[inst_pins_mask] = flat_pin_values[
                pin2libpin_flat_ids[inst_pins_mask]
            ].to(property_dtype)
        current_values_list.append(current_values)

    size_var_getter = getattr(data_collections, "get_size_var", None)
    vt_var_getter = getattr(data_collections, "get_vt_var", None)
    size_var = None if size_var_getter is None else size_var_getter()
    vt_var = None if vt_var_getter is None else vt_var_getter()
    if (
        size_var is None
        or vt_var is None
        or getattr(data_collections, "flat_libcell_info", None) is None
        or getattr(data_collections, "inst_is_sizeable", None) is None
    ):
        return current_values_list

    size_var = size_var.float()
    if cache is None:
        cache = _get_size_interpolated_pin_property_cache(
            data_collections,
            pin2libpin_flat_ids,
            inst_pins_mask,
            pin_nodes,
        )
    if cache is None:
        return current_values_list
    num_vt_classes = cache["num_vt_classes"]
    sizeable_mask = cache["sizeable_mask"]
    if not cache["has_sizeable"] or num_vt_classes <= 0:
        return current_values_list

    interpolated_values_list = [
        current_values.clone() for current_values in current_values_list
    ]
    sizeable_pin_nodes = cache["sizeable_pin_nodes"]
    active_sizeable_pin_mask = None
    if active_pin_mask is not None:
        active_sizeable_pin_mask = active_pin_mask[cache["sizeable_mask"]]
        if not torch.any(active_sizeable_pin_mask):
            return interpolated_values_list
        sizeable_pin_nodes = sizeable_pin_nodes[active_sizeable_pin_mask]
    sizeable_vt_probs = _vt_probability_rows(
        vt_var,
        sizeable_pin_nodes,
        num_vt_classes,
        pin_device,
        torch.float32,
    )
    if sizeable_vt_probs is None:
        return current_values_list

    vt_prob_offset = 0
    for group in cache["main_groups"]:
        cached_local_pin_ids = group["local_pin_ids"]
        cached_local_pin_nodes = group["local_pin_nodes"]
        local_count = cached_local_pin_nodes.numel()
        if active_sizeable_pin_mask is None:
            local_active_mask = None
            local_pin_ids = cached_local_pin_ids
            local_pin_nodes = cached_local_pin_nodes
        else:
            local_active_mask = active_pin_mask[cached_local_pin_ids]
            if not torch.any(local_active_mask):
                continue
            local_pin_ids = cached_local_pin_ids[local_active_mask]
            local_pin_nodes = cached_local_pin_nodes[local_active_mask]
        local_sizes = size_var[local_pin_nodes].to(device=pin_device, dtype=torch.float32)
        if local_active_mask is not None:
            active_local_count = int(torch.count_nonzero(local_active_mask).detach().item())
            local_vt_probs = sizeable_vt_probs[
                vt_prob_offset : vt_prob_offset + active_local_count
            ]
            vt_prob_offset += active_local_count
        else:
            local_vt_probs = sizeable_vt_probs[
                vt_prob_offset : vt_prob_offset + local_count
            ]
            vt_prob_offset += local_count
        weighted_sums = [
            torch.zeros_like(local_sizes, dtype=property_dtype)
            for property_dtype in property_dtypes
        ]
        total_weights = [
            torch.zeros_like(local_sizes, dtype=property_dtype)
            for property_dtype in property_dtypes
        ]

        for vt_group in group["vt_groups"]:
            vt_value = vt_group["vt_value"]
            if local_active_mask is None:
                candidate_libpin_ids = vt_group["candidate_libpin_ids"]
                actual_dims = vt_group["actual_dims"]
            else:
                candidate_libpin_ids = vt_group["candidate_libpin_ids"][local_active_mask]
                actual_dims = vt_group["actual_dims"][local_active_mask]
            for prop_idx, (flat_pin_values, property_dtype) in enumerate(
                zip(flat_pin_values_list, property_dtypes)
            ):
                typed_local_sizes = local_sizes.to(
                    device=pin_device,
                    dtype=property_dtype,
                )
                typed_vt_sizes = vt_group["vt_sizes"].to(device=pin_device, dtype=property_dtype)
                x_table = typed_vt_sizes.unsqueeze(0).expand(local_sizes.numel(), -1)
                y_table = flat_pin_values[candidate_libpin_ids].to(property_dtype)
                vt_interp = _interpolate_1d_piecewise_clamped(
                    typed_local_sizes,
                    x_table,
                    y_table,
                    actual_dims,
                )
                vt_weight = (
                    local_vt_probs[:, vt_value].to(property_dtype)
                    if vt_value < local_vt_probs.size(1)
                    else torch.zeros_like(typed_local_sizes)
                )
                weighted_sums[prop_idx] = weighted_sums[prop_idx] + vt_weight * vt_interp
                total_weights[prop_idx] = total_weights[prop_idx] + vt_weight

        for prop_idx, interpolated_values in enumerate(interpolated_values_list):
            total_weight = total_weights[prop_idx]
            weighted_sum = weighted_sums[prop_idx]
            interpolated_values[local_pin_ids] = torch.where(
                total_weight > 0,
                weighted_sum / total_weight.clamp_min(1e-12),
                current_values_list[prop_idx][local_pin_ids],
            )

    return interpolated_values_list


def compute_size_interpolated_pin_properties(data_collections, flat_pin_values_list, pin_mask=None):
    native_results = _compute_size_interpolated_pin_properties_native(
        data_collections,
        flat_pin_values_list,
        pin_mask=pin_mask,
    )
    if native_results is not None:
        return native_results
    return _compute_size_interpolated_pin_properties_legacy(
        data_collections,
        flat_pin_values_list,
        pin_mask=pin_mask,
    )


def warmup_size_interpolated_pin_native_cache(data_collections):
    native_mode = _resolve_size_interpolated_pin_native_op_mode(data_collections)
    if native_mode == "off":
        return None
    if size_interpolated_pin is None:
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "extension is not available",
        )
    pin2libpin_flat_ids, inst_pins_mask, pin_nodes = compute_current_pin2libpin_flat_ids(
        data_collections
    )
    if pin2libpin_flat_ids is None:
        if native_mode == "on":
            raise RuntimeError(
                "size_interpolated_pin_native_op=on requires pin-to-libpin "
                "mapping tensors"
            )
        return None
    cache = _get_size_interpolated_pin_native_cache(
        data_collections,
        pin2libpin_flat_ids,
        inst_pins_mask,
        pin_nodes,
    )
    if cache is None:
        return _size_interpolated_pin_native_unavailable(
            native_mode,
            "failed to build native cache",
        )
    payload = _get_size_interpolated_pin_native_payload(data_collections, cache)
    if payload is None and native_mode == "on":
        raise RuntimeError(
            "size_interpolated_pin_native_op=on requires a native payload"
        )
    return payload


def compute_size_interpolated_pin_property(data_collections, flat_pin_values):
    results = compute_size_interpolated_pin_properties(data_collections, [flat_pin_values])
    return None if not results else results[0]


def _pin_property_at_size(
    data_collections,
    flat_pin_values,
    main_id: int,
    pin_offset: int,
    size_value: float,
    vt_probs: torch.Tensor | None,
):
    cell_info = data_collections.flat_libcell_info
    pin_device = cell_info.device
    property_dtype = (
        flat_pin_values.dtype if torch.is_floating_point(flat_pin_values) else torch.float32
    )
    cell_sizes = cell_info[:, 2].float().to(device=pin_device)
    cell_vts = cell_info[:, 3].long().to(device=pin_device)

    cell_start = data_collections.main_id_2_cell_id_start[main_id].long()
    cell_end = data_collections.main_id_2_cell_id_start[main_id + 1].long()
    candidate_cell_ids = torch.arange(
        cell_start,
        cell_end,
        device=pin_device,
        dtype=torch.long,
    )
    if candidate_cell_ids.numel() == 0:
        return None

    size_tensor = torch.tensor([float(size_value)], device=pin_device, dtype=property_dtype)
    if vt_probs is None:
        num_vt_classes = int(cell_vts.max().item()) + 1 if cell_vts.numel() else 0
        vt_probs = torch.ones((1, num_vt_classes), device=pin_device, dtype=property_dtype)
    else:
        vt_probs = vt_probs.reshape(1, -1).to(device=pin_device, dtype=property_dtype)

    candidate_sizes = cell_sizes[candidate_cell_ids]
    candidate_vts = cell_vts[candidate_cell_ids]
    weighted_sum = torch.zeros_like(size_tensor)
    total_weight = torch.zeros_like(size_tensor)

    for vt_class in torch.unique(candidate_vts):
        vt_value = int(vt_class.item())
        vt_mask = candidate_vts == vt_class
        vt_cell_ids = candidate_cell_ids[vt_mask]
        if vt_cell_ids.numel() == 0:
            continue
        vt_sizes = candidate_sizes[vt_mask]
        order = torch.argsort(vt_sizes)
        vt_cell_ids = vt_cell_ids[order]
        vt_sizes = vt_sizes[order].to(device=pin_device, dtype=property_dtype)
        candidate_libpin_ids = (
            data_collections.cell_id_2_libpin_id_start[vt_cell_ids].long()
            + int(pin_offset)
        )
        y_table = flat_pin_values[candidate_libpin_ids].reshape(1, -1).to(property_dtype)
        x_table = vt_sizes.unsqueeze(0)
        actual_dims = torch.tensor([vt_cell_ids.numel()], device=pin_device, dtype=torch.long)
        vt_interp = _interpolate_1d_piecewise_clamped(
            size_tensor,
            x_table,
            y_table,
            actual_dims,
        )
        vt_weight = (
            vt_probs[:, vt_value]
            if vt_value < vt_probs.size(1)
            else torch.zeros_like(size_tensor)
        )
        weighted_sum = weighted_sum + vt_weight * vt_interp
        total_weight = total_weight + vt_weight

    if torch.any(total_weight > 0):
        return float((weighted_sum / total_weight.clamp_min(1e-12))[0].item())
    return None


def build_size_interpolated_pin_property_debug(
    data_collections,
    pin_ids,
    property_specs,
):
    pin_ids_t = torch.as_tensor(pin_ids, dtype=torch.long)
    pin2node_map = getattr(data_collections, "pin2node_map", None)
    if pin2node_map is None or pin_ids_t.numel() == 0:
        return {"samples": []}

    pin2node_map = torch.as_tensor(pin2node_map, dtype=torch.long)
    size_var_getter = getattr(data_collections, "get_size_var", None)
    vt_var_getter = getattr(data_collections, "get_vt_var", None)
    size_var = None if size_var_getter is None else size_var_getter()
    vt_var = None if vt_var_getter is None else vt_var_getter()

    cell_info = getattr(data_collections, "flat_libcell_info", None)
    if cell_info is None:
        return {"samples": []}
    cell_info = torch.as_tensor(cell_info)
    cell_sizes = cell_info[:, 2].float()
    cell_vts = cell_info[:, 3].long()
    num_vt_classes = int(cell_vts.max().item()) + 1 if cell_vts.numel() else 0

    samples = []
    for pin_id in pin_ids_t.tolist():
        if pin_id < 0 or pin_id >= pin2node_map.numel():
            continue
        node_id = int(pin2node_map[pin_id].item())
        main_id = int(data_collections.inst_main_id[node_id].item())
        pin_offset = int(data_collections.pin_2_libpin_offset[pin_id].item())
        if main_id < 0 or pin_offset < 0:
            continue

        vt_probs = _vt_probability_rows(
            vt_var,
            torch.tensor([node_id], dtype=torch.long),
            num_vt_classes,
            cell_info.device,
            torch.float32,
        )
        current_size = None if size_var is None else float(size_var[node_id].item())
        cell_start = data_collections.main_id_2_cell_id_start[main_id].long()
        cell_end = data_collections.main_id_2_cell_id_start[main_id + 1].long()
        candidate_cell_ids = torch.arange(
            cell_start,
            cell_end,
            device=cell_info.device,
            dtype=torch.long,
        )
        legal_sizes = sorted(
            {
                float(cell_sizes[cell_id].item())
                for cell_id in candidate_cell_ids.tolist()
            }
        )
        if not legal_sizes:
            continue

        legal_rows = []
        for legal_size in legal_sizes:
            property_values = {}
            for property_name, spec in property_specs.items():
                property_values[property_name] = _pin_property_at_size(
                    data_collections,
                    spec["flat_pin_values"],
                    main_id,
                    pin_offset,
                    legal_size,
                    vt_probs,
                )
            legal_rows.append(
                {
                    "size": legal_size,
                    "properties": property_values,
                }
            )

        continuous_sizes = []
        if current_size is not None:
            continuous_sizes.append(current_size)
        else:
            for left_size, right_size in zip(legal_sizes[:-1], legal_sizes[1:]):
                continuous_sizes.append(0.5 * (left_size + right_size))
        continuous_rows = []
        seen_sizes = set()
        for sample_size in continuous_sizes:
            rounded_size = round(float(sample_size), 12)
            if rounded_size in seen_sizes:
                continue
            seen_sizes.add(rounded_size)
            property_values = {}
            for property_name, spec in property_specs.items():
                property_values[property_name] = _pin_property_at_size(
                    data_collections,
                    spec["flat_pin_values"],
                    main_id,
                    pin_offset,
                    sample_size,
                    vt_probs,
                )
            continuous_rows.append(
                {
                    "size": float(sample_size),
                    "properties": property_values,
                }
            )

        samples.append(
            {
                "pin_id": int(pin_id),
                "node_id": node_id,
                "main_id": main_id,
                "pin_offset": pin_offset,
                "current_size": current_size,
                "units": {
                    property_name: spec.get("unit")
                    for property_name, spec in property_specs.items()
                },
                "legal_size_points": legal_rows,
                "continuous_samples": continuous_rows,
            }
        )

    return {"samples": samples}
