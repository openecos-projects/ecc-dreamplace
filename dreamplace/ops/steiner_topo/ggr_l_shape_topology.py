import torch


H_FIRST = 0
V_FIRST = 1
STRAIGHT = 2

REQUIRED_FIELDS = (
    "pin_relate_x",
    "pin_relate_y",
    "net_steiner_start",
    "flat_pin_from",
    "flat_pin_to",
    "edge_l_directions",
    "pin_fa",
    "flat_pin_to_start",
    "net_flat_topo_sort",
    "net_flat_topo_sort_start",
    "net_vertex_start",
    "metadata",
)

REQUIRED_METADATA = (
    "schema_version",
    "num_pins",
    "num_vertices",
    "num_edges",
    "num_nets",
)


def use_ggr_l_shape_topology(params):
    return bool(getattr(params, "l_shape_use_ggr_topology", False))


def validate_ggr_l_shape_topology_params(params):
    capacity_al_enable = bool(getattr(params, "l_shape_capacity_al_enable", False))
    if capacity_al_enable and not use_ggr_l_shape_topology(params):
        raise RuntimeError(
            "l_shape_capacity_al_enable requires l_shape_use_ggr_topology=1"
        )
    if not use_ggr_l_shape_topology(params):
        return
    if not bool(getattr(params, "l_direction_use_gpugr", False)):
        raise RuntimeError(
            "l_shape_use_ggr_topology requires l_direction_use_gpugr=1"
        )
    if bool(getattr(params, "soft_l_assignment", False)):
        raise RuntimeError(
            "l_shape_use_ggr_topology is incompatible with soft_l_assignment=1"
        )


def _as_int32_tensor(pack, name, device):
    value = pack[name]
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype != torch.int32:
        raise RuntimeError(
            "GGR L-shape topology pack field '%s' has invalid dtype %s; expected int32"
            % (name, tensor.dtype)
        )
    if tensor.dim() != 1:
        raise RuntimeError(
            "GGR L-shape topology pack field '%s' must be a 1D tensor" % name
        )
    return tensor.contiguous()


def _require_monotonic(field_name, tensor):
    if tensor.numel() < 2:
        return
    if bool((tensor[1:] < tensor[:-1]).any().item()):
        raise RuntimeError(
            "GGR L-shape topology pack field '%s' must be monotonic" % field_name
        )


def _require_range(field_name, tensor, lower, upper, allow_negative_one=False):
    if tensor.numel() == 0:
        return
    valid = (tensor >= lower) & (tensor < upper)
    if allow_negative_one:
        valid = valid | (tensor == -1)
    if not bool(valid.all().item()):
        raise RuntimeError(
            "GGR L-shape topology pack field '%s' has index out of range [%d, %d)"
            % (field_name, lower, upper)
        )


def build_steiner_cache_from_ggr_pack(pack, pin_pos):
    missing = [name for name in REQUIRED_FIELDS if name not in pack]
    if missing:
        raise RuntimeError(
            "GGR L-shape topology pack missing field: %s" % ", ".join(missing)
        )
    metadata = dict(pack["metadata"])
    missing_meta = [name for name in REQUIRED_METADATA if name not in metadata]
    if missing_meta:
        raise RuntimeError(
            "GGR L-shape topology pack metadata missing field: %s"
            % ", ".join(missing_meta)
        )

    if pin_pos is None:
        raise RuntimeError("GGR L-shape topology pack loader requires pin_pos")
    if pin_pos.dim() != 1 or pin_pos.numel() % 2 != 0:
        raise RuntimeError("GGR L-shape topology pack loader requires flat pin_pos")

    device = pin_pos.device
    num_pins = int(metadata["num_pins"])
    num_vertices = int(metadata["num_vertices"])
    num_edges = int(metadata["num_edges"])
    num_nets = int(metadata["num_nets"])
    if num_pins <= 0 or num_vertices < num_pins or num_edges < 0 or num_nets < 0:
        raise RuntimeError("GGR L-shape topology pack metadata has invalid sizes")
    if pin_pos.numel() // 2 != num_pins:
        raise RuntimeError(
            "GGR L-shape topology pack metadata num_pins does not match pin_pos"
        )

    tensors = {
        name: _as_int32_tensor(pack, name, device)
        for name in REQUIRED_FIELDS
        if name != "metadata"
    }

    pin_relate_x = tensors["pin_relate_x"]
    pin_relate_y = tensors["pin_relate_y"]
    net_vertex_start = tensors["net_vertex_start"]
    net_steiner_start = tensors["net_steiner_start"]
    flat_pin_from = tensors["flat_pin_from"]
    flat_pin_to = tensors["flat_pin_to"]
    edge_l_directions = tensors["edge_l_directions"]
    pin_fa = tensors["pin_fa"]
    flat_pin_to_start = tensors["flat_pin_to_start"]
    net_flat_topo_sort = tensors["net_flat_topo_sort"]
    net_flat_topo_sort_start = tensors["net_flat_topo_sort_start"]

    expected_shapes = {
        "pin_relate_x": num_vertices,
        "pin_relate_y": num_vertices,
        "net_vertex_start": num_nets + 1,
        "net_steiner_start": num_nets + 1,
        "flat_pin_from": num_edges,
        "flat_pin_to": num_edges,
        "edge_l_directions": num_edges,
        "pin_fa": num_vertices,
        "flat_pin_to_start": num_vertices + 1,
        "net_flat_topo_sort": num_vertices,
        "net_flat_topo_sort_start": num_nets + 1,
    }
    for name, expected in expected_shapes.items():
        actual = int(tensors[name].numel())
        if actual != expected:
            raise RuntimeError(
                "GGR L-shape topology pack field '%s' has shape mismatch: got %d expected %d"
                % (name, actual, expected)
            )

    if int(net_steiner_start[0].item()) != num_pins:
        raise RuntimeError(
            "GGR L-shape topology pack net_steiner_start[0] must equal num_pins"
        )
    if int(net_steiner_start[-1].item()) != num_vertices:
        raise RuntimeError(
            "GGR L-shape topology pack net_steiner_start[-1] must equal num_vertices"
        )
    if int(flat_pin_to_start[-1].item()) != num_edges:
        raise RuntimeError(
            "GGR L-shape topology pack flat_pin_to_start[-1] must equal num_edges"
        )
    if int(net_flat_topo_sort_start[-1].item()) != num_vertices:
        raise RuntimeError(
            "GGR L-shape topology pack net_flat_topo_sort_start[-1] must equal num_vertices"
        )

    _require_monotonic("net_vertex_start", net_vertex_start)
    _require_monotonic("net_steiner_start", net_steiner_start)
    _require_monotonic("flat_pin_to_start", flat_pin_to_start)
    _require_monotonic("net_flat_topo_sort_start", net_flat_topo_sort_start)
    _require_range("pin_relate_x", pin_relate_x, 0, num_pins)
    _require_range("pin_relate_y", pin_relate_y, 0, num_pins)
    _require_range("flat_pin_from", flat_pin_from, 0, num_vertices)
    _require_range("flat_pin_to", flat_pin_to, 0, num_vertices)
    _require_range("pin_fa", pin_fa, 0, num_vertices, allow_negative_one=True)
    _require_range("net_flat_topo_sort", net_flat_topo_sort, 0, num_vertices)

    if num_pins > 0:
        identity = torch.arange(num_pins, dtype=torch.int32, device=device)
        if not bool(torch.equal(pin_relate_x[:num_pins], identity)):
            raise RuntimeError(
                "GGR L-shape topology pack pin_relate_x must be identity for original pins"
            )
        if not bool(torch.equal(pin_relate_y[:num_pins], identity)):
            raise RuntimeError(
                "GGR L-shape topology pack pin_relate_y must be identity for original pins"
            )

    allowed = (
        (edge_l_directions == H_FIRST)
        | (edge_l_directions == V_FIRST)
        | (edge_l_directions == STRAIGHT)
    )
    if not bool(allowed.all().item()):
        raise RuntimeError(
            "GGR L-shape topology pack contains invalid direction; allowed values are H_FIRST, V_FIRST, STRAIGHT"
        )

    pin_x = pin_pos[:num_pins]
    pin_y = pin_pos[num_pins:]
    newx = pin_x.index_select(0, pin_relate_x.to(dtype=torch.long))
    newy = pin_y.index_select(0, pin_relate_y.to(dtype=torch.long))

    cache_tuple = (
        newx.contiguous(),
        newy.contiguous(),
        pin_relate_x,
        pin_relate_y,
        net_vertex_start,
        net_steiner_start,
        pin_fa,
        flat_pin_to,
        flat_pin_from,
        flat_pin_to_start,
        net_flat_topo_sort,
        net_flat_topo_sort_start,
    )
    return cache_tuple, edge_l_directions.contiguous(), metadata
