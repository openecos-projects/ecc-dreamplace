import hashlib


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return _decode(value.item())
        except ValueError:
            pass
    return str(value)


def _sequence(pydb, attr_name):
    value = getattr(pydb, attr_name, None)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _int_sequence(pydb, attr_name):
    return [int(value) for value in _sequence(pydb, attr_name)]


def _optional_scalar(pydb, attr_name):
    if not hasattr(pydb, attr_name):
        return None
    value = getattr(pydb, attr_name)
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(pydb, attr_name):
    if not hasattr(pydb, attr_name):
        return None
    return _decode(getattr(pydb, attr_name))


def _optional_int(pydb, attr_name):
    value = _optional_scalar(pydb, attr_name)
    return None if value is None else int(value)


def _hash_items(items):
    digest = hashlib.sha1()
    for item in items:
        digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _name_hash(pydb, attr_name):
    return _hash_items(sorted(_decode(value) for value in _sequence(pydb, attr_name)))


def _generic_sequence_hash(pydb, attr_name):
    return _hash_items(_decode(value) for value in _sequence(pydb, attr_name))


def _text_sequence(pydb, attr_name):
    return [_decode(value) for value in _sequence(pydb, attr_name)]


def _set_delta(lhs_values, rhs_values, limit=10):
    lhs_set = set(lhs_values)
    rhs_set = set(rhs_values)
    return {
        "only_lhs_count": len(lhs_set - rhs_set),
        "only_rhs_count": len(rhs_set - lhs_set),
        "only_lhs_sample": sorted(lhs_set - rhs_set)[:limit],
        "only_rhs_sample": sorted(rhs_set - lhs_set)[:limit],
    }


def _sequence_diffs(lhs_values, rhs_values, limit=10):
    diffs = []
    for index, (lhs_value, rhs_value) in enumerate(zip(lhs_values, rhs_values)):
        if lhs_value == rhs_value:
            continue
        diffs.append(
            {
                "index": index,
                "lhs": lhs_value,
                "rhs": rhs_value,
            }
        )
        if len(diffs) >= limit:
            break
    return diffs


def _presence(pydb, attr_name):
    return len(_sequence(pydb, attr_name)) > 0


def _referenced_node_names(pydb):
    node_names = _text_sequence(pydb, "node_names")
    referenced = set()
    for node_id in _int_sequence(pydb, "pin2node_map"):
        if 0 <= node_id < len(node_names):
            referenced.add(node_names[node_id])
    return referenced


def _placement_relevant_node_names(pydb, peer_pydb=None):
    node_names = _text_sequence(pydb, "node_names")
    relevant = _referenced_node_names(pydb)
    if peer_pydb is not None:
        relevant.update(set(node_names) & set(_text_sequence(peer_pydb, "node_names")))
    return relevant


def _build_diagnostics(lhs_pydb, rhs_pydb):
    diagnostics = {
        "set_deltas": {},
        "placement_relevant_set_deltas": {},
        "first_sequence_diffs": {},
        "optional_sequence_presence": {},
    }
    for attr_name in ("node_names", "pin_names", "net_names"):
        lhs_values = _text_sequence(lhs_pydb, attr_name)
        rhs_values = _text_sequence(rhs_pydb, attr_name)
        diagnostics["set_deltas"][attr_name] = _set_delta(lhs_values, rhs_values)
        diagnostics["first_sequence_diffs"][attr_name] = _sequence_diffs(
            lhs_values,
            rhs_values,
        )

    diagnostics["placement_relevant_set_deltas"]["node_names"] = _set_delta(
        _placement_relevant_node_names(lhs_pydb, rhs_pydb),
        _placement_relevant_node_names(rhs_pydb, lhs_pydb),
    )

    for attr_name in (
        "node_is_buffer",
        "node_master_names",
        "inst_libcell_offset",
        "flat_libcell_names",
        "flat_libcell_info",
        "flat_libcell_width",
        "flat_libcell_height",
        "flat_libcell_leakage",
        "flat_libcell_main_id2size_vt_limit",
    ):
        diagnostics["optional_sequence_presence"][attr_name] = {
            "lhs": _presence(lhs_pydb, attr_name),
            "rhs": _presence(rhs_pydb, attr_name),
        }
    return diagnostics


def _canonical_net_connectivity(pydb):
    pin_names = _text_sequence(pydb, "pin_names")
    net_names = _text_sequence(pydb, "net_names")
    flat_net2pin = _int_sequence(pydb, "flat_net2pin_map")
    flat_net2pin_start = _int_sequence(pydb, "flat_net2pin_start_map")
    if len(flat_net2pin_start) != len(net_names) + 1:
        return None

    rows = []
    for net_id, net_name in enumerate(net_names):
        start = flat_net2pin_start[net_id]
        end = flat_net2pin_start[net_id + 1]
        if start < 0 or end < start or end > len(flat_net2pin):
            return None
        pins = []
        for flat_id in range(start, end):
            pin_id = flat_net2pin[flat_id]
            if pin_id < 0 or pin_id >= len(pin_names):
                return None
            pins.append(pin_names[pin_id])
        rows.append((net_name, tuple(sorted(pins))))
    return tuple(sorted(rows))


def _pin_mapping_hash(pydb, attr_name, target_attr_name):
    pin_names = _text_sequence(pydb, "pin_names")
    target_names = _text_sequence(pydb, target_attr_name)
    mapping = _int_sequence(pydb, attr_name)
    if len(mapping) != len(pin_names):
        return None
    rows = []
    for pin_name, mapped_id in zip(pin_names, mapping):
        if mapped_id < 0 or mapped_id >= len(target_names):
            return None
        rows.append((pin_name, target_names[mapped_id]))
    return _hash_items(sorted(rows))


def _pin_name_subset_hash(pydb, attr_name):
    pin_names = _text_sequence(pydb, "pin_names")
    pin_ids = _int_sequence(pydb, attr_name)
    if not pin_ids:
        return None
    rows = []
    for pin_id in pin_ids:
        if pin_id < 0 or pin_id >= len(pin_names):
            return None
        rows.append(pin_names[pin_id])
    return _hash_items(sorted(rows))


def _count_invalid_pin_refs(pydb, attr_name):
    pin_count = len(_sequence(pydb, "pin_names"))
    invalid = 0
    for pin_id in _int_sequence(pydb, attr_name):
        if pin_id < 0 or pin_id >= pin_count:
            invalid += 1
    return invalid


def _node_size_hash(pydb, attr_name, included_node_names=None):
    node_names = _text_sequence(pydb, "node_names")
    sizes = _sequence(pydb, attr_name)
    if len(node_names) != len(sizes):
        return None
    included_node_names = included_node_names or set(node_names)
    return _hash_items(
        sorted(
            (node_name, float(size))
            for node_name, size in zip(node_names, sizes)
            if node_name in included_node_names
        )
    )


def _positive_area_node_names(pydb):
    node_names = _text_sequence(pydb, "node_names")
    size_x = _sequence(pydb, "node_size_x")
    size_y = _sequence(pydb, "node_size_y")
    if len(node_names) != len(size_x) or len(node_names) != len(size_y):
        return set()
    positive = set()
    for node_name, sx, sy in zip(node_names, size_x, size_y):
        if float(sx) > 0.0 and float(sy) > 0.0:
            positive.add(node_name)
    return positive


def _timing_check_class_histogram(pydb):
    histogram = {}
    for arc in _sequence(pydb, "endpoints_timing_check_arcs"):
        if len(arc) <= 6:
            continue
        key = str(int(arc[6]))
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def _timing_snapshot(pydb):
    end_points = _sequence(pydb, "end_points")
    start_points = _sequence(pydb, "start_points")
    endpoint_r_rat = _sequence(pydb, "endpoints_rRAT")
    endpoint_f_rat = _sequence(pydb, "endpoints_fRAT")
    return {
        "presence": {
            "start_points": hasattr(pydb, "start_points"),
            "end_points": hasattr(pydb, "end_points"),
            "endpoints_constraint_arcs": hasattr(pydb, "endpoints_constraint_arcs"),
            "endpoints_timing_check_arcs": hasattr(pydb, "endpoints_timing_check_arcs"),
            "endpoints_rRAT": hasattr(pydb, "endpoints_rRAT"),
            "endpoints_fRAT": hasattr(pydb, "endpoints_fRAT"),
        },
        "counts": {
            "start_points": len(start_points),
            "end_points": len(end_points),
            "endpoints_constraint_arcs": len(_sequence(pydb, "endpoints_constraint_arcs")),
            "endpoints_timing_check_arcs": len(_sequence(pydb, "endpoints_timing_check_arcs")),
            "endpoints_rRAT": len(endpoint_r_rat),
            "endpoints_fRAT": len(endpoint_f_rat),
            "invalid_start_point_pin_refs": _count_invalid_pin_refs(pydb, "start_points"),
            "invalid_end_point_pin_refs": _count_invalid_pin_refs(pydb, "end_points"),
        },
        "checks": {
            "endpoint_rat_length_match": (
                len(end_points) == len(endpoint_r_rat) == len(endpoint_f_rat)
            ),
        },
        "hashes": {
            "start_point_pin_names": _pin_name_subset_hash(pydb, "start_points"),
            "end_point_pin_names": _pin_name_subset_hash(pydb, "end_points"),
        },
        "metadata": {
            "timing_check_class_histogram": _timing_check_class_histogram(pydb),
        },
    }


def build_pydb_parity_snapshot(pydb, peer_pydb=None):
    node_names = _sequence(pydb, "node_names")
    pin_names = _sequence(pydb, "pin_names")
    net_names = _sequence(pydb, "net_names")
    net_connectivity = _canonical_net_connectivity(pydb)
    placement_relevant_node_names = _placement_relevant_node_names(pydb, peer_pydb)
    size_comparable_node_names = set(placement_relevant_node_names)
    if peer_pydb is not None:
        size_comparable_node_names &= _positive_area_node_names(pydb)
        size_comparable_node_names &= _positive_area_node_names(peer_pydb)
    hashes = {
        "node_names": _hash_items(sorted(placement_relevant_node_names)),
        "pin_names": _name_hash(pydb, "pin_names"),
        "net_names": _name_hash(pydb, "net_names"),
        "net_connectivity": None if net_connectivity is None else _hash_items(net_connectivity),
        "pin2node": _pin_mapping_hash(pydb, "pin2node_map", "node_names"),
        "pin2net": _pin_mapping_hash(pydb, "pin2net_map", "net_names"),
        "node_size_x": _node_size_hash(
            pydb,
            "node_size_x",
            size_comparable_node_names,
        ),
        "node_size_y": _node_size_hash(
            pydb,
            "node_size_y",
            size_comparable_node_names,
        ),
        "node_is_buffer": _generic_sequence_hash(pydb, "node_is_buffer"),
        "node_master_names": _generic_sequence_hash(pydb, "node_master_names"),
        "inst_libcell_offset": _generic_sequence_hash(pydb, "inst_libcell_offset"),
        "flat_libcell_names": _generic_sequence_hash(pydb, "flat_libcell_names"),
        "flat_libcell_info": _generic_sequence_hash(pydb, "flat_libcell_info"),
        "flat_libcell_width": _generic_sequence_hash(pydb, "flat_libcell_width"),
        "flat_libcell_height": _generic_sequence_hash(pydb, "flat_libcell_height"),
        "flat_libcell_leakage": _generic_sequence_hash(pydb, "flat_libcell_leakage"),
        "flat_libcell_main_id2size_vt_limit": _generic_sequence_hash(
            pydb,
            "flat_libcell_main_id2size_vt_limit",
        ),
    }
    return {
        "counts": {
            "nodes": int(getattr(pydb, "num_nodes", len(node_names))),
            "node_names": len(node_names),
            "placement_relevant_nodes": len(placement_relevant_node_names),
            "size_comparable_nodes": len(size_comparable_node_names),
            "pins": len(pin_names),
            "nets": len(net_names),
        },
        "geometry": {
            "die_bbox": {
                "xl": _optional_scalar(pydb, "xl"),
                "yl": _optional_scalar(pydb, "yl"),
                "xh": _optional_scalar(pydb, "xh"),
                "yh": _optional_scalar(pydb, "yh"),
            },
            "core_bbox": {
                "xl": _optional_scalar(pydb, "core_xl"),
                "yl": _optional_scalar(pydb, "core_yl"),
                "xh": _optional_scalar(pydb, "core_xh"),
                "yh": _optional_scalar(pydb, "core_yh"),
            },
            "site_width": _optional_scalar(pydb, "site_width"),
            "row_height": _optional_scalar(pydb, "row_height"),
        },
        "buffer": {
            "main_type_index": _optional_int(pydb, "buffer_main_type_index"),
            "main_type_status": _optional_text(pydb, "buffer_main_type_status"),
            "main_type_candidate_indices": _int_sequence(
                pydb,
                "buffer_main_type_candidate_indices",
            ),
        },
        "timing": _timing_snapshot(pydb),
        "hashes": hashes,
    }


def audit_pydb_parity(lhs_pydb, rhs_pydb, lhs_label="lhs", rhs_label="rhs"):
    lhs = build_pydb_parity_snapshot(lhs_pydb, peer_pydb=rhs_pydb)
    rhs = build_pydb_parity_snapshot(rhs_pydb, peer_pydb=lhs_pydb)
    core_mismatches = []
    metadata_mismatches = []
    timing_mismatches = []
    for key in ("placement_relevant_nodes", "pins", "nets"):
        if lhs["counts"].get(key) != rhs["counts"].get(key):
            core_mismatches.append(f"count_{key}")
    for key in ("node_names", "pin_names", "net_names", "pin2node", "pin2net"):
        if lhs["hashes"].get(key) != rhs["hashes"].get(key):
            core_mismatches.append(f"{key}_hash")
    if lhs["hashes"].get("net_connectivity") != rhs["hashes"].get("net_connectivity"):
        core_mismatches.append("net_connectivity_hash")
    for key in ("node_size_x", "node_size_y"):
        if lhs["hashes"].get(key) != rhs["hashes"].get(key):
            core_mismatches.append(f"{key}_hash")
    if lhs["geometry"] != rhs["geometry"]:
        core_mismatches.append("geometry")
    for key in ("start_points", "end_points"):
        if lhs["timing"]["presence"].get(key) != rhs["timing"]["presence"].get(key):
            timing_mismatches.append(f"{key}_presence")
        if lhs["timing"]["counts"].get(key) != rhs["timing"]["counts"].get(key):
            timing_mismatches.append(f"count_{key}")
    for key in ("invalid_start_point_pin_refs", "invalid_end_point_pin_refs"):
        if lhs["timing"]["counts"].get(key) != 0 or rhs["timing"]["counts"].get(key) != 0:
            timing_mismatches.append(key)
    if (
        lhs["timing"]["presence"].get("end_points")
        or rhs["timing"]["presence"].get("end_points")
    ) and (
        not lhs["timing"]["checks"].get("endpoint_rat_length_match")
        or not rhs["timing"]["checks"].get("endpoint_rat_length_match")
    ):
        timing_mismatches.append("endpoint_rat_length_match")
    for key in ("start_point_pin_names", "end_point_pin_names"):
        if lhs["timing"]["hashes"].get(key) != rhs["timing"]["hashes"].get(key):
            timing_mismatches.append(f"{key}_hash")
    if lhs["buffer"] != rhs["buffer"]:
        metadata_mismatches.append("buffer_metadata")
    for key in (
        "node_is_buffer",
        "node_master_names",
        "inst_libcell_offset",
        "flat_libcell_names",
        "flat_libcell_info",
        "flat_libcell_width",
        "flat_libcell_height",
        "flat_libcell_leakage",
        "flat_libcell_main_id2size_vt_limit",
    ):
        if lhs["hashes"].get(key) != rhs["hashes"].get(key):
            metadata_mismatches.append(f"{key}_hash")
    mismatches = core_mismatches + timing_mismatches + metadata_mismatches
    return {
        "artifact": "placeio_backend_pydb_parity_audit",
        "artifact_version": 3,
        "lhs_label": lhs_label,
        "rhs_label": rhs_label,
        "passed": not mismatches,
        "core_topology_passed": not core_mismatches,
        "timing_passed": not timing_mismatches,
        "metadata_passed": not metadata_mismatches,
        "mismatches": mismatches,
        "core_mismatches": core_mismatches,
        "timing_mismatches": timing_mismatches,
        "metadata_mismatches": metadata_mismatches,
        "lhs": lhs,
        "rhs": rhs,
        "diagnostics": _build_diagnostics(lhs_pydb, rhs_pydb),
    }
