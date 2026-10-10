import math


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _as_int_list(value):
    return [int(item) for item in _as_list(value)]


def _as_float_map_by_index(value):
    result = {}
    for index, item in enumerate(_as_list(value)):
        result[int(index)] = float(item)
    return result


def _as_int_map_by_index(value):
    result = {}
    for index, item in enumerate(_as_list(value)):
        result[int(index)] = int(item)
    return result


def build_criticality_maps_from_timing_outputs(
    *,
    pin_slack,
    endpoint_incidence_result=None,
):
    sink_slack_by_pin = _as_float_map_by_index(pin_slack)
    incidence = (
        getattr(endpoint_incidence_result, "pin_endpoint_incidence_count", None)
        if endpoint_incidence_result is not None
        else None
    )
    npath_by_pin = _as_int_map_by_index(incidence)
    summary = {
        "criticality_source": "timing_propagation",
        "sink_slack_map_entry_count": len(sink_slack_by_pin),
        "npath_map_entry_count": len(npath_by_pin),
        "active_endpoint_count": int(
            getattr(endpoint_incidence_result, "active_endpoint_count", 0) or 0
        ),
    }
    return {
        "sink_slack_by_pin": sink_slack_by_pin,
        "npath_by_pin": npath_by_pin,
    }, summary


def _derive_active_endpoint_ids_from_timing_op(timing_op):
    active_endpoint_ids = _as_int_list(getattr(timing_op, "last_active_endpoint_ids", []))
    if active_endpoint_ids:
        return active_endpoint_ids

    endpoint_ids = _as_int_list(getattr(timing_op, "last_endpoint_ids_tensor", None))
    endpoint_slack = _as_list(getattr(timing_op, "last_endpoint_slack_tensor", None))
    if not endpoint_ids or len(endpoint_ids) != len(endpoint_slack):
        return None

    finite_endpoints = []
    negative_endpoints = []
    for endpoint_id, slack in zip(endpoint_ids, endpoint_slack):
        slack = float(slack)
        if not math.isfinite(slack):
            continue
        finite_endpoints.append((slack, int(endpoint_id)))
        if slack < 0.0:
            negative_endpoints.append(int(endpoint_id))
    if negative_endpoints:
        return negative_endpoints
    if finite_endpoints:
        _, endpoint_id = min(finite_endpoints, key=lambda item: (item[0], item[1]))
        return [int(endpoint_id)]
    return None


def build_criticality_maps_from_timing_op(timing_op, active_endpoint_ids=None):
    pin_slack = timing_op.get_pin_slack()
    if active_endpoint_ids is None:
        active_endpoint_ids = _derive_active_endpoint_ids_from_timing_op(timing_op)
    if active_endpoint_ids is None:
        endpoint_incidence_result = timing_op.compute_active_endpoint_incidence()
    else:
        endpoint_incidence_result = timing_op.compute_active_endpoint_incidence(
            active_endpoint_ids
        )
    return build_criticality_maps_from_timing_outputs(
        pin_slack=pin_slack,
        endpoint_incidence_result=endpoint_incidence_result,
    )
