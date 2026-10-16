def refresh_buffering_runtime_state(data_collections, projection_result):
    if data_collections is None:
        return {"status": "skipped", "reason": "missing_data_collections"}
    if projection_result is None:
        return {"status": "skipped", "reason": "missing_projection_result"}

    projected_buffer_count = int(getattr(projection_result, "projected_buffer_count", 0))
    affected_nets = tuple(getattr(projection_result, "affected_nets", ()) or ())
    summary = {
        "status": "ok",
        "projected_buffer_count": projected_buffer_count,
        "affected_net_count": len(affected_nets),
    }
    data_collections.last_buffering_projection_result = projection_result
    data_collections.last_buffering_runtime_refresh_summary = summary

    payload = getattr(data_collections, "buffer_relaxed_timing_payload", None)
    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata", {}) or {})
        metadata["last_runtime_refresh"] = dict(summary)
        payload["metadata"] = metadata
        data_collections.buffer_relaxed_timing_payload = payload
    segment_payload = getattr(data_collections, "buffer_segment_count_payload", None)
    if isinstance(segment_payload, dict):
        metadata = dict(segment_payload.get("metadata", {}) or {})
        metadata["last_runtime_refresh"] = dict(summary)
        segment_payload["metadata"] = metadata
        data_collections.buffer_segment_count_payload = segment_payload
    return summary
