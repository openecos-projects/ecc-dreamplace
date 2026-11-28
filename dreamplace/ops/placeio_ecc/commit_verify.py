"""Transport physical writes and Python TP observations to native verification.

C++ owns grouping, acceptance, rollback and retry decisions. Python supplies
only the existing placeio_ecc database and TimingPropagation boundaries.
"""

from .buffer_session import action_digest
from .repair_profile import measure_phase
from .timing_carrier import refresh_python_timing_carrier, require_carrier_schema

ARTIFACT = "placeio_ecc_commit_verified_buffers"
ARTIFACT_VERSION = 2


def timing_reader(pydb, *, config=None):
    """Evaluate a fresh physical timing carrier with Python TimingPropagation."""
    from ..rebuffer.timing_model import RebufferTimingModel
    from ..rebuffer.timing_profile import placement_timing_kwargs
    from .electrical_observation import electrical_observation

    resolved = dict(config or {})
    model = RebufferTimingModel(pydb, **placement_timing_kwargs(resolved))
    return {
        **model.baseline.as_dict(),
        "electrical": electrical_observation(
            model,
            max_fanout=resolved.get("max_fanout", 32),
            cap_margin=resolved.get("cap_margin", 0.0),
            slew_margin=resolved.get("slew_margin", 0.0),
        ),
    }


def commit_buffer_actions_verified(
    raw_db,
    actions,
    *,
    chunk_size=0,
    wns_margin_ps=0.0,
    tns_margin_ps=0.0,
    reader=timing_reader,
    profile=None,
):
    """Verify complete native trees, preserving chains across source nets."""
    from ..rebuffer import rebuffer_cpp
    from .place_io import PlaceIOFunction

    def apply(values):
        items = list(values or ())
        has_resize = any(
            item.get("action_kind") in {"resize_cell", "resize_revert"}
            for item in items
        )
        if not has_resize:
            with measure_phase(profile, "native_buffer_commit_ms"):
                return dict(
                    PlaceIOFunction.apply_buffer_actions(raw_db, items, action_digest(items))
                )

        resize_kinds = {"resize_cell", "resize_revert"}
        resize_items = [item for item in items if item.get("action_kind") in resize_kinds]
        buffer_items = [item for item in items if item.get("action_kind") not in resize_kinds]
        is_revert = any(item.get("action_kind") == "resize_revert" for item in resize_items)
        by_node = {}
        node_order = []
        for item in resize_items:
            node_id = int(item["node_id"])
            if node_id not in by_node:
                by_node[node_id] = dict(item)
                node_order.append(node_id)
            else:
                prev = by_node[node_id]
                merged = dict(item)
                if "previous_cell_index" in prev:
                    merged["previous_cell_index"] = prev["previous_cell_index"]
                if "previous_master_name" in prev:
                    merged["previous_master_name"] = prev["previous_master_name"]
                by_node[node_id] = merged
        dedup_resizes = [by_node[nid] for nid in node_order]
        committed_resizes = []
        if dedup_resizes:
            with measure_phase(profile, "native_sizing_commit_ms"):
                if is_revert and hasattr(raw_db.pydb, "apply_sizing"):
                    node_ids = [int(item["node_id"]) for item in dedup_resizes]
                    target_names = [str(item["target_master_name"]) for item in dedup_resizes]
                    sizing_summary = dict(raw_db.pydb.apply_sizing(node_ids, target_names) or {})
                else:
                    sizing_summary = dict(
                        PlaceIOFunction.apply_sparse_sizing(raw_db, dedup_resizes) or {}
                    )
            if not bool(sizing_summary.get("ok")) or int(
                sizing_summary.get("accepted_count", 0) or 0
            ) != len(dedup_resizes):
                return {
                    "status": "rejected",
                    "accepted_count": 0,
                    "removed_count": 0,
                    "rejected_count": len(items),
                    "failed_count": 0,
                    "committed_buffer_instance_names": [],
                    "committed_resizes": [],
                }
            if not is_revert:
                pydb = raw_db.pydb
                for item in dedup_resizes:
                    node_id = int(item["node_id"])
                    prev_idx = item.get("previous_cell_index")
                    if prev_idx is None:
                        main_id = int(pydb.inst_main_id[node_id])
                        prev_idx = (
                            int(pydb.main_id_2_cell_id_start[main_id])
                            + int(pydb.inst_libcell_offset[node_id])
                        )
                    prev_name = item.get("previous_master_name") or str(
                        pydb.flat_libcell_names[int(prev_idx)]
                    )
                    committed_resizes.append(
                        {
                            "node_id": node_id,
                            "instance_name": str(
                                item.get("instance_name") or pydb.node_names[node_id]
                            ),
                            "previous_cell_index": int(prev_idx),
                            "previous_master_name": str(prev_name),
                            "target_cell_index": int(item["target_cell_index"]),
                            "target_master_name": str(item["target_master_name"]),
                        }
                    )

        buffer_summary = {
            "status": "accepted",
            "accepted_count": 0,
            "removed_count": 0,
            "rejected_count": 0,
            "failed_count": 0,
            "committed_buffer_instance_names": [],
        }
        if buffer_items:
            with measure_phase(profile, "native_buffer_commit_ms"):
                buffer_summary = dict(
                    PlaceIOFunction.apply_buffer_actions(
                        raw_db, buffer_items, action_digest(buffer_items)
                    )
                    or {}
                )
        buf_status = str(buffer_summary.get("status") or "")
        if buf_status not in {"accepted", "ok"}:
            return {
                **buffer_summary,
                "committed_resizes": committed_resizes,
            }
        if is_revert:
            return {
                **buffer_summary,
                "status": "accepted",
                "removed_count": int(buffer_summary.get("removed_count", 0) or 0)
                + len(dedup_resizes),
            }
        return {
            **buffer_summary,
            "status": "accepted",
            "accepted_count": int(buffer_summary.get("accepted_count", 0) or 0)
            + len(resize_items),
            "committed_buffer_instance_names": list(
                buffer_summary.get("committed_buffer_instance_names", ()) or ()
            ),
            "committed_resizes": committed_resizes,
        }

    def refresh():
        kwargs = {"profile": profile} if profile is not None else {}
        _, summary = refresh_python_timing_carrier(raw_db, **kwargs)
        return dict(summary)

    def counts():
        return {
            "nodes": len(raw_db.pydb.node_names),
            "pins": len(raw_db.pydb.pin_names),
            "nets": len(raw_db.pydb.net_names),
        }

    result = dict(
        rebuffer_cpp.run_verified_buffer_commit(
            [dict(action) for action in actions or ()],
            lambda: require_carrier_schema(raw_db.pydb),
            apply,
            refresh,
            lambda: dict(reader(raw_db.pydb)),
            counts,
            action_digest,
            chunk_size=int(chunk_size),
            wns_margin_ps=float(wns_margin_ps),
            tns_margin_ps=float(tns_margin_ps),
        )
    )
    # Preserve the historical timing_after schema for callers that compare
    # TimingSnapshot dictionaries; the native verifier keeps the richer
    # electrical observation in a sibling field and in each chunk record.
    for field in ("timing_before", "timing_after"):
        value = result.get(field)
        if isinstance(value, dict) and "electrical" in value:
            result[f"{field}_electrical"] = value.pop("electrical")
    result.update(
        artifact=ARTIFACT,
        artifact_version=ARTIFACT_VERSION,
        timing_evaluator="dreamplace_timing_propagation_after_native_commit",
        refresh_deferred=False,
        ecc_sta_run=False,
        external_sta_run=False,
    )
    return result


__all__ = ["ARTIFACT", "commit_buffer_actions_verified", "timing_reader"]
