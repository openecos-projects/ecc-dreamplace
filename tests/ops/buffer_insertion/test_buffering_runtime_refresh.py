from types import SimpleNamespace

from dreamplace.ops.buffer_insertion.projection import BufferingProjectionResult
from dreamplace.ops.buffer_insertion.runtime_refresh import (
    refresh_buffering_runtime_state,
)


def test_runtime_refresh_skips_without_data_collections():
    result = BufferingProjectionResult(mode="candidate")

    summary = refresh_buffering_runtime_state(None, result)

    assert summary["status"] == "skipped"
    assert summary["reason"] == "missing_data_collections"


def test_runtime_refresh_records_projection_without_openroad():
    projection = BufferingProjectionResult(
        mode="candidate",
        selected_actions=({"net_id": 7},),
        projected_buffer_count=1,
        affected_nets=(7,),
    )
    data = SimpleNamespace(buffer_relaxed_timing_payload={"metadata": {}})

    summary = refresh_buffering_runtime_state(data, projection)

    assert summary == {
        "status": "ok",
        "projected_buffer_count": 1,
        "affected_net_count": 1,
    }
    assert data.last_buffering_projection_result is projection
    assert data.last_buffering_runtime_refresh_summary == summary
    assert (
        data.buffer_relaxed_timing_payload["metadata"]["last_runtime_refresh"]
        == summary
    )
