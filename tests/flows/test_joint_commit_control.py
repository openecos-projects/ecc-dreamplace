"""Final request publication preserves the trace replacement boundary."""
from types import SimpleNamespace

from dreamplace.NonLinearPlace import NonLinearPlace


def test_finalization_replaces_an_existing_empty_trace():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    previous_trace = []
    placer.joint_buffer_commit_trace = previous_trace
    request = object()
    result = {"status": "request_ready", "projected_buffer_count": 1}
    placer.joint_coordinator = SimpleNamespace(
        is_segment_direct_joint=False, last_buffer_commit_request=request,
        prepare_buffer_commit_request=lambda **kwargs: result,
    )
    # The file-writing boundary is unrelated to trace ownership.
    placer._update_joint_flow_artifact = lambda params: None
    assert placer._finalize_joint_buffer_commit(SimpleNamespace(), iteration=4) == result
    assert placer.last_buffer_commit_request is request
    assert placer.joint_buffer_commit_trace == [result]
    assert placer.joint_buffer_commit_trace is not previous_trace
    assert previous_trace == []
