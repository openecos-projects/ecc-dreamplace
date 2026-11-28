"""Fresh TP roundoff must not discard a complete physical buffer bundle."""

import pytest
from dreamplace.ops.placeio_ecc.commit_verify import commit_buffer_actions_verified
from test_commit_verify import FakeBackend, _action, _install


@pytest.mark.parametrize("metric", ["wns_ps", "tns_ps"])
@pytest.mark.parametrize("delta,kept", [(-2.3e-10, 1), (-1.0e-3, 0)])
def test_commit_distinguishes_roundoff_from_timing_regression(monkeypatch, metric, delta, kept):
    change = (delta, 0.0) if metric == "wns_ps" else (0.0, delta)
    backend = FakeBackend({"wns_ps": -100.0, "tns_ps": -400000.0}, {0: change})
    _install(monkeypatch, backend)
    result = commit_buffer_actions_verified(backend, [_action(0)], reader=lambda pydb: pydb.timing)
    assert (result["status"], result["kept_count"], result["reverted_count"]) == (
        "ok",
        kept,
        1 - kept,
    )
    assert backend.committed == ({0} if kept else set())
