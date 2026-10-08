from types import SimpleNamespace

import pytest

from dreamplace.ops.placeio_openroad.place_io import PlaceIOFunction


class FakeOpenRoadBridge:
    def __init__(self):
        self.calls = 0
        self.pydb = SimpleNamespace(num_nodes=2, num_pins=3, num_nets=1)

    def sync_from_openroad(self):
        self.calls += 1
        return self.pydb


def test_sync_from_openroad_accepts_topo_mode_and_reports_summary():
    bridge = FakeOpenRoadBridge()

    pydb, summary = PlaceIOFunction.sync_from_openroad(
        bridge,
        refresh_mode="topo",
        return_summary=True,
    )

    assert pydb is bridge.pydb
    assert bridge.calls == 1
    assert summary["status"] == "ok"
    assert summary["requested_refresh_mode"] == "topo"
    assert summary["effective_refresh_mode"] == "topo"


def test_sync_from_openroad_accepts_all_mode():
    bridge = FakeOpenRoadBridge()

    pydb, summary = PlaceIOFunction.sync_from_openroad(
        bridge,
        refresh_mode="all",
        return_summary=True,
    )

    assert pydb is bridge.pydb
    assert summary["effective_refresh_mode"] == "all"


def test_sync_from_openroad_rejects_unsupported_mode():
    with pytest.raises(RuntimeError, match="unsupported OpenROAD refresh_mode"):
        PlaceIOFunction.sync_from_openroad(
            FakeOpenRoadBridge(),
            refresh_mode="affected_net",
        )
