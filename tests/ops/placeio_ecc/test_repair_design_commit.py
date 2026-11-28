from types import SimpleNamespace

import pytest

from dreamplace.ops.placeio_ecc.backend_caps import DEFAULT_BACKEND_CAPS
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
from dreamplace.ops.placeio_ecc.repair_profile import RepairProfile


class _FakePyDB:
    timing_schema_version = 2

    def __init__(self):
        self.inst_main_id = [0, 0, 1]
        self.main_id_2_cell_id_start = [0, 2, 3]
        self.inst_libcell_offset = [0, 1, 0]
        self.flat_libcell_names = ["A", "B", "C"]


@pytest.mark.parametrize("profiled", [False, True])
def test_apply_repair_design_keeps_unchanged_nodes_at_current_cell(monkeypatch, profiled):
    captured = {}

    def fake_apply_sizing(raw_db, cell_ids, cell_master_names, **kwargs):
        captured["cell_ids"] = list(cell_ids)
        captured["cell_master_names"] = list(cell_master_names)
        return {"ok": True, "accepted_count": 1, "requested_count": 1}

    monkeypatch.setattr(
        PlaceIOFunction,
        "apply_sizing",
        staticmethod(fake_apply_sizing),
    )
    raw_db = SimpleNamespace(
        timing_enabled=True,
        pydb=_FakePyDB(),
        backend_caps=DEFAULT_BACKEND_CAPS,
    )

    profile = RepairProfile() if profiled else None
    result = PlaceIOFunction.apply_repair_design(
        raw_db,
        {
            "resizes": [{"node_id": 1, "target_cell_index": 0}],
            "actions": [],
        },
        refresh=False,
        profile=profile,
    )

    assert captured["cell_ids"] == [0, 0, 2]
    assert captured["cell_master_names"] == ["A", "B", "C"]
    assert result == {
        "status": "committed",
        "sizing": {"ok": True, "accepted_count": 1, "requested_count": 1},
        "buffer": {"status": "skipped", "accepted_count": 0},
        "refresh": {"status": "skipped"},
        "requires_refresh": True,
    }
    if profile is not None:
        assert profile.commits == {"sizing": result["sizing"]}
        assert profile.timings_ms["native_sizing_commit_ms"] >= 0


def test_apply_repair_design_uses_sparse_sizing_and_syncs_carrier_positions():
    sizing_calls = []
    placement_calls = []

    class FakeEcc:
        @staticmethod
        def write_placement_back(dm, xs, ys):
            placement_calls.append((dm, list(xs), list(ys)))

    pydb = _FakePyDB()
    pydb.num_nodes = 3
    pydb.node_x = [1.0, 2.0, 3.0]
    pydb.node_y = [4.0, 5.0, 6.0]
    pydb.node_master_names = ["A", "B", "C"]
    pydb.apply_sizing = lambda node_ids, target_names: (
        sizing_calls.append((list(node_ids), list(target_names)))
        or {
            "ok": True,
            "accepted_count": len(node_ids),
            "requested_count": len(node_ids),
        }
    )

    raw_db = SimpleNamespace(
        timing_enabled=True,
        pydb=pydb,
        backend_caps=DEFAULT_BACKEND_CAPS,
        dm_inst="dm",
        dm_inst_ptr="dm",
        module=FakeEcc,
        ecc_module=SimpleNamespace(ecc=FakeEcc),
    )

    PlaceIOFunction.apply(raw_db, [10, 20], [40, 50])
    assert placement_calls == [("dm", [10, 20], [40, 50])]
    assert list(pydb.node_x) == [10.0, 20.0, 3.0]
    assert list(pydb.node_y) == [40.0, 50.0, 6.0]
    assert raw_db._carrier_positions_synced is True

    result = PlaceIOFunction.apply_repair_design(
        raw_db,
        {
            "resizes": [{"node_id": 1, "target_cell_index": 0}],
            "actions": [],
        },
        refresh=False,
    )
    assert sizing_calls == [([1], ["A"])]
    assert result["sizing"] == {
        "ok": True,
        "accepted_count": 1,
        "requested_count": 1,
    }

