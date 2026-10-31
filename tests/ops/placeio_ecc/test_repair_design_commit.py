from types import SimpleNamespace

from dreamplace.ops.placeio_ecc.backend_caps import DEFAULT_BACKEND_CAPS
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction


class _FakePyDB:
    timing_schema_version = 2

    def __init__(self):
        self.inst_main_id = [0, 0, 1]
        self.main_id_2_cell_id_start = [0, 2, 3]
        self.inst_libcell_offset = [0, 1, 0]
        self.flat_libcell_names = ["A", "B", "C"]


def test_apply_repair_design_keeps_unchanged_nodes_at_current_cell(monkeypatch):
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

    result = PlaceIOFunction.apply_repair_design(
        raw_db,
        {
            "resizes": [{"node_id": 1, "target_cell_index": 0}],
            "actions": [],
        },
        refresh=False,
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
