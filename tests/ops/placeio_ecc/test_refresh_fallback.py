from types import SimpleNamespace

from dreamplace.ops.placeio_ecc import refresh as refresh_module
from dreamplace.ops.placeio_ecc.backend_caps import DEFAULT_BACKEND_CAPS
from dreamplace.ops.placeio_ecc.export_options import PyDbExportOptions
from dreamplace.ops.placeio_ecc.refresh import RefreshController


class _Native:
    def __init__(self, calls):
        self.calls = calls

    def destroy_sta(self):
        self.calls.append("destroy_sta")
        return True

    def def_init(self, path):
        self.calls.append(("def_init", path))
        return True


class _Module:
    def __init__(self, calls, bad, good):
        self.calls = calls
        self.ecc = _Native(calls)
        self._place_timing_active = True
        self._pydb_calls = 0
        self._bad = bad
        self._good = good

    def get_dmInst_ptr(self):
        return "dm"

    def pydb(self, dm_inst, bins_x, bins_y, routability, *, with_sta, **kwargs):
        self._pydb_calls += 1
        return self._bad if self._pydb_calls == 1 else self._good

    def refresh_place_timing(self, inputs):
        self.calls.append(("refresh_place_timing", inputs))
        return {"rcx": {"status": "ok"}}

    def def_save(self, path):
        self.calls.append(("def_save", path))
        return True


def test_refresh_falls_back_to_committed_def_on_duplicate_endpoints(tmp_path, monkeypatch):
    calls = []
    raw_db = SimpleNamespace(
        pydb="old",
        timing_enabled=True,
        timing_inputs={"lib_paths": ["cells.lib"], "sdc_path": "design.sdc"},
        route_num_bins_x=4,
        route_num_bins_y=4,
        with_routability=False,
        export_options=PyDbExportOptions(),
        workspace=str(tmp_path),
        backend_caps=DEFAULT_BACKEND_CAPS,
        refresh_generation=0,
        sparse_timing_view=None,
    )
    bad = SimpleNamespace()
    good = SimpleNamespace(
        timing_schema_version=2,
        num_nodes=1,
        pin_names=["p"],
        net_names=["n"],
    )
    raw_db.module = _Module(calls, bad, good)

    def validate(pydb):
        if pydb is bad:
            raise RuntimeError("ECC timing endpoint identities are invalid")
        return "validated"

    monkeypatch.setattr(refresh_module, "validate_timing_schema", validate)

    new_pydb, summary = RefreshController.refresh(raw_db)

    assert new_pydb is good
    assert raw_db.pydb is good
    assert raw_db.refresh_generation == 1
    assert summary["refresh_path"] == "committed_def_reload"
    assert summary["committed_def_path"].endswith("committed_refresh.def")
    assert ("def_save", summary["committed_def_path"]) in calls
    assert "destroy_sta" in calls
    assert ("def_init", summary["committed_def_path"]) in calls
    assert raw_db.module._place_timing_active is False
    assert [
        call[0] for call in calls if isinstance(call, tuple) and call[0] == "refresh_place_timing"
    ] == [
        "refresh_place_timing",
        "refresh_place_timing",
    ]
