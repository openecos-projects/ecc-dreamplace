import json
from types import SimpleNamespace

from dreamplace import Placer
from dreamplace import placer_cli


def test_openroad_backend_does_not_load_ieda(monkeypatch):
    engine = Placer.PlacementEngine.__new__(Placer.PlacementEngine)
    engine.params = SimpleNamespace(with_sta=True, place_io_engine="openroad")
    data_manager = object()

    def fail_if_loaded():
        raise AssertionError("OpenROAD backend must not load AiEDA/iEDA")

    monkeypatch.setattr(Placer, "_load_placeio_ieda", fail_if_loaded)

    assert engine._prepare_ieda_data_manager_for_sta(data_manager) is data_manager
    assert not hasattr(Placer, "IEDAIO")
    assert not hasattr(Placer, "IEDASta")


def test_openroad_save_placement_does_not_write_ieda_tcl(tmp_path, monkeypatch):
    engine = Placer.PlacementEngine.__new__(Placer.PlacementEngine)
    engine.params = SimpleNamespace(
        result_dir=str(tmp_path),
        place_io_engine="openroad",
        design_name=lambda: "standalone",
    )
    written_defs = []

    monkeypatch.setattr(engine, "write_back", written_defs.append)
    monkeypatch.setattr(
        Placer,
        "_load_placeio_ieda",
        lambda: (_ for _ in ()).throw(
            AssertionError("OpenROAD save path must not load AiEDA/iEDA")
        ),
    )

    engine.save_placement()

    assert written_defs == [str(tmp_path / "standalone" / "standalone.gp.def")]
    assert not (tmp_path / "standalone" / "standalone.tcl").exists()


def test_canonical_main_resolves_once_and_writes_effective_params(tmp_path, monkeypatch):
    params_path = tmp_path / "params.json"
    result_dir = tmp_path / "result"
    workspace = tmp_path / "workspace"
    params_path.write_text(
        json.dumps({"design_name": "entry_unit", "result_dir": str(result_dir)}),
        encoding="utf-8",
    )
    calls = []
    original = placer_cli.apply_flow_defaults

    def counted(params):
        calls.append(params)
        return original(params)

    class FakeEngine:
        def __init__(self, params):
            self.placedb = None
            self.placer = None
            self.last_run_result = {"status": "ok"}

        def setup_rawdb(self, data_manager):
            pass

        def run(self):
            return self.last_run_result

        def write_back(self, output_def):
            raise AssertionError("STA flow must not write a DEF")

    monkeypatch.setattr(placer_cli, "apply_flow_defaults", counted)
    monkeypatch.setattr(Placer, "PlacementEngine", FakeEngine)

    status = Placer.main(
        [
            str(params_path),
            "--flow-kind",
            "sta",
            "--workspace",
            str(workspace),
        ]
    )

    manifest = json.loads((workspace / "effective_params.json").read_text())
    result_manifest = json.loads(
        (result_dir / "effective_params.json").read_text()
    )
    assert status == 0
    assert len(calls) == 1
    assert manifest["resolved_params"]["flow_kind"] == "sta"
    assert manifest["run_context"]["workspace"] == str(workspace)
    assert result_manifest["run_context"]["result_dir"] == str(result_dir)
    assert result_manifest["run_context"]["manifest_scope"] == "result"
    assert len(manifest["content_sha256"]) == 64
