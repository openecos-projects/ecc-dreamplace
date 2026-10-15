#!/usr/bin/env python3

import sys
import types
import unittest
from pathlib import Path
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))


class OptimizationFlowTest(unittest.TestCase):
    def test_sta_only_success_does_not_treat_hpwl_sentinel_as_failure(self):
        from dreamplace.flows.flow_config import FlowKind
        from dreamplace.flows.optimization_flow import _is_failed_run_result

        result = {"status": "ok", "hpwl": float("inf")}
        self.assertFalse(_is_failed_run_result(result, FlowKind.STA))
        self.assertTrue(_is_failed_run_result(result, FlowKind.PLACEMENT))

    def _resolved_params(self, **kwargs):
        from dreamplace.flows.flow_config import apply_flow_defaults

        params = types.SimpleNamespace(**kwargs)
        apply_flow_defaults(params)
        return params

    def test_run_optimization_flow_invokes_existing_engine_sequence(self):
        from dreamplace.flows.flow_config import FlowKind
        from dreamplace.flows.optimization_flow import run_optimization_flow

        calls = []
        created_engines = []

        class FakeEngine:
            def __init__(self, params):
                calls.append(("init", params))
                created_engines.append(self)

            def setup_rawdb(self, data_manager):
                calls.append(("setup_rawdb", data_manager.dir_workspace))

            def run(self):
                calls.append(("run",))

            def write_back(self, output_def):
                calls.append(("write_back", output_def))

            def write_verilog(self, output_verilog):
                calls.append(("write_verilog", output_verilog))

        params = self._resolved_params(placement_sizing_mode="size_only")
        launch = {
            "workspace": "/tmp/workspace",
            "output_def": "/tmp/out.def",
            "output_verilog": "/tmp/out.v",
        }

        result = run_optimization_flow(params, launch, engine_cls=FakeEngine)

        self.assertIs(created_engines[0], result.engine)
        self.assertEqual(FlowKind.SIZING, result.flow_kind)
        self.assertEqual("sizing", params.flow_kind)
        self.assertEqual(100, params.global_place_stages[0]["iteration"])
        self.assertEqual("discrete_gradient_topk", params.continuous_size_dynamics_mode)
        self.assertEqual("surrogate_only", params.timing_surrogate_mode)
        self.assertEqual(
            [
                ("init", params),
                ("setup_rawdb", "/tmp/workspace"),
                ("run",),
                ("write_back", "/tmp/out.def"),
                ("write_verilog", "/tmp/out.v"),
            ],
            calls,
        )

    def test_run_optimization_flow_skips_writeback_without_output_def(self):
        from dreamplace.flows.optimization_flow import run_optimization_flow

        calls = []

        class FakeEngine:
            def __init__(self, params):
                pass

            def setup_rawdb(self, data_manager):
                calls.append(("setup_rawdb", data_manager.dir_workspace))

            def run(self):
                calls.append(("run",))

            def write_back(self, output_def):
                calls.append(("write_back", output_def))

        result = run_optimization_flow(
            self._resolved_params(),
            {"workspace": "/tmp/workspace", "output_def": None},
            engine_cls=FakeEngine,
        )

        self.assertEqual([("setup_rawdb", "/tmp/workspace"), ("run",)], calls)
        self.assertIsInstance(result.flow_kind.value, str)

    def test_run_optimization_flow_skips_writeback_for_sta_flow(self):
        from dreamplace.flows.flow_config import FlowKind
        from dreamplace.flows.optimization_flow import run_optimization_flow

        calls = []

        class FakeEngine:
            def __init__(self, params):
                pass

            def setup_rawdb(self, data_manager):
                calls.append(("setup_rawdb", data_manager.dir_workspace))

            def run(self):
                calls.append(("run",))

            def write_back(self, output_def):
                calls.append(("write_back", output_def))

        result = run_optimization_flow(
            self._resolved_params(flow_kind="sta"),
            {"workspace": "/tmp/workspace", "output_def": "/tmp/out.def"},
            engine_cls=FakeEngine,
        )

        self.assertEqual(FlowKind.STA, result.flow_kind)
        self.assertEqual([("setup_rawdb", "/tmp/workspace"), ("run",)], calls)

    def test_run_optimization_flow_writes_initialization_profile_when_enabled(self):
        from dreamplace.flows.optimization_flow import run_optimization_flow

        result_dir = Path(tempfile.mkdtemp())
        calls = []

        class FakePlacer:
            initialization_profile = {
                "data_collection_ms": 1.25,
                "operator_build_ms": 2.5,
            }

        class FakeEngine:
            def __init__(self, params):
                calls.append(("init",))
                self.placer = None

            def setup_rawdb(self, data_manager):
                calls.append(("setup_rawdb", data_manager.dir_workspace))

            def run(self):
                calls.append(("run",))
                self.placer = FakePlacer()

            def write_back(self, output_def):
                calls.append(("write_back", output_def))

        params = self._resolved_params(
            flow_kind="sta",
            result_dir=str(result_dir),
            base_design_name="unit",
            initialization_profile=True,
        )

        run_optimization_flow(
            params,
            {"workspace": "/tmp/workspace", "output_def": None},
            engine_cls=FakeEngine,
        )

        profile_path = result_dir / "unit_initialization_profile_latest.json"
        self.assertTrue(profile_path.exists())
        text = profile_path.read_text()
        self.assertIn('"artifact": "initialization_profile_latest"', text)
        self.assertIn('"setup_rawdb_ms"', text)
        self.assertIn('"run_ms"', text)
        self.assertIn('"data_collection_ms": 1.25', text)
        self.assertIn('"operator_build_ms": 2.5', text)
        self.assertEqual(
            [("init",), ("setup_rawdb", "/tmp/workspace"), ("run",)],
            calls,
        )

    def test_run_optimization_flow_includes_rawdb_subprofile(self):
        from dreamplace.flows.optimization_flow import run_optimization_flow

        result_dir = Path(tempfile.mkdtemp())

        class FakePlaceDB:
            rawdb_initialization_profile = {
                "setup_rawdb_openroad_read_ms": 11.0,
                "setup_rawdb_pydb_export_ms": 22.0,
            }

        class FakeEngine:
            def __init__(self, params):
                self.placedb = FakePlaceDB()
                self.placer = None

            def setup_rawdb(self, data_manager):
                pass

            def run(self):
                pass

            def write_back(self, output_def):
                pass

        params = self._resolved_params(
            flow_kind="sta",
            result_dir=str(result_dir),
            base_design_name="unit",
            initialization_profile=True,
        )

        run_optimization_flow(
            params,
            {"workspace": "/tmp/workspace", "output_def": None},
            engine_cls=FakeEngine,
        )

        profile_path = result_dir / "unit_initialization_profile_latest.json"
        profile = profile_path.read_text()
        self.assertIn('"setup_rawdb_openroad_read_ms": 11.0', profile)
        self.assertIn('"setup_rawdb_pydb_export_ms": 22.0', profile)

    def test_run_optimization_flow_includes_pydb_export_subprofile(self):
        from dreamplace.flows.optimization_flow import run_optimization_flow

        result_dir = Path(tempfile.mkdtemp())

        class FakePlaceDB:
            rawdb_initialization_profile = {
                "setup_rawdb_pydb_export_ms": 22.0,
                "setup_rawdb_pydb_export_basic_timing_ms": 7.0,
                "setup_rawdb_pydb_export_graph_csr_ms": 3.0,
            }

        class FakeEngine:
            def __init__(self, params):
                self.placedb = FakePlaceDB()
                self.placer = None

            def setup_rawdb(self, data_manager):
                pass

            def run(self):
                pass

            def write_back(self, output_def):
                pass

        params = self._resolved_params(
            flow_kind="sta",
            result_dir=str(result_dir),
            base_design_name="unit",
            initialization_profile=True,
        )

        run_optimization_flow(
            params,
            {"workspace": "/tmp/workspace", "output_def": None},
            engine_cls=FakeEngine,
        )

        profile = (result_dir / "unit_initialization_profile_latest.json").read_text()
        self.assertIn('"setup_rawdb_pydb_export_basic_timing_ms": 7.0', profile)
        self.assertIn('"setup_rawdb_pydb_export_graph_csr_ms": 3.0', profile)

    def test_run_optimization_flow_rejects_unresolved_params(self):
        from dreamplace.flows.optimization_flow import run_optimization_flow

        with self.assertRaisesRegex(ValueError, "requires Params resolved"):
            run_optimization_flow(
                types.SimpleNamespace(flow_kind="sta"),
                {"workspace": "/tmp/workspace", "output_def": None},
                engine_cls=object,
            )

    def test_run_optimization_flow_writes_cli_manifest_without_reinterpreting_params(self):
        from dreamplace.flows.flow_config import resolved_params_manifest
        from dreamplace.flows.optimization_flow import run_optimization_flow

        workspace = Path(tempfile.mkdtemp())

        class FakeEngine:
            def __init__(self, params):
                self.placedb = None
                self.placer = None

            def setup_rawdb(self, data_manager):
                pass

            def run(self):
                pass

            def write_back(self, output_def):
                pass

        params = self._resolved_params(flow_kind="sta")
        run_optimization_flow(
            params,
            {
                "workspace": str(workspace),
                "output_def": None,
                "effective_params_manifest": resolved_params_manifest(params),
            },
            engine_cls=FakeEngine,
        )

        manifest = (workspace / "effective_params.json").read_text()
        self.assertIn('"content_sha256"', manifest)
        self.assertIn('"flow_kind": "sta"', manifest)


if __name__ == "__main__":
    unittest.main()
