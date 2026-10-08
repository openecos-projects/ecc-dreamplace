import json
import os
import sys
import tempfile
import types
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch


def _install_placer_import_stubs():
    modules = {
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
        "torch": types.ModuleType("torch"),
        "torch.autograd": types.ModuleType("torch.autograd"),
        "numpy": types.ModuleType("numpy"),
        "dreamplace.configure": types.ModuleType("dreamplace.configure"),
        "dreamplace.macroPlaceDB": types.ModuleType("dreamplace.macroPlaceDB"),
        "dreamplace.NonLinearPlace": types.ModuleType("dreamplace.NonLinearPlace"),
        "dreamplace.ops.placeio_openroad": types.ModuleType(
            "dreamplace.ops.placeio_openroad"
        ),
        "dreamplace.ops.placeio_openroad.place_io": types.ModuleType(
            "dreamplace.ops.placeio_openroad.place_io"
        ),
        "tools": types.ModuleType("tools"),
        "tools.iEDA": types.ModuleType("tools.iEDA"),
        "tools.iEDA.module": types.ModuleType("tools.iEDA.module"),
        "tools.iEDA.module.sta": types.ModuleType("tools.iEDA.module.sta"),
        "tools.iEDA.module.io": types.ModuleType("tools.iEDA.module.io"),
    }
    modules["dreamplace.configure"].compile_configurations = {"CUDA_FOUND": "FALSE"}
    modules["dreamplace.macroPlaceDB"].MacroPlaceDB = object
    modules["dreamplace.NonLinearPlace"].NonLinearPlace = object
    modules["dreamplace.ops.placeio_openroad"].__path__ = []
    modules["dreamplace.ops.placeio_openroad.place_io"].PlaceIOFunction = object
    modules["tools.iEDA.module.sta"].IEDASta = object
    modules["tools.iEDA.module.io"].IEDAIO = object
    torch = modules["torch"]
    torch.__path__ = []
    torch.autograd = modules["torch.autograd"]
    torch.autograd.__path__ = []
    torch.autograd.Function = object
    torch.manual_seed = lambda seed: None
    torch.cuda = types.SimpleNamespace(
        manual_seed=lambda seed: None,
        manual_seed_all=lambda seed: None,
    )
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False, deterministic=True)
    )
    torch.set_num_threads = lambda count: None
    numpy = modules["numpy"]
    numpy.random = types.SimpleNamespace(seed=lambda seed: None)
    previous = {}
    for name, module in modules.items():
        previous[name] = sys.modules.get(name)
        sys.modules[name] = module
    return previous


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_PREVIOUS_MODULES = _install_placer_import_stubs()
from dreamplace import Placer
_restore_modules(_PREVIOUS_MODULES)
sys.path.pop()


class PlacerCliTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.params_json = self.root / "params.json"
        self.params_json.write_text(
            json.dumps(
                {
                    "random_seed": 1000,
                    "num_threads": 1,
                    "gpu": 0,
                    "evaluate_pl": 0,
                    "global_place_stages": [{"iteration": 10}],
                    "target_density": 0.3,
                    "stop_overflow": 0.1,
                    "result_dir": str(self.root / "old_result"),
                    "base_design_name": "old_design",
                    "detailed_place_engine": "",
                    "get_congestion_map": 0,
                }
            )
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_build_effective_params_applies_openroad_handoff_overrides(self):
        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--place-io-engine",
                "openroad",
                "--workspace",
                str(self.root / "run"),
                "--result-dir",
                str(self.root / "run"),
                "--base-design-name",
                "new_design",
                "--output-def",
                str(self.root / "run" / "final.def"),
                "--def-input",
                "/design/input.def",
                "--verilog-input",
                "/design/input.v",
                "--tech-lef",
                "/tech/tech.lef",
                "--lef",
                "/tech/cell.lef",
                "--lib",
                "/tech/timing.lib",
                "--sdc",
                "/design/input.sdc",
                "--rc-tcl",
                "/tech/setRC.tcl",
                "--handoff-mode",
                "buffer-only",
                "--handoff-trigger-period",
                "50",
                "--max-handoff-overflow",
                "0.2",
                "--setup-margin",
                "0.125",
                "--target-density",
                "0.5",
                "--stop-overflow",
                "0.075",
                "--plot-interval",
                "25",
                "--handoff-restart-policy",
                "warm_schedule",
                "--with-sta",
            ]
        )

        params, launch = Placer.build_effective_params_from_args(args)

        self.assertEqual(params.place_io_engine, "openroad")
        self.assertEqual(params.result_dir, str(self.root / "run"))
        self.assertEqual(params.base_design_name, "new_design")
        self.assertEqual(params.target_density, 0.5)
        self.assertEqual(params.stop_overflow, 0.075)
        self.assertEqual(params.plot_interval, 25)
        self.assertEqual(params.handoff_restart_policy, "warm_schedule")
        self.assertEqual(params.with_sta, 1)
        self.assertEqual(params.def_input, "/design/input.def")
        self.assertEqual(params.verilog_input, "/design/input.v")
        self.assertEqual(params.lef_input, ["/tech/tech.lef", "/tech/cell.lef"])
        self.assertEqual(params.design_inputs["rc_tcl"], "/tech/setRC.tcl")
        self.assertEqual(
            params.openroad_handoff["trigger"]["guards"],
            [{"name": "overflow", "op": "<=", "value": 0.2}],
        )
        self.assertEqual(
            params.openroad_handoff["buffer_insertion"]["write_def_after"],
            str(self.root / "run" / "final.def"),
        )
        self.assertEqual(launch["workspace"], str(self.root / "run"))
        self.assertEqual(launch["output_def"], str(self.root / "run" / "final.def"))
        self.assertEqual(launch["setup_margin"], 0.125)

    def test_workspace_config_supplies_design_inputs_for_openroad(self):
        workspace = self.root / "run"
        config = workspace / "config"
        config.mkdir(parents=True)
        (config / "tech_path.json").write_text(
            json.dumps(
                {
                    "tech_lef_path": "/tech/ws_tech.lef",
                    "lef_paths": ["/tech/ws_cell_a.lef", "/tech/ws_cell_b.lef"],
                    "lib_paths": ["/tech/ws.lib"],
                    "rc_tcl": "/tech/ws.setRC.tcl",
                }
            )
        )
        (config / "design_path.json").write_text(
            json.dumps(
                {"def_input_path": "/design/ws.def", "sdc_path": "/design/ws.sdc"}
            )
        )

        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--place-io-engine",
                "openroad",
                "--workspace",
                str(workspace),
            ]
        )
        params, launch = Placer.build_effective_params_from_args(args)

        self.assertEqual(params.design_inputs["tech_lef"], ["/tech/ws_tech.lef"])
        self.assertEqual(
            params.design_inputs["lef"],
            ["/tech/ws_cell_a.lef", "/tech/ws_cell_b.lef"],
        )
        self.assertEqual(params.design_inputs["lib"], ["/tech/ws.lib"])
        self.assertEqual(params.design_inputs["rc_tcl"], "/tech/ws.setRC.tcl")
        self.assertEqual(params.design_inputs["def"], "/design/ws.def")
        self.assertEqual(params.design_inputs["sdc"], "/design/ws.sdc")
        self.assertEqual(launch["workspace"], str(workspace))

    def test_explicit_design_inputs_override_workspace_config(self):
        workspace = self.root / "run"
        config = workspace / "config"
        config.mkdir(parents=True)
        (config / "tech_path.json").write_text(
            json.dumps({"tech_lef_path": "/tech/ws_tech.lef"})
        )

        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--place-io-engine",
                "openroad",
                "--workspace",
                str(workspace),
                "--tech-lef",
                "/tech/cli_tech.lef",
            ]
        )
        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual(params.design_inputs["tech_lef"], ["/tech/cli_tech.lef"])

    def test_dry_run_prints_effective_json_without_constructing_engine(self):
        output = StringIO()
        with patch.object(Placer, "PlacementEngine") as engine_cls:
            rc = Placer.main(
                [
                    str(self.params_json),
                    "--place-io-engine",
                    "openroad",
                    "--workspace",
                    str(self.root / "run"),
                    "--output-def",
                    str(self.root / "final.def"),
                    "--handoff-mode",
                    "no-handoff",
                    "--dry-run-config",
                ],
                output_stream=output,
            )

        self.assertEqual(rc, 0)
        engine_cls.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["place_io_engine"], "openroad")
        self.assertEqual(payload["openroad_handoff"]["enabled"], False)

    def test_build_effective_params_applies_legacy_defaults(self):
        args = Placer.build_arg_parser().parse_args([str(self.params_json)])

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertFalse(params.auto_adjust_bins)
        # differentiable_timing_obj is declared in params.json (default 1), so the
        # legacy placement default only fills keys the schema leaves absent;
        # params.json wins for this one.
        self.assertEqual(1, params.differentiable_timing_obj)

    def test_buffering_help_keeps_canonical_knobs_and_hides_legacy_knobs(self):
        parser = Placer.build_arg_parser()
        help_text = parser.format_help()

        self.assertIn("--flow-kind", help_text)
        self.assertIn("--buffering-mode", help_text)
        self.assertIn("--buffering-continuous-steps", help_text)
        self.assertIn("--buffering-continuous-lr", help_text)
        self.assertIn("--buffering-max-selected-actions", help_text)
        self.assertIn("--buffering-max-repeaters-per-segment", help_text)
        self.assertNotIn("--du-buffering-continuous-steps", help_text)
        self.assertNotIn("--du-buffering-continuous-lr", help_text)
        self.assertNotIn("--du-buffering-max-selected-actions", help_text)
        self.assertNotIn("--du-buffering-segment-count-max-repeater-count", help_text)
        self.assertNotIn("--du-buffering-autograd-batch-mode", help_text)
        self.assertNotIn("--du-buffering-relaxed-smoke-net-cap", help_text)
        self.assertNotIn("--du-buffering-global-joint-gradient", help_text)

    def test_build_effective_params_accepts_sizing_flow_kind(self):
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sizing"]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual("sizing", params.flow_kind)
        self.assertEqual("size_only", params.placement_sizing_mode)
        self.assertEqual(100, params.global_place_stages[0]["iteration"])
        self.assertEqual("discrete_gradient_topk", params.continuous_size_dynamics_mode)
        self.assertEqual("surrogate_only", params.timing_surrogate_mode)
        self.assertEqual("nearest_size_round", params.projection_resolver)
        self.assertEqual("off", params.critical_endpoint_pruning_mode)
        self.assertTrue(params.production_fast_loop)
        self.assertEqual("auto", params.timing_lut_2d_native_op)
        self.assertEqual("auto", params.size_interpolated_pin_native_op)
        self.assertEqual(30.0, params.discrete_gradient_topk_up_percent)
        self.assertEqual(0.0, params.discrete_gradient_topk_down_percent)

    def test_build_effective_params_accepts_sta_flow_kind(self):
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sta"]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual("sta", params.flow_kind)
        self.assertEqual("size_only", params.placement_sizing_mode)
        self.assertEqual(1, params.with_sta)
        self.assertEqual("lut_only", params.timing_surrogate_mode)
        self.assertEqual("auto", params.timing_lut_2d_native_op)
        self.assertEqual("auto", params.size_interpolated_pin_native_op)
        self.assertEqual(0, params.global_place_stages[0]["iteration"])

    def test_shared_cell_budget_cli_uses_one_percent_sizing_default_and_explicit_overrides(self):
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sizing"]
        )
        params, _ = Placer.build_effective_params_from_args(args)
        self.assertEqual(1, params.discrete_gradient_topk_shared_budget)
        self.assertEqual(1.0, params.discrete_gradient_topk_shared_budget_percent)
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sizing",
             "--discrete-gradient-topk-shared-budget", "0"]
        )
        params, _ = Placer.build_effective_params_from_args(args)
        self.assertEqual(0, params.discrete_gradient_topk_shared_budget)
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sizing",
             "--discrete-gradient-topk-shared-budget", "1",
             "--discrete-gradient-topk-shared-budget-percent", "0"]
        )
        params, _ = Placer.build_effective_params_from_args(args)
        self.assertEqual(0.0, params.discrete_gradient_topk_shared_budget_percent)

        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sizing",
             "--discrete-gradient-topk-shared-budget", "1",
             "--discrete-gradient-topk-oscillation-veto", "1"]
        )
        params, _ = Placer.build_effective_params_from_args(args)
        self.assertEqual(1, params.discrete_gradient_topk_oscillation_veto)

    def test_build_effective_params_accepts_buffering_flow_kind(self):
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "buffering"]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual("buffering", params.flow_kind)
        self.assertEqual("segment", params.buffering_mode)
        self.assertEqual(1, params.buffering_continuous_relaxed_optimization)
        self.assertEqual(1, params.buffering_segment_count_tns_gradient)
        self.assertEqual("segment_only", params.buffering_candidate_policy)
        self.assertEqual(3, params.buffering_max_repeaters_per_segment)
        self.assertEqual(0, params.buffering_include_tree_node_candidates)
        self.assertEqual(
            "dynamic_net_provider",
            params.buffering_relaxed_timing_integration_mode,
        )

    def test_build_effective_params_rejects_physical_eco_flow_kind(self):
        # The canonical flow only accepts the flow kinds it implements: the
        # physical-eco kind still has profile tables but no canonical flow path,
        # so the CLI must surface the rejection instead of handing back a
        # parameter set the flow layer refuses to run.
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "physical_eco"]
        )

        with self.assertRaises(ValueError) as error:
            Placer.build_effective_params_from_args(args)

        self.assertIn("physical_eco", str(error.exception))

    def test_buffering_flow_kind_preserves_cli_tuning_overrides(self):
        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--flow-kind",
                "buffering",
                "--buffering-mode",
                "candidate",
                "--buffering-continuous-steps",
                "7",
                "--buffering-continuous-lr",
                "0.2",
                "--buffering-max-repeaters-per-segment",
                "6",
            ]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual("candidate", params.buffering_mode)
        self.assertEqual(7, params.buffering_continuous_steps)
        self.assertEqual(7, params.buffering_continuous_steps)
        self.assertEqual(0.2, params.buffering_continuous_lr)
        self.assertEqual(0.2, params.buffering_continuous_lr)
        self.assertEqual(6, params.buffering_max_repeaters_per_segment)

    def test_buffering_canonical_aliases_reach_canonical_params(self):
        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--flow-kind",
                "buffering",
                "--buffering-continuous-steps",
                "11",
                "--buffering-max-selected-actions",
                "9",
                "--buffering-max-repeaters-per-segment",
                "6",
            ]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual(11, params.buffering_continuous_steps)
        self.assertEqual(11, params.buffering_continuous_steps)
        self.assertEqual(9, params.buffering_max_selected_actions)
        self.assertEqual(9, params.buffering_max_selected_actions)
        self.assertEqual(6, params.buffering_max_repeaters_per_segment)

    def test_buffering_mode_rejects_invalid_value(self):
        with self.assertRaises(SystemExit):
            Placer.build_arg_parser().parse_args(
                [str(self.params_json), "--buffering-mode", "unknown"]
            )

    def test_legacy_buffering_cli_args_are_rejected(self):
        with self.assertRaises(SystemExit):
            Placer.build_arg_parser().parse_args(
                [str(self.params_json), "--du-buffering-continuous-steps", "7"]
            )

    def test_build_effective_params_accepts_size_interpolated_pin_native_op_override(self):
        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--flow-kind",
                "sizing",
                "--size-interpolated-pin-native-op",
                "off",
            ]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual("off", params.size_interpolated_pin_native_op)

    def test_build_effective_params_preserves_explicit_iteration_override(self):
        args = Placer.build_arg_parser().parse_args(
            [str(self.params_json), "--flow-kind", "sizing", "--iterations", "7"]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual(7, params.global_place_stages[0]["iteration"])
        self.assertNotIn("_global_place_stages_explicit", params.toJson())

    def test_build_effective_params_accepts_sizing_topk_overrides(self):
        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--flow-kind",
                "sizing",
                "--discrete-gradient-topk-up-percent",
                "30",
                "--discrete-gradient-topk-down-percent",
                "0.1",
                "--discrete-gradient-topk-vt-percent",
                "7.5",
            ]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual("sizing", params.flow_kind)
        self.assertEqual("size_only", params.placement_sizing_mode)
        self.assertEqual(30.0, params.discrete_gradient_topk_up_percent)
        self.assertEqual(0.1, params.discrete_gradient_topk_down_percent)
        self.assertEqual(7.5, params.discrete_gradient_topk_vt_percent)

    def test_main_runs_engine_with_simple_openroad_workspace(self):
        output_def = self.root / "final.def"
        fake_engine = MagicMock()
        fake_engine.run.return_value = {"hpwl": 1.0}
        with patch.object(Placer, "PlacementEngine", return_value=fake_engine):
            rc = Placer.main(
                [
                    str(self.params_json),
                    "--place-io-engine",
                    "openroad",
                    "--workspace",
                    str(self.root / "run"),
                    "--output-def",
                    str(output_def),
                    "--handoff-mode",
                    "no-handoff",
                ]
            )

        self.assertEqual(rc, 0)
        fake_engine.setup_rawdb.assert_called_once()
        data_manager = fake_engine.setup_rawdb.call_args.kwargs["data_manager"]
        self.assertEqual(data_manager.dir_workspace, str(self.root / "run"))
        fake_engine.run.assert_called_once()
        fake_engine.write_back.assert_called_once_with(str(output_def))

    def test_output_def_overrides_existing_config_only_handoff_write_path(self):
        self.params_json.write_text(
            json.dumps(
                {
                    "random_seed": 1000,
                    "num_threads": 1,
                    "gpu": 0,
                    "evaluate_pl": 0,
                    "result_dir": str(self.root / "run"),
                    "openroad_handoff": {
                        "enabled": True,
                        "trigger": {"mode": "disabled"},
                        "buffer_insertion": {
                            "enabled": True,
                            "strategy": "buffer-only",
                            "options": {"pre_repair_tcl": ["source old.tcl"]},
                            "write_def_after": "/old/final.def",
                        },
                    },
                }
            )
        )

        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--output-def",
                str(self.root / "new.def"),
            ]
        )

        params, launch = Placer.build_effective_params_from_args(args)

        self.assertEqual(launch["output_def"], str(self.root / "new.def"))
        self.assertEqual(
            params.openroad_handoff["buffer_insertion"]["write_def_after"],
            str(self.root / "new.def"),
        )

    def test_preserves_existing_openroad_handoff_when_no_handoff_overrides(self):
        handoff_config = {
            "enabled": True,
            "trigger": {
                "mode": "interval",
                "interval": 100,
                "dedupe_by": "absolute_iteration",
            },
            "buffer_insertion": {
                "enabled": True,
                "strategy": "buffer-only",
                "options": {"pre_repair_tcl": ["source custom.tcl"]},
                "write_def_after": "/old/final.def",
            },
        }
        self.params_json.write_text(
            json.dumps(
                {
                    "random_seed": 1000,
                    "num_threads": 1,
                    "gpu": 0,
                    "evaluate_pl": 0,
                    "result_dir": str(self.root / "run"),
                    "openroad_handoff": handoff_config,
                }
            )
        )
        args = Placer.build_arg_parser().parse_args([str(self.params_json)])

        params, launch = Placer.build_effective_params_from_args(args)

        self.assertEqual(params.openroad_handoff, handoff_config)
        self.assertEqual(launch["output_def"], "/old/final.def")

    def test_existing_custom_pre_repair_tcl_allows_buffer_only_without_rc_tcl(self):
        self.params_json.write_text(
            json.dumps(
                {
                    "random_seed": 1000,
                    "num_threads": 1,
                    "gpu": 0,
                    "evaluate_pl": 0,
                    "result_dir": str(self.root / "run"),
                    "openroad_handoff": {
                        "enabled": True,
                        "trigger": {"mode": "disabled"},
                        "buffer_insertion": {
                            "enabled": True,
                            "strategy": "buffer-only",
                            "options": {"pre_repair_tcl": ["source custom.tcl"]},
                            "write_def_after": "/old/final.def",
                        },
                    },
                }
            )
        )
        args = Placer.build_arg_parser().parse_args(
            [
                str(self.params_json),
                "--handoff-mode",
                "buffer-only",
                "--output-def",
                str(self.root / "new.def"),
            ]
        )

        params, _ = Placer.build_effective_params_from_args(args)

        self.assertEqual(
            params.openroad_handoff["buffer_insertion"]["options"]["pre_repair_tcl"],
            ["source custom.tcl"],
        )

    def test_max_handoff_overflow_rejects_out_of_range_numeric_values(self):
        parser = Placer.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args([str(self.params_json), "--max-handoff-overflow", "1.5"])


if __name__ == "__main__":
    unittest.main()
