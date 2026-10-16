import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cx55_ablation_iter100_matrix as matrix
sys.path.pop()


class Cx55AblationIter100MatrixHelperTest(unittest.TestCase):
    def test_stdout_stderr_tee_can_disable_terminal_mirroring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "run.log")
            saved_stdout_fd = os.dup(1)
            outer_read_fd, outer_write_fd = os.pipe()
            try:
                os.dup2(outer_write_fd, 1)
                os.close(outer_write_fd)
                outer_write_fd = None

                with matrix.StdoutStderrTee(log_path, mirror_to_stdout=False):
                    os.write(1, b"log only\n")

                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
                saved_stdout_fd = None
                mirrored = os.read(outer_read_fd, 1024)
            finally:
                if saved_stdout_fd is not None:
                    os.dup2(saved_stdout_fd, 1)
                    os.close(saved_stdout_fd)
                if outer_write_fd is not None:
                    os.close(outer_write_fd)
                os.close(outer_read_fd)

            self.assertEqual(mirrored, b"")
            with open(log_path, "rb") as stream:
                self.assertEqual(stream.read(), b"log only\n")

    def test_prepare_runtime_modules_loads_source_placer_module(self):
        loaded_modules = []
        touched_modules = (
            "dreamplace",
            "dreamplace.configure",
            "dreamplace.Params",
            "dreamplace.ops.openroad_handoff",
            "dreamplace.ops.openroad_handoff.controller",
            "dreamplace.ops.openroad_handoff.session",
            "dreamplace.BasicPlace",
            "dreamplace.macroPlaceDB",
            "dreamplace.NonLinearPlace",
            "dreamplace.Placer",
            "dreamplace.ops",
            "dreamplace.ops.placeio_openroad",
            "dreamplace.ops.placeio_openroad.place_io",
        )
        previous_modules = {
            module_name: sys.modules.get(module_name) for module_name in touched_modules
        }
        previous_path = list(sys.path)

        def install_module(module_name, module):
            sys.modules[module_name] = module
            parent_name, _, child_name = module_name.rpartition(".")
            if parent_name and parent_name in sys.modules:
                setattr(sys.modules[parent_name], child_name, module)
            return module

        def fake_load_source_module(module_name, relative_path):
            loaded_modules.append((module_name, relative_path))
            module = types.ModuleType(module_name)
            if module_name == "dreamplace.Params":
                module.Params = type("Params", (), {})
            return install_module(module_name, module)

        try:
            dreamplace = install_module("dreamplace", types.ModuleType("dreamplace"))
            dreamplace.__path__ = []
            configure = install_module(
                "dreamplace.configure", types.ModuleType("dreamplace.configure")
            )
            configure.compile_configurations = {"CUDA_FOUND": "TRUE"}
            ops = install_module("dreamplace.ops", types.ModuleType("dreamplace.ops"))
            ops.__path__ = []
            placeio_pkg = install_module(
                "dreamplace.ops.placeio_openroad",
                types.ModuleType("dreamplace.ops.placeio_openroad"),
            )
            placeio_pkg.__path__ = []
            placeio_module = install_module(
                "dreamplace.ops.placeio_openroad.place_io",
                types.ModuleType("dreamplace.ops.placeio_openroad.place_io"),
            )

            class FakePlaceIOFunction:
                @staticmethod
                def _build_tcl_command():
                    return ""

            placeio_module.PlaceIOFunction = FakePlaceIOFunction

            with patch.object(
                matrix.glob,
                "glob",
                return_value=["/tmp/placeio_openroad_cpp.so"],
            ), patch.object(matrix, "require_paths"), patch.object(
                matrix,
                "load_source_module",
                side_effect=fake_load_source_module,
            ):
                _Params, Placer = matrix.prepare_runtime_modules("/tmp/mpl-test")

            self.assertIn(
                ("dreamplace.Placer", "dreamplace/Placer.py"),
                loaded_modules,
            )
            self.assertIs(Placer, sys.modules["dreamplace.Placer"])
        finally:
            sys.path[:] = previous_path
            for module_name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = module

    def test_exact_once_trigger_fires_only_at_requested_absolute_iteration(self):
        class FakeController:
            def should_handoff(self, event):
                return False, None

        placedb = types.SimpleNamespace(
            create_openroad_handoff_controller=lambda params: FakeController()
        )

        matrix.install_exact_once_trigger(placedb, target_iter=100)
        controller = placedb.create_openroad_handoff_controller({})

        self.assertEqual(controller.should_handoff({"absolute_iteration": 99}), (False, None))
        self.assertEqual(
            controller.should_handoff({"absolute_iteration": 100}),
            (
                True,
                {
                    "trigger_mode": "exact_iteration",
                    "trigger_reason": "exact_iteration@iter=100",
                },
            ),
        )
        self.assertEqual(controller.should_handoff({"absolute_iteration": 100}), (False, None))

    def test_periodic_trigger_fires_once_per_matching_absolute_iteration(self):
        class FakeController:
            def should_handoff(self, event):
                return False, None

        placedb = types.SimpleNamespace(
            create_openroad_handoff_controller=lambda params: FakeController()
        )

        matrix.install_periodic_trigger(placedb, trigger_period=50)
        controller = placedb.create_openroad_handoff_controller({})

        self.assertEqual(controller.should_handoff({"absolute_iteration": 49}), (False, None))
        self.assertEqual(
            controller.should_handoff({"absolute_iteration": 50}),
            (
                True,
                {
                    "trigger_mode": "periodic_iteration",
                    "trigger_reason": "periodic_iteration@period=50@iter=50",
                },
            ),
        )
        self.assertEqual(controller.should_handoff({"absolute_iteration": 50}), (False, None))
        self.assertEqual(
            controller.should_handoff({"absolute_iteration": 100}),
            (
                True,
                {
                    "trigger_mode": "periodic_iteration",
                    "trigger_reason": "periodic_iteration@period=50@iter=100",
                },
            ),
        )

    def test_session_summary_counts_status_mutation_and_buffer_churn(self):
        events = [
            {
                "status": "success",
                "absolute_iteration": 100,
                "metric_snapshot": {
                    "hpwl": 10.0,
                    "overflow": 0.1,
                    "objective": 5.0,
                    "wns": -0.05,
                    "tns": -1.0,
                    "ws": -0.04,
                    "ts": 0.0,
                    "max_slew_violation": 0.01,
                    "max_load_cap_violation": 0.02,
                },
                "mutation_kind": "topology_changed",
                "added_buffer_count": 2,
                "removed_buffer_count": 1,
                "surviving_buffer_count": 5,
                "buffer_only_policy": {
                    "status": "violated",
                    "allowed_change_counts": {"added_buffer_count": 2},
                    "violation_counts": {
                        "non_buffer_master_change_count": 1,
                        "non_buffer_size_change_count": 1,
                        "non_buffer_orient_change_count": 0,
                        "added_non_buffer_count": 0,
                        "removed_non_buffer_count": 0,
                        "connectivity_violation_count": 0,
                        "total_violation_count": 2,
                    },
                    "summary": {
                        "non_buffer_changed_instance_count": 1,
                        "buffer_changed_instance_count": 0,
                        "unknown_reason_count": 0,
                    },
                    "unknown_reasons": [],
                    "violation_samples": [{"node_name": "u0"}],
                    "sample_limit": 50,
                    "samples_truncated": False,
                },
            },
            {
                "status": "skipped",
                "absolute_iteration": 101,
                "metric_snapshot": {
                    "hpwl": 9.0,
                    "overflow": 0.05,
                    "objective": 4.0,
                    "wns": -0.03,
                    "tns": -0.5,
                    "ws": -0.02,
                    "ts": 0.0,
                    "max_slew_violation": 0.0,
                    "max_load_cap_violation": 0.01,
                },
                "mutation_kind": "no_mutation",
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 5,
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {"added_buffer_count": 0},
                    "violation_counts": {
                        "non_buffer_master_change_count": 0,
                        "non_buffer_size_change_count": 0,
                        "non_buffer_orient_change_count": 0,
                        "added_non_buffer_count": 0,
                        "removed_non_buffer_count": 0,
                        "connectivity_violation_count": 0,
                        "total_violation_count": 0,
                    },
                    "summary": {
                        "non_buffer_changed_instance_count": 0,
                        "buffer_changed_instance_count": 0,
                        "unknown_reason_count": 0,
                    },
                    "unknown_reasons": [],
                    "violation_samples": [],
                    "sample_limit": 50,
                    "samples_truncated": False,
                },
            },
        ]

        summary = matrix.build_session_summary(events)

        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["event_iterations"], [100, 101])
        self.assertEqual(summary["status_counts"], {"success": 1, "skipped": 1})
        self.assertEqual(summary["mutation_kind_counts"]["topology_changed"], 1)
        self.assertEqual(summary["mutation_kind_counts"]["no_mutation"], 1)
        self.assertEqual(
            summary["buffer_churn_totals"],
            {
                "added_buffer_count": 2,
                "removed_buffer_count": 1,
                "surviving_buffer_count": 10,
            },
        )
        self.assertEqual(
            summary["buffer_only_policy_status_counts"],
            {"clean": 1, "violated": 1, "unknown": 0},
        )
        self.assertEqual(
            summary["buffer_only_policy_violation_totals"]["total_violation_count"],
            2,
        )
        self.assertEqual(
            summary["buffer_only_policy_violation_totals"][
                "non_buffer_master_change_count"
            ],
            1,
        )
        self.assertEqual(
            summary["buffer_only_policy_unknown_reason_counts"],
            {},
        )
        self.assertEqual(
            summary["latest_metric_snapshot"],
            {
                "hpwl": 9.0,
                "overflow": 0.05,
                "objective": 4.0,
                "wns": -0.03,
                "tns": -0.5,
                "ws": -0.02,
                "ts": 0.0,
                "max_slew_violation": 0.0,
                "max_load_cap_violation": 0.01,
            },
        )

    def test_session_summary_counts_unavailable_policy_as_unknown(self):
        events = [
            {"status": "success", "buffer_only_policy": None},
            {"status": "success", "buffer_only_policy": "malformed"},
            {
                "status": "success",
                "buffer_only_policy": {
                    "status": "deferred",
                    "violation_counts": "malformed",
                    "unknown_reasons": [
                        {"reason": "missing_baseline"},
                        {"reason": "missing_baseline"},
                        {"reason": "bad_diff"},
                        {"reason": []},
                        {"reason": 3},
                        {"reason": ""},
                        {},
                        "malformed",
                    ],
                },
            },
            {
                "status": "success",
                "buffer_only_policy": {
                    "status": ["violated"],
                    "unknown_reasons": 7,
                },
            },
            {
                "status": "success",
                "buffer_only_policy": {
                    "status": {"state": "clean"},
                },
            },
            {
                "status": "success",
                "buffer_only_policy": {
                    "unknown_reasons": [],
                },
            },
        ]

        summary = matrix.build_session_summary(events)

        self.assertEqual(
            summary["buffer_only_policy_status_counts"],
            {"clean": 0, "violated": 0, "unknown": 6},
        )
        self.assertEqual(
            summary["buffer_only_policy_unknown_reason_counts"],
            {"bad_diff": 1, "missing_baseline": 2},
        )
        self.assertEqual(
            summary["buffer_only_policy_violation_totals"],
            matrix.empty_policy_violation_totals(),
        )

    def test_session_summary_skips_non_dict_events(self):
        valid_event = {
            "status": "success",
            "absolute_iteration": 100,
            "metric_snapshot": {"hpwl": 1.0, "wns": -0.1},
            "mutation_kind": "placement_only",
            "added_buffer_count": 1,
            "removed_buffer_count": 0,
            "surviving_buffer_count": 2,
            "buffer_only_policy": {
                "status": "clean",
                "violation_counts": {"total_violation_count": 0},
                "unknown_reasons": [],
            },
        }
        events = ["malformed", valid_event]

        summary = matrix.build_session_summary(events)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["event_count"], 1)
        self.assertEqual(summary["event_iterations"], [100])
        self.assertEqual(summary["status_counts"], {"success": 1})
        self.assertEqual(summary["mutation_kind_counts"]["placement_only"], 1)
        self.assertEqual(
            summary["buffer_churn_totals"],
            {
                "added_buffer_count": 1,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 2,
            },
        )
        self.assertEqual(
            summary["buffer_only_policy_status_counts"],
            {"clean": 1, "violated": 0, "unknown": 0},
        )
        self.assertEqual(summary["latest_metric_snapshot"], {"hpwl": 1.0, "wns": -0.1})
        self.assertEqual(summary["events_tail"], [valid_event])

    def test_session_summary_treats_non_iterable_history_as_empty(self):
        for event_history in (None, 7):
            summary = matrix.build_session_summary(event_history)

            self.assertEqual(summary["status"], "not_started")
            self.assertEqual(summary["event_count"], 0)
            self.assertEqual(summary["event_iterations"], [])
            self.assertEqual(summary["status_counts"], {})
            self.assertEqual(summary["events_tail"], [])

    def test_session_summary_skips_malformed_status_and_iteration_fields(self):
        events = [
            {
                "status": [],
                "absolute_iteration": [],
                "iteration": "bad",
                "mutation_kind": "placement_only",
                "added_buffer_count": 3,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 4,
                "buffer_only_policy": None,
            },
            {
                "status": "success",
                "absolute_iteration": "101",
                "mutation_kind": "no_mutation",
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 4,
                "buffer_only_policy": {
                    "status": "clean",
                    "violation_counts": {"total_violation_count": 0},
                    "unknown_reasons": [],
                },
            },
        ]

        summary = matrix.build_session_summary(events)

        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["event_iterations"], [101])
        self.assertEqual(summary["status_counts"], {"success": 1})
        self.assertEqual(summary["mutation_kind_counts"]["placement_only"], 1)
        self.assertEqual(summary["mutation_kind_counts"]["no_mutation"], 1)

    def test_finalized_session_status_uses_valid_event_count(self):
        self.assertEqual(
            matrix.finalized_session_status(
                {"event_count": 1, "status_counts": {"success": 1}}
            ),
            "completed",
        )
        self.assertEqual(
            matrix.finalized_session_status(
                {"event_count": 1, "status_counts": {"success": 0}}
            ),
            "partial",
        )
        self.assertEqual(
            matrix.finalized_session_status(
                {"event_count": 0, "status_counts": {"success": 0}}
            ),
            "not_started",
        )
        for malformed_summary in (
            {"event_count": None, "status_counts": {"success": 1}},
            {"event_count": "1", "status_counts": {"success": 1}},
            {"event_count": [], "status_counts": {"success": 1}},
            {"event_count": True, "status_counts": {"success": 1}},
            {"event_count": -1, "status_counts": {"success": 1}},
        ):
            self.assertEqual(
                matrix.finalized_session_status(malformed_summary),
                "not_started",
            )
        for malformed_status_counts in (None, "malformed"):
            self.assertEqual(
                matrix.finalized_session_status(
                    {"event_count": 1, "status_counts": malformed_status_counts}
                ),
                "partial",
            )
        for malformed_success_count in (True, 1.0, "1"):
            self.assertEqual(
                matrix.finalized_session_status(
                    {
                        "event_count": 1,
                        "status_counts": {"success": malformed_success_count},
                    }
                ),
                "partial",
            )

    def test_diagnostic_evidence_extracts_move_sequence_and_conservative_labels(self):
        log_text = "\n".join(
            [
                "[INFO RSZ-0100] Repair move sequence: UnbufferMove BufferMove SplitLoadMove ",
                "[INFO RSZ-0038] Inserted 0 buffers in 0 nets.",
                "[INFO RSZ-0039] Resized 2 instances.",
            ]
        )
        events = [
            {
                "buffer_insertion_strategy": "repair_design",
                "mutation_kind": "geometry_changed",
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 5,
                "buffer_only_policy": {
                    "status": "violated",
                    "violation_counts": {
                        "non_buffer_master_change_count": 0,
                        "non_buffer_size_change_count": 2,
                        "total_violation_count": 2,
                    },
                    "allowed_change_counts": {"resized_buffer_count": 0},
                    "summary": {"unknown_reason_count": 0},
                    "unknown_reasons": [],
                },
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "mutation_kind": "placement_only",
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 5,
                "buffer_only_policy": {
                    "status": "clean",
                    "violation_counts": {"total_violation_count": 0},
                    "allowed_change_counts": {
                        "added_buffer_count": 0,
                        "removed_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                    "summary": {"unknown_reason_count": 0},
                    "unknown_reasons": [],
                },
            },
        ]

        evidence = matrix.extract_openroad_diagnostic_evidence(log_text, events)

        self.assertEqual(
            evidence["repair_move_sequences"],
            ["UnbufferMove BufferMove SplitLoadMove"],
        )
        self.assertEqual(evidence["inserted_buffer_log_count"], 0)
        self.assertEqual(evidence["resized_instance_log_count"], 2)
        self.assertIn(
            "consistent_with_repair_design_driver_resize",
            evidence["diagnostic_labels"],
        )
        self.assertIn("clean_no_buffer_churn", evidence["diagnostic_labels"])
        self.assertIn(
            "buffer_moves_available_but_not_accepted",
            evidence["diagnostic_labels"],
        )
        self.assertIn(
            "insufficient_timing_evidence",
            evidence["diagnostic_labels"],
        )

    def test_diagnostic_evidence_ignores_malformed_events_and_policies(self):
        events = [
            "malformed",
            {
                "buffer_insertion_strategy": "repair_design",
                "buffer_only_policy": "malformed",
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": "malformed",
                    "violation_counts": "malformed",
                },
            },
            {
                "buffer_insertion_strategy": "repair_design",
                "buffer_only_policy": {
                    "status": "violated",
                    "allowed_change_counts": {},
                    "violation_counts": "malformed",
                },
            },
        ]

        evidence = matrix.extract_openroad_diagnostic_evidence("", events)

        self.assertEqual(
            evidence["diagnostic_labels"],
            ["insufficient_timing_evidence"],
        )

    def test_diagnostic_evidence_treats_non_iterable_history_as_empty(self):
        for event_history in (None, 7):
            evidence = matrix.extract_openroad_diagnostic_evidence("", event_history)

            self.assertEqual(evidence["repair_move_sequences"], [])
            self.assertEqual(evidence["inserted_buffer_log_count"], 0)
            self.assertEqual(
                evidence["diagnostic_labels"],
                ["insufficient_timing_evidence"],
            )

    def test_diagnostic_evidence_requires_resize_specific_repair_evidence(self):
        log_text = "\n".join(
            [
                "[INFO RSZ-0039] Resized 2 instances.",
                "WNS 0.0",
            ]
        )
        events = [
            {
                "buffer_insertion_strategy": "repair_design",
                "buffer_only_policy": {
                    "status": "violated",
                    "violation_counts": {
                        "connectivity_violation_count": 1,
                        "added_non_buffer_count": 1,
                        "total_violation_count": 2,
                    },
                },
            }
        ]

        evidence = matrix.extract_openroad_diagnostic_evidence(log_text, events)

        self.assertNotIn(
            "consistent_with_repair_design_driver_resize",
            evidence["diagnostic_labels"],
        )

    def test_diagnostic_evidence_requires_explicit_zero_allowed_buffer_churn(self):
        log_text = "[INFO RSZ-0100] Repair move sequence: BufferMove"
        events = [
            {
                "buffer_insertion_strategy": "buffer-only",
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {
                        "added_buffer_count": 1,
                        "removed_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                },
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {
                        "added_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                },
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "added_buffer_count": 1,
                "removed_buffer_count": 0,
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {
                        "added_buffer_count": 0,
                        "removed_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                },
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "added_buffer_count": 0,
                "removed_buffer_count": 1,
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {
                        "added_buffer_count": 0,
                        "removed_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                },
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "added_buffer_count": True,
                "removed_buffer_count": 0,
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {
                        "added_buffer_count": 0,
                        "removed_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                },
            },
            {
                "buffer_insertion_strategy": "buffer-only",
                "added_buffer_count": "1",
                "removed_buffer_count": "0",
                "buffer_only_policy": {
                    "status": "clean",
                    "allowed_change_counts": {
                        "added_buffer_count": 0,
                        "removed_buffer_count": 0,
                        "resized_buffer_count": 0,
                    },
                },
            },
        ]

        evidence = matrix.extract_openroad_diagnostic_evidence(log_text, events)

        self.assertNotIn("clean_no_buffer_churn", evidence["diagnostic_labels"])
        self.assertNotIn(
            "buffer_moves_available_but_not_accepted",
            evidence["diagnostic_labels"],
        )

    def test_diagnostic_evidence_requires_run_level_zero_event_churn(self):
        log_text = "[INFO RSZ-0100] Repair move sequence: BufferMove"
        clean_zero_event = {
            "buffer_insertion_strategy": "buffer-only",
            "added_buffer_count": 0,
            "removed_buffer_count": 0,
            "buffer_only_policy": {
                "status": "clean",
                "allowed_change_counts": {
                    "added_buffer_count": 0,
                    "removed_buffer_count": 0,
                    "resized_buffer_count": 0,
                },
            },
        }
        for churn_event in (
            {"added_buffer_count": 1, "removed_buffer_count": 0},
            {"added_buffer_count": True, "removed_buffer_count": 0},
            {"added_buffer_count": 0, "removed_buffer_count": "0"},
            {"added_buffer_count": 0},
        ):
            evidence = matrix.extract_openroad_diagnostic_evidence(
                log_text,
                [
                    clean_zero_event,
                    {
                        "buffer_insertion_strategy": "buffer-only",
                        "buffer_only_policy": {
                            "status": "clean",
                            "allowed_change_counts": {
                                "added_buffer_count": 0,
                                "removed_buffer_count": 0,
                                "resized_buffer_count": 0,
                            },
                        },
                        **churn_event,
                    },
                ],
            )

            self.assertNotIn("clean_no_buffer_churn", evidence["diagnostic_labels"])
            self.assertNotIn(
                "buffer_moves_available_but_not_accepted",
                evidence["diagnostic_labels"],
            )


if __name__ == "__main__":
    unittest.main()
