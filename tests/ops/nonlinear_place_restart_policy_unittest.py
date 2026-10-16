import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.openroad_handoff import PlacementHandoffSession
sys.path.pop()


_tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _tests_dir)
from _import_stubs import auto_stub, register_stub_modules
sys.path.remove(_tests_dir)

# The stubs below isolate the heavy op layer. auto_stub fabricates whatever the
# product imports from dreamplace.ops, so a new op import in the product cannot
# rot this list again (it used to fail with "'dreamplace.ops' is not a package").
_OPS_STUBS = auto_stub("dreamplace.ops")


def _install_nonlinearplace_import_stubs():
    _OPS_STUBS.__enter__()
    modules = {
        "dreamplace.BasicPlace": types.ModuleType("dreamplace.BasicPlace"),
        "dreamplace.PlaceObj": types.ModuleType("dreamplace.PlaceObj"),
        "dreamplace.NesterovAcceleratedGradientOptimizer": types.ModuleType(
            "dreamplace.NesterovAcceleratedGradientOptimizer"
        ),
        "dreamplace.EvalMetrics": types.ModuleType("dreamplace.EvalMetrics"),
        "dreamplace.ops.fence_region": types.ModuleType("dreamplace.ops.fence_region"),
        "dreamplace.ops.fence_region.fence_region": types.ModuleType(
            "dreamplace.ops.fence_region.fence_region"
        ),
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
    }
    modules["dreamplace.BasicPlace"].BasicPlace = type("BasicPlace", (), {})
    modules["dreamplace.BasicPlace"].size_var_to_logits = mock.Mock(
        side_effect=AssertionError("size conversion is outside the restart fixture")
    )
    return register_stub_modules(modules)


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
    _OPS_STUBS.__exit__(None, None, None)


def _load_nonlinearplace_class():
    previous = _install_nonlinearplace_import_stubs()
    previous_nonlinearplace = sys.modules.get("dreamplace.NonLinearPlace")
    sys.modules.pop("dreamplace.NonLinearPlace", None)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    previous_path = list(sys.path)
    sys.path.append(repo_root)
    try:
        from dreamplace.NonLinearPlace import NonLinearPlace
    finally:
        if previous_nonlinearplace is None:
            sys.modules.pop("dreamplace.NonLinearPlace", None)
        else:
            sys.modules["dreamplace.NonLinearPlace"] = previous_nonlinearplace
        sys.path[:] = previous_path
        _restore_modules(previous)
    return NonLinearPlace


class NonLinearPlaceRestartPolicyTest(unittest.TestCase):
    def test_handoff_restart_policy_defaults_to_cold_when_missing(self):
        NonLinearPlace = _load_nonlinearplace_class()

        self.assertEqual(NonLinearPlace._handoff_restart_policy(types.SimpleNamespace()), "cold")

    def test_handoff_restart_policy_rejects_unknown_policy(self):
        NonLinearPlace = _load_nonlinearplace_class()

        with self.assertRaisesRegex(RuntimeError, "unknown handoff_restart_policy"):
            NonLinearPlace._handoff_restart_policy(
                types.SimpleNamespace(handoff_restart_policy="hot")
            )

    def test_capture_warm_schedule_state_clones_selected_fields(self):
        NonLinearPlace = _load_nonlinearplace_class()
        model = types.SimpleNamespace(
            density_weight=torch.tensor([1.0, 2.0]),
            density_weight_u=np.array([3.0, 4.0]),
            density_weight_step_size=5.0,
            momentum=torch.tensor([9.0]),
        )

        state = NonLinearPlace._capture_warm_schedule_state(model)

        self.assertEqual(set(state.keys()), {
            "density_weight",
            "density_weight_u",
            "density_weight_step_size",
        })
        self.assertTrue(torch.equal(state["density_weight"], torch.tensor([1.0, 2.0])))
        self.assertIsNot(state["density_weight"], model.density_weight)
        self.assertTrue(np.array_equal(state["density_weight_u"], np.array([3.0, 4.0])))
        self.assertIsNot(state["density_weight_u"], model.density_weight_u)
        self.assertEqual(state["density_weight_step_size"], 5.0)

        model.density_weight[0] = 99.0
        model.density_weight_u[0] = 88.0
        self.assertTrue(torch.equal(state["density_weight"], torch.tensor([1.0, 2.0])))
        self.assertTrue(np.array_equal(state["density_weight_u"], np.array([3.0, 4.0])))

    def test_apply_warm_schedule_state_skips_incompatible_shapes_and_reports(self):
        NonLinearPlace = _load_nonlinearplace_class()
        model = types.SimpleNamespace(
            density_weight=torch.tensor([0.0, 0.0]),
            density_weight_u=np.array([0.0, 0.0, 0.0]),
            density_weight_step_size=0.0,
        )
        state = {
            "density_weight": torch.tensor([1.0]),
            "density_weight_u": np.array([2.0, 3.0]),
            "density_weight_step_size": 4.0,
        }

        report = NonLinearPlace._apply_warm_schedule_state(model, state)

        self.assertEqual(report, {
            "applied": ["density_weight_step_size"],
            "skipped": ["density_weight", "density_weight_u"],
        })
        self.assertTrue(torch.equal(model.density_weight, torch.tensor([0.0, 0.0])))
        self.assertTrue(np.array_equal(model.density_weight_u, np.array([0.0, 0.0, 0.0])))
        self.assertEqual(model.density_weight_step_size, 4.0)

    def test_apply_warm_schedule_state_skips_incompatible_container_types(self):
        NonLinearPlace = _load_nonlinearplace_class()
        model = types.SimpleNamespace(
            density_weight=torch.tensor([0.0]),
            density_weight_step_size=0.0,
        )
        state = {
            "density_weight": np.array([1.0]),
            "density_weight_step_size": 4.0,
        }

        report = NonLinearPlace._apply_warm_schedule_state(model, state)

        self.assertEqual(report, {
            "applied": ["density_weight_step_size"],
            "skipped": ["density_weight"],
        })
        self.assertIsInstance(model.density_weight, torch.Tensor)
        self.assertTrue(torch.equal(model.density_weight, torch.tensor([0.0])))
        self.assertEqual(model.density_weight_step_size, 4.0)

    def test_restart_probe_payload_extracts_metric_values(self):
        NonLinearPlace = _load_nonlinearplace_class()
        metric = types.SimpleNamespace(
            hpwl=torch.tensor(12.5, dtype=torch.float64),
            overflow=torch.tensor([0.1, 0.4, 0.2], dtype=torch.float64),
            density_weight=torch.tensor([3.0, 4.0], dtype=torch.float64),
        )

        payload = NonLinearPlace._restart_probe_payload(
            "warm_schedule", "global", 7, metric
        )

        self.assertEqual(payload, {
            "restart_policy": "warm_schedule",
            "phase": "global",
            "handoff_seq": 7,
            "hpwl": 12.5,
            "overflow": 0.4,
            "density_weight": [3.0, 4.0],
        })

    def test_prepare_restart_state_stores_warm_state_when_policy_enabled(self):
        NonLinearPlace = _load_nonlinearplace_class()
        params = types.SimpleNamespace(handoff_restart_policy="warm_schedule")
        placedb = types.SimpleNamespace(
            pending_continuation_seed=None,
            pending_warm_schedule_state=None,
            num_nodes=2,
            num_movable_nodes=1,
            num_physical_nodes=2,
            num_filler_nodes=0,
            node_names=["u0", "macro0"],
        )
        engine = NonLinearPlace.__new__(NonLinearPlace)
        engine._build_continuation_seed = (
            lambda placedb, result, pos, seq: {"seed": True}
        )
        model = types.SimpleNamespace(density_weight=torch.tensor([2.0]))

        result = engine._prepare_restart_state(
            params,
            placedb,
            {"mutation_kind": "topology_changed"},
            pos=np.array([1.0, 2.0, 3.0, 4.0]),
            handoff_seq=1,
            model=model,
        )

        self.assertEqual(result["restart_policy"], "warm_schedule")
        self.assertEqual(placedb.pending_continuation_seed, {"seed": True})
        self.assertTrue(result["warm_schedule_state_captured"])
        self.assertTrue(
            torch.equal(
                placedb.pending_warm_schedule_state["density_weight"],
                torch.tensor([2.0]),
            )
        )

    def test_handoff_transaction_commits_restart_metadata(self):
        NonLinearPlace = _load_nonlinearplace_class()
        engine = NonLinearPlace.__new__(NonLinearPlace)
        engine._current_placement_counts_for_continuation_seed = lambda placedb: {
            "num_nodes": 10,
            "num_movable_nodes": 8,
            "num_physical_nodes": 10,
            "num_filler_nodes": 0,
        }
        engine._restart_after_topology_sync = (
            lambda params, placedb_arg, result, pos=None, handoff_seq=None: {
                "runtimedb": "rebuilt"
            }
        )

        class FakePlaceDB:
            def __init__(self):
                self.handoff_session = None
                self.openroad_bridge = types.SimpleNamespace(
                    session_identity="bridge-1"
                )

            def create_or_reset_handoff_session(self, session_id=None):
                self.handoff_session = PlacementHandoffSession(
                    session_id=session_id or "run-1"
                )
                return self.handoff_session

            def _get_openroad_session_identity(self):
                return self.openroad_bridge.session_identity

            def current_topology_counts(self):
                return {"nodes": 10, "pins": 20, "nets": 5}

            def current_snapshot_fingerprint(self, counts=None):
                return "nodes=10|pins=20|nets=5"

            def execute_openroad_handoff(self, params, pos, handoff_event):
                return {
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "post_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    "post_snapshot_fingerprint": "nodes=12|pins=24|nets=6",
                    "added_buffer_count": 2,
                    "removed_buffer_count": 1,
                    "surviving_buffer_count": 7,
                    "sync_contract": {
                        "mutation_kind": "topology_changed",
                        "requires_runtimedb_rebuild": True,
                        "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    },
                }

        placedb = FakePlaceDB()

        engine._execute_openroad_handoff_transaction(
            params=types.SimpleNamespace(handoff_restart_policy="warm_schedule"),
            placedb=placedb,
            pos="pos-token",
            run_event={"absolute_iteration": 50, "phase": 0},
            trigger_decision={
                "trigger_mode": "interval",
                "trigger_reason": "interval=50",
            },
            handoff_event={"absolute_iteration": 50, "phase": 0},
        )

        event = placedb.handoff_session.event_history[-1]
        self.assertIn("restart_policy", event)
        self.assertIn("restart_probes", event)
        self.assertIn("restart_probe_target_iterations", event)
        self.assertEqual(event["restart_policy"], "warm_schedule")
        self.assertEqual(event["restart_probes"], [])
        self.assertEqual(event["restart_probe_target_iterations"], [0, 50])

    def test_execute_handoff_sets_placedb_generation_before_rebuilt_call(self):
        NonLinearPlace = _load_nonlinearplace_class()
        engine = object.__new__(NonLinearPlace)
        params = types.SimpleNamespace(handoff_restart_policy="warm_schedule")
        observed = {}

        class FakePlaceDB:
            def __init__(self):
                self.handoff_session = None
                self.runtimedb_generation = 0

            def create_or_reset_handoff_session(self):
                self.handoff_session = PlacementHandoffSession("plot-clock")
                return self.handoff_session

            def current_topology_counts(self):
                return {"nodes": 10, "pins": 20, "nets": 5}

            def current_snapshot_fingerprint(self, counts):
                return "nodes=%d|pins=%d|nets=%d" % (
                    counts["nodes"],
                    counts["pins"],
                    counts["nets"],
                )

            def _get_openroad_session_identity(self):
                return "bridge-1"

            def execute_openroad_handoff(self, params, pos, handoff_event):
                return {
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "post_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    "post_snapshot_fingerprint": "nodes=12|pins=24|nets=6",
                    "added_buffer_count": 2,
                    "removed_buffer_count": 0,
                    "surviving_buffer_count": 2,
                    "sync_contract": {
                        "mutation_kind": "topology_changed",
                        "requires_runtimedb_rebuild": True,
                    },
                }

        def fake_restart(params, placedb, handoff_result, pos=None, handoff_seq=None, model=None):
            def rebuilt_call(call_params, call_placedb):
                observed["generation"] = call_placedb.runtimedb_generation
                observed["model_iteration"] = getattr(model, "iteration", None)
                observed["placedb_iteration"] = getattr(call_placedb, "iteration", None)
                return "continued"
            return rebuilt_call

        placedb = FakePlaceDB()
        engine._restart_after_topology_sync = fake_restart
        model = types.SimpleNamespace(iteration=123)

        result = engine._execute_openroad_handoff_transaction(
            params=params,
            placedb=placedb,
            pos="pos-token",
            run_event={"absolute_iteration": 50, "phase": 0},
            trigger_decision={
                "trigger_mode": "periodic_iteration",
                "trigger_reason": "periodic_iteration@period=50@iter=50",
            },
            handoff_event={"absolute_iteration": 50, "phase": 0},
            model=model,
        )

        self.assertEqual(result, "continued")
        self.assertEqual(observed["generation"], 1)
        self.assertEqual(observed["model_iteration"], 123)
        self.assertIsNone(observed["placedb_iteration"])


if __name__ == "__main__":
    unittest.main()
