#!/usr/bin/env python3

import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dreamplace.macroPlaceDB import MacroPlaceDB  # noqa: E402
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction  # noqa: E402
from dreamplace.ops.placeio_ecc.timing_generation import timing_generation_lease  # noqa: E402


class FakeEccModule:
    def __init__(self):
        self.calls = []
        self.dir_workspace = "/tmp/ecc-workspace"

    def get_dmInst_ptr(self):
        self.calls.append(("get_dmInst_ptr",))
        return "dm-inst"

    def pydb(self, dm_inst, bins_x, bins_y, routability, with_sta,
             include_m2_pg_rail_blockage=False, include_m2_pg_rail_density=True,
             timing_model_only=False):
        self.export_options = (include_m2_pg_rail_blockage, include_m2_pg_rail_density)
        self.calls.append(
            ("pydb", dm_inst, bins_x, bins_y, routability, with_sta, timing_model_only)
        )
        return types.SimpleNamespace(
            num_nodes=2,
            timing_schema_version=2 if with_sta else 0,
            pin_names=[],
            end_points=[],
            clock_pins=[],
            endpoints_constraint_arcs=[],
            endpoints_timing_check_arcs=[],
            endpoints_max_valid=[],
            endpoints_max_reason=[],
            endpoints_constraint_max_valid=[],
            endpoints_constraint_max_reason=[],
            endpoints_timing_check_max_valid=[],
            endpoints_timing_check_max_reason=[],
            net_names=[],
            inst_main_id=[0, 0],
            inst_libcell_offset=[0, 0],
            main_id_2_cell_id_start=[0, 2],
        )

    def write_placement_back(self, dm_inst, node_x, node_y):
        self.calls.append(("write_placement_back", dm_inst, node_x, node_y))

    def def_save(self, path):
        self.calls.append(("def_save", path))
        return True

    def tcl_save(self, path):
        self.calls.append(("tcl_save", path))
        return True

    def prepare_place_timing(self, inputs):
        self.calls.append(("prepare_place_timing", dict(inputs)))
        return {"status": "ok"}

    def refresh_place_timing(self, inputs):
        self.calls.append(("refresh_place_timing", dict(inputs)))
        return {"status": "ok", "mode": "full_rebuild", "rcx": {"status": "ok"}}

    def verilog_save(self, path):
        self.calls.append(("verilog_save", path))
        return True


class EccPlaceIOTest(unittest.TestCase):
    def test_geometry_read_and_writeback_use_native_wrapper(self):
        module = FakeEccModule()
        params = types.SimpleNamespace(
            with_sta=0,
            route_num_bins_x=64,
            route_num_bins_y=32,
            routability_opt_flag=1,
        )

        raw_db = PlaceIOFunction.read(params, module)

        self.assertEqual("dm-inst", raw_db.dm_inst)
        self.assertEqual(2, raw_db.pydb.num_nodes)
        self.assertEqual(
            ("pydb", "dm-inst", 64, 32, True, False, False),
            module.calls[1],
        )
        self.assertEqual("ecc", PlaceIOFunction.backend_caps(raw_db)["backend"])

        PlaceIOFunction.apply(raw_db, [1.0, 2.0], [3.0, 4.0])
        PlaceIOFunction.write_def(raw_db, "/tmp/out.def")
        PlaceIOFunction.write_tcl(raw_db, "/tmp/out.tcl")

        self.assertEqual(
            ("write_placement_back", "dm-inst", [1.0, 2.0], [3.0, 4.0]),
            module.calls[2],
        )
        self.assertEqual(("def_save", "/tmp/out.def"), module.calls[3])
        self.assertEqual(("tcl_save", "/tmp/out.tcl"), module.calls[4])

    def test_timing_export_uses_native_timing_session(self):
        module = FakeEccModule()
        params = types.SimpleNamespace(
            with_sta=1,
            design_inputs={"lib": ["cells.lib"], "sdc": "design.sdc"},
        )

        raw_db = PlaceIOFunction.read(params, module)

        self.assertTrue(raw_db.timing_enabled)
        self.assertTrue(PlaceIOFunction.backend_caps(raw_db)["supports_sta_pydb"])
        self.assertEqual(
            ("prepare_place_timing", {"lib": ["cells.lib"], "sdc": "design.sdc"}),
            module.calls[0],
        )
        self.assertEqual(
            ("pydb", "dm-inst", 512, 512, False, True, False),
            module.calls[2],
        )

    def test_timing_refresh_replaces_pydb(self):
        module = FakeEccModule()
        params = types.SimpleNamespace(
            with_sta=1,
            design_inputs={"lib": ["cells.lib"], "sdc": "design.sdc"},
        )

        raw_db = PlaceIOFunction.read(params, module)
        old_pydb = raw_db.pydb
        new_pydb, summary = PlaceIOFunction.refresh(raw_db)

        self.assertIs(raw_db.pydb, new_pydb)
        self.assertIsNot(old_pydb, new_pydb)
        self.assertEqual(1, raw_db.refresh_generation)
        self.assertEqual("ok", summary["status"])
        self.assertEqual("full_rebuild", summary["refresh_mode"])
        self.assertTrue(summary["pydb_replaced"])

    def test_native_full_refresh_uses_full_macro_placedb_rebuild(self):
        module = FakeEccModule()
        raw_db = PlaceIOFunction.read(
            types.SimpleNamespace(with_sta=1, design_inputs={"lib": ["cells.lib"]}),
            module,
        )
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.rawdb = raw_db
        placedb.pydb = raw_db.pydb
        rebuild_modes = []
        placedb.rebuild_from_pydb = lambda _pydb, rebuild_mode: (
            rebuild_modes.append(rebuild_mode) or {"requested_rebuild_mode": rebuild_mode}
        )

        summary = placedb.refresh_from_ecc_backend()

        self.assertEqual(["all"], rebuild_modes)
        self.assertEqual("all", summary["rebuild"]["requested_rebuild_mode"])
        self.assertTrue(summary["macro_pydb_replaced"])

    def test_refresh_without_rcx_does_not_claim_full_capability(self):
        module = FakeEccModule()
        module.refresh_place_timing = lambda _inputs: {
            "status": "ok",
            "rcx": {"status": "skipped", "reason": "no_rcx_config"},
        }
        raw_db = PlaceIOFunction.read(types.SimpleNamespace(with_sta=1), module)

        _, summary = PlaceIOFunction.refresh(raw_db)

        self.assertTrue(summary["pydb_replaced"])
        self.assertFalse(summary["backend_capabilities"]["supports_committed_refresh"])
        self.assertEqual(
            "native_rebuilt_without_fresh_rc",
            summary["backend_capabilities"]["sta_state_status"],
        )

    def test_failed_refresh_invalidates_old_timing_model(self):
        module = FakeEccModule()
        raw_db = PlaceIOFunction.read(
            types.SimpleNamespace(with_sta=1, design_inputs={"lib": ["cells.lib"]}),
            module,
        )
        old_pydb = raw_db.pydb
        lease = timing_generation_lease(types.SimpleNamespace(rawdb=raw_db, pydb=old_pydb))
        module.refresh_place_timing = lambda _inputs: (_ for _ in ()).throw(
            RuntimeError("native rebuild failed")
        )

        with self.assertRaisesRegex(RuntimeError, "native rebuild failed"):
            PlaceIOFunction.refresh(raw_db)

        self.assertIsNone(raw_db.pydb)
        self.assertIsNot(raw_db.pydb, old_pydb)
        self.assertFalse(raw_db.timing_enabled)
        self.assertIn("native rebuild failed", raw_db.refresh_invalidated_reason)
        self.assertFalse(PlaceIOFunction.backend_caps(raw_db)["supports_sta_pydb"])
        self.assertFalse(PlaceIOFunction.backend_caps(raw_db)["supports_committed_refresh"])
        self.assertEqual(
            "invalidated_refresh_failure",
            PlaceIOFunction.backend_caps(raw_db)["sta_state_status"],
        )
        self.assertEqual(0, raw_db.refresh_generation)
        with self.assertRaisesRegex(RuntimeError, "invalid or replaced native generation"):
            lease.require_current()
        with self.assertRaisesRegex(RuntimeError, "does not support sizing writeback"):
            PlaceIOFunction.apply_sizing(raw_db, [1, 0], ["BUFX1", "BUFX2"])

    def test_two_refreshes_invalidate_each_prior_tensor_generation(self):
        module = FakeEccModule()
        raw_db = PlaceIOFunction.read(types.SimpleNamespace(with_sta=1), module)
        leases = []
        original_refresh = module.refresh_place_timing

        def refresh(inputs):
            self.assertFalse(raw_db.timing_enabled)
            with self.assertRaisesRegex(RuntimeError, "invalid or replaced native generation"):
                leases[-1].require_current()
            return original_refresh(inputs)

        module.refresh_place_timing = refresh
        for generation in range(3):
            self.assertEqual(generation, raw_db.refresh_generation)
            current = timing_generation_lease(
                types.SimpleNamespace(rawdb=raw_db, pydb=raw_db.pydb)
            )
            current.require_current()
            for previous in leases:
                with self.assertRaisesRegex(RuntimeError, "invalid or replaced native generation"):
                    previous.require_current()
            leases.append(current)
            if generation < 2:
                PlaceIOFunction.refresh(raw_db)
        self.assertTrue(raw_db.timing_enabled)

    def test_operator_rejects_stale_tensor_generation_before_timing_work(self):
        import torch
        from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation

        module = FakeEccModule()
        raw_db = PlaceIOFunction.read(types.SimpleNamespace(with_sta=1), module)
        placedb = types.SimpleNamespace(rawdb=raw_db, pydb=raw_db.pydb)
        timing = TimingPropagation.__new__(TimingPropagation)
        torch.nn.Module.__init__(timing)
        timing.timing_model_lease = timing_generation_lease(placedb)
        PlaceIOFunction.refresh(raw_db)
        with self.assertRaisesRegex(RuntimeError, "invalid or replaced native generation"):
            timing({}, {}, {})
        with self.assertRaisesRegex(RuntimeError, "invalid or replaced native generation"):
            timing_generation_lease(placedb)

    def test_mutation_capabilities_promote_after_native_round_trips(self):
        module = FakeEccModule()
        params = types.SimpleNamespace(
            with_sta=1,
            design_inputs={"lib": ["cells.lib"], "sdc": "design.sdc"},
        )

        raw_db = PlaceIOFunction.read(params, module)
        self.assertFalse(PlaceIOFunction.backend_caps(raw_db)["supports_apply_sizing"])
        self.assertFalse(PlaceIOFunction.backend_caps(raw_db)["supports_buffer_commit"])
        self.assertFalse(
            PlaceIOFunction.backend_caps(raw_db)["supports_committed_refresh"]
        )

        actions = []
        raw_db.pydb.apply_sizing = lambda ids, names: (
            actions.append((ids, names)) or {"ok": True, "accepted_count": 1}
        )
        sizing = PlaceIOFunction.apply_sizing(raw_db, [1, 0], ["BUFX1", "BUFX2"])
        self.assertTrue(sizing["ok"])
        self.assertEqual([([0], ["BUFX2"])], actions)
        self.assertEqual(1, sizing["requested_count"])
        self.assertTrue(PlaceIOFunction.backend_caps(raw_db)["supports_apply_sizing"])

        buffer_calls = []

        def native_buffer(actions, digest):
            self.assertIsInstance(actions, list)
            buffer_calls.append((actions, digest))
            return {"status": "accepted", "accepted_count": 1,
                    "rejected_count": 0, "failed_count": 0}

        raw_db.pydb.apply_buffer_actions = native_buffer
        buffered = PlaceIOFunction.apply_buffer_actions(raw_db, ({"action_id": 0},), "digest")
        self.assertEqual(buffer_calls, [([{"action_id": 0}], "digest")])
        self.assertEqual(buffered["accepted_action_count"], 1)
        self.assertEqual("accepted", buffered["status"])
        self.assertTrue(PlaceIOFunction.backend_caps(raw_db)["supports_buffer_commit"])

        PlaceIOFunction.refresh(raw_db)
        capabilities = PlaceIOFunction.backend_caps(raw_db)
        self.assertTrue(capabilities["supports_committed_refresh"])
        self.assertEqual("validated_native_full_rebuild", capabilities["sta_state_status"])

    def test_missing_native_method_is_reported(self):
        params = types.SimpleNamespace(with_sta=0)

        with self.assertRaisesRegex(TypeError, "write_placement_back"):
            PlaceIOFunction.read(params, types.SimpleNamespace(pydb=lambda *args: None))

    def test_rejected_sizing_writeback_cannot_be_reported_as_success(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.num_physical_nodes = 1
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.node_x = np.array([0.0])
        placedb.node_y = np.array([0.0])
        placedb.write_sizing_back = lambda: {
            "ok": False,
            "accepted_count": 0,
            "requested_count": 1,
        }
        params = types.SimpleNamespace(
            placement_sizing_mode="size_only",
            place_io_engine="ecc",
            scale_factor=1.0,
            shift_factor=[0.0, 0.0],
        )

        with self.assertRaisesRegex(RuntimeError, "ECC sizing writeback rejected"):
            placedb.apply(params, np.array([0.0]), np.array([0.0]))

    def test_size_only_preserves_integer_dbu_after_float32_scaling(self):
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.num_physical_nodes = 1
        placedb.num_terminals = placedb.num_terminal_NIs = 0
        placedb.node_x = np.zeros(1, dtype=np.float32)
        placedb.node_y = np.zeros(1, dtype=np.float32)
        placedb.write_sizing_back = lambda: {"ok": True, "accepted_count": 0, "requested_count": 0}
        placedb.rawdb = types.SimpleNamespace(timing_enabled=False)
        positions = []
        placedb.write_placement_back = lambda x, y, **kwargs: positions.append(
            (x.tolist(), y.tolist())
        )
        params = types.SimpleNamespace(
            place_io_engine="ecc", placement_sizing_mode="size_only",
            scale_factor=0.005, shift_factor=[1000, 1400],
        )
        placedb.apply(params, np.array([(63872 - 1000) * 0.005], dtype=np.float32),
                      np.array([(9744 - 1400) * 0.005], dtype=np.float32))
        self.assertEqual(positions, [([63872], [9744])])


if __name__ == "__main__":
    unittest.main()
