import os
import sys
import types
import unittest

import torch


def _install_placeobj_import_stubs():
    modules = {
        "dreamplace.BasicPlace": types.ModuleType("dreamplace.BasicPlace"),
        "dreamplace.ops": types.ModuleType("dreamplace.ops"),
        "dreamplace.ops.weighted_average_wirelength": types.ModuleType("dreamplace.ops.weighted_average_wirelength"),
        "dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength": types.ModuleType("dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength"),
        "dreamplace.ops.logsumexp_wirelength": types.ModuleType("dreamplace.ops.logsumexp_wirelength"),
        "dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength": types.ModuleType("dreamplace.ops.logsumexp_wirelength.logsumexp_wirelength"),
        "dreamplace.ops.electric_potential": types.ModuleType("dreamplace.ops.electric_potential"),
        "dreamplace.ops.electric_potential.electric_potential": types.ModuleType("dreamplace.ops.electric_potential.electric_potential"),
        "dreamplace.ops.electric_potential.electric_overflow": types.ModuleType("dreamplace.ops.electric_potential.electric_overflow"),
        "dreamplace.ops.density_overflow": types.ModuleType("dreamplace.ops.density_overflow"),
        "dreamplace.ops.density_overflow.density_overflow": types.ModuleType("dreamplace.ops.density_overflow.density_overflow"),
        "dreamplace.ops.density_potential": types.ModuleType("dreamplace.ops.density_potential"),
        "dreamplace.ops.density_potential.density_potential": types.ModuleType("dreamplace.ops.density_potential.density_potential"),
        "dreamplace.ops.rudy": types.ModuleType("dreamplace.ops.rudy"),
        "dreamplace.ops.rudy.rudy": types.ModuleType("dreamplace.ops.rudy.rudy"),
        "dreamplace.ops.rudy.rudy_macros": types.ModuleType("dreamplace.ops.rudy.rudy_macros"),
        "dreamplace.ops.pin_utilization": types.ModuleType("dreamplace.ops.pin_utilization"),
        "dreamplace.ops.pin_utilization.pin_utilization": types.ModuleType("dreamplace.ops.pin_utilization.pin_utilization"),
        "dreamplace.ops.nctugr_binary": types.ModuleType("dreamplace.ops.nctugr_binary"),
        "dreamplace.ops.nctugr_binary.nctugr_binary": types.ModuleType("dreamplace.ops.nctugr_binary.nctugr_binary"),
        "dreamplace.ops.irt_egr": types.ModuleType("dreamplace.ops.irt_egr"),
        "dreamplace.ops.irt_egr.irt_egr": types.ModuleType("dreamplace.ops.irt_egr.irt_egr"),
        "dreamplace.ops.adjust_node_area": types.ModuleType("dreamplace.ops.adjust_node_area"),
        "dreamplace.ops.adjust_node_area.adjust_node_area": types.ModuleType("dreamplace.ops.adjust_node_area.adjust_node_area"),
        "dreamplace.ops.macro_overlap": types.ModuleType("dreamplace.ops.macro_overlap"),
        "dreamplace.ops.macro_overlap.macro_overlap": types.ModuleType("dreamplace.ops.macro_overlap.macro_overlap"),
        "dreamplace.ops.macro_refinement": types.ModuleType("dreamplace.ops.macro_refinement"),
        "dreamplace.ops.macro_refinement.macro_refinement": types.ModuleType("dreamplace.ops.macro_refinement.macro_refinement"),
        "dreamplace.ops.timing_propagation": types.ModuleType("dreamplace.ops.timing_propagation"),
        "dreamplace.ops.timing_propagation.timing_propagation": types.ModuleType("dreamplace.ops.timing_propagation.timing_propagation"),
        "dreamplace.ops.timing_propagation.crash_stage_marker": types.ModuleType("dreamplace.ops.timing_propagation.crash_stage_marker"),
        "dreamplace.ops.cell_modeling": types.ModuleType("dreamplace.ops.cell_modeling"),
        "dreamplace.ops.cell_modeling.cell_modeling": types.ModuleType("dreamplace.ops.cell_modeling.cell_modeling"),
        "dreamplace.ops.rc_timing": types.ModuleType("dreamplace.ops.rc_timing"),
        "dreamplace.ops.rc_timing.rc_timing": types.ModuleType("dreamplace.ops.rc_timing.rc_timing"),
        "dreamplace.ops.pin2pin_attraction": types.ModuleType("dreamplace.ops.pin2pin_attraction"),
        "dreamplace.ops.pin2pin_attraction.pin2pin_attraction": types.ModuleType("dreamplace.ops.pin2pin_attraction.pin2pin_attraction"),
        "tools": types.ModuleType("tools"),
        "tools.iEDA": types.ModuleType("tools.iEDA"),
        "tools.iEDA.module": types.ModuleType("tools.iEDA.module"),
        "tools.iEDA.module.sta": types.ModuleType("tools.iEDA.module.sta"),
    }
    modules["dreamplace.BasicPlace"].PlaceDataCollection = object
    timing_module = modules["dreamplace.ops.timing_propagation.timing_propagation"]
    timing_module.TimingPropagation = object
    timing_module.SMOOTH_MAX_ALPHA = 10.0
    timing_module.build_pin_violation_detail_rows = lambda *args, **kwargs: []
    timing_module.smooth_max = lambda value, *args, **kwargs: value
    timing_module.write_pin_violation_detail_csv = lambda *args, **kwargs: None
    modules[
        "dreamplace.ops.timing_propagation.crash_stage_marker"
    ].write_crash_stage_marker = lambda *args, **kwargs: None
    modules["dreamplace.ops.cell_modeling.cell_modeling"].CellModeling = object
    modules["dreamplace.ops.rc_timing.rc_timing"].RCTiming = object
    modules["tools.iEDA.module.sta"].IEDASta = object
    previous = {}
    package_names = {
        "dreamplace.ops",
        "dreamplace.ops.weighted_average_wirelength",
        "dreamplace.ops.logsumexp_wirelength",
        "dreamplace.ops.electric_potential",
        "dreamplace.ops.density_overflow",
        "dreamplace.ops.density_potential",
        "dreamplace.ops.rudy",
        "dreamplace.ops.pin_utilization",
        "dreamplace.ops.nctugr_binary",
        "dreamplace.ops.irt_egr",
        "dreamplace.ops.adjust_node_area",
        "dreamplace.ops.macro_overlap",
        "dreamplace.ops.macro_refinement",
        "dreamplace.ops.timing_propagation",
        "dreamplace.ops.cell_modeling",
        "dreamplace.ops.rc_timing",
        "dreamplace.ops.pin2pin_attraction",
        "tools",
        "tools.iEDA",
        "tools.iEDA.module",
    }
    for name, module in modules.items():
        previous[name] = sys.modules.get(name)
        if name in package_names:
            module.__path__ = []
        sys.modules[name] = module
    # The sizing utilities are pure Python (torch only) and live under
    # dreamplace.ops.size_interpolated_pin: keep the real package path reachable
    # so they load from the source tree instead of being stubbed.
    modules["dreamplace.ops"].__path__ = [
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "dreamplace",
            "ops",
        )
    ]
    return previous


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_placeobj_class():
    previous = _install_placeobj_import_stubs()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from dreamplace.PlaceObj import PlaceObj
    finally:
        sys.path.pop()
        _restore_modules(previous)
    return PlaceObj


class PlaceObjTimingSnapshotTest(unittest.TestCase):
    def test_pin_2_libpin_ids_excludes_negative_pin_offsets(self):
        PlaceObj = _load_placeobj_class()
        place_obj = PlaceObj.__new__(PlaceObj)
        inst_size = torch.tensor([0, 1, 2], dtype=torch.long)
        data_collections = types.SimpleNamespace(
            pin2node_map=torch.tensor([0, 1, 2], dtype=torch.long),
            inst_main_id=torch.tensor([0, 0, 0], dtype=torch.long),
            main_id_2_cell_id_start=torch.tensor([0], dtype=torch.long),
            pin_2_libpin_offset=torch.tensor([0, -1, 2], dtype=torch.long),
            cell_id_2_libpin_id_start=torch.tensor(
                [100, 200, 300],
                dtype=torch.long,
            ),
        )

        inst_pins_mask, libpin_ids = place_obj.pin_2_libpin_ids(
            inst_size,
            data_collections,
        )

        self.assertTrue(
            torch.equal(
                inst_pins_mask,
                torch.tensor([True, False, True], dtype=torch.bool),
            )
        )
        self.assertTrue(
            torch.equal(libpin_ids, torch.tensor([100, 302], dtype=torch.long))
        )

    def test_pin_caps_op_initializes_pin_library_map_with_current_torch_api(self):
        PlaceObj = _load_placeobj_class()
        place_obj = PlaceObj.__new__(PlaceObj)
        place_obj.placedb = types.SimpleNamespace(pin_names=["pin0", "pin1", "pin2"])
        inst_size = torch.tensor([0, 1], dtype=torch.long)
        data_collections = types.SimpleNamespace(
            pin2node_map=torch.tensor([0, 1, 2], dtype=torch.long),
            inst_main_id=torch.tensor([0, 0, -1], dtype=torch.long),
            main_id_2_cell_id_start=torch.tensor([0], dtype=torch.long),
            pin_2_libpin_offset=torch.tensor([0, 1, -1], dtype=torch.long),
            cell_id_2_libpin_id_start=torch.tensor([0, 2], dtype=torch.long),
            flat_lib_pin_cap=torch.tensor([0.5, 0.6, 0.7, 0.8], dtype=torch.float64),
            flat_lib_pin_rcap=torch.tensor([0.4, 0.5, 0.6, 0.7], dtype=torch.float64),
            flat_lib_pin_fcap=torch.tensor([0.3, 0.4, 0.5, 0.6], dtype=torch.float64),
            end_points=torch.tensor([2], dtype=torch.long),
            outcaps=torch.tensor([1.25], dtype=torch.float64),
            get_size_var=lambda: None,
            get_vt_var=lambda: None,
        )

        pin_cap_base, pin_rcap_base, pin_fcap_base = place_obj.pin_caps_op(
            inst_size,
            data_collections,
        )

        self.assertTrue(
            torch.equal(
                place_obj.inst_pins_mask,
                torch.tensor([True, True, False], dtype=torch.bool),
            )
        )
        self.assertTrue(
            torch.equal(
                place_obj.pin2libpin_flat_ids[place_obj.inst_pins_mask],
                torch.tensor([0, 3], dtype=torch.long),
            )
        )
        self.assertEqual(pin_cap_base.dtype, torch.float64)
        self.assertTrue(
            torch.equal(
                pin_cap_base,
                torch.tensor([0.5, 0.8, 1.25], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                pin_rcap_base,
                torch.tensor([0.4, 0.7, 1.25], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                pin_fcap_base,
                torch.tensor([0.3, 0.6, 1.25], dtype=torch.float64),
            )
        )

    def test_mapped_instance_pin_limits_returns_none_on_shape_mismatch(self):
        PlaceObj = _load_placeobj_class()
        place_obj = PlaceObj.__new__(PlaceObj)

        result = place_obj._mapped_instance_pin_limits(
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([True, False, True], dtype=torch.bool),
            torch.tensor([0, 1], dtype=torch.long),
        )

        self.assertIsNone(result)

    def test_timing_metric_snapshot_maps_instance_pins_to_library_pin_limits(self):
        PlaceObj = _load_placeobj_class()
        place_obj = PlaceObj.__new__(PlaceObj)
        place_obj.wns = torch.tensor(-0.11)
        place_obj.tns = torch.tensor(-1.2)
        place_obj.ws = torch.tensor(-0.09)
        place_obj.ts = torch.tensor(0.0)
        place_obj.inst_pins_mask = torch.tensor([True, False, True, True], dtype=torch.bool)
        place_obj.pin2libpin_flat_ids = torch.tensor([4, 1, 0, 2], dtype=torch.long)
        place_obj.data_collections = types.SimpleNamespace(
            flat_lib_pin_slew_limit=torch.tensor([0.05, 0.11, 0.07, 0.13, 0.09], dtype=torch.float32),
            flat_lib_pin_cap_limit=torch.tensor([0.10, 0.50, 0.20, 0.30, 0.12], dtype=torch.float32),
        )
        place_obj.op_collections = types.SimpleNamespace(
            timing_propagation_op=types.SimpleNamespace(
                pin_rtran=torch.tensor([0.08, 9.50, 0.02, 0.06], dtype=torch.float32),
                pin_ftran=torch.tensor([0.12, 9.75, 0.10, 0.05], dtype=torch.float32),
                pin_net_cap_rise=torch.tensor([0.14, 8.00, 0.08, 0.18], dtype=torch.float32),
                pin_net_cap_fall=torch.tensor([0.11, 8.50, 0.15, 0.24], dtype=torch.float32),
            )
        )

        snapshot = place_obj.timing_metric_snapshot()

        self.assertEqual(
            snapshot,
            {
                "wns": -0.10999999940395355,
                "tns": -1.2000000476837158,
                "ws": -0.09000000357627869,
                "ts": 0.0,
                "max_slew_violation": 0.05000000074505806,
                "max_load_cap_violation": 0.05000000447034836,
            },
        )


if __name__ == "__main__":
    unittest.main()
