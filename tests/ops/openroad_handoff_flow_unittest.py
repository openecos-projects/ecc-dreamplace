import os
import numpy as np
import re
import sys
import types
import unittest
import tempfile
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.openroad_handoff import OpenRoadHandoffController
from dreamplace.ops.openroad_handoff import PlacementHandoffSession
from dreamplace.ops.routability.cooptimization_area import initialize_overflow_reference
sys.path.pop()

_tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _tests_dir)
from _import_stubs import auto_stub, register_stub_modules
sys.path.remove(_tests_dir)

# The hand-written stubs below isolate the heavy op layer; auto_stub fabricates
# whatever the product imports from dreamplace.ops so a new op import cannot rot
# these lists again (they used to fail with "'dreamplace.ops' is not a package").
# `dreamplace.ops.openroad_handoff` must stay real: several tests assert against
# its concrete controller/session classes and read its recorded event history.
_OPS_STUBS = auto_stub("dreamplace.ops", except_=("dreamplace.ops.openroad_handoff",))

def _install_macroplacedb_import_stubs():
    _OPS_STUBS.__enter__()
    tool_modules = {
        "tools": types.ModuleType("tools"),
        "tools.iEDA": types.ModuleType("tools.iEDA"),
        "tools.iEDA.data": types.ModuleType("tools.iEDA.data"),
        "tools.iEDA.data.design": types.ModuleType("tools.iEDA.data.design"),
        "tools.iEDA.module": types.ModuleType("tools.iEDA.module"),
        "tools.iEDA.module.io": types.ModuleType("tools.iEDA.module.io"),
        "dreamplace.ops": types.ModuleType("dreamplace.ops"),
        "dreamplace.ops.fence_region": types.ModuleType("dreamplace.ops.fence_region"),
        "dreamplace.ops.fence_region.fence_region": types.ModuleType("dreamplace.ops.fence_region.fence_region"),
    }
    tool_modules["tools.iEDA.data.design"].IEDADesign = object
    tool_modules["tools.iEDA.module.io"].IEDAIO = object
    return register_stub_modules(tool_modules)


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            if module is None:
                parent.__dict__.pop(child, None)
            else:
                setattr(parent, child, module)
    _OPS_STUBS.__exit__(None, None, None)


def _load_macroplacedb_class():
    previous = _install_macroplacedb_import_stubs()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from dreamplace.macroPlaceDB import MacroPlaceDB
    finally:
        sys.path.pop()
        _restore_modules(previous)
    return MacroPlaceDB


def _install_nonlinearplace_import_stubs():
    _OPS_STUBS.__enter__()
    tool_modules = {
        "dreamplace.BasicPlace": types.ModuleType("dreamplace.BasicPlace"),
        "dreamplace.PlaceObj": types.ModuleType("dreamplace.PlaceObj"),
        "dreamplace.NesterovAcceleratedGradientOptimizer": types.ModuleType(
            "dreamplace.NesterovAcceleratedGradientOptimizer"
        ),
        "dreamplace.EvalMetrics": types.ModuleType("dreamplace.EvalMetrics"),
        "dreamplace.ops": types.ModuleType("dreamplace.ops"),
        "dreamplace.ops.fence_region": types.ModuleType("dreamplace.ops.fence_region"),
        "dreamplace.ops.fence_region.fence_region": types.ModuleType(
            "dreamplace.ops.fence_region.fence_region"
        ),
        "dreamplace.ops.timing_propagation": types.ModuleType(
            "dreamplace.ops.timing_propagation"
        ),
        "dreamplace.ops.timing_propagation.crash_stage_marker": types.ModuleType(
            "dreamplace.ops.timing_propagation.crash_stage_marker"
        ),
        "dreamplace.ops.timing_net_weighting": types.ModuleType(
            "dreamplace.ops.timing_net_weighting"
        ),
        "dreamplace.ops.timing_net_weighting.net_weighting": types.ModuleType(
            "dreamplace.ops.timing_net_weighting.net_weighting"
        ),
        "dreamplace.ops.gate_projection": types.ModuleType("dreamplace.ops.gate_projection"),
        "dreamplace.ops.gate_projection.gate_projection": types.ModuleType(
            "dreamplace.ops.gate_projection.gate_projection"
        ),
        "dreamplace.ops.discrete_gradient_topk": types.ModuleType(
            "dreamplace.ops.discrete_gradient_topk"
        ),
        "dreamplace.ops.size_interpolated_pin.sizing_limit_utils": types.ModuleType("dreamplace.ops.size_interpolated_pin.sizing_limit_utils"),
        "matplotlib": types.ModuleType("matplotlib"),
        "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
    }
    tool_modules["dreamplace.BasicPlace"].BasicPlace = type("BasicPlace", (), {})
    tool_modules["dreamplace.BasicPlace"].size_var_to_logits = mock.Mock(
        side_effect=AssertionError("size conversion is outside the handoff fixture")
    )
    tool_modules[
        "dreamplace.ops.timing_propagation.crash_stage_marker"
    ].write_crash_stage_marker = lambda *_args, **_kwargs: None
    tool_modules[
        "dreamplace.ops.timing_net_weighting.net_weighting"
    ].update_net_weights_from_tp = lambda *_args, **_kwargs: None
    tool_modules[
        "dreamplace.ops.gate_projection.gate_projection"
    ].GateProjectionOp = type("GateProjectionOp", (), {})
    tool_modules[
        "dreamplace.ops.gate_projection.gate_projection"
    ].ProjectionContext = type("ProjectionContext", (), {})
    tool_modules[
        "dreamplace.ops.discrete_gradient_topk"
    ].apply_discrete_gradient_topk_update = lambda *_args, **_kwargs: None
    tool_modules[
        "dreamplace.ops.discrete_gradient_topk"
    ].build_discrete_gradient_topk_candidate_cache = lambda *_args, **_kwargs: None
    tool_modules[
        "dreamplace.ops.size_interpolated_pin.sizing_limit_utils"
    ].build_size_interpolated_pin_property_debug = lambda *_args, **_kwargs: {}
    tool_modules[
        "dreamplace.ops.size_interpolated_pin.sizing_limit_utils"
    ].compute_current_pin2libpin_flat_ids = lambda *_args, **_kwargs: (None, None, None)
    tool_modules[
        "dreamplace.ops.size_interpolated_pin.sizing_limit_utils"
    ].compute_size_interpolated_pin_properties = lambda *_args, **_kwargs: (None, None)
    return register_stub_modules(tool_modules)


def _load_nonlinearplace_class():
    previous = _install_nonlinearplace_import_stubs()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from dreamplace.NonLinearPlace import NonLinearPlace
    finally:
        sys.path.pop()
        _restore_modules(previous)
    return NonLinearPlace


def _install_basicplace_import_stubs():
    _OPS_STUBS.__enter__()
    module_names = [
        "dreamplace.ops",
        "dreamplace.ops.move_boundary",
        "dreamplace.ops.move_boundary.move_boundary",
        "dreamplace.ops.hpwl",
        "dreamplace.ops.hpwl.hpwl",
        "dreamplace.ops.rmst_wl",
        "dreamplace.ops.rmst_wl.rmst_wl",
        "dreamplace.ops.macro_legalize",
        "dreamplace.ops.macro_legalize.macro_legalize",
        "dreamplace.ops.greedy_legalize",
        "dreamplace.ops.greedy_legalize.greedy_legalize",
        "dreamplace.ops.abacus_legalize",
        "dreamplace.ops.abacus_legalize.abacus_legalize",
        "dreamplace.ops.legality_check",
        "dreamplace.ops.legality_check.legality_check",
        "dreamplace.ops.draw_place",
        "dreamplace.ops.draw_place.draw_place",
        "dreamplace.ops.pin_pos",
        "dreamplace.ops.pin_pos.pin_pos",
        "dreamplace.ops.global_swap",
        "dreamplace.ops.global_swap.global_swap",
        "dreamplace.ops.k_reorder",
        "dreamplace.ops.k_reorder.k_reorder",
        "dreamplace.ops.independent_set_matching",
        "dreamplace.ops.independent_set_matching.independent_set_matching",
        "dreamplace.ops.pin_weight_sum",
        "dreamplace.ops.pin_weight_sum.pin_weight_sum",
        "dreamplace.ops.steiner_topo",
        "dreamplace.ops.steiner_topo.steiner_topo",
        "dreamplace.ops.cell_modeling",
        "dreamplace.ops.cell_modeling.cell_modeling",
        "dreamplace.ops.gate_projection",
        "dreamplace.ops.gate_projection.gate_projection",
        "dreamplace.ops.timing_propagation",
        "dreamplace.ops.timing_propagation.crash_stage_marker",
        "dreamplace.ops.timing_propagation.timing_propagation",
    ]
    previous = register_stub_modules(
        {name: types.ModuleType(name) for name in module_names}
    )
    sys.modules[
        "dreamplace.ops.timing_propagation.timing_propagation"
    ].ARCS_INFO = object()
    sys.modules[
        "dreamplace.ops.timing_propagation.timing_propagation"
    ].LUTS_INFO = object()
    class FakePinWeightSum:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    sys.modules[
        "dreamplace.ops.pin_weight_sum.pin_weight_sum"
    ].PinWeightSum = FakePinWeightSum
    sys.modules[
        "dreamplace.ops.cell_modeling.cell_modeling"
    ].CellModeling = type("CellModeling", (), {})
    gate_projection_module = sys.modules["dreamplace.ops.gate_projection.gate_projection"]
    gate_projection_module.ArgminProjectionResolver = type("ArgminProjectionResolver", (), {})
    gate_projection_module.GateProjectionOp = type("GateProjectionOp", (), {})
    gate_projection_module.NearestSizeProjectionResolver = type(
        "NearestSizeProjectionResolver", (), {}
    )
    gate_projection_module.StableCurrentCellResolver = type(
        "StableCurrentCellResolver", (), {}
    )
    sys.modules[
        "dreamplace.ops.timing_propagation.crash_stage_marker"
    ].write_crash_stage_marker = lambda *_args, **_kwargs: None
    return previous


def _load_basicplace_module():
    previous = _install_basicplace_import_stubs()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        import dreamplace.BasicPlace as BasicPlaceModule
    finally:
        sys.path.pop()
        _restore_modules(previous)
    return BasicPlaceModule


def _read_openroad_place_io_cpp():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(
        repo_root,
        "dreamplace",
        "ops",
        "placeio_openroad",
        "src",
        "openroad_place_io.cpp",
    )
    with open(source_path, "r") as source_file:
        return source_file.read()


def _read_openroad_pyplacedb_export_h():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(
        repo_root,
        "dreamplace",
        "ops",
        "placeio_openroad",
        "src",
        "openroad_pyplacedb_export.h",
    )
    with open(source_path, "r") as source_file:
        return source_file.read()


def _read_openroad_pyplacedb_export_impl_cpp():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(
        repo_root,
        "dreamplace",
        "ops",
        "placeio_openroad",
        "src",
        "openroad_pyplacedb_export_impl.cpp",
    )
    with open(source_path, "r") as source_file:
        return source_file.read()


def _read_openroad_sizing_cell_classification_h():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(
        repo_root,
        "dreamplace",
        "ops",
        "placeio_openroad",
        "src",
        "openroad_sizing_cell_classification.h",
    )
    with open(source_path, "r") as source_file:
        return source_file.read()


def _read_openroad_place_io_impl_bridge_h():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(
        repo_root,
        "dreamplace",
        "ops",
        "placeio_openroad",
        "src",
        "openroad_place_io_impl_bridge.h",
    )
    with open(source_path, "r") as source_file:
        return source_file.read()


def _strip_cpp_comments(source):
    return re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)


def _load_openroad_cpp_extension_or_none():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from dreamplace.ops.placeio_openroad import placeio_openroad_cpp
    except (ImportError, OSError):
        return None
    finally:
        sys.path.pop()
    return placeio_openroad_cpp


def _load_place_io_module_with_cpp_stub():
    cpp_module_name = "dreamplace.ops.placeio_openroad.placeio_openroad_cpp"
    previous_cpp_module = sys.modules.get(cpp_module_name)
    sys.modules[cpp_module_name] = types.ModuleType(cpp_module_name)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from dreamplace.ops.placeio_openroad import place_io
    finally:
        sys.path.pop()
        if previous_cpp_module is None:
            sys.modules.pop(cpp_module_name, None)
        else:
            sys.modules[cpp_module_name] = previous_cpp_module
    return place_io


def _assert_openroad_cpp_source_exports_snapshot_node_metadata(test_case):
    header_source = _strip_cpp_comments(_read_openroad_pyplacedb_export_h())
    impl_source = _strip_cpp_comments(_read_openroad_pyplacedb_export_impl_cpp())
    binding_source = _strip_cpp_comments(_read_openroad_place_io_cpp())
    struct_match = re.search(r"struct\s+PyPlaceDB\s*\{(?P<body>.*?)\n\};", header_source, re.DOTALL)
    test_case.assertIsNotNone(struct_match)
    struct_body = struct_match.group("body")
    test_case.assertRegex(struct_body, r"py::list\s+node_names\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_master_names\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_is_buffer\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_x\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_y\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_orient\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_size_x\s*;")
    test_case.assertRegex(struct_body, r"py::list\s+node_size_y\s*;")

    for field_name in ("node_master_names", "node_is_buffer"):
        test_case.assertRegex(
            binding_source,
            r"\.def_readwrite\s*\(\s*\"%s\"\s*,\s*&PyPlaceDB::%s\s*\)"
            % (field_name, field_name),
        )
        test_case.assertRegex(impl_source, r"pydb\.%s\.append\s*\(" % field_name)
        test_case.assertRegex(impl_source, r"assert\s*\(\s*py::len\s*\(\s*pydb\.%s\s*\)" % field_name)


def _assert_openroad_cpp_source_exports_libcell_names(test_case):
    header_source = _strip_cpp_comments(_read_openroad_pyplacedb_export_h())
    impl_source = _strip_cpp_comments(_read_openroad_pyplacedb_export_impl_cpp())
    binding_source = _strip_cpp_comments(_read_openroad_place_io_cpp())

    struct_match = re.search(r"struct\s+PyPlaceDB\s*\{(?P<body>.*?)\n\};", header_source, re.DOTALL)
    test_case.assertIsNotNone(struct_match)
    struct_body = struct_match.group("body")
    test_case.assertRegex(struct_body, r"py::list\s+flat_libcell_names\s*;")
    test_case.assertRegex(
        binding_source,
        r"\.def_readwrite\s*\(\s*\"flat_libcell_names\"\s*,\s*&PyPlaceDB::flat_libcell_names\s*\)",
    )
    test_case.assertRegex(impl_source, r"pydb\.flat_libcell_names\.append\s*\(\s*cell_name\s*\)")
    test_case.assertLess(
        impl_source.find("pydb.flat_libcell_names.append"),
        impl_source.find("pydb.flat_libcell_info.append"),
    )


def _assert_openroad_cpp_source_exports_size_and_vt_axes(test_case):
    source = _strip_cpp_comments(_read_openroad_pyplacedb_export_impl_cpp())
    classifier = _strip_cpp_comments(
        _read_openroad_sizing_cell_classification_h()
    )
    test_case.assertIn("CellClass classifyCell", classifier)
    test_case.assertIn("configured_vt_suffixes", classifier)
    test_case.assertIn("endsWith(cell_name, suffix)", classifier)
    test_case.assertIn('"--vt_suffix"', _read_openroad_place_io_cpp())
    test_case.assertIn("vtSuffixes()", source)
    test_case.assertIn("orderedAxis(drive_groups)", source)
    test_case.assertIn("orderedAxis(vt_groups)", source)
    test_case.assertRegex(
        source,
        r"cell_info\.append\s*\(\s*static_cast<double>\s*\(\s*"
        r"ordered_cells_timing_coordinate\[cell_id\]\s*\)\s*\)",
    )
    test_case.assertRegex(
        source,
        r"cell_info\.append\s*\(\s*ordered_cells_vt\[cell_id\]\s*\)",
    )
    test_case.assertIn("has_size_choices || has_vt_choices", source)


def _assert_openroad_cpp_source_rebuilds_node_index_after_buffer_insertion(test_case):
    source = _strip_cpp_comments(_read_openroad_place_io_impl_bridge_h())
    method_match = re.search(
        r"std::string\s+runBufferInsertion\s*\([^)]*\)\s*\{(?P<body>.*?)\n\s*(?:py::dict\s+runOneNetBuffer|void\s+setNodeOrient)",
        source,
        re.DOTALL,
    )
    test_case.assertIsNotNone(method_match)
    body = method_match.group("body")
    eval_match = re.search(
        r"auto\s+(?P<result>\w+)\s*=\s*design_->evalTclString\s*\(\s*cmd\s*\)\s*;",
        body,
    )
    test_case.assertIsNotNone(eval_match)
    result_name = eval_match.group("result")
    test_case.assertRegex(body, r"rebuildNodeInstIndex\s*\(\s*\)\s*;")
    test_case.assertRegex(body, r"return\s+%s\s*;" % result_name)
    test_case.assertLess(
        body.index("design_->evalTclString"),
        body.index("rebuildNodeInstIndex"),
    )
    test_case.assertLess(
        body.index("rebuildNodeInstIndex"),
        body.index("return %s" % result_name),
    )


def _make_topology_snapshot(
    node_names=("u0", "u1"),
    node_master_names=("BUF_X1", "INV_X1"),
    node_is_buffer=None,
    include_node_master_names=True,
    node_size_x=(1.0, 2.0),
    node_size_y=(1.5, 2.5),
    node_orient=("N", "N"),
    node_x=(10.0, 20.0),
    node_y=(30.0, 40.0),
    pin_names=("u0/A", "u1/Y"),
    net_names=("n0",),
    node_name2id_map=None,
    pin2node_map=None,
    pin2net_map=None,
    flat_net2pin_map=None,
    flat_net2pin_start_map=None,
):
    if node_is_buffer is None:
        node_is_buffer = tuple(
            "BUF" in str(master_name).upper() for master_name in node_master_names
        )
    snapshot = types.SimpleNamespace(
        num_nodes=len(node_names),
        node_names=list(node_names),
        node_size_x=list(node_size_x),
        node_size_y=list(node_size_y),
        node_orient=list(node_orient),
        node_x=list(node_x),
        node_y=list(node_y),
        node_is_buffer=list(node_is_buffer),
        pin_names=list(pin_names),
        net_names=list(net_names),
        node_name2id_map=(
            node_name2id_map
            if node_name2id_map is not None
            else {name: node_id for node_id, name in enumerate(node_names)}
        ),
    )
    if include_node_master_names:
        snapshot.node_master_names = list(node_master_names)
    optional_attrs = {
        "pin2node_map": pin2node_map,
        "pin2net_map": pin2net_map,
        "flat_net2pin_map": flat_net2pin_map,
        "flat_net2pin_start_map": flat_net2pin_start_map,
    }
    for attr_name, attr_value in optional_attrs.items():
        if attr_value is not None:
            setattr(snapshot, attr_name, attr_value)
    return snapshot


class OpenRoadHandoffControllerTest(unittest.TestCase):
    def test_interval_trigger_returns_structured_decision(self):
        controller = OpenRoadHandoffController(
            {"enabled": True, "trigger": {"mode": "interval", "interval": 10}}
        )
        accepted, decision = controller.should_handoff(
            {"absolute_iteration": 10, "iteration": 10, "phase": 0}
        )
        self.assertTrue(accepted)
        self.assertEqual(
            decision,
            {"trigger_mode": "interval", "trigger_reason": "interval=10@iter=10"},
        )

    def test_interval_trigger_falls_back_to_legacy_iteration(self):
        controller = OpenRoadHandoffController(
            {"enabled": True, "trigger": {"mode": "interval", "interval": 5}}
        )
        accepted, decision = controller.should_handoff({"iteration": 15})
        self.assertTrue(accepted)
        self.assertEqual(
            decision,
            {"trigger_mode": "interval", "trigger_reason": "interval=5@iter=15"},
        )

    def test_repeated_matching_events_are_still_accepted_without_dedupe(self):
        controller = OpenRoadHandoffController(
            {"enabled": True, "trigger": {"mode": "interval", "interval": 10}}
        )
        event = {"absolute_iteration": 20, "iteration": 20, "phase": 0}

        first_accepted, first_decision = controller.should_handoff(event)
        second_accepted, second_decision = controller.should_handoff(event)

        self.assertTrue(first_accepted)
        self.assertTrue(second_accepted)
        self.assertEqual(
            first_decision,
            {"trigger_mode": "interval", "trigger_reason": "interval=10@iter=20"},
        )
        self.assertEqual(first_decision, second_decision)

    def test_phase_trigger_matches_session_phase_and_iteration_in_stage(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "phase",
                    "phases": [{"phase": 2, "at_stage_start": True}],
                },
            }
        )
        accepted, decision = controller.should_handoff(
            {"phase": 2, "iteration_in_stage": 0}
        )
        self.assertTrue(accepted)
        self.assertEqual(
            decision,
            {"trigger_mode": "phase", "trigger_reason": "phase=2@start"},
        )

    def test_metric_trigger_returns_structured_decision(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "metric",
                    "metrics": [{"name": "overflow", "op": ">=", "value": 0.25}],
                },
            }
        )
        accepted, decision = controller.should_handoff(
            {"metric": type("Metric", (), {"overflow": 0.25})()}
        )
        self.assertTrue(accepted)
        self.assertEqual(
            decision,
            {"trigger_mode": "metric", "trigger_reason": "overflow>=0.25"},
        )

    def test_composite_trigger_returns_structured_decision_for_metric_rule(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "composite",
                    "phases": [{"phase": 3, "at_stage_start": True}],
                    "metrics": [{"name": "objective", "op": "<=", "value": 10.0}],
                },
            }
        )
        accepted, decision = controller.should_handoff(
            {
                "phase": 1,
                "iteration_in_stage": 0,
                "metric": type("Metric", (), {"objective": 9.5})(),
            }
        )
        self.assertTrue(accepted)
        self.assertEqual(
            decision,
            {"trigger_mode": "metric", "trigger_reason": "objective<=10.0"},
        )

    def test_composite_trigger_returns_structured_decision_for_phase_rule(self):
        controller = OpenRoadHandoffController(
            {
                "enabled": True,
                "trigger": {
                    "mode": "composite",
                    "phases": [{"phase": 2, "at_stage_start": True}],
                    "metrics": [{"name": "overflow", "op": ">=", "value": 0.5}],
                },
            }
        )
        accepted, decision = controller.should_handoff(
            {"phase": 2, "iteration_in_stage": 0}
        )
        self.assertTrue(accepted)
        self.assertEqual(
            decision,
            {"trigger_mode": "phase", "trigger_reason": "phase=2@start"},
        )

    def test_build_handoff_event_normalizes_legacy_stage_and_iteration(self):
        controller = OpenRoadHandoffController(
            {"enabled": True, "trigger": {"mode": "interval", "interval": 10}}
        )
        event = {
            "stage": 0,
            "iteration": 10,
            "metric": type(
                "Metric",
                (),
                {
                    "hpwl": 1.5,
                    "overflow": 0.25,
                    "objective": 9.0,
                    "wns": -0.11,
                    "tns": -1.2,
                    "ws": -0.09,
                    "ts": 0.0,
                    "max_slew_violation": 0.03,
                    "max_load_cap_violation": 0.04,
                },
            )(),
        }
        decision = {"trigger_mode": "interval", "trigger_reason": "interval=10@iter=10"}

        handoff_event = controller.build_handoff_event(event, decision)

        self.assertEqual(handoff_event["trigger_mode"], "interval")
        self.assertEqual(handoff_event["trigger_reason"], "interval=10@iter=10")
        self.assertEqual(handoff_event["reason"], "interval=10@iter=10")
        self.assertEqual(handoff_event["stage"], 0)
        self.assertEqual(handoff_event["iteration"], 10)
        self.assertEqual(handoff_event["phase"], 0)
        self.assertEqual(handoff_event["absolute_iteration"], 10)
        self.assertEqual(
            handoff_event["metric_snapshot"],
            {
                "hpwl": 1.5,
                "overflow": 0.25,
                "objective": 9.0,
                "wns": -0.11,
                "tns": -1.2,
                "ws": -0.09,
                "ts": 0.0,
                "max_slew_violation": 0.03,
                "max_load_cap_violation": 0.04,
            },
        )

    def test_build_handoff_event_backfills_legacy_stage_and_iteration_from_session_fields(self):
        controller = OpenRoadHandoffController(
            {"enabled": True, "trigger": {"mode": "interval", "interval": 10}}
        )
        event = {
            "phase": 3,
            "absolute_iteration": 20,
            "metric": type(
                "Metric",
                (),
                {
                    "hpwl": 2.5,
                    "overflow": 0.5,
                    "objective": 8.0,
                    "wns": -0.2,
                    "tns": -2.5,
                    "ws": -0.15,
                    "ts": 0.0,
                    "max_slew_violation": 0.05,
                    "max_load_cap_violation": 0.06,
                },
            )(),
        }
        decision = {"trigger_mode": "interval", "trigger_reason": "interval=10@iter=20"}

        handoff_event = controller.build_handoff_event(event, decision)

        self.assertEqual(handoff_event["phase"], 3)
        self.assertEqual(handoff_event["absolute_iteration"], 20)
        self.assertEqual(handoff_event["stage"], 3)
        self.assertEqual(handoff_event["iteration"], 20)
        self.assertEqual(
            handoff_event["metric_snapshot"],
            {
                "hpwl": 2.5,
                "overflow": 0.5,
                "objective": 8.0,
                "wns": -0.2,
                "tns": -2.5,
                "ws": -0.15,
                "ts": 0.0,
                "max_slew_violation": 0.05,
                "max_load_cap_violation": 0.06,
            },
        )


class MacroPlaceDBOpenRoadHandoffFlowTest(unittest.TestCase):
    def _build_contract(self, old_pydb, new_pydb):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        return placedb._build_topology_sync_contract(old_pydb, new_pydb)

    def _snapshot_fingerprint(self, pydb):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.pydb = pydb
        counts = placedb._build_handoff_counts_summary()
        return placedb._build_handoff_snapshot_fingerprint(counts)

    def test_openroad_pydb_export_exposes_snapshot_node_metadata(self):
        extension = _load_openroad_cpp_extension_or_none()
        if extension is not None:
            pydb = extension.PyPlaceDB()
            for field_name in (
                "node_names",
                "node_master_names",
                "node_is_buffer",
                "node_x",
                "node_y",
                "node_orient",
                "node_size_x",
                "node_size_y",
            ):
                self.assertTrue(hasattr(pydb, field_name))
            return

        # Unit jobs in this repo often run without the compiled OpenROAD
        # extension.  Fall back to a comment-stripped structural source check.
        _assert_openroad_cpp_source_exports_snapshot_node_metadata(self)

    def test_openroad_pydb_export_exposes_libcell_names(self):
        extension = _load_openroad_cpp_extension_or_none()
        if extension is not None:
            pydb = extension.PyPlaceDB()
            self.assertTrue(hasattr(pydb, "flat_libcell_names"))
            return

        # Unit jobs in this repo often run without the compiled OpenROAD
        # extension.  Fall back to a comment-stripped structural source check.
        _assert_openroad_cpp_source_exports_libcell_names(self)

    def test_openroad_pydb_export_exposes_independent_size_and_vt_axes(self):
        _assert_openroad_cpp_source_exports_size_and_vt_axes(self)

    def test_run_buffer_insertion_rebuilds_node_index_after_successful_tcl(self):
        _assert_openroad_cpp_source_rebuilds_node_index_after_buffer_insertion(self)

    def test_one_net_buffer_python_wrapper_forwards_to_raw_db(self):
        place_io = _load_place_io_module_with_cpp_stub()

        class RawDB:
            def __init__(self):
                self.calls = []

            def run_one_net_buffer(self, net_name, config):
                self.calls.append((net_name, dict(config)))
                return {
                    "artifact": "buffer_insertion_one_net_result",
                    "artifact_version": 1,
                    "net_name": net_name,
                    "status": "unsupported",
                }

        raw_db = RawDB()
        result = place_io.PlaceIOFunction.run_one_net_buffer(
            raw_db,
            "net_a",
            {"max_loop_nets": 1},
        )

        self.assertEqual([("net_a", {"max_loop_nets": 1})], raw_db.calls)
        self.assertEqual("unsupported", result["status"])

    def test_buffer_insertion_command_keeps_repair_design_default_options(self):
        place_io = _load_place_io_module_with_cpp_stub()

        command = place_io.PlaceIOFunction.build_buffer_insertion_command(
            "repair_design",
            {"max_wire_length": 10},
        )

        self.assertEqual(command, "repair_design -max_wire_length {10}")

    def test_buffer_insertion_command_expands_experimental_buffer_only_alias(self):
        place_io = _load_place_io_module_with_cpp_stub()

        command = place_io.PlaceIOFunction.build_buffer_insertion_command(
            "buffer-only"
        )

        self.assertEqual(
            command,
            'repair_timing -setup -sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap",
        )

    def test_buffer_insertion_command_ignores_options_for_experimental_buffer_only_alias(self):
        place_io = _load_place_io_module_with_cpp_stub()

        command = place_io.PlaceIOFunction.build_buffer_insertion_command(
            "buffer-only",
            {"max_wire_length": 10},
        )

        self.assertEqual(
            command,
            'repair_timing -setup -sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap",
        )
        self.assertNotIn("-max_wire_length", command)

    def test_buffer_insertion_command_runs_pre_repair_tcl_before_experimental_alias(self):
        place_io = _load_place_io_module_with_cpp_stub()

        command = place_io.PlaceIOFunction.build_buffer_insertion_command(
            "buffer-only",
            {
                "pre_repair_tcl": [
                    "set_wire_rc -signal -layer MET3",
                    "estimate_parasitics -placement",
                ],
                "max_wire_length": 10,
            },
        )

        self.assertEqual(
            command,
            "set_wire_rc -signal -layer MET3\n"
            "estimate_parasitics -placement\n"
            'repair_timing -setup -sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap",
        )
        self.assertNotIn("-max_wire_length", command)

    def test_buffer_insertion_command_normalizes_experimental_buffer_only_aliases(self):
        place_io = _load_place_io_module_with_cpp_stub()
        expected_command = (
            'repair_timing -setup -sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap"
        )

        for strategy in ("Buffer-Only", "BUFFER_ONLY", "buffer only"):
            with self.subTest(strategy=strategy):
                profile = place_io.PlaceIOFunction.describe_buffer_insertion_strategy(
                    strategy
                )
                command = place_io.PlaceIOFunction.build_buffer_insertion_command(
                    strategy
                )

                self.assertEqual(profile["profile"], "buffer-only")
                self.assertEqual(profile["kind"], "experimental")
                self.assertTrue(profile["experimental"])
                self.assertEqual(command, expected_command)

    def test_buffer_insertion_command_preserves_unknown_strategy_as_custom_tcl(self):
        place_io = _load_place_io_module_with_cpp_stub()

        profile = place_io.PlaceIOFunction.describe_buffer_insertion_strategy(
            "repair_timing -setup"
        )
        command = place_io.PlaceIOFunction.build_buffer_insertion_command(
            "repair_timing -setup"
        )

        self.assertEqual(profile["profile"], "repair_timing -setup")
        self.assertEqual(profile["kind"], "custom")
        self.assertEqual(command, "repair_timing -setup")

    def test_openroad_sizing_writeback_uses_bridge_apply_sizing(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.params = types.SimpleNamespace(place_io_engine="openroad")
        placedb.inst_cell_id = np.array([1, -1, 0], dtype=np.int32)
        placedb.flat_libcell_names = np.array(
            [b"INV_X1", b"INV_X2"], dtype=np.bytes_
        )

        class FakePlaceIOFunction:
            @staticmethod
            def apply_sizing(raw_db, cell_ids, cell_master_names):
                calls.append((raw_db, list(cell_ids), list(cell_master_names)))
                return {"applied": 2, "skipped": 1, "missing_masters": 0}

        calls = []
        fake_place_io = types.ModuleType("dreamplace.ops.placeio_openroad.place_io")
        fake_place_io.PlaceIOFunction = FakePlaceIOFunction
        fake_package = types.ModuleType("dreamplace.ops.placeio_openroad")
        fake_package.__path__ = []
        fake_package.place_io = fake_place_io
        fake_ops = types.ModuleType("dreamplace.ops")
        fake_ops.__path__ = []
        fake_ops.placeio_openroad = fake_package
        placedb.openroad_bridge = object()

        previous_modules = {
            name: sys.modules.get(name)
            for name in (
                "dreamplace.ops",
                "dreamplace.ops.placeio_openroad",
                "dreamplace.ops.placeio_openroad.place_io",
            )
        }
        import dreamplace

        previous_ops_attr = getattr(dreamplace, "ops", None)
        previous_had_ops_attr = hasattr(dreamplace, "ops")
        dreamplace.ops = fake_ops
        sys.modules["dreamplace.ops"] = fake_ops
        sys.modules["dreamplace.ops.placeio_openroad"] = fake_package
        sys.modules["dreamplace.ops.placeio_openroad.place_io"] = fake_place_io
        try:
            result = placedb.write_sizing_back()
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            if previous_had_ops_attr:
                dreamplace.ops = previous_ops_attr
            else:
                delattr(dreamplace, "ops")

        self.assertEqual(result["applied"], 2)
        self.assertEqual(placedb.last_sizing_writeback_summary, result)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], placedb.openroad_bridge)
        self.assertEqual(calls[0][1], [1, -1, 0])
        self.assertEqual(calls[0][2], ["INV_X1", "INV_X2"])

    def test_openroad_aimp_db_summary_records_macroplacedb_counts(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.num_physical_nodes = 5
        placedb.num_terminals = 2
        placedb.num_terminal_NIs = 0
        placedb.pin_names = np.array([b"p0", b"p1", b"p2"])
        placedb.net_names = np.array([b"n0", b"n1"])
        placedb.flat_libcell_info = np.zeros((4, 4), dtype=np.float32)
        placedb.flat_libarc_info = np.zeros((7, 6), dtype=np.int32)
        placedb.flat_lib_pin_cap = np.array([1.0, 0.0, 2.0])
        placedb.flat_lib_pin_rcap = np.array([0.1, 0.0, 0.2])
        placedb.flat_lib_pin_fcap = np.array([0.1, 0.0, 0.2])
        placedb.flat_lib_pin_cap_limit = np.array([3.0, 0.0, 0.0])
        placedb.flat_lib_pin_slew_limit = np.array([4.0, 5.0, 0.0])
        placedb.flat_inst_arcs_by_level = np.arange(8, dtype=np.int32)
        placedb.flat_inst_arcs_by_level_start = np.array([0, 8], dtype=np.int32)
        placedb.main_id_2_cell_id_start = np.array([0, 2, 4], dtype=np.int32)
        placedb.inst_is_sizeable = np.array([True, False, True, True, False])

        with tempfile.TemporaryDirectory() as tmpdir:
            params = types.SimpleNamespace(
                place_io_engine="openroad",
                result_dir=tmpdir,
                design_name=lambda: "unit_design",
                openroad_aimp_debug_artifacts=False,
                design_inputs={
                    "def": "/tmp/in.def",
                    "sdc": "/tmp/in.sdc",
                    "rc_tcl": "/tmp/setRC.tcl",
                },
            )

            path = placedb._write_openroad_aimp_db_summary(params)
            summary = __import__("json").loads(open(path, encoding="utf-8").read())

        self.assertEqual(summary["aimp_db_source"], "openroad")
        self.assertTrue(summary["openroad_backed_aimp_db"])
        self.assertEqual(summary["counts"]["instances"], 5)
        self.assertEqual(summary["counts"]["movable_instances"], 3)
        self.assertEqual(summary["counts"]["libcells"], 4)
        self.assertEqual(summary["counts"]["libarcs"], 7)
        self.assertEqual(summary["counts"]["libpins"], 3)
        self.assertEqual(summary["counts"]["timing_edges"], 8)
        self.assertEqual(summary["counts"]["equiv_classes"], 2)
        self.assertEqual(summary["counts"]["sizeable_instances"], 3)
        self.assertEqual(summary["coverage"]["lib_pins_with_cap"], 2)
        self.assertEqual(summary["coverage"]["lib_pins_with_cap_limit"], 1)
        self.assertEqual(summary["coverage"]["lib_pins_with_slew_limit"], 2)
        self.assertEqual(summary["paths"]["sdc"], "/tmp/in.sdc")
        self.assertEqual(summary["paths"]["rc_tcl"], "/tmp/setRC.tcl")
        self.assertEqual(summary["parasitics"]["initialization"], "placement")
        self.assertTrue(summary["parasitics"]["rc_tcl_configured"])

    def test_run_openroad_buffer_insertion_records_experimental_buffer_only_metadata(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        params = types.SimpleNamespace(
            place_io_engine="openroad",
            openroad_handoff={
                "buffer_insertion": {
                    "enabled": True,
                    "strategy": "buffer-only",
                    "options": {"max_wire_length": 10},
                },
            },
        )

        class FakeOpenRoadBridge:
            def sync_to_openroad(self, node_x, node_y):
                calls.append(("sync", list(node_x), list(node_y)))

            def run_buffer_insertion(self, command):
                calls.append(("buffer", command))
                return "OpenROAD output"

        placedb.openroad_bridge = FakeOpenRoadBridge()
        placedb._extract_unscaled_movable_positions = lambda pos: (
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
        )
        placedb.synchronize_topology_from_openroad = lambda params, pos=None: {
            "mutation_kind": "topology_changed",
            "requires_runtimedb_rebuild": True,
            "new_counts": {"nodes": 2, "pins": 3, "nets": 1},
            "added_buffer_count": 1,
            "removed_buffer_count": 0,
            "surviving_buffer_count": 0,
        }

        experimental_command = (
            'repair_timing -setup -sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap"
        )
        calls = []
        placeio_module = _load_place_io_module_with_cpp_stub()
        ops_package = types.ModuleType("dreamplace.ops")
        ops_package.__path__ = []
        placeio_package = types.ModuleType("dreamplace.ops.placeio_openroad")
        placeio_package.__path__ = []
        placeio_package.place_io = placeio_module
        previous_modules = {
            name: sys.modules.get(name)
            for name in (
                "dreamplace.ops",
                "dreamplace.ops.placeio_openroad",
                "dreamplace.ops.placeio_openroad.place_io",
            )
        }
        import dreamplace

        previous_ops_attr = getattr(dreamplace, "ops", None)
        previous_had_ops_attr = hasattr(dreamplace, "ops")
        ops_package.placeio_openroad = placeio_package
        dreamplace.ops = ops_package
        sys.modules["dreamplace.ops"] = ops_package
        sys.modules["dreamplace.ops.placeio_openroad"] = placeio_package
        sys.modules["dreamplace.ops.placeio_openroad.place_io"] = placeio_module
        try:
            result = placedb.run_openroad_buffer_insertion(
                params,
                np.array([1.0, 2.0], dtype=np.float64),
                {"trigger_reason": "unit"},
            )
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            if previous_had_ops_attr:
                dreamplace.ops = previous_ops_attr
            else:
                delattr(dreamplace, "ops")

        self.assertEqual(
            calls,
            [
                ("sync", [1.0], [2.0]),
                ("buffer", experimental_command),
            ],
        )
        self.assertTrue(result["executed"])
        self.assertEqual(result["buffer_insertion_strategy"], "buffer-only")
        self.assertEqual(result["buffer_insertion_command"], experimental_command)
        self.assertEqual(result["buffer_insertion_strategy_profile"], "buffer-only")
        self.assertEqual(result["buffer_insertion_strategy_kind"], "experimental")
        self.assertTrue(result["buffer_insertion_strategy_experimental"])
        self.assertEqual(
            result["buffer_insertion_strategy_validation"],
            "log_validated_only",
        )
        self.assertIs(
            result.get("buffer_insertion_strategy_non_buffer_mutation_free"),
            None,
        )

    def test_topology_sync_contract_classifies_unchanged_snapshot_as_no_mutation(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot()

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "no_mutation")
        self.assertFalse(contract["requires_runtimedb_rebuild"])

    def test_topology_sync_contract_classifies_coordinate_only_change_as_placement_only(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot(node_x=(11.0, 20.0), node_y=(30.0, 41.0))

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "placement_only")
        self.assertFalse(contract["requires_runtimedb_rebuild"])

    def test_topology_sync_contract_classifies_surviving_master_change_as_geometry_changed(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot(
            node_master_names=("BUF_X2", "INV_X1"),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.5, 2.5),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "geometry_changed")
        self.assertTrue(contract["requires_runtimedb_rebuild"])

    def test_snapshot_fingerprint_changes_when_node_master_changes(self):
        baseline = _make_topology_snapshot()
        changed = _make_topology_snapshot(
            node_master_names=("BUF_X2", "INV_X1"),
        )

        self.assertNotEqual(
            self._snapshot_fingerprint(baseline),
            self._snapshot_fingerprint(changed),
        )

    def test_snapshot_fingerprint_changes_when_node_size_changes(self):
        baseline = _make_topology_snapshot()
        changed = _make_topology_snapshot(node_size_x=(1.25, 2.0))

        self.assertNotEqual(
            self._snapshot_fingerprint(baseline),
            self._snapshot_fingerprint(changed),
        )

    def test_snapshot_fingerprint_changes_when_node_orient_changes(self):
        baseline = _make_topology_snapshot()
        changed = _make_topology_snapshot(node_orient=("S", "N"))

        self.assertNotEqual(
            self._snapshot_fingerprint(baseline),
            self._snapshot_fingerprint(changed),
        )

    def test_snapshot_fingerprint_changes_when_connectivity_changes(self):
        baseline = _make_topology_snapshot(
            pin_names=("u0/A", "u1/Y"),
            net_names=("n0", "n1"),
            pin2node_map=np.array([0, 1], dtype=np.int32),
            pin2net_map=np.array([0, 1], dtype=np.int32),
            flat_net2pin_map=np.array([0, 1], dtype=np.int32),
            flat_net2pin_start_map=np.array([0, 1, 2], dtype=np.int32),
        )
        changed = _make_topology_snapshot(
            pin_names=("u0/A", "u1/Y"),
            net_names=("n0", "n1"),
            pin2node_map=np.array([1, 0], dtype=np.int32),
            pin2net_map=np.array([0, 1], dtype=np.int32),
            flat_net2pin_map=np.array([0, 1], dtype=np.int32),
            flat_net2pin_start_map=np.array([0, 1, 2], dtype=np.int32),
        )

        self.assertNotEqual(
            self._snapshot_fingerprint(baseline),
            self._snapshot_fingerprint(changed),
        )

    def test_topology_sync_contract_classifies_added_instance_as_topology_changed(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot(
            node_names=("u0", "u1", "u2"),
            node_master_names=("BUF_X1", "INV_X1", "BUF_X1"),
            node_size_x=(1.0, 2.0, 1.0),
            node_size_y=(1.5, 2.5, 1.5),
            node_orient=("N", "N", "N"),
            node_x=(10.0, 20.0, 25.0),
            node_y=(30.0, 40.0, 45.0),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertTrue(contract["requires_runtimedb_rebuild"])
        self.assertEqual(contract["added_names"]["nodes"], ["u2"])

    def test_topology_sync_contract_classifies_removed_instance_as_topology_changed(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot(
            node_names=("u0",),
            node_master_names=("BUF_X1",),
            node_size_x=(1.0,),
            node_size_y=(1.5,),
            node_orient=("N",),
            node_x=(10.0,),
            node_y=(30.0,),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertTrue(contract["requires_runtimedb_rebuild"])
        self.assertEqual(contract["removed_names"]["nodes"], ["u1"])

    def test_topology_sync_contract_preserves_placement_only_for_logically_equal_reordered_connectivity(self):
        old_pydb = _make_topology_snapshot(
            node_names=("u0", "u1"),
            node_master_names=("BUF_X1", "INV_X1"),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.5, 2.5),
            node_orient=("N", "S"),
            node_x=(10.0, 20.0),
            node_y=(30.0, 40.0),
            pin_names=("u0/A", "u1/Y", "u0/Z"),
            net_names=("n0", "n1"),
            pin2node_map=np.array([0, 1, 0], dtype=np.int32),
            pin2net_map=np.array([0, 0, 1], dtype=np.int32),
            flat_net2pin_map=np.array([0, 1, 2], dtype=np.int32),
            flat_net2pin_start_map=np.array([0, 2, 3], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("u1", "u0"),
            node_master_names=("INV_X1", "BUF_X1"),
            node_size_x=(2.0, 1.0),
            node_size_y=(2.5, 1.5),
            node_orient=("S", "N"),
            node_x=(20.0, 11.0),
            node_y=(40.0, 30.0),
            pin_names=("u0/Z", "u1/Y", "u0/A"),
            net_names=("n1", "n0"),
            pin2node_map=np.array([1, 0, 1], dtype=np.int32),
            pin2net_map=np.array([0, 1, 1], dtype=np.int32),
            flat_net2pin_map=np.array([0, 1, 2], dtype=np.int32),
            flat_net2pin_start_map=np.array([0, 1, 3], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "placement_only")
        self.assertFalse(contract["requires_runtimedb_rebuild"])

    def test_topology_sync_contract_uses_raw_connectivity_fallback_when_pin_names_are_missing(self):
        old_pydb = _make_topology_snapshot(
            pin_names=(),
            net_names=(),
            pin2node_map=np.array([0], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            pin_names=(),
            net_names=(),
            pin2node_map=np.array([1], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertTrue(contract["requires_runtimedb_rebuild"])

    def test_topology_sync_contract_accepts_equivalent_list_tuple_and_numpy_snapshot_values(self):
        old_pydb = _make_topology_snapshot(
            node_size_x=(np.float64(1.0), 2.0),
            node_size_y=(1.5, np.float64(2.5)),
            node_orient=(b"N", b"N"),
            node_x=(np.float64(10.0), 20.0),
            node_y=(30.0, np.float64(40.0)),
            pin2node_map=[0, 1],
            pin2net_map=[0, 0],
            flat_net2pin_map=[0, 1],
            flat_net2pin_start_map=[0, 2],
        )
        new_pydb = _make_topology_snapshot(
            node_size_x=np.array([1.0, 2.0], dtype=np.float64),
            node_size_y=np.array([1.5, 2.5], dtype=np.float64),
            node_orient=np.array([b"N", b"N"]),
            node_x=np.array([10.0, 20.0], dtype=np.float64),
            node_y=np.array([30.0, 40.0], dtype=np.float64),
            pin2node_map=np.array([0, 1], dtype=np.int32),
            pin2net_map=np.array([0, 0], dtype=np.int32),
            flat_net2pin_map=np.array([0, 1], dtype=np.int32),
            flat_net2pin_start_map=np.array([0, 2], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "no_mutation")
        self.assertFalse(contract["requires_runtimedb_rebuild"])

    def test_topology_sync_contract_uses_bytes_keys_in_new_node_name_map(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot(
            node_names=("u1", "u0"),
            node_master_names=("INV_X1", "BUF_X1"),
            node_size_x=(2.0, 1.0),
            node_size_y=(2.5, 1.5),
            node_x=(20.0, 10.0),
            node_y=(40.0, 30.0),
            node_name2id_map={b"u1": 0, b"u0": 1},
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "no_mutation")
        self.assertEqual(contract["surviving_name_to_new_id"], {"u0": 1, "u1": 0})

    def test_topology_sync_contract_handles_missing_master_names_with_available_geometry(self):
        old_pydb = _make_topology_snapshot(include_node_master_names=False)
        new_pydb = _make_topology_snapshot(
            include_node_master_names=False,
            node_size_x=(1.0, 2.25),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "geometry_changed")
        self.assertTrue(contract["requires_runtimedb_rebuild"])

    def test_topology_sync_contract_counts_unbuffer_only_buffer_churn(self):
        old_pydb = _make_topology_snapshot(
            node_names=("buf0", "logic0"),
            node_master_names=("BUF_X1", "NAND_X1"),
            node_is_buffer=(True, False),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.0, 2.0),
            node_orient=("N", "N"),
            node_x=(10.0, 20.0),
            node_y=(30.0, 40.0),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("logic0",),
            node_master_names=("NAND_X1",),
            node_is_buffer=(False,),
            node_size_x=(2.0,),
            node_size_y=(2.0,),
            node_orient=("N",),
            node_x=(20.0,),
            node_y=(40.0,),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(contract["removed_names"]["nodes"], ["buf0"])
        self.assertEqual(contract["added_buffer_count"], 0)
        self.assertEqual(contract["removed_buffer_count"], 1)
        self.assertEqual(contract["surviving_buffer_count"], 0)
        self.assertEqual(contract["surviving_name_to_new_id"], {"logic0": 0})

    def test_topology_sync_contract_counts_rebuffer_as_removed_and_added_buffers(self):
        old_pydb = _make_topology_snapshot(
            node_names=("buf_old", "logic0"),
            node_master_names=("BUF_X1", "NAND_X1"),
            node_is_buffer=(True, False),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.0, 2.0),
            node_orient=("N", "N"),
            node_x=(10.0, 20.0),
            node_y=(30.0, 40.0),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("logic0", "buf_new"),
            node_master_names=("NAND_X1", "BUF_X2"),
            node_is_buffer=(False, True),
            node_size_x=(2.0, 1.0),
            node_size_y=(2.0, 1.0),
            node_orient=("N", "N"),
            node_x=(20.0, 25.0),
            node_y=(40.0, 45.0),
        )

        contract = self._build_contract(old_pydb, new_pydb)

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(contract["removed_names"]["nodes"], ["buf_old"])
        self.assertEqual(contract["added_names"]["nodes"], ["buf_new"])
        self.assertEqual(contract["added_buffer_count"], 1)
        self.assertEqual(contract["removed_buffer_count"], 1)
        self.assertEqual(contract["surviving_buffer_count"], 0)
        self.assertEqual(contract["surviving_name_to_new_id"], {"logic0": 0})

    def test_buffer_only_policy_reports_non_buffer_master_change_as_violation(self):
        old_pydb = _make_topology_snapshot(
            node_names=("u0", "u1"),
            node_master_names=("NAND_X1", "INV_X1"),
            node_is_buffer=(False, False),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.5, 2.5),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("u0", "u1"),
            node_master_names=("NAND_X4", "INV_X1"),
            node_is_buffer=(False, False),
            node_size_x=(1.75, 2.0),
            node_size_y=(1.5, 2.5),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "geometry_changed")
        self.assertTrue(contract["requires_runtimedb_rebuild"])
        self.assertEqual(policy["status"], "violated")
        self.assertEqual(
            policy["violation_counts"]["non_buffer_master_change_count"], 1
        )
        self.assertEqual(
            policy["violation_counts"]["non_buffer_size_change_count"], 1
        )
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 2)
        self.assertEqual(
            policy["summary"]["non_buffer_changed_instance_count"], 1
        )
        self.assertEqual(policy["violation_samples"][0]["node_name"], "u0")
        self.assertEqual(
            set(policy["violation_samples"][0]["change_kinds"]),
            {"master", "size"},
        )

    def test_buffer_only_policy_accepts_surviving_buffer_resize(self):
        old_pydb = _make_topology_snapshot(
            node_names=("u0", "u1"),
            node_master_names=("BUF_X1", "INV_X1"),
            node_is_buffer=(True, False),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.5, 2.5),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("u0", "u1"),
            node_master_names=("BUF_X4", "INV_X1"),
            node_is_buffer=(True, False),
            node_size_x=(2.0, 2.0),
            node_size_y=(1.5, 2.5),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "geometry_changed")
        self.assertEqual(policy["status"], "clean")
        self.assertEqual(policy["allowed_change_counts"]["resized_buffer_count"], 1)
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 0)
        self.assertEqual(policy["summary"]["buffer_changed_instance_count"], 1)

    def test_buffer_only_policy_reports_added_and_removed_non_buffers(self):
        old_pydb = _make_topology_snapshot(
            node_names=("u0", "old_logic"),
            node_master_names=("BUF_X1", "NAND_X1"),
            node_is_buffer=(True, False),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.5, 2.5),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("u0", "new_logic"),
            node_master_names=("BUF_X1", "NOR_X1"),
            node_is_buffer=(True, False),
            node_size_x=(1.0, 2.0),
            node_size_y=(1.5, 2.5),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(policy["status"], "violated")
        self.assertEqual(policy["violation_counts"]["added_non_buffer_count"], 1)
        self.assertEqual(policy["violation_counts"]["removed_non_buffer_count"], 1)
        self.assertEqual(policy["summary"]["non_buffer_changed_instance_count"], 2)
        sample_kinds = {
            tuple(sample["change_kinds"]): sample["node_name"]
            for sample in policy["violation_samples"]
        }
        self.assertEqual(sample_kinds[("added_non_buffer",)], "new_logic")
        self.assertEqual(sample_kinds[("removed_non_buffer",)], "old_logic")

    def test_buffer_only_policy_flags_mixed_buffer_add_and_connectivity_change(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "other"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1"),
            node_is_buffer=(False, False, False),
            node_size_x=(1.0, 2.0, 3.0),
            node_size_y=(1.5, 2.5, 3.5),
            node_orient=("N", "N", "N"),
            node_x=(10.0, 20.0, 30.0),
            node_y=(30.0, 40.0, 50.0),
            pin_names=("driver/Y", "sink/A", "other/A"),
            net_names=("n0", "n1", "n2", "n3"),
            pin2node_map=np.array([0, 1, 2], dtype=np.int32),
            pin2net_map=np.array([0, 0, 2], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "other", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1", "BUF_X1"),
            node_is_buffer=(False, False, False, True),
            node_size_x=(1.0, 2.0, 3.0, 1.0),
            node_size_y=(1.5, 2.5, 3.5, 1.5),
            node_orient=("N", "N", "N", "N"),
            node_x=(10.0, 20.0, 30.0, 40.0),
            node_y=(30.0, 40.0, 50.0, 60.0),
            pin_names=("driver/Y", "sink/A", "other/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "n1", "n2", "n3"),
            pin2node_map=np.array([0, 1, 2, 3, 3], dtype=np.int32),
            pin2net_map=np.array([0, 1, 3, 0, 1], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(policy["allowed_change_counts"]["added_buffer_count"], 1)
        self.assertEqual(policy["status"], "violated")
        self.assertEqual(
            policy["violation_counts"]["connectivity_violation_count"], 1
        )
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 1)
        self.assertEqual(len(policy["violation_samples"]), 1)
        self.assertEqual(
            policy["violation_samples"][0],
            {
                "change_kinds": ["connectivity"],
                "reason": "non_buffer_pin_net_changed",
                "pin_name": "other/A",
                "old_node_name": "other",
                "new_node_name": "other",
                "old_net_name": "n2",
                "new_net_name": "n3",
                "old_node_is_buffer": False,
                "new_node_is_buffer": False,
                "old_net_is_buffer_churn_affected": False,
                "new_net_is_buffer_churn_affected": False,
                "affected_net_names_sample": ["n0", "n1"],
                "affected_net_count": 2,
                "affected_net_source_samples": [
                    {
                        "source": "added_buffer_pin",
                        "pin_name": "buf0/A",
                        "node_name": "buf0",
                        "net_name": "n0",
                    },
                    {
                        "source": "added_buffer_pin",
                        "pin_name": "buf0/Y",
                        "node_name": "buf0",
                        "net_name": "n1",
                    },
                ],
            },
        )

    def test_buffer_only_policy_reports_unknown_when_connectivity_metadata_is_missing(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink"),
            node_master_names=("NAND_X1", "INV_X1"),
            node_is_buffer=(False, False),
            pin_names=("driver/Y", "sink/A"),
            net_names=("n0",),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "BUF_X1"),
            node_is_buffer=(False, False, True),
            node_size_x=(1.0, 2.0, 1.0),
            node_size_y=(1.5, 2.5, 1.5),
            node_orient=("N", "N", "N"),
            node_x=(10.0, 20.0, 30.0),
            node_y=(30.0, 40.0, 50.0),
            pin_names=("driver/Y", "sink/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "n1"),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]
        unknown_reasons = {
            unknown.get("reason") for unknown in policy["unknown_reasons"]
        }

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(policy["status"], "unknown")
        self.assertEqual(policy["allowed_change_counts"]["added_buffer_count"], 1)
        self.assertIn("unattributed_connectivity_change", unknown_reasons)

    def test_buffer_only_policy_keeps_connectivity_unknown_when_added_status_missing(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink"),
            node_master_names=("NAND_X1", "INV_X1"),
            node_is_buffer=(False, False),
            pin_names=("driver/Y", "sink/A"),
            net_names=("n0",),
            pin2node_map=np.array([0, 1], dtype=np.int32),
            pin2net_map=np.array([0, 0], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "maybe_buf"),
            node_master_names=("NAND_X1", "INV_X1", "BUF_X1"),
            node_is_buffer=(False, False, "unknown"),
            node_size_x=(1.0, 2.0, 1.0),
            node_size_y=(1.5, 2.5, 1.5),
            node_orient=("N", "N", "N"),
            node_x=(10.0, 20.0, 30.0),
            node_y=(30.0, 40.0, 50.0),
            pin_names=("driver/Y", "sink/A", "maybe_buf/A", "maybe_buf/Y"),
            net_names=("n0", "n1"),
            pin2node_map=np.array([0, 1, 2, 2], dtype=np.int32),
            pin2net_map=np.array([0, 1, 0, 1], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]
        unknown_reasons = {
            unknown.get("reason") for unknown in policy["unknown_reasons"]
        }

        self.assertEqual(policy["status"], "unknown")
        self.assertEqual(policy["violation_counts"]["connectivity_violation_count"], 0)
        self.assertIn("missing_added_node_buffer_status", unknown_reasons)
        self.assertIn("unattributed_connectivity_change", unknown_reasons)

    def test_buffer_only_policy_keeps_connectivity_unknown_for_unknown_survivor_status(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink"),
            node_master_names=("NAND_X1", "INV_X1"),
            node_is_buffer=(False, "unknown"),
            pin_names=("driver/Y", "sink/A"),
            net_names=("n0", "n1"),
            pin2node_map=np.array([0, 1], dtype=np.int32),
            pin2net_map=np.array([0, 0], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "BUF_X1"),
            node_is_buffer=(False, "unknown", True),
            node_size_x=(1.0, 2.0, 1.0),
            node_size_y=(1.5, 2.5, 1.5),
            node_orient=("N", "N", "N"),
            node_x=(10.0, 20.0, 30.0),
            node_y=(30.0, 40.0, 50.0),
            pin_names=("driver/Y", "sink/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "n1"),
            pin2node_map=np.array([0, 1, 2, 2], dtype=np.int32),
            pin2net_map=np.array([0, 1, 0, 1], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]
        unknown_reasons = {
            unknown.get("reason") for unknown in policy["unknown_reasons"]
        }

        self.assertEqual(policy["status"], "unknown")
        self.assertEqual(policy["violation_counts"]["connectivity_violation_count"], 0)
        self.assertIn("missing_surviving_node_buffer_status", unknown_reasons)
        self.assertIn("unattributed_connectivity_change", unknown_reasons)

    def test_buffer_only_policy_flags_one_sided_affected_net_change(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "other"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1"),
            node_is_buffer=(False, False, False),
            node_size_x=(1.0, 2.0, 3.0),
            node_size_y=(1.5, 2.5, 3.5),
            node_orient=("N", "N", "N"),
            node_x=(10.0, 20.0, 30.0),
            node_y=(30.0, 40.0, 50.0),
            pin_names=("driver/Y", "sink/A", "other/A"),
            net_names=("n0", "n1", "n2"),
            pin2node_map=np.array([0, 1, 2], dtype=np.int32),
            pin2net_map=np.array([0, 0, 2], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink", "other", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1", "BUF_X1"),
            node_is_buffer=(False, False, False, True),
            node_size_x=(1.0, 2.0, 3.0, 1.0),
            node_size_y=(1.5, 2.5, 3.5, 1.5),
            node_orient=("N", "N", "N", "N"),
            node_x=(10.0, 20.0, 30.0, 40.0),
            node_y=(30.0, 40.0, 50.0, 60.0),
            pin_names=("driver/Y", "sink/A", "other/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "n1", "n2"),
            pin2node_map=np.array([0, 1, 2, 3, 3], dtype=np.int32),
            pin2net_map=np.array([0, 1, 0, 0, 1], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(policy["status"], "violated")
        self.assertEqual(
            policy["violation_counts"]["connectivity_violation_count"], 1
        )

    def test_buffer_only_policy_accepts_surviving_buffer_rewire_split(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink0", "sink1", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1", "BUF_X1"),
            node_is_buffer=(False, False, False, True),
            node_size_x=(1.0, 2.0, 3.0, 1.0),
            node_size_y=(1.5, 2.5, 3.5, 1.5),
            node_orient=("N", "N", "N", "N"),
            node_x=(10.0, 20.0, 30.0, 40.0),
            node_y=(30.0, 40.0, 50.0, 60.0),
            pin_names=("driver/Y", "sink0/A", "sink1/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "buf_out"),
            pin2node_map=np.array([0, 1, 2, 3, 3], dtype=np.int32),
            pin2net_map=np.array([0, 1, 1, 0, 1], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink0", "sink1", "buf0", "buf1"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1", "BUF_X1", "BUF_X1"),
            node_is_buffer=(False, False, False, True, True),
            node_size_x=(1.0, 2.0, 3.0, 1.0, 1.0),
            node_size_y=(1.5, 2.5, 3.5, 1.5, 1.5),
            node_orient=("N", "N", "N", "N", "N"),
            node_x=(10.0, 20.0, 30.0, 40.0, 45.0),
            node_y=(30.0, 40.0, 50.0, 60.0, 65.0),
            pin_names=(
                "driver/Y",
                "sink0/A",
                "sink1/A",
                "buf0/A",
                "buf0/Y",
                "buf1/A",
                "buf1/Y",
            ),
            net_names=("n0", "buf_out", "split_n0"),
            pin2node_map=np.array([0, 1, 2, 3, 3, 4, 4], dtype=np.int32),
            pin2net_map=np.array([0, 2, 2, 0, 2, 0, 1], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(policy["status"], "clean")
        self.assertEqual(policy["allowed_change_counts"]["added_buffer_count"], 1)
        self.assertEqual(policy["violation_counts"]["connectivity_violation_count"], 0)
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 0)

    def test_buffer_only_policy_accepts_pure_surviving_buffer_rewire(self):
        old_pydb = _make_topology_snapshot(
            node_names=("driver", "sink0", "sink1", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1", "BUF_X1"),
            node_is_buffer=(False, False, False, True),
            node_size_x=(1.0, 2.0, 3.0, 1.0),
            node_size_y=(1.5, 2.5, 3.5, 1.5),
            node_orient=("N", "N", "N", "N"),
            node_x=(10.0, 20.0, 30.0, 40.0),
            node_y=(30.0, 40.0, 50.0, 60.0),
            pin_names=("driver/Y", "sink0/A", "sink1/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "buf_out"),
            pin2node_map=np.array([0, 1, 2, 3, 3], dtype=np.int32),
            pin2net_map=np.array([0, 1, 1, 0, 1], dtype=np.int32),
        )
        new_pydb = _make_topology_snapshot(
            node_names=("driver", "sink0", "sink1", "buf0"),
            node_master_names=("NAND_X1", "INV_X1", "NOR_X1", "BUF_X1"),
            node_is_buffer=(False, False, False, True),
            node_size_x=(1.0, 2.0, 3.0, 1.0),
            node_size_y=(1.5, 2.5, 3.5, 1.5),
            node_orient=("N", "N", "N", "N"),
            node_x=(10.0, 20.0, 30.0, 40.0),
            node_y=(30.0, 40.0, 50.0, 60.0),
            pin_names=("driver/Y", "sink0/A", "sink1/A", "buf0/A", "buf0/Y"),
            net_names=("n0", "buf_out", "split_n0"),
            pin2node_map=np.array([0, 1, 2, 3, 3], dtype=np.int32),
            pin2net_map=np.array([0, 2, 2, 0, 2], dtype=np.int32),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "topology_changed")
        self.assertEqual(policy["status"], "clean")
        self.assertEqual(policy["allowed_change_counts"]["added_buffer_count"], 0)
        self.assertEqual(policy["allowed_change_counts"]["removed_buffer_count"], 0)
        self.assertEqual(policy["violation_counts"]["connectivity_violation_count"], 0)
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 0)

    def test_buffer_only_policy_reports_unknown_when_buffer_metadata_is_missing(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot(node_master_names=("BUF_X2", "INV_X1"))
        del old_pydb.node_is_buffer
        del new_pydb.node_is_buffer

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(contract["mutation_kind"], "geometry_changed")
        self.assertEqual(policy["status"], "unknown")
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 0)
        self.assertGreater(policy["summary"]["unknown_reason_count"], 0)

    def test_buffer_only_policy_reports_unknown_when_coordinate_metadata_is_missing(self):
        old_pydb = _make_topology_snapshot()
        new_pydb = _make_topology_snapshot()
        del old_pydb.node_x
        del new_pydb.node_y

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]
        unknown_fields = {
            unknown.get("field_name")
            for unknown in policy["unknown_reasons"]
            if unknown.get("reason") == "missing_surviving_node_metadata"
        }

        self.assertEqual(policy["status"], "unknown")
        self.assertEqual(policy["violation_counts"]["total_violation_count"], 0)
        self.assertIn("node_x", unknown_fields)
        self.assertIn("node_y", unknown_fields)

    def test_buffer_only_policy_truncates_violation_samples(self):
        old_names = tuple("u%d" % index for index in range(60))
        new_names = old_names
        old_pydb = _make_topology_snapshot(
            node_names=old_names,
            node_master_names=tuple("NAND_X1" for _ in old_names),
            node_is_buffer=tuple(False for _ in old_names),
            node_size_x=tuple(1.0 for _ in old_names),
            node_size_y=tuple(1.0 for _ in old_names),
            node_orient=tuple("N" for _ in old_names),
            node_x=tuple(float(index) for index in range(60)),
            node_y=tuple(float(index) for index in range(60)),
        )
        new_pydb = _make_topology_snapshot(
            node_names=new_names,
            node_master_names=tuple("NAND_X2" for _ in new_names),
            node_is_buffer=tuple(False for _ in new_names),
            node_size_x=tuple(2.0 for _ in new_names),
            node_size_y=tuple(1.0 for _ in new_names),
            node_orient=tuple("N" for _ in new_names),
            node_x=tuple(float(index) for index in range(60)),
            node_y=tuple(float(index) for index in range(60)),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(policy["status"], "violated")
        self.assertEqual(len(policy["violation_samples"]), 50)
        self.assertTrue(policy["samples_truncated"])
        self.assertEqual(policy["sample_limit"], 50)

    def test_buffer_only_policy_truncates_when_total_violations_exceed_limit(self):
        names = tuple("u%d" % index for index in range(26))
        old_pydb = _make_topology_snapshot(
            node_names=names,
            node_master_names=tuple("NAND_X1" for _ in names),
            node_is_buffer=tuple(False for _ in names),
            node_size_x=tuple(1.0 for _ in names),
            node_size_y=tuple(1.0 for _ in names),
            node_orient=tuple("N" for _ in names),
            node_x=tuple(float(index) for index in range(26)),
            node_y=tuple(float(index) for index in range(26)),
        )
        new_pydb = _make_topology_snapshot(
            node_names=names,
            node_master_names=tuple("NAND_X2" for _ in names),
            node_is_buffer=tuple(False for _ in names),
            node_size_x=tuple(2.0 for _ in names),
            node_size_y=tuple(1.0 for _ in names),
            node_orient=tuple("N" for _ in names),
            node_x=tuple(float(index) for index in range(26)),
            node_y=tuple(float(index) for index in range(26)),
        )

        contract = self._build_contract(old_pydb, new_pydb)
        policy = contract["buffer_only_policy"]

        self.assertEqual(policy["violation_counts"]["total_violation_count"], 52)
        self.assertEqual(len(policy["violation_samples"]), 26)
        self.assertTrue(policy["samples_truncated"])
        self.assertEqual(
            policy["samples_truncated"],
            policy["violation_counts"]["total_violation_count"]
            > policy["sample_limit"],
        )

    def test_synchronize_topology_ignores_deprecated_mutation_hint(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.openroad_bridge = None

        contract = placedb.synchronize_topology_from_openroad(
            params=object(),
            mutation_hint="topology_changed",
        )

        self.assertEqual(contract["mutation_kind"], "placement_only")
        self.assertFalse(contract["requires_runtimedb_rebuild"])

    def test_create_or_reset_handoff_session_sets_session_on_placedb(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)

        first_session = placedb.create_or_reset_handoff_session(session_id="run-1")
        self.assertIsInstance(first_session, PlacementHandoffSession)
        self.assertIs(first_session, placedb.handoff_session)

        second_session = placedb.create_or_reset_handoff_session(session_id="run-2")

        self.assertIsInstance(second_session, PlacementHandoffSession)
        self.assertIs(second_session, placedb.handoff_session)
        self.assertIsNot(first_session, second_session)
        self.assertEqual(second_session.session_id, "run-2")

    def test_execute_openroad_handoff_returns_structured_mutation_result(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.handoff_session = PlacementHandoffSession(session_id="run-1")
        placedb.num_physical_nodes = 10
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.openroad_bridge = type(
            "Bridge",
            (),
            {
                "session_identity": "bridge-1",
                "write_def": lambda self, output: None,
            },
        )()
        placedb.run_openroad_buffer_insertion = lambda params, pos, handoff_event: {
            "triggered": True,
            "executed": True,
            "requires_sync_back": True,
            "requires_runtimedb_rebuild": True,
            "mutation_kind": "topology_changed",
            "sync_contract": {
                "mutation_kind": "topology_changed",
                "requires_runtimedb_rebuild": True,
                "old_counts": {"nodes": 10, "pins": 20, "nets": 5},
                "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
                "added_names": {"nodes": ["u10", "u11"], "pins": ["p20"], "nets": ["n5"]},
                "removed_names": {"nodes": [], "pins": [], "nets": []},
                "added_buffer_count": 2,
                "removed_buffer_count": 1,
                "surviving_buffer_count": 7,
                "surviving_name_to_new_id": {"u0": 0, "u1": 1},
                "stable_identity_key": "name",
                "legacy_ids_stable": False,
                "runtimedb_rebuild_owner": "dreamplace_runtimedb",
            },
        }

        params = type(
            "Params",
            (),
            {
                "place_io_engine": "openroad",
                "openroad_handoff": {
                    "enabled": True,
                    "buffer_insertion": {"enabled": True},
                },
            },
        )()
        handoff_event = {
            "absolute_iteration": 10,
            "phase": 1,
            "trigger_mode": "interval",
            "trigger_reason": "interval=10@iter=10",
        }
        decision = {
            "trigger_mode": "interval",
            "trigger_reason": "interval=10@iter=10",
        }

        result = placedb.execute_openroad_handoff(
            params=params,
            pos=None,
            handoff_event=handoff_event,
        )

        self.assertEqual(result["mutation_kind"], "topology_changed")
        self.assertTrue(result["requires_runtimedb_rebuild"])
        self.assertEqual(
            result["post_counts"],
            {"nodes": 12, "pins": 24, "nets": 6},
        )
        self.assertEqual(
            result["identity_summary"],
            {
                "stable_identity_key": "name",
                "legacy_ids_stable": False,
                "surviving_name_to_new_id": {"u0": 0, "u1": 1},
            },
        )
        self.assertEqual(
            result["added_names"],
            {"nodes": ["u10", "u11"], "pins": ["p20"], "nets": ["n5"]},
        )
        self.assertEqual(
            result["removed_names"],
            {"nodes": [], "pins": [], "nets": []},
        )
        self.assertEqual(
            result["runtimedb_rebuild_owner"],
            "dreamplace_runtimedb",
        )
        self.assertEqual(result["added_buffer_count"], 2)
        self.assertEqual(result["removed_buffer_count"], 1)
        self.assertEqual(result["surviving_buffer_count"], 7)
        self.assertEqual(
            result["sync_contract"]["runtimedb_rebuild_owner"],
            "dreamplace_runtimedb",
        )
        self.assertEqual(
            placedb.handoff_session.event_history,
            [],
        )
        self.assertNotIn("handoff_seq", result)
        self.assertNotIn("handoff_session", result)

    def test_topology_changed_handoff_rejects_missing_buffer_churn_counts(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        handoff_result = {
            "executed": True,
            "sync_contract": {
                "mutation_kind": "topology_changed",
                "requires_runtimedb_rebuild": True,
                "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "buffer churn"):
            placedb._build_mutation_result_from_handoff(
                handoff_result,
                pre_counts={"nodes": 10, "pins": 20, "nets": 5},
            )

    def test_topology_changed_handoff_rejects_invalid_buffer_churn_counts(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        invalid_values = (None, True, -1, 1.5, "2")
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                handoff_result = {
                    "executed": True,
                    "sync_contract": {
                        "mutation_kind": "topology_changed",
                        "requires_runtimedb_rebuild": True,
                        "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
                        "added_buffer_count": invalid_value,
                        "removed_buffer_count": 1,
                        "surviving_buffer_count": 7,
                    },
                }

                with self.assertRaisesRegex(RuntimeError, "buffer churn"):
                    placedb._build_mutation_result_from_handoff(
                        handoff_result,
                        pre_counts={"nodes": 10, "pins": 20, "nets": 5},
                    )

    def test_refresh_topology_restores_positions_when_node_map_uses_bytes_keys(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb._restore_topology_refresh_params = lambda params: None

        def _initialize_from_rawdb(pydb, params):
            placedb.node_name2id_map = {b"u0": 0, b"u1": 1}
            placedb.num_physical_nodes = 2
            placedb.node_x = [0.0, 0.0]
            placedb.node_y = [0.0, 0.0]

        placedb.initialize_from_rawdb = _initialize_from_rawdb
        placedb.initialize = lambda params: None

        placedb.refresh_topology_from_openroad_snapshot(
            params=object(),
            new_pydb=object(),
            old_positions={"u0": (1.5, 2.5)},
        )

        self.assertEqual(placedb.node_x[0], 1.5)
        self.assertEqual(placedb.node_y[0], 2.5)

    def test_refresh_topology_restores_unchanged_non_buffer_instance_after_buffer_churn(self):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb._restore_topology_refresh_params = lambda params: None

        def _initialize_from_rawdb(pydb, params):
            placedb.node_name2id_map = {"logic0": 0, "buf_new": 1}
            placedb.num_physical_nodes = 2
            placedb.node_x = [0.0, 0.0]
            placedb.node_y = [0.0, 0.0]

        placedb.initialize_from_rawdb = _initialize_from_rawdb
        placedb.initialize = lambda params: None

        placedb.refresh_topology_from_openroad_snapshot(
            params=object(),
            new_pydb=_make_topology_snapshot(
                node_names=("logic0", "buf_new"),
                node_master_names=("NAND_X1", "BUF_X2"),
                node_is_buffer=(False, True),
                node_size_x=(2.0, 1.0),
                node_size_y=(2.0, 1.0),
                node_orient=("N", "N"),
                node_x=(0.0, 0.0),
                node_y=(0.0, 0.0),
            ),
            old_positions={"logic0": (42.0, 43.0), "buf_old": (10.0, 11.0)},
        )

        self.assertEqual(placedb.node_x[0], 42.0)
        self.assertEqual(placedb.node_y[0], 43.0)
        self.assertEqual(placedb.node_x[1], 0.0)
        self.assertEqual(placedb.node_y[1], 0.0)


class BasicPlaceContinuationSeedTest(unittest.TestCase):
    def _make_params(self):
        return types.SimpleNamespace(
            init_loc_perc_x=0.5,
            init_loc_perc_y=0.5,
            global_place_flag=True,
            random_center_init_flag=True,
            gpu=False,
            gpu_id=0,
            with_sta=False,
            macro_place_flag=False,
            routability_opt_flag=False,
        )

    def _make_placedb(self):
        return types.SimpleNamespace(
            total_movable_node_area=2.0,
            num_nodes=5,
            num_physical_nodes=3,
            num_movable_nodes=2,
            num_filler_nodes=2,
            dtype=np.float64,
            node_x=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            node_y=np.array([4.0, 5.0, 6.0], dtype=np.float64),
            node_names=np.array([b"u0", b"u1", b"fixed0"], dtype=np.bytes_),
            node_name2id_map={b"u0": 0, b"u1": 1, b"fixed0": 2},
            node_size_x=np.ones(5, dtype=np.float64),
            node_size_y=np.ones(5, dtype=np.float64),
            xl=0.0,
            xh=100.0,
            yl=0.0,
            yh=100.0,
            regions=[],
        )

    def _make_init_only_basic_place_class(self):
        BasicPlaceModule = _load_basicplace_module()
        # Keep area initialization real while native operators are stubbed.
        BasicPlaceModule.initialize_overflow_reference = initialize_overflow_reference

        class InitOnlyBasicPlace(BasicPlaceModule.BasicPlace):
            def build_pin_pos(self, *args):
                return None

            def build_move_boundary(self, *args):
                return None

            def build_hpwl(self, *args):
                return None

            def build_weight_hpwl(self, *args):
                return None

            def build_rsmt_wl(self, *args):
                return None

            def build_legality_check(self, *args):
                return None

            def build_legalization(self, *args):
                return None

            def build_detailed_placement(self, *args):
                return None

            def build_draw_placement(self, *args):
                return None

        class StubPlaceDataCollection:
            def __init__(self, pos, params, placedb, device):
                self.pos = pos
                self.flat_node2pin_map = np.array([], dtype=np.int32)
                self.flat_node2pin_start_map = np.zeros(
                    int(placedb.num_nodes) + 1, dtype=np.int32
                )
                self.pin2net_map = np.array([], dtype=np.int32)

            def build_gate_projection_op(self):
                return None

        return BasicPlaceModule, InitOnlyBasicPlace, StubPlaceDataCollection

    def test_pending_continuation_seed_populates_init_pos_without_random_reinitialization(self):
        (
            BasicPlaceModule,
            InitOnlyBasicPlace,
            StubPlaceDataCollection,
        ) = self._make_init_only_basic_place_class()
        placedb = self._make_placedb()
        placedb.pending_continuation_seed = {
            "overflow_reference_area": 1.5,
            "movable_x": [11.0, 12.0],
            "movable_y": [21.0, 22.0],
            "movable_node_names": ["u0", "u1"],
            "filler_x": [31.0, 32.0],
            "filler_y": [41.0, 42.0],
            "source_handoff_seq": 7,
            "source_snapshot_fingerprint": "post-fp",
            "source_topology_epoch": 3,
        }

        with mock.patch.object(
            BasicPlaceModule,
            "PlaceDataCollection",
            StubPlaceDataCollection,
        ), mock.patch.object(
            BasicPlaceModule.np.random,
            "normal",
            wraps=BasicPlaceModule.np.random.normal,
        ) as normal_mock, mock.patch.object(
            BasicPlaceModule.np.random,
            "uniform",
            wraps=BasicPlaceModule.np.random.uniform,
        ) as uniform_mock, self.assertLogs(level="INFO") as log_context:
            place = InitOnlyBasicPlace(self._make_params(), placedb, timer=None)

        self.assertFalse(
            any(
                "move cells to location" in message
                and "with random noise" in message
                for message in log_context.output
            )
        )
        normal_mock.assert_not_called()
        uniform_mock.assert_not_called()
        self.assertIsNone(placedb.pending_continuation_seed)
        self.assertEqual(placedb.overflow_reference_area, 1.5)
        np.testing.assert_allclose(place.init_pos[:2], [11.0, 12.0])
        np.testing.assert_allclose(place.init_pos[2:3], [3.0])
        np.testing.assert_allclose(place.init_pos[3:5], [31.0, 32.0])
        np.testing.assert_allclose(place.init_pos[5:7], [21.0, 22.0])
        np.testing.assert_allclose(place.init_pos[7:8], [6.0])
        np.testing.assert_allclose(place.init_pos[8:10], [41.0, 42.0])

    def test_legacy_continuation_seed_without_names_keeps_restored_positions_after_reorder(self):
        (
            BasicPlaceModule,
            InitOnlyBasicPlace,
            StubPlaceDataCollection,
        ) = self._make_init_only_basic_place_class()
        placedb = self._make_placedb()
        placedb.node_x = np.array([200.0, 100.0, 300.0], dtype=np.float64)
        placedb.node_y = np.array([220.0, 110.0, 330.0], dtype=np.float64)
        placedb.pending_continuation_seed = {
            "movable_x": [100.0, 200.0],
            "movable_y": [110.0, 220.0],
            "filler_x": [31.0, 32.0],
            "filler_y": [41.0, 42.0],
            "source_handoff_seq": 7,
            "source_snapshot_fingerprint": "pre-fp",
            "source_topology_epoch": 3,
        }

        with mock.patch.object(
            BasicPlaceModule,
            "PlaceDataCollection",
            StubPlaceDataCollection,
        ), mock.patch.object(
            BasicPlaceModule.np.random,
            "uniform",
            wraps=BasicPlaceModule.np.random.uniform,
        ) as uniform_mock:
            place = InitOnlyBasicPlace(self._make_params(), placedb, timer=None)

        uniform_mock.assert_not_called()
        self.assertIsNone(placedb.pending_continuation_seed)
        np.testing.assert_allclose(place.init_pos[:3], [200.0, 100.0, 300.0])
        np.testing.assert_allclose(place.init_pos[5:8], [220.0, 110.0, 330.0])
        np.testing.assert_allclose(place.init_pos[3:5], [31.0, 32.0])
        np.testing.assert_allclose(place.init_pos[8:10], [41.0, 42.0])

    def test_named_movable_continuation_seed_maps_reordered_nodes_by_name(self):
        (
            BasicPlaceModule,
            InitOnlyBasicPlace,
            StubPlaceDataCollection,
        ) = self._make_init_only_basic_place_class()
        placedb = self._make_placedb()
        placedb.node_names = np.array([b"u1", b"u0", b"fixed0"], dtype=np.bytes_)
        placedb.node_name2id_map = {b"u1": 0, b"u0": 1, b"fixed0": 2}
        placedb.node_x = np.array([200.0, 100.0, 300.0], dtype=np.float64)
        placedb.node_y = np.array([220.0, 110.0, 330.0], dtype=np.float64)
        placedb.pending_continuation_seed = {
            "movable_x": [101.0, 202.0],
            "movable_y": [111.0, 222.0],
            "movable_node_names": ["u0", "u1"],
            "filler_x": [31.0, 32.0],
            "filler_y": [41.0, 42.0],
        }

        with mock.patch.object(
            BasicPlaceModule,
            "PlaceDataCollection",
            StubPlaceDataCollection,
        ), mock.patch.object(
            BasicPlaceModule.np.random,
            "uniform",
            wraps=BasicPlaceModule.np.random.uniform,
        ) as uniform_mock:
            place = InitOnlyBasicPlace(self._make_params(), placedb, timer=None)

        uniform_mock.assert_not_called()
        self.assertIsNone(placedb.pending_continuation_seed)
        np.testing.assert_allclose(place.init_pos[:3], [202.0, 101.0, 300.0])
        np.testing.assert_allclose(place.init_pos[5:8], [222.0, 111.0, 330.0])
        np.testing.assert_allclose(place.init_pos[3:5], [31.0, 32.0])
        np.testing.assert_allclose(place.init_pos[8:10], [41.0, 42.0])

    def test_named_movable_continuation_seed_leaves_new_nodes_at_restored_positions(self):
        (
            BasicPlaceModule,
            InitOnlyBasicPlace,
            StubPlaceDataCollection,
        ) = self._make_init_only_basic_place_class()
        placedb = self._make_placedb()
        placedb.node_names = np.array([b"u1", b"u2", b"fixed0"], dtype=np.bytes_)
        placedb.node_name2id_map = {b"u1": 0, b"u2": 1, b"fixed0": 2}
        placedb.node_x = np.array([200.0, 12.0, 300.0], dtype=np.float64)
        placedb.node_y = np.array([220.0, 22.0, 330.0], dtype=np.float64)
        placedb.pending_continuation_seed = {
            "movable_x": [100.0, 202.0],
            "movable_y": [110.0, 222.0],
            "movable_node_names": ["u0", "u1"],
            "filler_x": [31.0, 32.0],
            "filler_y": [41.0, 42.0],
        }

        with mock.patch.object(
            BasicPlaceModule,
            "PlaceDataCollection",
            StubPlaceDataCollection,
        ), mock.patch.object(
            BasicPlaceModule.np.random,
            "uniform",
            wraps=BasicPlaceModule.np.random.uniform,
        ) as uniform_mock:
            place = InitOnlyBasicPlace(self._make_params(), placedb, timer=None)

        uniform_mock.assert_not_called()
        self.assertIsNone(placedb.pending_continuation_seed)
        np.testing.assert_allclose(place.init_pos[:3], [202.0, 12.0, 300.0])
        np.testing.assert_allclose(place.init_pos[5:8], [222.0, 22.0, 330.0])
        np.testing.assert_allclose(place.init_pos[3:5], [31.0, 32.0])
        np.testing.assert_allclose(place.init_pos[8:10], [41.0, 42.0])

    def test_continuation_seed_initializes_uncovered_target_fillers_without_overwriting_seed(self):
        (
            BasicPlaceModule,
            InitOnlyBasicPlace,
            StubPlaceDataCollection,
        ) = self._make_init_only_basic_place_class()
        placedb = self._make_placedb()
        placedb.num_nodes = 6
        placedb.num_filler_nodes = 3
        placedb.node_size_x = np.ones(6, dtype=np.float64)
        placedb.node_size_y = np.ones(6, dtype=np.float64)
        placedb.pending_continuation_seed = {
            "movable_x": [11.0, 12.0],
            "movable_y": [21.0, 22.0],
            "filler_x": [31.0, 32.0],
            "filler_y": [41.0, 42.0],
            "source_handoff_seq": 7,
            "source_snapshot_fingerprint": "pre-fp",
            "source_topology_epoch": 3,
        }

        with mock.patch.object(
            BasicPlaceModule,
            "PlaceDataCollection",
            StubPlaceDataCollection,
        ), mock.patch.object(
            BasicPlaceModule.np.random,
            "uniform",
            side_effect=[np.array([71.0]), np.array([81.0])],
        ) as uniform_mock:
            place = InitOnlyBasicPlace(self._make_params(), placedb, timer=None)

        self.assertEqual(uniform_mock.call_count, 2)
        self.assertIsNone(placedb.pending_continuation_seed)
        np.testing.assert_allclose(place.init_pos[3:6], [31.0, 32.0, 71.0])
        np.testing.assert_allclose(place.init_pos[9:12], [41.0, 42.0, 81.0])

    def test_malformed_continuation_seed_remains_pending_when_initialization_raises(self):
        (
            BasicPlaceModule,
            InitOnlyBasicPlace,
            StubPlaceDataCollection,
        ) = self._make_init_only_basic_place_class()
        placedb = self._make_placedb()
        placedb.pending_continuation_seed = {
            "movable_x": [11.0, 12.0],
            "movable_y": [21.0, 22.0],
            "filler_x": ["not-a-number"],
            "filler_y": [41.0],
            "source_handoff_seq": 7,
            "source_snapshot_fingerprint": "pre-fp",
            "source_topology_epoch": 3,
        }

        with mock.patch.object(
            BasicPlaceModule,
            "PlaceDataCollection",
            StubPlaceDataCollection,
        ), self.assertRaises(ValueError):
            InitOnlyBasicPlace(self._make_params(), placedb, timer=None)

        self.assertEqual(
            placedb.pending_continuation_seed,
            {
                "movable_x": [11.0, 12.0],
                "movable_y": [21.0, 22.0],
                "filler_x": ["not-a-number"],
                "filler_y": [41.0],
                "source_handoff_seq": 7,
                "source_snapshot_fingerprint": "pre-fp",
                "source_topology_epoch": 3,
            },
        )


class NonLinearPlaceOpenRoadHandoffFlowTest(unittest.TestCase):
    def _make_engine(self):
        NonLinearPlace = _load_nonlinearplace_class()
        return NonLinearPlace.__new__(NonLinearPlace)

    def _default_run_event(self):
        return {"absolute_iteration": 10, "phase": 1}

    def _default_trigger_decision(self):
        return {
            "trigger_mode": "interval",
            "trigger_reason": "interval=10@iter=10",
        }

    def _default_handoff_event(self):
        return {
            "absolute_iteration": 10,
            "phase": 1,
            "trigger_mode": "interval",
            "trigger_reason": "interval=10@iter=10",
        }

    def test_openroad_backed_flow_skips_ieda_final_check_log(self):
        engine = self._make_engine()
        engine.op_collections = types.SimpleNamespace(steiner_topo_op=object())
        params = types.SimpleNamespace(
            place_io_engine="openroad",
            openroad_backed_aimp_db=True,
            with_sta=True,
        )

        class FakeModel:
            def __init__(self):
                self.check_log_calls = 0

            def timing_obj(self, _pos):
                raise AssertionError("OpenROAD-backed finalization should not retime through iEDA")

            def check_log(self, *_args):
                self.check_log_calls += 1

        model = FakeModel()

        engine._finalize_after_runtime_projection_refresh(
            params=params,
            placedb=object(),
            cur_pos=None,
            model=model,
            projection_runtime_summary={"num_changed_instances": 0},
        )

        self.assertEqual(model.check_log_calls, 0)

    def test_openroad_backed_flow_skips_post_projection_ieda_refresh(self):
        engine = self._make_engine()
        params = types.SimpleNamespace(
            place_io_engine="openroad",
            openroad_backed_aimp_db=True,
            result_dir="/tmp/should-not-be-used",
            design_name=lambda: "unit_design",
        )
        summary = {
            "pre_projection": {"wns": 0.0, "tns": 0.0},
            "post_projection": {"wns": 0.0, "tns": 0.0},
            "post_legalization": {"wns": 0.0, "tns": 0.0},
        }

        result = engine._refresh_post_projection_timing_stage_artifact(
            params,
            timing_stage_summary=summary,
        )

        self.assertIs(result, summary)

    def _execute_default_handoff_transaction(self, engine, placedb):
        return engine._execute_openroad_handoff_transaction(
            params=object(),
            placedb=placedb,
            pos="pos-token",
            run_event=self._default_run_event(),
            trigger_decision=self._default_trigger_decision(),
            handoff_event=self._default_handoff_event(),
        )

    def _make_placedb(self, mutation_result=None, handoff_error=None):
        class FakePlaceDB:
            def __init__(self, mutation_result, handoff_error):
                self.handoff_session = None
                self.openroad_bridge = type(
                    "Bridge",
                    (),
                    {"session_identity": "bridge-1"},
                )()
                self._mutation_result = mutation_result or {
                    "pre_counts": {"nodes": 10, "pins": 20, "nets": 5},
                    "pre_snapshot_fingerprint": "nodes=10|pins=20|nets=5",
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "post_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    "post_snapshot_fingerprint": "nodes=12|pins=24|nets=6",
                    "identity_summary": {
                        "stable_identity_key": "name",
                        "legacy_ids_stable": False,
                        "surviving_name_to_new_id": {"u0": 0, "u1": 1},
                    },
                    "added_names": {
                        "nodes": ["u10", "u11"],
                        "pins": ["p20"],
                        "nets": ["n5"],
                    },
                    "removed_names": {
                        "nodes": [],
                        "pins": [],
                        "nets": [],
                    },
                    "added_buffer_count": 2,
                    "removed_buffer_count": 1,
                    "surviving_buffer_count": 7,
                    "runtimedb_rebuild_owner": "dreamplace_runtimedb",
                    "sync_contract": {
                        "mutation_kind": "topology_changed",
                        "requires_runtimedb_rebuild": True,
                        "old_counts": {"nodes": 10, "pins": 20, "nets": 5},
                        "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    },
                }
                self._handoff_error = handoff_error
                self.handoff_events = []

            def create_or_reset_handoff_session(self, session_id=None):
                if session_id is None:
                    session_id = "run-1"
                self.handoff_session = PlacementHandoffSession(session_id=session_id)
                return self.handoff_session

            def _get_openroad_session_identity(self):
                return self.openroad_bridge.session_identity

            def current_topology_counts(self):
                return {"nodes": 10, "pins": 20, "nets": 5}

            def current_snapshot_fingerprint(self, counts=None):
                return "nodes=10|pins=20|nets=5"

            def execute_openroad_handoff(self, params, pos, handoff_event):
                self.handoff_events.append(handoff_event)
                if self._handoff_error is not None:
                    raise self._handoff_error
                return dict(self._mutation_result)

        return FakePlaceDB(mutation_result, handoff_error)

    def _make_real_macroplacedb(self, handoff_error=None):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        placedb.handoff_session = None
        placedb.num_physical_nodes = 10
        placedb.num_terminals = 0
        placedb.num_terminal_NIs = 0
        placedb.pin_names = ["p%s" % i for i in range(20)]
        placedb.net_names = ["n%s" % i for i in range(5)]
        placedb.openroad_bridge = type(
            "Bridge",
            (),
            {
                "session_identity": "bridge-1",
                "write_def": lambda self, output: None,
            },
        )()

        def _run_openroad_buffer_insertion(params, pos, handoff_event):
            if handoff_error is not None:
                raise handoff_error
            return {
                "triggered": True,
                "executed": True,
                "requires_sync_back": True,
                "requires_runtimedb_rebuild": True,
                "sync_contract": {
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "old_counts": {"nodes": 10, "pins": 20, "nets": 5},
                    "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    "added_names": {"nodes": ["u10", "u11"], "pins": ["p20"], "nets": ["n5"]},
                    "removed_names": {"nodes": [], "pins": [], "nets": []},
                    "added_buffer_count": 2,
                    "removed_buffer_count": 1,
                    "surviving_buffer_count": 7,
                    "surviving_name_to_new_id": {"u0": 0, "u1": 1},
                    "stable_identity_key": "name",
                    "legacy_ids_stable": False,
                    "runtimedb_rebuild_owner": "dreamplace_runtimedb",
                },
            }

        placedb.run_openroad_buffer_insertion = _run_openroad_buffer_insertion
        return placedb

    def _make_seed_capture_placedb(self):
        return types.SimpleNamespace(
            num_nodes=5,
            num_physical_nodes=3,
            num_movable_nodes=2,
            num_filler_nodes=2,
            node_names=np.array([b"u0", b"u1", b"fixed0"], dtype=np.bytes_),
            pending_continuation_seed=None,
            handoff_session=types.SimpleNamespace(topology_epoch=3),
        )

    def _make_live_pos(self):
        return np.array(
            [10.0, 20.0, 30.0, 40.0, 50.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            dtype=np.float64,
        )

    def _execute_real_refresh_failure_handoff(self, refresh_failure_mode):
        MacroPlaceDB = _load_macroplacedb_class()
        placedb = MacroPlaceDB.__new__(MacroPlaceDB)
        params = types.SimpleNamespace(
            place_io_engine="openroad",
            scale_factor=1.0,
            shift_factor=(0.0, 0.0),
            openroad_handoff={
                "buffer_insertion": {
                    "enabled": True,
                    "strategy": "repair_design",
                    "options": {"max_wire_length": 10},
                    "write_def_after": (
                        "out.def" if refresh_failure_mode == "write_def" else None
                    ),
                },
            },
        )
        placedb.params = params
        placedb.handoff_session = None
        placedb.num_physical_nodes = 1
        placedb.num_terminal_NIs = 0
        placedb.num_terminals = 0
        placedb.pin_names = []
        placedb.net_names = []
        placedb._extract_unscaled_movable_positions = lambda pos: (
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
        )

        class Bridge:
            session_identity = "bridge-1"

            def write_def(self, output):
                raise RuntimeError("write_def boom")

        placedb.openroad_bridge = Bridge()
        placedb.current_topology_counts = lambda: {
            "nodes": 10,
            "pins": 20,
            "nets": 5,
        }
        placedb.current_snapshot_fingerprint = lambda counts=None: (
            "nodes=10|pins=20|nets=5"
        )
        if refresh_failure_mode == "sync_contract":
            placedb.synchronize_topology_from_openroad = (
                lambda params, pos=None: (_ for _ in ()).throw(
                    RuntimeError("refresh boom")
                )
            )
        else:
            placedb.synchronize_topology_from_openroad = lambda params, pos=None: {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
                "new_counts": {"nodes": 10, "pins": 20, "nets": 5},
                "added_buffer_count": 0,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 0,
            }

        placeio_module = types.ModuleType("dreamplace.ops.placeio_openroad.place_io")

        class FakePlaceIOFunction:
            @staticmethod
            def sync_to_openroad(raw_db, node_x, node_y):
                return None

            @staticmethod
            def describe_buffer_insertion_strategy(strategy=None):
                return {
                    "profile": "repair_design",
                    "kind": "baseline",
                    "experimental": False,
                    "validation": "default_baseline",
                }

            @staticmethod
            def build_buffer_insertion_command(strategy, options=None):
                return "repair_design -max_wire_length {10}"

            @staticmethod
            def buffer_insertion(raw_db, strategy="repair_design", options=None):
                return "OpenROAD output"

        placeio_module.PlaceIOFunction = FakePlaceIOFunction
        ops_package = types.ModuleType("dreamplace.ops")
        ops_package.__path__ = []
        placeio_package = types.ModuleType("dreamplace.ops.placeio_openroad")
        placeio_package.__path__ = []
        placeio_package.place_io = placeio_module
        previous_modules = {
            name: sys.modules.get(name)
            for name in (
                "dreamplace.ops",
                "dreamplace.ops.placeio_openroad",
                "dreamplace.ops.placeio_openroad.place_io",
            )
        }
        import dreamplace

        previous_ops_attr = getattr(dreamplace, "ops", None)
        previous_had_ops_attr = hasattr(dreamplace, "ops")
        ops_package.placeio_openroad = placeio_package
        dreamplace.ops = ops_package
        sys.modules["dreamplace.ops"] = ops_package
        sys.modules["dreamplace.ops.placeio_openroad"] = placeio_package
        sys.modules["dreamplace.ops.placeio_openroad.place_io"] = placeio_module
        try:
            engine = self._make_engine()
            engine._restart_after_topology_sync = (
                lambda params, placedb_arg, result, pos=None, handoff_seq=None: None
            )
            with self.assertRaisesRegex(RuntimeError, "boom"):
                engine._execute_openroad_handoff_transaction(
                    params=params,
                    placedb=placedb,
                    pos=np.array([1.0, 2.0], dtype=np.float64),
                    run_event=self._default_run_event(),
                    trigger_decision=self._default_trigger_decision(),
                    handoff_event=self._default_handoff_event(),
                )
        finally:
            for name, module in previous_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            if previous_had_ops_attr:
                dreamplace.ops = previous_ops_attr
            else:
                delattr(dreamplace, "ops")
        return placedb.handoff_session.event_history[-1]

    def test_rebuild_restart_stores_live_position_continuation_seed_before_runtime_construction(self):
        NonLinearPlace = _load_nonlinearplace_class()
        constructed_seeds = []

        class RestartProbe(NonLinearPlace):
            def __init__(self, params, placedb, timer):
                constructed_seeds.append(placedb.pending_continuation_seed)

        engine = RestartProbe.__new__(RestartProbe)
        engine.timer = "timer-token"
        placedb = self._make_seed_capture_placedb()
        handoff_result = {
            "mutation_kind": "topology_changed",
            "requires_runtimedb_rebuild": True,
            "pre_snapshot_fingerprint": "pre-fingerprint",
            "post_snapshot_fingerprint": "post-fingerprint",
            "sync_contract": {
                "mutation_kind": "topology_changed",
                "requires_runtimedb_rebuild": True,
                "old_counts": {"nodes": 5, "pins": 10, "nets": 2},
                "new_counts": {"nodes": 5, "pins": 10, "nets": 2},
            },
        }

        restart_result = engine._restart_after_topology_sync(
            params=object(),
            placedb=placedb,
            handoff_result=handoff_result,
            pos=self._make_live_pos(),
            handoff_seq=42,
        )

        self.assertIsInstance(restart_result, RestartProbe)
        self.assertEqual(len(constructed_seeds), 1)
        self.assertIs(constructed_seeds[0], placedb.pending_continuation_seed)
        self.assertEqual(
            placedb.pending_continuation_seed,
            {
                "movable_x": [10.0, 20.0],
                "movable_y": [110.0, 120.0],
                "movable_node_names": ["u0", "u1"],
                "filler_x": [40.0, 50.0],
                "filler_y": [140.0, 150.0],
                "source_handoff_seq": 42,
                "source_snapshot_fingerprint": "pre-fingerprint",
                "source_topology_epoch": 3,
            },
        )

    def test_rebuild_seed_uses_source_runtime_layout_after_placedb_counts_change(self):
        NonLinearPlace = _load_nonlinearplace_class()

        class RestartProbe(NonLinearPlace):
            def __init__(self, params, placedb, timer):
                pass

        engine = RestartProbe.__new__(RestartProbe)
        engine.timer = "timer-token"
        placedb = types.SimpleNamespace(
            num_nodes=6,
            num_physical_nodes=4,
            num_movable_nodes=3,
            num_filler_nodes=2,
            pending_continuation_seed=None,
            handoff_session=types.SimpleNamespace(topology_epoch=3),
        )

        engine._restart_after_topology_sync(
            params=object(),
            placedb=placedb,
            handoff_result={
                "mutation_kind": "topology_changed",
                "requires_runtimedb_rebuild": True,
                "post_snapshot_fingerprint": "post-fingerprint",
                "_source_placement_counts": {
                    "num_nodes": 5,
                    "num_physical_nodes": 3,
                    "num_movable_nodes": 2,
                    "num_filler_nodes": 2,
                },
                "sync_contract": {
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "old_counts": {"nodes": 3, "pins": 10, "nets": 2},
                    "new_counts": {"nodes": 4, "pins": 12, "nets": 3},
                },
            },
            pos=self._make_live_pos(),
            handoff_seq=42,
        )

        self.assertEqual(placedb.pending_continuation_seed["movable_x"], [10.0, 20.0])
        self.assertEqual(placedb.pending_continuation_seed["movable_y"], [110.0, 120.0])
        self.assertEqual(placedb.pending_continuation_seed["filler_x"], [40.0, 50.0])
        self.assertEqual(placedb.pending_continuation_seed["filler_y"], [140.0, 150.0])

    def test_no_mutation_and_placement_only_restart_do_not_seed_or_rebuild(self):
        NonLinearPlace = _load_nonlinearplace_class()

        class RestartProbe(NonLinearPlace):
            def __init__(self, params, placedb, timer):
                raise AssertionError("runtime rebuild should not be constructed")

        for mutation_kind in ("no_mutation", "placement_only"):
            engine = RestartProbe.__new__(RestartProbe)
            engine.timer = "timer-token"
            placedb = self._make_seed_capture_placedb()
            placedb.pending_continuation_seed = {"stale": True}

            restart_result = engine._restart_after_topology_sync(
                params=object(),
                placedb=placedb,
                handoff_result={
                    "mutation_kind": mutation_kind,
                    "requires_runtimedb_rebuild": False,
                    "sync_contract": {
                        "mutation_kind": mutation_kind,
                        "requires_runtimedb_rebuild": False,
                    },
                },
                pos=self._make_live_pos(),
                handoff_seq=99,
            )

            self.assertIsNone(restart_result)
            self.assertIsNone(placedb.pending_continuation_seed)

    def test_handoff_transaction_supplies_live_pos_and_handoff_seq_to_restart(self):
        engine = self._make_engine()
        placedb = self._make_placedb()
        live_pos = object()
        captured = {}

        def _restart_after_topology_sync(params, placedb_arg, result, pos=None, handoff_seq=None):
            captured["pos"] = pos
            captured["handoff_seq"] = handoff_seq
            return {"runtimedb": "rebuilt"}

        engine._restart_after_topology_sync = _restart_after_topology_sync

        self._execute_default_handoff_transaction_with_pos(engine, placedb, live_pos)

        self.assertIs(captured["pos"], live_pos)
        self.assertEqual(
            captured["handoff_seq"],
            placedb.handoff_session.event_history[0]["handoff_seq"],
        )

    def _execute_default_handoff_transaction_with_pos(self, engine, placedb, pos):
        return engine._execute_openroad_handoff_transaction(
            params=object(),
            placedb=placedb,
            pos=pos,
            run_event=self._default_run_event(),
            trigger_decision=self._default_trigger_decision(),
            handoff_event=self._default_handoff_event(),
        )

    def test_topology_changed_handoff_commits_only_after_runtimedb_rebuild(self):
        engine = self._make_engine()
        placedb = self._make_placedb()

        def _restart_after_topology_sync(params, placedb_arg, result, pos=None, handoff_seq=None):
            def _continued_run(continued_params, continued_placedb):
                self.assertEqual(continued_placedb.handoff_session.status, "active")
                self.assertEqual(continued_placedb.handoff_session.runtimedb_generation, 1)
                return {"runtimedb": "rebuilt"}

            return _continued_run

        engine._restart_after_topology_sync = _restart_after_topology_sync

        restart_result = self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(restart_result, {"runtimedb": "rebuilt"})
        self.assertEqual(placedb.handoff_session.event_history[-1]["status"], "success")
        self.assertEqual(placedb.handoff_session.runtimedb_generation, 1)
        self.assertEqual(placedb.handoff_session.topology_epoch, 1)

    def test_handoff_transaction_records_buffer_churn_in_session_event(self):
        engine = self._make_engine()
        placedb = self._make_placedb()
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: {
            "runtimedb": "rebuilt"
        }

        self._execute_default_handoff_transaction(engine, placedb)

        event = placedb.handoff_session.event_history[-1]
        self.assertEqual(event["added_buffer_count"], 2)
        self.assertEqual(event["removed_buffer_count"], 1)
        self.assertEqual(event["surviving_buffer_count"], 7)

    def test_session_mutation_result_carries_buffer_only_policy(self):
        NonLinearPlace = _load_nonlinearplace_class()
        placer = NonLinearPlace.__new__(NonLinearPlace)
        policy = {
            "status": "clean",
            "allowed_change_counts": {"added_buffer_count": 1},
            "violation_counts": {"total_violation_count": 0},
            "summary": {"unknown_reason_count": 0},
            "violation_samples": [],
            "unknown_reasons": [],
            "sample_limit": 50,
            "samples_truncated": False,
        }

        result = placer._build_session_mutation_result(
            {
                "mutation_kind": "topology_changed",
                "requires_runtimedb_rebuild": True,
                "post_counts": {"nodes": 3, "pins": 2, "nets": 1},
                "added_buffer_count": 1,
                "removed_buffer_count": 0,
                "surviving_buffer_count": 1,
                "buffer_only_policy": policy,
            }
        )

        self.assertEqual(result["buffer_only_policy"], policy)
        self.assertIsNot(result["buffer_only_policy"], policy)

    def test_topology_changed_session_payload_rejects_missing_buffer_churn_counts(self):
        engine = self._make_engine()
        handoff_result = {
            "mutation_kind": "topology_changed",
            "requires_runtimedb_rebuild": True,
            "post_counts": {"nodes": 12, "pins": 24, "nets": 6},
        }

        with self.assertRaisesRegex(RuntimeError, "buffer churn"):
            engine._build_session_mutation_result(handoff_result)

    def test_topology_changed_session_payload_rejects_invalid_buffer_churn_counts(self):
        engine = self._make_engine()
        invalid_values = (None, True, -1, 1.5, "2")
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                handoff_result = {
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "post_counts": {"nodes": 12, "pins": 24, "nets": 6},
                    "added_buffer_count": invalid_value,
                    "removed_buffer_count": 1,
                    "surviving_buffer_count": 7,
                }

                with self.assertRaisesRegex(RuntimeError, "buffer churn"):
                    engine._build_session_mutation_result(handoff_result)

    def test_rebuild_continuation_can_begin_second_handoff(self):
        engine = self._make_engine()
        placedb = self._make_placedb()

        def _restart_after_topology_sync(params, placedb_arg, result, pos=None, handoff_seq=None):
            def _continued_run(continued_params, continued_placedb):
                continued_placedb.current_topology_counts = lambda: {
                    "nodes": 12,
                    "pins": 24,
                    "nets": 6,
                }
                continued_placedb.current_snapshot_fingerprint = lambda counts=None: (
                    "nodes=12|pins=24|nets=6"
                )
                seq = continued_placedb.handoff_session.begin_handoff(
                    run_event={"absolute_iteration": 11, "phase": 1},
                    trigger_decision={
                        "trigger_mode": "interval",
                        "trigger_reason": "interval=11@iter=11",
                    },
                    openroad_session_identity="bridge-1",
                    pre_counts={"nodes": 12, "pins": 24, "nets": 6},
                    pre_snapshot_fingerprint="nodes=12|pins=24|nets=6",
                )
                continued_placedb.handoff_session.record_mutation_result(
                    seq,
                    {
                        "mutation_kind": "no_mutation",
                        "requires_runtimedb_rebuild": False,
                    },
                )
                continued_placedb.handoff_session.commit_continuation(
                    seq,
                    {
                        "runtimedb_rebuild_performed": False,
                    },
                )
                return len(continued_placedb.handoff_session.event_history)

            return _continued_run

        engine._restart_after_topology_sync = _restart_after_topology_sync

        event_count = self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(event_count, 2)
        self.assertEqual(placedb.handoff_session.status, "active")

    def test_rebuild_continuation_failure_records_follow_up_failure(self):
        engine = self._make_engine()
        placedb = self._make_placedb()

        def _restart_after_topology_sync(params, placedb_arg, result, pos=None, handoff_seq=None):
            def _continued_run(continued_params, continued_placedb):
                continued_placedb.current_topology_counts = lambda: {
                    "nodes": 99,
                    "pins": 199,
                    "nets": 77,
                }
                continued_placedb.current_snapshot_fingerprint = lambda counts=None: (
                    "nodes=99|pins=199|nets=77"
                )
                raise RuntimeError("continued boom")

            return _continued_run

        engine._restart_after_topology_sync = _restart_after_topology_sync

        with self.assertRaisesRegex(RuntimeError, "continued boom"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "failed")
        self.assertEqual(len(placedb.handoff_session.event_history), 2)
        self.assertEqual(placedb.handoff_session.event_history[0]["status"], "success")
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["parent_handoff_seq"],
            placedb.handoff_session.event_history[0]["handoff_seq"],
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["pre_counts"],
            {"nodes": 12, "pins": 24, "nets": 6},
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["pre_snapshot_fingerprint"],
            "nodes=12|pins=24|nets=6",
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["failure_stage"],
            "rebuild",
        )

    def test_rebuild_continuation_preserves_nested_failure_after_session_failed(self):
        engine = self._make_engine()
        placedb = self._make_placedb()

        def _restart_after_topology_sync(params, placedb_arg, result, pos=None, handoff_seq=None):
            def _continued_run(continued_params, continued_placedb):
                continued_placedb.current_topology_counts = lambda: {
                    "nodes": 12,
                    "pins": 24,
                    "nets": 6,
                }
                continued_placedb.current_snapshot_fingerprint = lambda counts=None: (
                    "nodes=12|pins=24|nets=6"
                )
                nested_seq = continued_placedb.handoff_session.begin_handoff(
                    run_event={"absolute_iteration": 11, "phase": 1},
                    trigger_decision={
                        "trigger_mode": "interval",
                        "trigger_reason": "interval=11@iter=11",
                    },
                    openroad_session_identity="bridge-1",
                    pre_counts={"nodes": 12, "pins": 24, "nets": 6},
                    pre_snapshot_fingerprint="nodes=12|pins=24|nets=6",
                )
                continued_placedb.handoff_session.fail_current_handoff(
                    nested_seq,
                    "mutation",
                    "nested mutation failed",
                )
                raise RuntimeError("nested original failure")

            return _continued_run

        engine._restart_after_topology_sync = _restart_after_topology_sync

        with self.assertRaisesRegex(RuntimeError, "nested original failure"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "failed")
        self.assertEqual(len(placedb.handoff_session.event_history), 2)
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["error_summary"],
            "nested mutation failed",
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["failure_stage"],
            "mutation",
        )

    def test_real_macroplacedb_payload_is_compatible_with_session_recording(self):
        engine = self._make_engine()
        placedb = self._make_real_macroplacedb()
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: {
            "runtimedb": "rebuilt"
        }

        restart_result = self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(restart_result, {"runtimedb": "rebuilt"})
        self.assertEqual(placedb.handoff_session.event_history[-1]["status"], "success")
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["post_counts"],
            {"nodes": 12, "pins": 24, "nets": 6},
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["added_names"],
            {"nodes": ["u10", "u11"], "pins": ["p20"], "nets": ["n5"]},
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["removed_names"],
            {"nodes": [], "pins": [], "nets": []},
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["runtimedb_rebuild_owner"],
            "dreamplace_runtimedb",
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["added_buffer_count"],
            2,
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["removed_buffer_count"],
            1,
        )
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["surviving_buffer_count"],
            7,
        )

    def test_handoff_execution_failure_marks_session_failed(self):
        engine = self._make_engine()
        placedb = self._make_placedb(handoff_error=RuntimeError("boom"))
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "failed")
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["failure_stage"],
            "mutation",
        )

    def test_mutation_failure_records_attempted_strategy_command_in_session(self):
        handoff_error = RuntimeError("openroad failed")
        handoff_error.failure_stage = "mutation"
        handoff_error.handoff_payload = {
            "buffer_insertion_strategy": "repair_design",
            "buffer_insertion_command": "repair_design -max_wire_length {10}",
        }
        engine = self._make_engine()
        placedb = self._make_placedb(handoff_error=handoff_error)
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "openroad failed"):
            self._execute_default_handoff_transaction(engine, placedb)

        event = placedb.handoff_session.event_history[-1]
        self.assertEqual(event["status"], "mutation_failed")
        self.assertEqual(event["buffer_insertion_strategy"], "repair_design")
        self.assertEqual(
            event["buffer_insertion_command"],
            "repair_design -max_wire_length {10}",
        )

    def test_session_mutation_record_failure_preserves_attempted_strategy_command(self):
        mutation_result = {
            "pre_counts": {"nodes": 10, "pins": 20, "nets": 5},
            "pre_snapshot_fingerprint": "nodes=10|pins=20|nets=5",
            "mutation_kind": "topology_changed",
            "requires_runtimedb_rebuild": True,
            "post_counts": {"nodes": 12, "pins": 24, "nets": 6},
            "post_snapshot_fingerprint": "nodes=12|pins=24|nets=6",
            "buffer_insertion_strategy": "repair_design",
            "buffer_insertion_command": "repair_design -max_wire_length {10}",
            # Missing buffer churn counts make _build_session_mutation_result fail
            # after the OpenROAD command metadata has been returned.
        }
        engine = self._make_engine()
        placedb = self._make_placedb(mutation_result=mutation_result)
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "buffer churn"):
            self._execute_default_handoff_transaction(engine, placedb)

        event = placedb.handoff_session.event_history[-1]
        self.assertEqual(event["status"], "mutation_failed")
        self.assertEqual(event["buffer_insertion_strategy"], "repair_design")
        self.assertEqual(
            event["buffer_insertion_command"],
            "repair_design -max_wire_length {10}",
        )

    def test_macroplacedb_mutation_result_failure_preserves_attempted_strategy_metadata(self):
        engine = self._make_engine()
        placedb = self._make_real_macroplacedb()
        experimental_command = (
            'repair_timing -setup -sequence "unbuffer,buffer,split" '
            "-skip_last_gasp -skip_vt_swap -skip_crit_vt_swap"
        )

        def _run_openroad_buffer_insertion(params, pos, handoff_event):
            return {
                "triggered": True,
                "executed": True,
                "requires_sync_back": True,
                "buffer_insertion_strategy": "buffer-only",
                "buffer_insertion_command": experimental_command,
                "buffer_insertion_strategy_profile": "buffer-only",
                "buffer_insertion_strategy_kind": "experimental",
                "buffer_insertion_strategy_experimental": True,
                "buffer_insertion_strategy_validation": "log_validated_only",
                "buffer_insertion_strategy_non_buffer_mutation_free": None,
                "sync_contract": {
                    "mutation_kind": "topology_changed",
                    "requires_runtimedb_rebuild": True,
                    "old_counts": {"nodes": 10, "pins": 20, "nets": 5},
                    "new_counts": {"nodes": 12, "pins": 24, "nets": 6},
                },
            }

        placedb.run_openroad_buffer_insertion = _run_openroad_buffer_insertion
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "buffer churn"):
            self._execute_default_handoff_transaction(engine, placedb)

        event = placedb.handoff_session.event_history[-1]
        self.assertEqual(event["status"], "mutation_failed")
        self.assertEqual(event["buffer_insertion_strategy"], "buffer-only")
        self.assertEqual(event["buffer_insertion_command"], experimental_command)
        self.assertEqual(event["buffer_insertion_strategy_profile"], "buffer-only")
        self.assertEqual(event["buffer_insertion_strategy_kind"], "experimental")
        self.assertTrue(event["buffer_insertion_strategy_experimental"])
        self.assertEqual(
            event["buffer_insertion_strategy_validation"],
            "log_validated_only",
        )
        self.assertIs(
            event.get("buffer_insertion_strategy_non_buffer_mutation_free"),
            None,
        )

    def test_refresh_stage_error_marks_session_failed(self):
        engine = self._make_engine()
        refresh_error = RuntimeError("refresh boom")
        refresh_error.failure_stage = "refresh"
        placedb = self._make_placedb(handoff_error=refresh_error)
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "refresh boom"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "failed")
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["failure_stage"],
            "refresh",
        )

    def test_sync_contract_refresh_failure_records_attempted_strategy_command(self):
        event = self._execute_real_refresh_failure_handoff("sync_contract")

        self.assertEqual(event["status"], "refresh_failed")
        self.assertEqual(event["failure_stage"], "refresh")
        self.assertEqual(event["buffer_insertion_strategy"], "repair_design")
        self.assertEqual(
            event["buffer_insertion_command"],
            "repair_design -max_wire_length {10}",
        )

    def test_write_def_refresh_failure_records_attempted_strategy_command(self):
        event = self._execute_real_refresh_failure_handoff("write_def")

        self.assertEqual(event["status"], "refresh_failed")
        self.assertEqual(event["failure_stage"], "refresh")
        self.assertEqual(event["buffer_insertion_strategy"], "repair_design")
        self.assertEqual(
            event["buffer_insertion_command"],
            "repair_design -max_wire_length {10}",
        )

    def test_rebuild_failure_marks_session_failed(self):
        engine = self._make_engine()
        placedb = self._make_placedb()

        def _fail_rebuild(params, placedb_arg, result, pos=None, handoff_seq=None):
            raise RuntimeError("rebuild boom")

        engine._restart_after_topology_sync = _fail_rebuild

        with self.assertRaisesRegex(RuntimeError, "rebuild boom"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "failed")
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["failure_stage"],
            "rebuild",
        )

    def test_failed_session_is_preserved_instead_of_reset(self):
        engine = self._make_engine()
        placedb = self._make_placedb()
        session = placedb.create_or_reset_handoff_session(session_id="run-1")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 5, "phase": 0},
            trigger_decision={
                "trigger_mode": "interval",
                "trigger_reason": "interval=5@iter=5",
            },
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 10, "pins": 20, "nets": 5},
            pre_snapshot_fingerprint="nodes=10|pins=20|nets=5",
        )
        session.fail_current_handoff(seq, "mutation", "boom")
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "failed"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertIs(placedb.handoff_session, session)
        self.assertEqual(placedb.handoff_session.status, "failed")

    def test_pending_session_reentry_preserves_original_begin_handoff_error(self):
        engine = self._make_engine()
        placedb = self._make_placedb()
        session = placedb.create_or_reset_handoff_session(session_id="run-1")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 5, "phase": 0},
            trigger_decision={
                "trigger_mode": "interval",
                "trigger_reason": "interval=5@iter=5",
            },
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 10, "pins": 20, "nets": 5},
            pre_snapshot_fingerprint="nodes=10|pins=20|nets=5",
        )
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "not active"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "handoff_in_progress")
        self.assertEqual(len(placedb.handoff_session.event_history), 1)
        self.assertEqual(placedb.handoff_session.event_history[-1]["handoff_seq"], seq)

    def test_begin_handoff_invariant_failure_marks_session_failed(self):
        engine = self._make_engine()
        placedb = self._make_placedb()
        session = placedb.create_or_reset_handoff_session(session_id="run-1")
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 5, "phase": 0},
            trigger_decision={
                "trigger_mode": "interval",
                "trigger_reason": "interval=5@iter=5",
            },
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 10, "pins": 20, "nets": 5},
            pre_snapshot_fingerprint="nodes=10|pins=20|nets=5",
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": False,
            },
        )
        placedb.current_topology_counts = lambda: {"nodes": 11, "pins": 22, "nets": 6}
        placedb.current_snapshot_fingerprint = lambda counts=None: "nodes=11|pins=22|nets=6"
        engine._restart_after_topology_sync = lambda params, placedb_arg, result, pos=None, handoff_seq=None: None

        with self.assertRaisesRegex(RuntimeError, "pre_counts"):
            self._execute_default_handoff_transaction(engine, placedb)

        self.assertEqual(placedb.handoff_session.status, "failed")
        self.assertEqual(
            placedb.handoff_session.event_history[-1]["failure_stage"],
            "refresh",
        )


if __name__ == "__main__":
    unittest.main()
