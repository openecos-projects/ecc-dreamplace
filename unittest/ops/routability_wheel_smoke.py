"""Run release regressions against an installed wheel, outside its source tree."""

import argparse
import os
import sys
import sysconfig
import unittest
from pathlib import Path
from types import SimpleNamespace

TEST_FILES = (
    "nonlinear_place_plain_flow_unittest",
    "routability_controller_unittest",
    "routability_l_shape_inputs_unittest",
    "routability_l_shape_capacity_al_unittest",
    "routability_l_shape_segment_compaction_unittest",
    "steiner_topo_ggr_l_shape_topology_pack_unittest",
    "xplace_parser_cache_unittest",
    "inflation_target_density_sync_unittest",
    "gpugr_backend_select_unittest",
    "gpugr_route_grid_unittest",
    "gpugr_congestion_maps_unittest",
    "gpugr_routing_layers_unittest",
    "cell_padding_geometry_unittest",
    "post_legalization_adaptive_padding_unittest",
    "gpugr_evidence_unittest",
    "params_l_shape_preset_unittest",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    os.environ.pop("ECC_XPLACE_ROOT", None)

    # Load the installed package before tests that add their source root to sys.path.
    import dreamplace
    import torch
    from dreamplace import configure
    from dreamplace.NonLinearPlace import NonLinearPlace  # noqa: F401
    from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR

    package = Path(dreamplace.__file__).resolve()
    installed_root = Path(sysconfig.get_path("platlib")).resolve()
    assert package.parent == installed_root / "dreamplace", package
    assert torch.cuda.is_available() or not args.cuda, "CUDA runtime is required"
    assert (configure.compile_configurations["CUDA_FOUND"] == "TRUE") == args.cuda
    operator = XplaceGPUGR(SimpleNamespace(), SimpleNamespace())
    runtime = operator._ensure_xplace_python_path()
    assert runtime == installed_root / "thirdparty" / "xplace", runtime
    _, gpugr, _, _, _ = operator._import_xplace_modules()
    assert bool(gpugr.cuda_enabled()) == args.cuda
    if not args.cuda:
        extensions = runtime / "cpp_to_py" / "cpybin"
        assert not list(extensions.glob("*cuda*.so")), extensions
        assert not list(extensions.glob("gpudp*.so")), extensions
    os.environ["XPLACE_CPU_RUNTIME_ROOT"] = str(runtime)

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    test_dir = Path(__file__).resolve().parent
    for name in TEST_FILES:
        tests = loader.discover(str(test_dir), pattern=f"{name}.py")
        assert tests.countTestCases(), f"No release tests found for {name}"
        suite.addTests(tests)
    if not args.cuda:
        native_tests = test_dir.parents[1] / "thirdparty" / "xplace" / "cpp_to_py"
        for folder, filename in (
            ("gpugr", "unittest_cpu_pattern_route.py"),
            ("gpugr", "unittest_portless_io_pin_grnet.py"),
            ("io_parser", "unittest_portless_lef_pin.py"),
        ):
            tests = unittest.TestLoader().discover(str(native_tests / folder), pattern=filename)
            assert tests.countTestCases(), f"No native release tests found for {filename}"
            suite.addTests(tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    for name, module in sys.modules.copy().items():
        if name.startswith("dreamplace.") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(package.parent), module.__file__
    print(
        f"installed_routability_wheel_ok cuda={args.cuda} tests={result.testsRun} runtime={runtime}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
