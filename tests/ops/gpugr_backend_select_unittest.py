import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.gpugr import xplace_backend  # noqa: E402
from dreamplace.ops.gpugr.xplace_backend import (  # noqa: E402
    XplaceGPUGR,
    normalize_gpugr_backend,
    resolve_gpugr_backend,
    validate_gpugr_backend_request,
)


class GPUGRBackendSelectTest(unittest.TestCase):
    def test_optional_route_imports_restore_process_modules(self):
        real_parent = types.ModuleType("torchvision")
        for fail in (False, True):
            with self.subTest(fail=fail), mock.patch.dict(
                sys.modules, {"torchvision": real_parent}
            ), mock.patch(
                "importlib.import_module", side_effect=ImportError("fixture missing dependency")
            ):
                before = {name: sys.modules.get(name) for name in (
                    "seaborn", "torchvision", "torchvision.transforms"
                )}
                try:
                    with xplace_backend._route_force_import_dependencies():
                        self.assertIsNot(sys.modules["torchvision"], real_parent)
                        self.assertIs(
                            sys.modules["torchvision"].transforms,
                            sys.modules["torchvision.transforms"],
                        )
                        if fail:
                            raise RuntimeError("fixture helper import failed")
                except RuntimeError as exc:
                    self.assertEqual(str(exc), "fixture helper import failed")
                self.assertEqual(
                    {name: sys.modules.get(name) for name in before}, before
                )
                self.assertFalse(hasattr(real_parent, "transforms"))

    def test_installed_runtime_precedes_stale_checkout_extensions(self):
        with tempfile.TemporaryDirectory(prefix="xplace_runtime_lookup_") as tmp:
            root = Path(tmp)
            checkout = root / "checkout"
            platlib = root / "site-packages"
            installed = platlib / "thirdparty/xplace"
            source = checkout / "thirdparty/xplace"
            for candidate in (installed, source):
                extensions = candidate / "cpp_to_py/cpybin"
                extensions.mkdir(parents=True)
                (extensions / "gpugr.so").touch()
            with (
                mock.patch.dict(os.environ, {"ECC_XPLACE_ROOT": ""}),
                mock.patch.object(xplace_backend.sysconfig, "get_path", return_value=str(platlib)),
                mock.patch.object(
                    xplace_backend,
                    "__file__",
                    str(checkout / "dreamplace/ops/gpugr/xplace_backend.py"),
                ),
                mock.patch.object(sys, "path", list(sys.path)),
            ):
                operator = XplaceGPUGR(SimpleNamespace(), SimpleNamespace())
                self.assertEqual(operator._ensure_xplace_python_path(), installed)
                self.assertEqual(sys.path[0], str(installed))
                os.environ["ECC_XPLACE_ROOT"] = str(source)
                operator = XplaceGPUGR(SimpleNamespace(), SimpleNamespace())
                self.assertEqual(operator._ensure_xplace_python_path(), source)

    def test_missing_extensions_report_the_incomplete_runtime(self):
        with tempfile.TemporaryDirectory(prefix="xplace_runtime_missing_") as tmp:
            root = Path(tmp)
            (root / "cpp_to_py").mkdir()
            with mock.patch.object(
                XplaceGPUGR, "_xplace_root_candidates", return_value=iter([root])
            ):
                operator = XplaceGPUGR(SimpleNamespace(), SimpleNamespace())
                with self.assertRaisesRegex(RuntimeError, "Xplace gpugr extensions are not built"):
                    operator._import_xplace_modules()

    def test_normalize_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            normalize_gpugr_backend("gpu")

    def test_auto_uses_parallel_cpu_unless_cuda_is_ready(self):
        for cuda_available, extension_cuda_enabled, expected in (
            (True, True, "cuda"),
            (True, False, "cpu_pr_mt"),
            (False, True, "cpu_pr_mt"),
            (False, False, "cpu_pr_mt"),
        ):
            with self.subTest(
                cuda_available=cuda_available,
                extension_cuda_enabled=extension_cuda_enabled,
            ):
                self.assertEqual(
                    resolve_gpugr_backend(
                        "auto",
                        cuda_available=cuda_available,
                        extension_cuda_enabled=extension_cuda_enabled,
                    ),
                    expected,
                )

    def test_explicit_cuda_requires_cuda_readiness(self):
        self.assertEqual(
            resolve_gpugr_backend("cuda", cuda_available=True, extension_cuda_enabled=True),
            "cuda",
        )
        with self.assertRaises(RuntimeError):
            resolve_gpugr_backend("cuda", cuda_available=True, extension_cuda_enabled=False)
        with self.assertRaises(RuntimeError):
            resolve_gpugr_backend("cuda", cuda_available=False, extension_cuda_enabled=True)

    def test_explicit_cpu_pr_always_resolves(self):
        self.assertEqual(
            resolve_gpugr_backend("cpu_pr", cuda_available=False, extension_cuda_enabled=False),
            "cpu_pr",
        )

    def test_explicit_cpu_pr_mt_always_resolves(self):
        self.assertEqual(normalize_gpugr_backend("cpu_pr_mt"), "cpu_pr_mt")
        self.assertEqual(
            resolve_gpugr_backend("cpu_pr_mt", cuda_available=False, extension_cuda_enabled=False),
            "cpu_pr_mt",
        )

    def test_cpu_pr_rejects_rrr_iters(self):
        validate_gpugr_backend_request("cpu_pr", 0)
        with self.assertRaises(RuntimeError):
            validate_gpugr_backend_request("cpu_pr", 1)
        validate_gpugr_backend_request("cuda", 1)

    def test_cpu_pr_mt_rejects_rrr_iters(self):
        validate_gpugr_backend_request("cpu_pr_mt", 0)
        with self.assertRaisesRegex(RuntimeError, "backend=cpu_pr_mt"):
            validate_gpugr_backend_request("cpu_pr_mt", 1)

    def test_cpu_only_extension_stubs_cuda_dct_with_cuda_torch(self):
        fake_gpugr = types.SimpleNamespace(cuda_enabled=lambda: False)
        fake_cpp_to_py = types.ModuleType("cpp_to_py")
        fake_cpp_to_py.gpugr = fake_gpugr

        with tempfile.TemporaryDirectory(prefix="xplace_cpu_loader_") as tmp:
            root = Path(tmp)
            (root / "utils").mkdir()
            (root / "src/core").mkdir(parents=True)
            (root / "utils/io_parser.py").write_text("class IOParser: pass\n")
            (root / "src/core/flute.py").write_text("class Flute: pass\n")
            (root / "src/core/route_force.py").write_text(
                "from .dct2_fft2 import dct2\n"
                "def calc_gr_wl_via(*args): return (0, 0)\n"
                "def estimate_num_shorts(*args): return 0\n"
            )

            with (
                mock.patch.dict(sys.modules, {"cpp_to_py": fake_cpp_to_py}),
                mock.patch("torch.backends.cuda.is_built", return_value=True),
            ):
                _, loaded_gpugr, _, _, _ = XplaceGPUGR._load_xplace_python_modules(root)

        self.assertIs(loaded_gpugr, fake_gpugr)


if __name__ == "__main__":
    unittest.main()
