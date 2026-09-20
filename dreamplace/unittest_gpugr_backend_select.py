import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.gpugr.xplace_backend import (  # noqa: E402
    XplaceGPUGR,
    normalize_gpugr_backend,
    resolve_gpugr_backend,
    validate_gpugr_backend_request,
)


class GPUGRBackendSelectTest(unittest.TestCase):
    def test_normalize_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            normalize_gpugr_backend("gpu")

    def test_auto_prefers_cuda_when_ready(self):
        self.assertEqual(
            resolve_gpugr_backend("auto", cuda_available=True, extension_cuda_enabled=True),
            "cuda",
        )

    def test_auto_falls_back_to_cpu_pr_without_cuda_extension(self):
        self.assertEqual(
            resolve_gpugr_backend("auto", cuda_available=True, extension_cuda_enabled=False),
            "cpu_pr",
        )
        self.assertEqual(
            resolve_gpugr_backend("auto", cuda_available=False, extension_cuda_enabled=True),
            "cpu_pr",
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

    def test_explicit_cugr2_does_not_depend_on_cuda_readiness(self):
        self.assertEqual(
            resolve_gpugr_backend("cugr2", cuda_available=False, extension_cuda_enabled=False),
            "cugr2",
        )

    def test_explicit_cugr_does_not_depend_on_cuda_readiness(self):
        self.assertEqual(
            resolve_gpugr_backend("cugr", cuda_available=False, extension_cuda_enabled=False),
            "cugr",
        )

    def test_factory_returns_cugr_operator_without_fallback(self):
        from types import SimpleNamespace

        from dreamplace.ops.gpugr.backend_select import create_gpugr_backend
        from dreamplace.ops.gpugr.cugr_backend import CugrGPUGR

        operator = create_gpugr_backend(SimpleNamespace(gpugr_backend="cugr"), object())
        self.assertIsInstance(operator, CugrGPUGR)

    def test_factory_routes_cugr2_to_xplace_without_fallback(self):
        from types import SimpleNamespace

        from dreamplace.ops.gpugr.backend_select import create_gpugr_backend

        operator = create_gpugr_backend(SimpleNamespace(gpugr_backend="cugr2"), object())
        self.assertIsInstance(operator, XplaceGPUGR)

    def test_cpu_pr_rejects_rrr_iters(self):
        validate_gpugr_backend_request("cpu_pr", 0)
        with self.assertRaises(RuntimeError):
            validate_gpugr_backend_request("cpu_pr", 1)
        validate_gpugr_backend_request("cuda", 1)

    def test_cugr2_rejects_rrr_iters(self):
        validate_gpugr_backend_request("cugr2", 0)
        with self.assertRaisesRegex(RuntimeError, "backend=cugr2"):
            validate_gpugr_backend_request("cugr2", 1)

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
