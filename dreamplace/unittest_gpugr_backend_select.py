import os
import sys
import unittest

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.gpugr.xplace_backend import (  # noqa: E402
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

    def test_cpu_pr_rejects_rrr_iters(self):
        validate_gpugr_backend_request("cpu_pr", 0)
        with self.assertRaises(RuntimeError):
            validate_gpugr_backend_request("cpu_pr", 1)
        validate_gpugr_backend_request("cuda", 1)


if __name__ == "__main__":
    unittest.main()
