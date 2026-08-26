import unittest

import torch

from dreamplace.ops.gpugr.xplace_parser_cache import XplaceParserCacheMixin


class _FakeGPDB:
    def __init__(self, node_lpos):
        self._node_lpos = torch.tensor(node_lpos, dtype=torch.float32)

    def node_lpos_tensor(self):
        return self._node_lpos


class XplaceParserCacheTest(unittest.TestCase):
    def test_validation_accepts_def_integer_rounding(self):
        mixin = XplaceParserCacheMixin()
        mixin._validate_parser_cache_node_lpos(
            _FakeGPDB([[100.0, 200.0]]),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[101.0, 200.0]], dtype=torch.float32),
            "test",
        )

    def test_validation_rejects_larger_coordinate_mismatch(self):
        mixin = XplaceParserCacheMixin()
        with self.assertRaises(RuntimeError):
            mixin._validate_parser_cache_node_lpos(
                _FakeGPDB([[100.0, 200.0]]),
                torch.tensor([0], dtype=torch.long),
                torch.tensor([[101.01, 200.0]], dtype=torch.float32),
                "test",
            )


if __name__ == "__main__":
    unittest.main()
