import unittest
from types import SimpleNamespace

import torch

from dreamplace.ops.gpugr.xplace_parser_cache import XplaceParserCacheMixin
from dreamplace.ops.routability.gpugr_context import invalidate_route_state


class _FakeGPDB:
    def __init__(self, node_lpos):
        self._node_lpos = torch.tensor(node_lpos, dtype=torch.float32)

    def node_lpos_tensor(self):
        return self._node_lpos


class XplaceParserCacheTest(unittest.TestCase):
    def test_master_refresh_invalidates_unchanged_node_set_and_retained_operator(self):
        mixin = XplaceParserCacheMixin()
        mixin.params = SimpleNamespace(result_dir="/tmp/route-refresh", place_io_engine="ecc")
        mixin.placedb = SimpleNamespace(topology_generation=0)
        mixin._resolve_lefs = lambda: ["/tmp/cells.lef"]
        names = ("U1", "U2")
        key = mixin._build_parser_cache_key("custom", "gcd", mixin._resolve_lefs())
        mixin._parser_db_cache = {"key": key, "node_names": names, "node_count": 2}
        self.assertTrue(mixin.parser_cache_would_hit("custom", "gcd", names, 2))
        mixin.placedb.topology_generation += 1
        self.assertFalse(mixin.parser_cache_would_hit("custom", "gcd", names, 2))
        mixin.placedb._autodmp_gpugr_op = mixin
        mixin.placedb._gpugr_topology_pin_name_to_id_cache = {"old": 0}
        invalidate_route_state(mixin.placedb)
        self.assertIsNone(mixin._parser_db_cache)
        self.assertEqual(vars(mixin.placedb), {"topology_generation": 1})

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
