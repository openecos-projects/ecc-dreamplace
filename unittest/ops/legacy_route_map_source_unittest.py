import sys
import types
import unittest

from dreamplace.NonLinearPlace import (
    _ensure_modularity_inflation_contract,
    _resolve_route_map_source,
)


class LegacyRouteMapSourceTest(unittest.TestCase):
    def test_legacy_area_key_selects_ecc_egr(self):
        params = types.SimpleNamespace(
            adjust_gpugr_area_flag=0,
            adjust_nctugr_area_flag=1,
        )

        with self.assertLogs(level="INFO") as logs:
            source = _resolve_route_map_source(params)

        self.assertEqual(source, "irt_egr")
        self.assertIn("NCTUgr is not invoked", "\n".join(logs.output))
        self.assertNotIn("nctugr_binary", sys.modules)
        self.assertNotIn("place_io", sys.modules)

    def test_gpugr_takes_precedence_over_legacy_key(self):
        params = types.SimpleNamespace(
            adjust_gpugr_area_flag=1,
            adjust_nctugr_area_flag=1,
        )

        self.assertEqual(_resolve_route_map_source(params), "gpugr")

    def test_modularity_rejects_legacy_route_key(self):
        params = types.SimpleNamespace(
            modularity_inflation_flag=1,
            routability_opt_flag=1,
            modularity_require_gpugr_flag=1,
            adjust_gpugr_area_flag=1,
            adjust_nctugr_area_flag=1,
            adjust_rudy_area_flag=0,
        )

        with self.assertRaisesRegex(
            RuntimeError, "does not support adjust_nctugr_area_flag"
        ):
            _ensure_modularity_inflation_contract(params, route_map_source="irt_egr")


if __name__ == "__main__":
    unittest.main()
