import unittest
from types import SimpleNamespace

from dreamplace.ops.timing_propagation.criticality import (
    build_criticality_maps_from_timing_op,
    build_criticality_maps_from_timing_outputs,
)


class TestTimingCriticalityMaps(unittest.TestCase):
    def test_builds_maps_from_timing_outputs(self):
        maps, summary = build_criticality_maps_from_timing_outputs(
            pin_slack=[0.0, -4.5, 2.0],
            endpoint_incidence_result=SimpleNamespace(
                pin_endpoint_incidence_count=[0, 3, 1],
                active_endpoint_count=2,
            ),
        )

        self.assertEqual(summary["criticality_source"], "timing_propagation")
        self.assertEqual(summary["sink_slack_map_entry_count"], 3)
        self.assertEqual(summary["npath_map_entry_count"], 3)
        self.assertEqual(summary["active_endpoint_count"], 2)
        self.assertEqual(maps["sink_slack_by_pin"], {0: 0.0, 1: -4.5, 2: 2.0})
        self.assertEqual(maps["npath_by_pin"], {0: 0, 1: 3, 2: 1})

    def test_derives_active_endpoints_from_negative_slack(self):
        calls = []

        def compute_active_endpoint_incidence(active_endpoint_ids=None):
            calls.append(list(active_endpoint_ids or []))
            return SimpleNamespace(
                pin_endpoint_incidence_count=[0, 2, 1, 2],
                active_endpoint_count=len(active_endpoint_ids or []),
            )

        timing_op = SimpleNamespace(
            get_pin_slack=lambda: [1.0, -2.0, 3.0, -0.5],
            last_endpoint_ids_tensor=[1, 3],
            last_endpoint_slack_tensor=[-2.0, -0.5],
            last_active_endpoint_ids=[],
            compute_active_endpoint_incidence=compute_active_endpoint_incidence,
        )

        maps, summary = build_criticality_maps_from_timing_op(timing_op)

        self.assertEqual(calls, [[1, 3]])
        self.assertEqual(summary["npath_map_entry_count"], 4)
        self.assertEqual(summary["active_endpoint_count"], 2)
        self.assertEqual(maps["npath_by_pin"], {0: 0, 1: 2, 2: 1, 3: 2})


if __name__ == "__main__":
    unittest.main()
