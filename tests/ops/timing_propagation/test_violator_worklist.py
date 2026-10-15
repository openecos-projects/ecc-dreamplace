import unittest
from types import SimpleNamespace


class ViolatorWorklistTest(unittest.TestCase):
    def test_parses_check_type_violators_from_timing_side_owner(self):
        from dreamplace.ops.timing_propagation.violator_worklist import (
            parse_report_check_types_violators,
        )

        records = parse_report_check_types_violators(
            """
max capacitance

Pin                                    Limit    Cap    Slack
------------------------------------------------------------
g1/A                                  20.00  41.25  -21.25 (VIOLATED)
g2/B                                  20.00  19.00    1.00
"""
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["pin_name"], "g1/A")
        self.assertEqual(records[0]["limit"], 20.0)
        self.assertEqual(records[0]["value"], 41.25)
        self.assertEqual(records[0]["slack"], -21.25)

    def test_maps_report_pin_aliases_to_pydb_nets(self):
        from dreamplace.ops.timing_propagation.violator_worklist import (
            map_violator_pins_to_nets,
        )

        pydb = SimpleNamespace(
            pin_names=["drv:Y", "g1:A", "r0[3]:D"],
            pin2net_map=[0, 1, 2],
            net_names=["driver_net", "n_critical", "n_reg"],
        )

        mapped = map_violator_pins_to_nets(
            pydb,
            [
                {"pin_name": "g1/A", "slack": -1.0},
                {"pin_name": "r0[3]/D", "slack": -2.0},
            ],
            check_type="max_slew",
        )

        self.assertEqual([record["net_name"] for record in mapped], ["n_critical", "n_reg"])
        self.assertEqual([record["pin_id"] for record in mapped], [1, 2])


if __name__ == "__main__":
    unittest.main()
