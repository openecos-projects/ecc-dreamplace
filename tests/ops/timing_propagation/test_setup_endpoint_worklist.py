import json
import tempfile
import unittest
from pathlib import Path

from dreamplace.ops.timing_propagation.setup_endpoint_worklist import (
    collect_opensta_setup_endpoint_worklist,
    parse_report_checks_setup_endpoints,
    write_setup_endpoint_worklist,
)


class TestSetupEndpointWorklist(unittest.TestCase):
    def test_parses_report_checks_endpoints_and_slack(self):
        records = parse_report_checks_setup_endpoints(
            """
Startpoint: in0
Endpoint: u1/D (rising edge-triggered flip-flop clocked by clk)
Path Group: clk
slack (VIOLATED) -0.125

Startpoint: in1
Endpoint: u2/D
slack (MET) 0.250
"""
        )

        self.assertEqual(records[0], {"pin_name": "u1/D", "slack": -0.125, "npath": 1})
        self.assertEqual(records[1], {"pin_name": "u2/D", "slack": 0.25, "npath": 1})

    def test_parses_opensta_numeric_slack_prefix_format(self):
        records = parse_report_checks_setup_endpoints(
            """
Endpoint: r7_t1_t2_s0_out_reg[5]
          (rising edge-triggered flip-flop clocked by clk)
-----------------------------------------------------------------
            -266.214508   slack (VIOLATED)
"""
        )

        self.assertEqual(
            records,
            [
                {
                    "pin_name": "r7_t1_t2_s0_out_reg[5]",
                    "slack": -266.214508,
                    "npath": 1,
                }
            ],
        )

    def test_prefers_endpoint_pin_from_path_detail_when_endpoint_is_instance(self):
        records = parse_report_checks_setup_endpoints(
            """
Endpoint: r7_t1_t2_s0_out_reg[5]
          (rising edge-triggered flip-flop clocked by clk)
  42.672642  797.123901 ^ r7_t1_t2_s0_g30102/Y (NOR4xp25_ASAP7_75t_R)
   0.334344  797.458252 ^ r7_t1_t2_s0_out_reg[5]/D (DFFHQNx1_ASAP7_75t_R)
             797.458252   data arrival time
             550.000000 ^ r7_t1_t2_s0_out_reg[5]/CLK (DFFHQNx1_ASAP7_75t_R)
            -266.214508   slack (VIOLATED)
"""
        )

        self.assertEqual(records[0]["pin_name"], "r7_t1_t2_s0_out_reg[5]/D")
        self.assertEqual(records[0]["slack"], -266.214508)

    def test_aggregates_duplicate_endpoints_by_worst_slack_and_path_count(self):
        records = parse_report_checks_setup_endpoints(
            """
Endpoint: u1/D
slack (VIOLATED) -0.125
Endpoint: u1/D
slack (VIOLATED) -0.300
Endpoint: u2/D
slack (VIOLATED) -0.050
"""
        )

        self.assertEqual(records[0], {"pin_name": "u1/D", "slack": -0.3, "npath": 2})
        self.assertEqual(records[1], {"pin_name": "u2/D", "slack": -0.05, "npath": 1})

    def test_limits_output_after_aggregation(self):
        records = parse_report_checks_setup_endpoints(
            """
Endpoint: u1/D
slack (VIOLATED) -0.100
Endpoint: u2/D
slack (VIOLATED) -0.300
Endpoint: u3/D
slack (VIOLATED) -0.200
""",
            max_endpoints=2,
        )

        self.assertEqual([record["pin_name"] for record in records], ["u2/D", "u3/D"])

    def test_writes_setup_endpoint_json_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "setup_endpoints.json"
            summary = write_setup_endpoint_worklist(
                [
                    {"pin_name": "u1/D", "slack": -0.125, "npath": 2},
                ],
                path,
            )

            payload = json.loads(path.read_text())
            self.assertEqual(
                payload,
                {
                    "setup_endpoints": [
                        {"pin_name": "u1/D", "slack": -0.125, "npath": 2},
                    ]
                },
            )
            self.assertEqual(summary["setup_endpoint_count"], 1)
            self.assertEqual(summary["setup_endpoint_json_path"], str(path))

    def test_collects_opensta_report_into_artifacts(self):
        class FakeRawDb:
            def __init__(self):
                self.commands = []

            def eval_tcl_string(self, command):
                self.commands.append(command)
                report_path = command.split("{", 1)[1].split("}", 1)[0]
                Path(report_path).write_text(
                    """
Endpoint: u2/D
slack (VIOLATED) -0.300
Endpoint: u1/D
slack (VIOLATED) -0.100
"""
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_db = FakeRawDb()
            summary = collect_opensta_setup_endpoint_worklist(
                raw_db,
                output_dir=tmpdir,
                group_count=7,
                endpoint_count=2,
                max_endpoints=1,
            )

            self.assertEqual(summary["status"], "ok")
            self.assertIn("report_checks -path_delay max", raw_db.commands[0])
            self.assertIn("-group_count 7", raw_db.commands[0])
            self.assertIn("-endpoint_count 2", raw_db.commands[0])
            payload = json.loads(Path(summary["setup_endpoint_json_path"]).read_text())
            self.assertEqual(payload["setup_endpoints"][0]["pin_name"], "u2/D")
            self.assertEqual(summary["setup_endpoint_count"], 1)
            self.assertTrue(Path(summary["summary_path"]).exists())


if __name__ == "__main__":
    unittest.main()
