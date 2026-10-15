import argparse
import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.openroad_handoff import launcher
sys.path.pop()


class OpenRoadHandoffLauncherTest(unittest.TestCase):
    def test_buffer_only_repair_command_includes_margin_and_buffer_sequence(self):
        command = launcher.buffer_only_repair_command(0.125)

        self.assertIn("repair_timing -setup", command)
        self.assertIn("-setup_margin 0.125", command)
        self.assertIn('-sequence "unbuffer,buffer,split"', command)
        self.assertIn("-skip_pin_swap", command)
        self.assertIn("-skip_gate_cloning", command)
        self.assertIn("-skip_size_down", command)

    def test_optional_gate_float_accepts_none_aliases(self):
        self.assertIsNone(launcher.optional_gate_float("none"))
        self.assertIsNone(launcher.optional_gate_float("off"))
        self.assertIsNone(launcher.optional_gate_float("disabled"))
        self.assertEqual(launcher.optional_gate_float("0.2"), 0.2)
        with self.assertRaises(argparse.ArgumentTypeError):
            launcher.optional_gate_float("bad")

    def test_build_buffer_only_config_sets_interval_guard_and_write_def(self):
        config = launcher.build_openroad_handoff_config(
            output_def="/tmp/final.def",
            rc_tcl="/tmp/setRC.tcl",
            handoff_mode="buffer-only",
            trigger_period=50,
            max_handoff_overflow=0.2,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(config["trigger"]["mode"], "interval")
        self.assertEqual(config["trigger"]["interval"], 50)
        self.assertEqual(
            config["trigger"]["guards"],
            [{"name": "overflow", "op": "<=", "value": 0.2}],
        )
        self.assertEqual(config["trigger"]["dedupe_by"], "absolute_iteration")
        self.assertEqual(config["buffer_insertion"]["strategy"], "buffer-only")
        self.assertEqual(config["buffer_insertion"]["write_def_after"], "/tmp/final.def")
        self.assertEqual(
            config["buffer_insertion"]["options"]["pre_repair_tcl"],
            ["source /tmp/setRC.tcl", "estimate_parasitics -placement"],
        )

    def test_build_buffer_only_config_accepts_custom_pre_repair_tcl(self):
        config = launcher.build_openroad_handoff_config(
            output_def="/tmp/final.def",
            rc_tcl=None,
            handoff_mode="buffer-only",
            trigger_period=50,
            max_handoff_overflow=None,
            pre_repair_tcl=["source /tmp/custom.tcl"],
        )

        self.assertEqual(
            config["buffer_insertion"]["options"]["pre_repair_tcl"],
            ["source /tmp/custom.tcl"],
        )

    def test_build_no_handoff_config_disables_controller(self):
        config = launcher.build_openroad_handoff_config(
            output_def="/tmp/final.def",
            rc_tcl="/tmp/setRC.tcl",
            handoff_mode="no-handoff",
            trigger_period=50,
            max_handoff_overflow=0.2,
        )

        self.assertEqual(config, {"enabled": False, "trigger": {"mode": "disabled"}})

    def test_install_margin_patch_updates_buffer_only_aliases(self):
        aliases = {
            "buffer-only": {"command": "old"},
            "buffer_only": {"command": "old"},
        }
        fake_place_io = types.SimpleNamespace(
            _BUFFER_INSERTION_STRATEGY_ALIASES=aliases
        )
        with patch.dict(
            sys.modules,
            {"dreamplace.ops.placeio_openroad.place_io": fake_place_io},
        ):
            command = launcher.install_buffer_only_margin_command_patch(0.25)

        self.assertEqual(aliases["buffer-only"]["command"], command)
        self.assertEqual(aliases["buffer_only"]["command"], command)
        self.assertIn("-setup_margin 0.25", command)


if __name__ == "__main__":
    unittest.main()
