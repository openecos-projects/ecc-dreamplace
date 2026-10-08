#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("campaign_protocol.py")
SPEC = importlib.util.spec_from_file_location("campaign_protocol", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def _source_identity(seed: str = "a") -> dict:
    payload = {
        "artifact": protocol.SOURCE_IDENTITY_ARTIFACT,
        "artifact_version": protocol.SOURCE_IDENTITY_VERSION,
        "autodmp_root": "/autodmp",
        "git_head": seed * 64,
        "tracked_dirty": True,
        "tracked_diff_sha256": "b" * 64,
        "source_file_sha256": {"runner.py": "c" * 64},
        "native_library_sha256": {"timing.so": "d" * 64},
    }
    payload["identity_digest"] = protocol.digest_payload(
        protocol._source_identity_payload(payload)
    )
    return payload


class CampaignProtocolTest(unittest.TestCase):
    @staticmethod
    def _profile_path(version: int) -> Path:
        return (
            MODULE_PATH.with_name("profiles")
            / f"timing_physical_synthesis_v{version}.json"
        )

    def _fixture_root(self, root: Path, case: str = "aes_256") -> Path:
        design = root / "design" / case
        design.mkdir(parents=True)
        (design / f"{case}.def").write_text("VERSION 5.8 ;\n", encoding="utf-8")
        (design / f"{case}.v").write_text("module top; endmodule\n", encoding="utf-8")
        (design / f"{case}.sdc").write_text("create_clock -period 1 [get_ports clk]\n", encoding="utf-8")
        asap7 = root / "ASAP7"
        for relative in (
            "lef/asap7_tech_1x_201209.lef",
            "lef/asap7sc7p5t_27_R_1x_201211.lef",
            "lef/sram_asap7_16x256_1rw.lef",
            "lef/sram_asap7_32x256_1rw.lef",
            "lef/sram_asap7_64x256_1rw.lef",
            "lef/sram_asap7_64x64_1rw.lef",
            "lib/asap7sc7p5t_AO_RVT_FF_nldm_201020.lib",
            "lib/asap7sc7p5t_INVBUF_RVT_FF_nldm_201020.lib",
            "lib/asap7sc7p5t_OA_RVT_FF_nldm_201020.lib",
            "lib/asap7sc7p5t_SEQ_RVT_FF_nldm_201020.lib",
            "lib/asap7sc7p5t_SIMPLE_RVT_FF_nldm_201020.lib",
            "lib/sram_asap7_16x256_1rw.lib",
            "lib/sram_asap7_32x256_1rw.lib",
            "lib/sram_asap7_64x256_1rw.lib",
            "lib/sram_asap7_64x64_1rw.lib",
            "setRC.tcl",
        ):
            path = asap7 / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        return root

    def test_case_paths_use_design_case_layout(self):
        root = Path("/tmp") / "iccad24-benchmark-fixture"
        paths = protocol.case_paths(root, "ariane136")
        self.assertEqual(
            paths["def"], root / "design" / "ariane136" / "ariane136.def"
        )
        self.assertNotEqual(paths["def"], root / "ariane136.def")

    def test_benchmark_manifest_records_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_root(Path(tmp))
            manifest = protocol.validate_benchmark_inputs(root, ("aes_256",))
        self.assertEqual(manifest["cases"], ["aes_256"])
        self.assertEqual(
            manifest["case_inputs"]["aes_256"]["def"]["relative_to_benchmark_root"],
            "design/aes_256/aes_256.def",
        )
        self.assertEqual(
            manifest["case_inputs"]["aes_256"]["def"]["resolved_path"],
            manifest["case_inputs"]["aes_256"]["def"]["path"],
        )
        self.assertEqual(len(manifest["technology_inputs"]["libs"]), 9)

    def test_profile_rejects_flattened_design_inputs_and_fillers(self):
        profile_path = self._profile_path(2)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        flattened = copy.deepcopy(profile)
        flattened["benchmark_source"]["required_design_files"] = [
            "{case}.def",
            "{case}.v",
            "{case}.sdc",
        ]
        with self.assertRaisesRegex(ValueError, "design input layout"):
            protocol.validate_profile(flattened)
        with_fillers = copy.deepcopy(profile)
        with_fillers["initial_state"]["R0"]["enable_fillers"] = True
        with self.assertRaisesRegex(ValueError, "filler-free"):
            protocol.validate_profile(with_fillers)

    def test_v2_and_v3_profiles_validate_under_version_dispatch(self):
        v2 = protocol.load_profile(self._profile_path(2))
        v3 = protocol.load_profile(self._profile_path(3))

        self.assertEqual(v2["artifact_version"], 2)
        self.assertEqual(v2["profile_name"], "timing_physical_synthesis_v2")
        self.assertEqual(v3["artifact_version"], 3)
        self.assertEqual(v3["profile_name"], "timing_physical_synthesis_v3")
        self.assertEqual(v3["placement"]["pin2pin_activation_overflow"], 0.35)
        self.assertEqual(
            v3["coordinated_milestones"]["thresholds"],
            [0.30, 0.25, 0.20, 0.15, 0.10],
        )
        self.assertEqual(v3["coordinated_milestones"]["sizing_rounds"], 5)
        self.assertEqual(v3["coordinated_milestones"]["buffering_rounds"], 1)
        self.assertEqual(v3["mixed_terminal_refinement"]["sizing_iterations"], 50)

    def test_v3_profile_rejects_inconsistent_schedule_relationships(self):
        profile = json.loads(self._profile_path(3).read_text(encoding="utf-8"))

        ascending = copy.deepcopy(profile)
        ascending["coordinated_milestones"]["thresholds"] = [0.30, 0.31]
        with self.assertRaisesRegex(ValueError, "strictly descending"):
            protocol.validate_profile(ascending)

        above_activation = copy.deepcopy(profile)
        above_activation["coordinated_milestones"]["thresholds"][0] = 0.36
        with self.assertRaisesRegex(ValueError, "activation overflow"):
            protocol.validate_profile(above_activation)

        mismatched_tail = copy.deepcopy(profile)
        mismatched_tail["mixed_terminal_refinement"]["sizing_iterations"] = 51
        with self.assertRaisesRegex(ValueError, "terminal sizing"):
            protocol.validate_profile(mismatched_tail)

    def test_v3_profile_produces_a_distinct_campaign_identity(self):
        v2_path = self._profile_path(2)
        v3_path = self._profile_path(3)
        benchmark = {"artifact": "fixture", "cases": ["aes_256"]}

        v2_manifest = protocol.build_campaign_manifest(
            profile=protocol.load_profile(v2_path),
            benchmark_manifest=benchmark,
            profile_path=v2_path,
            source_identity=_source_identity(),
        )
        v3_manifest = protocol.build_campaign_manifest(
            profile=protocol.load_profile(v3_path),
            benchmark_manifest=benchmark,
            profile_path=v3_path,
            source_identity=_source_identity(),
        )

        self.assertNotEqual(v2_manifest["profile_digest"], v3_manifest["profile_digest"])
        self.assertNotEqual(v2_manifest["campaign_id"], v3_manifest["campaign_id"])

    def test_source_identity_is_part_of_campaign_identity(self):
        profile_path = self._profile_path(3)
        profile = protocol.load_profile(profile_path)
        benchmark = {"artifact": "fixture", "cases": ["aes_256"]}

        first = protocol.build_campaign_manifest(
            profile=profile,
            benchmark_manifest=benchmark,
            profile_path=profile_path,
            source_identity=_source_identity("a"),
        )
        second = protocol.build_campaign_manifest(
            profile=profile,
            benchmark_manifest=benchmark,
            profile_path=profile_path,
            source_identity=_source_identity("e"),
        )

        self.assertNotEqual(first["source_identity_digest"], second["source_identity_digest"])
        self.assertNotEqual(first["campaign_id"], second["campaign_id"])

    def test_dpost_manifest_uses_packaged_def_and_output_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._fixture_root(Path(tmp))
            benchmark = protocol.validate_benchmark_inputs(root, ("aes_256",))
            dpost = protocol.build_dpost_manifest(
                benchmark,
                Path(tmp) / "run" / "initial_state" / "D_post",
            )
        record = dpost["cases"]["aes_256"]
        self.assertEqual(record["producer"], "packaged_iccad24_def_passthrough")
        self.assertEqual(record["normalization"], "none")
        self.assertTrue(record["coordinates_fixed"])
        self.assertEqual(record["source_sha256"], record["normalized_sha256"])
        self.assertTrue(record["normalized_path"].endswith("/aes_256/aes_256.def"))


if __name__ == "__main__":
    unittest.main()
