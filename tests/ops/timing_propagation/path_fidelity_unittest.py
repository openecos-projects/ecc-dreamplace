import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_fidelity import (  # noqa: E402
    _derive_pairs,
    _pysta_sequential_launch_arcs_for_golden,
    build_canonical_pin_map,
    build_launch_stage_probe,
    compare_backend,
    compare_frame,
    def_components_sha256,
    summarize_endpoint_gba_alignment,
    summarize_old_new_endpoint_domain,
)


def test_structural_canonical_pin_mapping_handles_ports_and_instance_pins():
    placedb = SimpleNamespace(
        pin_names=[b"in", b"u:a:A", b"u:a:Y"],
        node_names=[b"in", b"u:a"],
        pin2node_map=[0, 1, 1],
    )

    records, names = build_canonical_pin_map(placedb)

    assert names == [
        "port:in",
        "inst:u:a/pin:A",
        "inst:u:a/pin:Y",
    ]
    assert records[1]["node_name"] == "u:a"


def test_structural_canonical_pin_mapping_rejects_unrelated_raw_name():
    placedb = SimpleNamespace(
        pin_names=["other:A"],
        node_names=["u0"],
        pin2node_map=[0],
    )

    with pytest.raises(ValueError, match="not structurally tied"):
        build_canonical_pin_map(placedb)


def test_pair_derivation_matches_min_accumulate_clamp_semantics():
    paths = [
        {
            "rank": 0,
            "slack_ps": -10.0,
            "points": [{"pin": "a"}, {"pin": "b"}, {"pin": "c"}],
        },
        {
            "rank": 1,
            "slack_ps": -5.0,
            "points": [{"pin": "a"}, {"pin": "b"}],
        },
    ]

    pairs = _derive_pairs(
        paths,
        [0, 1, 2],
        {"a": 0, "b": 1, "c": 2},
        wns_ps=-10.0,
        min_weight=10.0,
        max_weight=10.05,
        accumulate_weight=0.2,
    )

    assert pairs == [
        {
            "clamped": True,
            "dst_pin": "b",
            "first_path_rank": 0,
            "occurrence_count": 2,
            "src_pin": "a",
            "weight": 10.05,
        },
        {
            "clamped": False,
            "dst_pin": "c",
            "first_path_rank": 0,
            "occurrence_count": 1,
            "src_pin": "b",
            "weight": 10.0,
        },
    ]


def test_backend_comparison_preserves_all_denominators():
    golden_path = {
        "split": "max",
        "endpoint_pin": "b",
        "endpoint_transition": "rise",
        "points": [
            {"pin": "a", "transition": "fall"},
            {"pin": "b", "transition": "rise"},
        ],
    }
    golden_pair = {
        "src_pin": "a",
        "dst_pin": "b",
        "occurrence_count": 2,
        "weight": 10.0,
    }

    result = compare_backend(
        [golden_path],
        [golden_pair],
        [dict(golden_path)],
        [dict(golden_pair)],
    )

    assert result["selected_state_match_count"] == 1
    assert result["selected_state_denominator"] == 1
    assert result["exact_path_match_count"] == 1
    assert result["exact_path_denominator"] == 1
    assert result["ordered_pair_intersection"] == 2
    assert result["ordered_pair_union"] == 2
    assert result["weighted_pair_jaccard"] == pytest.approx(1.0)


def test_endpoint_domain_summary_separates_async_legacy_endpoints_from_duplicates():
    old_paths = [
        {
            "endpoint_pin": "inst:u0/pin:D",
            "endpoint_transition": "unknown",
            "slack_ps": -10.0,
        },
        {
            "endpoint_pin": "inst:u0/pin:RESET",
            "endpoint_transition": "unknown",
            "slack_ps": -20.0,
        },
    ]
    new_paths = [
        {
            "endpoint_pin": "inst:u0/pin:D",
            "endpoint_transition": "rise",
            "slack_ps": -8.0,
        },
        {
            "endpoint_pin": "inst:u0/pin:D",
            "endpoint_transition": "rise",
            "slack_ps": -7.0,
        },
        {
            "endpoint_pin": "inst:u0/pin:D",
            "endpoint_transition": "fall",
            "slack_ps": -6.0,
        },
    ]
    pysta_states = [
        {
            "endpoint_pin": "inst:u0/pin:D",
            "endpoint_transition": "rise",
            "violating": True,
        }
    ]

    summary = summarize_old_new_endpoint_domain(
        old_paths,
        new_paths,
        pysta_states,
    )

    assert summary["common_endpoint_pin_count"] == 1
    assert summary["new_duplicate_test_count"] == 1
    assert summary["new_endpoint_transition_count"] == 2
    assert summary["old_only_endpoint_pin_count"] == 1
    assert summary["old_only_absent_from_pysta_state_domain_count"] == 1
    assert summary["old_only_local_pin_counts"] == {"RESET": 1}
    assert summary["old_only_slack_sum_ps"] == -20.0


def test_endpoint_gba_alignment_does_not_use_recovered_path_slack():
    golden = [
        {
            "split": "max",
            "endpoint_pin": "b",
            "endpoint_transition": "rise",
            "endpoint_gba_aat_ps": 1075.0,
            "endpoint_gba_rat_ps": 1000.0,
            "endpoint_gba_slack_ps": -75.0,
            "endpoint_cppr_credit_ps": 0.0,
            "recovered_path_slack_ps": -1000.0,
        }
    ]
    pysta = [
        {
            "split": "max",
            "endpoint_pin": "b",
            "endpoint_transition": "rise",
            "aat_ps": 1075.0,
            "rat_ps": 1100.0,
            "slack_ps": 25.0,
        }
    ]

    summary = summarize_endpoint_gba_alignment(golden, pysta)

    assert summary["matched_state_count"] == 1
    assert summary["sign_mismatch_count"] == 1
    assert summary["pysta_minus_golden_median_ps"] == pytest.approx(100.0)
    assert summary["pysta_minus_golden_aat_median_ps"] == 0.0
    assert summary["pysta_minus_golden_rat_median_ps"] == 100.0
    assert summary["sign_mismatch_golden_endpoint_gba_slack_median_ps"] == -75.0


def test_endpoint_gba_alignment_rejects_ambiguous_legacy_slack():
    golden = [
        {
            "split": "max",
            "endpoint_pin": "b",
            "endpoint_transition": "rise",
            "slack_ps": -1000.0,
        }
    ]

    with pytest.raises(ValueError, match="endpoint_gba_slack_ps"):
        summarize_endpoint_gba_alignment(golden, [])


def test_launch_stage_probe_finds_sequential_output_boundary():
    golden = [
        {
            "split": "max",
            "endpoint_pin": "inst:cap/pin:D",
            "endpoint_transition": "rise",
            "endpoint_gba_slack_ps": -100.0,
            "points": [
                {
                    "pin": "port:rst",
                    "transition": "rise",
                    "point_gba_aat_ps": 0.0,
                    "recovered_path_aat_ps": 0.0,
                },
                {
                    "pin": "inst:launch/pin:RESET",
                    "transition": "rise",
                    "point_gba_aat_ps": 10.0,
                    "recovered_path_aat_ps": 10.0,
                },
                {
                    "pin": "inst:launch/pin:Q",
                    "transition": "rise",
                    "point_gba_aat_ps": 210.0,
                    "recovered_path_aat_ps": 210.0,
                },
                {
                    "pin": "inst:logic/pin:A",
                    "transition": "rise",
                    "point_gba_aat_ps": 215.0,
                    "recovered_path_aat_ps": 215.0,
                },
                {
                    "pin": "inst:cap/pin:D",
                    "transition": "rise",
                    "point_gba_aat_ps": 500.0,
                    "recovered_path_aat_ps": 500.0,
                },
            ],
        }
    ]
    pysta_states = [
        {
            "split": "max",
            "endpoint_pin": "inst:cap/pin:D",
            "endpoint_test_id": 7,
            "endpoint_transition": "rise",
            "slack_ps": -10.0,
        }
    ]
    pysta_points = [
        {
            "pin": point["pin"],
            "transition": point["transition"],
            "point_gba_aat_ps": point["point_gba_aat_ps"]
            + (0.0 if index < 2 else -160.0),
        }
        for index, point in enumerate(golden[0]["points"])
    ]

    probe = build_launch_stage_probe(golden, pysta_states, pysta_points)

    representative = probe["representatives"][0]
    assert representative["first_large_delta_index"] == 2
    assert representative["points"][2]["role"] == "sequential_control_to_output"
    assert representative["points"][3]["role"] == "first_post_sequential_net_sink"


def test_sequential_launch_arc_probe_reports_control_namespace_candidate():
    placedb = SimpleNamespace(
        flat_inst_arcs_by_level=[
            [0, 2, 0, 3, 0, 1, 3, 0],
            [1, 2, 0, 4, -1, 1, 4, 0],
        ],
        flat_inst_arcs_by_level_start=[0, 2],
        clk_pin_names=["u0:CLK", "u0:RESET"],
        clk_pin_rtran=[0.0, 5.0],
        clk_pin_ftran=[0.0, 6.0],
        flat_libcell_names=["ASYNC_DFF"],
    )
    snapshot = {
        "pin_rise_aat": torch.tensor([0.0, 0.0, 50.0]),
        "pin_fall_aat": torch.tensor([0.0, 0.0, 200.0]),
    }
    golden_paths = [
        {
            "points": [
                {"pin": "inst:u0/pin:RESET", "transition": "rise"},
                {"pin": "inst:u0/pin:QN", "transition": "fall"},
            ]
        }
    ]

    records = _pysta_sequential_launch_arcs_for_golden(
        placedb,
        snapshot,
        {"inst:u0/pin:QN": 2},
        golden_paths,
    )

    assert len(records) == 2
    reset = next(record for record in records if record["source_pin"] == "RESET")
    assert reset["target_pin"] == "inst:u0/pin:QN"
    assert reset["timing_sense"] == -1
    assert reset["source_rise_slew_ps"] == 5.0
    assert reset["target_fall_gba_aat_ps"] == 200.0


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="ascii",
    )


def test_compare_frame_fails_closed_on_coordinate_mismatch(tmp_path):
    autodmp_dir = tmp_path / "autodmp"
    golden_dir = tmp_path / "golden"
    autodmp_dir.mkdir()
    golden_dir.mkdir()
    common = {
        "analysis": {"ideal_clock": True, "setup_only": True, "split": "max"},
        "effective_sdc_sha256": "sdc",
        "frame_id": "f0",
        "input_def_sha256": "def",
        "liberty_sha256": ["lib"],
        "rc": {
            "wire_capacitance_per_micron": 1.0,
            "wire_resistance_per_micron": 2.0,
        },
    }
    (autodmp_dir / "autodmp_manifest.json").write_text(
        json.dumps({**common, "coordinate_sha256": "a"}), encoding="ascii"
    )
    (golden_dir / "golden_manifest.json").write_text(
        json.dumps({**common, "coordinate_sha256": "b"}), encoding="ascii"
    )

    with pytest.raises(ValueError, match="coordinate_sha256"):
        compare_frame(autodmp_dir, golden_dir, tmp_path / "comparison.json")


def test_def_component_hash_ignores_non_component_sections(tmp_path):
    lhs = tmp_path / "lhs.def"
    rhs = tmp_path / "rhs.def"
    components = "COMPONENTS 1 ;\n- u0 BUF + PLACED ( 1 2 ) N ;\nEND COMPONENTS\n"
    lhs.write_text("VERSION 5.8 ;\n" + components + "END DESIGN\n", encoding="ascii")
    rhs.write_text("VERSION 5.7 ;\n" + components + "# tail\nEND DESIGN\n", encoding="ascii")

    assert def_components_sha256(lhs) == def_components_sha256(rhs)
