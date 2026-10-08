import json
from pathlib import Path
import sys

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from path_fidelity import (  # noqa: E402
    _derive_pairs,
    _new_paths_from_batch,
    compare_backend,
    sha256_file,
)
from dreamplace.ops.timing_net_weighting import timing_net_weighting_cpp  # noqa: E402
from dreamplace.ops.timing_propagation import timing_propagation  # noqa: E402


FIXTURE = SCRIPT_DIR / "fixtures" / "path_parity"
GOLDEN = FIXTURE / "golden"


CANONICAL_NAMES = [
    "port:a",
    "port:b",
    "inst:u_buf0/pin:A",
    "inst:u_buf0/pin:Y",
    "inst:u_inv0/pin:A",
    "inst:u_inv0/pin:Y",
    "inst:u_xor0/pin:A",
    "inst:u_xor0/pin:B",
    "inst:u_xor0/pin:Y",
    "inst:u_buf1/pin:A",
    "inst:u_buf1/pin:Y",
    "port:y_buf",
    "inst:u_inv1/pin:A",
    "inst:u_inv1/pin:Y",
    "port:y_inv",
]


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def _build_predecessor_csr(num_pins, edges):
    predecessors = [[] for _ in range(num_pins)]
    for source, target, arc_id in edges:
        predecessors[target].append((source, arc_id))
    offsets = [0]
    pins = []
    arc_ids = []
    for records in predecessors:
        for source, arc_id in records:
            pins.append(source)
            arc_ids.append(arc_id)
        offsets.append(len(pins))
    return (
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(pins, dtype=torch.int32),
        torch.tensor(arc_ids, dtype=torch.int32),
    )


def _extract_fixture_paths():
    module = timing_propagation._tp_cpp
    assert module is not None and hasattr(module, "SetupCriticalPathExtractor")
    flat_arcs = torch.tensor(
        [
            [2, 3, 0, 0, 1],
            [4, 5, 1, 0, -1],
            [6, 8, 2, 0, 0],
            [7, 8, 2, 1, 0],
            [9, 10, 0, 0, 1],
            [12, 13, 1, 0, -1],
        ],
        dtype=torch.int32,
    )
    edges = [
        (0, 2, -1),
        (2, 3, 0),
        (3, 4, -1),
        (4, 5, 1),
        (5, 6, -1),
        (1, 7, -1),
        (6, 8, 2),
        (7, 8, 3),
        (8, 9, -1),
        (9, 10, 4),
        (10, 11, -1),
        (8, 12, -1),
        (12, 13, 5),
        (13, 14, -1),
    ]
    pred_start, pred_pin, pred_arc_id = _build_predecessor_csr(
        len(CANONICAL_NAMES), edges
    )
    extractor = module.SetupCriticalPathExtractor(
        flat_arcs,
        pred_start,
        pred_pin,
        pred_arc_id,
        torch.tensor([0, 1], dtype=torch.int32),
        0,
    )
    rise_aat = torch.tensor(
        [0, 0, 0, 10, 10, 32, 32, 0, 62, 62, 72, 72, 62, 84, 84],
        dtype=torch.float32,
    )
    fall_aat = torch.tensor(
        [0, 0, 0, 12, 12, 32, 32, 0, 64, 64, 76, 76, 64, 84, 84],
        dtype=torch.float32,
    )
    batch = extractor.extract(
        torch.tensor([11, 14], dtype=torch.int32),
        torch.tensor([-1, -1], dtype=torch.int64),
        torch.tensor([-67.0, -79.0]),
        torch.tensor([-71.0, -79.0]),
        rise_aat,
        fall_aat,
        torch.zeros(len(CANONICAL_NAMES)),
        torch.zeros(len(CANONICAL_NAMES)),
        torch.tensor([10, 0, 30, 30, 10, 0], dtype=torch.float32),
        torch.tensor([0, 20, 30, 30, 0, 20], dtype=torch.float32),
        torch.tensor([0, 22, 32, 32, 0, 22], dtype=torch.float32),
        torch.tensor([12, 0, 32, 32, 12, 0], dtype=torch.float32),
        0,
        32,
        1.0e-4,
    )
    snapshot = {"pin_rise_aat": rise_aat, "pin_fall_aat": fall_aat}
    paths, invalid = _new_paths_from_batch(batch, snapshot, CANONICAL_NAMES)
    assert invalid == []
    return batch, paths


def test_fixture_manifest_and_opentimer_golden_are_current():
    manifest = json.loads(
        (GOLDEN / "golden_manifest.json").read_text(encoding="ascii")
    )
    assert manifest["schema_version"] == 3
    assert manifest["opentimer_metadata"]["fep"] == 4
    assert manifest["opentimer_metadata"]["endpoint_gba_wns_ps"] == -79.0
    for name, expected_sha in manifest["fixture_inputs"].items():
        assert sha256_file(FIXTURE / name) == expected_sha
    for key, name in (
        ("canonical_pin_map_sha256", "canonical_pin_map.jsonl"),
        ("golden_pairs_sha256", "golden_pairs.jsonl"),
        ("golden_paths_sha256", "golden_paths.jsonl"),
    ):
        assert sha256_file(GOLDEN / name) == manifest["artifacts"][key]


def test_native_extractor_matches_typed_opentimer_paths_and_pairs_exactly():
    golden_paths = _load_jsonl(GOLDEN / "golden_paths.jsonl")
    golden_pairs = _load_jsonl(GOLDEN / "golden_pairs.jsonl")
    batch, paths = _extract_fixture_paths()
    canonical_to_id = {
        canonical_name: pin_id
        for pin_id, canonical_name in enumerate(CANONICAL_NAMES)
    }
    pin_to_node = [0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 7, 6, 6, 8]
    pairs = _derive_pairs(
        paths,
        pin_to_node,
        canonical_to_id,
        wns_ps=-79.0,
        min_weight=10.0,
        max_weight=50.0,
        accumulate_weight=0.2,
    )

    comparison = compare_backend(golden_paths, golden_pairs, paths, pairs)
    assert comparison["selected_state_recall"] == 1.0
    assert comparison["selected_state_precision"] == 1.0
    assert comparison["exact_path_match_rate"] == 1.0
    assert comparison["ordered_pair_overlap"] == 1.0
    assert comparison["weighted_pair_jaccard"] == 1.0
    assert comparison["normalized_pair_weight_error"] == 0.0

    native_pairs = timing_net_weighting_cpp.accumulate_pin2pin_pairs(
        batch.path_offsets,
        batch.path_pins,
        batch.endpoint_slacks,
        batch.path_valid,
        torch.tensor(pin_to_node, dtype=torch.int64),
        torch.empty((0, 2), dtype=torch.int64),
        torch.empty((0,), dtype=torch.float32),
        -79.0,
        10.0,
        50.0,
        0.2,
    )
    native_weight_by_key = {
        (CANONICAL_NAMES[int(source)], CANONICAL_NAMES[int(target)]): float(weight)
        for (source, target), weight in zip(
            native_pairs.pair_keys.tolist(), native_pairs.pair_weights.tolist()
        )
    }
    golden_weight_by_key = {
        (record["src_pin"], record["dst_pin"]): float(record["weight"])
        for record in golden_pairs
    }
    assert native_weight_by_key.keys() == golden_weight_by_key.keys()
    for key, expected_weight in golden_weight_by_key.items():
        assert native_weight_by_key[key] == pytest.approx(expected_weight)
