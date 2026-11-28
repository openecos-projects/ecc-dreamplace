from collections import Counter
from pathlib import Path

import pytest


pytest.importorskip("dreamplace.ops.rebuffer.rebuffer_steiner_cpp")

from dreamplace.ops.rebuffer import (  # noqa: E402
    build_openroad_flute_tree,
    build_openroad_pd_tree,
    build_openroad_stt_tree,
    lut_paths,
)


def _canonical(edges):
    result = Counter()
    for x1, y1, x2, y2 in edges:
        if (x1, y1) == (x2, y2):
            continue
        result[tuple(sorted(((x1, y1), (x2, y2))))] += 1
    return result


def test_openroad_lut_data_is_present():
    paths = lut_paths()
    assert Path(paths["post"]).is_file()
    assert Path(paths["powv"]).is_file()


def test_prim_dijkstra_matches_openroad_duplicate_pin_case():
    tree = build_openroad_pd_tree(
        [0, 10, 10, 10],
        [0, 10, 20, 10],
        0,
        alpha=0.4,
    )

    assert tree["wirelength"] == 30
    assert _canonical(tree["edges"]) == {
        ((0, 0), (10, 10)): 1,
        ((10, 10), (10, 20)): 1,
    }


def test_flute_matches_openroad_low_degree_tree():
    tree = build_openroad_flute_tree(
        [0, 0, 10, 10],
        [0, 10, 10, 0],
        accuracy=3,
    )

    assert tree["wirelength"] == 30
    assert _canonical(tree["edges"]) == {
        ((0, 0), (0, 10)): 1,
        ((0, 0), (10, 0)): 1,
        ((0, 10), (10, 10)): 1,
    }


def test_stt_uses_flute_when_alpha_is_zero():
    tree = build_openroad_stt_tree(
        [0, 0, 10, 10],
        [0, 10, 10, 0],
        0,
        alpha=0.0,
    )

    assert tree["provider"] == "openroad_flute"
    assert tree["wirelength"] == 30
