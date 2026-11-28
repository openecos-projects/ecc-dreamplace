import pytest

from dreamplace.ops.buffer_insertion.candidates import sample_buffer_candidates


def _tree(start, end):
    return {
        "status": "ok",
        "net_id": 10,
        "root_node_id": 0,
        "node_ids": [0, 1],
        "coordinates": {0: start, 1: end},
        "children_by_node": {0: [1], 1: []},
        "parent_by_node": {0: None, 1: 0},
    }


def _sample(start, end, *, k=3):
    return sample_buffer_candidates(
        _tree(start, end),
        buffer_main_type_index=7,
        dbu=1000,
        max_candidates_per_segment=k,
    )


def test_short_nonzero_segment_gets_k_equal_count_candidates():
    candidates = _sample((0, 0), (4, 0))

    assert [(row["x_dbu"], row["y_dbu"]) for row in candidates] == [
        (1, 0),
        (2, 0),
        (3, 0),
    ]
    assert [row["segment_split_ratio"] for row in candidates] == [0.25, 0.5, 0.75]
    assert all(row["segment_split_count_on_edge"] == 3 for row in candidates)


def test_equal_count_summary_accounts_for_each_nonzero_edge():
    candidates, summary = sample_buffer_candidates(
        _tree((0, 0), (4, 0)),
        buffer_main_type_index=7,
        dbu=1000,
        max_candidates_per_segment=3,
        return_summary=True,
    )

    assert len(candidates) == 3
    assert summary == {
        "total_tree_edge_count": 1,
        "coordinate_supported_edge_count": 1,
        "nonzero_tree_edge_count": 1,
        "edge_with_candidate_count": 1,
        "edge_without_candidate_count": 0,
        "candidate_attempt_count": 3,
        "rounded_endpoint_skip_count": 0,
        "coordinate_dedup_count": 0,
        "candidate_count": 3,
    }


def test_long_segment_is_capped_at_k_candidates():
    candidates = _sample((0, 0), (100_000, 0), k=3)

    assert len(candidates) == 3
    assert [row["x_dbu"] for row in candidates] == [25_000, 50_000, 75_000]


def test_zero_length_segment_has_no_candidate():
    assert _sample((7, 9), (7, 9)) == []


def test_dbu_rounding_deduplicates_and_never_exceeds_k():
    first = _sample((0, 0), (2, 0), k=3)
    second = _sample((0, 0), (2, 0), k=3)

    assert len(first) == 1
    assert first == second
    assert first[0]["candidate_id"] == 0
    assert (first[0]["x_dbu"], first[0]["y_dbu"]) == (1, 0)


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_candidate_count_is_rejected(k):
    with pytest.raises(ValueError, match="max_candidates_per_segment must be positive"):
        _sample((0, 0), (4, 0), k=k)


def _branch_tree():
    # Driver 0 -> Steiner 5 -> sink pins 1 and 2.
    return {
        "status": "ok",
        "net_id": 10,
        "root_node_id": 0,
        "sink_pin_ids": [1, 2],
        "node_ids": [0, 1, 2, 5],
        "coordinates": {0: (0, 0), 5: (1000, 0), 1: (1000, 400), 2: (1800, 0)},
        "children_by_node": {0: [5], 5: [1, 2], 1: [], 2: []},
        "parent_by_node": {0: None, 5: 0, 1: 5, 2: 5},
    }


def _rebuffer_sites(tree):
    return sample_buffer_candidates(
        tree,
        buffer_main_type_index=7,
        dbu=1000,
        candidate_generation_policy="rebuffer_sites",
    )


def test_rebuffer_sites_isolate_each_branch_at_the_shared_parent():
    rows = _rebuffer_sites(_branch_tree())

    assert sorted(
        (row["parent_node_id"], row["child_node_ids"][0], row["segment_split_ratio"])
        for row in rows
    ) == [
        (0, 5, 0.001),
        (0, 5, 0.5),
        (0, 5, 0.999),
        (5, 1, 0.0025),
        (5, 1, 0.5),
        (5, 2, 0.00125),
        (5, 2, 0.5),
    ]
    parent_end = [row for row in rows if row["parent_node_id"] == 5 and row["segment_split_ratio"] < 0.5]
    assert {(row["x_dbu"], row["y_dbu"]) for row in parent_end} == {(1000, 1), (1001, 0)}


def test_rebuffer_sites_cap_the_parent_end_ratio_on_short_edges():
    rows = _rebuffer_sites({**_tree((0, 0), (40, 0)), "sink_pin_ids": [1]})

    assert [row["segment_split_ratio"] for row in rows] == [0.01, 0.5]
