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
