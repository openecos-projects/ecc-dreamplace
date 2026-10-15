from types import SimpleNamespace

import pytest

from dreamplace.ops.buffer_insertion.buffer_selection import (
    resolve_fixed_buffer_selection,
)


BUFFER_LIBRARY = {
    3: {"master_name": "BUFX2H7H"},
    11: {"master_name": "BUFX4H7H"},
}


def test_auto_x4_resolves_runtime_legal_buffer_index():
    result = resolve_fixed_buffer_selection(
        SimpleNamespace(buffering_fixed_buffer_master="auto:x4"),
        None,
        BUFFER_LIBRARY,
    )

    assert result == (11, "BUFX4H7H", "auto_x4")


def test_auto_x4_prefers_nominal_vt_on_multi_vt_libraries():
    multi_vt_library = {
        7: {"master_name": "BUFX4H7L", "delay": 0.018},
        11: {"master_name": "BUFX4H7R", "delay": 0.024},
        15: {"master_name": "BUFX4H7H", "delay": 0.035},
    }

    result = resolve_fixed_buffer_selection(
        SimpleNamespace(buffering_fixed_buffer_master="auto:x4"),
        None,
        multi_vt_library,
    )

    assert result == (11, "BUFX4H7R", "auto_x4")


def test_exact_master_resolution_is_case_insensitive():
    result = resolve_fixed_buffer_selection(
        SimpleNamespace(buffering_fixed_buffer_master="bufx4h7h"),
        None,
        BUFFER_LIBRARY,
    )

    assert result == (11, "BUFX4H7H", "exact_master")


def test_explicit_index_overrides_default_master_selector():
    result = resolve_fixed_buffer_selection(
        SimpleNamespace(
            buffering_fixed_buffer_master="auto:x4",
            buffering_fixed_bsu_index=3,
            _buffering_fixed_bsu_index_explicit=True,
        ),
        None,
        BUFFER_LIBRARY,
    )

    assert result == (3, "BUFX2H7H", "bsu_index")


def test_conflicting_explicit_master_and_index_are_rejected():
    params = SimpleNamespace(
        buffering_fixed_buffer_master="BUFX4H7H",
        buffering_fixed_bsu_index=3,
        _buffering_fixed_buffer_master_explicit=True,
        _buffering_fixed_bsu_index_explicit=True,
    )

    with pytest.raises(ValueError, match="resolve to different legal buffers"):
        resolve_fixed_buffer_selection(params, None, BUFFER_LIBRARY)
