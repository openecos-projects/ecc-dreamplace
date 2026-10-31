import pytest

from dreamplace.ops.buffer_insertion import fast_buffer_types
from dreamplace.ops.buffer_insertion.fast_buffer_types import (
    fast_buffer_drive_type_table,
    fast_buffer_load_limits,
    fast_buffer_master_names,
)


@pytest.fixture
def characterized(monkeypatch):
    rows = (
        {"name": "BUFX1H7R", "intrinsic_delay": 24.1, "target_load": 0.0055},
        {"name": "DLY1X6H7L", "intrinsic_delay": 62.6, "target_load": 0.0206},
        {"name": "BUFX16H7L", "intrinsic_delay": 22.6, "target_load": 0.0709},
    )
    monkeypatch.setattr(fast_buffer_types, "fast_buffers", lambda pydb: rows)


def test_delay_cells_are_left_out_of_the_candidate_masters(characterized):
    assert fast_buffer_master_names(object()) == ("BUFX1H7R", "BUFX16H7L")


def test_load_limits_follow_the_requested_master_order(characterized):
    assert fast_buffer_load_limits(object(), ("BUFX16H7L", "BUFX1H7R")) == [0.0709, 0.0055]


def test_drive_type_table_maps_masters_to_legal_library_rows():
    library = {
        0: {"master_name": "BUFX16H7L", "cell_id": 607, "buffer_main_type_index": 25},
        1: {"master_name": "BUFX1H7R", "cell_id": 576, "buffer_main_type_index": 25},
    }

    table = fast_buffer_drive_type_table(("BUFX1H7R", "BUFX16H7L"), buffer_library=library)

    assert [(e.master_name, e.bsu_index, e.cell_id) for e in table.entries] == [
        ("BUFX1H7R", 1, 576),
        ("BUFX16H7L", 0, 607),
    ]


def test_drive_type_table_rejects_a_master_outside_the_legal_library():
    with pytest.raises(ValueError, match="absent from legal buffer library: BUFX1H7R"):
        fast_buffer_drive_type_table(("BUFX1H7R",), buffer_library={})
