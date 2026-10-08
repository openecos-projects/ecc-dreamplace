from types import SimpleNamespace

import torch

from dreamplace.ops.buffer_insertion.contract import BufferFamilyLegalTable


def _multi_vt_metadata():
    return SimpleNamespace(
        flat_libcell_info=torch.tensor(
            [
                [0, 4, 1.0, 0],
                [1, 4, 1.0, 1],
                [2, 4, 2.0, 0],
                [3, 4, 2.0, 1],
            ],
            dtype=torch.float32,
        ),
        flat_libcell_names=[
            "BUFX1H7H",
            "BUFX1H7R",
            "BUFX2H7H",
            "BUFX2H7R",
        ],
    )


def test_exact_master_selects_its_vt_subfamily():
    table, reasons = BufferFamilyLegalTable.from_metadata(
        _multi_vt_metadata(),
        4,
        preferred_master_name="BUFX1H7H",
    )

    assert reasons == []
    assert table.legal_master_names == ["BUFX1H7H", "BUFX2H7H"]


def test_multi_vt_family_remains_unsupported_without_exact_master():
    table, reasons = BufferFamilyLegalTable.from_metadata(
        _multi_vt_metadata(),
        4,
    )

    assert table is None
    assert reasons == ["unsupported_multiple_buffer_vt"]


def test_exact_master_matching_is_case_insensitive_and_trimmed():
    table, reasons = BufferFamilyLegalTable.from_metadata(
        _multi_vt_metadata(),
        4,
        preferred_master_name=" bufx1h7r ",
    )

    assert reasons == []
    assert table.legal_master_names == ["BUFX1H7R", "BUFX2H7R"]


def test_auto_selector_does_not_resolve_multi_vt_ambiguity():
    table, reasons = BufferFamilyLegalTable.from_metadata(
        _multi_vt_metadata(),
        4,
        preferred_master_name="auto:x4",
    )

    assert table is None
    assert reasons == ["unsupported_multiple_buffer_vt"]


def test_unknown_exact_master_does_not_pick_an_arbitrary_vt_subfamily():
    table, reasons = BufferFamilyLegalTable.from_metadata(
        _multi_vt_metadata(),
        4,
        preferred_master_name="BUFX4H7L",
    )

    assert table is None
    assert reasons == ["unsupported_multiple_buffer_vt"]
