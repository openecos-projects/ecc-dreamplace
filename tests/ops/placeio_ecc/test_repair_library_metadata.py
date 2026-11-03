"""Library declaration inputs survive the placeio_ecc carrier boundary."""

from types import SimpleNamespace

import pytest
from dreamplace.ops.placeio_ecc.backend_caps import DEFAULT_BACKEND_CAPS
from dreamplace.ops.placeio_ecc.library_metadata import (
    attach_repair_library_source,
    ensure_repair_library_metadata,
)
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction
from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.liberty import build_liberty_view_from_pydb
from test_timing_carrier import _base


def test_library_sources_are_lazy_and_reused_after_physical_carrier_refresh(tmp_path, monkeypatch):
    liberty = tmp_path / "cells.lib"
    liberty.write_text(
        'library (test) { leakage_power_unit : "1nW"; '
        "cell (BUF1) { pin (A) { direction : input; } "
        'pin (Y) { direction : output; function : "A"; } } '
        "cell (BUF2) { pin (A) { direction : input; } "
        'pin (Y) { direction : output; function : "A"; } } }'
    )
    lef = tmp_path / "cells.lef"
    lef.write_text("MACRO BUF1 CLASS CORE ; END BUF1\nMACRO BUF2 CLASS CORE ; END BUF2\n")
    base = _base()
    carrier = CommittedTimingPyDB(base, base)
    attach_repair_library_source(
        carrier,
        SimpleNamespace(lef_input=[str(lef)]),
        {"lib_paths": [str(liberty)]},
    )
    assert carrier.repair_library_metadata is None
    imported = ensure_repair_library_metadata(carrier)

    def forbidden(*_args):
        pytest.fail("carrier refresh parsed immutable library metadata again")

    monkeypatch.setattr(rebuffer_cpp, "read_repair_library_metadata", forbidden)
    refreshed = CommittedTimingPyDB(carrier, base)

    assert ensure_repair_library_metadata(refreshed) is imported
    assert build_liberty_view_from_pydb(refreshed)["_native_cell_metadata"] is imported
    with pytest.raises(ValueError, match="identities changed"):
        refreshed.repair_library_source.read(["BUF1"])


def test_target_selection_rejects_a_pydb_without_source_library_inputs():
    with pytest.raises(ValueError, match="placeio_ecc library sources"):
        ensure_repair_library_metadata(_base())


def _cross_family_database(tmp_path, *, equivalent=True):
    liberty = tmp_path / "cells.lib"
    target_function = "A" if equivalent else "!A"
    liberty.write_text(
        "library (test) { cell (BUF1) { pin (A) { direction : input; } "
        'pin (Y) { direction : output; function : "A"; } } '
        "cell (BUF2) { pin (A) { direction : input; } "
        f'pin (Y) {{ direction : output; function : "{target_function}"; }} }} }}'
    )
    lef = tmp_path / "cells.lef"
    lef.write_text("MACRO BUF1 CLASS CORE ; END BUF1\nMACRO BUF2 CLASS CORE ; END BUF2\n")
    base = _base()
    base.flat_libcell_info[1] = ["BUF2", 1, 1, 0]
    base.main_id_2_cell_id_start = [0, 1, 2]
    base.buffer_main_type_candidate_indices = [0, 1]
    attach_repair_library_source(
        base, SimpleNamespace(lef_input=[str(lef)]), {"lib_paths": [str(liberty)]}
    )
    calls = []

    def apply(node_ids, masters):
        calls.append((node_ids, masters))
        return dict(ok=True, accepted_count=len(node_ids), rejected_count=0)

    base.apply_sizing = apply
    return SimpleNamespace(pydb=base, timing_enabled=True, backend_caps=DEFAULT_BACKEND_CAPS), calls


def test_repair_commit_qualifies_equivalent_buffer_across_physical_families(tmp_path):
    raw_db, calls = _cross_family_database(tmp_path)

    result = PlaceIOFunction.apply_repair_design(
        raw_db,
        dict(actions=[], resizes=[dict(node_id=0, target_cell_index=1)]),
        refresh=False,
    )

    assert result["status"] == "committed"
    assert calls == [([0], ["BUF2"])]


def test_cross_family_sizing_rejects_non_equivalent_function_before_native_write(tmp_path):
    raw_db, calls = _cross_family_database(tmp_path, equivalent=False)

    with pytest.raises(ValueError, match="swappable buffer"):
        PlaceIOFunction.apply_repair_design(
            raw_db,
            dict(actions=[], resizes=[dict(node_id=0, target_cell_index=1)]),
            refresh=False,
        )

    assert calls == []
