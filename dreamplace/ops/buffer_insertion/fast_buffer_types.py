"""Typed candidate drive types taken from the OpenROAD-style fast buffer list.

``four_strength_table`` fixes the candidate masters to BUFX8/12/16/20H7L.  The
``fast_buffers`` mode instead uses the masters that the rebuffer
characterization (``findBuffers``/``findFastBuffers``) selects from Liberty,
which includes the small single-sink isolation buffers that repair uses most.
"""

from .drive_type_table import DriveTypeEntry, DriveTypeTable

FAST_BUFFERS_MODE = "fast_buffers"
TYPED_DRIVE_TYPE_MODES = ("four_strength_table", FAST_BUFFERS_MODE)
DELAY_CELL_INTRINSIC_RATIO = 2.0

# Characterization depends only on Liberty, so one result serves every
# contract and state build on the same database.
_FAST_BUFFER_CACHE = {}


def fast_buffers(pydb):
    """Return characterized fast buffer rows for ``pydb`` in characterization order."""

    key = id(pydb)
    cached = _FAST_BUFFER_CACHE.get(key)
    if cached is not None and cached[0] is pydb:
        return cached[1]
    from dreamplace.ops.rebuffer import characterize_buffers_from_pydb

    result = characterize_buffers_from_pydb(pydb)
    if result.get("status") != "ok" or not result.get("fast_buffers"):
        raise ValueError(
            "fast_buffers requires an ok buffer characterization: "
            f"{result.get('unsupported_reasons', result.get('status'))}"
        )
    rows = tuple(dict(row) for row in result["fast_buffers"])
    _FAST_BUFFER_CACHE[key] = (pydb, rows)
    return rows


def fast_buffer_master_names(pydb):
    """Return the candidate masters for ``pydb`` in characterization order.

    Delay cells (intrinsic delay above twice the fastest buffer's) are left
    out: on BM64 the gradient picked DLY1X6H7L for a third of all actions,
    mostly to fix slew, and WNS ended 65 ps worse than without it.
    """

    rows = fast_buffers(pydb)
    fastest = min(float(row["intrinsic_delay"]) for row in rows)
    return tuple(
        str(row["name"])
        for row in rows
        if float(row["intrinsic_delay"]) <= DELAY_CELL_INTRINSIC_RATIO * fastest
    )


def fast_buffer_load_limits(pydb, master_names):
    """Return the load (pF) each master may drive, in ``master_names`` order.

    The limit is the characterization target load: the load at which the
    buffer output reaches the target slew.  A weak buffer beyond it degrades
    the downstream slew so much that a first-order gradient cannot judge it.
    """

    by_name = {str(row["name"]): row for row in fast_buffers(pydb)}
    return [float(by_name[name]["target_load"]) for name in master_names]


def fast_buffer_drive_type_table(master_names, *, buffer_library):
    """Build the typed drive table over ``master_names`` from the legal library."""

    by_master = {
        str(entry.get("master_name", "")): (int(bsu), dict(entry))
        for bsu, entry in dict(buffer_library or {}).items()
    }
    entries = []
    for drive_type_id, master_name in enumerate(master_names):
        if master_name not in by_master:
            raise ValueError(f"fast buffer is absent from legal buffer library: {master_name}")
        bsu_index, entry = by_master[master_name]
        entries.append(
            DriveTypeEntry(
                drive_type_id=drive_type_id,
                master_name=master_name,
                # Ordering only; timing uses each master's own Liberty tables.
                drive_strength=float(drive_type_id),
                cell_id=int(entry.get("cell_id", bsu_index)),
                bsu_index=bsu_index,
                buffer_main_type_index=int(entry.get("buffer_main_type_index", -1)),
                input_cap=entry.get("input_cap"),
                area=entry.get("area"),
                width=entry.get("width"),
            )
        )
    return DriveTypeTable(entries=tuple(entries), source="rebuffer_fast_buffers")


__all__ = [
    "FAST_BUFFERS_MODE",
    "TYPED_DRIVE_TYPE_MODES",
    "fast_buffer_drive_type_table",
    "fast_buffer_load_limits",
    "fast_buffer_master_names",
    "fast_buffers",
]
