"""Drive-strength table loading for typed discrete candidate buffering.

The table is an input contract for the four candidate buffer masters.  Liberty
timing data remains owned by the native metadata/buffer library; this module
only validates the external type identity and maps each master to its legal
family-local BSU index.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


from dreamplace.flows.buffer_drive_table import (
    DRIVE_TYPE_TABLE_SCHEMA,
    EXPECTED_DRIVE_TYPE_MASTERS,
)


@dataclass(frozen=True)
class DriveTypeEntry:
    drive_type_id: int
    master_name: str
    drive_strength: float
    cell_id: int
    bsu_index: int
    buffer_main_type_index: int = -1
    input_cap: float | None = None
    area: float | None = None
    width: float | None = None
    site: str = ""
    vt: str = ""

    def to_dict(self):
        payload = {
            "drive_type_id": int(self.drive_type_id),
            "master_name": str(self.master_name),
            "drive_strength": float(self.drive_strength),
            "cell_id": int(self.cell_id),
            "bsu_index": int(self.bsu_index),
            "buffer_main_type_index": int(self.buffer_main_type_index),
        }
        for name in ("input_cap", "area", "width"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = float(value)
        if self.site:
            payload["site"] = str(self.site)
        if self.vt:
            payload["vt"] = str(self.vt)
        return payload


@dataclass(frozen=True)
class DriveTypeTable:
    entries: tuple[DriveTypeEntry, ...]
    source: str
    path: str = ""
    sha256: str = ""
    units: dict | None = None

    @property
    def count(self):
        return len(self.entries)

    @property
    def master_names(self):
        return tuple(entry.master_name for entry in self.entries)

    @property
    def bsu_indices(self):
        return tuple(int(entry.bsu_index) for entry in self.entries)

    @property
    def cell_ids(self):
        return tuple(int(entry.cell_id) for entry in self.entries)

    def by_id(self, drive_type_id):
        drive_type_id = int(drive_type_id)
        if drive_type_id < 0 or drive_type_id >= len(self.entries):
            raise ValueError(f"drive_type_id={drive_type_id} is outside the table")
        return self.entries[drive_type_id]

    def to_metadata(self):
        return {
            "schema": DRIVE_TYPE_TABLE_SCHEMA,
            "source": str(self.source),
            "path": str(self.path),
            "sha256": str(self.sha256),
            "units": dict(self.units or {}),
            "count": int(self.count),
            "masters": [entry.to_dict() for entry in self.entries],
        }


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _load_payload(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"drive-strength table does not exist: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid drive-strength table JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("drive-strength table root must be an object")
    return payload, hashlib.sha256(raw).hexdigest(), str(path)


def load_drive_type_table(
    path,
    *,
    buffer_library,
    expected_masters=EXPECTED_DRIVE_TYPE_MASTERS,
):
    """Load and validate the external table against the legal buffer library."""

    payload, sha256, path_string = _load_payload(path)
    if payload.get("schema") != DRIVE_TYPE_TABLE_SCHEMA:
        raise ValueError(
            "unsupported drive-strength table schema: "
            f"{payload.get('schema')!r}"
        )
    units = payload.get("units")
    if not isinstance(units, dict):
        raise ValueError("drive-strength table requires units")
    rows = payload.get("buffers")
    if not isinstance(rows, list):
        raise ValueError("drive-strength table requires a buffers list")
    expected_masters = tuple(str(name) for name in expected_masters)
    if len(rows) != len(expected_masters):
        raise ValueError(
            "drive-strength table must contain exactly "
            f"{len(expected_masters)} buffers"
        )

    by_bsu = {
        int(bsu): dict(entry or {})
        for bsu, entry in dict(buffer_library or {}).items()
    }
    by_master = {
        str(entry.get("master_name", "")): (int(bsu), entry)
        for bsu, entry in by_bsu.items()
        if str(entry.get("master_name", ""))
    }
    entries = []
    seen_ids = set()
    seen_masters = set()
    for expected_id, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"drive-strength row {expected_id} must be an object")
        drive_type_id = row.get("drive_type_id")
        if int(drive_type_id) != expected_id:
            raise ValueError(
                "drive_type_id must be contiguous and ordered: "
                f"expected {expected_id}, received {drive_type_id!r}"
            )
        master_name = str(row.get("master_name", "")).strip()
        if master_name != expected_masters[expected_id]:
            raise ValueError(
                f"drive type {expected_id} must be {expected_masters[expected_id]}, "
                f"received {master_name!r}"
            )
        if master_name in seen_masters:
            raise ValueError(f"duplicate drive-strength master: {master_name}")
        seen_ids.add(expected_id)
        seen_masters.add(master_name)
        if master_name not in by_master:
            raise ValueError(
                f"drive-strength master is absent from legal buffer library: {master_name}"
            )
        bsu_index, library_entry = by_master[master_name]
        main_type_index = int(library_entry.get("buffer_main_type_index", -1))
        library_cell_id = int(library_entry.get("cell_id", bsu_index))
        if row.get("cell_id") is None:
            cell_id = library_cell_id
        else:
            cell_id = int(row["cell_id"])
            if library_cell_id != cell_id:
                raise ValueError(
                    f"cell_id mismatch for {master_name}: table={cell_id}, "
                    f"library={library_entry.get('cell_id')}"
                )
        strength_value = row.get("drive_strength")
        if strength_value is None:
            raise ValueError(f"drive_strength is missing for {master_name}")
        entries.append(
            DriveTypeEntry(
                drive_type_id=expected_id,
                master_name=master_name,
                drive_strength=_finite_float(
                    strength_value,
                    f"drive_strength[{expected_id}]",
                ),
                cell_id=cell_id,
                bsu_index=bsu_index,
                buffer_main_type_index=main_type_index,
                input_cap=(
                    _finite_float(row["input_cap"], f"input_cap[{expected_id}]")
                    if row.get("input_cap") is not None
                    else None
                ),
                area=(
                    _finite_float(row["area"], f"area[{expected_id}]")
                    if row.get("area") is not None
                    else None
                ),
                width=(
                    _finite_float(row["width"], f"width[{expected_id}]")
                    if row.get("width") is not None
                    else None
                ),
                site=str(row.get("site", "") or ""),
                vt=str(row.get("vt", "") or ""),
            )
        )
    if set(seen_ids) != set(range(len(expected_masters))):
        raise ValueError("drive-strength table has non-contiguous type IDs")
    return DriveTypeTable(
        entries=tuple(entries),
        source=str(payload.get("source", "external")),
        path=path_string,
        sha256=sha256,
        units=dict(units),
    )


def write_case_drive_type_table(path, *, buffer_library, source="case-buffer-list"):
    """Write the fixed ICS55 four-master table for a case preflight artifact.

    This helper is deliberately explicit: it does not infer arbitrary strength
    types from suffixes.  It only emits the four approved masters and uses the
    known drive-strength values as the external table's ordering contract.
    """

    by_master = {
        str(entry.get("master_name", "")): (int(bsu), dict(entry))
        for bsu, entry in dict(buffer_library or {}).items()
    }
    rows = []
    for drive_type_id, (master_name, strength) in enumerate(
        zip(EXPECTED_DRIVE_TYPE_MASTERS, (8.0, 12.0, 16.0, 20.0))
    ):
        if master_name not in by_master:
            raise ValueError(f"missing approved case buffer: {master_name}")
        bsu, entry = by_master[master_name]
        row = {
            "drive_type_id": drive_type_id,
            "master_name": master_name,
            "drive_strength": strength,
            "cell_id": int(entry.get("cell_id", bsu)),
            "bsu_index": bsu,
            "buffer_main_type_index": int(
                entry.get("buffer_main_type_index", -1)
            ),
        }
        for name in ("input_cap", "area", "width"):
            if entry.get(name) is not None:
                row[name] = float(entry[name])
        rows.append(row)
    payload = {
        "schema": DRIVE_TYPE_TABLE_SCHEMA,
        "source": str(source),
        "library": "ics55",
        "units": {"drive_strength": "relative", "capacitance": "library"},
        "buffers": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return load_drive_type_table(path, buffer_library=buffer_library)
