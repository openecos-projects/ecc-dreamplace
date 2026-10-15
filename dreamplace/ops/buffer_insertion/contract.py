import json
from dataclasses import dataclass
from pathlib import Path


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _as_rows(value):
    rows = _as_list(value)
    return [list(row) for row in rows]


def _finite_len(value):
    return len(_as_list(value))


def _nonempty(value):
    return _finite_len(value) > 0


def _fallback_family_candidate_table(flat_libcell_info, preserve_vt=True):
    families = {}
    for cell_id, row in enumerate(_as_rows(flat_libcell_info)):
        if len(row) < 4:
            continue
        main_id = int(round(float(row[1])))
        size = float(row[2])
        vt = int(round(float(row[3]))) if len(row) > 3 else 0
        key = (main_id, vt) if preserve_vt else (main_id, None)
        families.setdefault(key, []).append((size, cell_id, vt, main_id))
    table = {}
    for key, rows in families.items():
        rows = sorted(rows, key=lambda item: (item[0], item[1]))
        table[key] = {
            "sizes": [row[0] for row in rows],
            "cell_ids": [row[1] for row in rows],
            "vts": [row[2] for row in rows],
            "main_ids": [row[3] for row in rows],
        }
    return table


def _family_candidate_table(flat_libcell_info):
    if hasattr(flat_libcell_info, "detach"):
        try:
            from dreamplace.ops.discrete_gradient_topk.discrete_gradient_topk import (
                build_family_candidate_table,
            )

            table = build_family_candidate_table(flat_libcell_info, preserve_vt=True)
            return {
                key: {
                    "sizes": _as_list(value["sizes"]),
                    "cell_ids": _as_list(value["cell_ids"]),
                    "vts": _as_list(value["vts"]),
                    "main_ids": _as_list(value["main_ids"]),
                }
                for key, value in table.items()
            }
        except Exception:
            pass
    return _fallback_family_candidate_table(flat_libcell_info, preserve_vt=True)


@dataclass(frozen=True)
class BufferFamilyLegalTable:
    main_id: int
    vt: int
    legal_cell_ids: list
    legal_master_names: list
    legal_size_values: list

    @classmethod
    def from_metadata(cls, metadata, main_id, preferred_master_name=""):
        flat_info = getattr(metadata, "flat_libcell_info", None)
        family_table = _family_candidate_table(flat_info)
        matching_keys = [key for key in family_table if int(key[0]) == int(main_id)]
        if not matching_keys:
            return None, ["empty_buffer_legal_table"]
        if len(matching_keys) > 1:
            preferred = str(preferred_master_name or "").strip().casefold()
            if preferred and not preferred.startswith("auto:"):
                names = _as_list(getattr(metadata, "flat_libcell_names", []))
                preferred_keys = []
                for key in matching_keys:
                    rows = family_table[key]
                    cell_ids = [int(cell_id) for cell_id in rows["cell_ids"]]
                    if any(
                        0 <= cell_id < len(names)
                        and str(names[cell_id]).casefold() == preferred
                        for cell_id in cell_ids
                    ):
                        preferred_keys.append(key)
                if len(preferred_keys) == 1:
                    matching_keys = preferred_keys
                else:
                    return None, ["unsupported_multiple_buffer_vt"]
            else:
                return None, ["unsupported_multiple_buffer_vt"]

        key = matching_keys[0]
        rows = family_table[key]
        cell_ids = [int(cell_id) for cell_id in rows["cell_ids"]]
        names = _as_list(getattr(metadata, "flat_libcell_names", []))
        legal_names = [
            str(names[cell_id]) if 0 <= cell_id < len(names) else ""
            for cell_id in cell_ids
        ]
        sizes = [float(size) for size in rows["sizes"]]
        return cls(
            main_id=int(main_id),
            vt=int(key[1] if key[1] is not None else 0),
            legal_cell_ids=cell_ids,
            legal_master_names=legal_names,
            legal_size_values=sizes,
        ), []


class BufferFamilyContract:
    def __init__(self, metadata, backend="unknown", preferred_master_name=""):
        self.metadata = metadata
        self.backend = backend
        self.preferred_master_name = str(preferred_master_name or "").strip()

    @classmethod
    def from_buffering_params(cls, metadata, params, backend="unknown"):
        return cls(
            metadata,
            backend=backend,
            preferred_master_name=getattr(
                params,
                "buffering_fixed_buffer_master",
                "",
            ),
        )

    def _basic_reasons(self, legal_table):
        reasons = []
        if legal_table is None:
            return reasons

        required_vector_fields = (
            "flat_libcell_names",
            "flat_libcell_width",
            "flat_libcell_height",
            "flat_libcell_leakage",
            "cell_id_2_libpin_id_start",
            "flat_lib_pin_cap",
            "cell_id_2_arc_id_start",
            "flat_libarc_info",
        )
        for field_name in required_vector_fields:
            if not _nonempty(getattr(self.metadata, field_name, None)):
                reasons.append("missing_" + field_name)

        for field_name in (
            "f_delay_flat_luts_values",
            "r_delay_flat_luts_values",
            "f_trans_flat_luts_values",
            "r_trans_flat_luts_values",
        ):
            if not _nonempty(getattr(self.metadata, field_name, None)):
                reasons.append("missing_" + field_name)

        names = _as_list(getattr(self.metadata, "flat_libcell_names", []))
        if any(cell_id >= len(names) or not legal_table.legal_master_names[idx]
               for idx, cell_id in enumerate(legal_table.legal_cell_ids)):
            reasons.append("missing_master_name")

        cell_pin_start = [int(value) for value in _as_list(getattr(self.metadata, "cell_id_2_libpin_id_start", []))]
        pin_caps = [float(value) for value in _as_list(getattr(self.metadata, "flat_lib_pin_cap", []))]
        input_offsets = []
        for cell_id in legal_table.legal_cell_ids:
            if cell_id + 1 >= len(cell_pin_start):
                reasons.append("missing_libpin_offsets")
                continue
            begin = cell_pin_start[cell_id]
            end = cell_pin_start[cell_id + 1]
            caps = pin_caps[begin:end]
            positive_offsets = [offset for offset, cap in enumerate(caps) if cap > 0.0]
            if not positive_offsets:
                reasons.append("missing_input_cap")
            else:
                input_offsets.append(positive_offsets[0])
        if input_offsets and len(set(input_offsets)) > 1:
            reasons.append("unstable_input_pin_offset")

        arc_start = [int(value) for value in _as_list(getattr(self.metadata, "cell_id_2_arc_id_start", []))]
        arc_offsets = []
        for cell_id in legal_table.legal_cell_ids:
            if cell_id + 1 >= len(arc_start):
                reasons.append("missing_arc_offsets")
                continue
            if arc_start[cell_id + 1] <= arc_start[cell_id]:
                reasons.append("missing_timing_arc")
            else:
                arc_offsets.append(0)
        if arc_offsets and len(set(arc_offsets)) > 1:
            reasons.append("unstable_timing_arc_offset")

        return sorted(set(reasons))

    def build_artifact(self):
        main_id = int(getattr(self.metadata, "buffer_main_type_index", -1))
        export_status = str(getattr(self.metadata, "buffer_main_type_status", "unknown"))
        if main_id < 0 and self.preferred_master_name:
            candidates = _as_list(getattr(self.metadata, "buffer_main_type_candidate_indices", []))
            names = _as_list(getattr(self.metadata, "flat_libcell_names", []))
            rows = _as_rows(getattr(self.metadata, "flat_libcell_info", []))
            for cell_id, name in enumerate(names):
                if str(name).casefold() == self.preferred_master_name.casefold():
                    candidate = int(rows[cell_id][1])
                    if candidate in candidates:
                        main_id = candidate
                        export_status = "ok"
                    break
        unsupported_reasons = []
        if main_id < 0:
            unsupported_reasons.append(export_status if export_status != "ok" else "invalid_buffer_main_type_index")

        legal_table = None
        if main_id >= 0:
            legal_table, table_reasons = BufferFamilyLegalTable.from_metadata(
                self.metadata,
                main_id,
                preferred_master_name=self.preferred_master_name,
            )
            unsupported_reasons.extend(table_reasons)
            unsupported_reasons.extend(self._basic_reasons(legal_table))

        status = "ok" if not unsupported_reasons and legal_table is not None else "unsupported"
        legal_cell_ids = legal_table.legal_cell_ids if legal_table is not None else []
        artifact = {
            "artifact": "buffering_buffer_family_contract",
            "artifact_version": 1,
            "status": status,
            "backend": self.backend,
            "buffer_main_type_index": main_id,
            "buffer_main_type_status": export_status,
            "buffer_vt": legal_table.vt if legal_table is not None else None,
            "bsu_semantics": "diff_sizing_family_local_legal_index",
            "legal_cell_ids": legal_cell_ids,
            "legal_master_names": legal_table.legal_master_names if legal_table is not None else [],
            "legal_size_values": legal_table.legal_size_values if legal_table is not None else [],
            "bsu_to_cell_id": {
                str(index): int(cell_id)
                for index, cell_id in enumerate(legal_cell_ids)
            },
            "unsupported_reasons": sorted(set(reason for reason in unsupported_reasons if reason)),
        }
        return artifact

    def write_artifact(self, output_path):
        artifact = self.build_artifact()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        return artifact
