"""Parser-cache machinery for the Xplace GPUGR backend.

The cache keeps the parsed LEF/DEF databases (rawdb/gpdb) alive across gpugr
invocations so that later calls only re-apply movable-node coordinates instead
of re-parsing (and re-exporting) the DEF. Mixed into XplaceGPUGR.

Ported from the pinned AiEDA reference
(tools/iEDA/module/gpugr.py @ 9b8aa46, sha256 b3981e71...); the only change is
that workspace-derived values are replaced by the params/placedb carried by the
owning operator.
"""

from pathlib import Path

import torch

PARSER_CACHE_LPOS_TOLERANCE_DBU = 1.0


class XplaceParserCacheMixin:
    # The owning class provides:
    #   self._parser_db_cache, self.params, self._cache_scope_key()

    def _cache_scope_key(self):
        return str(Path(self.params.result_dir).expanduser().resolve())

    def _build_parser_cache_key(self, benchmark: str, design_name: str, lefs):
        return (
            self._cache_scope_key(),
            str(benchmark),
            str(design_name),
            tuple(str(Path(lef).expanduser().resolve()) for lef in lefs),
            int(getattr(self.placedb, "topology_generation", 0)),
        )

    def _normalize_parser_cache_lpos(self, node_lpos, node_names):
        if node_lpos is None:
            return None, None
        if isinstance(node_lpos, torch.Tensor):
            lpos = node_lpos.detach().to(device="cpu", dtype=torch.float32).contiguous()
        else:
            lpos = torch.as_tensor(node_lpos, dtype=torch.float32, device="cpu").contiguous()
        if lpos.dim() != 2 or lpos.size(1) != 2:
            raise ValueError(
                f"parser cache node_lpos must have shape [N, 2], got {tuple(lpos.shape)}"
            )
        if lpos.dtype != torch.float32:
            raise ValueError(f"parser cache node_lpos must be float32, got {lpos.dtype}")

        if node_names is None:
            return lpos, None
        names = tuple(self._normalize_name(name) for name in node_names)
        if len(names) != int(lpos.size(0)):
            raise ValueError(
                f"parser cache node name count ({len(names)}) does not match "
                f"node_lpos rows ({lpos.size(0)})"
            )
        return lpos, names

    def _parser_cache_matches(self, cache_key, node_names, node_count=None):
        cache = self._parser_db_cache
        if not cache:
            return False
        if cache.get("key") != cache_key:
            return False
        if node_names is None:
            return False
        if node_count is not None and cache.get("node_count") != int(node_count):
            return False
        return cache.get("node_names") == tuple(node_names)

    def parser_cache_would_hit(
        self, benchmark: str = "", design_name: str = "", node_names=None, node_count=None
    ):
        if getattr(self.params, "macro_only", False):
            return False
        cache = self._parser_db_cache
        if not cache:
            return False
        if node_names is None:
            return False
        if node_count is not None and cache.get("node_count") != int(node_count):
            return False
        cached_node_names = cache.get("node_names")
        if cached_node_names is node_names:
            names = cached_node_names
        else:
            names = tuple(self._normalize_name(name) for name in node_names)
        if cached_node_names != names:
            return False
        if not design_name:
            design_name = self.params.design_name()
        if not benchmark:
            benchmark = "custom"
        cache_key = cache.get("key")
        if not cache_key or len(cache_key) < 3:
            return False
        return (
            cache_key == self._build_parser_cache_key(
                benchmark, design_name, self._resolve_lefs()
            )
        )

    def _gpdb_node_type_by_id(self, gpdb):
        node_type_by_id = {}
        for start, end, node_type in gpdb.node_type_indices():
            for node_id in range(int(start), int(end)):
                node_type_by_id[node_id] = str(node_type)
        return node_type_by_id

    def _gpdb_movable_node_ids(self, gpdb):
        movable_node_ids = []
        for start, end, node_type in gpdb.node_type_indices():
            if str(node_type) in ("Mov", "FloatMov"):
                movable_node_ids.extend(range(int(start), int(end)))
        return movable_node_ids

    def _validate_parser_cache_movable_names(
        self, node_names, gpdb_node_names, gpdb_movable_node_ids
    ):
        gpdb_movable_names = tuple(gpdb_node_names[node_id] for node_id in gpdb_movable_node_ids)
        autodmp_names = set(node_names)
        gpdb_names = set(gpdb_movable_names)

        missing_in_gpdb_movable = sorted(autodmp_names - gpdb_names)
        extra_in_gpdb_movable = sorted(gpdb_names - autodmp_names)
        if missing_in_gpdb_movable or extra_in_gpdb_movable:
            details = []
            if missing_in_gpdb_movable:
                details.append(
                    "AutoDMP-only movable names: " + ", ".join(missing_in_gpdb_movable[:5])
                )
            if extra_in_gpdb_movable:
                details.append("GPDB-only movable names: " + ", ".join(extra_in_gpdb_movable[:5]))
            raise ValueError(
                "Cannot cache gpugr parser DB because AutoDMP and gpdb movable-name sets differ; "
                + "; ".join(details)
            )
        return gpdb_movable_names

    def _validate_parser_cache_node_lpos(self, gpdb, movable_node_ids, expected_lpos, context):
        current_lpos = gpdb.node_lpos_tensor().to(dtype=torch.float32).contiguous()
        if expected_lpos.dtype != torch.float32:
            raise RuntimeError(
                f"gpugr parser cache {context} expected_lpos must be float32, "
                f"got {expected_lpos.dtype}"
            )
        mapped_lpos = current_lpos.index_select(0, movable_node_ids)
        diff = torch.abs(mapped_lpos - expected_lpos)
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        # DEF coordinates are integral DBU, so exporting and reparsing a
        # floating-point placement can differ by one DBU through truncation.
        if max_abs_diff > PARSER_CACHE_LPOS_TOLERANCE_DBU:
            max_index = int(torch.argmax(diff.reshape(-1)).item()) if diff.numel() else 0
            row = max_index // 2
            coord = max_index % 2
            raise RuntimeError(
                f"gpugr parser cache {context} lpos validation failed: "
                f"max_abs_diff={max_abs_diff:.6g} row={row} coord={coord} "
                f"expected={float(expected_lpos[row, coord].item()):.6g} "
                f"actual={float(mapped_lpos[row, coord].item()):.6g}"
            )

    def _store_parser_cache(self, cache_key, rawdb, gpdb, node_names, node_lpos):
        if node_names is None:
            raise ValueError("Cannot cache gpugr parser DB without AutoDMP node-name mapping")
        node_names = tuple(node_names)
        node_count = int(node_lpos.size(0))
        if len(node_names) != int(node_count):
            raise ValueError(
                f"Cannot cache gpugr parser DB because node name count ({len(node_names)}) "
                f"does not match node_lpos rows ({node_count})"
            )
        seen_input_names = set()
        duplicate_input_names = set()
        for name in node_names:
            if name in seen_input_names:
                duplicate_input_names.add(name)
            else:
                seen_input_names.add(name)
        duplicate_input_names = sorted(duplicate_input_names)
        if duplicate_input_names:
            sample = ", ".join(duplicate_input_names[:5])
            raise ValueError(
                f"Cannot cache gpugr parser DB because AutoDMP node names are duplicated: {sample}"
            )

        gpdb_node_names = [self._normalize_name(name) for name in gpdb.node_names()]
        name_to_gpdb_id = {}
        duplicate_gpdb_names = set()
        for node_id, name in enumerate(gpdb_node_names):
            if name in name_to_gpdb_id:
                duplicate_gpdb_names.add(name)
            else:
                name_to_gpdb_id[name] = node_id
        if duplicate_gpdb_names:
            sample = ", ".join(sorted(duplicate_gpdb_names)[:5])
            raise ValueError(
                f"Cannot cache gpugr parser DB because gpdb node names are duplicated: {sample}"
            )

        missing = [name for name in node_names if name not in name_to_gpdb_id]
        if missing:
            sample = ", ".join(missing[:5])
            raise ValueError(
                f"Cannot cache gpugr parser DB because AutoDMP nodes are missing in gpdb: {sample}"
            )

        movable_node_id_list = [name_to_gpdb_id[name] for name in node_names]
        if len(set(movable_node_id_list)) != len(movable_node_id_list):
            raise ValueError(
                "Cannot cache gpugr parser DB because AutoDMP names map to duplicate gpdb node ids"
            )

        node_type_by_id = self._gpdb_node_type_by_id(gpdb)
        non_movable = [
            f"{node_names[index]}:{node_type_by_id.get(node_id, '<missing>')}"
            for index, node_id in enumerate(movable_node_id_list)
            if node_type_by_id.get(node_id) not in ("Mov", "FloatMov")
        ]
        if non_movable:
            sample = ", ".join(non_movable[:5])
            raise ValueError(
                f"Cannot cache gpugr parser DB because mapped gpdb nodes are not movable: {sample}"
            )

        gpdb_movable_node_ids = self._gpdb_movable_node_ids(gpdb)
        gpdb_movable_node_names = self._validate_parser_cache_movable_names(
            node_names,
            gpdb_node_names,
            gpdb_movable_node_ids,
        )

        movable_node_ids = torch.tensor(movable_node_id_list, dtype=torch.long, device="cpu")
        self._validate_parser_cache_node_lpos(
            gpdb,
            movable_node_ids,
            node_lpos,
            "initial parse",
        )
        full_node_lpos = gpdb.node_lpos_tensor().to(dtype=torch.float32).contiguous()
        if full_node_lpos.dtype != torch.float32:
            raise RuntimeError(
                f"gpugr parser cache full_node_lpos must be float32, got {full_node_lpos.dtype}"
            )
        self._parser_db_cache = {
            "key": cache_key,
            "rawdb": rawdb,
            "gpdb": gpdb,
            "row_y_orient_pairs": tuple(
                (int(row_y), int(orient)) for row_y, orient in gpdb.row_y_orient_pairs()
            ),
            "node_names": node_names,
            "node_count": int(node_count),
            "movable_node_ids": movable_node_ids,
            "mapped_node_names": tuple(
                gpdb_node_names[node_id] for node_id in movable_node_id_list
            ),
            "gpdb_node_name_count": len(gpdb_node_names),
            "gpdb_movable_node_count": len(gpdb_movable_node_ids),
            "gpdb_movable_node_names": gpdb_movable_node_names,
            "full_node_lpos": full_node_lpos,
            "full_node_lpos_dtype": str(full_node_lpos.dtype),
            "num_gpdb_nodes": int(full_node_lpos.size(0)),
        }
        return self._parser_db_cache

    def _apply_parser_cache_node_lpos(self, gpdb, node_lpos, node_names):
        cache = self._parser_db_cache
        if not cache:
            raise RuntimeError("gpugr parser cache is not initialized")
        if cache.get("node_count") != int(node_lpos.size(0)):
            raise RuntimeError("gpugr parser cache node count is stale")
        if node_names is None:
            raise RuntimeError(
                "gpugr parser cache requires node names for name-based coordinate updates"
            )
        node_names = tuple(node_names)
        if cache.get("node_names") != node_names:
            raise RuntimeError("gpugr parser cache node-name mapping is stale")
        if cache.get("mapped_node_names") != node_names:
            raise RuntimeError("gpugr parser cache mapped gpdb node names are stale")
        movable_node_ids = cache["movable_node_ids"]
        if int(movable_node_ids.numel()) != int(node_lpos.size(0)):
            raise RuntimeError("gpugr parser cache movable id mapping length is stale")
        if not bool(cache.get("static_apply_validation_done", False)):
            gpdb_node_names = [self._normalize_name(name) for name in gpdb.node_names()]
            if int(cache.get("gpdb_node_name_count", -1)) != len(gpdb_node_names):
                raise RuntimeError("gpugr parser cache gpdb node-name count is stale")
            mapped_node_names = tuple(
                gpdb_node_names[int(node_id)] for node_id in movable_node_ids.tolist()
            )
            if mapped_node_names != node_names:
                raise RuntimeError("gpugr parser cache gpdb node-name mapping changed")
            gpdb_movable_node_ids = self._gpdb_movable_node_ids(gpdb)
            if int(cache.get("gpdb_movable_node_count", -1)) != len(gpdb_movable_node_ids):
                raise RuntimeError("gpugr parser cache gpdb movable count is stale")
            gpdb_movable_node_names = tuple(
                gpdb_node_names[node_id] for node_id in gpdb_movable_node_ids
            )
            if cache.get("gpdb_movable_node_names") != gpdb_movable_node_names:
                raise RuntimeError("gpugr parser cache gpdb movable-name set is stale")
            cache["static_apply_validation_done"] = True
        full_node_lpos = cache["full_node_lpos"]
        if str(full_node_lpos.dtype) != cache.get("full_node_lpos_dtype", "torch.float32"):
            raise RuntimeError(
                f"gpugr parser cache full_node_lpos dtype changed: "
                f"{full_node_lpos.dtype} vs {cache.get('full_node_lpos_dtype')}"
            )
        current_num_lpos_nodes = int(gpdb.node_lpos_tensor().size(0))
        if int(full_node_lpos.size(0)) != current_num_lpos_nodes:
            full_node_lpos = gpdb.node_lpos_tensor().to(dtype=torch.float32).contiguous()
            if full_node_lpos.dtype != torch.float32:
                raise RuntimeError(
                    f"gpugr parser cache full_node_lpos must be float32, got {full_node_lpos.dtype}"
                )
            cache["full_node_lpos"] = full_node_lpos
            cache["full_node_lpos_dtype"] = str(full_node_lpos.dtype)
            cache["num_gpdb_nodes"] = int(full_node_lpos.size(0))
        full_node_lpos.index_copy_(0, movable_node_ids, node_lpos)
        row_y_orient_pairs = cache.get("row_y_orient_pairs")
        if row_y_orient_pairs is None:
            row_y_orient_pairs = tuple(
                (int(row_y), int(orient)) for row_y, orient in gpdb.row_y_orient_pairs()
            )
            cache["row_y_orient_pairs"] = row_y_orient_pairs
        if self.params.place_io_engine == "openroad":
            gpdb.apply_node_lpos_keep_orient(full_node_lpos)
        else:
            gpdb.apply_node_lpos_like_ieda_writeback(full_node_lpos, list(row_y_orient_pairs))
        gpdb.reset()
        gpdb.setup()
        if not bool(cache.get("dynamic_apply_validation_done", False)):
            self._validate_parser_cache_node_lpos(
                gpdb,
                movable_node_ids,
                node_lpos,
                "apply",
            )
            cache["dynamic_apply_validation_done"] = True
