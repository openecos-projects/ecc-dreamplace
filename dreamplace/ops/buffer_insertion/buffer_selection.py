import re


_AUTO_X4_MASTER_PATTERN = re.compile(r"^buf_?x4(?:$|[^0-9])", re.IGNORECASE)


def resolve_fixed_buffer_selection(params, config, buffer_library):
    requested_master = str(
        getattr(config, "fixed_buffer_master", "")
        or getattr(params, "buffering_fixed_buffer_master", "")
        or ""
    ).strip()
    requested_index = getattr(config, "fixed_bsu_index", None)
    if requested_index is None:
        requested_index = getattr(params, "buffering_fixed_bsu_index", None)
    index_explicit = bool(
        getattr(params, "_buffering_fixed_bsu_index_explicit", False)
    )
    master_explicit = bool(
        getattr(params, "_buffering_fixed_buffer_master_explicit", False)
    )
    if index_explicit and not master_explicit:
        requested_master = ""

    entries = [
        (int(bsu), str(entry.get("master_name", "") or ""))
        for bsu, entry in sorted(buffer_library.items())
    ]
    if requested_master:
        if requested_master.casefold() == "auto:x4":
            matches = [
                (bsu, name)
                for bsu, name in entries
                if _AUTO_X4_MASTER_PATTERN.match(name)
            ]
            source = "auto_x4"
            if len(matches) > 1:
                # Multi-VT libraries expose one master per VT class at the
                # same drive strength. Default to the nominal-VT variant,
                # approximated by the median-delay candidate (LVT is fastest,
                # HVT slowest), without hardcoding PDK-specific VT suffixes.
                def _match_delay(item):
                    bsu, _name = item
                    try:
                        return float(
                            buffer_library[int(bsu)].get("delay", float("inf"))
                        )
                    except (TypeError, ValueError):
                        return float("inf")

                matches = sorted(matches, key=_match_delay)
                matches = [matches[len(matches) // 2]]
        else:
            requested_casefold = requested_master.casefold()
            matches = [
                (bsu, name)
                for bsu, name in entries
                if name.casefold() == requested_casefold
            ]
            source = "exact_master"
        if len(matches) != 1:
            candidates = ", ".join(name for _bsu, name in matches) or "<none>"
            raise ValueError(
                f"buffering_fixed_buffer_master={requested_master!r} matched "
                f"{len(matches)} legal masters: {candidates}"
            )
        resolved_index, resolved_master = matches[0]
        if (
            index_explicit
            and requested_index is not None
            and int(requested_index) != resolved_index
        ):
            raise ValueError(
                "buffering_fixed_buffer_master and explicit "
                "buffering_fixed_bsu_index resolve to different legal buffers: "
                f"{resolved_master} uses index {resolved_index}, got "
                f"{int(requested_index)}"
            )
        return resolved_index, resolved_master, source

    if requested_index is None:
        return None, "", "optimized"
    resolved_index = int(requested_index)
    by_index = dict(entries)
    if resolved_index not in by_index:
        raise ValueError(
            f"buffering_fixed_bsu_index={resolved_index} is outside legal "
            f"buffer table [0, {len(entries) - 1}]"
        )
    return resolved_index, by_index[resolved_index], "bsu_index"
