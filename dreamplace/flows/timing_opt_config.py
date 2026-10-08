"""Canonical names and compatibility handling for placement timing options."""

import copy

TIMING_OPT_ALIASES = (
    ("timing_opt_enabled", "inflation_s5b1_enabled"),
    ("timing_opt_buffering_enabled", "inflation_s5b1_buffering_enabled"),
    ("timing_opt_max_windows", "inflation_s5b1_max_windows"),
    ("timing_opt_overflow_milestones", "inflation_s5b1_overflow_milestones"),
    ("timing_opt_sizing_rounds", "inflation_sizing_rounds"),
)

_DEFAULTS = {
    "timing_opt_enabled": 0,
    "timing_opt_buffering_enabled": 1,
    "timing_opt_max_windows": 5,
    "timing_opt_overflow_milestones": [],
    "timing_opt_sizing_rounds": 5,
}
_LEGACY_NAMES = {legacy for _, legacy in TIMING_OPT_ALIASES}


def _values_equal(left, right):
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return list(left) == list(right)
    return left == right


def normalize_timing_opt_params(params):
    """Normalize old placement window fields to the ``timing_opt_*`` names.

    Existing profiles may still contain ``inflation_s5b1_*`` fields.  A
    canonical schema default may coexist with one of those legacy values; the
    explicit legacy value then replaces that default.  If both spellings are
    explicitly supplied with different values, fail before flow defaults can
    make a silent choice.
    """
    for canonical, legacy in TIMING_OPT_ALIASES:
        has_canonical = hasattr(params, canonical)
        has_legacy = hasattr(params, legacy)
        canonical_explicit = bool(
            getattr(params, f"_{canonical}_explicit", False)
        )
        legacy_explicit = bool(getattr(params, f"_{legacy}_explicit", False))
        default = _DEFAULTS[canonical]

        if has_canonical and has_legacy:
            canonical_value = getattr(params, canonical)
            legacy_value = getattr(params, legacy)
            if not _values_equal(canonical_value, legacy_value):
                if legacy_explicit and not canonical_explicit:
                    canonical_value = legacy_value
                elif canonical_explicit and not legacy_explicit:
                    pass
                elif not canonical_explicit and not legacy_explicit and _values_equal(
                    canonical_value, default
                ):
                    canonical_value = legacy_value
                else:
                    raise ValueError(
                        "conflicting timing optimization parameters: "
                        f"{canonical}={canonical_value!r} and "
                        f"{legacy}={legacy_value!r}"
                    )
        elif has_legacy:
            canonical_value = getattr(params, legacy)
        elif has_canonical:
            canonical_value = getattr(params, canonical)
        else:
            canonical_value = copy.deepcopy(default)

        setattr(params, canonical, canonical_value)
        if legacy_explicit:
            setattr(params, f"_{canonical}_explicit", True)
        # Consume input aliases once. A stale legacy value must not undo an
        # internal canonical update, such as disabling windows for legalization.
        if has_legacy:
            delattr(params, legacy)
        if hasattr(params, f"_{legacy}_explicit"):
            delattr(params, f"_{legacy}_explicit")
    return params


def is_legacy_timing_opt_name(name):
    return name in _LEGACY_NAMES
