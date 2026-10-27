"""Validate the atomic schema-v3 sparse view before native materialization.

Integer identity and four-bit transition arrays are int64, electrical data
are float64 ps, state masks are uint8 (0/1). The producer remaps native identities
into placement pin and runtime base-arc domains. Digests cover the contiguous,
little-endian bytes.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

INTEGER_COLUMNS = {
    "pin_state_start": 0,
    "state_pin": 0,
    "state_domain": 0,
    "state_tag": 0,
    "state_launch_tag": 0,
    "state_source_type": 0,
    "state_role": 0,
    "state_level": 0,
    "state_slew_group": 0,
    "cell_edges": 3,
    "slew_only_cell_edges": 3,
    "net_edges": 3,
    "cell_transition_mask": 0,
    "slew_only_cell_transition_mask": 0,
    "net_transition_mask": 0,
    "check_index": 7,
    "context_state": 0,
}
OTHER_COLUMNS = {
    "net_threshold_adjustments": (np.dtype("float64"), 4),
    "pin_reference_cap": (np.dtype("float64"), 2),
    "state_valid": (np.dtype("uint8"), 2),
    "state_seed": (np.dtype("uint8"), 2),
    "state_reference": (np.dtype("float64"), 4),
    "check_values": (np.dtype("float64"), 4),
}
STATE_KEYS = (
    "pin_state_start",
    "state_pin",
    "state_domain",
    "state_tag",
    "state_launch_tag",
    "state_source_type",
    "state_role",
    "state_slew_group",
    "state_valid",
)
MODEL_KEYS = (
    "pin_reference_cap",
    "pin_state_start",
    "state_pin",
    "state_domain",
    "state_tag",
    "state_launch_tag",
    "state_source_type",
    "state_role",
    "state_level",
    "state_slew_group",
    "state_valid",
    "state_seed",
    "cell_edges",
    "slew_only_cell_edges",
    "net_edges",
    "cell_transition_mask",
    "slew_only_cell_transition_mask",
    "net_transition_mask",
    "net_threshold_adjustments",
    "check_index",
    "context_state",
    "check_values",
)
CAPABILITIES = {"native_required_affine_v1", "native_transition_mask_v1"}


def sparse_digest(arrays, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for name, value in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(memoryview(value))
    return digest.hexdigest()


def model_rule_units(values):
    """Integer 1e-9 ps rule units, rounded like the producer's std::llround.

    llround rounds halves away from zero; np.rint rounds halves to even and
    disagrees with the producer on exact .5 values.
    """
    scaled = values * 1e9
    return (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).astype(np.int64)


def model_digest(arrays, metadata):
    """Digest structure and rules, excluding references and float roundoff."""
    model_arrays = {name: arrays[name] for name in MODEL_KEYS if name != "check_values"}
    # Native model preparation can differ from complete STA by sub-femtosecond
    # rounding in the affine intercept. Match the producer's 1e-9 ps integer
    # canonicalization while retaining sensitivity to meaningful rule changes.
    model_arrays["check_values"] = model_rule_units(arrays["check_values"][:, :3])
    return sparse_digest(model_arrays, metadata)


def _require(condition, message):
    if not condition:
        raise RuntimeError(f"ECC sparse timing schema: {message}")


def _range(values, count, name, minimum=0):
    _require(np.all((values >= minimum) & (values < count)), f"invalid {name} identity")


@dataclass(frozen=True)
class SparseTimingView:
    arrays: dict[str, np.ndarray]
    domains: tuple[str, ...]
    tags: tuple[str, ...]
    unsupported: tuple[str, ...]
    semantic_digest: str
    state_key_digest: str
    model_digest: str = ""
    launch_tags: tuple[str, ...] = ()
    early: bool = False

    def require_supported(self):
        _require(not self.unsupported, "; ".join(self.unsupported))

    def statistics(self):
        arrays = self.arrays
        counts = np.diff(arrays["pin_state_start"])
        kinds = arrays["check_index"][:, 5]
        return {
            "num_pins": len(counts),
            "num_states": len(arrays["state_pin"]),
            "states_per_pin_mean": float(counts.mean()) if len(counts) else 0.0,
            "states_per_pin_max": int(counts.max(initial=0)),
            "num_launch_domains": len(self.domains),
            "num_propagation_tags": len(self.tags),
            "num_slew_groups": int(arrays["state_slew_group"].max(initial=-1)) + 1,
            "num_tagged_cell_arcs": len(arrays["cell_edges"]),
            "num_slew_only_cell_arcs": len(arrays["slew_only_cell_edges"]),
            "num_tagged_net_arcs": len(arrays["net_edges"]),
            "num_endpoint_candidates": len(kinds),
            "num_endpoint_contexts": len(arrays["context_state"]),
            "num_legal_endpoint_candidates": int(arrays["check_index"][:, 6].sum()),
            "rule_counts": {
                name: int(np.count_nonzero(kinds == code))
                for code, name in enumerate(("normal", "max_delay", "multicycle", "false_path"))
            },
            "array_bytes": sum(a.nbytes for a in arrays.values()),
            "unsupported": list(self.unsupported),
            "semantic_digest": self.semantic_digest,
            "state_key_digest": self.state_key_digest,
            "model_digest": self.model_digest,
        }


def read_sparse_timing_view(pydb, *, analysis="max"):
    view = pydb.timing_min_sparse if analysis == "min" else pydb.timing_sparse
    metadata, arrays = view["metadata"], view["arrays"]
    format_version = metadata["format_version"]
    _require(format_version in (2, 3), "unsupported view format")
    _require(metadata["analysis"] == analysis, "wrong analysis domain")
    _require(
        CAPABILITIES | {f"sparse_{analysis}_v1"} <= set(metadata["capabilities"]),
        "missing producer capability",
    )
    _require(
        set(arrays) == INTEGER_COLUMNS.keys() | OTHER_COLUMNS.keys(),
        "array field mismatch",
    )
    for name, columns in INTEGER_COLUMNS.items():
        value = arrays[name]
        _require(
            isinstance(value, np.ndarray) and value.dtype == np.int64,
            f"{name} must be int64",
        )
        _require(value.ndim == (2 if columns else 1), f"invalid {name} rank")
        if columns:
            _require(value.shape[1] == columns, f"invalid {name} columns")
    for name, (dtype, columns) in OTHER_COLUMNS.items():
        value = arrays[name]
        _require(
            isinstance(value, np.ndarray) and value.dtype == dtype,
            f"invalid {name} dtype",
        )
        _require(value.ndim == 2 and value.shape[1] == columns, f"invalid {name} shape")
        _require(np.all(np.isfinite(value)), f"nonfinite {name}")
    _require(all(a.flags.c_contiguous for a in arrays.values()), "arrays must be contiguous")
    _require(
        all(a.dtype.byteorder in ("=", "<", "|") for a in arrays.values()),
        "unsupported endianness",
    )
    domains, tags, launch_tags = metadata["domains"], metadata["tags"], metadata["launch_tags"]
    _require(
        len(set(domains)) == len(domains)
        and len(set(tags)) == len(tags)
        and len(set(launch_tags)) == len(launch_tags),
        "duplicate domain/tag",
    )
    pin_count = len(pydb.pin_names)
    _require(
        arrays["pin_reference_cap"].shape == (pin_count, 2),
        "pin capacitance rows mismatch",
    )
    state_pin = arrays["state_pin"]
    states = len(state_pin)
    offsets = arrays["pin_state_start"]
    _require(offsets.shape == (pin_count + 1,), "pin CSR size mismatch")
    _require(
        offsets[0] == 0 and offsets[-1] == states and np.all(np.diff(offsets) >= 0),
        "invalid pin CSR offsets",
    )
    _range(state_pin, pin_count, "state pin")
    _require(
        np.array_equal(state_pin, np.repeat(np.arange(pin_count), np.diff(offsets))),
        "pin CSR ownership mismatch",
    )
    for name in (
        "state_domain",
        "state_tag",
        "state_launch_tag",
        "state_source_type",
        "state_role",
        "state_level",
        "state_slew_group",
        "state_valid",
        "state_seed",
        "state_reference",
    ):
        _require(len(arrays[name]) == states, f"{name} row mismatch")
    _range(arrays["state_domain"], len(domains), "domain")
    _range(arrays["state_tag"], len(tags), "tag")
    _range(arrays["state_launch_tag"], len(launch_tags), "launch tag")
    _range(arrays["state_source_type"], 3, "source type", minimum=0 if format_version >= 3 else 1)
    electrical_only = arrays["state_source_type"] == 0
    _require(
        not np.any(arrays["state_valid"][electrical_only])
        and not np.any(arrays["state_seed"][electrical_only]),
        "electrical-only state has an arrival transition or seed",
    )
    _range(arrays["state_role"], 2, "pin role")
    _require(np.all(arrays["state_level"] >= 0), "negative state level")
    _require(np.all(arrays["state_slew_group"] >= 0), "negative slew group")
    _range(
        arrays["state_slew_group"],
        int(arrays["state_slew_group"].max(initial=-1)) + 1,
        "slew group",
    )
    _require(
        np.all(arrays["state_valid"] <= 1)
        and np.all(arrays["state_seed"] <= arrays["state_valid"]),
        "invalid state/seed mask",
    )
    keys = np.column_stack(
        (
            state_pin,
            arrays["state_domain"],
            arrays["state_tag"],
            arrays["state_source_type"],
        )
    )
    _require(
        np.array_equal(np.lexsort(keys[:, ::-1].T), np.arange(states)),
        "unsorted state identities",
    )
    _require(not np.any(np.all(keys[1:] == keys[:-1], axis=1)), "duplicate state identity")
    cell = np.asarray(pydb.flat_inst_arcs_by_level, dtype=np.int64).reshape(-1, 8)
    _require(
        len(arrays["net_threshold_adjustments"]) == len(arrays["net_edges"])
        and np.all(arrays["net_threshold_adjustments"][:, 2:] > 0),
        "invalid net threshold adjustments",
    )
    for name, base_count in (
        ("cell_edges", len(cell)),
        ("slew_only_cell_edges", len(cell)),
        ("net_edges", pin_count),
    ):
        edges = arrays[name]
        mask_name = name.replace("edges", "transition_mask")
        masks = arrays[mask_name]
        _require(len(masks) == len(edges), f"{name} transition rows mismatch")
        minimum_mask = 0 if format_version >= 3 and name == "net_edges" else 1
        _require(np.all((masks >= minimum_mask) & (masks <= 15)), f"{name} invalid transition mask")
        _range(edges[:, 0], base_count, f"{name} base")
        _range(edges[:, 1:], states, f"{name} state")
        src, dst = edges[:, 1], edges[:, 2]
        arrival_edges = masks != 0
        if name != "slew_only_cell_edges":
            _require(
                np.array_equal(arrays["state_domain"][src[arrival_edges]], arrays["state_domain"][dst[arrival_edges]]),
                f"{name} changes domain",
            )
            _require(
                np.array_equal(
                    arrays["state_source_type"][src[arrival_edges]],
                    arrays["state_source_type"][dst[arrival_edges]],
                ),
                f"{name} changes source type",
            )
            _require(
                np.array_equal(arrays["state_launch_tag"][src[arrival_edges]], arrays["state_launch_tag"][dst[arrival_edges]]),
                f"{name} changes launch identity",
            )
            _require(
                not np.any(electrical_only[src[arrival_edges]])
                and not np.any(electrical_only[dst[arrival_edges]]),
                "electrical-only state transports an arrival",
            )
        for input_edge in range(2):
            for output_edge in range(2):
                used = masks & (1 << (2 * input_edge + output_edge)) != 0
                _require(
                    name == "slew_only_cell_edges" or np.all(
                        ~used
                        | (
                            arrays["state_valid"][src, input_edge]
                            & arrays["state_valid"][dst, output_edge]
                        ).astype(bool)
                    ),
                    f"{name} transition uses an invalid state lane",
                )
        if name == "net_edges":
            _require(np.all(masks & ~9 == 0), "net edge changes transition")
            _require(np.array_equal(edges[:, 0], state_pin[dst]), "net base sink mismatch")
            net_arcs = np.asarray(pydb.net_flat_arcs, dtype=np.int64).reshape(-1, 2)
            base_keys = net_arcs[:, 0] * pin_count + net_arcs[:, 1]
            _require(
                np.all(np.isin(state_pin[src] * pin_count + state_pin[dst], base_keys)),
                "net edge has no base arc",
            )
        else:
            base = cell[edges[:, 0]]
            legal_masks = np.where(base[:, 4] == 1, 9, np.where(base[:, 4] == -1, 6, 15))
            triggers = np.where(base[:, 5] == 1, 3, np.where(base[:, 5] == -1, 12, 15))
            _require(
                np.all(masks & ~(legal_masks & triggers) == 0),
                "cell transition contradicts sense/trigger",
            )
            sources = base[:, 0].copy()
            clock = edges[:, 0] < pydb.flat_inst_arcs_by_level_start[1]
            sources[clock] = np.asarray(pydb.clock_pins)[sources[clock]]
            _require(
                np.array_equal(sources, state_pin[src])
                and np.array_equal(base[:, 1], state_pin[dst]),
                "cell base/state pin mismatch",
            )
    checks = arrays["check_index"]
    _range(arrays["context_state"], states, "context state")
    _require(not np.any(electrical_only[arrays["context_state"]]), "hold context uses an electrical-only state")
    _require(
        len(arrays["context_state"]) == (len(pydb.hold_contexts) if analysis == "min" else 0),
        "context rows mismatch",
    )
    _require(len(checks) == len(arrays["check_values"]), "check value rows mismatch")
    _range(checks[:, 0], states, "check launch state")
    _require(not np.any(electrical_only[checks[:, 0]]), "check uses an electrical-only state")
    _range(checks[:, 1], len(domains), "check capture domain")
    _range(checks[:, 2], len(pydb.endpoints_timing_check_arcs), "endpoint arc", minimum=-1)
    _range(checks[:, 3], 2, "check transition")
    _range(checks[:, 5], 4, "rule kind")
    _range(checks[:, 6], 2, "check legal mask")
    _require(
        np.all((checks[:, 5] == 3) == (checks[:, 6] == 0)),
        "false path legality mismatch",
    )
    _require(
        np.all(arrays["state_valid"][checks[:, 0], checks[:, 3]]),
        "check uses invalid transition",
    )
    _require(
        np.all(np.isin(state_pin[checks[:, 0]], pydb.end_points)),
        "check pin is not an endpoint",
    )
    _require(
        sparse_digest(arrays, metadata) == view["semantic_digest"],
        "semantic digest mismatch",
    )
    _require("model_digest" in view, "missing model digest")
    _require(
        model_digest(arrays, metadata) == view["model_digest"],
        "model digest mismatch",
    )
    state_digest = sparse_digest(
        {key: arrays[key] for key in STATE_KEYS},
        {"domains": domains, "tags": tags, "launch_tags": launch_tags},
    )
    _require(state_digest == view["state_key_digest"], "state key digest mismatch")
    return SparseTimingView(
        arrays,
        tuple(domains),
        tuple(tags),
        tuple(metadata["unsupported"]),
        view["semantic_digest"],
        state_digest,
        view["model_digest"],
        tuple(launch_tags),
        analysis == "min",
    )
