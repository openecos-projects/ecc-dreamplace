"""Validate the atomic schema-v3 sparse view before native materialization.

Integer identity arrays are int64, electrical data are float64 ps, masks are
uint8 (0/1). The producer remaps native identities into placement pin and
runtime base-arc domains. Digests cover the contiguous, little-endian bytes.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np


INTEGER_COLUMNS = {
    "pin_state_start": 0,
    "state_pin": 0,
    "state_domain": 0,
    "state_tag": 0,
    "state_source_type": 0,
    "state_level": 0,
    "state_slew_group": 0,
    "cell_edges": 3,
    "slew_only_cell_edges": 3,
    "net_edges": 3,
    "check_index": 7,
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
    "state_source_type",
    "state_slew_group",
    "state_valid",
)
CAPABILITIES = {"sparse_max_v1", "native_required_affine_v1"}


def sparse_digest(arrays, metadata):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for name, value in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(memoryview(value))
    return digest.hexdigest()


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
            "num_legal_endpoint_candidates": int(arrays["check_index"][:, 6].sum()),
            "rule_counts": {
                name: int(np.count_nonzero(kinds == code))
                for code, name in enumerate(
                    ("normal", "max_delay", "multicycle", "false_path")
                )
            },
            "array_bytes": sum(a.nbytes for a in arrays.values()),
            "unsupported": list(self.unsupported),
            "semantic_digest": self.semantic_digest,
            "state_key_digest": self.state_key_digest,
        }


def read_sparse_timing_view(pydb):
    view = pydb.timing_sparse
    metadata, arrays = view["metadata"], view["arrays"]
    _require(metadata["format_version"] == 1, "unsupported view format")
    _require(
        CAPABILITIES <= set(metadata["capabilities"]), "missing producer capability"
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
    _require(
        all(a.flags.c_contiguous for a in arrays.values()), "arrays must be contiguous"
    )
    _require(
        all(a.dtype.byteorder in ("=", "<", "|") for a in arrays.values()),
        "unsupported endianness",
    )
    domains, tags = metadata["domains"], metadata["tags"]
    _require(
        len(set(domains)) == len(domains) and len(set(tags)) == len(tags),
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
        "state_source_type",
        "state_level",
        "state_slew_group",
        "state_valid",
        "state_seed",
        "state_reference",
    ):
        _require(len(arrays[name]) == states, f"{name} row mismatch")
    _range(arrays["state_domain"], len(domains), "domain")
    _range(arrays["state_tag"], len(tags), "tag")
    _range(arrays["state_source_type"], 3, "source type", minimum=1)
    _require(np.all(arrays["state_level"] >= 0), "negative state level")
    _require(np.all(arrays["state_slew_group"] >= 0), "negative slew group")
    _range(arrays["state_slew_group"], int(arrays["state_slew_group"].max(initial=-1)) + 1, "slew group")
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
    _require(
        not np.any(np.all(keys[1:] == keys[:-1], axis=1)), "duplicate state identity"
    )
    cell = np.asarray(pydb.flat_inst_arcs_by_level, dtype=np.int64).reshape(-1, 8)
    _require(
        len(arrays["net_threshold_adjustments"]) == len(arrays["net_edges"])
        and np.all(arrays["net_threshold_adjustments"][:, 2:] > 0),
        "invalid net threshold adjustments",
    )
    for name, base_count in (("cell_edges", len(cell)), ("slew_only_cell_edges", len(cell)), ("net_edges", pin_count)):
        edges = arrays[name]
        _range(edges[:, 0], base_count, f"{name} base")
        _range(edges[:, 1:], states, f"{name} state")
        src, dst = edges[:, 1], edges[:, 2]
        if name != "slew_only_cell_edges":
            for field, message in (("state_domain", "changes domain"), ("state_tag", "needs unsupported tag transition"), ("state_source_type", "changes source type")):
                _require(np.array_equal(arrays[field][src], arrays[field][dst]), f"{name} {message}")
        if name == "net_edges":
            _require(
                np.array_equal(edges[:, 0], state_pin[dst]), "net base sink mismatch"
            )
            net_arcs = np.asarray(pydb.net_flat_arcs, dtype=np.int64).reshape(-1, 2)
            base_keys = net_arcs[:, 0] * pin_count + net_arcs[:, 1]
            _require(
                np.all(np.isin(state_pin[src] * pin_count + state_pin[dst], base_keys)),
                "net edge has no base arc",
            )
        else:
            base = cell[edges[:, 0]]
            sources = base[:, 0].copy()
            clock = edges[:, 0] < pydb.flat_inst_arcs_by_level_start[1]
            sources[clock] = np.asarray(pydb.clock_pins)[sources[clock]]
            _require(
                np.array_equal(sources, state_pin[src])
                and np.array_equal(base[:, 1], state_pin[dst]),
                "cell base/state pin mismatch",
            )
    checks = arrays["check_index"]
    _require(len(checks) == len(arrays["check_values"]), "check value rows mismatch")
    _range(checks[:, 0], states, "check launch state")
    _range(checks[:, 1], len(domains), "check capture domain")
    _range(
        checks[:, 2], len(pydb.endpoints_timing_check_arcs), "endpoint arc", minimum=-1
    )
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
    state_digest = sparse_digest(
        {key: arrays[key] for key in STATE_KEYS}, {"domains": domains, "tags": tags}
    )
    _require(state_digest == view["state_key_digest"], "state key digest mismatch")
    return SparseTimingView(
        arrays,
        tuple(domains),
        tuple(tags),
        tuple(metadata["unsupported"]),
        view["semantic_digest"],
        state_digest,
    )
