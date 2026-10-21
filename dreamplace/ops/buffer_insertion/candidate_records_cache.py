"""Signatures for reusing candidate records across geometry refreshes."""

import hashlib
import json

import torch

_STRUCTURE_TENSOR_NAMES = (
    "net_flat_topo_sort",
    "net_flat_topo_sort_start",
    "pin_fa",
    "flat_pin_to",
    "flat_pin_to_start",
)

_GEOMETRY_TENSOR_NAMES = (
    "node_x_dbu",
    "node_y_dbu",
)


def candidate_topology_signature(
    topology,
    topology_epoch=None,
    *,
    include_geometry=True,
):
    """Return a stable signature for the candidate-expanded topology domain."""

    digest = hashlib.sha256()
    digest.update(
        str(int((topology or {}).get("topology_generation", 0) or 0)).encode(
            "ascii"
        )
    )
    digest.update(
        str(
            json.dumps(
                topology_epoch,
                sort_keys=True,
                separators=(",", ":"),
            )
            if topology_epoch is not None
            else None
        ).encode("utf-8")
    )
    names = _STRUCTURE_TENSOR_NAMES + (
        _GEOMETRY_TENSOR_NAMES if include_geometry else ()
    )
    for name in names:
        value = (topology or {}).get(name)
        digest.update(name.encode("ascii"))
        if value is None:
            digest.update(b"<none>")
            continue
        tensor = torch.as_tensor(value).detach().cpu().contiguous().reshape(-1)
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def candidate_topology_structure_signature(
    topology,
    topology_epoch=None,
):
    """Return a signature that changes only with candidate tree structure."""

    return candidate_topology_signature(
        topology,
        topology_epoch=topology_epoch,
        include_geometry=False,
    )


__all__ = [
    "candidate_topology_signature",
    "candidate_topology_structure_signature",
]
