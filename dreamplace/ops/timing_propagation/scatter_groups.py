"""Stable reduction groups owned by the source topology tensor.

Views of the same immutable topology share groups. In-place topology edits
invalidate all views; stream-local entries need no cross-stream synchronization.
The cache retains only derived indices, so it cannot keep its owner alive.
"""

from typing import NamedTuple

import torch


class ScatterGroups(NamedTuple):
    order: torch.Tensor
    offsets: torch.Tensor


class _GroupCache(NamedTuple):
    version: int
    views: dict[tuple, ScatterGroups]


def _build(index, destinations):
    index = index.long().contiguous()
    order = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=destinations)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return ScatterGroups(order, offsets)


def scatter_groups(index, destinations):
    """Group valid destination indices in their original source order."""
    if index.is_inference():
        return _build(index, destinations)
    owner = index
    while owner._base is not None:
        owner = owner._base
    version = index._version
    cache = getattr(owner, "_timing_scatter_groups", None)
    if cache is None or cache.version != version:
        cache = _GroupCache(version, {})
        owner._timing_scatter_groups = cache
    stream = torch.cuda.current_stream(index.device).cuda_stream if index.is_cuda else None
    key = (
        index.data_ptr(),
        tuple(index.shape),
        tuple(index.stride()),
        index.dtype,
        index.device,
        destinations,
        stream,
    )
    if key not in cache.views:
        cache.views[key] = _build(index, destinations)
    return cache.views[key]
