"""Single factory for DreamPlace GPUGR backend ownership."""

from .xplace_backend import XplaceGPUGR, normalize_gpugr_backend


def create_gpugr_backend(params, placedb):
    backend = normalize_gpugr_backend(getattr(params, "gpugr_backend", "auto"))
    return XplaceGPUGR(params, placedb)


__all__ = ["create_gpugr_backend"]
