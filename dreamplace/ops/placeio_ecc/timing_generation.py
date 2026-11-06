"""Bind timing tensors to the validated native database generation they use."""

from dataclasses import dataclass

from .place_io import ECCPlaceIOBackend


@dataclass(frozen=True)
class TimingGenerationLease:
    backend: ECCPlaceIOBackend
    pydb_id: int
    generation: int

    def require_current(self):
        if (
            not self.backend.timing_enabled
            or id(self.backend.pydb) != self.pydb_id
            or self.backend.refresh_generation != self.generation
        ):
            raise RuntimeError(
                "ECC timing tensors belong to an invalid or replaced native generation; "
                "rebuild the placement timing operators after refresh"
            )


def timing_generation_lease(placedb):
    backend = getattr(placedb, "rawdb", None)
    if not isinstance(backend, ECCPlaceIOBackend):
        return None
    # Keep identity without retaining an entire superseded geometry snapshot.
    # The generation check also rejects identity reuse after its release.
    lease = TimingGenerationLease(backend, id(placedb.pydb), backend.refresh_generation)
    lease.require_current()
    return lease
