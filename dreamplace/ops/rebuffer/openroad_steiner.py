"""OpenROAD-compatible placement Steiner provider.

The tree search is implemented in ``rebuffer_steiner_cpp``.  This module only
resolves the LUT paths and converts Python inputs to the native boundary.
"""

from pathlib import Path

ARTIFACT = "openroad_placement_steiner"
ARTIFACT_VERSION = 1

_DATA_DIR = Path(__file__).resolve().parents[3] / "thirdparty" / "xplace" / "thirdparty" / "flute"
_POST_LUT = _DATA_DIR / "POST9.dat"
_POWV_LUT = _DATA_DIR / "POWV9.dat"


def _extension():
    try:
        import dreamplace.ops.rebuffer.rebuffer_steiner_cpp as native
    except ImportError as exc:
        raise RuntimeError(
            "rebuffer_steiner_cpp is not built; the OpenROAD-compatible "
            "Steiner provider is unavailable"
        ) from exc
    return native


def lut_paths():
    return {"post": str(_POST_LUT), "powv": str(_POWV_LUT)}


def build_openroad_stt_tree(x, y, driver_index, *, alpha=0.3):
    """Return OpenROAD's default ``makeSteinerTree`` result.

    ``alpha > 0`` uses ``pdr::primDijkstra`` and falls back to OpenROAD
    FLUTE accuracy 3 when the PD tree fails OpenROAD's crossing check.
    """

    native = _extension()
    return dict(
        native.stt(
            [int(value) for value in x],
            [int(value) for value in y],
            int(driver_index),
            float(alpha),
            str(_POST_LUT),
            str(_POWV_LUT),
        )
    )


def build_openroad_flute_tree(x, y, *, accuracy=3):
    """Return OpenROAD ``stt::flt::Flute`` using its own LUT data."""

    native = _extension()
    return dict(
        native.flute(
            [int(value) for value in x],
            [int(value) for value in y],
            int(accuracy),
            str(_POST_LUT),
            str(_POWV_LUT),
        )
    )


def build_openroad_pd_tree(x, y, driver_index, *, alpha=0.3):
    """Return the raw PD tree without the FLUTE fallback."""

    native = _extension()
    return dict(
        native.prim_dijkstra(
            [int(value) for value in x],
            [int(value) for value in y],
            int(driver_index),
            float(alpha),
        )
    )
