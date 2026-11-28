"""Source-library inputs owned by the placeio_ecc PyDB metadata boundary."""


def _paths(value):
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(path) for path in (value or ()))


class RepairLibrarySource:
    """Lazily import immutable native metadata once across carrier refreshes."""

    def __init__(self, liberties, lefs):
        self.liberties = _paths(liberties)
        self.lefs = _paths(lefs)
        self._names = None
        self._native = None

    def read(self, names):
        from dreamplace.ops.rebuffer import rebuffer_cpp

        names = tuple(str(name) for name in names)
        if self._native is None:
            self._native = rebuffer_cpp.read_repair_library_metadata(
                self.liberties, self.lefs, names
            )
            self._names = names
        elif names != self._names:
            raise ValueError("repair library cell identities changed after metadata import")
        return self._native


def attach_repair_library_source(pydb, params, inputs):
    pydb.repair_library_source = RepairLibrarySource(
        inputs.get("lib_paths") or inputs.get("lib"),
        inputs.get("lef_paths") or inputs.get("lef") or getattr(params, "lef_input", ()),
    )


def ensure_repair_library_metadata(pydb):
    source = getattr(pydb, "repair_library_source", None)
    if source is None:
        raise ValueError("target-cell selection requires placeio_ecc library sources")
    pydb.repair_library_metadata = source.read(pydb.flat_libcell_names)
    return pydb.repair_library_metadata


def validate_buffer_resize_pairs(pydb, sources, targets, cell_master_names):
    """Qualify cross-family writes before the existing native batch mutation."""
    from dreamplace.ops.rebuffer import rebuffer_cpp
    from dreamplace.ops.rebuffer.liberty import build_liberty_view_from_pydb

    if list(cell_master_names) != list(pydb.flat_libcell_names):
        raise ValueError("buffer sizing cell names differ from PyDB Liberty identities")
    ensure_repair_library_metadata(pydb)
    rebuffer_cpp.validate_buffer_resize_pairs(build_liberty_view_from_pydb(pydb), sources, targets)
