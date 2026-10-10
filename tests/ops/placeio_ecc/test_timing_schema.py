from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.timing_schema import (
    read_qualification_arrays,
    validate_timing_schema,
)


@pytest.fixture
def pydb():
    return SimpleNamespace(
        timing_schema_version=2,
        pin_names=["clk", "q", "d", "unclocked"],
        clock_pins=[0],
        end_points=[2, 3],
        endpoints_constraint_arcs=[[0, 2, 0, 0, 1, 1, 1, 0]],
        endpoints_timing_check_arcs=[[0, 2, 0, 0, 1, 1, 1, 0]],
        endpoints_max_valid=[[True, False], [False, False]],
        endpoints_max_reason=[[0, 3], [2, 2]],
        endpoints_constraint_max_valid=[[True, False]],
        endpoints_constraint_max_reason=[[0, 3]],
        endpoints_timing_check_max_valid=[[True, False]],
        endpoints_timing_check_max_reason=[[0, 3]],
    )


def test_schema_preserves_row_edge_qualification(pydb):
    validate_timing_schema(pydb)
    arrays = read_qualification_arrays(pydb)
    np.testing.assert_array_equal(arrays["endpoints_max_valid"], [[True, False], [False, False]])
    assert arrays["endpoints_max_reason"].dtype == np.int32


@pytest.mark.parametrize(
    "field,value",
    [
        ("timing_schema_version", 1),
        ("endpoints_max_valid", [[True, False]]),
        ("endpoints_max_reason", [[0, 0], [2, 2]]),
        ("endpoints_constraint_arcs", [[2, 2, 0, 0, 1, 1, 1, 0]]),
        ("endpoints_constraint_arcs", [[0, 3, 0, 0, 1, 1, 1, 0]]),
        ("endpoints_constraint_max_reason", [[0, 6]]),
    ],
)
def test_schema_rejects_stale_or_misaligned_native_metadata(pydb, field, value):
    setattr(pydb, field, value)
    with pytest.raises(RuntimeError):
        validate_timing_schema(pydb)


def test_schema_requires_registered_fields(pydb):
    del pydb.endpoints_max_valid
    with pytest.raises(RuntimeError, match="missing endpoints_max_valid"):
        validate_timing_schema(pydb)


def test_empty_native_lists_convert_to_two_edge_arrays(pydb):
    pydb.end_points = []
    pydb.endpoints_constraint_arcs = []
    pydb.endpoints_timing_check_arcs = []
    for name in read_qualification_arrays(pydb):
        setattr(pydb, name, [])
    validate_timing_schema(pydb)
    assert all(array.shape == (0, 2) for array in read_qualification_arrays(pydb).values())
