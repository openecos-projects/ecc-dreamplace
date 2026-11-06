from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.hold_schema import read_hold_timing_schema


@pytest.fixture
def pydb():
    return SimpleNamespace(
        timing_schema_version=2,
        hold_schema_version=2,
        timing_clock_names=["clk"],
        pin_names=["clk", "q", "d"],
        flat_lib_pin_cap=[0.002],
        flat_lib_pin_min_rcap=[0.0018],
        flat_lib_pin_min_fcap=[0.0017],
        flat_lib_pin_rcap=[0.0028],
        flat_lib_pin_fcap=[0.0027],
        flat_libarc_info=[[0, 2, 0, 0, 1, 1]],
        flat_libarc_check_types=[2],
        flat_libarc_is_latch_data=[False],
        clock_pins=[0],
        start_points=[1],
        end_points=[2],
        min_outcaps_rise=[0.0],
        min_outcaps_fall=[0.0],
        endpoints_constraint_arcs=[],
        endpoints_constraint_max_valid=[],
        endpoints_constraint_max_reason=[],
        endpoints_timing_check_arcs=[[0, 2, 0, 0, 1, 1, 2, 0]],
        endpoints_max_valid=[[False, False]],
        endpoints_max_reason=[[4, 4]],
        endpoints_timing_check_max_valid=[[False, False]],
        endpoints_timing_check_max_reason=[[4, 4]],
        endpoints_min_valid=[[True, False]],
        endpoints_min_reason=[[0, 3]],
        endpoints_timing_check_min_valid=[[True, False]],
        endpoints_timing_check_min_reason=[[0, 3]],
        startpoints_min_valid=[[True, True]],
        min_inrdelays=[5.0],
        min_infdelays=[6.0],
        min_inrtrans=[2.0],
        min_inftrans=[3.0],
        hold_launches=[
            dict(
                start_pin=1,
                launch_pin=0,
                launch_clock="clk",
                path_tag="clk",
                source_type=2,
                data_transition=1,
                launch_transition=1,
                arrival_ps=0.0,
                slew_ps=1.0,
                launch_time_ps=0.0,
                launch_slew_ps=1.0,
            )
        ],
        hold_contexts=[
            dict(
                start_pin=1,
                native_start_pin=0,
                launch_pin=0,
                endpoint_pin=2,
                capture_pin=0,
                launch_clock="clk",
                capture_clock="clk",
                path_tag="clk",
                source_type=2,
                check_type=2,
                data_transition=1,
                launch_transition=1,
                capture_transition=1,
                capture_pin_transition=1,
                capture_edge_index=-1,
                selected_check_arc_idx=-1,
                arrival_ps=10.0,
                required_ps=19.0,
                data_slew_ps=2.0,
                normal_required_ps=0.0,
                launch_time_ps=0.0,
                launch_clock_arrival_ps=0.0,
                launch_clock_slew_ps=1.0,
                capture_edge_ps=0.0,
                capture_arrival_ps=20.0,
                capture_slew_ps=3.0,
                check_time_ps=-2.0,
                uncertainty_ps=4.0,
                cppr_ps=3.0,
            )
        ],
    )


def test_min_only_endpoint_and_signed_check_keep_paired_state(pydb):
    result = read_hold_timing_schema(pydb)
    expected = dict(pydb.hold_contexts[0])
    expected.update(capture_pin_transition=1, capture_edge_index=-1, selected_check_arc_idx=-1, normal_required_ps=0.0)
    assert asdict(result.contexts[0]) == expected
    assert result.contexts[0].required_base_ps == 21.0
    np.testing.assert_array_equal(result.endpoint_valid, [[True, False]])
    assert not result.input_arrival.flags.writeable


def test_hold_pin_capacitance_uses_min_range_independently_of_max(pydb):
    result = read_hold_timing_schema(pydb)
    np.testing.assert_array_equal(result.min_pin_caps, [[0.0018], [0.0017]])
    assert all(not table.flags.writeable for table in result.min_pin_caps)


def test_multiple_capture_contexts_are_not_collapsed_to_one_arrival(pydb):
    second = dict(pydb.hold_contexts[0], arrival_ps=20.0, required_ps=29.0, capture_arrival_ps=30.0)
    pydb.hold_contexts.append(second)
    result = read_hold_timing_schema(pydb)
    expected = [dict(row) for row in pydb.hold_contexts]
    for row in expected:
        row.update(capture_pin_transition=1, capture_edge_index=-1, selected_check_arc_idx=-1, normal_required_ps=0.0)
    assert [asdict(context) for context in result.contexts] == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("hold_schema_version", 0),
        ("endpoints_min_valid", [[True, True]]),
        ("endpoints_min_reason", [[0, 7]]),
        ("endpoints_timing_check_min_reason", [[0, 6]]),
        ("min_inrdelays", []),
    ],
)
def test_stale_missing_or_misaligned_min_payload_is_rejected(pydb, field, value):
    setattr(pydb, field, value)
    with pytest.raises(RuntimeError):
        read_hold_timing_schema(pydb)


@pytest.mark.parametrize(
    "field,value",
    [
        ("required_ps", 17.0),
        ("capture_pin", 1),
        ("endpoint_pin", 3),
        ("capture_clock", "undeclared"),
        ("data_transition", -1),
        ("arrival_ps", float("nan")),
        ("check_type", 1),
    ],
)
def test_context_identity_edge_and_required_components_are_checked(pydb, field, value):
    pydb.hold_contexts[0][field] = value
    with pytest.raises(RuntimeError):
        read_hold_timing_schema(pydb)


def test_qualified_endpoint_requires_real_context(pydb):
    pydb.hold_contexts = []
    with pytest.raises(RuntimeError, match="cover"):
        read_hold_timing_schema(pydb)


def test_each_min_check_group_requires_its_own_capture_context(pydb):
    pydb.flat_libarc_info.append([1, 2, 0, 1, 1, -1])
    pydb.flat_libarc_check_types.append(4)
    pydb.flat_libarc_is_latch_data.append(False)
    pydb.endpoints_timing_check_arcs.append([1, 2, 0, 1, 1, -1, 4, 0])
    pydb.endpoints_timing_check_min_valid.append([True, False])
    pydb.endpoints_timing_check_min_reason.append([0, 3])
    pydb.endpoints_timing_check_max_valid.append([False, False])
    pydb.endpoints_timing_check_max_reason.append([4, 4])
    with pytest.raises(RuntimeError, match="check group.*capture context"):
        read_hold_timing_schema(pydb)
    pydb.hold_contexts.append(dict(
        pydb.hold_contexts[0], capture_pin=1, capture_transition=-1, check_type=4,
    ))
    actual = read_hold_timing_schema(pydb)
    expected = [dict(row) for row in pydb.hold_contexts]
    for row in expected:
        row.update(capture_pin_transition=1, capture_edge_index=-1, selected_check_arc_idx=-1, normal_required_ps=0.0)
    assert [asdict(context) for context in actual.contexts] == expected


@pytest.mark.parametrize("endpoint_reason,check_reason", [(5, 3), (8, 8)])
def test_known_inactive_min_domain_has_empty_contexts(pydb, endpoint_reason, check_reason):
    pydb.endpoints_min_valid = [[False, False]]
    pydb.endpoints_min_reason = [[endpoint_reason, endpoint_reason]]
    pydb.endpoints_timing_check_min_valid = [[False, False]]
    pydb.endpoints_timing_check_min_reason = [[check_reason, check_reason]]
    pydb.hold_contexts = []
    assert read_hold_timing_schema(pydb).contexts == ()
