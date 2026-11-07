from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.timing_carrier import (
    CommittedTimingPyDB,
    require_carrier_schema,
)
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel


def _base():
    """One real combinational cell between constrained input/output ports."""
    fields = dict(
        timing_schema_version=2,
        dbu=1000,
        num_nodes=3,
        num_terminals=0,
        num_terminal_NIs=2,
        node_names=["u", "IN", "OUT"],
        node_master_names=["BUF1", "", ""],
        node_x=[10000.0, 0.0, 100000.0],
        node_y=[0.0, 0.0, 0.0],
        node_size_x=[1000.0, 0.0, 0.0],
        node_size_y=[1000.0, 0.0, 0.0],
        pin_names=["IN", "u:A", "u:Y", "OUT"],
        pin2node_map=[1, 0, 0, 2],
        pin_offset_x=[0.0] * 4,
        pin_offset_y=[0.0] * 4,
        pin_direct=["R0"] * 4,
        net_names=["n0", "n1"],
        clock_net_names=[],
        pin2net_map=[0, 0, 1, 1],
        flat_net2pin_map=[0, 1, 2, 3],
        flat_net2pin_start_map=[0, 2, 4],
        net2driver_pin_map=[0, 2],
        net_flat_arcs_start=[0, 1, 2],
        net_flat_arcs=[[0, 1], [2, 3]],
        start_points=[0],
        end_points=[3],
        endpoint_pin_ids=[3],
        start_pin_ids=[0],
        clock_pins=[],
        FF_ids=[],
        clk_pin_rtran=[],
        clk_pin_ftran=[],
        inrdelays=[2.0],
        infdelays=[3.0],
        inrtrans=[15.0],
        inftrans=[20.0],
        outcaps=[0.0],
        endpoints_rRAT=[0.0],
        endpoints_fRAT=[0.0],
        endpoints_max_valid=np.ones((1, 2), dtype=bool),
        endpoints_max_reason=np.zeros((1, 2), dtype=np.int32),
        endpoints_constraint_arcs=np.empty((0, 8), dtype=np.int64),
        endpoints_timing_check_arcs=np.empty((0, 8), dtype=np.int64),
        endpoints_constraint_max_valid=np.empty((0, 2), dtype=bool),
        endpoints_constraint_max_reason=np.empty((0, 2), dtype=np.int32),
        endpoints_timing_check_max_valid=np.empty((0, 2), dtype=bool),
        endpoints_timing_check_max_reason=np.empty((0, 2), dtype=np.int32),
        flat_inst_arcs_by_level=[[1, 2, 0, 0, 1, 0, 0, 0]],
        flat_inst_arcs_by_level_start=[0, 0, 1],
        inst_flat_arcs=[0],
        inst_flat_arcs_start=[0, 1, 1, 1],
        flat_pin_to_graph=[1, 2, 3],
        flat_pin_to_graph_start=[0, 1, 2, 3, 3],
        flat_pin_to_graph_reverse=[0, 1, 2],
        flat_pin_to_graph_start_reverse=[0, 0, 1, 2, 3],
        pin_pred_start=[0, 0, 1, 2, 3],
        pin_pred_pin=[0, 1, 2],
        pin_pred_arc_id=[-1, 0, -1],
        pin_succ_start=[0, 1, 2, 3, 3],
        pin_succ_pin=[1, 2, 3],
        pin_succ_arc_id=[-1, 0, -1],
        arc_level_start=[0, 0, 1],
        arc_src_pin=[1, 0, 2],
        arc_dst_pin=[2, 1, 3],
        arc_inst_id=[0, -1, -1],
        arc_libcell_id=[0, -1, -1],
        arc_libarc_id=[0, -1, -1],
        arc_sense=[1, 0, 0],
        arc_type=[0, 0, 0],
        arc_offset=[0, -1, -1],
        pin_to_inst_id=[1, 0, 0, 2],
        pin_to_node_id=[1, 0, 0, 2],
        inst_topo_start=[0, 0, 1],
        inst_topo_ids=[0],
        flat_libcell_names=["BUF1", "BUF2"],
        flat_libcell_info=[["BUF1", 0, 1, 0], ["BUF2", 0, 2, 0]],
        flat_libcell_width=[1000.0, 2000.0],
        flat_libcell_height=[1000.0, 1000.0],
        flat_libcell_leakage=[0.0, 0.0],
        main_id_2_cell_id_start=[0, 2],
        inst_main_id=[0, -1, -1],
        inst_libcell_offset=[0, 0, 0],
        cell_id_2_arc_id_start=[0, 1, 2],
        cell_id_2_libpin_id_start=[0, 2, 4],
        pin_2_libpin_offset=[-1, 0, 1, -1],
        flat_lib_pin_names=["A", "Y", "A", "Y"],
        flat_lib_pin_cap=[0.005, 0.0, 0.01, 0.0],
        flat_lib_pin_rcap=[0.005, 0.0, 0.01, 0.0],
        flat_lib_pin_fcap=[0.006, 0.0, 0.012, 0.0],
        flat_lib_pin_cap_limit=[0.0, 1.0, 0.0, 1.0],
        flat_lib_pin_slew_limit=[100.0] * 4,
        flat_libarc_info=[["A", "Y", 0, 0, 1, 0], ["A", "Y", 1, 0, 1, 0]],
        buffer_main_type_candidate_indices=[0],
    )
    for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
        values = [[10.0, 110.0, 20.0, 120.0], [5.0, 55.0, 10.0, 60.0]]
        if prefix.endswith("trans"):
            values = [[10.0, 30.0, 30.0, 50.0], [5.0, 15.0, 15.0, 25.0]]
        fields.update(
            {
                f"{prefix}_flat_luts_values": values,
                f"{prefix}_flat_luts_trans_table": [[0.0, 100.0]] * 2,
                f"{prefix}_flat_luts_cap_table": [[0.0, 1.0]] * 2,
                f"{prefix}_flat_luts_dim": [[2, 2]] * 2,
            }
        )
    return SimpleNamespace(**fields)


def _physical_after():
    # Physical export reorders nodes, pins, and nets. Tests must prove that
    # timing references follow names rather than retaining numeric identities.
    return SimpleNamespace(
        num_nodes=4,
        num_terminals=0,
        num_terminal_NIs=2,
        dbu=1000,
        node_names=["b", "u", "OUT", "IN"],
        node_master_names=["BUF2", "BUF1", "", ""],
        node_x=[50000.0, 10000.0, 100000.0, 0.0],
        node_y=[0.0] * 4,
        pin_names=["OUT", "b:Y", "u:A", "b:A", "u:Y", "IN"],
        pin2node_map=[2, 0, 1, 0, 1, 3],
        pin_offset_x=[0.0] * 6,
        pin_offset_y=[0.0] * 6,
        pin_direct=["R0"] * 6,
        net_names=["n1_child", "n0", "n1"],
        clock_net_names=[],
        flat_net2pin_map=[1, 0, 5, 2, 4, 3],
        flat_net2pin_start_map=[0, 2, 4, 6],
    )


def test_native_mutation_state_follows_the_live_physical_owner():
    base, physical = _base(), _physical_after()
    base.native_mutation_epoch, base.native_state_dirty = 7, False
    physical.native_mutation_epoch, physical.native_state_dirty = 1, True
    carrier = CommittedTimingPyDB(base, physical)
    assert (carrier.native_mutation_epoch, carrier.native_state_dirty) == (1, True)
    physical.native_mutation_epoch, physical.native_state_dirty = 2, False
    assert (carrier.native_mutation_epoch, carrier.native_state_dirty) == (2, False)


@pytest.mark.parametrize("capture_deadline", [False, True])
def test_recovery_reference_convention_survives_carrier_rebuild(capture_deadline):
    base, physical = _base(), _physical_after()
    first = CommittedTimingPyDB(base, physical, recovery_rat_is_capture_deadline=capture_deadline)
    rebuilt = CommittedTimingPyDB(first, physical)
    assert (first.recovery_rat_is_capture_deadline, rebuilt.recovery_rat_is_capture_deadline) == (
        capture_deadline, capture_deadline,
    )


def test_output_sdc_load_is_included_in_rise_and_fall_rc_and_python_timing():
    base = _base()
    unloaded = RebufferTimingModel(base, use_pin_offset=False)
    base.outcaps = [0.03]
    base.out_rcaps = [0.02]
    base.out_fcaps = [0.04]
    loaded = RebufferTimingModel(base, use_pin_offset=False)

    assert loaded.lib_pin_caps[3] == pytest.approx(0.03)
    assert loaded.lib_pin_rcaps[3] == pytest.approx(0.02)
    assert loaded.lib_pin_fcaps[3] == pytest.approx(0.04)
    assert float(loaded.loads["rise"][2] - unloaded.loads["rise"][2]) == pytest.approx(0.02)
    assert float(loaded.loads["fall"][2] - unloaded.loads["fall"][2]) == pytest.approx(0.04)
    assert loaded.baseline.wns_ps < unloaded.baseline.wns_ps


@pytest.mark.parametrize("trigger", ["repair_buffering_enable", "repair_buffering_midplace_enable"])
def test_repair_read_and_refresh_use_python_carrier_without_ecc_sta(trigger):
    from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction

    calls = []
    base = _base()

    class MetadataModule:
        def get_dmInst_ptr(self):
            return "physical_db"

        def prepare_placement_metadata(self, inputs):
            calls.append(("metadata", inputs))
            return {"parasitics_initialization": "python_placement_rc_pending"}

        def pydb(self, *args, **kwargs):
            calls.append(("pydb", kwargs))
            return base

        def write_placement_back(self, *args):
            pass

        def def_save(self, *args):
            pass

    raw = PlaceIOFunction.read(
        SimpleNamespace(with_sta=True, **{trigger: True}), MetadataModule()
    )
    assert isinstance(raw.pydb, CommittedTimingPyDB)
    assert raw.backend_caps.sta_reference_role == "dreamplace_timing_propagation"
    assert calls[0] == ("metadata", {})
    assert calls[1][1]["timing_metadata_only"] is True
    before = RebufferTimingModel(raw.pydb, use_pin_offset=False)
    refreshed, summary = PlaceIOFunction.refresh(raw)
    assert calls[-1][1]["with_sta"] is False
    assert calls[-1][1]["include_clock_nets"] is True
    assert summary["ecc_sta_run"] is False
    assert RebufferTimingModel(refreshed, use_pin_offset=False).baseline == before.baseline


def _reference_after(base):
    """Independent, manually indexed two-buffer timing graph."""
    ref = deepcopy(base)
    ref.num_nodes = 4
    ref.node_names = ["u", "b", "IN", "OUT"]
    ref.node_x, ref.node_y = [10000.0, 50000.0, 0.0, 100000.0], [0.0] * 4
    ref.pin_names = ["IN", "u:A", "u:Y", "b:A", "b:Y", "OUT"]
    ref.pin2node_map = ref.pin_to_node_id = ref.pin_to_inst_id = [2, 0, 0, 1, 1, 3]
    ref.pin_offset_x = ref.pin_offset_y = [0.0] * 6
    ref.inst_main_id, ref.inst_libcell_offset = [0, 0, -1, -1], [0, 1, 0, 0]
    ref.pin_2_libpin_offset = [-1, 0, 1, 0, 1, -1]
    ref.pin2net_map = [0, 0, 1, 1, 2, 2]
    ref.net_names = ["n0", "n1", "n1_child"]
    ref.flat_net2pin_map = [0, 1, 2, 3, 4, 5]
    ref.flat_net2pin_start_map = ref.net_flat_arcs_start = [0, 2, 4, 6]
    ref.net_flat_arcs_start = [0, 1, 2, 3]
    ref.net_flat_arcs = [[0, 1], [2, 3], [4, 5]]
    ref.net2driver_pin_map = [0, 2, 4]
    ref.end_points = ref.endpoint_pin_ids = [5]
    ref.flat_inst_arcs_by_level = [[1, 2, 0, 0, 1, 0, 0, 0], [3, 4, 1, 1, 1, 0, 0, 1]]
    ref.flat_inst_arcs_by_level_start = ref.arc_level_start = [0, 0, 1, 2]
    ref.inst_flat_arcs, ref.inst_flat_arcs_start = [0, 1], [0, 1, 2, 2, 2]
    ref.inst_topo_start, ref.inst_topo_ids = [0, 0, 1, 2], [0, 1]
    ref.flat_pin_to_graph = ref.pin_succ_pin = [1, 2, 3, 4, 5]
    ref.flat_pin_to_graph_start = ref.pin_succ_start = [0, 1, 2, 3, 4, 5, 5]
    ref.flat_pin_to_graph_reverse = ref.pin_pred_pin = [0, 1, 2, 3, 4]
    ref.flat_pin_to_graph_start_reverse = ref.pin_pred_start = [0, 0, 1, 2, 3, 4, 5]
    ref.pin_pred_arc_id = ref.pin_succ_arc_id = [-1, 0, -1, 1, -1]
    ref.arc_src_pin, ref.arc_dst_pin = [1, 3, 0, 2, 4], [2, 4, 1, 3, 5]
    ref.arc_inst_id, ref.arc_libcell_id, ref.arc_libarc_id = (
        [0, 1, -1, -1, -1],
        [0, 1, -1, -1, -1],
        [0, 1, -1, -1, -1],
    )
    return ref


def _timing(pydb):
    return RebufferTimingModel(pydb, use_pin_offset=False)


def test_native_carrier_remaps_graph_and_uses_actual_inserted_master():
    carrier = CommittedTimingPyDB(_base(), _physical_after())

    assert {
        "starts": carrier.start_points,
        "endpoints": carrier.end_points,
        "cells": carrier.flat_inst_arcs_by_level,
        "schedule": carrier.flat_inst_arcs_by_level_start,
        "net_drivers": carrier.net2driver_pin_map,
        "lib_pin_offsets": carrier.pin_2_libpin_offset,
        "masters": carrier.inst_libcell_offset,
    } == {
        "starts": [5],
        "endpoints": [0],
        "cells": [[2, 4, 0, 0, 1, 0, 0, 1], [3, 1, 1, 1, 1, 0, 0, 0]],
        "schedule": [0, 0, 1, 2],
        "net_drivers": [1, 5, 4],
        "lib_pin_offsets": [-1, 1, 0, 0, 1, -1],
        "masters": [1, 0, 0, 0],
    }
    assert carrier.backend_endpoint_rAAT == []
    assert carrier.timing_backend_snapshot_valid is False


def test_rebuilt_carrier_runs_real_timing_propagation_like_independent_graph():
    base = _base()
    rebuilt = _timing(CommittedTimingPyDB(base, _physical_after()))
    reference = _timing(_reference_after(base))
    before = _timing(base)

    assert rebuilt.baseline.as_dict() == pytest.approx(
        reference.baseline.as_dict(), abs=1e-8, rel=0
    )
    assert rebuilt.baseline.wns_ps != pytest.approx(before.baseline.wns_ps)
    by_name = dict(zip(rebuilt.pydb.pin_names, rebuilt.model.get_pin_slack().numpy(), strict=True))
    assert by_name == pytest.approx(
        dict(
            zip(
                reference.pydb.pin_names,
                reference.model.get_pin_slack().numpy(),
                strict=True,
            )
        ),
        abs=1e-8,
        rel=0,
    )


def test_carrier_rejects_missing_endpoint_before_publishing_a_graph():
    physical = _physical_after()
    physical.pin_names[0] = "unrelated_output"
    with pytest.raises(ValueError, match="end_points"):
        rebuffer_cpp.rebuild_timing_carrier(_base(), physical)


def test_unchanged_export_and_insert_remove_round_trip_preserve_full_timing():
    base = _base()
    before = _timing(base)
    unchanged = CommittedTimingPyDB(base, deepcopy(base))
    inserted = CommittedTimingPyDB(unchanged, _physical_after())
    removed = CommittedTimingPyDB(inserted, deepcopy(base))

    assert _timing(unchanged).baseline.as_dict() == pytest.approx(
        before.baseline.as_dict(), abs=1e-8, rel=0
    )
    restored = _timing(removed)
    np.testing.assert_allclose(
        restored.model.get_pin_slack().numpy(),
        before.model.get_pin_slack().numpy(),
        atol=1e-8,
        rtol=0,
    )
    assert removed._metadata is base
    assert (removed.num_nodes, removed.num_pins, removed.num_nets) == (3, 4, 2)


def test_resize_reads_current_master_and_capacitance():
    base = _base()
    resized = deepcopy(base)
    resized.node_master_names[0] = "BUF2"
    carrier = CommittedTimingPyDB(base, resized)
    reference = deepcopy(base)
    reference.inst_libcell_offset[0] = 1
    reference.flat_inst_arcs_by_level[0][2:4] = [1, 1]

    assert carrier.flat_inst_arcs_by_level == [[1, 2, 1, 1, 1, 0, 0, 0]]
    assert _timing(carrier).baseline.as_dict() == pytest.approx(
        _timing(reference).baseline.as_dict(), abs=1e-8, rel=0
    )
    assert _timing(carrier).baseline.wns_ps != pytest.approx(_timing(base).baseline.wns_ps)


def test_clock_local_indices_and_qualification_survive_pin_reordering():
    base = _base()
    base.clock_pins, base.FF_ids = [0], [2]
    base.flat_inst_arcs_by_level = [[0, 2, 0, 0, 1, 0, 0, 0]]
    base.flat_inst_arcs_by_level_start = [0, 1]
    base.endpoints_constraint_arcs = [[0, 3, -1, -1, 0, 0, 0, -1]]
    base.endpoints_constraint_max_valid = np.ones((1, 2), dtype=bool)
    base.endpoints_constraint_max_reason = np.zeros((1, 2), dtype=np.int32)

    carrier = CommittedTimingPyDB(base, _physical_after())

    assert {
        "clocks": carrier.clock_pins,
        "FF": carrier.FF_ids,
        "launch_arc": carrier.flat_inst_arcs_by_level[0],
        "checks": carrier.endpoints_constraint_arcs,
    } == {
        "clocks": [5],
        "FF": [4],
        "launch_arc": [0, 4, 0, 0, 1, 0, 0, 1],
        "checks": [[0, 0, -1, -1, 0, 0, 0, -1]],
    }
    np.testing.assert_array_equal(
        carrier.endpoints_constraint_max_valid, base.endpoints_constraint_max_valid
    )


def test_refresh_reads_physical_connectivity_without_requesting_ecc_sta():
    from dreamplace.ops.placeio_ecc.place_io import ECCPlaceIOBackend, PlaceIOFunction
    from dreamplace.ops.placeio_ecc.timing_carrier import refresh_python_timing_carrier

    calls = []
    physical = _physical_after()

    def export(dm, x, y, routability, **options):
        calls.append(options)
        assert options == {
            "with_sta": False,
            "include_clock_nets": True,
            "include_m2_pg_rail_blockage": False,
            "include_m2_pg_rail_density": True,
        }
        return physical

    def forbidden(*args, **kwargs):
        pytest.fail("repair refresh called ECC STA")

    native = SimpleNamespace(pydb=export, init_sta=forbidden, run_sta=forbidden)
    wrapper = SimpleNamespace(ecc=native, pydb=forbidden, refresh_place_timing=forbidden)
    raw = ECCPlaceIOBackend(wrapper, ".", object(), _base(), timing_enabled=True)
    carrier, summary = refresh_python_timing_carrier(raw)

    assert raw.pydb is carrier
    assert summary["ecc_sta_run"] is False
    assert raw.refresh_generation == 1
    assert len(calls) == 1
    assert _timing(carrier).baseline.as_dict() == pytest.approx(
        _timing(_reference_after(_base())).baseline.as_dict(), abs=1e-8, rel=0
    )
    # Placement reentry uses the common refresh API. It must retain the
    # active Python carrier rather than fall back to the old ECC STA path.
    reentered, summary = PlaceIOFunction.refresh(raw)
    assert reentered is raw.pydb and reentered is not carrier
    assert reentered._metadata is carrier._metadata
    assert raw.refresh_generation == len(calls) == 2
    assert summary["ecc_sta_run"] is False
    assert raw.backend_caps.sta_reference_role == "dreamplace_timing_propagation"


def test_verifier_scores_the_rebuilt_physical_graph_with_real_timing(monkeypatch):
    from dreamplace.ops.placeio_ecc.commit_verify import commit_buffer_actions_verified
    from dreamplace.ops.placeio_ecc.place_io import ECCPlaceIOBackend, PlaceIOFunction

    base, after = _base(), _physical_after()
    state = {"physical": base}

    def export(*args, **kwargs):
        assert kwargs["with_sta"] is False
        return state["physical"]

    def commit(raw_db, actions, digest):
        state["physical"] = after
        return {
            "status": "accepted",
            "accepted_count": 1,
            "rejected_count": 0,
            "failed_count": 0,
            "committed_buffer_instance_names": ["b"],
            "action_digest": digest,
        }

    raw = ECCPlaceIOBackend(SimpleNamespace(pydb=export), ".", object(), base, timing_enabled=True)
    monkeypatch.setattr(PlaceIOFunction, "apply_buffer_actions", staticmethod(commit))
    result = commit_buffer_actions_verified(
        raw,
        [{"action_kind": "buffer_insert", "net_name": "n1"}],
        wns_margin_ps=1e6,
        tns_margin_ps=1e6,
    )

    expected = _timing(_reference_after(base)).baseline.as_dict()
    assert result["timing_after"] == pytest.approx(expected, abs=1e-8, rel=0)
    assert result["timing_after"] != result["timing_before"]
    assert result["native_counts_after"] == {"nodes": 4, "pins": 6, "nets": 3}
    assert result["kept_count"] == 1
    assert raw.pydb.pin_names == after.pin_names


def test_resize_matches_liberty_arcs_by_ports_when_local_order_changes():
    base = _base()
    # BUF2 has a negative-sense arc before the equivalent positive arc.
    base.cell_id_2_arc_id_start = [0, 1, 3]
    base.flat_libarc_info = [
        ["A", "Y", 0, 0, 1, 0],
        ["A", "Y", 1, 0, -1, 0],
        ["A", "Y", 1, 1, 1, 0],
    ]
    physical = deepcopy(base)
    physical.node_master_names[0] = "BUF2"

    carrier = CommittedTimingPyDB(base, physical)

    assert carrier.flat_inst_arcs_by_level == [[1, 2, 1, 2, 1, 0, 1, 0]]


def test_resize_maps_repeated_equivalent_arcs_by_occurrence():
    base = _base()
    # Both masters repeat A->Y positive (state-dependent `when` arcs), and
    # BUF2 lists a negative arc first. The instance times through BUF1's
    # second repeat, so the resize must pick BUF2's second repeat.
    base.cell_id_2_arc_id_start = [0, 2, 5]
    base.flat_libarc_info = [
        ["A", "Y", 0, 0, 1, 0],
        ["A", "Y", 0, 1, 1, 0],
        ["A", "Y", 1, 0, -1, 0],
        ["A", "Y", 1, 1, 1, 0],
        ["A", "Y", 1, 2, 1, 0],
    ]
    base.flat_inst_arcs_by_level = [[1, 2, 0, 1, 1, 0, 1, 0]]
    physical = deepcopy(base)
    physical.node_master_names[0] = "BUF2"

    carrier = CommittedTimingPyDB(base, physical)

    assert carrier.flat_inst_arcs_by_level == [[1, 2, 1, 4, 1, 0, 2, 0]]


def test_python_timing_and_planner_keep_only_qualified_endpoint_transitions():
    from dreamplace.ops.placeio_ecc.repair_timing import _slack_vector

    base = _base()
    unqualified = _timing(base)
    base.endpoints_max_valid = np.array([[True, False]])
    base.endpoints_max_reason = np.array([[0, 1]], dtype=np.int32)
    model = _timing(CommittedTimingPyDB(base, deepcopy(base)))
    expected = float(model.model.pin_rRAT_live_snapshot[3] - model.model.pin_rAAT_live_snapshot[3])

    assert model.baseline.wns_ps == pytest.approx(expected, abs=1e-8, rel=0)
    assert model.baseline.wns_ps > unqualified.baseline.wns_ps
    assert _slack_vector(model)[3] == pytest.approx(expected, abs=1e-8, rel=0)

    base.endpoints_max_valid = np.zeros((1, 2), dtype=bool)
    base.endpoints_max_reason = np.ones((1, 2), dtype=np.int32)
    masked = _timing(CommittedTimingPyDB(base, deepcopy(base)))
    assert masked.baseline.as_dict() == {
        "wns_ps": 0.0,
        "tns_ps": 0.0,
        "worst_slack_ps": 0.0,
        "endpoint_count": 0,
        "violating_endpoint_count": 0,
    }
    assert np.isinf(_slack_vector(masked)[3])


@pytest.mark.parametrize("version", [2, 3])
def test_carrier_accepts_schema_two_and_three(version):
    # Schema 3 only adds the sparse constraint view; the per-pin, per-arc and
    # per-endpoint rows the carrier rebuild consumes are unchanged.
    require_carrier_schema(SimpleNamespace(timing_schema_version=version))


def test_carrier_rejects_unknown_schema():
    with pytest.raises(ValueError, match="schema 2 or 3"):
        require_carrier_schema(SimpleNamespace(timing_schema_version=4))


def test_inplace_carrier_sizing_matches_full_rebuild_without_reexport():
    from dreamplace.ops.placeio_ecc.place_io import ECCPlaceIOBackend
    from dreamplace.ops.placeio_ecc.timing_carrier import apply_inplace_carrier_sizing

    base = _base()
    base.flat_lib_pin_offset_x = [10.0, 20.0, 30.0, 40.0]
    base.flat_lib_pin_offset_y = [5.0, 15.0, 25.0, 35.0]
    physical = deepcopy(base)
    physical.native_state_dirty = True
    carrier = CommittedTimingPyDB(base, physical)

    def forbidden_export(*args, **kwargs):
        pytest.fail("size-only in-place carrier sizing must not call C++ pydb export")

    raw = ECCPlaceIOBackend(
        SimpleNamespace(ecc=SimpleNamespace(pydb=forbidden_export)),
        ".",
        object(),
        carrier,
        timing_enabled=True,
    )

    updated, summary = apply_inplace_carrier_sizing(
        raw,
        [{"node_id": 0, "target_cell_index": 1, "target_master_name": "BUF2"}],
    )

    assert updated is carrier
    assert summary["refresh_path"] == "inplace_carrier_sizing"
    assert summary["pydb_replaced"] is False
    assert raw.refresh_generation == 1
    assert carrier.native_state_dirty is False
    assert carrier.node_master_names[0] == "BUF2"
    assert carrier.inst_libcell_offset[0] == 1
    assert carrier.node_size_x[0] == pytest.approx(2000.0)
    assert carrier.node_size_y[0] == pytest.approx(1000.0)
    assert list(carrier.pin_offset_x[1:3]) == [30.0, 40.0]
    assert list(carrier.pin_offset_y[1:3]) == [25.0, 35.0]
    assert carrier.flat_inst_arcs_by_level == [[1, 2, 1, 1, 1, 0, 0, 0]]
    assert list(carrier.arc_libcell_id) == [1, -1, -1]
    assert list(carrier.arc_libarc_id) == [1, -1, -1]

    resized_physical = deepcopy(base)
    resized_physical.node_master_names[0] = "BUF2"
    resized_physical.node_size_x[0] = 2000.0
    resized_physical.pin_offset_x[1:3] = [30.0, 40.0]
    resized_physical.pin_offset_y[1:3] = [25.0, 35.0]
    full_rebuilt = CommittedTimingPyDB(base, resized_physical)

    assert _timing(carrier).baseline.as_dict() == pytest.approx(
        _timing(full_rebuilt).baseline.as_dict(), abs=1e-8, rel=0
    )


def test_refresh_skips_routability_export_when_metadata_owns_static_routability():
    from dreamplace.ops.placeio_ecc.place_io import ECCPlaceIOBackend
    from dreamplace.ops.placeio_ecc.timing_carrier import refresh_python_timing_carrier

    base = _base()
    base.num_routing_grids_x = 4
    base.num_routing_grids_y = 4
    base.unit_horizontal_capacities = [1.0]
    base.unit_vertical_capacities = [1.0]
    base.initial_horizontal_demand_map = [0.0] * 16
    base.initial_vertical_demand_map = [0.0] * 16

    seen_routability_flags = []

    def export(dm, x, y, routability, **options):
        seen_routability_flags.append(bool(routability))
        return _physical_after()

    raw = ECCPlaceIOBackend(
        SimpleNamespace(ecc=SimpleNamespace(pydb=export)),
        ".",
        object(),
        base,
        timing_enabled=True,
    )
    carrier, _ = refresh_python_timing_carrier(raw)
    assert seen_routability_flags == [False]
    assert carrier.num_routing_grids_x == 4
    assert carrier.unit_horizontal_capacities == [1.0]

