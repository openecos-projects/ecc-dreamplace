from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.ops.placeio_ecc.place_io import ECCPlaceIOBackend, PlaceIOFunction
from dreamplace.ops.placeio_ecc.timing_schema import validate_timing_schema
from tests.ops.placeio_ecc.test_timing_carrier import _physical_after, _schema3_base


def _attach_mutations(pydb, commit_calls, sizing_calls):
    def apply_buffer_actions(actions, digest):
        commit_calls.append((list(actions), digest))
        results = [
            {
                "accepted": True,
                "inserted_buffer_name": f"b{len(commit_calls)}_{idx}",
            }
            for idx, _ in enumerate(actions)
        ]
        return {
            "status": "accepted",
            "accepted_count": len(actions),
            "rejected_count": 0,
            "failed_count": 0,
            "committed_buffer_instance_names": [
                item["inserted_buffer_name"] for item in results
            ],
            "results": results,
        }

    def apply_sizing(node_ids, target_names):
        sizing_calls.append((list(node_ids), list(target_names)))
        return {"ok": True, "accepted_count": len(node_ids)}

    pydb.apply_buffer_actions = apply_buffer_actions
    pydb.apply_sizing = apply_sizing
    return pydb


def _chained_second_buffer():
    return SimpleNamespace(
        num_nodes=5,
        num_terminals=0,
        num_terminal_NIs=2,
        dbu=1000,
        node_names=["b2", "b", "u", "OUT", "IN"],
        node_master_names=["BUF1", "BUF2", "BUF1", "", ""],
        node_x=[75000.0, 50000.0, 10000.0, 100000.0, 0.0],
        node_y=[0.0] * 5,
        node_size_x=[1000.0, 2000.0, 1000.0, 0.0, 0.0],
        node_size_y=[1000.0, 1000.0, 1000.0, 0.0, 0.0],
        pin_names=["OUT", "b2:Y", "b2:A", "b:Y", "u:A", "b:A", "u:Y", "IN"],
        pin2node_map=[3, 0, 0, 1, 2, 1, 2, 4],
        pin_offset_x=[0.0] * 8,
        pin_offset_y=[0.0] * 8,
        pin_direct=["R0"] * 8,
        net_names=["n1_leaf", "n1_child", "n0", "n1"],
        clock_net_names=[],
        flat_net2pin_map=[1, 0, 3, 2, 7, 4, 6, 5],
        flat_net2pin_start_map=[0, 2, 4, 6, 8],
    )


def _sync_placedb_view(placedb, pydb):
    movable_count = int(pydb.num_nodes) - int(pydb.num_terminals) - int(
        pydb.num_terminal_NIs
    )
    placedb.pydb = pydb
    placedb.num_nodes = int(pydb.num_nodes)
    placedb.num_physical_nodes = int(pydb.num_nodes)
    placedb.num_movable_nodes = movable_count
    placedb.num_filler_nodes = 0
    placedb.node_names = list(pydb.node_names)
    placedb.node_x = np.asarray(pydb.node_x, dtype=np.float64).copy()
    placedb.node_y = np.asarray(pydb.node_y, dtype=np.float64).copy()
    placedb.node_size_x = np.asarray(pydb.node_size_x, dtype=np.float64).copy()
    placedb.node_size_y = np.asarray(pydb.node_size_y, dtype=np.float64).copy()
    placedb.node2orig_node_map = np.arange(placedb.num_nodes, dtype=np.int32)
    placedb.node2fence_region_map = np.full(
        placedb.num_nodes, np.iinfo(np.int32).max, dtype=np.int32
    )
    pins_per_node = [[] for _ in range(placedb.num_nodes)]
    for pin_id, node_id in enumerate(pydb.pin2node_map):
        pins_per_node[int(node_id)].append(pin_id)
    flat_pins = []
    starts = [0]
    for group in pins_per_node:
        flat_pins.extend(group)
        starts.append(len(flat_pins))
    placedb.flat_node2pin_map = np.asarray(flat_pins, dtype=np.int32)
    placedb.flat_node2pin_start_map = np.asarray(starts, dtype=np.int32)
    placedb.pin2node_map = np.asarray(pydb.pin2node_map, dtype=np.int32)
    placedb.flat_libcell_names = list(pydb.flat_libcell_names)


@pytest.fixture
def ecc_schema3_carrier_harness():
    commit_calls = []
    sizing_calls = []
    refresh_records = []
    sta_calls = []

    base = _attach_mutations(_schema3_base(), commit_calls, sizing_calls)
    initial_view = validate_timing_schema(base)
    first_buf = _physical_after()
    first_buf.node_size_x = [2000.0, 1000.0, 0.0, 0.0]
    first_buf.node_size_y = [1000.0, 1000.0, 0.0, 0.0]
    snapshots = [
        _attach_mutations(first_buf, commit_calls, sizing_calls),
        _attach_mutations(_chained_second_buffer(), commit_calls, sizing_calls),
    ]

    def export_physical(*_args, **kwargs):
        if kwargs.get("with_sta"):
            sta_calls.append(dict(kwargs))
        idx = min(max(len(commit_calls) - 1, 0), len(snapshots) - 1)
        return snapshots[idx]

    rawdb = ECCPlaceIOBackend(
        SimpleNamespace(
            ecc=SimpleNamespace(
                pydb=export_physical,
                run_sta=lambda *a, **k: sta_calls.append(("run_sta", a, k)),
            )
        ),
        ".",
        object(),
        base,
        timing_enabled=True,
        sparse_timing_view=initial_view,
    )

    placedb = SimpleNamespace(
        rawdb=rawdb,
        dbu=1000,
        dtype=np.float64,
        area=1.0e10,
        total_fixed_node_area=0.0,
        total_space_area=1.0e10,
        regions=[],
        xl=0.0,
        yl=0.0,
        xh=200000.0,
        yh=200000.0,
        topology_generation=0,
        pending_continuation_seed=None,
        pending_warm_schedule_state=None,
        cooptimization_geometry=None,
        backend_caps=PlaceIOFunction.backend_caps(rawdb),
    )
    _sync_placedb_view(placedb, base)

    def write_placement_back(node_x, node_y, **_kwargs):
        count = len(node_x)
        placedb.node_x[:count] = np.asarray(node_x, dtype=np.float64)
        placedb.node_y[:count] = np.asarray(node_y, dtype=np.float64)

    def refresh_from_ecc_backend(*, refresh_mode="full_rebuild", rebuild_mode="full_rebuild"):
        pre_nodes = int(placedb.num_nodes)
        new_pydb, summary = PlaceIOFunction.refresh(
            rawdb, refresh_mode=refresh_mode, rebuild_mode=rebuild_mode
        )
        _sync_placedb_view(placedb, new_pydb)
        placedb.topology_generation += 1
        placedb.backend_caps = PlaceIOFunction.backend_caps(rawdb)
        out = {
            **dict(summary),
            "rebuild": {
                "pre_counts": {"nodes": pre_nodes},
                "post_counts": {"nodes": int(placedb.num_nodes)},
            },
        }
        refresh_records.append(out)
        return out

    def make_drive_type_table(tmp_dir):
        from dreamplace.ops.buffer_insertion.drive_type_table import (
            write_case_drive_type_table,
        )

        library = {
            idx: {
                "cell_id": idx % 2,
                "master_name": name,
                "input_cap": 0.005,
                "delay": 10.0,
                "output_slew": 10.0,
            }
            for idx, name in enumerate(
                ("BUFX8H7L", "BUFX12H7L", "BUFX16H7L", "BUFX20H7L")
            )
        }
        return write_case_drive_type_table(
            tmp_dir / "drive_strength.json",
            buffer_library=library,
            drive_strengths=[
                {"master": name, "strength": strength}
                for name, strength in zip(
                    ("BUFX8H7L", "BUFX12H7L", "BUFX16H7L", "BUFX20H7L"),
                    (8.0, 12.0, 16.0, 20.0),
                    strict=True,
                )
            ],
        )

    placedb.write_placement_back = write_placement_back
    placedb.refresh_from_ecc_backend = refresh_from_ecc_backend
    return SimpleNamespace(
        placedb=placedb,
        rawdb=rawdb,
        commit_calls=commit_calls,
        sizing_calls=sizing_calls,
        refresh_records=refresh_records,
        sta_calls=sta_calls,
        make_drive_type_table=make_drive_type_table,
    )
