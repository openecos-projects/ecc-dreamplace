"""Native queue snapshot construction and driver identity."""

from dreamplace.ops.rebuffer import rebuffer_cpp
from test_repair_design import _design, _liberty, _timing


def test_native_queue_snapshot_ranks_path_nets_and_roots_in_cpp():
    timing = _timing()
    timing["pin_slack"] = [0.0, -0.2]
    result = rebuffer_cpp.build_native_timing_queue_snapshot(
        _design(),
        timing,
        _liberty(),
        [
            {
                "endpoint_pin": 1,
                "endpoint_pin_name": "load:D",
                "slack_ps": -0.2,
                "path_pins": [0, 1],
                "path_timing": {},
            }
        ],
        [0],
    )

    assert result["status"] == "ok"
    assert result["items"] == [
        {
            "endpoint_pin": 1,
            "endpoint_pin_name": "load:D",
            "slack_ps": -0.2,
            "path_pins": [0, 1],
            "net_ids": [0],
            "root_net_id": 0,
            "driver_pin_id": 0,
            "tree_key": "0:0",
        }
    ]
    assert result["snapshot_statistics"] == {
        "item_count": 1,
        "endpoint_count": 1,
        "root_count": 1,
        "driver_count": 1,
        "buffer_output_root_count": 0,
        "path_driver_item_count": 1,
    }


def test_native_queue_snapshot_emits_each_ranked_path_driver_root():
    design = _design()
    design.update(
        {
            "num_nets": 3,
            "num_pins": 3,
            "num_nodes": 3,
            "flat_net2pin_map": [0, 1, 2],
            "flat_net2pin_start_map": [0, 1, 2, 3],
            "net2driver_pin_map": [0, 1, 2],
            "pin2node_map": [0, 1, 2],
            "pin2net_map": [0, 1, 2],
            "pin_offset_x": [0.0, 0.0, 0.0],
            "pin_offset_y": [0.0, 0.0, 0.0],
            "node_x": [0.0, 100.0, 200.0],
            "node_y": [0.0, 0.0, 0.0],
            "inst_main_id": [0, 0, 0],
            "inst_libcell_offset": [0, 0, 0],
            "pin_2_libpin_offset": [0, 0, 0],
        }
    )
    timing = _timing()
    timing.update(
        {
            "pin_slew_rise": [0.2, 0.2, 0.2],
            "pin_slew_fall": [0.2, 0.2, 0.2],
            "pin_load": [0.0, 0.05, 0.05],
            "pin_cap": [0.0, 0.001, 0.001],
            "pin_max_slew": [0.1, 0.1, 0.1],
            "pin_max_cap": [0.01, 0.01, 0.01],
        }
    )

    result = rebuffer_cpp.build_native_timing_queue_snapshot(
        design,
        timing,
        _liberty(),
        [
            {
                "endpoint_pin": 2,
                "slack_ps": -0.2,
                "path_pins": [0, 1, 2],
                "path_timing": {},
            }
        ],
        [10, 11, 12],
    )

    assert result["status"] == "ok"
    assert [
        (item["root_net_id"], item["driver_pin_id"], item["net_ids"]) for item in result["items"]
    ] == [(2, 2, [12]), (1, 1, [11])]
