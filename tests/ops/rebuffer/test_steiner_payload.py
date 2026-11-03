from types import SimpleNamespace

from dreamplace.ops.rebuffer.steiner_payload import (
    build_openroad_steiner_payload,
)


def _pydb():
    return SimpleNamespace(
        pin_names=["drv:Y", "load:A"],
        net_names=["n"],
        pin2node_map=[0, 1],
        node_x=[0.0, 100.0],
        node_y=[0.0, 0.0],
        pin_offset_x=[0.0, 0.0],
        pin_offset_y=[0.0, 0.0],
        flat_net2pin_map=[0, 1],
        flat_net2pin_start_map=[0, 2],
        net2driver_pin_map=[0],
        dbu=1000,
    )


def test_openroad_steiner_payload_roots_two_pin_net_at_driver():
    payload = build_openroad_steiner_payload(_pydb(), r_unit=0.914, dbu=1000)

    assert payload == {
        "artifact": "rebuffer_openroad_steiner_payload",
        "artifact_version": 1,
        "status": "ok",
        "provider": "openroad_stt",
        "steiner_node_base": 2,
        "net_ids": [0],
        "net_flat_topo_sort_start": [0, 2],
        "net_flat_topo_sort": [0, 1],
        "pin_fa": [-1, 0],
        "flat_pin_to_start": [0, 1, 1],
        "flat_pin_to": [1],
        "flat_pin_from": [0],
        "node_x": [0.0, 100.0],
        "node_y": [0.0, 0.0],
        "edge_resistance": [0.0, 0.09140000000000001],
        "node_capacitance": [0.0, 0.0],
    }


def test_coincident_terminal_net_keeps_zero_length_connections():
    pydb = _pydb()
    pydb.node_x = [100.0, 100.0]
    payload = build_openroad_steiner_payload(pydb)
    assert payload["status"] == "ok", payload
    assert {key: payload[key] for key in (
        "net_flat_topo_sort", "pin_fa", "flat_pin_from", "flat_pin_to", "edge_resistance"
    )} == {
        "net_flat_topo_sort": [0, 1], "pin_fa": [-1, 0],
        "flat_pin_from": [0], "flat_pin_to": [1], "edge_resistance": [0.0, 0.0]
    }
