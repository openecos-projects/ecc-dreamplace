from types import SimpleNamespace

import pytest

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


def _with_package_port(pad_pin="io_pad:PAD", driver=-1):
    pydb = _pydb()
    pydb.pin_names += ["package_out", pad_pin]
    pydb.pin2node_map += [3, 2]
    pydb.node_x += [1000.0, 5000.0]
    pydb.node_y += [0.0, 0.0]
    pydb.pin_offset_x += [0.0, 0.0]
    pydb.pin_offset_y += [0.0, 0.0]
    pydb.num_terminals, pydb.num_terminal_NIs = 1, 1
    pydb.net_names.insert(0, "package_link")
    pydb.net2driver_pin_map.insert(0, driver)
    pydb.flat_net2pin_map = [2, 3, 0, 1]
    pydb.flat_net2pin_start_map = [0, 2, 4]
    return pydb


@pytest.mark.parametrize("driver", [-1, 2, 3])
def test_package_pad_link_keeps_its_net_slot_without_a_tree(driver):
    payload = build_openroad_steiner_payload(_with_package_port(driver=driver))

    assert payload["status"] == "ok"
    assert payload["net_ids"] == [0, 1]
    assert payload["net_flat_topo_sort_start"] == [0, 0, 2]
    assert payload["net_flat_topo_sort"] == [0, 1]
    assert payload["pin_fa"] == [-1, 0, -1, -1]
    assert payload["flat_pin_to"] == [1]
    assert payload["edge_resistance"][2:] == [0.0, 0.0]


def test_fixed_cell_core_pin_still_receives_placement_rc():
    payload = build_openroad_steiner_payload(_with_package_port("io_pad:C", driver=2))

    assert payload["status"] == "ok"
    assert payload["net_flat_topo_sort"] == [2, 3, 0, 1]
    assert payload["edge_resistance"][3] > 0.0


@pytest.mark.parametrize(("pin_name", "driver"), [("io_pad:C", -1), ("io_pad:PAD", 0)])
def test_non_package_undriven_or_foreign_driver_is_an_error(pin_name, driver):
    payload = build_openroad_steiner_payload(_with_package_port(pin_name, driver))

    assert payload["status"] == "unsupported"
    assert payload["unsupported_reasons"] == ["driver_not_in_net"]
