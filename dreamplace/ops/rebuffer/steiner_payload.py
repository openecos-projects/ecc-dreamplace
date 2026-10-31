"""Build a native ``SteinerView`` payload from OpenROAD's STT provider.

The conversion is deliberately limited to topology marshalling: pin
coordinates, edge endpoints, rooting, and CSR construction.  Tree search is
performed by ``rebuffer_steiner_cpp`` and repair decisions remain in
``rebuffer_cpp``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .openroad_steiner import build_openroad_stt_tree

ARTIFACT = "rebuffer_openroad_steiner_payload"
ARTIFACT_VERSION = 1


def _pin_coordinates(pydb):
    pin2node = np.asarray(pydb.pin2node_map, dtype=np.int64)
    node_x = np.asarray(pydb.node_x, dtype=np.float64)
    node_y = np.asarray(pydb.node_y, dtype=np.float64)
    pin_offset_x = np.asarray(pydb.pin_offset_x, dtype=np.float64)
    pin_offset_y = np.asarray(pydb.pin_offset_y, dtype=np.float64)
    x = np.rint(node_x[pin2node] + pin_offset_x).astype(np.int64)
    y = np.rint(node_y[pin2node] + pin_offset_y).astype(np.int64)
    return x, y


def _empty_payload(pydb, reason):
    num_pins = len(pydb.pin_names)
    num_nets = len(pydb.net_names)
    return {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "status": "unsupported",
        "unsupported_reasons": [reason],
        "steiner_node_base": num_pins,
        "net_ids": list(range(num_nets)),
        "net_flat_topo_sort_start": [0] * (num_nets + 1),
        "net_flat_topo_sort": [],
        "pin_fa": [-1] * num_pins,
        "flat_pin_to_start": [0] * (num_pins + 1),
        "flat_pin_to": [],
        "node_x": [0.0] * num_pins,
        "node_y": [0.0] * num_pins,
        "edge_resistance": [0.0] * num_pins,
        "node_capacitance": [0.0] * num_pins,
    }


def build_openroad_steiner_payload(pydb, *, r_unit=0.914, dbu=None):
    """Return a ``SteinerView``-compatible payload using OpenROAD STT.

    Pin nodes keep their global pin ids.  Synthetic Steiner nodes are
    allocated above ``len(pydb.pin_names)``.  Pins that share a coordinate are
    connected to the primary pin with zero-length edges.
    """

    if pydb is None:
        return _empty_payload(pydb, "pydb_is_none")
    dbu = int(dbu if dbu is not None else getattr(pydb, "dbu", 0) or 0)
    if dbu <= 0:
        return _empty_payload(pydb, "invalid_dbu")

    num_pins = len(pydb.pin_names)
    num_nets = len(pydb.net_names)
    if num_pins <= 0 or num_nets <= 0:
        return _empty_payload(pydb, "empty_design")

    pin_x, pin_y = _pin_coordinates(pydb)
    flat_map = np.asarray(pydb.flat_net2pin_map, dtype=np.int64)
    flat_start = np.asarray(pydb.flat_net2pin_start_map, dtype=np.int64)
    drivers = np.asarray(pydb.net2driver_pin_map, dtype=np.int64)

    steiner_base = num_pins
    node_x = {index: int(pin_x[index]) for index in range(num_pins)}
    node_y = {index: int(pin_y[index]) for index in range(num_pins)}
    parent = {}
    children = defaultdict(list)
    edge_resistance = {}
    topo = []
    topo_start = [0]
    net_ids = []
    next_steiner = steiner_base

    for net_id in range(num_nets):
        begin = int(flat_start[net_id])
        end = int(flat_start[net_id + 1])
        pins = [int(value) for value in flat_map[begin:end]]
        if len(pins) < 2:
            topo_start.append(len(topo))
            net_ids.append(net_id)
            continue
        driver = int(drivers[net_id])
        if driver not in pins:
            return _empty_payload(pydb, "driver_not_in_net")
        driver_index = pins.index(driver)
        xs = [int(pin_x[pin]) for pin in pins]
        ys = [int(pin_y[pin]) for pin in pins]
        tree = build_openroad_stt_tree(xs, ys, driver_index, alpha=0.3)
        edges = tree.get("edges", ())
        if not edges:
            return _empty_payload(pydb, "openroad_stt_empty_tree")

        coord_pins = defaultdict(list)
        for pin in pins:
            coord_pins[(int(pin_x[pin]), int(pin_y[pin]))].append(pin)
        coord_node = {}
        for coord, coord_pin_ids in coord_pins.items():
            coord_node[coord] = (
                driver if driver in coord_pin_ids else coord_pin_ids[0]
            )
        for x1, y1, x2, y2 in edges:
            for coord in ((int(x1), int(y1)), (int(x2), int(y2))):
                if coord in coord_node:
                    continue
                coord_node[coord] = next_steiner
                node_x[next_steiner] = coord[0]
                node_y[next_steiner] = coord[1]
                next_steiner += 1

        adjacency = defaultdict(set)
        for x1, y1, x2, y2 in edges:
            left = coord_node[(int(x1), int(y1))]
            right = coord_node[(int(x2), int(y2))]
            if left == right:
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
        for coord, coord_pin_ids in coord_pins.items():
            primary = coord_node[coord]
            for pin in coord_pin_ids:
                if pin == primary:
                    continue
                adjacency[primary].add(pin)
                adjacency[pin].add(primary)

        if driver not in adjacency:
            return _empty_payload(pydb, "driver_not_in_openroad_tree")
        order = []
        stack = [driver]
        seen = {driver}
        parent[driver] = -1
        while stack:
            node = stack.pop()
            order.append(node)
            for child in sorted(adjacency[node], reverse=True):
                if child in seen:
                    continue
                seen.add(child)
                parent[child] = node
                children[node].append(child)
                stack.append(child)
        expected = set(coord_node.values()) | set(pins)
        if seen != expected:
            return _empty_payload(pydb, "openroad_stt_disconnected")

        for node in order:
            for child in children[node]:
                length = abs(node_x[node] - node_x[child]) + abs(
                    node_y[node] - node_y[child]
                )
                edge_resistance[child] = (
                    float(length) / float(dbu) * float(r_unit)
                )
        topo.extend(order)
        topo_start.append(len(topo))
        net_ids.append(net_id)

    max_node = max(node_x) if node_x else num_pins - 1
    size = max_node + 1
    pin_fa = [-1] * size
    node_x_values = [0.0] * size
    node_y_values = [0.0] * size
    edge_resistance_values = [0.0] * size
    for node in range(size):
        node_x_values[node] = float(node_x.get(node, 0))
        node_y_values[node] = float(node_y.get(node, 0))
        edge_resistance_values[node] = float(edge_resistance.get(node, 0.0))
        pin_fa[node] = int(parent.get(node, -1))

    flat_pin_to_start = [0] * (size + 1)
    flat_pin_to = []
    for node in range(size):
        flat_pin_to_start[node] = len(flat_pin_to)
        flat_pin_to.extend(children.get(node, ()))
    flat_pin_to_start[size] = len(flat_pin_to)

    return {
        "artifact": ARTIFACT,
        "artifact_version": ARTIFACT_VERSION,
        "status": "ok",
        "provider": "openroad_stt",
        "steiner_node_base": steiner_base,
        "net_ids": net_ids,
        "net_flat_topo_sort_start": topo_start,
        "net_flat_topo_sort": topo,
        "pin_fa": pin_fa,
        "flat_pin_to_start": flat_pin_to_start,
        "flat_pin_to": flat_pin_to,
        "node_x": node_x_values,
        "node_y": node_y_values,
        "edge_resistance": edge_resistance_values,
        "node_capacitance": [0.0] * size,
    }


__all__ = ["build_openroad_steiner_payload"]
