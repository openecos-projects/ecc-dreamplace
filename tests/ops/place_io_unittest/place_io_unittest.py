##
# @file   place_io_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os
import sys
import re
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from dreamplace.ops.place_io import place_io
sys.path.pop()


class Params (object):
    def __init__(self):
        self.aux_input = None


def name2id_map2str(m):
    id2name_map = [None]*len(m)
    for k in m.keys():
        id2name_map[m[k]] = k
    content = ""
    for i in range(len(m)):
        if i:
            content += ", "
        content += "%s : %d" % (id2name_map[i], i)
    return "{%s}" % (content)


def array2str(a):
    content = ""
    for v in a:
        if content:
            content += ", "
        content += "%s" % (v)
    return "[%s]" % (content)


def split_top_level(s):
    """Split "a, b, [c, d]" into its top-level comma-separated items."""
    items = []
    depth = 0
    cur = ""
    for ch in s:
        if ch in "[{(":
            depth += 1
            cur += ch
        elif ch in "]})":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    return items


def parse_value(token):
    token = token.strip()
    if token.startswith("["):
        return [parse_value(t) for t in split_top_level(token[1:-1])]
    if token.startswith("("):
        return tuple(parse_value(t) for t in split_top_level(token[1:-1]))
    if token.startswith("{"):
        m = {}
        for part in split_top_level(token[1:-1]):
            k, v = part.split(":")
            m[k.strip()] = int(v.strip())
        return m
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return token


def parse_dump(text):
    """Parse a place_io pydb dump into a {section: parsed value} dict."""
    dump = {}
    for line in text.strip().splitlines():
        name, value = line.split("=", 1)
        dump[name.strip()] = parse_value(value.strip())
    return dump


def canonicalize(dump):
    """Rewrite a parsed dump into an order-independent canonical form.

    pydb assigns net ids, and (following them) pin ids, in the order the reader
    walks the nets.  That walk order is controlled by the `sort_nets_by_degree`
    option, which this test does not set, so the net/pin ordering is an internal
    detail and must not be pinned by a byte-exact comparison.  Renumber nets by
    sorted net name and pins by their position within each net, so that every
    section can be compared by value while still checking all design content
    (net connectivity, pin node/direction/offset, node and row data).
    """
    net_name2id_map = dump["net_name2id_map"]
    net_id2name = {i: name for name, i in net_name2id_map.items()}
    net_names_sorted = sorted(net_id2name.values())
    new_net_id = {name: i for i, name in enumerate(net_names_sorted)}

    # pin order: walk nets in canonical order, keep each net's stored pin order
    net2pin_map = dump["net2pin_map"]
    pin_order = []
    for name in net_names_sorted:
        pin_order.extend(net2pin_map[net_name2id_map[name]])
    assert sorted(pin_order) == list(range(len(pin_order))), \
        "net2pin_map is not a permutation of the pin ids"
    pin2new = {old: new for new, old in enumerate(pin_order)}

    canonical = {}
    # node-indexed sections have a stable order
    for key in ("num_nodes", "num_terminals", "node_name2id_map", "node_names",
                "node_x", "node_y", "node_orient", "node_size_x", "node_size_y",
                "rows", "xl", "yl", "xh", "yh", "row_height", "site_width",
                "num_movable_pins"):
        canonical[key] = dump[key]

    canonical["net_name2id_map"] = {name: new_net_id[name] for name in net_names_sorted}
    canonical["net_names"] = net_names_sorted
    canonical["net2pin_map"] = [
        [pin2new[p] for p in net2pin_map[net_name2id_map[name]]]
        for name in net_names_sorted
    ]
    canonical["flat_net2pin_map"] = [p for pins in canonical["net2pin_map"] for p in pins]
    flat_start = [0]
    for pins in canonical["net2pin_map"]:
        flat_start.append(flat_start[-1] + len(pins))
    canonical["flat_net2pin_start_map"] = flat_start

    canonical["pin2net_map"] = [new_net_id[net_id2name[dump["pin2net_map"][p]]] for p in pin_order]
    canonical["pin2node_map"] = [dump["pin2node_map"][p] for p in pin_order]
    canonical["pin_direct"] = [dump["pin_direct"][p] for p in pin_order]
    canonical["pin_offset_x"] = [dump["pin_offset_x"][p] for p in pin_order]
    canonical["pin_offset_y"] = [dump["pin_offset_y"][p] for p in pin_order]

    canonical["node2pin_map"] = [[pin2new[p] for p in pins] for pins in dump["node2pin_map"]]
    canonical["flat_node2pin_map"] = [p for pins in canonical["node2pin_map"] for p in pins]
    flat_start = [0]
    for pins in canonical["node2pin_map"]:
        flat_start.append(flat_start[-1] + len(pins))
    canonical["flat_node2pin_start_map"] = flat_start

    return canonical


class PlaceIOOpTest(unittest.TestCase):
    def test_simple(self):
        params = Params()
        design = os.path.dirname(os.path.realpath(__file__))
        params.aux_input = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join(design, "simple/simple.aux")))

        db = place_io.PlaceIOFunction.read(params)
        pydb = place_io.PlaceIOFunction.pydb(db)

        content = ""
        content += "num_nodes = %s\n" % (pydb.num_nodes)
        content += "num_terminals = %s\n" % (pydb.num_terminals)
        content += "node_name2id_map = %s\n" % (name2id_map2str(pydb.node_name2id_map))
        content += "node_names = %s\n" % (array2str(pydb.node_names))
        content += "node_x = %s\n" % (pydb.node_x)
        content += "node_y = %s\n" % (pydb.node_y)
        content += "node_orient = %s\n" % (array2str(pydb.node_orient))
        content += "node_size_x = %s\n" % (pydb.node_size_x)
        content += "node_size_y = %s\n" % (pydb.node_size_y)
        content += "pin_direct = %s\n" % (array2str(pydb.pin_direct))
        content += "pin_offset_x = %s\n" % (pydb.pin_offset_x)
        content += "pin_offset_y = %s\n" % (pydb.pin_offset_y)
        content += "net_name2id_map = %s\n" % (name2id_map2str(pydb.net_name2id_map))
        content += "net_names = %s\n" % (array2str(pydb.net_names))
        content += "net2pin_map = %s\n" % (pydb.net2pin_map)
        content += "flat_net2pin_map = %s\n" % (pydb.flat_net2pin_map)
        content += "flat_net2pin_start_map = %s\n" % (pydb.flat_net2pin_start_map)
        content += "node2pin_map = %s\n" % (pydb.node2pin_map)
        content += "flat_node2pin_map = %s\n" % (pydb.flat_node2pin_map)
        content += "flat_node2pin_start_map = %s\n" % (pydb.flat_node2pin_start_map)
        content += "pin2node_map = %s\n" % (pydb.pin2node_map)
        content += "pin2net_map = %s\n" % (pydb.pin2net_map)
        content += "rows = %s\n" % (pydb.rows)
        content += "xl = %s\n" % (pydb.xl)
        content += "yl = %s\n" % (pydb.yl)
        content += "xh = %s\n" % (pydb.xh)
        content += "yh = %s\n" % (pydb.yh)
        content += "row_height = %s\n" % (pydb.row_height)
        content += "site_width = %s\n" % (pydb.site_width)
        content += "num_movable_pins = %s\n" % (pydb.num_movable_pins)

        with open(os.path.join(design, "simple.golden"), "r") as f:
            golden = f.read()

        # Compare the parsed pydb semantically instead of byte-for-byte: the
        # net- and pin-indexed sections of both dumps are renumbered into a
        # canonical net order (see canonicalize), because the golden was
        # recorded with nets sorted by degree while this test runs with the
        # default unordered read.  Only that internal ordering is released;
        # every section's content is still compared exactly.
        actual_db = canonicalize(parse_dump(content))
        golden_db = canonicalize(parse_dump(golden))

        self.assertEqual(sorted(actual_db.keys()), sorted(golden_db.keys()),
                         "dump sections differ")
        for key in sorted(actual_db.keys()):
            self.assertEqual(actual_db[key], golden_db[key],
                             "pydb section '%s' differs" % key)


if __name__ == "__main__":
    unittest.main()