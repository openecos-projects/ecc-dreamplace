"""Rebuffer-style candidate sites on one rooted Steiner edge.

The rebuffer tree VGDP inserts almost all of its buffers at the parent end of
an edge, isolating that child branch from the trunk.  Equal-count sampling at
1/4, 1/2, 3/4 cannot express that move, so this policy places one site next to
the parent, one at the midpoint and, below Steiner vertices only, one next to
the child.  A site next to a pin child would only re-drive that single pin.
"""

PARENT_END_MAX_RATIO = 0.01


def rebuffer_site_ratios(length_dbu, *, child_is_pin):
    """Return increasing split ratios in (0, 1) for one nonzero edge."""

    length_dbu = float(length_dbu)
    if not length_dbu > 0.0:
        return []
    # One DBU of upstream wire, capped so short edges keep a distinct midpoint.
    end = min(PARENT_END_MAX_RATIO, 1.0 / length_dbu)
    ratios = [end, 0.5]
    if not child_is_pin:
        ratios.append(1.0 - end)
    return ratios


def _edge_cap(edge_rc, parent, child):
    record = edge_rc.get((parent, child)) or edge_rc.get((child, parent)) or {}
    return float(record.get("c", 0.0) or 0.0)


def downstream_loads(candidates, rc_tree):
    """Return the capacitance each segment candidate would drive.

    The load is the child subtree's pin and wire capacitance plus the part of
    the candidate's own edge below the split.  This is the quantity rebuffer
    checks before letting a buffer drive a branch.
    """

    node_cap = {int(node): float(cap) for node, cap in (rc_tree.get("node_cap") or {}).items()}
    children_by_node = {
        int(node): [int(child) for child in children]
        for node, children in (rc_tree.get("children_by_node") or {}).items()
    }
    edge_rc = rc_tree.get("edge_rc") or {}
    subtree = {}

    def subtree_cap(root):
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if node in subtree:
                continue
            children = children_by_node.get(node, [])
            if expanded:
                subtree[node] = node_cap.get(node, 0.0) + sum(
                    _edge_cap(edge_rc, node, child) + subtree[child] for child in children
                )
            else:
                stack.append((node, True))
                stack.extend((child, False) for child in children)
        return subtree[root]

    loads = []
    for candidate in candidates:
        parent = int(candidate["parent_node_id"])
        child = int(candidate["child_node_ids"][0])
        ratio = float(candidate["segment_split_ratio"])
        loads.append(subtree_cap(child) + _edge_cap(edge_rc, parent, child) * (1.0 - ratio))
    return loads


__all__ = ["downstream_loads", "rebuffer_site_ratios"]
