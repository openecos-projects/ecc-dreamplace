"""Incremental 2n sub-net candidate topology and state refresh between B1 rounds."""

from collections import deque

from dreamplace.ops.buffer_insertion.candidate_sites import downstream_loads
from dreamplace.ops.buffer_insertion.candidates import sample_buffer_candidates
from dreamplace.ops.buffer_insertion.fast_buffer_types import (
    FAST_BUFFERS_MODE,
    fast_buffer_load_limits,
)
from dreamplace.ops.buffer_insertion.incremental_compact_payload import (
    DirtyNetCandidateEntry,
    DirtyNetCompactPack,
    LockedBufferInfo,
    splice_compact_payload_and_state,
)
from dreamplace.ops.steiner_topo.timing_rooted_tree_view import (
    build_timing_rooted_tree_view,
)


def _decode_name(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _is_segment_candidate(record):
    return (
        record.get("tree_node_id") is None
        and record.get("node_id") is None
        and record.get("parent_node_id") is not None
        and len(record.get("child_node_ids") or ()) == 1
        and record.get("segment_split_ratio") is not None
    )


def _metadata_for_fast_buffers(data_collections, placedb):
    for container in (data_collections, placedb):
        if container is None:
            continue
        for attr in ("buffering_metadata", "metadata", "pydb"):
            val = getattr(container, attr, None)
            if val is not None:
                return val
    return data_collections if data_collections is not None else placedb


def _extract_orig_net_ids(payload, state):
    metadata = payload.get("metadata") or {}
    if metadata.get("net_ids") is not None:
        return [int(value) for value in metadata["net_ids"]]
    seen = set()
    ordered = []
    for rec in getattr(state, "candidate_records", ()) or ():
        nid = int(rec.get("net_id", rec.get("affected_net_id", -1)))
        if nid >= 0 and nid not in seen:
            seen.add(nid)
            ordered.append(nid)
    return ordered


def _build_pin_name_lookup(placedb, state):
    pin_name_by_id = {}
    raw_names = getattr(placedb, "pin_names", None) if placedb is not None else None
    if raw_names is not None:
        for pin_id, raw_name in enumerate(raw_names):
            name = _decode_name(raw_name)
            if name:
                pin_name_by_id[int(pin_id)] = name
    for rec in getattr(state, "candidate_records", ()) or ():
        drv_id = rec.get("driver_pin_id")
        drv_name = rec.get("driver_pin_name")
        if drv_id is not None and drv_name:
            pin_name_by_id.setdefault(int(drv_id), str(drv_name))
        load_id = rec.get("load_pin_id")
        load_name = rec.get("load_pin_name")
        if load_id is not None and load_name:
            pin_name_by_id.setdefault(int(load_id), str(load_name))
        for pid, pname in zip(
            rec.get("downstream_pin_ids") or (),
            rec.get("downstream_pin_names") or (),
            strict=False,
        ):
            if pid is not None and pname:
                pin_name_by_id.setdefault(int(pid), str(pname))
    return pin_name_by_id


class IncrementalCandidateSession:
    """Tracks split C++ sub-nets per original net across B1 rounds.

    Only the 2n C++ sub-nets touched by newly accepted buffers regenerate
    Steiner trees, candidates, and compact timing subgraphs between rounds;
    all untouched original nets retain their existing slices and masks.
    """

    def __init__(self, *, lane, model, params, incremental_gate):
        self.lane = lane
        self.model = model
        self.params = params
        self.incremental_gate = incremental_gate
        self.data_collections = getattr(model, "data_collections", None)
        self.placedb = getattr(model, "placedb", None)

        state = lane._current_state()
        payload = lane.buffer_relaxed_timing_payload
        self.orig_net_ids = _extract_orig_net_ids(payload, state)
        self.orig_net_row = {nid: idx for idx, nid in enumerate(self.orig_net_ids)}
        timing_op = getattr(
            getattr(model, "op_collections", None), "timing_propagation_op", None
        )
        self.orig_num_pins = int(
            getattr(self.placedb, "num_pins", 0)
            or getattr(timing_op, "num_pins", 0)
            or 0
        )
        self.dbu = int(
            getattr(self.data_collections, "dbu", getattr(params, "dbu", 1000)) or 1000
        )
        self.max_candidates_per_segment = int(
            getattr(params, "buffering_max_candidates_per_segment", 3) or 3
        )
        self.include_tree_node_candidates = bool(
            int(getattr(params, "buffering_include_tree_node_candidates", 0) or 0)
        )
        self.candidate_policy = str(
            getattr(params, "buffering_candidate_policy", "rebuffer_sites")
            or "rebuffer_sites"
        )
        self.drive_type_mode = str(
            getattr(params, "buffering_candidate_drive_type_mode", "single_fixed")
            or "single_fixed"
        )
        self.pin_name_by_id = _build_pin_name_lookup(self.placedb, state)

        # static_payload["sink_net_index"] stores design-level orig_net_id.
        self.orig_pin_cap = {}
        node_caps = payload["node_capacitance"].detach().cpu().tolist()
        sink_nodes = payload["sink_node_id"].detach().cpu().tolist()
        sink_pins = payload["sink_pin_id"].detach().cpu().tolist()
        sink_orig_nets = payload["sink_net_index"].detach().cpu().tolist()
        self.orig_sinks_by_net = {nid: set() for nid in self.orig_net_ids}
        for snode, spin, sink_orig_nid in zip(
            sink_nodes, sink_pins, sink_orig_nets, strict=True
        ):
            spin_i = int(spin)
            orig_nid = int(sink_orig_nid)
            self.orig_pin_cap[spin_i] = float(node_caps[int(snode)])
            if orig_nid in self.orig_sinks_by_net:
                self.orig_sinks_by_net[orig_nid].add(spin_i)

        self.orig_driver_by_net = {}
        self.orig_net_name = {}
        self.orig_driver_name_by_net = {}
        for rec in state.candidate_records:
            nid = int(rec.get("net_id", rec.get("affected_net_id", -1)))
            if nid < 0:
                continue
            if (
                nid not in self.orig_driver_by_net
                and rec.get("driver_pin_id") is not None
            ):
                self.orig_driver_by_net[nid] = int(rec["driver_pin_id"])
            if nid not in self.orig_net_name and rec.get("net_name"):
                self.orig_net_name[nid] = str(rec["net_name"])
            if nid not in self.orig_driver_name_by_net and rec.get("driver_pin_name"):
                self.orig_driver_name_by_net[nid] = str(rec["driver_pin_name"])

        existing_cids = state.candidate_ids.detach().cpu().tolist()
        self.next_candidate_id = (max(existing_cids) + 1) if existing_cids else 1

        self.subnets_by_orig: dict[int, dict[int, dict]] = {}
        self.locked_buffers_by_orig: dict[int, list[LockedBufferInfo]] = {}
        self.cpp_to_orig_net: dict[int, int] = {}

        self._input_cap_by_dtype = []
        if (
            getattr(state, "is_typed_candidate", False)
            and "per_size_input_cap" in payload
        ):
            per_size_cap = payload["per_size_input_cap"][0].detach().cpu().tolist()
            for bsu in state.drive_type_bsu_index.detach().cpu().tolist():
                self._input_cap_by_dtype.append(float(per_size_cap[int(bsu)]))
        elif "per_size_input_cap" in payload:
            per_size_cap = payload["per_size_input_cap"][0].detach().cpu().tolist()
            bsu = int(getattr(state, "fixed_bsu_index", 0) or 0)
            self._input_cap_by_dtype.append(float(per_size_cap[bsu]))

    def invalidate_model_caches(self):
        model = self.model
        if model is None:
            return
        model._relaxed_buffer_dynamic_net_provider = None
        model._relaxed_buffer_dynamic_net_provider_key = None
        model._timing_geometry_cache = None
        model._virtual_cell_density_view = None
        if self.lane is not None:
            self.lane._full_candidate_topology_signature = None
            self.lane._full_candidate_structure_signature = None

    def _fast_limits(self, state):
        if (
            self.drive_type_mode != FAST_BUFFERS_MODE
            or not getattr(state, "is_typed_candidate", False)
            or (self.data_collections is None and self.placedb is None)
        ):
            return None
        return fast_buffer_load_limits(
            _metadata_for_fast_buffers(self.data_collections, self.placedb),
            state.drive_type_master_names,
        )

    def _view_node_id(self, cpp_net_id, global_v, pin_id):
        if int(pin_id) >= 0:
            return int(pin_id)
        return -((int(cpp_net_id) + 1) * 100000 + int(global_v) + 1)

    def _compute_buffer_tree_metadata(self, orig_net_id):
        """Compute transitive real sinks and topological buffer-tree depth."""
        subnets = self.subnets_by_orig[orig_net_id]
        locked = self.locked_buffers_by_orig[orig_net_id]
        buf_by_pin_in = {buf.p_in: buf for buf in locked}
        subnet_by_drv = {int(t["driver_pin_id"]): t for t in subnets.values()}
        orig_sinks = self.orig_sinks_by_net.get(orig_net_id, set())

        root_drv = self.orig_driver_by_net.get(orig_net_id)
        if root_drv is None or root_drv not in subnet_by_drv:
            locked_p_outs = {buf.p_out for buf in locked}
            roots = [t for drv, t in subnet_by_drv.items() if drv not in locked_p_outs]
            root_drv = (
                int(roots[0]["driver_pin_id"])
                if roots
                else next(iter(subnet_by_drv))
            )

        subnet_buf_depth = {}
        subtree_sinks = {}
        visited_subnets = set()

        def visit_subnet(drv_pin_id, buf_depth):
            t = subnet_by_drv.get(int(drv_pin_id))
            if t is None:
                return ()
            cid = int(t["net_id"])
            if cid in visited_subnets:
                return ()
            visited_subnets.add(cid)
            subnet_buf_depth[cid] = int(buf_depth)

            topo_gv = [int(gv) for gv in t["topo"]]
            k_of_gv = {gv: k for k, gv in enumerate(topo_gv)}
            parents_gv = [int(p) for p in t["parent"]]
            children = {gv: [] for gv in topo_gv}
            for k, gv in enumerate(topo_gv):
                p_gv = parents_gv[k]
                if p_gv >= 0 and p_gv in children:
                    children[p_gv].append(gv)

            v_pins = [int(p) for p in t["vertex_pin_id"]]
            for gv in reversed(topo_gv):
                k = k_of_gv[gv]
                pin_v = v_pins[k]
                sinks = set()
                if pin_v in orig_sinks or (
                    pin_v >= 0
                    and pin_v != int(t["driver_pin_id"])
                    and pin_v not in buf_by_pin_in
                    and pin_v < self.orig_num_pins
                ):
                    sinks.add(pin_v)
                elif pin_v in buf_by_pin_in:
                    buf = buf_by_pin_in[pin_v]
                    child_sinks = visit_subnet(buf.p_out, buf_depth + 1)
                    buf.buffer_depth = int(buf_depth)
                    buf.downstream_pin_ids = tuple(child_sinks)
                    sinks.update(child_sinks)
                for ch_gv in children[gv]:
                    sinks.update(subtree_sinks[(cid, ch_gv)])
                subtree_sinks[(cid, gv)] = tuple(sorted(sinks))
            return subtree_sinks[(cid, int(t["root_vertex"]))]

        visit_subnet(root_drv, 0)

        for buf in locked:
            b_sinks = buf.downstream_pin_ids
            b_names = tuple(
                self.pin_name_by_id[p] for p in b_sinks if p in self.pin_name_by_id
            )
            rec = buf.record
            rec["segment_tree_depth"] = int(buf.buffer_depth)
            rec["downstream_pin_ids"] = b_sinks
            rec["downstream_pin_names"] = b_names
            rec["downstream_pin_count"] = len(b_sinks)
            if b_sinks:
                rec["load_pin_id"] = b_sinks[0]
                if b_sinks[0] in self.pin_name_by_id:
                    rec["load_pin_name"] = self.pin_name_by_id[b_sinks[0]]

        return subnet_buf_depth, subtree_sinks, buf_by_pin_in

    def _annotate_downstream_fields(self, item, down_pins, buf_depth):
        item["segment_tree_depth"] = int(buf_depth)
        item["downstream_pin_ids"] = tuple(down_pins)
        item["downstream_pin_names"] = tuple(
            self.pin_name_by_id[p] for p in down_pins if p in self.pin_name_by_id
        )
        item["downstream_pin_count"] = len(down_pins)
        if down_pins:
            item["load_pin_id"] = down_pins[0]
            if down_pins[0] in self.pin_name_by_id:
                item["load_pin_name"] = self.pin_name_by_id[down_pins[0]]

    def _build_dirty_net_pack(self, *, orig_net_id, state, fast_limits):
        """Stitch all active C++ sub-nets of orig_net_id into one net-local compact tree."""
        subnets = self.subnets_by_orig[orig_net_id]
        locked = self.locked_buffers_by_orig[orig_net_id]
        subnet_buf_depth, subtree_sinks, buf_by_pin_in = (
            self._compute_buffer_tree_metadata(orig_net_id)
        )
        locked_p_outs = {buf.p_out for buf in locked}

        root_subnets = [
            cid
            for cid, t in sorted(subnets.items())
            if int(t["driver_pin_id"]) not in locked_p_outs
        ]
        if not root_subnets:
            raise RuntimeError(f"Missing root sub-net for original net {orig_net_id}")
        root_cid = root_subnets[0]

        buffer_main_type_index = int(state.buffer_main_type_index)
        net_name = self.orig_net_name.get(orig_net_id, "")
        drv_pin_id = int(
            self.orig_driver_by_net.get(orig_net_id, subnets[root_cid]["driver_pin_id"])
        )
        drv_pin_name = self.orig_driver_name_by_net.get(
            orig_net_id, self.pin_name_by_id.get(drv_pin_id, "")
        )

        local_parents = []
        local_edge_r = []
        local_edge_c = []
        local_node_cap = []
        local_orig_node = []
        sink_pairs = []
        net_candidates = []

        subnet_by_p_out = {
            int(t["driver_pin_id"]): int(cid) for cid, t in subnets.items()
        }
        num_dtypes = (
            int(state.drive_type_count)
            if getattr(state, "is_typed_candidate", False)
            else 1
        )

        def add_compact_node(parent_local, edge_r, edge_c, node_cap, orig_node):
            idx = len(local_parents)
            local_parents.append(int(parent_local))
            local_edge_r.append(float(edge_r))
            local_edge_c.append(float(edge_c))
            local_node_cap.append(float(node_cap))
            local_orig_node.append(int(orig_node))
            return idx

        def stitch_subnet(cid, parent_compact_for_root):
            t = subnets[cid]
            topo_gv = [int(gv) for gv in t["topo"]]
            k_of_gv = {gv: k for k, gv in enumerate(topo_gv)}
            parents_gv = [int(p) for p in t["parent"]]
            edge_r_v = [float(x) for x in t["edge_r"]]
            edge_c_v = [float(x) for x in t["edge_c"]]
            vx_v = [float(x) for x in t["vertex_x_dbu"]]
            vy_v = [float(x) for x in t["vertex_y_dbu"]]
            vpin_v = [int(x) for x in t["vertex_pin_id"]]
            root_gv = int(t["root_vertex"])
            buf_depth = int(subnet_buf_depth.get(cid, 0))

            view_id_of_gv = {
                gv: self._view_node_id(cid, gv, vpin_v[k])
                for k, gv in enumerate(topo_gv)
            }
            gv_by_view_id = {vid: gv for gv, vid in view_id_of_gv.items()}
            edges = []
            children_gv = {gv: [] for gv in topo_gv}
            rc_edge_map = {}
            node_cap_map = {}
            coords_dbu = {}
            net_pin_ids = []
            for k, gv in enumerate(topo_gv):
                vid = view_id_of_gv[gv]
                coords_dbu[vid] = (vx_v[k], vy_v[k])
                pv = vpin_v[k]
                if pv >= 0:
                    net_pin_ids.append(vid)
                if pv in self.orig_pin_cap and gv != root_gv:
                    node_cap_map[vid] = float(self.orig_pin_cap[pv])
                elif pv in buf_by_pin_in and gv != root_gv:
                    dt_id = buf_by_pin_in[pv].drive_type_id
                    if 0 <= dt_id < len(self._input_cap_by_dtype):
                        node_cap_map[vid] = float(self._input_cap_by_dtype[dt_id])
                p_gv = parents_gv[k]
                if p_gv >= 0 and p_gv in children_gv:
                    edges.append((view_id_of_gv[p_gv], vid))
                    children_gv[p_gv].append(gv)
                    rc_edge_map[(view_id_of_gv[p_gv], vid)] = {
                        "r": edge_r_v[k],
                        "c": edge_c_v[k],
                    }

            tree_view = build_timing_rooted_tree_view(
                net_id=orig_net_id,
                net_pin_ids=net_pin_ids,
                driver_pin_id=view_id_of_gv[root_gv],
                undirected_edges=edges,
                coordinates=coords_dbu,
            )
            generated = []
            if tree_view.get("status") == "ok":
                generated = sample_buffer_candidates(
                    tree_view,
                    buffer_main_type_index=buffer_main_type_index,
                    dbu=self.dbu,
                    max_candidates_per_segment=self.max_candidates_per_segment,
                    include_tree_node_candidates=self.include_tree_node_candidates,
                    candidate_generation_policy=self.candidate_policy,
                    return_summary=False,
                )

            seg_cands_by_edge = {}
            tree_cands_by_vertex = {}
            if generated:
                rc_tree = {
                    "node_cap": node_cap_map,
                    "children_by_node": {
                        int(k): [int(c) for c in ch]
                        for k, ch in tree_view["children_by_node"].items()
                    },
                    "edge_rc": rc_edge_map,
                }
                seg_recs = [cand for cand in generated if _is_segment_candidate(cand)]
                loads = (
                    downstream_loads(seg_recs, rc_tree)
                    if seg_recs and fast_limits is not None
                    else [0.0] * len(seg_recs)
                )
                seg_load_by_idx = {
                    id(cand): float(ld)
                    for cand, ld in zip(seg_recs, loads, strict=True)
                }

                for cand in generated:
                    item = dict(cand)
                    cid_val = self.next_candidate_id
                    self.next_candidate_id += 1
                    item["candidate_id"] = int(cid_val)
                    item["net_id"] = int(orig_net_id)
                    item["net_name"] = net_name
                    item["driver_pin_id"] = drv_pin_id
                    item["driver_pin_name"] = drv_pin_name
                    item["cpp_net_id"] = int(cid)
                    item["cpp_subnet_refreshed"] = True

                    if _is_segment_candidate(item):
                        p_gv = gv_by_view_id[int(item["parent_node_id"])]
                        c_gv = gv_by_view_id[int(item["child_node_ids"][0])]
                        item["cpp_parent_vertex"] = int(p_gv)
                        item["cpp_child_vertices"] = (int(c_gv),)
                        down_pins = subtree_sinks.get((cid, c_gv), ())
                        self._annotate_downstream_fields(item, down_pins, buf_depth)
                        load_val = seg_load_by_idx.get(id(cand), 0.0)
                        allowed = (
                            [load_val <= float(lim) for lim in fast_limits]
                            if fast_limits is not None
                            else [True] * num_dtypes
                        )
                        seg_cands_by_edge.setdefault((p_gv, c_gv), []).append(
                            (float(item["segment_split_ratio"]), item, allowed)
                        )
                    else:
                        t_gv = gv_by_view_id[
                            int(item.get("tree_node_id", item.get("node_id")))
                        ]
                        item["cpp_parent_vertex"] = int(t_gv)
                        item["cpp_child_vertices"] = ()
                        down_pins = subtree_sinks.get((cid, t_gv), ())
                        self._annotate_downstream_fields(item, down_pins, buf_depth)
                        tree_cands_by_vertex.setdefault(t_gv, []).append(
                            (item, [True] * num_dtypes)
                        )

            for key in seg_cands_by_edge:
                seg_cands_by_edge[key].sort(key=lambda pair: pair[0])

            local_compact_of_gv = {}
            if parent_compact_for_root is None:
                root_pin = vpin_v[k_of_gv[root_gv]]
                root_compact = add_compact_node(
                    -1,
                    0.0,
                    0.0,
                    0.0,
                    root_pin if 0 <= root_pin < self.orig_num_pins else -1,
                )
            else:
                root_compact = int(parent_compact_for_root)
            local_compact_of_gv[root_gv] = root_compact

            queue = deque([root_gv])
            while queue:
                u_gv = queue.popleft()
                u_compact = local_compact_of_gv[u_gv]
                for item, allowed in tree_cands_by_vertex.get(u_gv, ()):
                    net_candidates.append(
                        DirtyNetCandidateEntry(u_compact, item, None, allowed)
                    )

                for v_gv in children_gv[u_gv]:
                    vk = k_of_gv[v_gv]
                    total_r = edge_r_v[vk]
                    total_c = edge_c_v[vk]
                    curr_parent = u_compact
                    prev_ratio = 0.0
                    for ratio, item, allowed in seg_cands_by_edge.get((u_gv, v_gv), ()):
                        delta = max(0.0, float(ratio) - prev_ratio)
                        split_compact = add_compact_node(
                            curr_parent,
                            total_r * delta,
                            total_c * delta,
                            0.0,
                            -1,
                        )
                        net_candidates.append(
                            DirtyNetCandidateEntry(split_compact, item, None, allowed)
                        )
                        curr_parent = split_compact
                        prev_ratio = float(ratio)

                    rem_ratio = max(0.0, 1.0 - prev_ratio)
                    vp = vpin_v[vk]
                    if vp in buf_by_pin_in:
                        buf = buf_by_pin_in[vp]
                        junction = add_compact_node(
                            curr_parent,
                            total_r * rem_ratio,
                            total_c * rem_ratio,
                            0.0,
                            -1,
                        )
                        local_compact_of_gv[v_gv] = junction
                        buf_compact = junction
                        if children_gv[v_gv]:
                            # Same-subnet wire branches stay on the input side.
                            buf_compact = add_compact_node(junction, 0.0, 0.0, 0.0, -1)
                            queue.append(v_gv)
                        # Another insertion upstream can move this input to a new subnet.
                        buf.upstream_subnet = int(cid)
                        buf.record["cpp_net_id"] = int(cid)
                        dt_id = buf.drive_type_id
                        allowed = [idx == dt_id for idx in range(num_dtypes)]
                        net_candidates.append(
                            DirtyNetCandidateEntry(
                                buf_compact, buf.record, dt_id, allowed
                            )
                        )
                        child_cid = subnet_by_p_out.get(buf.p_out)
                        if child_cid is not None:
                            stitch_subnet(child_cid, buf_compact)
                    else:
                        is_orig_pin = 0 <= vp < self.orig_num_pins
                        is_sink = is_orig_pin and vp != drv_pin_id
                        ncap = float(self.orig_pin_cap.get(vp, 0.0)) if is_sink else 0.0
                        v_compact = add_compact_node(
                            curr_parent,
                            total_r * rem_ratio,
                            total_c * rem_ratio,
                            ncap,
                            vp if is_orig_pin else -1,
                        )
                        local_compact_of_gv[v_gv] = v_compact
                        if is_sink:
                            sink_pairs.append((v_compact, vp))
                        queue.append(v_gv)

        stitch_subnet(root_cid, None)

        num_local = len(local_parents)
        children_by_local = [[] for _ in range(num_local)]
        for idx, p in enumerate(local_parents):
            if p >= 0:
                children_by_local[p].append(idx)

        return DirtyNetCompactPack(
            topo=list(range(num_local)),
            pin_fa=local_parents,
            children_by_local=children_by_local,
            edge_r=local_edge_r,
            edge_c=local_edge_c,
            node_cap=local_node_cap,
            orig_node=local_orig_node,
            sinks=sink_pairs,
            candidates=net_candidates,
        )

    def refresh_round(self, *, accepted_action_infos, dirty_nets):
        """Incrementally refresh only the 2n C++ sub-nets touched in the round."""
        dirty_nets = [int(nid) for nid in (dirty_nets or ()) if int(nid) >= 0]
        accepted_action_infos = list(accepted_action_infos or ())
        if not dirty_nets or not accepted_action_infos:
            return {"status": "skipped", "reason": "no_dirty_subnets"}

        state = self.lane._current_state()
        payload = self.lane.buffer_relaxed_timing_payload
        fast_limits = self._fast_limits(state)
        is_typed = bool(getattr(state, "is_typed_candidate", False))

        old_cand_ids = state.candidate_ids.detach().cpu().tolist()
        old_row_by_cid = {int(cid): row for row, cid in enumerate(old_cand_ids)}

        for info in accepted_action_infos:
            orig_net_id = int(info.get("orig_net_id", -1))
            active_net = int(info.get("active_net", -1))
            new_net = int(info.get("new_net", -1))
            if orig_net_id < 0:
                continue
            self.cpp_to_orig_net[orig_net_id] = orig_net_id
            if active_net >= 0:
                self.cpp_to_orig_net[active_net] = orig_net_id
            if new_net >= 0:
                self.cpp_to_orig_net[new_net] = orig_net_id

            cid_val = int(info["candidate_id"])
            old_row = old_row_by_cid.get(cid_val, int(info.get("row_idx", -1)))
            base_rec = (
                dict(state.candidate_records[old_row])
                if 0 <= old_row < len(state.candidate_records)
                else {}
            )
            if info.get("x_dbu") is not None and info.get("y_dbu") is not None:
                base_rec["x_dbu"] = float(info["x_dbu"])
                base_rec["y_dbu"] = float(info["y_dbu"])
                base_rec["candidate_location_x"] = float(info["x_dbu"]) / float(
                    self.dbu
                )
                base_rec["candidate_location_y"] = float(info["y_dbu"]) / float(
                    self.dbu
                )
            base_rec["cpp_subnet_refreshed"] = True
            base_rec["cpp_net_id"] = (
                int(active_net) if active_net >= 0 else int(orig_net_id)
            )
            self.locked_buffers_by_orig.setdefault(orig_net_id, []).append(
                LockedBufferInfo(
                    candidate_id=cid_val,
                    drive_type_id=int(info.get("drive_type_id") or 0),
                    p_in=int(info["p_in"]),
                    p_out=int(info["p_out"]),
                    upstream_subnet=active_net,
                    downstream_subnet=new_net,
                    buf_node=int(info["buf_node"]),
                    record=base_rec,
                )
            )

        refreshed_trees = self.incremental_gate.refresh_dirty_subnets(dirty_nets)
        dirty_orig_nets = set()
        for tree in refreshed_trees:
            cid = int(tree["net_id"])
            orig_net_id = self.cpp_to_orig_net.get(cid, cid)
            self.subnets_by_orig.setdefault(orig_net_id, {})[cid] = tree
            dirty_orig_nets.add(orig_net_id)

        if not dirty_orig_nets:
            return {"status": "skipped", "reason": "no_dirty_orig_nets"}

        dirty_packs_by_orig_net = {
            orig_nid: self._build_dirty_net_pack(
                orig_net_id=orig_nid,
                state=state,
                fast_limits=fast_limits,
            )
            for orig_nid in sorted(dirty_orig_nets)
            if orig_nid in self.orig_net_row
        }
        if not dirty_packs_by_orig_net:
            return {"status": "skipped", "reason": "dirty_nets_not_in_payload"}

        new_payload, new_state, old_to_new_row, group_count = (
            splice_compact_payload_and_state(
                orig_net_ids=self.orig_net_ids,
                dirty_packs_by_orig_net=dirty_packs_by_orig_net,
                payload=payload,
                state=state,
                old_row_by_cid=old_row_by_cid,
            )
        )

        self.incremental_gate.remap_rejection_masks(
            is_typed=is_typed,
            new_shape=tuple(new_state.activation_param.shape),
            old_to_new_rows=old_to_new_row,
            dirty_orig_nets=dirty_orig_nets,
            new_candidate_orig_nets=new_state.candidate_net_id.detach().cpu().tolist(),
        )

        self.lane.buffer_optimization_state = new_state
        self.lane.buffer_relaxed_timing_payload = new_payload
        if self.data_collections is not None:
            self.lane.attach_state(self.data_collections)
        self.lane.register_model_parameters(self.model)
        self.invalidate_model_caches()
        return {
            "status": "ok",
            "dirty_subnet_count": len(refreshed_trees),
            "dirty_orig_net_count": len(dirty_packs_by_orig_net),
            "candidate_count": int(new_state.candidate_ids.numel()),
            "subnet_group_count": int(group_count),
        }
