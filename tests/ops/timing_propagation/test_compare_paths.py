import unittest
import torch
import importlib
import logging
import sys
from pathlib import Path

# this test lives outside the dreamplace package: make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dreamplace.ops.timing_propagation import timing_propagation as tp_mod


def python_extract_critical_paths(endpoints, start_points, pin_rAAT, pin_fAAT, pin_rslack, pin_fslack,
                                  flat_rev, flat_rev_start, max_depth, slack_eps=1e-3):
    """Pure-Python implementation copied from TimingPropagation.get_critical_paths fallback.
    All inputs are CPU torch tensors with appropriate dtypes.
    Returns: list of list of ints (paths)
    """
    endpoints_cpu = endpoints.contiguous()
    start_points_cpu = start_points.contiguous() if start_points is not None else torch.empty(0, dtype=torch.int32)

    rev_offsets = flat_rev_start.contiguous()
    rev_edges = flat_rev.contiguous()
    num_pins = rev_offsets.numel() - 1

    rslack = pin_rslack.detach().cpu().to(torch.float64)
    fslack = pin_fslack.detach().cpu().to(torch.float64)
    raat = pin_rAAT.detach().cpu().to(torch.float64)
    faat = pin_fAAT.detach().cpu().to(torch.float64)

    start_mask = torch.zeros(num_pins, dtype=torch.bool)
    if start_points_cpu is not None and start_points_cpu.numel() > 0:
        sidx = start_points_cpu.detach().cpu().to(torch.int64)
        sidx = sidx[(sidx >= 0) & (sidx < num_pins)]
        if sidx.numel() > 0:
            start_mask[sidx] = True

    depth_limit = max_depth if max_depth > 0 else num_pins
    paths = []
    endpoints_list = endpoints_cpu.tolist()
    for sink in endpoints_list:
        if sink < 0 or sink >= num_pins:
            paths.append([])
            continue

        use_rise = float(rslack[sink]) <= float(fslack[sink])
        current_slack = float(rslack[sink]) if use_rise else float(fslack[sink])
        current_pin = int(sink)
        path = [current_pin]
        visited = {current_pin}

        steps = 0
        while steps < depth_limit:
            if current_pin < 0 or current_pin >= num_pins:
                break
            if start_mask[current_pin]:
                break

            begin = int(rev_offsets[current_pin])
            end = int(rev_offsets[current_pin + 1])
            if begin >= end:
                break

            best_pred = None
            best_slack = float("inf")
            best_gap = float("inf")
            best_arrival = float("-inf")

            for idx in range(begin, end):
                pred = int(rev_edges[idx])
                if pred < 0 or pred >= num_pins or pred in visited:
                    continue

                pred_slack = float(rslack[pred]) if use_rise else float(fslack[pred])
                slack_gap = abs(pred_slack - current_slack)
                arrival = float(raat[pred] if use_rise else faat[pred])

                improved = False
                if pred_slack < best_slack - slack_eps:
                    improved = True
                elif abs(pred_slack - best_slack) <= slack_eps:
                    if slack_gap < best_gap - slack_eps:
                        improved = True
                    elif abs(slack_gap - best_gap) <= slack_eps and arrival > best_arrival + slack_eps:
                        improved = True

                if improved:
                    best_pred = pred
                    best_slack = pred_slack
                    best_gap = slack_gap
                    best_arrival = arrival

            if best_pred is None:
                break

            current_pin = best_pred
            current_slack = float(rslack[best_pred]) if use_rise else float(fslack[best_pred])
            path.append(current_pin)
            visited.add(current_pin)
            steps += 1

        paths.append(list(reversed(path)))

    return paths


class TestExtractCriticalPathsParity(unittest.TestCase):

    def test_cpp_vs_python_extract_paths(self):
        # If C++ extension not available, skip test
        tp_cpp = getattr(tp_mod, "_tp_cpp", None)
        if tp_cpp is None or not hasattr(tp_cpp, "extract_critical_paths"):
            self.skipTest("C++ timing_propagation extension not available; skipping parity test")

        # Build a small linear graph: 0 -> 1 -> 2 -> 3 -> 4
        num_pins = 5
        # Reverse adjacency: preds per pin: 0:[], 1:[0], 2:[1], 3:[2], 4:[3]
        rev_edges = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        rev_start = torch.tensor([0, 0, 1, 2, 3, 4], dtype=torch.int32)

        # endpoints: test sink at pin 4
        endpoints = torch.tensor([4], dtype=torch.int32)

        # start points: pin 0 is a start
        start_points = torch.tensor([0], dtype=torch.int32)

        # AAT and slack values: make slack strictly decreasing along chain so path chosen is unique
        pin_rAAT = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        pin_fAAT = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        # Choose rslack smaller than fslack so rise used; make pred slack improve backwards
        pin_rslack = torch.tensor([10.0, 9.0, 8.0, 7.0, -1.0], dtype=torch.float32)
        pin_fslack = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
        pin_slack = torch.tensor([0.0] * num_pins, dtype=torch.float32)

        max_depth = -1
        slack_eps = 1e-3

        # Prepare empty pin-pair arc mapping tensors (C++ will treat as not provided)
        pin_pair_keys = torch.empty((0, 2), dtype=torch.int32)
        pin_pair_start = torch.empty((0,), dtype=torch.int32)
        pin_pair_indices = torch.empty((0,), dtype=torch.int32)

        # Call C++ implementation
        cpp_paths, cpp_arcs = tp_cpp.extract_critical_paths(
            endpoints,
            1,
            start_points,
            pin_rAAT,
            pin_fAAT,
            pin_rslack,
            pin_fslack,
            pin_slack,
            rev_edges,
            rev_start,
            pin_pair_keys,
            pin_pair_start,
            pin_pair_indices,
            int(max_depth),
            float(slack_eps),
        )

        # Normalize C++ output to python lists of int
        cpp_paths_py = [[int(x) for x in p] for p in cpp_paths]

        # Compute Python fallback's result using the same algorithm
        py_paths = python_extract_critical_paths(
            endpoints,
            start_points,
            pin_rAAT,
            pin_fAAT,
            pin_rslack,
            pin_fslack,
            rev_edges,
            rev_start,
            max_depth,
            slack_eps,
        )

        # Compare
        logging.info("C++ paths: %s", cpp_paths_py)
        logging.info("Python paths: %s", py_paths)
        self.assertEqual(cpp_paths_py, py_paths)


class TestCriticalEndpointTraversalPruner(unittest.TestCase):

    def _require_pruner(self):
        tp_cpp = getattr(tp_mod, "_tp_cpp", None)
        if tp_cpp is None or not hasattr(tp_cpp, "CriticalEndpointTraversalPruner"):
            self.skipTest("C++ traversal-pruning extension not available")
        return tp_cpp

    def test_extension_module_reused_across_package_alias_imports(self):
        from dreamplace.ops.timing_propagation import timing_propagation as canonical_tp

        if tp_mod._tp_cpp is None or canonical_tp._tp_cpp is None:
            self.skipTest("C++ traversal-pruning extension not available")
        self.assertIs(tp_mod._tp_cpp, canonical_tp._tp_cpp)

    def _build_pruner(self):
        tp_cpp = self._require_pruner()
        flat_inst_arcs_by_level = torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 1],
                [2, 3, 0, 2, 0, 0, 2],
            ],
            dtype=torch.int32,
        )
        flat_inst_arcs_by_level_start = torch.tensor(
            [0, 0, 2, 2, 3],
            dtype=torch.int32,
        )
        flat_pin_to_graph_reverse = torch.tensor([0, 1, 2], dtype=torch.int32)
        flat_pin_to_graph_start_reverse = torch.tensor(
            [0, 0, 1, 2, 3],
            dtype=torch.int32,
        )
        pin_pair_arc_keys = torch.tensor(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=torch.int32,
        )
        flat_pin_pair_arc_start = torch.tensor([0, 2, 3], dtype=torch.int32)
        flat_pin_pair_arc_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
        start_points = torch.tensor([0], dtype=torch.int32)
        pin2node_map = torch.tensor([10, 10, 20, 20], dtype=torch.int32)
        return tp_cpp.CriticalEndpointTraversalPruner(
            flat_inst_arcs_by_level,
            flat_inst_arcs_by_level_start,
            flat_pin_to_graph_reverse,
            flat_pin_to_graph_start_reverse,
            pin_pair_arc_keys,
            flat_pin_pair_arc_start,
            flat_pin_pair_arc_indices,
            start_points,
            pin2node_map,
        )

    def test_identity_refresh_keeps_all_arcs_in_original_order(self):
        pruner = self._build_pruner()
        result = pruner.refresh(torch.tensor([3], dtype=torch.int32))

        self.assertEqual(result.kept_flat_arc_indices.tolist(), [0, 1, 2])
        self.assertEqual(result.kept_counts_by_level.tolist(), [0, 2, 0, 1])
        self.assertEqual(result.kept_level_offsets.tolist(), [0, 0, 2, 2, 3])
        self.assertEqual(result.active_arc_count, 3)
        self.assertEqual(result.dropped_arc_count, 0)
        self.assertEqual(result.active_inst_count, 2)

    def test_refresh_is_deterministic_for_repeated_calls(self):
        pruner = self._build_pruner()
        first = pruner.refresh(torch.tensor([3], dtype=torch.int64))
        second = pruner.refresh(torch.tensor([3], dtype=torch.int64))

        self.assertEqual(first.kept_flat_arc_indices.tolist(), second.kept_flat_arc_indices.tolist())
        self.assertEqual(first.kept_counts_by_level.tolist(), second.kept_counts_by_level.tolist())
        self.assertEqual(first.kept_level_offsets.tolist(), second.kept_level_offsets.tolist())

    def test_empty_levels_are_preserved_for_partial_cone(self):
        pruner = self._build_pruner()
        result = pruner.refresh(torch.tensor([1], dtype=torch.int32))

        self.assertEqual(result.kept_flat_arc_indices.tolist(), [0, 1])
        self.assertEqual(result.kept_counts_by_level.tolist(), [0, 2, 0, 0])
        self.assertEqual(result.kept_level_offsets.tolist(), [0, 0, 2, 2, 2])
        self.assertEqual(pruner.num_levels(), 4)

    def test_multi_arc_same_pair_retains_all_arc_ids(self):
        pruner = self._build_pruner()
        result = pruner.refresh(torch.tensor([1], dtype=torch.int32))

        self.assertEqual(result.kept_flat_arc_indices.tolist(), [0, 1])
        self.assertEqual(result.active_arc_count, 2)

    def test_compact_predecessor_edges_retain_multi_arcs_and_traverse_nets(self):
        tp_cpp = self._require_pruner()
        flat_inst_arcs_by_level = torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 1],
                [2, 3, 0, 2, 0, 0, 2],
            ],
            dtype=torch.int32,
        )
        flat_inst_arcs_by_level_start = torch.tensor(
            [0, 0, 2, 2, 3],
            dtype=torch.int32,
        )
        pin_pred_start = torch.tensor([0, 0, 2, 3, 4], dtype=torch.int32)
        pin_pred_pin = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
        pin_pred_arc_id = torch.tensor([0, 1, -1, 2], dtype=torch.int32)
        start_points = torch.tensor([0], dtype=torch.int32)
        pin2node_map = torch.tensor([10, 10, 20, 20], dtype=torch.int32)

        pruner = tp_cpp.CriticalEndpointTraversalPruner(
            flat_inst_arcs_by_level,
            flat_inst_arcs_by_level_start,
            pin_pred_start,
            pin_pred_pin,
            pin_pred_arc_id,
            start_points,
            pin2node_map,
        )
        result = pruner.refresh(torch.tensor([3], dtype=torch.int32))

        self.assertEqual(result.kept_flat_arc_indices.tolist(), [0, 1, 2])
        self.assertEqual(result.kept_counts_by_level.tolist(), [0, 2, 0, 1])
        self.assertEqual(result.kept_level_offsets.tolist(), [0, 0, 2, 2, 3])
        self.assertEqual(result.active_arc_count, 3)

    def test_endpoint_incidence_counts_each_endpoint_cone_once_per_pin(self):
        pruner = self._build_pruner()
        result = pruner.endpoint_incidence(torch.tensor([1, 3], dtype=torch.int32))

        self.assertEqual(result.pin_endpoint_incidence_count.tolist(), [2, 2, 1, 1])
        self.assertEqual(result.arc_endpoint_incidence_count.tolist(), [2, 2, 1])
        self.assertEqual(result.active_endpoint_count, 2)
        self.assertEqual(result.pin_incidence_nonzero_count, 4)
        self.assertEqual(result.arc_incidence_nonzero_count, 3)

    def test_refresh_reports_taskflow_parallel_chunks_for_many_endpoints(self):
        tp_cpp = self._require_pruner()
        num_cones = 8
        flat_rows = []
        pin_pred_start = []
        pin_pred_pin = []
        pin_pred_arc_id = []
        start_points = []
        endpoint_ids = []
        pin2node_map = []

        edge_count = 0
        for cone_idx in range(num_cones):
            src_pin = cone_idx * 2
            dst_pin = src_pin + 1
            flat_rows.append([src_pin, dst_pin, 0, cone_idx, 0, 0, cone_idx])
            start_points.append(src_pin)
            endpoint_ids.append(dst_pin)
            pin2node_map.extend([cone_idx, cone_idx])

            pin_pred_start.append(edge_count)
            pin_pred_start.append(edge_count)
            pin_pred_pin.append(src_pin)
            pin_pred_arc_id.append(cone_idx)
            edge_count += 1
        pin_pred_start.append(edge_count)

        flat_inst_arcs_by_level = torch.tensor(flat_rows, dtype=torch.int32)
        flat_inst_arcs_by_level_start = torch.tensor(
            [0, 0, num_cones],
            dtype=torch.int32,
        )
        pruner = tp_cpp.CriticalEndpointTraversalPruner(
            flat_inst_arcs_by_level,
            flat_inst_arcs_by_level_start,
            torch.tensor(pin_pred_start, dtype=torch.int32),
            torch.tensor(pin_pred_pin, dtype=torch.int32),
            torch.tensor(pin_pred_arc_id, dtype=torch.int32),
            torch.tensor(start_points, dtype=torch.int32),
            torch.tensor(pin2node_map, dtype=torch.int32),
        )
        result = pruner.refresh(torch.tensor(endpoint_ids, dtype=torch.int32))

        self.assertEqual(result.kept_flat_arc_indices.tolist(), list(range(num_cones)))
        self.assertGreater(result.parallel_task_count, 1)


if __name__ == "__main__":
    unittest.main()
