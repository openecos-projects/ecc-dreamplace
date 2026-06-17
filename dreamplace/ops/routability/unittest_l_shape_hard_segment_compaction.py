#!/usr/bin/python3

import os
import sys
import unittest
import importlib.util
import ast
from unittest import mock

import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTODMP_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)
MODULE_PATH = os.path.join(CURRENT_DIR, "l_shape_segment.py")
ELECTRIC_POTENTIAL_PATH = os.path.join(CURRENT_DIR, "l_shape_electric_potential.py")
SPEC = importlib.util.spec_from_file_location("l_shape_segment", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load l_shape_segment.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

H_FIRST = MODULE.H_FIRST
LShapeSegmentOp = MODULE.LShapeSegmentOp
STRAIGHT = MODULE.STRAIGHT
V_FIRST = MODULE.V_FIRST


def _make_inputs(dtype=torch.float32):
    newx = torch.tensor(
        [0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 3.0, 3.0],
        dtype=dtype,
        requires_grad=True,
    )
    newy = torch.tensor(
        [0.0, 0.0, 2.0, 5.0, 1.0, 4.0, 3.0, 3.0],
        dtype=dtype,
        requires_grad=True,
    )
    flat_from = torch.tensor([0, 2, 4, 6], dtype=torch.long)
    flat_to = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    l_directions = torch.tensor([STRAIGHT, H_FIRST, V_FIRST, STRAIGHT], dtype=torch.long)
    return newx, newy, flat_from, flat_to, l_directions


def _run_op(wire_width=0.2, wire_width_h=None, wire_width_v=None, deterministic_backward=True):
    newx, newy, flat_from, flat_to, l_directions = _make_inputs()
    op = LShapeSegmentOp(
        wire_width=wire_width,
        wire_width_h=wire_width_h,
        wire_width_v=wire_width_v,
        deterministic_backward=deterministic_backward,
    )
    result = op(newx, newy, flat_from, flat_to, l_directions)
    return result, newx, newy


def _run_reference_hard_compaction(wire_width=0.2, wire_width_h=None, wire_width_v=None):
    newx, newy, flat_from, flat_to, l_directions = _make_inputs()
    op = LShapeSegmentOp(
        wire_width=wire_width,
        wire_width_h=wire_width_h,
        wire_width_v=wire_width_v,
        deterministic_backward=True,
    )
    topo = op._compute_topology(flat_from, flat_to, l_directions, len(newx), newx.device)

    valid_from = topo["valid_from"]
    valid_to = topo["valid_to"]
    valid_edge_idx = topo["valid_edge_idx"]
    valid_l_dir = l_directions[topo["valid_mask"]]
    is_h_first = (valid_l_dir == H_FIRST) | (valid_l_dir == MODULE.FAKE_STRAIGHT)
    is_v_first = valid_l_dir == V_FIRST
    is_straight_by_dir = valid_l_dir == STRAIGHT

    x1 = newx[valid_from]
    y1 = newy[valid_from]
    x2 = newx[valid_to]
    y2 = newy[valid_to]

    is_horizontal_line = torch.abs(y1 - y2) < 1e-4
    is_vertical_line = torch.abs(x1 - x2) < 1e-4
    diag_straight = is_straight_by_dir & ~(is_horizontal_line | is_vertical_line)
    is_straight = (is_horizontal_line | is_vertical_line) | (is_straight_by_dir & ~diag_straight)
    is_upper_l = (~is_straight) & (is_h_first | diag_straight)
    is_lower_l = (~is_straight) & is_v_first
    corner_x = torch.where(is_upper_l, x2, torch.where(is_lower_l, x1, x1))
    corner_y = torch.where(is_upper_l, y1, torch.where(is_lower_l, y2, y1))

    seg1_x2 = torch.where(is_straight, x2, corner_x)
    seg1_y2 = torch.where(is_straight, y2, corner_y)
    seg1_min_x = torch.minimum(x1, seg1_x2)
    seg1_max_x = torch.maximum(x1, seg1_x2)
    seg1_min_y = torch.minimum(y1, seg1_y2)
    seg1_max_y = torch.maximum(y1, seg1_y2)
    seg1_is_h = torch.abs(seg1_x2 - x1) >= torch.abs(seg1_y2 - y1)

    seg2_min_x = torch.minimum(corner_x, x2)
    seg2_max_x = torch.maximum(corner_x, x2)
    seg2_min_y = torch.minimum(corner_y, y2)
    seg2_max_y = torch.maximum(corner_y, y2)
    seg2_is_h = torch.abs(x2 - corner_x) >= torch.abs(y2 - corner_y)

    if wire_width_h is not None or wire_width_v is not None:
        width_h = float(wire_width if wire_width_h is None else wire_width_h)
        width_v = float(wire_width if wire_width_v is None else wire_width_v)
        half_width_h = width_h / 2.0
        half_width_v = width_v / 2.0
        seg1_llx = torch.where(seg1_is_h, seg1_min_x, seg1_min_x - half_width_v)
        seg1_lly = torch.where(seg1_is_h, seg1_min_y - half_width_h, seg1_min_y)
        seg1_size_x = torch.where(seg1_is_h, seg1_max_x - seg1_min_x, torch.full_like(seg1_min_x, width_v))
        seg1_size_y = torch.where(seg1_is_h, torch.full_like(seg1_min_x, width_h), seg1_max_y - seg1_min_y)
        seg2_llx = torch.where(seg2_is_h, seg2_min_x, seg2_min_x - half_width_v)
        seg2_lly = torch.where(seg2_is_h, seg2_min_y - half_width_h, seg2_min_y)
        seg2_size_x = torch.where(seg2_is_h, seg2_max_x - seg2_min_x, torch.full_like(seg2_min_x, width_v))
        seg2_size_y = torch.where(seg2_is_h, torch.full_like(seg2_min_x, width_h), seg2_max_y - seg2_min_y)
    else:
        half_width = wire_width / 2.0
        seg1_llx = seg1_min_x - half_width
        seg1_lly = seg1_min_y - half_width
        seg1_size_x = (seg1_max_x - seg1_min_x) + 2 * half_width
        seg1_size_y = (seg1_max_y - seg1_min_y) + 2 * half_width
        seg2_llx = seg2_min_x - half_width
        seg2_lly = seg2_min_y - half_width
        seg2_size_x = (seg2_max_x - seg2_min_x) + 2 * half_width
        seg2_size_y = (seg2_max_y - seg2_min_y) + 2 * half_width

    is_l_shape = is_upper_l | is_lower_l
    seg1_valid = is_straight | is_l_shape
    seg2_valid = is_l_shape & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
    segment_llx = torch.cat((seg1_llx[seg1_valid], seg2_llx[seg2_valid]))
    segment_lly = torch.cat((seg1_lly[seg1_valid], seg2_lly[seg2_valid]))
    segment_size_x = torch.cat((seg1_size_x[seg1_valid], seg2_size_x[seg2_valid]))
    segment_size_y = torch.cat((seg1_size_y[seg1_valid], seg2_size_y[seg2_valid]))
    segment_edge_idx = torch.cat((valid_edge_idx[seg1_valid], valid_edge_idx[seg2_valid]))
    segment_is_horizontal = torch.cat((seg1_is_h[seg1_valid], seg2_is_h[seg2_valid]))
    segment_weight = torch.cat((torch.ones_like(seg1_size_x[seg1_valid]), torch.ones_like(seg2_size_x[seg2_valid])))

    if wire_width_h is not None or wire_width_v is not None:
        valid_seg = (segment_size_x > 1e-6) & (segment_size_y > 1e-6)
    else:
        min_size = max((wire_width / 2.0) * 2, 1e-6)
        valid_seg = (segment_size_x > min_size) | (segment_size_y > min_size)

    result = {
        "segment_llx": segment_llx[valid_seg],
        "segment_lly": segment_lly[valid_seg],
        "segment_size_x": segment_size_x[valid_seg],
        "segment_size_y": segment_size_y[valid_seg],
        "segment_edge_idx": segment_edge_idx[valid_seg],
        "segment_is_horizontal": segment_is_horizontal[valid_seg],
        "segment_weight": segment_weight[valid_seg],
    }
    result["num_segments"] = result["segment_llx"].numel()
    return result, newx, newy


class TestLShapeHardSegmentCompaction(unittest.TestCase):
    def assert_forward_matches_expected(self, result, expected):
        for key, value in expected.items():
            actual = result[key]
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    self.assertTrue(
                        torch.allclose(actual, value, atol=1e-7, rtol=0),
                        msg=f"{key}: {actual} != {value}",
                    )
                else:
                    self.assertTrue(torch.equal(actual, value), msg=f"{key}: {actual} != {value}")
            else:
                self.assertEqual(actual, value, msg=key)

    def test_hard_forward_preserves_segment_order_and_masks(self):
        result, _, _ = _run_op(wire_width=0.2)

        expected = {
            "segment_llx": torch.tensor([-0.1, 1.9, 0.9, 3.9, 0.9], dtype=torch.float32),
            "segment_lly": torch.tensor([-0.1, 1.9, 0.9, 1.9, 3.9], dtype=torch.float32),
            "segment_size_x": torch.tensor([2.2, 2.2, 0.2, 0.2, 2.2], dtype=torch.float32),
            "segment_size_y": torch.tensor([0.2, 0.2, 3.2, 3.2, 0.2], dtype=torch.float32),
            "segment_edge_idx": torch.tensor([0, 1, 2, 1, 2], dtype=torch.long),
            "segment_is_horizontal": torch.tensor([True, True, False, False, True]),
            "segment_weight": torch.ones(5, dtype=torch.float32),
            "num_segments": 5,
        }
        self.assert_forward_matches_expected(result, expected)

    def test_directional_width_forward_preserves_segment_order_and_masks(self):
        result, _, _ = _run_op(wire_width=0.0, wire_width_h=0.4, wire_width_v=0.6)

        expected = {
            "segment_llx": torch.tensor([0.0, 2.0, 0.7, 3.7, 1.0], dtype=torch.float32),
            "segment_lly": torch.tensor([-0.2, 1.8, 1.0, 2.0, 3.8], dtype=torch.float32),
            "segment_size_x": torch.tensor([2.0, 2.0, 0.6, 0.6, 2.0], dtype=torch.float32),
            "segment_size_y": torch.tensor([0.4, 0.4, 3.0, 3.0, 0.4], dtype=torch.float32),
            "segment_edge_idx": torch.tensor([0, 1, 2, 1, 2], dtype=torch.long),
            "segment_is_horizontal": torch.tensor([True, True, False, False, True]),
            "segment_weight": torch.ones(5, dtype=torch.float32),
            "num_segments": 5,
        }
        self.assert_forward_matches_expected(result, expected)

    def test_diagonal_straight_direction_falls_back_to_h_first_l_shape(self):
        newx = torch.tensor([0.0, 2.0], dtype=torch.float32, requires_grad=True)
        newy = torch.tensor([0.0, 2.0], dtype=torch.float32, requires_grad=True)
        flat_from = torch.tensor([0], dtype=torch.long)
        flat_to = torch.tensor([1], dtype=torch.long)
        l_directions = torch.tensor([STRAIGHT], dtype=torch.long)
        op = LShapeSegmentOp(wire_width=0.2, deterministic_backward=True)

        result = op(newx, newy, flat_from, flat_to, l_directions)

        expected = {
            "segment_llx": torch.tensor([-0.1, 1.9], dtype=torch.float32),
            "segment_lly": torch.tensor([-0.1, -0.1], dtype=torch.float32),
            "segment_size_x": torch.tensor([2.2, 0.2], dtype=torch.float32),
            "segment_size_y": torch.tensor([0.2, 2.2], dtype=torch.float32),
            "segment_edge_idx": torch.tensor([0, 0], dtype=torch.long),
            "segment_is_horizontal": torch.tensor([True, False]),
            "segment_weight": torch.ones(2, dtype=torch.float32),
            "num_segments": 2,
        }
        self.assert_forward_matches_expected(result, expected)

    def test_size_and_weight_are_forward_only_in_hard_active_path(self):
        result, _, _ = _run_op(wire_width=0.2)

        self.assertIn("HardSegmentPositionCompaction", type(result["segment_llx"].grad_fn).__name__)
        self.assertIn("HardSegmentPositionCompaction", type(result["segment_lly"].grad_fn).__name__)
        self.assertFalse(result["segment_size_x"].requires_grad)
        self.assertFalse(result["segment_size_y"].requires_grad)
        self.assertFalse(result["segment_weight"].requires_grad)
        self.assertIsNone(result["segment_size_x"].grad_fn)
        self.assertIsNone(result["segment_size_y"].grad_fn)
        self.assertIsNone(result["segment_weight"].grad_fn)

    def test_active_electric_backward_contract_returns_no_size_or_weight_gradients(self):
        with open(ELECTRIC_POTENTIAL_PATH, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=ELECTRIC_POTENTIAL_PATH)

        target_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "SegmentElectricPotentialFunction":
                target_class = node
                break
        self.assertIsNotNone(target_class)

        forward_fn = next(
            node for node in target_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        backward_fn = next(
            node for node in target_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "backward"
        )
        forward_args = [arg.arg for arg in forward_fn.args.args]
        self.assertEqual(forward_args[2:6], [
            "segment_size_x",
            "segment_size_y",
            "segment_is_horizontal",
            "segment_weight",
        ])

        return_nodes = [node for node in ast.walk(backward_fn) if isinstance(node, ast.Return)]
        self.assertTrue(
            any(
                isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Add)
                and isinstance(node.value.right, ast.BinOp)
                and isinstance(node.value.right.op, ast.Mult)
                and isinstance(node.value.right.left, ast.Tuple)
                and len(node.value.right.left.elts) == 1
                and isinstance(node.value.right.left.elts[0], ast.Constant)
                and node.value.right.left.elts[0].value is None
                and isinstance(node.value.right.right, ast.Constant)
                and node.value.right.right.value == 48
                for node in return_nodes
            )
        )

    def test_boolean_mask_compaction_indices_are_unique(self):
        newx, newy, flat_from, flat_to, l_directions = _make_inputs()
        op = LShapeSegmentOp(wire_width=0.2, deterministic_backward=True)
        topo = op._compute_topology(flat_from, flat_to, l_directions, len(newx), newx.device)

        valid_from = topo["valid_from"]
        valid_to = topo["valid_to"]
        valid_l_dir = l_directions[topo["valid_mask"]]
        is_h_first = (valid_l_dir == H_FIRST) | (valid_l_dir == MODULE.FAKE_STRAIGHT)
        is_v_first = valid_l_dir == V_FIRST
        is_straight_by_dir = valid_l_dir == STRAIGHT

        x1 = newx[valid_from]
        y1 = newy[valid_from]
        x2 = newx[valid_to]
        y2 = newy[valid_to]
        is_horizontal_line = torch.abs(y1 - y2) < 1e-4
        is_vertical_line = torch.abs(x1 - x2) < 1e-4
        diag_straight = is_straight_by_dir & ~(is_horizontal_line | is_vertical_line)
        is_straight = (is_horizontal_line | is_vertical_line) | (is_straight_by_dir & ~diag_straight)
        is_upper_l = (~is_straight) & (is_h_first | diag_straight)
        is_lower_l = (~is_straight) & is_v_first
        corner_x = torch.where(is_upper_l, x2, torch.where(is_lower_l, x1, x1))
        corner_y = torch.where(is_upper_l, y1, torch.where(is_lower_l, y2, y1))

        seg1_x2 = torch.where(is_straight, x2, corner_x)
        seg1_y2 = torch.where(is_straight, y2, corner_y)
        seg1_min_x = torch.minimum(x1, seg1_x2)
        seg1_max_x = torch.maximum(x1, seg1_x2)
        seg1_min_y = torch.minimum(y1, seg1_y2)
        seg1_max_y = torch.maximum(y1, seg1_y2)

        seg2_min_x = torch.minimum(corner_x, x2)
        seg2_max_x = torch.maximum(corner_x, x2)
        seg2_min_y = torch.minimum(corner_y, y2)
        seg2_max_y = torch.maximum(corner_y, y2)

        half_width = 0.1
        seg1_size_x = (seg1_max_x - seg1_min_x) + 2 * half_width
        seg1_size_y = (seg1_max_y - seg1_min_y) + 2 * half_width
        seg2_size_x = (seg2_max_x - seg2_min_x) + 2 * half_width
        seg2_size_y = (seg2_max_y - seg2_min_y) + 2 * half_width
        is_l_shape = is_upper_l | is_lower_l
        seg1_valid = is_straight | is_l_shape
        seg2_valid = is_l_shape & (seg2_size_x > 1e-6) & (seg2_size_y > 1e-6)
        min_size = 0.2
        seg1_final_valid = seg1_valid & ((seg1_size_x > min_size) | (seg1_size_y > min_size))
        seg2_final_valid = seg2_valid & ((seg2_size_x > min_size) | (seg2_size_y > min_size))

        for indices in (
            torch.nonzero(seg1_final_valid, as_tuple=False).flatten(),
            torch.nonzero(seg2_final_valid, as_tuple=False).flatten(),
        ):
            self.assertEqual(indices.numel(), torch.unique(indices).numel())

    def test_active_position_gradients_accumulate_like_reference(self):
        result, newx, newy = _run_op(wire_width=0.2)

        upstream_llx = torch.tensor([0.5, -1.0, 2.0, 3.0, -0.25], dtype=torch.float32)
        upstream_lly = torch.tensor([-0.75, 1.25, -1.5, 0.5, 2.5], dtype=torch.float32)
        cost = (
            result["segment_llx"].mul(upstream_llx).sum()
            + result["segment_lly"].mul(upstream_lly).sum()
        )
        cost.backward()

        expected_newx_grad = torch.tensor([0.5, 0.0, -1.0, 3.0, 1.75, 0.0, 0.0, 0.0])
        expected_newy_grad = torch.tensor([-0.375, -0.375, 1.75, 0.0, -1.5, 2.5, 0.0, 0.0])

        self.assertTrue(torch.allclose(newx.grad, expected_newx_grad, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(newy.grad, expected_newy_grad, atol=1e-6, rtol=0))

    def test_optimized_path_matches_reference_forward_and_active_vjp(self):
        for kwargs in (
            {"wire_width": 0.2},
            {"wire_width": 0.0, "wire_width_h": 0.4, "wire_width_v": 0.6},
        ):
            optimized, opt_newx, opt_newy = _run_op(**kwargs)
            reference, ref_newx, ref_newy = _run_reference_hard_compaction(**kwargs)

            self.assertEqual(optimized["num_segments"], reference["num_segments"])
            for key in ("segment_llx", "segment_lly", "segment_size_x", "segment_size_y", "segment_weight"):
                self.assertTrue(
                    torch.allclose(optimized[key], reference[key].detach(), atol=1e-7, rtol=0),
                    msg=f"{key} mismatch for {kwargs}",
                )
            for key in ("segment_edge_idx", "segment_is_horizontal"):
                self.assertTrue(torch.equal(optimized[key], reference[key]), msg=f"{key} mismatch for {kwargs}")

            upstream_llx = torch.linspace(-0.75, 0.85, optimized["num_segments"])
            upstream_lly = torch.linspace(0.45, -0.65, optimized["num_segments"])
            opt_cost = (
                optimized["segment_llx"].mul(upstream_llx).sum()
                + optimized["segment_lly"].mul(upstream_lly).sum()
            )
            ref_cost = (
                reference["segment_llx"].mul(upstream_llx).sum()
                + reference["segment_lly"].mul(upstream_lly).sum()
            )
            opt_cost.backward()
            ref_cost.backward()

            self.assertTrue(torch.allclose(opt_newx.grad, ref_newx.grad, atol=1e-6, rtol=0), msg=str(kwargs))
            self.assertTrue(torch.allclose(opt_newy.grad, ref_newy.grad, atol=1e-6, rtol=0), msg=str(kwargs))

    def test_internal_env_reference_route_matches_optimized_forward_and_vjp(self):
        with mock.patch.dict(os.environ, {"DREAMPLACE_L_SHAPE_HARD_SEGMENT_COMPACTION_REFERENCE": "1"}):
            reference_env, ref_newx, ref_newy = _run_op(wire_width=0.2)
        optimized, opt_newx, opt_newy = _run_op(wire_width=0.2)

        self.assertNotIn("HardSegmentPositionCompaction", type(reference_env["segment_llx"].grad_fn).__name__)
        self.assertTrue(reference_env["segment_size_x"].requires_grad)
        self.assertFalse(optimized["segment_size_x"].requires_grad)

        for key in ("segment_llx", "segment_lly", "segment_size_x", "segment_size_y", "segment_weight"):
            self.assertTrue(torch.allclose(optimized[key], reference_env[key].detach(), atol=1e-7, rtol=0), msg=key)
        for key in ("segment_edge_idx", "segment_is_horizontal"):
            self.assertTrue(torch.equal(optimized[key], reference_env[key]), msg=key)

        upstream_llx = torch.linspace(-0.3, 0.7, optimized["num_segments"])
        upstream_lly = torch.linspace(0.8, -0.2, optimized["num_segments"])
        opt_cost = (
            optimized["segment_llx"].mul(upstream_llx).sum()
            + optimized["segment_lly"].mul(upstream_lly).sum()
        )
        ref_cost = (
            reference_env["segment_llx"].mul(upstream_llx).sum()
            + reference_env["segment_lly"].mul(upstream_lly).sum()
        )
        opt_cost.backward()
        ref_cost.backward()

        self.assertTrue(torch.allclose(opt_newx.grad, ref_newx.grad, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(opt_newy.grad, ref_newy.grad, atol=1e-6, rtol=0))

    def test_non_deterministic_setting_matches_reference_forward_and_vjp(self):
        with mock.patch.dict(os.environ, {"DREAMPLACE_L_SHAPE_HARD_SEGMENT_COMPACTION_REFERENCE": "1"}):
            reference_env, ref_newx, ref_newy = _run_op(
                wire_width=0.2,
                deterministic_backward=False,
            )
        optimized, opt_newx, opt_newy = _run_op(
            wire_width=0.2,
            deterministic_backward=False,
        )

        for key in ("segment_llx", "segment_lly", "segment_size_x", "segment_size_y", "segment_weight"):
            self.assertTrue(torch.allclose(optimized[key], reference_env[key].detach(), atol=1e-7, rtol=0), msg=key)
        for key in ("segment_edge_idx", "segment_is_horizontal"):
            self.assertTrue(torch.equal(optimized[key], reference_env[key]), msg=key)

        upstream_llx = torch.tensor([0.25, -0.5, 0.75, 1.25, -1.5], dtype=torch.float32)
        upstream_lly = torch.tensor([-1.0, 0.4, 0.9, -0.2, 1.1], dtype=torch.float32)
        opt_cost = (
            optimized["segment_llx"].mul(upstream_llx).sum()
            + optimized["segment_lly"].mul(upstream_lly).sum()
        )
        ref_cost = (
            reference_env["segment_llx"].mul(upstream_llx).sum()
            + reference_env["segment_lly"].mul(upstream_lly).sum()
        )
        opt_cost.backward()
        ref_cost.backward()

        self.assertTrue(torch.allclose(opt_newx.grad, ref_newx.grad, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(opt_newy.grad, ref_newy.grad, atol=1e-6, rtol=0))

    def test_snapshot_replay_preserves_forward_and_vjp(self):
        import tempfile

        newx, newy, flat_from, flat_to, l_directions = _make_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = os.path.join(tmpdir, "snapshot.pt")
            MODULE.save_hard_segment_snapshot(
                snapshot_path,
                newx,
                newy,
                flat_from,
                flat_to,
                l_directions,
                wire_width=0.2,
                wire_width_h=None,
                wire_width_v=None,
                deterministic_backward=True,
            )

            stats = MODULE.replay_hard_segment_snapshot_vjp(snapshot_path)

        self.assertLessEqual(stats["forward_max_abs"], 1e-7)
        self.assertLessEqual(stats["grad_newx_max_abs"], 1e-6)
        self.assertLessEqual(stats["grad_newy_max_abs"], 1e-6)
        self.assertLessEqual(stats["grad_relative_norm"], 1e-6)
        self.assertGreaterEqual(stats["grad_cosine_similarity"], 0.999999)
        self.assertEqual(stats["num_segments"], 5)

    def test_routability_env_snapshot_capture_writes_replayable_snapshot(self):
        import tempfile

        newx, newy, flat_from, flat_to, l_directions = _make_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = os.path.join(tmpdir, "realish_snapshot.pt")
            saved = MODULE.maybe_save_hard_segment_snapshot(
                snapshot_path,
                None,
                330,
                False,
                False,
                newx,
                newy,
                flat_from,
                flat_to,
                l_directions,
                wire_width=0.2,
                wire_width_h=None,
                wire_width_v=None,
                deterministic_backward=True,
            )

            self.assertTrue(saved)
            self.assertTrue(os.path.exists(snapshot_path))
            stats = MODULE.replay_hard_segment_snapshot_vjp(snapshot_path)

        self.assertEqual(stats["num_segments"], 5)
        self.assertLessEqual(stats["grad_relative_norm"], 1e-6)

    def test_deterministic_backward_repeats_exactly_for_fixed_input(self):
        def run_once():
            result, newx, newy = _run_op(wire_width=0.2)
            upstream_llx = torch.linspace(-1.0, 1.0, result["num_segments"])
            upstream_lly = torch.linspace(0.75, -0.25, result["num_segments"])
            cost = (
                result["segment_llx"].mul(upstream_llx).sum()
                + result["segment_lly"].mul(upstream_lly).sum()
            )
            cost.backward()
            return newx.grad.clone(), newy.grad.clone()

        first_x, first_y = run_once()
        second_x, second_y = run_once()

        self.assertTrue(torch.equal(first_x, second_x))
        self.assertTrue(torch.equal(first_y, second_y))


if __name__ == "__main__":
    unittest.main()
