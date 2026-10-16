import unittest

import torch

from dreamplace.ops.net_subgraph_timing.segment_count_live_geometry import (
    build_live_edge_geometry,
)


class SegmentCountLiveGeometryTest(unittest.TestCase):
    def test_builds_prepared_order_rc_and_coordinate_gradients(self):
        new_x = torch.tensor(
            [0.0, 10.0, 10.0, 10.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        new_y = torch.tensor(
            [0.0, 0.0, 4.0, 4.0],
            dtype=torch.float64,
            requires_grad=True,
        )
        result = build_live_edge_geometry(
            new_x,
            new_y,
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 2, 3]),
            r_unit=3.0,
            c_unit=5.0,
            scale_factor=2.0,
            dbu=1.0,
            require_axis_aligned=True,
        )

        torch.testing.assert_close(
            result["length"],
            torch.tensor([5.0, 2.0, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result["edge_resistance"],
            torch.tensor([15.0, 6.0, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result["edge_capacitance"],
            torch.tensor([25.0, 10.0, 0.0], dtype=torch.float64),
        )
        self.assertEqual(
            result["census"],
            {
                "edge_count": 3,
                "horizontal_edge_count": 1,
                "vertical_edge_count": 1,
                "diagonal_edge_count": 0,
                "zero_length_edge_count": 1,
            },
        )

        objective = result["edge_resistance"].sum() + result[
            "edge_capacitance"
        ].sum()
        grad_x, grad_y = torch.autograd.grad(objective, (new_x, new_y))
        torch.testing.assert_close(
            grad_x,
            torch.tensor([-4.0, 4.0, 0.0, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            grad_y,
            torch.tensor([0.0, -4.0, 4.0, 0.0], dtype=torch.float64),
        )

    def test_rejects_diagonal_edge_when_rectilinear_geometry_is_required(self):
        with self.assertRaisesRegex(ValueError, "diagonal edge"):
            build_live_edge_geometry(
                torch.tensor([0.0, 2.0], dtype=torch.float64),
                torch.tensor([0.0, 3.0], dtype=torch.float64),
                torch.tensor([0]),
                torch.tensor([1]),
                r_unit=1.0,
                c_unit=1.0,
                scale_factor=1.0,
                dbu=1.0,
                require_axis_aligned=True,
            )

    def test_models_diagonal_edge_as_deterministic_rectilinear_manhattan_path(self):
        new_x = torch.tensor([0.0, 2.0], dtype=torch.float64, requires_grad=True)
        new_y = torch.tensor([0.0, 3.0], dtype=torch.float64, requires_grad=True)

        result = build_live_edge_geometry(
            new_x,
            new_y,
            torch.tensor([0]),
            torch.tensor([1]),
            r_unit=2.0,
            c_unit=4.0,
            scale_factor=1.0,
            dbu=1.0,
            require_axis_aligned=False,
        )

        torch.testing.assert_close(
            result["length"], torch.tensor([5.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result["edge_resistance"], torch.tensor([10.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result["edge_capacitance"], torch.tensor([20.0], dtype=torch.float64)
        )
        self.assertEqual(result["rectilinear_path_policy"], "x_then_y")
        self.assertEqual(result["census"]["diagonal_edge_count"], 1)
        grad_x, grad_y = torch.autograd.grad(
            result["edge_resistance"].sum(),
            (new_x, new_y),
        )
        torch.testing.assert_close(
            grad_x, torch.tensor([-2.0, 2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            grad_y, torch.tensor([-2.0, 2.0], dtype=torch.float64)
        )

    def test_rejects_node_ids_outside_coordinate_domain(self):
        with self.assertRaisesRegex(IndexError, "coordinate domain"):
            build_live_edge_geometry(
                torch.tensor([0.0], dtype=torch.float64),
                torch.tensor([0.0], dtype=torch.float64),
                torch.tensor([0]),
                torch.tensor([1]),
                r_unit=1.0,
                c_unit=1.0,
                scale_factor=1.0,
                dbu=1.0,
            )


if __name__ == "__main__":
    unittest.main()
