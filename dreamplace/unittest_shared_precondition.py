import os
import sys
import types
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import torch


AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)


def _install_ieda_stubs():
    tools = types.ModuleType("tools")
    ieda = types.ModuleType("tools.iEDA")
    module = types.ModuleType("tools.iEDA.module")
    sta = types.ModuleType("tools.iEDA.module.sta")
    gpugr = types.ModuleType("tools.iEDA.module.gpugr")

    class IEDASta:
        pass

    class IEDAGPUGR:
        pass

    sta.IEDASta = IEDASta
    gpugr.IEDAGPUGR = IEDAGPUGR
    sys.modules.setdefault("tools", tools)
    sys.modules.setdefault("tools.iEDA", ieda)
    sys.modules.setdefault("tools.iEDA.module", module)
    sys.modules.setdefault("tools.iEDA.module.sta", sta)
    sys.modules.setdefault("tools.iEDA.module.gpugr", gpugr)


_install_ieda_stubs()

from dreamplace.PlaceObj import PlaceObj, PreconditionOp  # noqa: E402


def _make_precondition_op(regions=(), pin_counts=None):
    if pin_counts is None:
        pin_counts = torch.zeros(4)
    placedb = types.SimpleNamespace(
        num_nodes=4,
        num_movable_nodes=2,
        num_physical_nodes=3,
        num_filler_nodes=1,
        regions=list(regions),
        filler_start_map=[0, 0, 0, 1],
    )
    data_collections = types.SimpleNamespace(
        node_areas=torch.tensor([2.0, 4.0, 8.0, 16.0]),
        num_pins_in_nodes=pin_counts,
        pos=[torch.zeros(8)],
        node2fence_region_map=torch.tensor([0, 1, 0, 0], dtype=torch.long),
    )
    op_collections = types.SimpleNamespace()
    return PreconditionOp(placedb, data_collections, op_collections)


class SharedPreconditionTest(unittest.TestCase):
    def test_single_density_denominator_includes_raw_pin_count(self):
        op = _make_precondition_op(pin_counts=torch.tensor([3.0, 1.0, 0.0, 0.0]))

        precond = op._build_precondition(torch.tensor([2.0]))

        torch.testing.assert_close(
            precond,
            torch.tensor([7.0, 9.0, 16.0, 32.0]),
        )

    def test_multi_fence_denominator_includes_raw_pin_count(self):
        op = _make_precondition_op(
            regions=(object(), object()),
            pin_counts=torch.tensor([3.0, 1.0, 0.0, 0.0]),
        )

        precond = op._build_precondition(torch.tensor([0.5, 1.5, 2.0]))

        torch.testing.assert_close(
            precond,
            torch.tensor([5.0, 5.0, 8.0, 32.0]),
        )

    def test_apply_components_uses_one_state_step_and_one_denominator(self):
        op = _make_precondition_op()
        density_weight = torch.tensor([2.0])
        grad_a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        grad_b = grad_a * 2.0

        out_a, out_b = op.apply_components([grad_a, grad_b], density_weight)

        expected_a = torch.tensor(
            [0.25, 0.25, 0.0, 0.125, 1.25, 0.75, 0.0, 0.25]
        )
        self.assertEqual(op.iteration, 1)
        self.assertTrue(torch.allclose(out_a, expected_a))
        self.assertTrue(torch.allclose(out_b, expected_a * 2.0))
        self.assertTrue(torch.allclose(grad_a, torch.arange(1.0, 9.0)))

    def test_single_gradient_call_matches_one_component_api(self):
        density_weight = torch.tensor([2.0])
        grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        single_op = _make_precondition_op()
        component_op = _make_precondition_op()

        single = single_op(grad.clone(), density_weight)
        component = component_op.apply_components([grad], density_weight)[0]

        self.assertEqual(single_op.iteration, 1)
        self.assertEqual(component_op.iteration, 1)
        self.assertTrue(torch.allclose(single, component))

    def test_apply_components_shares_fixed_update_and_fix_node_masks(self):
        op = _make_precondition_op(regions=(object(), object()))
        density_weight = torch.tensor([2.0])
        update_mask = torch.tensor([True, False, True])
        fix_nodes_mask = torch.tensor([True, False, False, False])
        grad_a = torch.ones(8)
        grad_b = torch.ones(8) * 2.0

        out_a, out_b = op.apply_components(
            [grad_a, grad_b],
            density_weight,
            update_mask=update_mask,
            fix_nodes_mask=fix_nodes_mask,
        )

        expected_a = torch.tensor([0.0, 0.0, 0.0, 1.0 / 32.0, 0.0, 0.0, 0.0, 1.0 / 32.0])
        self.assertEqual(op.iteration, 1)
        self.assertTrue(torch.allclose(out_a, expected_a))
        self.assertTrue(torch.allclose(out_b, expected_a * 2.0))

    def test_target_weight_follows_preconditioned_norms(self):
        op = _make_precondition_op()
        density_weight = torch.tensor([2.0])
        base_raw = torch.tensor([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        l_shape_raw = torch.tensor([0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        base_grad, l_shape_grad = op.apply_components(
            [base_raw, l_shape_raw], density_weight
        )
        target_weight = PlaceObj._compute_l_shape_target_weight(
            base_grad.norm().item(), l_shape_grad.norm().item(), 0.2
        )
        raw_weight = PlaceObj._compute_l_shape_target_weight(
            base_raw.norm().item(), l_shape_raw.norm().item(), 0.2
        )

        self.assertAlmostEqual(target_weight, 0.4)
        self.assertAlmostEqual(raw_weight, 0.2)
        self.assertNotAlmostEqual(target_weight, raw_weight)


if __name__ == "__main__":
    unittest.main()
