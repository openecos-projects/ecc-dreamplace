from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dreamplace.flows.hold_objective import HoldObjective
from dreamplace.flows.hold_rc import HoldRC


class TestModel:
    def _current_timing_surrogate_mode(self):
        return getattr(self.params, "timing_surrogate_mode", None)

    def pin_2_libpin_ids(self, inst_libcell_offset, data):
        nodes = data.pin2node_map
        offsets = data.pin_2_libpin_offset
        main = data.inst_main_id[nodes]
        mask = (main >= 0) & (offsets >= 0)
        cells = data.main_id_2_cell_id_start[main[mask]] + inst_libcell_offset[nodes[mask]].long()
        return mask, data.cell_id_2_libpin_id_start[cells] + offsets[mask]

    def _get_sizing_pin_cap_static_cache(self, data):
        cached = getattr(data, "_sizing_pin_cap_static_cache", None)
        if cached is not None:
            return cached
        nodes = data.pin2node_map
        offsets = data.pin_2_libpin_offset
        main = data.inst_main_id[nodes]
        mask = (main >= 0) & data.inst_is_sizeable[nodes] & (offsets >= 0)
        pin_nodes = nodes[mask]
        pin_main = main[mask].long()
        pin_offsets = offsets[mask].long()
        pin_ids = torch.nonzero(mask, as_tuple=False).flatten()
        groups = []
        for main_id in torch.unique(pin_main):
            local = pin_main == main_id
            cells = torch.arange(
                data.main_id_2_cell_id_start[main_id],
                data.main_id_2_cell_id_start[main_id + 1],
            )
            libpin_ids = data.cell_id_2_libpin_id_start[cells].unsqueeze(0) + pin_offsets[local].unsqueeze(1)
            groups.append({
                "main_mask": local,
                "candidate_sizes": data.flat_libcell_info[cells, 2].float(),
                "candidate_vts": data.flat_libcell_info[cells, 3].long(),
                "candidate_libpin_ids": libpin_ids,
            })
        cached = {
            "has_sizeable": bool(groups),
            "pin_nodes": pin_nodes,
            "sizeable_pin_ids": pin_ids,
            "groups": groups,
        }
        data._sizing_pin_cap_static_cache = cached
        return cached


def make_model(size=None):
    model = TestModel()
    model.params = SimpleNamespace(timing_surrogate_mode="lut_only" if size is None else "auto")
    model.data_collections = SimpleNamespace(
        pin2node_map=torch.tensor([0]),
        inst_main_id=torch.tensor([0]),
        inst_is_sizeable=torch.tensor([True]),
        inst_libcell_offset=torch.tensor([0]),
        pin_2_libpin_offset=torch.tensor([0]),
        main_id_2_cell_id_start=torch.tensor([0, 2]),
        cell_id_2_libpin_id_start=torch.tensor([0, 1, 2]),
        flat_libcell_info=torch.tensor([[0, 0, 1, 0], [0, 0, 2, 0]]),
        flat_lib_pin_cap=torch.tensor([3.0, 6.0]),
        flat_lib_pin_rcap=torch.tensor([3.0, 6.0]),
        flat_lib_pin_fcap=torch.tensor([3.0, 6.0]),
        get_size_var=lambda: size,
        get_vt_var=lambda: torch.ones((1, 1)),
        end_points=torch.empty(0, dtype=torch.long),
        outcaps=torch.empty(0),
    )
    model._timing_rc_cap_input_tensor = lambda cap: cap
    return model


def test_min_capacitance_tracks_current_master_without_rebuilding_late_tables():
    model = make_model()
    rc = HoldRC(SimpleNamespace(
        min_pin_caps=(np.array([1.0, 4.0]), np.array([0.5, 2.0])),
        output_caps=(np.zeros(0), np.zeros(0)),
    ))
    base = torch.tensor([3.0])
    first = rc.pin_caps(model, base)
    model.data_collections.inst_libcell_offset[0] = 1
    second = rc.pin_caps(model, base)
    torch.testing.assert_close(torch.stack(first), torch.tensor([[3.0], [1.0], [0.5]]))
    torch.testing.assert_close(torch.stack(second), torch.tensor([[3.0], [4.0], [2.0]]))
    torch.testing.assert_close(model.data_collections.flat_lib_pin_rcap, torch.tensor([3.0, 6.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_min_capacitance_moves_static_indices_to_cuda_timing_device():
    model = make_model()
    rc = HoldRC(SimpleNamespace(
        min_pin_caps=(np.array([1.0, 4.0]), np.array([0.5, 2.0])),
        output_caps=(np.zeros(0), np.zeros(0)),
    ))
    base, rise, fall = rc.pin_caps(model, torch.tensor([3.0], device="cuda"))
    assert base.device.type == rise.device.type == fall.device.type == "cuda"
    torch.testing.assert_close(
        torch.stack((base, rise, fall)).cpu(),
        torch.tensor([[3.0], [1.0], [0.5]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sizing_min_capacitance_moves_cached_indices_to_cuda_and_keeps_gradient():
    size = torch.tensor([1.5], device="cuda", requires_grad=True)
    model = make_model(size)
    rc = HoldRC(SimpleNamespace(
        min_pin_caps=(np.array([1.0, 4.0]), np.array([0.5, 2.0])),
        output_caps=(np.zeros(0), np.zeros(0)),
    ))
    base, rise, fall = rc.pin_caps(
        model, torch.tensor([3.0], device="cuda")
    )
    assert base.device.type == rise.device.type == fall.device.type == "cuda"
    torch.testing.assert_close(
        torch.stack((base, rise, fall)).cpu(),
        torch.tensor([[3.0], [2.5], [1.25]]),
    )
    (gradient,) = torch.autograd.grad(rise.sum(), size)
    torch.testing.assert_close(gradient.cpu(), torch.tensor([6.0]))


def test_min_capacitance_has_size_gradient_and_preserves_late_candidate_cache():
    size = torch.tensor([1.5], requires_grad=True)
    model = make_model(size)
    cache = model._get_sizing_pin_cap_static_cache(model.data_collections)
    rc = HoldRC(SimpleNamespace(
        min_pin_caps=(np.array([1.0, 4.0]), np.array([0.5, 2.0])),
        output_caps=(np.zeros(0), np.zeros(0)),
    ))
    base, rise, fall = rc.pin_caps(model, torch.tensor([3.0]))
    torch.testing.assert_close(
        torch.stack((base, rise, fall)), torch.tensor([[3.0], [2.5], [1.25]])
    )
    (gradient,) = torch.autograd.grad(rise.sum(), size)
    torch.testing.assert_close(gradient, torch.tensor([6.0]))
    assert model._get_sizing_pin_cap_static_cache(model.data_collections) is cache


def test_report_only_rc_keeps_values_and_reenables_gradients_for_active_phase():
    size = torch.tensor([1.5], requires_grad=True)
    model = make_model(size)
    model.placedb = SimpleNamespace(gr_sizing=object())
    model._timing_rc_cap_input_tensor = lambda cap: cap

    def rc_operator(base, rise, fall):
        values = {"rise": rise * 2, "fall": fall * 3}
        return None, values, values, None, None, values

    model.op_collections = SimpleNamespace(elmore_delay_op=rc_operator)
    rc = HoldRC(SimpleNamespace(
        min_pin_caps=(np.array([1., 4.]), np.array([.5, 2.])),
        output_caps=(np.zeros(0), np.zeros(0)),
    ))
    model.hold_objective = HoldObjective(0, 0, 0)
    zero = torch.zeros(1)
    report = rc.evaluate(model, zero, zero, zero)
    assert not report.gradients_enabled
    assert not report.delays["rise"].requires_grad
    model.hold_objective = HoldObjective(1, .01, 0)
    active = rc.evaluate(model, zero, zero, zero)
    assert active.gradients_enabled
    torch.testing.assert_close(active.delays, report.delays)
    gradient, = torch.autograd.grad(active.delays["rise"].sum(), size)
    torch.testing.assert_close(gradient, torch.tensor([12.]))


@pytest.mark.parametrize("sizeable", [True, False])
def test_min_load_preserves_sdc_output_load_on_non_library_ports(sizeable):
    model = make_model(torch.tensor([1.5], requires_grad=True))
    data = model.data_collections
    data.inst_is_sizeable.fill_(sizeable)
    data.pin2node_map = torch.tensor([0, 0])
    data.pin_2_libpin_offset = torch.tensor([0, -1])
    data.end_points = torch.tensor([1])
    data.outcaps = torch.tensor([.75])
    rc = HoldRC(SimpleNamespace(
        min_pin_caps=(np.array([1., 4.]), np.array([.5, 2.])),
        output_caps=(np.array([.25]), np.array([.125])),
    ))
    base, rise, fall = rc.pin_caps(model, torch.tensor([3., .75]))
    expected = [[3., .75], [2.5, .25], [1.25, .125]] if sizeable else [
        [3., .75], [1., .25], [.5, .125],
    ]
    torch.testing.assert_close(torch.stack((base, rise, fall)), torch.tensor(expected))
    if sizeable:
        gradient, = torch.autograd.grad(rise.sum() + fall.sum(), data.get_size_var())
        torch.testing.assert_close(gradient, torch.tensor([9.]))
