"""Sizing snapshots preserve the geometry used to score a phase best state."""

from types import SimpleNamespace

import torch
from dreamplace.flows.discrete_sizing_state import SizingState
from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState


def test_sizing_restore_preserves_imported_pin_geometry_and_native_view():
    data = SimpleNamespace(
        real_size=torch.nn.Parameter(torch.tensor([1.0])),
        vt_logits=torch.nn.Parameter(torch.tensor([[0.0]])),
        inst_cell_id=torch.tensor([0]),
        pin_offset_x=torch.tensor([0.7]),
        pin_offset_y=torch.tensor([0.8]),
    )
    data.get_continuous_size_parameter = lambda: data.real_size
    db = SimpleNamespace(
        pin_offset_x=data.pin_offset_x.numpy().copy(),
        pin_offset_y=data.pin_offset_y.numpy().copy(),
    )
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    owner.data, owner.placedb = data, db
    owner.oscillation = CellOscillationState()
    owner.refresh_candidates_after_sizing = False
    best = SizingState.capture(data, owner.oscillation)

    # Runtime master sync reconstructs LEF pin centers; they may differ from
    # the imported DEF geometry that was used to evaluate the initial state.
    def sync(model, summary, geometry):
        data.inst_cell_id[:] = torch.tensor(summary["applied_cell_ids"])
        data.pin_offset_x[:] = 0.25
        data.pin_offset_y[:] = 0.5
        data.original_pin_offset_x = data.pin_offset_x.clone()
        data.original_pin_offset_y = data.pin_offset_y.clone()

    owner._sync = sync
    with torch.no_grad():
        data.inst_cell_id[:] = 1
        data.real_size[:] = 2.0
        data.vt_logits[:] = 3.0
    model = SimpleNamespace(_timing_geometry_cache=object())
    owner.restore_sizing_state(model, best, object())

    assert {
        "masters": data.inst_cell_id.tolist(),
        "size": data.real_size.tolist(),
        "vt": data.vt_logits.tolist(),
        "pin_x": data.pin_offset_x.tolist(),
        "pin_y": data.pin_offset_y.tolist(),
        "db_x": db.pin_offset_x.tolist(),
        "db_y": db.pin_offset_y.tolist(),
    } == {
        "masters": best.masters.tolist(),
        "size": best.size.tolist(),
        "vt": best.vt.tolist(),
        "pin_x": best.pin_geometry["pin_offset_x"].tolist(),
        "pin_y": best.pin_geometry["pin_offset_y"].tolist(),
        "db_x": best.pin_geometry["pin_offset_x"].tolist(),
        "db_y": best.pin_geometry["pin_offset_y"].tolist(),
    }
    assert not hasattr(data, "original_pin_offset_x")
    assert not hasattr(data, "original_pin_offset_y")
    assert model._timing_geometry_cache is None
