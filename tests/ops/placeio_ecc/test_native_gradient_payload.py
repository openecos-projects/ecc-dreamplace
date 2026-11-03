"""Gradient transport refreshes physical identities before the C++ queue reads it."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from dreamplace.ops.placeio_ecc import gradient_order


def test_native_gradient_probe_materializes_current_names_and_epoch(monkeypatch):
    source = SimpleNamespace(pin_names=["a:A", "a:Y"])
    updated = SimpleNamespace(pin_names=["a:A", "new_buffer:Y"])
    owner = SimpleNamespace(active_pydb=source, r_unit=0.65, c_unit=0.000189)
    overlay = object()
    bridge = SimpleNamespace(graph=SimpleNamespace(epoch=3))
    bridge.materialize = Mock(side_effect=lambda current: setattr(owner, "active_pydb", updated))
    constructor = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr("dreamplace.ops.rebuffer.timing_model.RebufferTimingModel", constructor)
    probes = []

    def probe(model):
        assert torch.is_grad_enabled()
        probes.append(model)
        return {"status": "ok", "loss": 2.0, "driver_scores": {1: 7.5}}

    monkeypatch.setattr(gradient_order, "compute_timing_gradient_scores", probe)
    with torch.no_grad():
        result = gradient_order.native_timing_gradient_payload(owner, bridge, overlay)
        assert not torch.is_grad_enabled()
    assert result == {
        "gradient_score_epoch": 3,
        "gradient_score_by_driver_name": {"new_buffer:Y": 7.5},
        "gradient_loss": 2.0,
        "gradient_steiner_provider": "ecc",
    }
    bridge.materialize.assert_called_once_with(overlay)
    constructor.assert_called_once_with(
        updated,
        use_pin_offset=False,
        steiner_provider="ecc",
        r_unit=0.65,
        c_unit=0.000189,
    )
    bridge.graph.epoch = 4
    gradient_order.native_timing_gradient_payload(owner, bridge, overlay)
    assert len(probes) == 2


def test_fresh_gradient_probe_after_native_mutation_preserves_hard_timing_state():
    from tests.ops.placeio_ecc.test_native_timing_campaign import _action, _session

    owner, session, bridge = _session()
    session.apply([_action(session.overlay)])
    session.evaluate()
    before = {key: np.array(value, copy=True) for key, value in bridge.graph.state_view().items()}
    epoch = int(bridge.graph.epoch)
    payload = gradient_order.native_timing_gradient_payload(owner, bridge, session.overlay)
    assert payload["gradient_score_epoch"] == epoch == bridge.graph.epoch
    assert set(payload["gradient_score_by_driver_name"]) <= set(owner.active_pydb.pin_names)
    for key, value in bridge.graph.state_view().items():
        np.testing.assert_array_equal(value, before[key], err_msg=key)
