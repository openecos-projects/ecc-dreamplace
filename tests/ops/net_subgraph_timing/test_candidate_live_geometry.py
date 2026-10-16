import pytest
import torch

from dreamplace.ops.net_subgraph_timing.candidate_live_geometry import (
    CandidateLiveGeometry,
)


def _payload():
    return {
        "metadata": {"compact_node_to_original": [0, -1, 1]},
        "pin_fa": torch.tensor([-1, 0, 1], dtype=torch.long),
        "sink_node_id": torch.tensor([2], dtype=torch.long),
        "sink_pin_id": torch.tensor([1], dtype=torch.long),
        "node_capacitance": torch.zeros(3),
        "candidate_node_id": torch.tensor([1], dtype=torch.long),
    }


def test_candidate_live_geometry_resolves_synthetic_node_from_segment_endpoints():
    state = type(
        "State",
        (),
        {
            "candidate_records": [
                {
                    "parent_node_id": 0,
                    "child_node_ids": [1],
                    "segment_split_ratio": 0.5,
                }
            ]
        },
    )()
    geometry = CandidateLiveGeometry(_payload(), state)

    geometry.bind(
        new_x=torch.tensor([0.0, 10.0]),
        new_y=torch.tensor([0.0, 0.0]),
        pin_capacitance_by_sense={"base": torch.tensor([0.0, 0.2])},
        r_unit=1.0,
        c_unit=1.0,
        scale_factor=1.0,
        dbu=1.0,
    )

    assert geometry.rc["edge_resistance"].shape == (3,)
    assert torch.allclose(geometry.rc["edge_resistance"], torch.tensor([0.0, 5.0, 5.0]))


def test_candidate_live_geometry_rejects_stale_positive_node_id():
    payload = _payload()
    payload["metadata"] = {"compact_node_to_original": [0, 2, 1]}
    state = type("State", (), {"candidate_records": []})()
    geometry = CandidateLiveGeometry(payload, state)

    with pytest.raises(IndexError, match="outside live coordinate domain"):
        geometry.bind(
            new_x=torch.tensor([0.0, 10.0]),
            new_y=torch.tensor([0.0, 0.0]),
            pin_capacitance_by_sense={"base": torch.tensor([0.0, 0.2])},
            r_unit=1.0,
            c_unit=1.0,
            scale_factor=1.0,
            dbu=1.0,
        )


def test_discrete_probe_freezes_geometry_graph_without_changing_rc_values():
    state = type("State", (), {"candidate_records": []})()
    payload = _payload()
    payload["metadata"]["compact_node_to_original"] = [0, 1, 2]
    geometry = CandidateLiveGeometry(payload, state)
    x = torch.tensor([0.0, 5.0, 10.0], requires_grad=True)
    caps = torch.tensor([0.0, 0.2, 0.4], requires_grad=True)
    kwargs = {
        "new_x": x,
        "new_y": torch.zeros_like(x),
        "pin_capacitance_by_sense": {"base": caps},
        "r_unit": 1.0,
        "c_unit": 1.0,
        "scale_factor": 1.0,
        "dbu": 1.0,
    }
    geometry.bind(**kwargs)
    gp_values = geometry.forward_overrides(
        active_view=None,
        device=x.device,
        dtype=x.dtype,
        sense="base",
    )
    assert all(value.requires_grad for value in gp_values.values())
    state._gradient_drive_type = 0
    geometry.bind(**kwargs)
    probe_values = geometry.forward_overrides(
        active_view=None,
        device=x.device,
        dtype=x.dtype,
        sense="base",
    )
    for name, value in probe_values.items():
        assert not value.requires_grad
        torch.testing.assert_close(value, gp_values[name])
