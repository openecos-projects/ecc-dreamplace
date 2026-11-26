"""Reconcile electrical timing drivers with the router's LEF-only directions."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from dreamplace.ops.gr_parasitics.rc_parameters import RCParameters
from dreamplace.ops.gr_parasitics.route_snapshot import PinMapping, RouteIdentity, prepare_snapshot


def test_inout_output_pad_uses_timing_driver_and_keeps_its_rc(native_output_pad_route):
    route, lef = native_output_pad_route
    assert dict(zip(route["pin_names"], route["pin_is_driver"], strict=True)) == {
        "OUT": 0,
        "U1:PAD": 0,
    }
    mapping = PinMapping(
        {"OUT": 0, "U1:PAD": 1},
        {"N1": 0},
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([True]),
    )
    rc = RCParameters.from_lefs([lef], route["layer_names"])
    identity = RouteIdentity("output-pad", "native-route", 1)
    tree = prepare_snapshot(route, rc, mapping, identity, dtype=torch.float64)
    assert tree.root_vertex.tolist() == [tree.pin_to_vertex[1].item()]
    assert tree.valid_pin_ids.tolist() == [0, 1]
    assert tree.filtered_net_count == 0
    assert tree.wire_cap.sum().item() > 0

    # The same physical route with an explicit PAD driver is the reference.
    flags = np.zeros_like(route["pin_is_driver"])
    flags[route["pin_names"].index("U1:PAD")] = 1
    explicit = dict(route, pin_is_driver=flags)
    expected = prepare_snapshot(explicit, rc, mapping, identity, dtype=torch.float64)
    torch.testing.assert_close(
        {key: value for key, value in vars(tree).items() if isinstance(value, torch.Tensor)},
        {key: value for key, value in vars(expected).items() if isinstance(value, torch.Tensor)},
        rtol=0,
        atol=0,
    )
    assert tree.model_report == expected.model_report


@pytest.mark.parametrize("physical_drivers", [0, 2])
def test_no_timing_driver_requires_one_physical_driver(snapshot_inputs, physical_drivers):
    route, rc, mapping, identity = snapshot_inputs
    flags = np.zeros_like(route["pin_is_driver"])
    flags[:physical_drivers] = 1
    route = dict(route, pin_is_driver=flags)
    mapping = replace(mapping, driver_pin=torch.tensor([-1], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="requires exactly one actual driver"):
        prepare_snapshot(route, rc, mapping, identity)


def test_timing_driver_does_not_hide_an_extra_physical_driver(snapshot_inputs):
    route, rc, mapping, identity = snapshot_inputs
    flags = np.ones_like(route["pin_is_driver"])
    with pytest.raises(RuntimeError, match="native/PyDB driver mismatch"):
        prepare_snapshot(dict(route, pin_is_driver=flags), rc, mapping, identity)
