"""Frozen native hold/action replay with live pin loads and zero wire RC.

The shared action fixture owns master metadata and setup probing. This class
adds the min corner, input loads and early timing model used by hold tests.
"""

import torch
from dreamplace.flows.hold_rc import HoldRC
from dreamplace.ops.placeio_ecc.hold_schema import read_hold_timing_schema
from dreamplace.ops.timing_propagation.early_propagation import EarlyPropagation
from native_action_case import NativeActionCase


class NativeHoldCase(NativeActionCase):
    def __init__(self, pydb, view, base, *, device="cpu"):
        super().__init__(pydb, view, base, device=device)
        self.hold = read_hold_timing_schema(pydb)
        self.rc = HoldRC(self.hold)
        self.early = EarlyPropagation(self.hold)
        self.sinks = torch.unique(self.timing.net_flat_arcs[:, 1])

    def loads(self):
        _, *caps = self.rc.pin_caps(self.model, self.zeros)
        nets = self.data.pin2net_map
        return tuple(
            cap.new_zeros(self.data.net2driver_pin_map.numel()).scatter_add(
                0, nets[self.sinks], cap[self.sinks]
            )[nets]
            for cap in caps
        )

    def forward(self, beta):
        self.context.beta = beta
        return self.early.forward(
            self.timing, (self.zeros, self.zeros), (self.zeros, self.zeros), self.loads()
        )
