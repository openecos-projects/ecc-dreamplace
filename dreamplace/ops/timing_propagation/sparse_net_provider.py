"""Lift the existing buffer transfer into launch states without rerunning trees.

The non-inverting buffer providers are translation invariant in arrival time.
Evaluate their electrical tree once with zero driver arrival, then add each
launch state's own arrival. Slew and load retain physical ownership.
"""

import torch

from .hard_scatter_native import scatter_slew_max
from .net_aat_native import _NetAat


class SparseNetProvider:
    def __init__(self, timing, schedule):
        self.timing = timing
        self.levels = {
            id(level): (
                index,
                timing.pin_net[level.pins[level.net[:, 1]]],
            )
            for index, level in enumerate(schedule.levels)
        }
        self.provider = None

    def bind(self, provider):
        if provider is not None and not getattr(
            provider, "supports_sparse_timing", False
        ):
            raise RuntimeError(
                f"{type(provider).__name__} has no sparse timing capability"
            )
        self.provider = provider

    def __call__(self, level, lanes, delays, impulses):
        output = _NetAat.apply(
            level.net,
            *lanes,
            *(value[level.pins] for value in (*delays, *impulses)),
            1e-8,
        )
        provider = self.provider
        if provider is None:
            return output
        index, row_net_ids = self.levels[id(level)]
        net_ids = torch.unique(row_net_ids)
        epoch = self.timing.critical_path_topology_epoch
        if not provider.has_dynamic_nets(
            net_ids, level_id=index, level_view_epoch=epoch
        ):
            return output
        drivers = level.pins[level.net[:, 0]]
        sinks = level.pins[level.net[:, 1]]
        zeros = lanes[0].new_zeros(self.timing.num_pins)
        driver_slew = tuple(
            scatter_slew_max(zeros, drivers, value[level.net[:, 0]])
            for value in lanes[2:]
        )

        transfer = provider.propagate_net_aat_level(
            net_ids=net_ids,
            pin_rAAT=zeros,
            pin_fAAT=zeros,
            pin_rtran=driver_slew[0],
            pin_ftran=driver_slew[1],
            base_pin_net_delay_rise=delays[0],
            base_pin_net_delay_fall=delays[1],
            base_pin_net_impulse_rise=impulses[0],
            base_pin_net_impulse_fall=impulses[1],
            static_calculate_net_aat_level=self.timing.calculate_net_aat_level,
            level_id=index,
            level_view_epoch=epoch,
        )
        active = torch.isin(
            row_net_ids,
            row_net_ids.new_tensor(sorted(provider.affected_net_ids)),
        )
        return tuple(
            torch.where(
                active,
                transfer[lane][sinks]
                + (lanes[lane][level.net[:, 0]] if lane < 2 else 0),
                value,
            )
            for lane, value in enumerate(output)
        )
