import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer.analytic import (
    analytic_segment_transfer,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.schema import (
    BufferDeviceLut,
    SegmentTransferInput,
)


def default_fake_buffer_device(dtype=torch.float64):
    input_cap_by_size = torch.tensor([0.4, 0.8], dtype=dtype)
    input_slew_axis = torch.tensor([0.0, 10.0, 20.0], dtype=dtype)
    output_load_axis = torch.tensor([0.0, 5.0, 10.0], dtype=dtype)
    delay = torch.empty(2, 3, 3, dtype=dtype)
    output_slew = torch.empty(2, 3, 3, dtype=dtype)
    for size in range(2):
        for slew in range(3):
            for load in range(3):
                delay[size, slew, load] = 1.0 + size + 0.1 * slew + 0.2 * load
                output_slew[size, slew, load] = 0.5 + 0.3 * size + 0.2 * slew + 0.1 * load
    return BufferDeviceLut(
        input_cap_by_size=input_cap_by_size,
        input_slew_axis=input_slew_axis,
        output_load_axis=output_load_axis,
        delay_lut=delay,
        output_slew_lut=output_slew,
        source="deterministic_fake_lut",
    )


def build_deterministic_segment_transfer_dataset(dtype=torch.float64):
    device = default_fake_buffer_device(dtype=dtype)
    rows = []
    for repeater_count in (0, 1):
        for input_slew in (1.0, 6.0):
            transfer_input = SegmentTransferInput(
                input_arrival=0.0,
                input_slew=torch.tensor(input_slew, dtype=dtype),
                downstream_load=torch.tensor(3.0, dtype=dtype),
                edge_resistance=torch.tensor(2.0, dtype=dtype),
                edge_capacitance=torch.tensor(1.5, dtype=dtype),
                repeater_count=repeater_count,
                bsu_index=torch.tensor(0.5, dtype=dtype),
                buffer_device=device if repeater_count else None,
            )
            result = analytic_segment_transfer(transfer_input)
            rows.append(
                {
                    "input": transfer_input,
                    "label": {
                        "upstream_visible_load": result.upstream_visible_load.detach().clone(),
                        "segment_delay": result.segment_delay.detach().clone(),
                        "output_arrival": result.output_arrival.detach().clone(),
                        "output_slew": result.output_slew.detach().clone(),
                    },
                }
            )
    return rows
