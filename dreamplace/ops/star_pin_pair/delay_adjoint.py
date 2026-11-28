"""Extract star-pair delay adjoints from the timing VJP."""

from __future__ import annotations

from typing import Dict

import torch


def pair_delay_adjoint_from_pin_grads(
    sink_pin_ids: torch.Tensor,
    grad_delay_rise: torch.Tensor,
    grad_delay_fall: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Gather pin-state net-delay adjoints onto physical star pairs."""
    sinks = torch.as_tensor(sink_pin_ids).reshape(-1).long()
    rise = torch.as_tensor(grad_delay_rise).reshape(-1)
    fall = torch.as_tensor(grad_delay_fall).reshape(-1)
    if int(sinks.numel()) == 0:
        empty = rise.new_zeros(0)
        return {"rise": empty, "fall": empty, "pair": empty}
    if int(rise.numel()) != int(fall.numel()):
        raise ValueError("rise/fall delay adjoints must have the same length")
    if bool(torch.any((sinks < 0) | (sinks >= int(rise.numel()))).item()):
        raise ValueError("sink_pin_ids contains out-of-range pin ids")
    pair_rise = rise[sinks]
    pair_fall = fall[sinks]
    return {
        "rise": pair_rise,
        "fall": pair_fall,
        "pair": pair_rise + pair_fall,
    }
