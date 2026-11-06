"""Clock identities excluded from data-net buffering, without changing GP/STA."""

import torch


def decode_name(name):
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


_decode_name = decode_name


def clock_net_ids(*owners):
    """Use native net types and STA clock pins, never a net-name heuristic."""
    excluded = set()
    for owner in owners:
        names = getattr(owner, "clock_net_names", None)
        if names is not None and len(names) > 0:
            clock_names = {_decode_name(name) for name in names}
            net_names = getattr(owner, "net_names", None)
            if net_names is not None:
                excluded.update(
                    index for index, name in enumerate(net_names)
                    if _decode_name(name) in clock_names
                )
        pins = getattr(owner, "clock_pins", None)
        pin2net = getattr(owner, "pin2net_map", None)
        if pins is not None and pin2net is not None:
            pins = torch.as_tensor(pins, dtype=torch.long)
            pin2net = torch.as_tensor(pin2net, device=pins.device)
            excluded.update(int(net) for net in pin2net[pins].detach().cpu().tolist())
    return excluded
