import torch


def _abs_diff(left, right):
    return torch.abs(torch.as_tensor(left) - torch.as_tensor(right))


def compare_transfer_results(reference, candidate):
    delay_diff = _abs_diff(reference.segment_delay, candidate.segment_delay)
    slew_diff = _abs_diff(reference.output_slew, candidate.output_slew)
    load_diff = _abs_diff(reference.upstream_visible_load, candidate.upstream_visible_load)
    return {
        "delay_abs_diff": delay_diff,
        "output_slew_abs_diff": slew_diff,
        "upstream_visible_load_abs_diff": load_diff,
        "max_abs_diff": torch.maximum(torch.maximum(delay_diff, slew_diff), load_diff),
    }
