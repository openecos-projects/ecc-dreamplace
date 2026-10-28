"""Common size/VT weights for max and min pin capacitance interpolation."""

import torch


def sizing_pin_candidate_weights(
    sizes, vt, candidate_sizes, candidate_vts, temperature=4.0
):
    """Return differentiable weights over the legal pin-cap candidates."""
    vt_scores = vt[:, candidate_vts].clamp_min(1e-9)
    size_scores = -temperature * torch.abs(
        sizes.unsqueeze(1) - candidate_sizes.unsqueeze(0)
    )
    return torch.softmax(torch.log(vt_scores) + size_scores, dim=1)
