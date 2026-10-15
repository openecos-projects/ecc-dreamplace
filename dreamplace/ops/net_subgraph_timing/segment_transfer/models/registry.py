from .residual_mlp import ResidualSegmentTransferMLP


def build_segment_transfer_model(name, **kwargs):
    if name == "residual_mlp":
        return ResidualSegmentTransferMLP(**kwargs)
    raise ValueError(f"unknown segment transfer model: {name}")
