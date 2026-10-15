from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentTransferDatasetSchema:
    feature_names: tuple[str, ...] = (
        "input_arrival",
        "input_slew",
        "downstream_load",
        "edge_resistance",
        "edge_capacitance",
        "upstream_retained_capacitance",
        "repeater_count",
        "bsu_index",
    )
    label_names: tuple[str, ...] = (
        "upstream_visible_load",
        "segment_delay",
        "output_arrival",
        "output_slew",
    )
