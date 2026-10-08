def build_segment_transfer_feature_row(
    transfer_input,
    result=None,
    *,
    label=None,
):
    row = {
        "input_arrival": transfer_input.input_arrival,
        "input_slew": transfer_input.input_slew,
        "downstream_load": transfer_input.downstream_load,
        "edge_resistance": transfer_input.edge_resistance,
        "edge_capacitance": transfer_input.edge_capacitance,
        "upstream_retained_capacitance": transfer_input.upstream_retained_capacitance,
        "repeater_count": int(transfer_input.repeater_count),
        "bsu_index": transfer_input.bsu_index,
        "split_fractions": transfer_input.split_fractions,
    }
    if result is not None:
        row.update(
            {
                "upstream_visible_load": result.upstream_visible_load,
                "segment_delay": result.segment_delay,
                "output_slew": result.output_slew,
                "output_arrival": result.output_arrival,
            }
        )
    if label is not None:
        row["label"] = label
    return row
