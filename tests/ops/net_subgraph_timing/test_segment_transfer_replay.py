import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    replay_local_segment_transfer,
    replay_probe_row_segment_transfer,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


def test_local_replay_reports_n0_n1_delta_and_ac10_diagnostics():
    device = default_fake_buffer_device()
    summary = replay_local_segment_transfer(
        base_input=SegmentTransferInput(
            input_arrival=torch.tensor(0.0, dtype=torch.float64),
            input_slew=torch.tensor(4.0, dtype=torch.float64),
            downstream_load=torch.tensor(3.0, dtype=torch.float64),
            edge_resistance=torch.tensor(2.0, dtype=torch.float64),
            edge_capacitance=torch.tensor(1.0, dtype=torch.float64),
            repeater_count=0,
        ),
        buffer_device=device,
        bsu_index=torch.tensor(0.5, dtype=torch.float64),
        split_fractions=(0.25, 0.75),
        upstream_retained_capacitance=torch.tensor(0.25, dtype=torch.float64),
    )

    assert summary["status"] == "local_segment_transfer_replay"
    assert set(summary) >= {"n0", "n1", "delta_n1_minus_n0"}
    for key in (
        "upstream_wire_delay",
        "downstream_wire_delay",
        "buffer_input_slew",
        "buffer_output_load",
        "buffer_input_cap",
        "buffer_delay",
        "buffer_output_slew",
        "upstream_visible_load",
    ):
        if key == "upstream_visible_load":
            assert key in summary["n1"]
        else:
            assert key in summary["n1"]["diagnostics"]
    assert summary["n1"]["diagnostics"]["upstream_retained_capacitance"] == 0.25
    assert summary["n1"]["diagnostics"]["split_fractions"] == [0.25, 0.75]
    assert summary["delta_n1_minus_n0"]["upstream_visible_load"] != 0.0


def test_probe_row_replay_uses_probe_split_and_identity_metadata():
    device = default_fake_buffer_device()
    probe_row = {
        "global_segment_id": 27429,
        "net_id": 123,
        "net_name": "n_24317",
        "segment_parent_arrival": 10.0,
        "segment_parent_slew": 4.0,
        "segment_downstream_load": 3.0,
        "edge_rc": {"r": 2.0, "c": 1.0},
        "fractions_by_repeater_count": {"0": [1.0], "1": [0.25, 0.75]},
        "z_value": 1.0,
        "bsu_index": 0.5,
    }

    summary = replay_probe_row_segment_transfer(
        probe_row,
        buffer_device=device,
        upstream_retained_capacitance=0.125,
    )

    assert summary["probe"]["segment_id"] == 27429
    assert summary["probe"]["net_name"] == "n_24317"
    assert summary["n1"]["diagnostics"]["split_fractions"] == [0.25, 0.75]
    assert summary["n1"]["diagnostics"]["upstream_retained_capacitance"] == 0.125
    assert summary["n1"]["output_arrival"] == (
        summary["n1"]["segment_delay"] + probe_row["segment_parent_arrival"]
    )
