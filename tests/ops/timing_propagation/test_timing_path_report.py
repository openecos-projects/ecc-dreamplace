from types import SimpleNamespace

import pytest
import torch
from dreamplace.ops.timing_propagation.timing_path_report import (
    build_critical_path_rows,
    critical_path_summary,
    timing_index_tensor,
)


def _batch():
    return SimpleNamespace(
        path_offsets=torch.tensor([0, 4]),
        path_pins=torch.tensor([0, 1, 2, 3]),
        path_transitions=torch.tensor([1, 1, 1, 0]),
        path_arc_ids=torch.tensor([0, -1, 1]),
        endpoint_pins=torch.tensor([3]),
        endpoint_transitions=torch.tensor([0]),
        endpoint_slacks=torch.tensor([-5.0]),
        path_valid=torch.tensor([True]),
        invalid_reason=torch.tensor([0]),
        max_residual_ps=torch.tensor([0.0]),
        failing_state_count=2,
        selected_state_count=1,
        valid_path_count=1,
        invalid_path_count=0,
        topology_epoch=7,
        extraction_runtime_ms=0.25,
    )


def _timing_op():
    return SimpleNamespace(
        pin_rAAT_live_snapshot=torch.tensor([10.0, 15.0, 18.0, 42.0]),
        pin_fAAT_live_snapshot=torch.tensor([20.0, 27.0, 31.0, 31.0]),
        pin_rtran_live=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        pin_ftran_live=torch.tensor([5.0, 6.0, 7.0, 8.0]),
        pin_net_cap_rise_live=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        pin_net_cap_fall_live=torch.tensor([0.5, 0.6, 0.7, 0.8]),
        pin_net_delay_rise_live=torch.tensor([0.0, 0.0, 3.0, 0.0]),
        pin_net_delay_fall_live=torch.tensor([0.0, 0.0, 4.0, 0.0]),
        cell_arc_rr_delays=torch.tensor([5.0, 0.0]),
        cell_arc_fr_delays=torch.tensor([0.0, 11.0]),
        cell_arc_rf_delays=torch.tensor([0.0, 13.0]),
        cell_arc_ff_delays=torch.tensor([7.0, 0.0]),
    )


def test_build_critical_path_rows_reconstructs_cell_and_net_edges():
    rows = build_critical_path_rows(
        batch=_batch(),
        timing_op=_timing_op(),
        pin_names=["start", "cell0_out", "cell1_in", "endpoint"],
        flat_inst_arcs_by_level=[
            [0, 1, 0, 10, 1, 0, 0],
            [2, 3, 1, 11, -1, 0, 1],
        ],
        flat_libcell_names=["BUF", "INV"],
        flat_libarc_names=[()] * 10 + [("A", "Y"), ("A", "Y")],
    )

    assert [row["incoming_arc_kind"] for row in rows] == [
        "startpoint",
        "cell",
        "net",
        "cell",
    ]
    assert [row["incoming_delay_ps"] for row in rows[1:]] == pytest.approx([7.0, 4.0, 11.0])
    assert [row["cumulative_arrival_ps"] for row in rows] == pytest.approx([20.0, 27.0, 31.0, 42.0])
    assert [row["incoming_residual_ps"] for row in rows[1:]] == pytest.approx([0.0, 0.0, 0.0])
    assert rows[3]["lib_cell_name"] == "INV"
    assert rows[3]["incoming_source_slew_ps"] == pytest.approx(7.0)
    assert rows[3]["incoming_output_load_pf"] == pytest.approx(0.4)
    assert rows[3]["pin_slew_ps"] == pytest.approx(4.0)
    assert rows[3]["pin_net_cap_pf"] == pytest.approx(0.4)

    summary = critical_path_summary(_batch(), rows)
    assert summary == {
        "status": "ok",
        "failing_state_count": 2,
        "selected_state_count": 1,
        "valid_path_count": 1,
        "invalid_path_count": 0,
        "row_count": 4,
        "max_recomputed_edge_residual_ps": 0.0,
        "topology_epoch": 7,
        "extraction_runtime_ms": 0.25,
    }


def test_timing_index_tensor_follows_reference_device():
    reference = torch.zeros(2)
    result = timing_index_tensor(torch.tensor([1], dtype=torch.int32), reference)
    assert result.dtype == torch.long
    assert result.device == reference.device
    assert result.tolist() == [1]

    if torch.cuda.is_available():
        cuda_values = torch.tensor([1], dtype=torch.int32, device="cuda")
        cpu_result = timing_index_tensor(cuda_values, reference)
        cuda_reference = torch.zeros(2, device="cuda")
        cuda_result = timing_index_tensor(cuda_values.cpu(), cuda_reference)
        assert cpu_result.device.type == "cpu"
        assert cuda_result.device.type == "cuda"
        assert torch.equal(cuda_reference[cuda_result], torch.zeros(1, device="cuda"))
