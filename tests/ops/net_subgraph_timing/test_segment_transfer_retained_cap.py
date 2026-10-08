import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    alpha_sample_from_probe_row,
    estimate_retained_cap_candidates_from_probe_row,
    fit_alpha_for_retained_cap_samples,
    retained_cap_from_net_edge_cap_fraction,
    retained_cap_from_target_parent_load,
    summarize_retained_cap_candidate_errors,
    zero_retained_upstream_cap,
)


def test_zero_retained_upstream_cap_returns_segment_tensor():
    values = zero_retained_upstream_cap(
        3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert tuple(values.shape) == (3,)
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, torch.zeros(3, dtype=torch.float32))


def test_retained_cap_from_target_parent_load_clamps_negative_gap():
    target = torch.tensor([0.10, 0.20, 0.30], dtype=torch.float64)
    analytic = torch.tensor([0.04, 0.25, 0.10], dtype=torch.float64)

    retained = retained_cap_from_target_parent_load(
        target_parent_visible_load=target,
        analytic_parent_visible_load=analytic,
    )

    torch.testing.assert_close(
        retained,
        torch.tensor([0.06, 0.0, 0.20], dtype=torch.float64),
    )


def test_probe_row_retained_cap_candidates_are_ranked_by_target_error():
    probe_row = {
        "global_segment_id": 27429,
        "net_name": "n_24317",
        "edge_rc": {"c": 0.00001395144},
        "fractions_by_repeater_count": {"1": [0.5, 0.5]},
        "driver_zero_z_net_cap": 0.057361502,
        "static_buffer_input_cap": 0.000731310,
        "segment_downstream_load": 0.057351038,
        "analytic_transfer_upstream_visible_input_cap": 0.000731310,
        "root_load_decomposition": {
            "edge_cap_sum": 0.025077713,
            "total_node_cap_sum": 0.032283793,
            "manual_root_load_node_plus_edge": 0.057361506,
        },
    }

    candidates = estimate_retained_cap_candidates_from_probe_row(probe_row)
    assert candidates["zero"] == 0.0
    assert abs(candidates["local_upstream_wire_cap"] - 0.00000697572) < 1e-12

    summary = summarize_retained_cap_candidate_errors(
        probe_row=probe_row,
        opensta_upstream_net_cap=0.010885200,
    )

    assert summary["status"] == "candidate_estimator_calibration"
    assert summary["best_candidate"]["name"] == "edge_cap_sum_fraction"
    assert summary["best_candidate"]["abs_error"] < 0.003
    assert summary["candidates"][-1]["name"] in {
        "zero_z_minus_buffer_input",
        "root_load_fraction",
    }


def test_net_edge_cap_fraction_estimator_builds_segment_tensor():
    nets = [
        {
            "net_id": 7,
            "rc_tree": {
                "children_by_node": {1: [2, 3], 2: [], 3: []},
                "edge_rc": {
                    (1, 2): {"c": 0.2},
                    (1, 3): {"c": 0.4},
                },
            },
        }
    ]
    segment_rows = (
        {
            "segment_id": 0,
            "net_id": 7,
            "parent_x_dbu": 0,
            "parent_y_dbu": 0,
            "child_x_dbu": 100,
            "child_y_dbu": 0,
        },
        {
            "segment_id": 1,
            "net_id": 7,
            "fractions_by_repeater_count": {"1": [0.25, 0.75]},
        },
    )

    retained = retained_cap_from_net_edge_cap_fraction(
        nets,
        segment_rows,
        alpha=0.5,
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        retained,
        torch.tensor([0.15, 0.075], dtype=torch.float64),
    )


def test_alpha_fit_for_retained_cap_samples():
    summary = fit_alpha_for_retained_cap_samples(
        [
            {
                "segment_id": 1,
                "net_name": "n1",
                "analytic_parent_visible_load": 1.0,
                "target_parent_visible_load": 3.0,
                "base_retained_cap": 4.0,
            },
            {
                "segment_id": 2,
                "net_name": "n2",
                "analytic_parent_visible_load": 2.0,
                "target_parent_visible_load": 5.0,
                "base_retained_cap": 6.0,
            },
        ]
    )

    assert summary["status"] == "ok"
    assert summary["sample_count"] == 2
    assert summary["alpha"] == 0.5
    assert summary["rmse"] == 0.0
    assert summary["samples"][0]["predicted_parent_visible_load"] == 3.0


def test_alpha_sample_from_probe_row_uses_named_base_candidate():
    probe_row = {
        "global_segment_id": 27429,
        "net_name": "n_24317",
        "analytic_transfer_upstream_visible_input_cap": 0.000731310,
        "edge_rc": {"c": 0.00001395144},
        "fractions_by_repeater_count": {"1": [0.5, 0.5]},
        "root_load_decomposition": {
            "edge_cap_sum": 0.025077713,
        },
    }

    sample = alpha_sample_from_probe_row(
        probe_row=probe_row,
        target_parent_visible_load=0.010885200,
    )
    summary = fit_alpha_for_retained_cap_samples([sample])

    assert sample["base_candidate_name"] == "edge_cap_sum_fraction"
    assert abs(summary["alpha"] - 0.809793939) < 1e-6
    assert summary["rmse"] < 1e-12
