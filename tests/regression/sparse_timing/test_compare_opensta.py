from compare_opensta import compare, read_golden


def test_group_order_does_not_replace_worst_endpoint(tmp_path):
    report = tmp_path / "golden.tsv"
    report.write_text(
        "endpoint\tlaunch_clock\tcapture_clock\tslack_ns\n"
        "core/dff/D\tjtag\tsys\t-0.010\n"
        "core/dff/D\tsys\tsys\t0.024\n"
        "out\tsys\tsys\t0.002\n"
    )
    assert read_golden(report) == {
        "core/dff:D": {"slack_ps": -10.0, "launch_clock": "jtag", "capture_clock": "sys"},
        "out": {"slack_ps": 2.0, "launch_clock": "sys", "capture_clock": "sys"},
    }


def test_small_slack_does_not_get_an_extra_one_ns_floor():
    golden = {"p": {"slack_ps": 10, "launch_clock": "sys", "capture_clock": "sys"}}
    summary, rows = compare({"p": 13}, golden)
    assert (summary["aggregate_passed"], summary["endpoint_gate_passed"], summary["slack_failures"]) == (True, False, 1)
    assert (rows[0]["error_ps"], rows[0]["tolerance_ps"]) == (3, 2)


def test_endpoint_membership_is_exact_even_when_tns_matches():
    golden = {"p": {"slack_ps": -100, "launch_clock": "sys", "capture_clock": "sys"}}
    summary, _ = compare({"other": -100}, golden)
    assert (summary["missing_endpoints"], summary["extra_endpoints"], summary["endpoint_gate_passed"]) == (["p"], ["other"], False)
