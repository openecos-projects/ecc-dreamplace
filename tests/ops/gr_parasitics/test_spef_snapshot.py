"""Unit tests for post-route SPEF RC tree packing and GRParasiticsOp evaluation."""

import gzip
from pathlib import Path

import pytest
import torch
from dreamplace.ops.gr_parasitics.gr_parasitics import GRParasiticsOp
from dreamplace.ops.gr_parasitics.route_snapshot import (
    PinMapping,
    RouteIdentity,
    prepare_spef_snapshot,
)


def _identity() -> RouteIdentity:
    return RouteIdentity(input_sha256="spef_sha", geometry_sha256="geom_sha", generation=1)


def test_spef_and_spef_gz_starrc_escaping_units_and_symmetric_coupling(tmp_path: Path):
    """Verify .spef/.spef.gz equivalence, StarRC escaping, KOHM/FF scaling, and 1x coupling."""
    spef_text = """*SPEF "IEEE 1481-1999"
*DESIGN "escape_demo"
*DIVIDER /
*DELIMITER :
*T_UNIT 1.0 NS
*C_UNIT 1.0 FF
*R_UNIT 0.001 KOHM

*NAME_MAP
*10 i_soc/gen_bank\\[1\\]\\.u_drv
*11 i_soc/gen_bank\\[1\\]\\.u_sink
*20 i_soc/gen_bank\\[1\\]\\.n_victim
*21 i_soc/n_aggr
*22 clk_net

*PORTS
in_p I *C 0.0 0.0

*D_NET *20 25.0
*CONN
*I *10:Q[3] O *C 10.0 20.0 *D BUF_X2
*I *11:D[0] I *C 40.0 20.0
*N *20:1 *C 25.0 20.0
*CAP
1 *20:1 *21:2 10.0
2 *10:Q[3] 2.0
3 *20:1 8.0
4 *11:D[0] 5.0
*RES
1 *10:Q[3] *20:1 4000.0
2 *20:1 *11:D[0] 6000.0
*END

*D_NET *21 15.0
*CONN
*P in_p I *C 0.0 0.0
*I u_other:A I *C 15.0 0.0
*N *21:2 *C 8.0 0.0
*CAP
1 *21:2 *20:1 10.0
2 in_p 1.0
3 *21:2 3.0
4 u_other:A 1.0
*RES
1 in_p *21:2 2000.0
2 *21:2 u_other:A 3000.0
*END

*D_NET *22 10.0
*CONN
*P clk_i I *C 0.0 5.0
*I u_ff:CK I *C 5.0 5.0
*RES
1 clk_i u_ff:CK 1000.0
*END
"""
    plain_path = tmp_path / "design.spef"
    gz_path = tmp_path / "design.spef.gz"
    plain_path.write_text(spef_text, encoding="utf-8")
    with gzip.open(gz_path, "wt", encoding="utf-8") as gz_stream:
        gz_stream.write(spef_text)

    # PyPlaceDB escapes '[' and ']' on DEF buses, while StarRC also escapes '.'.
    pin_names = {
        "i_soc/gen_bank\\[1\\].u_drv:Q[3]": 0,
        "i_soc/gen_bank\\[1\\].u_sink:D[0]": 1,
        "in_p": 2,
        "u_other:A": 3,
        "clk_i": 4,
        "u_ff:CK": 5,
    }
    net_names = {
        "i_soc/gen_bank\\[1\\].n_victim": 0,
        "i_soc/n_aggr": 1,
        "clk_net": 2,
    }
    mapping = PinMapping(
        pin_names=pin_names,
        net_names=net_names,
        pin_net=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
        driver_pin=torch.tensor([0, 2, 4], dtype=torch.int32),
        eligible_nets=torch.tensor([True, True, False], dtype=torch.bool),
    )

    snap_plain = prepare_spef_snapshot(
        plain_path, mapping, _identity(), coupling_cap_scale=1.0, dtype=torch.float64
    )
    snap_gz = prepare_spef_snapshot(
        gz_path, mapping, _identity(), coupling_cap_scale=1.0, dtype=torch.float64
    )
    snap_zero_cc = prepare_spef_snapshot(
        gz_path, mapping, _identity(), coupling_cap_scale=0.0, dtype=torch.float64
    )

    assert snap_plain.rc_parameters is None
    assert snap_plain.filtered_net_count == 1
    assert snap_plain.net_ids.tolist() == [0, 1]
    assert snap_plain.valid_pin_ids.tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(snap_plain.wire_cap, snap_gz.wire_cap, rtol=0, atol=0)
    torch.testing.assert_close(
        snap_plain.incoming_resistance, snap_gz.incoming_resistance, rtol=0, atol=0
    )
    torch.testing.assert_close(snap_plain.vertex_xy_um, snap_gz.vertex_xy_um, rtol=0, atol=0)

    # Grounded cap: net0=(2+8+5)=15 fF, net1=(1+3+1)=5 fF => 0.020 pF total.
    # Symmetric coupling cap: 10 fF in *20 AND 10 fF in *21 => 0.010 pF per net (not 2x!).
    assert pytest.approx(float(snap_zero_cc.wire_cap.sum()), rel=0, abs=1e-12) == 0.020
    assert pytest.approx(float(snap_plain.wire_cap.sum()), rel=0, abs=1e-12) == 0.040
    assert snap_plain.parent.numel() == 6  # 3 nodes on net0 + 3 nodes on net1, zero phantom nodes

    # Verify Elmore delay and autograd gradients through GRParasiticsOp.
    op = GRParasiticsOp(snap_plain)
    pin_cap = torch.tensor(
        [0.0, 0.010, 0.0, 0.005, 0.002, 0.004], dtype=torch.float64, requires_grad=True
    )
    _, loads, delays, _, betas, impulses = op(pin_cap, pin_cap, pin_cap)
    # Net 0: R_UNIT = 0.001 KOHM = 1 Ohm => R1 = 4000*1 = 4000? Wait: 0.001 KOHM = 1 Ohm,
    # so 4000.0 * 1 Ohm = 4000 Ohm? Let's verify exact Elmore math:
    # R1 = 4000 Ohm, R2 = 6000 Ohm; C(drv)=0.002 pF, C(mid)=0.008+0.010=0.018 pF,
    # C(sink)=0.005 + 0.010(pin) = 0.015 pF.
    # Delay(mid) = 4000 * (0.018 + 0.015) = 132.0 ps
    # Delay(sink) = 132.0 + 6000 * 0.015 = 222.0 ps
    # Load(drv) = 0.002 + 0.018 + 0.015 = 0.035 pF
    assert pytest.approx(float(loads["generic"][0].detach()), rel=0, abs=1e-12) == 0.035
    assert pytest.approx(float(delays["generic"][1].detach()), rel=0, abs=1e-9) == 222.0
    # Excluded clock net pins keep local pin cap load and zero wire delay.
    assert pytest.approx(float(loads["generic"][5].detach()), rel=0, abs=1e-12) == 0.004
    assert float(delays["generic"][5].detach()) == 0.0
    assert bool((impulses["generic"] >= 0).all())

    loss = delays["generic"][1] + betas["generic"][1]
    (grad,) = torch.autograd.grad(loss, pin_cap)
    # d(delay_sink)/d(pin_cap_sink) = R1 + R2 = 10000 ps/pF > 0, and beta > 0.
    assert float(grad[1]) > 10000.0
    assert float(grad[4]) == 0.0
    assert float(grad[5]) == 0.0


def test_spef_parallel_via_ladder_series_parallel_reduction_and_zero_ohm_tie(tmp_path: Path):
    """Parallel resistor merge + degree-2 ladder contraction match exact Elmore delay."""
    # Topology on net n0:
    #   u0:Z --(0 Ohm)-- n0:a
    #   Between n0:a and n0:b:
    #     direct resistor R_ab = 20 Ohm
    #     parallel ladder via n0:w with R_aw = 10 Ohm, R_wb = 10 Ohm, C_w = 10 fF
    #     duplicate parallel resistor R_ab2 = 20 Ohm
    #   Exact equivalent between a and b:
    #     G_ab_initial = 1/20 + 1/20 = 0.1 S (R = 10 Ohm)
    #     Ladder a-w-b adds G_series = 1/(10+10) = 0.05 S, splitting C_w equally
    #     Total R_eq(a, b) = 1 / 0.15 = 20/3 Ohm!
    #   n0:b --(30 Ohm)-- u1:A
    spef_path = tmp_path / "ladder.spef"
    spef_path.write_text(
        """*SPEF "IEEE 1481-1999"
*DESIGN "ladder"
*DELIMITER :
*R_UNIT 1.0 OHM
*C_UNIT 1.0 FF
*D_NET n0 10.0
*CONN
*I u0:Z O *C 0.0 0.0
*I u1:A I *C 10.0 0.0
*CAP
1 n0:w 10.0
2 u1:A 20.0
*RES
1 u0:Z n0:a 0.0
2 n0:a n0:b 20.0
3 n0:a n0:b 20.0
4 n0:a n0:w 10.0
5 n0:w n0:b 10.0
6 n0:b u1:A 30.0
*END
""",
        encoding="utf-8",
    )
    mapping = PinMapping(
        pin_names={"u0:Z": 0, "u1:A": 1},
        net_names={"n0": 0},
        pin_net=torch.tensor([0, 0], dtype=torch.int32),
        driver_pin=torch.tensor([0], dtype=torch.int32),
        eligible_nets=torch.tensor([True], dtype=torch.bool),
    )
    snap = prepare_spef_snapshot(spef_path, mapping, _identity(), dtype=torch.float64)
    assert snap.model_report["parallel_resistor_merges"] == 1
    assert snap.model_report["series_parallel_reductions"] == 1
    assert snap.model_report["loop_count"] == 0

    # Topological invariant: root first, and every parent appears strictly before child.
    topo = snap.topo_order.tolist()
    pos_in_topo = {v: idx for idx, v in enumerate(topo)}
    parent = snap.parent.tolist()
    assert parent[snap.root_vertex.item()] == -1
    for v_idx, p_idx in enumerate(parent):
        if p_idx >= 0:
            assert pos_in_topo[p_idx] < pos_in_topo[v_idx]

    op = GRParasiticsOp(snap)
    zero_pin_cap = torch.zeros(2, dtype=torch.float64)
    _, loads, delays, _, _, _ = op(zero_pin_cap, zero_pin_cap, zero_pin_cap)
    # Total wire cap = 10 fF + 20 fF = 0.030 pF
    # C_a = 5 fF (at R=0 from root), C_b = 5 fF (at R=20/3), C_sink = 20 fF (at R=20/3 + 30)
    # Delay(b) = (20/3) * (0.005 + 0.020) = 5/30 = 0.166666666667 ps
    # Delay(sink) = Delay(b) + 30 * 0.020 = 0.166666666667 + 0.6 = 0.766666666667 ps
    expected_delay = (20.0 / 3.0) * 0.025 + 0.6
    assert pytest.approx(float(loads["generic"][0]), rel=0, abs=1e-12) == 0.030
    assert pytest.approx(float(delays["generic"][1]), rel=0, abs=1e-10) == expected_delay
