from types import SimpleNamespace

import pytest
from dreamplace.flows.post_gp.diff_opt import DiffOptController
from dreamplace.flows.config.specs import parse_diff_opt_flows


def test_diff_opt_default_modes_are_step_local():
    mode, flows = parse_diff_opt_flows("buffering")
    assert mode == "buffering"
    assert [flow.to_list() for flow in flows] == [
        ["bf", 10, {"wns": 500.0, "tns": 5.0, "slew": 1.0, "cap": 1.0}]
    ]

    mode, flows = parse_diff_opt_flows("joint")
    assert mode == "joint"
    assert [flow.action for flow in flows] == ["sz", "bf"]


def test_diff_opt_rejects_mode_action_mismatch_and_empty_flow():
    with pytest.raises(ValueError, match="does not allow"):
        parse_diff_opt_flows("sizing", [["bf", 1, [1, 1, 1, 1]]])
    with pytest.raises(ValueError, match="non-empty"):
        parse_diff_opt_flows("joint", [])


def test_diff_opt_coefficients_are_canonicalized():
    _mode, flows = parse_diff_opt_flows(
        "joint",
        [["sz", 2, [100, 1, 1, 1]], ["bf", 1, {"wns": 500, "tns": 5}]],
    )
    assert flows[0].coefficients.to_dict() == {
        "wns": 100.0,
        "tns": 1.0,
        "slew": 1.0,
        "cap": 1.0,
    }
    assert flows[1].coefficients.to_dict() == {
        "wns": 500.0,
        "tns": 5.0,
        "slew": 1.0,
        "cap": 1.0,
    }


def test_diff_opt_buffer_phase_rebuilds_full_domain_and_binds_timing_loss():
    calls = []

    class Lane:
        timing_loss = None

        @staticmethod
        def _current_state():
            return object()

    lane = Lane()

    class Owner:
        def prepare(self, model, pos, *, active_only):
            calls.append(active_only)
            self.lane = lane

    controller = DiffOptController.__new__(DiffOptController)
    controller.owner = Owner()
    controller.params = SimpleNamespace(enable_relaxed_buffer_timing=False)
    def timing_loss(wns, tns, worst_slack, total_slack):
        return wns + tns
    model = SimpleNamespace(_timing_loss=timing_loss)

    controller._ensure_lane(model, object())

    assert calls == [False]
    assert lane.timing_loss is timing_loss
    assert controller.params.enable_relaxed_buffer_timing is True


def test_diff_opt_each_phase_uses_its_four_terms_and_restores_model_state(monkeypatch, tmp_path):
    import torch
    from dreamplace.flows.post_gp import diff_opt
    from dreamplace.flows.discrete.owner import DiscreteTimingOwner
    from dreamplace.PlaceObj import PlaceObj

    params = SimpleNamespace(
        placement_sizing_mode="place_only",
        timing_objective_lane="timing_only",
        differentiable_timing_obj=0,
        timing_opt_sizing_rounds=2,
        buffering_continuous_steps=3,
        diff_optimization_mode="joint",
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0]),
        real_size=torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False),
        vt_logits=torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False),
    )
    data.get_continuous_size_parameter = lambda: data.real_size
    timing = SimpleNamespace(
        last_total_slew_violation_tensor=torch.tensor(1000.0),
        last_total_cap_violation_tensor=torch.tensor(1000.0),
    )
    model = PlaceObj.__new__(PlaceObj)
    model.params = params
    model.op_collections = SimpleNamespace(timing_propagation_op=timing)
    model.timing_wns_coeff, model.timing_tns_coeff = 7.0, 8.0
    model.timing_slew_weight, model.timing_cap_weight = 9.0, 10.0
    model.timing_grad_balance_weight, model.timing_leakage_weight = 11.0, 0.0
    expected_original = (7.0, 8.0, 9.0, 10.0, 11.0)
    model.timing_obj = lambda pos: tuple(torch.tensor(v) for v in (-2.0, -3.0, 0.0, 0.0))
    observed = []
    preparations = []

    class PhaseOwner(DiscreteTimingOwner):
        def __init__(self, *args, **kwargs):
            self.params, self.data, self.capacity = params, data, 100.0
            self.lane = None

        def prepare(self, model, pos, *, active_only):
            preparations.append(active_only)
            self.lane = SimpleNamespace(
                _current_state=lambda: object(),
                summarize=lambda: {
                    "candidate_count": 4,
                    "net_count": 2,
                    "state_kind": "candidate",
                    "candidate_geometry_refresh": {
                        "candidate_topology_generation": 1,
                        "candidate_geometry_generation": 2,
                        "candidate_geometry_digest": "abc",
                        "dirty_node_ids": [1, 2, 3],
                        "dirty_edge_ids": [4, 5, 6],
                    },
                },
            )

        def probe(self, model, pos, capacity):
            assert params.timing_objective_lane == "timing_slew_cap"
            assert params.differentiable_timing_obj == 1
            assert data.real_size.requires_grad and data.vt_logits.requires_grad
            loss = model._timing_loss(*model.timing_obj(pos))
            terms = model.last_timing_objective_terms
            observed.append(dict(terms["weighted_terms"]))
            assert terms["weighted_terms"]["slew"] > 0
            assert terms["weighted_terms"]["cap"] > 0
            return {"best_restore": {"best_score": float(loss)}}

        def run_sizing(self, model, pos, geometry, capacity):
            return self.probe(model, pos, capacity)

        def run_buffering(self, model, pos, capacity):
            import hashlib
            import json

            assert self.lane.timing_loss == model._timing_loss
            result = self.probe(model, pos, capacity)
            path = tmp_path / "candidate_discrete_scheduler_trace.json"
            path.write_text(json.dumps(observed[-1]))
            result.update(
                trace_path=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            return result

        def retain_buffered_nets(self, model):
            pass

        def publish_geometry(self, model):
            return {"status": "ok"}

        def sync_native_sizing(self):
            return {"ok": True}

        def prepare_terminal_request(self, *args, **kwargs):
            return {"status": "request_ready"}

    monkeypatch.setattr(diff_opt, "DiscreteTimingOwner", PhaseOwner)
    monkeypatch.setattr(
        "dreamplace.ops.routability.cooptimization_area.SizingWindowGeometry",
        lambda *_args: None,
    )
    controller = DiffOptController(params, None, data, None, None)
    summary = controller.run(
        model,
        torch.zeros(2),
        flows=[["sz", 1, [100, 1, 2, 3]], ["bf", 1, [200, 2, 4, 6]], ["bf", 1, [300, 3, 6, 9]]],
    )
    assert len(observed) == 3
    assert preparations == [True, False, False]
    for name in ("timing", "slew", "cap"):
        assert [terms[name] for terms in observed] == pytest.approx(
            [observed[0][name] * factor for factor in (1, 2, 3)]
        )
    assert [phase["action"] for phase in summary["phases"]] == ["sz", "bf", "bf"]
    assert (
        model.timing_wns_coeff,
        model.timing_tns_coeff,
        model.timing_slew_weight,
        model.timing_cap_weight,
        model.timing_grad_balance_weight,
    ) == expected_original
    assert (
        params.timing_objective_lane,
        params.differentiable_timing_obj,
        params.placement_sizing_mode,
    ) == ("timing_only", 0, "place_only")
    assert not data.real_size.requires_grad and not data.vt_logits.requires_grad

    import hashlib
    from pathlib import Path

    traces = [phase for phase in summary["phases"] if phase["action"] == "bf"]
    assert len({phase["trace_path"] for phase in traces}) == 2
    for phase in traces:
        assert phase["candidate_state"]["candidate_geometry_refresh"] == {
            "candidate_topology_generation": 1,
            "candidate_geometry_generation": 2,
            "candidate_geometry_digest": "abc",
        }
        assert (
            hashlib.sha256(Path(phase["trace_path"]).read_bytes()).hexdigest()
            == phase["trace_sha256"]
        )


def test_diff_opt_direct_b1_commits_to_ecc_carrier_rebuilds_for_next_sz_and_aggregates(
    ecc_schema3_carrier_harness, monkeypatch, tmp_path
):
    import torch
    from dreamplace.flows.post_gp import diff_opt
    from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
    from dreamplace.ops.buffer_insertion.optimization_state import (
        build_typed_discrete_candidate_state,
    )
    from dreamplace.ops.placeio_ecc.timing_carrier import CommittedTimingPyDB
    from dreamplace.PlaceObj import PlaceObj

    placedb = ecc_schema3_carrier_harness.placedb
    rawdb = ecc_schema3_carrier_harness.rawdb
    params = SimpleNamespace(
        place_io_engine="ecc",
        placement_sizing_mode="place_only",
        timing_objective_lane="timing_only",
        differentiable_timing_obj=0,
        timing_opt_sizing_rounds=1,
        buffering_continuous_steps=1,
        diff_optimization_mode="joint",
        enable_relaxed_buffer_timing=False,
        cell_padding_x=0.0,
        scale_factor=1.0,
        shift_factor=(0.0, 0.0),
        result_dir=str(tmp_path),
    )

    table = ecc_schema3_carrier_harness.make_drive_type_table(tmp_path)

    def _make_runtime():
        n_mov = placedb.num_movable_nodes
        pydb = placedb.pydb
        inst_cell_id = torch.tensor(
            [
                -1
                if int(m) < 0
                else int(pydb.main_id_2_cell_id_start[int(m)])
                + int(pydb.inst_libcell_offset[i])
                for i, m in enumerate(pydb.inst_main_id)
            ],
            dtype=torch.long,
        )
        data = SimpleNamespace(
            sizing_parameterization="real_size",
            flat_libcell_info=torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 0.0]]),
            flat_libcell_leakage=torch.tensor([1.0, 2.0]),
            flat_libcell_width=torch.tensor([1000.0, 2000.0]),
            flat_libcell_height=torch.tensor([1000.0, 1000.0]),
            inst_size_lower=torch.ones(n_mov),
            inst_size_upper=torch.full((n_mov,), 2.0),
            inst_vt_mask=torch.ones((n_mov, 1), dtype=torch.bool),
            inst_is_sizeable=torch.ones(n_mov, dtype=torch.bool),
            inst_cell_id=inst_cell_id,
            real_size=torch.nn.Parameter(torch.ones(n_mov), requires_grad=False),
            vt_logits=torch.nn.Parameter(torch.zeros(n_mov, 1), requires_grad=False),
            node_size_x=torch.as_tensor(placedb.node_size_x, dtype=torch.float64),
            node_size_y=torch.as_tensor(placedb.node_size_y, dtype=torch.float64),
            original_node_size_x=torch.as_tensor(placedb.node_size_x, dtype=torch.float64),
            original_node_size_y=torch.as_tensor(placedb.node_size_y, dtype=torch.float64),
            node_areas=torch.as_tensor(
                placedb.node_size_x * placedb.node_size_y, dtype=torch.float64
            ),
            target_density=torch.tensor([0.5], dtype=torch.float64),
            sorted_node_map=torch.arange(n_mov, dtype=torch.int32),
            pin_offset_x=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            pin_offset_y=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            original_pin_offset_x=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            original_pin_offset_y=torch.zeros(len(placedb.pydb.pin_names), dtype=torch.float64),
            flat_node2pin_map=torch.as_tensor(placedb.flat_node2pin_map, dtype=torch.long),
            flat_node2pin_start_map=torch.as_tensor(
                placedb.flat_node2pin_start_map, dtype=torch.long
            ),
        )
        data.get_continuous_size_parameter = lambda: data.real_size
        pos = torch.tensor(
            list(placedb.node_x) + list(placedb.node_y), dtype=torch.float64
        )
        timing = SimpleNamespace(
            last_total_slew_violation_tensor=torch.tensor(0.0),
            last_total_cap_violation_tensor=torch.tensor(0.0),
            last_total_slew_violation=0.0,
            last_total_cap_violation=0.0,
        )
        ops = SimpleNamespace(
            timing_propagation_op=timing,
            steiner_topo_op=SimpleNamespace(topology_generation=1, frozen_nets=None),
            virtual_cell_density_op=None,
        )
        model = PlaceObj.__new__(PlaceObj)
        model.params = params
        model.op_collections = ops
        model.timing_wns_coeff, model.timing_tns_coeff = 100.0, 1.0
        model.timing_slew_weight, model.timing_cap_weight = 1.0, 1.0
        model.timing_grad_balance_weight, model.timing_leakage_weight = 1.0, 0.0
        model._timing_geometry_cache = None
        model._virtual_cell_density_view = None
        model._candidate_density_view = None
        model.refresh_after_geometry_change = lambda: None
        model.timing_obj = lambda _pos: tuple(
            torch.tensor(v) for v in (-10.0, -20.0, 0.0, 0.0)
        )
        from dreamplace.ops.buffer_insertion.buffering_config import (
            build_buffering_config_from_params,
        )

        config = build_buffering_config_from_params(params, flow_kind="buffering")
        lane = BufferingOptimizationLane(config, placedb=placedb)
        lane.data_collections = data
        return model, pos, data, ops, (lambda _pos: None), lane

    sz_observations = []
    virtual_prepare_calls = []

    def _stub_prepare(self, model, pos, *, active_only=True, **_kwargs):
        del model, pos
        virtual_prepare_calls.append(active_only)
        records = [
            {
                "candidate_id": 0,
                "net_id": 0,
                "candidate_node_id": 1,
                "buffer_main_type_index": 0,
                "parent_node_id": 0,
                "child_node_ids": [1],
                "segment_split_ratio": 0.5,
                "x_dbu": 50000,
                "y_dbu": 0,
            }
        ]
        state = build_typed_discrete_candidate_state(
            records,
            buffer_main_type_index=0,
            legal_buffer_count=4,
            drive_type_table=table,
        )
        payload = {
            "metadata": {"state_kind": "candidate"},
            "buffer_legal_table": {
                "legal_cell_ids": [0, 1, 0, 1],
                "legal_master_names": ["BUF1", "BUF2", "BUF1", "BUF2"],
                "legal_size_values": [1.0, 2.0, 1.0, 2.0],
                "buffer_main_type_indices": [0, 0, 0, 0],
            },
        }
        self.lane.buffer_optimization_state = state
        self.lane.buffer_relaxed_timing_payload = payload
        self.lane.attach_state(self.data)

    def _stub_run_sizing(self, model, pos, geometry, capacity):
        del model, pos, geometry, capacity
        sz_observations.append(
            {
                "relaxed_timing": bool(self.params.enable_relaxed_buffer_timing),
                "node_names": list(self.placedb.node_names),
                "num_movable": int(self.placedb.num_movable_nodes),
                "is_carrier": isinstance(self.placedb.rawdb.pydb, CommittedTimingPyDB),
                "sparse_states": self.placedb.rawdb.sparse_timing_view.statistics()[
                    "num_states"
                ],
            }
        )
        u_idx = self.placedb.node_names.index("u")
        self.data.inst_cell_id[u_idx] = 1
        return {"best_round": 1, "rounds": [{"accepted": 1}]}

    def _stub_run_buffering(self, model, pos, capacity):
        del model, pos, capacity
        state = self.lane._current_state()
        with torch.no_grad():
            state.activation_param.zero_()
            state.activation_param[0, 0] = 1.0
        return {"status": "ok", "scheduled": 1, "accepted": 1}

    rebuild_calls = []

    def _rebuild():
        rebuild_calls.append(len(placedb.node_names))
        return _make_runtime()

    monkeypatch.setattr(diff_opt.DiscreteTimingOwner, "prepare", _stub_prepare)
    monkeypatch.setattr(diff_opt.DiscreteTimingOwner, "run_sizing", _stub_run_sizing)
    monkeypatch.setattr(
        diff_opt.DiscreteTimingOwner, "run_buffering", _stub_run_buffering
    )

    model, pos, data, ops, refresh_topo, lane = _make_runtime()
    controller = diff_opt.DiffOptController(
        params,
        placedb,
        data,
        ops,
        refresh_topo,
        lane=lane,
        rebuild_runtime_fn=_rebuild,
    )
    assert controller.direct_commit_enabled is True
    summary = controller.run(
        model,
        pos,
        flows=[
            ["bf", 1, [100, 1, 1, 1]],
            ["sz", 1, [100, 1, 1, 1]],
            ["bf", 1, [100, 1, 1, 1]],
        ],
    )

    assert ecc_schema3_carrier_harness.sta_calls == []
    assert len(ecc_schema3_carrier_harness.commit_calls) == 2
    assert len(ecc_schema3_carrier_harness.refresh_records) == 2
    assert all(
        rec["ecc_sta_run"] is False
        for rec in ecc_schema3_carrier_harness.refresh_records
    )
    assert rebuild_calls == [4]
    assert virtual_prepare_calls == [False, False]
    assert sz_observations == [
        {
            "relaxed_timing": False,
            "node_names": ["b", "u", "OUT", "IN"],
            "num_movable": 2,
            "is_carrier": True,
            "sparse_states": 12,
        }
    ]
    assert len(ecc_schema3_carrier_harness.sizing_calls) == 1
    assert rawdb.sparse_timing_view is not None
    assert rawdb.sparse_timing_view.statistics()["num_states"] == 16
    assert params.enable_relaxed_buffer_timing is False
    assert controller.owner.lane._current_state() is None
    assert controller.owner.terminal_request.is_noop is True
    assert summary["direct_committed_buffer_count"] == 2
    assert summary["refresh_counts"]["full_rebuild_count"] == 2
    assert summary["terminal_request"]["direct_committed_count"] == 2
