import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from dreamplace.flows import alternating_discrete
from dreamplace.ops.buffer_insertion.buffering_lane import BufferingOptimizationLane
from dreamplace.ops.buffer_insertion.drive_type_table import write_case_drive_type_table
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_typed_discrete_candidate_state,
)


class _FixtureOwner(alternating_discrete.AlternatingSizingOwner):
    def sync_native_sizing(self):
        return {"ok": True}

    def prepare(self, model, pos, *, active_only=True):
        return None

    def retain_buffered_nets(self, model):
        return None

    def _fits(self, model, capacity):
        return True

    def virtual_area(self, model, state=None):
        return 0.0

    def _sync(self, model, summary, geometry):
        rows = torch.tensor(summary["applied_instance_ids"], dtype=torch.long)
        cells = torch.tensor(summary["applied_cell_ids"], dtype=torch.long)
        self.data.inst_cell_id[rows] = cells
        widths = self.data.flat_libcell_info[cells, 2]
        geometry.apply_sizes(rows, widths, torch.ones_like(widths))

    def prepare_terminal_request(self, model, pos, *, iteration):
        projection = self.lane.project(iteration=iteration, final=True)
        self.terminal_request = self.lane.build_commit_request(
            projection,
            iteration=iteration,
            reason="fixture_final_only",
            commit_enabled=False,
        )
        return {
            "status": "request_ready",
            "iteration": iteration,
            "projected_buffer_count": projection.projected_buffer_count,
            "commit_request": self.terminal_request.to_summary(),
        }


class _FixtureLane(BufferingOptimizationLane):
    def objective(self, _model, pos=None):
        state = self._current_state()
        from dreamplace.ops.net_subgraph_timing import (
            build_relaxed_buffer_dynamic_net_arc_inputs,
        )

        dynamic_payload = {
            "per_size_input_cap": torch.tensor(
                [[0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]],
                dtype=torch.float32,
            ),
            "per_size_delay": torch.tensor(
                [[10.0, 8.0, 6.0, 4.0], [10.0, 8.0, 6.0, 4.0]],
                dtype=torch.float32,
            ),
            "per_size_output_slew": torch.tensor(
                [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
                dtype=torch.float32,
            ),
            "sink_pin_id": torch.tensor([2], dtype=torch.int32),
            "fill_value": 0.0,
            "net_flat_topo_sort": torch.tensor([0, 1], dtype=torch.int32),
            "net_flat_topo_sort_start": torch.tensor([0, 2], dtype=torch.int32),
            "pin_fa": torch.tensor([-1, 0], dtype=torch.int32),
            "flat_pin_to_start": torch.tensor([0, 1, 1], dtype=torch.int32),
            "flat_pin_to": torch.tensor([1], dtype=torch.int32),
            "edge_resistance": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "node_capacitance": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "edge_capacitance": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "driver_arrival": torch.tensor([0.0], dtype=torch.float32),
            "driver_slew": torch.tensor([0.1], dtype=torch.float32),
            "candidate_node_id": torch.tensor([1, 1], dtype=torch.int32),
            "sink_node_id": torch.tensor([1], dtype=torch.int32),
            "sink_net_index": torch.tensor([0], dtype=torch.int32),
        }
        dynamic = build_relaxed_buffer_dynamic_net_arc_inputs(
            buffer_state=state,
            num_pins=3,
            **dynamic_payload,
        )
        # Treat a lower sink delay as a better objective.  The tiny fixture
        # deliberately uses a negated delay so the scheduler has a positive
        # predicted improvement to commit and exercises the full virtual
        # activation -> projection path.
        loss = -dynamic["pin_net_delays"]["rise"][2]
        return loss, {"loss": loss.detach(), "tns": -loss.detach(), "wns": -loss.detach()}


@pytest.mark.parametrize("mode", ["standalone", "overflow", "inflation", "partial_inflation"])
def test_window_schedules_execute_discrete_phases_and_preserve_candidates(tmp_path, mode):
    params = SimpleNamespace(
        timing_opt_overflow_milestones=(0.30, 0.25, 0.20, 0.15, 0.10) if mode == "overflow" else (),
        timing_opt_max_windows=5,
        timing_opt_sizing_rounds=5,
        discrete_gradient_topk_shared_budget_percent=1.0,
        placement_sizing_mode="place_only",
        timing_objective_lane="timing_only",
        differentiable_timing_obj=0,
        sizing_parameterization="real_size",
        cell_padding_x=0.0,
        result_dir=str(tmp_path),
        design_name=lambda: "fixture",
        enable_relaxed_buffer_timing=True,
    )
    data = SimpleNamespace(
        flat_libcell_info=torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
        ),
        flat_libcell_leakage=torch.tensor([1.0, 2.0, 3.0]),
        flat_libcell_width=torch.ones(14),
        flat_libcell_height=torch.ones(14),
        inst_size_lower=torch.tensor([1.0]),
        inst_size_upper=torch.tensor([3.0]),
        inst_vt_mask=torch.tensor([[True]]),
        inst_is_sizeable=torch.tensor([True]),
        inst_cell_id=torch.tensor([0]),
        real_size=torch.nn.Parameter(torch.tensor([1.0])),
        vt_logits=torch.nn.Parameter(torch.zeros(1, 1)),
        node_size_x=torch.tensor([1.0]),
        node_size_y=torch.tensor([1.0]),
        original_node_size_x=torch.tensor([1.0]),
        original_node_size_y=torch.tensor([1.0]),
        node_areas=torch.tensor([1.0]),
        target_density=torch.tensor([0.25]),
        sorted_node_map=torch.tensor([0], dtype=torch.int32),
    )
    data.get_continuous_size_parameter = lambda: data.real_size
    data.sizing_parameterization = "real_size"
    placedb = SimpleNamespace(
        num_movable_nodes=1,
        num_filler_nodes=0,
        area=4.0,
        total_fixed_node_area=0.0,
        total_space_area=4.0,
        regions=[],
        node_size_x=torch.tensor([1.0]).numpy(),
        node_size_y=torch.tensor([1.0]).numpy(),
    )
    model = SimpleNamespace(
        _candidate_density_view=None,
        _timing_geometry_cache=None,
        timing_wns_coeff=1.0,
        timing_tns_coeff=0.01,
        timing_slew_weight=1.0,
        timing_cap_weight=1.0,
        timing_grad_balance_weight=1.0,
        refresh_after_geometry_change=lambda: None,
        refresh_routability_operators=lambda: None,
        inflation_state=SimpleNamespace(target_area=None),
        params=params,
        data_collections=data,
        placedb=placedb,
        _relaxed_buffer_dynamic_net_provider=SimpleNamespace(forward_backend="python_relaxed"),
    )

    def timing_obj(pos, **kwargs):
        error = (data.real_size - 2.4).square().sum() + data.vt_logits.sum() * 0.0
        return -error, -100.0 * error, error * 0.0, error * 0.0

    model.timing_obj = timing_obj
    model._timing_loss = lambda wns, tns, ws, ts: wns

    controller = object.__new__(alternating_discrete.AlternatingDiscreteController)
    owner = _FixtureOwner(
        params,
        placedb,
        data,
        SimpleNamespace(),
        lambda pos: None,
        execute_window=controller.run_outer_round,
        window_limit=5,
    )
    records = [
        {
            "candidate_id": index,
            "net_id": 0,
            "candidate_node_id": index + 1,
            "buffer_main_type_index": 4,
            "parent_node_id": 0,
            "child_node_ids": [1],
            "segment_split_ratio": 0.5,
            "x_dbu": 100 + index * 100,
            "y_dbu": 200,
        }
        for index in range(2)
    ]
    library = {
        index: {
            "cell_id": 10 + index,
            "master_name": name,
            "input_cap": 0.1 + index,
            "delay": 1.0 + index,
            "output_slew": 0.5 + index,
        }
        for index, name in enumerate(("BUFX8H7L", "BUFX12H7L", "BUFX16H7L", "BUFX20H7L"))
    }
    table = write_case_drive_type_table(tmp_path / "drive_strength.json", buffer_library=library)
    state = build_typed_discrete_candidate_state(
        records,
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
    )
    config = SimpleNamespace(
        mode="candidate",
        candidate_strategy="discrete_net_gradient",
        active_strategy="discrete_net_gradient",
        candidate_drive_type_mode="four_strength_table",
        route_b_selection_fraction=0.001,
        continuous_steps=1,
        output_dir=str(tmp_path / "buffering"),
        commit_enabled=False,
        segment_capacity_enabled=False,
        max_selected_actions=None,
    )
    payload = {
        "metadata": {"state_kind": "candidate"},
        "buffer_legal_table": {
            "legal_cell_ids": [10, 11, 12, 13],
            "legal_master_names": list(library[index]["master_name"] for index in range(4)),
            "legal_size_values": [1.0, 2.0, 3.0, 4.0],
            "buffer_main_type_indices": [4, 4, 4, 4],
        },
    }
    lane = _FixtureLane(
        config,
        buffer_optimization_state=state,
        buffer_relaxed_timing_payload=payload,
        placedb=placedb,
    )
    lane.data_collections = data
    owner.lane = lane

    controller.params = params
    controller.placedb = placedb
    controller.data = data
    controller.owner = owner
    controller.config = alternating_discrete.AlternatingDiscreteConfig()
    controller.outer_rounds = 5
    controller.sizing_rounds = 5
    controller.buffering_rounds = 1
    controller.sizing_budget_percent = 1.0
    controller.buffering_fraction = 0.001
    controller.placement_step_count = 0
    controller.completed_rounds = 0
    controller.trace = []
    controller.last_result = None
    controller.last_finalize_result = None

    pos = torch.tensor([10.0, 20.0])
    optimizer = SimpleNamespace(rebase_objective_state=lambda **kwargs: kwargs)
    if mode == "standalone":
        result = alternating_discrete.run_alternating_discrete_flow(
            controller, model, pos, iteration=0, finalize=True
        )
    else:
        if mode == "overflow":
            metric = SimpleNamespace(overflow=torch.tensor([1.0]))
            metric.evaluate = lambda *args: None
            for iteration, stage, overflow in [
                (1, 0, 0.31),
                (2, 0, 0.30),
                (3, 1, 0.4),
                (4, 1, 0.19),
                (5, 2, 0.4),
                (6, 2, 0.09),
            ]:
                metric.overflow = torch.tensor([overflow], dtype=torch.float64)
                controller.before_step(
                    model, pos, optimizer, metric, {}, iteration=iteration, stage=stage
                )
                controller.before_inflation(
                    model, pos, iteration=iteration, stage=stage, optimizer=optimizer
                )
            assert [r["trigger"]["threshold"] for r in controller.trace] == [
                0.30,
                0.25,
                0.20,
                0.15,
                0.10,
            ]
        else:
            events = [(1, 0), (1, 0), (10, 0), (20, 1), (20, 1), (30, 1), (40, 2), (50, 3)]
            if mode == "partial_inflation":
                events = events[:3]
            for iteration, stage in events:
                controller.before_inflation(
                    model, pos, iteration=iteration, stage=stage, optimizer=optimizer
                )
            assert [r["trigger"]["iteration"] for r in controller.trace] == (
                [1, 10] if mode == "partial_inflation" else [1, 10, 20, 30, 40]
            )
        result = controller.finalize(model, pos, iteration=100)

    expected_windows = 2 if mode == "partial_inflation" else 5
    assert result["status"] == "request_ready"
    assert controller.completed_rounds == expected_windows
    assert (
        sum(len(record["sizing"]["rounds"]) for record in controller.trace) == 5 * expected_windows
    )
    assert sum(record["buffering"]["scheduled"] for record in controller.trace) == expected_windows
    assert controller.trace[-1]["buffering"]["accepted_action_digest"]
    assert result["projected_buffer_count"] > 0
    summary_path = Path(controller.summary_path)
    artifact = json.loads(summary_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "request_ready"
    assert artifact["sizing_attempt_count"] == 5 * expected_windows
    assert artifact["buffering_attempt_count"] == expected_windows
    assert artifact["virtual_buffer_count"] == result["projected_buffer_count"]
    assert artifact["untriggered_window_count"] == 5 - expected_windows


def test_live_rc_rebuilds_candidate_payload_only_when_topology_changes(monkeypatch):
    calls = []
    state = SimpleNamespace(activation_param=torch.tensor([[1.0]]))
    lane = SimpleNamespace(
        _current_state=lambda: state,
        refresh_candidate_geometry=lambda **kwargs: calls.append(kwargs["reason"])
        or {"status": "ok"},
    )
    controller = object.__new__(alternating_discrete.AlternatingDiscreteController)
    controller.owner = SimpleNamespace(lane=lane)
    controller.params = SimpleNamespace(buffering_segment_live_geometry=True)
    topo = SimpleNamespace(topology_generation=2, frozen_nets=SimpleNamespace(candidate_state=None))
    controller.ops = SimpleNamespace(steiner_topo_op=topo)
    controller.completed_rounds = 1
    controller._candidate_payload_topology_generation = None
    installs = []
    monkeypatch.setattr(
        alternating_discrete,
        "install_candidate_density",
        lambda model, state: installs.append(state),
    )

    for iteration in [25, 26]:
        controller.refresh_live_geometry(None, None, iteration=iteration)
    topo.topology_generation += 1
    controller.refresh_live_geometry(None, None, iteration=30)

    assert calls == ["placement_geometry_iteration_25", "placement_geometry_iteration_30"]
    assert len(installs) == 3
    assert topo.frozen_nets.candidate_state is state
