"""ECC projection temporary timing-state contracts."""

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from dreamplace.NonLinearPlace import NonLinearPlace


class ProjectionTimingSnapshotTest(unittest.TestCase):
    def test_compute_post_projection_timing_snapshot_uses_projected_discrete_state(
        self,
    ):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.pos = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.0], dtype=torch.float32))]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            params = SimpleNamespace(
                result_dir=tmpdir,
                design_name=lambda: "FFT",
                placement_sizing_mode="size_only",
            )

            with open(
                os.path.join(tmpdir, "FFT_endpoint_timing_compare_summary.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "ieda_wns": -80.0,
                        "ieda_tns": -700.0,
                        "num_ieda_negative_slack": 1,
                        "python_wns": -75.0,
                        "python_tns": -650.0,
                    },
                    f,
                )
            with open(
                os.path.join(tmpdir, "FFT_endpoint_timing_compare.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["pin_name", "cpp_slack", "py_slack", "slack_delta"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pin_name": "u_proj/Y",
                        "cpp_slack": -80.0,
                        "py_slack": -75.0,
                        "slack_delta": -5.0,
                    }
                )

            original_size_var = torch.tensor([1.35], dtype=torch.float32)
            original_vt_var = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
            placer.data_collections = SimpleNamespace(
                get_size_var=lambda: original_size_var,
                get_vt_var=lambda: original_vt_var,
                inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
                inst_cell_id=torch.tensor([10], dtype=torch.int64),
            )
            placer.op_collections = SimpleNamespace(
                timing_propagation_op=SimpleNamespace(
                    last_total_slew_violation=0.55,
                    last_total_cap_violation=0.11,
                    last_total_leakage=0.14,
                )
            )
            placer.last_projection_result = SimpleNamespace(
                inst_ids=torch.tensor([0], dtype=torch.int64),
                has_legal_candidate=torch.tensor([True], dtype=torch.bool),
                projected_size=torch.tensor([1.0], dtype=torch.float32),
                projected_vt=torch.tensor([0], dtype=torch.int64),
                projected_cell_id=torch.tensor([11], dtype=torch.int64),
                projected_libcell_offset=torch.tensor([1], dtype=torch.int64),
            )

            captured = {}

            class _ModelStub:
                def __init__(self, data_collections):
                    self.data_collections = data_collections
                    self.timing_call_count = 0

                def timing_obj(self, pos):
                    self.timing_call_count += 1
                    captured["size_var"] = self.data_collections.get_size_var().clone()
                    captured["vt_var"] = self.data_collections.get_vt_var().clone()
                    captured["inst_libcell_offset"] = (
                        self.data_collections.inst_libcell_offset.clone()
                    )
                    captured["inst_cell_id"] = (
                        self.data_collections.inst_cell_id.clone()
                    )
                    return -80.0, -700.0, -7.0, -0.5

            placer.last_global_place_model = _ModelStub(placer.data_collections)

            snapshot = placer._compute_post_projection_timing_snapshot(
                params=params,
                place_record={"wns": -100.0},
                legalization_record={"wns": -90.0},
                projection_summary={"total_projected_leakage": 0.14},
            )

            self.assertEqual(snapshot["record"]["wns"], -80.0)
            self.assertEqual(snapshot["record"]["slew_violation"], 0.55)
            self.assertEqual(snapshot["top_endpoints"], [])
            self.assertEqual(
                snapshot["endpoint_payload"],
                {
                    "source": "backend_endpoint_debug",
                    "available": False,
                    "summary_path": None,
                    "csv_path": None,
                    "wns": None,
                    "tns": None,
                    "num_negative_endpoints": None,
                    "python_wns": None,
                    "python_tns": None,
                },
            )
            self.assertTrue(
                torch.equal(
                    captured["size_var"], torch.tensor([1.0], dtype=torch.float32)
                )
            )
            self.assertTrue(
                torch.equal(
                    captured["vt_var"], torch.tensor([[1.0, 0.0]], dtype=torch.float32)
                )
            )
            self.assertTrue(
                torch.equal(
                    captured["inst_libcell_offset"],
                    torch.tensor([1], dtype=torch.int64),
                )
            )
            self.assertTrue(
                torch.equal(
                    captured["inst_cell_id"], torch.tensor([11], dtype=torch.int64)
                )
            )
            self.assertTrue(
                torch.equal(
                    placer.data_collections.inst_libcell_offset,
                    torch.tensor([0], dtype=torch.int64),
                )
            )
            self.assertTrue(
                torch.equal(placer.data_collections.get_size_var(), original_size_var)
            )
            self.assertTrue(
                torch.equal(placer.data_collections.get_vt_var(), original_vt_var)
            )
            self.assertEqual(placer.last_global_place_model.timing_call_count, 1)

            previous_summary = {
                "pre_projection": {"wns": -100.0},
                "post_projection": {"wns": -123.0},
                "post_legalization": {"wns": -90.0},
            }
            refreshed = placer._refresh_post_projection_timing_stage_artifact(
                params, previous_summary
            )
            self.assertEqual(placer.last_global_place_model.timing_call_count, 2)
            self.assertEqual(refreshed["post_projection"]["wns"], -80.0)
            self.assertEqual(previous_summary["post_projection"], {"wns": -123.0})
            with open(
                os.path.join(tmpdir, "FFT_timing_stage_summary.json"),
                encoding="utf-8",
            ) as stream:
                self.assertEqual(json.load(stream), refreshed)
            self.assertTrue(
                torch.equal(placer.data_collections.get_size_var(), original_size_var)
            )


def test_projection_snapshot_failure_restores_runtime_views_and_metrics(caplog):
    engine = NonLinearPlace.__new__(NonLinearPlace)
    nn.Module.__init__(engine)
    get_size = lambda: torch.tensor([1.5])
    get_vt = lambda: torch.tensor([[0.25, 0.75]])
    data = SimpleNamespace(
        get_size_var=get_size,
        get_vt_var=get_vt,
        inst_cell_id=torch.tensor([0]),
        inst_libcell_offset=torch.tensor([0]),
    )
    timing = SimpleNamespace(
        last_total_slew_violation=1.0,
        last_total_cap_violation=2.0,
        last_total_leakage=3.0,
        last_runtime_seconds=4.0,
    )
    engine.data_collections = data
    engine.op_collections = SimpleNamespace(timing_propagation_op=timing)
    engine.last_timing_metrics = {"wns": -1.0}
    engine.last_projection_result = SimpleNamespace(
        inst_ids=torch.tensor([0]),
        has_legal_candidate=torch.tensor([True]),
        projected_size=torch.tensor([2.0]),
        projected_vt=torch.tensor([0]),
        projected_cell_id=torch.tensor([1]),
        projected_libcell_offset=torch.tensor([1]),
    )

    def fail_timing(pos):
        assert data.inst_cell_id.tolist() == [1]
        timing.last_total_slew_violation = 99.0
        engine.last_timing_metrics = {"wns": -99.0}
        raise RuntimeError("snapshot fixture failure")

    engine.last_global_place_model = SimpleNamespace(timing_obj=fail_timing)
    result = engine._compute_post_projection_timing_snapshot(
        SimpleNamespace(), {}, {}, {}
    )

    assert result is None
    assert "snapshot fixture failure" in caplog.text
    assert data.get_size_var is get_size
    assert data.get_vt_var is get_vt
    assert {
        "cell_ids": data.inst_cell_id.tolist(),
        "offsets": data.inst_libcell_offset.tolist(),
        "timing_metrics": engine.last_timing_metrics,
        "timing_scalars": vars(timing),
    } == {
        "cell_ids": [0],
        "offsets": [0],
        "timing_metrics": {"wns": -1.0},
        "timing_scalars": {
            "last_total_slew_violation": 1.0,
            "last_total_cap_violation": 2.0,
            "last_total_leakage": 3.0,
            "last_runtime_seconds": 4.0,
        },
    }
