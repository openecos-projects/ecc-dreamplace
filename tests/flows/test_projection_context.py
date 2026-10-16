"""ECC projection scoring contracts."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from dreamplace.NonLinearPlace import NonLinearPlace


class ProjectionContextTest(unittest.TestCase):
    def test_build_projection_context_adds_explicit_slew_cap_penalties_for_heavy_profile(
        self,
    ):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.pos = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.0], dtype=torch.float32))]
        )

        size_var = torch.tensor([1.5, 2.0], dtype=torch.float32)
        vt_var = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        placer.data_collections = SimpleNamespace(
            inst_main_id=torch.tensor([0, 1], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True, True], dtype=torch.bool),
            pin2node_map=torch.tensor([0, 0, 1], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1, 0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64),
            flat_lib_pin_cap=torch.tensor(
                [0.020, 0.000, 0.018, 0.000, 0.030, 0.000, 0.025, 0.000],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [100.0, 100.0, 95.0, 95.0, 120.0, 120.0, 120.0, 120.0],
                dtype=torch.float32,
            ),
            flat_lib_pin_cap_limit=torch.tensor(
                [0.050, 0.050, 0.050, 0.050, 0.060, 0.060, 0.060, 0.060],
                dtype=torch.float32,
            ),
            inst_libcell_offset=torch.tensor([0, 0], dtype=torch.int64),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )
        placer.op_collections = SimpleNamespace(
            timing_propagation_op=SimpleNamespace(
                pin_rtran=torch.tensor([180.0, 110.0, 118.0], dtype=torch.float32),
                pin_ftran=torch.tensor([150.0, 90.0, 105.0], dtype=torch.float32),
                pin_net_cap_rise=torch.tensor(
                    [0.020, 0.090, 0.030], dtype=torch.float32
                ),
                pin_net_cap_fall=torch.tensor(
                    [0.018, 0.080, 0.020], dtype=torch.float32
                ),
            )
        )
        placer._build_endpoint_stage_payload = lambda params_arg, stage_name: ({}, [])

        class _ProjectionOpStub:
            class _CandidateProvider:
                def enumerate(self, data_collections, inst_ids):
                    return SimpleNamespace(
                        candidate_sizes=torch.tensor(
                            [[1.0, 1.5], [2.0, 2.5]],
                            dtype=torch.float32,
                        ),
                        candidate_leakages=torch.tensor(
                            [[0.10, 0.12], [0.20, 0.24]],
                            dtype=torch.float32,
                        ),
                    )

            def __init__(self):
                self.candidate_provider = self._CandidateProvider()

            def _build_request(self, candidates, size_var, vt_var):
                return SimpleNamespace(
                    continuous_sizes=size_var.clone(),
                    current_leakage=torch.tensor([0.11, 0.21], dtype=torch.float32),
                )

        context = placer._build_projection_context(
            _ProjectionOpStub(),
            inst_ids=torch.tensor([0, 1], dtype=torch.int64),
            size_var=size_var,
            vt_var=vt_var,
            params=SimpleNamespace(
                projection_objective_profile="slew_cap_heavy",
                design_name=lambda: "FFT",
            ),
        )

        self.assertIn("slew_penalty", context.candidate_terms)
        self.assertIn("cap_penalty", context.candidate_terms)
        self.assertGreater(context.term_weights["slew_penalty"], 0.0)
        self.assertGreater(context.term_weights["cap_penalty"], 0.0)
        self.assertGreater(context.candidate_terms["slew_penalty"][0, 0].item(), 0.0)
        self.assertGreater(context.candidate_terms["cap_penalty"][0, 0].item(), 0.0)
        self.assertEqual(context.candidate_terms["slew_penalty"][1, 0].item(), 0.0)
        self.assertEqual(context.candidate_terms["cap_penalty"][1, 0].item(), 0.0)

    def test_current_pin_limit_context_interpolates_size_dependent_limits(self):
        placer = NonLinearPlace.__new__(NonLinearPlace)
        nn.Module.__init__(placer)
        placer.pos = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.0], dtype=torch.float32))]
        )

        size_var = torch.tensor([1.5], dtype=torch.float32)
        vt_var = torch.tensor([[1.0]], dtype=torch.float32)
        placer.data_collections = SimpleNamespace(
            pin2node_map=torch.tensor([0, 0], dtype=torch.int64),
            inst_main_id=torch.tensor([0], dtype=torch.int64),
            inst_is_sizeable=torch.tensor([True], dtype=torch.bool),
            inst_libcell_offset=torch.tensor([0], dtype=torch.int64),
            main_id_2_cell_id_start=torch.tensor([0, 2], dtype=torch.int64),
            cell_id_2_libpin_id_start=torch.tensor([0, 2, 4], dtype=torch.int64),
            pin_2_libpin_offset=torch.tensor([0, 1], dtype=torch.int64),
            flat_lib_pin_cap=torch.tensor(
                [0.020, 0.000, 0.020, 0.000],
                dtype=torch.float32,
            ),
            flat_lib_pin_slew_limit=torch.tensor(
                [80.0, 100.0, 80.0, 120.0],
                dtype=torch.float32,
            ),
            flat_lib_pin_cap_limit=torch.tensor(
                [0.040, 0.050, 0.040, 0.070],
                dtype=torch.float32,
            ),
            flat_libcell_info=torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            get_size_var=lambda: size_var,
            get_vt_var=lambda: vt_var,
        )

        context = placer._current_pin_limit_context()

        self.assertTrue(
            torch.equal(context["output_pin_mask"], torch.tensor([False, True]))
        )
        self.assertAlmostEqual(float(context["slew_limits"][0].item()), 80.0, places=6)
        self.assertAlmostEqual(float(context["cap_limits"][0].item()), 0.040, places=6)
        self.assertAlmostEqual(float(context["slew_limits"][1].item()), 110.0, places=6)
        self.assertAlmostEqual(float(context["cap_limits"][1].item()), 0.060, places=6)
