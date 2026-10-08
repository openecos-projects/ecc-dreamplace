import unittest
from pathlib import Path

import torch
import dreamplace.ops.net_subgraph_timing.net_subgraph_timing_cpp as native_cpp
import dreamplace.ops.net_subgraph_timing.segment_count_native as segment_count_native

from dreamplace.ops.buffer_insertion.segment_count_state import (
    build_segment_count_state,
)
from dreamplace.ops.net_subgraph_timing.segment_count_native import (
    segment_count_forward_cuda_segment_transfer_explicit_autograd,
    segment_count_driver_cap_native_cuda_autograd,
    segment_count_forward_native_explicit_autograd,
    segment_count_forward_native,
    segment_count_forward_native_recompute_autograd,
    segment_count_transfer_backward_native,
    segment_count_transfer_backward_native_cuda,
    segment_count_transfer_forward_native,
    segment_count_transfer_forward_native_cuda,
    segment_count_forward_native_segment_transfer_explicit_autograd,
    segment_count_forward_native_segment_transfer_recompute_autograd,
)
from dreamplace.ops.net_subgraph_timing.segment_count_prepared_timing import (
    segment_count_prepared_relaxed_timing,
)
from dreamplace.ops.net_subgraph_timing.segment_count_relaxed_timing import (
    _interpolate_size_tables,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer.data.sampler import (
    default_fake_buffer_device,
)


class SegmentCountNativeTest(unittest.TestCase):
    def _branch_net(self):
        return {
            "net_id": 42,
            "net_name": "branch",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0), 2: (0, 100), 3: (150, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1, 2], 1: [3], 2: [], 3: []},
                "edge_rc": {
                    (0, 1): {"r": 10.0, "c": 0.5},
                    (0, 2): {"r": 6.0, "c": 0.25},
                    (1, 3): {"r": 4.0, "c": 0.125},
                },
                "node_cap": {0: 0.0, 1: 1.0, 2: 5.0, 3: 3.0},
                "sink_nodes": [2, 3],
            },
        }

    def _tables(self, segment_count):
        input_cap = torch.tensor([0.3, 0.4, 0.6, 0.9], dtype=torch.float64).repeat(segment_count, 1)
        delay = torch.tensor([1.0, 2.0, 3.5, 5.0], dtype=torch.float64).repeat(segment_count, 1)
        output_slew = torch.tensor([0.10, 0.20, 0.35, 0.50], dtype=torch.float64).repeat(segment_count, 1)
        return input_cap, delay, output_slew

    def test_native_segment_transfer_zero_count_matches_unbuffered_elmore_load(self):
        net = {
            "net_id": 7,
            "net_name": "line",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 2.0, "c": 0.5}},
                "node_cap": {0: 0.0, 1: 3.0},
                "sink_nodes": [1],
            },
        }
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=0.0,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        kwargs = {
            "segment_state": state,
            "per_size_input_cap": tables[0],
            "per_size_delay": tables[1],
            "per_size_output_slew": tables[2],
            "driver_arrival": torch.tensor([4.0], dtype=torch.float64),
            "driver_slew": torch.tensor([0.2], dtype=torch.float64),
        }
        unbuffered = segment_count_prepared_relaxed_timing(inputs, **kwargs)
        native = segment_count_forward_native_segment_transfer_explicit_autograd(
            inputs,
            buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
            **kwargs,
        )

        torch.testing.assert_close(native["segment_delay"], unbuffered["segment_delay"])
        torch.testing.assert_close(native["sink_arrival"], unbuffered["sink_arrival"])
        torch.testing.assert_close(native["sink_slew"], unbuffered["sink_slew"])

    def _assert_explicit_autograd_matches_prepared_python(self, initial_z):
        net = self._branch_net()
        reference_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=initial_z,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        native_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=initial_z,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(reference_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], reference_state, dtype=torch.float64)
        driver_arrival_ref = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew_ref = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        driver_arrival_native = driver_arrival_ref.detach().clone().requires_grad_(True)
        driver_slew_native = driver_slew_ref.detach().clone().requires_grad_(True)

        reference = segment_count_prepared_relaxed_timing(
            inputs,
            segment_state=reference_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_ref,
            driver_slew=driver_slew_ref,
        )
        native = segment_count_forward_native_explicit_autograd(
            inputs,
            segment_state=native_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_native,
            driver_slew=driver_slew_native,
        )

        reference_loss = (
            reference["sink_arrival"].sum()
            + 0.7 * reference["sink_slew"].sum()
            + 0.3 * reference["sink_load"].sum()
        )
        native_loss = (
            native["sink_arrival"].sum()
            + 0.7 * native["sink_slew"].sum()
            + 0.3 * native["sink_load"].sum()
        )
        reference_loss.backward()
        native_loss.backward()

        for key in ("sink_arrival", "sink_slew", "sink_load"):
            torch.testing.assert_close(native[key], reference[key], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(native_state.z_param.grad, reference_state.z_param.grad)
        torch.testing.assert_close(
            native_state.bsu_index_param.grad,
            reference_state.bsu_index_param.grad,
        )
        torch.testing.assert_close(driver_arrival_native.grad, driver_arrival_ref.grad)
        torch.testing.assert_close(driver_slew_native.grad, driver_slew_ref.grad)

    def test_native_forward_matches_prepared_python(self):
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64)
        driver_slew = torch.tensor([0.2], dtype=torch.float64)
        buffer_tensors = _interpolate_size_tables(
            state.bsu_index(),
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
        )

        reference = segment_count_prepared_relaxed_timing(
            inputs,
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
        )
        native = segment_count_forward_native(
            net_topo_start=inputs["net_topo_start"],
            flat_topo_node_id=inputs["flat_topo_node_id"],
            edge_start=inputs["edge_start"],
            edge_parent_compact_id=inputs["edge_parent_compact_id"],
            edge_child_compact_id=inputs["edge_child_compact_id"],
            edge_resistance=inputs["edge_resistance"],
            edge_capacitance=inputs["edge_capacitance"],
            node_capacitance=inputs["node_capacitance"],
            edge_to_segment_id=inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=state.z_value(),
            buffer_input_cap=buffer_tensors["buffer_input_cap"],
            buffer_delay=buffer_tensors["buffer_delay"],
            buffer_output_slew=buffer_tensors["buffer_output_slew"],
            parent_cap_fraction=inputs["parent_cap_fraction"],
            child_cap_fraction=inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=inputs["segment_sub_resistance_fraction"],
            sink_node_id=inputs["sink_node_id"],
            sink_net_index=inputs["sink_net_id"],
            sink_node_compact_id=inputs["sink_node_compact_id"],
        )

        for key in (
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "segment_delay",
            "segment_output_slew",
            "segment_upstream_visible_input_cap",
        ):
            torch.testing.assert_close(native[key], reference[key], atol=1e-7, rtol=1e-7)

    def test_native_segment_transfer_live_rc_gradients_match_reference_and_finite_difference(self):
        net = self._branch_net()
        device_lut = default_fake_buffer_device(dtype=torch.float64)

        def evaluate(initial_z, edge_r, edge_c, *, native):
            state = build_segment_count_state(
                [net],
                buffer_main_type_index=7,
                legal_buffer_count=4,
                max_repeater_count=3,
                initial_z=initial_z,
                initial_bsu_index=1.5,
                dtype=torch.float64,
            )
            tables = self._tables(state.num_segments)
            inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
            kwargs = {
                "segment_state": state,
                "per_size_input_cap": tables[0],
                "per_size_delay": tables[1],
                "per_size_output_slew": tables[2],
                "driver_arrival": torch.tensor([4.0], dtype=torch.float64),
                "driver_slew": torch.tensor([0.2], dtype=torch.float64),
                "buffer_device_lut": device_lut,
                "edge_resistance_override": edge_r,
                "edge_capacitance_override": edge_c,
            }
            if native:
                result = segment_count_forward_native_segment_transfer_explicit_autograd(
                    inputs,
                    canonical_equal_spacing=True,
                    **kwargs,
                )
            else:
                kwargs.pop("buffer_device_lut")
                result = segment_count_prepared_relaxed_timing(
                    inputs,
                    transfer_backend="segment_transfer_python",
                    buffer_device_lut=device_lut,
                    canonical_equal_spacing=True,
                    **kwargs,
                )
            loss = (
                result["sink_arrival"].sum()
                + 0.7 * result["sink_slew"].sum()
                + 0.3 * result["sink_load"].sum()
                + 0.2 * result["driver_net_cap"].sum()
            )
            return loss

        base_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=0.0,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        base_inputs = build_segment_count_timing_inputs(
            [net],
            base_state,
            dtype=torch.float64,
        )
        for initial_z in (0.0, 1.0, 2.0, 3.0, 1.25):
            with self.subTest(initial_z=initial_z):
                ref_r = base_inputs["edge_resistance"].clone().requires_grad_(True)
                ref_c = base_inputs["edge_capacitance"].clone().requires_grad_(True)
                native_r = ref_r.detach().clone().requires_grad_(True)
                native_c = ref_c.detach().clone().requires_grad_(True)
                ref_grad = torch.autograd.grad(
                    evaluate(initial_z, ref_r, ref_c, native=False),
                    (ref_r, ref_c),
                )
                native_grad = torch.autograd.grad(
                    evaluate(initial_z, native_r, native_c, native=True),
                    (native_r, native_c),
                )
                torch.testing.assert_close(native_grad[0], ref_grad[0], rtol=1e-8, atol=1e-9)
                torch.testing.assert_close(native_grad[1], ref_grad[1], rtol=1e-8, atol=1e-9)
                self.assertTrue(bool((native_grad[0] != 0.0).any()))
                self.assertTrue(bool((native_grad[1] != 0.0).any()))

                epsilon = 1.0e-6
                for tensor_index, analytical in ((0, native_grad[0]), (1, native_grad[1])):
                    plus_r = native_r.detach().clone()
                    minus_r = native_r.detach().clone()
                    plus_c = native_c.detach().clone()
                    minus_c = native_c.detach().clone()
                    if tensor_index == 0:
                        plus_r[0] += epsilon
                        minus_r[0] -= epsilon
                    else:
                        plus_c[0] += epsilon
                        minus_c[0] -= epsilon
                    finite_difference = (
                        evaluate(initial_z, plus_r, plus_c, native=True)
                        - evaluate(initial_z, minus_r, minus_c, native=True)
                    ) / (2.0 * epsilon)
                    torch.testing.assert_close(
                        analytical[0],
                        finite_difference,
                        rtol=1e-5,
                        atol=1e-6,
                    )

    def test_native_recompute_autograd_matches_prepared_python_gradients(self):
        net = self._branch_net()
        reference_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        native_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(reference_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], reference_state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64)
        driver_slew = torch.tensor([0.2], dtype=torch.float64)

        reference = segment_count_prepared_relaxed_timing(
            inputs,
            segment_state=reference_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
        )
        native = segment_count_forward_native_recompute_autograd(
            inputs,
            segment_state=native_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
        )

        (reference["sink_arrival"].sum() + reference["sink_slew"].sum()).backward()
        (native["sink_arrival"].sum() + native["sink_slew"].sum()).backward()

        torch.testing.assert_close(native_state.z_param.grad, reference_state.z_param.grad)
        torch.testing.assert_close(
            native_state.bsu_index_param.grad,
            reference_state.bsu_index_param.grad,
        )

    def test_native_segment_transfer_recompute_autograd_matches_prepared_high_fidelity(self):
        net = self._branch_net()
        reference_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        native_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(reference_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], reference_state, dtype=torch.float64)
        driver_arrival_ref = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew_ref = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        driver_arrival_native = driver_arrival_ref.detach().clone().requires_grad_(True)
        driver_slew_native = driver_slew_ref.detach().clone().requires_grad_(True)
        device_lut = default_fake_buffer_device(dtype=torch.float64)

        reference = segment_count_prepared_relaxed_timing(
            inputs,
            segment_state=reference_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_ref,
            driver_slew=driver_slew_ref,
            transfer_backend="segment_transfer_native",
            buffer_device_lut=device_lut,
        )
        native = segment_count_forward_native_segment_transfer_recompute_autograd(
            inputs,
            segment_state=native_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_native,
            driver_slew=driver_slew_native,
            buffer_device_lut=device_lut,
        )

        reference_loss = (
            reference["sink_arrival"].sum()
            + 0.7 * reference["sink_slew"].sum()
            + 0.3 * reference["sink_load"].sum()
            + 0.2 * reference["driver_net_cap"].sum()
        )
        native_loss = (
            native["sink_arrival"].sum()
            + 0.7 * native["sink_slew"].sum()
            + 0.3 * native["sink_load"].sum()
            + 0.2 * native["driver_net_cap"].sum()
        )
        reference_loss.backward()
        native_loss.backward()

        for key in (
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "segment_delay",
            "segment_output_slew",
            "segment_upstream_visible_input_cap",
            "driver_net_cap",
        ):
            torch.testing.assert_close(native[key], reference[key], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(native_state.z_param.grad, reference_state.z_param.grad)
        torch.testing.assert_close(
            native_state.bsu_index_param.grad,
            reference_state.bsu_index_param.grad,
        )
        torch.testing.assert_close(driver_arrival_native.grad, driver_arrival_ref.grad)
        torch.testing.assert_close(driver_slew_native.grad, driver_slew_ref.grad)
        self.assertEqual(
            native["metadata"]["backend"],
            "cpp_cpu_segment_transfer_recompute_autograd",
        )

    def test_native_segment_transfer_explicit_autograd_matches_forward_and_finite_difference(self):
        net = self._branch_net()
        recompute_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        explicit_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(recompute_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], recompute_state, dtype=torch.float64)
        driver_arrival_recompute = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew_recompute = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        driver_arrival_explicit = driver_arrival_recompute.detach().clone().requires_grad_(True)
        driver_slew_explicit = driver_slew_recompute.detach().clone().requires_grad_(True)
        device_lut = default_fake_buffer_device(dtype=torch.float64)

        recompute = segment_count_forward_native_segment_transfer_recompute_autograd(
            inputs,
            segment_state=recompute_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_recompute,
            driver_slew=driver_slew_recompute,
            buffer_device_lut=device_lut,
        )
        explicit = segment_count_forward_native_segment_transfer_explicit_autograd(
            inputs,
            segment_state=explicit_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_explicit,
            driver_slew=driver_slew_explicit,
            buffer_device_lut=device_lut,
        )

        recompute_loss = (
            recompute["sink_arrival"].sum()
            + 0.7 * recompute["sink_slew"].sum()
            + 0.3 * recompute["sink_load"].sum()
            + 0.2 * recompute["driver_net_cap"].sum()
        )
        explicit_loss = (
            explicit["sink_arrival"].sum()
            + 0.7 * explicit["sink_slew"].sum()
            + 0.3 * explicit["sink_load"].sum()
            + 0.2 * explicit["driver_net_cap"].sum()
        )
        recompute_loss.backward()
        explicit_loss.backward()

        for key in (
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "segment_delay",
            "segment_output_slew",
            "segment_upstream_visible_input_cap",
            "driver_net_cap",
        ):
            torch.testing.assert_close(explicit[key], recompute[key], atol=1e-7, rtol=1e-7)

        def explicit_loss_at(z_value, bsu_value):
            state = build_segment_count_state(
                [net],
                buffer_main_type_index=7,
                legal_buffer_count=4,
                max_repeater_count=3,
                initial_z=1.25,
                initial_bsu_index=1.5,
                dtype=torch.float64,
            )
            with torch.no_grad():
                state.z_param.copy_(z_value)
                state.bsu_index_param.copy_(bsu_value)
            replay = segment_count_forward_native_segment_transfer_explicit_autograd(
                inputs,
                segment_state=state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_arrival=torch.tensor([4.0], dtype=torch.float64),
                driver_slew=torch.tensor([0.2], dtype=torch.float64),
                buffer_device_lut=device_lut,
            )
            return (
                replay["sink_arrival"].sum()
                + 0.7 * replay["sink_slew"].sum()
                + 0.3 * replay["sink_load"].sum()
                + 0.2 * replay["driver_net_cap"].sum()
            )

        eps = 1e-5
        z_base = explicit_state.z_param.detach()
        bsu_base = explicit_state.bsu_index_param.detach()
        fd_z = []
        fd_bsu = []
        for index in range(int(z_base.numel())):
            z_plus = z_base.clone()
            z_minus = z_base.clone()
            z_plus[index] += eps
            z_minus[index] -= eps
            fd_z.append(
                (
                    explicit_loss_at(z_plus, bsu_base).item()
                    - explicit_loss_at(z_minus, bsu_base).item()
                )
                / (2.0 * eps)
            )
        for index in range(int(bsu_base.numel())):
            bsu_plus = bsu_base.clone()
            bsu_minus = bsu_base.clone()
            bsu_plus[index] += eps
            bsu_minus[index] -= eps
            fd_bsu.append(
                (
                    explicit_loss_at(z_base, bsu_plus).item()
                    - explicit_loss_at(z_base, bsu_minus).item()
                )
                / (2.0 * eps)
            )
        torch.testing.assert_close(
            explicit_state.z_param.grad,
            torch.tensor(fd_z, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            explicit_state.bsu_index_param.grad,
            torch.tensor(fd_bsu, dtype=torch.float64),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(
            explicit["metadata"]["backend"],
            "cpp_cpu_segment_transfer_explicit_autograd",
        )

    def test_cuda_segment_count_transfer_forward_matches_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64)
        driver_slew = torch.tensor([0.2], dtype=torch.float64)
        device_lut = default_fake_buffer_device(dtype=torch.float64)
        lut_tensors = device_lut.tensors(dtype=torch.float64, device=torch.device("cpu"))
        load_buffer_tensors = _interpolate_size_tables(
            state.bsu_index(),
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
        )
        retained_cap = torch.zeros_like(state.z_param)

        cpu = segment_count_transfer_forward_native(
            net_topo_start=inputs["net_topo_start"],
            flat_topo_node_id=inputs["flat_topo_node_id"],
            edge_start=inputs["edge_start"],
            edge_parent_compact_id=inputs["edge_parent_compact_id"],
            edge_child_compact_id=inputs["edge_child_compact_id"],
            edge_resistance=inputs["edge_resistance"],
            edge_capacitance=inputs["edge_capacitance"],
            node_capacitance=inputs["node_capacitance"],
            edge_to_segment_id=inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=state.z_value(),
            bsu_index=state.bsu_index(),
            load_input_cap=load_buffer_tensors["buffer_input_cap"],
            buffer_input_cap_by_size=lut_tensors["input_cap_by_size"],
            buffer_slew_axis=lut_tensors["input_slew_axis"],
            buffer_load_axis=lut_tensors["output_load_axis"],
            buffer_delay_lut=lut_tensors["delay_lut"],
            buffer_output_slew_lut=lut_tensors["output_slew_lut"],
            parent_cap_fraction=inputs["parent_cap_fraction"],
            child_cap_fraction=inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=inputs["segment_sub_resistance_fraction"],
            segment_retained_upstream_cap=retained_cap,
            sink_node_id=inputs["sink_node_id"],
            sink_net_index=inputs["sink_net_id"],
            sink_node_compact_id=inputs["sink_node_compact_id"],
        )

        def cuda_tensor(value):
            return value.cuda() if torch.is_tensor(value) else value

        try:
            cuda = segment_count_transfer_forward_native_cuda(
                net_topo_start=inputs["net_topo_start"].cuda(),
                flat_topo_node_id=inputs["flat_topo_node_id"].cuda(),
                edge_start=inputs["edge_start"].cuda(),
                edge_parent_compact_id=inputs["edge_parent_compact_id"].cuda(),
                edge_child_compact_id=inputs["edge_child_compact_id"].cuda(),
                edge_resistance=inputs["edge_resistance"].cuda(),
                edge_capacitance=inputs["edge_capacitance"].cuda(),
                node_capacitance=inputs["node_capacitance"].cuda(),
                edge_to_segment_id=inputs["edge_to_segment_id"].cuda(),
                driver_arrival=driver_arrival.cuda(),
                driver_slew=driver_slew.cuda(),
                z_value=state.z_value().cuda(),
                bsu_index=state.bsu_index().cuda(),
                load_input_cap=load_buffer_tensors["buffer_input_cap"].cuda(),
                buffer_input_cap_by_size=cuda_tensor(lut_tensors["input_cap_by_size"]),
                buffer_slew_axis=cuda_tensor(lut_tensors["input_slew_axis"]),
                buffer_load_axis=cuda_tensor(lut_tensors["output_load_axis"]),
                buffer_delay_lut=cuda_tensor(lut_tensors["delay_lut"]),
                buffer_output_slew_lut=cuda_tensor(lut_tensors["output_slew_lut"]),
                parent_cap_fraction=inputs["parent_cap_fraction"].cuda(),
                child_cap_fraction=inputs["child_cap_fraction"].cuda(),
                segment_sub_resistance_fraction=inputs[
                    "segment_sub_resistance_fraction"
                ].cuda(),
                segment_retained_upstream_cap=retained_cap.cuda(),
                sink_node_id=inputs["sink_node_id"].cuda(),
                sink_net_index=inputs["sink_net_id"].cuda(),
                sink_node_compact_id=inputs["sink_node_compact_id"].cuda(),
            )
            torch.cuda.synchronize()
        except ImportError as exc:
            self.skipTest(f"CUDA extension is not built: {exc}")

        for key in (
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "segment_delay",
            "segment_output_slew",
            "segment_upstream_visible_input_cap",
            "node_load",
            "node_arrival",
            "node_slew",
            "effective_node_cap",
        ):
            torch.testing.assert_close(
                cuda[key].cpu(),
                cpu[key],
                atol=1e-7,
                rtol=1e-7,
            )

    def test_cuda_segment_count_transfer_backward_matches_cpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64)
        driver_slew = torch.tensor([0.2], dtype=torch.float64)
        device_lut = default_fake_buffer_device(dtype=torch.float64)
        lut_tensors = device_lut.tensors(dtype=torch.float64, device=torch.device("cpu"))
        load_buffer_tensors = _interpolate_size_tables(
            state.bsu_index(),
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
        )
        forward_result = segment_count_transfer_forward_native(
            net_topo_start=inputs["net_topo_start"],
            flat_topo_node_id=inputs["flat_topo_node_id"],
            edge_start=inputs["edge_start"],
            edge_parent_compact_id=inputs["edge_parent_compact_id"],
            edge_child_compact_id=inputs["edge_child_compact_id"],
            edge_resistance=inputs["edge_resistance"],
            edge_capacitance=inputs["edge_capacitance"],
            node_capacitance=inputs["node_capacitance"],
            edge_to_segment_id=inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=state.z_value(),
            bsu_index=state.bsu_index(),
            load_input_cap=load_buffer_tensors["buffer_input_cap"],
            buffer_input_cap_by_size=lut_tensors["input_cap_by_size"],
            buffer_slew_axis=lut_tensors["input_slew_axis"],
            buffer_load_axis=lut_tensors["output_load_axis"],
            buffer_delay_lut=lut_tensors["delay_lut"],
            buffer_output_slew_lut=lut_tensors["output_slew_lut"],
            parent_cap_fraction=inputs["parent_cap_fraction"],
            child_cap_fraction=inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=inputs["segment_sub_resistance_fraction"],
            segment_retained_upstream_cap=torch.zeros_like(state.z_param),
            sink_node_id=inputs["sink_node_id"],
            sink_net_index=inputs["sink_net_id"],
            sink_node_compact_id=inputs["sink_node_compact_id"],
        )
        grad_segment_delay = torch.full_like(forward_result["segment_delay"], 0.11)
        grad_segment_slew = torch.full_like(forward_result["segment_output_slew"], 0.17)
        grad_segment_upstream = torch.full_like(
            forward_result["segment_upstream_visible_input_cap"],
            0.19,
        )
        grad_sink_arrival = torch.ones_like(forward_result["sink_arrival"])
        grad_sink_slew = torch.full_like(forward_result["sink_slew"], 0.7)
        grad_sink_load = torch.full_like(forward_result["sink_load"], 0.3)

        cpu = segment_count_transfer_backward_native(
            net_topo_start=inputs["net_topo_start"],
            edge_start=inputs["edge_start"],
            edge_parent_compact_id=inputs["edge_parent_compact_id"],
            edge_child_compact_id=inputs["edge_child_compact_id"],
            edge_resistance=inputs["edge_resistance"],
            edge_capacitance=inputs["edge_capacitance"],
            edge_to_segment_id=inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            z_value=state.z_value(),
            bsu_index=state.bsu_index(),
            load_input_cap=load_buffer_tensors["buffer_input_cap"],
            buffer_input_cap_by_size=lut_tensors["input_cap_by_size"],
            buffer_slew_axis=lut_tensors["input_slew_axis"],
            buffer_load_axis=lut_tensors["output_load_axis"],
            buffer_delay_lut=lut_tensors["delay_lut"],
            buffer_output_slew_lut=lut_tensors["output_slew_lut"],
            parent_cap_fraction=inputs["parent_cap_fraction"],
            child_cap_fraction=inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=inputs["segment_sub_resistance_fraction"],
            sink_node_compact_id=inputs["sink_node_compact_id"],
            node_load=forward_result["node_load"],
            node_slew=forward_result["node_slew"],
            effective_node_cap=forward_result["effective_node_cap"],
            grad_segment_delay=grad_segment_delay,
            grad_segment_output_slew=grad_segment_slew,
            grad_segment_upstream_visible_input_cap=grad_segment_upstream,
            grad_sink_arrival=grad_sink_arrival,
            grad_sink_slew=grad_sink_slew,
            grad_sink_load=grad_sink_load,
        )

        try:
            cuda = segment_count_transfer_backward_native_cuda(
                net_topo_start=inputs["net_topo_start"].cuda(),
                edge_start=inputs["edge_start"].cuda(),
                edge_parent_compact_id=inputs["edge_parent_compact_id"].cuda(),
                edge_child_compact_id=inputs["edge_child_compact_id"].cuda(),
                edge_resistance=inputs["edge_resistance"].cuda(),
                edge_capacitance=inputs["edge_capacitance"].cuda(),
                edge_to_segment_id=inputs["edge_to_segment_id"].cuda(),
                driver_arrival=driver_arrival.cuda(),
                driver_slew=driver_slew.cuda(),
                z_value=state.z_value().cuda(),
                bsu_index=state.bsu_index().cuda(),
                load_input_cap=load_buffer_tensors["buffer_input_cap"].cuda(),
                buffer_input_cap_by_size=lut_tensors["input_cap_by_size"].cuda(),
                buffer_slew_axis=lut_tensors["input_slew_axis"].cuda(),
                buffer_load_axis=lut_tensors["output_load_axis"].cuda(),
                buffer_delay_lut=lut_tensors["delay_lut"].cuda(),
                buffer_output_slew_lut=lut_tensors["output_slew_lut"].cuda(),
                parent_cap_fraction=inputs["parent_cap_fraction"].cuda(),
                child_cap_fraction=inputs["child_cap_fraction"].cuda(),
                segment_sub_resistance_fraction=inputs[
                    "segment_sub_resistance_fraction"
                ].cuda(),
                sink_node_compact_id=inputs["sink_node_compact_id"].cuda(),
                node_load=forward_result["node_load"].cuda(),
                node_slew=forward_result["node_slew"].cuda(),
                effective_node_cap=forward_result["effective_node_cap"].cuda(),
                grad_segment_delay=grad_segment_delay.cuda(),
                grad_segment_output_slew=grad_segment_slew.cuda(),
                grad_segment_upstream_visible_input_cap=grad_segment_upstream.cuda(),
                grad_sink_arrival=grad_sink_arrival.cuda(),
                grad_sink_slew=grad_sink_slew.cuda(),
                grad_sink_load=grad_sink_load.cuda(),
            )
            torch.cuda.synchronize()
        except ImportError as exc:
            self.skipTest(f"CUDA extension is not built: {exc}")

        for key in (
            "grad_z",
            "grad_bsu",
            "grad_load_input_cap",
            "grad_driver_arrival",
            "grad_driver_slew",
        ):
            torch.testing.assert_close(cuda[key].cpu(), cpu[key], atol=1e-7, rtol=1e-7)
        self.assertGreater(cuda["temporary_allocation_bytes"], 0)
        self.assertGreaterEqual(cuda["temporary_allocation_cpu_ms"], 0.0)

    def _assert_cuda_explicit_autograd_matches_cpu_explicit(self, initial_z):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        net = self._branch_net()
        cpu_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=initial_z,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        hybrid_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=initial_z,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(cpu_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], cpu_state, dtype=torch.float64)
        driver_arrival_cpu = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew_cpu = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        driver_arrival_hybrid = driver_arrival_cpu.detach().clone().requires_grad_(True)
        driver_slew_hybrid = driver_slew_cpu.detach().clone().requires_grad_(True)
        edge_resistance_cpu = inputs["edge_resistance"].clone().requires_grad_(True)
        edge_capacitance_cpu = inputs["edge_capacitance"].clone().requires_grad_(True)
        edge_resistance_hybrid = edge_resistance_cpu.detach().clone().requires_grad_(True)
        edge_capacitance_hybrid = edge_capacitance_cpu.detach().clone().requires_grad_(True)
        device_lut = default_fake_buffer_device(dtype=torch.float64)

        cpu = segment_count_forward_native_segment_transfer_explicit_autograd(
            inputs,
            segment_state=cpu_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_cpu,
            driver_slew=driver_slew_cpu,
            buffer_device_lut=device_lut,
            edge_resistance_override=edge_resistance_cpu,
            edge_capacitance_override=edge_capacitance_cpu,
            canonical_equal_spacing=True,
        )
        hybrid = segment_count_forward_cuda_segment_transfer_explicit_autograd(
            inputs,
            segment_state=hybrid_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_hybrid,
            driver_slew=driver_slew_hybrid,
            buffer_device_lut=device_lut,
            edge_resistance_override=edge_resistance_hybrid,
            edge_capacitance_override=edge_capacitance_hybrid,
            canonical_equal_spacing=True,
        )
        cpu_loss = (
            cpu["sink_arrival"].sum()
            + 0.7 * cpu["sink_slew"].sum()
            + 0.3 * cpu["sink_load"].sum()
            + 0.2 * cpu["driver_net_cap"].sum()
        )
        hybrid_loss = (
            hybrid["sink_arrival"].sum()
            + 0.7 * hybrid["sink_slew"].sum()
            + 0.3 * hybrid["sink_load"].sum()
            + 0.2 * hybrid["driver_net_cap"].sum()
        )
        cpu_loss.backward()
        hybrid_loss.backward()

        for key in (
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "segment_delay",
            "segment_output_slew",
            "segment_upstream_visible_input_cap",
            "driver_net_cap",
        ):
            torch.testing.assert_close(hybrid[key], cpu[key], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(hybrid_state.z_param.grad, cpu_state.z_param.grad)
        torch.testing.assert_close(
            hybrid_state.bsu_index_param.grad,
            cpu_state.bsu_index_param.grad,
        )
        torch.testing.assert_close(driver_arrival_hybrid.grad, driver_arrival_cpu.grad)
        torch.testing.assert_close(driver_slew_hybrid.grad, driver_slew_cpu.grad)
        torch.testing.assert_close(edge_resistance_hybrid.grad, edge_resistance_cpu.grad)
        torch.testing.assert_close(edge_capacitance_hybrid.grad, edge_capacitance_cpu.grad)
        self.assertEqual(
            hybrid["metadata"]["backend"],
            "cpp_cuda_segment_transfer_explicit_autograd",
        )
        self.assertIsNone(hybrid["metadata"]["backward_fallback_reason"])
        self.assertEqual(
            hybrid["metadata"]["backward"],
            "cpp_cuda_segment_count_transfer_explicit",
        )

    def test_cuda_explicit_autograd_matches_cpu_at_route_b_count_states(self):
        for initial_z in (0.0, 1.0, 2.0, 3.0, 1.25):
            self._assert_cuda_explicit_autograd_matches_cpu_explicit(initial_z)

    def test_cuda_explicit_autograd_keeps_gpu_resident_outputs_and_gradients(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        device = torch.device("cuda", torch.cuda.current_device())
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
            device=device,
        )
        tables = tuple(table.to(device=device) for table in self._tables(len(state.segment_rows)))
        inputs = build_segment_count_timing_inputs(
            [net],
            state,
            dtype=torch.float64,
            device=device,
        )
        driver_arrival = torch.tensor([4.0], dtype=torch.float64, device=device, requires_grad=True)
        driver_slew = torch.tensor([0.2], dtype=torch.float64, device=device, requires_grad=True)
        edge_resistance = inputs["edge_resistance"].clone().requires_grad_(True)
        edge_capacitance = inputs["edge_capacitance"].clone().requires_grad_(True)
        result = segment_count_forward_cuda_segment_transfer_explicit_autograd(
            inputs,
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
            buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
            edge_resistance_override=edge_resistance,
            edge_capacitance_override=edge_capacitance,
            canonical_equal_spacing=True,
        )

        for key in (
            "segment_delay",
            "segment_output_slew",
            "sink_arrival",
            "sink_slew",
            "sink_load",
            "sink_node_id",
            "sink_net_index",
            "driver_net_cap",
        ):
            assert result[key].device == device
        (
            result["sink_arrival"].sum()
            + result["sink_slew"].sum()
            + result["driver_net_cap"].sum()
        ).backward()
        assert state.z_param.grad.device == device
        assert state.bsu_index_param.grad.device == device
        assert driver_arrival.grad.device == device
        assert driver_slew.grad.device == device
        assert edge_resistance.grad.device == device
        assert edge_capacitance.grad.device == device
        assert bool((edge_resistance.grad != 0.0).any())
        assert bool((edge_capacitance.grad != 0.0).any())

    def _assert_cuda_cap_only_matches_full_transfer(self, initial_z, initial_bsu_index):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        device = torch.device("cuda", torch.cuda.current_device())
        net = self._branch_net()

        def make_state_and_inputs():
            state = build_segment_count_state(
                [net],
                buffer_main_type_index=7,
                legal_buffer_count=4,
                max_repeater_count=3,
                initial_z=initial_z,
                initial_bsu_index=initial_bsu_index,
                dtype=torch.float64,
                device=device,
            )
            tables = tuple(
                table.to(device=device)
                for table in self._tables(len(state.segment_rows))
            )
            return state, build_segment_count_timing_inputs(
                [net], state, dtype=torch.float64, device=device
            ), tables

        cap_state, cap_inputs, cap_tables = make_state_and_inputs()
        full_state, full_inputs, full_tables = make_state_and_inputs()
        device_lut = default_fake_buffer_device(dtype=torch.float64)
        cap_runtime_profile = {}
        cap = segment_count_driver_cap_native_cuda_autograd(
            cap_inputs,
            segment_state=cap_state,
            per_size_input_cap=cap_tables[0],
            buffer_input_cap_by_size=device_lut.input_cap_by_size,
            runtime_profile=cap_runtime_profile,
        )
        full = segment_count_forward_cuda_segment_transfer_explicit_autograd(
            full_inputs,
            segment_state=full_state,
            per_size_input_cap=full_tables[0],
            per_size_delay=full_tables[1],
            per_size_output_slew=full_tables[2],
            driver_arrival=torch.tensor(
                [4.0], dtype=torch.float64, device=device, requires_grad=True
            ),
            driver_slew=torch.tensor(
                [0.2], dtype=torch.float64, device=device, requires_grad=True
            ),
            buffer_device_lut=device_lut,
        )
        cap["driver_net_cap"].sum().backward()
        full["driver_net_cap"].sum().backward()

        torch.testing.assert_close(
            cap["driver_net_cap"], full["driver_net_cap"], atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            cap_state.z_param.grad,
            full_state.z_param.grad,
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            cap_state.bsu_index_param.grad,
            full_state.bsu_index_param.grad,
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertEqual(cap_runtime_profile["native_cap_backward_invocation_count"], 1)
        self.assertGreater(
            cap_runtime_profile["native_cap_backward_temporary_allocation_bytes"],
            0,
        )
        self.assertGreaterEqual(
            cap_runtime_profile["native_cap_backward_temporary_allocation_cpu_ms"],
            0.0,
        )

    def test_cuda_cap_only_matches_full_transfer_root_cap_and_vjp(self):
        for initial_z in (0.0, 1.0, 2.0, 3.0, 1.25):
            for initial_bsu_index in (0.0, 1.0, 1.5):
                self._assert_cuda_cap_only_matches_full_transfer(
                    initial_z,
                    initial_bsu_index,
                )

    def test_cuda_cap_only_root_cap_gradient_matches_finite_difference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        device = torch.device("cuda", torch.cuda.current_device())
        net = self._branch_net()
        device_lut = default_fake_buffer_device(dtype=torch.float64)

        def evaluate(initial_z, initial_bsu_index, *, backward=False):
            state = build_segment_count_state(
                [net],
                buffer_main_type_index=7,
                legal_buffer_count=4,
                max_repeater_count=3,
                initial_z=1.25,
                initial_bsu_index=0.5,
                dtype=torch.float64,
                device=device,
            )
            with torch.no_grad():
                state.z_param[0] = float(initial_z)
                state.bsu_index_param[0] = float(initial_bsu_index)
            inputs = build_segment_count_timing_inputs(
                [net], state, dtype=torch.float64, device=device
            )
            tables = tuple(
                table.to(device=device)
                for table in self._tables(len(state.segment_rows))
            )
            result = segment_count_driver_cap_native_cuda_autograd(
                inputs,
                segment_state=state,
                per_size_input_cap=tables[0],
                buffer_input_cap_by_size=device_lut.input_cap_by_size,
            )
            objective = result["driver_net_cap"].sum()
            if backward:
                objective.backward()
                return (
                    float(objective.detach().cpu().item()),
                    float(state.z_param.grad[0].detach().cpu().item()),
                    float(state.bsu_index_param.grad[0].detach().cpu().item()),
                )
            return float(objective.detach().cpu().item())

        epsilon = 1.0e-4
        _, grad_z, grad_bsu = evaluate(1.25, 0.5, backward=True)
        finite_z = (evaluate(1.25 + epsilon, 0.5) - evaluate(1.25 - epsilon, 0.5)) / (
            2.0 * epsilon
        )
        finite_bsu = (evaluate(1.25, 0.5 + epsilon) - evaluate(1.25, 0.5 - epsilon)) / (
            2.0 * epsilon
        )
        self.assertAlmostEqual(grad_z, finite_z, places=6)
        self.assertAlmostEqual(grad_bsu, finite_bsu, places=6)

    def test_cuda_extension_uses_current_stream_and_aten_scratch_storage(self):
        repo_root = Path(__file__).resolve().parents[3]
        source_root = repo_root / "dreamplace" / "ops" / "net_subgraph_timing" / "src"
        wrapper = (source_root / "segment_transfer_cuda.cpp").read_text(encoding="utf-8")
        kernel = (source_root / "segment_transfer_cuda_kernel.cu").read_text(
            encoding="utf-8"
        )

        self.assertIn("CUDAGuard", wrapper)
        self.assertIn("getCurrentCUDAStream", wrapper)
        self.assertIn("at::zeros_like(node_load)", wrapper)
        self.assertIn("<<<blocks, threads, 0, stream>>>", kernel)
        self.assertNotIn("cudaMalloc", kernel)
        self.assertNotIn("cudaFree", kernel)

    def test_cuda_explicit_autograd_runs_on_nondefault_device_and_stream(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires at least two CUDA devices")
        source_device = 0
        target_device = torch.device("cuda", 1)
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
            device=target_device,
        )
        tables = tuple(
            table.to(device=target_device)
            for table in self._tables(len(state.segment_rows))
        )
        inputs = build_segment_count_timing_inputs(
            [net],
            state,
            dtype=torch.float64,
            device=target_device,
        )
        driver_arrival = torch.tensor(
            [4.0], dtype=torch.float64, device=target_device, requires_grad=True
        )
        driver_slew = torch.tensor(
            [0.2], dtype=torch.float64, device=target_device, requires_grad=True
        )
        edge_resistance = inputs["edge_resistance"].clone().requires_grad_(True)
        edge_capacitance = inputs["edge_capacitance"].clone().requires_grad_(True)
        torch.cuda.synchronize(target_device)
        previous_stream = torch.cuda.current_stream(target_device)
        stream = torch.cuda.Stream(device=target_device)
        completion = torch.cuda.Event()
        try:
            torch.cuda.set_stream(stream)
            torch.cuda.set_device(source_device)
            self.assertEqual(torch.cuda.current_device(), source_device)
            result = segment_count_forward_cuda_segment_transfer_explicit_autograd(
                inputs,
                segment_state=state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                buffer_device_lut=default_fake_buffer_device(dtype=torch.float64),
                edge_resistance_override=edge_resistance,
                edge_capacitance_override=edge_capacitance,
                canonical_equal_spacing=True,
            )
            (
                result["sink_arrival"].sum()
                + result["sink_slew"].sum()
                + result["driver_net_cap"].sum()
            ).backward()
            completion.record(stream)
        finally:
            torch.cuda.set_stream(previous_stream)
            torch.cuda.set_device(source_device)
        completion.synchronize()

        self.assertEqual(torch.cuda.current_device(), source_device)
        for key in ("sink_arrival", "sink_slew", "sink_load", "driver_net_cap"):
            self.assertEqual(result[key].device, target_device)
            self.assertTrue(torch.isfinite(result[key]).all())
        for grad in (
            state.z_param.grad,
            state.bsu_index_param.grad,
            driver_arrival.grad,
            driver_slew.grad,
            edge_resistance.grad,
            edge_capacitance.grad,
        ):
            self.assertEqual(grad.device, target_device)
            self.assertTrue(torch.isfinite(grad).all())

    def test_cuda_forward_rejects_mixed_device_inputs(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires at least two CUDA devices")
        source_device = torch.device("cuda", 0)
        target_device = torch.device("cuda", 1)
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
            device=target_device,
        )
        inputs = build_segment_count_timing_inputs(
            [net],
            state,
            dtype=torch.float64,
            device=target_device,
        )
        per_size_input_cap, per_size_delay, per_size_output_slew = (
            table.to(device=target_device)
            for table in self._tables(len(state.segment_rows))
        )
        load_input_cap = _interpolate_size_tables(
            state.bsu_index(),
            per_size_input_cap=per_size_input_cap,
            per_size_delay=per_size_delay,
            per_size_output_slew=per_size_output_slew,
        )["buffer_input_cap"]
        device_lut = default_fake_buffer_device(dtype=torch.float64).tensors(
            dtype=torch.float64,
            device=target_device,
        )

        with self.assertRaisesRegex(RuntimeError, "driver_arrival must be on"):
            segment_count_transfer_forward_native_cuda(
                net_topo_start=inputs["net_topo_start"],
                flat_topo_node_id=inputs["flat_topo_node_id"],
                edge_start=inputs["edge_start"],
                edge_parent_compact_id=inputs["edge_parent_compact_id"],
                edge_child_compact_id=inputs["edge_child_compact_id"],
                edge_resistance=inputs["edge_resistance"],
                edge_capacitance=inputs["edge_capacitance"],
                node_capacitance=inputs["node_capacitance"],
                edge_to_segment_id=inputs["edge_to_segment_id"],
                driver_arrival=torch.tensor(
                    [4.0], dtype=torch.float64, device=source_device
                ),
                driver_slew=torch.tensor(
                    [0.2], dtype=torch.float64, device=target_device
                ),
                z_value=state.z_param,
                bsu_index=state.bsu_index_param,
                load_input_cap=load_input_cap,
                buffer_input_cap_by_size=device_lut["input_cap_by_size"],
                buffer_slew_axis=device_lut["input_slew_axis"],
                buffer_load_axis=device_lut["output_load_axis"],
                buffer_delay_lut=device_lut["delay_lut"],
                buffer_output_slew_lut=device_lut["output_slew_lut"],
                parent_cap_fraction=inputs["parent_cap_fraction"],
                child_cap_fraction=inputs["child_cap_fraction"],
                segment_sub_resistance_fraction=inputs[
                    "segment_sub_resistance_fraction"
                ],
                segment_retained_upstream_cap=torch.zeros_like(state.z_param),
                sink_node_id=inputs["sink_node_id"],
                sink_net_index=inputs["sink_net_id"],
                sink_node_compact_id=inputs["sink_node_compact_id"],
            )

    def test_native_explicit_autograd_matches_prepared_python_gradients(self):
        net = self._branch_net()
        reference_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=0.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        native_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=0.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(reference_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], reference_state, dtype=torch.float64)
        driver_arrival_ref = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew_ref = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        driver_arrival_native = driver_arrival_ref.detach().clone().requires_grad_(True)
        driver_slew_native = driver_slew_ref.detach().clone().requires_grad_(True)

        reference = segment_count_prepared_relaxed_timing(
            inputs,
            segment_state=reference_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_ref,
            driver_slew=driver_slew_ref,
        )
        native = segment_count_forward_native_explicit_autograd(
            inputs,
            segment_state=native_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival_native,
            driver_slew=driver_slew_native,
        )

        reference_loss = (
            reference["sink_arrival"].sum()
            + 0.7 * reference["sink_slew"].sum()
            + 0.3 * reference["sink_load"].sum()
        )
        native_loss = (
            native["sink_arrival"].sum()
            + 0.7 * native["sink_slew"].sum()
            + 0.3 * native["sink_load"].sum()
        )
        reference_loss.backward()
        native_loss.backward()

        for key in ("sink_arrival", "sink_slew", "sink_load"):
            torch.testing.assert_close(native[key], reference[key], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(native_state.z_param.grad, reference_state.z_param.grad)
        torch.testing.assert_close(
            native_state.bsu_index_param.grad,
            reference_state.bsu_index_param.grad,
        )
        torch.testing.assert_close(driver_arrival_native.grad, driver_arrival_ref.grad)
        torch.testing.assert_close(driver_slew_native.grad, driver_slew_ref.grad)

    def test_native_explicit_autograd_keeps_general_z_domain(self):
        self._assert_explicit_autograd_matches_prepared_python(initial_z=1.25)

    def test_native_explicit_autograd_matches_prepared_python_for_multi_repeater_z(self):
        self._assert_explicit_autograd_matches_prepared_python(initial_z=2.25)

    def test_native_explicit_autograd_matches_prepared_python_at_z_boundaries(self):
        for initial_z in (0.0, 1.0, 2.0, 3.0):
            self._assert_explicit_autograd_matches_prepared_python(initial_z=initial_z)

    def test_native_explicit_autograd_matches_prepared_python_outside_z_bounds(self):
        self._assert_explicit_autograd_matches_prepared_python(initial_z=-0.25)
        self._assert_explicit_autograd_matches_prepared_python(initial_z=3.25)

    def test_native_explicit_autograd_matches_prepared_python_at_bsu_boundaries(self):
        for bsu_index in (0.0, 3.0):
            net = self._branch_net()
            reference_state = build_segment_count_state(
                [net],
                buffer_main_type_index=7,
                legal_buffer_count=4,
                max_repeater_count=3,
                initial_z=1.25,
                initial_bsu_index=bsu_index,
                dtype=torch.float64,
            )
            native_state = build_segment_count_state(
                [net],
                buffer_main_type_index=7,
                legal_buffer_count=4,
                max_repeater_count=3,
                initial_z=1.25,
                initial_bsu_index=bsu_index,
                dtype=torch.float64,
            )
            tables = self._tables(len(reference_state.segment_rows))
            inputs = build_segment_count_timing_inputs([net], reference_state, dtype=torch.float64)
            driver_arrival_ref = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
            driver_slew_ref = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
            driver_arrival_native = driver_arrival_ref.detach().clone().requires_grad_(True)
            driver_slew_native = driver_slew_ref.detach().clone().requires_grad_(True)

            reference = segment_count_prepared_relaxed_timing(
                inputs,
                segment_state=reference_state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_arrival=driver_arrival_ref,
                driver_slew=driver_slew_ref,
            )
            native = segment_count_forward_native_explicit_autograd(
                inputs,
                segment_state=native_state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_arrival=driver_arrival_native,
                driver_slew=driver_slew_native,
            )

            reference_loss = (
                reference["sink_arrival"].sum()
                + 0.7 * reference["sink_slew"].sum()
                + 0.3 * reference["sink_load"].sum()
            )
            native_loss = (
                native["sink_arrival"].sum()
                + 0.7 * native["sink_slew"].sum()
                + 0.3 * native["sink_load"].sum()
            )
            reference_loss.backward()
            native_loss.backward()

            for key in ("sink_arrival", "sink_slew", "sink_load"):
                torch.testing.assert_close(native[key], reference[key], atol=1e-7, rtol=1e-7)
            torch.testing.assert_close(native_state.z_param.grad, reference_state.z_param.grad)
            torch.testing.assert_close(
                native_state.bsu_index_param.grad,
                reference_state.bsu_index_param.grad,
            )
            torch.testing.assert_close(driver_arrival_native.grad, driver_arrival_ref.grad)
            torch.testing.assert_close(driver_slew_native.grad, driver_slew_ref.grad)

    def test_native_explicit_autograd_does_not_recompute_for_general_z(self):
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=1.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)

        old_recompute = segment_count_native._recompute_backward_grads

        def fail_if_called(**kwargs):
            raise AssertionError("general-z explicit native backward must not fallback to recompute")

        segment_count_native._recompute_backward_grads = fail_if_called
        try:
            native = segment_count_forward_native_explicit_autograd(
                inputs,
                segment_state=state,
                per_size_input_cap=tables[0],
                per_size_delay=tables[1],
                per_size_output_slew=tables[2],
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
            )
            loss = (
                native["sink_arrival"].sum()
                + 0.7 * native["sink_slew"].sum()
                + 0.3 * native["sink_load"].sum()
            )
            loss.backward()
        finally:
            segment_count_native._recompute_backward_grads = old_recompute

    def test_native_cpp_explicit_backward_matches_python_explicit_backward(self):
        net = self._branch_net()
        python_state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=0.25,
            initial_bsu_index=1.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(python_state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], python_state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        python_result = segment_count_forward_native_explicit_autograd(
            inputs,
            segment_state=python_state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
        )
        python_loss = (
            python_result["sink_arrival"].sum()
            + 0.7 * python_result["sink_slew"].sum()
            + 0.3 * python_result["sink_load"].sum()
        )
        python_loss.backward()

        buffer_tensors = _interpolate_size_tables(
            python_state.bsu_index(),
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
        )
        forward_result = segment_count_forward_native(
            net_topo_start=inputs["net_topo_start"],
            flat_topo_node_id=inputs["flat_topo_node_id"],
            edge_start=inputs["edge_start"],
            edge_parent_compact_id=inputs["edge_parent_compact_id"],
            edge_child_compact_id=inputs["edge_child_compact_id"],
            edge_resistance=inputs["edge_resistance"],
            edge_capacitance=inputs["edge_capacitance"],
            node_capacitance=inputs["node_capacitance"],
            edge_to_segment_id=inputs["edge_to_segment_id"],
            driver_arrival=driver_arrival.detach(),
            driver_slew=driver_slew.detach(),
            z_value=python_state.z_value().detach(),
            buffer_input_cap=buffer_tensors["buffer_input_cap"].detach(),
            buffer_delay=buffer_tensors["buffer_delay"].detach(),
            buffer_output_slew=buffer_tensors["buffer_output_slew"].detach(),
            parent_cap_fraction=inputs["parent_cap_fraction"],
            child_cap_fraction=inputs["child_cap_fraction"],
            segment_sub_resistance_fraction=inputs["segment_sub_resistance_fraction"],
            sink_node_id=inputs["sink_node_id"],
            sink_net_index=inputs["sink_net_id"],
            sink_node_compact_id=inputs["sink_node_compact_id"],
        )
        cpp_grads = native_cpp.segment_count_backward(
            inputs["net_topo_start"].contiguous(),
            inputs["edge_start"].contiguous(),
            inputs["edge_parent_compact_id"].contiguous(),
            inputs["edge_child_compact_id"].contiguous(),
            inputs["edge_resistance"].contiguous(),
            inputs["edge_capacitance"].contiguous(),
            inputs["edge_to_segment_id"].contiguous(),
            driver_arrival.detach().contiguous(),
            driver_slew.detach().contiguous(),
            python_state.z_value().detach().contiguous(),
            python_state.bsu_index().detach().contiguous(),
            tables[0].contiguous(),
            tables[1].contiguous(),
            tables[2].contiguous(),
            buffer_tensors["buffer_input_cap"].detach().contiguous(),
            buffer_tensors["buffer_delay"].detach().contiguous(),
            buffer_tensors["buffer_output_slew"].detach().contiguous(),
            inputs["parent_cap_fraction"].contiguous(),
            inputs["child_cap_fraction"].contiguous(),
            inputs["segment_sub_resistance_fraction"].contiguous(),
            inputs["sink_node_compact_id"].contiguous(),
            forward_result["node_load"].detach().contiguous(),
            forward_result["node_slew"].detach().contiguous(),
            forward_result["effective_node_cap"].detach().contiguous(),
            torch.zeros_like(forward_result["segment_delay"]),
            torch.zeros_like(forward_result["segment_output_slew"]),
            torch.zeros_like(forward_result["segment_upstream_visible_input_cap"]),
            torch.ones_like(forward_result["sink_arrival"]),
            torch.full_like(forward_result["sink_slew"], 0.7),
            torch.full_like(forward_result["sink_load"], 0.3),
        )

        torch.testing.assert_close(cpp_grads["grad_z"], python_state.z_param.grad)
        torch.testing.assert_close(cpp_grads["grad_bsu"], python_state.bsu_index_param.grad)
        torch.testing.assert_close(cpp_grads["grad_driver_arrival"], driver_arrival.grad)
        torch.testing.assert_close(cpp_grads["grad_driver_slew"], driver_slew.grad)

    def test_native_explicit_autograd_zeroes_gradients_outside_clamped_domains(self):
        net = self._branch_net()
        state = build_segment_count_state(
            [net],
            buffer_main_type_index=7,
            legal_buffer_count=4,
            max_repeater_count=3,
            initial_z=-0.25,
            initial_bsu_index=-0.5,
            dtype=torch.float64,
        )
        tables = self._tables(len(state.segment_rows))
        inputs = build_segment_count_timing_inputs([net], state, dtype=torch.float64)
        driver_arrival = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        driver_slew = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)

        native = segment_count_forward_native_explicit_autograd(
            inputs,
            segment_state=state,
            per_size_input_cap=tables[0],
            per_size_delay=tables[1],
            per_size_output_slew=tables[2],
            driver_arrival=driver_arrival,
            driver_slew=driver_slew,
        )
        loss = (
            native["sink_arrival"].sum()
            + 0.7 * native["sink_slew"].sum()
            + 0.3 * native["sink_load"].sum()
        )
        loss.backward()

        torch.testing.assert_close(state.z_param.grad, torch.zeros_like(state.z_param))
        torch.testing.assert_close(
            state.bsu_index_param.grad,
            torch.zeros_like(state.bsu_index_param),
        )


if __name__ == "__main__":
    unittest.main()
