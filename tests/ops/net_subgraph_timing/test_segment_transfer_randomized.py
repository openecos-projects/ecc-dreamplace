import math
import random

import torch

from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    BufferDeviceLut,
    SegmentTransferInput,
    analytic_segment_transfer,
)


def _affine_device(dtype=torch.float64):
    slew_axis = torch.tensor([0.0, 50.0, 150.0, 400.0], dtype=dtype)
    load_axis = torch.tensor([0.0, 10.0, 50.0, 150.0], dtype=dtype)
    input_cap_by_size = torch.tensor([0.10, 0.25, 0.60], dtype=dtype)
    delay = torch.empty(3, 4, 4, dtype=dtype)
    output_slew = torch.empty(3, 4, 4, dtype=dtype)
    for size in range(3):
        for slew_index, slew in enumerate(slew_axis):
            for load_index, load in enumerate(load_axis):
                delay[size, slew_index, load_index] = (
                    1.5 + 0.7 * size + 0.03 * slew + 0.11 * load
                )
                output_slew[size, slew_index, load_index] = (
                    0.8 + 0.4 * size + 0.09 * slew + 0.05 * load
                )
    return BufferDeviceLut(
        input_cap_by_size=input_cap_by_size,
        input_slew_axis=slew_axis,
        output_load_axis=load_axis,
        delay_lut=delay,
        output_slew_lut=output_slew,
        source="affine_randomized_test_lut",
    )


def _interp_size_affine(values, bsu_index):
    clipped = max(0.0, min(float(bsu_index), float(len(values) - 1)))
    lo = int(math.floor(clipped))
    hi = min(lo + 1, len(values) - 1)
    alpha = clipped - lo
    return (1.0 - alpha) * float(values[lo]) + alpha * float(values[hi])


def _affine_delay(*, bsu_index, input_slew, output_load):
    return 1.5 + 0.7 * float(bsu_index) + 0.03 * float(input_slew) + 0.11 * float(output_load)


def _affine_output_slew(*, bsu_index, input_slew, output_load):
    return 0.8 + 0.4 * float(bsu_index) + 0.09 * float(input_slew) + 0.05 * float(output_load)


def _wire_slew(input_slew, wire_delay):
    return math.sqrt(float(input_slew) ** 2 + (math.log(10.0) * float(wire_delay)) ** 2)


def _random_split_fractions(rng, count):
    if count == 0:
        return (1.0,)
    cuts = sorted(rng.uniform(0.02, 0.98) for _ in range(count))
    fractions = []
    previous = 0.0
    for cut in cuts:
        fractions.append(cut - previous)
        previous = cut
    fractions.append(1.0 - previous)
    return tuple(fractions)


def test_randomized_n0_matches_independent_elmore_formula():
    rng = random.Random(1701)
    for _ in range(100):
        input_arrival = rng.uniform(0.0, 100.0)
        input_slew = rng.uniform(0.01, 80.0)
        downstream_load = rng.uniform(0.001, 30.0)
        edge_resistance = rng.uniform(0.001, 15.0)
        edge_capacitance = rng.uniform(0.0, 5.0)

        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(input_arrival, dtype=torch.float64),
                input_slew=torch.tensor(input_slew, dtype=torch.float64),
                downstream_load=torch.tensor(downstream_load, dtype=torch.float64),
                edge_resistance=torch.tensor(edge_resistance, dtype=torch.float64),
                edge_capacitance=torch.tensor(edge_capacitance, dtype=torch.float64),
                repeater_count=0,
            )
        )

        expected_delay = edge_resistance * (downstream_load + 0.5 * edge_capacitance)
        expected_slew = _wire_slew(input_slew, expected_delay)
        expected_load = downstream_load + edge_capacitance
        torch.testing.assert_close(
            result.segment_delay,
            torch.tensor(expected_delay, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.output_arrival,
            torch.tensor(input_arrival + expected_delay, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.output_slew,
            torch.tensor(expected_slew, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.upstream_visible_load,
            torch.tensor(expected_load, dtype=torch.float64),
        )


def test_randomized_multi_n_matches_independent_affine_lut_formula():
    rng = random.Random(2701)
    device = _affine_device()
    input_caps = [0.10, 0.25, 0.60]
    for _ in range(100):
        repeater_count = rng.randint(1, 4)
        input_arrival = rng.uniform(0.0, 100.0)
        input_slew = rng.uniform(0.01, 35.0)
        downstream_load = rng.uniform(0.001, 16.0)
        edge_resistance = rng.uniform(0.001, 12.0)
        edge_capacitance = rng.uniform(0.0, 4.0)
        split_fractions = _random_split_fractions(rng, repeater_count)
        bsu_index = rng.uniform(0.0, 2.0)
        retained_cap = rng.uniform(0.0, 1.0)

        result = analytic_segment_transfer(
            SegmentTransferInput(
                input_arrival=torch.tensor(input_arrival, dtype=torch.float64),
                input_slew=torch.tensor(input_slew, dtype=torch.float64),
                downstream_load=torch.tensor(downstream_load, dtype=torch.float64),
                edge_resistance=torch.tensor(edge_resistance, dtype=torch.float64),
                edge_capacitance=torch.tensor(edge_capacitance, dtype=torch.float64),
                upstream_retained_capacitance=torch.tensor(retained_cap, dtype=torch.float64),
                repeater_count=repeater_count,
                bsu_index=torch.tensor(bsu_index, dtype=torch.float64),
                split_fractions=split_fractions,
                buffer_device=device,
            )
        )

        buffer_input_cap = _interp_size_affine(input_caps, bsu_index)
        segment_resistances = [edge_resistance * fraction for fraction in split_fractions]
        segment_capacitances = [edge_capacitance * fraction for fraction in split_fractions]
        expected_delay = 0.0
        slew = input_slew
        expected_wire_delays = []
        expected_buffer_input_slews = []
        expected_buffer_output_loads = []
        expected_buffer_delays = []
        expected_buffer_output_slews = []
        for index in range(repeater_count):
            wire_delay = segment_resistances[index] * (
                buffer_input_cap + 0.5 * segment_capacitances[index]
            )
            expected_wire_delays.append(wire_delay)
            expected_delay += wire_delay
            slew = _wire_slew(slew, wire_delay)
            expected_buffer_input_slews.append(slew)
            output_load = (
                buffer_input_cap
                if index + 1 < repeater_count
                else downstream_load
            ) + segment_capacitances[index + 1]
            expected_buffer_output_loads.append(output_load)
            buffer_delay = _affine_delay(
                bsu_index=bsu_index,
                input_slew=slew,
                output_load=output_load,
            )
            buffer_output_slew = _affine_output_slew(
                bsu_index=bsu_index,
                input_slew=slew,
                output_load=output_load,
            )
            expected_buffer_delays.append(buffer_delay)
            expected_buffer_output_slews.append(buffer_output_slew)
            expected_delay += buffer_delay
            slew = buffer_output_slew
        downstream_wire_delay = segment_resistances[-1] * (
            downstream_load + 0.5 * segment_capacitances[-1]
        )
        expected_wire_delays.append(downstream_wire_delay)
        expected_delay += downstream_wire_delay
        expected_output_slew = _wire_slew(slew, downstream_wire_delay)
        expected_load = buffer_input_cap + segment_capacitances[0] + retained_cap

        torch.testing.assert_close(
            result.segment_delay,
            torch.tensor(expected_delay, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.output_arrival,
            torch.tensor(input_arrival + expected_delay, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.output_slew,
            torch.tensor(expected_output_slew, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.upstream_visible_load,
            torch.tensor(expected_load, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.diagnostics["buffer_input_slew"],
            torch.tensor(expected_buffer_input_slews[0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.diagnostics["buffer_delay"],
            torch.tensor(expected_buffer_delays[0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.diagnostics["buffer_output_slew"],
            torch.tensor(expected_buffer_output_slews[0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.stack(result.diagnostics["wire_delays"]),
            torch.tensor(expected_wire_delays, dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.stack(result.diagnostics["buffer_input_slews"]),
            torch.tensor(expected_buffer_input_slews, dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.stack(result.diagnostics["buffer_output_loads"]),
            torch.tensor(expected_buffer_output_loads, dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.stack(result.diagnostics["buffer_delays"]),
            torch.tensor(expected_buffer_delays, dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.stack(result.diagnostics["buffer_output_slews"]),
            torch.tensor(expected_buffer_output_slews, dtype=torch.float64),
        )
