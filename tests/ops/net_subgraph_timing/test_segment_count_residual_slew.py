import torch
from dreamplace.ops.net_subgraph_timing.segment_count_dynamic_provider import SegmentCountDynamicNetProvider

def _residual(base_slew, current_slew, zero_slew):
    provider = object.__new__(SegmentCountDynamicNetProvider)
    provider.metadata = {}
    pin = torch.tensor([0, 1], dtype=torch.long)
    arrival = torch.tensor([100.0, 200.0], dtype=torch.float64)
    return provider._residualize_sink_timing(
        static_pin_aat=arrival,
        static_pin_slew=base_slew,
        result={"sink_node_id": pin, "sink_arrival": arrival + 5.0, "sink_slew": current_slew},
        zero_result={"sink_node_id": pin, "sink_arrival": arrival, "sink_slew": zero_slew},
    )

def test_segment_residual_slew_stays_physical_and_preserves_valid_gradients():
    # Real mempool probe: 107.16043 + 57.845596 - 182.103851 < 0.
    current = torch.tensor([57.845596, 80.0], dtype=torch.float64, requires_grad=True)
    base = torch.tensor([107.160430, 100.0], dtype=torch.float64)
    zero = torch.tensor([182.103851, 120.0], dtype=torch.float64)
    result = _residual(base, current, zero)
    torch.testing.assert_close(result['sink_slew'], torch.tensor([0.0, 60.0], dtype=torch.float64))
    result['sink_slew'].sum().backward()
    torch.testing.assert_close(current.grad, torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(result['sink_arrival'], torch.tensor([105.0, 205.0], dtype=torch.float64))

def test_segment_residual_slew_zero_state_preserves_static_baseline():
    base = torch.tensor([107.160430, 100.0], dtype=torch.float64)
    zero = torch.tensor([182.103851, 120.0], dtype=torch.float64)
    result = _residual(base, zero, zero)
    torch.testing.assert_close(result['sink_slew'], base)


def test_direct_segment_driver_cap_uses_current_load_and_gradient():
    provider = object.__new__(SegmentCountDynamicNetProvider)
    provider.metadata = {}
    pin = torch.tensor([0, 1], dtype=torch.long)
    current = torch.tensor([2.0, 3.0], requires_grad=True)
    zero = torch.tensor([4.0, 5.0])
    static = torch.tensor([4.5, 4.0])
    cap = {"driver_pin_id": pin, "driver_net_cap": current}
    baseline = {"driver_pin_id": pin, "driver_net_cap": zero}
    provider.driver_cap_mode = "residual"
    residual = provider._residualize_driver_cap(static, cap, baseline)
    torch.testing.assert_close(residual["driver_net_cap"], torch.tensor([2.5, 2.0]))
    provider.driver_cap_mode = "direct"
    direct = provider._residualize_driver_cap(static, cap, baseline)
    torch.testing.assert_close(direct["driver_net_cap"], current)
    direct["driver_net_cap"].sum().backward()
    torch.testing.assert_close(current.grad, torch.ones_like(current))


def test_zero_cap_residual_preserves_double_static_load_and_small_changes():
    provider = object.__new__(SegmentCountDynamicNetProvider)
    provider.metadata = {}
    provider.driver_cap_mode = "residual"
    baseline = {
        "driver_pin_id": torch.tensor([0, 1]),
        "driver_net_cap": torch.tensor([0.04343845, 0.04672001]),
    }
    static = torch.tensor([0.043438451234, 0.046720012345], dtype=torch.float64, requires_grad=True)

    def evaluate(load):
        return provider._residualize_driver_cap(load, baseline, baseline)["driver_net_cap"]

    torch.testing.assert_close(evaluate(static), static, atol=0, rtol=0)
    assert torch.autograd.gradcheck(evaluate, (static,), eps=1e-9, atol=1e-7, rtol=1e-7)


def test_zero_timing_residual_preserves_double_static_arrival_and_slew():
    provider = object.__new__(SegmentCountDynamicNetProvider)
    provider.metadata = {}
    baseline = {
        "sink_node_id": torch.tensor([0, 1]),
        "sink_arrival": torch.tensor([100.0, 200.0]),
        "sink_slew": torch.tensor([70.0, 80.0]),
    }
    arrival = torch.tensor([100.123456789, 200.987654321], dtype=torch.float64, requires_grad=True)
    slew = torch.tensor([69.123456789, 89.987654321], dtype=torch.float64, requires_grad=True)

    def evaluate(aat, transition):
        result = provider._residualize_sink_timing(
            static_pin_aat=aat, static_pin_slew=transition,
            result=baseline, zero_result=baseline,
        )
        return result["sink_arrival"], result["sink_slew"]

    torch.testing.assert_close(evaluate(arrival, slew), (arrival, slew), atol=0, rtol=0)
    assert torch.autograd.gradcheck(evaluate, (arrival, slew), eps=1e-6, atol=1e-7, rtol=1e-7)
