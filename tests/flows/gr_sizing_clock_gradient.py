"""Check real exported Liberty FF arcs at the normal GR sizing boundary."""

import torch


def check_clock_gradient(model):
    db, data = model.placedb, model.data_collections
    timing = model.op_collections.timing_propagation_op
    assert timing._resolve_surrogate_mode(model.params.timing_surrogate_mode) == (True, True)

    def names(values):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values]

    node = names(db.node_names).index("launch")
    pins = names(db.pin_names)
    clock, output = pins.index("launch:CK"), pins.index("launch:Q")
    # Level-0 arc sources address the clock-pin domain; sinks address PyDB
    # physical pins. The native adapter intentionally keeps these IDs separate.
    clock_index = timing.clock_pins.tolist().index(clock)
    arcs = timing.flat_inst_arcs_by_level[
        timing.flat_inst_arcs_by_level_start[0] : timing.flat_inst_arcs_by_level_start[1]
    ]
    matching = (arcs[:, 0] == clock_index) & (arcs[:, 1] == output)
    assert int(matching.sum()) == 1 and int(arcs[matching][0, 5]) == 1, {
        "clock_output": [clock, output],
        "level0_arcs": arcs.tolist(),
        "pins": pins,
    }
    caps = model.pin_caps_op(data.inst_libcell_offset, data)
    groups = model.op_collections.elmore_delay_op(*caps)
    rise, fall = groups[1]["rise"].detach(), groups[1]["fall"].detach()
    sizes, probabilities = data.get_size_var().detach(), data.get_vt_var().detach()
    assert probabilities.shape[1] == 2
    cell_names = names(db.flat_libcell_names)
    vt_codes = {
        suffix: int(db.flat_libcell_info[cell_names.index(f"DFFX1H7{suffix}"), 3])
        for suffix in ("R", "L")
    }
    assert set(vt_codes.values()) == {0, 1}
    original_size, original_vt = timing.size_var_getter, timing.vt_var_getter
    node_id = torch.tensor([node], dtype=torch.long)
    values = torch.tensor([1.4, 0.3], dtype=sizes.dtype, requires_grad=True)

    def evaluate(point):
        size = sizes.index_copy(0, node_id, point[:1])
        vt = probabilities.index_copy(0, node_id, torch.stack((1 - point[1], point[1]))[None])
        timing.size_var_getter, timing.vt_var_getter = lambda: size, lambda: vt
        zeros = torch.zeros_like(rise)
        _, raat, _, _, _ = timing.calculate_clk2q_aat(
            zeros, zeros, zeros, zeros, rise, fall, use_surrogate=True
        )
        return raat[output]

    try:
        delay = evaluate(values)
        gradient = torch.autograd.grad(delay, values)[0]
        assert torch.isfinite(delay) and bool(torch.isfinite(gradient).all())
        assert gradient[0] < 0
        # Slot order is exported metadata. R is slower than L in this fixture;
        # increasing probability of slot 1 has the corresponding signed effect.
        assert gradient[1] * (1 if vt_codes["R"] == 1 else -1) > 0, gradient
        fd = []
        for index in range(2):
            plus, minus = values.detach().clone(), values.detach().clone()
            plus[index] += 1e-4
            minus[index] -= 1e-4
            fd.append((evaluate(plus) - evaluate(minus)) / (plus[index] - minus[index]))
        fd = torch.stack(fd)
        torch.testing.assert_close(gradient, fd, rtol=1e-3, atol=1e-6)
        return {
            "arc": arcs[matching][0].tolist(),
            "clock_pin": clock,
            "clock_input_index": clock_index,
            "output_pin": output,
            "delay_ps": float(delay.detach()),
            "size_vt_gradient": gradient.tolist(),
            "size_vt_finite_difference": fd.tolist(),
            "root_load_pf": float(rise[output]),
            "vt_codes": vt_codes,
        }
    finally:
        timing.size_var_getter, timing.vt_var_getter = original_size, original_vt
