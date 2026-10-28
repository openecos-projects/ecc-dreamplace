"""Evaluate min RC on the current tree and the current size/VT pin caps."""

from dataclasses import dataclass

import torch

from dreamplace.ops.size_interpolated_pin.candidate_weights import sizing_pin_candidate_weights


@dataclass
class EarlyRCInputs:
    delays: dict[str, torch.Tensor]
    impulses: dict[str, torch.Tensor]
    caps: dict[str, torch.Tensor]
    pin_caps: dict[str, torch.Tensor]
    gradients_enabled: bool = True


class HoldRC:
    def __init__(self, data):
        self.tables = tuple(torch.from_numpy(cap.copy()) for cap in data.min_pin_caps)
        self.output_caps = tuple(torch.from_numpy(cap.copy()) for cap in data.output_caps)

    def pin_caps(self, model, base):
        data = model.data_collections
        mask, ids = model.pin_2_libpin_ids(data.inst_libcell_offset, data)
        # The placement database keeps static pin/master maps on CPU even when
        # the timing lane runs on CUDA.  Index tensors must follow the cap
        # table/device used by this timing evaluation.
        mask = mask.to(device=base.device)
        ids = ids.to(device=base.device)
        tables = tuple(table.to(device=base.device, dtype=base.dtype) for table in self.tables)
        caps = tuple(torch.zeros_like(base) for _ in range(2))
        for cap, table in zip(caps, tables, strict=True):
            cap[mask] = table[ids]
        size, vt = None, None
        if model._current_timing_surrogate_mode() != "lut_only":
            size, vt = data.get_size_var(), data.get_vt_var()
        if size is not None and vt is not None:
            cache = model._get_sizing_pin_cap_static_cache(data)
            if cache is None:
                raise RuntimeError(
                    "min sizing capacitance requires the canonical pin candidate cache"
                )
            if cache["has_sizeable"]:
                device = base.device
                nodes = cache["pin_nodes"].to(device=device)
                sizeable_pin_ids = cache["sizeable_pin_ids"].to(device=device)
                sizes = size.float().to(device=device)[nodes]
                vts = vt.float().to(device=device)[nodes]
                for group in cache["groups"]:
                    selected = group["main_mask"].to(device=device)
                    candidate_sizes = group["candidate_sizes"].to(
                        device=device, dtype=sizes.dtype
                    )
                    candidate_vts = group["candidate_vts"].to(device=device)
                    candidate_libpin_ids = group["candidate_libpin_ids"].to(
                        device=device, dtype=torch.long
                    )
                    weights = sizing_pin_candidate_weights(
                        sizes[selected],
                        vts[selected],
                        candidate_sizes,
                        candidate_vts,
                    )
                    pins = sizeable_pin_ids[selected]
                    for cap, table in zip(caps, tables, strict=True):
                        cap[pins] = (weights * table[candidate_libpin_ids]).sum(1)
        from dreamplace.ops.timing_propagation.action_pin_caps import apply_pin_caps

        caps = apply_pin_caps(model, caps, data, tables)
        # SDC output loads are independent of Liberty's min/max pin caps.
        # Add them after interpolation, matching the canonical late cap path.
        for cap, output in zip(caps, self.output_caps, strict=True):
            end_points = data.end_points.to(device=cap.device)
            cap[end_points] += output.to(cap)
        return base, *caps

    def evaluate(self, model, x, y, base):
        gradients_enabled = torch.is_grad_enabled() and model.hold_objective.active
        with torch.set_grad_enabled(gradients_enabled):
            caps = tuple(
                model._timing_rc_cap_input_tensor(cap.to(x.device))
                for cap in self.pin_caps(model, base)
            )
            rc = model.op_collections.elmore_delay_op
            if model.placedb.gr_sizing is not None:
                _, loads, delays, _, _, impulses = rc(*caps)
            else:
                data = model.data_collections
                _, loads, delays, _, _, impulses = rc(
                    x,
                    y,
                    data.net_flat_topo_sort,
                    data.net_flat_topo_sort_start,
                    data.pin_fa,
                    data.flat_pin_to_start,
                    data.flat_pin_to,
                    data.flat_pin_from,
                    *caps,
                )
            return EarlyRCInputs(
                delays, impulses, loads, dict(zip(("base", "rise", "fall"), caps)),
                gradients_enabled,
            )
