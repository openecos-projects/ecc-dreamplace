"""Live hold/removal size and VT queries through the shared piecewise evaluator."""

import torch

from dreamplace.ops.cell_modeling.additional_lut_datasets import (
    register_piecewise_lut_dataset,
)


class HoldCheckModel:
    def __init__(self, timing, check_types):
        model = timing.cell_modeling_op
        data = model.data_collections
        arcs = data.flat_libarc_info.detach().cpu().long()
        cells = data.flat_libcell_info.detach().cpu()
        classes = torch.as_tensor(check_types.copy()).long()
        # Master arc ordering need not match. Match the same pin/sense/trigger,
        # check class and duplicate ordinal before interpolating size or VT.
        signatures, offsets, ranks = {}, [], {}
        for arc_id, arc in enumerate(arcs.tolist()):
            cell, main = arc[2], int(cells[arc[2], 1])
            signature = (arc[0], arc[1], arc[4], arc[5], int(classes[arc_id]))
            key = (cell, signature)
            rank = ranks.get(key, 0)
            ranks[key] = rank + 1
            family = signatures.setdefault(main, {})
            identity = (*signature, rank)
            offsets.append(family.setdefault(identity, len(family)))
        offsets = torch.tensor(offsets, device=timing.device, dtype=torch.long)
        self.offsets = offsets
        self.names = []
        max_offset = max(map(len, signatures.values()), default=1)
        for sense in ("r", "f"):
            name = f"{sense}_min_check"
            lut = getattr(timing.arcs_info, f"{sense}_check_luts")
            valid = lut.valid_arc_mask & torch.isin(
                classes.to(timing.device), classes.new_tensor([2, 4]).to(timing.device)
            )
            ids = torch.nonzero(valid, as_tuple=False).flatten()
            info = data.flat_libarc_info.to(timing.device)[ids]
            libcells = data.flat_libcell_info.to(timing.device)[info[:, 2].long()]
            if name not in model.piecewise_state:
                register_piecewise_lut_dataset(
                    model, name, f"{sense}_check_luts",
                    libcell_main_ids=libcells[:, 1].long(), arc_offsets=offsets[ids],
                    sizes=libcells[:, 2].float(), vts=libcells[:, 3].float(),
                    num_main_ids=model.num_libcell_main_ids, max_arc_offset=max_offset,
                    vt_order=model.vt_order_tensor.detach().cpu().tolist(), arc_indices=ids,
                )
            self.names.append(name)

    def query(self, timing, rows, first, second, edge):
        model = timing.cell_modeling_op
        state = timing._get_surrogate_state()
        if state is None:
            raise RuntimeError("hold check sizing requires current size/VT state")
        pins = rows[:, 1].long()
        nodes = timing.pin2node_map[pins].long()
        selected = timing.inst_is_sizeable[nodes].bool()
        arcs = rows[:, 3].long()
        lut = (timing.arcs_info.r_check_luts, timing.arcs_info.f_check_luts)[edge]
        value = timing.lut_entry_vectorized(rows[:, 2].long(), first, second, arcs, lut)
        if not bool(selected.any()):
            return value
        nodes, arcs = nodes[selected], arcs[selected]
        main, offsets = timing.inst_main_id[nodes].long(), self.offsets[arcs]
        candidates = model.piecewise_state[self.names[edge]]["vt_available"][main, offsets]
        if not bool(candidates.any(1).all()):
            raise RuntimeError("hold check has no qualified size/VT LUT candidates")
        result = model._piecewise_forward_dataset(
            self.names[edge], main, offsets, timing._vt_scalar(state["vt_var"])[nodes],
            state["size_var"][nodes], first[selected], second[selected],
        )
        return value.index_copy(
            0, torch.nonzero(selected, as_tuple=False).flatten(), result.to(value),
        )
