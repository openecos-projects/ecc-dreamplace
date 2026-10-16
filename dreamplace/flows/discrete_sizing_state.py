"""Phase-local sizing snapshots include the exact imported pin geometry."""

from dataclasses import dataclass

import torch


@dataclass
class SizingState:
    size: torch.Tensor
    vt: torch.Tensor
    masters: torch.Tensor
    oscillation: dict
    pin_geometry: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, data, oscillation):
        return cls(
            data.get_continuous_size_parameter().detach().clone(),
            data.vt_logits.detach().clone(),
            data.inst_cell_id.detach().clone(),
            oscillation.snapshot(),
            {
                name: getattr(data, name).detach().clone()
                for name in (
                    "pin_offset_x",
                    "pin_offset_y",
                    "original_pin_offset_x",
                    "original_pin_offset_y",
                )
                if hasattr(data, name)
            },
        )

    def restore_parameters(self, data):
        changed = torch.nonzero(data.inst_cell_id != self.masters).flatten()
        with torch.no_grad():
            data.get_continuous_size_parameter().copy_(self.size)
            data.vt_logits.copy_(self.vt)
        return changed

    def restore_pin_geometry(self, data, placedb):
        # Master sync reconstructs library offsets. Restore the imported
        # geometry used to evaluate the snapshot, including inflation.
        changed = False
        with torch.no_grad():
            for name, values in self.pin_geometry.items():
                changed |= not torch.equal(getattr(data, name), values)
                getattr(data, name).copy_(values)
                if name in {"pin_offset_x", "pin_offset_y"}:
                    getattr(placedb, name)[:] = values.detach().cpu().numpy()
            for name in ("original_pin_offset_x", "original_pin_offset_y"):
                if name not in self.pin_geometry and hasattr(data, name):
                    delattr(data, name)
        return changed
