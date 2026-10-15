from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiscreteVtCommit:
    data_collections: object
    target_vts: torch.Tensor | None
    require_exact_logits: bool

    @classmethod
    def from_summary(cls, data_collections, cell_ids, summary):
        flat_info = getattr(data_collections, "flat_libcell_info", None)
        target_vts = None
        if flat_info is not None and flat_info.ndim == 2 and flat_info.shape[1] > 3:
            target_vts = flat_info[cell_ids.to(device=flat_info.device), 3].long()

        applied_vts = summary.get("applied_vts") or []
        if applied_vts:
            if len(applied_vts) != int(cell_ids.numel()):
                raise RuntimeError("discrete sizing applied_vts must align with applied cells")
            if target_vts is None:
                raise RuntimeError("discrete sizing VT commit requires flat library cell metadata")
            reported_vts = torch.as_tensor(
                applied_vts,
                dtype=torch.long,
                device=target_vts.device,
            )
            if not torch.equal(reported_vts, target_vts):
                raise RuntimeError(
                    "discrete sizing target VT does not match the selected library cell"
                )
        return cls(
            data_collections=data_collections,
            target_vts=target_vts,
            require_exact_logits=bool(applied_vts),
        )

    def apply_data_collections(self, inst_ids, cell_ids):
        data = self.data_collections
        inst_vt_init = getattr(data, "inst_vt_init", None)
        if inst_vt_init is not None and self.target_vts is not None:
            rows = inst_ids.to(device=inst_vt_init.device)
            vt_ids = self.target_vts.to(device=inst_vt_init.device)
            inst_vt_init[rows] = 0.0
            inst_vt_init[rows, vt_ids] = 1.0

        inst_leakage_init = getattr(data, "inst_leakage_init", None)
        flat_leakage = getattr(data, "flat_libcell_leakage", None)
        if inst_leakage_init is not None and flat_leakage is not None:
            inst_leakage_init[inst_ids.to(device=inst_leakage_init.device)] = flat_leakage[
                cell_ids.to(device=flat_leakage.device)
            ].to(
                device=inst_leakage_init.device,
                dtype=inst_leakage_init.dtype,
            )

    def apply_placedb(self, placedb, inst_ids):
        if placedb is None:
            return
        inst_cpu = inst_ids.detach().cpu().numpy()
        if getattr(placedb, "inst_vt_init", None) is not None and self.target_vts is not None:
            target_vts_cpu = self.target_vts.detach().cpu().numpy()
            placedb.inst_vt_init[inst_cpu] = 0.0
            placedb.inst_vt_init[inst_cpu, target_vts_cpu] = 1.0

        data_leakage = getattr(self.data_collections, "inst_leakage_init", None)
        if getattr(placedb, "inst_leakage_init", None) is not None and data_leakage is not None:
            placedb.inst_leakage_init[inst_cpu] = (
                data_leakage[inst_ids.to(data_leakage.device)].detach().cpu().numpy()
            )

    def probability_gap(self, inst_ids):
        if self.target_vts is None or not self.require_exact_logits:
            return 0.0
        vt_var_getter = getattr(self.data_collections, "get_vt_var", None)
        vt_var = vt_var_getter() if callable(vt_var_getter) else None
        if vt_var is None:
            return 0.0
        expected = torch.zeros(
            (int(inst_ids.numel()), int(vt_var.shape[1])),
            dtype=vt_var.dtype,
            device=vt_var.device,
        )
        expected.scatter_(
            1,
            self.target_vts.to(device=vt_var.device).reshape(-1, 1),
            1.0,
        )
        actual = vt_var[inst_ids.to(device=vt_var.device)]
        gap = float((actual - expected).abs().max().cpu().item())
        if gap > 1.0e-4:
            raise RuntimeError("discrete sizing VT logits do not match selected legal VTs")
        return gap
