"""Compact density view for accepted typed candidates during ordinary GP."""

import torch

from .virtual_cell_density import build_combined_virtual_position


class CandidateDensityView:
    """Reuse ElectricPotential with live edge positions and selected master sizes."""

    def __init__(self, model, state):
        self.model = model
        data = model.data_collections
        rows = torch.nonzero(state.activation_param.detach().sum(dim=1) > 0).flatten()
        records = [state.candidate_records[index] for index in rows.cpu().tolist()]
        raw_parent = torch.tensor(
            [int(record.get("parent_node_id", -1)) for record in records],
            dtype=torch.long,
        )
        raw_child = torch.tensor(
            [
                int(record["child_node_ids"][0])
                if record.get("child_node_ids")
                else -1
                for record in records
            ],
            dtype=torch.long,
        )
        self.raw_parent = raw_parent
        self.raw_child = raw_child
        self.parent = raw_parent.clamp(min=0)
        self.child = raw_child.clamp(min=0)
        self.ratio = torch.tensor(
            [float(record.get("segment_split_ratio", 0.5) or 0.5) for record in records]
        )
        self.fixed_x_dbu = torch.tensor(
            [
                float(record.get("x_dbu", record.get("candidate_location_x", 0.0)) or 0.0)
                for record in records
            ],
            dtype=torch.float64,
        )
        self.fixed_y_dbu = torch.tensor(
            [
                float(record.get("y_dbu", record.get("candidate_location_y", 0.0)) or 0.0)
                for record in records
            ],
            dtype=torch.float64,
        )
        self.prefer_fixed_xy = torch.tensor(
            [bool(record.get("cpp_subnet_refreshed", False)) for record in records],
            dtype=torch.bool,
        )
        types = state.selected_drive_type().index_select(0, rows)
        cells = state.drive_type_cell_ids.index_select(0, types).to(data.flat_libcell_width.device)
        self.width = data.flat_libcell_width[cells].detach() + 2 * model.params.cell_padding_x
        self.height = data.flat_libcell_height[cells].detach()
        self.count = int(rows.numel())
        self._density_ops = {}

    def _op(self, pos):
        key = (pos.device, pos.dtype)
        if key in self._density_ops:
            return self._density_ops[key]
        model = self.model
        data, db = model.data_collections, model.placedb
        count = db.num_movable_nodes
        widths = torch.cat((data.node_size_x[:count], self.width.to(pos), data.node_size_x[count:]))
        heights = torch.cat(
            (data.node_size_y[:count], self.height.to(pos), data.node_size_y[count:])
        )
        masks = torch.cat(
            (
                data.movable_macro_mask[:count],
                torch.zeros(self.count, dtype=torch.bool, device=pos.device),
            )
        )
        op = model.build_electric_potential(
            model.params,
            db,
            data,
            model.num_bins_x,
            model.num_bins_y,
            name="Global placement: candidate buffer density",
            node_size_x=widths,
            node_size_y=heights,
            num_movable_nodes=count + self.count,
            num_terminals=db.num_terminals,
            num_filler_nodes=db.num_filler_nodes,
            sorted_node_map=torch.argsort(widths[: count + self.count]).to(torch.int32),
            movable_macro_mask=masks,
        )
        self._density_ops[key] = op
        return op

    def __call__(self, pos, *, mode=None):
        model = self.model
        db = model.placedb
        _, x, y = model._timing_geometry_for_pos(pos)
        coord_count = int(x.numel())
        raw_parent = self.raw_parent.to(pos.device)
        raw_child = self.raw_child.to(pos.device)
        live_mask = (
            (~self.prefer_fixed_xy.to(pos.device))
            & (raw_parent >= 0)
            & (raw_parent < coord_count)
            & (raw_child >= 0)
            & (raw_child < coord_count)
        )
        parent = raw_parent.clamp(min=0, max=max(0, coord_count - 1))
        child = raw_child.clamp(min=0, max=max(0, coord_count - 1))
        ratio = self.ratio.to(pos)
        live_x = (1 - ratio) * x[parent].to(pos) + ratio * x[child].to(pos)
        live_y = (1 - ratio) * y[parent].to(pos) + ratio * y[child].to(pos)
        if not bool(live_mask.all().item()):
            scale = float(getattr(model.params, "scale_factor", 1.0) or 1.0)
            shift = getattr(model.params, "shift_factor", (0.0, 0.0)) or (0.0, 0.0)
            fixed_x = (
                (self.fixed_x_dbu.to(device=pos.device, dtype=torch.float64) - float(shift[0]))
                * scale
            ).to(pos)
            fixed_y = (
                (self.fixed_y_dbu.to(device=pos.device, dtype=torch.float64) - float(shift[1]))
                * scale
            ).to(pos)
            live_x = torch.where(live_mask, live_x, fixed_x)
            live_y = torch.where(live_mask, live_y, fixed_y)
        vx = live_x - self.width.to(pos) / 2
        vy = live_y - self.height.to(pos) / 2
        combined = build_combined_virtual_position(
            pos,
            vx,
            vy,
            self.width,
            self.height,
            num_movable_nodes=db.num_movable_nodes,
            num_terminals=db.num_terminals,
            num_filler_nodes=db.num_filler_nodes,
            num_fixed_nodes=db.num_physical_nodes - db.num_movable_nodes,
        )
        op = self._op(pos)
        return op(combined) if mode is None else op(combined, mode=mode)


def install_candidate_density(model, state):
    previous = model._candidate_density_view
    if not bool((state.activation_param.detach().sum(dim=1) > 0).any()):
        model._candidate_density_view = None
        return
    view = CandidateDensityView(model, state)
    if (
        previous is not None
        and torch.equal(previous.width, view.width)
        and torch.equal(previous.height, view.height)
    ):
        view._density_ops = previous._density_ops
    model._candidate_density_view = view
