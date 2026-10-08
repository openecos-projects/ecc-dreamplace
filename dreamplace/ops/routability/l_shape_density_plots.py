"""L-shape density diagnostics, separate from the optimization operator."""

import logging

import torch

from .l_shape_electric_potential import compute_track_rho_components

logger = logging.getLogger(__name__)


def plot_l_shape_electric_overflow_map(l_shape_op, output_path, title_prefix="L-shape Electric Overflow"):
    """Plot GGR overflow plus surrogate density/overflow diagnostics."""
    import matplotlib.pyplot as plt
    import numpy as np

    plot_payload = _build_l_shape_electric_plot_maps(l_shape_op)
    blockage_initial_density = plot_payload.get("blockage_initial_density", False)
    ggr_overflow_map = plot_payload["ggr_overflow_map"]
    ggr_overflow_map_h = plot_payload["ggr_overflow_map_h"]
    ggr_overflow_map_v = plot_payload["ggr_overflow_map_v"]
    overflow_map = plot_payload["surrogate_overflow_map"]
    overflow_map_h = plot_payload["surrogate_overflow_map_h"]
    overflow_map_v = plot_payload["surrogate_overflow_map_v"]
    density_map = plot_payload["surrogate_density_map"]
    density_map_h = plot_payload["surrogate_density_map_h"]
    density_map_v = plot_payload["surrogate_density_map_v"]
    split_available = plot_payload["split_available"]
    unit_label = plot_payload.get("unit_label", "area units")
    show_density_row = bool(blockage_initial_density)

    def compute_limits(*map_tensors):
        valid_maps = [tensor.numpy() for tensor in map_tensors if isinstance(tensor, torch.Tensor)]
        if not valid_maps:
            return 0.0, 1.0
        stacked = np.concatenate([m.reshape(-1) for m in valid_maps])
        if float(np.min(stacked)) < -1e-9:
            vmax = max(1e-6, float(np.percentile(np.abs(stacked), 99)))
            return -vmax, vmax
        vmax = max(1e-6, float(np.percentile(stacked, 99)))
        return 0.0, vmax

    def render_map(ax, map_tensor, title, vmin, vmax, cmap=None):
        map_np = map_tensor.numpy()
        if cmap is None and vmin < 0.0:
            im = ax.imshow(
                map_np.T,
                origin="lower",
                cmap="coolwarm",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            im = ax.imshow(
                map_np.T,
                origin="lower",
                cmap=cmap or "plasma",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
        positive_ratio = float(np.mean(map_np > 0))
        ax.set_title(
            f"{title}\nmean={float(map_np.mean()):.4e} "
            f"max={float(map_np.max()):.4e} pos={positive_ratio:.1%}"
        )
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        return im

    if split_available and isinstance(overflow_map_h, torch.Tensor) and isinstance(overflow_map_v, torch.Tensor):
        num_rows = 3 if show_density_row else 2
        fig, axes = plt.subplots(num_rows, 3, figsize=(24, 7 * num_rows), constrained_layout=True)
        if num_rows == 2:
            axes = np.asarray(axes).reshape(2, 3)
        ggr_limits_h = compute_limits(ggr_overflow_map_h)
        ggr_limits_v = compute_limits(ggr_overflow_map_v)
        ggr_limits_t = compute_limits(ggr_overflow_map)
        overflow_limits_h = compute_limits(overflow_map_h)
        overflow_limits_v = compute_limits(overflow_map_v)
        overflow_limits_t = compute_limits(overflow_map)
        im_h_ggr = render_map(axes[0, 0], ggr_overflow_map_h, "GGR Overflow H", *ggr_limits_h, cmap="plasma")
        im_v_ggr = render_map(axes[0, 1], ggr_overflow_map_v, "GGR Overflow V", *ggr_limits_v, cmap="plasma")
        im_t_ggr = render_map(axes[0, 2], ggr_overflow_map, "GGR Overflow Total", *ggr_limits_t, cmap="plasma")
        next_row = 1
        if show_density_row and isinstance(density_map_h, torch.Tensor) and isinstance(density_map_v, torch.Tensor):
            density_limits_h = compute_limits(density_map_h)
            density_limits_v = compute_limits(density_map_v)
            density_limits_t = compute_limits(density_map)
            im_h_density = render_map(axes[next_row, 0], density_map_h, "Surrogate Density H", *density_limits_h, cmap="magma")
            im_v_density = render_map(axes[next_row, 1], density_map_v, "Surrogate Density V", *density_limits_v, cmap="magma")
            im_t_density = render_map(axes[next_row, 2], density_map, "Surrogate Density Total", *density_limits_t, cmap="magma")
            next_row += 1
        im_h_overflow = render_map(axes[next_row, 0], overflow_map_h, "Surrogate Overflow H", *overflow_limits_h, cmap="plasma")
        im_v_overflow = render_map(axes[next_row, 1], overflow_map_v, "Surrogate Overflow V", *overflow_limits_v, cmap="plasma")
        im_t_overflow = render_map(axes[next_row, 2], overflow_map, "Surrogate Overflow Total", *overflow_limits_t, cmap="plasma")
        axes[0, 0].set_ylabel("Bin Y\nGGR")
        if show_density_row:
            axes[1, 0].set_ylabel("Bin Y\nDensity")
            axes[2, 0].set_ylabel("Bin Y\nOverflow")
        else:
            axes[1, 0].set_ylabel("Bin Y\nSurrogate")
        fig.colorbar(im_h_ggr, ax=axes[0, :], fraction=0.025, pad=0.02, label=f"Overflow ({unit_label})")
        if show_density_row:
            fig.colorbar(im_h_density, ax=axes[1, :], fraction=0.025, pad=0.02, label=f"Density ({unit_label})")
            fig.colorbar(im_h_overflow, ax=axes[2, :], fraction=0.025, pad=0.02, label=f"Overflow ({unit_label})")
        else:
            fig.colorbar(im_h_overflow, ax=axes[1, :], fraction=0.025, pad=0.02, label=f"Overflow ({unit_label})")
    else:
        num_rows = 3 if show_density_row else 2
        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 8 * num_rows), constrained_layout=True)
        axes = np.atleast_1d(axes)
        ggr_limits = compute_limits(ggr_overflow_map)
        overflow_limits = compute_limits(overflow_map)
        im_ggr = render_map(axes[0], ggr_overflow_map, "GGR Overflow", *ggr_limits, cmap="plasma")
        next_row = 1
        if show_density_row and isinstance(density_map, torch.Tensor):
            density_limits = compute_limits(density_map)
            im_density = render_map(axes[next_row], density_map, "Surrogate Density", *density_limits, cmap="magma")
            next_row += 1
        im_overflow = render_map(axes[next_row], overflow_map, "Surrogate Overflow", *overflow_limits, cmap="plasma")
        fig.colorbar(im_ggr, ax=axes[0], label=f"Overflow ({unit_label})")
        if show_density_row:
            fig.colorbar(im_density, ax=axes[1], label=f"Density ({unit_label})")
            fig.colorbar(im_overflow, ax=axes[2], label=f"Overflow ({unit_label})")
        else:
            fig.colorbar(im_overflow, ax=axes[1], label=f"Overflow ({unit_label})")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape electric overflow plot saved to %s", output_path)



def plot_l_shape_supply_maps(l_shape_op, output_path, title_prefix="L-shape Supply Debug"):
    import matplotlib.pyplot as plt

    plot_payload = _build_l_shape_electric_plot_maps(l_shape_op)
    supply_original = plot_payload["supply_original"]
    supply_original_h = plot_payload["supply_original_h"]
    supply_original_v = plot_payload["supply_original_v"]
    supply_ggr = plot_payload["supply_ggr_map"]
    supply_ggr_h = plot_payload["supply_ggr_map_h"]
    supply_ggr_v = plot_payload["supply_ggr_map_v"]
    fix_usage = plot_payload["fix_usage_map"]
    fix_usage_h = plot_payload["fix_usage_map_h"]
    fix_usage_v = plot_payload["fix_usage_map_v"]
    split_available = plot_payload["split_available"]

    if not isinstance(supply_ggr, torch.Tensor):
        raise ValueError("Missing GGR supply map for L-shape supply plotting")
    if not isinstance(supply_original, torch.Tensor):
        raise ValueError("Missing theoretical supply_original map for L-shape supply plotting")

    def compute_limits(*maps):
        valid = [m for m in maps if isinstance(m, torch.Tensor)]
        vmax = max(float(m.max().item()) for m in valid) if valid else 1.0
        return 0.0, max(vmax, 1e-6)

    def maps_are_equivalent(lhs, rhs, atol=1e-6):
        if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            return False
        if lhs.shape != rhs.shape:
            return False
        return bool(torch.allclose(lhs, rhs, atol=atol, rtol=0.0))

    def render_map(ax, map_tensor, title, vmin, vmax):
        map_np = map_tensor.numpy()
        im = ax.imshow(
            map_np.T,
            origin="lower",
            cmap="viridis",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"{title}\nmean={float(map_np.mean()):.4e} max={float(map_np.max()):.4e}"
        )
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        return im

    if (
        split_available
        and isinstance(supply_original_h, torch.Tensor)
        and isinstance(supply_original_v, torch.Tensor)
        and isinstance(supply_ggr_h, torch.Tensor)
        and isinstance(supply_ggr_v, torch.Tensor)
    ):
        has_fix_usage = (
            isinstance(fix_usage_h, torch.Tensor)
            and isinstance(fix_usage_v, torch.Tensor)
            and isinstance(fix_usage, torch.Tensor)
        )
        ggr_is_redundant = (
            maps_are_equivalent(supply_original_h, supply_ggr_h)
            and maps_are_equivalent(supply_original_v, supply_ggr_v)
            and maps_are_equivalent(supply_original, supply_ggr)
        )
        show_ggr_row = not ggr_is_redundant
        num_rows = 1 + int(has_fix_usage) + int(show_ggr_row)
        fig, axes = plt.subplots(num_rows, 3, figsize=(24, 6 + 4 * num_rows), constrained_layout=True)
        limits_h = compute_limits(
            supply_original_h,
            supply_ggr_h if show_ggr_row else None,
            fix_usage_h if has_fix_usage else None,
        )
        limits_v = compute_limits(
            supply_original_v,
            supply_ggr_v if show_ggr_row else None,
            fix_usage_v if has_fix_usage else None,
        )
        limits_t = compute_limits(
            supply_original,
            supply_ggr if show_ggr_row else None,
            fix_usage if has_fix_usage else None,
        )
        im_h = render_map(axes[0, 0], supply_original_h, "Supply Original H", *limits_h)
        im_v = render_map(axes[0, 1], supply_original_v, "Supply Original V", *limits_v)
        im_t = render_map(axes[0, 2], supply_original, "Supply Original Total", *limits_t)
        current_row = 1
        if has_fix_usage:
            render_map(axes[1, 0], fix_usage_h, "Fix Usage H", *limits_h)
            render_map(axes[1, 1], fix_usage_v, "Fix Usage V", *limits_v)
            render_map(axes[1, 2], fix_usage, "Fix Usage Total", *limits_t)
            current_row = 2
        if show_ggr_row:
            render_map(axes[current_row, 0], supply_ggr_h, "Supply GGR H", *limits_h)
            render_map(axes[current_row, 1], supply_ggr_v, "Supply GGR V", *limits_v)
            render_map(axes[current_row, 2], supply_ggr, "Supply GGR Total", *limits_t)
        axes[0, 0].set_ylabel("Bin Y\nOriginal")
        if has_fix_usage:
            axes[1, 0].set_ylabel("Bin Y\nFixUsage")
        if show_ggr_row:
            axes[current_row, 0].set_ylabel("Bin Y\nGGR")
        fig.colorbar(im_h, ax=axes[:, 0], fraction=0.025, pad=0.02, label="Tracks")
        fig.colorbar(im_v, ax=axes[:, 1], fraction=0.025, pad=0.02, label="Tracks")
        fig.colorbar(im_t, ax=axes[:, 2], fraction=0.025, pad=0.02, label="Tracks")
    else:
        has_fix_usage = isinstance(fix_usage, torch.Tensor)
        ggr_is_redundant = maps_are_equivalent(supply_original, supply_ggr)
        show_ggr_row = not ggr_is_redundant
        num_rows = 1 + int(has_fix_usage) + int(show_ggr_row)
        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 6 + 6 * num_rows), constrained_layout=True)
        limits = compute_limits(
            supply_original,
            supply_ggr if show_ggr_row else None,
            fix_usage if has_fix_usage else None,
        )
        im_original = render_map(axes[0], supply_original, "Supply Original", *limits)
        current_row = 1
        if has_fix_usage:
            render_map(axes[1], fix_usage, "Fix Usage", *limits)
            current_row = 2
        if show_ggr_row:
            render_map(axes[current_row], supply_ggr, "Supply GGR", *limits)
        fig.colorbar(im_original, ax=axes, label="Tracks")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape supply debug plot saved to %s", output_path)



def plot_l_shape_initial_density_map(l_shape_op, output_path, title_prefix="L-shape Initial Density"):
    """Plot blockage-backed initial density maps in track-space units."""
    import matplotlib.pyplot as plt

    plot_payload = _build_l_shape_electric_plot_maps(l_shape_op)
    initial_density_map = plot_payload["initial_density_map"]
    initial_density_map_h = plot_payload["initial_density_map_h"]
    initial_density_map_v = plot_payload["initial_density_map_v"]
    base_fixed_usage_map = plot_payload.get("base_fixed_usage_map")
    base_fixed_usage_map_h = plot_payload.get("base_fixed_usage_map_h")
    base_fixed_usage_map_v = plot_payload.get("base_fixed_usage_map_v")
    macro_usage_map = plot_payload.get("macro_usage_map")
    macro_usage_map_h = plot_payload.get("macro_usage_map_h")
    macro_usage_map_v = plot_payload.get("macro_usage_map_v")
    boundary_usage_map = plot_payload.get("boundary_usage_map")
    boundary_usage_map_h = plot_payload.get("boundary_usage_map_h")
    boundary_usage_map_v = plot_payload.get("boundary_usage_map_v")
    split_available = plot_payload["split_available"]
    unit_label = plot_payload.get("unit_label", "tracks")

    if not isinstance(initial_density_map, torch.Tensor):
        raise ValueError("Missing initial density map for L-shape initial-density plotting")

    def compute_limits(*maps):
        valid = [m for m in maps if isinstance(m, torch.Tensor)]
        vmax = max(float(m.max().item()) for m in valid) if valid else 1.0
        return 0.0, max(vmax, 1e-6)

    def render_map(ax, map_tensor, title, vmin, vmax):
        map_np = map_tensor.numpy()
        im = ax.imshow(
            map_np.T,
            origin="lower",
            cmap="magma",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(
            f"{title}\nmean={float(map_np.mean()):.4e} max={float(map_np.max()):.4e}"
        )
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        return im

    if (
        split_available
        and isinstance(initial_density_map_h, torch.Tensor)
        and isinstance(initial_density_map_v, torch.Tensor)
    ):
        macro_available = (
            isinstance(base_fixed_usage_map_h, torch.Tensor)
            and isinstance(base_fixed_usage_map_v, torch.Tensor)
            and isinstance(base_fixed_usage_map, torch.Tensor)
            and isinstance(macro_usage_map_h, torch.Tensor)
            and isinstance(macro_usage_map_v, torch.Tensor)
            and isinstance(macro_usage_map, torch.Tensor)
        )
        if macro_available:
            rows = [
                (
                    f"{title_prefix} Base Fixed",
                    base_fixed_usage_map_h,
                    base_fixed_usage_map_v,
                    base_fixed_usage_map,
                ),
                (
                    f"{title_prefix} Macro Usage",
                    macro_usage_map_h,
                    macro_usage_map_v,
                    macro_usage_map,
                ),
            ]
            if (
                isinstance(boundary_usage_map_h, torch.Tensor)
                and isinstance(boundary_usage_map_v, torch.Tensor)
                and isinstance(boundary_usage_map, torch.Tensor)
                and bool((boundary_usage_map > 0).any().item())
            ):
                rows.append(
                    (
                        f"{title_prefix} Boundary Usage",
                        boundary_usage_map_h,
                        boundary_usage_map_v,
                        boundary_usage_map,
                    )
                )
            rows.append(
                (
                    f"{title_prefix} Effective Fixed",
                    initial_density_map_h,
                    initial_density_map_v,
                    initial_density_map,
                )
            )
            fig, axes = plt.subplots(
                len(rows), 3, figsize=(24, 8 * len(rows)), constrained_layout=True
            )
            for row_idx, (row_title, map_h, map_v, map_t) in enumerate(rows):
                limits = compute_limits(map_h, map_v, map_t)
                im_h = render_map(axes[row_idx, 0], map_h, f"{row_title} H", *limits)
                im_v = render_map(axes[row_idx, 1], map_v, f"{row_title} V", *limits)
                im_t = render_map(axes[row_idx, 2], map_t, f"{row_title} Total", *limits)
                for ax, im in (
                    (axes[row_idx, 0], im_h),
                    (axes[row_idx, 1], im_v),
                    (axes[row_idx, 2], im_t),
                ):
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Density ({unit_label})")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
            limits_h = compute_limits(initial_density_map_h)
            limits_v = compute_limits(initial_density_map_v)
            limits_t = compute_limits(initial_density_map)
            im_h = render_map(axes[0], initial_density_map_h, f"{title_prefix} H", *limits_h)
            im_v = render_map(axes[1], initial_density_map_v, f"{title_prefix} V", *limits_v)
            im_t = render_map(axes[2], initial_density_map, f"{title_prefix} Total", *limits_t)
            fig.colorbar(im_h, ax=axes[0], fraction=0.046, pad=0.04, label=f"Density ({unit_label})")
            fig.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04, label=f"Density ({unit_label})")
            fig.colorbar(im_t, ax=axes[2], fraction=0.046, pad=0.04, label=f"Density ({unit_label})")
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        limits = compute_limits(initial_density_map)
        im = render_map(ax, initial_density_map, title_prefix, *limits)
        fig.colorbar(im, ax=ax, label=f"Density ({unit_label})")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape initial density plot saved to %s", output_path)



def _build_l_shape_electric_plot_maps(l_shape_op):
    density_map = getattr(l_shape_op, "cached_density_map", None)
    density_map_h = getattr(l_shape_op, "cached_density_map_h", None)
    density_map_v = getattr(l_shape_op, "cached_density_map_v", None)
    density_op = getattr(l_shape_op, "density_op", None)
    if density_map is None or density_op is None:
        raise ValueError("Missing cached density map or density op for electric plotting")

    def prepare_tensor(value):
        if not isinstance(value, torch.Tensor):
            return None
        if value.requires_grad:
            value = value.detach()
        if value.is_cuda:
            value = value.cpu()
        return value.to(dtype=torch.float32)

    density_map = prepare_tensor(density_map)
    density_map_h = prepare_tensor(density_map_h)
    density_map_v = prepare_tensor(density_map_v)
    target_density = prepare_tensor(getattr(density_op, "target_density", None))
    target_demand = prepare_tensor(getattr(density_op, "target_demand", None))
    raw_wire_demand_map = prepare_tensor(getattr(density_op, "raw_wire_demand_map", None))
    supply_original = prepare_tensor(getattr(density_op, "supply_original", None))
    target_density_h = prepare_tensor(getattr(density_op, "target_density_h", None))
    target_density_v = prepare_tensor(getattr(density_op, "target_density_v", None))
    target_demand_h = prepare_tensor(getattr(density_op, "target_demand_h", None))
    target_demand_v = prepare_tensor(getattr(density_op, "target_demand_v", None))
    raw_wire_demand_map_h = prepare_tensor(getattr(density_op, "raw_wire_demand_map_h", None))
    raw_wire_demand_map_v = prepare_tensor(getattr(density_op, "raw_wire_demand_map_v", None))
    supply_original_h = prepare_tensor(getattr(density_op, "supply_original_h", None))
    supply_original_v = prepare_tensor(getattr(density_op, "supply_original_v", None))
    fix_usage_map = prepare_tensor(getattr(l_shape_op, "fix_usage_map", None))
    fix_usage_map_h = prepare_tensor(getattr(l_shape_op, "fix_usage_map_h", None))
    fix_usage_map_v = prepare_tensor(getattr(l_shape_op, "fix_usage_map_v", None))
    blockage_initial_density = bool(getattr(density_op, "blockage_initial_density", False))

    if not isinstance(target_density, torch.Tensor) or target_density.dim() != 2:
        raise TypeError(
            "L-shape electric plotting expects a 2D routing supply tensor in density_op.target_density"
        )
    if not isinstance(target_demand, torch.Tensor) or target_demand.dim() != 2:
        raise TypeError(
            "L-shape electric plotting expects a 2D routing demand tensor in density_op.target_demand"
        )

    surrogate_overflow_map = density_map.clone()
    surrogate_overflow_map_h = None
    surrogate_overflow_map_v = None
    surrogate_density_map = density_map.clone()
    surrogate_density_map_h = None
    surrogate_density_map_v = None
    surrogate_rho_map = density_map.clone()
    surrogate_rho_map_h = None
    surrogate_rho_map_v = None
    initial_density_map = torch.zeros_like(density_map)
    initial_density_map_h = None
    initial_density_map_v = None
    base_fixed_usage_map = None
    base_fixed_usage_map_h = None
    base_fixed_usage_map_v = None
    macro_usage_map = None
    macro_usage_map_h = None
    macro_usage_map_v = None
    boundary_usage_map = None
    boundary_usage_map_h = None
    boundary_usage_map_v = None
    ggr_overflow_map = torch.zeros_like(density_map)
    ggr_overflow_map_h = None
    ggr_overflow_map_v = None
    supply_map = target_density
    bin_area = float(density_op.bin_size_x * density_op.bin_size_y)
    unit_label = "tracks" if blockage_initial_density else "area units"

    split_available = (
        isinstance(density_map_h, torch.Tensor)
        and isinstance(density_map_v, torch.Tensor)
        and isinstance(target_density_h, torch.Tensor)
        and isinstance(target_density_v, torch.Tensor)
        and density_map_h.shape == target_density_h.shape
        and density_map_v.shape == target_density_v.shape
    )

    if blockage_initial_density:
        if split_available and isinstance(supply_original_h, torch.Tensor) and isinstance(supply_original_v, torch.Tensor) and isinstance(fix_usage_map_h, torch.Tensor) and isinstance(fix_usage_map_v, torch.Tensor):
            macro_source_map = prepare_tensor(getattr(density_op, "macro_source_map", None))
            boundary_source_map = prepare_tensor(getattr(density_op, "boundary_source_map", None))
            components_h = compute_track_rho_components(
                density_map_h,
                supply_original_h,
                fix_usage_map_h,
                bin_area,
                macro_source_map=macro_source_map,
                boundary_source_map=boundary_source_map,
            )
            components_v = compute_track_rho_components(
                density_map_v,
                supply_original_v,
                fix_usage_map_v,
                bin_area,
                macro_source_map=macro_source_map,
                boundary_source_map=boundary_source_map,
            )
            density_seg_h_tracks = components_h["density_seg_tracks"]
            density_seg_v_tracks = components_v["density_seg_tracks"]
            initial_density_map_h = components_h["initial_density_tracks"]
            initial_density_map_v = components_v["initial_density_tracks"]
            base_fixed_usage_map_h = components_h["base_fixed_usage"]
            base_fixed_usage_map_v = components_v["base_fixed_usage"]
            macro_usage_map_h = components_h["macro_usage"]
            macro_usage_map_v = components_v["macro_usage"]
            boundary_usage_map_h = components_h["boundary_usage"]
            boundary_usage_map_v = components_v["boundary_usage"]
            base_fixed_usage_map = (
                fix_usage_map.clamp(min=0)
                if isinstance(fix_usage_map, torch.Tensor)
                else base_fixed_usage_map_h + base_fixed_usage_map_v
            )
            macro_usage_map = macro_usage_map_h + macro_usage_map_v
            boundary_usage_map = boundary_usage_map_h + boundary_usage_map_v
            initial_density_map = (
                base_fixed_usage_map + macro_usage_map + boundary_usage_map
            )
            surrogate_occ_map_h = components_h["occupancy_tracks"]
            surrogate_occ_map_v = components_v["occupancy_tracks"]
            surrogate_density_map_h = surrogate_occ_map_h
            surrogate_density_map_v = surrogate_occ_map_v
            surrogate_density_map = surrogate_occ_map_h + surrogate_occ_map_v
            surrogate_rho_map_h = components_h["residual_tracks"]
            surrogate_rho_map_v = components_v["residual_tracks"]
            surrogate_overflow_map_h = components_h["overflow_map"]
            surrogate_overflow_map_v = components_v["overflow_map"]
            surrogate_rho_map = surrogate_rho_map_h + surrogate_rho_map_v
            surrogate_overflow_map = surrogate_overflow_map_h + surrogate_overflow_map_v
            if isinstance(target_demand_h, torch.Tensor) and target_demand_h.dim() == 2:
                ggr_overflow_map_h = torch.relu(target_demand_h - target_density_h)
            else:
                ggr_overflow_map_h = torch.zeros_like(surrogate_overflow_map_h)
            if isinstance(target_demand_v, torch.Tensor) and target_demand_v.dim() == 2:
                ggr_overflow_map_v = torch.relu(target_demand_v - target_density_v)
            else:
                ggr_overflow_map_v = torch.zeros_like(surrogate_overflow_map_v)
            ggr_overflow_map = ggr_overflow_map_h + ggr_overflow_map_v
        elif isinstance(supply_original, torch.Tensor) and isinstance(fix_usage_map, torch.Tensor):
            macro_source_map = prepare_tensor(getattr(density_op, "macro_source_map", None))
            boundary_source_map = prepare_tensor(getattr(density_op, "boundary_source_map", None))
            components = compute_track_rho_components(
                density_map,
                supply_original,
                fix_usage_map,
                bin_area,
                macro_source_map=macro_source_map,
                boundary_source_map=boundary_source_map,
            )
            density_seg_tracks = components["density_seg_tracks"]
            initial_density_map = components["initial_density_tracks"]
            base_fixed_usage_map = components["base_fixed_usage"]
            macro_usage_map = components["macro_usage"]
            boundary_usage_map = components["boundary_usage"]
            surrogate_occ_map = components["occupancy_tracks"]
            surrogate_density_map = surrogate_occ_map
            surrogate_rho_map = components["residual_tracks"]
            surrogate_overflow_map = components["overflow_map"]
            ggr_overflow_map = torch.relu(target_demand - supply_map)
    else:
        area_per_track = prepare_tensor(getattr(density_op, "area_per_track", None))
        if isinstance(area_per_track, torch.Tensor) and area_per_track.numel() == 1 and float(area_per_track.item()) > 0:
            calibrated_area_per_track = float(area_per_track.item())
        else:
            total_density = float(density_map.sum().item())
            calibration_map = raw_wire_demand_map if isinstance(raw_wire_demand_map, torch.Tensor) else target_demand
            total_demand = float(calibration_map.sum().item())
            calibrated_area_per_track = (
                total_density / total_demand if total_density > 0 and total_demand > 0 else 0.0
            )

        if calibrated_area_per_track > 0:
            if split_available:
                calibrated_area_per_track_h = calibrated_area_per_track
                calibrated_area_per_track_v = calibrated_area_per_track
                calibration_map_h = raw_wire_demand_map_h if isinstance(raw_wire_demand_map_h, torch.Tensor) else target_demand_h
                calibration_map_v = raw_wire_demand_map_v if isinstance(raw_wire_demand_map_v, torch.Tensor) else target_demand_v
                if isinstance(calibration_map_h, torch.Tensor) and calibration_map_h.dim() == 2:
                    total_density_h = float(density_map_h.sum().item())
                    total_demand_h = float(calibration_map_h.sum().item())
                    if total_density_h > 0 and total_demand_h > 0:
                        calibrated_area_per_track_h = total_density_h / total_demand_h
                if isinstance(calibration_map_v, torch.Tensor) and calibration_map_v.dim() == 2:
                    total_density_v = float(density_map_v.sum().item())
                    total_demand_v = float(calibration_map_v.sum().item())
                    if total_density_v > 0 and total_demand_v > 0:
                        calibrated_area_per_track_v = total_density_v / total_demand_v

                demand_h_in_tracks = density_map_h / calibrated_area_per_track_h
                demand_v_in_tracks = density_map_v / calibrated_area_per_track_v
                surrogate_rho_map_h = (demand_h_in_tracks - target_density_h) * calibrated_area_per_track_h
                surrogate_rho_map_v = (demand_v_in_tracks - target_density_v) * calibrated_area_per_track_v
                surrogate_overflow_map_h = torch.relu(demand_h_in_tracks - target_density_h) * calibrated_area_per_track_h
                surrogate_overflow_map_v = torch.relu(demand_v_in_tracks - target_density_v) * calibrated_area_per_track_v
                surrogate_rho_map_h = surrogate_rho_map_h - surrogate_rho_map_h.mean()
                surrogate_rho_map_v = surrogate_rho_map_v - surrogate_rho_map_v.mean()
                surrogate_rho_map = surrogate_rho_map_h + surrogate_rho_map_v
                surrogate_overflow_map = surrogate_overflow_map_h + surrogate_overflow_map_v
                if isinstance(target_demand_h, torch.Tensor) and target_demand_h.dim() == 2:
                    ggr_overflow_map_h = torch.relu(target_demand_h - target_density_h) * calibrated_area_per_track_h
                else:
                    ggr_overflow_map_h = torch.zeros_like(surrogate_overflow_map_h)
                if isinstance(target_demand_v, torch.Tensor) and target_demand_v.dim() == 2:
                    ggr_overflow_map_v = torch.relu(target_demand_v - target_density_v) * calibrated_area_per_track_v
                else:
                    ggr_overflow_map_v = torch.zeros_like(surrogate_overflow_map_v)
                ggr_overflow_map = ggr_overflow_map_h + ggr_overflow_map_v
            else:
                demand_in_tracks = density_map / calibrated_area_per_track
                surrogate_rho_map = (demand_in_tracks - supply_map) * calibrated_area_per_track
                overflow_in_tracks = torch.relu(demand_in_tracks - supply_map)
                surrogate_overflow_map = overflow_in_tracks * calibrated_area_per_track
                surrogate_rho_map = surrogate_rho_map - surrogate_rho_map.mean()
                ggr_overflow_map = torch.relu(target_demand - supply_map) * calibrated_area_per_track

    return {
        "density_map": density_map,
        "density_map_h": density_map_h,
        "density_map_v": density_map_v,
        "supply_original": supply_original,
        "supply_original_h": supply_original_h,
        "supply_original_v": supply_original_v,
        "fix_usage_map": fix_usage_map,
        "fix_usage_map_h": fix_usage_map_h,
        "fix_usage_map_v": fix_usage_map_v,
        "supply_ggr_map": supply_map,
        "supply_ggr_map_h": target_density_h,
        "supply_ggr_map_v": target_density_v,
        "surrogate_rho_map": surrogate_rho_map,
        "surrogate_rho_map_h": surrogate_rho_map_h,
        "surrogate_rho_map_v": surrogate_rho_map_v,
        "surrogate_density_map": surrogate_density_map,
        "surrogate_density_map_h": surrogate_density_map_h,
        "surrogate_density_map_v": surrogate_density_map_v,
        "initial_density_map": initial_density_map,
        "initial_density_map_h": initial_density_map_h,
        "initial_density_map_v": initial_density_map_v,
        "base_fixed_usage_map": base_fixed_usage_map,
        "base_fixed_usage_map_h": base_fixed_usage_map_h,
        "base_fixed_usage_map_v": base_fixed_usage_map_v,
        "macro_usage_map": macro_usage_map,
        "macro_usage_map_h": macro_usage_map_h,
        "macro_usage_map_v": macro_usage_map_v,
        "boundary_usage_map": boundary_usage_map,
        "boundary_usage_map_h": boundary_usage_map_h,
        "boundary_usage_map_v": boundary_usage_map_v,
        "surrogate_overflow_map": surrogate_overflow_map,
        "surrogate_overflow_map_h": surrogate_overflow_map_h,
        "surrogate_overflow_map_v": surrogate_overflow_map_v,
        "ggr_overflow_map": ggr_overflow_map,
        "ggr_overflow_map_h": ggr_overflow_map_h,
        "ggr_overflow_map_v": ggr_overflow_map_v,
        "split_available": split_available,
        "density_op": density_op,
        "unit_label": unit_label,
        "blockage_initial_density": blockage_initial_density,
    }
