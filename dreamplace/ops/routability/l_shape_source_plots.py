"""L-shape source diagnostics, separate from the optimization operator."""

import logging

import torch

from .l_shape_electric_potential import SegmentElectricPotentialFunction

logger = logging.getLogger(__name__)


def _prepare_plot_tensor(value, dtype=torch.float32):
    if not isinstance(value, torch.Tensor):
        return None
    if value.requires_grad:
        value = value.detach()
    if value.is_cuda:
        value = value.cpu()
    return value.to(dtype=dtype)



def _build_l_shape_true_source_plot_maps(l_shape_op):
    """Collect the actual forward source maps and macro masks for plotting."""
    density_op = getattr(l_shape_op, "density_op", None)
    if density_op is None:
        raise ValueError("Missing density op for L-shape true-source plotting")

    source_map = _prepare_plot_tensor(SegmentElectricPotentialFunction.last_rho_map)
    source_map_h = _prepare_plot_tensor(SegmentElectricPotentialFunction.last_rho_map_h)
    source_map_v = _prepare_plot_tensor(SegmentElectricPotentialFunction.last_rho_map_v)
    split_available = (
        isinstance(source_map_h, torch.Tensor)
        and isinstance(source_map_v, torch.Tensor)
        and source_map_h.shape == source_map_v.shape
    )
    if not isinstance(source_map, torch.Tensor) and split_available:
        source_map = source_map_h + source_map_v
    if not isinstance(source_map, torch.Tensor):
        raise ValueError(
            "Missing SegmentElectricPotentialFunction.last_rho_map for L-shape true-source plotting"
        )

    macro_source_map = _prepare_plot_tensor(getattr(density_op, "macro_source_map", None))
    macro_body_source_map = _prepare_plot_tensor(
        getattr(density_op, "macro_body_source_map", None)
    )
    macro_halo_source_map = _prepare_plot_tensor(
        getattr(density_op, "macro_halo_source_map", None)
    )
    macro_mask = None
    if isinstance(macro_source_map, torch.Tensor):
        macro_mask = macro_source_map > 0

    return {
        "source_name": "forward_rho",
        "source_map": source_map,
        "source_map_h": source_map_h,
        "source_map_v": source_map_v,
        "split_available": split_available,
        "density_op": density_op,
        "macro_source_map": macro_source_map,
        "macro_body_source_map": macro_body_source_map,
        "macro_halo_source_map": macro_halo_source_map,
        "macro_mask": macro_mask,
    }



def _compute_robust_limits(map_tensor, *, symmetric=False, lower_pct=1.0, upper_pct=99.0):
    import numpy as np

    map_np = map_tensor.detach().cpu().to(dtype=torch.float32).numpy()
    finite = map_np[np.isfinite(map_np)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = float(np.percentile(np.abs(finite), upper_pct))
        vmax = max(vmax, 1e-6)
        return -vmax, vmax
    vmin = float(np.percentile(finite, lower_pct))
    vmax = float(np.percentile(finite, upper_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax



def _render_l_shape_map(ax, map_tensor, title, *, cmap=None, vmin=None, vmax=None, mask=False):
    import numpy as np

    map_np = map_tensor.detach().cpu().to(dtype=torch.float32).numpy()
    if cmap is None:
        cmap = "coolwarm" if float(np.nanmin(map_np)) < -1e-9 else "viridis"
    if vmin is None or vmax is None:
        symmetric = cmap == "coolwarm"
        vmin, vmax = _compute_robust_limits(map_tensor, symmetric=symmetric)
    im = ax.imshow(
        map_np.T,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    finite = map_np[np.isfinite(map_np)]
    if finite.size == 0:
        stats = "no finite values"
    else:
        stats = (
            "mean=%.4e max=%.4e min=%.4e p99=%.4e"
            % (
                float(finite.mean()),
                float(finite.max()),
                float(finite.min()),
                float(np.percentile(finite, 99.0)),
            )
        )
    if mask:
        active = int((map_np > 0).sum())
        stats = "%s active=%d" % (stats, active)
    ax.set_title("%s\n%s" % (title, stats))
    ax.set_xlabel("Bin X")
    ax.set_ylabel("Bin Y")
    return im



def plot_l_shape_electric_potential_map(l_shape_op, output_path, title_prefix="L-shape Electric Potential"):
    """Plot potential reconstructed from the actual forward source maps."""
    import matplotlib.pyplot as plt

    plot_payload = _build_l_shape_true_source_plot_maps(l_shape_op)
    rho_map = plot_payload["source_map"]
    rho_map_h = plot_payload["source_map_h"]
    rho_map_v = plot_payload["source_map_v"]
    split_available = plot_payload["split_available"]
    density_op = plot_payload["density_op"]
    if not density_op.energy_valid:
        raise RuntimeError(
            "L-shape electric potential plot requires full energy; disable l_shape_fast_mode"
        )

    ref_tensor = getattr(density_op, "bin_center_x", None)
    if not isinstance(ref_tensor, torch.Tensor):
        target_density = getattr(density_op, "target_density", None)
        if not isinstance(target_density, torch.Tensor):
            raise ValueError("L-shape electric potential plot requires density_op.target_density")
        density_op._init_bins(target_density.device, target_density.dtype)
        ref_tensor = getattr(density_op, "bin_center_x", None)
    if not isinstance(ref_tensor, torch.Tensor):
        raise ValueError("L-shape electric potential plot requires initialized density_op bin centers")

    if (
        getattr(density_op, "idct2", None) is None
        or getattr(density_op, "dct2", None) is None
        or getattr(density_op, "inv_wu2_plus_wv2", None) is None
    ):
        density_op._init_dct(ref_tensor.device, ref_tensor.dtype)
    device = ref_tensor.device
    dtype = ref_tensor.dtype
    bin_area = float(density_op.bin_size_x * density_op.bin_size_y)

    def compute_potential(map_tensor):
        map_tensor = map_tensor.to(device=device, dtype=dtype)
        rho_map_normalized = map_tensor * (1.0 / bin_area)
        auv = density_op.dct2.forward(rho_map_normalized)
        potential_map = density_op.idct2.forward(auv * density_op.inv_wu2_plus_wv2)
        potential_map = potential_map * bin_area
        return potential_map.detach().cpu().to(dtype=torch.float32)

    if split_available and isinstance(rho_map_h, torch.Tensor) and isinstance(rho_map_v, torch.Tensor):
        potential_map_h = compute_potential(rho_map_h)
        potential_map_v = compute_potential(rho_map_v)
        potential_map = potential_map_h + potential_map_v
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        im_h = _render_l_shape_map(axes[0], potential_map_h, f"{title_prefix} H")
        im_v = _render_l_shape_map(axes[1], potential_map_v, f"{title_prefix} V")
        im_t = _render_l_shape_map(axes[2], potential_map, f"{title_prefix} Total")
        fig.colorbar(im_h, ax=axes[0], fraction=0.046, pad=0.04, label="Potential")
        fig.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04, label="Potential")
        fig.colorbar(im_t, ax=axes[2], fraction=0.046, pad=0.04, label="Potential")
    else:
        potential_map = compute_potential(rho_map)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = _render_l_shape_map(ax, potential_map, title_prefix)
        fig.colorbar(im, ax=ax, label="Potential")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape electric potential plot saved to %s", output_path)



def plot_l_shape_true_source_maps(l_shape_op, output_path, title_prefix="L-shape True Source"):
    """Plot the actual forward source maps after macro occupancy and AL."""
    import matplotlib.pyplot as plt

    payload = _build_l_shape_true_source_plot_maps(l_shape_op)
    source_map = payload["source_map"]
    source_map_h = payload["source_map_h"]
    source_map_v = payload["source_map_v"]
    split_available = payload["split_available"]

    if split_available and isinstance(source_map_h, torch.Tensor) and isinstance(source_map_v, torch.Tensor):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
        limits_h = _compute_robust_limits(source_map_h)
        limits_v = _compute_robust_limits(source_map_v)
        limits_t = _compute_robust_limits(source_map)
        im_h = _render_l_shape_map(
            axes[0], source_map_h, f"{title_prefix} H", cmap="magma", vmin=limits_h[0], vmax=limits_h[1]
        )
        im_v = _render_l_shape_map(
            axes[1], source_map_v, f"{title_prefix} V", cmap="magma", vmin=limits_v[0], vmax=limits_v[1]
        )
        im_t = _render_l_shape_map(
            axes[2], source_map, f"{title_prefix} Total", cmap="magma", vmin=limits_t[0], vmax=limits_t[1]
        )
        fig.colorbar(im_h, ax=axes[0], fraction=0.046, pad=0.04, label="Source")
        fig.colorbar(im_v, ax=axes[1], fraction=0.046, pad=0.04, label="Source")
        fig.colorbar(im_t, ax=axes[2], fraction=0.046, pad=0.04, label="Source")
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        limits = _compute_robust_limits(source_map)
        im = _render_l_shape_map(
            ax, source_map, title_prefix, cmap="magma", vmin=limits[0], vmax=limits[1]
        )
        fig.colorbar(im, ax=ax, label="Source")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape true source plot saved to %s", output_path)



def plot_l_shape_macro_source_maps(l_shape_op, output_path, title_prefix="L-shape Macro Source"):
    """Plot fixed-macro source maps and the active macro-source mask."""
    import matplotlib.pyplot as plt

    payload = _build_l_shape_true_source_plot_maps(l_shape_op)
    macro_source = payload["macro_source_map"]
    macro_body = payload["macro_body_source_map"]
    macro_halo = payload["macro_halo_source_map"]
    macro_mask = payload["macro_mask"]
    if not isinstance(macro_source, torch.Tensor):
        raise ValueError("Missing density_op.macro_source_map for macro source plotting")

    maps = [macro_source]
    titles = [f"{title_prefix} Total"]
    cmaps = ["viridis"]
    if isinstance(macro_body, torch.Tensor):
        maps.append(macro_body)
        titles.append(f"{title_prefix} Body")
        cmaps.append("viridis")
    if isinstance(macro_halo, torch.Tensor):
        maps.append(macro_halo)
        titles.append(f"{title_prefix} Halo")
        cmaps.append("viridis")
    if isinstance(macro_mask, torch.Tensor):
        maps.append(macro_mask.to(dtype=torch.float32))
        titles.append(f"{title_prefix} Mask")
        cmaps.append("gray_r")

    fig, axes = plt.subplots(1, len(maps), figsize=(8 * len(maps), 8), constrained_layout=True)
    if len(maps) == 1:
        axes = [axes]
    numeric_limits = _compute_robust_limits(macro_source)
    for ax, map_tensor, title, cmap in zip(axes, maps, titles, cmaps):
        if cmap == "gray_r":
            im = _render_l_shape_map(
                ax, map_tensor, title, cmap=cmap, vmin=0.0, vmax=1.0, mask=True
            )
            label = "Mask"
        else:
            im = _render_l_shape_map(
                ax,
                map_tensor,
                title,
                cmap=cmap,
                vmin=numeric_limits[0],
                vmax=numeric_limits[1],
            )
            label = "Macro source"
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("L-shape macro source plot saved to %s", output_path)



def plot_segment_density_map(density_map, output_path, title="Segment Density Map", colormap="hot"):
    """绘制segment密度图"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # 确保tensor可以转换为numpy
    if density_map.requires_grad:
        density_map = density_map.detach()
    if density_map.is_cuda:
        density_map = density_map.cpu()

    im = ax.imshow(
        density_map.numpy().T,  # 转置使x轴在水平方向
        origin='lower',
        cmap=colormap,
        aspect='equal'
    )

    ax.set_title(title)
    ax.set_xlabel('Bin X')
    ax.set_ylabel('Bin Y')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Density')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Density map saved to {output_path}")
