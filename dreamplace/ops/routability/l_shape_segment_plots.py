"""L-shape segment diagnostics, separate from the optimization operator."""

import logging

import torch

from .l_shape_segment import H_FIRST, V_FIRST, STRAIGHT, FAKE_STRAIGHT, UNKNOWN

logger = logging.getLogger(__name__)


def plot_l_shape_segments(segments, newx=None, newy=None, flat_from=None, flat_to=None, l_directions=None,
                          output_path=None, placedb=None, params=None):
    """
    绘制L形segments

    Args:
        segments: LShapeSegmentOp的输出（包含缓存的原始数据）
        newx, newy: 顶点坐标（可选，优先使用segments中缓存的数据）
        flat_from, flat_to: 边信息（可选，优先使用segments中缓存的数据）
        l_directions: L方向（可选，优先使用segments中缓存的数据）
        output_path: 输出路径
        placedb, params: 用于坐标转换
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    # 优先使用segments中缓存的原始数据（确保一致性）
    if 'newx' in segments:
        newx = segments['newx']
    if 'newy' in segments:
        newy = segments['newy']
    if 'flat_from' in segments:
        flat_from = segments['flat_from']
    if 'flat_to' in segments:
        flat_to = segments['flat_to']
    if 'l_directions' in segments:
        l_directions = segments['l_directions']

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制segments
    if segments['num_segments'] > 0:
        seg_llx = segments['segment_llx'].detach().cpu().numpy()
        seg_lly = segments['segment_lly'].detach().cpu().numpy()
        seg_size_x = segments['segment_size_x'].detach().cpu().numpy()
        seg_size_y = segments['segment_size_y'].detach().cpu().numpy()
        seg_is_h = segments['segment_is_horizontal'].detach().cpu().numpy()

        patches = []
        colors = []

        for i in range(len(seg_llx)):
            rect = Rectangle(
                (seg_llx[i], seg_lly[i]),
                seg_size_x[i], seg_size_y[i]
            )
            patches.append(rect)
            # 水平segment用蓝色，垂直segment用绿色
            colors.append('blue' if seg_is_h[i] else 'green')

        pc = PatchCollection(patches, alpha=0.3, edgecolor='black', linewidth=0.5)
        pc.set_facecolor(colors)
        ax.add_collection(pc)

    # 绘制原始边（与segment构建逻辑保持一致）
    if newx is None or newy is None or flat_from is None or flat_to is None or l_directions is None:
        logger.warning("Missing edge data for plotting L-shape lines")
    else:
        newx_np = newx.detach().cpu().numpy()
        newy_np = newy.detach().cpu().numpy()
        flat_from_np = flat_from.detach().cpu().numpy()
        flat_to_np = flat_to.detach().cpu().numpy()
        l_dir_np = l_directions.detach().cpu().numpy()

        for i in range(len(flat_from_np)):
            f, t = flat_from_np[i], flat_to_np[i]
            if f < 0 or t < 0 or f >= len(newx_np) or t >= len(newx_np):
                continue

            x1, y1 = newx_np[f], newy_np[f]
            x2, y2 = newx_np[t], newy_np[t]
            l_dir = l_dir_np[i]

            # 与segment构建逻辑保持一致：先检查几何上是否是直线
            is_horizontal_line = abs(y1 - y2) < 1e-4
            is_vertical_line = abs(x1 - x2) < 1e-4
            is_straight = is_horizontal_line or is_vertical_line or (l_dir == STRAIGHT)

            if is_straight:
                # 几何上是水平或垂直线，直接画直线
                ax.plot([x1, x2], [y1, y2], 'k--', linewidth=0.5, alpha=0.5)
            elif l_dir == V_FIRST:
                # Lower-L: 先垂直后水平，拐点在 (x1, y2)
                corner_x, corner_y = x1, y2
                ax.plot([x1, corner_x], [y1, corner_y], 'm-', linewidth=0.8, alpha=0.7)
                ax.plot([corner_x, x2], [corner_y, y2], 'm-', linewidth=0.8, alpha=0.7)
            elif l_dir == UNKNOWN:
                # 残余 UNKNOWN 与 segment 构建保持一致：跳过，不画 L 形
                continue
            else:
                # Upper-L (包括 H_FIRST 和残余 FAKE_STRAIGHT): 先水平后垂直
                corner_x, corner_y = x2, y1
                ax.plot([x1, corner_x], [y1, corner_y], 'r-', linewidth=0.8, alpha=0.7)
                ax.plot([corner_x, x2], [corner_y, y2], 'r-', linewidth=0.8, alpha=0.7)

    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title('L-shape Segments (blue=horizontal, green=vertical)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', label='Upper-L (H→V)'),
        Line2D([0], [0], color='m', label='Lower-L (V→H)'),
        Line2D([0], [0], color='k', linestyle='--', label='Straight'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"L-shape segments plot saved to {output_path}")



def plot_soft_l_intermediate(segments, output_path, max_diagonal_edges=4000):
    """Plot soft-L candidate paths and edge-level probability statistics."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    required = ("newx", "newy", "flat_from", "flat_to", "soft_l_weights")
    missing = [key for key in required if key not in segments]
    if missing:
        raise ValueError(f"Missing soft-L plotting data: {missing}")

    newx = segments["newx"].detach().cpu().numpy()
    newy = segments["newy"].detach().cpu().numpy()
    flat_from = segments["flat_from"].detach().cpu().numpy()
    flat_to = segments["flat_to"].detach().cpu().numpy()
    soft_l_weights = segments["soft_l_weights"].detach().cpu().numpy()

    valid = (
        (flat_from >= 0)
        & (flat_to >= 0)
        & (flat_from < len(newx))
        & (flat_to < len(newx))
    )
    if not np.any(valid):
        raise ValueError("No valid edges available for soft-L plotting")

    edge_idx = np.nonzero(valid)[0]
    x1 = newx[flat_from[valid]]
    y1 = newy[flat_from[valid]]
    x2 = newx[flat_to[valid]]
    y2 = newy[flat_to[valid]]
    weights = np.clip(soft_l_weights[valid], 0.0, None)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weights = np.where(weight_sum > 1e-12, weights / np.clip(weight_sum, 1e-12, None), 0.5)

    is_diagonal = (np.abs(x1 - x2) >= 1e-4) & (np.abs(y1 - y2) >= 1e-4)
    diag_idx = np.nonzero(is_diagonal)[0]
    if diag_idx.size == 0:
        raise ValueError("No diagonal edges available for soft-L plotting")

    if diag_idx.size > max_diagonal_edges:
        take = np.linspace(0, diag_idx.size - 1, max_diagonal_edges, dtype=int)
        diag_idx = diag_idx[take]

    diag_x1 = x1[diag_idx]
    diag_y1 = y1[diag_idx]
    diag_x2 = x2[diag_idx]
    diag_y2 = y2[diag_idx]
    diag_weights = weights[diag_idx]

    full_diag_weights = weights[is_diagonal]
    full_p_h = full_diag_weights[:, 0]
    full_entropy = -np.sum(
        np.where(full_diag_weights > 1e-12, full_diag_weights * np.log2(np.clip(full_diag_weights, 1e-12, None)), 0.0),
        axis=1,
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.2], height_ratios=[1.0, 1.0])
    ax_paths = fig.add_subplot(gs[:, 0])
    ax_prob = fig.add_subplot(gs[0, 1])
    ax_entropy = fig.add_subplot(gs[1, 1])

    for i in range(diag_idx.size):
        h_weight = float(diag_weights[i, 0])
        v_weight = float(diag_weights[i, 1])
        x1_i, y1_i = diag_x1[i], diag_y1[i]
        x2_i, y2_i = diag_x2[i], diag_y2[i]

        h_alpha = min(0.95, 0.05 + 0.90 * h_weight)
        v_alpha = min(0.95, 0.05 + 0.90 * v_weight)
        h_width = 0.4 + 1.4 * h_weight
        v_width = 0.4 + 1.4 * v_weight

        ax_paths.plot([x1_i, x2_i], [y1_i, y1_i], color="tab:red", alpha=h_alpha, linewidth=h_width)
        ax_paths.plot([x2_i, x2_i], [y1_i, y2_i], color="tab:red", alpha=h_alpha, linewidth=h_width)

        ax_paths.plot([x1_i, x1_i], [y1_i, y2_i], color="tab:purple", alpha=v_alpha, linewidth=v_width)
        ax_paths.plot([x1_i, x2_i], [y2_i, y2_i], color="tab:purple", alpha=v_alpha, linewidth=v_width)

    ax_paths.set_aspect("equal")
    ax_paths.autoscale()
    ax_paths.set_title(
        f"Soft L Candidate Paths (sampled {diag_idx.size}/{np.count_nonzero(is_diagonal)} diagonal edges)"
    )
    ax_paths.set_xlabel("X")
    ax_paths.set_ylabel("Y")
    ax_paths.legend(
        handles=[
            Line2D([0], [0], color="tab:red", label="H-path candidate"),
            Line2D([0], [0], color="tab:purple", label="V-path candidate"),
        ],
        loc="upper right",
    )

    ax_prob.hist(full_p_h, bins=20, color="tab:blue", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_prob.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax_prob.set_title("p(H-first) Distribution")
    ax_prob.set_xlabel("p_H")
    ax_prob.set_ylabel("Edge Count")

    ax_entropy.hist(full_entropy, bins=20, color="tab:orange", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax_entropy.set_title("Soft-L Entropy")
    ax_entropy.set_xlabel("Entropy (bits)")
    ax_entropy.set_ylabel("Edge Count")
    ambiguous_ratio = float(np.mean((full_p_h >= 0.4) & (full_p_h <= 0.6)))
    decisive_ratio = float(np.mean((full_p_h <= 0.1) | (full_p_h >= 0.9)))
    stats_text = "\n".join(
        [
            f"diagonal_edges = {int(np.count_nonzero(is_diagonal))}",
            f"sampled_edges = {int(diag_idx.size)}",
            f"mean_p_H = {float(full_p_h.mean()):.3f}",
            f"mean_entropy = {float(full_entropy.mean()):.3f}",
            f"ambiguous[0.4,0.6] = {ambiguous_ratio:.1%}",
            f"decisive<=0.1|>=0.9 = {decisive_ratio:.1%}",
        ]
    )
    ax_entropy.text(
        0.98,
        0.98,
        stats_text,
        transform=ax_entropy.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Soft L intermediate plot saved to %s", output_path)



def plot_soft_l_scoring_maps(soft_debug, output_path, title_prefix="Soft L Scoring"):
    """Plot cached directional cost and hotspot maps used for soft-L scoring."""
    import matplotlib.pyplot as plt
    import numpy as np

    required = ("cost_map_h", "cost_map_v", "hotspot_map_h", "hotspot_map_v")
    missing = [key for key in required if key not in soft_debug]
    if missing:
        raise ValueError(f"Missing soft-L scoring maps: {missing}")

    has_ggr_overflow = ("ggr_overflow_h" in soft_debug and isinstance(soft_debug["ggr_overflow_h"], torch.Tensor) and
                        "ggr_overflow_v" in soft_debug and isinstance(soft_debug["ggr_overflow_v"], torch.Tensor))
    nrows = 3 if has_ggr_overflow else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5 * nrows), constrained_layout=True)
    plots = [
        ("cost_map_h", "Horizontal-Leg Cost", "viridis"),
        ("cost_map_v", "Vertical-Leg Cost", "viridis"),
        ("hotspot_map_h", "Horizontal Hotspot Map", "magma"),
        ("hotspot_map_v", "Vertical Hotspot Map", "magma"),
    ]
    if has_ggr_overflow:
        plots.extend([
            ("ggr_overflow_h", "GGR H Overflow Ratio", "plasma"),
            ("ggr_overflow_v", "GGR V Overflow Ratio", "plasma"),
        ])

    for ax, (key, title, cmap) in zip(axes.flat, plots):
        value = soft_debug[key]
        if value.requires_grad:
            value = value.detach()
        if value.is_cuda:
            value = value.cpu()
        kwargs = {}
        if key.startswith("ggr_overflow_"):
            vmax = max(1e-6, float(np.percentile(value.numpy(), 99)))
            kwargs = {"vmin": 0.0, "vmax": vmax}
        im = ax.imshow(value.numpy().T, origin="lower", cmap=cmap, aspect="equal", **kwargs)
        ax.set_title(title)
        ax.set_xlabel("Bin X")
        ax.set_ylabel("Bin Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    extra = []
    if "effective_hotspot_weight" in soft_debug:
        extra.append(f"effective_hotspot_weight={soft_debug['effective_hotspot_weight']:.3f}")
    if "overflow_ratio" in soft_debug:
        extra.append(f"overflow_ratio={soft_debug['overflow_ratio']:.4f}")
    fig.suptitle(f"{title_prefix}" + (f" ({', '.join(extra)})" if extra else ""))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Soft L scoring map plot saved to %s", output_path)
