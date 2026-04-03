import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_density_map(density_map, title="Density Map", save_path=None):
    """
    可视化 density map，适合 debug 用
    :param density_map: 2D numpy array or torch.Tensor
    :param title: 图标题
    :param save_path: 如果指定则保存图片，否则直接显示
    """
    if hasattr(density_map, "detach"):
        density_map = density_map.detach().cpu().numpy()

    plt_density_map = density_map.transpose()  # 转置以符合 (x, y) 方向
    plt.figure(figsize=(8, 6))
    plt.imshow(plt_density_map, cmap="binary", interpolation="nearest", origin="lower")
    plt.colorbar(label="Density")
    plt.title(title)
    plt.xlabel("Bin X")
    plt.ylabel("Bin Y")

    # num_bins_y, num_bins_x = density_map.shape
    # for i in range(num_bins_y):
    #     for j in range(num_bins_x):
    #         plt.text(j, i, f"{density_map[i, j]:.0f}", ha="center", va="center", color="black", fontsize=7)

    plt.savefig(save_path, bbox_inches="tight")
    print(f"Density map saved to {save_path}")


def plot_potential_map(potential_map, title="Potential Map", save_path=None, vmin_p=5, vmax_p=95):
    """
    可视化 potential map，使用百分位裁剪以避免极值拉伸色条
    :param potential_map: 2D numpy array or torch.Tensor
    :param title: 图标题
    :param save_path: 如果指定则保存图片，否则直接显示
    :param vmin_p: vmin 百分位
    :param vmax_p: vmax 百分位
    """
    if hasattr(potential_map, "detach"):
        potential_map = potential_map.detach().cpu().numpy()

    plt_map = potential_map.transpose()
    vmin = np.percentile(plt_map, vmin_p)
    vmax = np.percentile(plt_map, vmax_p)
    plt.figure(figsize=(8, 6))
    plt.imshow(plt_map, cmap="binary", interpolation="nearest", origin="lower",
               vmin=vmin, vmax=vmax)
    plt.colorbar(label="Potential")
    plt.title(title)
    plt.xlabel("Bin X")
    plt.ylabel("Bin Y")

    plt.savefig(save_path, bbox_inches="tight")
    print(f"Potential map saved to {save_path}")

def plot_bboxes_on_die(
    pos,
    node_size_x_clamped,
    node_size_y_clamped,
    xl, yl, xh, yh,
    bin_size_x, bin_size_y,
    title="BBoxes on Die",
    save_path=None
):
    """
    可视化所有bbox在die上的分布
    :param pos: [2*N] tensor, 前N为x，后N为y
    :param node_size_x_clamped: [N] tensor
    :param node_size_y_clamped: [N] tensor
    :param xl, yl, xh, yh: die边界
    :param bin_size_x, bin_size_y: bin尺寸（可选，用于画网格）
    :param title: 图标题
    :param save_path: 保存路径
    """
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    if hasattr(node_size_x_clamped, "detach"):
        node_size_x_clamped = node_size_x_clamped.detach().cpu().numpy()
    if hasattr(node_size_y_clamped, "detach"):
        node_size_y_clamped = node_size_y_clamped.detach().cpu().numpy()

    num_bboxes = pos.shape[0] // 2
    bbox_x = pos[:num_bboxes]
    bbox_y = pos[num_bboxes:]
    bbox_w = node_size_x_clamped
    bbox_h = node_size_y_clamped

    fig, ax = plt.subplots(figsize=(8, 8))
    # 画die边界
    ax.add_patch(plt.Rectangle((xl, yl), xh-xl, yh-yl, fill=False, edgecolor='black', linewidth=2, label='Die'))
    # 画bin网格
    num_bins_x = int((xh - xl) / bin_size_x)
    num_bins_y = int((yh - yl) / bin_size_y)
    for i in range(num_bins_x + 1):
        ax.plot([xl + i * bin_size_x, xl + i * bin_size_x], [yl, yh], color='gray', linewidth=0.5, alpha=0.5)
    for j in range(num_bins_y + 1):
        ax.plot([xl, xh], [yl + j * bin_size_y, yl + j * bin_size_y], color='gray', linewidth=0.5, alpha=0.5)
    # 画所有bbox
    for i in range(num_bboxes):
        rect = plt.Rectangle((bbox_x[i], bbox_y[i]), bbox_w[i], bbox_h[i], fill=False, color='blue', alpha=0.4)
        ax.add_patch(rect)
    ax.set_xlim(xl-5, xh+5)
    ax.set_ylim(yl-5, yh+5)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"BBox map saved to {save_path}")
    else:
        plt.show()


def plot_density_with_bboxes(
    density_map, pos, node_size_x, node_size_y,
    xl, yl, xh, yh, bin_size_x, bin_size_y, title="Density+BBoxes", save_path=None
):
    """
    density_map: 2D numpy array or tensor
    pos: [2*N] tensor or array, 前N为x，后N为y
    node_size_x: [N]
    node_size_y: [N]
    xl, yl, xh, yh: die边界
    bin_size_x, bin_size_y: bin尺寸
    """
    if hasattr(density_map, "detach"):
        density_map = density_map.detach().cpu().numpy()
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    if hasattr(node_size_x, "detach"):
        node_size_x = node_size_x.detach().cpu().numpy()
    if hasattr(node_size_y, "detach"):
        node_size_y = node_size_y.detach().cpu().numpy()

    num_bboxes = pos.shape[0] // 2
    bbox_x = pos[:num_bboxes]
    bbox_y = pos[num_bboxes:]
    bbox_w = node_size_x
    bbox_h = node_size_y

    plt.figure(figsize=(8, 8))

    plt_density_map = density_map.transpose()  # 转置以符合 (x, y) 方向

    plt.imshow(
        plt_density_map,
        cmap="binary",
        interpolation="nearest",
        origin="lower",
        extent=(xl, xh, yl, yh)
    )

    plt.colorbar(label="Density")
    ax = plt.gca()
    # 画 die 边界
    ax.add_patch(plt.Rectangle((xl, yl), xh-xl, yh-yl, fill=False, edgecolor='black', linewidth=2))
    # 叠加所有 bbox
    for x, y, w, h in zip(bbox_x, bbox_y, bbox_w, bbox_h):
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=0.8, alpha=0.3)
        ax.add_patch(rect)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
        

def plot_node_grad_directions(pos, grad_1, grad_2 = None, grad_3 = None, title="Node Gradient Directions", save_path="node_grad_direction_map.png"):
    grad_1_x = grad_1[:len(grad_1)//2].cpu().numpy()
    grad_1_y = grad_1[len(grad_1)//2:].cpu().numpy()

    if grad_2 is None:
        grad_2 = torch.zeros_like(grad_1)
    if grad_3 is None:
        grad_3 = torch.zeros_like(grad_1)
        
    grad_2_x = grad_2[:len(grad_2)//2].cpu().numpy()
    grad_2_y = grad_2[len(grad_2)//2:].cpu().numpy()

    grad_3_x = grad_3[:len(grad_3)//2].cpu().numpy()
    grad_3_y = grad_3[len(grad_3)//2:].cpu().numpy()

    pos = pos.detach().cpu().numpy()
    node_x = pos[:len(pos)//2]
    node_y = pos[len(pos)//2:]

    plt.figure(figsize=(32,32))
    N = len(node_x)
    scale_factor = 50
    plt.scatter(node_x[:N], node_y[:N], s=6, color='gray')
    plt.quiver(
        node_x[:N], node_y[:N], grad_1_x[:N] * scale_factor, grad_1_y[:N] * scale_factor,
        color='red', scale=1, scale_units='xy', width=0.0008, headwidth=4, headlength=5, headaxislength=4, label='Gradient 1'
    )
    plt.quiver(
        node_x[:N], node_y[:N], grad_2_x[:N] * scale_factor, grad_2_y[:N] * scale_factor,
        color='blue', scale=1, scale_units='xy', width=0.0008, headwidth=4, headlength=5, headaxislength=4, label='Gradient 2'
    )
    plt.quiver(
        node_x[:N], node_y[:N], grad_3_x[:N] * scale_factor, grad_3_y[:N] * scale_factor,
        color='green', scale=1, scale_units='xy', width=0.0008, headwidth=4, headlength=5, headaxislength=4, label='Gradient 3'
    )
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Node gradient direction map saved to {save_path}")
