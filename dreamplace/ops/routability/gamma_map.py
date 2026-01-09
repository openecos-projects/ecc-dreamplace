import torch
import matplotlib.pyplot as plt
import numpy as np
import time

class RobustGammaModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

    def path_count_density_torch(
        self,
        x: torch.Tensor, y: torch.Tensor,
        W: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-8
        n1 = x + y
        k1 = x
        log_comb_1 = torch.lgamma(n1 + 1 + eps) - torch.lgamma(k1 + 1 + eps) - torch.lgamma(n1 - k1 + 1 + eps)

        n2 = (W - x) + (H - y)
        k2 = W - x
        log_comb_2 = torch.lgamma(n2 + 1 + eps) - torch.lgamma(k2 + 1 + eps) - torch.lgamma(n2 - k2 + 1 + eps)

        n3 = W + H
        k3 = W
        log_comb_3 = torch.lgamma(n3 + 1 + eps) - torch.lgamma(k3 + 1 + eps) - torch.lgamma(n3 - k3 + 1 + eps)

        demand = log_comb_1 + log_comb_2 - log_comb_3
        demand = torch.exp(demand)
        return torch.nan_to_num(demand, nan=-np.inf, posinf=0, neginf=-np.inf)

    def generate_demand_map_final(
        self,
        edges: torch.Tensor,
        die_xl: float, die_yl: float, die_xh: float, die_yh: float,
        bin_num_x: int, bin_num_y: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        
        num_edges = edges.shape[0]

        # --- 1. 准备所有bin和edge的坐标信息 (一次性) ---
        bin_size_x = (die_xh - die_xl) / bin_num_x
        bin_size_y = (die_yh - die_yl) / bin_num_y

        # bin中心点坐标 (1D张量)
        # shape: [num_bins_x]
        bin_centers_x = die_xl + (torch.arange(bin_num_x, device=self.device, dtype=dtype) + 0.5) * bin_size_x
        # shape: [num_bins_y]
        bin_centers_y = die_yl + (torch.arange(bin_num_y, device=self.device, dtype=dtype) + 0.5) * bin_size_y

        s1, s2 = edges[:, :2], edges[:, 2:]
        
        # BBox边界和尺寸 (1D张量)
        # shape: [num_edges]
        llx = torch.min(s1[:, 0], s2[:, 0])
        lly = torch.min(s1[:, 1], s2[:, 1])
        urx = torch.max(s1[:, 0], s2[:, 0])
        ury = torch.max(s1[:, 1], s2[:, 1])
        W_real = urx - llx
        H_real = ury - lly

        # --- 2. 向量化坐标变换 ---
        # 使用广播计算所有bin相对于所有BBox的默认局部坐标 (LL->UR)
        # a. 扩展维度以启用广播
        #    real_x: [num_bins_x] -> [1, num_bins_x]
        #    llx, W_real: [num_edges] -> [num_edges, 1]
        norm_x_base = (bin_centers_x.unsqueeze(0) - llx.unsqueeze(1)) / torch.clamp(W_real.unsqueeze(1), min=1e-6)
        norm_y_base = (bin_centers_y.unsqueeze(0) - lly.unsqueeze(1)) / torch.clamp(H_real.unsqueeze(1), min=1e-6)

        # b. 创建方向判断的布尔掩码 (boolean masks)
        # shape: [num_edges]
        start_at_right = s1[:, 0] > s2[:, 0]
        start_at_top = s1[:, 1] > s2[:, 1]

        # c. 使用torch.where根据掩码进行条件翻转，这等效于if/elif逻辑
        #    掩码扩展维度: [num_edges] -> [num_edges, 1]
        norm_x = torch.where(start_at_right.unsqueeze(1), 1.0 - norm_x_base, norm_x_base)
        norm_y = torch.where(start_at_top.unsqueeze(1), 1.0 - norm_y_base, norm_y_base)
        
        # --- 3. 在单位盒子中计算log-demand ---
        # 扩展维度以进行最终计算
        # norm_x: [num_edges, num_bins_x] -> [num_edges, 1, num_bins_x]
        # norm_y: [num_edges, num_bins_y] -> [num_edges, num_bins_y, 1]
        # 最终广播后的形状为 [num_edges, num_bins_y, num_bins_x]
        unit_W = torch.tensor(1.0, dtype=dtype, device=self.device)
        unit_H = torch.tensor(1.0, dtype=dtype, device=self.device)

        demand_shape = self.path_count_density_torch(
            norm_x.unsqueeze(1), norm_y.unsqueeze(2), unit_W, unit_H
        )
        

        # --- 4. 创建BBox掩码并应用 ---
        # is_in_x: [num_edges, num_bins_x]
        is_in_x = (bin_centers_x.unsqueeze(0) >= llx.unsqueeze(1)) & (bin_centers_x.unsqueeze(0) <= urx.unsqueeze(1))
        is_in_y = (bin_centers_y.unsqueeze(0) >= lly.unsqueeze(1)) & (bin_centers_y.unsqueeze(0) <= ury.unsqueeze(1))
        # mask: [num_edges, num_bins_y, num_bins_x]
        mask = is_in_y.unsqueeze(2) & is_in_x.unsqueeze(1)
        
        # 将BBox外的demand设为负无穷，使其在求和中贡献为0
        demand_shape[~mask] = 0

        # --- 5. 缩放并累加 ---
        # magnitude = W_real + H_real
        # # 在对数空间中，乘法变成加法
        # final_demand = demand_shape + torch.log(torch.clamp(magnitude, min=1e-6)).view(num_edges, 1, 1)

        # 在线性空间中，使用乘法进行缩放
        # final_demand = demand_shape * magnitude.view(num_edges, 1, 1)   
        
        # 沿着edge的维度(dim=0)求和，得到最终的2D map
        demand_map = torch.sum(demand_shape, dim=0)
        
        
        return demand_map
    
    def generate_demand_map_final_(
        self,
        edges: torch.Tensor,
        die_xl: float, die_yl: float, die_xh: float, die_yh: float,
        bin_num_x: int, bin_num_y: int,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        
        die_width = die_xh - die_xl
        die_height = die_yh - die_yl
        bin_width = die_width / bin_num_x
        bin_height = die_height / bin_num_y
        
        demand_map = torch.zeros((bin_num_y, bin_num_x), dtype=dtype, device=self.device)
        
        for i in range(edges.shape[0]):
            edge = edges[i]
            s1, s2 = edge[:2], edge[2:]

            if s1[0] < s2[0] and s1[1] < s2[1]:
                flag = 0
                # ll -> ur
            elif s1[0] < s2[0] and s1[1] > s2[1]:
                flag = 1
                # ul -> lr
            elif s1[0] > s2[0] and s1[1] < s2[1]:
                flag = 2
                # lr -> ul
            elif s1[0] > s2[0] and s1[1] > s2[1]:
                flag = 3
                # ur -> ll

            llx, lly = min(s1[0].item(), s2[0].item()), min(s1[1].item(), s2[1].item())
            urx, ury = max(s1[0].item(), s2[0].item()), max(s1[1].item(), s2[1].item())
            W_real, H_real = urx - llx, ury - lly

            if W_real < 1e-6 or H_real < 1e-6:
                continue

            start_bin_x = max(0, int((llx - die_xl) / bin_width))
            end_bin_x = min(bin_num_x, int(torch.ceil(torch.tensor((urx - die_xl) / bin_width)).item()))
            start_bin_y = max(0, int((lly - die_yl) / bin_height))
            end_bin_y = min(bin_num_y, int(torch.ceil(torch.tensor((ury - die_yl) / bin_height)).item()))
            
            if start_bin_x >= end_bin_x or start_bin_y >= end_bin_y:
                continue

            for bin_x in range(start_bin_x, end_bin_x):
                for bin_y in range(start_bin_y, end_bin_y):
                    bin_center_x = die_xl + (bin_x + 0.5) * bin_width
                    bin_center_y = die_yl + (bin_y + 0.5) * bin_height
                    rel_x = (bin_center_x - llx) / W_real 
                    rel_y = (bin_center_y - lly) / H_real
                    if flag == 1:
                        rel_y = 1.0 - rel_y
                    elif flag == 2:
                        rel_x = 1.0 - rel_x
                    elif flag == 3:
                        rel_x = 1.0 - rel_x
                        rel_y = 1.0 - rel_y
                    demand = self.path_count_density_torch(
                        torch.tensor(rel_x, dtype=dtype, device=self.device),
                        torch.tensor(rel_y, dtype=dtype, device=self.device),
                        torch.tensor(1.0, dtype=dtype, device=self.device),
                        torch.tensor(1.0, dtype=dtype, device=self.device)
                    )
                    demand_map[bin_y, bin_x] += demand
        
        return demand_map



# --- 测试代码 ---
if __name__ == '__main__':
    model = RobustGammaModel()
    DIE_XL, DIE_YL, DIE_XH, DIE_YH = 0.0, 0.0, 1000.0, 1000.0
    BIN_NUM_X, BIN_NUM_Y = 200, 200

    edges_to_test = torch.tensor([
        [100, 100, 800, 900],  # LL -> UR
        [450, 450, 550, 580],  # 一个小的 LL -> UR
        [100, 900, 400, 600],  # UL -> LR
        [900, 100, 600, 400],  # LR -> UL

        # [100, 100, 100, 800],
    ], dtype=torch.float32, device=model.device)

    start_time = time.time()
    
    the_log_map = model.generate_demand_map_final(
        edges_to_test,
        DIE_XL, DIE_YL, DIE_XH, DIE_YH,
        BIN_NUM_X, BIN_NUM_Y
    )
    
    end_time = time.time()

    print(f"Demand map generated in {end_time - start_time:.4f} seconds.")


    # start_time = time.time()
    # model.generate_demand_map_final_(
    #     edges_to_test,
    #     DIE_XL, DIE_YL, DIE_XH, DIE_YH,
    #     BIN_NUM_X, BIN_NUM_Y 
    # )

    # end_time = time.time()
    # print(f"(Naive) Demand map generated in {end_time - start_time:.4f} seconds.")

    # --- 可视化 ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    # 因为输出已经是log-demand，我们直接绘制即可，不再需要LogNorm
    map_to_plot = the_log_map.cpu().numpy()
    
    im = ax.imshow(map_to_plot, cmap='inferno', origin='lower', 
                   extent=(DIE_XL, DIE_XH, DIE_YL, DIE_YH))
                   
    for edge in edges_to_test.cpu().numpy():
        ax.plot([edge[0], edge[2]], [edge[1], edge[3]], color='cyan', linewidth=1.0, linestyle='--')
    
    ax.set_title('Log Demand Map (Gamma Model in Unit Box)')
    ax.set_xlabel('Die X Coordinate')
    ax.set_ylabel('Die Y Coordinate')
    ax.set_xlim(DIE_XL, DIE_XH)
    ax.set_ylim(DIE_YL, DIE_YH)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Log Demand')
    plt.grid(False)
    plt.savefig("robust_gamma_demand_map.png", bbox_inches="tight", dpi=300)