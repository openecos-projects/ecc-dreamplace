import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
np.Inf = np.inf

# 尝试导入 TensorFlow Lattice
try:
    # 强制使用 Legacy Keras (Keras 2) 以兼容 TensorFlow Lattice
    # 必须在导入 tensorflow 之前设置，解决 "KerasTensor cannot be used as input to a TensorFlow function" 错误
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    import tensorflow as tf
    import tensorflow_lattice as tfl
    # 禁用 GPU 以避免显存占用过高，因为这是轻量级任务
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    TFL_AVAILABLE = True
    print("TensorFlow Lattice is available. TFL models will be trained.")
except ImportError:
    TFL_AVAILABLE = False
    print("Warning: TensorFlow Lattice not found. TFL modeling will be skipped.")

# ==========================================
# 工具函数
# ==========================================

def parse_semicolon_list(s):
    """解析分号分隔的字符串为浮点列表"""
    if pd.isna(s) or s == "":
        return []
    try:
        return [float(x) for x in str(s).split(';') if x.strip()]
    except ValueError:
        return []

def get_safe_filename(name):
    return str(name).replace(":", "_").replace(" ", "_").replace("/", "_")

# ==========================================
# 可视化模块 (增强版)
# ==========================================

def visualize_physics_view(group_df, group_name, output_dir="plots"):
    """
    物理视角可视化：
    1. 传统的 3D 视图
    2. 关键：固定 Slew 的截面图 (Cross-section)，展示 Size 对驱动能力的影响
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    safe_name = get_safe_filename(group_name)
    print(f"Plotting for group: {group_name}...")

    # 设置绘图风格
    sns.set_theme(style="whitegrid")

    # --- 图 1: 物理截面图 (Cross-section Analysis) ---
    # 逻辑：选取 Low, Med, High 三个 Input Slew，分别固定它们
    # 在同一张图上使用子图 (Subplots) 展示不同 VT 的 Delay vs Cap 曲线
    
    # 获取所有 VT
    vts = sorted(group_df['VT'].unique()) if 'VT' in group_df.columns else ['N/A']
    num_vts = len(vts)
    
    if num_vts == 0:
        return

    # 确定 Target Slews
    unique_slews = sorted(group_df['axis1_val'].unique())
    if len(unique_slews) >= 3:
        target_slews = [unique_slews[0], unique_slews[len(unique_slews)//2], unique_slews[-1]]
    else:
        target_slews = unique_slews
    
    for slew_val in target_slews:
        # 创建 Subplots: 1 行 num_vts 列
        # 动态调整图片宽度
        fig, axes = plt.subplots(1, num_vts, figsize=(6 * num_vts, 6), sharey=False)
        if num_vts == 1:
            axes = [axes] # 统一为列表处理
            
        fig.suptitle(f'Delay vs Cap (Fixed Slew={slew_val:.4g}) - Group: {group_name}', fontsize=16)
        
        has_data = False
        
        for i, vt in enumerate(vts):
            ax = axes[i]
            
            # 筛选特定 VT 和 Slew 的数据
            if vt == 'N/A':
                subset = group_df[np.isclose(group_df['axis1_val'], slew_val)]
            else:
                subset = group_df[(group_df['VT'] == vt) & np.isclose(group_df['axis1_val'], slew_val)]
            
            if not subset.empty:
                has_data = True
                # 使用 hue='cell_size' 区分不同尺寸
                sns.lineplot(data=subset, x='axis2_val', y='value', 
                             hue='cell_size', 
                             markers=True, dashes=False, palette="viridis", ax=ax)
                
                ax.set_title(f'VT: {vt}')
                ax.set_xlabel('Output Load Capacitance')
                if i == 0:
                    ax.set_ylabel('Cell Delay')
                else:
                    ax.set_ylabel('') # 只有第一个图显示 Y 轴标签
                
                # 图例处理：只在最后一个子图显示图例，避免拥挤
                if i == num_vts - 1:
                    ax.legend(title='Cell Size', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    if ax.get_legend():
                        ax.get_legend().remove()
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        
        if has_data:
            plt.tight_layout()
            # 调整顶部空间以容纳 suptitle
            plt.subplots_adjust(top=0.9)
            safe_name = get_safe_filename(f"{group_name}_slew_{slew_val:.4g}_multi_vt")
            plt.savefig(f"{output_dir}/{safe_name}_delay_vs_cap.png")
        
        plt.close()

    # --- 图 2: Delay vs Slew (Fixed Cap) ---
    # 逻辑：选取 Low, Med, High 三个 Output Cap，分别固定它们
    # 在同一张图上使用子图 (Subplots) 展示不同 VT 的 Delay vs Slew 曲线
    
    # 确定 Target Caps
    unique_caps = sorted(group_df['axis2_val'].unique())
    if len(unique_caps) >= 3:
        target_caps = [unique_caps[0], unique_caps[len(unique_caps)//2], unique_caps[-1]]
    else:
        target_caps = unique_caps
        
    for cap_val in target_caps:
        # 创建 Subplots: 1 行 num_vts 列
        fig, axes = plt.subplots(1, num_vts, figsize=(6 * num_vts, 6), sharey=False)
        if num_vts == 1:
            axes = [axes]
            
        fig.suptitle(f'Delay vs Slew (Fixed Cap={cap_val:.4g}) - Group: {group_name}', fontsize=16)
        
        has_data = False
        
        for i, vt in enumerate(vts):
            ax = axes[i]
            
            # 筛选特定 VT 和 Cap 的数据
            if vt == 'N/A':
                subset = group_df[np.isclose(group_df['axis2_val'], cap_val)]
            else:
                subset = group_df[(group_df['VT'] == vt) & np.isclose(group_df['axis2_val'], cap_val)]
            
            if not subset.empty:
                has_data = True
                # 使用 hue='cell_size' 区分不同尺寸
                sns.lineplot(data=subset, x='axis1_val', y='value', 
                             hue='cell_size', 
                             markers=True, dashes=False, palette="viridis", ax=ax)
                
                ax.set_title(f'VT: {vt}')
                ax.set_xlabel('Input Slew (axis1)')
                if i == 0:
                    ax.set_ylabel('Cell Delay')
                else:
                    ax.set_ylabel('')
                
                if i == num_vts - 1:
                    ax.legend(title='Cell Size', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    if ax.get_legend():
                        ax.get_legend().remove()
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        
        if has_data:
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            safe_name = get_safe_filename(f"{group_name}_cap_{cap_val:.4g}_multi_vt")
            plt.savefig(f"{output_dir}/{safe_name}_delay_vs_slew.png")
        
        plt.close()

    # --- 图 3: Electrical Effort 视图 (Delay vs Cap/Size) ---
    # 理论上，如果器件设计合理，不同 Size 的点应该重合在一条线上（归一化）
    # group_df = group_df.copy() # 防止SettingWithCopyWarning
    # group_df['eff_effort'] = group_df['axis2_val'] / group_df['cell_size'].replace(0, np.nan)
    
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=group_df, x='eff_effort', y='value', 
    #                 hue='cell_size', palette='coolwarm', alpha=0.6)
    # plt.title(f'Logical Effort View: Delay vs Cap/Size\nGroup: {group_name}')
    # plt.xlabel('Electrical Effort (Cap / Size)')
    # plt.ylabel('Cell Delay')
    # plt.grid(True)
    # plt.savefig(f"{output_dir}/{safe_name}_logical_effort.png")
    # plt.close()

def check_and_plot_size_monotonicity(group_df, group_name, output_dir="plots/monotonicity"):
    """
    检查并绘制固定 Cap 和 Slew 下，Delay 随 Size 变化的单调性
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保按 VT 分组处理
    vts = group_df['VT'].unique() if 'VT' in group_df.columns else ['N/A']
    
    for vt in vts:
        if vt == 'N/A':
            vt_group = group_df
        else:
            vt_group = group_df[group_df['VT'] == vt]
            
        if vt_group.empty:
            continue
            
        # 辅助列用于分组 (避免浮点精度问题)
        vt_group = vt_group.copy()
        # Cap 和 Slew 可能非常小，先放缩再取整，避免精度丢失
        # 使用 1e15 作为缩放因子，足以覆盖 fF/fs 级别的精度
        scale_factor = 1e15
        vt_group['slew_idx'] = (vt_group['axis1_val'])
        vt_group['cap_idx'] = (vt_group['axis2_val'])
        
        # 统计每个 (Slew, Cap) 组合下的 Size 数量
        counts = vt_group.groupby(['slew_idx', 'cap_idx'])['cell_size'].nunique()
        valid_indices = counts[counts >= 3].index # 至少3个点才能看趋势
        
        if len(valid_indices) == 0:
            continue
            
        non_monotonic_count = 0
        total_checks = 0
        plot_data = []
        
        # 选取用于绘图的代表性 Slew (低、中、高)
        unique_slew_idxs = sorted(vt_group['slew_idx'].unique())
        if len(unique_slew_idxs) >= 3:
            target_slew_idxs = [unique_slew_idxs[0], unique_slew_idxs[len(unique_slew_idxs)//2], unique_slew_idxs[-1]]
        else:
            target_slew_idxs = unique_slew_idxs
        
        # 选取用于绘图的代表性 Cap (均匀选取 5 个点)
        unique_cap_idxs = sorted(vt_group['cap_idx'].unique())
        if len(unique_cap_idxs) >= 5:
            indices = np.linspace(0, len(unique_cap_idxs)-1, 5, dtype=int)
            target_cap_idxs = [unique_cap_idxs[i] for i in indices]
        elif len(unique_cap_idxs) >= 3:
            target_cap_idxs = [unique_cap_idxs[0], unique_cap_idxs[len(unique_cap_idxs)//2], unique_cap_idxs[-1]]
        else:
            target_cap_idxs = unique_cap_idxs

        # 遍历所有组合检查单调性
        for (s_idx, c_idx), sub in vt_group.groupby(['slew_idx', 'cap_idx']):
            if len(sub) < 3:
                continue
                
            # 按 Size 排序
            sub = sub.sort_values('cell_size')
            delays = sub['value'].values
            
            # 检查单调性 (递增或递减均可)
            is_increasing = np.all(np.diff(delays) >= -1e-9) # 允许微小误差
            is_decreasing = np.all(np.diff(delays) <= 1e-9)
            
            if not (is_increasing or is_decreasing):
                non_monotonic_count += 1
            
            total_checks += 1
            
            # 收集绘图数据 (匹配索引)
            if s_idx in target_slew_idxs and c_idx in target_cap_idxs:
                 plot_data.append(sub)

        # assert len(plot_data) == len(target_slew_idxs) * len(target_cap_idxs), "No data collected for plotting."
        # 打印统计信息
        if total_checks > 0:
            ratio = non_monotonic_count / total_checks
            if ratio > 0.1: # 如果超过 10% 不单调，打印警告
                print(f"  [Monotonicity] Group: {group_name}, VT: {vt} -> Non-monotonic: {non_monotonic_count}/{total_checks} ({ratio:.1%})")

        # 绘图
        if plot_data:
            plt.figure(figsize=(12, 8))
            combined = pd.concat(plot_data)
            
            # 优化图例显示：Hue=Cap, Style=Slew
            # 使用原始值显示 Label
            combined['Slew_Label'] = combined['axis1_val'].apply(lambda x: f"Slew={x:.4g}")
            combined['Cap_Label'] = combined['axis2_val'].apply(lambda x: f"Cap={x:.4g}")
            
            sns.lineplot(
                data=combined, 
                x='cell_size', 
                y='value', 
                hue='Cap_Label', 
                style='Slew_Label', 
                markers=True, 
                dashes=False,
                palette="viridis"
            )
            
            plt.title(f'Size vs Delay Monotonicity Check (Multi-Cap/Slew)\nGroup: {group_name}, VT: {vt}')
            plt.xlabel('Cell Size')
            plt.ylabel('Delay')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            
            safe_name = get_safe_filename(f"{group_name}_{vt}")
            plt.savefig(f"{output_dir}/{safe_name}_size_monotonicity.png")
            plt.close()

# ==========================================
# 建模模块 (物理模型 vs 纯数学模型)
# ==========================================

def fit_and_evaluate(X, y, model_name):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return model, r2, rmse

def perform_physics_modeling(df):
    """
    对比两种建模方式：
    1. Baseline: 简单的多项式回归
    2. Physics-Aware: 基于 Elmore Delay 和 Logical Effort 的特征工程
    """
    print("\nPerforming Physics-Aware Modeling Analysis...")
    
    # 数据展开 (Expand)
    print("Expanding LUT data...")
    expanded_rows = []
    
    # 使用 itertuples 加速遍历
    for row in df.itertuples():
        a1_list = row.axis1_parsed # Slew
        a2_list = row.axis2_parsed # Cap
        vals = row.values_parsed   # Delay
        
        # 检查维度一致性
        expected_len = len(a1_list) * len(a2_list)
        
        # 严格断言：确保 LUT 值数量等于两个轴维度的乘积
        assert len(vals) == expected_len, f"Dimension mismatch in row {row.Index}: values length {len(vals)} != axis1 length {len(a1_list)} * axis2 length {len(a2_list)}"

        if len(vals) == 0:
            continue
            
        for i, slew in enumerate(a1_list):
            for j, cap in enumerate(a2_list):
                idx = i * len(a2_list) + j
                if idx < len(vals):
                    expanded_rows.append({
                        'arc': f"{row.pin_name}->{row.related_pin_name}",
                        'type': f"{row.main_type}_{row.table_type}",
                        'cell_size': float(row.cell_size),
                        'VT': row.VT if hasattr(row, 'VT') else 'N/A',
                        'idx_slew':i,
                        'idx_cap': j,
                        'axis1_val': slew, # Input Slew
                        'axis2_val': cap,  # Output Cap
                        'value': vals[idx] # Delay
                    })

    if not expanded_rows:
        print("No valid data found.")
        return

    df_exp = pd.DataFrame(expanded_rows)
    
    # 分组建模
    groups = df_exp.groupby(['arc', 'type'])
    results = []
    low_r2_groups = []
    
    visualized_types = set()

    print(f"\n{'Group Name':<40} | {'Poly R2':<8} | {'Phys R2':<8} | {'Pruned R2':<9} | {'TFL R2':<8} | {'Impv':<5} | {'Key Coeff (1/Size)'}")
    print("-" * 120)

    for name, group in groups:
        if len(group) < 20: continue # 忽略数据太少的组
        
        group_id = f"{name[0]}:{name[1]}"
        
        # 1. 抽样可视化 (每种类型画一次)
        table_type_key = name[1]
        # if table_type_key not in visualized_types and len(group) > 50:
            # visualize_physics_view(group, group_id)
            # check_and_plot_size_monotonicity(group, group_id)
            # visualized_types.add(table_type_key)

        # 准备基础特征
        group = group.copy()
        
        # --- Model A: 纯数学多项式 (Baseline) ---
        # Features: Slew, Cap, Size, Slew^2, Cap^2, Slew*Cap
        Xa = group[['axis1_val', 'axis2_val', 'cell_size']].copy()
        Xa['slew_sq'] = Xa['axis1_val'] ** 2
        Xa['cap_sq'] = Xa['axis2_val'] ** 2
        Xa['interact'] = Xa['axis1_val'] * Xa['axis2_val']
        # One-hot VT
        Xa = pd.concat([Xa, pd.get_dummies(group['VT'], prefix='VT')], axis=1)
        
        model_a, r2_a, rmse_a = fit_and_evaluate(Xa, group['value'], "Poly")

        # --- Model B: 物理感知模型 (Physics-Based) ---
        # Features: Slew, Cap, 1/Size, Cap/Size
        # 理论: Delay ~ R_driver * C_load + Intrinsic
        # R_driver ~ 1/Size, 所以主要项是 Cap/Size
        Xb = group[['axis1_val', 'axis2_val']].copy()
        
        # 避免除以0
        size_safe = group['cell_size'].replace(0, 1e-9)
        
        Xb['inv_size'] = 1.0 / size_safe          # 物理含义: Intrinsic Delay 随尺寸的变化
        Xb['cap_div_size'] = Xb['axis2_val'] / size_safe  # 物理含义: 归一化的 RC 延时 (Logical Effort)
        
        # One-hot VT (VT 影响阈值电压，从而影响延迟基数)
        Xb = pd.concat([Xb, pd.get_dummies(group['VT'], prefix='VT')], axis=1)
        
        model_b, r2_b, rmse_b = fit_and_evaluate(Xb, group['value'], "Physics")

        # --- Model C: Physics (Pruned - Solution A) ---
        # 剔除 Delay 最大的 10% 数据点，去除 "曲棍球棒" 效应 (High Load Filtering)
        limit = group['value'].quantile(0.90)
        group_pruned = group[group['value'] < limit].copy()
        
        Xb_pruned = group_pruned[['axis1_val', 'axis2_val']].copy()
        size_safe_pruned = group_pruned['cell_size'].replace(0, 1e-9)
        Xb_pruned['inv_size'] = 1.0 / size_safe_pruned
        Xb_pruned['cap_div_size'] = Xb_pruned['axis2_val'] / size_safe_pruned
        Xb_pruned = pd.concat([Xb_pruned, pd.get_dummies(group_pruned['VT'], prefix='VT')], axis=1)
        
        # Align columns (fill missing VT columns with 0)
        for col in Xb.columns:
            if col not in Xb_pruned.columns:
                Xb_pruned[col] = 0
        Xb_pruned = Xb_pruned[Xb.columns]

        model_c, r2_c, rmse_c = fit_and_evaluate(Xb_pruned, group_pruned['value'], "Physics_Pruned")
        
        # --- Model D: TensorFlow Lattice (Convex Guaranteed) ---
        model_tfl, r2_tfl, rmse_tfl = train_tfl_model(group, target_col='value', table_type=name[1])
        # model_tfl, r2_tfl, rmse_tfl = 0, 0, 0
        impv = r2_c - r2_b

        # 过滤 R2 < 0.8 的部分并收集
        if r2_b < 0.8:
            low_r2_groups.append({
                'group': group_id,
                'poly_r2': r2_a,
                'physics_r2': r2_b,
                'pruned_physics_r2': r2_c,
                'tfl_r2': r2_tfl,
                'poly_rmse': rmse_a,
                'physics_rmse': rmse_b,
                'pruned_physics_rmse': rmse_c,
                'tfl_rmse': rmse_tfl
            })

        # 打印对比结果
        # 获取 Model B 中 'inv_size' 的系数，观察内阻特性
        inv_size_coef = 0
        if 'inv_size' in Xb.columns:
            # 找到 inv_size 在列中的位置
            col_idx = list(Xb.columns).index('inv_size')
            inv_size_coef = model_b.coef_[col_idx]

        tfl_str = f"{r2_tfl:.4f}" if not np.isnan(r2_tfl) else "N/A"
        print(f"{group_id[:40]:<40} | {r2_a:.4f}   | {r2_b:.4f}   | {r2_c:.4f}    | {tfl_str:<8} | {impv:+.2f} | {inv_size_coef:.4f}")
        
        results.append({
            'group': group_id,
            'poly_r2': r2_a,
            'physics_r2': r2_b,
            'pruned_physics_r2': r2_c,
            'tfl_r2': r2_tfl,
            'poly_rmse': rmse_a,
            'physics_rmse': rmse_b,
            'pruned_physics_rmse': rmse_c,
            'tfl_rmse': rmse_tfl,
            'physics_coeffs': str(dict(zip(Xb.columns, model_b.coef_)))
        })

    if results:
        pd.DataFrame(results).to_csv("modeling_comparison.csv", index=False)
        print("\nModel comparison saved to modeling_comparison.csv")

    if low_r2_groups:
        pd.DataFrame(low_r2_groups).to_csv("low_r2_cases.csv", index=False)
        print(f"\nFound {len(low_r2_groups)} groups with Physics R2 < 0.8. Saved to low_r2_cases.csv")

# ==========================================
# 主流程
# ==========================================

def analyze_lut_data(csv_path="lut_data.csv", output_excel="lut_analysis.xlsx"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 提取 VT (假设 VT 是 cell_name 的最后一个字符，根据你的数据调整)
    # 例如 AND2X1H7R -> R (或 H7R? 视你的命名规则而定)
    # 这里简单取最后一个字符作为 VT 标识
    if 'VT' not in df.columns:
        df['VT'] = df['cell_name'].str[-1]

    # 解析列表列
    print("Parsing array columns...")
    cols_to_parse = ['axis1', 'axis2', 'values']
    for col in cols_to_parse:
        df[f'{col}_parsed'] = df[col].apply(parse_semicolon_list)
    
    # 执行物理建模和可视化
    perform_physics_modeling(df)

    # 保存基础分析到 Excel
    print(f"\nWriting summary to {output_excel}...")
    try:
        # 只保存原始列，不保存解析后的长列表，保持 Excel 整洁
        df_out = df.drop(columns=[f'{c}_parsed' for c in cols_to_parse])
        df_out.to_excel(output_excel, index=False)
        print("Done.")
    except Exception as e:
        print(f"Excel write failed: {e}. Saving CSV.")
        df.to_csv(output_excel.replace('.xlsx', '.csv'), index=False)

def train_tfl_model(group_df, target_col='value', table_type=''):
    """
    使用 TensorFlow Lattice 训练保证凸性的模型 (增强版)
    改进点：
    1. Log 空间变换
    2. Quantile Keypoints 初始化
    3. 正则化 (Torsion/Laplacian)
    """
    if not TFL_AVAILABLE:
        return None, np.nan, np.nan

    # 识别是否为时序约束表
    is_constraint = any(x in table_type.lower() for x in ['setup', 'hold', 'recovery', 'removal'])

    # ---------------------------------------------------------
    # 改进 1: 数据预处理 (Log 变换)
    # ---------------------------------------------------------
    # 物理上 Delay vs Slew/Cap 在对数域更接近线性/凸函数
    # 使用 log1p 避免 log(0) 问题
    
    # 提取原始数据
    raw_slew = group_df['axis1_val'].values.astype(np.float32)
    raw_cap = group_df['axis2_val'].values.astype(np.float32)
    raw_size = group_df['cell_size'].values.astype(np.float32)
    raw_y = group_df[target_col].values.astype(np.float32)
    
    if np.any(raw_y <= -1):
        print("Warning: raw_y contains values <= -1, log1p will fail.")
        raw_y.clip(min=-0.999, out=raw_y)

    # 可以选择 clip 或者过滤
    # 转换为 Log 域输入
    # 注意：Size 不需要 Log，因为我们用的是 1/Size，本身就是非线性变换
    X_slew_log = np.log1p(raw_slew)
    X_cap_log = np.log1p(raw_cap)
    
    # 计算 1/Size (保持不变)
    size_safe = np.where(raw_size < 1e-9, 1e-9, raw_size)
    inv_size_val = 1.0 / size_safe
    
    # 目标值也 Log 变换
    y_log = np.log1p(raw_y)

    # VT One-Hot
    if 'VT' in group_df.columns:
        vt_dummies = pd.get_dummies(group_df['VT'], prefix='VT')
        X_vt = vt_dummies.values.astype(np.float32)
    else:
        X_vt = np.zeros((len(group_df), 1), dtype=np.float32)

    # ---------------------------------------------------------
    # 辅助函数: Quantile Keypoints 初始化
    # ---------------------------------------------------------
    def get_quantile_keypoints(data, num_points=5):
        # 确保数据唯一并排序，避免分位数重叠
        unique_data = np.unique(data)
        if len(unique_data) < num_points:
            return np.linspace(unique_data.min(), unique_data.max(), num_points)
        # 均匀分布的分位数 (0%, 25%, 50%, 75%, 100%)
        return np.percentile(unique_data, np.linspace(0, 100, num_points))

    # ---------------------------------------------------------
    # 构建模型
    # ---------------------------------------------------------
    input_slew = tf.keras.Input(shape=(1,), name='slew_log')
    input_cap = tf.keras.Input(shape=(1,), name='cap_log')
    input_inv_size = tf.keras.Input(shape=(1,), name='inv_size')
    input_vt = tf.keras.Input(shape=(X_vt.shape[1],), name='vt')

    # 约束设置
    mono_slew = 'increasing'
    conv_slew = 'convex'
    mono_cap = 'increasing'
    conv_cap = 'convex'
    mono_inv_size = 'increasing' # Size越大 -> 1/Size越小 -> Delay越小 (即随 1/Size 递增)
    conv_inv_size = 'convex'

    # Calibrators (使用 Quantile Keypoints)
    calib_slew = tfl.layers.PWLCalibration(
        input_keypoints=get_quantile_keypoints(X_slew_log, 5), # 改进 2
        dtype=tf.float32,
        monotonicity=mono_slew,
        convexity=conv_slew,
        output_min=0.0,
        output_max=1.0
    )(input_slew)

    calib_cap = tfl.layers.PWLCalibration(
        input_keypoints=get_quantile_keypoints(X_cap_log, 5), # 改进 2
        dtype=tf.float32,
        monotonicity=mono_cap,
        convexity=conv_cap,
        output_min=0.0,
        output_max=1.0
    )(input_cap)

    # 1/Size 的 Keypoints 初始化
    if len(np.unique(inv_size_val)) > 1:
        kp_inv_size = get_quantile_keypoints(inv_size_val, 5)
    else:
        kp_inv_size = np.linspace(inv_size_val.min(), inv_size_val.max(), 2)

    calib_inv_size = tfl.layers.PWLCalibration(
        input_keypoints=kp_inv_size,
        dtype=tf.float32,
        monotonicity=mono_inv_size,
        convexity=conv_inv_size,
        output_min=0.0,
        output_max=1.0
    )(input_inv_size)

    # Lattice Layer with Regularization
    lattice_monos = ['increasing', 'increasing', 'increasing'] if not is_constraint else None
    my_lattice_sizes = [3, 3, 3]
    
    # 改进 3: 添加正则化以处理脏数据
    # Torsion: 抑制 Lattice 网格的扭曲，使曲面平滑
    # Laplacian: 抑制局部曲率过大
    # 正则化器位于 tfl.lattice_layer 模块中
    reg_torsion = tfl.lattice_layer.TorsionRegularizer(
        lattice_sizes=my_lattice_sizes, # 必须传入！
        l2=1e-4)
    reg_laplacian = tfl.lattice_layer.LaplacianRegularizer(
        lattice_sizes=my_lattice_sizes, # 必须传入！
        l2=1e-4)
    lattice_out = tfl.layers.Lattice(
        lattice_sizes=my_lattice_sizes, 
        monotonicities=lattice_monos, 
        output_min=0.0,
        kernel_regularizer=reg_torsion # 应用正则化
    )([calib_slew, calib_cap, calib_inv_size])

    # VT Effect
    vt_effect = tf.keras.layers.Dense(1, use_bias=False)(input_vt)
    
    combined = tf.keras.layers.Add()([lattice_out, vt_effect])

    # Final Output
    # 注意：因为目标是 Log(Delay)，这里的 Bias 初始化应该是 Log(Delay) 的均值
    output = tf.keras.layers.Dense(
        1, 
        use_bias=True,
        bias_initializer=tf.keras.initializers.Constant(np.mean(y_log))
    )(combined)

    model = tf.keras.Model(
        inputs=[input_slew, input_cap, input_inv_size, input_vt],
        outputs=output
    )

    # 训练配置
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.05))

    train_inputs = {
        'slew_log': X_slew_log,
        'cap_log': X_cap_log,
        'inv_size': inv_size_val,
        'vt': X_vt
    }

    # 训练 (拟合 Log 目标)
    model.fit(train_inputs, y_log, epochs=500, verbose=0, batch_size=128)

    # 预测并还原
    y_pred_log = model.predict(train_inputs, verbose=0).flatten()
    y_pred = np.expm1(y_pred_log) # 还原 Log 变换

    # 评估 (在原始物理空间评估)
    r2 = r2_score(raw_y, y_pred)
    rmse = np.sqrt(mean_squared_error(raw_y, y_pred))

    return model, r2, rmse

if __name__ == "__main__":
    # 请修改为你的实际路径
    csv_path = "/home/zhaoxueyan/code/benchmark/baseline/lut_data.csv"
    output_excel = "lut_analysis.xlsx"
    analyze_lut_data(csv_path, output_excel)