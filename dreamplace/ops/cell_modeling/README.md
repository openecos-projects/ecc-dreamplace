# Cell Modeling 统一建模方案

本方案旨在通过聚合相同功能类型（Main Type）但不同工艺变体（VT, Size）的 Standard Cell Arc 数据，构建统一的时序模型 $Value = f(VT, Size, Input\_Slew, Output\_Cap)$。

## 1. 目标

*   **归类**: 利用 `main_idx` (Main Cell Type ID) 和 `libarc_offset` (Arc 在 Cell 中的索引) 将所有物理 Cell 的 Arc 进行逻辑归类。
*   **统一建模**: 不再对每个具体的物理 Cell (如 `AND2_X1_LVT`) 单独查表，而是对逻辑 Arc (如 `AND2` 的 `A->Z`) 建立包含 VT 和 Size 维度的统一模型。
*   **物理感知回归**: 采用基于物理公式的线性回归模型 (Physics-Aware Regression)，替代传统的查表插值。
*   **支持梯度**: 模型需支持对 `Input Slew`, `Output Cap`, `Size`, `VT` 的微分，以支持时序驱动的布局优化。

## 2. 建模方法 (Physics-Aware Regression)

参考 `analyze_lut_data.py` 中的 Model B，我们采用以下公式拟合 Delay：

$$
Delay = w_0 + w_1 \cdot Slew + w_2 \cdot Cap + w_3 \cdot \frac{1}{Size} + w_4 \cdot \frac{Cap}{Size} + w_5 \cdot VT
$$

*   **训练**: 在初始化阶段，针对每个 `(MainType, ArcOffset)` 组合，收集所有对应的物理 Cell (不同 Size/VT) 的 LUT 数据点，使用 Ridge Regression 求解系数 $w_0, ..., w_5$。
*   **推理**: 运行时根据输入的 `MainType` 和 `ArcOffset` 查找对应的系数向量，并结合当前的 `Slew`, `Cap`, `Size`, `VT` 计算 Delay。

## 3. 数据接口 (PlaceDataCollection)

`PlaceDataCollection` 中已包含以下关键数据结构 (来自 C++ 导出)：

1.  **`flat_libarc_info` (List/Tensor)**
    *   **内容**: `[src, dst, libcell_idx, libarc_offset]`
    *   **用途**: 确定 Arc 所属的 Cell 和 Arc 索引。

2.  **`flat_libcell_info` (List/Tensor)**
    *   **内容**: `[libcell_name, libcell_main_id, libcell_size, libcell_vt]`
    *   **用途**: 获取 Cell 的物理属性 (Size, VT) 和逻辑分类 (Main Type)。

3.  **`arcs_info` (LUTs)**
    *   包含 `f_delay_luts`, `r_delay_luts`, `f_trans_luts`, `r_trans_luts` 等原始查表数据，用于训练回归模型。

## 4. 实现细节 (`ops/cell_modeling/cell_modeling.py`)

*   **`build_models`**: 
    *   加载数据并转换为 Tensor。
    *   构建特征矩阵 $X = [Slew, Cap, 1/Size, Cap/Size, VT, 1]$。
    *   对每个 `(MainType, ArcOffset)` 分组进行线性回归拟合。
    *   保存系数张量 `coeff_*`，形状为 `[NumMainTypes, MaxArcOffset, 6]`。

*   **`forward`**:
    *   **输入**: `libcell_main_id`, `arc_offset`, `vt`, `size`, `input_slew`, `out_cap`。
    *   **输出**: 计算得到的 Delay 或 Slew 值。
    *   支持 PyTorch 自动微分。

## 5. 预期收益

*   **可微性**: 模型对 Size 和 VT 完全可微，支持基于梯度的 Cell Sizing 和 VT Assignment。
*   **内存优化**: 相比存储庞大的 4D LUT，仅需存储少量回归系数 (每个 Arc 仅 6 个 float)。
*   **计算效率**: 推理过程仅涉及简单的向量点积，速度极快。

## 6. 独立误差分析与主流程复用

当前代码库额外提供了一个独立的误差分析入口：

```bash
/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python \
  baseline/run_cell_modeling_regression.py \
  --workspace /path/to/workspace_case/FFT \
  --top-k 200 \
  --dump-all-points
```

它的职责是：

*   **单独启动**：不跑整条 placement，只装载 `placedb`、`CellModeling` 和原始 LUT。
*   **工艺分目录存储**：结果落到 `baseline/cell_modeling_regression/<process_node>/<design>/<lib_hash>/`。
*   **误差分析**：输出 `cell_modeling_error_summary.json`、`cell_modeling_error_topk.csv`，并可选输出 `cell_modeling_error_points.csv.gz`。
*   **主流程复用**：独立回归不会产生另一套模型格式，而是复用 `CellModeling` 当前的 `surrogate_cache/<lib_hash>`。因此单独分析阶段构建出的模型，主流程在相同 `lib_hash` 下会直接命中并复用。注意磁盘缓存是 opt-in：需要给主流程和独立回归设置相同的 `surrogate_cache_root`（JSON 参数或 `--surrogate-cache-root`），否则每次启动都在内存中重新拟合（约数秒），不落盘。
*   **输入链一致**：独立回归的 DREAMPlace 输入应与主 `place-only` 流程一致，直接复用 `fixFanout -> place` 的 DEF/Verilog 解析链，而不是旁路构造另一份输入上下文。
*   **复用校验**：可通过 `--require-main-flow-cache-match` 强制要求 standalone regression 与当前主流程 `surrogate_cache_summary` 使用同一 `lib_hash/cache_dir`。

建议把它当作 `delay/slew surrogate` 的独立 debug 入口：

*   调整回归形式或特征后，先跑这个入口看 `delay` / `slew` 误差。
*   确认误差和 worst-case case 后，再回到 `size_only` / `joint` 跑整条 placement。
