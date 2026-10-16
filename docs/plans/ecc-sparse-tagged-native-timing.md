# ECC sparse tagged native timing 方案

## 目标

在当前 `fafb03e1` 基线之上，为复杂 SDC 引入按实际可达状态建立的 sparse tagged graph based timing。核心目标是同时满足以下条件：

1. 传播状态只覆盖实际可达的 `(pin, launch domain)`；exception profile 默认留在 endpoint legality 关系中，只有会改变传播 eligibility 或 source seed 的约束才提升为 state tag，不分配 `clock × profile × pin` 的 dense 张量。
2. AAT、slew、RAT 和梯度计算直接接入现有 CPU/CUDA native timing operators，Python 只负责状态编排、静态索引和目标聚合。
3. 普通单 domain 设计继续使用当前 scalar fast path，不因为增加复杂 SDC 支持而变慢。
4. PULP 这类存在跨时钟 max-delay、multicycle、false path 混合端点的设计，在共享组合逻辑处保留不同语义状态，并在 endpoint 使用对应的 required-time 规则。
5. 保持 graph based analysis。状态是约束语义的压缩标签，不是逐条路径展开，也不引入通用 PBA。

本方案只覆盖 timing state、native propagation、endpoint objective 和梯度接口。宏布置、全局布线、buffer 候选生成、SDC 文本前端重写和外部 OpenROAD flow 不属于本方案。

本方案的唯一主验证对象固定为 PULP `innovus_1013` case，golden 固定为 OpenSTA。所有数值对齐都必须使用同一份 placement DEF、Verilog、SDC、Liberty 和 RC/寄生输入；synthetic graph 只用于单元测试和结构验证，不能作为最终 golden。

## 修订后的设计决策和执行闸门

本版本先冻结以下决策，后续实现不得在 Phase 2 或 Phase 3 中重新改变 state key：

1. propagation state 使用 `(pin, launch_domain_id)`。endpoint exception 使用独立的 sparse check relation：

   ```text
   (launch_domain_id, capture_domain_id, check_class, transition)
       -> exception_profile_id / legal mask
   ```

2. 只有 exception 会改变内部 arc eligibility、source arrival seed 或 state transition 时，才允许把等价的 `propagation_tag_id` 加入 state key。这个例外必须在 schema 中显式标记，不能因为 endpoint profile 数量增加就复制所有 pin state。
3. 第一阶段只实现 native 已经能够完整解释的约束。无法表达的 selector、required time 或 edge relation 必须在 rawdb 阶段标记为 `unsupported`，禁止自动近似成 `normal_setup`。
4. Phase 0 必须先对真实 PULP 输入完成只读 preflight，证明 state/arc 数量和 profile 覆盖范围，再决定是否进入 native kernel 实现。
5. 每个阶段都有独立的 semantic gate、numeric gate 和 performance gate。任何 gate 未通过时只能停留在 diagnostic mode，不能接入 placement 或 diff sizing。

## 基线和当前可复用能力

实现基于 worktree：

```text
路径：/home/zhaoxueyan/code/ipd-timing-opt-ecc/ecc-dreamplace-sparse-lane
分支：codex/pulp-sparse-tagged
基线：fafb03e1
```

当前版本已经具备以下可复用基础：

- [timing_lanes_native.py](../dreamplace/ops/timing_propagation/timing_lanes_native.py) 已经将四个电气传播通道融合到一个 CPU/CUDA native forward/backward：rise AAT、fall AAT、rise slew、fall slew。
- [cell_aat_native.py](../dreamplace/ops/timing_propagation/cell_aat_native.py) 已经把 LUT 查询、cell arc update 和四通道聚合连接起来。
- [net_aat_native.py](../dreamplace/ops/timing_propagation/net_aat_native.py) 已经提供 net AAT/slew 的 native 路径。
- [timing_lanes_cpu.cpp](../dreamplace/ops/timing_propagation/src/timing_lanes_cpu.cpp) 和 CUDA 对应实现已经提供 smooth/hard aggregation 的反向传播。
- `d94a4d03` 已经完成 CPU/CUDA native timing propagation 的共享基础。
- 当前 `TimingPropagation` 的 scalar AAT/RAT、recovery check、critical endpoint pruning 和 smooth aggregation 仍然是可工作的基线。

当前 timing lane 的语义是“四个 rise/fall 电气通道”，并不包含 launch domain 或 SDC exception。新实现需要在它上面增加 sparse state 维度，不能把这两种 lane 混为一谈。

## 核心数据模型

### Domain

`launch_domain_id` 表示数据到达的来源类别。第一版只保留 native 能确定的时钟来源和未时钟输入：

```text
0 ... num_real_clocks - 1   真实 launch clock
num_real_clocks              unclocked input domain
```

同一 domain 内的多个起点可以在 graph based propagation 中合并。domain 不保存具体 startpoint，也不保存完整路径。

### Exception profile

`exception_profile_id` 是经过 native SDC 解析后得到的紧凑约束语义 ID，表示一个 endpoint check 应该采用哪种 required-time 规则。它默认不进入内部 propagation state；它通过 endpoint check relation 关联到可达的 launch/capture domain。第一版支持：

```text
normal_setup
clock_pair_max_delay
multicycle_setup
multicycle_hold
false_path
```

profile 至少记录：

```text
kind
applies_to_setup
applies_to_hold
max_delay_ps
ignore_clock_latency
setup_multiplier
hold_multiplier
transition_mask
```

profile 是语义分类，不是用户调优参数，也不是每条真实路径的唯一编号。等价的 SDC 约束共享一个 profile；不支持的 `-through` 或无法唯一解析的 selector 继续报告 unsupported。

profile 的 required-time 规则必须由 native clock-edge/check API 提供，不能在 DreamPlace Python 中重新推导时钟相位或路径例外优先级。schema 必须保存可复核的规则输入：

```text
launch_edge_class
capture_edge_class
clock_latency_policy
max_delay_ps
setup_capture_edge_offset
hold_capture_edge_offset
transition_mask
constraint_precedence
```

规则选择固定为：先过滤 false path，再按同一 launch/capture/check class 消费 native 导出的 chosen rule；如果 native 不能给出唯一结果，则输出 `ambiguous_exception` 并停在 diagnostic mode。`max-delay`、`multicycle` 和 normal setup 不允许由 profile ID 的数值顺序隐式决定优先级，也不能在 DreamPlace 侧用一个未经定义的“最严格”排序替代 native precedence。

### State CSR

每个实际可达的 pin/domain 组合建立一个 propagation state row：

```text
tag_pin_state_start[num_pins + 1]
tag_state_pin_id[num_states]
tag_state_domain_id[num_states]
# optional; present only when preflight proves propagation tags are required
tag_state_propagation_tag_id[num_states]
```

同一 pin 的 state 按 `domain_id` 排序并去重。`tag_pin_state_start` 是唯一的 pin 到 state 映射，不允许 Python 在 forward 中遍历字典或构造可变列表。若 Phase 0 证明某个传播例外确实需要额外标签，则追加稳定的 `tag_state_propagation_tag_id`，并在 metrics 中单独报告其扩散原因。

每个 state row 内仍然使用当前四个电气向量：

```text
state_rAAT[num_states]
state_fAAT[num_states]
state_rSlew[num_states]
state_fSlew[num_states]
```

endpoint relation 单独使用 CSR，避免把 endpoint-only profile 复制到所有内部 pin：

```text
tag_check_start[num_checks + 1]
tag_check_endpoint_pin_id[num_check_rows]
tag_check_launch_state_id[num_check_rows]
tag_check_capture_domain_id[num_check_rows]
tag_check_profile_id[num_check_rows]
tag_check_valid_transition_mask[num_check_rows]
```

`tag_check_launch_state_id` 必须指向 endpoint pin 上真实可达的 propagation state；构造阶段发现 dangling state、重复 check 或 profile 不可判定时必须拒绝 schema。

### Tagged cell/net arc CSR

基础 topology 仍使用当前 scalar arc ID，新增 tagged edge 只保存状态映射：

```text
tagged_inst_arc_level_start[num_levels + 1]
tagged_inst_arc_base_arc_id[num_tagged_cell_arcs]
tagged_inst_arc_src_state[num_tagged_cell_arcs]
tagged_inst_arc_dst_state[num_tagged_cell_arcs]

tagged_net_arc_level_start[num_levels + 1]
tagged_net_arc_base_sink_pin_id[num_tagged_net_arcs]
tagged_net_arc_src_state[num_tagged_net_arcs]
tagged_net_arc_dst_state[num_tagged_net_arcs]
```

只有 base arc 的 source pin 存在某个 domain state 时才建立 tagged edge。默认 cell/net arc 保持 launch domain identity；如果某个 native exception 明确要求传播状态转移，则使用显式的 propagation-tag transition 表，不允许通过 profile ID 的隐式复制实现。所有 tagged edge 必须能回溯到一个 base arc 和一条合法的 source/destination state relation。

如果确实存在需要传播的语义标签，schema 额外保存：

```text
tagged_state_transition_start[num_states + 1]
tagged_state_transition_dst[num_state_transitions]
tagged_state_transition_reason[num_state_transitions]
```

该表只描述编译阶段已经证明必要的 state transition，不把 endpoint-only exception 扩散到整个 graph。Phase 0 必须报告 transition 数量；数量非零时，必须有专门的 two-state transition fixture 和 CPU/CUDA parity 测试。

### Endpoint check metadata

每个 setup/recovery/output check row 需要关联：

```text
endpoint_pin_id
capture_domain_id
check_class
profile_id 或 profile candidate list
valid transition mask
chosen rule id
source constraint ids
```

endpoint objective 不直接使用全局 scalar RAT，而是对可达的 state/check pair：

1. 过滤 false path 和非法 launch/capture pair；
2. 按 profile 计算 required time；
3. 得到 rise/fall slack；
4. 对相同 endpoint/check class/transition 的候选，只保留 native chosen rule 对应的最差合法 slack；
5. 按 OpenSTA 对应的 check-row 口径去重后计算 WNS/TNS。

这样可以避免同一个 endpoint 因为多个 launch domain、重复 profile 或 rise/fall 记录重复计入 TNS。验收 artifact 必须同时保存“候选 state/check pair 数量”和“最终计入 WNS/TNS 的 check row 数量”。

## Native SDC 编译和导出

### 约束来源

约束 profile 必须来自 native iSTA 已解析的 clock/path exception state，不在 DreamPlace Python 侧重新解析 SDC 文本。native 已经拥有 clock group、max-delay、multicycle 和 false-path 的语义对象；导出层只负责把它们压缩成稳定的 schema。

### 支持范围

第一版支持：

- 普通同步 setup/recovery；
- clock pair 间的显式 max-delay，包括 `-ignore_clock_latency`；
- 起点或终点集合可静态归类的 multicycle setup/hold；
- source-only、endpoint-only 和 clock-pair false path；
- `set_clock_groups -asynchronous` 形成的 clock-pair 合法性矩阵；
- PULP 当前 SDC 中可以由 native selector 静态归类、且不含 `-through` 的约束形式。

PULP 的 input delay、output delay、reset false path、SRAM MAR false path 和 clock group 必须逐类统计。它们不能因为同属于“path exception”就共用一个 profile。每类约束在 preflight manifest 中标记为：

```text
supported_in_sparse
diagnostic_only
unsupported_fail_fast
```

第一版不把 `input_delay` 或 `output_delay` 当作普通内部 launch domain。它们必须通过 output/input check row 和 native edge relation 表示；如果 native 导出只提供了 scalar required time，则只能保留为 diagnostic-only。

以下情况必须显式记录 unsupported 并阻止 sparse mode 误用：

- 任意 `-through` 约束；
- 无法唯一确定 launch domain 的起点；
- 需要按具体 pin/instance 组合动态匹配的 profile；
- 同一 state 在传播过程中需要改变 exception identity 的约束；
- native 导出缺少 required-time 数值或 check row 的约束。

`unsupported` 不是静默降级。rawdb 必须同时保存原始 reason、selector 摘要、受影响的 endpoint 数量和建议的外部 OpenSTA 命令。placement/diff-sizing 默认在发现 `unsupported_fail_fast` 时停止；只有显式的 diagnostic-only 模式才允许继续运行 scalar timing，并且结果不得进入 QoR 结论。

### Required-time 语义契约

Phase 1 在实现前必须从 native iSTA/OpenSTA 导出一张可审计的规则表，至少覆盖：

| check class | required-time 来源 | 必须固定的语义 |
|---|---|---|
| normal setup/recovery | native capture edge/check arc | launch/capture edge、clock latency、uncertainty、transition |
| clock-pair max-delay | native max-delay check | budget、`-ignore_clock_latency`、launch/capture pair |
| multicycle setup | native selected capture edge | setup multiplier 对应的 capture edge |
| multicycle hold | native selected hold edge | hold multiplier 与 setup multiplier 的关系 |
| false path/clock group | native legality matrix | invalid pair，不生成可计分 slack |
| input/output delay | native I/O check row | external reference edge、delay、transition |

DreamPlace 只消费这张规则表，不在 Python 侧重新推导 multiplier、时钟相位或 exception precedence。若同一 check 同时命中多个约束，native 必须导出最终的 chosen rule 和被淘汰的 rule IDs；无法产生唯一 chosen rule 时返回 `ambiguous_exception`。

### Schema 版本

新增字段必须注册为新的 timing schema 版本。不能让 schema v2 的消费者静默忽略 tagged state。建议：

```text
schema v2：当前 scalar timing
schema v3：scalar timing + sparse tagged timing metadata
```

schema v3 仍然保留全部 v2 scalar 字段，便于诊断、scalar fast path 和逐步迁移。`timing_schema.py` 应先验证 CSR offset、state identity、endpoint relation、base arc 映射、profile ID 范围和 semantic digest，再允许构造 sparse `TimingPropagation`。

### 双仓库 ownership 和兼容策略

schema v3 的 producer 是 `ecc-tools` native iSTA export，consumer 是 `ecc-dreamplace`。两边的职责固定为：

| 仓库 | 负责内容 | 独立验收 |
|---|---|---|
| ecc-tools | exception/profile 编译、domain/check relation、required-time rule、unsupported diagnostics、schema producer | 用 PULP SDC 生成稳定 manifest、profile/state digest 和 v2 scalar 对照 |
| ecc-dreamplace | schema v3 校验、CSR materialization、CPU/CUDA propagation、RAT、objective/backward | 用 frozen schema fixture 做 scalar/tagged、CPU/CUDA、gradient parity |
| ecc | 只负责依赖版本、flow 参数和 gitlink 组合 | 在安装环境中验证 producer/consumer 版本匹配及端到端 rawdb/placement 入口 |

v3 消费者必须检查 `schema_version`、`producer_capabilities` 和 `semantic_digest`。v2 consumer 遇到 v3 必须明确拒绝；v3 consumer 遇到缺失 sparse fields 只能走 v2 scalar path。父 ECC 的 gitlink 只能在两个子仓库的目标 commit 已经可被 CI 获取后更新，不能依赖尚未合并的临时 PR ref。

## Native propagation 设计

### 四通道聚合复用

当前 `timing_lanes_forward/backward` 已经对四个电气向量做 fused aggregation。新实现复用它的 destination-indexed reduction，但不把它误认为完整的 sparse semantic propagation：

```text
destinations: state-level four vectors
sources: tagged edge update four vectors
index: destination state ID
```

也就是说，现有 operator 的四个 lane 继续表示：

```text
rise AAT / fall AAT / rise slew / fall slew
```

而 sparse semantic state 由 `state_id` 和 tagged CSR 表示。不要把 exception profile 追加成第五个电气 lane。

### Cell AAT

新增独立的 tagged cell AAT native entry point；只有在现有 ABI 可以保持清晰且不引入隐式分支时，才扩展现有 `cell_aat_forward`。接口必须接受：

```text
base_arc_id
source_state_id
destination_state_id
```

LUT 查询使用 source state 的 slew、destination load/cap 和 base arc 的 Liberty index。每条 tagged edge 输出四个 state-local update，再调用现有 `timing_lanes_forward` 聚合到 destination state。

必须避免以下低效实现：

- Python 对每个 domain/profile 逐次调用现有 cell operator；
- 为每个 state 构造完整的 dense pin 向量；
- 在每个 level 中重复转换 arc index dtype；
- forward 每次重新排序或构造 tagged CSR。

### Net AAT

新增 tagged net AAT native entry point，输入为：

```text
tagged_net_arc_base_sink_pin_id
tagged_net_arc_src_state
tagged_net_arc_dst_state
net delay / impulse / cap
```

同一 net 的 RC 数值可以复用，但 source/destination state 的 AAT 和 slew 必须独立。输出继续使用四通道 native scatter。

### RAT 和 profile rule

RAT 需要单独的 profile-aware native path，不能把 scalar RAT 复制到每一个 state 后再用 Python 修正。profile-aware RAT 至少需要：

- normal setup/recovery 的 capture clock 规则；
- max-delay 的固定 budget 和 `ignore_clock_latency`；
- multicycle setup/hold multiplier；
- false path 的 invalid mask；
- rise/fall 和 timing-check arc class。

RAT 的 state-level结果可以用 sparse endpoint rows 存储，不要求每个内部 pin 都保存完整 required time矩阵。

### Backward

native backward 必须直接从 endpoint legal reduction 回传到：

```text
state AAT
state slew
cell delay/slew LUT 输入
net delay/impulse/cap
placement position
```

不得先把 tagged state 投影为 scalar，再从 scalar slack 反推 profile 梯度。scalar projection 只允许用于诊断或旧版兼容输出。

## Runtime mode 和效率策略

### Scalar fast path

当 native export 判断没有可表示的多 state 约束时：

```text
不创建 tagged CSR
不创建 state tensors
继续走当前 scalar CPU/CUDA path
```

这是现有 BM64/CX55 等普通 case 的性能保护条件。

### Sparse mode

当 preflight 证明存在多 launch domain、传播 tag 或多个合法 endpoint rule 时：

```text
构造一次静态 tagged CSR
将 index/offset/base arc metadata 常驻 CPU/CUDA device
每次 forward 只更新 RC、slew、AAT 和 LUT 数值
```

静态结构不能在每个 placement iteration 重新编译。只有 topology 或 cell master 发生结构性变化时才允许刷新。

运行模式由 preflight capability 决定，而不是由“能否构造部分 CSR”决定：

| preflight 结果 | rawdb 行为 | placement/diff-sizing 行为 |
|---|---|---|
| 全部约束 `supported_in_sparse` | 构造 v3 sparse view | 允许 sparse timing |
| 存在 `diagnostic_only`、无 `unsupported_fail_fast` | 保留 v3 metadata | 只允许显式 diagnostic-only scalar 运行，禁止作为 QoR 结果 |
| 存在 `unsupported_fail_fast` 或 digest/schema 不一致 | 保存诊断并停止 | 不进入 timing optimization |

任何 provider（dynamic RC、critical pruning、smooth aggregation、recovery check）如果没有声明 sparse capability，都必须在 setup 阶段触发同一套 fail-fast，而不是在 forward 中隐式投影为 scalar。

### State count metrics

每次 sparse setup 必须记录：

```text
num_pins
num_states
states_per_pin_mean
states_per_pin_max
scalar_cell_arc_count
tagged_cell_arc_count
scalar_net_arc_count
tagged_net_arc_count
tagged_cell_arc_ratio
tagged_net_arc_ratio
num_profiles
num_launch_domains
num_propagation_transitions
num_endpoint_checks
num_supported_constraints
num_diagnostic_only_constraints
num_unsupported_constraints
state_key_digest
semantic_digest
```

这些指标用于直接判断 sparse 是否真的节省了内存和计算，不能只报告“sparse mode enabled”。

### Device policy

- CPU 和 CUDA 使用相同的 state CSR 和 profile metadata；
- index/offset 只在 setup 时转换一次；
- CUDA 不允许因为 sparse mode 自动退回 Python 循环；
- deterministic mode 下保留现有 deterministic native policy；
- CPU/CUDA parity 必须比较 state-level AAT、endpoint legal slack 和最终梯度，而不仅是 scalar WNS/TNS。

## 与现有 placement/diff-sizing 的接入

接入边界如下：

1. `macroPlaceDB.py` 接收并保存 schema-v3 sparse metadata。
2. `BasicPlace.py` 将静态 sparse arrays 放入 `data_collections`，只做一次 device transfer。
3. `PlaceObj.py` 构造 `TimingPropagation` 时绑定 sparse metadata 和 mode。
4. `TimingPropagation.forward` 根据 mode 分派 scalar 或 sparse schedule。
5. placement 的 timing objective 复用 sparse endpoint reduction。
6. diff sizing 复用相同的 sparse forward/backward，不另建一套 sizing-only timing model。
7. discrete sizing、candidate buffering 和 dynamic RC provider 必须声明是否支持 sparse state；不支持时提前 fail-fast，不能静默回退为错误的 scalar timing。

## 验证计划

### A. Schema 和 CSR

用小图覆盖：

- 一个 domain、一个 profile 的 scalar parity；
- 两个 domain 在共享 cell/net 后仍保持两个 state；
- 同 domain 多 source 合并为一个 destination state；
- state offset、pin ID、profile ID、base arc 映射错误时拒绝；
- 空 sparse view 正确退回 scalar mode；
- tagged CSR 重复 materialization 的 digest 稳定。

### B. SDC profile

分别构造：

```text
normal setup
source-only false path
clock-pair false path
clock-pair 3 ns max-delay
setup multicycle = 3
hold multicycle = 2
```

检查每类 profile 的合法性、required time 和 endpoint filtering。包含一个 endpoint 同时收到普通 setup 和 max-delay/multicycle 路径的共享逻辑图。
另加一个同一 launch/capture/check class 命中多个 exception 的 fixture，验证 chosen rule、被淘汰 rule IDs 和 `ambiguous_exception` 行为；input/output delay、reset/SRAM false path 和 clock-group 也必须有 supported/diagnostic/unsupported 分类测试。

### C. Native CPU/CUDA

对相同 schema 和相同 RC 输入，比较：

- scalar vs one-domain tagged AAT/slew；
- CPU vs CUDA state-level AAT/slew/RAT；
- hard vs smooth aggregation；
- forward/backward finite difference；
- invalid state 的 sentinel、slew 和 gradient 必须保持为无效值/零。

state-level AAT/slew/RAT 和 CPU/CUDA forward 数值采用与 D 节相同的 3% 相对容差并设置 2 ps 绝对下限；invalid mask、state ordering、gradient 非零位置和 endpoint legal mask 仍要求严格一致。

### D. PULP/OpenSTA golden 对齐

先对 PULP `innovus_1013` 的固定输入做只读 preflight，记录 domain、profile、constraint class、state、transition 和 tagged arc 数量。只有 preflight manifest 固定并确认 schema capability 后，才开始 native propagation。PULP placement output 作为唯一固定输入，OpenSTA 作为 golden。第一轮不改变 geometry 和 RC，也不执行 sizing 或 buffering。对同一份 DEF、Verilog、SDC、Liberty 和 RC/寄生输入，分别比较：

- normal setup；
- clock-pair max-delay；
- multicycle；
- false path；
- endpoint WNS/TNS；
- legal endpoint membership。

PULP case 的比较必须按路径语义拆开记录：

```text
normal setup
clock-pair max-delay
multicycle setup/hold
false-path exclusion
recovery/removal（若该 case 的 golden 报告包含）
```

最终才运行完整 PULP diff sizing，并再次用 OpenSTA golden 评估结果，避免把 placement、RC 和 sparse semantics 的差异混在一个结果里。

验收记录必须同时保存：

```text
PULP 输入 manifest 和哈希
OpenSTA 命令脚本及报告
ECC native export 的 profile/state digest
ECC sparse forward 的 endpoint/profile 报告
OpenSTA 与 ECC 的 WNS/TNS、endpoint membership、path-class 差异
```

验收门槛必须在第一次 baseline 运行后写入 manifest，不能只写“结果接近”。当前阶段采用 3% 的数值容差；合法性和规则身份仍然要求严格一致：

```text
endpoint legal membership：逐 check row 严格一致
chosen profile/rule：逐 check row 严格一致
required time：绝对误差 <= max(2 ps, 3% * max(1 ps, abs(OpenSTA required time)))
WNS：绝对误差 <= max(2 ps, 3% * max(1 ps, abs(OpenSTA WNS)))
TNS：绝对误差 <= max(2 ps, 3% * max(1 ps, abs(OpenSTA TNS)))
计入 TNS 的 check-row 数量：严格一致
```

3% 只适用于 required time、slack 和 aggregate timing 数值，不适用于 endpoint legality、chosen rule、state identity 或 check-row membership。若 Liberty/RC 的数值精度不足以支撑 3%，必须在第一次 baseline 中记录实际精度并冻结，不能在比较失败后临时放宽。

如果差异超过门槛，报告必须按 `constraint class -> endpoint -> transition -> launch/capture domain` 定位，不能用 aggregate WNS/TNS 掩盖 membership 差异。所有 diagnostic-only 或 unsupported constraint 都必须从“通过”结论中单独列出。

### E. 性能门槛

在 scalar case 上：

```text
runtime regression < 5%
peak memory regression < 5%
```

在 sparse PULP case 上至少报告：

```text
scalar estimated state count
actual sparse state count
tagged/scalar arc ratio
setup export time
forward time
backward time
peak memory
CPU/CUDA speed ratio
```

不能在没有这些数据时声称 sparse 优于 dense。

性能实验固定以下协议：同一 commit、同一 Nix/编译配置、同一 CPU 线程数和 CUDA device、同一输入 manifest；首次 import/build 和一次 kernel warm-up 不计入 steady-state。每个测量至少重复 5 次，报告中位数和最大偏差，分别测量 setup、单次 forward、单次 backward、完整 timing iteration 和峰值 RSS/GPU memory。3% 是 timing semantic/QoR 数值门槛；runtime 和 peak memory 继续使用独立的 5% scalar regression 门槛。两类门槛都满足，才允许合并 sparse 默认路径。

## 实施阶段

### Phase 0：基线和计数器

- 在当前 worktree 固定 scalar CPU/CUDA 测试结果。
- 为 `TimingPropagation` 增加只读 profile/metadata 统计接口。
- 对 PULP `innovus_1013` 的真实 DEF/Verilog/SDC/Liberty/RC 输入运行只读 native preflight，输出 constraint class、supported/diagnostic/unsupported 分类和 sparse state/arc 数量。
- 通过一个 synthetic schema 计算 sparse state、endpoint relation 和 tagged arc 数量，验证 CSR digest 稳定性。

验收：不改生产行为，现有 timing tests 和 native CPU/CUDA tests 全部通过；PULP preflight 生成 manifest、state key digest、semantic digest，并明确给出是否进入 sparse mode 的 go/no-go。若实际 state 数量接近 dense 估计值，先回到 state key 设计，不进入 Phase 2。

### Phase 1：schema-v3 和 native static compiler

- 在 ecc-tools 的 TimingSnapshot/PyPlaceDB 增加 sparse fields。
- 从 native iSTA exception state 编译 domain、endpoint check relation、required-time rule 和 permission matrix。
- 加入 schema、CSR、semantic digest 和 unsupported diagnostics。

验收：PULP SDC 可以在 rawdb 阶段输出明确的 supported/diagnostic/unsupported 统计、chosen rule 和 selector 摘要，不再只报 `endpoints_max_reason`；ecc-dreamplace 能拒绝 schema/digest/CSR 不一致，并对 v2 输入保持 scalar path。

### Phase 2：sparse AAT/slew native path

- 引入 tagged cell/net arc state IDs。
- 复用现有四通道 `timing_lanes` native aggregation。
- 增加 CPU/CUDA tagged forward/backward。
- scalar projection 仅作为诊断。

验收：one-domain parity、two-domain reconvergence、必要的 propagation-tag transition fixture、CPU/CUDA 和 gradient tests 通过；所有结果先对 state-level AAT/slew 做数值门槛检查，再进入 endpoint objective。

### Phase 3：profile-aware RAT and endpoint objective

- 加入 normal/max-delay/multicycle/false-path profile rules。
- 实现 endpoint legal pair reduction。
- 保证一个 endpoint 只计一次 WNS/TNS。

验收：synthetic mixed endpoint 与 OpenSTA 的路径类别、chosen rule、required time、legal endpoint membership 和 check-row aggregation 一致；达到 D 节数值门槛后，才允许处理真实 PULP placement output。

### Phase 4：ECC placement/diff-sizing integration

- 接入 scalar/sparse 自动分派。
- 让 diff sizing 共享 sparse timing state。
- 适配 dynamic RC、critical endpoint pruning、smooth objective 和 recovery checks。
- 对不支持 sparse 的动态 provider 直接 fail-fast，并给出具体 capability。

验收：PULP `innovus_1013` fixed placement 与 OpenSTA golden 的 hard STA 对齐后，再进行 diff sizing 端到端验证；任何不支持的 provider 或 constraint 都在 setup 阶段可见，并且不会生成可宣称为 QoR 的 sparse 结果。

### Phase 5：性能和长期维护

- 根据 state/arc 计数器优化 CSR layout、device transfer 和 native kernel。
- 评估 tagged state 的缓存和 CUDA occupancy。
- 将 sparse schema、profile digest 和 state metrics 写入运行 artifact。

验收：scalar case 无明显回归，PULP sparse case 的性能和内存有可复现报告。

## 主要风险和处理方式

| 风险 | 处理方式 |
|---|---|
| profile 数量因复杂 SDC 爆炸 | profile 默认只存在 endpoint relation；native 编译阶段按等价 required-time/legality 语义合并，并设置 check/profile 数量上限 |
| exception profile 被错误地扩散到内部 state | 固定 `(pin, launch_domain)` state key；只有 preflight 证明需要传播的 tag 才进入 transition CSR |
| max-delay/multicycle/false-path 同时命中时语义不确定 | 由 native 导出 chosen rule、淘汰 rule IDs 和 precedence；无法唯一选择时 `ambiguous_exception` fail-fast |
| `-through` 或 pin-specific exception 无法压缩 | 明确 unsupported，保留原始 SDC 外部评估，不静默近似 |
| tagged cell LUT 查询变成 Python 循环 | 以 base arc + src/dst state CSR 批量进入 native operator |
| sparse state 在 forward 中反复 materialize | 静态 topology epoch 下缓存 CSR 和 device tensor |
| dynamic RC provider 仍是 scalar | 增加 sparse capability contract，未支持时 fail-fast |
| smooth aggregation 对无效 state 泄漏梯度 | invalid state 使用统一 sentinel，并在 native forward/backward 中显式屏蔽 |
| scalar projection 掩盖 profile 错误 | sparse objective 不依赖 scalar projection；projection 只用于诊断 |
| CPU/CUDA state ordering 不一致 | schema digest、排序规则和 state ID golden fixture 固化 |
| ecc-tools/ecc-dreamplace 版本错配 | producer capability、schema version、semantic digest 三重校验；父 ECC 只引用 CI 可获取的子仓库 commit |
| scalar 性能门槛无法复现 | 固定 build/device/input，重复 5 次并报告中位数、偏差、setup/forward/backward/iteration 分项 |
| endpoint WNS/TNS 与 OpenSTA 统计口径不同 | 固化 check-row aggregation contract，保存候选数量、最终计数数量和逐 row membership |

## 完成定义

方案完成需要同时满足：

1. 普通 scalar case 继续走原有 CPU/CUDA path，性能和结果没有回归。
2. PULP 的普通 setup、clock-pair max-delay、multicycle 和 false path 都有明确 profile 统计。
3. PULP 的 input/output delay、reset/SRAM false path 和 clock-group 约束都有明确的 supported/diagnostic/unsupported 状态。
4. 混合 endpoint 的共享逻辑保留多个 launch-domain sparse state，没有提前 scalar 合并；只有经 preflight 证明必要的 propagation tag 才进入 state。
5. AAT、slew、RAT 和 sizing gradient 都直接经过 native-compatible sparse operators。
6. CPU/CUDA、hard/smooth、forward/backward 和 OpenSTA endpoint legality 均有测试证据，并达到 D 节的数值容差。
7. scalar case 满足 runtime/peak-memory 门槛，PULP case 保存完整 state/arc/performance manifest。
8. 完整 PULP `innovus_1013` diff sizing 运行不再因为粗粒度 `reason 6` guard 提前退出，且所有仍不支持的约束被显式记录。
9. 最终 QoR 和语义结论只以 OpenSTA golden 对比为准；synthetic fixture 不得替代 PULP/OpenSTA 结果。

## 执行进度（2026-10-15）

当前已完成 schema v3、native sparse schedule 和真实 PULP 固定 DEF 的
OpenSTA **逐端点数值 gate**。这不等于整个计划完成：生产 dispatch、动态
provider、sizing master/VT 梯度和端到端 diff sizing 尚未接入；逐 check row 的
chosen-rule/exception identity 验收也尚未全部完成。

### 已落地的修复

- 保留 ideal clock 的物理 predecessor provenance，沿完整 pin/transition 链回溯
  source clock edge。launch/capture waveform 使用 source edge，LUT 和 CPPR
  继续使用物理 CK transition；launch tag 保留 source polarity。
- 导入 Liberty `when`，用 instance case/tie constants 在 setup 编译 variant mask，
  delay/slew 查询和 native export 共用该 mask。
- 跨库 net 接收端使用 Liberty slew span/derate 换算，并保留 input/output threshold
  差异对应的 arrival 修正。native DCTask 和 sparse export 共用计算 helper；
  sparse forward 对换算保留梯度。
- 修复同名 port/net 被 `set_load [get_ports ...]` 同时命中而重复施加载荷的问题，
  并计入 top-level inout driver 自身的 external port load。
- `set_driving_cell` 默认输入按 Liberty 声明顺序选择；matching arc 的外部 drive
  annotation 按 OpenSTA `findInputDriverDelay` 的覆盖顺序处理。
- `set_max_delay -ignore_clock_latency` 保留 setup/external margin，排除 clock
  uncertainty 和 CPPR，修复 PULP CDC required time 额外扣减 100 ps 的问题。
- 区分 function-insensitive arrival arc 与 electrical arc。额外 slew influence
  使用独立稀疏关系，和普通 cell edge 在同一次 native query 中计算；它不会创建
  arrival candidate。PULP 只增加 142 条 electrical relation。
- 保留先前 PAD I/O role、clock-to-Q sensitivity、recovery LUT 轴顺序和 required
  符号、一维 constrained-data LUT 轴的修复。

### 固定 PULP/OpenSTA 验证

输入为 `innovus_1013` 的原始 placement DEF、原始 SDC 和同一组 Liberty/LEF；
本轮没有 SPEF，双方均为 zero-wire/frozen-load STA，不是 placement RC 或 GR
评测。完整输入哈希保存在 `/tmp/ecc-pulp-phase0-evidence/run-final-23/manifest.json`。

OpenSTA golden 使用 `golden/golden_detailed.tcl` 和 `golden/golden_detailed.tsv`。
跨 path group 的重复 endpoint 必须取最小 slack，不能用文件最后一行覆盖。
`tests/regression/sparse_timing/compare_opensta.py` 固定采用
`max(2 ps, 0.03 * abs(golden))`，没有额外的 1 ns tolerance floor。

| 指标 | Sparse CPU | OpenSTA |
| --- | ---: | ---: |
| Unique legal endpoints | 41687 | 41687 |
| Violating endpoints | 9670 | 9670 |
| WNS (ns) | -252.069738487 | -252.069794000 |
| TNS (ns) | -15726.894882654 | -15726.897089000 |

逐 endpoint slack 超差数为 0，missing/extra endpoint 均为 0；selected endpoint
candidate 的 required-time 超差数为 0。最大 slack 绝对误差为 26.102 ps，仍在
对应 endpoint 的 3% 容差内。完整对照为
`/tmp/ecc-pulp-phase0-evidence/compare-final-23/{summary.json,endpoints.tsv}`。

同一 sparse export 的 CPU/CUDA state AAT、state slew、required 和 endpoint slack
最大差均为 0。CUDA 独立重建记录为 `build-cuda-timing.log`，测试为
`cuda-sparse-tests-4.log`（41 passed，覆盖 hard/smooth、finite difference、直接
CPU/CUDA state/gradient parity）；ecc-tools focused native CTest 为 9/9 passed。
所有运行环境均为独立诊断环境，没有替换共享 ECC 安装。

PULP sparse state 计数为 886257 / 883334 pins，mean 1.003309、max 3；
4 个 domain、6 个 propagation tag。Native rule candidates 为 normal 64193、
max-delay 1726、multicycle 216、false-path 223；这些是 native 分类统计，不应
表述成已经逐 row 与 OpenSTA rule ID 完成对齐。

`fixed_sta.py` 中的 `native_reference` 是旧 native path-state cache，保留作诊断；
它不是 OpenSTA golden，不能据此宣称生产 native path map 已经全部对齐。

### 尚未完成的验收

1. 逐 check row 的 legality/chosen rule/exception precedence 及分类 golden fixture。
2. schema v3 接入正式 PlaceIO、macroPlaceDB、BasicPlace、PlaceObj 和
   TimingPropagation dispatch。当前生产入口仍拒绝 v3/reason 6。
3. Dynamic RC、critical pruning、candidate buffering 和 sizing refresh 的 sparse
   capability/epoch contract；unsupported provider 必须 setup fail-fast。
4. Placement/diff-sizing 的 master/VT 梯度、完整 PULP diff sizing，以及优化后
   OpenSTA 评测。小图 RC 梯度通过不等于 sizing 已经交付。
5. Scalar runtime/memory 5% regression gate，以及 PULP 生产 master/VT 优化迭代
   性能测量。下文的完整 RC adjoint 测量只覆盖 frozen timing graph。

因此当前状态为“固定 DEF 的 sparse STA 数值对齐通过，生产优化流程待接入”。

最终 native build 的复核为 `run-final-23` / `fixed-sta-final-23`，同样逐端点
0 failures、selected required 0 failures；输入哈希与 `run-final-22` 完全一致，
semantic digest 均为
`df56e7eac407ac5c82e21781c317ae1a82fd3829449be520ef606e7e838df14f`。
最新 native module SHA256 为
`57cdcecedb6ade426440c996f2aeb6bb28c2ac4af000eb25382632ff98e5154b`。

合并 cell/electrical query batch 后，同一 export 的 AAT/slew/required/slack 与
合并前逐值相同；CPU controlled repeat 的 RSS 从 4990624 KiB 降为
4051808 KiB，forward 中位数从 8.343 s 降为 7.466 s。最终 `fixed-sta-final-23`
另一次 5-sample CPU 中位数为 6.575 s；`fixed-sta-final-22-cuda-batched` 的
CUDA 中位数为 0.052669 s，GPU peak allocated 为 1942361600 bytes。
CPU profile 的主要热点为 hard scatter 和 timing-lane aggregation。
这些测量尚未满足同 build、同环境的 scalar 5% regression 验收协议，因此不能
据此宣称 CPU 性能 gate 通过。

### 稀疏执行范围压缩与 RC adjoint 验证（2026-10-15）

已复用现有 CPU/CUDA cell AAT、net AAT 和四通道聚合算子，在 setup 为每层
建立紧凑的 state、physical pin 和 electrical slew-group 映射。Native 输入和
聚合只覆盖本层涉及的 state，结果按映射写回全局数组；rise/fall 索引和写回
使用同一个二维操作。仅 forward 的 slew max 使用已有无需反向计数的路径。
这次没有修改 native ABI 或重建二进制，正式 scalar 执行路径也没有改动。

当前检查到的 C++ level-offset/剪枝结构属于 scalar topology helper，完整的
sparse forward level 循环仍在 Python；这里复用的是现有 native 算子，没有
把这些 helper 表述为完整的 C++ sparse DAG executor。

最终证据目录为 `/tmp/ecc-pulp-phase0-evidence/compact-final-validation-56fapcsa/`，
包含源码快照、运行环境和命令 manifest、测试日志、timing/gradient arrays、
OpenSTA 对照和逐数组 parity 报告。统一使用 PyTorch 2.11.0+cu128、float64、
8 CPU 线程和 GPU 0；预热一次，每组计时五次，CUDA 计时前后同步。

| 固定 PULP 测量 | 优化前中位数 | 优化后中位数 |
| --- | ---: | ---: |
| CPU，仅 forward | 3.456574 s | 0.572471 s |
| CUDA，仅 forward | 50.938588 ms | 39.276702 ms |
| CPU，RC forward + backward | 4.423088 s | 1.346895 s |
| CUDA，RC forward + backward | 190.024357 ms | 134.366925 ms |

共享机器负载有波动：前一轮同输入 CPU forward 为 5.726377 s，表中采用
最终旧实现复测的 3.456574 s。CPU RC adjoint 峰值 RSS 从 21165976 KiB
降到 3742304 KiB；CUDA RC adjoint peak allocated 从 18993360896 bytes
降到 1613069824 bytes。仅 forward 和保留梯度图的 forward 属于不同测量。

优化前后及 CPU/CUDA 的全量 AAT、slew、required、check slack 和 endpoint
slack 逐值相同；CPU 六组 physical RC 梯度逐值相同，CUDA/CPU 梯度的最大
绝对差为 1.863e-9，均通过 atol=1e-8、rtol=1e-7。六组输入为 rise/fall
capacitance、net delay 和 impulse；它们不等于实际 cell master/VT 梯度。
新增 inference/有梯度 forward 一致性和 invalid-transition/shared-slew 边界
覆盖后，相关 CPU/CUDA/schema/comparator 测试为 49 passed。最终 OpenSTA
41687 个 endpoint 的 slack 和 selected required 超差数均为 0。

生产 dispatch、动态 RC/provider epoch、master/VT 梯度、完整 diff sizing
以及 scalar 5% regression gate 仍按上方未完成项验收。

### 生产优化功能接入前核对（2026-10-15，历史状态）

以下核对记录的是生产接入前的状态；后续进展以文末的“统一生产入口接入”记录为准。
当时的结论为：sparse timing 仍是独立诊断/回归入口，尚未接入正式优化调用链。
`SparseTimingSchedule` 的生产目录引用只有其自身定义；实际构造和调用位于
`tests/regression/sparse_timing/fixed_sta.py` 以及算子测试。以下状态不能由
数值对齐、RC adjoint 通过或性能测量替代。

| 边界/功能 | 当前状态 | 所需适配 |
| --- | --- | --- |
| PlaceIO 初始读取和 committed refresh | 校验仅接受 v2，v3 在入口拒绝 | schema/capability 分派；不能仅绕过 reason 6 |
| macroPlaceDB / BasicPlace | 未保存或 materialize sparse view | 原子导入 view、device schedule 和 topology/master epoch |
| PlaceObj / TimingPropagation | 仍构造并执行 scalar timing | sparse dispatch、目标输出和物理电气/报告状态的显式边界 |
| Hard/smooth CPU/CUDA、endpoint objective | 独立 sparse 验证通过 | 接入正式 timing 配置、指标和 artifact 调用链 |
| Direct-loss timing-driven GP | 验证了六组 RC 输入梯度，未运行 GP | 接入实时 RC/position graph 和 slew/cap 目标 |
| Continuous / discrete size、VT、S10/S50 | 未接入 sparse master/VT 梯度 | 复用 piecewise/native cell model、动态 check LUT 和投影后 master 刷新 |
| GR/RRR3 diff sizing | 原有 flow/config 存在，未调用 sparse | 先在 frozen GR RC 下打通 size/VT 更新、refresh 和最终外部 STA |
| Segment / candidate buffering | provider 仍消费 scalar pin rise/fall 向量 | 稀疏 driver/sink state 映射、共享 electrical view、buffer/geometry epoch |
| Pin2Pin、critical pruning/path extraction | 仍依赖 scalar pin/path 数据 | 保留 state/domain/rule 的路径与 pruning 适配 |
| 最终实体化、合法化后报告 | 未有 sparse 端到端验收 | commit 后重建 view/schedule，刷新 STA 并外部复核 |

Producer 的 `needsSparse` 在多时钟、setup exception 或禁止路径的 clock group
存在时导出 v3；不需要 sparse 的 case 保留 v2。旧 scalar 功能的代码仍在，
但 v3 不会自动继承它们。对当前 worktree 的入口校验直接输入 version=3，
实际报错仍为 `ECC timing requires schema 2, got 3; rebuild/reinstall ecc-tools`。

Sparse cell forward 目前使用固定 master 的 `_CellAat`，其 backward 只返回
AAT/slew/cap 的导数，不对 LUT tables 或 size/VT 变量求导。`_levels` 还复制了
base arc 的 Liberty 行，所以 master 投影后不能继续复用旧 schedule；需要明确
live binding 或 master epoch 刷新。已有六组 RC 梯度验证不代表 sizing 梯度。

后续执行顺序为：正式 schema/view/dispatch 与 read-only STA；fixed-GR diff
sizing 的 size/VT 和 master refresh；实时 placement RC/position/DRV 目标；
buffer provider、Pin2Pin/pruning 和 commit 后完整重建。每一步保留 v2 scalar
分派，对尚无 sparse capability 的组合在 setup 显式拒绝。


### 统一生产入口接入（2026-10-15）

代码位于同一隔离 worktree，尚未提交、发布或替换共享 ECC 安装。下列记录
区分已接线和已验证的能力，不能用固定输入 STA 或单元测试替代优化 QoR。

架构固定为一个 `TimingPropagation` 入口。Schema v2 保留 scalar 执行，v3
通过 PlaceIO → macroPlaceDB → BasicPlace → PlaceObj 传入验证后的 sparse view，
在原 forward 的动态 RC overlay/device/surrogate 边界后执行 sparse state schedule。
未新增 sizing optimizer、buffer optimizer、独立 cell model 或另一套候选决策。

- 原 `calculate_cell_aat_level` 的参考 cell 查询移入 `cell_aat_queries.py`，
  scalar 和 sparse 共用；固定 master 仍使用原 `_CellAat` CPU/CUDA native 算子。
  Sparse 的物理 pin 映射将 size/VT 查询交给原 piecewise/native cell model。
  Master 更新仅重绑 Liberty 行；view、device 或 topology epoch 变化重建 schedule。
- Candidate 和 segment provider 复用原有 cap overlay、树求解、live geometry、
  虚拟 buffer DRV、动态 delay snapshot 回收。非反相 buffer 在零 driver arrival
  下每层求解一次，再加回各 launch state 的 arrival，不按 domain 重算树。
  未声明 sparse capability 的 provider 会明确拒绝。
- Pin2Pin 路径提取和剪枝使用原 C++ extractor/pruner。仅对 state/base arc 身份
  做映射；pruner 新增导出其已经遍历过的 pin 集合。物理 endpoint incidence
  按合法 state cone 取并集，防止同一 endpoint 因多个 launch state 重复计数。
- Arrival 剪枝保留完整电气 slew，选中端点的值和梯度不依赖被剪掉的 arrival。
  未选端点保持上次完整刷新值；selector/full-refresh 周期及路径快照强制全图刷新。
  因电气 slew 仍需全图计算，不能据此宣称完整 cell 查询或 buffer 树开销被剪掉。
- 原完整 RAT 模式复用原 cell RAT operator，结果按匹配的 state AAT/RAT 投影。
  Fast loop 继续沿用跳过内部完整 RAT 的策略。Endpoint 报告采用合法 check 的
  qualification，路径报告使用提取路径自己的 state arrival 和有效 delay。
- Profile/device contract、setup/recovery scope、CPU/CUDA parity API 复用原入口。
  Setup/recovery 分类引用原 `recovery_checks.py` 的常量，避免重复定义编号。
  原有 surrogate parity 的限制不变：真实 model 的跨设备检查单独进行，不能
  通过 deepcopy live optimizer/model 状态替代。
- ECC committed refresh 先校验新 PyDB/view，再替换 backend 引用；原调用链
  重建 timing owner 时导入新 view。这里只验证了交接与失效逻辑，尚未完成
  真实 PULP buffer 实体化后整条数据库刷新/合法化的端到端验收。

验证证据位于 `/tmp/ecc-pulp-phase0-evidence/`：

| 验证 | 结果 | 证据 |
| --- | --- | --- |
| 生产入口 CPU/CUDA sparse、路径和原路径报告 | 80 passed | `production-final-tests-cuda.log` |
| CPU timing/schema/cell-query/provider/pruning 回归 | 766 passed，569 skipped，2 subtests passed；跳过项主要为 CUDA 参数化测试 | `production-final-tests-cpu.log` |
| 原 piecewise/native model 连续三次 size/VT 更新 | 梯度有限且非零、目标下降、native forward/backward 为 cpp | `test_sparse_optimization.py` |
| 四 drive type 离散 candidate 梯度 | 复用原 native_explicit_autograd provider，四个 probe 均有梯度 | 同上 |
| Segment 零状态校准和 live geometry | 零状态等于静态值，位置/count/strength 梯度有限且非零 | 同上 |
| Master 投影/view 替换 | Liberty 行更新、必要时重建 schedule | 同上及 `test_sparse_paths.py` |
| Full PULP 生产 CPU / CUDA / 完整 RAT | WNS −252.069738486637 ns、TNS −15726.894882654 ns，41687 endpoints、9670 violating | `production-final-{cpu,cuda,paths}/` |
| Full PULP OpenSTA endpoint + selected required | 三组均 0 failures，3% / 2 ps 门槛通过 | `production-final-opensta-{cpu,cuda,paths}/` |
| Full PULP 原 native extractor 路径 | 64/64 valid，0 invalid，最大残差 0 ps | `production-final-paths/paths.json` |

本轮最后一次固定输入测量 CPU forward 中位数约 0.417 s、CUDA 40.799 ms，
CPU 完整 RAT forward 约 0.658 s；测试期间有并发负载。这些数值只描述
frozen-load、zero-wire PULP STA，不是 placement 或 S50 的每步耗时，也不证明
scalar 5% regression gate 已通过。它们与历史 compact snapshot 的耗时不能
直接作为受控性能对比。

仍需独立验收：完整 PULP placement、S10/S50 和 buffering 的真实动作/实体化，
优化后 DEF 的同输入外部 STA，以及真实动态 RC/GR/RRR/provider 的端到端性能。
本轮没有重新构建并验证 CUDA buffer provider 的原生二进制；buffer 接口与
native candidate probe 的新增验证在 CPU，不能外推为完整 GPU buffering 已合格。
逐 check rule/exception precedence 的分类 golden 及 scalar 5% performance gate
仍为未完成项。

## 2026-10-15：旧设计真实优化流程验收

本轮在隔离 CPU runtime 中，从相同的 preplace_sizer 输入分别运行普通 scalar 与强制 sparse 两条路径。完整执行 placement（direct_loss、S10B0）→ placement-parasitic candidate B10（0.1%）→ frozen GGR/RRR3 离散 S50（无 continuous warm-up）→ 零 padding 合法化 → 外部 placement RC STA / OpenROAD GR50 STA。当前对照表 6/6 完成；发现 capture-clock 问题后另补跑 eth_top、stage_d，历史结果保留。

| Case（sparse） | GP iterations | GP overflow | placement GR50 WNS / TNS (ns) | S50 GR50 WNS / TNS (ns) | TNS 幅值改善 |
| --- | --- | --- | --- | --- | --- |
| BM64 | 556 | 0.091304 | -1.812 / -600.148 | -1.246 / -426.834 | 28.88% |
| eth_top | 443 | 0.092859 | -2.127 / -684.142 | -0.988 / -103.757 | 84.83% |
| stage_d_ysyx_24080018 | 385 | 0.086651 | -1.612 / -302.610 | -1.104 / -98.086 | 67.59% |

本轮修复两处 native 问题：

- buffer 拆分输出端口后，旧上游 net 占用端口名。native buffer commit 成功后先复用 iDB canonicalizeNetNames，再构建 timing；journal 使用 net 指针，返回规范化后的实际名字。
- sparse check 导出不能用“设计第一个 clock”的回退替代真实 capture clock。复用 scalar qualification 的 capture-clock state 判断，按 capture clock 与 capture edge 验证。eth_top 的旧 sparse 结果多计 865 个端点，其中 532 个负 slack，额外 TNS 约 −69.207 ns。固定 DEF 修复后 hard TNS −235.116 ns，外部 placement TNS −236.058 ns。

此外，隔离 launcher 在载入 ECC 后清除 LD_PRELOAD，解决系统 tee 继承 Nix Tcl 后的 GLIBC 冲突；原 GR BrokenPipe 出现在日志刷新收尾，路由本身已返回。

三个 sparse placement 的内部 WNS/TNS 与外部 placement RC STA 聚合差距均在 3% 内。内部 GGR/RRR3 与 OpenROAD GR50 使用不同路由及 RC，不能用其差值代替固定 RC 的 STA 资格判断。

最新 native 构建的 port buffer mutation / STA refresh / DEF-Verilog round-trip 回归通过；两个已有 Verilog net-name 测试通过；包含无 capture-clock 场景的 sparse required-rule 测试通过。重新导出 PULP 并调用 production CPU sparse STA，OpenSTA golden 的 41,687 个端点全部匹配，slack 与所选 required-time 通过 3% / 2 ps，missing/extra/slack_failures/required_failures 均为 0。

覆盖边界：本轮验证 CPU direct_loss + S10B0 + candidate B10 + GR/RRR3 S50；未运行 CTS/detail-route，也不把本轮结果表述为 CUDA 优化流程验收。BM64 的 capture-clock 修复前后固定 DEF 逐端点身份和 slack 完全一致，因此复用已完成的端到端结果；其余两例完整重跑。

完整结果与最终 DEF/Verilog 路径、hash、backend 记录：`/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc-sparse-e2e-old3-20261015-cw7av75i/e2e-results-current.md`、`e2e-results-current.json`、`e2e-results-current.tsv`。历史试验：`e2e-results-all-runs.json`。

## 2026-10-15：scalar recovery 与 sparse 的固定输入对齐

修复旧 scalar recovery 的四项错误：check LUT 按统一的 related-clock /
constrained-data 顺序查询；max required time 使用 capture anchor 减去 check；
live check 替换参考 check，不再把参考 RAT 一起取最小值；按每条 check 的
rise/fall qualification 选择参与归约的候选。保留现有 native LUT、scatter
和 autograd 算子，本轮没有修改或重建 native 二进制。

同一份 eth_top 诊断 DEF 的 hard TNS：scalar 修复前 −222.038609 ns，
修复后 −235.120500 ns，sparse −235.116484 ns，外部 placement RC OpenSTA
−236.058319 ns。724 个共同 RN/SN 端点的 scalar/sparse slack 最大差异为
0.000244141 ps；普通 D 端点仍有独立的 arrival 差异，不能据聚合 TNS
宣称所有端点均已与 OpenSTA 对齐。

三个已完成 placement DEF 的受控 hard STA / 位置梯度对照：

| Case | scalar TNS (ns) | sparse TNS (ns) | TNS 相对差距 | 位置梯度 cosine |
| --- | --- | --- | --- | --- |
| BM64 | -443.814000 | -443.833969 | 0.0045% | 0.999999999 |
| eth_top | -407.935063 | -407.779094 | 0.0382% | 0.999987844 |
| stage_d_ysyx_24080018 | -96.533469 | -96.491195 | 0.0438% | 0.999919796 |

CPU timing/schema 回归 674 passed、564 skipped、4 subtests passed；另在
GPU 1 验证 recovery hard/smooth 值和反向梯度。eth_top 的选定端点额外完成
OpenSTA 路径查询。修复后的真实 placement/B10/S50 收敛尚未重跑，因此上节
历史优化 QoR 不应视为本轮 scalar 修复后的新结果。

隔离 runtime、逐 case 日志、梯度、源码 hash 和验证摘要保存在
`/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc-recovery-alignment-20261015-ojbrsb_u/validation.json`、`validation.md`。

## 2026-10-15：整理提交与配套版本

本轮 native 提交按依赖关系组织：`439c247fa908f5b7259292399cdc118d59acc164`
修复 SDC、电气与 required-time 语义，`6b4b98973b80d6a388b78e3e84562e151deedd2c`
增加 sparse schema/export，`e737d48cb78cb69363da9bc6d8d8d46ba30e75ab`
修复 buffer 输出端口的 net 命名并纳入回归 helper。DreamPlace 的 scalar
recovery 修复为 `481d9bcc7e9bcfb3b31c015d50b4ebaaaf995894`；它依赖前述
native LUT 轴与 required-time 语义，发布及父仓库引用时必须配套。

整理时补齐被 test 白名单忽略的 `test/data/buffer_port_mutation.py`，清理
新增代码的 lambda lint 和公共查询抽取后的无用变量。重新通过 CPU
timing/schema/golden comparator 回归 677 passed、564 skipped、4 subtests
passed；CUDA sparse/paths/optimization/schema 回归 84 passed、4 subtests
passed。Nix 重新构建 `ecc_py` 与 9 个 native 测试目标，CTest 9/9 通过；
用本次重新构建的 native 完成 buffer 拒绝/回滚、STA refresh 与 DEF/Verilog
往返回归（1 passed、8 deselected）。本节没有增加新的端到端 QoR 结论。

提交清单、源码 diff、Nix 构建、测试与 native import 证据位于
`/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc-sparse-commit-prep-20261015-x7q48qqv/`。
