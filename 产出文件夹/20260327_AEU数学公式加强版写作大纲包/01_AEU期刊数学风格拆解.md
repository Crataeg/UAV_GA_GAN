# AEU 期刊数学风格拆解

## 1. 本次判断所依赖的本地知识库材料

### AEU 专门目录

- `参考文献/论文文献/AEU/20260324_近年无人机相关文献/README.md`
- `参考文献/论文文献/AEU/20260324_近年无人机相关文献/aeu_uav_recent.bib`
- `参考文献/论文文献/AEU/20260324_近年无人机相关文献/official_metadata/*.xml`

### 本地收藏的 AEU 相关 PDF

- `参考文献/论文文献/key reference/AEU/Joint optimization of altitude and resource allocation for aerial cooperative.pdf`
- `参考文献/论文文献/key reference/AEU/Outage performance analysis of antenna selection schemes in UAV-assisted.pdf`
- `参考文献/论文文献/key reference/AEU/Joint throughput and coverage maximization for moving users by optimum UAV positioning in the presence of underlaid D2D communications.pdf`
- `参考文献/论文文献/key reference/AEU/Mobile jammer enabled secure UAV communication with short packet transmission.pdf`
- `参考文献/论文文献/key reference/AEU/Optimization and performance analysis of UAV-based full-duplex systems.pdf`

## 2. 从知识库中能稳定看出的 AEU 数学偏好

即使不对所有 PDF 做全文逐页解析，仅从题目、元数据、本地分类方式和官方入口，也能稳定看出这一组 AEU 无人机论文的数学重心：

1. 偏好“可解析性能指标”，尤其是 `outage probability`、`coverage`、`throughput`、`short packet reliability`、`secrecy`、`antenna selection`。
2. 偏好“系统模型先行”，先把随机变量、信道、干扰、资源变量全部符号化，再写算法。
3. 偏好“优化问题显式写出”，包括目标函数、约束集、变量域、求解步骤。
4. 偏好“半闭式或闭式表达”，哪怕最后需要数值积分，也通常会先给出数学形式。
5. 偏好“高度、资源分配、天线选择、硬件非理想、短包长度”等参数敏感性分析。

## 3. 对应到论文题材的复杂度判断

| 本地论文对象 | 从知识库能判断出的数学主线 | 复杂度判断 | 对我们的启发 |
| --- | --- | --- | --- |
| `2023 altitude + resource allocation + short packet` | 高度优化、资源分配、短包可靠性/时延耦合 | 高 | 我们不能只写 `SINR` 定义，必须把高度与资源/风险联系起来 |
| `2024 outage + antenna selection` | 以 `outage probability` 为核心的解析性能推导 | 高 | 我们非常适合补一个基于 `Rayleigh + Gamma/Nakagami aggregate` 的闭式 outage |
| `2023 throughput + coverage + underlaid D2D` | 位置优化、覆盖-吞吐权衡、干扰约束 | 中高 | 我们可以把 `GA` 搜索问题写成带约束的 worst-case scene optimization |
| `2022 mobile jammer + short packet` | 安全/干扰/短包块长下的可靠性分析 | 高 | 我们应把 `BLER` 或短包可靠性写成独立评价层，而非简单图表 |
| `2025 SM-MIMO + hardware impairments + FD UAV` | 硬件非理想和选择机制下的性能表达 | 高 | AEU 审稿口味接受“工程非理想 + 数学分析”的组合，不排斥工程性，但必须数学化 |

## 4. 因而，AEU 并不喜欢什么

### 不喜欢的写法

- 只有流程图和模块图，没有清晰的随机变量定义
- 只有 `GA / GAN` 算法流程，没有明确目标函数
- 只有 `SINR = S/(I+N)` 一条定义，没有后续性能推导
- 只给仿真均值曲线，不给解析式、命题、渐近结论或至少半闭式积分形式
- 把经验评分函数写成主要科学结论

### 对我们当前旧稿的直接警告

- 如果继续以 `avg_total_deg` 为中心写，AEU 审稿人很容易把它看成“经验打分 + 启发式搜索”
- 如果不把 `Nakagami aggregate interference` 写成一个明确的统计近似推导，而只写成代码开关，论文会显得数学密度不足
- 如果 measured 分支不写成 `field reconstruction + poor-region discovery` 的数学问题，就会显得它只是附加图像展示

## 5. 我们当前工程在 AEU 语境下最有价值的数学资源

### 资源 1：`Rayleigh` 有用信道 + `Gamma/Nakagami` 聚合干扰

这是最关键的一条。因为它允许我们推导：

- 条件 outage
- 平均 outage
- 基于 MGF 的闭式或半闭式表达

这比单纯写 `SINR` 定义要强很多。

### 资源 2：高度显式进入几何和 LoS 概率

当前代码里高度会影响：

- 三维距离
- 仰角
- `P_LoS`
- 路径损耗
- 干扰耦合
- 旋翼推进功率

这意味着我们可以把“高度”写成真正的耦合变量，而不是场景参数。

### 资源 3：`FPC + slow-timescale CLPC`

这给了我们一个非常适合 AEU 的写法：

- `UE UL` 用标准启发的 `FPC`
- `AP / BS` 用慢时标 `CLPC`
- 二者都在统一功率域里进入 `SINR`

这会让论文呈现出“通信系统专业性”，而不只是几何仿真。

### 资源 4：实测场验证不是假的

`AERPAW / Dryad` 虽然不能写成真实基站图，但它非常适合写成：

- `measured communication field`
- `field risk reconstruction`
- `poor-region hit rate`

这让论文不仅有推导，还有外部验证闭环。

## 6. 对 AEU 投稿最合适的重写路线

建议把论文结构从旧版的“工程说明书”改成下面这种顺序：

1. 城市场景与语义受限部署的随机系统模型
2. A2G/U2U 几何、LoS/NLoS、FPC/CLPC、Nakagami 聚合干扰
3. `SINR -> outage / rate / throughput / EE` 的解析或半解析推导
4. 最坏工况搜索问题的明确目标函数与约束集
5. `GA / GAN / Random` 分别作为求解器和基线
6. `measured-map` 作为外部验证问题
7. 最后再把 `avg_total_deg` 留作工程辅助指标

## 7. 最终数学复杂度目标

### 建议至少达到的层级

- `B+` 级：系统模型完整，关键性能指标有闭式或半闭式表达，算法问题写清楚

### 最理想的层级

- `C-` 级：在 `B+` 基础上，再补一个能体现论文味道的命题，例如：
  - `Proposition 1`：在 `Rayleigh desired + Gamma aggregate interference` 下的闭式 outage
  - `Proposition 2`：高度对平均路径损耗的导数结构
  - `Corollary 1`：当 `m_eq -> +∞` 时聚合干扰退化为确定性均值

## 8. 对本项目最直接的结论

这篇稿子如果想投 AEU，最该补的不是“再讲 GA/GAN 多厉害”，而是：

- 把 `SINR-first` 写成真正的通信系统模型
- 把 `Nakagami aggregate` 写成统计推导
- 把 `outage / throughput / EE` 写成解析评价层
- 把 `measured-map` 写成验证问题

只要这四件事到位，当前工程完全有机会从“工程型稿件”抬成“通信系统导向、数学表达更强”的 AEU 风格稿件。
