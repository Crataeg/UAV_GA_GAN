# AEU 数学公式加强版写作大纲

## 0. 题目候选

### 候选题目 1

`Worst-Case Communication Degradation Assessment for Urban UAV Networks via SINR-First Modeling, Moment-Matched Aggregate Interference, and Measured-Field Validation`

### 候选题目 2

`A SINR-First Worst-Case Scene Discovery Framework for Urban UAV Communications With FPC/CLPC-Aware Interference Modeling and Measured-Map Validation`

### 候选题目 3

`Mathematical Modeling and Worst-Case Search of Urban UAV Communication Risk Under Semantic Interference Deployment and Nakagami Aggregate Interference`

## 1. 摘要应该怎么写

摘要建议压成 5 句：

1. 城市低空无人机通信同时受 `3D geometry + LoS/NLoS + heterogeneous interference + power control + propulsion energy` 影响。
2. 本文建立一个 `SINR-first` 数学框架，把统计城市形态、语义受限干扰部署、`FPC/CLPC` 和 `Nakagami aggregate interference` 统一到同一功率域模型。
3. 在此基础上推导 `outage / rate / BLER-adjusted throughput / energy efficiency`，并把最坏工况发现写成约束优化问题。
4. 采用 `GA / GAN / Random` 在 synthetic 场景和 `AERPAW / Dryad measured communication field` 上比较。
5. 结果表明 `GA` 更适合作为 worst-case 搜索器，`GAN` 更适合作为高风险分布补样器，且 `GA` 在 measured poor-region hit rate 上显著优于随机基线。

## 2. 引言

### 2.1 第一段不要再写“工程平台介绍”

开篇要直接进入通信系统问题：

- 低空 UAV 链路不是简单的地面蜂窝链路，因为高度同时改变 `distance`、`elevation angle`、`LoS probability`、`interference exposure` 和 `propulsion cost`
- 因而 worst-case communication scene 不是单一几何极值问题，而是多耦合变量下的风险搜索问题

### 2.2 第二段建立文献缺口

建议按三组缺口写：

1. 很多工作只做部署/轨迹优化，没有把 `power -> SINR -> outage/rate/throughput/EE` 的链条统一写清
2. 很多工作做优化，但把异构干扰写得过于粗略，缺少 `FPC/CLPC + source semantics + aggregate interference` 的统一处理
3. 很多工作只有 synthetic 仿真，缺少 measured communication field 上的外部验证

### 2.3 第三段给出贡献

建议压成 4 点：

1. 提出一个面向城市低空 UAV 的 `SINR-first` 数学建模框架。
2. 在当前工程上推导 `Rayleigh desired signal + Gamma/Nakagami moment-matched aggregate interference` 下的闭式 outage 结构。
3. 把最坏工况发现写成受约束场景优化问题，并比较 `GA / GAN / Random`。
4. 将公开测量数据写成 `measured communication field` 验证问题，而不是虚构的完整基站真值图。

## 3. 系统模型与场景建模

### 3.1 城市形态统计建模

这一节必须先给出城市随机形态。

建议写：

`(1) A = L^2`

`(2) N_b = floor(beta * A / 10^6)`

`(3) H_n ~ Rayleigh(gamma), n = 1, ..., N_b`

`(4) E[H_n] = gamma * sqrt(pi / 2)`

`(5) W = 1000 * sqrt(alpha / beta),  S = (1000 / sqrt(beta)) * (1 - sqrt(alpha))`

要点：

- `alpha` 是建筑覆盖率
- `beta` 是建筑密度
- `gamma` 是建筑高度尺度
- 这里不是做真实地图复刻，而是做 defended statistical city

工程锚点：

- `UAV_GA.py:1080-1165`
- `obsidian知识库/30_写作基础/技术涉及文献与报告总结.md`

### 3.2 语义受限部署模型

这节用来把“工程抽象”变成论文里的数学对象。

定义每个干扰源：

`(6) s_i = {tau_i, q_i, P_i, f_i, a_i, R_i, xi_i}`

其中：

- `tau_i` 为源类型
- `q_i` 为语义位置标签，如 `indoor / roof / street / pole`
- `P_i` 为发射功率语义
- `f_i` 为频率
- `a_i` 为平均活动因子
- `R_i` 为覆盖半径
- `xi_i` 为附加传播因子

再写受限部署：

`(7) q_i in Gamma(tau_i)`

如果要更像论文，可以再补：

`(8) Pr(q_i = m | tau_i) = pi_{tau_i,m},  sum_m pi_{tau_i,m} = 1`

工程锚点：

- `INTERFERENCE_SOURCE_TABLE.md`
- `UAV_GA.py:1510-1585`

### 3.3 A2G/U2U 几何与 LoS/NLoS

这一节必须把高度变量真正写进公式。

`(9) d_{s,k} = ||u_k - g_s||_2`

`(10) r_{s,k} = ||u_{k,xy} - g_{s,xy}||_2`

`(11) theta_{s,k} = arctan(|h_k - h_s| / r_{s,k})`

LoS 概率建议写成：

`(12) p_{s,k}^{LoS} = 1 / (1 + C * exp(-B * (theta_{s,k}^{deg} - C)))`

平均路径损耗写成：

`(13) L_{s,k} = FSPL(d_{s,k}, f_c) + p_{s,k}^{LoS} * eta_L + (1 - p_{s,k}^{LoS}) * eta_N`

对于干扰链路：

`(14) L_{i,k}^{int} = FSPL(d_{i,k}, f_i) + p_{i,k}^{LoS} * eta_L + (1 - p_{i,k}^{LoS}) * eta_N + 10 log10(xi_i) + L_i^{indoor}`

这一节最好直接补一个导数：

`(15) dL_{s,k}/dh_k = 20 / ln(10) * h_k / (r_{s,k}^2 + h_k^2) - (eta_N - eta_L) * dp_{s,k}^{LoS}/dtheta * dtheta_{s,k}/dh_k`

这条导数非常重要，因为它能把“最佳高度/风险高度”说得像通信论文，而不是经验观察。

工程锚点：

- `UAV_GA.py:2350-2595`

### 3.4 功率控制语义

#### UE 上行 FPC

`(16) P_i^{UE} = min(P_max, P_0 + alpha * L_i^{serv} + 10 log10(M_i) + Delta_TF + f(Delta_TPC))`

其中：

`(17) L_i^{serv} approx FSPL(d_i^{serv}, f_i) + L_i^{extra}`

#### AP/BS 慢时标 CLPC

`(18) ell_i = (1 - zeta) * alpha_t + zeta * (u_i / c_i)`

`(19) P_{i}^{CLPC} =`

- `P_{i,min}, if ell_i <= ell_sleep`
- `P_{i,max}, if ell_i >= ell_full`
- `P_{i,min} + ((ell_i - ell_sleep) / (ell_full - ell_sleep)) * (P_{i,max} - P_{i,min}), otherwise`

这一节的价值在于：

- 让功率不再只是“随机参数”
- 让工程代码里的 `FPC + CLPC` 进入论文主线

工程锚点：

- `UAV_GA.py:2024-2065`
- `UAV_GA.py:3247-3305`
- `project_defaults.py`

## 4. SINR-first 通信性能分析

### 4.1 有用信号功率

`(20) S_{s,k} = barS_{s,k} * H_{s,k}`

`(21) barS_{s,k}[mW] = 10^((P_t - L_{s,k}) / 10)`

`(22) H_{s,k} ~ Exp(1)`

如果正文想对接代码，可写成 dBm 模板：

`(23) P_{rx,s,k}[dBm] = P_t[dBm] - L_{s,k}[dB] + 10 log10(H_{s,k})`

工程锚点：

- `UAV_GA.py:2161-2180`
- `UAV_GA.py:2846-2896`

### 4.2 单个干扰源与聚合干扰

定义单个干扰项：

`(24) I_{i,k} = barOmega_{i,k} * G_{i,k}`

`(25) barOmega_{i,k} = a_i * rho(f_i, f_c) * 10^((P_i - L_{i,k}^{int}) / 10) * chi(d_{i,k}, R_i)`

其中：

- `rho(f_i, f_c)` 是频谱耦合因子
- `chi(d_{i,k}, R_i)` 是超覆盖半径修正

若采用 Nakagami 聚合，则：

`(26) G_{i,k} ~ Gamma(m_i, 1 / m_i),  E[G_{i,k}] = 1, Var[G_{i,k}] = 1 / m_i`

总干扰：

`(27) I_k = sum_i I_{i,k}`

矩匹配得到：

`(28) Omega_{Sigma,k} = sum_i barOmega_{i,k}`

`(29) sigma_{I,k}^2 = sum_i barOmega_{i,k}^2 / m_i`

`(30) m_{eq,k} = Omega_{Sigma,k}^2 / sigma_{I,k}^2`

`(31) theta_{eq,k} = sigma_{I,k}^2 / Omega_{Sigma,k} = Omega_{Sigma,k} / m_{eq,k}`

`(32) I_k approx Gamma(m_{eq,k}, theta_{eq,k})`

工程锚点：

- `UAV_GA.py:2611-2713`
- `model_source_registry.json`

### 4.3 噪声、SINR 与闭式 outage

噪声：

`(33) N_k[dBm] = N_0[dBm/Hz] + 10 log10(B) + NF`

`(34) N_k[mW] = 10^(N_k[dBm] / 10)`

SINR：

`(35) gamma_{s,k} = S_{s,k} / (I_k + N_k)`

### 4.4 这里一定要放主命题

#### Proposition 1

在 `H_{s,k} ~ Exp(1)` 且 `I_k approx Gamma(m_{eq,k}, theta_{eq,k})` 时，门限 `tau` 下的 outage probability 可写为：

`(36) P_out^{s,k}(tau) = 1 - exp(-tau * N_k / barS_{s,k}) * (1 + theta_{eq,k} * tau / barS_{s,k})^(-m_{eq,k})`

这是整篇稿子最值得“抬出来”的公式，因为它把当前工程里的：

- `Rayleigh` 有用信道
- `Nakagami/Gamma` 聚合干扰
- `SINR-first` 口径

三件事真正连成了 AEU 喜欢的闭式性能分析。

### 4.5 Rate、BLER-adjusted throughput 与 EE

`(37) R_{s,k} = B log2(1 + gamma_{s,k})`

如果采用平均速率表达，可再给：

`(38) barR_{s,k} = B / ln(2) * int_0^inf (1 - P_out^{s,k}(t)) / (1 + t) dt`

BLER 层建议从工程 LUT 拟合一个可写入正文的平滑式：

`(39) BLER(gamma_dB) approx 1 / (1 + exp(a_b * (gamma_dB - b_b)))`

然后写：

`(40) T_{s,k} = R_{s,k} * (1 - BLER(gamma_{s,k}))`

能效：

`(41) EE = sum_{k,s} T_{s,k} / (sum_k P_prop(v_k) + K * P_tx)`

旋翼推进功率：

`(42) P_prop(v) = P_0 * (1 + 3v^2 / U_tip^2) + P_i * sqrt(sqrt(1 + v^4 / (4v_0^4)) - v^2 / (2v_0^2)) + 0.5 * d_0 * rho * s * A * v^3`

工程锚点：

- `evaluate.py:101-319`
- `kpi.py:56-125`
- `UAV_GA.py:3077-3150`

### 4.6 工程优化层指标要降级表述

可以保留，但必须放到后面：

`(43) D_comm(M) = 1 / (1 + exp((M - theta_M) / s_M))`

`(44) D_speed(v) = 1 - P_prop(0) / P_prop(v)`

`(45) D_total = (w_c * D_comm + w_v * D_speed) / (w_c + w_v)`

其中：

`(46) M = P_rx[dBm] - P_IN[dBm]`

论文里必须明确：

- `D_comm / D_total` 是工程优化层指标
- `SINR / outage / throughput / EE` 才是主物理结论

## 5. Worst-case 搜索与生成补样

### 5.1 决策向量必须明确定义

`(47) x = [h_1, ..., h_K, v_1, ..., v_K, x_1^u, y_1^u, ..., tau_1, ..., P_1, ..., q_1, ..., x_1^i, y_1^i, ...]`

变量包括：

- UAV 高度
- UAV 速度
- UAV 水平位置
- 干扰源类型
- 干扰源功率
- 干扰源位置语义
- 干扰源二维坐标

### 5.2 受约束优化问题

建议正文主问题这样写：

`(48) maximize_x J(x) = 1 / (K * S) * sum_{k=1}^K sum_{s=1}^S D_total^{s,k}(x)`

subject to:

`(49) h_min <= h_k <= h_max`

`(50) 0 <= v_k <= v_max`

`(51) P_i_min(tau_i) <= P_i <= P_i_max(tau_i)`

`(52) q_i in Gamma(tau_i)`

`(53) (x_i, y_i) in A(q_i)`

如果想更 AEU 一点，可在正文中把主问题写成：

`(54) maximize_x J_1(x) = lambda_1 * barP_out(x) + lambda_2 * barD_comm(x) - lambda_3 * barEE(x)`

但这条更适合扩展版，主文仍建议忠实于现有工程。

### 5.3 GA、GAN、Random 的写法

#### GA

- 不要写成“我们提出了 GA”
- 应写成“用 GA 求解上述非凸、混合离散-连续 worst-case search problem”

#### GAN

GAN 训练目标：

`(55) min_G max_D E_{x~p_GA}[log D(x)] + E_{z~p(z)}[log(1 - D(G(z)))]`

正文一定要诚实：

- `GA` 是直接 worst-case optimizer
- `GAN` 是高风险分布学习器和补样器
- 当前结果下不能把 `GAN` 写成优于 `GA` 的最坏工况搜索器

工程锚点：

- `gan_uav_pipeline.py:79-190`
- `compare_random_ga_gan.py:1-220`

## 6. Measured communication field 验证

### 6.1 坐标对齐与 measured field 重建

`(56) p_n^a = p_n^{local} + Delta`

`(57) haty(x) = sum_{n in N_k(x)} w_n(x) y_n / sum_{n in N_k(x)} w_n(x)`

`(58) w_n(x) = 1 / max(||x - p_n^a||, 1)^p`

### 6.2 风险得分

令：

- `R_metric(x)` 表示测量指标诱导的风险
- `R_poor(x)` 表示到差点云的接近度
- `C(x)` 表示全局置信度

则当前工程可写成：

`(59) R_poor(x) = exp(-d_poor(x) / sigma_c)`

`(60) C(x) = exp(-d_meas(x) / sigma_c)`

`(61) Score_field(x) = C(x) * (0.85 * R_metric(x) + 0.15 * R_poor(x))`

### 6.3 measured-map 上的搜索评价

`(62) hit_rate_delta = 1 / N * sum_{n=1}^N 1{d_poor(x_n) <= delta}`

建议正文主打：

- `best_score`
- `hit_rate_15m`
- `hit_rate_25m`
- `hit_rate_40m`

工程锚点：

- `compare_measured_map_search.py:227-265`
- `compare_measured_map_search.py:533-535`

## 7. 实验设计与结果呈现顺序

### 7.1 Synthetic 结果顺序

固定顺序：

1. `mean / p10 / p50 / p90 SINR`
2. `outage`
3. `rate`
4. `BLER-adjusted throughput`
5. `EE`
6. 再补 `avg_total_deg`

### 7.2 当前结果可以直接写成一句话

#### Synthetic

- `GA` 平均 `SINR = 0.359 dB`
- `Random` 平均 `SINR = 3.645 dB`
- `GAN` 平均 `SINR = 5.634 dB`

所以：

- `GA` 在 synthetic 场景中最能把系统推向差状态
- `GAN` 没有超过 `GA`

#### Measured-map

- `AERPAW: GA hit_rate_15m = 1.0, Random = 0.1167`
- `Dryad:  GA hit_rate_15m = 1.0, Random = 0.1833`

所以：

- `GA` 对实测差区域的发现能力非常强
- 这能支撑“算法不是只在 synthetic 场景里有效”

### 7.3 建议图表

- `publication_synthetic_overview.png`
- `publication_link_cdfs.png`
- `publication_measured_map.png`
- `publication_iterative_benchmark.png`

## 8. 讨论

必须诚实写清的三条边界：

1. `measured open data` 不是完整的 `BS/interferer/building truth map`
2. `gnss_jammer / industrial_device / satellite_ground` 不是 defended mainline 默认背景源
3. `D_total` 是工程优化层目标，不是主物理定律

## 9. 结论

结论建议压成三句：

1. 我们建立了一个 `SINR-first` 的 UAV 通信风险评估框架，并把 `FPC / CLPC / Nakagami aggregate interference` 纳入统一功率域模型。
2. 在该模型上可得到面向 `outage / throughput / EE` 的解析或半解析评价，并把 worst-case 场景发现写成受约束优化问题。
3. 当前结果表明 `GA` 是最有效的 worst-case 搜索器，`GAN` 更适合作为高风险分布补样器，且 `GA` 在 measured communication field 上同样表现稳定。
