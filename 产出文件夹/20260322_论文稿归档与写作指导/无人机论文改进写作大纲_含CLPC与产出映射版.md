# 无人机论文改进写作大纲（含 CLPC 与产出映射版）

生成时间：2026-03-22  
用途：在原《无人机论文写作指导大纲_技术回顾版》的基础上，吸收本次新增的 `CLPC 闭环功控接口实验` 与 `CLPC 技术对齐后的全工程输出`，形成一份可直接指导正式论文重写的新版大纲。

---

## 0. 这次必须纳入论文的新产出

### 0.1 已归档的 4 份论文稿

- `D:\论文无人机\产出文件夹\20260322_论文稿归档与写作指导\uav_ga_gan_sci_manuscript.docx`
- `D:\论文无人机\产出文件夹\20260322_论文稿归档与写作指导\uav_ga_gan_sci_manuscript.pdf`
- `D:\论文无人机\产出文件夹\20260322_论文稿归档与写作指导\uav_ga_gan_sci_manuscript_zh.docx`
- `D:\论文无人机\产出文件夹\20260322_论文稿归档与写作指导\uav_ga_gan_sci_manuscript_zh.pdf`

### 0.2 闭环功控接口实验产出

- `D:\论文无人机\产出文件夹\20260322_CLPC闭环功控接口实验\README.md`
- `D:\论文无人机\产出文件夹\20260322_CLPC闭环功控接口实验\clpc_interface_summary.json`
- `D:\论文无人机\产出文件夹\20260322_CLPC闭环功控接口实验\figures\figure_01_clpc_layout.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC闭环功控接口实验\figures\figure_02_clpc_metric_boxes.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC闭环功控接口实验\figures\figure_03_clpc_ue_power_boxes.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC闭环功控接口实验\figures\figure_04_clpc_power_vs_margin.png`

### 0.3 CLPC 技术对齐后的全工程论文图

- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\README.md`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\comparison_summary.json`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\comparison_means.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\comparison_box.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\comparison_total_deg_overview.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\ga_top_view.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\ga_side_view.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\ga_height_analysis.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\ga_communication_analysis.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\ga_interference_location.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\aerpaw_ga_spatial_compare.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\aerpaw_ga_metric_compare.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\dryad_ga_spatial_compare.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\dryad_ga_metric_compare.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\kpi_outage_curve.png`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\figures\kpi_rate_stats.png`

### 0.4 新增参考支撑文件

- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\references\极简_工程仿真模型改进_CLPC_Nakagami.pptx`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\references\EARTH_olsson_E3F_workshop_2012.pdf`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\references\ETSI_ES_202_706_V1_4_1_2014-12.pdf`
- `D:\论文无人机\产出文件夹\20260322_CLPC技术对齐_全工程论文图\references\New_results_on_the_sum_of_Gamma_random_variates_arXiv_1202_2576.pdf`

---

## 1. 现在论文主线要怎么改

旧主线偏“城市 UAV 通信 + GA/GAN 最坏场景搜索”。  
新主线必须升级为：

> 面向城市无人机通信最坏退化边界，构建“地图感知传播 + 干扰源标定 + AP/BS 型 CLPC + UE UL 的 FPC/CLPC 接口 + Nakagami-m 聚合干扰 + 功率域劣化算法 + GA/GAN + 公共实测校核”的一体化工程框架。

这个主线相比旧稿多了两层新技术：

- 第一层：`AP / BS` 的 `CLPC` 发射功率近似已经按 PPT 技术思路接入工程。
- 第二层：`UE UL` 的 `FPC/CLPC` 接口仍然保留，不能被新 CLPC 覆盖掉。

因此，论文里必须明确写出“闭环功控有两层”：

- `UE UL`：走 FPC 公式，保留 `f(ΔTPC)` 项。
- `WiFi / 4G / 5G`：走时段-负载 `CLPC` 发射功率近似。

---

## 2. 新版论文总纲

## 第一章 引言

这一章的目标不是重复讲 UAV 应用背景，而是把研究问题收束到“最坏退化边界”。

建议写法：

- 平均覆盖与平均路径损耗不是本文主问题。
- 本文主问题是：在地图、遮挡、异构干扰、时段-负载功控和随机衰落共同存在时，哪里是最坏边界。
- 当前工作不是单一优化器论文，而是一个工程方法链。

对应产出：

- 4 份归档论文稿用于继承引言骨架。
- `comparison_means.png`
- `comparison_box.png`

本章必须点出的新贡献：

- 地图存在时的确定性 LoS。
- 干扰源的类型化标定。
- AP/BS 的 CLPC 与 UE UL 的 FPC/CLPC 并存。
- Nakagami-m 聚合干扰。
- 功率域劣化算法。

## 第二章 无人机通信特点分析与研究必要性

这是最需要加强的一章，必须单列。

建议结构：

- 2.1 无人机通信为什么不同于地面无线
- 2.2 高度 h 同时作用于服务链路与干扰链路
- 2.3 地图感知 LoS 使链路变成分段函数
- 2.4 高度、频谱耦合、干扰活动度共同导致非单调性
- 2.5 因而必须搜索最坏边界，而不能靠经验判断

对应产出：

- `ga_height_analysis.png`
- 若仍保留旧版第二版稿中的高度非单调示意图，也应与这里合并使用

本章必须保留的公式：

- `d(h)`
- `χ_LoS(h)` 或几何 LoS 指示量
- `PL_s(h)`
- `P_s(h)`
- `PL_i(h)`
- `I(h)`
- `dPL_i/dh`
- `dI/dh`

## 第三章 城市几何、地图感知传播与组件化链路预算

建议结构：

- 3.1 α–β–γ 城市形态参数
- 3.2 建筑生成、语义层和场景观测量
- 3.3 几何 LoS 与概率 LoS 回退
- 3.4 P.1238 / P.2109 与室内/穿透传播
- 3.5 频谱耦合与 ACIR 进入链路预算

对应产出：

- `ga_top_view.png`
- `ga_side_view.png`
- `ga_interference_location.png`

本章要明确写出：

- 有地图时主引擎是连线交点法。
- 概率 LoS 只是在无地图或特定模式下的后备。
- 传播链不是单一 A2G 公式，而是组件化链路预算。

## 第四章 干扰源建模、标定与闭环功控技术对齐

这是新版论文新增重点章节。

建议结构：

- 4.1 统一功率口径：EIRP-equivalent dBm
- 4.2 WiFi / 4G / 5G / UE UL / GNSS / ISM / Ku 各类源的标定链
- 4.3 AP/BS 型 CLPC 的技术背景
- 4.4 UE UL 型 FPC/CLPC 接口
- 4.5 两类功控在工程中的分工与边界

对应产出：

- `20260322_CLPC技术对齐_全工程论文图\README.md`
- `20260322_CLPC技术对齐_全工程论文图\references\*.pdf`
- `20260322_CLPC闭环功控接口实验\figure_03_clpc_ue_power_boxes.png`
- `20260322_CLPC闭环功控接口实验\figure_04_clpc_power_vs_margin.png`

本章必须明确写的技术事实：

- `AP / BS CLPC` 来源于本次 PPT 技术对齐。
- `UE UL` 仍走 `_ue_ul_fpc_power_dbm()`。
- 这两类功控不要混写成同一机制。

## 第五章 Nakagami-m 聚合干扰与功率域劣化算法

这是“数学性和技术性”最核心的一章。

建议结构：

- 5.1 单源干扰功率的线性域表达
- 5.2 频谱耦合、活动度和路径损耗如何进入 `Ω_i`
- 5.3 Nakagami-m 单源建模
- 5.4 多源干扰的矩匹配与等效 `m_eq`
- 5.5 功率裕量 `M`、`SINR`、`Rate`、`Outage`
- 5.6 综合退化度 `D_total`

对应产出：

- `figure_02_clpc_metric_boxes.png`
- `comparison_total_deg_overview.png`
- `comparison_total_deg_density.png`
- `kpi_outage_curve.png`
- `kpi_rate_stats.png`

这一章必须写成完整链条：

- `P_tx`
- `PL`
- `P_rx`
- `I_y`
- `N_y`
- `P_IN`
- `M`
- `SINR`
- `Rate`
- `Outage`
- `D_comm`
- `D_total`

不要再停留在“综合指标”这种抽象表述。

## 第六章 GA-GAN 最坏场景搜索与全工程结果

建议结构：

- 6.1 GA 搜索边界样本
- 6.2 GAN 学习边界邻域分布
- 6.3 CLPC 对齐后的全工程比较结果
- 6.4 GA / GAN / Random 的分工和局限

对应产出：

- `comparison_means.png`
- `comparison_box.png`
- `comparison_summary.json`

这一章要写清：

- 现在结果是 “CLPC 对齐版全工程输出”
- `GA avg_total_deg = 0.4225`
- `Random avg_total_deg = 0.3523`
- `GAN avg_total_deg = 0.2309`

并解释：

- GA 仍然最擅长找最坏边界。
- GAN 在当前样本量下仍更像边界邻域生成器。

## 第七章 AERPAW / Dryad 实测对齐与边界外推

建议结构：

- 7.1 AERPAW 对齐
- 7.2 Dryad 对齐
- 7.3 `measured / random / model_worst_region` 三组比较
- 7.4 为什么这是“边界外推”，不是“轨迹回放”

对应产出：

- `aerpaw_ga_spatial_compare.png`
- `aerpaw_ga_metric_compare.png`
- `dryad_ga_spatial_compare.png`
- `dryad_ga_metric_compare.png`

本章建议强调：

- 这些图现在已经是 CLPC 对齐版全工程输出的一部分。
- 论文里可以把它们解释为 “技术对齐后方法链仍保持对最坏边界的识别能力”。

## 第八章 结论与后续工作

建议结构：

- 8.1 本文构建了什么样的方法链
- 8.2 CLPC 对齐后新增了什么技术价值
- 8.3 当前还缺什么

必须明确指出的后续工作：

- AP/BS CLPC 还只是静态时段-负载近似，不是完整时序闭环。
- UE UL 的 `f(ΔTPC)` 仍是接口级，而不是完整网络控制序列。
- Nakagami 的 `m` 参数还未完成外场回归。
- 干扰源密度/活动度/覆盖半径中仍有场景默认值，需要继续实测标定。

---

## 3. 这次改大纲后，论文插图如何布置

### 第一组：方法总览与算法比较

- `comparison_means.png`
- `comparison_box.png`
- `comparison_total_deg_overview.png`

### 第二组：空间布局与传播机制

- `ga_top_view.png`
- `ga_side_view.png`
- `ga_interference_location.png`
- `ga_height_analysis.png`

### 第三组：闭环功控专用图

- `figure_01_clpc_layout.png`
- `figure_02_clpc_metric_boxes.png`
- `figure_03_clpc_ue_power_boxes.png`
- `figure_04_clpc_power_vs_margin.png`

### 第四组：实测对齐图

- `aerpaw_ga_spatial_compare.png`
- `aerpaw_ga_metric_compare.png`
- `dryad_ga_spatial_compare.png`
- `dryad_ga_metric_compare.png`

---

## 4. 这版正式写作一定要避免的错误

- 不要把 AP/BS 的 CLPC 和 UE UL 的 FPC/CLPC 混成一个概念。
- 不要再写“闭环功控已经完整实现”。
- 不要把 `Nakagami` 写成已经完成参数标定。
- 不要把有地图时的 LoS 主引擎写成概率抽样。
- 不要只写 GA/GAN，而忽略“CLPC 对齐后再跑了一次全工程”的事实。

---

## 5. 一句话总纲

如果把新版论文压缩成一句话，主旨应写成：

> 本文围绕城市无人机通信最坏退化边界，构建并验证了一个融合地图感知传播、异构干扰源标定、AP/BS 型 CLPC、UE 上行 FPC/CLPC 接口、Nakagami-m 聚合干扰以及功率域劣化算法的全工程方法链，并通过 GA/GAN 与 AERPAW/Dryad 对齐结果证明该框架具备识别保守脆弱边界的能力。

