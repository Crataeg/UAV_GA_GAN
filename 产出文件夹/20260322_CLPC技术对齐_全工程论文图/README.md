# CLPC 技术对齐与全工程输出说明

生成时间：2026-03-22

## 1. 找到并对齐的 PPT

- 本次用于技术对齐的 PPT 为：
  - `C:\Users\52834\Downloads\极简_工程仿真模型改进_CLPC_Nakagami.pptx`
- 已归档副本：
  - [极简_工程仿真模型改进_CLPC_Nakagami.pptx](./references/极简_工程仿真模型改进_CLPC_Nakagami.pptx)

PPT 核心观点有两条：

- `CLPC` 的研究主语是 `WiFi/AP/蜂窝基站` 的时段-负载闭环发射功率，而不只是 UE 上行 `FPC`。
- `Nakagami-m` 应用于聚合干扰的分布刻画，而不只是把所有干扰线性加总成一个固定均值。

## 2. 下载并归档的相关文件

- [EARTH_olsson_E3F_workshop_2012.pdf](./references/EARTH_olsson_E3F_workshop_2012.pdf)
  - 用于对齐 `24 小时 traffic profile / load levels / BS output power` 的研究背景。
- [ETSI_ES_202_706_V1_4_1_2014-12.pdf](./references/ETSI_ES_202_706_V1_4_1_2014-12.pdf)
  - 用于对齐无线接入设备能效/功率评估背景。
- [New_results_on_the_sum_of_Gamma_random_variates_arXiv_1202_2576.pdf](./references/New_results_on_the_sum_of_Gamma_random_variates_arXiv_1202_2576.pdf)
  - 用于补充 `Nakagami / Gamma 和求和近似` 的理论背景。

## 3. 技术对齐后做出的工程修正

这次没有再把“闭环功控”只理解成 UE 上行 `ΔTPC` 接口，而是按 PPT 的主语修正到了 `AP / BS` 发射功率侧。

已修正位置：

- `D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py`

具体改动：

- 新增 `clpc_enabled / clpc_hour_of_day / clpc_blend_alpha / clpc_users_by_type / clpc_capacity_by_type / clpc_pmin_frac_by_type` 等参数。
- 新增 `_apply_clpc_power_control()`：
  - 输入：`hour-of-day`、`base_load`、`users/capacity`
  - 融合：`effective_load = (1-α) * base_load + α * user_load`
  - 输出：基于 `deep_sleep / dynamic_adjust / full_power` 三状态的 `tx_dbm`
- 将该 CLPC 逻辑接到 `_calibrate_source_profile_and_power()` 后段：
  - 对 `wifi_2_4g / wifi_5_8g / cellular_4g / cellular_5g` 生效
  - 不作用于 `cellular_ue_ul`
  - `cellular_ue_ul` 仍走原有 `FPC` 接口

这意味着：

- 现在工程里同时存在两类“功控”：
  - `UE 上行 FPC/CLPC 接口`
  - `AP / BS 的时段-负载 CLPC 发射功率近似`

## 4. 全工程运行方式

本次完整运行参数文件：

- [clpc_fullrun_params.json](./clpc_fullrun_params.json)

完整流程运行目录在工程内：

- `D:\论文无人机\成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\output\run_20260322_clpc_full`

归档到本目录的核心统计文件：

- [comparison_summary.json](./comparison_summary.json)

## 5. 结果摘要

本次 `CLPC 对齐版` 全工程输出中：

- `GA` 平均综合退化度：`0.4225`
- `GAN` 平均综合退化度：`0.2309`
- `Random` 平均综合退化度：`0.3523`
- `GA` 平均功率裕量：`-0.61 dB`
- `GAN` 平均功率裕量：`5.61 dB`
- `Random` 平均功率裕量：`2.53 dB`

说明在启用 `CLPC + Nakagami` 的配置下：

- `GA` 仍然最容易把场景推进到更差的功率裕量区间。
- `GAN` 在当前样本量下仍更像“边界邻域生成器”，而不是最坏边界发现器。

## 6. 论文建议优先使用的图片

### 6.1 方法/结果总览图

- [comparison_means.png](./figures/comparison_means.png)
- [comparison_box.png](./figures/comparison_box.png)
- [comparison_total_deg_overview.png](./figures/comparison_total_deg_overview.png)
- [comparison_total_deg_density.png](./figures/comparison_total_deg_density.png)

### 6.2 工程场景与通信状态图

- [ga_top_view.png](./figures/ga_top_view.png)
- [ga_side_view.png](./figures/ga_side_view.png)
- [ga_interference_location.png](./figures/ga_interference_location.png)
- [ga_height_analysis.png](./figures/ga_height_analysis.png)
- [ga_communication_analysis.png](./figures/ga_communication_analysis.png)

### 6.3 实测对齐图

- [aerpaw_ga_spatial_compare.png](./figures/aerpaw_ga_spatial_compare.png)
- [aerpaw_ga_metric_compare.png](./figures/aerpaw_ga_metric_compare.png)
- [dryad_ga_spatial_compare.png](./figures/dryad_ga_spatial_compare.png)
- [dryad_ga_metric_compare.png](./figures/dryad_ga_metric_compare.png)

### 6.4 KPI 图

- [kpi_outage_curve.png](./figures/kpi_outage_curve.png)
- [kpi_rate_stats.png](./figures/kpi_rate_stats.png)

## 7. 写作边界

这次对齐后，论文可以更准确地写成：

- 已实现 `AP / BS` 的时段-负载 CLPC 发射功率近似。
- 已实现 `UE UL` 的 FPC/CLPC 修正接口。
- 已实现 `Nakagami-m` 聚合干扰机制。

但仍不能写成：

- 已完成真实网络时序级闭环控制仿真。
- 已完成 CLPC 参数的外场标定。
- 已完成 Nakagami `m` 参数的测量回归。

