# 20260325 可信来源与非臆造主链

生成日期：`2026-03-25`

## 这次解决了什么

- 根目录主链已回收 `CLPC + measured-map` 版本，不再只有 synthetic `GA/GAN/Random` 对比。
- 根目录默认干扰源范围已收紧到 `0,1,2,3,7`，避免把来源不足的类型继续混入主结论。
- 场景 JSON 现在带 `model_provenance` 与 `enabled_interference_type_ids`，结果文件可追溯。
- 开源对比主线已统一为 `measured-map-driven`：明确声明“不重建虚构 BS / 热点 / 干扰源”。

## 本次主链运行

- 运行目录：`output/run_20260325_mainline_final`
- synthetic 摘要：`comparison_summary.json`
- integrated 摘要：`final_integrated_report.json`
- measured-map 摘要：`measured_map_summary.json`

## synthetic 主链结果

- `GA avg_total_deg = 0.4145`
- `GAN avg_total_deg = 0.2204`
- `Random avg_total_deg = 0.3359`
- `GA mean SINR = 0.3590 dB`
- `GAN mean SINR = 5.7163 dB`
- `Random mean SINR = 3.4512 dB`
- `GA mean throughput_A = 19.00 Mbps`
- `GAN mean throughput_A = 32.65 Mbps`
- `Random mean throughput_A = 33.45 Mbps`

解释：

- `GA` 仍然最擅长把 synthetic 场景推向更差的综合退化区间。
- `GAN` 在 synthetic 主链里仍然更像“高风险边界分布生成器”，不是比 `GA` 更强的最坏场景优化器。

## 开源 measured-map 结果

- `AERPAW`
  - `GA best_score = 1.2599`
  - `GAN best_score = 0.9669`
  - `Random best_score = 0.7824`
  - `GA hit_rate_15m = 1.00`
  - `GAN hit_rate_15m = 0.60`
  - `Random hit_rate_15m = 0.25`

- `Dryad`
  - `GA best_score = 1.3718`
  - `GAN best_score = 1.2024`
  - `Random best_score = 0.6512`
  - `GA hit_rate_15m = 1.00`
  - `GAN hit_rate_15m = 0.85`
  - `Random hit_rate_15m = 0.15`

解释：

- 这组结果现在可以写成“GA / GAN / Random 在 measured communication field 上搜索最差区域的能力比较”。
- 这组结果不能写成“公开数据给出了完整真实 BS / 干扰源地图后，我们直接在真实地图上优化”。

## 关联来源

- 模型注册表：`model_source_registry.json`
- 网页与标准归档：`参考文献/论文文献/key reference/20260325_全网补证_方法来源`

## 推荐写法

- synthetic 主链：写“统计城市先验 + defended source types + CLPC/Nakagami aggregate”
- measured 主链：写“open measured field + worst-region search”
- 不再把两条链混成一个“全都是真实城市、真实干扰源”的叙述
