# 20260325 全网补证方法来源清单

访问日期：`2026-03-25`

## 本次补证目的

- 把主链里真正要写进论文的方法来源、标准来源、开源数据来源固定下来。
- 把仍然缺乏稳健来源支撑的干扰源类型显式降级为“默认不启用”，避免继续混入主线结果。

## 已保存网页 / PDF

- `47_CFR_15_247.html`
  - 来源：<https://www.law.cornell.edu/cfr/text/47/15.247>
  - 用途：`wifi_2_4g / wifi_5_8g` 与 `915 MHz ISM` 类设备的频段与发射功率语义锚点。

- `Gamma_Sum_Nakagami_arXiv_1202_2576.pdf`
  - 来源：<https://arxiv.org/pdf/1202.2576.pdf>
  - 用途：聚合干扰 Gamma / Nakagami 近似的理论锚点。

- `3GPP_TR_38_901_v190200p.pdf`
  - 来源：<https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.02.00_60/tr_138901v190200p.pdf>
  - 用途：城市蜂窝布局先验与 3GPP 场景口径锚点。

- `3GPP_TS_36_213_v160200p.pdf`
  - 来源：<https://www.etsi.org/deliver/etsi_ts/136200_136299/136213/16.02.00_60/ts_136213v160200p.pdf>
  - 用途：UE 上行 FPC 语义锚点。

- `3GPP_TS_36_101_v161600p.pdf`
  - 来源：<https://www.etsi.org/deliver/etsi_ts/136100_136199/136101/16.16.00_60/ts_136101v161600p.pdf>
  - 用途：LTE 频段口径锚点。

- `3GPP_TS_38_104_v170900p.pdf`
  - 来源：<https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/17.09.00_60/ts_138104v170900p.pdf>
  - 用途：NR 频段口径锚点。

- `AERPAW_dataset_page.html`
  - 来源：<https://aerpaw.org/dataset/aerpaw-ericsson-5g-uav-experiment/>
  - 用途：AERPAW 开源数据来源页面。

- `Dryad_10_5061_dryad_wh70rxx06.html`
  - 来源：<https://datadryad.org/dataset/doi%3A10.5061/dryad.wh70rxx06>
  - 用途：Dryad 开源数据 DOI 页面。

## 本次应用到主链的结论

- 根目录主链默认只启用 `wifi_2_4g / wifi_5_8g / cellular_4g / cellular_5g / cellular_ue_ul`。
- `gnss_jammer / industrial_device / satellite_ground` 不再作为主线默认干扰源写法；它们保留在代码里，但默认不启用。
- measured 对比主线统一改为 `measured-map-driven` 口径，不再把公开数据描述成真实完整的 `BS / 干扰源 / 建筑附着热点` 地图。

## 写作边界

- `WiFi AP` 密度仍是项目场景先验，不是实测城市事实。
- `4G / 5G` 背景站点密度现在按 3GPP 场景先验解释，不再写成来历不明的常数。
- `CLPC` 只能写成 `slow-timescale AP/BS power-control approximation`。
- `Nakagami` 只能写成 `aggregate interference approximation`，不能写成“所有干扰源统一真实信道模型”。
