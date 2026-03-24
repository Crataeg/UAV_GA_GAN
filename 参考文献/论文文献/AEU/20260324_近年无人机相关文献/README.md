# AEU 近年无人机相关文献

整理日期：2026-03-24

目标：为 `AEU - International Journal of Electronics and Communications` 投稿方向补充近年与无人机通信、UAV-assisted、中继、短包、NOMA、定位跟踪相关的官方文献入口。

本目录当前包含两类内容：

- `official_metadata/`：从 Elsevier 官方接口抓取的 XML 元数据
- `official_links/`：指向 ScienceDirect 官方文章页的 Windows 快捷链接

当前自动化状态：

- 已成功下载 7 篇论文的 Elsevier 官方 XML 元数据
- 已成功生成 7 个 ScienceDirect 官方跳转文件
- 受 Elsevier 权限与反爬限制影响，命令行无法稳定直接下载 PDF
- 其中 `Impact of hardware impairments and BS antenna tilt on 3D drone localization and tracking` 在官方元数据中标记为开放获取，后续可优先在浏览器中手动打开该官方链接尝试保存 PDF

建议优先阅读顺序：

1. `2023` 短包与资源分配：最贴近你当前短包、链路劣化和资源优化写法
2. `2023-2024` NOMA / UAV-assisted / 天线选择：补强系统模型与性能分析对比
3. `2024` 3D drone localization and tracking：补定位与硬件不理想因素
4. `2025/2026 issue` full-duplex UAV + SM MIMO：补最新方向和审稿期刊口味

文献清单：

1. `2026-01-31` / `2025` 录用后见刊
   Title: Enhancing SM MIMO performance under non-ideal transceiver hardware via full-duplex UAV and antenna selection
   DOI: `10.1016/j.aeue.2025.156122`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841125004637)
   Relevance: 最新的 UAV 中继、天线选择、硬件非理想建模，可用于讨论工程可实现性与实际硬件损伤。

2. `2024-12-31`
   Title: Impact of hardware impairments and BS antenna tilt on 3D drone localization and tracking
   DOI: `10.1016/j.aeue.2024.155543`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841124004291)
   Open access: `true`
   Relevance: 对无人机三维定位、基站天线倾角、硬件不理想因素的分析很适合补充实验和系统假设部分。

3. `2024-05-31`
   Title: Outage performance analysis of antenna selection schemes in UAV-assisted networks
   DOI: `10.1016/j.aeue.2024.155278`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841124001638)
   Relevance: 适合补 UAV-assisted 网络中的 outage probability、天线选择与性能边界比较。

4. `2023-11-30`
   Title: Combining transmit antenna selection and full-duplex aerial relay drone for performance enhancement of multiuser NOMA millimeter-wave communications
   DOI: `10.1016/j.aeue.2023.154928`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841123004028)
   Relevance: 对 NOMA、毫米波、无人机中继的组合建模比较完整，适合补 related work。

5. `2023-10-31`
   Title: Joint optimization of altitude and resource allocation for aerial cooperative short packet communication system
   DOI: `10.1016/j.aeue.2023.154858`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841123003321)
   Relevance: 与你当前短包通信、资源分配、最坏工况设计的相关性最高，建议重点看。

6. `2023-03-31`
   Title: Joint throughput and coverage maximization for moving users by optimum UAV positioning in the presence of underlaid D2D communications
   DOI: `10.1016/j.aeue.2023.154541`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841123000158)
   Relevance: 适合补 UAV 位置优化、吞吐量和覆盖权衡，以及 D2D 干扰背景。

7. `2022-12-31`
   Title: Mobile jammer enabled secure UAV communication with short packet transmission
   DOI: `10.1016/j.aeue.2022.154434`
   Official URL: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1434841122003041)
   Relevance: 虽然略早，但和短包、安全、轨迹优化关系很强，适合补引言或 related work 中的直接对标。
