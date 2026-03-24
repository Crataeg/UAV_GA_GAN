# 20260324 WiFi 干扰源 CLPC 与 Nakagami 研究

## 1. 全网研究结论

### 1.1 WiFi/AP 做闭环功控是“有依据，但不是蜂窝式那种强闭环”

- Cisco `Radio Resource Management / Transmit Power Control`
  - 结论：在企业级控制器管理 WiFi 网络里，AP 发射功率可以被周期性评估和调整
  - 含义：如果工程里的 WiFi 干扰源代表“企业级 / 受控 AP”，那么给它们加 CLPC/TPC 近似是有现实依据的

- Cisco `RRM White Paper - TPC Algorithm`
  - 结论：AP 功率调节是按 AP / radio 周期运行的，不是每个包、每个时隙都闭环
  - 含义：WiFi 的“闭环功控”更像慢时标控制，不应该照搬蜂窝 UE 的快时标功控语义

- `User-aware WLAN Transmit Power Control in the Wild`
  - 结论：研究上已经把 WLAN AP 功率控制和用户吞吐、负载、干扰联系起来
  - 含义：如果要把 WiFi 干扰源换成 CLPC，应理解为“负载 / 覆盖 / 干扰折中”的控制，而不是固定公式套用

- `Understanding the Limitations of Transmit Power Control`
  - 结论：TPC 并不是对所有 WLAN 场景都一定更好
  - 含义：不应把所有 WiFi 干扰源默认都改成 CLPC；至少要区分受控 AP 和非受控背景 WiFi

### 1.2 WiFi 相关干扰“也需要换成 Nakagami 衰落”这件事，不应一刀切

- `A Tractable Framework for Coverage Analysis of Cellular-Connected UAV Networks`
  - 结论：Nakagami 很适合做 UAV 网络里的总干扰统计近似
  - 含义：它适合“聚合干扰建模”，不自动等于“每一个 WiFi 干扰源都必须改成 Nakagami”

- `UAV-Assisted Superposition Coded Cooperation: Hybrid Relaying and Nakagami-m Channel for UL/DL`
  - 结论：Nakagami 在 UAV 协作 / 中继类理论分析里常用
  - 含义：如果你要做解析型或统计型对比，Nakagami 是可接受的

- `Characterization of A2G UAV communication channels under Rician fading conditions`
  - 结论：实测 LoS A2G 场景下，Rician 仍然很重要
  - 含义：如果 WiFi AP 到 UAV 的链路更接近强 LoS，就不能简单说 Nakagami 一定比 Rician 更合理

## 2. 对当前工程代码的映射

### 2.1 WiFi 干扰源现在已经具备进入 CLPC 的入口

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:60-101`
  - WiFi 2.4G / 5G AP 类型已经被单独定义

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:664-666`
  - `clpc_apply_types` 默认就包含 `wifi_2_4g`、`wifi_5_8g`

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:2096-2100`
  - WiFi 源在功率标定时会先被压到 `eirp_min_dbm` 门槛

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:3217-3282`
  - `_apply_clpc_power_control()` 对 AP / BS 类源统一施加慢时标负载闭环近似

结论：

- 从代码上说，WiFi 干扰源“已经能走 CLPC”
- 问题不在“能不能加”，而在“该不该默认加、给谁加”

### 2.2 Nakagami 在当前代码里也是“可用但非唯一”

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:817-825`
  - 提供 `interference_aggregation_model` 和 `nakagami_m_*` 系列参数

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:2265-2271`
  - 总干扰预计算可以走 Nakagami 聚合

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:2611-2679`
  - `m` 可以按源类型 / LoS-NLoS 分开设置

- `成果本身\代码工程\无人机通信测试评估技术研究_代码与设计\0315\UAV_GA.py:2114-2118`
  - 线性路径仍然保留 Rayleigh 小尺度衰落

结论：

- 当前代码已经具备“对 WiFi 单独设 Nakagami-m”的接口
- 但它不是“全局唯一 Nakagami”

## 3. 建议

### 3.1 对 WiFi 相关干扰源，不建议全部默认改成 CLPC

更合理的区分：

- 企业级 / 控制器管理 AP
  - 可以用 CLPC/TPC 近似
  - 语义是慢时标的 RRM / TPC

- 背景杂散 WiFi / 家用 AP / 不可控热点
  - 更适合固定功率或有界随机功率
  - 不宜统一套 CLPC

### 3.2 对 WiFi 相关干扰源，也不建议一刀切全部改成 Nakagami

更合理的区分：

- 聚合干扰分析
  - Nakagami 合理

- 单个 LoS 强的 AP 到 UAV 干扰链路
  - Rician 或高 m Nakagami 更合理

- 室内 / 穿墙 / 遮挡较重的 WiFi 泄漏干扰
  - Nakagami 可以作为可选统计近似

### 3.3 最稳妥的工程策略

- 不动原工程主线
- 后续如果要试：
  - 只对 `wifi_2_4g / wifi_5_8g` 单独做 `clpc_apply_types` 配置实验
  - 只对 WiFi 类型单独给 `nakagami_m_by_type`
  - 保留一个对照组：
    - `fixed WiFi power + current fading`
    - `WiFi CLPC + current fading`
    - `WiFi CLPC + WiFi-only Nakagami`

## 4. 适合论文里的结论

- “WiFi 干扰源采用慢时标闭环功控近似”是可辩护的，但前提是将其解释为受控 AP 网络
- “所有 WiFi 干扰源统一改成 Nakagami 衰落”不够稳妥
- 更好的写法是：
  - WiFi 功率控制和衰落模型都作为受控建模假设与敏感性分析项
  - 不作为唯一主线前提
