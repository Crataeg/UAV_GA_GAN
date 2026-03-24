# WiFi CLPC And Nakagami Line-By-Line

## What Changed In This Project

- `project_defaults.py`
  - enforces `clpc_enabled=True`
  - preserves `interference_aggregation_model='nakagami'`
  - makes Wi-Fi CLPC active by default through `clpc_apply_types` and Wi-Fi load/capacity defaults

- `gan_uav_pipeline.py`
  - `default_user_params()` now calls `apply_project_defaults(...)`

- `compare_random_ga_gan.py`
  - `default_user_params()` now calls `apply_project_defaults(...)`
  - still uses `compare_measured_map_search.py` for the measured-map path

- `evaluate.py`
  - `default_user_params()` now calls `apply_project_defaults(...)`
  - measured comparison import now points to `compare_measured_map_search.py`

## Interference Aggregation In The Core Model

- `UAV_GA.py:2265-2271`
  - total interference precomputation selects the aggregate model
- `UAV_GA.py:2611-2679`
  - Nakagami-m equivalent aggregation is implemented here
- `UAV_GA.py:2726-2728`
  - runtime interference power calculation also routes to Nakagami when configured

Interpretation:

- this project does not change the aggregate interference mechanism
- Nakagami stays as the default aggregate interference model

## WiFi CLPC In The Core Model

- `UAV_GA.py:60-101`
  - Wi-Fi 2.4 GHz and 5 GHz AP source types are defined
- `UAV_GA.py:2096-2100`
  - Wi-Fi sources are power-calibrated before CLPC
- `UAV_GA.py:3217-3282`
  - AP/BS closed-loop-like power control is applied here

Interpretation:

- the original model already had the CLPC machinery
- this project only turns that machinery on by default for Wi-Fi relevant entrypoints

## External Research Saved

- `D:\论文无人机\参考文献\论文文献\key reference\wifi_clpc_nakagami_20260324\Cisco_RRM_TPC_Guide_17_18.pdf`
- `D:\论文无人机\参考文献\论文文献\key reference\wifi_clpc_nakagami_20260324\Cisco_RRM_TPC_Overview.html`
- `D:\论文无人机\参考文献\论文文献\key reference\wifi_clpc_nakagami_20260324\User_aware_WLAN_TPC_in_the_Wild.url`
- `D:\论文无人机\参考文献\论文文献\key reference\wifi_clpc_nakagami_20260324\TPC_Limitations_Indoor_WLANs.url`
- `D:\论文无人机\参考文献\论文文献\key reference\wifi_clpc_nakagami_20260324\UAV_Nakagami_Coverage_Framework.url`

## Why This Is The Final Choice

1. Wi-Fi AP power control has real engineering basis in controller-managed WLANs.
2. The current project already supports CLPC-like AP/BS power adaptation.
3. Nakagami aggregate interference is already implemented and enabled by default.
4. Therefore, the minimal and defensible change is:
   - force Wi-Fi CLPC on
   - leave Nakagami aggregation unchanged
