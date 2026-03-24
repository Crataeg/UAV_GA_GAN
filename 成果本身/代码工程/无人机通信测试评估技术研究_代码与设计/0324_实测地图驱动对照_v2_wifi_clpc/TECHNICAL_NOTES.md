# Technical Notes

## Scope

This project combines:

- the measured-map-driven comparison workflow from `0324_实测地图驱动对照_v1`
- enforced Wi-Fi CLPC defaults for the synthetic GA/GAN/random engineering path
- unchanged Nakagami-m aggregate interference modeling

## Interference Aggregation

In the synthetic engineering path, interference power aggregation is handled in `UAV_GA.py`.

- `UAV_GA.py:2265-2271`
  - the per-drone precomputation path checks `interference_aggregation_model`
  - if it is `nakagami`, the code uses `_nakagami_aggregate_interference_mw()`

- `UAV_GA.py:2611-2679`
  - `_nakagami_m_for_source()` selects the per-source `m`
  - `_nakagami_aggregate_interference_mw()` computes the equivalent aggregate Nakagami-m interference

- `UAV_GA.py:2726-2728`
  - `calculate_interference_power_mw()` also routes to the same Nakagami aggregate path

Interpretation:

- the project already supports Nakagami-m aggregate interference
- this project keeps that modeling path unchanged

## Why Wi-Fi CLPC Is Enforced Here

The original code already supports CLPC-like control for AP/BS sources:

- `UAV_GA.py:654-666`
  - CLPC parameters and applicable source types are defined
- `UAV_GA.py:3217-3282`
  - `_apply_clpc_power_control()` applies a slow-timescale load-driven power adaptation

However, the original default entrypoints did not force `clpc_enabled=True`.
This project changes only the entrypoint defaults, via `project_defaults.py`, so that:

- Wi-Fi 2.4 GHz AP
- Wi-Fi 5 GHz AP

are guaranteed to participate in CLPC by default.

## Why Nakagami Is Not Further Changed

This project does not switch to an even stronger Nakagami-only assumption because:

1. Nakagami aggregate interference is already the default in the synthetic path.
2. The measured-map path does not reconstruct explicit interferer objects at all.
3. Mixing additional channel-model changes into the measured-map path would make interpretation worse, not better.

## Default Entry Points Changed

- `gan_uav_pipeline.py`
- `compare_random_ga_gan.py`
- `evaluate.py`

All three now import `project_defaults.py` and inherit:

- `clpc_enabled=True`
- `clpc_apply_types` including both Wi-Fi bands
- preserved `interference_aggregation_model='nakagami'`

## External Research Saved

See:

- `D:\论文无人机\参考文献\论文文献\key reference\wifi_clpc_nakagami_20260324`

These materials are used only as justification and documentation. They do not modify the original project code.
