# SINR First Model And Line Map

## Core Position

This project should be written as a `SINR-first` communication degradation study.

The intended chain is:

1. source power and propagation assumptions
2. received useful power `P_rx`
3. aggregate interference `I_y` and noise `N_y`
4. margin `M_db = P_rx - P_IN`
5. `SINR = 10^(M_db/10)`
6. SINR-derived or SINR-monotonic metrics:
   - outage
   - rate
   - BLER-adjusted throughput
   - energy efficiency

The score or degradation formula is only a configurable wrapper around this chain. It is not the scientific center of the paper.

## Code Line Map

### Source power and power control

- `UAV_GA.py:2024`
  - `_ue_ul_fpc_power_dbm()`
  - UE uplink FPC interface

- `UAV_GA.py:3247`
  - `_apply_clpc_power_control()`
  - slow-timescale AP/BS CLPC approximation

### Aggregate interference

- `UAV_GA.py:2657`
  - `_nakagami_aggregate_interference_mw()`
  - Gamma/Nakagami aggregate-interference approximation

### Power-domain SINR core

- `UAV_GA.py:2846`
  - `calculate_power_margin_components()`
  - outputs `P_rx`, `I_y`, `N_y`, `P_IN`, `M_db`, `SINR_linear`, `R_bps`

### SINR to communication degradation

- `UAV_GA.py:2975`
  - `calculate_comm_degradation()`
  - maps margin/SINR to a configurable degradation score

- `UAV_GA.py:3756`
  - `calculate_link_degradation()`
  - combines communication degradation with speed-energy degradation

### UAV-specific energy layer

- `UAV_GA.py:3077`
  - `calculate_speed_energy_efficiency_degradation()`
  - rotary-wing propulsion penalty

### KPI extraction and reporting

- `evaluate.py:101`
  - `extract_link_records()`
  - extracts per-link `sinr_db`, `rate_bps`, `ee_bpj`

- `evaluate.py:242`
  - `evaluate_groups_from_samples()`
  - writes KPI and throughput report per group

- `kpi.py:56`
  - `compute_kpis()`
  - SINR/rate/EE summary and outage

- `kpi.py:99`
  - `compute_throughput_kpis()`
  - BLER LUT plus throughput summary

- `plot_results.py:47`
  - `plot_kpi_comparison()`
  - includes `kpi_sinr_stats.png`, rate, EE, tail risk, throughput

## Writing Rule

- Lead with SINR.
- Treat rate and throughput as derived evaluation layers.
- Keep `avg_total_deg` as an optimization score or engineering summary, not as the main physical claim.
