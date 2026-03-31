# Paper Writing Guide

## Recommended Paper Story

### Main statement

This is a power-domain UAV communication degradation framework in which:

- source power and propagation assumptions produce `P_rx`, `I`, and `N`
- SINR is the physical center
- outage, rate, BLER-adjusted throughput, and EE are downstream evaluation layers

### Two result lines

1. `Synthetic engineering line`
   - defended city prior
   - defended source types
   - GA/GAN/random comparison

2. `Measured open-data line`
   - AERPAW and Dryad
   - measured communication field
   - GA/GAN/random worst-region search

## Recommended Claim Language

Use:

- `SINR-first communication degradation model`
- `slow-timescale CLPC approximation`
- `aggregate Nakagami-m interference approximation`
- `measured-map-driven worst-region search`

Avoid:

- `real full city interferer map`
- `fully calibrated real closed-loop controller`
- `all interferers are truly Nakagami channels`
- `GAN is always the strongest worst-case optimizer`

## Suggested Result Order

1. mean SINR
2. outage
3. mean rate
4. BLER-adjusted throughput
5. EE
6. optional score / degradation summary

## Final Folder Reading Order

1. `00_README.md`
2. `05_writing_guide/PAPER_WRITING_GUIDE.md`
3. `02_line_level_analysis/SINR_FIRST_MODEL_AND_LINE_MAP.md`
4. `02_line_level_analysis/RESEARCH_THOUGHT_TO_CODE_MAP.md`
5. `04_outputs/final_integrated_report.json`
6. `04_outputs/sinr_focus_summary.json`
