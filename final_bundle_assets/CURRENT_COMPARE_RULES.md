# Current Compare Rules

## Synthetic Compare

The synthetic comparison always uses the same engineering model and the same variable bounds for three groups:

- `GA`
  - direct worse-scene optimizer
- `GAN`
  - generator trained on GA elite samples
- `Random`
  - uninformed baseline using the same bounds

### Priority of interpretation

1. `SINR`
2. `outage`
3. `rate`
4. `BLER-adjusted throughput`
5. `EE`
6. optional score layer such as `avg_total_deg`

### Current reading rule

- lower mean SINR means stronger degradation search
- if rate / throughput move consistently with SINR, they are supporting layers

## Measured Open-Data Compare

Datasets:

- `AERPAW`
- `Dryad`

Mode:

- `measured_map_driven`

Meaning:

- public open data are treated as measured communication fields
- they are not treated as complete explicit BS/interferer/building truth maps

### Current measured effect metrics

- `best_score`
- `hit_rate_15m`
- `hit_rate_25m`
- `hit_rate_40m`

### Current reading rule

- higher hit rate near measured poor cloud means better worst-region discovery
- higher best score means stronger search into the measured poor region

## Optional Source Rule

- `paper_mainline`
  - only defended source classes are enabled by default

- `optional_all_controlled`
  - classes `4/5/6` become available as controlled-source search types
  - no background PPP prevalence is asserted for them

This keeps them optional instead of silently disabled, while avoiding unsupported prevalence claims.
