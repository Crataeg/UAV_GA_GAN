# Research Thought To Code Map

## Thought 1: Urban communication degradation should be evaluated from power and SINR

- code anchor:
  - `UAV_GA.py:2846`
  - `UAV_GA.py:2975`
  - `evaluate.py:101`
  - `kpi.py:56`

- meaning:
  - power assumptions feed the received-power and interference model
  - SINR is computed in the power domain
  - all final communication metrics should be interpreted relative to SINR first

## Thought 2: UAV communication is not only terrestrial cellular communication

- code anchor:
  - `UAV_GA.py:671`
  - `UAV_GA.py:854`
  - `UAV_GA.py:3077`

- meaning:
  - urban morphology, LoS/NLoS, UAV altitude, and rotary-wing propulsion are UAV-specific layers
  - they are the reason this is not a plain terrestrial KPI exercise

## Thought 3: Closed-loop-like power control and aggregate interference should be explicit knobs

- code anchor:
  - `UAV_GA.py:2024`
  - `UAV_GA.py:3247`
  - `UAV_GA.py:2657`

- meaning:
  - UE UL uses an FPC-style interface
  - AP/BS uses a slow-timescale CLPC approximation
  - aggregate interference can use a Nakagami/Gamma approximation

## Thought 4: GA and GAN do different jobs

- code anchor:
  - `gan_uav_pipeline.py:78`
  - `gan_uav_pipeline.py:142`
  - `compare_random_ga_gan.py:296`

- meaning:
  - GA is the direct optimizer for worse synthetic scenes
  - GAN learns from GA samples and generates nearby high-risk scenarios
  - the paper should not over-claim GAN as a better worst-case optimizer unless the results support it

## Thought 5: Open data should be used as measured communication fields, not fictional full maps

- code anchor:
  - `compare_measured_map_search.py`
  - `compare_random_ga_gan.py:349`
  - `compare_random_ga_gan.py:382`

- meaning:
  - AERPAW and Dryad are used as measured SINR/throughput fields
  - they are not treated as full explicit BS/interferer/building truth maps

## Writing Consequence

- synthetic chain:
  - optimization on a defended engineering model

- measured chain:
  - search on a measured communication field

- do not collapse these into one stronger claim than the data supports
