# Measured-Map Variant Guide

## Purpose

This folder is a copied variant of `0315`, created to avoid changing the original code.

Its open-dataset comparison path has been changed from:

- reconstructing a fictional serving BS proxy
- attaching empirical hotspot interferers
- optimizing a synthetic interference scene on top of measured data

to:

- treating AERPAW and Dryad as measured communication maps
- interpolating a measured risk field directly from observed SINR / throughput samples
- running GA / GAN / random search on the measured field itself

This `v2_wifi_clpc` variant also enforces Wi-Fi CLPC defaults for the synthetic engineering path while keeping Nakagami aggregate interference unchanged.

## Why This Variant Exists

The public datasets provide measured trajectories or measured map points, not a full annotated map of:

- explicit serving BS geometry
- explicit interferer positions
- explicit building-attached hotspot sources

Therefore, plotting or optimizing with fictional BS / interferer objects can overstate what the dataset actually supports.

This variant keeps the algorithm meaningful by changing the question:

- old question: "What synthetic interference scene makes the model worst?"
- new question: "Where is the worst communication region in the measured map, and can GA / GAN find it better than random search?"

## Main Files

- `compare_measured_map_search.py`
  - new measured-map-driven comparison module
- `project_defaults.py`
  - shared defaults enforcing Wi-Fi CLPC while preserving Nakagami aggregation
- `compare_random_ga_gan.py`
  - same main entrypoint style as before, but now calls the measured-map-driven module
- `compare_measured_cornercases.py`
  - legacy measured-corner-case reconstruction logic kept for reference only

## New Comparison Meaning

For AERPAW and Dryad:

1. Load measured SINR / throughput points
2. Align them into a local positive XY plane
3. Build an interpolated measured risk field
4. Use GA to search high-risk locations on that field
5. Use GAN to learn the GA-discovered high-risk region distribution
6. Compare against random samples on the same field
7. Compare all groups against the real measured poor-point cloud

For the synthetic GA/GAN/random engineering path:

1. Wi-Fi AP source types are forced to use CLPC-capable defaults
2. Nakagami aggregate interference remains the default
3. The measured-map path stays free of fictional BS / hotspot reconstruction

This makes the comparison interpretable:

- `Measured poor cloud`: real low-quality measured region
- `Random`: uninformed search baseline
- `GA`: optimization-based search on measured field
- `GAN`: learned generative search on measured field
- `Field worst region`: dense-grid worst area of the interpolated field

## Expected Outputs

- `output/<run_id>/compare/measured_map_compare/<dataset>/measured_map_report.json`
- `output/<run_id>/compare/measured_map_compare/<dataset>/grid_map.csv`
- `output/<run_id>/compare/measured_map_compare/<dataset>/field_worst_region.csv`
- `output/<run_id>/compare/measured_map_compare/<dataset>/ga_samples_scored.csv`
- `output/<run_id>/compare/measured_map_compare/<dataset>/gan_samples_scored.csv`
- `output/<run_id>/compare/measured_map_compare/<dataset>/random_samples_scored.csv`
- `output/<run_id>/compare/measured_map_compare/<dataset>/<dataset>_measured_map_search.png`
- `output/<run_id>/compare/measured_map_compare/<dataset>/<dataset>_measured_map_compare.png`

## Recommended Commands

- Run the full copied workflow:
  - `& .\.venv\Scripts\python.exe .\compare_random_ga_gan.py --run_id measured_map_demo`

- Run only the measured-map comparison:
  - `& .\.venv\Scripts\python.exe .\compare_measured_map_search.py`

## Interpretation Rule

When using outputs from this folder in writing:

- do not describe the map as a true BS / interferer map from the dataset
- describe it as a measured communication field or measured degradation field
- describe GA / GAN as searching worst regions in a measured-map-derived surrogate field
- describe Wi-Fi CLPC as a slow-timescale AP power-control approximation, not as packet-level closed-loop control
