import json
import os
from typing import Dict


def compare_rules_dict() -> Dict:
    return {
        "synthetic_compare": {
            "groups": ["GA", "GAN", "Random"],
            "same_model_same_bounds": True,
            "ga_definition": "GA searches the synthetic engineering model for worse communication scenes.",
            "gan_definition": "GAN learns from GA elite samples and generates nearby high-risk scenarios.",
            "random_definition": "Random uses the same variable bounds without optimization.",
            "core_metric_priority": [
                "SINR",
                "outage",
                "rate",
                "BLER-adjusted throughput",
                "energy efficiency",
                "optional score layer",
            ],
            "current_summary_files": [
                "comparison_summary.json",
                "kpi/kpi_report.json",
                "sinr_focus_summary.json",
            ],
        },
        "measured_map_compare": {
            "datasets": ["AERPAW", "Dryad"],
            "mode": "measured_map_driven",
            "rule": (
                "Public open data are treated as measured communication fields, not as complete explicit "
                "BS/interferer/building truth maps."
            ),
            "groups": [
                "measured_poor_cloud",
                "measured_selected",
                "random",
                "ga",
                "gan",
                "field_worst_region",
            ],
            "main_effect_metrics": [
                "best_score",
                "hit_rate_15m",
                "hit_rate_25m",
                "hit_rate_40m",
            ],
        },
        "optional_source_policy": {
            "paper_mainline": "Only defended source classes are enabled by default.",
            "optional_profiles": {
                "optional_all_controlled": "Enable 4/5/6 as optional controlled-source classes without asserting background PPP prevalence.",
                "optional_gnss_controlled": "Enable class 4 only as optional controlled source.",
                "optional_industrial_controlled": "Enable class 5 only as optional controlled source.",
                "optional_satellite_controlled": "Enable class 6 only as optional controlled source.",
            },
        },
    }


def write_compare_rules_json(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(compare_rules_dict(), handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    write_compare_rules_json(os.path.join("output", "current_compare_rules.json"))
