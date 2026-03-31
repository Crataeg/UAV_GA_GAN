from typing import Dict, List


EXPERIMENT_PROFILES: Dict[str, Dict] = {
    "paper_mainline": {
        "enabled_interference_type_ids": [0, 1, 2, 3, 7],
        "bg_lambda_km2_4": 0.0,
        "bg_lambda_km2_5": 0.0,
        "bg_lambda_km2_6": 0.0,
        "profile_note": (
            "Defended paper mainline. Optional source classes 4/5/6 stay out of the "
            "default synthetic chain."
        ),
    },
    "optional_all_controlled": {
        "enabled_interference_type_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "bg_lambda_km2_4": 0.0,
        "bg_lambda_km2_5": 0.0,
        "bg_lambda_km2_6": 0.0,
        "profile_note": (
            "Optional source classes 4/5/6 are available to the controlled-source search, "
            "but no background PPP prior is asserted for them."
        ),
    },
    "optional_gnss_controlled": {
        "enabled_interference_type_ids": [0, 1, 2, 3, 4, 7],
        "bg_lambda_km2_4": 0.0,
        "bg_lambda_km2_5": 0.0,
        "bg_lambda_km2_6": 0.0,
        "profile_note": "Enable GNSS jammer as an optional controlled source only.",
    },
    "optional_industrial_controlled": {
        "enabled_interference_type_ids": [0, 1, 2, 3, 5, 7],
        "bg_lambda_km2_4": 0.0,
        "bg_lambda_km2_5": 0.0,
        "bg_lambda_km2_6": 0.0,
        "profile_note": "Enable industrial ISM interferer as an optional controlled source only.",
    },
    "optional_satellite_controlled": {
        "enabled_interference_type_ids": [0, 1, 2, 3, 6, 7],
        "bg_lambda_km2_4": 0.0,
        "bg_lambda_km2_5": 0.0,
        "bg_lambda_km2_6": 0.0,
        "profile_note": "Enable Ku-band satellite-ground source as an optional controlled source only.",
    },
}


def available_profiles() -> List[str]:
    return list(EXPERIMENT_PROFILES.keys())


def apply_experiment_profile(params: Dict) -> Dict:
    out = dict(params)
    profile_name = str(out.get("experiment_profile", "paper_mainline")).strip() or "paper_mainline"
    profile = dict(EXPERIMENT_PROFILES.get(profile_name, EXPERIMENT_PROFILES["paper_mainline"]))
    for key, value in profile.items():
        if key == "profile_note":
            continue
        out.setdefault(key, value)
    out.setdefault("experiment_profile", profile_name)
    out.setdefault("experiment_profile_note", str(profile.get("profile_note", "")))
    return out
