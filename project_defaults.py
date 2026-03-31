import os
from typing import Dict

from experiment_profiles import apply_experiment_profile


def apply_project_defaults(params: Dict) -> Dict:
    out = dict(params)
    out = apply_experiment_profile(out)

    # Keep the current project behavior: aggregate interference with Nakagami-m
    # unless the caller explicitly overrides it.
    out.setdefault("interference_aggregation_model", "nakagami")

    # Enforce slow-timescale closed-loop-like power control for controller-managed AP/BS sources.
    out["clpc_enabled"] = True
    out.setdefault("clpc_hour_of_day", 18.0)
    out.setdefault("clpc_blend_alpha", 0.5)
    out.setdefault("clpc_sleep_threshold", 0.10)
    out.setdefault("clpc_full_threshold", 0.90)
    out["clpc_apply_types"] = ["wifi_2_4g", "wifi_5_8g", "cellular_4g", "cellular_5g"]

    # Replace previously opaque cellular PPP defaults with explicit 3GPP-style
    # layout priors that can be overridden by the caller.
    # 4G macro prior: UMa-like inter-site distance ~500 m => ~4.62 sites/km^2
    # 5G small-cell prior: UMi-like inter-site distance ~200 m => ~28.87 sites/km^2
    out.setdefault("bg_lambda_km2_2", 4.62)
    out.setdefault("bg_lambda_km2_3", 28.87)
    out.setdefault("bg_lambda_km2_4", 0.0)
    out.setdefault("bg_lambda_km2_5", 0.0)
    out.setdefault("bg_lambda_km2_6", 0.0)

    out.setdefault(
        "model_source_registry_json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_source_registry.json"),
    )

    # Wi-Fi-specific load/capacity knobs to make Wi-Fi CLPC actually active by default.
    users_by_type = dict(out.get("clpc_users_by_type", {}) or {})
    users_by_type.setdefault("wifi_2_4g", 8.0)
    users_by_type.setdefault("wifi_5_8g", 8.0)
    out["clpc_users_by_type"] = users_by_type

    capacity_by_type = dict(out.get("clpc_capacity_by_type", {}) or {})
    capacity_by_type.setdefault("wifi_2_4g", 16.0)
    capacity_by_type.setdefault("wifi_5_8g", 16.0)
    out["clpc_capacity_by_type"] = capacity_by_type

    pmin_frac_by_type = dict(out.get("clpc_pmin_frac_by_type", {}) or {})
    pmin_frac_by_type.setdefault("wifi_2_4g", 0.20)
    pmin_frac_by_type.setdefault("wifi_5_8g", 0.20)
    out["clpc_pmin_frac_by_type"] = pmin_frac_by_type

    return out
