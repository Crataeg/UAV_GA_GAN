import argparse
import json
import os
from typing import Dict, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(report: Mapping, group: str, key: str) -> float:
    return float(report["groups"][group]["kpis"][key]["mean"])


def _throughput_a_mean(report: Mapping, group: str) -> float:
    return float(report["groups"][group]["throughput_A"]["throughput_bps"]["mean"])


def _outage_at(report: Mapping, group: str, threshold_db: float) -> float:
    outage = report["groups"][group]["kpis"]["outage"]
    return float(outage[str(float(threshold_db))])


def build_sinr_focus_summary(kpi_report: Mapping, measured_map_summary: Mapping) -> Dict:
    groups = ["ga", "gan", "random"]
    synthetic = {}
    for group in groups:
        synthetic[group] = {
            "mean_sinr_db": _mean(kpi_report, group, "sinr_db"),
            "mean_rate_mbps": _mean(kpi_report, group, "rate_bps") / 1e6,
            "mean_throughput_a_mbps": _throughput_a_mean(kpi_report, group) / 1e6,
            "outage_at_0db": _outage_at(kpi_report, group, 0.0),
            "outage_at_5db": _outage_at(kpi_report, group, 5.0),
        }

    measured = {}
    for dataset_id, dataset in measured_map_summary.get("datasets", {}).items():
        measured[dataset_id] = {
            "ga_best_score": float(dataset["search_groups"]["ga"]["best_score"]),
            "gan_best_score": float(dataset["search_groups"]["gan"]["best_score"]),
            "random_best_score": float(dataset["search_groups"]["random"]["best_score"]),
            "ga_hit_rate_15m": float(dataset["groups"]["ga"]["hit_rate_15m"]),
            "gan_hit_rate_15m": float(dataset["groups"]["gan"]["hit_rate_15m"]),
            "random_hit_rate_15m": float(dataset["groups"]["random"]["hit_rate_15m"]),
        }

    ranking = {
        "synthetic_worst_by_mean_sinr": sorted(groups, key=lambda g: synthetic[g]["mean_sinr_db"]),
        "synthetic_best_by_mean_rate": sorted(groups, key=lambda g: synthetic[g]["mean_rate_mbps"], reverse=True),
        "synthetic_best_by_mean_throughput_a": sorted(groups, key=lambda g: synthetic[g]["mean_throughput_a_mbps"], reverse=True),
    }

    return {
        "focus": "SINR-first communication degradation interpretation. Other metrics are treated as SINR-derived or SINR-monotonic evaluation layers.",
        "synthetic_summary": synthetic,
        "measured_map_summary": measured,
        "ranking": ranking,
    }


def _bar_colors(n: int) -> Iterable[str]:
    palette = ["#D62728", "#1F77B4", "#7F7F7F", "#2CA02C"]
    return palette[:n]


def plot_synthetic_focus(summary: Mapping, out_path: str) -> None:
    groups = ["ga", "gan", "random"]
    labels = [g.upper() for g in groups]
    sinr = [float(summary["synthetic_summary"][g]["mean_sinr_db"]) for g in groups]
    rate = [float(summary["synthetic_summary"][g]["mean_rate_mbps"]) for g in groups]
    thr = [float(summary["synthetic_summary"][g]["mean_throughput_a_mbps"]) for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(groups))
    colors = list(_bar_colors(len(groups)))

    axes[0].bar(x, sinr, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("dB")
    axes[0].set_title("Mean SINR")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, rate, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mbps")
    axes[1].set_title("Mean Rate")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, thr, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("Mbps")
    axes[2].set_title("Mean Throughput (BLER_A)")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("SINR-First Synthetic Comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_monotonic_relation(summary: Mapping, out_path: str) -> None:
    groups = ["ga", "gan", "random"]
    labels = [g.upper() for g in groups]
    sinr = np.asarray([float(summary["synthetic_summary"][g]["mean_sinr_db"]) for g in groups], dtype=float)
    rate = np.asarray([float(summary["synthetic_summary"][g]["mean_rate_mbps"]) for g in groups], dtype=float)
    thr = np.asarray([float(summary["synthetic_summary"][g]["mean_throughput_a_mbps"]) for g in groups], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = list(_bar_colors(len(groups)))
    for idx, label in enumerate(labels):
        axes[0].scatter(sinr[idx], rate[idx], s=80, color=colors[idx], label=label)
        axes[0].annotate(label, (sinr[idx], rate[idx]), textcoords="offset points", xytext=(5, 5))
        axes[1].scatter(sinr[idx], thr[idx], s=80, color=colors[idx], label=label)
        axes[1].annotate(label, (sinr[idx], thr[idx]), textcoords="offset points", xytext=(5, 5))

    axes[0].set_xlabel("Mean SINR (dB)")
    axes[0].set_ylabel("Mean Rate (Mbps)")
    axes[0].set_title("SINR -> Rate")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Mean SINR (dB)")
    axes[1].set_ylabel("Mean Throughput_A (Mbps)")
    axes[1].set_title("SINR -> Throughput")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_measured_focus(summary: Mapping, out_path: str) -> None:
    datasets = list(summary.get("measured_map_summary", {}).keys())
    if not datasets:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(datasets))
    width = 0.25

    ga_best = [float(summary["measured_map_summary"][ds]["ga_best_score"]) for ds in datasets]
    gan_best = [float(summary["measured_map_summary"][ds]["gan_best_score"]) for ds in datasets]
    rnd_best = [float(summary["measured_map_summary"][ds]["random_best_score"]) for ds in datasets]

    axes[0].bar(x - width, ga_best, width=width, label="GA")
    axes[0].bar(x, gan_best, width=width, label="GAN")
    axes[0].bar(x + width, rnd_best, width=width, label="Random")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ds.upper() for ds in datasets])
    axes[0].set_title("Measured-Field Best Score")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    ga_hit = [float(summary["measured_map_summary"][ds]["ga_hit_rate_15m"]) for ds in datasets]
    gan_hit = [float(summary["measured_map_summary"][ds]["gan_hit_rate_15m"]) for ds in datasets]
    rnd_hit = [float(summary["measured_map_summary"][ds]["random_hit_rate_15m"]) for ds in datasets]

    axes[1].bar(x - width, ga_hit, width=width, label="GA")
    axes[1].bar(x, gan_hit, width=width, label="GAN")
    axes[1].bar(x + width, rnd_hit, width=width, label="Random")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ds.upper() for ds in datasets])
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Measured Poor-Region Hit Rate @15m")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SINR-first summary from the compare outputs.")
    parser.add_argument("--compare_dir", required=True, help="Path to output/<run_id>/compare")
    args = parser.parse_args()

    compare_dir = os.path.abspath(args.compare_dir)
    kpi_report = _load_json(os.path.join(compare_dir, "kpi", "kpi_report.json"))
    measured_map_summary = _load_json(os.path.join(compare_dir, "measured_map_compare", "measured_map_summary.json"))

    summary = build_sinr_focus_summary(kpi_report, measured_map_summary)
    out_json = os.path.join(compare_dir, "sinr_focus_summary.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    plot_synthetic_focus(summary, os.path.join(compare_dir, "sinr_focus_synthetic.png"))
    plot_monotonic_relation(summary, os.path.join(compare_dir, "sinr_focus_monotonic.png"))
    plot_measured_focus(summary, os.path.join(compare_dir, "sinr_focus_measured_map.png"))

    print("SINR focus summary:", out_json)


if __name__ == "__main__":
    main()
