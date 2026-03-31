import argparse
import csv
import json
import os
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ga": "#A12424",
    "gan": "#245A9C",
    "random": "#6E6E6E",
}
LABELS = {
    "ga": "GA",
    "gan": "GAN",
    "random": "Random",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_links_csv(path: str) -> Dict[str, np.ndarray]:
    cols = {
        "sinr_db": [],
        "rate_bps": [],
        "ee_bpj": [],
    }
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cols["sinr_db"].append(float(row["sinr_db"]))
            cols["rate_bps"].append(float(row["rate_bps"]))
            cols["ee_bpj"].append(float(row["ee_bpj"]))
    return {key: np.asarray(val, dtype=float) for key, val in cols.items()}


def _load_bler_lut(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            xs.append(float(row["sinr_db"]))
            ys.append(float(row["bler"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _throughput_from_links(links: Mapping[str, np.ndarray], bler_x: np.ndarray, bler_y: np.ndarray) -> np.ndarray:
    sinr_db = np.asarray(links["sinr_db"], dtype=float)
    rate_bps = np.asarray(links["rate_bps"], dtype=float)
    bler = np.interp(sinr_db, bler_x, bler_y, left=float(bler_y[0]), right=float(bler_y[-1]))
    return rate_bps * (1.0 - bler)


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
    return arr, y


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _stat_triplet(report: Mapping, key: str, groups: Sequence[str], branch: str = "kpis") -> Tuple[List[float], List[float], List[float]]:
    means: List[float] = []
    p10: List[float] = []
    p90: List[float] = []
    for group in groups:
        stats = report["groups"][group][branch][key]
        means.append(float(stats["mean"]))
        p10.append(float(stats["p10"]))
        p90.append(float(stats["p90"]))
    return means, p10, p90


def _bar_with_quantiles(ax, groups: Sequence[str], means: Sequence[float], p10: Sequence[float], p90: Sequence[float], ylabel: str, title: str, scale: float = 1.0) -> None:
    x = np.arange(len(groups))
    colors = [COLORS[g] for g in groups]
    means_arr = np.asarray(means, dtype=float) / float(scale)
    p10_arr = np.asarray(p10, dtype=float) / float(scale)
    p90_arr = np.asarray(p90, dtype=float) / float(scale)
    ax.bar(x, means_arr, color=colors, width=0.68)
    ax.errorbar(
        x,
        means_arr,
        yerr=[means_arr - p10_arr, p90_arr - means_arr],
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=0.9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y")


def plot_publication_synthetic_overview(kpi_report: Mapping, out_path: str) -> None:
    groups = ["ga", "gan", "random"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.6))

    sinr_mean, sinr_p10, sinr_p90 = _stat_triplet(kpi_report, "sinr_db", groups, branch="kpis")
    _bar_with_quantiles(axes[0, 0], groups, sinr_mean, sinr_p10, sinr_p90, "dB", "Mean SINR")
    _panel_label(axes[0, 0], "(a)")

    rate_mean, rate_p10, rate_p90 = _stat_triplet(kpi_report, "rate_bps", groups, branch="kpis")
    _bar_with_quantiles(axes[0, 1], groups, rate_mean, rate_p10, rate_p90, "Mbps", "Mean Rate", scale=1e6)
    _panel_label(axes[0, 1], "(b)")

    thr_mean, thr_p10, thr_p90 = _stat_triplet(kpi_report, "throughput_bps", groups, branch="throughput_A")
    _bar_with_quantiles(axes[0, 2], groups, thr_mean, thr_p10, thr_p90, "Mbps", "Mean Throughput", scale=1e6)
    _panel_label(axes[0, 2], "(c)")

    ee_mean, ee_p10, ee_p90 = _stat_triplet(kpi_report, "ee_bpj", groups, branch="kpis")
    _bar_with_quantiles(axes[1, 0], groups, ee_mean, ee_p10, ee_p90, "bit/J", "Mean Energy Efficiency")
    _panel_label(axes[1, 0], "(d)")

    ax = axes[1, 1]
    for group in groups:
        outage = kpi_report["groups"][group]["kpis"]["outage"]
        xs = np.asarray([float(key) for key in outage.keys()], dtype=float)
        ys = np.asarray([float(outage[key]) for key in outage.keys()], dtype=float)
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker="o", linewidth=2.0, color=COLORS[group], label=LABELS[group])
    ax.set_xlabel("SINR Threshold (dB)")
    ax.set_ylabel("Outage Probability")
    ax.set_title("Multi-threshold Outage")
    ax.grid(True)
    ax.legend(frameon=False)
    _panel_label(ax, "(e)")

    ax = axes[1, 2]
    tail_sinr = [float(kpi_report["groups"][g]["kpis"]["tail"]["tail_mean_sinr_db"]) for g in groups]
    tail_rate = [float(kpi_report["groups"][g]["kpis"]["tail"]["tail_mean_rate_bps"]) / 1e6 for g in groups]
    x = np.arange(len(groups))
    width = 0.36
    ax.bar(x - width / 2, tail_sinr, width=width, color=[COLORS[g] for g in groups], alpha=0.92, label="Worst 5% SINR")
    ax2 = ax.twinx()
    ax2.plot(x + width / 2, tail_rate, color="#222222", marker="s", linewidth=1.8, label="Worst 5% Rate")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[g] for g in groups])
    ax.set_ylabel("Worst 5% SINR (dB)")
    ax2.set_ylabel("Worst 5% Rate (Mbps)")
    ax.set_title("Tail Risk")
    ax.grid(True, axis="y")
    _panel_label(ax, "(f)")

    fig.suptitle("Synthetic Communication Comparison", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)


def plot_publication_link_cdfs(kpi_dir: str, out_path: str) -> None:
    groups = ["ga", "gan", "random"]
    links = {g: _load_links_csv(os.path.join(kpi_dir, f"links_{g}.csv")) for g in groups}
    bler_x, bler_y = _load_bler_lut(os.path.join(kpi_dir, "BLER_A.csv"))
    throughputs = {g: _throughput_from_links(links[g], bler_x, bler_y) for g in groups}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    panels = [
        ("sinr_db", "SINR (dB)", "(a)"),
        ("rate_bps", "Rate (Mbps)", "(b)"),
        ("throughput_A", "Throughput (Mbps)", "(c)"),
    ]
    for ax, (key, xlabel, panel) in zip(axes, panels):
        for group in groups:
            if key == "throughput_A":
                x, y = _ecdf(throughputs[group] / 1e6)
            elif key == "rate_bps":
                x, y = _ecdf(links[group][key] / 1e6)
            else:
                x, y = _ecdf(links[group][key])
            ax.plot(x, y, linewidth=2.0, color=COLORS[group], label=LABELS[group])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECDF")
        ax.grid(True)
        _panel_label(ax, panel)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Link-level Distribution Comparison", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def plot_publication_measured_map(measured_summary: Mapping, out_path: str) -> None:
    datasets = ["aerpaw", "dryad"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    x = np.arange(len(datasets))
    width = 0.24

    ga_best = [float(measured_summary["datasets"][ds]["search_groups"]["ga"]["best_score"]) for ds in datasets]
    gan_best = [float(measured_summary["datasets"][ds]["search_groups"]["gan"]["best_score"]) for ds in datasets]
    rnd_best = [float(measured_summary["datasets"][ds]["search_groups"]["random"]["best_score"]) for ds in datasets]

    ax = axes[0]
    ax.bar(x - width, ga_best, width=width, color=COLORS["ga"], label="GA")
    ax.bar(x, gan_best, width=width, color=COLORS["gan"], label="GAN")
    ax.bar(x + width, rnd_best, width=width, color=COLORS["random"], label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets])
    ax.set_ylabel("Best Measured-field Score")
    ax.set_title("Worst-region Search Score")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    _panel_label(ax, "(a)")

    ga_hit = [float(measured_summary["datasets"][ds]["groups"]["ga"]["hit_rate_15m"]) for ds in datasets]
    gan_hit = [float(measured_summary["datasets"][ds]["groups"]["gan"]["hit_rate_15m"]) for ds in datasets]
    rnd_hit = [float(measured_summary["datasets"][ds]["groups"]["random"]["hit_rate_15m"]) for ds in datasets]

    ax = axes[1]
    ax.bar(x - width, ga_hit, width=width, color=COLORS["ga"], label="GA")
    ax.bar(x, gan_hit, width=width, color=COLORS["gan"], label="GAN")
    ax.bar(x + width, rnd_hit, width=width, color=COLORS["random"], label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Hit Rate @ 15 m")
    ax.set_title("Measured Poor-region Localization")
    ax.grid(True, axis="y")
    _panel_label(ax, "(b)")

    fig.suptitle("Open-data Measured-field Comparison", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def plot_publication_iterative_benchmark(iterative_summary: Sequence[Mapping], out_path: str) -> None:
    if not iterative_summary:
        return
    labels = [f"{row['profile']}|{row['seed']}" for row in iterative_summary]
    x = np.arange(len(labels))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ga = [float(row["synthetic"]["ga"]["mean_sinr_db"]) for row in iterative_summary]
    gan = [float(row["synthetic"]["gan"]["mean_sinr_db"]) for row in iterative_summary]
    rnd = [float(row["synthetic"]["random"]["mean_sinr_db"]) for row in iterative_summary]

    ax = axes[0]
    ax.bar(x - width, ga, width=width, color=COLORS["ga"], label="GA")
    ax.bar(x, gan, width=width, color=COLORS["gan"], label="GAN")
    ax.bar(x + width, rnd, width=width, color=COLORS["random"], label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Mean SINR (dB)")
    ax.set_title("Synthetic Robustness Across Runs")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    _panel_label(ax, "(a)")

    datasets = ["aerpaw", "dryad"]
    ax = axes[1]
    ga_hit = [np.mean([float(row["measured"][ds]["ga_hit_rate_15m"]) for row in iterative_summary]) for ds in datasets]
    gan_hit = [np.mean([float(row["measured"][ds]["gan_hit_rate_15m"]) for row in iterative_summary]) for ds in datasets]
    rnd_hit = [np.mean([float(row["measured"][ds]["random_hit_rate_15m"]) for row in iterative_summary]) for ds in datasets]
    ds_x = np.arange(len(datasets))
    ax.bar(ds_x - width, ga_hit, width=width, color=COLORS["ga"], label="GA")
    ax.bar(ds_x, gan_hit, width=width, color=COLORS["gan"], label="GAN")
    ax.bar(ds_x + width, rnd_hit, width=width, color=COLORS["random"], label="Random")
    ax.set_xticks(ds_x)
    ax.set_xticklabels([ds.upper() for ds in datasets])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean Hit Rate @ 15 m")
    ax.set_title("Measured-field Robustness Across Runs")
    ax.grid(True, axis="y")
    _panel_label(ax, "(b)")

    fig.suptitle("Algorithm Superiority Across Profiles and Seeds", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style figures for the UAV paper project.")
    parser.add_argument("--compare_dir", required=True, help="Path to output/<run_id>/compare")
    parser.add_argument("--iterative_summary_json", default="", help="Optional iterative benchmark summary json")
    args = parser.parse_args()

    _setup_style()
    compare_dir = os.path.abspath(args.compare_dir)
    kpi_dir = os.path.join(compare_dir, "kpi")
    kpi_report = _load_json(os.path.join(kpi_dir, "kpi_report.json"))
    measured_summary = _load_json(os.path.join(compare_dir, "measured_map_compare", "measured_map_summary.json"))

    plot_publication_synthetic_overview(kpi_report, os.path.join(compare_dir, "publication_synthetic_overview.png"))
    plot_publication_link_cdfs(kpi_dir, os.path.join(compare_dir, "publication_link_cdfs.png"))
    plot_publication_measured_map(measured_summary, os.path.join(compare_dir, "publication_measured_map.png"))

    if args.iterative_summary_json:
        iterative_summary = _load_json(os.path.abspath(args.iterative_summary_json))
        plot_publication_iterative_benchmark(iterative_summary, os.path.join(compare_dir, "publication_iterative_benchmark.png"))

    print("Publication figures written to:", compare_dir)


if __name__ == "__main__":
    main()
