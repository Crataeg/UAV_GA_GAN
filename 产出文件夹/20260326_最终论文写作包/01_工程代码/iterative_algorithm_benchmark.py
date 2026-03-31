import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from current_compare_rules import compare_rules_dict, write_compare_rules_json
from experiment_profiles import available_profiles


ROOT = Path(__file__).resolve().parent


def _write_params(profile: str, seed: int, out_dir: Path) -> Path:
    params = {
        "experiment_profile": str(profile),
        "random_seed": int(seed),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"params_{profile}_seed{seed}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, ensure_ascii=False, indent=2)
    return path


def _run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_synthetic(compare_dir: Path) -> Dict:
    return _load_json(compare_dir / "sinr_focus_summary.json")["synthetic_summary"]


def _collect_measured(compare_dir: Path) -> Dict:
    return _load_json(compare_dir / "sinr_focus_summary.json")["measured_map_summary"]


def plot_synthetic_benchmark(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    labels = [f"{row['profile']}|{row['seed']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.25

    ga = [float(row["synthetic"]["ga"]["mean_sinr_db"]) for row in rows]
    gan = [float(row["synthetic"]["gan"]["mean_sinr_db"]) for row in rows]
    rnd = [float(row["synthetic"]["random"]["mean_sinr_db"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.8), 4.8))
    ax.bar(x - width, ga, width=width, label="GA", color="#D62728")
    ax.bar(x, gan, width=width, label="GAN", color="#1F77B4")
    ax.bar(x + width, rnd, width=width, label="Random", color="#7F7F7F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean SINR (dB)")
    ax.set_title("Iterative Synthetic Benchmark: Lower SINR Means Stronger Degradation Search")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_measured_benchmark(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    datasets = sorted({dataset for row in rows for dataset in row["measured"].keys()})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.6))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        labels = [f"{row['profile']}|{row['seed']}" for row in rows]
        x = np.arange(len(rows))
        width = 0.25
        ga = [float(row["measured"][dataset]["ga_hit_rate_15m"]) for row in rows]
        gan = [float(row["measured"][dataset]["gan_hit_rate_15m"]) for row in rows]
        rnd = [float(row["measured"][dataset]["random_hit_rate_15m"]) for row in rows]
        ax.bar(x - width, ga, width=width, label="GA", color="#D62728")
        ax.bar(x, gan, width=width, label="GAN", color="#1F77B4")
        ax.bar(x + width, rnd, width=width, label="Random", color="#7F7F7F")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{dataset.upper()} hit rate @ 15m")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively benchmark GA/GAN/random across profiles and seeds.")
    parser.add_argument("--python", default=r"D:\UAV_GA\12\.venv\Scripts\python.exe")
    parser.add_argument("--profiles", nargs="+", default=["paper_mainline", "optional_all_controlled"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027])
    parser.add_argument("--ga_runs", type=int, default=3)
    parser.add_argument("--ga_maxgen", type=int, default=3)
    parser.add_argument("--ga_nind", type=int, default=16)
    parser.add_argument("--gan_epochs", type=int, default=6)
    parser.add_argument("--gan_samples", type=int, default=10)
    parser.add_argument("--gan_batch", type=int, default=8)
    parser.add_argument("--gan_latent", type=int, default=16)
    parser.add_argument("--random_n", type=int, default=24)
    parser.add_argument("--viz_count", type=int, default=0)
    parser.add_argument("--corner_random_points", type=int, default=40)
    parser.add_argument("--corner_worst_top_k", type=int, default=20)
    parser.add_argument("--corner_grid_size", type=int, default=40)
    parser.add_argument("--out_dir", default=str(ROOT / "产出文件夹" / "20260326_算法优越性迭代基准"))
    args = parser.parse_args()

    for profile in args.profiles:
        if profile not in available_profiles():
            raise ValueError(f"Unknown profile: {profile}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params_dir = out_dir / "params"
    results: List[Dict] = []

    write_compare_rules_json(str(out_dir / "current_compare_rules.json"))
    with open(out_dir / "benchmark_setup.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    for profile in args.profiles:
        for seed in args.seeds:
            params_path = _write_params(profile, int(seed), params_dir)
            run_id = f"bench_{profile}_s{int(seed)}"

            _run(
                [
                    str(args.python),
                    str(ROOT / "gan_uav_pipeline.py"),
                    "--run_id",
                    run_id,
                    "--params",
                    str(params_path),
                    "--ga_runs",
                    str(int(args.ga_runs)),
                    "--ga_maxgen",
                    str(int(args.ga_maxgen)),
                    "--ga_nind",
                    str(int(args.ga_nind)),
                    "--gan_epochs",
                    str(int(args.gan_epochs)),
                    "--gan_samples",
                    str(int(args.gan_samples)),
                    "--gan_batch",
                    str(int(args.gan_batch)),
                    "--gan_latent",
                    str(int(args.gan_latent)),
                ],
                ROOT,
            )

            _run(
                [
                    str(args.python),
                    str(ROOT / "compare_random_ga_gan.py"),
                    "--run_id",
                    run_id,
                    "--params",
                    str(params_path),
                    "--random_n",
                    str(int(args.random_n)),
                    "--random_seed",
                    str(int(seed)),
                    "--viz_count",
                    str(int(args.viz_count)),
                    "--corner_random_points",
                    str(int(args.corner_random_points)),
                    "--corner_worst_top_k",
                    str(int(args.corner_worst_top_k)),
                    "--corner_grid_size",
                    str(int(args.corner_grid_size)),
                ],
                ROOT,
            )

            _run(
                [
                    str(args.python),
                    str(ROOT / "sinr_focus_report.py"),
                    "--compare_dir",
                    str(ROOT / "output" / run_id / "compare"),
                ],
                ROOT,
            )

            compare_dir = ROOT / "output" / run_id / "compare"
            row = {
                "profile": profile,
                "seed": int(seed),
                "run_id": run_id,
                "synthetic": _collect_synthetic(compare_dir),
                "measured": _collect_measured(compare_dir),
            }
            results.append(row)

    with open(out_dir / "iterative_benchmark_summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    plot_synthetic_benchmark(results, out_dir / "iterative_synthetic_sinr.png")
    plot_measured_benchmark(results, out_dir / "iterative_measured_hit_rate.png")
    print("Benchmark summary:", out_dir / "iterative_benchmark_summary.json")


if __name__ == "__main__":
    main()
