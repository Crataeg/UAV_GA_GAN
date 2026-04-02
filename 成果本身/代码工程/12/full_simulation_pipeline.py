from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent

PROFILE_PRESETS: Dict[str, Dict[str, object]] = {
    "quick": {
        "ga_runs": 2,
        "ga_maxgen": 2,
        "ga_nind": 10,
        "ga_seed": 123,
        "gan_epochs": 2,
        "gan_batch": 8,
        "gan_latent": 16,
        "gan_samples": 2,
        "device": "cpu",
        "random_n": 10,
        "random_seed": 2024,
        "viz_count": 2,
        "skip_measured_corner_compare": True,
        "corner_random_points": 20,
        "corner_worst_top_k": 5,
        "corner_grid_size": 20,
        "corner_tx_power_dbm": 43.0,
        "corner_bandwidth_hz": 100e6,
        "corner_disable_hotspots": False,
    },
    "balanced": {
        "ga_runs": 6,
        "ga_maxgen": 6,
        "ga_nind": 16,
        "ga_seed": 123,
        "gan_epochs": 30,
        "gan_batch": 16,
        "gan_latent": 16,
        "gan_samples": 8,
        "device": "cpu",
        "random_n": 20,
        "random_seed": 2024,
        "viz_count": 2,
        "skip_measured_corner_compare": False,
        "corner_random_points": 40,
        "corner_worst_top_k": 10,
        "corner_grid_size": 25,
        "corner_tx_power_dbm": 43.0,
        "corner_bandwidth_hz": 100e6,
        "corner_disable_hotspots": False,
    },
    "full": {
        "ga_runs": 40,
        "ga_maxgen": 40,
        "ga_nind": 30,
        "ga_seed": 123,
        "gan_epochs": 300,
        "gan_batch": 32,
        "gan_latent": 32,
        "gan_samples": 20,
        "device": "cpu",
        "random_n": 50,
        "random_seed": 2024,
        "viz_count": 5,
        "skip_measured_corner_compare": False,
        "corner_random_points": 120,
        "corner_worst_top_k": 25,
        "corner_grid_size": 55,
        "corner_tx_power_dbm": 43.0,
        "corner_bandwidth_hz": 100e6,
        "corner_disable_hotspots": False,
    },
}


def script_path(name: str) -> str:
    return str(ROOT / name)


def default_project_python() -> str:
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def run_command(cmd: List[str], cwd: Path, dry_run: bool = False) -> None:
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_copy_params(params_path: str, pipeline_dir: Path) -> str:
    if not params_path:
        return ""
    src = Path(params_path).resolve()
    dst = pipeline_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def status_payload(
    run_id: str,
    profile: str,
    project_python: str,
    current_step: str,
    step_index: int,
    step_count: int,
    completed_steps: List[str],
    failed_step: str = "",
    completed: bool = False,
    failed: bool = False,
) -> Dict:
    return {
        "run_id": run_id,
        "profile": profile,
        "subprocess_python": project_python,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_step": current_step,
        "step_index": int(step_index),
        "step_count": int(step_count),
        "completed_steps": list(completed_steps),
        "completed": bool(completed),
        "failed": bool(failed),
        "failed_step": failed_step,
    }


def resolve_value(args, name: str, profile: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return PROFILE_PRESETS[profile][name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click full simulation pipeline for the UAV GA/GAN project."
    )
    parser.add_argument("--params", default="", help="Path to JSON with user_params overrides")
    parser.add_argument("--run_id", default="", help="Output run folder id; defaults to run_YYYYMMDD_HHMMSS")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=tuple(PROFILE_PRESETS.keys()),
        help="Preset workload profile. Use quick for smoke test, balanced for normal one-click runs, full for the heaviest formal run.",
    )
    parser.add_argument(
        "--python",
        default="",
        help="Python interpreter used for subprocess steps; defaults to local .venv if present",
    )

    parser.add_argument("--ga_runs", type=int, default=None)
    parser.add_argument("--ga_maxgen", type=int, default=None)
    parser.add_argument("--ga_nind", type=int, default=None)
    parser.add_argument("--ga_seed", type=int, default=None)

    parser.add_argument("--gan_epochs", type=int, default=None)
    parser.add_argument("--gan_batch", type=int, default=None)
    parser.add_argument("--gan_latent", type=int, default=None)
    parser.add_argument("--gan_samples", type=int, default=None)
    parser.add_argument("--device", default=None)

    parser.add_argument("--random_n", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--viz_count", type=int, default=None)

    parser.add_argument("--skip_measured_corner_compare", action="store_true")
    parser.add_argument("--aerpaw_zip", default="")
    parser.add_argument("--dryad_zip", default="")
    parser.add_argument("--corner_random_points", type=int, default=None)
    parser.add_argument("--corner_worst_top_k", type=int, default=None)
    parser.add_argument("--corner_grid_size", type=int, default=None)
    parser.add_argument("--corner_tx_power_dbm", type=float, default=None)
    parser.add_argument("--corner_bandwidth_hz", type=float, default=None)
    parser.add_argument("--corner_disable_hotspots", action="store_true")

    parser.add_argument("--gen_bler_b", action="store_true", help="Optionally regenerate BLER_B via Sionna")
    parser.add_argument("--sionna_python", default="", help="python.exe with TF+Sionna installed")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_id = args.run_id.strip() or time.strftime("run_%Y%m%d_%H%M%S")
    run_root = ROOT / "output" / run_id
    pipeline_dir = run_root / "pipeline"
    ensure_dir(pipeline_dir)
    project_python = str(args.python).strip() or default_project_python()
    profile = str(args.profile).strip().lower()

    ga_runs = int(resolve_value(args, "ga_runs", profile))
    ga_maxgen = int(resolve_value(args, "ga_maxgen", profile))
    ga_nind = int(resolve_value(args, "ga_nind", profile))
    ga_seed = int(resolve_value(args, "ga_seed", profile))
    gan_epochs = int(resolve_value(args, "gan_epochs", profile))
    gan_batch = int(resolve_value(args, "gan_batch", profile))
    gan_latent = int(resolve_value(args, "gan_latent", profile))
    gan_samples = int(resolve_value(args, "gan_samples", profile))
    device = str(resolve_value(args, "device", profile))
    random_n = int(resolve_value(args, "random_n", profile))
    random_seed = int(resolve_value(args, "random_seed", profile))
    viz_count = int(resolve_value(args, "viz_count", profile))
    skip_measured_corner_compare = bool(args.skip_measured_corner_compare) or bool(PROFILE_PRESETS[profile]["skip_measured_corner_compare"])
    corner_random_points = int(resolve_value(args, "corner_random_points", profile))
    corner_worst_top_k = int(resolve_value(args, "corner_worst_top_k", profile))
    corner_grid_size = int(resolve_value(args, "corner_grid_size", profile))
    corner_tx_power_dbm = float(resolve_value(args, "corner_tx_power_dbm", profile))
    corner_bandwidth_hz = float(resolve_value(args, "corner_bandwidth_hz", profile))
    corner_disable_hotspots = bool(args.corner_disable_hotspots) or bool(PROFILE_PRESETS[profile]["corner_disable_hotspots"])

    copied_params = maybe_copy_params(str(args.params).strip(), pipeline_dir) if not args.dry_run else str(args.params).strip()

    config_payload = {
        "run_id": run_id,
        "profile": profile,
        "workspace": str(ROOT),
        "subprocess_python": project_python,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params_path": copied_params,
        "ga": {
            "runs": ga_runs,
            "maxgen": ga_maxgen,
            "nind": ga_nind,
            "seed": ga_seed,
        },
        "gan": {
            "epochs": gan_epochs,
            "batch": gan_batch,
            "latent": gan_latent,
            "samples": gan_samples,
            "device": device,
        },
        "compare": {
            "random_n": random_n,
            "random_seed": random_seed,
            "viz_count": viz_count,
            "skip_measured_corner_compare": skip_measured_corner_compare,
        },
        "measured_cornercases": {
            "aerpaw_zip": str(args.aerpaw_zip),
            "dryad_zip": str(args.dryad_zip),
            "random_points": corner_random_points,
            "worst_top_k": corner_worst_top_k,
            "grid_size": corner_grid_size,
            "tx_power_dbm": corner_tx_power_dbm,
            "bandwidth_hz": corner_bandwidth_hz,
            "disable_hotspots": corner_disable_hotspots,
        },
        "extra_evaluate": {
            "gen_bler_b": bool(args.gen_bler_b),
            "sionna_python": str(args.sionna_python),
        },
    }
    if not args.dry_run:
        write_json(pipeline_dir / "pipeline_config.json", config_payload)

    commands: List[Tuple[str, List[str]]] = []

    gan_cmd = [
        project_python,
        "-u",
        script_path("gan_uav_pipeline.py"),
        "--run_id",
        run_id,
        "--ga_runs",
        str(ga_runs),
        "--ga_maxgen",
        str(ga_maxgen),
        "--ga_nind",
        str(ga_nind),
        "--ga_seed",
        str(ga_seed),
        "--gan_epochs",
        str(gan_epochs),
        "--gan_batch",
        str(gan_batch),
        "--gan_latent",
        str(gan_latent),
        "--gan_samples",
        str(gan_samples),
        "--device",
        device,
    ]
    if copied_params:
        gan_cmd += ["--params", copied_params]
    commands.append(("gan_uav_pipeline", gan_cmd))

    compare_cmd = [
        project_python,
        "-u",
        script_path("compare_random_ga_gan.py"),
        "--run_id",
        run_id,
        "--random_n",
        str(random_n),
        "--random_seed",
        str(random_seed),
        "--viz_count",
        str(viz_count),
        "--corner_random_points",
        str(corner_random_points),
        "--corner_worst_top_k",
        str(corner_worst_top_k),
        "--corner_grid_size",
        str(corner_grid_size),
        "--corner_tx_power_dbm",
        str(corner_tx_power_dbm),
        "--corner_bandwidth_hz",
        str(corner_bandwidth_hz),
    ]
    if copied_params:
        compare_cmd += ["--params", copied_params]
    if skip_measured_corner_compare:
        compare_cmd += ["--skip_measured_corner_compare"]
    if args.aerpaw_zip:
        compare_cmd += ["--aerpaw_zip", str(args.aerpaw_zip)]
    if args.dryad_zip:
        compare_cmd += ["--dryad_zip", str(args.dryad_zip)]
    if corner_disable_hotspots:
        compare_cmd += ["--corner_disable_hotspots"]
    commands.append(("compare_random_ga_gan", compare_cmd))

    if args.gen_bler_b:
        evaluate_cmd = [
            project_python,
            "-u",
            script_path("evaluate.py"),
            "--run_id",
            run_id,
            "--random_n",
            str(random_n),
            "--random_seed",
            str(random_seed),
            "--out_dir",
            str(run_root / "compare" / "kpi"),
            "--gen_bler_b",
        ]
        if copied_params:
            evaluate_cmd += ["--params", copied_params]
        if args.sionna_python:
            evaluate_cmd += ["--sionna_python", str(args.sionna_python)]
        commands.append(("evaluate_bler_b", evaluate_cmd))

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    completed = False
    failed = False
    failed_step = ""
    completed_steps: List[str] = []
    step_count = len(commands)
    if not args.dry_run:
        write_json(
            pipeline_dir / "pipeline_status.json",
            status_payload(
                run_id=run_id,
                profile=profile,
                project_python=project_python,
                current_step="pending",
                step_index=0,
                step_count=step_count,
                completed_steps=[],
            ),
        )
    try:
        for idx, (step_name, cmd) in enumerate(commands, start=1):
            print(f"[PIPELINE] Step {idx}/{step_count}: {step_name}")
            if not args.dry_run:
                write_json(
                    pipeline_dir / "pipeline_status.json",
                    status_payload(
                        run_id=run_id,
                        profile=profile,
                        project_python=project_python,
                        current_step=step_name,
                        step_index=idx,
                        step_count=step_count,
                        completed_steps=completed_steps,
                    ),
                )
            run_command(cmd, ROOT, dry_run=bool(args.dry_run))
            completed_steps.append(step_name)
        completed = True
        return 0
    except Exception:
        failed = True
        if len(completed_steps) < step_count:
            failed_step = commands[len(completed_steps)][0]
        raise
    finally:
        manifest = {
            "run_id": run_id,
            "profile": profile,
            "workspace": str(ROOT),
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed": completed,
            "failed": failed,
            "failed_step": failed_step,
            "dry_run": bool(args.dry_run),
            "commands": [{"step": step_name, "cmd": " ".join(cmd)} for step_name, cmd in commands],
            "run_root": str(run_root),
            "key_outputs": {
                "gan_root": str(run_root / "gan"),
                "compare_root": str(run_root / "compare"),
                "ga_samples": str(run_root / "gan" / "arrays" / "ga_samples.npy"),
                "gan_samples": str(run_root / "gan" / "arrays" / "gan_samples.npy"),
                "comparison_summary": str(run_root / "compare" / "comparison_summary.json"),
                "kpi_report": str(run_root / "compare" / "kpi" / "kpi_report.json"),
                "measured_cornercases": str(run_root / "compare" / "measured_cornercases"),
                "pipeline_config": str(pipeline_dir / "pipeline_config.json"),
            },
        }
        if not args.dry_run:
            write_json(pipeline_dir / "pipeline_manifest.json", manifest)
            write_json(
                pipeline_dir / "pipeline_status.json",
                status_payload(
                    run_id=run_id,
                    profile=profile,
                    project_python=project_python,
                    current_step="done" if completed else failed_step or "failed",
                    step_index=len(completed_steps),
                    step_count=step_count,
                    completed_steps=completed_steps,
                    failed_step=failed_step,
                    completed=completed,
                    failed=failed,
                ),
            )
            if completed:
                print(f"Full pipeline completed successfully. Output root: output/{run_id}/")
            else:
                print(f"Full pipeline stopped before completion. Output root: output/{run_id}/")
                if failed_step:
                    print("Failed step:", failed_step)
            print("Pipeline manifest:", pipeline_dir / "pipeline_manifest.json")
        else:
            print(f"Dry run only. Planned output root: output/{run_id}/")


if __name__ == "__main__":
    raise SystemExit(main())
