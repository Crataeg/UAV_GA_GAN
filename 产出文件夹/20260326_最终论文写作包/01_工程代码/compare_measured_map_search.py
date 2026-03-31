import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import geatpy as ea
except Exception:
    ea = None

from measured_dataset_loaders import (
    DEFAULT_AERPAW_ZIP,
    DEFAULT_DRYAD_ZIP,
    DatasetBundle,
    MeasuredPoint,
    extract_dense_reference_points,
    load_aerpaw_bundle,
    load_dryad_bundle,
    metric_suffix,
    write_csv,
)
import gan_uav_pipeline as gan_pipeline
from kpi import summarize_distribution

LEGACY_AERPAW_ZIP = DEFAULT_AERPAW_ZIP
LEGACY_DRYAD_ZIP = DEFAULT_DRYAD_ZIP


def _default_zip_candidates(filename: str) -> Tuple[str, ...]:
    home = str(Path.home())
    return (
        os.path.join(home, "Downloads", filename),
        os.path.join("D:\\", "Downloads", filename),
        os.path.join("D:\\", "下载", filename),
        os.path.join("C:\\Users\\52834\\Downloads", filename),
    )


def _pick_existing_path(candidates: Sequence[str], fallback: str) -> str:
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return fallback


DEFAULT_AERPAW_ZIP = _pick_existing_path(
    _default_zip_candidates("aerpaw-dataset-24.zip"),
    LEGACY_AERPAW_ZIP,
)
DEFAULT_DRYAD_ZIP = _pick_existing_path(
    _default_zip_candidates("doi_10_5061_dryad_wh70rxx06__v20250521.zip"),
    LEGACY_DRYAD_ZIP,
)


@dataclass(frozen=True)
class MeasuredMapBundle:
    dataset_id: str
    family: str
    notes: str
    area_size_m: float
    shift_xy_m: Tuple[float, float]
    all_points: Tuple[MeasuredPoint, ...]
    selected_points: Tuple[MeasuredPoint, ...]
    poor_cloud_points: Tuple[MeasuredPoint, ...]
    spacing_ref_m: float
    low_quantile: float


def _safe_summary(values: Sequence[float], percentiles: Sequence[int] = (10, 50, 90)) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return summarize_distribution(arr.tolist(), percentiles=tuple(percentiles))


def _summarize_search_scores(scores: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(scores), dtype=float).reshape((-1,))
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _median_spacing(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] <= 1:
        return 25.0
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 25.0
    return float(max(np.median(finite), 5.0))


def _align_bundle(bundle: DatasetBundle, area_margin_m: float = 60.0, low_quantile: float = 0.10) -> MeasuredMapBundle:
    all_points = list(bundle.all_points)
    selected_points = list(bundle.selected_points)
    xs = np.asarray([p.x_local_m for p in all_points], dtype=float)
    ys = np.asarray([p.y_local_m for p in all_points], dtype=float)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    span_x = max_x - min_x
    span_y = max_y - min_y
    span = max(span_x, span_y, 100.0)
    area_size = float(span + 2.0 * area_margin_m)
    shift_x = float(area_margin_m - min_x + 0.5 * max(0.0, span - span_x))
    shift_y = float(area_margin_m - min_y + 0.5 * max(0.0, span - span_y))

    def shift_point(point: MeasuredPoint) -> MeasuredPoint:
        return replace(
            point,
            x_aligned_m=float(point.x_local_m + shift_x),
            y_aligned_m=float(point.y_local_m + shift_y),
        )

    aligned_all = tuple(shift_point(point) for point in all_points)
    aligned_selected = tuple(shift_point(point) for point in selected_points)
    poor_cloud = tuple(extract_dense_reference_points(aligned_all, quantile=float(low_quantile)))

    spacing = _median_spacing(np.asarray([[p.x_aligned_m, p.y_aligned_m] for p in aligned_all], dtype=float))
    notes = (
        "Measured-map-driven comparison. The open dataset is treated as a measured field only. "
        "No explicit serving base station, hotspot, or fictional interferer reconstruction is used "
        "for visualization or optimization."
    )
    return MeasuredMapBundle(
        dataset_id=bundle.dataset_id,
        family=bundle.family,
        notes=notes,
        area_size_m=area_size,
        shift_xy_m=(shift_x, shift_y),
        all_points=aligned_all,
        selected_points=aligned_selected,
        poor_cloud_points=poor_cloud,
        spacing_ref_m=spacing,
        low_quantile=float(low_quantile),
    )


class MeasuredRiskField:
    def __init__(self, bundle: MeasuredMapBundle):
        self.bundle = bundle
        self.area_size_m = float(bundle.area_size_m)
        self.spacing_ref_m = float(bundle.spacing_ref_m)
        self.conf_scale_m = float(max(self.spacing_ref_m * 3.0, 20.0))
        self.score_scales = {"sinr": 0.5, "throughput": 0.5}
        self.metric_fields: Dict[str, Dict[str, object]] = {}
        self.all_xy = np.asarray([[p.x_aligned_m, p.y_aligned_m] for p in bundle.all_points], dtype=float)
        self.poor_xy = (
            np.asarray([[p.x_aligned_m, p.y_aligned_m] for p in bundle.poor_cloud_points], dtype=float)
            if bundle.poor_cloud_points
            else np.zeros((0, 2), dtype=float)
        )

        for suffix in ("sinr", "throughput"):
            subset = [p for p in bundle.all_points if metric_suffix(p.metric_name) == suffix]
            if not subset:
                continue
            xy = np.asarray([[p.x_aligned_m, p.y_aligned_m] for p in subset], dtype=float)
            values = np.asarray([p.raw_metric_value for p in subset], dtype=float)
            self.metric_fields[suffix] = {
                "points": tuple(subset),
                "xy": xy,
                "values": values,
                "low_ref": float(np.percentile(values, 10)),
                "high_ref": float(np.percentile(values, 90)),
            }

        active = [key for key in self.score_scales if key in self.metric_fields]
        if active:
            weight = 1.0 / float(len(active))
            self.score_scales = {key: (weight if key in active else 0.0) for key in self.score_scales}

    def _idw_predict(
        self,
        train_xy: np.ndarray,
        train_values: np.ndarray,
        query_xy: np.ndarray,
        k: int = 12,
        power: float = 1.7,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        diff = query_xy[:, None, :] - train_xy[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        if train_xy.shape[0] > int(k):
            idx = np.argpartition(dist, kth=int(k) - 1, axis=1)[:, : int(k)]
            local_dist = np.take_along_axis(dist, idx, axis=1)
            local_vals = train_values[idx]
        else:
            local_dist = dist
            local_vals = np.broadcast_to(train_values[None, :], dist.shape)

        exact = local_dist <= 1e-9
        weights = 1.0 / np.maximum(local_dist, 1.0) ** float(power)
        pred = np.sum(weights * local_vals, axis=1) / np.maximum(np.sum(weights, axis=1), 1e-12)
        if np.any(exact):
            rows = np.where(np.any(exact, axis=1))[0]
            for row in rows:
                pred[row] = float(local_vals[row][exact[row]][0])

        nearest = np.min(local_dist, axis=1)
        confidence = np.exp(-nearest / float(self.conf_scale_m))
        return pred, nearest, confidence

    def evaluate_xy(self, query_xy: np.ndarray) -> List[Dict[str, float]]:
        xy = np.asarray(query_xy, dtype=float).reshape((-1, 2))
        global_nearest = np.min(np.linalg.norm(xy[:, None, :] - self.all_xy[None, :, :], axis=2), axis=1)
        global_conf = np.exp(-global_nearest / float(self.conf_scale_m))

        if self.poor_xy.size:
            nearest_poor = np.min(np.linalg.norm(xy[:, None, :] - self.poor_xy[None, :, :], axis=2), axis=1)
        else:
            nearest_poor = np.full((xy.shape[0],), np.inf, dtype=float)
        poor_proximity = np.exp(-np.minimum(nearest_poor, 10.0 * self.conf_scale_m) / float(self.conf_scale_m))

        predicted: Dict[str, np.ndarray] = {}
        risks: Dict[str, np.ndarray] = {}
        metric_confidences: Dict[str, np.ndarray] = {}
        for suffix, field in self.metric_fields.items():
            pred, _, conf = self._idw_predict(
                np.asarray(field["xy"], dtype=float),
                np.asarray(field["values"], dtype=float),
                xy,
            )
            low_ref = float(field["low_ref"])
            high_ref = float(field["high_ref"])
            span = max(high_ref - low_ref, 1e-6)
            risk = np.clip((high_ref - pred) / span, 0.0, 1.5)
            predicted[suffix] = pred
            risks[suffix] = risk
            metric_confidences[suffix] = conf

        combined_metric_risk = np.zeros((xy.shape[0],), dtype=float)
        combined_conf = np.zeros((xy.shape[0],), dtype=float)
        for suffix, weight in self.score_scales.items():
            if suffix not in risks or weight <= 0.0:
                continue
            combined_metric_risk += float(weight) * risks[suffix]
            combined_conf += float(weight) * metric_confidences[suffix]
        if not np.any(combined_conf):
            combined_conf = global_conf.copy()

        field_score = global_conf * (0.85 * combined_metric_risk + 0.15 * poor_proximity)

        rows: List[Dict[str, float]] = []
        for idx in range(xy.shape[0]):
            rows.append(
                {
                    "x_m": float(xy[idx, 0]),
                    "y_m": float(xy[idx, 1]),
                    "field_score": float(field_score[idx]),
                    "combined_metric_risk": float(combined_metric_risk[idx]),
                    "global_confidence": float(global_conf[idx]),
                    "metric_confidence": float(combined_conf[idx]),
                    "poor_proximity": float(poor_proximity[idx]),
                    "nearest_measured_dist_m": float(global_nearest[idx]),
                    "nearest_poor_dist_m": float(nearest_poor[idx]),
                    "predicted_sinr_db": float(predicted["sinr"][idx]) if "sinr" in predicted else float("nan"),
                    "predicted_throughput_mbps": float(predicted["throughput"][idx]) if "throughput" in predicted else float("nan"),
                }
            )
        return rows

    def score_array(self, samples_xy: np.ndarray) -> np.ndarray:
        rows = self.evaluate_xy(samples_xy)
        return np.asarray([float(item["field_score"]) for item in rows], dtype=float)


class MeasuredMapProblem(ea.Problem if ea is not None else object):
    def __init__(self, field: MeasuredRiskField):
        if ea is None:
            raise RuntimeError("geatpy is required for measured-map GA search")
        self.field = field
        self.num_drones = 0
        self.num_interference = 0
        super().__init__(
            "measured_map_worst_region_search",
            1,
            [-1],
            2,
            [0, 0],
            [0.0, 0.0],
            [float(field.area_size_m), float(field.area_size_m)],
        )

    def evalVars(self, Vars: np.ndarray):
        scores = self.field.score_array(np.asarray(Vars, dtype=float)).reshape((-1, 1))
        return scores, None


def _random_candidate_samples(bounds: Tuple[np.ndarray, np.ndarray], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lb = np.asarray(bounds[0], dtype=float)
    ub = np.asarray(bounds[1], dtype=float)
    return rng.uniform(lb, ub, size=(int(n), lb.size))


def _fallback_ga_samples(
    field: MeasuredRiskField,
    bounds: Tuple[np.ndarray, np.ndarray],
    runs: int,
    maxgen: int,
    nind: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lb = np.asarray(bounds[0], dtype=float)
    ub = np.asarray(bounds[1], dtype=float)
    area_size = float(ub[0] - lb[0])
    samples: List[np.ndarray] = []

    for run_idx in range(int(runs)):
        pop = rng.uniform(lb, ub, size=(int(nind), lb.size))
        best_x = np.asarray(pop[0], dtype=float)
        best_score = -np.inf
        sigma0 = max(area_size * 0.18, 20.0)

        for gen_idx in range(max(int(maxgen), 1)):
            scores = field.score_array(pop)
            elite_n = max(2, int(nind) // 4)
            elite_idx = np.argsort(scores)[-elite_n:]
            elites = np.asarray(pop[elite_idx], dtype=float)
            elite_scores = np.asarray(scores[elite_idx], dtype=float)
            if float(elite_scores[-1]) > float(best_score):
                best_score = float(elite_scores[-1])
                best_x = np.asarray(elites[-1], dtype=float)
            if gen_idx == int(maxgen) - 1:
                break
            sigma = sigma0 * (0.85 ** float(gen_idx))
            parents = elites[rng.integers(0, elite_n, size=int(nind))]
            children = parents + rng.normal(0.0, sigma, size=parents.shape)
            children = np.clip(children, lb, ub)
            children[:elite_n] = elites
            pop = children

        samples.append(np.asarray(best_x, dtype=float))
        rng = np.random.default_rng(int(seed) + int(run_idx) + 1)

    return np.asarray(samples, dtype=float)


def _fallback_generative_samples(
    seed_samples: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    n_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lb = np.asarray(bounds[0], dtype=float)
    ub = np.asarray(bounds[1], dtype=float)
    if seed_samples.size == 0:
        return rng.uniform(lb, ub, size=(int(n_samples), lb.size))

    seeds = np.asarray(seed_samples, dtype=float)
    scales = np.std(seeds, axis=0)
    min_scale = np.maximum((ub - lb) * 0.05, 8.0)
    scales = np.maximum(scales, min_scale)

    base = seeds[rng.integers(0, seeds.shape[0], size=int(n_samples))]
    noise = rng.normal(0.0, scales, size=base.shape)
    out = np.clip(base + noise, lb, ub)
    return np.asarray(out, dtype=float)


def _run_search_groups(
    field: MeasuredRiskField,
    seed: int,
    ga_runs: int,
    ga_maxgen: int,
    ga_nind: int,
    gan_epochs: int,
    gan_batch: int,
    gan_latent: int,
    gan_samples_n: int,
    random_search_n: int,
    device: str,
) -> Dict[str, Dict[str, object]]:
    bounds = (
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([float(field.area_size_m), float(field.area_size_m)], dtype=float),
    )
    empty_idx = np.asarray([], dtype=int)

    random_samples = _random_candidate_samples(bounds, random_search_n, seed + 7919)
    random_scores = field.score_array(random_samples)
    random_best_idx = int(np.argmax(random_scores)) if random_scores.size else -1
    out: Dict[str, Dict[str, object]] = {
        "random": {
            "samples": random_samples,
            "scores": random_scores,
            "best_x": np.asarray(random_samples[random_best_idx], dtype=float) if random_best_idx >= 0 else np.zeros((2,), dtype=float),
            "best_score": float(random_scores[random_best_idx]) if random_best_idx >= 0 else 0.0,
            "score_stats": _summarize_search_scores(random_scores.tolist()),
        }
    }

    ga_fallback = ea is None
    if ea is not None:
        try:
            problem = MeasuredMapProblem(field)
            ga_samples = gan_pipeline.run_ga_samples(problem, ga_runs, ga_maxgen, ga_nind, seed)
        except Exception:
            ga_fallback = True
            ga_samples = _fallback_ga_samples(field, bounds, ga_runs, ga_maxgen, ga_nind, seed + 12345)
    else:
        ga_samples = _fallback_ga_samples(field, bounds, ga_runs, ga_maxgen, ga_nind, seed + 12345)
    ga_scores = field.score_array(ga_samples)
    ga_best_idx = int(np.argmax(ga_scores)) if ga_scores.size else -1
    out["ga"] = {
        "samples": ga_samples,
        "scores": ga_scores,
        "best_x": np.asarray(ga_samples[ga_best_idx], dtype=float) if ga_best_idx >= 0 else np.zeros((2,), dtype=float),
        "best_score": float(ga_scores[ga_best_idx]) if ga_best_idx >= 0 else 0.0,
        "score_stats": _summarize_search_scores(ga_scores.tolist()),
        "search_note": "fallback_evolutionary_search" if ga_fallback else "geatpy_ga",
    }

    lb = np.asarray(bounds[0], dtype=float)
    ub = np.asarray(bounds[1], dtype=float)
    if gan_pipeline.torch is not None and ga_samples.size > 0:
        gen, gan_log = gan_pipeline.train_gan(
            ga_samples,
            lb,
            ub,
            int(gan_epochs),
            int(gan_batch),
            int(gan_latent),
            str(device),
        )
        gan_samples = gan_pipeline.generate_gan_samples(
            gen,
            int(gan_samples_n),
            int(gan_latent),
            lb,
            ub,
            empty_idx,
            empty_idx,
            str(device),
        )
        gan_samples = np.clip(np.asarray(gan_samples, dtype=float), lb, ub)
        gan_scores = field.score_array(gan_samples)
        gan_best_idx = int(np.argmax(gan_scores)) if gan_scores.size else -1
        out["gan"] = {
            "samples": gan_samples,
            "scores": gan_scores,
            "best_x": np.asarray(gan_samples[gan_best_idx], dtype=float) if gan_best_idx >= 0 else np.zeros((2,), dtype=float),
            "best_score": float(gan_scores[gan_best_idx]) if gan_best_idx >= 0 else 0.0,
            "score_stats": _summarize_search_scores(gan_scores.tolist()),
            "gan_training": gan_log,
        }
    else:
        gan_samples = _fallback_generative_samples(ga_samples, bounds, gan_samples_n, seed + 24680)
        gan_scores = field.score_array(gan_samples)
        gan_best_idx = int(np.argmax(gan_scores)) if gan_scores.size else -1
        out["gan"] = {
            "samples": gan_samples,
            "scores": gan_scores,
            "best_x": np.asarray(gan_samples[gan_best_idx], dtype=float) if gan_best_idx >= 0 else np.zeros((2,), dtype=float),
            "best_score": float(gan_scores[gan_best_idx]) if gan_best_idx >= 0 else 0.0,
            "score_stats": _summarize_search_scores(gan_scores.tolist()),
            "gan_training": {"mode": "fallback_gaussian_resample"},
        }
    return out


def _evaluate_grid(
    field: MeasuredRiskField,
    area_size_m: float,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    xs = np.linspace(0.0, float(area_size_m), int(grid_size))
    ys = np.linspace(0.0, float(area_size_m), int(grid_size))
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    samples = np.column_stack([mesh_x.reshape((-1,)), mesh_y.reshape((-1,))])
    rows = field.evaluate_xy(samples)
    score_map = np.asarray([float(item["field_score"]) for item in rows], dtype=float).reshape(mesh_x.shape)
    return xs, ys, score_map, rows


def _worst_region_from_grid(grid_rows: Sequence[Mapping[str, float]], top_k: int) -> List[Dict[str, float]]:
    ranked = sorted(grid_rows, key=lambda item: float(item["field_score"]), reverse=True)
    return list(ranked[: int(top_k)])


def _measured_rows(points: Sequence[MeasuredPoint], field_rows: Sequence[Mapping[str, float]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for point, field_row in zip(points, field_rows):
        row = asdict(point)
        row.update({key: field_row[key] for key in field_row})
        rows.append(row)
    return rows


def _group_summary(rows: Sequence[Mapping[str, float]]) -> Dict[str, object]:
    field_scores = [float(item["field_score"]) for item in rows]
    sinr_values = [float(item["predicted_sinr_db"]) for item in rows if np.isfinite(float(item["predicted_sinr_db"]))]
    thr_values = [
        float(item["predicted_throughput_mbps"])
        for item in rows
        if np.isfinite(float(item["predicted_throughput_mbps"]))
    ]
    poor_dist = [float(item["nearest_poor_dist_m"]) for item in rows if np.isfinite(float(item["nearest_poor_dist_m"]))]
    conf_values = [float(item["global_confidence"]) for item in rows]
    return {
        "count": int(len(rows)),
        "field_score": _safe_summary(field_scores),
        "predicted_sinr_db": _safe_summary(sinr_values),
        "predicted_throughput_mbps": _safe_summary(thr_values),
        "nearest_poor_dist_m": _safe_summary(poor_dist),
        "global_confidence": _safe_summary(conf_values),
        "hit_rate_15m": float(np.mean([dist <= 15.0 for dist in poor_dist])) if poor_dist else 0.0,
        "hit_rate_25m": float(np.mean([dist <= 25.0 for dist in poor_dist])) if poor_dist else 0.0,
        "hit_rate_40m": float(np.mean([dist <= 40.0 for dist in poor_dist])) if poor_dist else 0.0,
    }


def _top_rows(rows: Sequence[Mapping[str, float]], n: int) -> List[Mapping[str, float]]:
    ranked = sorted(rows, key=lambda item: float(item["field_score"]), reverse=True)
    return list(ranked[: int(n)])


def _plot_field_heatmap(
    bundle: MeasuredMapBundle,
    field: MeasuredRiskField,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    score_map: np.ndarray,
    overlay_groups: Mapping[str, Sequence[Mapping[str, float]]],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    extent = [float(grid_x[0]), float(grid_x[-1]), float(grid_y[0]), float(grid_y[-1])]
    img = ax.imshow(score_map, origin="lower", extent=extent, cmap="magma", aspect="auto")
    plt.colorbar(img, ax=ax, label="Measured-field risk score")

    ax.scatter(
        [p.x_aligned_m for p in bundle.all_points],
        [p.y_aligned_m for p in bundle.all_points],
        s=8,
        c="#D9D9D9",
        alpha=0.25,
        label="Measured samples",
    )
    if bundle.poor_cloud_points:
        ax.scatter(
            [p.x_aligned_m for p in bundle.poor_cloud_points],
            [p.y_aligned_m for p in bundle.poor_cloud_points],
            s=18,
            c="#111111",
            alpha=0.55,
            label="Measured poor cloud",
        )
    if bundle.selected_points:
        ax.scatter(
            [p.x_aligned_m for p in bundle.selected_points],
            [p.y_aligned_m for p in bundle.selected_points],
            s=55,
            c="#FFFFFF",
            edgecolors="#000000",
            linewidths=0.8,
            label="Measured anchors",
        )
        for point in bundle.selected_points:
            if point.selection_id:
                ax.annotate(point.selection_id, (point.x_aligned_m, point.y_aligned_m), textcoords="offset points", xytext=(4, 3), fontsize=8)

    styles = {
        "random": {"marker": "o", "color": "#2CA02C", "label": "Random"},
        "ga": {"marker": "D", "color": "#D62728", "label": "GA"},
        "gan": {"marker": "s", "color": "#1F77B4", "label": "GAN"},
        "field_worst_region": {"marker": "x", "color": "#FFD700", "label": "Field worst grid"},
    }
    for key, rows in overlay_groups.items():
        if not rows or key not in styles:
            continue
        style = styles[key]
        top_rows = _top_rows(rows, 12)
        ax.scatter(
            [float(item["x_m"]) for item in top_rows],
            [float(item["y_m"]) for item in top_rows],
            s=70 if key != "field_worst_region" else 28,
            marker=style["marker"],
            c=style["color"],
            linewidths=0.9,
            alpha=0.85,
            label=style["label"],
        )

    ax.set_xlim(0.0, float(bundle.area_size_m))
    ax.set_ylim(0.0, float(bundle.area_size_m))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{bundle.dataset_id.upper()}: measured-map-driven search")
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_group_comparison(
    dataset_id: str,
    group_rows: Mapping[str, Sequence[Mapping[str, float]]],
    out_path: str,
) -> None:
    groups = {key: rows for key, rows in group_rows.items() if rows}
    labels = list(groups.keys())
    if not labels:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    score_data = [[float(item["field_score"]) for item in groups[label]] for label in labels]
    sinr_data = [
        [float(item["predicted_sinr_db"]) for item in groups[label] if np.isfinite(float(item["predicted_sinr_db"]))]
        for label in labels
    ]
    thr_data = [
        [float(item["predicted_throughput_mbps"]) for item in groups[label] if np.isfinite(float(item["predicted_throughput_mbps"]))]
        for label in labels
    ]
    dist_data = [
        [float(item["nearest_poor_dist_m"]) for item in groups[label] if np.isfinite(float(item["nearest_poor_dist_m"]))]
        for label in labels
    ]

    axes[0, 0].boxplot(score_data, labels=labels, showfliers=False)
    axes[0, 0].set_title("Measured-field risk score")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].boxplot(sinr_data, labels=labels, showfliers=False)
    axes[0, 1].set_title("Predicted SINR (dB)")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    axes[1, 0].boxplot(thr_data, labels=labels, showfliers=False)
    axes[1, 0].set_title("Predicted throughput (Mbps)")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    axes[1, 1].boxplot(dist_data, labels=labels, showfliers=False)
    axes[1, 1].set_title("Distance to measured poor cloud (m)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{dataset_id.upper()}: measured-map group comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _dataset_report(
    bundle: DatasetBundle,
    map_bundle: MeasuredMapBundle,
    search_groups: Mapping[str, Dict[str, object]],
    group_rows: Mapping[str, Sequence[Mapping[str, float]]],
) -> Dict[str, object]:
    return {
        "dataset_id": bundle.dataset_id,
        "family": bundle.family,
        "methodology": {
            "mode": "measured_map_driven",
            "notes": map_bundle.notes,
            "low_quantile": float(map_bundle.low_quantile),
            "spacing_ref_m": float(map_bundle.spacing_ref_m),
            "shift_xy_m": [float(map_bundle.shift_xy_m[0]), float(map_bundle.shift_xy_m[1])],
            "area_size_m": float(map_bundle.area_size_m),
            "no_explicit_bs_or_interferer_reconstruction": True,
        },
        "search_groups": {
            key: {
                "best_score": float(item.get("best_score", 0.0)),
                "score_stats": dict(item.get("score_stats", {})),
                "num_samples": int(len(item.get("samples", []))),
                "search_note": item.get("search_note", ""),
                "gan_training": dict(item.get("gan_training", {})) if isinstance(item.get("gan_training", {}), dict) else {},
            }
            for key, item in search_groups.items()
        },
        "groups": {key: _group_summary(rows) for key, rows in group_rows.items()},
    }


def run_comparison(
    aerpaw_zip: str,
    dryad_zip: str,
    out_dir: str,
    random_points: int,
    worst_top_k: int,
    grid_size: int,
    tx_power_dbm: float,
    bandwidth_hz: float,
    use_empirical_hotspots: bool,
    seed: int,
    ga_runs: int = 40,
    ga_maxgen: int = 40,
    ga_nind: int = 30,
    gan_epochs: int = 300,
    gan_batch: int = 32,
    gan_latent: int = 32,
    gan_samples_n: int = 20,
    random_search_n: int = 40,
    device: str = "cpu",
) -> Dict[str, object]:
    del tx_power_dbm
    del bandwidth_hz
    del use_empirical_hotspots

    os.makedirs(out_dir, exist_ok=True)
    bundles = (load_aerpaw_bundle(aerpaw_zip), load_dryad_bundle(dryad_zip))
    overall_summary: Dict[str, object] = {
        "reference_files": {
            "aerpaw_zip": aerpaw_zip,
            "dryad_zip": dryad_zip,
        },
        "datasets": {},
    }

    for offset, bundle in enumerate(bundles):
        dataset_out_dir = os.path.join(out_dir, bundle.dataset_id)
        os.makedirs(dataset_out_dir, exist_ok=True)

        map_bundle = _align_bundle(bundle, area_margin_m=60.0, low_quantile=0.10)
        field = MeasuredRiskField(map_bundle)
        search_groups = _run_search_groups(
            field=field,
            seed=int(seed + offset * 1000),
            ga_runs=int(ga_runs),
            ga_maxgen=int(ga_maxgen),
            ga_nind=int(ga_nind),
            gan_epochs=int(gan_epochs),
            gan_batch=int(gan_batch),
            gan_latent=int(gan_latent),
            gan_samples_n=int(gan_samples_n),
            random_search_n=int(random_points if int(random_points) > 0 else random_search_n),
            device=str(device),
        )

        poor_cloud_rows = field.evaluate_xy(np.asarray([[p.x_aligned_m, p.y_aligned_m] for p in map_bundle.poor_cloud_points], dtype=float))
        selected_rows = field.evaluate_xy(np.asarray([[p.x_aligned_m, p.y_aligned_m] for p in map_bundle.selected_points], dtype=float))
        grid_x, grid_y, score_map, grid_rows = _evaluate_grid(field, map_bundle.area_size_m, grid_size)
        field_worst_rows = _worst_region_from_grid(grid_rows, worst_top_k)
        random_rows = field.evaluate_xy(np.asarray(search_groups["random"]["samples"], dtype=float))
        ga_rows = field.evaluate_xy(np.asarray(search_groups["ga"]["samples"], dtype=float))
        gan_rows: List[Dict[str, float]] = []
        if "gan" in search_groups:
            gan_rows = field.evaluate_xy(np.asarray(search_groups["gan"]["samples"], dtype=float))

        group_rows = {
            "measured_poor_cloud": poor_cloud_rows,
            "measured_selected": selected_rows,
            "random": random_rows,
            "ga": ga_rows,
            "gan": gan_rows,
            "field_worst_region": field_worst_rows,
        }

        write_csv(os.path.join(dataset_out_dir, "measured_poor_cloud.csv"), _measured_rows(map_bundle.poor_cloud_points, poor_cloud_rows))
        write_csv(os.path.join(dataset_out_dir, "measured_selected_points.csv"), _measured_rows(map_bundle.selected_points, selected_rows))
        write_csv(os.path.join(dataset_out_dir, "random_samples_scored.csv"), random_rows)
        write_csv(os.path.join(dataset_out_dir, "ga_samples_scored.csv"), ga_rows)
        if gan_rows:
            write_csv(os.path.join(dataset_out_dir, "gan_samples_scored.csv"), gan_rows)
        write_csv(os.path.join(dataset_out_dir, "field_worst_region.csv"), field_worst_rows)
        write_csv(os.path.join(dataset_out_dir, "grid_map.csv"), grid_rows)

        _plot_field_heatmap(
            bundle=map_bundle,
            field=field,
            grid_x=grid_x,
            grid_y=grid_y,
            score_map=score_map,
            overlay_groups={
                "random": random_rows,
                "ga": ga_rows,
                "gan": gan_rows,
                "field_worst_region": field_worst_rows,
            },
            out_path=os.path.join(dataset_out_dir, f"{bundle.dataset_id}_measured_map_search.png"),
        )
        _plot_group_comparison(
            dataset_id=bundle.dataset_id,
            group_rows=group_rows,
            out_path=os.path.join(dataset_out_dir, f"{bundle.dataset_id}_measured_map_compare.png"),
        )

        report = _dataset_report(bundle, map_bundle, search_groups, group_rows)
        with open(os.path.join(dataset_out_dir, "measured_map_report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        overall_summary["datasets"][bundle.dataset_id] = report

    with open(os.path.join(out_dir, "measured_map_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, ensure_ascii=False, indent=2)
    return overall_summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measured-map-driven GA/GAN/random comparison on AERPAW and Dryad datasets.")
    parser.add_argument("--aerpaw_zip", default=DEFAULT_AERPAW_ZIP)
    parser.add_argument("--dryad_zip", default=DEFAULT_DRYAD_ZIP)
    parser.add_argument("--out_dir", default=os.path.join("output", "measured_map_compare"))
    parser.add_argument("--random_points", type=int, default=120)
    parser.add_argument("--worst_top_k", type=int, default=25)
    parser.add_argument("--grid_size", type=int, default=55)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--ga_runs", type=int, default=40)
    parser.add_argument("--ga_maxgen", type=int, default=40)
    parser.add_argument("--ga_nind", type=int, default=30)
    parser.add_argument("--gan_epochs", type=int, default=300)
    parser.add_argument("--gan_batch", type=int, default=32)
    parser.add_argument("--gan_latent", type=int, default=32)
    parser.add_argument("--gan_samples_n", type=int, default=20)
    parser.add_argument("--random_search_n", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    summary = run_comparison(
        aerpaw_zip=str(args.aerpaw_zip),
        dryad_zip=str(args.dryad_zip),
        out_dir=str(args.out_dir),
        random_points=int(args.random_points),
        worst_top_k=int(args.worst_top_k),
        grid_size=int(args.grid_size),
        tx_power_dbm=0.0,
        bandwidth_hz=0.0,
        use_empirical_hotspots=False,
        seed=int(args.seed),
        ga_runs=int(args.ga_runs),
        ga_maxgen=int(args.ga_maxgen),
        ga_nind=int(args.ga_nind),
        gan_epochs=int(args.gan_epochs),
        gan_batch=int(args.gan_batch),
        gan_latent=int(args.gan_latent),
        gan_samples_n=int(args.gan_samples_n),
        random_search_n=int(args.random_search_n),
        device=str(args.device),
    )
    print(json.dumps({"out_dir": args.out_dir, "datasets": list(summary.get("datasets", {}).keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
