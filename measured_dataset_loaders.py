import csv
import io
import math
import os
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


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
    r"d:\Downloads\aerpaw-dataset-24.zip",
)
DEFAULT_DRYAD_ZIP = _pick_existing_path(
    _default_zip_candidates("doi_10_5061_dryad_wh70rxx06__v20250521.zip"),
    r"d:\Downloads\doi_10_5061_dryad_wh70rxx06__v20250521.zip",
)

AERPAW_NR_FILES = (
    "aerpaw-dataset-24/Logs/pawprints_5G_NR_altitude30m_flight1.csv",
    "aerpaw-dataset-24/Logs/pawprints_5G_NR_altitude30m_flight2.csv",
)
AERPAW_THR_FILES = ("aerpaw-dataset-24/Logs/pawprints_iperf_throughput_altitude30m_flight1.csv",)
DRYAD_SINR_FILES = (
    "Dryad/yaw45/inputf1_sinr_with_header.csv",
    "Dryad/yaw315/inputf1_sinr_with_header.csv",
)
DRYAD_THR_FILES = (
    "Dryad/yaw45/input_throughput_with_header.csv",
    "Dryad/yaw315/input_throughput_with_header.csv",
)


@dataclass(frozen=True)
class MeasuredPoint:
    dataset_id: str
    track_id: str
    selection_id: str
    selection_role: str
    source_zip_path: str
    internal_path: str
    row_number: int
    metric_name: str
    raw_metric_value: float
    longitude: float
    latitude: float
    altitude_m: float
    x_local_m: float
    y_local_m: float
    z_local_m: float
    x_aligned_m: float = 0.0
    y_aligned_m: float = 0.0
    bs_distance_m: float = float("nan")
    bs_bearing_deg: float = float("nan")
    bs_elevation_deg: float = float("nan")
    note: str = ""


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    family: str
    reference_altitude_m: float
    station_height_m: float
    comm_freq_hz: float
    notes: str
    all_points: Tuple[MeasuredPoint, ...]
    selected_points: Tuple[MeasuredPoint, ...]
    station_xy_local_m: Tuple[float, float]


def metric_suffix(metric_name: str) -> str:
    return "sinr" if "sinr" in str(metric_name).lower() else "throughput"


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def csv_rows_from_zip(zip_path: str, internal_path: str) -> Iterable[Tuple[int, Mapping[str, str]]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(internal_path) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            for idx, row in enumerate(reader, start=2):
                yield idx, row


def csv_rows_from_nested_zip(outer_zip_path: str, nested_zip_name: str, internal_path: str) -> Iterable[Tuple[int, Mapping[str, str]]]:
    with zipfile.ZipFile(outer_zip_path) as outer:
        data = io.BytesIO(outer.read(nested_zip_name))
    with zipfile.ZipFile(data) as inner:
        with inner.open(internal_path) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            for idx, row in enumerate(reader, start=2):
                yield idx, row


def bearing_distance_to_local_xy(distance_m: float, bearing_deg: float, elevation_deg: float) -> Tuple[float, float]:
    horizontal = float(distance_m) * math.cos(math.radians(float(elevation_deg)))
    bearing_rad = math.radians(float(bearing_deg))
    x = horizontal * math.sin(bearing_rad)
    y = horizontal * math.cos(bearing_rad)
    return float(x), float(y)


def latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    r_earth = 6371000.0
    lat_rad = np.deg2rad(np.asarray(lat, dtype=float))
    lon_rad = np.deg2rad(np.asarray(lon, dtype=float))
    lat0_rad = math.radians(float(lat0))
    lon0_rad = math.radians(float(lon0))
    x = r_earth * (lon_rad - lon0_rad) * math.cos(lat0_rad)
    y = r_earth * (lat_rad - math.radians(float(lat0)))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def spatial_distance(a: MeasuredPoint, b: MeasuredPoint) -> float:
    return float(math.hypot(float(a.x_local_m) - float(b.x_local_m), float(a.y_local_m) - float(b.y_local_m)))


def pick_unique(
    points: Sequence[MeasuredPoint],
    sort_key,
    reverse: bool,
    min_separation_m: float,
    existing: Optional[Sequence[MeasuredPoint]] = None,
    limit: int = 1,
) -> List[MeasuredPoint]:
    chosen: List[MeasuredPoint] = []
    occupied = list(existing or [])
    ranked = sorted(points, key=sort_key, reverse=reverse)
    for point in ranked:
        ok = True
        for prev in occupied:
            if spatial_distance(point, prev) < float(min_separation_m):
                ok = False
                break
        if ok:
            chosen.append(point)
            occupied.append(point)
        if len(chosen) >= int(limit):
            break
    return chosen


def extract_dense_reference_points(points: Sequence[MeasuredPoint], quantile: float = 0.1) -> List[MeasuredPoint]:
    dense: List[MeasuredPoint] = []
    for suffix in ("sinr", "throughput"):
        subset = [p for p in points if metric_suffix(p.metric_name) == suffix]
        if not subset:
            continue
        values = np.asarray([p.raw_metric_value for p in subset], dtype=float)
        threshold = float(np.quantile(values, float(quantile)))
        dense.extend([p for p in subset if p.raw_metric_value <= threshold])
    return dense


def infer_aerpaw_station_height(zip_path: str) -> float:
    heights: List[float] = []
    for internal_path in AERPAW_NR_FILES:
        for _, row in csv_rows_from_zip(zip_path, internal_path):
            alt = safe_float(row.get("altitude"))
            dist = safe_float(row.get("bs_distance"))
            elev = safe_float(row.get("bs_elevation"))
            if not np.isfinite(alt) or not np.isfinite(dist) or not np.isfinite(elev):
                continue
            horizontal = dist * math.cos(math.radians(elev))
            station_height = alt - horizontal * math.tan(math.radians(elev))
            if np.isfinite(station_height):
                heights.append(float(station_height))
    if not heights:
        return 12.0
    return float(np.median(np.asarray(heights, dtype=float)))


def load_aerpaw_bundle(zip_path: str) -> DatasetBundle:
    station_height_m = infer_aerpaw_station_height(zip_path)
    points: List[MeasuredPoint] = []

    for internal_path in AERPAW_NR_FILES:
        track_id = Path(internal_path).stem
        for row_number, row in csv_rows_from_zip(zip_path, internal_path):
            metric = safe_float(row.get("ss_sinr"))
            dist = safe_float(row.get("bs_distance"))
            bearing = safe_float(row.get("bs_bearing"))
            elev = safe_float(row.get("bs_elevation"))
            if not np.isfinite(metric) or not np.isfinite(dist) or not np.isfinite(bearing):
                continue
            x_m, y_m = bearing_distance_to_local_xy(dist, bearing, elev if np.isfinite(elev) else 0.0)
            points.append(
                MeasuredPoint(
                    dataset_id="aerpaw",
                    track_id=track_id,
                    selection_id="",
                    selection_role="candidate",
                    source_zip_path=zip_path,
                    internal_path=internal_path,
                    row_number=row_number,
                    metric_name="measured_sinr_db",
                    raw_metric_value=float(metric),
                    longitude=safe_float(row.get("longitude")),
                    latitude=safe_float(row.get("latitude")),
                    altitude_m=safe_float(row.get("altitude")),
                    x_local_m=float(x_m),
                    y_local_m=float(y_m),
                    z_local_m=safe_float(row.get("altitude")),
                    bs_distance_m=dist,
                    bs_bearing_deg=bearing,
                    bs_elevation_deg=elev,
                )
            )

    for internal_path in AERPAW_THR_FILES:
        track_id = Path(internal_path).stem
        for row_number, row in csv_rows_from_zip(zip_path, internal_path):
            metric = safe_float(row.get("iperf_client_mbps"))
            dist = safe_float(row.get("bs_distance"))
            bearing = safe_float(row.get("bs_bearing"))
            elev = safe_float(row.get("bs_elevation"))
            if not np.isfinite(metric) or not np.isfinite(dist) or not np.isfinite(bearing):
                continue
            x_m, y_m = bearing_distance_to_local_xy(dist, bearing, elev if np.isfinite(elev) else 0.0)
            points.append(
                MeasuredPoint(
                    dataset_id="aerpaw",
                    track_id=track_id,
                    selection_id="",
                    selection_role="candidate",
                    source_zip_path=zip_path,
                    internal_path=internal_path,
                    row_number=row_number,
                    metric_name="measured_throughput_mbps",
                    raw_metric_value=float(metric),
                    longitude=safe_float(row.get("longitude")),
                    latitude=safe_float(row.get("latitude")),
                    altitude_m=safe_float(row.get("altitude")),
                    x_local_m=float(x_m),
                    y_local_m=float(y_m),
                    z_local_m=safe_float(row.get("altitude")),
                    bs_distance_m=dist,
                    bs_bearing_deg=bearing,
                    bs_elevation_deg=elev,
                )
            )

    selected: List[MeasuredPoint] = []
    poor_f1 = [p for p in points if p.metric_name == "measured_sinr_db" and "flight1" in p.track_id]
    poor_f2 = [p for p in points if p.metric_name == "measured_sinr_db" and "flight2" in p.track_id]
    poor_thr = [p for p in points if p.metric_name == "measured_throughput_mbps"]

    pick = pick_unique(poor_f1, lambda item: item.raw_metric_value, False, 40.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_P1", selection_role="poor_sinr", note="flight1 poorest SINR point"))

    pick = pick_unique(poor_f2, lambda item: item.raw_metric_value, False, 40.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_P2", selection_role="poor_sinr", note="flight2 poorest SINR point"))

    pick = pick_unique(poor_thr, lambda item: item.raw_metric_value, False, 35.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_P3", selection_role="poor_throughput", note="poorest throughput point"))

    pick = pick_unique(points, lambda item: safe_float(item.bs_distance_m, -1.0), True, 35.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="AE_C1", selection_role="classic_edge", note="cell-edge point"))

    pick = pick_unique(
        [p for p in points if np.isfinite(p.bs_elevation_deg)],
        lambda item: item.bs_elevation_deg,
        False,
        35.0,
        selected,
        1,
    )
    if pick:
        selected.append(replace(pick[0], selection_id="AE_C2", selection_role="classic_low_elevation", note="lowest elevation point"))

    altitude_values = [p.altitude_m for p in points if np.isfinite(p.altitude_m)]
    reference_altitude_m = float(np.median(np.asarray(altitude_values, dtype=float))) if altitude_values else 30.0
    return DatasetBundle(
        dataset_id="aerpaw",
        family="AERPAW 5G NSA",
        reference_altitude_m=reference_altitude_m,
        station_height_m=station_height_m,
        comm_freq_hz=3.6e9,
        notes="AERPAW measured trajectory bundle. Relative XY is reconstructed from measured BS distance and bearing fields only.",
        all_points=tuple(points),
        selected_points=tuple(selected),
        station_xy_local_m=(0.0, 0.0),
    )


def load_dryad_bundle(zip_path: str) -> DatasetBundle:
    records: List[Dict[str, object]] = []
    for internal_path in DRYAD_SINR_FILES:
        track_id = Path(internal_path).parent.name + "_" + Path(internal_path).stem
        for row_number, row in csv_rows_from_nested_zip(zip_path, "Ericsson_Amir.zip", internal_path):
            metric = safe_float(row.get("LTE SINR"))
            if not np.isfinite(metric):
                continue
            records.append(
                {
                    "track_id": track_id,
                    "internal_path": internal_path,
                    "row_number": row_number,
                    "metric_name": "measured_sinr_db",
                    "raw_metric_value": float(metric),
                    "longitude": safe_float(row.get("Longitude")),
                    "latitude": safe_float(row.get("Latitude")),
                    "altitude_m": safe_float(row.get("Altitude")),
                }
            )
    for internal_path in DRYAD_THR_FILES:
        track_id = Path(internal_path).parent.name + "_" + Path(internal_path).stem
        for row_number, row in csv_rows_from_nested_zip(zip_path, "Ericsson_Amir.zip", internal_path):
            metric = safe_float(row.get("Throughput"))
            if not np.isfinite(metric):
                continue
            records.append(
                {
                    "track_id": track_id,
                    "internal_path": internal_path,
                    "row_number": row_number,
                    "metric_name": "measured_throughput_mbps",
                    "raw_metric_value": float(metric),
                    "longitude": safe_float(row.get("Longitude")),
                    "latitude": safe_float(row.get("Latitude")),
                    "altitude_m": safe_float(row.get("Altitude")),
                }
            )

    lat = np.asarray([float(item["latitude"]) for item in records], dtype=float)
    lon = np.asarray([float(item["longitude"]) for item in records], dtype=float)
    lat0 = float(np.mean(lat))
    lon0 = float(np.mean(lon))
    x_m, y_m = latlon_to_xy_m(lat, lon, lat0, lon0)

    points: List[MeasuredPoint] = []
    for idx, item in enumerate(records):
        points.append(
            MeasuredPoint(
                dataset_id="dryad",
                track_id=str(item["track_id"]),
                selection_id="",
                selection_role="candidate",
                source_zip_path=zip_path,
                internal_path=str(item["internal_path"]),
                row_number=int(item["row_number"]),
                metric_name=str(item["metric_name"]),
                raw_metric_value=float(item["raw_metric_value"]),
                longitude=float(item["longitude"]),
                latitude=float(item["latitude"]),
                altitude_m=float(item["altitude_m"]),
                x_local_m=float(x_m[idx]),
                y_local_m=float(y_m[idx]),
                z_local_m=float(item["altitude_m"]),
            )
        )

    sinr_points = [p for p in points if p.metric_name == "measured_sinr_db"]
    thr_points = [p for p in points if p.metric_name == "measured_throughput_mbps"]

    good_points: List[MeasuredPoint] = []
    for group in (sinr_points, thr_points):
        if not group:
            continue
        values = np.asarray([p.raw_metric_value for p in group], dtype=float)
        cutoff = float(np.percentile(values, 90.0))
        good_points.extend([p for p in group if p.raw_metric_value >= cutoff])
    if good_points:
        station_x = float(np.mean([p.x_local_m for p in good_points]))
        station_y = float(np.mean([p.y_local_m for p in good_points]))
    else:
        station_x = 0.0
        station_y = 0.0

    selected: List[MeasuredPoint] = []
    poor_sinr_45 = [p for p in sinr_points if "yaw45" in p.track_id]
    poor_sinr_315 = [p for p in sinr_points if "yaw315" in p.track_id]
    poor_thr_45 = [p for p in thr_points if "yaw45" in p.track_id]
    poor_thr_315 = [p for p in thr_points if "yaw315" in p.track_id]

    pick = pick_unique(poor_sinr_45, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P1", selection_role="poor_sinr", note="yaw45 poorest SINR point"))

    pick = pick_unique(poor_sinr_315, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P2", selection_role="poor_sinr", note="yaw315 poorest SINR point"))

    pick = pick_unique(poor_thr_45, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P3", selection_role="poor_throughput", note="yaw45 poorest throughput point"))

    pick = pick_unique(poor_thr_315, lambda item: item.raw_metric_value, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_P4", selection_role="poor_throughput", note="yaw315 poorest throughput point"))

    pick = pick_unique(points, lambda item: item.x_local_m + item.y_local_m, False, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_C1", selection_role="classic_corner", note="south-west track corner"))

    pick = pick_unique(points, lambda item: item.x_local_m + item.y_local_m, True, 20.0, selected, 1)
    if pick:
        selected.append(replace(pick[0], selection_id="DR_C2", selection_role="classic_corner", note="north-east track corner"))

    altitude_values = [p.altitude_m for p in points if np.isfinite(p.altitude_m)]
    reference_altitude_m = float(np.median(np.asarray(altitude_values, dtype=float))) if altitude_values else 30.0
    return DatasetBundle(
        dataset_id="dryad",
        family="Dryad Ericsson 5G NSA",
        reference_altitude_m=reference_altitude_m,
        station_height_m=12.0,
        comm_freq_hz=3.6e9,
        notes="Dryad measured map bundle. XY is reconstructed from logged latitude and longitude only.",
        all_points=tuple(points),
        selected_points=tuple(selected),
        station_xy_local_m=(station_x, station_y),
    )


def write_csv(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
