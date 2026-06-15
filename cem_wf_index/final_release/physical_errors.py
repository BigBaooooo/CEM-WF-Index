"""Physical consistency metrics for final typhoon analogue rankings.

These functions use CMA records only for evaluation/reporting. They are not
called by the online inference path and do not provide ranking features.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COAST_ANCHORS: tuple[tuple[float, float], ...] = (
    (18.2, 109.5),
    (22.3, 114.2),
    (24.0, 118.3),
    (25.0, 121.5),
    (28.0, 121.0),
    (31.2, 121.5),
    (35.0, 139.7),
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if not all(math.isfinite(float(v)) for v in vals):
        return float("nan")
    radius = 6371.0088
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))))


def _resample_numeric(values: Any, *, n_points: int = 16) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.full(n_points, np.nan, dtype=float)
    if arr.size == 1:
        return np.full(n_points, float(arr[0]), dtype=float)
    x = np.arange(arr.size, dtype=float)
    return np.interp(np.linspace(0.0, float(arr.size - 1), n_points), x, arr).astype(float)


def _closest_coast_point(frame: pd.DataFrame, anchors: tuple[tuple[float, float], ...]) -> tuple[float, float, float]:
    best_lat = float("nan")
    best_lon = float("nan")
    best_offset_h = float("nan")
    best_dist = float("inf")
    if frame.empty:
        return best_lat, best_lon, best_offset_h
    start_time = pd.to_datetime(frame["time_utc"], utc=True, errors="coerce").min() if "time_utc" in frame.columns else pd.NaT
    for row in frame[["time_utc", "lat", "lon"]].itertuples(index=False):
        lat = pd.to_numeric(row.lat, errors="coerce")
        lon = pd.to_numeric(row.lon, errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        for anchor_lat, anchor_lon in anchors:
            dist = _haversine_km(lat, lon, anchor_lat, anchor_lon)
            if math.isfinite(dist) and dist < best_dist:
                best_lat = float(lat)
                best_lon = float(lon)
                best_dist = float(dist)
                time_value = pd.Timestamp(row.time_utc)
                if pd.notna(start_time) and pd.notna(time_value):
                    best_offset_h = float((time_value - start_time).total_seconds() / 3600.0)
    return best_lat, best_lon, best_offset_h


def _typhoon_cma_event_cache(
    root: Path,
    details_dir: Path,
    descriptor: pd.DataFrame,
    event_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not event_ids:
        return {}
    required = {"event_id", "tc_id", "window_start", "window_end"}
    if not required.issubset(descriptor.columns):
        return {}
    mapping_path = details_dir / "typhoon_era5_tracklet_cma_eval_mapping_v6.csv"
    cma_path = root / "outputs" / "v5_migration" / "cma_best_track_table.csv"
    if not mapping_path.exists() or not cma_path.exists():
        return {}
    mapping = pd.read_csv(mapping_path, usecols=["tracklet_id", "tc_id"], low_memory=False).drop_duplicates("tracklet_id")
    tracklet_to_cma = {str(row.tracklet_id): str(row.tc_id) for row in mapping.itertuples(index=False)}
    wanted_descriptor = descriptor[descriptor["event_id"].astype(str).isin(event_ids)].copy()
    wanted_cma = {tracklet_to_cma.get(str(value), "") for value in wanted_descriptor["tc_id"].astype(str)}
    wanted_cma.discard("")
    if not wanted_cma:
        return {}
    cma = pd.read_csv(cma_path, usecols=["tc_id", "time_utc", "lat", "lon", "pres", "wnd"], low_memory=False)
    cma = cma[cma["tc_id"].astype(str).isin(wanted_cma)].copy()
    cma["time_utc"] = pd.to_datetime(cma["time_utc"], utc=True, errors="coerce")
    cma = cma.dropna(subset=["tc_id", "time_utc"])
    grouped = {str(tc_id): frame.sort_values("time_utc") for tc_id, frame in cma.groupby(cma["tc_id"].astype(str), sort=False)}

    cache: dict[str, dict[str, Any]] = {}
    for row in wanted_descriptor[["event_id", "tc_id", "window_start", "window_end"]].itertuples(index=False):
        cma_tc_id = tracklet_to_cma.get(str(row.tc_id), "")
        frame = grouped.get(cma_tc_id, pd.DataFrame())
        if not frame.empty:
            start = pd.Timestamp(row.window_start)
            end = pd.Timestamp(row.window_end)
            start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
            end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
            window = frame[(frame["time_utc"] >= start) & (frame["time_utc"] <= end)].copy()
            if window.empty:
                window = frame
        else:
            window = frame
        landfall_lat, landfall_lon, landfall_offset_h = _closest_coast_point(window, COAST_ANCHORS)
        cache[str(row.event_id)] = {
            "lat": _resample_numeric(window.get("lat", [])),
            "lon": _resample_numeric(window.get("lon", [])),
            "pres": _resample_numeric(window.get("pres", [])),
            "wnd": _resample_numeric(window.get("wnd", [])),
            "landfall_lat": landfall_lat,
            "landfall_lon": landfall_lon,
            "landfall_offset_h": landfall_offset_h,
        }
    return cache


def typhoon_top1_physical_errors_from_ranked(
    root: Path,
    details_dir: Path,
    descriptor: pd.DataFrame,
    ranked_by_query: dict[str, list[str]],
) -> dict[str, float | str]:
    pairs = {
        str(query_id): str(ranked[0])
        for query_id, ranked in ranked_by_query.items()
        if ranked and str(query_id) and str(ranked[0])
    }
    event_ids = set(pairs.keys()).union(pairs.values())
    cache = _typhoon_cma_event_cache(root, details_dir, descriptor, event_ids)
    track: list[float] = []
    pressure: list[float] = []
    wind: list[float] = []
    landfall: list[float] = []
    landfall_time: list[float] = []
    for query_id, candidate_id in pairs.items():
        query = cache.get(query_id)
        candidate = cache.get(candidate_id)
        if query is None or candidate is None:
            continue
        distances = np.asarray(
            [
                _haversine_km(q_lat, q_lon, c_lat, c_lon)
                for q_lat, q_lon, c_lat, c_lon in zip(query["lat"], query["lon"], candidate["lat"], candidate["lon"])
            ],
            dtype=float,
        )
        if np.isfinite(distances).any():
            track.append(float(np.sqrt(np.nanmean(np.square(distances)))))
        pressure_diff = np.abs(np.asarray(query["pres"], dtype=float) - np.asarray(candidate["pres"], dtype=float))
        if np.isfinite(pressure_diff).any():
            pressure.append(float(np.nanmean(pressure_diff)))
        wind_diff = np.abs(np.asarray(query["wnd"], dtype=float) - np.asarray(candidate["wnd"], dtype=float))
        if np.isfinite(wind_diff).any():
            wind.append(float(np.nanmean(wind_diff)))
        landfall_dist = _haversine_km(
            query["landfall_lat"],
            query["landfall_lon"],
            candidate["landfall_lat"],
            candidate["landfall_lon"],
        )
        if math.isfinite(landfall_dist):
            landfall.append(float(landfall_dist))
        if math.isfinite(query["landfall_offset_h"]) and math.isfinite(candidate["landfall_offset_h"]):
            landfall_time.append(float(abs(query["landfall_offset_h"] - candidate["landfall_offset_h"])))
    return {
        "Track RMSE/km": float(np.nanmean(track)) if track else "NOT_MEASURED_CURRENT",
        "PRES MAE/hPa": float(np.nanmean(pressure)) if pressure else "NOT_MEASURED_CURRENT",
        "WND MAE/m/s": float(np.nanmean(wind)) if wind else "NOT_MEASURED_CURRENT",
        "Landfall dist/km": float(np.nanmean(landfall)) if landfall else "NOT_MEASURED_CURRENT",
        "Landfall time err/h": float(np.nanmean(landfall_time)) if landfall_time else "NOT_MEASURED_CURRENT",
    }
