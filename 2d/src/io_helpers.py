"""Input/output helpers for 2D reproducibility files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _config_to_json(config: Optional[Mapping[str, Any] | str]) -> str:
    if config is None:
        return "{}"
    if isinstance(config, str):
        return config
    return json.dumps(dict(config), sort_keys=True)


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.shape == () else value)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_point_sets(
    path: str | Path,
    *,
    points: np.ndarray,
    target: str,
    method: str,
    seed: Optional[int],
    config: Optional[Mapping[str, Any] | str] = None,
    uvs: Optional[np.ndarray] = None,
) -> Path:
    """Save point sets with shape (R, N, 2) to a compressed NPZ file."""
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim == 2 and points_array.shape[1] == 2:
        points_array = points_array[None, ...]
    if points_array.ndim != 3 or points_array.shape[2] != 2:
        raise ValueError(f"Expected points with shape (N, 2) or (R, N, 2), got {points_array.shape}.")
    if not np.isfinite(points_array).all():
        raise ValueError("points contains non-finite values.")

    output_path = Path(path)
    ensure_dir(output_path.parent)

    arrays: dict[str, Any] = {
        "points": points_array,
        "target": np.asarray(str(target)),
        "method": np.asarray(str(method)),
        "seed": np.asarray(-1 if seed is None else int(seed), dtype=np.int64),
        "config_json": np.asarray(_config_to_json(config)),
    }
    if uvs is not None:
        uvs_array = np.asarray(uvs, dtype=np.float64)
        if uvs_array.ndim == 2 and uvs_array.shape[1] == 2:
            uvs_array = uvs_array[None, ...]
        if uvs_array.shape != points_array.shape:
            raise ValueError(f"uvs must have shape {points_array.shape}, got {uvs_array.shape}.")
        arrays["uvs"] = uvs_array

    np.savez_compressed(output_path, **arrays)
    return output_path


def load_point_sets(path: str | Path) -> dict[str, Any]:
    """Load a point-set NPZ file."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        if "points" not in data:
            keys = ", ".join(sorted(data.files))
            raise ValueError(f"{input_path} does not contain 'points'. Available arrays: {keys}.")

        result: dict[str, Any] = {
            "points": np.asarray(data["points"], dtype=np.float64),
            "target": _scalar_string(data["target"]) if "target" in data else "",
            "method": _scalar_string(data["method"]) if "method" in data else "",
            "seed": int(data["seed"].item()) if "seed" in data else None,
            "config_json": _scalar_string(data["config_json"]) if "config_json" in data else "{}",
        }
        if "uvs" in data:
            result["uvs"] = np.asarray(data["uvs"], dtype=np.float64)

    points = result["points"]
    if points.ndim == 2 and points.shape[1] == 2:
        result["points"] = points[None, ...]
    elif points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(f"Expected points with shape (N, 2) or (R, N, 2), got {points.shape}.")

    if result["seed"] == -1:
        result["seed"] = None
    return result


def save_discrepancies(
    path: str | Path,
    *,
    ns: np.ndarray | list[int],
    l2_mean: np.ndarray | list[float],
    star_mean: np.ndarray | list[float],
    target: str,
    method: str,
    config: Optional[Mapping[str, Any] | str] = None,
    l2_values: Optional[np.ndarray] = None,
    star_values: Optional[np.ndarray] = None,
) -> Path:
    """Save discrepancy summaries to a compressed NPZ file."""
    ns_array = np.asarray(ns, dtype=np.int64)
    l2_mean_array = np.asarray(l2_mean, dtype=np.float64)
    star_mean_array = np.asarray(star_mean, dtype=np.float64)

    if ns_array.ndim != 1:
        raise ValueError("ns must be one-dimensional.")
    if l2_mean_array.shape != ns_array.shape or star_mean_array.shape != ns_array.shape:
        raise ValueError("l2_mean and star_mean must have the same shape as ns.")
    if not np.isfinite(l2_mean_array).all() or not np.isfinite(star_mean_array).all():
        raise ValueError("discrepancy arrays contain non-finite values.")

    output_path = Path(path)
    ensure_dir(output_path.parent)

    arrays: dict[str, Any] = {
        "ns": ns_array,
        "l2_mean": l2_mean_array,
        "star_mean": star_mean_array,
        "target": np.asarray(str(target)),
        "method": np.asarray(str(method)),
        "config_json": np.asarray(_config_to_json(config)),
    }
    if l2_values is not None:
        arrays["l2_values"] = np.asarray(l2_values, dtype=np.float64)
    if star_values is not None:
        arrays["star_values"] = np.asarray(star_values, dtype=np.float64)

    np.savez_compressed(output_path, **arrays)
    return output_path


def load_discrepancies(path: str | Path) -> dict[str, Any]:
    """Load a discrepancy summary NPZ file."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        required = {"ns", "l2_mean", "star_mean"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{input_path} is missing required arrays: {', '.join(missing)}.")

        result: dict[str, Any] = {
            "ns": np.asarray(data["ns"], dtype=np.int64),
            "l2_mean": np.asarray(data["l2_mean"], dtype=np.float64),
            "star_mean": np.asarray(data["star_mean"], dtype=np.float64),
            "target": _scalar_string(data["target"]) if "target" in data else "",
            "method": _scalar_string(data["method"]) if "method" in data else "",
            "config_json": _scalar_string(data["config_json"]) if "config_json" in data else "{}",
        }
        if "l2_values" in data:
            result["l2_values"] = np.asarray(data["l2_values"], dtype=np.float64)
        if "star_values" in data:
            result["star_values"] = np.asarray(data["star_values"], dtype=np.float64)

    return result


def merge_discrepancy_series(
    existing: Optional[dict[str, Any]],
    *,
    ns: np.ndarray | list[int],
    l2_mean: np.ndarray | list[float],
    star_mean: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge new discrepancy values with an existing loaded summary by N."""
    combined: dict[int, tuple[float, float]] = {}
    if existing is not None:
        for n, l2, star in zip(existing["ns"], existing["l2_mean"], existing["star_mean"]):
            combined[int(n)] = (float(l2), float(star))

    for n, l2, star in zip(ns, l2_mean, star_mean):
        combined[int(n)] = (float(l2), float(star))

    sorted_ns = np.asarray(sorted(combined.keys()), dtype=np.int64)
    sorted_l2 = np.asarray([combined[int(n)][0] for n in sorted_ns], dtype=np.float64)
    sorted_star = np.asarray([combined[int(n)][1] for n in sorted_ns], dtype=np.float64)
    return sorted_ns, sorted_l2, sorted_star
