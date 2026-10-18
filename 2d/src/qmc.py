"""QMC and random point-set generation for 2D reproducibility experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import qmcpy as qp
import torch

from targets import Target2D


SequenceName = Literal["random", "sobol", "halton"]
VariantName = Literal["base", "shifted", "scrambled"]


@dataclass(frozen=True)
class PointSet2D:
    """A generated 2D point set and the UV points used to produce it."""

    points: np.ndarray
    uvs: np.ndarray
    sequence: str
    variant: str
    seed: Optional[int]
    shift: Optional[np.ndarray] = None


def _as_numpy(points: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def _make_qmcpy_distribution(sequence: str, randomize: Optional[str], seed: Optional[int]):
    if sequence == "sobol":
        return qp.DigitalNetB2(dimension=2, randomize=randomize, seed=seed)
    if sequence == "halton":
        return qp.Halton(dimension=2, randomize=randomize, seed=seed)
    raise ValueError(f"Unknown QMC sequence '{sequence}'. Expected sobol or halton.")


def _gen_qmcpy_samples(distribution, n: int) -> np.ndarray:
    """Generate samples from either old-style or new-style QMCPy objects."""
    try:
        return np.asarray(distribution(n, warn=False), dtype=np.float64)
    except Exception:
        pass

    if hasattr(distribution, "gen_samples"):
        try:
            return np.asarray(distribution.gen_samples(n, warn=False), dtype=np.float64)
        except TypeError:
            return np.asarray(distribution.gen_samples(n), dtype=np.float64)

    raise RuntimeError("QMCPy distribution supports neither call nor gen_samples.")


def generate_uv_points(
    sequence: SequenceName,
    n: int,
    *,
    variant: VariantName = "base",
    seed: Optional[int] = None,
    scramble: str = "LMS_DS",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate points in [0, 1)^2 before any target-density mapping."""
    if n < 1:
        raise ValueError("n must be positive.")

    if sequence == "random":
        rng = np.random.default_rng(seed)
        return rng.random((int(n), 2), dtype=np.float64), None

    if variant == "base":
        distribution = _make_qmcpy_distribution(sequence, randomize=None, seed=None)
        return _gen_qmcpy_samples(distribution, int(n)), None

    if variant == "scrambled":
        distribution = _make_qmcpy_distribution(sequence, randomize=scramble, seed=seed)
        return _gen_qmcpy_samples(distribution, int(n)), None

    if variant == "shifted":
        distribution = _make_qmcpy_distribution(sequence, randomize=None, seed=None)
        points = _gen_qmcpy_samples(distribution, int(n))
        rng = np.random.default_rng(seed)
        shift = rng.random(2, dtype=np.float64)
        return np.mod(points + shift, 1.0), shift

    raise ValueError(f"Unknown variant '{variant}'. Expected base, shifted, or scrambled.")


def map_to_target(
    uvs: np.ndarray,
    target: Optional[Target2D],
    *,
    device: str | torch.device,
) -> np.ndarray:
    """Map UV-space points through a target sampler, or return UVs unchanged."""
    if target is None or target.name == "uniform":
        return np.asarray(uvs, dtype=np.float64)

    target_device = torch.device(device)
    uv_tensor = torch.as_tensor(uvs, dtype=torch.float64, device=target_device)
    with torch.no_grad():
        points, _ = target.sample(uv_tensor)
    return _as_numpy(points).astype(np.float64, copy=False)


def generate_point_set(
    sequence: SequenceName,
    n: int,
    *,
    variant: VariantName = "base",
    seed: Optional[int] = None,
    target: Optional[Target2D] = None,
    device: str | torch.device,
    scramble: str = "LMS_DS",
) -> PointSet2D:
    """Generate one 2D point set, optionally mapped to a target distribution."""
    uvs, shift = generate_uv_points(
        sequence,
        n,
        variant=variant,
        seed=seed,
        scramble=scramble,
    )
    points = map_to_target(uvs, target, device=device)
    return PointSet2D(
        points=points,
        uvs=uvs,
        sequence=sequence,
        variant=variant,
        seed=seed,
        shift=shift,
    )


def generate_point_sets(
    sequence: SequenceName,
    n: int,
    *,
    instances: int,
    variant: VariantName = "base",
    seed: int = 0,
    target: Optional[Target2D] = None,
    device: str | torch.device,
    scramble: str = "LMS_DS",
) -> list[PointSet2D]:
    """Generate repeated point sets with deterministic per-instance seeds."""
    if instances < 1:
        raise ValueError("instances must be positive.")

    rng = np.random.default_rng(seed)
    point_sets: list[PointSet2D] = []
    for _ in range(int(instances)):
        instance_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        point_sets.append(
            generate_point_set(
                sequence,
                n,
                variant=variant,
                seed=instance_seed,
                target=target,
                device=device,
                scramble=scramble,
            )
        )
    return point_sets


def stack_points(point_sets: list[PointSet2D]) -> np.ndarray:
    """Stack generated point sets into an array with shape (R, N, 2)."""
    if len(point_sets) == 0:
        raise ValueError("point_sets must be non-empty.")
    return np.stack([point_set.points for point_set in point_sets], axis=0)
