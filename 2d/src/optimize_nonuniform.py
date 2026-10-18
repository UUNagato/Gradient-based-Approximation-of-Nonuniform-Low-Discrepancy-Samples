"""Direct optimization of non-uniform 2D point sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import trange

from discrepancy import (
    _iter_slices,
    estimate_nonuniform_discrepancies,
    rectangle_areas_bilinear,
    rectangle_areas_monte_carlo,
    rectangle_areas_uniform,
    sample_all_corner_anchored_rectangles,
)
from optimize_uniform import (
    compute_tau,
    indicator_func,
    make_log_lr_lambda,
    point_corner_anchored_rectangle_distance,
)
from qmc import generate_uv_points
from targets import Target2D


UVBoundaryMode = Literal["clamp", "wrap"]
PointSetSelection = Literal["best", "final"]


@dataclass(frozen=True)
class NonuniformOptimizationConfig:
    """Configuration for repeated non-uniform 2D point-set optimization."""

    device: str | torch.device
    target: Target2D
    npoints: int
    instances: int = 16
    nregions: int = 4096
    nspace_samples: int = 262144
    nepochs: int = 1000
    nepochs_params: int = 500
    lr_init: float = 1e-4
    lr_final: Optional[float] = 1e-7
    tau0: float = 20.0
    tau1: float = 2000.0
    uv_boundary_mode: UVBoundaryMode = "clamp"
    pointset_selection: PointSetSelection = "best"
    clamp_eps: float = 1e-12
    gt_nregions: int = 32768
    gt_nspace_samples: int = 262144
    gt_reps: int = 8
    areas_batch_size: int = 8192
    areas_space_batch_size: Optional[int] = 131072
    point_chunk_size: Optional[int] = None
    region_chunk_size: Optional[int] = 8192
    seed: int = 0
    evaluate_discrepancy: bool = True
    show_progress: bool = False


@dataclass(frozen=True)
class NonuniformOptimizationResult:
    """Optimized non-uniform point sets and optional discrepancy values."""

    points: np.ndarray
    l2_values: Optional[np.ndarray]
    star_values: Optional[np.ndarray]

    @property
    def l2_mean(self) -> Optional[float]:
        if self.l2_values is None:
            return None
        return float(np.mean(self.l2_values))

    @property
    def star_mean(self) -> Optional[float]:
        if self.star_values is None:
            return None
        return float(np.mean(self.star_values))


def _z_to_uvs(z: torch.Tensor, boundary_mode: UVBoundaryMode, clamp_eps: float) -> torch.Tensor:
    if boundary_mode == "clamp":
        return torch.clamp(z, 0.0, 1.0 - clamp_eps)
    if boundary_mode == "wrap":
        return torch.remainder(z, 1.0)
    raise ValueError(f"Unknown UV boundary mode: {boundary_mode}")


def _uvs_to_points(uvs: torch.Tensor, target: Target2D) -> torch.Tensor:
    points, _ = target.sample(uvs)
    return points


def _materialize_points(
    z: torch.Tensor,
    target: Target2D,
    *,
    uv_boundary_mode: UVBoundaryMode,
    clamp_eps: float,
    point_chunk_size: Optional[int],
) -> torch.Tensor:
    point_blocks: list[torch.Tensor] = []
    with torch.no_grad():
        for point_slice in _iter_slices(int(z.shape[0]), point_chunk_size):
            uvs = _z_to_uvs(z[point_slice], uv_boundary_mode, clamp_eps)
            point_blocks.append(_uvs_to_points(uvs, target).detach())
    return torch.cat(point_blocks, dim=0)


def _rectangle_masses(
    rectangles: torch.Tensor,
    target: Target2D,
    *,
    nspace_samples: int,
    areas_batch_size: int,
    areas_space_batch_size: Optional[int],
) -> torch.Tensor:
    """Compute or estimate target probability mass inside each rectangle."""
    if target.name == "uniform":
        return rectangle_areas_uniform(rectangles)
    if target.name == "bilinear":
        return rectangle_areas_bilinear(rectangles)

    space_samples = torch.rand(
        (int(nspace_samples), 2),
        dtype=rectangles.dtype,
        device=rectangles.device,
    )
    return rectangle_areas_monte_carlo(
        rectangles,
        target,
        space_samples,
        batch_size=areas_batch_size,
        space_batch_size=areas_space_batch_size,
    )


def _accumulate_contribution_sums(
    z: torch.Tensor,
    rectangles: torch.Tensor,
    anchors: torch.Tensor,
    target: Target2D,
    *,
    tau: float,
    uv_boundary_mode: UVBoundaryMode,
    clamp_eps: float,
    point_chunk_size: Optional[int],
    region_chunk_size: Optional[int],
) -> torch.Tensor:
    """
    Count soft rectangle membership after target mapping without storing gradients.

    The optimization loss depends on mapped points, but the full
    point-by-rectangle graph is large. This pass computes only the contribution
    sums; `_backward_contribution_sums` then rebuilds chunked graphs for the
    gradient pass.
    """
    contribution_sums = torch.zeros(rectangles.shape[0], dtype=z.dtype, device=z.device)

    with torch.no_grad():
        for point_slice in _iter_slices(int(z.shape[0]), point_chunk_size):
            uvs = _z_to_uvs(z[point_slice], uv_boundary_mode, clamp_eps)
            points = _uvs_to_points(uvs, target)
            for region_slice in _iter_slices(int(rectangles.shape[0]), region_chunk_size):
                distances = point_corner_anchored_rectangle_distance(
                    points,
                    rectangles[region_slice],
                    anchors[region_slice],
                )
                contribution_sums[region_slice] += indicator_func(distances, tau).sum(dim=0)

    return contribution_sums


def _backward_contribution_sums(
    z: torch.Tensor,
    rectangles: torch.Tensor,
    anchors: torch.Tensor,
    target: Target2D,
    *,
    tau: float,
    uv_boundary_mode: UVBoundaryMode,
    clamp_eps: float,
    upstream_grad: torch.Tensor,
    point_chunk_size: Optional[int],
    region_chunk_size: Optional[int],
) -> None:
    """
    Recompute mapped contribution graphs and backpropagate a precomputed loss gradient.

    This keeps the memory footprint bounded by the chunk sizes while preserving
    gradients through both the target sampler and the soft rectangle indicator.
    """
    for region_slice in _iter_slices(int(rectangles.shape[0]), region_chunk_size):
        grad_block = upstream_grad[region_slice]
        rect_block = rectangles[region_slice]
        anchor_block = anchors[region_slice]

        for point_slice in _iter_slices(int(z.shape[0]), point_chunk_size):
            uvs = _z_to_uvs(z[point_slice], uv_boundary_mode, clamp_eps)
            points = _uvs_to_points(uvs, target)
            distances = point_corner_anchored_rectangle_distance(points, rect_block, anchor_block)
            indicator_func(distances, tau).sum(dim=0).backward(grad_block)


def optimize_nonuniform_point_set(
    npoints: int,
    target: Target2D,
    *,
    device: str | torch.device,
    nregions: int = 4096,
    nspace_samples: int = 262144,
    nepochs: int = 1000,
    nepochs_params: int = 500,
    lr_init: float = 1e-4,
    lr_final: Optional[float] = 1e-7,
    tau0: float = 20.0,
    tau1: float = 2000.0,
    uv_boundary_mode: UVBoundaryMode = "clamp",
    pointset_selection: PointSetSelection = "best",
    clamp_eps: float = 1e-12,
    areas_batch_size: int = 8192,
    areas_space_batch_size: Optional[int] = 131072,
    point_chunk_size: Optional[int] = None,
    region_chunk_size: Optional[int] = 8192,
    seed: Optional[int] = None,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> torch.Tensor:
    """Optimize one mapped point set for a non-uniform target."""
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    if nregions < 1 or nspace_samples < 1 or nepochs < 1 or nepochs_params < 1:
        raise ValueError("nregions, nspace_samples, nepochs, and nepochs_params must be positive.")
    if clamp_eps < 0.0 or clamp_eps >= 1.0:
        raise ValueError("clamp_eps must satisfy 0 <= clamp_eps < 1.")
    if uv_boundary_mode not in ("clamp", "wrap"):
        raise ValueError("uv_boundary_mode must be 'clamp' or 'wrap'.")
    if pointset_selection not in ("best", "final"):
        raise ValueError("pointset_selection must be 'best' or 'final'.")

    dev = torch.device(device)
    lr_final = float(lr_final) if lr_final is not None else float(lr_init) * 1e-3

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    uvs, _ = generate_uv_points("sobol", int(npoints), variant="scrambled", seed=seed)
    z = torch.as_tensor(uvs, dtype=torch.float64, device=dev)
    z.requires_grad = True

    optimizer = optim.Adam([z], lr=float(lr_init))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_log_lr_lambda(
            lr_init=float(lr_init),
            lr_final=lr_final,
            total_steps=int(nepochs_params),
        ),
    )

    best_loss = float("inf")
    best_points: Optional[torch.Tensor] = None
    use_final_points = pointset_selection == "final"
    epoch_iter = trange(
        int(nepochs),
        leave=False,
        disable=not show_progress,
        desc=progress_desc or f"{target.name} opt",
    )

    for epoch in epoch_iter:
        rectangles, anchors = sample_all_corner_anchored_rectangles(
            int(nregions) * 4,
            dtype=torch.float64,
            device=dev,
        )
        with torch.no_grad():
            masses = _rectangle_masses(
                rectangles,
                target,
                nspace_samples=int(nspace_samples),
                areas_batch_size=int(areas_batch_size),
                areas_space_batch_size=areas_space_batch_size,
            )

        tau = compute_tau(epoch, int(nepochs_params), float(tau0), float(tau1))
        contribution_sums = _accumulate_contribution_sums(
            z,
            rectangles,
            anchors,
            target,
            tau=tau,
            uv_boundary_mode=uv_boundary_mode,
            clamp_eps=float(clamp_eps),
            point_chunk_size=point_chunk_size,
            region_chunk_size=region_chunk_size,
        )
        loss = torch.sqrt(torch.square(contribution_sums / int(npoints) - masses).mean())
        loss_value = float(loss.item())

        if not use_final_points and loss_value < best_loss:
            best_loss = loss_value
            best_points = _materialize_points(
                z,
                target,
                uv_boundary_mode=uv_boundary_mode,
                clamp_eps=float(clamp_eps),
                point_chunk_size=point_chunk_size,
            )

        optimizer.zero_grad()
        contribution_leaf = contribution_sums.detach().requires_grad_(True)
        proxy_loss = torch.sqrt(torch.square(contribution_leaf / int(npoints) - masses).mean())
        (upstream_grad,) = torch.autograd.grad(proxy_loss, contribution_leaf)
        _backward_contribution_sums(
            z,
            rectangles,
            anchors,
            target,
            tau=tau,
            uv_boundary_mode=uv_boundary_mode,
            clamp_eps=float(clamp_eps),
            upstream_grad=upstream_grad,
            point_chunk_size=point_chunk_size,
            region_chunk_size=region_chunk_size,
        )
        optimizer.step()
        scheduler.step()

    if use_final_points or best_points is None:
        best_points = _materialize_points(
            z,
            target,
            uv_boundary_mode=uv_boundary_mode,
            clamp_eps=float(clamp_eps),
            point_chunk_size=point_chunk_size,
        )
    return best_points.detach()


def optimize_nonuniform(config: NonuniformOptimizationConfig) -> NonuniformOptimizationResult:
    """Run repeated non-uniform point-set optimizations."""
    if config.instances < 1:
        raise ValueError("instances must be positive.")

    rng = np.random.default_rng(int(config.seed))
    point_sets: list[np.ndarray] = []
    l2_values: list[float] = []
    star_values: list[float] = []

    for instance_index in range(int(config.instances)):
        instance_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        points = optimize_nonuniform_point_set(
            int(config.npoints),
            config.target,
            device=config.device,
            nregions=int(config.nregions),
            nspace_samples=int(config.nspace_samples),
            nepochs=int(config.nepochs),
            nepochs_params=int(config.nepochs_params),
            lr_init=float(config.lr_init),
            lr_final=config.lr_final,
            tau0=float(config.tau0),
            tau1=float(config.tau1),
            uv_boundary_mode=config.uv_boundary_mode,
            pointset_selection=config.pointset_selection,
            clamp_eps=float(config.clamp_eps),
            areas_batch_size=int(config.areas_batch_size),
            areas_space_batch_size=config.areas_space_batch_size,
            point_chunk_size=config.point_chunk_size,
            region_chunk_size=config.region_chunk_size,
            seed=instance_seed,
            show_progress=config.show_progress,
            progress_desc=f"N={config.npoints} instance={instance_index + 1}/{config.instances}",
        )
        point_sets.append(points.cpu().numpy())

        if config.evaluate_discrepancy:
            l2, star = estimate_nonuniform_discrepancies(
                points,
                config.target,
                nrectangles=int(config.gt_nregions),
                nspace_samples=int(config.gt_nspace_samples),
                reps=int(config.gt_reps),
                areas_batch_size=int(config.areas_batch_size),
                areas_space_batch_size=config.areas_space_batch_size,
                point_chunk_size=config.point_chunk_size,
                region_chunk_size=config.region_chunk_size,
            )
            l2_values.append(l2)
            star_values.append(star)

    return NonuniformOptimizationResult(
        points=np.stack(point_sets, axis=0),
        l2_values=np.asarray(l2_values, dtype=np.float64) if config.evaluate_discrepancy else None,
        star_values=np.asarray(star_values, dtype=np.float64) if config.evaluate_discrepancy else None,
    )
