"""Direct optimization of uniform 2D point sets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import trange

from discrepancy import (
    _iter_slices,
    l2_discrepancy_warnock,
    rectangle_areas_uniform,
    sample_all_corner_anchored_rectangles,
    star_discrepancy_2d,
)
from qmc import generate_uv_points


@dataclass(frozen=True)
class UniformOptimizationConfig:
    """Configuration for repeated uniform 2D point-set optimization."""

    device: str | torch.device
    npoints: int
    instances: int = 16
    nregions: int = 8192
    nepochs: int = 1000
    nepochs_params: int = 500
    lr_init: float = 1e-3
    lr_final: Optional[float] = 1e-7
    tau0: float = 20.0
    tau1: float = 2000.0
    clamp_eps: float = 1e-12
    pair_batch_size: int = 4096
    point_chunk_size: Optional[int] = 1024
    region_chunk_size: Optional[int] = 8192
    seed: int = 0
    evaluate_discrepancy: bool = True
    show_progress: bool = False


@dataclass(frozen=True)
class UniformOptimizationResult:
    """Optimized uniform point sets and optional discrepancy values."""

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


class _Indicator(torch.autograd.Function):
    """Hard indicator in the forward pass with a sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.scale = scale
        return x.ge(0).type_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        sigx = torch.sigmoid(x * ctx.scale)
        grad = sigx * (1.0 - sigx)
        return grad * grad_output, None


indicator_func = _Indicator.apply


def compute_tau(epoch: int, nepochs_params: int, tau0: float, tau1: float) -> float:
    """Linearly ramp the indicator scale from tau0 to tau1."""
    denom = max(1, int(nepochs_params) - 1)
    capped_epoch = min(int(epoch), int(nepochs_params) - 1)
    return float(tau0) + (float(tau1) - float(tau0)) * (capped_epoch / denom)


def make_log_lr_lambda(
    *,
    lr_init: float,
    lr_final: float,
    total_steps: int,
    alpha: float = 0.9,
):
    """Create the log-shaped learning-rate schedule used by the original experiments."""
    ratio = float(lr_final) / float(lr_init)
    normalizer = -math.log(1.0 - float(alpha))
    total_steps = max(1, int(total_steps))

    def lr_lambda(current_step: int) -> float:
        progress_input = min(max(int(current_step), 0), total_steps) / float(total_steps)
        if progress_input == 0.0:
            return 1.0
        progress = -math.log(1.0 - float(alpha) * progress_input) / normalizer
        return 1.0 - (1.0 - ratio) * progress

    return lr_lambda


def point_corner_anchored_rectangle_distance(
    points: torch.Tensor,
    rectangles: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Signed distance to rectangles anchored at specified unit-square corners."""
    pts = points.unsqueeze(1)
    px = pts[:, :, 0]
    py = pts[:, :, 1]

    opposite = torch.where(anchors == 0, rectangles[:, 1, :], rectangles[:, 0, :]).unsqueeze(0)
    dx = torch.where(anchors[:, 0] == 0, opposite[:, :, 0] - px, px - opposite[:, :, 0])
    dy = torch.where(anchors[:, 1] == 0, opposite[:, :, 1] - py, py - opposite[:, :, 1])

    inside = torch.where(anchors[:, 0] == 0, px <= opposite[:, :, 0], px >= opposite[:, :, 0])
    inside = inside & torch.where(anchors[:, 1] == 0, py <= opposite[:, :, 1], py >= opposite[:, :, 1])
    inside_distance = torch.minimum(dx, dy)

    clamped_x = torch.clamp(
        px,
        torch.minimum(anchors[:, 0], opposite[:, :, 0]),
        torch.maximum(anchors[:, 0], opposite[:, :, 0]),
    )
    clamped_y = torch.clamp(
        py,
        torch.minimum(anchors[:, 1], opposite[:, :, 1]),
        torch.maximum(anchors[:, 1], opposite[:, :, 1]),
    )
    outside_distance = torch.sqrt(torch.clamp_min((px - clamped_x).square() + (py - clamped_y).square(), 1e-15))
    return torch.where(inside, inside_distance, -outside_distance)


def _materialize_points(
    z: torch.Tensor,
    *,
    clamp_eps: float,
    point_chunk_size: Optional[int],
) -> torch.Tensor:
    point_blocks: list[torch.Tensor] = []
    with torch.no_grad():
        for point_slice in _iter_slices(int(z.shape[0]), point_chunk_size):
            point_blocks.append(torch.clamp(z[point_slice], 0.0, 1.0 - clamp_eps).detach())
    return torch.cat(point_blocks, dim=0)


def _accumulate_contribution_sums(
    z: torch.Tensor,
    rectangles: torch.Tensor,
    anchors: torch.Tensor,
    *,
    tau: float,
    clamp_eps: float,
    point_chunk_size: Optional[int],
    region_chunk_size: Optional[int],
) -> torch.Tensor:
    """
    Count soft rectangle membership in chunks without keeping the autograd graph.

    The full point-by-rectangle tensor can be large. This pass computes the
    forward contribution sums cheaply, then `_backward_contribution_sums` rebuilds
    chunked graphs only for the gradient pass.
    """
    contribution_sums = torch.zeros(rectangles.shape[0], dtype=z.dtype, device=z.device)

    with torch.no_grad():
        for point_slice in _iter_slices(int(z.shape[0]), point_chunk_size):
            points = torch.clamp(z[point_slice], 0.0, 1.0 - clamp_eps)
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
    *,
    tau: float,
    clamp_eps: float,
    upstream_grad: torch.Tensor,
    point_chunk_size: Optional[int],
    region_chunk_size: Optional[int],
) -> None:
    """
    Recompute chunked contribution graphs and backpropagate a precomputed loss gradient.

    This avoids storing one large autograd graph for all point/rectangle pairs
    while preserving the gradient of the discrepancy loss with respect to points.
    """
    for region_slice in _iter_slices(int(rectangles.shape[0]), region_chunk_size):
        grad_block = upstream_grad[region_slice]
        rect_block = rectangles[region_slice]
        anchor_block = anchors[region_slice]

        for point_slice in _iter_slices(int(z.shape[0]), point_chunk_size):
            points = torch.clamp(z[point_slice], 0.0, 1.0 - clamp_eps)
            distances = point_corner_anchored_rectangle_distance(points, rect_block, anchor_block)
            indicator_func(distances, tau).sum(dim=0).backward(grad_block)


def optimize_uniform_point_set(
    npoints: int,
    *,
    device: str | torch.device,
    nregions: int = 8192,
    nepochs: int = 1000,
    nepochs_params: int = 500,
    lr_init: float = 1e-3,
    lr_final: Optional[float] = 1e-7,
    tau0: float = 20.0,
    tau1: float = 2000.0,
    clamp_eps: float = 1e-12,
    point_chunk_size: Optional[int] = 1024,
    region_chunk_size: Optional[int] = 8192,
    seed: Optional[int] = None,
    show_progress: bool = False,
    progress_desc: str = "uniform opt",
) -> torch.Tensor:
    """Optimize one uniform point set and return the best points as a Torch tensor."""
    if npoints < 1:
        raise ValueError("npoints must be positive.")
    if nregions < 1 or nepochs < 1 or nepochs_params < 1:
        raise ValueError("nregions, nepochs, and nepochs_params must be positive.")
    if clamp_eps < 0.0 or clamp_eps >= 1.0:
        raise ValueError("clamp_eps must satisfy 0 <= clamp_eps < 1.")

    dev = torch.device(device)
    lr_final = float(lr_final) if lr_final is not None else float(lr_init) * 1e-3

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
    epoch_iter = trange(int(nepochs), leave=False, disable=not show_progress, desc=progress_desc)

    for epoch in epoch_iter:
        rectangles, anchors = sample_all_corner_anchored_rectangles(
            int(nregions) * 4,
            dtype=torch.float64,
            device=dev,
        )
        areas = rectangle_areas_uniform(rectangles)
        tau = compute_tau(epoch, int(nepochs_params), float(tau0), float(tau1))

        contribution_sums = _accumulate_contribution_sums(
            z,
            rectangles,
            anchors,
            tau=tau,
            clamp_eps=float(clamp_eps),
            point_chunk_size=point_chunk_size,
            region_chunk_size=region_chunk_size,
        )
        loss = torch.sqrt(torch.square(contribution_sums / int(npoints) - areas).mean())
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_points = _materialize_points(
                z,
                clamp_eps=float(clamp_eps),
                point_chunk_size=point_chunk_size,
            )

        optimizer.zero_grad()
        contribution_leaf = contribution_sums.detach().requires_grad_(True)
        proxy_loss = torch.sqrt(torch.square(contribution_leaf / int(npoints) - areas).mean())
        (upstream_grad,) = torch.autograd.grad(proxy_loss, contribution_leaf)
        _backward_contribution_sums(
            z,
            rectangles,
            anchors,
            tau=tau,
            clamp_eps=float(clamp_eps),
            upstream_grad=upstream_grad,
            point_chunk_size=point_chunk_size,
            region_chunk_size=region_chunk_size,
        )
        optimizer.step()
        scheduler.step()

    if best_points is None:
        best_points = _materialize_points(z, clamp_eps=float(clamp_eps), point_chunk_size=point_chunk_size)
    return best_points.detach()


def optimize_uniform(config: UniformOptimizationConfig) -> UniformOptimizationResult:
    """Run repeated uniform point-set optimizations."""
    if config.instances < 1:
        raise ValueError("instances must be positive.")

    rng = np.random.default_rng(int(config.seed))
    point_sets: list[np.ndarray] = []
    l2_values: list[float] = []
    star_values: list[float] = []

    for instance_index in range(int(config.instances)):
        instance_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        points = optimize_uniform_point_set(
            int(config.npoints),
            device=config.device,
            nregions=int(config.nregions),
            nepochs=int(config.nepochs),
            nepochs_params=int(config.nepochs_params),
            lr_init=float(config.lr_init),
            lr_final=config.lr_final,
            tau0=float(config.tau0),
            tau1=float(config.tau1),
            clamp_eps=float(config.clamp_eps),
            point_chunk_size=config.point_chunk_size,
            region_chunk_size=config.region_chunk_size,
            seed=instance_seed,
            show_progress=config.show_progress,
            progress_desc=f"N={config.npoints} instance={instance_index + 1}/{config.instances}",
        )
        point_sets.append(points.cpu().numpy())

        if config.evaluate_discrepancy:
            l2 = l2_discrepancy_warnock(points, pair_batch_size=int(config.pair_batch_size))
            star = star_discrepancy_2d(points)
            l2_values.append(float(l2.cpu().item()))
            star_values.append(float(star.cpu().item()))

    return UniformOptimizationResult(
        points=np.stack(point_sets, axis=0),
        l2_values=np.asarray(l2_values, dtype=np.float64) if config.evaluate_discrepancy else None,
        star_values=np.asarray(star_values, dtype=np.float64) if config.evaluate_discrepancy else None,
    )
