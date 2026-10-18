"""Reference and memory-batched spherical discrepancy optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from sphere import (
    cap_indicator,
    latent_to_direction,
    sample_spherical_caps,
)
from targets import estimate_cap_masses


@dataclass(frozen=True)
class OptimizationConfig:
    npoints: int = 256
    epochs: int = 2500
    nregions: int = 2048
    target_samples: int = 8192
    target_sample_chunk_size: int = 8192
    min_cap_hits: int = 64
    fallback_samples: int = 64
    learning_rate: float = 5e-4
    final_learning_rate: float = 5e-7
    tau_start: float = 20.0
    tau_end: float = 1000.0
    point_chunk_size: int = 1024
    cap_chunk_size: int = 1024
    backend: str = "batched"
    seed: int = 0
    init: str = "sobol"
    loss_epsilon: float = 1e-12
    snapshot_every: int = 0


@dataclass
class OptimizationResult:
    latents: torch.Tensor
    directions: torch.Tensor
    losses: list[float]
    snapshot_epochs: list[int] = field(default_factory=list)
    latent_snapshots: torch.Tensor | None = None


def _iter_slices(total_size: int, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, total_size, chunk_size):
        yield slice(start, min(start + chunk_size, total_size))


def discrepancy_loss(
    contribution_sums: torch.Tensor,
    cap_masses: torch.Tensor,
    npoints: int,
    epsilon: float,
) -> torch.Tensor:
    residual = contribution_sums / npoints - cap_masses
    return torch.sqrt(residual.square().mean() + epsilon)


def reference_loss(
    latents: torch.Tensor,
    caps: torch.Tensor,
    cap_masses: torch.Tensor,
    *,
    tau: float,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Readable implementation that materializes the complete [N,R] graph."""
    directions = latent_to_direction(latents)
    contribution_sums = cap_indicator(caps, directions, tau).sum(dim=0)
    return discrepancy_loss(contribution_sums, cap_masses, latents.shape[0], epsilon)


@torch.no_grad()
def _accumulate_contribution_sums(
    latents: torch.Tensor,
    caps: torch.Tensor,
    *,
    tau: float,
    point_chunk_size: int,
    cap_chunk_size: int,
) -> torch.Tensor:
    contribution_sums = torch.zeros(caps.shape[0], dtype=latents.dtype, device=latents.device)
    for point_slice in _iter_slices(latents.shape[0], point_chunk_size):
        directions = latent_to_direction(latents[point_slice])
        for cap_slice in _iter_slices(caps.shape[0], cap_chunk_size):
            contribution_sums[cap_slice] += cap_indicator(caps[cap_slice], directions, tau).sum(dim=0)
    return contribution_sums


def batched_loss_and_backward(
    latents: torch.Tensor,
    caps: torch.Tensor,
    cap_masses: torch.Tensor,
    *,
    tau: float,
    point_chunk_size: int,
    cap_chunk_size: int,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Two-pass backward without retaining the complete point-by-cap graph."""
    contribution_sums = _accumulate_contribution_sums(
        latents,
        caps,
        tau=tau,
        point_chunk_size=point_chunk_size,
        cap_chunk_size=cap_chunk_size,
    )

    contribution_leaf = contribution_sums.detach().requires_grad_(True)
    proxy_loss = discrepancy_loss(
        contribution_leaf,
        cap_masses,
        latents.shape[0],
        epsilon,
    )
    (upstream_gradient,) = torch.autograd.grad(proxy_loss, contribution_leaf)

    for cap_slice in _iter_slices(caps.shape[0], cap_chunk_size):
        cap_block = caps[cap_slice]
        cap_gradient = upstream_gradient[cap_slice]
        for point_slice in _iter_slices(latents.shape[0], point_chunk_size):
            directions = latent_to_direction(latents[point_slice])
            contribution_sums_block = cap_indicator(cap_block, directions, tau).sum(dim=0)
            contribution_sums_block.backward(cap_gradient)

    return proxy_loss.detach()


def _learning_rate_factor(
    step: int,
    *,
    total_steps: int,
    initial_rate: float,
    final_rate: float,
    alpha: float = 0.9,
) -> float:
    if total_steps <= 0:
        return 1.0
    progress_linear = min(max(step, 0), total_steps) / total_steps
    progress_curved = -math.log(1.0 - alpha * progress_linear) / -math.log(1.0 - alpha)
    ratio = final_rate / initial_rate
    return 1.0 - (1.0 - ratio) * progress_curved


def _initial_latents(
    config: OptimizationConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if config.init == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=config.seed)
        values = engine.draw(config.npoints).to(device=device, dtype=dtype)
    elif config.init == "random":
        values = torch.rand(
            (config.npoints, 2),
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        raise ValueError("init must be 'sobol' or 'random'.")
    return values.detach().requires_grad_(True)


def _validate_config(config: OptimizationConfig) -> None:
    positive_integer_fields = (
        "npoints",
        "epochs",
        "nregions",
        "target_samples",
        "target_sample_chunk_size",
        "fallback_samples",
        "point_chunk_size",
        "cap_chunk_size",
    )
    for field_name in positive_integer_fields:
        if getattr(config, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive.")
    if config.min_cap_hits < 0:
        raise ValueError("min_cap_hits cannot be negative.")
    if config.learning_rate <= 0 or config.final_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if config.snapshot_every < 0:
        raise ValueError("snapshot_every cannot be negative.")
    if config.backend not in ("reference", "batched"):
        raise ValueError("backend must be 'reference' or 'batched'.")


def optimize_pointset(
    target,
    config: OptimizationConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    progress: bool = True,
) -> OptimizationResult:
    _validate_config(config)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    latents = _initial_latents(
        config,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    optimizer = torch.optim.Adam([latents], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _learning_rate_factor(
            step,
            total_steps=config.epochs,
            initial_rate=config.learning_rate,
            final_rate=config.final_learning_rate,
        ),
    )

    losses: list[float] = []
    snapshot_epochs: list[int] = []
    latent_snapshots: list[torch.Tensor] = []
    if config.snapshot_every > 0:
        snapshot_epochs.append(0)
        latent_snapshots.append(latents.detach().cpu().clone())
    report_every = max(1, config.epochs // 10)
    for epoch in range(config.epochs):
        caps = sample_spherical_caps(
            config.nregions,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        cap_masses = estimate_cap_masses(
            target,
            caps,
            sample_count=config.target_samples,
            cap_chunk_size=config.cap_chunk_size,
            sample_chunk_size=config.target_sample_chunk_size,
            min_cap_hits=config.min_cap_hits,
            fallback_samples=config.fallback_samples,
            generator=generator,
        )
        tau = config.tau_start + (config.tau_end - config.tau_start) * (
            epoch / max(1, config.epochs - 1)
        )

        optimizer.zero_grad(set_to_none=True)
        if config.backend == "reference":
            loss = reference_loss(
                latents,
                caps,
                cap_masses,
                tau=tau,
                epsilon=config.loss_epsilon,
            )
            loss.backward()
            detached_loss = loss.detach()
        else:
            detached_loss = batched_loss_and_backward(
                latents,
                caps,
                cap_masses,
                tau=tau,
                point_chunk_size=config.point_chunk_size,
                cap_chunk_size=config.cap_chunk_size,
                epsilon=config.loss_epsilon,
            )

        if not torch.isfinite(detached_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
        if latents.grad is None or not torch.isfinite(latents.grad).all():
            raise FloatingPointError(f"Non-finite or missing latent gradient at epoch {epoch}.")

        optimizer.step()
        scheduler.step()
        loss_value = float(detached_loss.cpu())
        losses.append(loss_value)
        completed_epoch = epoch + 1
        if config.snapshot_every > 0 and (
            completed_epoch % config.snapshot_every == 0
            or completed_epoch == config.epochs
        ):
            snapshot_epochs.append(completed_epoch)
            latent_snapshots.append(latents.detach().cpu().clone())
        if progress and (epoch % report_every == 0 or epoch == config.epochs - 1):
            print(f"epoch {epoch + 1:5d}/{config.epochs}: loss={loss_value:.8f}")

    with torch.no_grad():
        final_latents = latents.detach().clone()
        final_directions = latent_to_direction(final_latents)
    stacked_snapshots = (
        torch.stack(latent_snapshots, dim=0) if latent_snapshots else None
    )
    return OptimizationResult(
        final_latents,
        final_directions,
        losses,
        snapshot_epochs,
        stacked_snapshots,
    )
