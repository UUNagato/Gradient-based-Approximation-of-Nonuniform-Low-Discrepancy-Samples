"""Target distributions and spherical-cap probability estimation."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from envmap import EnvironmentMap
from sphere import (
    sample_uniform_in_caps,
    uniform_cap_probability,
    uniform_sample_hemisphere,
)


class UniformTarget:
    name = "uniform"

    def density(self, directions: torch.Tensor) -> torch.Tensor:
        return torch.full(
            directions.shape[:-1],
            1.0 / (4.0 * math.pi),
            dtype=directions.dtype,
            device=directions.device,
        )

    def analytic_cap_masses(self, caps: torch.Tensor) -> torch.Tensor:
        return uniform_cap_probability(caps)


class CosineTarget:
    name = "cosine"

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def density(self, directions: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(directions[..., 2], 0.0) / math.pi

    def sample_proposal(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        random_values = torch.rand(
            (count, 2),
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        directions = uniform_sample_hemisphere(random_values)
        proposal_pdf = torch.full(
            (count,),
            1.0 / (2.0 * math.pi),
            device=self.device,
            dtype=self.dtype,
        )
        return directions, proposal_pdf


class EnvmapTarget:
    name = "envmap"

    def __init__(
        self,
        path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.environment = EnvironmentMap(path, device=device, dtype=dtype)

    def density(self, directions: torch.Tensor) -> torch.Tensor:
        return self.environment.pdf(directions)

    def sample_proposal(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.environment.sample(count, generator=generator)


def make_target(
    case: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    envmap_path: str | Path | None = None,
):
    if case == "uniform":
        return UniformTarget()
    if case == "cosine":
        return CosineTarget(device=device, dtype=dtype)
    if case == "envmap":
        if envmap_path is None:
            raise ValueError("envmap_path is required for the envmap case.")
        return EnvmapTarget(envmap_path, device=device, dtype=dtype)
    raise ValueError(f"Unknown case: {case}")


@torch.no_grad()
def estimate_cap_masses(
    target,
    caps: torch.Tensor,
    *,
    sample_count: int,
    cap_chunk_size: int,
    sample_chunk_size: int,
    min_cap_hits: int,
    fallback_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Unbiased IS/cap-local estimate using an independent pilot for fallback.

    Finite-sample estimates may exceed one; clipping would introduce bias.
    """
    analytic = getattr(target, "analytic_cap_masses", None)
    if analytic is not None:
        return analytic(caps)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive for non-uniform targets.")
    if cap_chunk_size <= 0 or sample_chunk_size <= 0:
        raise ValueError("Chunk sizes must be positive.")

    cap_count = caps.shape[0]
    weighted_sum = torch.zeros(cap_count, dtype=caps.dtype, device=caps.device)
    inside_count = torch.zeros(cap_count, dtype=torch.long, device=caps.device)
    use_fallback = min_cap_hits > 0 and fallback_samples > 0

    for sample_start in range(0, sample_count, sample_chunk_size):
        current_sample_count = min(sample_chunk_size, sample_count - sample_start)
        directions, proposal_pdf = target.sample_proposal(
            current_sample_count, generator=generator
        )
        density = target.density(directions)
        if not torch.all(torch.isfinite(proposal_pdf) & (proposal_pdf > 0)):
            raise FloatingPointError("Proposal must return positive finite sample PDFs.")
        importance_weight = density / proposal_pdf
        if use_fallback:
            pilot, _ = target.sample_proposal(current_sample_count, generator=generator)

        for cap_start in range(0, cap_count, cap_chunk_size):
            cap_end = min(cap_start + cap_chunk_size, cap_count)
            cap_block = caps[cap_start:cap_end]
            cosine = directions @ cap_block[:, :3].transpose(0, 1)
            inside = cosine >= torch.cos(cap_block[:, 3]).unsqueeze(0)
            weighted_sum[cap_start:cap_end] += (
                inside.to(caps.dtype) * importance_weight.unsqueeze(1)
            ).sum(dim=0)
            if use_fallback:
                pilot_inside = pilot @ cap_block[:, :3].T >= torch.cos(cap_block[:, 3])[None]
                inside_count[cap_start:cap_end] += pilot_inside.sum(dim=0)

    masses = weighted_sum / sample_count
    if not use_fallback:
        return masses

    fallback_indices = torch.nonzero(inside_count < min_cap_hits, as_tuple=False).flatten()
    for start in range(0, fallback_indices.numel(), cap_chunk_size):
        indices = fallback_indices[start : start + cap_chunk_size]
        cap_block = caps[indices]
        local_directions = sample_uniform_in_caps(
            cap_block,
            fallback_samples,
            generator=generator,
        )
        local_density = target.density(local_directions.reshape(-1, 3)).reshape(
            cap_block.shape[0], fallback_samples
        )
        solid_angle = 4.0 * math.pi * uniform_cap_probability(cap_block)
        masses[indices] = local_density.mean(dim=1) * solid_angle

    return masses
