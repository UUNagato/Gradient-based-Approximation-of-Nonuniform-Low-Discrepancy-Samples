"""Small lat-long environment-map reader, PDF, and importance sampler."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


class EnvironmentMap:
    """Piecewise-constant spherical distribution built from a NumPy envmap."""

    def __init__(
        self,
        path: str | Path | np.ndarray,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(path, np.ndarray):
            image = path
        else:
            path = Path(path)
            if path.suffix.lower() == ".npy":
                image = np.load(path)
            elif path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                import matplotlib.image as mpimg
                image = mpimg.imread(path)
            else:
                raise ValueError("Environment maps must be .npy, .png, .jpg, or .jpeg files.")
        image = np.array(image, copy=True)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected envmap shape [H,W], [H,W,3], or [H,W,4], got {image.shape}.")
        if not np.isfinite(image).all():
            raise ValueError(f"Environment map contains NaN or Inf: {path}")

        self.path = path if isinstance(path, Path) else None
        self.device = device
        self.dtype = dtype
        self.image = torch.as_tensor(image, device=device, dtype=dtype)
        self.height, self.width, _ = self.image.shape
        if self.height == 0 or self.width == 0:
            raise ValueError("Environment map dimensions must be nonzero.")

        luminance = (
            0.212671 * self.image[..., 0]
            + 0.715160 * self.image[..., 1]
            + 0.072169 * self.image[..., 2]
        )
        self.luminance = torch.clamp_min(luminance, 0.0)
        # Double precision avoids cancellation near the poles and in the CDF.
        theta_edges = torch.arange(self.height + 1, device=device, dtype=torch.float64)
        theta_edges = theta_edges * (math.pi / self.height)
        self.cos_theta_edges = torch.cos(theta_edges)
        self.texel_solid_angle = (2.0 * math.pi / self.width) * (
            self.cos_theta_edges[:-1] - self.cos_theta_edges[1:]
        )
        weights = self.luminance.double() * self.texel_solid_angle[:, None]
        total_weight = weights.sum()
        if not torch.isfinite(total_weight) or total_weight <= 0:
            raise ValueError(f"Environment map has no positive luminance: {path}")

        self.texel_probability = weights / total_weight
        self.texel_pdf = (self.luminance.double() / total_weight).to(dtype)
        self.cdf = self.texel_probability.flatten().cumsum(0)
        # Normalize the whole CDF rather than changing only its final element.
        self.cdf = self.cdf / self.cdf[-1]

    def _uv_to_direction(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        theta = v * math.pi
        phi = u * (2.0 * math.pi)
        sin_theta = torch.sin(theta)
        return torch.stack(
            (sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)),
            dim=-1,
        )

    def direction_to_uv(self, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = torch.atan2(torch.linalg.vector_norm(directions[..., :2], dim=-1), directions[..., 2])
        phi = torch.remainder(torch.atan2(directions[..., 1], directions[..., 0]), 2.0 * math.pi)
        return phi / (2.0 * math.pi), theta / math.pi

    def pdf(self, directions: torch.Tensor) -> torch.Tensor:
        u, v = self.direction_to_uv(directions)
        column = torch.clamp((u * self.width).to(torch.long), 0, self.width - 1)
        row = torch.clamp((v * self.height).to(torch.long), 0, self.height - 1)
        return self.texel_pdf[row, column]

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        random_values = torch.rand(
            (count, 3),
            device=self.device,
            dtype=torch.float64,
            generator=generator,
        )
        return self.sample_from_uniform(random_values)

    def sample_from_uniform(self, random_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map [N,3] uniforms to texels, then sample phi and cos(theta) there."""
        random_values = random_values.to(device=self.device, dtype=torch.float64)
        if random_values.ndim != 2 or random_values.shape[1] != 3:
            raise ValueError("Environment sampling requires [N,3] uniforms.")
        if not torch.all(torch.isfinite(random_values) & (random_values >= 0) & (random_values <= 1)):
            raise ValueError("Uniform samples must be finite and within [0,1].")
        one = random_values.new_tensor(1.0)
        choice = random_values[:, 0].clamp_max(torch.nextafter(one, one * 0))
        # right=True skips zero-probability CDF plateaus, including a leading zero.
        index = torch.searchsorted(self.cdf, choice.contiguous(), right=True)
        row, column = index // self.width, index % self.width
        # Endpoint uniforms represent measure-zero boundaries. Move only those
        # endpoints to the interior, rather than clipping a finite probability band.
        jitter = random_values[:, 1:]
        jitter = torch.where((jitter == 0) | (jitter == 1), 0.5, jitter)
        phi = (column + jitter[:, 0]) * (2.0 * math.pi / self.width)
        top, bottom = self.cos_theta_edges[row], self.cos_theta_edges[row + 1]
        z = top + (bottom - top) * jitter[:, 1]
        radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        directions = torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), -1)
        directions = directions.to(self.dtype)
        # Casting to float32 can round an interior point into a neighboring texel.
        # Correct only those unrepresentable boundary samples to the texel center.
        u, v = self.direction_to_uv(directions)
        actual_row = (v * self.height).long().clamp(0, self.height - 1)
        actual_column = (u * self.width).long().clamp(0, self.width - 1)
        rounded_outside = (actual_row != row) | (actual_column != column)
        center_z = 0.5 * (top + bottom)
        center_radius = torch.sqrt((1.0 - center_z.square()).clamp_min(0))
        center_phi = (column + 0.5) * (2.0 * math.pi / self.width)
        centers = torch.stack((center_radius * torch.cos(center_phi),
                               center_radius * torch.sin(center_phi), center_z), -1).to(self.dtype)
        directions = torch.where(rounded_outside[:, None], centers, directions)
        return directions, self.texel_pdf[row, column]
