"""Target densities and samplers for the 2D reproducibility experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class Target2D:
    """A normalized 2D target distribution on the unit square."""

    name: str
    density: Callable[[Tensor], Tensor]
    sample: Callable[[Tensor], Tuple[Tensor, Tensor]]


def _inside_unit_square(points: Tensor) -> Tensor:
    return (
        (points[..., 0] >= 0.0)
        & (points[..., 0] <= 1.0)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= 1.0)
    )


def uniform_density(points: Tensor) -> Tensor:
    """Uniform density on [0, 1]^2."""
    inside = _inside_unit_square(points)
    return torch.where(inside, torch.ones_like(points[..., 0]), torch.zeros_like(points[..., 0]))


def sample_uniform(uv: Tensor) -> Tuple[Tensor, Tensor]:
    """Identity sampler for the unit square."""
    points = uv
    pdf = torch.ones_like(uv[..., 0])
    return points, pdf


def bilinear_density(points: Tensor) -> Tensor:
    """Density p(x, y) = 4xy on [0, 1]^2."""
    inside = _inside_unit_square(points)
    pdf = 4.0 * points[..., 0] * points[..., 1]
    return torch.where(inside, pdf, torch.zeros_like(pdf))


def sample_bilinear(uv: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Inverse-CDF sampler for p(x, y) = 4xy."""
    u = torch.clamp(uv[..., 0], min=eps)
    v = torch.clamp(uv[..., 1], min=eps)
    x = torch.sqrt(u)
    y = torch.sqrt(v)
    points = torch.stack([x, y], dim=-1)
    pdf = 4.0 * x * y
    return points, pdf


def quarter_disk_density(points: Tensor) -> Tensor:
    """Uniform density on the quarter disk x^2 + y^2 < 2/pi."""
    r2 = points[..., 0] ** 2 + points[..., 1] ** 2
    inside = (points[..., 0] >= 0.0) & (points[..., 1] >= 0.0) & (r2 < (2.0 / np.pi))
    return torch.where(inside, torch.full_like(r2, 2.0), torch.zeros_like(r2))


def sample_quarter_disk(uv: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Area-preserving sampler for the normalized first-quadrant disk target."""
    radius = np.sqrt(2.0 / np.pi) * torch.sqrt(torch.clamp(uv[..., 0], min=eps))
    theta = 0.5 * np.pi * uv[..., 1]
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y], dim=-1)
    pdf = torch.full_like(radius, 2.0)
    return points, pdf


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA images to scalar luminance values."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64, copy=False)
    if arr.ndim != 3:
        raise ValueError("Image target must be a grayscale, RGB, or RGBA image.")

    if arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float64, copy=False)
        weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        return np.tensordot(rgb, weights, axes=([-1], [0]))

    if arr.shape[0] in (3, 4):
        rgb = arr[:3, ...].astype(np.float64, copy=False)
        weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        return np.tensordot(weights, rgb, axes=([0], [0]))

    if arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float64, copy=False)

    raise ValueError("Could not identify image color channels.")


class PiecewiseImageTarget:
    """Piecewise-constant image density and inverse-CDF sampler."""

    def __init__(
        self,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        flip_y: bool = True,
        clip_high_quantile: Optional[float] = None,
        eps: float = 1e-12,
    ) -> None:
        self.path = str(path)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.eps = float(eps)

        image = iio.imread(self.path)
        gray = _to_grayscale(image)
        if flip_y:
            gray = np.flip(gray, axis=0).copy()

        weights = torch.as_tensor(gray, dtype=torch.float64, device=self.device)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)

        if clip_high_quantile is not None:
            if not (0.0 < clip_high_quantile <= 1.0):
                raise ValueError("clip_high_quantile must be in (0, 1].")
            weights = torch.clamp(weights, max=torch.quantile(weights, float(clip_high_quantile)))

        total = weights.sum()
        if total <= 0:
            raise ValueError(f"Image target has zero total intensity: {self.path}")

        self.weights = weights / total
        self.height = int(weights.shape[0])
        self.width = int(weights.shape[1])

        row_masses = self.weights.sum(dim=1)
        self.row_cdf = torch.cumsum(row_masses, dim=0)
        self.row_cdf[-1] = 1.0

        self.col_cdfs = torch.empty_like(self.weights)
        nonzero_rows = row_masses > 0
        self.col_cdfs[nonzero_rows] = torch.cumsum(
            self.weights[nonzero_rows] / row_masses[nonzero_rows, None],
            dim=1,
        )
        self.col_cdfs[nonzero_rows, -1] = 1.0
        if torch.any(~nonzero_rows):
            uniform_cdf = torch.linspace(
                1.0 / self.width,
                1.0,
                self.width,
                dtype=torch.float64,
                device=self.device,
            )
            self.col_cdfs[~nonzero_rows] = uniform_cdf

    def density(self, points: Tensor) -> Tensor:
        """Evaluate the normalized piecewise-constant image density."""
        pts = points.to(device=self.device, dtype=torch.float64)
        x = pts[..., 0]
        y = pts[..., 1]
        inside = _inside_unit_square(pts)

        col = torch.clamp((x * self.width).floor().to(torch.long), 0, self.width - 1)
        row = torch.clamp((y * self.height).floor().to(torch.long), 0, self.height - 1)
        pdf = self.weights[row, col] * float(self.height * self.width)
        pdf = torch.where(inside, pdf, torch.zeros_like(pdf))
        return pdf.to(dtype=points.dtype, device=points.device)

    def sample(self, uv: Tensor) -> Tuple[Tensor, Tensor]:
        """Map uniform samples to the image target."""
        uvs = torch.clamp(uv.to(device=self.device, dtype=torch.float64), 0.0, 1.0 - self.eps)
        orig_shape = uvs.shape[:-1]
        flat = uvs.reshape(-1, 2)
        u = flat[:, 0].contiguous()
        v = flat[:, 1].contiguous()

        row = torch.searchsorted(self.row_cdf, v, right=True)
        row = torch.clamp(row, 0, self.height - 1)
        prev_row = torch.where(row == 0, torch.zeros_like(v), self.row_cdf[row - 1])
        next_row = self.row_cdf[row]
        local_y = (v - prev_row) / torch.clamp(next_row - prev_row, min=self.eps)

        row_cdf = self.col_cdfs[row]
        col = torch.searchsorted(row_cdf, u[:, None], right=True).squeeze(-1)
        col = torch.clamp(col, 0, self.width - 1)
        prev_col = torch.where(
            col == 0,
            torch.zeros_like(u),
            row_cdf.gather(1, (col - 1).clamp_min(0)[:, None]).squeeze(-1),
        )
        next_col = row_cdf.gather(1, col[:, None]).squeeze(-1)
        local_x = (u - prev_col) / torch.clamp(next_col - prev_col, min=self.eps)

        x = (col.to(torch.float64) + local_x) / float(self.width)
        y = (row.to(torch.float64) + local_y) / float(self.height)
        points = torch.stack([x, y], dim=-1).reshape(*orig_shape, 2)
        pdf = self.density(points).reshape(*orig_shape)
        return points.to(dtype=uv.dtype, device=uv.device), pdf.to(dtype=uv.dtype, device=uv.device)


def image_target(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    flip_y: bool = True,
    clip_high_quantile: Optional[float] = None,
) -> Target2D:
    """Build an image-backed target."""
    target = PiecewiseImageTarget(
        path,
        device=device,
        flip_y=flip_y,
        clip_high_quantile=clip_high_quantile,
    )
    return Target2D(name="image", density=target.density, sample=target.sample)


def get_target(
    name: str,
    *,
    image_path: str | Path | None = None,
    device: str | torch.device | None = None,
    image_flip_y: bool = True,
    image_clip_high_quantile: Optional[float] = None,
) -> Target2D:
    """Return a supported target by name."""
    key = name.strip().lower()
    if key == "uniform":
        return Target2D(name="uniform", density=uniform_density, sample=sample_uniform)
    if key == "bilinear":
        return Target2D(name="bilinear", density=bilinear_density, sample=sample_bilinear)
    if key in {"quarter_disk", "quarter-disk"}:
        return Target2D(name="quarter_disk", density=quarter_disk_density, sample=sample_quarter_disk)
    if key == "image":
        if image_path is None:
            raise ValueError("image target requires image_path.")
        return image_target(
            image_path,
            device=device,
            flip_y=image_flip_y,
            clip_high_quantile=image_clip_high_quantile,
        )
    raise ValueError(f"Unknown target '{name}'. Expected uniform, bilinear, quarter_disk, or image.")
