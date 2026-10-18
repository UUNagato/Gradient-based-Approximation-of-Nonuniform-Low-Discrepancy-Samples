"""Plotting helpers for 2D reproducibility results."""

from __future__ import annotations

from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from targets import Target2D


def evaluate_density_grid(
    target: Target2D,
    *,
    resolution: int | tuple[int, int] = 256,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a target density on a regular grid over [0, 1]^2."""
    if isinstance(resolution, int):
        height = width = int(resolution)
    else:
        height, width = int(resolution[0]), int(resolution[1])
    if height < 1 or width < 1:
        raise ValueError("resolution must be positive.")

    dev = torch.device(device)
    xs = torch.linspace(0.0, 1.0, width, dtype=torch.float64, device=dev)
    ys = torch.linspace(0.0, 1.0, height, dtype=torch.float64, device=dev)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    with torch.no_grad():
        density = target.density(points).reshape(height, width)

    return (
        xs.detach().cpu().numpy(),
        ys.detach().cpu().numpy(),
        density.detach().cpu().numpy(),
    )


def plot_density(
    target: Target2D,
    *,
    resolution: int | tuple[int, int] = 256,
    device: str | torch.device,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
):
    """Plot a target density over the unit square."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    _, _, density = evaluate_density_grid(target, resolution=resolution, device=device)
    image = ax.imshow(
        density,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap=cmap,
        aspect="equal",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"{target.name} density")
    return ax, image


def plot_point_set(
    points: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    marker_size: float = 12.0,
    color: Optional[str] = None,
    alpha: float = 0.85,
):
    """Scatter plot one 2D point set."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {pts.shape}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    scatter_kwargs = {"s": marker_size, "alpha": alpha, "linewidths": 0}
    if color is not None:
        scatter_kwargs["c"] = color
    ax.scatter(pts[:, 0], pts[:, 1], **scatter_kwargs)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def plot_point_sets_grid(
    point_sets: Mapping[str, np.ndarray],
    *,
    max_columns: int = 4,
    marker_size: float = 12.0,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
):
    """Plot multiple named point sets in a compact grid."""
    if len(point_sets) == 0:
        raise ValueError("point_sets must be non-empty.")

    names = list(point_sets.keys())
    nplots = len(names)
    ncols = min(int(max_columns), nplots)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for index, name in enumerate(names):
        row = index // ncols
        col = index % ncols
        plot_point_set(point_sets[name], ax=axes[row][col], title=name, marker_size=marker_size)

    for index in range(nplots, nrows * ncols):
        row = index // ncols
        col = index % ncols
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig, axes


def plot_discrepancy_curves(
    series: Mapping[str, Mapping[str, np.ndarray]],
    *,
    metric: str,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xscale: str = "log",
    yscale: str = "log",
):
    """Plot discrepancy curves from loaded summary dictionaries."""
    if metric not in {"l2_mean", "star_mean"}:
        raise ValueError("metric must be 'l2_mean' or 'star_mean'.")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    for label, values in series.items():
        ns = np.asarray(values["ns"], dtype=np.float64)
        ys = np.asarray(values[metric], dtype=np.float64)
        ax.plot(ns, ys, marker="o", linewidth=1.5, markersize=4, label=label)

    ax.set_xlabel("N")
    ax.set_ylabel("L2 discrepancy" if metric == "l2_mean" else "Star discrepancy")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


def show_image_target(
    image: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "gray",
):
    """Display an image used as a target density."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.asarray(image), cmap=cmap, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
