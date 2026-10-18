"""2D discrepancy utilities for uniform and non-uniform point sets."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from targets import Target2D


def _iter_slices(total_size: int, chunk_size: Optional[int]):
    if total_size <= 0:
        return
    if chunk_size is None or chunk_size <= 0 or chunk_size >= total_size:
        yield slice(0, total_size)
        return
    for start in range(0, total_size, int(chunk_size)):
        yield slice(start, min(start + int(chunk_size), total_size))


def sample_anchored_rectangles(
    num: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device,
    min_size: float = 1e-8,
    max_size: float = 1.0,
) -> torch.Tensor:
    """Sample rectangles anchored at the origin in [0, 1]^2."""
    if num < 1:
        raise ValueError("num must be positive.")
    if min_size < 0 or max_size <= min_size:
        raise ValueError("Expected 0 <= min_size < max_size.")

    dev = torch.device(device)
    bottom_left = torch.zeros((int(num), 2), dtype=dtype, device=dev)
    top_right = torch.rand((int(num), 2), dtype=dtype, device=dev)
    top_right = top_right * (max_size - min_size) + min_size
    return torch.stack([bottom_left, top_right], dim=1)


def sample_all_corner_anchored_rectangles(
    num: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device,
    min_size: float = 1e-8,
    max_size: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample rectangles anchored at all four unit-square corners."""
    if num < 1:
        raise ValueError("num must be positive.")
    if num % 4 != 0:
        raise ValueError("num must be divisible by 4.")
    if min_size < 0 or max_size <= min_size:
        raise ValueError("Expected 0 <= min_size < max_size.")

    dev = torch.device(device)
    base_count = int(num) // 4
    point = torch.rand((base_count, 2), dtype=dtype, device=dev)
    point = point * (max_size - min_size) + min_size

    anchors = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=dtype,
        device=dev,
    )
    anchors = anchors.repeat(base_count, 1)
    repeated_points = point.repeat_interleave(4, dim=0)

    bottom_left = torch.minimum(anchors, repeated_points)
    top_right = torch.maximum(anchors, repeated_points)
    rectangles = torch.stack([bottom_left, top_right], dim=1)
    return rectangles, anchors


def outside_rectangle_mask(points: torch.Tensor, rectangles: torch.Tensor) -> torch.Tensor:
    """Return a mask with shape (P, R), true where each point is outside each rectangle."""
    bottom_left = rectangles[:, 0, :].unsqueeze(0)
    top_right = rectangles[:, 1, :].unsqueeze(0)
    pts = points.unsqueeze(1)
    inside_x = (pts[..., 0] >= bottom_left[..., 0]) & (pts[..., 0] <= top_right[..., 0])
    inside_y = (pts[..., 1] >= bottom_left[..., 1]) & (pts[..., 1] <= top_right[..., 1])
    return ~(inside_x & inside_y)


def rectangle_counts(
    points: torch.Tensor,
    rectangles: torch.Tensor,
    *,
    point_chunk_size: Optional[int] = None,
    region_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Count points inside each rectangle without materializing all pairs if chunked."""
    pts = points.to(dtype=torch.float64)
    rects = rectangles.to(device=pts.device, dtype=pts.dtype)
    counts = torch.zeros(rects.shape[0], dtype=pts.dtype, device=pts.device)

    for region_slice in _iter_slices(int(rects.shape[0]), region_chunk_size):
        rect_block = rects[region_slice]
        block_counts = torch.zeros(rect_block.shape[0], dtype=pts.dtype, device=pts.device)
        for point_slice in _iter_slices(int(pts.shape[0]), point_chunk_size):
            inside = ~outside_rectangle_mask(pts[point_slice], rect_block)
            block_counts += inside.sum(dim=0).to(dtype=pts.dtype)
        counts[region_slice] = block_counts

    return counts


def rectangle_areas_uniform(rectangles: torch.Tensor) -> torch.Tensor:
    """Compute Lebesgue area for axis-aligned rectangles."""
    bottom_left = rectangles[:, 0, :]
    top_right = rectangles[:, 1, :]
    side_lengths = top_right - bottom_left
    return side_lengths[:, 0] * side_lengths[:, 1]


def rectangle_areas_bilinear(rectangles: torch.Tensor) -> torch.Tensor:
    """Compute exact masses for p(x, y) = 4xy."""
    bottom_left = rectangles[:, 0, :]
    top_right = rectangles[:, 1, :]
    x0, y0 = bottom_left[:, 0], bottom_left[:, 1]
    x1, y1 = top_right[:, 0], top_right[:, 1]
    x0, x1 = torch.minimum(x0, x1), torch.maximum(x0, x1)
    y0, y1 = torch.minimum(y0, y1), torch.maximum(y0, y1)
    return (x1.square() - x0.square()) * (y1.square() - y0.square())


def rectangle_areas_monte_carlo(
    rectangles: torch.Tensor,
    target: Target2D,
    space_samples: torch.Tensor,
    *,
    batch_size: int = 2048,
    space_batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Estimate target mass inside rectangles from uniform space samples."""
    rects = rectangles.to(dtype=torch.float64)
    samples = space_samples.to(device=rects.device, dtype=rects.dtype)
    rho = target.density(samples).to(device=rects.device, dtype=rects.dtype)

    areas = []
    for region_slice in _iter_slices(int(rects.shape[0]), batch_size):
        rect_block = rects[region_slice]
        uniform_area = rectangle_areas_uniform(rect_block)
        weighted_sum = torch.zeros(rect_block.shape[0], dtype=rects.dtype, device=rects.device)
        inside_count = torch.zeros(rect_block.shape[0], dtype=rects.dtype, device=rects.device)

        for point_slice in _iter_slices(int(samples.shape[0]), space_batch_size):
            inside = ~outside_rectangle_mask(samples[point_slice], rect_block)
            weighted_sum += (inside * rho[point_slice, None]).sum(dim=0)
            inside_count += inside.sum(dim=0).to(dtype=rects.dtype)

        areas.append(weighted_sum * uniform_area / torch.clamp_min(inside_count, 1.0))

    return torch.cat(areas, dim=0)


def l2_discrepancy_warnock(points: torch.Tensor, *, pair_batch_size: int = 2048) -> torch.Tensor:
    """Warnock-style L2 discrepancy for points in [0, 1]^2."""
    pts = points.to(dtype=torch.float64)
    npoints = int(pts.shape[0])
    dim = int(pts.shape[1])
    if npoints < 1:
        raise ValueError("points must be non-empty.")

    prod1 = torch.prod(1.0 - pts.square(), dim=1)
    sum1 = torch.sum(prod1)

    pair_sum = torch.zeros((), dtype=pts.dtype, device=pts.device)
    pair_batch_size = max(1, int(pair_batch_size))
    for i_slice in _iter_slices(npoints, pair_batch_size):
        pi = pts[i_slice]
        for j_slice in _iter_slices(npoints, pair_batch_size):
            pj = pts[j_slice]
            pairwise_max = torch.maximum(pi[:, None, :], pj[None, :, :])
            pair_sum += torch.prod(1.0 - pairwise_max, dim=2).sum()

    value = (
        math.pow(3.0, -dim)
        - (1.0 / npoints) * math.pow(2.0, 1.0 - dim) * sum1
        + (1.0 / (npoints * npoints)) * pair_sum
    )
    return torch.sqrt(torch.clamp(value, min=0.0))


def star_discrepancy_2d(points: torch.Tensor) -> torch.Tensor:
    """Exact O(N^2) anchored star discrepancy for 2D points in [0, 1]^2."""
    device = points.device
    point_set = points.to(dtype=torch.float64)
    npoints = point_set.size(0)

    if npoints == 0:
        return torch.tensor(0.0, device=device, dtype=point_set.dtype)

    inv_n = 1.0 / npoints
    x = point_set[:, 0]
    y = point_set[:, 1]

    ys = torch.unique(torch.cat([y, torch.tensor([1.0], device=device, dtype=point_set.dtype)]))
    ys, _ = torch.sort(ys)
    num_y = ys.numel()

    sort_idx = torch.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    y_idx = torch.searchsorted(ys, y_sorted, right=False)

    x_unique, counts = torch.unique_consecutive(x_sorted, return_counts=True)
    start_idx = torch.cumsum(
        torch.cat([torch.tensor([0], device=device, dtype=counts.dtype), counts[:-1]]),
        dim=0,
    )

    xs_candidates = torch.unique(
        torch.cat([x_unique, torch.tensor([1.0], device=device, dtype=point_set.dtype)])
    )
    xs_candidates, _ = torch.sort(xs_candidates)

    bins = torch.zeros(num_y, device=device, dtype=point_set.dtype)
    prefix_le = torch.zeros(num_y, device=device, dtype=point_set.dtype)
    discrepancy = torch.tensor(0.0, device=device, dtype=point_set.dtype)

    group_ptr = 0
    num_groups = x_unique.numel()

    for u in xs_candidates:
        prefix_lt = prefix_le.clone()

        if group_ptr < num_groups and (x_unique[group_ptr] == u).item():
            start = int(start_idx[group_ptr].item())
            end = start + int(counts[group_ptr].item())
            idxs = y_idx[start:end]
            bins.index_add_(0, idxs, torch.ones_like(idxs, dtype=bins.dtype))
            prefix_le = torch.cumsum(bins, dim=0)
            group_ptr += 1

        cnt_lt_lt = torch.cat(
            [torch.zeros(1, device=device, dtype=point_set.dtype), prefix_lt[:-1]],
            dim=0,
        )
        cnt_lt_le = prefix_lt
        cnt_le_lt = torch.cat(
            [torch.zeros(1, device=device, dtype=point_set.dtype), prefix_le[:-1]],
            dim=0,
        )
        cnt_le_le = prefix_le

        volume = u * ys
        d1 = torch.abs(cnt_lt_lt * inv_n - volume)
        d2 = torch.abs(cnt_lt_le * inv_n - volume)
        d3 = torch.abs(cnt_le_lt * inv_n - volume)
        d4 = torch.abs(cnt_le_le * inv_n - volume)

        discrepancy = torch.maximum(discrepancy, torch.stack([d1, d2, d3, d4]).max())

    return discrepancy


def uniform_discrepancies(
    points: torch.Tensor,
    *,
    pair_batch_size: int = 2048,
) -> Tuple[float, float]:
    """Return mean-square L2 and star discrepancy values as Python floats."""
    l2 = l2_discrepancy_warnock(points, pair_batch_size=pair_batch_size)
    star = star_discrepancy_2d(points)
    return float(l2.detach().cpu().item()), float(star.detach().cpu().item())


def estimate_nonuniform_discrepancies(
    points: torch.Tensor,
    target: Target2D,
    *,
    nrectangles: int,
    nspace_samples: int,
    reps: int = 1,
    areas_batch_size: int = 2048,
    areas_space_batch_size: Optional[int] = None,
    point_chunk_size: Optional[int] = None,
    region_chunk_size: Optional[int] = None,
) -> Tuple[float, float]:
    """Estimate L2 and star discrepancy against a non-uniform target."""
    if nrectangles < 1 or nspace_samples < 1 or reps < 1:
        raise ValueError("nrectangles, nspace_samples, and reps must be positive.")

    pts = points.to(dtype=torch.float64)
    npoints = int(pts.shape[0])
    l2_total = torch.zeros((), dtype=pts.dtype, device=pts.device)
    star_total = torch.zeros((), dtype=pts.dtype, device=pts.device)

    with torch.no_grad():
        for _ in range(int(reps)):
            rectangles = sample_anchored_rectangles(
                int(nrectangles),
                dtype=pts.dtype,
                device=pts.device,
            )

            if target.name == "uniform":
                masses = rectangle_areas_uniform(rectangles)
            elif target.name == "bilinear":
                masses = rectangle_areas_bilinear(rectangles)
            else:
                space_samples = torch.rand((int(nspace_samples), 2), dtype=pts.dtype, device=pts.device)
                masses = rectangle_areas_monte_carlo(
                    rectangles,
                    target,
                    space_samples,
                    batch_size=areas_batch_size,
                    space_batch_size=areas_space_batch_size,
                )

            empirical = rectangle_counts(
                pts,
                rectangles,
                point_chunk_size=point_chunk_size,
                region_chunk_size=region_chunk_size,
            ) / npoints
            diff = empirical - masses
            l2_total += torch.sqrt(diff.square().mean())
            star_total += torch.abs(diff).max()

    l2 = l2_total / int(reps)
    star = star_total / int(reps)
    return float(l2.cpu().item()), float(star.cpu().item())
