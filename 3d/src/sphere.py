"""Minimal spherical geometry used by the 3D discrepancy optimizer."""

from __future__ import annotations

import math

import torch


def _gradient_abs(value: torch.Tensor) -> torch.Tensor:
    """Absolute value with a non-zero one-sided derivative at zero."""
    return torch.where(value < 0, -value, value)


def _gradient_sign(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))


def _soft_sqrt_01(value: torch.Tensor) -> torch.Tensor:
    epsilon = torch.as_tensor(1e-6, dtype=value.dtype, device=value.device)
    return torch.sqrt((value + epsilon) / (1.0 + epsilon))


def mirror_repeat_warping(latents: torch.Tensor) -> torch.Tensor:
    """Mirror arbitrary 2D coordinates back into the unit square."""
    floor_latents = torch.floor(latents)
    mirror_times = torch.abs(floor_latents).sum(dim=-1).to(torch.long)
    odd = (mirror_times % 2) == 1
    return torch.where(
        odd.unsqueeze(-1),
        1.0 - (latents - floor_latents),
        latents - floor_latents,
    )


def octahedral_square_to_sphere(latents: torch.Tensor) -> torch.Tensor:
    """Differentiable equal-area-style octahedral square-to-sphere mapping."""
    uv = latents * 2.0 - 1.0
    abs_uv = _gradient_abs(uv)
    diagonal = 1.0 - abs_uv.sum(dim=-1)
    radius = 1.0 - _gradient_abs(diagonal)
    radius_squared = radius.square()

    z = torch.where(diagonal < 0, radius_squared - 1.0, 1.0 - radius_squared)
    phi_prime = torch.zeros_like(radius)
    valid = radius > 1e-6
    phi_prime[valid] = 0.25 * math.pi * (
        (abs_uv[valid, 1] - abs_uv[valid, 0]) / radius[valid] + 1.0
    )

    sin_phi = _gradient_sign(uv[:, 1]) * torch.sin(phi_prime)
    cos_phi = _gradient_sign(uv[:, 0]) * torch.cos(phi_prime)
    sin_theta = radius * _soft_sqrt_01(2.0 - radius_squared)
    directions = torch.stack((cos_phi * sin_theta, sin_phi * sin_theta, z), dim=-1)
    return torch.nn.functional.normalize(directions, dim=-1)


def latent_to_direction(latents: torch.Tensor) -> torch.Tensor:
    return octahedral_square_to_sphere(mirror_repeat_warping(latents))


def uniform_sample_sphere(random_values: torch.Tensor) -> torch.Tensor:
    if random_values.shape[-1] != 2:
        raise ValueError("Sphere sampling requires random values with shape [..., 2].")
    z = 1.0 - 2.0 * random_values[..., 0]
    radius = torch.sqrt(torch.clamp_min(1.0 - z.square(), 0.0))
    phi = 2.0 * math.pi * random_values[..., 1]
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=-1)


def uniform_sample_hemisphere(random_values: torch.Tensor) -> torch.Tensor:
    if random_values.shape[-1] != 2:
        raise ValueError("Hemisphere sampling requires random values with shape [..., 2].")
    z = random_values[..., 0]
    radius = torch.sqrt(torch.clamp_min(1.0 - z.square(), 0.0))
    phi = 2.0 * math.pi * random_values[..., 1]
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=-1)


def sample_spherical_caps(
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return caps as [center_x, center_y, center_z, angular_radius]."""
    center_random = torch.rand((count, 2), device=device, dtype=dtype, generator=generator)
    centers = uniform_sample_sphere(center_random)
    angular_radius = torch.rand((count, 1), device=device, dtype=dtype, generator=generator)
    angular_radius = torch.clamp_min(angular_radius * math.pi, 1e-8)
    return torch.cat((centers, angular_radius), dim=-1)


def uniform_cap_probability(caps: torch.Tensor) -> torch.Tensor:
    """Normalized spherical-cap area under the uniform sphere distribution."""
    return torch.sin(0.5 * caps[:, 3]).square()


class _StableAcos(torch.autograd.Function):
    """Exact forward angles; bound only the derivative at parallel directions."""

    @staticmethod
    def forward(ctx, cosine):
        cosine = cosine.clamp(-1.0, 1.0)
        ctx.save_for_backward(cosine)
        return torch.acos(cosine)

    @staticmethod
    def backward(ctx, gradient):
        (cosine,) = ctx.saved_tensors
        return -gradient / torch.sqrt((1.0 - cosine.square()).clamp_min(2e-4))


def point_to_cap_distance(caps: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Signed angular distance [num_points, num_caps], positive inside a cap."""
    dot_value = points @ caps[:, :3].transpose(0, 1)
    radians = _StableAcos.apply(dot_value)
    return caps[:, 3].unsqueeze(0) - radians


def sample_uniform_in_caps(
    caps: torch.Tensor,
    samples_per_cap: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniformly sample directions inside each cap, returning [R, S, 3]."""
    count = caps.shape[0]
    random_values = torch.rand(
        (count, samples_per_cap, 2),
        dtype=caps.dtype,
        device=caps.device,
        generator=generator,
    )
    z_min = torch.cos(caps[:, 3]).unsqueeze(-1)
    local_z = z_min + (1.0 - z_min) * random_values[..., 0]
    phi = 2.0 * math.pi * random_values[..., 1]
    local_radius = torch.sqrt(torch.clamp_min(1.0 - local_z.square(), 0.0))
    local_x = local_radius * torch.cos(phi)
    local_y = local_radius * torch.sin(phi)

    centers = caps[:, :3]
    helper_z = torch.zeros_like(centers)
    helper_z[:, 2] = 1.0
    helper_x = torch.zeros_like(centers)
    helper_x[:, 0] = 1.0
    near_z = torch.abs(centers[:, 2]) > 0.999
    helper = torch.where(near_z.unsqueeze(-1), helper_x, helper_z)
    tangent_x = torch.nn.functional.normalize(torch.cross(helper, centers, dim=-1), dim=-1)
    tangent_y = torch.cross(centers, tangent_x, dim=-1)

    return (
        local_x.unsqueeze(-1) * tangent_x.unsqueeze(1)
        + local_y.unsqueeze(-1) * tangent_y.unsqueeze(1)
        + local_z.unsqueeze(-1) * centers.unsqueeze(1)
    )


class HardIndicator(torch.autograd.Function):
    """Hard forward with the original scaled STE (deliberately omits tau)."""

    @staticmethod
    def forward(ctx, distance: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.save_for_backward(distance)
        ctx.scale = scale
        return distance.ge(0).to(distance.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (distance,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(distance * ctx.scale)
        surrogate = sigmoid * (1.0 - sigmoid)
        return surrogate * grad_output, None


hard_indicator = HardIndicator.apply


def cap_indicator(caps: torch.Tensor, points: torch.Tensor, tau: float) -> torch.Tensor:
    """Exact dot-product membership, with a bounded angular surrogate backward."""
    cosine = points @ caps[:, :3].T
    inside = (cosine >= torch.cos(caps[:, 3])[None]).to(points.dtype)
    distance = caps[:, 3][None] - _StableAcos.apply(cosine)
    surrogate = hard_indicator(distance, tau)
    return surrogate + (inside - surrogate).detach()
