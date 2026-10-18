"""Train a conditional neural point-set generator for BRDF or environment lighting.

BRDF points live in local space with normal +Z. Encoded inputs concatenate
``[offset_xy, (wo_xy + 1)/2]``; Env uses ``[offset_xy, (normal_xyz + 1)/2]``.
BRDF view directions vary over the upper local hemisphere.
The network predicts a residual warp of a shifted Sobol point set.  Training
minimizes spherical-cap discrepancy with a memory-batched backward pass.

This compact example is extracted from:
  - exps/tests_3d/spherical_caps_non_uniform_rendering.py
  - exps/discrepancy/train_environmentmap.py
  - exps/discrepancy/train_disneybrdf_viewdir.py
  - models/pointsetmodel.py (AutoDecoderPointsetsV2cmp)

BRDF training uses shared scrambled Sobol integration and independent fixed
hemisphere quadrature for normalization, without a polar-only LUT.
"""

from __future__ import annotations

from pointsetmodel import AutoDecoderPointsetsV2cmp
from brdf_validation import BRDFQuadrature, HeldOutValidation, shared_sobol_masses

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from envmap import EnvironmentMap
from sphere import (
    cap_indicator,
    latent_to_direction,
    sample_spherical_caps,
    uniform_sample_sphere,
    uniform_sample_hemisphere,
)


THIS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENVMAP = THIS_DIR / "data" / "targets" / "example_envmap.npy"


def slices(total: int, chunk_size: int):
    for start in range(0, total, chunk_size):
        yield slice(start, min(start + chunk_size, total))


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vector, dim=-1, eps=1e-8)


def tangent_frame(normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a stable right-handed frame whose z axis is ``normal``."""
    normal = normalize(normal)
    helper_z = torch.zeros_like(normal)
    helper_z[..., 2] = 1.0
    helper_x = torch.zeros_like(normal)
    helper_x[..., 0] = 1.0
    helper = torch.where((normal[..., 2].abs() > 0.999).unsqueeze(-1), helper_x, helper_z)
    tangent = normalize(torch.cross(helper, normal, dim=-1))
    bitangent = torch.cross(normal, tangent, dim=-1)
    return tangent, bitangent


def world_to_local(directions: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    tangent, bitangent = tangent_frame(normal)
    return torch.stack(
        (
            (directions * tangent).sum(dim=-1),
            (directions * bitangent).sum(dim=-1),
            (directions * normal).sum(dim=-1),
        ),
        dim=-1,
    )


def schlick_fresnel(cosine: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(1.0 - cosine, 0.0, 1.0)
    return value.pow(5)


def disney_brdf(
    wi: torch.Tensor,
    wo: torch.Tensor,
    *,
    roughness: float,
    metallic: float,
    anisotropic: float,
    subsurface: float,
) -> torch.Tensor:
    """Scalar Disney BRDF in the local frame; specular is always zero."""
    ndotl = wi[..., 2]
    ndotv = wo[..., 2]
    half_vector = normalize(wi + wo)
    ndoth = half_vector[..., 2]
    ldoth = (wi * half_vector).sum(dim=-1)

    fl = schlick_fresnel(ndotl)
    fv = schlick_fresnel(ndotv)
    fd90 = 0.5 + 2.0 * ldoth.square() * roughness
    diffuse = torch.lerp(torch.ones_like(fd90), fd90, fl)
    diffuse = diffuse * torch.lerp(torch.ones_like(fd90), fd90, fv)

    fss90 = ldoth.square() * roughness
    fss = torch.lerp(torch.ones_like(fss90), fss90, fl)
    fss = fss * torch.lerp(torch.ones_like(fss90), fss90, fv)
    denominator = torch.clamp_min(ndotl + ndotv, 1e-5)
    subsurface_term = 1.25 * (fss * (1.0 / denominator - 0.5) + 0.5)

    aspect = math.sqrt(max(0.1, 1.0 - anisotropic * 0.9))
    alpha_x = max(0.001, roughness * roughness / aspect)
    alpha_y = max(0.001, roughness * roughness * aspect)
    hdotx, hdoty = half_vector[..., 0], half_vector[..., 1]
    distribution_denominator = math.pi * alpha_x * alpha_y * (
        (hdotx / alpha_x).square() + (hdoty / alpha_y).square() + ndoth.square()
    ).square()
    distribution = 1.0 / torch.clamp_min(distribution_denominator, 1e-8)

    def smith_g(vector: torch.Tensor) -> torch.Tensor:
        ndot = vector[..., 2]
        denominator_g = ndot + torch.sqrt(
            (vector[..., 0] * alpha_x).square()
            + (vector[..., 1] * alpha_y).square()
            + ndot.square()
        )
        return 1.0 / torch.clamp_min(denominator_g, 1e-4)

    fresnel = torch.lerp(
        torch.zeros_like(ldoth),
        torch.ones_like(ldoth),
        metallic + (1.0 - metallic) * schlick_fresnel(ldoth),
    )
    diffuse_or_subsurface = torch.lerp(diffuse, subsurface_term, subsurface)
    value = (
        (1.0 - metallic) * diffuse_or_subsurface / math.pi
        + distribution * smith_g(wi) * smith_g(wo) * fresnel
    )
    valid = (ndotl > 0.0) & (ndotv > 0.0)
    return torch.where(valid, value, torch.zeros_like(value))


class ConditionalTarget:
    """BRDF conditioned by local wo, or Env conditioned by world normal."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.args = args
        self.case = args.case
        self.envmap = None
        if self.case == "env":
            self.envmap = EnvironmentMap(args.envmap, device=device)

    @torch.no_grad()
    def weights(self, directions: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = normalize(condition.to(directions.dtype))
        if self.case == "env":
            assert self.envmap is not None
            cosine = torch.clamp_min((directions * condition).sum(dim=-1), 0.0)
            # Dividing radiance by its sphere integral cancels in normalization,
            # while making the estimator independent of the image exposure scale.
            return self.envmap.pdf(directions) * cosine

        if condition[2] <= 0:
            raise ValueError("BRDF wo must lie in the upper local hemisphere.")
        brdf = disney_brdf(
            directions,
            condition.expand_as(directions),
            roughness=self.args.roughness,
            metallic=self.args.metallic,
            anisotropic=self.args.anisotropic,
            subsurface=self.args.subsurface,
        )
        return brdf * torch.clamp_min(directions[..., 2], 0.0)

    def _ggx_parameters(self) -> tuple[float, float]:
        aspect = math.sqrt(1.0 - 0.9 * self.args.anisotropic)
        alpha = self.args.roughness ** 2
        return max(0.001, alpha / aspect), max(0.001, alpha * aspect)

    def _ggx_pdf(self, directions: torch.Tensor, wo: torch.Tensor) -> torch.Tensor:
        ax, ay = self._ggx_parameters()
        half = normalize(directions + wo)
        half = torch.where((half[:, 2] < 0)[:, None], -half, half)
        denominator = ((half[:, 0] / ax).square() + (half[:, 1] / ay).square()
                       + half[:, 2].square()).square()
        distribution = 1.0 / (math.pi * ax * ay * denominator.clamp_min(1e-30))
        jacobian = 4.0 * (half * wo).sum(-1).abs()
        return distribution * half[:, 2] / jacobian.clamp_min(1e-30)

    @torch.no_grad()
    def sample_proposal(self, count: int, condition: torch.Tensor, *, generator: torch.Generator):
        """Stratified 50/50 target-specific and cosine mixture, with full PDF.

        GGX NDF reflection may generate below-surface points: retain them with
        zero target weight rather than silently conditioning the proposal.
        """
        if count < 2:
            raise ValueError("target-samples must be at least 2 for the mixture.")
        # Float32 reflection can round a near-perpendicular GGX sample to
        # exactly -wo, destroying its half-vector and giving a zero PDF.
        condition = normalize(condition.double())
        primary_count = count // 2
        cosine_count = count - primary_count
        random = torch.rand((count, 2), device=condition.device, dtype=condition.dtype,
                            generator=generator)
        r = torch.sqrt(random[:cosine_count, 0])
        phi = 2.0 * math.pi * random[:cosine_count, 1]
        cosine_dirs = torch.stack((r * torch.cos(phi), r * torch.sin(phi),
                                   torch.sqrt(1.0 - random[:cosine_count, 0])), -1)
        if self.case == "env":
            tangent, bitangent = tangent_frame(condition)
            cosine_dirs = (cosine_dirs[:, :1] * tangent + cosine_dirs[:, 1:2] * bitangent
                           + cosine_dirs[:, 2:] * condition)
            primary_dirs, _ = self.envmap.sample(primary_count, generator=generator)
        else:
            ax, ay = self._ggx_parameters()
            uv = random[cosine_count:]
            slope = torch.sqrt(uv[:, 0] / (1.0 - uv[:, 0]))
            phi = 2.0 * math.pi * uv[:, 1]
            half = normalize(torch.stack((ax * slope * torch.cos(phi),
                                          ay * slope * torch.sin(phi), torch.ones_like(slope)), -1))
            primary_dirs = normalize(2.0 * (half * condition).sum(-1, keepdim=True) * half - condition)
        directions = torch.cat((primary_dirs, cosine_dirs))
        if self.case == "env":
            primary_pdf = self.envmap.pdf(directions)
            cosine_pdf = (directions * condition).sum(-1).clamp_min(0.0) / math.pi
        else:
            primary_pdf = self._ggx_pdf(directions, condition)
            cosine_pdf = directions[:, 2].clamp_min(0.0) / math.pi
        fraction = primary_count / count
        return directions, fraction * primary_pdf + (1.0 - fraction) * cosine_pdf


def condition_dim(args: argparse.Namespace) -> int:
    direction_dim = (2 if args.view_condition else 0) if args.case == "brdf" else (3 if args.normal_condition else 0)
    return direction_dim + (2 if args.offset_condition else 0)


def sample_condition(args: argparse.Namespace, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Local wo for BRDF; world surface normal for Env."""
    if args.case == "brdf":
        if args.fixed_view is not None:
            return normalize(torch.tensor(args.fixed_view, device=device, dtype=torch.float32))
        if args.view_angle is not None:
            theta = math.radians(args.view_angle)
            return torch.tensor([math.sin(theta), 0.0, math.cos(theta)], device=device)
        random_values = torch.rand((1, 2), device=device, generator=generator)
        random_values[:, 0] = 1.0 - random_values[:, 0]  # strictly positive local z
        return uniform_sample_hemisphere(random_values)[0]
    if args.fixed_normal is not None:
        normal = torch.tensor(args.fixed_normal, device=device, dtype=torch.float32)
        if normal.norm() < 1e-8:
            raise ValueError("--fixed-normal must be non-zero.")
        return normalize(normal)
    random_values = torch.rand((1, 2), device=device, generator=generator)
    return uniform_sample_sphere(random_values)[0]


def sample_offset(args: argparse.Namespace, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    if args.fixed_offset is not None:
        return torch.remainder(torch.tensor(args.fixed_offset, device=device), 1.0)
    return torch.rand(2, device=device, generator=generator)


@torch.no_grad()
def estimate_cap_masses(
    target: ConditionalTarget,
    caps: torch.Tensor,
    space_directions: torch.Tensor,
    condition: torch.Tensor,
    cap_chunk_size: int,
    proposal_pdf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Self-normalized IS estimate, consistent but finite-sample biased.

    Omit proposal_pdf only for uniform-sphere samples (the constant cancels).
    """
    weights = torch.clamp_min(target.weights(space_directions, condition), 0.0).double()
    if proposal_pdf is not None:
        if not torch.all(torch.isfinite(proposal_pdf) & (proposal_pdf > 0)):
            raise FloatingPointError("Integration proposal PDFs must be finite and positive.")
        weights = weights / proposal_pdf.double()
    total_weight = weights.sum()
    if not torch.isfinite(total_weight) or total_weight <= 0:
        raise FloatingPointError("Target has no sampled positive mass; check illumination/condition or increase target-samples.")
    target.last_effective_sample_size = float((total_weight.square() / weights.square().sum()).cpu())
    masses = torch.empty(len(caps), device=caps.device, dtype=caps.dtype)
    # Keep the high-precision directions for PDF/weight evaluation above;
    # cap membership uses the cap geometry's precision.
    cap_directions = space_directions.to(caps.dtype)
    for cap_slice in slices(len(caps), cap_chunk_size):
        cap_block = caps[cap_slice]
        inside = cap_directions @ cap_block[:, :3].T >= torch.cos(cap_block[:, 3])[None]
        masses[cap_slice] = (inside * weights[:, None]).sum(dim=0) / total_weight
    return masses


def model_directions(
    model: AutoDecoderPointsetsV2cmp,
    base_uv: torch.Tensor,
    condition: torch.Tensor,
    offset: torch.Tensor,
    point_slice: slice,
    args: argparse.Namespace,
) -> torch.Tensor:
    # Original input order and scaling: offset, then direction mapped to [0,1].
    use_direction = args.view_condition if args.case == "brdf" else args.normal_condition
    direction = condition[:2] if args.case == "brdf" else condition
    param = direction * 0.5 + 0.5 if use_direction else condition.new_empty(0)
    model_offset = offset if args.offset_condition else offset.new_empty(0)
    residual = model(model_offset, param)
    latents = base_uv + offset + residual
    return latent_to_direction(latents[point_slice])


@torch.no_grad()
def accumulate_cap_sums(
    model: AutoDecoderPointsetsV2cmp,
    base_uv: torch.Tensor,
    condition: torch.Tensor,
    offset: torch.Tensor,
    caps: torch.Tensor,
    tau: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    sums = torch.zeros(len(caps), device=base_uv.device, dtype=base_uv.dtype)
    for point_slice in slices(len(base_uv), args.point_chunk_size):
        directions = model_directions(model, base_uv, condition, offset, point_slice, args)
        for cap_slice in slices(len(caps), args.cap_chunk_size):
            sums[cap_slice] += cap_indicator(caps[cap_slice], directions, tau).sum(dim=0)
    return sums


def batched_loss_and_backward(
    model: AutoDecoderPointsetsV2cmp,
    base_uv: torch.Tensor,
    condition: torch.Tensor,
    offset: torch.Tensor,
    caps: torch.Tensor,
    cap_masses: torch.Tensor,
    tau: float,
    gradient_scale: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Two-pass backward that never retains the complete point-by-cap graph."""
    directions = model_directions(model, base_uv, condition, offset, slice(None), args)
    points_leaf = directions.detach().requires_grad_(True)
    with torch.no_grad():
        sums = torch.zeros(len(caps), device=base_uv.device, dtype=base_uv.dtype)
        for cap_slice in slices(len(caps), args.cap_chunk_size):
            for point_slice in slices(len(base_uv), args.point_chunk_size):
                sums[cap_slice] += cap_indicator(caps[cap_slice], points_leaf[point_slice], tau).sum(0)
    sums_leaf = sums.detach().requires_grad_(True)
    residual = sums_leaf / len(base_uv) - cap_masses
    loss = torch.sqrt(residual.square().mean() + 1e-12)
    (upstream,) = torch.autograd.grad(loss * gradient_scale, sums_leaf)
    for cap_slice in slices(len(caps), args.cap_chunk_size):
        for point_slice in slices(len(base_uv), args.point_chunk_size):
            contribution = cap_indicator(caps[cap_slice], points_leaf[point_slice], tau).sum(0)
            contribution.backward(upstream[cap_slice])
    directions.backward(points_leaf.grad)
    return loss.detach()


def learning_rate_factor(step: int, total_steps: int, final_ratio: float = 1e-3) -> float:
    linear = min(max(step / max(total_steps, 1), 0.0), 1.0)
    curved = -math.log(1.0 - 0.9 * linear) / -math.log(0.1)
    return 1.0 - (1.0 - final_ratio) * curved


def train(args: argparse.Namespace, device: torch.device):
    validate_args(args)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sobol = torch.quasirandom.SobolEngine(2, scramble=False)
    base_uv = sobol.draw(args.npoints).to(device)
    target = ConditionalTarget(args, device)
    model = AutoDecoderPointsetsV2cmp(condition_dim(args), args.hidden_sizes, 2, args.npoints, args.encoding_scale, args.mapped_dim).to(device)
    model.initialize_residual()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, args.epochs)
    )

    losses: list[float] = []
    start_epoch = 0
    validation_history = []
    best_validation = float('inf')
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        validate_args(argparse.Namespace(**state['args']))
        ignored = {'resume', 'output_dir', 'device', 'no_visualize', 'stop_after'}
        for key, value in vars(args).items():
            if key not in ignored and state['args'].get(key) != value:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(state['model_state_dict'])
        base_uv = state['base_uv'].to(device)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        generator.set_state(state['generator_state'].cpu())
        losses = state['losses']
        start_epoch = state['completed_epochs']
        validation_history = state.get('validation_history', [])
        best_validation = state.get('best_validation', float('inf'))
    quadrature = BRDFQuadrature(device, args.normalization_resolution) if args.case == 'brdf' else None
    validation = HeldOutValidation(target, device, quadrature) if args.case == 'brdf' and args.validation_every else None
    if validation is not None and not validation_history:
        baseline = validation.evaluate(model, base_uv, args)
        validation_history.append({'epoch': 0, **baseline})
        print(f"initial held-out cap L2={baseline['mean_cap_l2']:.8f}", flush=True)
    report_every = max(1, min(100, args.epochs // 10))
    for epoch in range(start_epoch, min(args.epochs, args.stop_after or args.epochs)):
        # 1. Resample spherical caps (the angular-radius measure is unchanged).
        caps = sample_spherical_caps(
            args.nregions, device=device, dtype=torch.float32, generator=generator
        )

        optimizer.zero_grad(set_to_none=True)
        instance_losses = []
        tau = args.tau_start + (args.tau_end - args.tau_start) * (
            epoch / max(1, args.epochs - 1)
        )
        conditions = [sample_condition(args, generator, device) for _ in range(args.condition_batch_size)]
        offsets = [sample_offset(args, generator, device) for _ in conditions]
        if args.case == 'brdf':
            all_masses = shared_sobol_masses(target, caps, conditions,
                sample_count=args.target_samples, seed=args.seed + epoch + 104729,
                chunk_size=args.cap_chunk_size, quadrature=quadrature)
        for index, (condition, offset) in enumerate(zip(conditions, offsets)):
            # 2. Draw a target condition and the matching randomization offset.
            if args.case == 'brdf':
                cap_masses = all_masses[:, index]
            else:
                space_directions, proposal_pdf = target.sample_proposal(
                    args.target_samples, condition, generator=generator)
                cap_masses = estimate_cap_masses(
                    target, caps, space_directions, condition, args.cap_chunk_size, proposal_pdf)
            # 4-5. Generate conditional points, evaluate discrepancy, and backpropagate.
            loss = batched_loss_and_backward(
                model, base_uv, condition, offset, caps, cap_masses, tau,
                1.0 / args.condition_batch_size, args,
            )
            instance_losses.append(float(loss.cpu()))

        # 6. One shared network update learns across all sampled conditions.
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError(f"Non-finite gradient at epoch {epoch + 1}.")
        optimizer.step()
        scheduler.step()
        losses.append(float(np.mean(instance_losses)))
        if validation is not None and ((epoch + 1) % args.validation_every == 0 or epoch + 1 == args.epochs):
            metrics = validation.evaluate(model, base_uv, args)
            validation_history.append({'epoch': epoch + 1, **metrics})
            print(f"epoch {epoch + 1}: held-out cap L2={metrics['mean_cap_l2']:.8f}", flush=True)
            if args.output_dir is not None:
                recovery_dir = Path(args.output_dir)
                recovery_dir.mkdir(parents=True, exist_ok=True)
                (recovery_dir / 'validation.json').write_text(json.dumps(validation_history, indent=2))
                if metrics['mean_cap_l2'] < best_validation:
                    best_validation = metrics['mean_cap_l2']
                    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                    config.update(schema_version=3, architecture='AutoDecoderPointsetsV2cmp',
                                  fixed_specular=0.0, completed_epochs=epoch + 1)
                    torch.save({'model_state_dict': model.state_dict(), 'base_uv': base_uv.cpu(),
                                'config': config, 'losses': losses}, recovery_dir / 'best_model.pt')
        if args.output_dir is not None and ((epoch + 1) % 500 == 0 or epoch + 1 == args.epochs or epoch + 1 == args.stop_after):
            # Recovery state for long experiments; final model.pt remains the
            # evaluated checkpoint and is written only after training completes.
            recovery_dir = Path(args.output_dir)
            recovery_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "base_uv": base_uv.cpu(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "generator_state": generator.get_state(), "losses": losses,
                        "completed_epochs": epoch + 1, "args": vars(args),
                        "validation_history": validation_history, "best_validation": best_validation},
                       recovery_dir / "training_state.tmp")
            (recovery_dir / "training_state.tmp").replace(recovery_dir / "training_state.pt")
        if epoch % report_every == 0 or epoch + 1 == args.epochs:
            print(f"epoch {epoch + 1:5d}/{args.epochs}: loss={losses[-1]:.8f} "
                  f"last_condition_ESS={target.last_effective_sample_size:.1f}/{args.target_samples}")
    return model, target, base_uv, losses


@torch.no_grad()
def evaluate_conditions(
    model: AutoDecoderPointsetsV2cmp,
    base_uv: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conditions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.70, 0.0, 0.714], [0.0, 0.70, 0.714]],
        device=device,
    )
    conditions = normalize(conditions)
    offsets = torch.tensor([[0.0, 0.0], [0.37, 0.19], [0.11, 0.61]], device=device)
    if args.case == "brdf" and (args.fixed_view is not None or args.view_angle is not None):
        fixed_view = sample_condition(args, torch.Generator(device=device), device)
        conditions = fixed_view.unsqueeze(0).expand_as(conditions)
    elif args.case == "env" and args.fixed_normal is not None:
        fixed_normal = normalize(torch.tensor(args.fixed_normal, device=device))
        conditions = fixed_normal.unsqueeze(0).expand_as(conditions)
    if args.fixed_offset is not None:
        fixed_offset = torch.remainder(torch.tensor(args.fixed_offset, device=device), 1.0)
        offsets = fixed_offset.unsqueeze(0).expand_as(offsets)
    all_directions = []
    full_slice = slice(0, len(base_uv))
    for condition, offset in zip(conditions, offsets):
        all_directions.append(model_directions(model, base_uv, condition, offset, full_slice, args))
    return conditions.cpu().numpy(), offsets.cpu().numpy(), torch.stack(all_directions).cpu().numpy()


def direction_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    v = (torch.arange(height, device=device) + 0.5) / height
    u = (torch.arange(width, device=device) + 0.5) / width
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    theta, phi = vv * math.pi, uu * (2.0 * math.pi)
    return torch.stack(
        (torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)),
        dim=-1,
    )


def directions_to_uv(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.mod(np.arctan2(directions[..., 1], directions[..., 0]), 2.0 * np.pi) / (2.0 * np.pi)
    v = np.arccos(np.clip(directions[..., 2], -1.0, 1.0)) / np.pi
    return u, v


def save_visualization(
    output_path: Path,
    model: AutoDecoderPointsetsV2cmp,
    target: ConditionalTarget,
    base_uv: torch.Tensor,
    losses: list[float],
    conditions: np.ndarray,
    offsets: np.ndarray,
    directions: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    grid = direction_grid(72, 144, device)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    axes[0].plot(losses, color="#2455d6")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Spherical-cap discrepancy")
    axes[0].grid(alpha=0.25)
    for index in range(3):
        condition = torch.as_tensor(conditions[index], device=device)
        values = target.weights(grid.reshape(-1, 3), condition).reshape(72, 144).cpu().numpy()
        values = values / max(float(values.max()), 1e-8)
        u, v = directions_to_uv(directions[index])
        axes[index + 1].imshow(values, extent=(0, 1, 1, 0), cmap="magma", aspect="auto")
        axes[index + 1].scatter(u, v, s=10, c="#42e8f5", edgecolors="black", linewidths=0.25)
        axes[index + 1].set(
            title=f"{'local wo' if args.case == 'brdf' else 'world n'}={np.round(conditions[index], 2)}\noffset={np.round(offsets[index], 2)}",
            xlabel="u",
            ylabel="v",
            xlim=(0, 1),
            ylim=(1, 0),
        )
    fig.suptitle(f"Conditional neural point sets — {args.case.upper()}")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one conditional neural point-set generator for BRDF or Env targets."
    )
    parser.add_argument("--case", choices=("brdf", "env"), required=True)
    parser.add_argument("--envmap", type=Path, default=DEFAULT_ENVMAP)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume an unchanged training recipe from training_state.pt")
    parser.add_argument("--stop-after", type=int, default=None, help="Save recovery state and stop after this total epoch count")
    parser.add_argument("--validation-every", type=int, default=500, help="Held-out BRDF check interval; 0 disables")
    parser.add_argument("--normalization-resolution", type=int, default=256, help="BRDF hemisphere quadrature z nodes (twice as many azimuth nodes)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--npoints", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--condition-batch-size", type=int, default=16)
    parser.add_argument("--nregions", type=int, default=4096)
    parser.add_argument("--target-samples", type=int, default=16384)
    parser.add_argument("--point-chunk-size", type=int, default=512)
    parser.add_argument("--cap-chunk-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--encoding-scale", type=float, default=0.1)
    parser.add_argument("--mapped-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tau-start", type=float, default=20.0)
    parser.add_argument("--tau-end", type=float, default=1000.0)
    parser.add_argument("--normal-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--view-condition", action=argparse.BooleanOptionalAction, default=True,
                        help="BRDF: condition on local wo_x, wo_y (normal remains +Z)")
    parser.add_argument("--offset-condition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed-normal", type=float, nargs=3, default=None, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--fixed-offset", type=float, nargs=2, default=None, metavar=("OX", "OY"))
    parser.add_argument("--fixed-view", type=float, nargs=3, default=None, metavar=("WX", "WY", "WZ"),
                        help="BRDF: fix the local outgoing direction; WZ must be positive")
    parser.add_argument("--view-angle", type=float, default=None,
                        help="BRDF: fix polar view angle at azimuth 0; default samples local wo")
    parser.add_argument("--roughness", type=float, default=0.7)
    parser.add_argument("--metallic", type=float, default=0.2)
    parser.add_argument("--anisotropic", type=float, default=0.0)
    parser.add_argument("--subsurface", type=float, default=0.0)
    parser.add_argument("--no-visualize", action="store_true")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "specular", 0.0) != 0.0:
        raise ValueError("The open-source example fixes specular to 0.")
    if getattr(args, 'normalization_resolution', 128) < 8:
        raise ValueError('normalization-resolution must be at least 8.')
    if getattr(args, 'validation_every', 0) < 0:
        raise ValueError('validation-every must be nonnegative.')
    if getattr(args, 'stop_after', None) is not None and not 0 < args.stop_after <= args.epochs:
        raise ValueError('stop-after must be within the configured training epochs.')
    if not args.hidden_sizes or any(width <= 0 for width in args.hidden_sizes):
        raise ValueError("Hidden sizes must be positive.")
    if not math.isfinite(args.encoding_scale) or args.encoding_scale <= 0:
        raise ValueError("Encoding scale must be finite and positive.")
    for name in ("npoints", "epochs", "condition_batch_size", "nregions", "point_chunk_size",
                 "cap_chunk_size", "mapped_dim"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if args.target_samples < 2:
        raise ValueError("target-samples must be at least 2.")
    for name in ("learning_rate", "tau_start", "tau_end"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if args.view_angle is not None and not 0.0 <= args.view_angle < 90.0:
        raise ValueError("--view-angle must be in [0, 90) degrees.")
    if not 0.0 < args.roughness <= 1.0:
        raise ValueError("--roughness must be in (0,1].")
    for name in ("metallic", "anisotropic", "subsurface"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"{name} must be in [0,1].")
    for name in ("fixed_view", "fixed_normal"):
        vector = getattr(args, name)
        if vector is not None and (not all(math.isfinite(x) for x in vector)
                                   or sum(x*x for x in vector) < 1e-16):
            raise ValueError(f"{name} must be a finite nonzero vector.")
    if args.fixed_view is not None and args.fixed_view[2] <= 0:
        raise ValueError("--fixed-view requires positive local WZ.")
    if args.fixed_view is not None and args.view_angle is not None:
        raise ValueError("Choose either --fixed-view or --view-angle.")
    if args.case == "brdf" and args.fixed_normal is not None:
        raise ValueError("BRDF uses fixed local normal +Z; use --fixed-view instead of --fixed-normal.")
    if args.case == "brdf" and not args.normal_condition:
        raise ValueError("Use --no-view-condition for BRDF ablation; normal is always local +Z.")
    if args.case == "env" and (args.fixed_view is not None or args.view_angle is not None):
        raise ValueError("View controls are BRDF-only; Env uses --fixed-normal.")
    if args.fixed_offset is not None and not all(math.isfinite(x) for x in args.fixed_offset):
        raise ValueError("--fixed-offset must be finite.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir or Path("outputs") / f"neural_{args.case}"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    recipe = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    recipe.update(fixed_specular=0.0, architecture='AutoDecoderPointsetsV2cmp',
                  base_sobol_scrambled=False,
                  target_integration='shared_scrambled_sobol_fixed_quadrature' if args.case == 'brdf' else 'importance_self_normalized')
    (output_dir / 'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')

    print(
        f"case={args.case} device={device} condition="
        f"{('local_view ' if args.view_condition else '') if args.case == 'brdf' else ('world_normal ' if args.normal_condition else '')}"
        f"{'offset' if args.offset_condition else ''}"
    )
    model, target, base_uv, losses = train(args, device)
    if len(losses) < args.epochs:
        print(f"Paused at epoch {len(losses)}; resume from {output_dir / 'training_state.pt'}")
        return
    conditions, offsets, directions = evaluate_conditions(model, base_uv, args, device)

    configuration = vars(args).copy()
    configuration["envmap"] = str(args.envmap)
    configuration["output_dir"] = str(output_dir)
    configuration["resume"] = str(args.resume) if args.resume is not None else None
    configuration["fixed_specular"] = 0.0
    configuration["target_integration"] = "shared_scrambled_sobol_fixed_quadrature" if args.case == 'brdf' else 'importance_self_normalized'
    configuration["schema_version"] = 3
    configuration["architecture"] = "AutoDecoderPointsetsV2cmp"
    configuration["coordinate_space"] = "local" if args.case == "brdf" else "world"
    configuration["direction_condition"] = "local_wo_xy" if args.case == "brdf" else "world_normal_xyz"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_uv": base_uv.cpu(),
            "config": configuration,
            "losses": losses,
        },
        output_dir / "model.pt",
    )
    np.savez_compressed(
        output_dir / "examples.npz",
        **{("view_directions" if args.case == "brdf" else "normals"): conditions},
        offsets=offsets,
        directions=directions,
        losses=np.asarray(losses, dtype=np.float32),
        config_json=np.asarray(json.dumps(configuration, indent=2)),
    )
    if not args.no_visualize:
        save_visualization(
            output_dir / "training.png", model, target, base_uv, losses,
            conditions, offsets, directions, args, device,
        )
    print(f"Saved model and examples to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
