"""Evaluate a BRDF checkpoint on held-out caps, with per-view direct optimization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import train_neural as neural
from optimizer import batched_loss_and_backward
from pointsetmodel import AutoDecoderPointsetsV2cmp
from sphere import latent_to_direction, sample_spherical_caps


@torch.no_grad()
def hard_fractions(points, caps):
    return torch.cat([
        (points @ block[:, :3].T >= torch.cos(block[:, 3])).double().mean(0)
        for block in caps.split(256)
    ])


def direct_optimize(base, condition, target, args, epochs, seed):
    generator = torch.Generator(device=base.device).manual_seed(seed)
    latents = base.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latents], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: neural.learning_rate_factor(step, epochs)
    )
    losses = []
    for epoch in range(epochs):
        caps = sample_spherical_caps(2048, device=base.device, dtype=base.dtype, generator=generator)
        space, pdf = target.sample_proposal(8192, condition, generator=generator)
        masses = neural.estimate_cap_masses(target, caps, space, condition, 512, pdf)
        optimizer.zero_grad(set_to_none=True)
        tau = args.tau_start + (args.tau_end - args.tau_start) * epoch / max(1, epochs - 1)
        loss = batched_loss_and_backward(latents, caps, masses, tau=tau,
                                        point_chunk_size=1024, cap_chunk_size=512)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))
        if epoch % 1000 == 0 or epoch + 1 == epochs:
            print(f"direct epoch {epoch + 1}/{epochs}: loss={float(loss):.8f}", flush=True)
    return latent_to_direction(latents.detach()), np.asarray(losses)


def sphere_panel(ax, points, target, condition, title):
    # Orthographic view of the +Z hemisphere; no geographic labels.
    q = torch.linspace(-1.02, 1.02, 500, device=condition.device)
    yy, xx = torch.meshgrid(q.flip(0), q, indexing="ij")
    r2 = xx.square() + yy.square()
    grid = torch.stack((xx, yy, (1-r2).clamp_min(0).sqrt()), -1)
    with torch.no_grad():
        values = target.weights(grid.reshape(-1, 3), condition).reshape(r2.shape)
        values = values / values[r2 <= 1].max()
    colors = plt.get_cmap("magma")(values.cpu().numpy())
    colors[r2.cpu().numpy() > 1] = 1
    ax.imshow(colors, extent=(-1.02, 1.02, -1.02, 1.02))
    points = points.cpu().numpy()
    visible = points[:, 2] >= 0
    ax.scatter(points[visible, 0], points[visible, 1], s=4, c="#53e3ee", linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--direct-epochs", type=int, default=5000)
    parser.add_argument("--reference-samples", type=int, default=262144)
    parser.add_argument("--evaluation-caps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=271828)
    options = parser.parse_args()
    device = neural.resolve_device(options.device)
    saved = torch.load(options.checkpoint, map_location=device, weights_only=False)
    args = argparse.Namespace(**saved["config"])
    neural.validate_args(args)
    if args.case != "brdf" or args.schema_version != 3:
        raise ValueError("Expected a schema 3 BRDF AutoDecoderPointsetsV2cmp checkpoint.")
    model = AutoDecoderPointsetsV2cmp(neural.condition_dim(args), args.hidden_sizes, 2,
                                     args.npoints, args.encoding_scale, args.mapped_dim).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    base = saved["base_uv"].to(device)
    target = neural.ConditionalTarget(args, device)
    out = options.output_dir or options.checkpoint.parent / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(options.seed)
    caps = sample_spherical_caps(options.evaluation_caps, device=device, dtype=base.dtype,
                                 generator=generator)
    offsets = torch.tensor([[0., 0.], [.37, .19], [.11, .61]], device=device)
    results, all_neural, all_direct, all_conditions, references, reference_runs = [], [], [], [], [], []
    direct_histories = []
    fig, axes = plt.subplots(3, 4, figsize=(14, 11), layout="constrained")
    for index, angle in enumerate((0, 45, 75)):
        theta = math.radians(angle)
        condition = torch.tensor([math.sin(theta), 0., math.cos(theta)], device=device)
        all_conditions.append(condition.cpu().numpy())
        estimates = []
        for _ in range(2):
            samples, pdf = target.sample_proposal(options.reference_samples, condition, generator=generator)
            estimates.append(neural.estimate_cap_masses(target, caps, samples, condition, 128, pdf).double())
        reference = torch.stack(estimates).mean(0)
        references.append(reference.cpu().numpy())
        reference_runs.append(torch.stack(estimates).cpu().numpy())
        noise = float(((estimates[0] - estimates[1]).square().mean()).sqrt() / 2)
        print(f"view={angle}: estimated reference MC noise RMS={noise:.8f}", flush=True)
        generated = []
        for offset in offsets:
            with torch.no_grad():
                generated.append(neural.model_directions(model, base, condition, offset, slice(None), args))
        direct, history = direct_optimize(base, condition, target, args, options.direct_epochs,
                                          options.seed + 100 + index)
        direct_histories.append(history)
        all_direct.append(direct.cpu().numpy())
        all_neural.append(torch.stack(generated).cpu().numpy())
        for j, (label, points) in enumerate(zip(("NN offset 0", "NN offset 1", "NN offset 2", "Direct"),
                                               [*generated, direct])):
            assert points.shape == (args.npoints, 3) and torch.isfinite(points).all()
            error = hard_fractions(points, caps) - reference
            metrics = {"view_angle_degrees": angle, "method": label,
                       "cap_l2": float(error.square().mean().sqrt()),
                       "sampled_cap_max_error": float(error.abs().max()),
                       "below_hemisphere": int((points[:, 2] < 0).sum()),
                       "reference_mc_noise_rms_estimate": noise}
            results.append(metrics)
            print(json.dumps(metrics), flush=True)
            np.savetxt(out / f"view{angle}_{j}_xyz.txt", points.cpu().numpy(), fmt="%.9g", header="x y z")
            sphere_panel(axes[index, j], points, target, condition,
                         f"view={angle} deg | {label}\ncap L2={metrics['cap_l2']:.5f}, below={metrics['below_hemisphere']}")
    fig.suptitle("BRDF target density + generated directions (+Z orthographic view)\n"
                 "Fixed held-out views and offsets; direct optimization retrains separately for each view")
    fig.savefig(out / "sphere_comparison.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(out / "pointsets.npz", neural=np.asarray(all_neural), direct=np.asarray(all_direct),
                        view_directions=np.asarray(all_conditions), offsets=offsets.cpu().numpy(),
                        evaluation_caps=caps.cpu().numpy(), reference_masses=np.asarray(references),
                        reference_estimates=np.asarray(reference_runs), direct_losses=np.asarray(direct_histories))
    report = {"evaluation": {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
              "training": saved["config"], "direct_target_samples": 8192, "direct_caps": 2048,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "notes": "Cap L2 is estimated on held-out random caps, not exact discrepancy. Reference masses use two independent self-normalized importance estimates. MC noise RMS is approximate; it does not measure estimator bias. Direct optimization has a separate point set per view; the neural network is shared.",
              "results": results}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
