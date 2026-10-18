"""Command-line entry point for the three minimal 3D reproduction cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from optimizer import OptimizationConfig, optimize_pointset
from targets import make_target
from visualize import render_pointset


THIS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENVMAP = THIS_DIR / "data" / "targets" / "example_envmap.npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a spherical point set for uniform, envmap, or cosine density."
    )
    parser.add_argument("--case", choices=("uniform", "envmap", "cosine"), required=True)
    parser.add_argument("--backend", choices=("batched", "reference"), default="batched")
    parser.add_argument("--envmap", type=Path, default=DEFAULT_ENVMAP)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or a torch device.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", choices=("sobol", "random"), default="sobol")
    parser.add_argument("--npoints", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--nregions", type=int, default=2048)
    parser.add_argument("--target-samples", type=int, default=8192)
    parser.add_argument("--target-sample-chunk-size", type=int, default=8192)
    parser.add_argument("--min-cap-hits", type=int, default=64)
    parser.add_argument("--fallback-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--final-learning-rate", type=float, default=5e-7)
    parser.add_argument("--tau-start", type=float, default=20.0)
    parser.add_argument("--tau-end", type=float, default=1000.0)
    parser.add_argument("--point-chunk-size", type=int, default=1024)
    parser.add_argument("--cap-chunk-size", type=int, default=1024)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Store latent snapshots in the output NPZ every N epochs; 0 disables it.",
    )
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--visual-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=900)
    parser.add_argument("--quiet", action="store_true")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    dtype = torch.float32
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = args.output or Path("outputs") / f"{args.case}_n{args.npoints}.npz"
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    target = make_target(
        args.case,
        device=device,
        dtype=dtype,
        envmap_path=args.envmap if args.case == "envmap" else None,
    )
    config = OptimizationConfig(
        npoints=args.npoints,
        epochs=args.epochs,
        nregions=args.nregions,
        target_samples=args.target_samples,
        target_sample_chunk_size=args.target_sample_chunk_size,
        min_cap_hits=args.min_cap_hits,
        fallback_samples=args.fallback_samples,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        point_chunk_size=args.point_chunk_size,
        cap_chunk_size=args.cap_chunk_size,
        backend=args.backend,
        seed=args.seed,
        init=args.init,
        snapshot_every=args.snapshot_every,
    )

    if not args.quiet:
        print(
            f"case={args.case} backend={args.backend} device={device} "
            f"npoints={args.npoints} epochs={args.epochs}"
        )
    result = optimize_pointset(
        target,
        config,
        device=device,
        dtype=dtype,
        progress=not args.quiet,
    )

    metadata = {
        **asdict(config),
        "case": args.case,
        "device": str(device),
        "dtype": str(dtype),
        "envmap": str(args.envmap.resolve()) if args.case == "envmap" else None,
    }
    output_arrays = {
        "latents": result.latents.cpu().numpy(),
        "directions": result.directions.cpu().numpy(),
        "losses": np.asarray(result.losses, dtype=np.float32),
        "config_json": np.asarray(json.dumps(metadata, indent=2)),
    }
    if result.latent_snapshots is not None:
        output_arrays["snapshot_epochs"] = np.asarray(result.snapshot_epochs, dtype=np.int64)
        output_arrays["latent_snapshots"] = result.latent_snapshots.numpy()
    np.savez_compressed(output, **output_arrays)
    print(f"Saved point set: {output.resolve()}")

    if not args.no_visualize:
        visual_output = args.visual_output or output.with_suffix(".png")
        render_pointset(
            result.directions.cpu().numpy(),
            visual_output,
            title=f"{args.case} — N={args.npoints}",
            image_size=args.image_size,
        )
        print(f"Saved visualization: {visual_output.resolve()}")


if __name__ == "__main__":
    main()
