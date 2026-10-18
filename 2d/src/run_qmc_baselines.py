#!/usr/bin/env python3
"""Generate and evaluate 2D QMC baseline point sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from discrepancy import estimate_nonuniform_discrepancies, uniform_discrepancies
from io_helpers import save_discrepancies, save_point_sets
from qmc import generate_point_sets, stack_points
from targets import get_target


DEFAULT_METHODS = [
    "random_base",
    "sobol_base",
    "sobol_shifted",
    "sobol_scrambled",
    "halton_base",
    "halton_shifted",
    "halton_scrambled",
]


def parse_ns(values: list[str]) -> list[int]:
    ns: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                ns.append(int(part))
    if not ns:
        raise ValueError("At least one N value is required.")
    return sorted(set(ns))


def parse_method(method: str) -> tuple[str, str]:
    parts = method.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Method must be '<sequence>_<variant>', got {method}.")
    sequence, variant = parts
    if sequence == "random":
        return sequence, "base"
    return sequence, variant


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="uniform", choices=["uniform", "bilinear", "quarter_disk", "image"])
    parser.add_argument("--image", type=Path, default=None, help="Image path when --target image.")
    parser.add_argument("--image_clip_high_quantile", type=float, default=None)
    parser.add_argument("--image_flip_y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--ns", nargs="+", default=["64"], help="Sample sizes, space or comma separated.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scramble", default="LMS_DS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gt_nregions", type=int, default=8192)
    parser.add_argument("--gt_nspace_samples", type=int, default=32768)
    parser.add_argument("--gt_reps", type=int, default=4)
    parser.add_argument("--areas_batch_size", type=int, default=2048)
    parser.add_argument("--areas_space_batch_size", type=int, default=None)
    parser.add_argument("--point_chunk_size", type=int, default=None)
    parser.add_argument("--region_chunk_size", type=int, default=None)
    parser.add_argument("--pair_batch_size", type=int, default=2048)
    parser.add_argument("--save_points", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ns = parse_ns(args.ns)
    if args.target == "image" and args.image is None:
        raise ValueError("--image is required when --target image.")

    target = get_target(
        args.target,
        image_path=args.image,
        device=args.device,
        image_flip_y=args.image_flip_y,
        image_clip_high_quantile=args.image_clip_high_quantile,
    )
    outdir = args.outdir or (ROOT / "data" / "baselines" / target.name)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    job_bar = tqdm(
        total=len(args.methods) * len(ns),
        desc="QMC baselines",
        disable=not args.show_progress,
    )

    for method in args.methods:
        sequence, variant = parse_method(method)
        l2_means: list[float] = []
        star_means: list[float] = []
        l2_values_by_n: list[np.ndarray] = []
        star_values_by_n: list[np.ndarray] = []

        for npoints in ns:
            job_bar.set_description(f"{method} N={npoints}")
            point_sets = generate_point_sets(
                sequence,
                npoints,
                instances=args.instances,
                variant=variant,
                seed=args.seed,
                target=target,
                device=args.device,
                scramble=args.scramble,
            )
            points_array = stack_points(point_sets)

            l2_values: list[float] = []
            star_values: list[float] = []
            for point_set in points_array:
                points = torch.as_tensor(point_set, dtype=torch.float64, device=dev)
                if target.name == "uniform":
                    l2, star = uniform_discrepancies(points, pair_batch_size=args.pair_batch_size)
                else:
                    l2, star = estimate_nonuniform_discrepancies(
                        points,
                        target,
                        nrectangles=args.gt_nregions,
                        nspace_samples=args.gt_nspace_samples,
                        reps=args.gt_reps,
                        areas_batch_size=args.areas_batch_size,
                        areas_space_batch_size=args.areas_space_batch_size,
                        point_chunk_size=args.point_chunk_size,
                        region_chunk_size=args.region_chunk_size,
                    )
                l2_values.append(l2)
                star_values.append(star)
                if args.show_progress:
                    job_bar.set_postfix_str(f"instance {len(l2_values)}/{args.instances}", refresh=True)

            l2_array = np.asarray(l2_values, dtype=np.float64)
            star_array = np.asarray(star_values, dtype=np.float64)
            l2_means.append(float(l2_array.mean()))
            star_means.append(float(star_array.mean()))
            l2_values_by_n.append(l2_array)
            star_values_by_n.append(star_array)

            if args.save_points:
                save_point_sets(
                    outdir / f"{method}_points_n{npoints}.npz",
                    points=points_array,
                    target=target.name,
                    method=method,
                    seed=args.seed,
                    config={"instances": args.instances, "scramble": args.scramble},
                )
            message = f"{method}, N={npoints}: L2={l2_means[-1]:.6e}, star={star_means[-1]:.6e}"
            if args.show_progress:
                tqdm.write(message)
            else:
                print(message)
            job_bar.update(1)

        save_discrepancies(
            outdir / f"{method}_discrepancies.npz",
            ns=ns,
            l2_mean=l2_means,
            star_mean=star_means,
            target=target.name,
            method=method,
            config={"seed": args.seed, "instances": args.instances, "scramble": args.scramble},
            l2_values=np.stack(l2_values_by_n, axis=0),
            star_values=np.stack(star_values_by_n, axis=0),
        )

    job_bar.close()


if __name__ == "__main__":
    main()
