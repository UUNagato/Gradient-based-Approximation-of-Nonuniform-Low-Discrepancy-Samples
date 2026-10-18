#!/usr/bin/env python3
"""Run direct uniform 2D point-set optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_helpers import save_discrepancies, save_point_sets
from optimize_uniform import UniformOptimizationConfig, optimize_uniform


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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="uniform", choices=["uniform"])
    parser.add_argument("--outdir", type=Path, default=ROOT / "data" / "optimized" / "uniform")
    parser.add_argument("--ns", nargs="+", default=["4", "8", "16", "32", "64", "128", "256", "512", "1024"], help="Sample sizes, space or comma separated.")
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nregions", type=int, default=8192)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--nepochs_params", type=int, default=500)
    parser.add_argument("--lr_init", type=float, default=1e-3)
    parser.add_argument("--lr_final", type=float, default=1e-7)
    parser.add_argument("--tau0", type=float, default=20.0)
    parser.add_argument("--tau1", type=float, default=2000.0)
    parser.add_argument("--clamp_eps", type=float, default=1e-12)
    parser.add_argument("--pair_batch_size", type=int, default=4096)
    parser.add_argument("--point_chunk_size", type=int, default=1024)
    parser.add_argument("--region_chunk_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_discrepancy_eval", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ns = parse_ns(args.ns)
    args.outdir.mkdir(parents=True, exist_ok=True)

    l2_means: list[float] = []
    star_means: list[float] = []
    l2_values_by_n: list[np.ndarray] = []
    star_values_by_n: list[np.ndarray] = []

    for npoints in tqdm(ns, desc="Ns", disable=not args.show_progress):
        config = UniformOptimizationConfig(
            device=args.device,
            npoints=npoints,
            instances=args.instances,
            nregions=args.nregions,
            nepochs=args.nepochs,
            nepochs_params=args.nepochs_params,
            lr_init=args.lr_init,
            lr_final=args.lr_final,
            tau0=args.tau0,
            tau1=args.tau1,
            clamp_eps=args.clamp_eps,
            pair_batch_size=args.pair_batch_size,
            point_chunk_size=args.point_chunk_size,
            region_chunk_size=args.region_chunk_size,
            seed=args.seed,
            evaluate_discrepancy=not args.skip_discrepancy_eval,
            show_progress=args.show_progress,
        )
        result = optimize_uniform(config)
        config_dict = {
            "npoints": npoints,
            "instances": args.instances,
            "nregions": args.nregions,
            "nepochs": args.nepochs,
            "nepochs_params": args.nepochs_params,
            "lr_init": args.lr_init,
            "lr_final": args.lr_final,
            "tau0": args.tau0,
            "tau1": args.tau1,
            "clamp_eps": args.clamp_eps,
            "seed": args.seed,
        }
        save_point_sets(
            args.outdir / f"points_n{npoints}.npz",
            points=result.points,
            target="uniform",
            method="direct_optimization",
            seed=args.seed,
            config=config_dict,
        )

        if not args.skip_discrepancy_eval:
            if result.l2_mean is None or result.star_mean is None:
                raise RuntimeError("Missing discrepancy values.")
            l2_means.append(result.l2_mean)
            star_means.append(result.star_mean)
            l2_values_by_n.append(result.l2_values)
            star_values_by_n.append(result.star_values)
            message = f"N={npoints}: L2={result.l2_mean:.6e}, star={result.star_mean:.6e}"
        else:
            message = f"N={npoints}: saved points"
        if args.show_progress:
            tqdm.write(message)
        else:
            print(message)

    if not args.skip_discrepancy_eval:
        save_discrepancies(
            args.outdir / "discrepancies.npz",
            ns=ns,
            l2_mean=l2_means,
            star_mean=star_means,
            target="uniform",
            method="direct_optimization",
            config={"seed": args.seed, "instances": args.instances},
            l2_values=np.stack(l2_values_by_n, axis=0),
            star_values=np.stack(star_values_by_n, axis=0),
        )


if __name__ == "__main__":
    main()
