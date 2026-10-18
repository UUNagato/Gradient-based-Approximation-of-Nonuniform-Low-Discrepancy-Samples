"""Static spherical point-set visualization and standalone CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sphere import latent_to_direction


def _select_instance(array: np.ndarray, instance: int) -> np.ndarray:
    if array.ndim == 3:
        if not 0 <= instance < array.shape[0]:
            raise IndexError(f"Instance {instance} is outside [0, {array.shape[0] - 1}].")
        array = array[instance]
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"Expected [N,2], [N,3], [B,N,2], or [B,N,3], got {array.shape}.")
    return array


def load_pointset(
    path: str | Path,
    *,
    key: str | None = None,
    instance: int = 0,
) -> np.ndarray:
    path = Path(path)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key is not None:
                if key not in loaded.files:
                    raise KeyError(f"Key '{key}' not found. Available keys: {loaded.files}")
                array = loaded[key]
            elif "directions" in loaded.files:
                array = loaded["directions"]
            elif "latents" in loaded.files:
                array = loaded["latents"]
            else:
                raise KeyError(
                    f"No 'directions' or 'latents' array in {path}. Available keys: {loaded.files}"
                )
        finally:
            loaded.close()
    else:
        array = loaded

    array = _select_instance(np.asarray(array), instance)
    if array.shape[1] == 2:
        latent_tensor = torch.from_numpy(array.astype(np.float32, copy=False))
        with torch.no_grad():
            array = latent_to_direction(latent_tensor).cpu().numpy()

    directions = np.asarray(array, dtype=np.float32)
    if not np.isfinite(directions).all():
        raise ValueError(f"Point set contains NaN or Inf: {path}")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Direction point set contains a zero-length vector.")
    return directions / norms


def render_pointset(
    directions: np.ndarray,
    output_path: str | Path,
    *,
    title: str | None = None,
    point_size: float = 18.0,
    image_size: int = 900,
    elevation: float = 22.0,
    azimuth: float = 35.0,
) -> Path:
    directions = np.asarray(directions, dtype=np.float32)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"Expected directions [N,3], got {directions.shape}.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100
    figure = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    longitude = np.linspace(0.0, 2.0 * np.pi, 72)
    latitude = np.linspace(0.0, np.pi, 36)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    sphere_x = np.cos(lon_grid) * np.sin(lat_grid)
    sphere_y = np.sin(lon_grid) * np.sin(lat_grid)
    sphere_z = np.cos(lat_grid)
    axis.plot_wireframe(
        sphere_x,
        sphere_y,
        sphere_z,
        rstride=3,
        cstride=3,
        color="#94a3b8",
        linewidth=0.45,
        alpha=0.32,
    )
    axis.scatter(
        directions[:, 0] * 1.012,
        directions[:, 1] * 1.012,
        directions[:, 2] * 1.012,
        s=point_size,
        c="#ef233c",
        edgecolors="#7f1d1d",
        linewidths=0.35,
        depthshade=True,
    )

    axis.set_xlim(-1.08, 1.08)
    axis.set_ylim(-1.08, 1.08)
    axis.set_zlim(-1.08, 1.08)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    if title:
        axis.set_title(title, pad=12)
    figure.tight_layout(pad=0.2)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a saved spherical point set.")
    parser.add_argument("input", type=Path, help="Input .npz or .npy point set.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--key", type=str, default=None, help="Array key for .npz input.")
    parser.add_argument("--instance", type=int, default=0, help="Instance index for [B,N,D] input.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--image-size", type=int, default=900)
    parser.add_argument("--elevation", type=float, default=22.0)
    parser.add_argument("--azimuth", type=float, default=35.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.input.with_suffix(".png")
    directions = load_pointset(args.input, key=args.key, instance=args.instance)
    render_pointset(
        directions,
        output,
        title=args.title,
        point_size=args.point_size,
        image_size=args.image_size,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    print(f"Saved visualization: {output.resolve()}")


if __name__ == "__main__":
    main()
