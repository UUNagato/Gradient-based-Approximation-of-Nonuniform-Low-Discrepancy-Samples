# 3D discrepancy-optimized point sets

## Directory layout

- `src/`: optimization, conditional neural training, visualization, and correctness tests.
- `scripts/`: editable Bash launchers; results are saved under `data/optimized/`.
- `notebooks/`: standalone Earth environment-map notebook, including embedded target data.
- `data/targets/`: example environment map and Earth image.

## Installation

Run all commands in this README from `3d/`. Python 3.10 or newer is recommended.
Install the shared dependencies from
this directory using the repository's [requirements.txt](../requirements.txt):

```bash
python -m pip install -r ../requirements.txt
```

## Run

```bash
bash scripts/run_uniform.sh
bash scripts/run_envmap.sh
bash scripts/run_cosine.sh
bash scripts/run_neural.sh --case env
jupyter lab notebooks/earth_fast_forward.ipynb
python src/test_minimal.py
```

Launchers accept additional command-line arguments, for example
`bash scripts/run_uniform.sh --device cpu --epochs 10`.
Bash launchers require a Bash shell; direct Python commands also work in PowerShell.
Set `PYTHON_BIN` to choose the Python executable (default: `python3`).
The uniform, envmap, and cosine launchers set `--backend reference` and
`--npoints 512`. Additional arguments override these launcher settings.
Running `src/optimize.py` directly retains its defaults: the batched backend,
256 points, and 2500 epochs. The launchers also use this default epoch count.
The neural launcher defaults to the BRDF case.
The BRDF implementation fixes `specular=0`; there is no configuration option.

## Direct optimization experiments

All cases optimize 2D latent coordinates, map them to spherical directions with
the same mirrored octahedral mapping, and minimize spherical-cap L2 discrepancy.

```bash
python src/optimize.py --case uniform
python src/optimize.py --case envmap
python src/optimize.py --case cosine
```

The three target distributions are:

- `uniform`: uniform spherical directions, with analytic spherical-cap masses.
- `envmap`: luminance-weighted lat-long environment-map sampling.
- `cosine`: normalized `max(z, 0) / pi`, estimated with a uniform-hemisphere proposal.

The bundled `data/targets/example_envmap.npy` makes the envmap command runnable
without external data. A custom environment map may be `.npy`, `.png`, `.jpg`,
or `.jpeg`. NumPy arrays may have shape `[H,W]`, `[H,W,3]`, or `[H,W,4]`:

```bash
python src/optimize.py --case envmap --envmap path/to/map.npy
```

Environment texels represent constant density per solid angle. Texels are selected
by luminance times their exact solid angle; within a texel, longitude and
`cos(theta)` are uniform. Thus even a 1-by-1 white map is a uniform sphere target.
Image channel values are used as supplied, without automatic sRGB decoding; use
linear RGB values when the image represents radiance rather than an arbitrary
density mask.

The non-uniform cap estimator uses independent pilot samples to decide when to
switch to uniform-in-cap integration. It does not clip Monte Carlo estimates to
`[0,1]`, since clipping would introduce bias. Enabling fallback therefore uses
`target-samples` pilot samples in addition to `target-samples` estimation samples.

### Direct optimization parameters

These are the defaults of `src/optimize.py`. The Bash launchers override
`--backend` to `reference` and `--npoints` to 512; appended arguments take
precedence over launcher settings.

| Parameter | Default |
| --- | --- |
| `--device` / `--seed` / `--init` | `auto` (CUDA when available, otherwise CPU) / 0 / `sobol` |
| `--backend` / `--npoints` / `--epochs` | `batched` / 256 / 2500 |
| `--nregions` / `--target-samples` | 2048 / 8192 |
| `--learning-rate` / `--final-learning-rate` | `5e-4` / `5e-7` |
| `--tau-start` / `--tau-end` | 20 / 1000 |
| `--point-chunk-size` / `--cap-chunk-size` | 1024 / 1024 |
| `--target-sample-chunk-size` | 8192 |
| `--min-cap-hits` / `--fallback-samples` | 64 / 64 |
| `--snapshot-every` | 0 (disabled) |

Use `python src/optimize.py --help` and `python src/train_neural.py --help`
for the complete option lists. Direct optimization writes to
`outputs/<case>_n<N>.npz` relative to the working directory unless `--output`
is supplied; the launchers write to `data/optimized/<case>/points.npz`.
The neural launcher writes to `data/optimized/neural/` unless `--output-dir`
is supplied.

## Batched and reference implementations

The default `batched` backend accumulates point contributions in chunks and then
recomputes small blocks during backward. It does not retain the full
`[num_points, num_caps]` autograd graph.

```bash
python src/optimize.py --case uniform --backend batched
python src/optimize.py --case uniform --backend reference
```

The `reference` backend is the direct, readable formulation and is intended for
small experiments and validation. `test_minimal.py` checks that both backends
produce matching losses and latent gradients.

Cap centers are uniform on the sphere and angular radii remain uniform in
`[0,pi]`, matching the research scripts. The hard forward pass uses exact
dot-product cap membership. Backward uses the original scaled sigmoid STE
(without multiplying by `tau`) and bounds only the angular derivative at
parallel directions.

For a quick CPU run:

```bash
python src/optimize.py --case uniform --device cpu \
  --npoints 32 --epochs 10 --nregions 128 \
  --target-samples 512 --output outputs/quick_uniform.npz
```

For larger experiments, increase `--npoints`, `--epochs`, `--nregions`, and
`--target-samples`; memory is controlled with `--point-chunk-size`,
`--cap-chunk-size`, and `--target-sample-chunk-size`.

## Conditional neural experiments

`src/train_neural.py` trains one residual point-set network across many target
conditions. It supports two target modes:

- `brdf`: Disney diffuse plus anisotropic GGX, multiplied by the incident
  cosine term. Both incident points and outgoing view directions are in local
  space, with a fixed normal `(0,0,1)`. The outgoing direction `wo` varies over
  the upper local hemisphere; BRDF parameters remain fixed for a training run.
- `env`: environment-map luminance multiplied by the cosine relative to the
  conditioned surface normal.

The release example fixes `specular=0`, matching the original BRDF experiments.
There is no specular configuration option. Earlier experimental checkpoints trained with
`specular=0.5` are historical diagnostics and must be retrained for this example;
changing checkpoint metadata would not make their learned distributions match.

The network is a standalone copy of `AutoDecoderPointsetsV2cmp`, matching the
original 3D experiment: hidden widths `[64, 128, 256, 512]`, Gaussian positional
encoding with 16 mapped dimensions and scale 0.1, ReLU after the three
concatenating hidden layers, and a single `2*N` output. There is no codebook.
The 32-dimensional encoding is concatenated again at each subsequent layer.
The output layer is zero-initialized.

Inputs concatenate the offset first, then the direction mapped from `[-1,1]`
to `[0,1]`: `[offset_xy, (wo_xy+1)/2]` for BRDF and
`[offset_xy, (normal_xyz+1)/2]` for Env. Positive `wo_z` follows from the unit
constraint. The predicted residual is added to the unwrapped base Sobol points
plus offset, then mirror-warped and mapped to the sphere, as in the original
experiment. Widths and encoding settings are configurable with `--hidden-sizes`,
`--mapped-dim`, and `--encoding-scale`.

```bash
python src/train_neural.py --case brdf
python src/train_neural.py --case env
```

Use `--no-view-condition` for BRDF, `--no-normal-condition` for Env, or
`--no-offset-condition` for offset ablations. Disabling the direction condition
does not stop the target direction from varying; select a fixed direction to
train a single target instead. For example:

```bash
python src/train_neural.py --case env \
  --fixed-normal 0 0 1 --fixed-offset 0.25 0.5
python src/train_neural.py --case brdf --fixed-view 0.6 0 0.8
python src/train_neural.py --case brdf --view-angle 35
```

`--view-angle` fixes the local polar angle at azimuth zero; when neither it nor
`--fixed-view` is set, local `wo` is resampled each training instance. A BRDF run
rejects `--fixed-normal`: the local normal is always `+Z`.

Independent BRDF evaluation still uses the GGX-reflection/cosine mixture with
self-normalized importance sampling; it reports reference noise. Env training
uses an environment-map/cosine mixture. These self-normalized estimates are
consistent but finite-sample biased. Training reports the last condition's
effective sample size (ESS). Invalid or zero sampled target mass raises an error.

GGX proposal generation and PDF evaluation use double precision to avoid
rounding near-antipodal reflections to a zero half-vector. Cap membership
retains the cap tensor's precision. With an explicit `--output-dir`, training
also saves `training_state.pt` every 500 epochs (weights, optimizer, scheduler,
sampling-generator state, base points, arguments, and loss history).

Each direct Python run saves `model.pt`, `examples.npz`, and `training.png` under
`outputs/neural_brdf` or `outputs/neural_env` unless `--output-dir` is given.
BRDF examples store `view_directions` and local-space `directions`; Env examples
store `normals` and world-space `directions`. Configuration metadata records
`schema_version=3`, `architecture=AutoDecoderPointsetsV2cmp`, `coordinate_space`,
and `direction_condition`. Earlier codebook-model checkpoints are incompatible
and must be retrained.

Evaluate a BRDF checkpoint on independent spherical caps and three held-out
view directions, with three offsets per view and separately optimized direct
point sets:

```bash
python src/evaluate_neural.py outputs/neural_brdf/model.pt --device cuda
```

This writes numerical metrics, reference integration estimates, point sets,
and spherical target-density overlays under the checkpoint's `evaluation/`
directory. Cap discrepancies are estimates, with an approximate reference
Monte Carlo noise diagnostic. The default direct comparison uses 5000 epochs
per view, 2048 caps, and 8192 integration samples per update.

### Neural parameters and training recipe

The neural defaults follow the archived 512-point BRDF comparison, with the
cap and integration-sample counts halved:

| Setting | Default |
| --- | --- |
| Points / epochs / conditions per epoch | 512 / 15000 / 16 |
| Spherical caps / integration samples | 4096 / 16384 |
| Point chunk / cap chunk | 512 / 4096 |
| Initial learning rate | `5e-4` (existing schedule decays to `5e-7`) |
| Hidden widths / mapped dimensions / encoding scale | `[64, 128, 256, 512]` / 16 / 0.1 |
| Roughness / metallic | 0.7 / 0.2 |
| Anisotropic / subsurface / specular | 0 / 0 / 0 (specular is fixed) |
| Normalization resolution / validation interval / seed | 256 / 500 / 0 |

The 16 conditions accumulate gradients sequentially before one Adam update;
their full computation graphs are not retained together. Halving the cap and
sample counts changes the stochastic estimates; the archived 15000-epoch
accuracy results do not validate this reduced-budget configuration.
The default chunks cover all points and caps in one block. The neural trainer
still uses its two-pass loss/backward implementation; chunking does not rerun
the network for each cap block. For a 4 GiB GPU, start with these defaults;
lower `--cap-chunk-size` to 2048 or 1024 if other GPU allocations leave too
little memory. The same cap chunk controls target integration and point loss.

For a 1024-point BRDF experiment and an independent evaluation:

```bash
python src/verify_brdf_normalization.py --case brdf --device cuda \
  --roughness 0.58 --metallic 0.2 --anisotropic 0.15 --subsurface 0.5 \
  --normalization-resolution 256 --output-dir data/optimized/neural_brdf_spec0_n1024_epochs15000
python src/train_neural.py --case brdf --device cuda --npoints 1024 --epochs 15000 \
  --condition-batch-size 16 --nregions 8192 --target-samples 32768 \
  --point-chunk-size 1024 --cap-chunk-size 1024 \
  --roughness 0.58 --metallic 0.2 --anisotropic 0.15 --subsurface 0.5 \
  --normalization-resolution 256 --output-dir data/optimized/neural_brdf_spec0_n1024_epochs15000
python src/evaluate_neural.py data/optimized/neural_brdf_spec0_n1024_epochs15000/best_model.pt
```

Evaluation compares three offsets at each of three fixed views against separate
direct optimizations; it saves sphere overlays, point sets, and estimated
discrepancy metrics. It does not rerun or modify the trained network.

BRDF training starts with an unscrambled Sobol base and shares a new scrambled
Sobol sphere integration set across conditions each epoch. Normalization uses
independent fixed Gauss-Legendre/azimuth quadrature, including the outgoing
azimuth for anisotropic materials. Run the normalization check for each material
and resolution before long training; increase resolution if it fails. This replaces the research
script's external polar-only LUT. The example uses direct shared-sample cap
integration, without the research script's adaptive small-cap fallback.
The material above comes from a local experiment, not explicit paper settings.

Every 500 epochs, `validation.json` records held-out cap errors for five views
and three offsets; `best_model.pt` records the best mean validation error.
`model.pt` and `examples.npz` contain the final epoch result. Final evaluation
uses separate caps and integration samples from checkpoint selection.
To resume, repeat the same training arguments and add
`--resume data/optimized/neural_brdf_spec0_n1024_epochs15000/training_state.pt`.
`--stop-after` saves an intermediate recovery state without creating a final model.

## Earth notebook

The demonstration includes saved training logs, loss and point-set plots, and a
static sphere preview. Set `RENDER_ANIMATION = True` to render the optional HTML
animation. To refresh the demonstration, restart the kernel, run all cells in
order, and save the notebook with outputs retained. Static plots need no active
kernel; animation playback requires a viewer with JavaScript/HTML support.
Saved outputs do not restore Python variables or replace generated data files.

`notebooks/earth_fast_forward.ipynb` is a fully standalone Earth-envmap experiment.
Upload this single file to Google Colab and choose **Runtime > Run all**; it does
not import this repository, read local assets, download data, or require Google
Drive. The notebook embeds its Earth target and contains both the direct
discrepancy loss and an optimizer with one backward pass per epoch. It records training snapshots,
plots the loss and initial/final point sets, optionally displays a fast-forward animation,
and writes a downloadable `.npz` result.

```bash
jupyter lab notebooks/earth_fast_forward.ipynb
```

## Outputs and visualization

Optimization writes a compressed `.npz` containing:

- `latents`: optimized `[N,2]` coordinates.
- `directions`: mapped `[N,3]` unit directions.
- `losses`: training loss history.
- `config_json`: the complete run configuration.
- `snapshot_epochs` and `latent_snapshots`: present when `--snapshot-every N` is used.

A PNG with the same basename is saved by default. Disable it with
`--no-visualize`, or select another path with `--visual-output`.

Saved `.npz` and raw `.npy` point sets can also be visualized independently:

```bash
python src/visualize.py outputs/quick_uniform.npz
python src/visualize.py directions.npy --output directions.png
python src/visualize.py batched_instances.npy --instance 2
```

The visualizer accepts `[N,2]` latents, `[N,3]` directions, and batched
`[B,N,2/3]` arrays.

## Tests

```bash
python src/test_minimal.py
```

The test suite covers mapping continuity, exact cap membership, white-map
uniformity and zero-probability texels, independent fallback estimation,
reference/batched latent and network gradients, BRDF local-view conditioning,
importance sampling of sparse environments and narrow GGX lobes, all three
optimization cases, notebook geometry, saved point sets, and headless PNG output.

## Feedback and reproducibility

When reporting an issue, include the command, Python and PyTorch versions,
runtime device, target data, and complete error message. Keep the configuration
and output files when sharing experimental results so that others can reproduce
and compare them.

## Release note

Add the intended open-source `LICENSE` at the repository root before publishing. No
license type is selected automatically by the extraction process.
