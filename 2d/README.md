# Gradient-based Approximation of Nonuniform Low-Discrepancy Samples: 2D

This directory contains the minimal 2D reproduction code for the
paper "Gradient-based Approximation of Nonuniform Low-Discrepancy Samples" 
for uniform and non-uniform point-set optimization experiments.

The bash scripts in `scripts/` are editable launch files for the optimization scripts

## Directory Layout

The bash scripts in `scripts/` are intended as editable launch files
for the programs in `src/`.
`data/` contains the results of the optimization and baseline discrepancy programs.
`targets/` contains files related to the target density functions.
`notebook/` contains the python notebook that visualizes some results.

## Installation

Python 3.10 or newer is recommended. Install the shared dependencies from
this directory using the repository's [requirements.txt](../requirements.txt):

```bash
python -m pip install -r ../requirements.txt
```

If PyTorch needs a CUDA-specific wheel, install the matching PyTorch build first
using the command recommended by PyTorch for the target machine, then run:

```bash
python -m pip install -r ../requirements.txt
```

## Targets

Following target functions are supported.

- `uniform`: uniform points optimized directly in the unit square.
- `bilinear`: non-uniform target with density proportional to `xy`.
- `quarter_disk`: non-uniform target supported on the first-quadrant disk.
- `image`: non-uniform target derived from an image intensity map.
