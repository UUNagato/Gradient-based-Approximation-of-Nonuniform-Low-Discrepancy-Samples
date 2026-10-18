# Gradient-based Approximation of Nonuniform Low-Discrepancy Samples

English | [简体中文](README_CN.md)

This is the open-source repository for **Gradient-based Approximation of
Nonuniform Low-Discrepancy Samples**, published at **SIGGRAPH 2026 (Conference
Papers)**, by Xiangyu Li, Ege Ciklabakkal, Daniel Ritchie, and Toshiya Hachisuka.
It contains standalone 2D and 3D implementations for reproducing the paper.

[Paper (PDF)](https://cs.uwaterloo.ca/~thachisu/gradnuld.pdf) | [ACM / DOI](https://doi.org/10.1145/3799902.3811140)

## Example results

**2D:** discrepancy comparison between optimized point sets and QMC baselines
for the quarter-disk target.

![2D quarter-disk discrepancy comparison](assets/2d-quarter-disk-discrepancy.png)

**3D:** training loss and initial/optimized point sets for the Earth target.

![3D Earth optimization: training loss and point sets before and after optimization](assets/3d-earth-optimization.png)

## Setup

Python 3.10 or newer is recommended. Run these commands from this directory
(`opensource/`). Bash launchers require a Bash shell, such as Git Bash or WSL.

```bash
python -m pip install -r requirements.txt
```

## Run experiments

2D:

```bash
bash 2d/scripts/run_uniform.sh
bash 2d/scripts/run_nonuniform.sh
bash 2d/scripts/run_qmc_baselines.sh
```

Edit the 2D launchers to select targets and parameters; they use CUDA by default.
See the [2D README](2d/README.md).

3D:

```bash
bash 3d/scripts/run_uniform.sh
bash 3d/scripts/run_envmap.sh
bash 3d/scripts/run_cosine.sh
bash 3d/scripts/run_neural.sh --case brdf
bash 3d/scripts/run_neural.sh --case env
```

3D launchers accept extra arguments, for example
`bash 3d/scripts/run_uniform.sh --device cpu --epochs 10`.
Set `PYTHON_BIN` to choose the Python executable (default: `python3`).
Launcher results are saved under `2d/data/` and `3d/data/optimized/`.
See the [3D README](3d/README.md) for experiment descriptions, parameters,
direct Python commands, output formats, and tests.

## Open notebooks

```bash
jupyter lab 2d/notebooks/compare_2d_results.ipynb
jupyter lab 3d/notebooks/earth_fast_forward.ipynb
```

Both notebooks include saved outputs for previewing results.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{10.1145/3799902.3811140,
author = {Li, Xiangyu and Ciklabakkal, Ege and Ritchie, Daniel and Hachisuka, Toshiya},
title = {Gradient-based Approximation of Nonuniform Low-Discrepancy Samples},
year = {2026},
isbn = {9798400725548},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3799902.3811140},
doi = {10.1145/3799902.3811140},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
articleno = {117},
numpages = {10},
series = {SIGGRAPH Conference Papers '26}
}
```
