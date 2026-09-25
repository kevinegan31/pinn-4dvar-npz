# PI-NPZ 4D-Var

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17538590.svg)](https://doi.org/10.5281/zenodo.17538590)

Code and data for:

**Physics-Informed Neural Networks as Differentiable Surrogates for 4D-Variational Data Assimilation**  
Kevin Egan & Brian Powell

## Overview
This repository demonstrates how physics-informed neural networks (PINNs) can serve as differentiable surrogates in strong-constraint 4D-Var for biogeochemical state estimation.

## Models
- **Traditional-NPZ**: RK4-integrated Franks et al. (1986) NPZ model (`models/traditional_npz.py`)
- **PI-NPZ**: physics-informed surrogate trained with data and physics losses (`models/pi_npz.py`)
- **NN-NPZ**: data-only surrogate trained without the physics residual (`models/nn_npz.py`)

## Repository structure
- `training/`: hyperparameter optimization and retraining scripts for PI-NPZ and NN-NPZ
- `assimilation_files/`: 4D-Var twin experiments for Traditional-NPZ and PI-NPZ
- `nn_assimilation_files/`: the same twin experiments for NN-NPZ
- `model_checkpoints/`: trained PI-NPZ and NN-NPZ models
- `data/`: training/validation/test sets and processed assimilation results
- `analysis/figures/`: notebooks reproducing all manuscript figures

## Twin experiments
| Paper | Label in code and data |
|---|---|
| T1 (dense) | `baseline` / `two_samples_per_day` |
| T2 (irregular) | `irreg_sparse` / `irregular_sparse` |
| T3 (sparse) | `field_informed` |

Raw 4D-Var outputs are archived separately on Zenodo (see the Data Availability Statement in the paper).