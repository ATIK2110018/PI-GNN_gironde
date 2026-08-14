# Gironde Estuary PINN

A physics-informed neural network (PINN) project for learning surrogate hydrodynamics of the Gironde estuary from D-Flow FM simulation data.

## Project Goal

This repository builds a parametric surrogate model that predicts the 2D shallow-water state of the Gironde estuary:

- water surface elevation `η`
- x-velocity `u`
- y-velocity `v`

The model is conditioned on time, spatial location, bathymetry, and dynamic boundary forcing.

## Input and Output

### Inputs
The model uses a 7D input:

`[t, x, y, z, H_ocean, Q_garonne, Q_dordogne]`

Where:
- `t` = time
- `x, y` = spatial coordinates
- `z` = bathymetry / bed elevation
- `H_ocean` = ocean boundary water level forcing
- `Q_garonne` = Garonne river discharge forcing
- `Q_dordogne` = Dordogne river discharge forcing

### Outputs
The model predicts:

`[η, u, v]`

## What the Repository Contains

The codebase includes:

- **FVM geometry extraction** from D-Flow FM NetCDF files
- **Hydrodynamic numerical model** components for shallow-water simulation
- **PINN training code** for learning from data and physics constraints
- **Lagged boundary-condition handling** for tidal and river forcing history
- **Training and validation plotting** for diagnostics and evaluation

## Method Overview

The project combines:

- supervised learning from D-Flow FM simulation outputs
- physics-informed residual losses based on the shallow water equations
- boundary-condition conditioning using lagged forcing histories
- mesh-aware modeling over an unstructured finite-volume grid

## Repository Structure

Typical key files are located under `code/fvm_pinn/`:

- `train.py` — main training / evaluation script
- `fvm_pinn_model.py` — PINN and graph-style message passing components
- `numerical_model.py` — GPU finite-volume hydrodynamic solver utilities
- `data_extractor.py` — NetCDF mesh and geometry extraction helpers
- `requirements.txt` — Python dependencies

## Requirements

Install the Python dependencies with:

```bash
pip install -r code/fvm_pinn/requirements.txt
```

## Notes

- The project expects D-Flow FM NetCDF data files as input.
- Some paths in the training script are configured for a Kaggle environment and may need to be updated for local use.
- The model is designed for unstructured mesh hydrodynamics and tidal boundary forcing.

## Getting Started

1. Clone the repository.
2. Install dependencies.
3. Provide the required NetCDF simulation data.
4. Update file paths in `train.py` if needed.
5. Run the training script.

## Citation / Acknowledgment

If you use this project, please acknowledge the original repository and the D-Flow FM simulation data used for training.
