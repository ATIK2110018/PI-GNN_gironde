# Gironde Estuary Parametric PI-GNN Surrogate for Tidal Hydrodynamics

## 1. Project Objective

Build a **parametric Physics-Informed Graph Neural Network (PI-GNN)** surrogate that learns the 2D Shallow Water Equations governing the Gironde estuary from D-Flow FM simulation data. The model takes a **7D input** `[t, x, y, z, H_ocean, Q_garonne, Q_dordogne]` and predicts the full hydrodynamic state `[η, u, v]` (water surface elevation, x-velocity, y-velocity) at any point in space-time, conditioned on dynamic boundary forcing.

## 2. Architecture

| Component | Description |
|-----------|-------------|
| **Input** | 7D: normalized time, spatial coords (x, y), bathymetry (z), boundary conditions (H, Q₁, Q₂) |
| **Fourier Features** | Random Fourier mapping (σ_t=30, σ_s=1) to overcome spectral bias for tidal oscillations |
| **Network** | 6-layer MLP (512 units each), SiLU activations, message passing over FVM faces |
| **Output** | 3D: water surface elevation (η), x-velocity (u), y-velocity (v) |
| **Mesh** | 36,271 unstructured FVM cells, 53,224 internal faces from D-Flow FM |

## 3. Loss Function

```
L_total = 10·L_data + 30·L_boundary + 5·L_velocity + 20·L_IC + λ_phys·L_SWE
```

| Term | Description |
|------|-------------|
| `L_data` | MSE of η at interior cells vs D-Flow FM ground truth |
| `L_boundary` | MSE of η at boundary cells (ocean, Garonne, Dordogne inlets) |
| `L_velocity` | MSE of (u, v) at interior cells vs D-Flow FM `mesh2d_ucx`/`mesh2d_ucy` |
| `L_IC` | MSE of η at t=0 (initial condition enforcement) |
| `L_SWE` | Autograd-based Shallow Water Equation residuals (mass + momentum) |

### SWE Physics Residuals (computed via `torch.autograd.grad` with chain-rule correction)

- **Mass**: `∂η/∂t + u·∂h/∂x + h·∂u/∂x + v·∂h/∂y + h·∂v/∂y = 0`
- **x-Momentum**: `∂u/∂t + u·∂u/∂x + v·∂u/∂y + g·∂η/∂x + friction = 0`
- **y-Momentum**: `∂v/∂t + u·∂v/∂x + v·∂v/∂y + g·∂η/∂y + friction = 0`

Bed slopes (`∂z/∂x`, `∂z/∂y`) are precomputed via Green-Gauss gradient reconstruction on the unstructured mesh.

## 4. Training Strategy

| Phase | Epochs | Physics Weight | Description |
|-------|--------|---------------|-------------|
| Data Pre-training | 1–10 | 0.0 | Pure data fitting (η, u, v) to establish base solution |
| Physics-Informed | 11–60 | 2.0 | Full SWE constraints activated |

- **Curriculum Learning**: Training window expands from 2,000 minutes to full dataset by Epoch 20
- **Steps per Epoch**: min(1000, window_size) random spacetime samples
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5) with ExponentialLR (γ=0.8)
- **Gradient Clipping**: max_norm=1.0

## 5. Validation

- **80/20 temporal train/test split**: First 80% (~210 hours) for training, last 20% (~55 hours) held out
- **Best model checkpoint**: Saved only during physics-constrained epochs (λ_phys > 0)
- **Metrics**: RMSE, R², NSE evaluated at 8 observation stations and 5 interior nodes

## 6. Domain Parameters

- **Manning's n**: 0.019 (uniform)
- **Gravity**: 9.81 m/s²
- **Coordinate system**: Lat/Lon scaled to meters (78,700 m/deg lon, 111,000 m/deg lat at ~45°N)
- **Minimum water depth**: 0.01 m (dry cell threshold)

## 7. Output Files

| File | Description |
|------|-------------|
| `fvm_pinn_model_best.pth` | Best checkpoint (physics-constrained epochs only) |
| `fvm_pinn_model_final.pth` | Final epoch checkpoint |
| `fvm_pinn_loss.png` | Training convergence (data + physics loss curves) |
| `after_training_timeseries.png` | η predictions at 5 interior nodes |
| `observation_points_timeseries.png` | η predictions at 8 observation stations |
| `spatial_field_comparison.png` | Spatial η field: true vs predicted vs error |
| `velocity_vector_field.png` | Predicted velocity vectors over water level |
| `water_level_simulation.gif` | Animated tidal simulation |
| `validation_timeseries.png` | Performance on UNSEEN test set (last 20%) |

## 8. Project Structure

```
code/fvm_pinn/
├── train.py               # Training + evaluation pipeline (entry point)
├── fvm_pinn_model.py      # HydroPI-GNN architecture + FVMPINNTrainer
├── numerical_model.py     # GPU FVM engine (bed slope computation)
├── data_extractor.py      # Unstructured mesh geometry extraction
└── requirements.txt       # Python dependencies
```

## 9. How to Run (Kaggle)

```bash
# Cell 1: Clone
!git clone https://github.com/ATIK2110018/PI-GNN_gironde.git code

# Cell 2: Train
%cd /kaggle/working/code/fvm_pinn
!python train.py
```

Requires Kaggle dataset: `atikurr/gironde-hydro-out` (containing `FlowFM_map.nc` and `boundary_conditions.csv`)
