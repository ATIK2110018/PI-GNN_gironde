# Gironde Estuary FVM-PINN: Requirements, Problems, and Progress

## 1. Clear Statement of Requirements (What the User Wants)
* **Numerical Solver Paradigm:** The Neural Network must act **exactly** like a traditional numerical hydrodynamic solver (forward-marching in time). It should not act like a naive global surrogate mapping function that randomly looks at past or future data.
* **Physics-Driven:** The explicit FVM physics (Shallow Water Equations) must be the primary driving force. The observational data is only meant to "guide" the model, not act as a shortcut for the model to blindly memorize the answers.
* **Geometry Extraction Compromise:** While strict data separation is ideal, the raw input file (`FlowFM_net.nc`) only contains 19,319 cells, whereas the target output data has 36,271 cells. Therefore, the FVM geometry (mesh topology, cell locations, and bed elevations) must be extracted from the Kaggle output file (`FlowFM_map.nc`) to ensure the spatial structure strictly matches the output arrays.
* **Domain Parameters:** The Manning's roughness coefficient ($n$) must be strictly set to `0.019` for the entire domain.

## 2. The Core Problem We Are Facing
The primary obstacle during training is the **"Hydrostatic Trap" (Flatline Phenomenon)**.
* When the Neural Network initially guesses the water depth, it tends to predict a completely flat water level with zero velocity.
* Unfortunately, a perfectly still, flat lake with zero velocity is an **exact mathematical solution** to the Shallow Water Equations. 
* Because it is a valid physical state, the rigorous FVM Physics Loss evaluates to exactly `0.0`. The Neural Network falls into this local minimum trap because predicting a flat line is mathematically infinitely easier than predicting a complex, propagating 265-hour tidal wave across 36,271 unstructured cells.

## 3. What We Have Tried (and Implemented) Till Now
1. **Explicit FVM Physics Engine:** Built the `GPUHydrodynamicModel` incorporating explicit Euler time-stepping and a Roe-flux Riemann solver to perfectly replicate standard numerical hydrodynamics on the GPU.
2. **Data Loss Correction (Depth vs Elevation):** Discovered the Neural Network was predicting Water Depth ($h$), but the Data Loss was penalizing it against Water Level ($WL$). Fixed this by calculating `Water Level = Depth + Bed Elevation` in the loss function to prevent the model from instantly draining the estuary.
3. **Manning's $n$ Correction:** Hardcoded Manning's $n$ to exactly `0.019` throughout the codebase.
4. **Fourier Features (Positional Encoding):** Injected Fourier Features into the Neural Network architecture with an extremely high frequency scale (`sigma=30.0`). Standard neural networks suffer from "Spectral Bias" and cannot bend fast enough to resolve 21 tidal cycles. This upgrade mathematically guarantees the network can draw sharp tidal peaks.
5. **Loss Balancing (100x Multipliers):** 
   * Multiplied the Boundary Data Loss by `100.0` to force the network to feel the ocean tide hitting the boundary.
   * Multiplied the Physics Loss by `100.0` to ensure the optimizer prioritizes gravity and momentum conservation, rather than lazily overfitting the data.
6. **Strict Forward Time-Marching (Numerical Solver Paradigm):** Removed all randomized training windows and "Replay Buffers". The model now trains purely sequentially at interpolated 1-minute temporal intervals, marching strictly forward in time without "cheating" by peeking at past data.
7. **Input Geometry Extraction:** Acknowledged the cell-count mismatch (19k vs 36k) and updated `data_extractor.py` and `train_fvm_pinn.py` to securely load the geometry directly from the output dataset (`FlowFM_map.nc`) to ensure perfect array alignment during training.
