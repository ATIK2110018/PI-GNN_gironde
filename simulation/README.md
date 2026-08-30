# PI-GNN Simulation: September 10-20, 2018

This folder contains the scripts and outputs for running the PI-GNN hydrodynamic surrogate model on the specific event of 10-20 September 2018.

## Files
- `run_sep2018_event.py`: The main script to run. It extracts the boundary conditions from `..\data\All Data.xlsx`, saves them as a CSV, and then executes the simulation script.
- `boundary_conditions_sep2018.csv`: Will be generated automatically by the script.
- `results/`: The output directory where the simulation GIF animation and observation time series plots will be saved.

## How to Run

1. Open your terminal or command prompt.
2. Activate your Python environment (the one with `pandas`, `torch`, `matplotlib`, etc. installed).
3. Navigate to this directory:
   ```bash
   cd simulation
   ```
4. Run the script:
   ```bash
   python run_sep2018_event.py
   ```

The script will extract the appropriate times and columns (Port Block, Garonne, Dordogne) from the excel file, prepare the data, and start inference with your best saved model (`fvm_pinn_model_best.pth`). 

Check the `results/` folder for your outputs!
