import pandas as pd
import numpy as np
import os
import subprocess
import sys
import argparse

def prepare_data(excel_path, output_bc_file):
    print(f"Reading {excel_path}...")
    
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Failed to read excel file: {e}")
        sys.exit(1)
    
    # Try to find the date column
    date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'time' in str(c).lower()), df.columns[0])
    
    # Ensure it's datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter for 10-20 September 2018
    # Using 10th Sep 00:00:00 to 20th Sep 23:59:59
    start_date = pd.to_datetime('2018-09-10 00:00:00')
    end_date = pd.to_datetime('2018-09-20 23:59:59')
    
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        print("Error: No data found for the date range 10-20 September 2018.")
        print(f"Available date range: {df[date_col].min()} to {df[date_col].max()}")
        sys.exit(1)
        
    print(f"Found {len(df_filtered)} rows for the specified date range.")
    
    # Find columns dynamically based on user description
    port_col = next((c for c in df.columns if 'port' in str(c).lower() or 'block' in str(c).lower()), None)
    gar_col = next((c for c in df.columns if 'garonne' in str(c).lower()), None)
    dor_col = next((c for c in df.columns if 'dordogne' in str(c).lower()), None)
    
    if not all([port_col, gar_col, dor_col]):
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Detected columns - Port Block: {port_col}, Garonne: {gar_col}, Dordogne: {dor_col}")
        print("Please ensure the excel file has identifiable column names for these stations.")
        sys.exit(1)
        
    print(f"Mapped columns: Time={date_col}, Downstream={port_col}, Upstream_1={gar_col}, Upstream_2={dor_col}")
    
    # Create the BC dataframe as expected by simulate.py
    bc_df = pd.DataFrame()
    
    # Time_s is seconds from the start of the simulation event (2018-09-10 00:00:00)
    bc_df['Time_s'] = (df_filtered[date_col] - start_date).dt.total_seconds()
    
    bc_df['H_ocean'] = df_filtered[port_col].values
    bc_df['Q_garonne'] = df_filtered[gar_col].values
    bc_df['Q_dordogne'] = df_filtered[dor_col].values
    
    # Drop rows with NaN if any exist in the required columns
    bc_df = bc_df.dropna()
    bc_df = bc_df.sort_values('Time_s')
    
    bc_df.to_csv(output_bc_file, index=False)
    print(f"Saved boundary conditions to {output_bc_file}")
    
    return output_bc_file

def run_simulation(bc_file, simulate_script, model_path, nc_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cmd = [
        sys.executable, simulate_script,
        "--model", model_path,
        "--nc_file", nc_file,
        "--bc_file", bc_file,
        "--output_dir", output_dir
    ]
    
    print(f"\nRunning simulation with command:\n{' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed with error code {e.returncode}")
    except Exception as e:
        print(f"Error running simulation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BCs from Excel and Run PI-GNN Simulation")
    
    # Default paths assume the script is run from the `simulation` folder locally
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    parser.add_argument("--excel_path", type=str, default=os.path.join(base_dir, "data", "All Data.xlsx"), help="Path to All Data.xlsx")
    parser.add_argument("--model_path", type=str, default=os.path.join(base_dir, "fvm_pinn_results", "fvm_pinn_model_best.pth"), help="Path to best model .pth")
    parser.add_argument("--nc_file", type=str, default=os.path.join(base_dir, "data", "input", "FlowFM_net.nc"), help="Path to FlowFM_net.nc")
    parser.add_argument("--simulate_script", type=str, default=os.path.join(base_dir, "code", "fvm_pinn", "simulate.py"), help="Path to simulate.py")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"), help="Directory to save simulation outputs")
    
    args = parser.parse_args()
    
    output_bc_file = os.path.join(args.output_dir, "boundary_conditions_sep2018.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    prepare_data(args.excel_path, output_bc_file)
    run_simulation(output_bc_file, args.simulate_script, args.model_path, args.nc_file, args.output_dir)

