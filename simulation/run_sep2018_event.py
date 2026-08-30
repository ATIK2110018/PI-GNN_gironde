import pandas as pd
import numpy as np
import os
import subprocess
import sys
import argparse

def prepare_data(excel_path, output_bc_file):
    print(f"Reading all sheets from {excel_path}...")
    
    try:
        # Read all sheets into a dictionary of dataframes
        dfs = pd.read_excel(excel_path, sheet_name=None)
    except Exception as e:
        print(f"Failed to read excel file: {e}")
        sys.exit(1)
        
    start_date = pd.to_datetime('2018-09-10 00:00:00')
    end_date = pd.to_datetime('2018-09-16 23:59:59') # 7 Days
    
    unified_df = None
    
    for sheet_name, df in dfs.items():
        # Find the time column for this specific sheet
        date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'time' in str(c).lower()), None)
        if not date_col:
            continue
            
        # Parse the datetime robustly
        df['Unified_Time'] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Unified_Time'])
        
        # CRITICAL FIX: Round time to nearest hour to force perfect alignment across sheets
        # This fixes any slight timestamp discrepancies (e.g., 10:01 vs 10:00) that would otherwise misalign rows!
        df['Unified_Time'] = df['Unified_Time'].dt.round('h')
        
        # Drop duplicates to prevent Scipy interpolation crashes (divide by zero)
        df = df.drop_duplicates(subset=['Unified_Time'], keep='first')
        
        # Filter for the event
        mask = (df['Unified_Time'] >= start_date) & (df['Unified_Time'] <= end_date)
        df_filtered = df.loc[mask].copy()
        
        if df_filtered.empty:
            continue
            
        # Prefix columns with sheet name
        rename_dict = {c: f"{sheet_name}_{c}" for c in df_filtered.columns if c != 'Unified_Time'}
        df_filtered = df_filtered.rename(columns=rename_dict)
        
        # Drop the original unrounded time column to avoid clutter
        original_time_col_renamed = f"{sheet_name}_{date_col}"
        if original_time_col_renamed in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=[original_time_col_renamed])
            
        # Merge on the perfectly rounded time!
        if unified_df is None:
            unified_df = df_filtered
        else:
            unified_df = pd.merge(unified_df, df_filtered, on='Unified_Time', how='outer')
            
    # Sort the final merged dataframe by time
    unified_df = unified_df.sort_values('Unified_Time').reset_index(drop=True)
    
    if unified_df.empty:
        print("Error: No data found for the date range 10-20 September 2018.")
        sys.exit(1)
        
    print(f"Unified data has {len(unified_df)} rows for the specified date range.")
    print(f"Available columns across all sheets: {unified_df.columns.tolist()}")
    
    # Find columns dynamically based on user description (exclude time/date columns)
    def get_data_col(keyword):
        return next((c for c in unified_df.columns if keyword in str(c).lower() and 'time' not in str(c).lower() and 'date' not in str(c).lower()), None)
        
    port_col = get_data_col('port')
    gar_col = get_data_col('garonne')
    dor_col = get_data_col('dordogne')
    
    if not all([port_col, gar_col, dor_col]):
        print(f"Detected columns - Port Block: {port_col}, Garonne: {gar_col}, Dordogne: {dor_col}")
        print("Please ensure the excel file has identifiable column names for these stations.")
        sys.exit(1)
        
    print(f"Mapped columns: Downstream={port_col}, Upstream_1={gar_col}, Upstream_2={dor_col}")
    
    # Create the BC dataframe as expected by simulate.py
    bc_df = pd.DataFrame()
    bc_df['Time_s'] = (unified_df['Unified_Time'] - start_date).dt.total_seconds()
    bc_df['H_ocean'] = unified_df[port_col].values
    bc_df['Q_garonne'] = unified_df[gar_col].values
    bc_df['Q_dordogne'] = unified_df[dor_col].values
    
    bc_df = bc_df.sort_values('Time_s')
    
    # The user noted timestamps might be slightly corrupted but serial is correct.
    # Interpolate to fill any NaN gaps that occurred during the outer merge!
    bc_df = bc_df.interpolate(method='linear').ffill().bfill()
    
    # Drop completely corrupted rows just in case and strictly enforce unique time!
    bc_df = bc_df.dropna(subset=['Time_s'])
    bc_df = bc_df.drop_duplicates(subset=['Time_s'], keep='first')
    
    bc_df.to_csv(output_bc_file, index=False)
    print(f"Saved boundary conditions to {output_bc_file}")
    
    # Also save the full unified dataframe for validation to use
    unified_out = os.path.join(os.path.dirname(output_bc_file), "unified_true_data.csv")
    unified_df.to_csv(unified_out, index=False)
    
    return output_bc_file

def plot_boundary_conditions(bc_file, output_dir):
    print("Plotting extracted Boundary Conditions...")
    import matplotlib.pyplot as plt
    bc_df = pd.read_csv(bc_file)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=150, sharex=True)
    
    t_hours = bc_df['Time_s'] / 3600.0
    
    axes[0].plot(t_hours, bc_df['H_ocean'], 'b-', linewidth=2)
    axes[0].set_title('Ocean Boundary (Port Bloc) - Water Level')
    axes[0].set_ylabel('WSE (m)')
    axes[0].grid(True)
    
    axes[1].plot(t_hours, bc_df['Q_garonne'], 'g-', linewidth=2)
    axes[1].set_title('Upstream Boundary - Garonne Discharge')
    axes[1].set_ylabel('Discharge (m³/s)')
    axes[1].grid(True)
    
    axes[2].plot(t_hours, bc_df['Q_dordogne'], 'r-', linewidth=2)
    axes[2].set_title('Upstream Boundary - Dordogne Discharge')
    axes[2].set_ylabel('Discharge (m³/s)')
    axes[2].set_xlabel('Time (Hours)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'extracted_boundary_conditions.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Boundary conditions plot saved to {plot_path}")

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
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False

def plot_validation(output_dir):
    print("Generating Validation Plots against True Observations...")
    import matplotlib.pyplot as plt
    
    pred_csv = os.path.join(output_dir, "station_predictions.csv")
    unified_csv = os.path.join(output_dir, "unified_true_data.csv")
    
    if not os.path.exists(pred_csv) or not os.path.exists(unified_csv):
        print("Predictions or True Data CSV not found. Skipping validation.")
        return
        
    df_pred = pd.read_csv(pred_csv)
    df_true_filtered = pd.read_csv(unified_csv)
    
    df_true_filtered['Unified_Time'] = pd.to_datetime(df_true_filtered['Unified_Time'])
    start_date = pd.to_datetime('2018-09-10 00:00:00')
    df_true_filtered['Time_s'] = (df_true_filtered['Unified_Time'] - start_date).dt.total_seconds()
    
    stations = []
    for c in df_pred.columns:
        if c.endswith('_WSE'):
            stations.append(c[:-4])
    
    fig, axes = plt.subplots(len(stations) // 2 + len(stations) % 2, 2, figsize=(20, 4 * (len(stations)//2)), dpi=150, sharex=True)
    axes = axes.flatten()
    
    for i, station in enumerate(stations):
        is_velocity = station in ['P1', 'P4']
        pred_col = f"{station}_U" if is_velocity else f"{station}_WSE"
        ylabel = "Velocity U (m/s)" if is_velocity else "Water Level (m)"
        
        # Try to find matching column in true data, excluding time, date, and ssc columns
        search_str = "for medoc" if station.lower() == "fort medoc" else station.lower()
        true_col = next((c for c in df_true_filtered.columns if search_str in str(c).lower() and 'time' not in str(c).lower() and 'date' not in str(c).lower() and 'ssc' not in str(c).lower()), None)
        
        ax = axes[i]
        ax.plot(df_pred['Time_Hours'], df_pred[pred_col], 'r-', label='PI-GNN Prediction', linewidth=2)
        
        if true_col:
            # Drop NaNs from true observations for accurate metrics
            valid_mask = ~df_true_filtered[true_col].isna()
            true_times = df_true_filtered.loc[valid_mask, 'Time_s'].values
            true_vals = df_true_filtered.loc[valid_mask, true_col].values
            
            if len(true_vals) > 5:
                # Interpolate prediction to the exact timestamps of true observations
                pred_interp = np.interp(true_times, df_pred['Time_s'].values, df_pred[pred_col].values)
                
                # Compute metrics
                var = np.sum((true_vals - np.mean(true_vals))**2)
                nse = 1 - np.sum((true_vals - pred_interp)**2) / (var + 1e-8)
                r2 = np.corrcoef(true_vals, pred_interp)[0, 1]**2 if var > 1e-6 else 0.0
                
                title = f'Station: {station} | R²: {r2:.3f} | NSE: {nse:.3f}'
                ax.plot(true_times / 3600.0, true_vals, 'k--', label='True Obs (Excel)', alpha=0.7)
            else:
                title = f'Station: {station} (Not enough valid true data)'
        else:
            title = f'Station: {station} (No true data found)'
            
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
    for ax in axes[-2:]:
        ax.set_xlabel('Time (Hours)')
        
    plt.tight_layout()
    val_path = os.path.join(output_dir, 'validation_vs_truth.png')
    plt.savefig(val_path)
    plt.close()
    print(f"Validation plot saved to {val_path}")

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
    plot_boundary_conditions(output_bc_file, args.output_dir)
    success = run_simulation(output_bc_file, args.simulate_script, args.model_path, args.nc_file, args.output_dir)
    
    if success:
        plot_validation(args.output_dir)


