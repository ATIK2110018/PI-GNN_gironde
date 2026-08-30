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
        
    # Sheets start exactly at 2018-08-01 00:00:00 with 1-hour intervals.
    # Ignore existing time columns.
    master_start_date = pd.to_datetime('2018-08-01 00:00:00')
    
    sim_start_date = pd.to_datetime('2018-09-09 18:00:00') # 6 hours spin-up
    eval_start_date = pd.to_datetime('2018-09-10 00:00:00')
    end_date = pd.to_datetime('2018-09-16 23:59:59') # 7 Days
    
    processed_dfs = []
    
    for sheet_name, df in dfs.items():
        # Drop time or date columns
        cols_to_drop = [c for c in df.columns if 'date' in str(c).lower() or 'time' in str(c).lower()]
        df_clean = df.drop(columns=cols_to_drop)
        
        # Prefix columns with sheet name to prevent overlapping names
        rename_dict = {c: f"{sheet_name}_{c}" for c in df_clean.columns}
        df_clean = df_clean.rename(columns=rename_dict).reset_index(drop=True)
        
        processed_dfs.append(df_clean)
            
    # Concatenate side-by-side (row 1 matches row 1, etc.)
    unified_df = pd.concat(processed_dfs, axis=1)
    
    # Generate uniform time index
    unified_df['Unified_Time'] = pd.date_range(start=master_start_date, periods=len(unified_df), freq='h')
    
    # Slice dataframe for the simulation window (including spin-up)
    mask = (unified_df['Unified_Time'] >= sim_start_date) & (unified_df['Unified_Time'] <= end_date)
    unified_df = unified_df.loc[mask].copy().reset_index(drop=True)
    
    if unified_df.empty:
        print("Error: No data found for the date range 10-20 September 2018.")
        sys.exit(1)
        
    print(f"Unified data has {len(unified_df)} rows for the specified date range.")
    print(f"Available columns across all sheets: {unified_df.columns.tolist()}")
    
    # Find columns dynamically based on user description
    def get_data_col(keyword):
        return next((c for c in unified_df.columns if keyword in str(c).lower()), None)
        
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
    bc_df['Time_s'] = (unified_df['Unified_Time'] - sim_start_date).dt.total_seconds()
    bc_df['H_ocean'] = unified_df[port_col].values
    bc_df['Q_garonne'] = unified_df[gar_col].values
    bc_df['Q_dordogne'] = unified_df[dor_col].values
    
    bc_df = bc_df.sort_values('Time_s')
    
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
    
    import matplotlib.dates as mdates
    date_fmt = mdates.DateFormatter('%b %d')
    sim_start_date = pd.to_datetime('2018-09-09 18:00:00')
    
    t_dates = sim_start_date + pd.to_timedelta(bc_df['Time_s'], unit='s')
    
    axes[0].plot(t_dates, bc_df['H_ocean'], 'b-', linewidth=2)
    axes[0].set_title('Ocean Boundary (Port Bloc) - Water Level')
    axes[0].set_ylabel('WSE (m)')
    axes[0].grid(True)
    
    axes[1].plot(t_dates, bc_df['Q_garonne'], 'g-', linewidth=2)
    axes[1].set_title('Upstream Boundary - Garonne Discharge')
    axes[1].set_ylabel('Discharge (m³/s)')
    axes[1].grid(True)
    
    axes[2].plot(t_dates, bc_df['Q_dordogne'], 'r-', linewidth=2)
    axes[2].set_title('Upstream Boundary - Dordogne Discharge')
    axes[2].set_ylabel('Discharge (m³/s)')
    axes[2].set_xlabel('Date')
    axes[2].xaxis.set_major_formatter(date_fmt)
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
    sim_start_date = pd.to_datetime('2018-09-09 18:00:00')
    eval_start_date = pd.to_datetime('2018-09-10 00:00:00')
    
    # Calculate Time_s relative to sim_start_date to match df_pred exactly
    df_true_filtered['Time_s'] = (df_true_filtered['Unified_Time'] - sim_start_date).dt.total_seconds()
    
    # Exclude the 6-hour spin-up period for plotting and metrics
    spinup_seconds = (eval_start_date - sim_start_date).total_seconds()
    
    df_pred_eval = df_pred[df_pred['Time_s'] >= spinup_seconds].copy()
    df_true_eval = df_true_filtered[df_true_filtered['Time_s'] >= spinup_seconds].copy()
    
    # Normalize time axes so 0 is exactly at eval_start_date
    df_pred_eval['Time_Hours_Norm'] = (df_pred_eval['Time_s'] - spinup_seconds) / 3600.0
    df_true_eval['Time_s_Norm'] = df_true_eval['Time_s'] - spinup_seconds
    
    stations = []
    for c in df_pred.columns:
        if c.endswith('_WSE'):
            stations.append(c[:-4])
    
    # Filter stations with valid true data and skip Le Marquis
    valid_stations = []
    station_data = {}
    
    for station in stations:
        if station.lower() == 'le marquis':
            continue
            
        is_velocity = station in ['P1', 'P4']
        pred_col = f"{station}_U" if is_velocity else f"{station}_WSE"
        ylabel = "Velocity U (m/s)" if is_velocity else "Water Level (m)"
        
        search_str = "for medoc" if station.lower() == "fort medoc" else station.lower()
        true_col = next((c for c in df_true_eval.columns if search_str in str(c).lower() and 'ssc' not in str(c).lower()), None)
        
        if true_col:
            valid_mask = ~df_true_eval[true_col].isna()
            true_times = df_true_eval.loc[valid_mask, 'Time_s_Norm'].values
            true_vals = df_true_eval.loc[valid_mask, true_col].values
            
            if len(true_vals) > 5:
                valid_stations.append(station)
                station_data[station] = {
                    'pred_col': pred_col,
                    'ylabel': ylabel,
                    'true_times': true_times,
                    'true_vals': true_vals
                }
                
    if not valid_stations:
        print("No stations with valid true data found. Skipping validation plot.")
        return
        
    # Set publication quality settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })
        
    # --- 1. Create Combined Figure ---
    import matplotlib.dates as mdates
    date_fmt = mdates.DateFormatter('%b %d')
    
    df_pred_eval['Datetime'] = eval_start_date + pd.to_timedelta(df_pred_eval['Time_s_Norm'], unit='s')
    
    num_stations = len(valid_stations)
    cols = 2
    rows = (num_stations + 1) // 2
    fig_comb, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharex=True)
    
    if num_stations == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()
        
    for i, station in enumerate(valid_stations):
        data = station_data[station]
        ax = axes[i]
        
        ax.plot(df_pred_eval['Datetime'], df_pred_eval[data['pred_col']], 'r-', label='PI-GNN Prediction', linewidth=2)
        
        true_times = data['true_times']
        true_vals = data['true_vals']
        
        pred_interp = np.interp(true_times, df_pred_eval['Time_s_Norm'].values, df_pred_eval[data['pred_col']].values)
        
        var = np.sum((true_vals - np.mean(true_vals))**2)
        nse = 1 - np.sum((true_vals - pred_interp)**2) / (var + 1e-8)
        r2 = np.corrcoef(true_vals, pred_interp)[0, 1]**2 if var > 1e-6 else 0.0
        
        title = f'Station: {station} | R²: {r2:.3f} | NSE: {nse:.3f}'
        
        true_datetimes = eval_start_date + pd.to_timedelta(true_times, unit='s')
        ax.plot(true_datetimes, true_vals, 'k--', label='True Obs (Excel)', alpha=0.8, linewidth=1.5)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(data['ylabel'], fontweight='bold')
        ax.xaxis.set_major_formatter(date_fmt)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='upper right')
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig_comb.delaxes(axes[j])
        
    for ax in axes[-2:]:
        ax.set_xlabel('Date', fontweight='bold')
        
    plt.tight_layout()
    val_path = os.path.join(output_dir, 'validation_combined.png')
    fig_comb.savefig(val_path, bbox_inches='tight')
    plt.close(fig_comb)
    print(f"Combined validation plot saved to {val_path}")
    
    # --- 2. Create Individual Publication Figures ---
    for station in valid_stations:
        data = station_data[station]
        
        fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
        
        ax_ind.plot(df_pred_eval['Datetime'], df_pred_eval[data['pred_col']], 'r-', label='PI-GNN Prediction', linewidth=2.5)
        
        true_times = data['true_times']
        true_vals = data['true_vals']
        
        pred_interp = np.interp(true_times, df_pred_eval['Time_s_Norm'].values, df_pred_eval[data['pred_col']].values)
        
        var = np.sum((true_vals - np.mean(true_vals))**2)
        nse = 1 - np.sum((true_vals - pred_interp)**2) / (var + 1e-8)
        r2 = np.corrcoef(true_vals, pred_interp)[0, 1]**2 if var > 1e-6 else 0.0
        
        true_datetimes = eval_start_date + pd.to_timedelta(true_times, unit='s')
        ax_ind.plot(true_datetimes, true_vals, 'k--', label='True Obs (Excel)', alpha=0.8, linewidth=2.0)
        
        ax_ind.set_title(f'Validation at {station}\nR²: {r2:.3f} | NSE: {nse:.3f}', fontweight='bold', pad=15)
        ax_ind.set_ylabel(data['ylabel'], fontweight='bold', labelpad=10)
        ax_ind.set_xlabel('Date', fontweight='bold', labelpad=10)
        ax_ind.xaxis.set_major_formatter(date_fmt)
        
        ax_ind.grid(True, linestyle=':', alpha=0.7)
        ax_ind.legend(loc='upper right', framealpha=0.9)
        
        # Format axes slightly better for individual plots
        ax_ind.spines['top'].set_visible(False)
        ax_ind.spines['right'].set_visible(False)
        
        plt.tight_layout()
        ind_path = os.path.join(output_dir, f'validation_{station.replace(" ", "_")}.png')
        fig_ind.savefig(ind_path, bbox_inches='tight', dpi=300)
        plt.close(fig_ind)
        print(f"Individual plot saved for {station} to {ind_path}")

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


