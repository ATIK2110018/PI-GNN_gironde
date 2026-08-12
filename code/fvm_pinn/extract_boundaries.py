import pandas as pd
import numpy as np
import os

def parse_bc_file(filepath):
    """Parses a D-Flow FM .bc file and returns a dictionary of {time: value} for the first data block."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data = {}
    in_data_block = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('['):
            continue
            
        if '=' in line:
            # Metadata line
            continue
            
        # If it's just numbers, it's data
        parts = line.split()
        if len(parts) >= 2:
            try:
                time_val = float(parts[0])
                val = float(parts[1])
                data[time_val] = val
            except ValueError:
                pass
                
        # We only need the first block (first boundary node) as representative
        # If we see time decrease, we've hit the next block
        if len(data) > 1:
            keys = list(data.keys())
            if keys[-1] < keys[-2]:
                del data[keys[-1]]
                break
                
    return data

def main():
    base_dir = r"c:\Users\atikr\Desktop\hydrodynamic\hydrodynamic_calibration - Copy\calibration.dsproj_data\FlowFM\data\input"
    
    wl_path = os.path.join(base_dir, "WaterLevel.bc")
    gar_path = os.path.join(base_dir, "garonne.bc")
    dor_path = os.path.join(base_dir, "dordogne.bc")
    
    print("Parsing boundary files...")
    wl_data = parse_bc_file(wl_path)
    gar_data = parse_bc_file(gar_path)
    dor_data = parse_bc_file(dor_path)
    
    # Get common time steps (they should be identical, but we'll use WL as base)
    times = sorted(list(wl_data.keys()))
    
    # Interpolate garonne and dordogne to match WL times just in case
    gar_times = sorted(list(gar_data.keys()))
    gar_vals = [gar_data[t] for t in gar_times]
    gar_interp = np.interp(times, gar_times, gar_vals)
    
    dor_times = sorted(list(dor_data.keys()))
    dor_vals = [dor_data[t] for t in dor_times]
    dor_interp = np.interp(times, dor_times, dor_vals)
    
    wl_vals = [wl_data[t] for t in times]
    
    df = pd.DataFrame({
        'Time_s': times,
        'H_ocean': wl_vals,
        'Q_garonne': gar_interp,
        'Q_dordogne': dor_interp
    })
    
    out_path = os.path.join(base_dir, "boundary_conditions.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully saved boundary timeseries to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
