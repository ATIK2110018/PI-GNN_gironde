import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
from scipy.interpolate import interp1d

from data_extractor import extract_fvm_geometry
from numerical_model import GPUHydrodynamicModel
from fvm_pinn_model import FVMPINNTrainer

def get_cells_near_line_dynamic(cell_coords_m, cell_areas, p1_deg, p2_deg):
    p1_m = p1_deg * np.array([78700.0, 111000.0])
    p2_m = p2_deg * np.array([78700.0, 111000.0])
    
    l2 = np.sum((p2_m - p1_m)**2)
    if l2 == 0: return np.zeros(cell_coords_m.shape[0], dtype=bool)
    
    t = np.sum((cell_coords_m - p1_m) * (p2_m - p1_m), axis=1) / l2
    t = np.clip(t, 0.0, 1.0)
    
    projection = p1_m + t[:, np.newaxis] * (p2_m - p1_m)
    dist_m = np.sqrt(np.sum((cell_coords_m - projection)**2, axis=1))
    
    local_threshold_m = np.sqrt(cell_areas.flatten()) * 2.0
    return dist_m < local_threshold_m

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simulate hydrodynamic event using trained FVM-PINN")
    parser.add_argument("--model", type=str, default=r"..\..\fvm_pinn_results\fvm_pinn_model_best.pth", help="Path to trained model (.pth)")
    parser.add_argument("--nc_file", type=str, default=r"..\..\data\input\FlowFM_net.nc", help="Path to NetCDF containing mesh")
    parser.add_argument("--bc_file", type=str, default=r"..\..\data\input\boundary_conditions.csv", help="Path to boundary conditions CSV")
    parser.add_argument("--output_dir", type=str, default=r"..\..\fvm_pinn_results\simulation", help="Output directory")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Extract Geometry
    cell_coords_t, cell_z_t, cell_areas_t, edge_index_t, edge_normals_t, edge_lengths_t, topo_boundary_mask = extract_fvm_geometry(args.nc_file, device=device)
    cell_coords_m = cell_coords_t.cpu().numpy()
    
    # Define exact boundaries like in training
    x_coords_m = cell_coords_m[:, 0] * 78700.0
    y_coords_m = cell_coords_m[:, 1] * 111000.0
    cell_coords_proj = np.column_stack((x_coords_m, y_coords_m))
    cell_areas_np = cell_areas_t.squeeze(1).cpu().numpy()
    
    p1_port = np.array([-1.055107109535667E+000, 4.558144911918696E+001])
    p2_port = np.array([-1.043691864509240E+000, 4.559334500610923E+001])
    port_mask = get_cells_near_line_dynamic(cell_coords_proj, cell_areas_np, p1_port, p2_port) & topo_boundary_mask
    
    p1_gar = np.array([-5.308167329151710E-001, 4.480884916128741E+001])
    p2_gar = np.array([-5.262550852925010E-001, 4.481051805675912E+001])
    gar_mask = get_cells_near_line_dynamic(cell_coords_proj, cell_areas_np, p1_gar, p2_gar) & topo_boundary_mask
    
    p1_dor = np.array([-2.586704969143130E-001, 4.491934439849670E+001])
    p2_dor = np.array([-2.586418807368147E-001, 4.491740422166230E+001])
    dor_mask = get_cells_near_line_dynamic(cell_coords_proj, cell_areas_np, p1_dor, p2_dor) & topo_boundary_mask
    
    exact_boundary_mask = port_mask | gar_mask | dor_mask
    boundary_mask_t = torch.tensor(exact_boundary_mask, dtype=torch.bool, device=device)
    
    fvm_model = GPUHydrodynamicModel(
        cell_coords=cell_coords_proj,
        cell_areas=cell_areas_np,
        cell_z=cell_z_t.cpu().numpy(),
        edge_index=edge_index_t.cpu().numpy(),
        edge_normals=edge_normals_t.cpu().numpy(),
        edge_lengths=edge_lengths_t.squeeze(1).cpu().numpy(),
        boundary_mask=boundary_mask_t,
        device=device
    )
    
    # 2. Load Model Checkpoint to get stats
    print(f"Loading checkpoint from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Retrieve scaling stats
    bc_mean = checkpoint['bc_mean']
    bc_std = checkpoint['bc_std']
    
    # 3. Load BCs
    print(f"Loading Boundary Conditions from {args.bc_file}")
    bc_df = pd.read_csv(args.bc_file)
    bc_interp = interp1d(bc_df['Time_s'].values, 
                         bc_df[['H_ocean', 'Q_garonne', 'Q_dordogne']].values, 
                         axis=0, kind='linear', fill_value="extrapolate")
                         
    # Create a 60s timestep array from start to end of BC
    t_start = bc_df['Time_s'].iloc[0]
    t_end = bc_df['Time_s'].iloc[-1]
    t_all_array = np.arange(t_start, t_end + 60, 60)
    bc_matrix = bc_interp(t_all_array)
    bc_matrix_norm = (bc_matrix - bc_mean) / bc_std
    
    # 4. Initialize Trainer (Wrapper)
    # Dummy matrices since it's inference
    dummy_wl = np.zeros((len(t_all_array), len(cell_coords_m)))
    
    trainer = FVMPINNTrainer(
        fvm_engine=fvm_model,
        cell_coords_m=torch.tensor(cell_coords_proj, dtype=torch.float32, device=device),
        true_wl_matrix=dummy_wl,
        times_seconds=t_all_array,
        boundary_mask=boundary_mask_t,
        bc_matrix_norm=bc_matrix_norm
    )
    
    # Override stats inside trainer from checkpoint
    trainer.coords_mean = torch.tensor(checkpoint['coords_mean'], device=device)
    trainer.coords_std = torch.tensor(checkpoint['coords_std'], device=device)
    trainer.z_mean = torch.tensor(checkpoint['z_mean'], device=device)
    trainer.z_std = torch.tensor(checkpoint['z_std'], device=device)
    
    # Update node embeddings norm
    coords_t = trainer.fvm.cell_coords
    trainer.norm_coords = (coords_t - trainer.coords_mean) / trainer.coords_std
    cell_z_flat = trainer.fvm.cell_z.squeeze(1)
    trainer.norm_z = ((cell_z_flat - trainer.z_mean) / trainer.z_std).unsqueeze(1)
    
    trainer.gnn.load_state_dict(checkpoint['model_state_dict'])
    trainer.gnn.eval()
    
    # 5. Run Inference
    print(f"Running Inference for {len(t_all_array)} time steps...")
    pred_wl_obs = []
    
    obs_points_deg = {
        'Lamena': [-0.795668429, 45.3363917],
        'Richard': [-0.923556245, 45.45450201],
        'Le Marquis': [-0.562383073, 45.00226741],
        'Fort Medoc': [-0.70027, 45.11798],
        'Pauillac': [-0.735575683, 45.18813893],
        'P4': [-0.711385121, 45.25392645],
        'P1': [-1.001201556, 45.59041047],
        'P2': [-0.909977411, 45.53429328]
    }
    obs_nodes = []
    obs_names = []
    for name, coords in obs_points_deg.items():
        c_m = np.array(coords) * np.array([78700.0, 111000.0])
        dist = np.sum((cell_coords_proj - c_m)**2, axis=1)
        obs_nodes.append(np.argmin(dist))
        obs_names.append(name)
        
    frames_for_anim = []
    
    with torch.no_grad():
        for i, t_val in enumerate(t_all_array):
            lagged_bc = trainer.get_lagged_bc(i)
            wl_all, _, _ = trainer.predict(lagged_bc=lagged_bc)
            wl_pred = wl_all[:, 0].cpu().numpy()
            
            pred_wl_obs.append(wl_pred[obs_nodes])
            
            if i % max(1, len(t_all_array) // 60) == 0:
                frames_for_anim.append((t_val, wl_pred))
                
            if (i+1) % 100 == 0:
                print(f"Step {i+1}/{len(t_all_array)} processed.")
                
    pred_wl_obs = np.array(pred_wl_obs)
    
    # 6. Save outputs
    print("Generating Observation Timeseries...")
    t_hours = t_all_array / 3600.0
    fig, axes = plt.subplots(4, 2, figsize=(20, 16), dpi=150, sharex=True)
    axes = axes.flatten()
    for i, (name, ax) in enumerate(zip(obs_names, axes)):
        ax.plot(t_hours, pred_wl_obs[:, i], 'r-', label='PINN Prediction', linewidth=2)
        ax.set_title(f'Station: {name}')
        ax.set_ylabel('Water Level (m)')
        ax.grid(True)
        ax.legend()
    axes[-2].set_xlabel('Time (Hours)')
    axes[-1].set_xlabel('Time (Hours)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'observation_predictions.png'))
    plt.close()
    
    print("Generating Animation...")
    triangulation = tri.Triangulation(cell_coords_proj[:, 0], cell_coords_proj[:, 1])
    
    x_tri = cell_coords_proj[triangulation.triangles, 0]
    y_tri = cell_coords_proj[triangulation.triangles, 1]
    edge1 = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
    edge2 = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
    edge3 = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
    max_edge = np.max(np.column_stack([edge1, edge2, edge3]), axis=1)
    triangulation.set_mask(max_edge > 2500.0)
    
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8), dpi=150)
    levels = np.linspace(-2.5, 3.5, 30)
    
    def update(frame_data):
        ax_anim.clear()
        t_val, wl_pred = frame_data
        t_hr = t_val / 3600.0
        tcf = ax_anim.tricontourf(triangulation, wl_pred, levels=levels, cmap='GnBu', extend='both')
        ax_anim.set_title(f"Simulation Water Level | Time: {t_hr:.2f} Hours")
        ax_anim.set_xlabel("X (m)")
        ax_anim.set_ylabel("Y (m)")
        return tcf,
        
    anim = animation.FuncAnimation(fig_anim, update, frames=frames_for_anim, interval=150)
    anim.save(os.path.join(args.output_dir, 'simulation_animation.gif'), writer='pillow')
    plt.close(fig_anim)
    
    print(f"Simulation finished! Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()
