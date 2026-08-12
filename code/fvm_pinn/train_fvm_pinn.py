import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting FVM-PINN Training on {device}")
    
    print("Extracting FVM Geometry...")
    nc_file = '/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc'
    cell_coords_t, cell_z, cell_areas, edge_index, edge_normals, edge_lengths, topo_boundary_mask = extract_fvm_geometry(nc_file, device=device)
    cell_coords = cell_coords_t.cpu().numpy()
    
    import netCDF4 as nc
    ds = nc.Dataset("/kaggle/input/datasets/atikurr/gironde-hydro-out/FlowFM_map.nc")
    
    raw_wl = ds.variables['mesh2d_s1'][:]
    if hasattr(raw_wl, 'filled'):
        raw_wl = raw_wl.filled(np.nan)
    true_wl_matrix = np.array(raw_wl, dtype=np.float32)
    
    cell_z_np = cell_z.cpu().numpy().flatten()
    invalid_mask = np.isnan(true_wl_matrix) | (true_wl_matrix < -900)
    true_wl_matrix[invalid_mask] = np.broadcast_to(cell_z_np, true_wl_matrix.shape)[invalid_mask]
    
    times_seconds = ds.variables['time'][:]
    ds.close()
    
    FAST_DEBUG_MODE = False
    
    if FAST_DEBUG_MODE:
        print("\n!!! FAST DEBUG MODE ENABLED !!!")
        print("Slicing data to 33.3 hours (120000 seconds) to yield exactly 2000 steps per epoch.")
        
        t_start = times_seconds[0]
        valid_indices = np.where(times_seconds <= t_start + 120000)[0]
        
        times_seconds = times_seconds[valid_indices]
        true_wl_matrix = true_wl_matrix[valid_indices, :]
    
    x_coords_m = cell_coords[:, 0] * 78700.0
    y_coords_m = cell_coords[:, 1] * 111000.0
    cell_coords_m = np.column_stack((x_coords_m, y_coords_m))
    cell_areas_np = cell_areas.squeeze(1).cpu().numpy()
    
    p1_port = np.array([-1.055107109535667E+000, 4.558144911918696E+001])
    p2_port = np.array([-1.043691864509240E+000, 4.559334500610923E+001])
    port_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_port, p2_port) & topo_boundary_mask
    
    p1_gar = np.array([-5.308167329151710E-001, 4.480884916128741E+001])
    p2_gar = np.array([-5.262550852925010E-001, 4.481051805675912E+001])
    gar_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_gar, p2_gar) & topo_boundary_mask
    
    p1_dor = np.array([-2.586704969143130E-001, 4.491934439849670E+001])
    p2_dor = np.array([-2.586418807368147E-001, 4.491740422166230E+001])
    dor_mask = get_cells_near_line_dynamic(cell_coords_m, cell_areas_np, p1_dor, p2_dor) & topo_boundary_mask
    
    exact_boundary_mask = port_mask | gar_mask | dor_mask
    boundary_mask_t = torch.tensor(exact_boundary_mask, device=device)
    
    fvm_model = GPUHydrodynamicModel(
        cell_coords=cell_coords_m,
        cell_areas=cell_areas_np,
        cell_z=cell_z.cpu().numpy(),
        edge_index=edge_index.cpu().numpy(),
        edge_normals=edge_normals.cpu().numpy(),
        edge_lengths=edge_lengths.squeeze(1).cpu().numpy(),
        boundary_mask=boundary_mask_t,
        device=device
    )
    
    print("Loading Boundary Conditions from CSV...")
    # Using Kaggle dataset structure: /kaggle/input/[dataset-name]/[filename]
    bc_df = pd.read_csv('/kaggle/input/datasets/atikurr/gironde-hydro-out/boundary_conditions.csv')
    
    # Interpolate BCs to match the exact times_seconds from NetCDF
    from scipy.interpolate import interp1d
    bc_interp = interp1d(bc_df['Time_s'].values, 
                         bc_df[['H_ocean', 'Q_garonne', 'Q_dordogne']].values, 
                         axis=0, kind='linear', fill_value="extrapolate")
    bc_matrix = bc_interp(np.array(times_seconds))
    
    # Normalize Boundary Conditions
    bc_mean = np.mean(bc_matrix, axis=0)
    bc_std = np.std(bc_matrix, axis=0) + 1e-6
    bc_matrix_norm = (bc_matrix - bc_mean) / bc_std
    
    trainer = FVMPINNTrainer(
        fvm_engine=fvm_model,
        cell_coords_m=torch.tensor(cell_coords_m, dtype=torch.float32, device=device),
        true_wl_matrix=true_wl_matrix,
        times_seconds=times_seconds,
        boundary_mask=boundary_mask_t,
        boundary_forcings=bc_matrix_norm
    )
    
    os.makedirs('/kaggle/working/outputs', exist_ok=True)
    
    plt.rcParams.update({
        'font.size': 14, 
        'axes.titlesize': 16, 
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    import matplotlib.tri as tri
    triangulation = tri.Triangulation(cell_coords_m[:, 0], cell_coords_m[:, 1])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    
    tcf = axes[0].tricontourf(triangulation, cell_z_np, levels=50, cmap='terrain')
    fig.colorbar(tcf, ax=axes[0], label='Bed Elevation (m)')
    axes[0].set_title("Estuary Bathymetry")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    
    manning_array = np.full_like(cell_z_np, 0.019)
    tcf2 = axes[1].tricontourf(triangulation, manning_array, levels=10, cmap='viridis')
    fig.colorbar(tcf2, ax=axes[1], label="Manning's n")
    
    axes[1].scatter(cell_coords_m[port_mask, 0], cell_coords_m[port_mask, 1], c='red', s=10, label='Ocean Boundary')
    axes[1].scatter(cell_coords_m[gar_mask, 0], cell_coords_m[gar_mask, 1], c='orange', s=10, label='Garonne Inflow')
    axes[1].scatter(cell_coords_m[dor_mask, 0], cell_coords_m[dor_mask, 1], c='magenta', s=10, label='Dordogne Inflow')
    
    axes[1].plot([p1_port[0]*78700, p2_port[0]*78700], [p1_port[1]*111000, p2_port[1]*111000], 'k-', linewidth=2, label='Boundary Segments')
    axes[1].plot([p1_gar[0]*78700, p2_gar[0]*78700], [p1_gar[1]*111000, p2_gar[1]*111000], 'k-', linewidth=2)
    axes[1].plot([p1_dor[0]*78700, p2_dor[0]*78700], [p1_dor[1]*111000, p2_dor[1]*111000], 'k-', linewidth=2)
    
    axes[1].set_title("Boundary Forcings & Friction")
    axes[1].set_xlabel("X (m)")
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/before_training_mesh.png', bbox_inches='tight')
    plt.close()
    
    print("Interpolating data to 1-minute (60s) intervals for continuous sequential training...")
    from scipy.interpolate import interp1d
    interp_func = interp1d(times_seconds, true_wl_matrix, axis=0, kind='linear')
    
    t_train_array = np.arange(times_seconds[0], times_seconds[-1] + 60, 60)
    
    t_train_array = t_train_array[t_train_array <= times_seconds[-1]]
    
    true_wl_interp = interp_func(t_train_array)
    bc_matrix_interp = bc_interp(t_train_array)
    bc_matrix_norm_interp = (bc_matrix_interp - bc_mean) / bc_std
    
    trainer.times_seconds = torch.tensor(t_train_array, dtype=torch.float32, device=device)
    trainer.true_wl_matrix = torch.tensor(true_wl_interp, dtype=torch.float32, device=device)
    trainer.boundary_forcings = torch.tensor(bc_matrix_norm_interp, dtype=torch.float32, device=device)
    
    loss_history_data = []
    loss_history_phys = []
    
    if 'FAST_DEBUG_MODE' in locals() and FAST_DEBUG_MODE:
        num_epochs = 5
    else:
        num_epochs = 50
        
    total_t_steps = len(t_train_array)
    
    print(f"Starting Randomized Spacetime Training ({num_epochs} Epochs over {total_t_steps} time steps)...")
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Starting Epoch {epoch}/{num_epochs} ---")
        
        epoch_int_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_ic_loss = 0.0
        epoch_phys_loss = 0.0
        
        # Expand window slightly every epoch so it reaches the full month around epoch 50
        window_size = int(min(2000 + (epoch - 1) * (total_t_steps / 50), total_t_steps))
        valid_t_indices = np.arange(window_size)
        
        # Option 2: Fast Epochs. We only sample 500 steps per epoch!
        steps_per_epoch = 500
        t_indices = np.random.choice(valid_t_indices, size=min(steps_per_epoch, window_size), replace=False)
        
        current_steps = len(t_indices)
        
        if epoch <= 10:
            current_phys_weight = 0.0
            print(f"  -> Epoch {epoch}: Purely Data-Driven Pre-training (Physics Weight = 0.0) | Window: {window_size} mins")
        else:
            current_phys_weight = 2.0
            print(f"  -> Epoch {epoch}: Full Physics Constraints (Physics Weight = 2.0) | Window: {window_size} mins")
        
        for step, t_idx in enumerate(t_indices):
            
            int_loss, bc_loss, ic_loss, p_loss = trainer.train_step(t_idx, phys_weight=current_phys_weight)
            
            epoch_int_loss += int_loss
            epoch_bc_loss += bc_loss
            epoch_ic_loss += ic_loss
            epoch_phys_loss += p_loss
            
            if (step + 1) % 100 == 0 or step == current_steps - 1:
                avg_int = epoch_int_loss / (step + 1)
                avg_bc = epoch_bc_loss / (step + 1)
                avg_ic = epoch_ic_loss / (step + 1)
                avg_phys = epoch_phys_loss / (step + 1)
                
                loss_history_data.append(avg_int + avg_bc + avg_ic)
                loss_history_phys.append(avg_phys)
                
                t_hr = t_train_array[t_idx] / 3600.0
                print(f"Epoch {epoch} | Step {step+1}/{current_steps} (Random Hour: {t_hr:.1f}) | Avg Data: {avg_int:.4f} | Avg BC: {avg_bc:.4f} | Avg IC: {avg_ic:.4f} | Avg Phys: {avg_phys:.4f}")
        
        trainer.scheduler.step()
        
        if avg_int < best_loss:
            best_loss = avg_int
            checkpoint = {
                'model_state_dict': trainer.pinn.state_dict(),
                't_min': trainer.t_min.item(),
                't_max': trainer.t_max.item(),
                'coords_mean': trainer.coords_mean.cpu().numpy(),
                'coords_std': trainer.coords_std.cpu().numpy(),
                'epoch': epoch,
                'loss': best_loss,
                'bc_mean': bc_mean,
                'bc_std': bc_std
            }
            torch.save(checkpoint, '/kaggle/working/outputs/fvm_pinn_model_best.pth')
            print(f"  -> Saved new best model checkpoint! (Data Loss: {best_loss:.4f})")
            
    final_checkpoint = {
        'model_state_dict': trainer.pinn.state_dict(),
        't_min': trainer.t_min.item(),
        't_max': trainer.t_max.item(),
        'coords_mean': trainer.coords_mean.cpu().numpy(),
        'coords_std': trainer.coords_std.cpu().numpy(),
        'bc_mean': bc_mean,
        'bc_std': bc_std,
        'epoch': num_epochs
    }
    torch.save(final_checkpoint, '/kaggle/working/outputs/fvm_pinn_model_final.pth')
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(loss_history_data, label='Data Loss', linewidth=2)
    plt.plot(loss_history_phys, label='FVM Physics Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('FVM-PINN Training Convergence')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/fvm_pinn_loss.png', bbox_inches='tight')
    plt.close()
    
    print("Evaluating full timeseries for 5 interior nodes...")
    trainer.pinn.eval()
    
    nodes_to_plot = [1000, 8000, 15000, 22000, 29000]
    times_hr = times_seconds / 3600.0
    
    pred_wl = np.zeros((len(times_seconds), len(nodes_to_plot)))
    
    with torch.no_grad():
        for t_idx, t_val in enumerate(times_seconds):
            norm_t = trainer.get_normalized_t(torch.tensor([t_val], dtype=torch.float32, device=device))
            node_coords_m = cell_coords_m[nodes_to_plot]
            norm_c = (torch.tensor(node_coords_m, dtype=torch.float32, device=device) - trainer.coords_mean) / trainer.coords_std
            norm_bc = torch.tensor(bc_matrix_norm[t_idx:t_idx+1], dtype=torch.float32, device=device)
            
            wl_pred_tensor, _, _ = trainer.predict(norm_t, norm_c, norm_bc)
            pred_wl[t_idx, :] = wl_pred_tensor.cpu().numpy().flatten()
            
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), dpi=300, sharex=True)
    for i, (node_id, ax) in enumerate(zip(nodes_to_plot, axes)):
        true_series = true_wl_matrix[:, node_id]
        pred_series = pred_wl[:, i]
        
        rmse = np.sqrt(np.mean((true_series - pred_series)**2))
        nse = 1 - np.sum((true_series - pred_series)**2) / np.sum((true_series - np.mean(true_series))**2)
        r2 = np.corrcoef(true_series, pred_series)[0, 1]**2
        
        ax.plot(times_hr, true_series, 'k-', label='True FVM Data', linewidth=2, alpha=0.7)
        ax.plot(times_hr, pred_series, 'r--', label='PINN Prediction', linewidth=2)
        ax.set_title(f'Water Level at Interior Node {node_id} | RMSE: {rmse:.3f} m | R²: {r2:.3f} | NSE: {nse:.3f}')
        ax.set_ylabel('Water Level (m)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    axes[-1].set_xlabel('Time (Hours)')
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/after_training_timeseries.png', bbox_inches='tight')
    plt.close()
    
    print("Evaluating full timeseries for 8 observation stations...")
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
        coords_m = np.array(coords) * np.array([78700.0, 111000.0])
        dist = np.sum((cell_coords_m - coords_m)**2, axis=1)
        nearest_node = np.argmin(dist)
        obs_nodes.append(nearest_node)
        obs_names.append(name)
        
    pred_wl_obs = np.zeros((len(times_seconds), len(obs_nodes)))
    
    with torch.no_grad():
        for t_idx, t_val in enumerate(times_seconds):
            norm_t = trainer.get_normalized_t(torch.tensor([t_val], dtype=torch.float32, device=device))
            node_coords_m = cell_coords_m[obs_nodes]
            norm_c = (torch.tensor(node_coords_m, dtype=torch.float32, device=device) - trainer.coords_mean) / trainer.coords_std
            norm_bc = torch.tensor(bc_matrix_norm[t_idx:t_idx+1], dtype=torch.float32, device=device)
            wl_pred_tensor, _, _ = trainer.predict(norm_t, norm_c, norm_bc)
            pred_wl_obs[t_idx, :] = wl_pred_tensor.cpu().numpy().flatten()
            
    fig, axes = plt.subplots(4, 2, figsize=(20, 16), dpi=300, sharex=True)
    axes = axes.flatten()
    for i, (node_id, name, ax) in enumerate(zip(obs_nodes, obs_names, axes)):
        true_series = true_wl_matrix[:, node_id]
        pred_series = pred_wl_obs[:, i]
        
        rmse = np.sqrt(np.mean((true_series - pred_series)**2))
        nse = 1 - np.sum((true_series - pred_series)**2) / np.sum((true_series - np.mean(true_series))**2)
        r2 = np.corrcoef(true_series, pred_series)[0, 1]**2
        
        ax.plot(times_hr, true_series, 'k-', label='True FVM Data', linewidth=2, alpha=0.7)
        ax.plot(times_hr, pred_series, 'r--', label='PINN Prediction', linewidth=2)
        ax.set_title(f'Station: {name} (Node {node_id}) | RMSE: {rmse:.3f} m | R²: {r2:.3f} | NSE: {nse:.3f}')
        ax.set_ylabel('Water Level (m)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
    axes[-2].set_xlabel('Time (Hours)')
    axes[-1].set_xlabel('Time (Hours)')
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/observation_points_timeseries.png', bbox_inches='tight')
    plt.close()
    
    print("Generating high-resolution spatial fields...")
    final_t_idx = len(times_seconds) - 1
    final_t_val = times_seconds[-1]
    
    with torch.no_grad():
        norm_t_final = trainer.get_normalized_t(torch.tensor([final_t_val], dtype=torch.float32, device=device))
        norm_bc_final = torch.tensor(bc_matrix_norm[final_t_idx:final_t_idx+1], dtype=torch.float32, device=device)
        wl_pred_final_tensor, u_pred_final_tensor, v_pred_final_tensor = trainer.predict(norm_t_final, trainer.norm_coords, norm_bc_final)
        wl_pred_final = wl_pred_final_tensor.cpu().numpy().flatten()
        u_pred_final = u_pred_final_tensor.cpu().numpy().flatten()
        v_pred_final = v_pred_final_tensor.cpu().numpy().flatten()
        
    true_wl_final = true_wl_matrix[final_t_idx]
    spatial_error = np.abs(wl_pred_final - true_wl_final)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=300)
    
    tcf1 = axes[0].tricontourf(triangulation, true_wl_final, levels=50, cmap='GnBu')
    fig.colorbar(tcf1, ax=axes[0], label='Water Level (m)')
    axes[0].set_title("True Water Level (Final Step)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    
    tcf2 = axes[1].tricontourf(triangulation, wl_pred_final, levels=50, cmap='GnBu')
    fig.colorbar(tcf2, ax=axes[1], label='Water Level (m)')
    axes[1].set_title("FVM-PINN Prediction")
    axes[1].set_xlabel("X (m)")
    
    vmax_err = np.percentile(spatial_error, 95)
    tcf3 = axes[2].tricontourf(triangulation, spatial_error, levels=50, cmap='Reds', vmax=vmax_err)
    fig.colorbar(tcf3, ax=axes[2], label='Absolute Error (m)')
    rmse_spatial = np.sqrt(np.mean(spatial_error**2))
    axes[2].set_title(f"Spatial Error | RMSE: {rmse_spatial:.3f} m")
    axes[2].set_xlabel("X (m)")
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/spatial_field_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("Generating velocity vector field...")
    plt.figure(figsize=(10, 8), dpi=300)
    
    step = max(1, len(cell_coords_m) // 3000)
    x_sub = cell_coords_m[::step, 0]
    y_sub = cell_coords_m[::step, 1]
    u_sub = u_pred_final[::step]
    v_sub = v_pred_final[::step]
    
    speed = np.sqrt(u_sub**2 + v_sub**2)
    
    plt.tricontourf(triangulation, wl_pred_final, levels=30, cmap='Blues', alpha=0.5)
    plt.colorbar(label='Water Level (m)')
    
    q = plt.quiver(x_sub, y_sub, u_sub, v_sub, speed, cmap='jet', scale=50, width=0.003)
    plt.colorbar(q, label='Velocity Magnitude (m/s)')
    plt.title('Predicted Velocity Vector Field')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/velocity_vector_field.png', bbox_inches='tight')
    plt.close()
    
    print("Generating water level simulation animation...")
    import matplotlib.animation as animation
    
    num_frames = min(60, len(times_seconds))
    frame_indices = np.linspace(0, len(times_seconds) - 1, num_frames, dtype=int)
    
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8), dpi=150)
    
    vmin = np.percentile(true_wl_matrix, 2)
    vmax = np.percentile(true_wl_matrix, 98)
    levels = np.linspace(vmin, vmax, 30)
    
    dummy_tcf = ax_anim.tricontourf(triangulation, np.full_like(wl_pred_final, vmin), levels=levels, cmap='GnBu')
    fig_anim.colorbar(dummy_tcf, ax=ax_anim, label='Water Level (m)')
    
    def update(frame_idx):
        ax_anim.clear()
        t_idx = frame_indices[frame_idx]
        t_val = times_seconds[t_idx]
        t_hr = t_val / 3600.0
        
        with torch.no_grad():
            norm_t = trainer.get_normalized_t(torch.tensor([t_val], dtype=torch.float32, device=device))
            norm_bc_anim = torch.tensor(bc_matrix_norm[t_idx:t_idx+1], dtype=torch.float32, device=device)
            wl_pred_tensor, _, _ = trainer.predict(norm_t, trainer.norm_coords, norm_bc_anim)
            wl_pred = wl_pred_tensor.cpu().numpy().flatten()
            
        tcf = ax_anim.tricontourf(triangulation, wl_pred, levels=levels, cmap='GnBu', extend='both')
        ax_anim.set_title(f"FVM-PINN Water Level Simulation | Time: {t_hr:.2f} Hours")
        ax_anim.set_xlabel("X (m)")
        ax_anim.set_ylabel("Y (m)")
        return tcf,
        
    anim = animation.FuncAnimation(fig_anim, update, frames=num_frames, interval=150)
    anim_file = '/kaggle/working/outputs/water_level_simulation.gif'
    try:
        anim.save(anim_file, writer='pillow')
        print(f"Animation saved to {anim_file}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
    plt.close(fig_anim)
    
    print("Training and Evaluation Complete! All plots saved to /kaggle/working/outputs")

if __name__ == "__main__":
    main()
