import os
import torch
import numpy as np
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
    # CRITICAL FALLBACK: FlowFM_net.nc only has 19,319 cells, but the target data has 36,271 cells.
    # We MUST use FlowFM_map.nc to ensure the geometry strictly matches the water level arrays.
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
    
    # 1. Instantiate the Differentiable FVM Physics Engine
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
    
    # 2. Instantiate the FVM-PINN Trainer
    trainer = FVMPINNTrainer(
        fvm_engine=fvm_model,
        cell_coords_m=torch.tensor(cell_coords_m, dtype=torch.float32, device=device),
        true_wl_matrix=true_wl_matrix,
        times_seconds=times_seconds,
        boundary_mask=boundary_mask_t
    )
    
    # ==========================================
    # PRE-TRAINING PLOTS (Mesh, Depth, Boundaries)
    # ==========================================
    os.makedirs('/kaggle/working/outputs', exist_ok=True)
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: Depth (Bed Elevation)
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(cell_coords_m[:, 0], cell_coords_m[:, 1], c=cell_z_np, cmap='terrain', s=1)
    plt.colorbar(sc1, label='Bed Elevation (m)')
    plt.title("Mesh & Depth (Cell Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    
    # Subplot 2: Boundary Lines & Manning's N
    plt.subplot(1, 2, 2)
    # Background mesh colored by constant manning's n
    manning_array = np.full_like(cell_z_np, 0.019)
    sc2 = plt.scatter(cell_coords_m[:, 0], cell_coords_m[:, 1], c=manning_array, cmap='viridis', s=1)
    plt.colorbar(sc2, label="Manning's n")
    
    # Overlay Boundaries
    plt.scatter(cell_coords_m[port_mask, 0], cell_coords_m[port_mask, 1], c='red', s=10, label='Ocean Boundary')
    plt.scatter(cell_coords_m[gar_mask, 0], cell_coords_m[gar_mask, 1], c='orange', s=10, label='Garonne Inflow')
    plt.scatter(cell_coords_m[dor_mask, 0], cell_coords_m[dor_mask, 1], c='magenta', s=10, label='Dordogne Inflow')
    
    # Also plot the original .pli line segments to verify exact extraction
    plt.plot([p1_port[0]*78700, p2_port[0]*78700], [p1_port[1]*111000, p2_port[1]*111000], 'k-', linewidth=2, label='.pli lines')
    plt.plot([p1_gar[0]*78700, p2_gar[0]*78700], [p1_gar[1]*111000, p2_gar[1]*111000], 'k-', linewidth=2)
    plt.plot([p1_dor[0]*78700, p2_dor[0]*78700], [p1_dor[1]*111000, p2_dor[1]*111000], 'k-', linewidth=2)
    
    plt.title("Boundary Cell Identification & Manning's n")
    plt.legend()
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/before_training_mesh.png')
    plt.close()
    
    # ==========================================
    # INTERPOLATE DATA TO 1-MINUTE TIME STEPS
    # ==========================================
    print("Interpolating data to 1-minute (60s) intervals for continuous sequential training...")
    from scipy.interpolate import interp1d
    interp_func = interp1d(times_seconds, true_wl_matrix, axis=0, kind='linear')
    
    t_train_array = np.arange(times_seconds[0], times_seconds[-1] + 60, 60)
    
    # Clip to avoid floating point overshoot
    t_train_array = t_train_array[t_train_array <= times_seconds[-1]]
    
    true_wl_interp = interp_func(t_train_array)
    
    # Update trainer with the new 1-minute resolution data
    trainer.times_seconds = torch.tensor(t_train_array, dtype=torch.float32, device=device)
    trainer.true_wl_matrix = torch.tensor(true_wl_interp, dtype=torch.float32, device=device)
    
    # ==========================================
    # TRAIN PINN (Sequential 1-Minute Time-Marching)
    # ==========================================
    
    loss_history_data = []
    loss_history_phys = []
    
    # =========================================================================
    # STRICT FORWARD TIME-MARCHING TRAINING (Numerical Solver Paradigm)
    # =========================================================================
    num_epochs = 5
    total_t_steps = len(t_train_array)
    
    print(f"Starting Strict Forward Time-Marching Training ({num_epochs} Epochs over {total_t_steps} time steps)...")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Starting Epoch {epoch}/{num_epochs} ---")
        
        # Reset accumulators for the epoch
        epoch_int_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_phys_loss = 0.0
        
        # STRICT FORWARD MARCHING: t_idx goes sequentially from 0 to max_time
        for t_idx in range(total_t_steps):
            
            int_loss, bc_loss, p_loss = trainer.train_step(t_idx)
            
            epoch_int_loss += int_loss
            epoch_bc_loss += bc_loss
            epoch_phys_loss += p_loss
            
            # Print average loss every 100 time steps so we can see progress faster
            if (t_idx + 1) % 100 == 0 or t_idx == total_t_steps - 1:
                avg_int = epoch_int_loss / (t_idx + 1)
                avg_bc = epoch_bc_loss / (t_idx + 1)
                avg_phys = epoch_phys_loss / (t_idx + 1)
                
                loss_history_data.append(avg_int + avg_bc)
                loss_history_phys.append(avg_phys)
                
                t_hr = t_train_array[t_idx] / 3600.0
                print(f"Epoch {epoch} | Step {t_idx+1}/{total_t_steps} (Hour {t_hr:.1f}) | Avg Data: {avg_int:.4f} | Avg BC: {avg_bc:.4f} | Avg Phys: {avg_phys:.4f}")
            
            
    # Save the trained model
    torch.save(trainer.pinn.state_dict(), '/kaggle/working/outputs/fvm_pinn_model.pth')
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history_data, label='Data Loss')
    plt.plot(loss_history_phys, label='FVM Physics Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('FVM-PINN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/outputs/fvm_pinn_loss.png')
    plt.close()
    
    # ==========================================
    # POST-TRAINING TIMESERIES COMPARISON
    # ==========================================
    
    print("Evaluating full timeseries for 3 interior nodes...")
    trainer.pinn.eval()
    
    nodes_to_plot = [5000, 15000, 25000] # Three distinct points in the estuary
    times_hr = times_seconds / 3600.0
    
    # We will build the predicted timeseries by querying the PINN
    pred_wl = np.zeros((len(times_seconds), len(nodes_to_plot)))
    
    with torch.no_grad():
        for t_idx, t_val in enumerate(times_seconds):
            # Force float32 to prevent Float vs Double dtype mismatch during inference
            norm_t = trainer.get_normalized_t(torch.tensor([t_val], dtype=torch.float32, device=device))
            
            # Extract coordinates for just the 3 nodes
            node_coords_m = cell_coords_m[nodes_to_plot]
            norm_c = (torch.tensor(node_coords_m, dtype=torch.float32, device=device) - trainer.coords_mean) / trainer.coords_std
            
            # Predict h
            h_pred, _, _ = trainer.pinn(norm_t, norm_c)
            # Water level = Depth (h) + Bed Elevation (cell_z)
            wl_pred = h_pred.cpu().numpy().flatten() + cell_z_np[nodes_to_plot]
            pred_wl[t_idx, :] = wl_pred
            
    plt.figure(figsize=(15, 10))
    for i, node_id in enumerate(nodes_to_plot):
        plt.subplot(3, 1, i+1)
        plt.plot(times_hr, true_wl_matrix[:, node_id], 'k--', label='True SRH-2D Data', linewidth=2)
        plt.plot(times_hr, pred_wl[:, i], 'r-', label='FVM-PINN Prediction', alpha=0.8, linewidth=2)
        plt.title(f'Water Level Timeseries at Interior Node {node_id}')
        plt.xlabel('Time (Hours)')
        plt.ylabel('Water Level (m)')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('/kaggle/working/outputs/after_training_timeseries.png')
    plt.close()
    
    print("Training and Evaluation Complete! All plots saved to /kaggle/working/outputs")

if __name__ == "__main__":
    main()
