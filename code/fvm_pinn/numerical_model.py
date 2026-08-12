import torch
from riemann_solver import roe_flux_2d
import numpy as np

class GPUHydrodynamicModel:
    """
    Pure Mathematical 2D Finite Volume Method (FVM) Solver running on the GPU.
    No Neural Networks. No ground truth required.
    Strictly integrates the Shallow Water Equations explicitly in time using 
    a well-balanced Roe/Rusanov Riemann solver.
    """
    def __init__(self, cell_coords, cell_areas, cell_z, edge_index, edge_normals, edge_lengths, boundary_mask, device='cuda'):
        self.device = device
        self.cell_areas = torch.tensor(cell_areas, dtype=torch.float32, device=device).unsqueeze(1)
        self.cell_z = torch.tensor(cell_z, dtype=torch.float32, device=device).unsqueeze(1)
        
        self.c_L = torch.tensor(edge_index[0, :], dtype=torch.long, device=device)
        self.c_R = torch.tensor(edge_index[1, :], dtype=torch.long, device=device)
        self.nx = torch.tensor(edge_normals[:, 0:1], dtype=torch.float32, device=device)
        self.ny = torch.tensor(edge_normals[:, 1:2], dtype=torch.float32, device=device)
        self.e_len = torch.tensor(edge_lengths, dtype=torch.float32, device=device).unsqueeze(1)
        
        c_coords = torch.tensor(cell_coords, dtype=torch.float32, device=device)
        self.d_LR = torch.norm(c_coords[self.c_R] - c_coords[self.c_L], dim=1)
        self.d_LR = torch.clamp(self.d_LR, min=1.0)
        
        self.boundary_mask = boundary_mask
        self.num_cells = cell_areas.shape[0]
        
        self.g = 9.81
        self.manning_n = 0.019
        
    def simulate_one_step(self, h, u, v, h_still, dt):
        """
        Differentiable FVM step for PINN Physics Loss.
        Takes the current state tensors and steps them forward by dt.
        Returns h_next, u_next, v_next.
        """
        h_L, h_R = h[self.c_L], h[self.c_R]
        u_L, u_R = u[self.c_L], u[self.c_R]
        v_L, v_R = v[self.c_L], v[self.c_R]
        h_still_L, h_still_R = h_still[self.c_L], h_still[self.c_R]
        
        F_mass, F_mom_x, F_mom_y, wave_speed = roe_flux_2d(
            h_L, h_R, u_L, u_R, v_L, v_R, 
            h_still_L, h_still_R, self.nx, self.ny, self.g
        )
        
        F_mass *= self.e_len
        F_mom_x *= self.e_len
        F_mom_y *= self.e_len
        
        div_mass = torch.zeros_like(h)
        div_mom_x = torch.zeros_like(h)
        div_mom_y = torch.zeros_like(h)
        
        div_mass.scatter_add_(0, self.c_L.unsqueeze(1), F_mass)
        div_mass.scatter_add_(0, self.c_R.unsqueeze(1), -F_mass)
        
        div_mom_x.scatter_add_(0, self.c_L.unsqueeze(1), F_mom_x)
        div_mom_x.scatter_add_(0, self.c_R.unsqueeze(1), -F_mom_x)
        
        div_mom_y.scatter_add_(0, self.c_L.unsqueeze(1), F_mom_y)
        div_mom_y.scatter_add_(0, self.c_R.unsqueeze(1), -F_mom_y)
        
        div_mass /= self.cell_areas
        div_mom_x /= self.cell_areas
        div_mom_y /= self.cell_areas
        
        h_next = h - dt * div_mass
        
        dry_mask = (h_next < 0.005)
        h_next = torch.clamp(h_next, min=0.005)
        
        velocity_mag = torch.sqrt(u**2 + v**2 + 1e-6)
        friction_coeff = self.g * (self.manning_n ** 2) / (h**(1/3) + 1e-6)
        
        S_fx = -friction_coeff * u * velocity_mag
        S_fy = -friction_coeff * v * velocity_mag
        
        hu_next = h*u - dt * div_mom_x + dt * S_fx
        hv_next = h*v - dt * div_mom_y + dt * S_fy
        
        hu_next[dry_mask] = 0.0
        hv_next[dry_mask] = 0.0
        
        u_next = hu_next / h_next
        v_next = hv_next / h_next
        
        u_next = torch.clamp(u_next, min=-15.0, max=15.0)
        v_next = torch.clamp(v_next, min=-15.0, max=15.0)
        
        u_mag_next = torch.sqrt(u_next**2 + v_next**2 + 1e-8)
        friction = self.g * self.manning_n**2 * u_mag_next / (h_next**(4/3) + 1e-8)
        u_next = u_next / (1.0 + dt * friction)
        v_next = v_next / (1.0 + dt * friction)
        
        return h_next, u_next, v_next, wave_speed

    def simulate(self, initial_wl, boundary_wl_matrix, times_seconds):
        print(f"Starting GPU Explicit FVM Simulation for {len(times_seconds)} time steps.")
        
        h = torch.tensor(initial_wl, dtype=torch.float32, device=self.device).unsqueeze(1) - self.cell_z
        h = torch.clamp(h, min=0.01)
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)
        h_still = h.clone()
        
        pred_wl_matrix = []
        
        current_time = times_seconds[0]
        output_idx = 0
        dt = 0.0
        
        while output_idx < len(times_seconds):
            target_time = times_seconds[output_idx]
            
            while current_time < target_time:
                
                h_L, h_R = h[self.c_L], h[self.c_R]
                u_L, u_R = u[self.c_L], u[self.c_R]
                v_L, v_R = v[self.c_L], v[self.c_R]
                h_still_L, h_still_R = h_still[self.c_L], h_still[self.c_R]
                
                F_mass, F_mom_x, F_mom_y, wave_speed = roe_flux_2d(
                    h_L, h_R, u_L, u_R, v_L, v_R, 
                    h_still_L, h_still_R, self.nx, self.ny, self.g
                )
                
                edge_dt = 0.2 * self.d_LR.unsqueeze(1) / wave_speed
                dynamic_dt = torch.min(edge_dt).item()
                dt = torch.clamp(torch.tensor(dynamic_dt), min=1e-5, max=target_time - current_time).item()
                
                F_mass *= self.e_len
                F_mom_x *= self.e_len
                F_mom_y *= self.e_len
                
                div_mass = torch.zeros_like(h)
                div_mom_x = torch.zeros_like(h)
                div_mom_y = torch.zeros_like(h)
                
                div_mass.scatter_add_(0, self.c_L.unsqueeze(1), F_mass)
                div_mass.scatter_add_(0, self.c_R.unsqueeze(1), -F_mass)
                div_mom_x.scatter_add_(0, self.c_L.unsqueeze(1), F_mom_x)
                div_mom_x.scatter_add_(0, self.c_R.unsqueeze(1), -F_mom_x)
                div_mom_y.scatter_add_(0, self.c_L.unsqueeze(1), F_mom_y)
                div_mom_y.scatter_add_(0, self.c_R.unsqueeze(1), -F_mom_y)
                
                div_mass /= self.cell_areas
                div_mom_x /= self.cell_areas
                div_mom_y /= self.cell_areas
                
                h_next = h - dt * div_mass
                dry_mask = (h_next < 0.005)
                h_next = torch.clamp(h_next, min=0.005)
                
                hu_next = h*u - dt * div_mom_x
                hv_next = h*v - dt * div_mom_y
                hu_next[dry_mask] = 0.0
                hv_next[dry_mask] = 0.0
                
                u_next = hu_next / h_next
                v_next = hv_next / h_next
                u_next = torch.clamp(u_next, min=-15.0, max=15.0)
                v_next = torch.clamp(v_next, min=-15.0, max=15.0)
                
                u_mag_next = torch.sqrt(u_next**2 + v_next**2 + 1e-8)
                friction = self.g * self.manning_n**2 * u_mag_next / (h_next**(4/3) + 1e-8)
                u_next = u_next / (1.0 + dt * friction)
                v_next = v_next / (1.0 + dt * friction)
                
                if output_idx < len(times_seconds) - 1:
                    t0 = times_seconds[output_idx]
                    t1 = times_seconds[output_idx + 1]
                    w0 = boundary_wl_matrix[output_idx]
                    w1 = boundary_wl_matrix[output_idx + 1]
                    alpha = (current_time - t0) / (t1 - t0)
                    bc_wl = w0 * (1 - alpha) + w1 * alpha
                else:
                    bc_wl = boundary_wl_matrix[-1]
                
                bc_wl_tensor = torch.tensor(bc_wl, dtype=torch.float32, device=self.device)
                bc_h = torch.clamp(bc_wl_tensor - self.cell_z[self.boundary_mask].squeeze(1), min=0.01)
                
                h_next[self.boundary_mask, 0] = bc_h
                hu_next[self.boundary_mask, 0] = 0.0
                hv_next[self.boundary_mask, 0] = 0.0
                
                h = h_next
                u = u_next
                v = v_next
                current_time += dt
                
            eta = (h + self.cell_z).cpu().numpy()
            pred_wl_matrix.append(eta)
            print(f"Time {target_time/3600.0:5.2f} hrs reached | dynamic_dt: {dt:.2f}s | Mean WL: {eta.mean():.2f}m")
            output_idx += 1
            
        return np.array(pred_wl_matrix)
