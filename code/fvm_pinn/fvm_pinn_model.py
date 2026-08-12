import torch
import torch.nn as nn
import numpy as np
from numerical_model import GPUHydrodynamicModel

class FourierFeatures(nn.Module):
    """
    Random Fourier Feature Mapping (Positional Encoding)
    Shatters the Spectral Bias so the network can learn high-frequency tidal waves.
    """
    def __init__(self, in_features=6, out_features=128, sigma_t=30.0, sigma_s=1.0):
        super().__init__()
        self.out_features = out_features
        B_t = torch.randn(1, out_features // 2) * sigma_t
        B_s = torch.randn(2, out_features // 2) * sigma_s
        B_bc = torch.randn(3, out_features // 2) * sigma_s
        self.B = nn.Parameter(torch.cat([B_t, B_s, B_bc], dim=0), requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class HydroPINN(nn.Module):
    """
    Neural Network predicting state (h, u, v) from (t, x, y)
    Uses Fourier Features to capture complex tidal cycles over 265 hours.
    """
    def __init__(self):
        super(HydroPINN, self).__init__()
        self.fourier = FourierFeatures(in_features=6, out_features=128, sigma_t=30.0, sigma_s=1.0)
        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 3)
        )
        
    def forward(self, inputs):
        features = self.fourier(inputs)
        out = self.net(features)
        wl = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        return wl, u, v

class FVMPINNTrainer:
    def __init__(self, fvm_engine: GPUHydrodynamicModel, cell_coords_m, true_wl_matrix, times_seconds, boundary_mask, boundary_forcings):
        self.fvm = fvm_engine
        self.device = fvm_engine.device
        
        self.boundary_mask = boundary_mask.clone().detach().to(dtype=torch.bool, device=self.device)
        self.interior_mask = ~self.boundary_mask
        
        coords_t = cell_coords_m.clone().detach().to(dtype=torch.float32, device=self.device)
        self.coords_mean = coords_t.mean(dim=0)
        self.coords_std = coords_t.std(dim=0)
        self.norm_coords = (coords_t - self.coords_mean) / self.coords_std
        
        self.t_min = times_seconds.min()
        self.t_max = times_seconds.max()
        
        self.true_wl_matrix = torch.tensor(true_wl_matrix, dtype=torch.float32, device=self.device)
        self.times_seconds = torch.tensor(times_seconds, dtype=torch.float32, device=self.device)
        self.boundary_forcings = torch.tensor(boundary_forcings, dtype=torch.float32, device=self.device)
        
        self.pinn = HydroPINN().to(self.device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
            self.pinn = nn.DataParallel(self.pinn)
            
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        
    def get_normalized_t(self, t):
        return (t - self.t_min) / (self.t_max - self.t_min)

    def predict(self, norm_t, norm_coords, norm_bc):
        t_expanded = norm_t.expand(norm_coords.size(0), 1)
        bc_expanded = norm_bc.expand(norm_coords.size(0), 3)
        inputs = torch.cat([t_expanded, norm_coords, bc_expanded], dim=1)
        return self.pinn(inputs)

    def compute_physics_loss(self, t_val, dt):
        norm_t_curr = self.get_normalized_t(t_val.unsqueeze(0))
        t_next = t_val + dt
        norm_t_next = self.get_normalized_t(t_next.unsqueeze(0))
        
        bc_curr = self.boundary_forcings[t_val == self.times_seconds][0].unsqueeze(0)
        # For simplicity in physics loss, we assume bc_curr is constant over dt=1.0
        
        wl_curr, u_curr, v_curr = self.predict(norm_t_curr, self.norm_coords, bc_curr)
        wl_next, u_next, v_next = self.predict(norm_t_next, self.norm_coords, bc_curr)
        
        h_curr = wl_curr - self.fvm.cell_z
        h_next = wl_next - self.fvm.cell_z
        
        h_curr_safe = torch.clamp(h_curr, min=0.005)
        
        h_fvm_next, u_fvm_next, v_fvm_next, _ = self.fvm.simulate_one_step(
            h_curr_safe, u_curr, v_curr, self.fvm.cell_z, dt
        )
        
        loss_h = nn.MSELoss()(h_next, h_fvm_next.detach())
        loss_u = nn.MSELoss()(u_next, u_fvm_next.detach())
        loss_v = nn.MSELoss()(v_next, v_fvm_next.detach())
        
        return loss_h + loss_u + loss_v

    def train_step(self, t_idx, phys_weight=2.0):
        self.optimizer.zero_grad()
        
        t_val = self.times_seconds[t_idx]
        true_h = self.true_wl_matrix[t_idx].unsqueeze(1)
        
        norm_t_curr = self.get_normalized_t(t_val.unsqueeze(0))
        bc_curr = self.boundary_forcings[t_idx].unsqueeze(0)
        
        wl_curr, u_curr, v_curr = self.predict(norm_t_curr, self.norm_coords, bc_curr)
        
        data_loss = nn.MSELoss()(wl_curr[self.interior_mask], true_h[self.interior_mask])
        boundary_loss = nn.MSELoss()(wl_curr[self.boundary_mask], true_h[self.boundary_mask])
        
        t_0 = self.times_seconds[0].unsqueeze(0)
        norm_t_0 = self.get_normalized_t(t_0)
        bc_0 = self.boundary_forcings[0].unsqueeze(0)
        wl_0, _, _ = self.predict(norm_t_0, self.norm_coords, bc_0)
        ic_loss = nn.MSELoss()(wl_0, self.true_wl_matrix[0].unsqueeze(1))
        
        pde_loss = self.compute_physics_loss(t_val, dt=1.0)
        
        total_loss = 10.0 * data_loss + 30.0 * boundary_loss + 20.0 * ic_loss + phys_weight * pde_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.optimizer.step()
        
        return data_loss.item(), boundary_loss.item(), ic_loss.item(), pde_loss.item()

