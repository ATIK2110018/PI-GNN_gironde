import torch
import torch.nn as nn
import numpy as np
from numerical_model import GPUHydrodynamicModel

class FourierFeatures(nn.Module):
    """
    Random Fourier Feature Mapping (Positional Encoding)
    Shatters the Spectral Bias so the network can learn high-frequency tidal waves.
    """
    def __init__(self, in_features=3, out_features=128, sigma=30.0):
        super().__init__()
        self.out_features = out_features
        # Fixed random matrix for projection
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * sigma, requires_grad=False)
        
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
        
        # CRITICAL FIX: The tidal wave has ~22 cycles over the normalized t=[0, 1].
        # If sigma=5.0, the network suffers from spectral bias and acts as a low-pass filter (flatlining).
        # We MUST set sigma >= 30.0 so the Fourier features can represent high-frequency sine waves.
        self.fourier = FourierFeatures(in_features=3, out_features=128, sigma=30.0)
        
        # Upgrade Architecture to handle complex FVM fluid dynamics (6 layers, 512 width)
        # We switch to SiLU (Swish) activation which has smoother 2nd derivatives and performs much better in PINNs
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
            nn.Linear(512, 3) # h, u, v
        )
        
    def forward(self, t, coords):
        t_expanded = t.expand(coords.size(0), 1)
        inputs = torch.cat([t_expanded, coords], dim=1)
        features = self.fourier(inputs)
        out = self.net(features)
        h = out[:, 0:1]
        u = out[:, 1:2]
        v = out[:, 2:3]
        return h, u, v

class FVMPINNTrainer:
    def __init__(self, fvm_engine: GPUHydrodynamicModel, cell_coords_m, true_wl_matrix, times_seconds, boundary_mask):
        self.fvm = fvm_engine
        self.device = fvm_engine.device
        
        # We need the boundary mask to apply strict Data Loss penalties at the boundaries
        self.boundary_mask = boundary_mask.clone().detach().to(dtype=torch.bool, device=self.device)
        self.interior_mask = ~self.boundary_mask
        
        # Coordinate Normalization
        coords_t = cell_coords_m.clone().detach().to(dtype=torch.float32, device=self.device)
        self.coords_mean = coords_t.mean(dim=0)
        self.coords_std = coords_t.std(dim=0)
        self.norm_coords = (coords_t - self.coords_mean) / self.coords_std
        
        self.t_min = times_seconds.min()
        self.t_max = times_seconds.max()
        
        self.true_wl_matrix = torch.tensor(true_wl_matrix, dtype=torch.float32, device=self.device)
        self.times_seconds = torch.tensor(times_seconds, dtype=torch.float32, device=self.device)
        
        self.pinn = HydroPINN().to(self.device)
        # Use AdamW for better regularization with GELU networks
        self.optimizer = torch.optim.AdamW(self.pinn.parameters(), lr=1e-3, weight_decay=1e-5)
        
    def get_normalized_t(self, t):
        return (t - self.t_min) / (self.t_max - self.t_min)

    def compute_physics_loss(self, t_val, dt):
        norm_t_curr = self.get_normalized_t(t_val.unsqueeze(0))
        t_next = t_val + dt
        norm_t_next = self.get_normalized_t(t_next.unsqueeze(0))
        
        h_curr, u_curr, v_curr = self.pinn(norm_t_curr, self.norm_coords)
        h_next, u_next, v_next = self.pinn(norm_t_next, self.norm_coords)
        
        # Clamp to avoid FVM crashes on dry cells or random negative initializations
        h_curr_safe = torch.clamp(h_curr, min=0.005)
        
        # Step the FVM explicitly to get EXACT physics future
        h_fvm_next, u_fvm_next, v_fvm_next, _ = self.fvm.simulate_one_step(
            h_curr_safe, u_curr, v_curr, self.fvm.cell_z, dt
        )
        
        # The NN's future prediction MUST exactly match the rigid FVM calculation
        loss_h = nn.MSELoss()(h_next, h_fvm_next.detach())
        loss_u = nn.MSELoss()(u_next, u_fvm_next.detach())
        loss_v = nn.MSELoss()(v_next, v_fvm_next.detach())
        
        return loss_h + loss_u + loss_v

    def train_step(self, t_idx, phys_weight=2.0):
        self.optimizer.zero_grad()
        
        t_val = self.times_seconds[t_idx]
        true_h = self.true_wl_matrix[t_idx].unsqueeze(1)
        
        # 1. Evaluate Data Loss
        norm_t_curr = self.get_normalized_t(t_val.unsqueeze(0))
        h_curr, u_curr, v_curr = self.pinn(norm_t_curr, self.norm_coords)
        
        # PINN predicts Depth (h). True data is Water Level (elevation).
        # Water Level = Depth + Bed Elevation (cell_z)
        wl_curr = h_curr + self.fvm.cell_z
        
        # CRITICAL FIX: Overwhelming boundary forcing!
        # If we don't force the PINN to respect the boundaries, it will predict a flat lake.
        loss_data_boundary = nn.MSELoss()(wl_curr[self.boundary_mask], true_h[self.boundary_mask])
        loss_data_interior = nn.MSELoss()(wl_curr[self.interior_mask], true_h[self.interior_mask])
        
        # We heavily weight the boundary condition so the wave is forced into the domain
        # Updated to match HydroNet standards: BC=30, Data=10
        data_loss = 10.0 * loss_data_interior + 30.0 * loss_data_boundary
        
        # 2. Evaluate Exact FVM Physics Loss
        phys_loss = self.compute_physics_loss(t_val, dt=1.0)
        
        # 3. Step Optimizer
        # Uses dynamic phys_weight based on paper's Warm-Up / PDD strategy
        total_loss = data_loss + phys_weight * phys_loss
        total_loss.backward()
        
        # Gradient clipping stabilizes stiff FVM gradients
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.optimizer.step()
        
        return loss_data_interior.item(), loss_data_boundary.item(), phys_loss.item()
