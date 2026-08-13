import torch
import torch.nn as nn
import numpy as np
from numerical_model import GPUHydrodynamicModel

class FourierFeatures(nn.Module):
    """
    Random Fourier Feature Mapping (Positional Encoding)
    Shatters the Spectral Bias so the network can learn high-frequency tidal waves.
    Input: [t, x, y, z, H_ocean, Q_garonne, Q_dordogne] = 7D
    """
    def __init__(self, in_features=7, out_features=128, sigma_t=30.0, sigma_s=1.0):
        super().__init__()
        self.out_features = out_features
        B_t = torch.randn(1, out_features // 2) * sigma_t       # time
        B_s = torch.randn(2, out_features // 2) * sigma_s       # space (x, y)
        B_z = torch.randn(1, out_features // 2) * sigma_s       # bathymetry
        B_bc = torch.randn(3, out_features // 2) * sigma_s      # boundary conditions
        self.B = nn.Parameter(torch.cat([B_t, B_s, B_z, B_bc], dim=0), requires_grad=False)
        
    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class HydroPINN(nn.Module):
    """
    Neural Network predicting state (eta, u, v) from (t, x, y, z, H, Q1, Q2).
    eta = water surface elevation, u = x-velocity, v = y-velocity.
    Uses Fourier Features to capture complex tidal cycles over 265 hours.
    """
    def __init__(self):
        super(HydroPINN, self).__init__()
        self.fourier = FourierFeatures(in_features=7, out_features=128, sigma_t=30.0, sigma_s=1.0)
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
        
        # Normalize bathymetry (cell_z) for network input
        cell_z_flat = self.fvm.cell_z.squeeze(1)  # [N]
        self.cell_z_flat = cell_z_flat  # physical units, for h = eta - z
        self.z_mean = cell_z_flat.mean()
        self.z_std = cell_z_flat.std() + 1e-6
        self.norm_z = ((cell_z_flat - self.z_mean) / self.z_std).unsqueeze(1)  # [N, 1]
        
        self.t_min = times_seconds.min()
        self.t_max = times_seconds.max()
        
        # Chain-rule scale factors for converting normalized derivatives to physical
        self.dt_scale = 1.0 / (self.t_max - self.t_min + 1e-8)
        self.dx_scale = 1.0 / self.coords_std[0]
        self.dy_scale = 1.0 / self.coords_std[1]
        
        self.true_wl_matrix = torch.tensor(true_wl_matrix, dtype=torch.float32, device=self.device)
        self.times_seconds = torch.tensor(times_seconds, dtype=torch.float32, device=self.device)
        self.boundary_forcings = torch.tensor(boundary_forcings, dtype=torch.float32, device=self.device)
        
        # Physics constants
        self.g = 9.81
        self.manning_n = 0.019
        
        # Precompute bed slopes for mass conservation equation
        self._precompute_bed_slopes()
        
        self.pinn = HydroPINN().to(self.device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
            self.pinn = nn.DataParallel(self.pinn)
            
        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
    
    def _precompute_bed_slopes(self):
        """Compute dz/dx and dz/dy at each cell using Green-Gauss gradient reconstruction."""
        z = self.fvm.cell_z  # [N, 1]
        dz_dx = torch.zeros_like(z)
        dz_dy = torch.zeros_like(z)
        
        z_face = 0.5 * (z[self.fvm.c_L] + z[self.fvm.c_R])  # [E, 1]
        flux_x = z_face * self.fvm.nx * self.fvm.e_len       # [E, 1]
        flux_y = z_face * self.fvm.ny * self.fvm.e_len       # [E, 1]
        
        dz_dx.scatter_add_(0, self.fvm.c_L.unsqueeze(1), flux_x)
        dz_dx.scatter_add_(0, self.fvm.c_R.unsqueeze(1), -flux_x)
        dz_dy.scatter_add_(0, self.fvm.c_L.unsqueeze(1), flux_y)
        dz_dy.scatter_add_(0, self.fvm.c_R.unsqueeze(1), -flux_y)
        
        dz_dx = dz_dx / self.fvm.cell_areas
        dz_dy = dz_dy / self.fvm.cell_areas
        
        self.bed_dz_dx = dz_dx.squeeze(1).detach()  # [N]
        self.bed_dz_dy = dz_dy.squeeze(1).detach()  # [N]
        print(f"Precomputed bed slopes: dz/dx range [{self.bed_dz_dx.min():.6f}, {self.bed_dz_dx.max():.6f}]")

    def get_normalized_t(self, t):
        return (t - self.t_min) / (self.t_max - self.t_min)

    def predict(self, norm_t, norm_coords, norm_bc, norm_z=None):
        """Standard forward pass for data loss and inference."""
        t_expanded = norm_t.expand(norm_coords.size(0), 1)
        bc_expanded = norm_bc.expand(norm_coords.size(0), 3)
        if norm_z is None:
            norm_z = self.norm_z  # Use all cells
        z_expanded = norm_z.expand(norm_coords.size(0), 1) if norm_z.dim() == 1 else norm_z[:norm_coords.size(0)]
        inputs = torch.cat([t_expanded, norm_coords, z_expanded, bc_expanded], dim=1)
        return self.pinn(inputs)

    def compute_physics_loss(self, t_idx):
        """
        Compute Shallow Water Equation residuals using torch.autograd.
        
        Mass:       deta/dt + d(hu)/dx + d(hv)/dy = 0
        x-Momentum: du/dt + u*du/dx + v*du/dy + g*deta/dx + friction_x = 0
        y-Momentum: dv/dt + u*dv/dx + v*dv/dy + g*deta/dy + friction_y = 0
        
        All derivatives are computed in PHYSICAL units via chain-rule correction.
        """
        # Sample collocation points for efficiency (not all 36k cells)
        n_colloc = min(4000, self.norm_coords.size(0))
        idx = torch.randperm(self.norm_coords.size(0), device=self.device)[:n_colloc]
        
        t_val = self.times_seconds[t_idx]
        bc_curr = self.boundary_forcings[t_idx]
        norm_t_val = self.get_normalized_t(t_val.unsqueeze(0))
        
        # Create inputs WITH requires_grad for autograd differentiation
        t_input = norm_t_val.expand(n_colloc, 1).clone().requires_grad_(True)
        xy_input = self.norm_coords[idx].clone().requires_grad_(True)
        z_input = self.norm_z[idx]  # no grad needed for z
        bc_input = bc_curr.unsqueeze(0).expand(n_colloc, 3)
        
        inputs = torch.cat([t_input, xy_input, z_input, bc_input], dim=1)
        eta, u, v = self.pinn(inputs)
        
        # Water depth h = eta - z_bed (physical units)
        z_phys = self.cell_z_flat[idx].unsqueeze(1)
        h = torch.clamp(eta - z_phys, min=0.01)
        
        ones = torch.ones_like(eta)
        
        # --- Autograd derivatives w.r.t. normalized time ---
        deta_dt_n = torch.autograd.grad(eta, t_input, ones, create_graph=True, retain_graph=True)[0]
        du_dt_n = torch.autograd.grad(u, t_input, ones, create_graph=True, retain_graph=True)[0]
        dv_dt_n = torch.autograd.grad(v, t_input, ones, create_graph=True, retain_graph=True)[0]
        
        # --- Autograd derivatives w.r.t. normalized (x, y) ---
        deta_dxy_n = torch.autograd.grad(eta, xy_input, ones, create_graph=True, retain_graph=True)[0]
        du_dxy_n = torch.autograd.grad(u, xy_input, ones, create_graph=True, retain_graph=True)[0]
        dv_dxy_n = torch.autograd.grad(v, xy_input, ones, create_graph=True, retain_graph=True)[0]
        
        # --- Chain-rule correction: normalized -> physical units ---
        deta_dt = deta_dt_n * self.dt_scale
        du_dt = du_dt_n * self.dt_scale
        dv_dt = dv_dt_n * self.dt_scale
        
        deta_dx = deta_dxy_n[:, 0:1] * self.dx_scale
        deta_dy = deta_dxy_n[:, 1:2] * self.dy_scale
        du_dx = du_dxy_n[:, 0:1] * self.dx_scale
        du_dy = du_dxy_n[:, 1:2] * self.dy_scale
        dv_dx = dv_dxy_n[:, 0:1] * self.dx_scale
        dv_dy = dv_dxy_n[:, 1:2] * self.dy_scale
        
        # --- Precomputed bed slopes at sampled cells ---
        dz_dx = self.bed_dz_dx[idx].unsqueeze(1)
        dz_dy = self.bed_dz_dy[idx].unsqueeze(1)
        
        # dh/dx = deta/dx - dz/dx (since h = eta - z_bed)
        dh_dx = deta_dx - dz_dx
        dh_dy = deta_dy - dz_dy
        
        # --- SWE Residuals ---
        # Mass conservation: deta/dt + d(hu)/dx + d(hv)/dy = 0
        # d(hu)/dx = u*dh/dx + h*du/dx
        # d(hv)/dy = v*dh/dy + h*dv/dy
        R_mass = deta_dt + u * dh_dx + h * du_dx + v * dh_dy + h * dv_dy
        
        # Manning friction: C_f = g * n^2 / h^(1/3)
        vel_mag = torch.sqrt(u**2 + v**2 + 1e-8)
        C_f = self.g * (self.manning_n ** 2) / (h ** (1.0/3.0) + 1e-8)
        
        # x-Momentum: du/dt + u*du/dx + v*du/dy + g*deta/dx + C_f*u*|V|/h = 0
        R_mom_x = du_dt + u * du_dx + v * du_dy + self.g * deta_dx + C_f * u * vel_mag / (h + 1e-8)
        
        # y-Momentum: dv/dt + u*dv/dx + v*dv/dy + g*deta/dy + C_f*v*|V|/h = 0
        R_mom_y = dv_dt + u * dv_dx + v * dv_dy + self.g * deta_dy + C_f * v * vel_mag / (h + 1e-8)
        
        loss_mass = torch.mean(R_mass ** 2)
        loss_mom_x = torch.mean(R_mom_x ** 2)
        loss_mom_y = torch.mean(R_mom_y ** 2)
        
        return loss_mass + loss_mom_x + loss_mom_y

    def train_step(self, t_idx, phys_weight=2.0):
        self.optimizer.zero_grad()
        
        t_val = self.times_seconds[t_idx]
        true_h = self.true_wl_matrix[t_idx].unsqueeze(1)
        true_u = self.true_ucx_matrix[t_idx].unsqueeze(1)
        true_v = self.true_ucy_matrix[t_idx].unsqueeze(1)
        
        norm_t_curr = self.get_normalized_t(t_val.unsqueeze(0))
        bc_curr = self.boundary_forcings[t_idx].unsqueeze(0)
        
        wl_curr, u_curr, v_curr = self.predict(norm_t_curr, self.norm_coords, bc_curr)
        
        data_loss = nn.MSELoss()(wl_curr[self.interior_mask], true_h[self.interior_mask])
        boundary_loss = nn.MSELoss()(wl_curr[self.boundary_mask], true_h[self.boundary_mask])
        
        # Velocity data loss (validated against D-Flow FM output)
        vel_loss = nn.MSELoss()(u_curr[self.interior_mask], true_u[self.interior_mask]) + \
                   nn.MSELoss()(v_curr[self.interior_mask], true_v[self.interior_mask])
        
        t_0 = self.times_seconds[0].unsqueeze(0)
        norm_t_0 = self.get_normalized_t(t_0)
        bc_0 = self.boundary_forcings[0].unsqueeze(0)
        wl_0, _, _ = self.predict(norm_t_0, self.norm_coords, bc_0)
        ic_loss = nn.MSELoss()(wl_0, self.true_wl_matrix[0].unsqueeze(1))
        
        if phys_weight > 0.0:
            pde_loss = self.compute_physics_loss(t_idx)
        else:
            pde_loss = torch.tensor(0.0, device=self.device)
        
        total_loss = 10.0 * data_loss + 30.0 * boundary_loss + 5.0 * vel_loss + 20.0 * ic_loss + phys_weight * pde_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.optimizer.step()
        
        return data_loss.item(), boundary_loss.item(), ic_loss.item(), pde_loss.item()
