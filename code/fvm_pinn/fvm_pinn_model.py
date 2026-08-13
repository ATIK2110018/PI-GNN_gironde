import torch
import torch.nn as nn
import numpy as np
from numerical_model import GPUHydrodynamicModel

# --- Lag Configuration ---
# H_ocean:    current + 13 hourly lags = 14 features (captures one full 12.4h M2 tidal cycle)
# Q_garonne:  current + 3 hourly lags = 4 features (river discharge changes slowly)
# Q_dordogne: current + 3 hourly lags = 4 features
N_H_LAGS = 14
N_QG_LAGS = 4
N_QD_LAGS = 4
N_BC_FEATURES = N_H_LAGS + N_QG_LAGS + N_QD_LAGS  # 22
N_SPATIAL = 3   # x, y, z
N_INPUTS = N_SPATIAL + N_BC_FEATURES  # 25
LAG_STEP = 60   # 60 time steps = 1 hour (data at 1-min intervals)
MIN_T_IDX = (N_H_LAGS - 1) * LAG_STEP  # 780 = need 13 hours of history


class SpatialFourierFeatures(nn.Module):
    """Random Fourier Features for spatial coordinates only."""
    def __init__(self, in_features=3, out_features=128, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * sigma, requires_grad=False)

    def forward(self, x):
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class HydroPINN(nn.Module):
    """
    Time-Independent Parametric Surrogate Model (DeepONet Architecture).
    """
    def __init__(self):
        super(HydroPINN, self).__init__()
        # 1. Spatial Trunk (processes x, y, z)
        self.spatial_enc = SpatialFourierFeatures(in_features=N_SPATIAL, out_features=128, sigma=1.0)
        self.trunk_net = nn.Sequential(
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU()
        )
        
        # 2. Parametric Branch (processes Boundary Conditions)
        self.branch_net = nn.Sequential(
            nn.Linear(N_BC_FEATURES, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU()
        )
        
        # 3. Decoder (combines trunk and branch)
        self.decoder = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 3)
        )

    def forward(self, inputs):
        # inputs: [N, N_SPATIAL + N_BC_FEATURES]
        coords = inputs[:, :N_SPATIAL]
        bcs = inputs[:, N_SPATIAL:]
        
        spatial_features = self.spatial_enc(coords)
        trunk_out = self.trunk_net(spatial_features)
        
        branch_out = self.branch_net(bcs)
        
        combined = torch.cat([trunk_out, branch_out], dim=-1)
        out = self.decoder(combined)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


class FVMPINNTrainer:
    def __init__(self, fvm_engine, cell_coords_m, true_wl_matrix, times_seconds,
                 boundary_mask, bc_matrix_norm):
        """
        Args:
            bc_matrix_norm: [T, 3] normalized BCs (H_ocean, Q_garonne, Q_dordogne)
        """
        self.fvm = fvm_engine
        self.device = fvm_engine.device

        self.boundary_mask = boundary_mask.clone().detach().to(dtype=torch.bool, device=self.device)
        self.interior_mask = ~self.boundary_mask

        # Normalize spatial coordinates
        coords_t = cell_coords_m.clone().detach().to(dtype=torch.float32, device=self.device)
        self.coords_mean = coords_t.mean(dim=0)
        self.coords_std = coords_t.std(dim=0)
        self.norm_coords = (coords_t - self.coords_mean) / self.coords_std

        # Normalize bathymetry
        cell_z_flat = self.fvm.cell_z.squeeze(1)
        self.cell_z_flat = cell_z_flat
        self.z_mean = cell_z_flat.mean()
        self.z_std = cell_z_flat.std() + 1e-6
        self.norm_z = ((cell_z_flat - self.z_mean) / self.z_std).unsqueeze(1)  # [N,1]

        # Store data tensors
        self.true_wl_matrix = torch.tensor(true_wl_matrix, dtype=torch.float32, device=self.device)
        self.times_seconds = torch.tensor(times_seconds, dtype=torch.float32, device=self.device)
        self.bc_matrix_norm = torch.tensor(bc_matrix_norm, dtype=torch.float32, device=self.device)

        # Chain-rule scale factors (spatial only — no time normalization needed)
        self.dx_scale = 1.0 / self.coords_std[0]
        self.dy_scale = 1.0 / self.coords_std[1]

        # Physics constants
        self.g = 9.81
        self.manning_n = 0.019

        self._precompute_bed_slopes()

        self.pinn = HydroPINN().to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel!")
            self.pinn = nn.DataParallel(self.pinn)

        self.optimizer = torch.optim.Adam(self.pinn.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def _precompute_bed_slopes(self):
        """Green-Gauss gradient reconstruction for dz/dx, dz/dy."""
        z = self.fvm.cell_z
        dz_dx = torch.zeros_like(z)
        dz_dy = torch.zeros_like(z)
        z_face = 0.5 * (z[self.fvm.c_L] + z[self.fvm.c_R])
        flux_x = z_face * self.fvm.nx * self.fvm.e_len
        flux_y = z_face * self.fvm.ny * self.fvm.e_len
        dz_dx.scatter_add_(0, self.fvm.c_L.unsqueeze(1), flux_x)
        dz_dx.scatter_add_(0, self.fvm.c_R.unsqueeze(1), -flux_x)
        dz_dy.scatter_add_(0, self.fvm.c_L.unsqueeze(1), flux_y)
        dz_dy.scatter_add_(0, self.fvm.c_R.unsqueeze(1), -flux_y)
        dz_dx = dz_dx / self.fvm.cell_areas
        dz_dy = dz_dy / self.fvm.cell_areas
        # Clamp extreme slopes from isolated boundary cells
        self.bed_dz_dx = dz_dx.squeeze(1).clamp(-0.3, 0.3).detach()
        self.bed_dz_dy = dz_dy.squeeze(1).clamp(-0.3, 0.3).detach()
        print(f"Precomputed bed slopes: dz/dx range [{self.bed_dz_dx.min():.6f}, {self.bed_dz_dx.max():.6f}]")

    # ----- Lagged BC Construction -----
    def get_lagged_bc(self, t_idx):
        """
        Build the 22-element lagged boundary condition vector for time index t_idx.
        Returns tensor of shape [22] on self.device.
        """
        h_lags = []
        for k in range(N_H_LAGS):
            idx = max(0, t_idx - k * LAG_STEP)
            h_lags.append(self.bc_matrix_norm[idx, 0])
        qg_lags = []
        for k in range(N_QG_LAGS):
            idx = max(0, t_idx - k * LAG_STEP)
            qg_lags.append(self.bc_matrix_norm[idx, 1])
        qd_lags = []
        for k in range(N_QD_LAGS):
            idx = max(0, t_idx - k * LAG_STEP)
            qd_lags.append(self.bc_matrix_norm[idx, 2])
        return torch.stack(h_lags + qg_lags + qd_lags)

    # ----- Prediction -----
    def predict(self, norm_coords, lagged_bc, norm_z=None):
        """Forward pass: no time input, just spatial + lagged BCs."""
        N = norm_coords.size(0)
        bc_expanded = lagged_bc.unsqueeze(0).expand(N, -1)
        if norm_z is None:
            norm_z = self.norm_z
        z_exp = norm_z.expand(N, 1) if norm_z.dim() == 1 else norm_z[:N]
        inputs = torch.cat([norm_coords, z_exp, bc_expanded], dim=1)  # [N, 18]
        return self.pinn(inputs)

    # ----- Physics Loss (SWE with finite-difference deta/dt) -----
    def compute_physics_loss(self, t_idx):
        """
        SWE residuals using autograd for spatial derivatives and
        finite-difference for the time derivative deta/dt.
        """
        n_colloc = min(4000, self.norm_coords.size(0))
        idx = torch.randperm(self.norm_coords.size(0), device=self.device)[:n_colloc]

        bc_curr = self.get_lagged_bc(t_idx)
        bc_prev = self.get_lagged_bc(max(0, t_idx - 1))

        # Current prediction (with grad for spatial derivatives)
        xy_input = self.norm_coords[idx].clone().requires_grad_(True)
        z_input = self.norm_z[idx]
        bc_input = bc_curr.unsqueeze(0).expand(n_colloc, -1)
        inputs = torch.cat([xy_input, z_input, bc_input], dim=1)
        eta, u, v = self.pinn(inputs)

        # Previous prediction (detached, for finite-difference deta/dt)
        with torch.no_grad():
            bc_input_prev = bc_prev.unsqueeze(0).expand(n_colloc, -1)
            inputs_prev = torch.cat([self.norm_coords[idx], self.norm_z[idx], bc_input_prev], dim=1)
            eta_prev, _, _ = self.pinn(inputs_prev)

        # Time derivative via finite difference (dt = 60 seconds)
        deta_dt = (eta - eta_prev) / 60.0

        # Spatial derivatives via autograd
        ones = torch.ones_like(eta)
        deta_dxy = torch.autograd.grad(eta, xy_input, ones, create_graph=True, retain_graph=True)[0]
        du_dxy = torch.autograd.grad(u, xy_input, ones, create_graph=True, retain_graph=True)[0]
        dv_dxy = torch.autograd.grad(v, xy_input, ones, create_graph=True, retain_graph=True)[0]

        deta_dx = deta_dxy[:, 0:1] * self.dx_scale
        deta_dy = deta_dxy[:, 1:2] * self.dy_scale
        du_dx = du_dxy[:, 0:1] * self.dx_scale
        du_dy = du_dxy[:, 1:2] * self.dy_scale
        dv_dx = dv_dxy[:, 0:1] * self.dx_scale
        dv_dy = dv_dxy[:, 1:2] * self.dy_scale

        dz_dx = self.bed_dz_dx[idx].unsqueeze(1)
        dz_dy = self.bed_dz_dy[idx].unsqueeze(1)
        dh_dx = deta_dx - dz_dx
        dh_dy = deta_dy - dz_dy

        z_phys = self.cell_z_flat[idx].unsqueeze(1)
        h = torch.clamp(eta - z_phys, min=0.01)

        vel_mag = torch.sqrt(u**2 + v**2 + 1e-8)
        C_f = self.g * (self.manning_n ** 2) / (h ** (1.0/3.0) + 1e-8)

        # Mass: deta/dt + d(hu)/dx + d(hv)/dy = 0
        R_mass = deta_dt + u * dh_dx + h * du_dx + v * dh_dy + h * dv_dy
        # x-Momentum (quasi-steady): u*du/dx + v*du/dy + g*deta/dx + friction = 0
        R_mom_x = u * du_dx + v * du_dy + self.g * deta_dx + C_f * u * vel_mag / (h + 1e-8)
        # y-Momentum (quasi-steady)
        R_mom_y = u * dv_dx + v * dv_dy + self.g * deta_dy + C_f * v * vel_mag / (h + 1e-8)

        return torch.mean(R_mass**2) + torch.mean(R_mom_x**2) + torch.mean(R_mom_y**2)

    # ----- Training Step -----
    def train_step(self, t_idx, phys_weight=2.0):
        self.optimizer.zero_grad()

        true_h = self.true_wl_matrix[t_idx].unsqueeze(1)
        true_u = self.true_ucx_matrix[t_idx].unsqueeze(1)
        true_v = self.true_ucy_matrix[t_idx].unsqueeze(1)

        bc_lagged = self.get_lagged_bc(t_idx)
        wl_curr, u_curr, v_curr = self.predict(self.norm_coords, bc_lagged)

        data_loss = nn.MSELoss()(wl_curr[self.interior_mask], true_h[self.interior_mask])
        boundary_loss = nn.MSELoss()(wl_curr[self.boundary_mask], true_h[self.boundary_mask])
        vel_loss = nn.MSELoss()(u_curr[self.interior_mask], true_u[self.interior_mask]) + \
                   nn.MSELoss()(v_curr[self.interior_mask], true_v[self.interior_mask])

        # IC loss at t=0 (use lagged BCs at index 0)
        bc_0 = self.get_lagged_bc(0)
        wl_0, _, _ = self.predict(self.norm_coords, bc_0)
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
