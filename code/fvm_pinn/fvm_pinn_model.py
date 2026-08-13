import torch
import torch.nn as nn
import numpy as np
from numerical_model import GPUHydrodynamicModel

# --- Lag Configuration ---
# H_ocean:    current + 13 hourly lags = 14 features (covers one full M2 tidal cycle ~12.4h)
# Q_garonne:  current + 3 hourly lags = 4 features
# Q_dordogne: current + 3 hourly lags = 4 features
N_H_LAGS = 14
N_QG_LAGS = 4
N_QD_LAGS = 4
N_BC_FEATURES = N_H_LAGS + N_QG_LAGS + N_QD_LAGS  # 22
N_SPATIAL = 3   # x_norm, y_norm, z_norm
LAG_STEP = 60   # 1 hour = 60 one-minute steps
MIN_T_IDX = (N_H_LAGS - 1) * LAG_STEP  # 780 steps = 13 hours of history

HIDDEN_DIM = 256
N_GNN_LAYERS = 6


def _mlp(in_dim, hidden_dim, out_dim, n_hidden=2):
    """Build a small MLP with SiLU activations."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GNNLayer(nn.Module):
    """
    Single message-passing layer using the FVM mesh edges.
    Uses scatter_add for aggregation — no torch_geometric required.
    
    Messages flow bidirectionally: c_L → c_R and c_R → c_L.
    Edge features carry the face geometry (nx, ny, e_len_norm).
    """
    def __init__(self, hidden_dim, edge_feat_dim):
        super().__init__()
        # Message function: takes source+target hidden states + edge features
        self.message_net = _mlp(2 * hidden_dim + edge_feat_dim, hidden_dim, hidden_dim)
        # Update function: takes node hidden state + aggregated messages
        self.update_net = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, src, dst, edge_feat):
        """
        Args:
            h:         [N_nodes, hidden_dim] node embeddings
            src:       [N_edges] source cell indices (c_L)
            dst:       [N_edges] destination cell indices (c_R)
            edge_feat: [N_edges, edge_feat_dim] face geometry features
        """
        N = h.size(0)

        # --- Forward direction: src → dst ---
        msg_fwd = self.message_net(torch.cat([h[src], h[dst], edge_feat], dim=-1))
        # --- Reverse direction: dst → src ---
        msg_rev = self.message_net(torch.cat([h[dst], h[src], edge_feat], dim=-1))

        # Aggregate at destination nodes
        agg = torch.zeros(N, msg_fwd.size(-1), device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg_fwd), msg_fwd)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(msg_rev), msg_rev)

        # Update with residual connection + LayerNorm
        h_new = self.update_net(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class HydroGNN(nn.Module):
    """
    Physics-Informed Graph Neural Network Hydrodynamic Surrogate.

    Architecture:
      - Node Encoder:  MLP([x_norm, y_norm, z_norm, lagged_BCs(22)]) → hidden_dim
      - Edge Encoder:  MLP([nx, ny, e_len_norm]) → hidden_dim
      - GNN Layers:    6 × bidirectional message passing with residuals
      - Decoder:       MLP(hidden_dim) → [η, u, v] per node

    The GNN explicitly propagates tidal wave information along the FVM mesh
    connectivity, learning physical phase lags from the ocean boundary inland.
    NO absolute time input → generalizes to any time period.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, n_layers=N_GNN_LAYERS):
        super().__init__()
        node_in = N_SPATIAL + N_BC_FEATURES  # 25
        edge_in = 3  # [nx, ny, e_len_norm]

        self.node_encoder = _mlp(node_in, hidden_dim, hidden_dim)
        self.edge_encoder = _mlp(edge_in, hidden_dim // 2, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.decoder = _mlp(hidden_dim, hidden_dim // 2, 3)

    def forward(self, node_feat, src, dst, edge_feat):
        """
        Args:
            node_feat: [N_nodes, 25]
            src:       [N_edges] c_L indices
            dst:       [N_edges] c_R indices
            edge_feat: [N_edges, 3]
        Returns:
            eta: [N_nodes, 1], u: [N_nodes, 1], v: [N_nodes, 1]
        """
        h = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)

        for layer in self.gnn_layers:
            h = layer(h, src, dst, e)

        out = self.decoder(h)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]


class FVMPINNTrainer:
    def __init__(self, fvm_engine, cell_coords_m, true_wl_matrix, times_seconds,
                 boundary_mask, bc_matrix_norm):
        """
        Args:
            fvm_engine:      GPUHydrodynamicModel (has c_L, c_R, nx, ny, e_len, cell_areas)
            bc_matrix_norm:  [T, 3] normalized BCs (H_ocean, Q_garonne, Q_dordogne)
        """
        self.fvm = fvm_engine
        self.device = fvm_engine.device

        self.boundary_mask = boundary_mask.clone().detach().to(dtype=torch.bool, device=self.device)
        self.interior_mask = ~self.boundary_mask

        # ---- Normalize spatial coordinates ----
        coords_t = cell_coords_m.clone().detach().to(dtype=torch.float32, device=self.device)
        self.coords_mean = coords_t.mean(dim=0)
        self.coords_std = coords_t.std(dim=0) + 1e-6
        self.norm_coords = (coords_t - self.coords_mean) / self.coords_std  # [N, 2]

        # ---- Normalize bathymetry ----
        cell_z_flat = self.fvm.cell_z.squeeze(1)
        self.cell_z_flat = cell_z_flat
        self.z_mean = cell_z_flat.mean()
        self.z_std = cell_z_flat.std() + 1e-6
        self.norm_z = ((cell_z_flat - self.z_mean) / self.z_std).unsqueeze(1)  # [N, 1]

        # ---- Store data tensors (will be overwritten in main after split) ----
        self.true_wl_matrix = torch.tensor(true_wl_matrix, dtype=torch.float32, device=self.device)
        self.times_seconds = torch.tensor(times_seconds, dtype=torch.float32, device=self.device)
        self.bc_matrix_norm = torch.tensor(bc_matrix_norm, dtype=torch.float32, device=self.device)

        # ---- Physics constants ----
        self.g = 9.81
        self.manning_n = 0.019

        # ---- Pre-build graph edge data ----
        self._build_graph()
        self._precompute_bed_slopes()

        # ---- Build Model ----
        # NOTE: nn.DataParallel is NOT compatible with GNNs because DataParallel
        # splits inputs on dim-0. This causes src/dst edge indices to go out of
        # bounds on each GPU's partial node tensor. GNNs must run on one GPU.
        self.gnn = HydroGNN(hidden_dim=HIDDEN_DIM, n_layers=N_GNN_LAYERS).to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"Note: {torch.cuda.device_count()} GPUs available. Running GNN on primary GPU (DataParallel incompatible with graph indexing).")

        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)

    def _build_graph(self):
        """Pre-compute edge indices and normalized edge features for message passing."""
        src = self.fvm.c_L.long()  # [N_edges]
        dst = self.fvm.c_R.long()  # [N_edges]
        self.src = src
        self.dst = dst

        # Edge features: [nx, ny, normalized edge length]
        nx = self.fvm.nx.squeeze(1)   # [N_edges]
        ny = self.fvm.ny.squeeze(1)   # [N_edges]
        e_len = self.fvm.e_len.squeeze(1)  # [N_edges]
        e_len_norm = (e_len - e_len.mean()) / (e_len.std() + 1e-6)

        self.edge_feat = torch.stack([nx, ny, e_len_norm], dim=1)  # [N_edges, 3]
        print(f"Graph built: {self.src.size(0)} edges, {self.norm_coords.size(0)} nodes.")

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

    def _build_node_features(self, lagged_bc):
        """
        Build the full [N_nodes, 25] node feature matrix by concatenating
        spatial features with the broadcast boundary condition vector.
        """
        N = self.norm_coords.size(0)
        # [x_norm, y_norm, z_norm] for all nodes
        spatial = torch.cat([self.norm_coords, self.norm_z], dim=1)  # [N, 3]
        # Broadcast lagged BC to all nodes
        bc_exp = lagged_bc.unsqueeze(0).expand(N, -1)  # [N, 22]
        return torch.cat([spatial, bc_exp], dim=1)  # [N, 25]

    def _forward_all(self, lagged_bc):
        """Run GNN forward pass for all nodes given a BC state."""
        node_feat = self._build_node_features(lagged_bc)
        return self.gnn(node_feat, self.src, self.dst, self.edge_feat)

    # ----- Physics Loss (FVM-based divergence) -----
    def compute_physics_loss(self, eta_t, u_t, v_t, eta_tm1):
        """
        SWE mass conservation residual using FVM-style flux divergence.
        Time derivative: finite difference between t and t-1.
        Spatial divergence: computed via scatter_add along mesh faces (same as FVM).
        This is identical to how the actual FVM solver computes mass conservation.
        """
        dt = 60.0  # seconds
        deta_dt = (eta_t - eta_tm1) / dt  # [N, 1]

        h = torch.clamp(eta_t - self.cell_z_flat.unsqueeze(1), min=0.01)  # water depth
        hu = h * u_t   # [N, 1]
        hv = h * v_t   # [N, 1]

        # Compute face fluxes (average of adjacent cells)
        hu_face = 0.5 * (hu[self.src] + hu[self.dst])  # [E, 1]
        hv_face = 0.5 * (hv[self.src] + hv[self.dst])

        nx = self.fvm.nx  # [E, 1]
        ny = self.fvm.ny
        e_len = self.fvm.e_len

        # Normal flux × edge length
        flux = (hu_face * nx + hv_face * ny) * e_len  # [E, 1]

        # Scatter fluxes to cells
        # In FVM convention: the flux leaving c_L is positive, entering c_R is positive
        # so c_L loses flux and c_R gains flux
        N = eta_t.size(0)
        div = torch.zeros(N, 1, device=self.device)
        div.scatter_add_(0, self.src.unsqueeze(1), -flux)   # c_L loses
        div.scatter_add_(0, self.dst.unsqueeze(1),  flux)   # c_R gains
        div = div / self.fvm.cell_areas  # [N, 1] → m/s

        # Continuity residual: dη/dt + div(h·U) = 0
        # Note: negative sign because div(hU) = -dη/dt
        R_mass = deta_dt - div

        # Depth-averaged momentum residual (linearised, quasi-steady)
        # Uses GNN-based surface gradient: Green-Gauss on predicted η
        eta_face = 0.5 * (eta_t[self.src] + eta_t[self.dst])  # [E, 1]
        d_eta_dx_flux = eta_face * nx * e_len
        d_eta_dy_flux = eta_face * ny * e_len

        grad_eta_x = torch.zeros(N, 1, device=self.device)
        grad_eta_y = torch.zeros(N, 1, device=self.device)
        grad_eta_x.scatter_add_(0, self.src.unsqueeze(1), d_eta_dx_flux)
        grad_eta_x.scatter_add_(0, self.dst.unsqueeze(1), -d_eta_dx_flux)
        grad_eta_y.scatter_add_(0, self.src.unsqueeze(1), d_eta_dy_flux)
        grad_eta_y.scatter_add_(0, self.dst.unsqueeze(1), -d_eta_dy_flux)
        grad_eta_x = grad_eta_x / self.fvm.cell_areas
        grad_eta_y = grad_eta_y / self.fvm.cell_areas

        # Manning friction
        vel_mag = torch.sqrt(u_t**2 + v_t**2 + 1e-8)
        C_f = self.g * (self.manning_n ** 2) / (h ** (1.0/3.0) + 1e-8)

        # x-momentum: g * dη/dx + C_f * u * |U| / h = 0
        R_mom_x = self.g * grad_eta_x + C_f * u_t * vel_mag / (h + 1e-8)
        # y-momentum: g * dη/dy + C_f * v * |U| / h = 0
        R_mom_y = self.g * grad_eta_y + C_f * v_t * vel_mag / (h + 1e-8)

        return (torch.mean(R_mass**2) +
                0.1 * torch.mean(R_mom_x**2) +
                0.1 * torch.mean(R_mom_y**2))

    # ----- Training Step -----
    def train_step(self, t_idx, phys_weight=2.0):
        self.optimizer.zero_grad()

        bc_curr = self.get_lagged_bc(t_idx)
        eta_t, u_t, v_t = self._forward_all(bc_curr)

        true_h = self.true_wl_matrix[t_idx].unsqueeze(1)   # [N, 1]
        true_u = self.true_ucx_matrix[t_idx].unsqueeze(1)
        true_v = self.true_ucy_matrix[t_idx].unsqueeze(1)

        # Data fidelity (interior + boundary separately weighted)
        data_loss = nn.MSELoss()(eta_t[self.interior_mask], true_h[self.interior_mask])
        bc_loss = nn.MSELoss()(eta_t[self.boundary_mask], true_h[self.boundary_mask])
        vel_loss = (nn.MSELoss()(u_t[self.interior_mask], true_u[self.interior_mask]) +
                    nn.MSELoss()(v_t[self.interior_mask], true_v[self.interior_mask]))

        # IC loss at t=0
        bc_0 = self.get_lagged_bc(0)
        eta_0, _, _ = self._forward_all(bc_0)
        ic_loss = nn.MSELoss()(eta_0, self.true_wl_matrix[0].unsqueeze(1))

        # Physics loss
        if phys_weight > 0.0 and t_idx > 0:
            with torch.no_grad():
                bc_prev = self.get_lagged_bc(t_idx - 1)
                eta_tm1, _, _ = self._forward_all(bc_prev)
            pde_loss = self.compute_physics_loss(eta_t, u_t, v_t, eta_tm1.detach())
        else:
            pde_loss = torch.tensor(0.0, device=self.device)

        total_loss = (10.0 * data_loss +
                      30.0 * bc_loss +
                      5.0  * vel_loss +
                      20.0 * ic_loss +
                      phys_weight * pde_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 1.0)
        self.optimizer.step()

        return data_loss.item(), bc_loss.item(), ic_loss.item(), pde_loss.item()

    # ----- Inference (for evaluation/validation) -----
    def predict(self, norm_coords=None, lagged_bc=None, norm_z=None):
        """
        Alias kept for compatibility with the evaluation code in train_fvm_pinn.py.
        Runs the GNN forward pass for all mesh nodes.
        """
        with torch.no_grad():
            return self._forward_all(lagged_bc)
