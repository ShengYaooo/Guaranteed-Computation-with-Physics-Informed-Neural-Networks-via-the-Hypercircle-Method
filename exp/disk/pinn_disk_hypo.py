import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. Global Settings
# ==========================================
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Physics & Geometry (Unit Disk, -Δu = 1)
# ==========================================
class ExactSolution:
    """ 
    Exact solution for -Δu = 1 with u=0 on boundary r=1:
    u = (1 - r^2) / 4
    grad(u) = (-x/2, -y/2)
    """
    def u_exact(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return 0.25 * (1.0 - (x**2 + y**2))

    def exact_gradient(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return -0.5 * x, -0.5 * y

    def q_particular(self, xy):
        """ Particular solution satisfying div(q_p) = -f = -1 """
        x, y = xy[:, 0:1], xy[:, 1:2]
        return -0.5 * x, -0.5 * y

solution = ExactSolution()

def disk_hard_bc(xy):
    """ Distance function to enforce u=0 on boundary r=1 """
    x, y = xy[:, 0:1], xy[:, 1:2]
    return 1.0 - (x**2 + y**2)

# ==========================================
# 2. Model Definitions
# ==========================================
class PrimalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xy):
        if not xy.requires_grad:
            xy.requires_grad_(True)
        # Enforce BCs: u = Distance * NN(x,y)
        return disk_hard_bc(xy) * self.net(xy)

class DualPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xy):
        if not xy.requires_grad:
            xy.requires_grad_(True)
        psi = self.net(xy)
        # Divergence-free field via curl: q0 = (d_psi/dy, -d_psi/dx)
        grads = torch.autograd.grad(
            psi, xy, torch.ones_like(psi), create_graph=True
        )[0]
        q_curl_x, q_curl_y = grads[:, 1:2], -grads[:, 0:1]

        # Total flux q = q_particular + q_divergence_free
        q_part_x, q_part_y = solution.q_particular(xy)
        return (q_part_x + q_curl_x), (q_part_y + q_curl_y)

# ==========================================
# 3. Training Function
# ==========================================
def train_jointly(primal, dual, xy_train, epochs=5000):
    params = list(primal.parameters()) + list(dual.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    for i in range(epochs + 1):
        optimizer.zero_grad()

        # Primal gradient: grad(u)
        u = primal(xy_train)
        grads_u = torch.autograd.grad(
            u, xy_train, torch.ones_like(u), create_graph=True
        )[0]
        grad_u_x, grad_u_y = grads_u[:, 0:1], grads_u[:, 1:2]

        # Dual flux: q
        q_x, q_y = dual(xy_train)

        # Hypercircle Loss: || grad(u) - q ||^2
        loss = torch.mean((grad_u_x - q_x)**2 + (grad_u_y - q_y)**2)

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i:5d} | Gap Loss: {loss.item():.2e}")

# ==========================================
# 4. Evaluation and Visualization
# ==========================================
if __name__ == "__main__":
    # 1. Generate Training Data (Uniform sampling in disk)
    N_samples = 8000
    r_samp = torch.sqrt(torch.rand(N_samples, 1, device=device))
    theta_samp = 2 * np.pi * torch.rand(N_samples, 1, device=device)
    x_samp = r_samp * torch.cos(theta_samp)
    y_samp = r_samp * torch.sin(theta_samp)
    xy_train = torch.cat([x_samp, y_samp], dim=1).requires_grad_(True)

    # 2. Training
    primal_model = PrimalPINN().to(device)
    dual_model = DualPINN().to(device)

    print(f"--- Training on {device} ---")
    start_time = time.perf_counter()
    train_jointly(primal_model, dual_model, xy_train)
    end_time = time.perf_counter()

    # 3. Evaluation on high-res grid
    grid_res = 120
    _x = np.linspace(-1, 1, grid_res)
    _y = np.linspace(-1, 1, grid_res)
    xx, yy = np.meshgrid(_x, _y)
    mask = (xx**2 + yy**2) <= 1.0

    xy_eval = torch.tensor(
        np.stack([xx[mask], yy[mask]], axis=1),
        device=device,
        requires_grad=True
    )

    u_pred = primal_model(xy_eval)

    gu_vec = torch.autograd.grad(
        u_pred, xy_eval, torch.ones_like(u_pred), create_graph=False
    )[0].detach().cpu().numpy()
    gu_x, gu_y = gu_vec[:, 0], gu_vec[:, 1]

    qx_t, qy_t = dual_model(xy_eval)
    qx, qy = qx_t.detach().cpu().numpy().flatten(), qy_t.detach().cpu().numpy().flatten()

    gt_qx_t, gt_qy_t = solution.exact_gradient(xy_eval)
    gt_qx, gt_qy = gt_qx_t.detach().cpu().numpy().flatten(), gt_qy_t.detach().cpu().numpy().flatten()

    # Calculate L2 errors squared
    gap_sq = np.mean((gu_x - qx)**2 + (gu_y - qy)**2)
    p_err_sq = np.mean((gu_x - gt_qx)**2 + (gu_y - gt_qy)**2)
    d_err_sq = np.mean((qx - gt_qx)**2 + (qy - gt_qy)**2)

    # 4. Final Report
    print("\n" + "="*50)
    print(f"{'HYPERCIRCLE VERIFICATION REPORT':^48}")
    print("-" * 50)
    print(f"Training Time:      {end_time - start_time:.2f} s")
    print(f"Energy Gap (Gap^2): {gap_sq:.2e}")
    print(f"Primal Error (P^2): {p_err_sq:.2e}")
    print(f"Dual Error (D^2):   {d_err_sq:.2e}")
    print(f"Sum (P^2 + D^2):    {p_err_sq + d_err_sq:.2e}")
    identity_check = abs(gap_sq - (p_err_sq + d_err_sq))
    print(f"Identity Check     {identity_check:.2e}")
    print("="*50)

    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def fill(vals):
        out = np.full_like(xx, np.nan, dtype=np.float64)
        out[mask] = vals
        return out

    # Predicted Potential u
    c1 = axes[0].pcolormesh(
        xx, yy,
        fill(u_pred.detach().cpu().numpy().flatten()),
        shading="auto",
        cmap="viridis"
    )
    axes[0].set_title("Predicted u (Primal)")
    plt.colorbar(c1, ax=axes[0])

    # Local Gap |grad u - q|
    gap_val = np.sqrt((gu_x - qx)**2 + (gu_y - qy)**2)
    c2 = axes[1].pcolormesh(
        xx, yy,
        fill(gap_val),
        shading="auto",
        cmap="magma"
    )
    axes[1].set_title("Local Gap Indicator")
    plt.colorbar(c2, ax=axes[1])

    # Error |u_pred - u_true|
    u_true = solution.u_exact(xy_eval).detach().cpu().numpy().flatten()
    u_err = np.abs(u_pred.detach().cpu().numpy().flatten() - u_true)
    c3 = axes[2].pcolormesh(
        xx, yy,
        fill(u_err),
        shading="auto",
        cmap="inferno"
    )
    axes[2].set_title("Absolute Error |u_pred - u_true|")
    plt.colorbar(c3, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()

    # ==========================================
    # 6. Save figure in the same folder as this script
    # ==========================================
    try:
        out_dir = Path(__file__).resolve().parent  # folder of this .py file
    except NameError:
        out_dir = Path(os.getcwd())                # fallback (e.g., notebook / interactive)

    out_path = out_dir / "HypoLoss.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")

    plt.show()
