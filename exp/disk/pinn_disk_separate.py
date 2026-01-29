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
# 1. Physics & Geometry (Unit Disk, f=1)
# ==========================================
class ExactSolution:
    """
    For -Δu = 1 in Unit Disk with u=0 on boundary:
    Exact u = (1 - r^2) / 4
    Exact Grad(u) = (-x/2, -y/2)
    """
    def u_exact(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return 0.25 * (1.0 - (x**2 + y**2))

    def exact_gradient(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return -0.5 * x, -0.5 * y

    def q_particular(self, xy):
        """ Particular solution satisfying div(q) = -f = -1 """
        x, y = xy[:, 0:1], xy[:, 1:2]
        return -0.5 * x, -0.5 * y

solution = ExactSolution()

def disk_hard_bc(xy):
    """ Distance function ensuring u=0 at r=1 """
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

    def forward(self, xy):
        if not xy.requires_grad:
            xy.requires_grad_(True)
        # u = Distance * NeuralNet
        return disk_hard_bc(xy) * self.net(xy)

class DualPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, xy):
        if not xy.requires_grad:
            xy.requires_grad_(True)
        psi = self.net(xy)
        grads = torch.autograd.grad(psi, xy, torch.ones_like(psi), create_graph=True)[0]
        q_curl_x, q_curl_y = grads[:, 1:2], -grads[:, 0:1]
        qp_x, qp_y = solution.q_particular(xy)
        return (qp_x + q_curl_x), (qp_y + q_curl_y)

# ==========================================
# 3. Separate Training Function
# ==========================================
def train_separately(primal, dual, xy_train, epochs=5000):
    print(f"--- Starting Separate Training on {device} ---")
    opt_p = torch.optim.Adam(primal.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(dual.parameters(), lr=1e-3)

    # --- Train Primal (PDE Residual) ---
    print("Training Primal Model...")
    start_p = time.time()
    for i in range(epochs + 1):
        opt_p.zero_grad()
        u = primal(xy_train)

        grads = torch.autograd.grad(u, xy_train, torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = grads[:, 0:1], grads[:, 1:2]
        u_xx = torch.autograd.grad(u_x, xy_train, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, xy_train, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        # Loss: PDE Residual -Δu = 1  ->  -(u_xx + u_yy) - 1 = 0
        loss_p = torch.mean((-(u_xx + u_yy) - 1.0) ** 2)
        loss_p.backward()
        opt_p.step()

        if i % 2000 == 0:
            print(f"  [Primal] Iter {i:5d} | Residual Loss: {loss_p.item():.2e}")
    time_p = time.time() - start_p

    # --- Train Dual (Energy Minimization) ---
    print("Training Dual Model...")
    start_d = time.time()
    for i in range(epochs + 1):
        opt_d.zero_grad()
        qx, qy = dual(xy_train)

        # Complementary energy: min 0.5 * ||q||^2  (constraint div(q)=-1 already enforced)
        loss_d = 0.5 * torch.mean(qx ** 2 + qy ** 2)
        loss_d.backward()
        opt_d.step()

        if i % 2000 == 0:
            print(f"  [Dual]   Iter {i:5d} | Energy Loss:   {loss_d.item():.2e}")
    time_d = time.time() - start_d

    return time_p, time_d

# ==========================================
# 4. Evaluation & Results
# ==========================================
if __name__ == "__main__":
    # Sampling (uniform in disk)
    N_samples = 8000
    r = torch.sqrt(torch.rand(N_samples, 1, device=device))
    theta = 2 * np.pi * torch.rand(N_samples, 1, device=device)
    xy_train = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1).requires_grad_(True)

    p_model, d_model = PrimalPINN().to(device), DualPINN().to(device)

    # Train
    tp, td = train_separately(p_model, d_model, xy_train)

    # Validation Grid
    res = 120
    _x = np.linspace(-1, 1, res)
    _y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(_x, _y)
    mask = (xx**2 + yy**2 <= 1.0)

    xy_eval = torch.tensor(np.stack([xx[mask], yy[mask]], axis=1),
                           device=device, requires_grad=True)

    # Predict
    u_pred = p_model(xy_eval)
    gu = torch.autograd.grad(u_pred, xy_eval, torch.ones_like(u_pred), create_graph=False)[0].detach().cpu().numpy()

    qx_p, qy_p = d_model(xy_eval)
    qx, qy = qx_p.detach().cpu().numpy().flatten(), qy_p.detach().cpu().numpy().flatten()

    # Exact gradients
    gt_qx_raw, gt_qy_raw = solution.exact_gradient(xy_eval)
    gt_qx, gt_qy = gt_qx_raw.detach().cpu().numpy().flatten(), gt_qy_raw.detach().cpu().numpy().flatten()

    # Metrics
    gap_sq = np.mean((gu[:, 0] - qx) ** 2 + (gu[:, 1] - qy) ** 2)
    p_err_sq = np.mean((gu[:, 0] - gt_qx) ** 2 + (gu[:, 1] - gt_qy) ** 2)
    d_err_sq = np.mean((qx - gt_qx) ** 2 + (qy - gt_qy) ** 2)
    identity_diff = abs(gap_sq - (p_err_sq + d_err_sq))

    print("\n" + "=" * 55)
    print(f"{'SEPARATE TRAINING PERFORMANCE REPORT':^55}")
    print("-" * 55)
    print(f"Primal Training Time (PDE):  {tp:10.2f} s")
    print(f"Dual Training Time (Energy): {td:10.2f} s")
    print(f"Total Computation Time:      {tp + td:10.2f} s")
    print("-" * 55)
    print(f"1. Gap Square (Measured):     {gap_sq:.2e}")
    print(f"2. Primal Error Square:       {p_err_sq:.2e}")
    print(f"3. Dual Error Square:         {d_err_sq:.2e}")
    print(f"4. Error Sum (2 + 3):         {p_err_sq + d_err_sq:.2e}")
    print("-" * 55)
    print(f"Identity Check     {identity_diff:.2e}")
    print("=" * 55)

    # Plotting (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def fill_disk(data_flat):
        full = np.full_like(xx, np.nan, dtype=np.float64)
        full[mask] = data_flat
        return full

    # 1) Predicted u
    u_pred_np = u_pred.detach().cpu().numpy().flatten()
    c1 = axes[0].pcolormesh(xx, yy, fill_disk(u_pred_np), cmap="viridis", shading="auto")
    axes[0].set_title("Predicted Displacement u")
    plt.colorbar(c1, ax=axes[0])

    # 2) Local gap |grad u - q|
    gap_val = np.sqrt((gu[:, 0] - qx) ** 2 + (gu[:, 1] - qy) ** 2)
    c2 = axes[1].pcolormesh(xx, yy, fill_disk(gap_val), cmap="magma", shading="auto")
    axes[1].set_title("Local Gap |∇u - q|")
    plt.colorbar(c2, ax=axes[1])

    # 3) Absolute error |u - u_exact|
    u_true = solution.u_exact(xy_eval).detach().cpu().numpy().flatten()
    u_err = np.abs(u_pred_np - u_true)
    c3 = axes[2].pcolormesh(xx, yy, fill_disk(u_err), cmap="inferno", shading="auto")
    axes[2].set_title("Absolute Error |u - u_exact|")
    plt.colorbar(c3, ax=axes[2])

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    # Save figure to the same folder
    try:
        out_dir = Path(__file__).resolve().parent
    except NameError:
        out_dir = Path(os.getcwd())

    out_path = out_dir / "Separate.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")

    plt.show()
