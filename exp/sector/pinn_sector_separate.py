import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. 全域設定
# ==========================================
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 物理與幾何定義 (沿用之前的 Singular Source)
# ==========================================
class ExactSolution:
    """ 真解 u = (r-1)^2 * r^(2/3) * sin(2*theta/3) """
    def u_exact(self, xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        r = torch.sqrt(x**2 + y**2 + 1e-12)
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2*np.pi, theta)
        return (r - 1.0)**2 * r**(2.0/3.0) * torch.sin(2.0 * theta / 3.0)

    def exact_gradient(self, xy):
        u = self.u_exact(xy)
        grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=False)[0]
        return grads[:, 0:1], grads[:, 1:2]

    def q_particular(self, xy):
        """ 特解: 讓 div(q) = -f（curl 部分自動 divergence-free）"""
        x, y = xy[:, 0:1], xy[:, 1:2]
        r = torch.sqrt(x**2 + y**2 + 1e-12)
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2*np.pi, theta)
        coef_1, coef_2 = 2.5, -2.8
        q_r = torch.sin(2.0 * theta / 3.0) * (coef_1 * r**(5.0/3.0) + coef_2 * r**(2.0/3.0))
        return q_r * torch.cos(theta), q_r * torch.sin(theta)

solution = ExactSolution()

def sector_hard_bc_singular(xy):
    """ 遮罩: 確保 u = 0 on boundary (強制滿足邊界) """
    x, y = xy[:, 0:1], xy[:, 1:2]
    r = torch.sqrt(x**2 + y**2 + 1e-12)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)
    return (1-r) * (r**(2/3)) * theta * (1.5 * np.pi - theta)

def forcing_f_from_qpart(xy):
    """
    已知 div(q) = -f 且 q = q_part + curl(psi)
    div(curl)=0 => div(q)=div(q_part)
    => f = -div(q_particular)
    """
    if not xy.requires_grad:
        xy.requires_grad_(True)
    qpx, qpy = solution.q_particular(xy)

    dqpx = torch.autograd.grad(qpx, xy, torch.ones_like(qpx), create_graph=True)[0]
    dqpy = torch.autograd.grad(qpy, xy, torch.ones_like(qpy), create_graph=True)[0]
    div_qp = dqpx[:, 0:1] + dqpy[:, 1:2]
    f = -div_qp
    return f

# ==========================================
# 2. 模型
# ==========================================
class PrimalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xy):
        if not xy.requires_grad:
            xy.requires_grad_(True)
        return sector_hard_bc_singular(xy) * self.net(xy)

class DualPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 1)
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
        grads = torch.autograd.grad(psi, xy, torch.ones_like(psi), create_graph=True)[0]
        q_curl_x, q_curl_y = grads[:, 1:2], -grads[:, 0:1]
        q_part_x, q_part_y = solution.q_particular(xy)
        return (q_part_x + q_curl_x), (q_part_y + q_curl_y)

# ==========================================
# 3. 分開訓練：Primal 用 PDE-Laplacian，Dual 用 Complementary Energy
# ==========================================
def laplacian_u(u, xy):
    grads = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    du_dx, du_dy = grads[:, 0:1], grads[:, 1:2]

    d2u_dx2 = torch.autograd.grad(du_dx, xy, torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(du_dy, xy, torch.ones_like(du_dy), create_graph=True)[0][:, 1:2]
    return d2u_dx2 + d2u_dy2

def train_primal_pde(primal, xy_train, iters=10001, lr=1e-3):
    print("\n--- Primal Training (PDE Residual: || Δu + f ||^2 ) ---")
    optimizer = torch.optim.Adam(primal.parameters(), lr=lr)
    history = []

    for i in range(iters):
        optimizer.zero_grad()

        u = primal(xy_train)
        lap_u = laplacian_u(u, xy_train)
        f = forcing_f_from_qpart(xy_train)

        res = lap_u + f
        loss = torch.mean(res**2)

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i}, Primal PDE Loss: {loss.item():.2e}")
            history.append(loss.item())

    return primal, history

def train_dual_complementary_energy(dual, xy_train, iters=10001, lr=1e-3):
    print("\n--- Dual Training (Complementary Energy: min 0.5*||q||^2, div(q)=-f by construction) ---")
    optimizer = torch.optim.Adam(dual.parameters(), lr=lr)
    history = []

    for i in range(iters):
        optimizer.zero_grad()

        qx, qy = dual(xy_train)
        loss = 0.5 * torch.mean(qx**2 + qy**2)   # <-- 這裡改成 complementary energy

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i}, Dual CompEnergy Loss: {loss.item():.2e}")
            history.append(loss.item())

    return dual, history

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    # --- Data ---
    N_samples = 10000
    r, th = np.sqrt(np.random.rand(N_samples)), np.random.rand(N_samples) * 1.5 * np.pi
    xy_np = np.stack([r*np.cos(th), r*np.sin(th)], axis=1)
    xy_train = torch.tensor(xy_np, dtype=torch.float64, device=device, requires_grad=True)

    primal_model = PrimalPINN().to(device)
    dual_model   = DualPINN().to(device)

    # --- Separate Training ---
    start_time = time.time()
    primal_model, primal_hist = train_primal_pde(primal_model, xy_train, iters=10001, lr=1e-3)
    dual_model, dual_hist     = train_dual_complementary_energy(dual_model, xy_train, iters=10001, lr=1e-3)
    print(f"\nTraining Time (Total): {time.time()-start_time:.2f}s")

    # --- Verification ---
    res = 2000
    xv, yv = np.meshgrid(np.linspace(-1.1, 1.1, res), np.linspace(-1.1, 1.1, res))
    x_flat, y_flat = xv.flatten(), yv.flatten()
    r_flat = np.sqrt(x_flat**2 + y_flat**2)
    th_flat = np.arctan2(y_flat, x_flat)
    th_flat[th_flat < 0] += 2*np.pi
    mask = (r_flat <= 1.0) & (th_flat <= 1.5*np.pi + 0.01)
    valid_pts = np.stack([x_flat[mask], y_flat[mask]], axis=1)

    xy_eval = torch.tensor(valid_pts, dtype=torch.float64, device=device, requires_grad=True)

    # Primal grad(u)
    u = primal_model(xy_eval)
    gu = torch.autograd.grad(u, xy_eval, torch.ones_like(u))[0].detach().cpu().numpy()

    # Dual q
    qx, qy = dual_model(xy_eval)
    q = np.stack([qx.detach().cpu().numpy().flatten(),
                  qy.detach().cpu().numpy().flatten()], axis=1)

    # Exact grad(u)
    gt_qx, gt_qy = solution.exact_gradient(xy_eval)
    gt = np.stack([gt_qx.detach().cpu().numpy().flatten(),
                   gt_qy.detach().cpu().numpy().flatten()], axis=1)

    # Errors
    gap_sq    = np.mean(np.sum((q - gu)**2, axis=1))
    primal_sq = np.mean(np.sum((gu - gt)**2, axis=1))
    dual_sq   = np.mean(np.sum((q - gt)**2, axis=1))

    rel_verif = abs(gap_sq - (primal_sq + dual_sq)) 

    print("\n" + "="*40)
    print("SEPARATE TRAINING HYPERCIRCLE RESULT")
    print("-" * 40)
    print("Primal Loss Used: || Δu + f ||^2   (PDE residual)")
    print("Dual Loss Used  : 0.5*||q||^2      (Complementary energy, div(q)=-f by construction)")
    print("-" * 40)
    print(f"1. Final Gap^2      : {gap_sq:.2e}")
    print(f"2. Primal Error^2   : {primal_sq:.2e}")
    print(f"3. Dual Error^2     : {dual_sq:.2e}")
    print(f"4. Sum (2+3)        : {primal_sq + dual_sq:.2e}")
    print(f"Verification Err    : {rel_verif:.2e}")
    print("="*40)

    # --- Plot ---
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(valid_pts[:, 0], valid_pts[:, 1])

    x_tri = valid_pts[triang.triangles].mean(axis=1)[:, 0]
    y_tri = valid_pts[triang.triangles].mean(axis=1)[:, 1]
    mask_tri = (x_tri**2 + y_tri**2 > 1.01) | ((np.arctan2(y_tri, x_tri) % (2*np.pi)) > 1.5*np.pi + 0.1)
    triang.set_mask(mask_tri)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def plot_t(ax, data, title):
        c = ax.tricontourf(triang, data, levels=60, cmap='viridis')
        ax.set_title(title)
        ax.axis('equal')
        plt.colorbar(c, ax=ax)

    plot_t(axes[0], np.sqrt(np.sum(gu**2, axis=1)), "Primal |grad u|")
    plot_t(axes[1], np.sqrt(np.sum(q**2, axis=1)),  "Dual |q| (comp energy-min)")
    plot_t(axes[2], np.sqrt(np.sum((q-gu)**2, axis=1)), "Gap |q - grad u|")

    plt.savefig("separate_training_comp_energy.png", dpi=300, bbox_inches='tight')
    plt.show()
