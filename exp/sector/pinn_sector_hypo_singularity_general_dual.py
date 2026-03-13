import os
from pathlib import Path
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
        """ 特解: 確保 div(q) = -f (強制滿足物理) """
        x, y = xy[:, 0:1], xy[:, 1:2]
        r = torch.sqrt(x**2 + y**2 + 1e-12)
        theta = torch.atan2(y, x)
        theta = torch.where(theta < 0, theta + 2*np.pi, theta)
        coef_1, coef_2 = 2.5, -2.8
        q_r = torch.sin(2.0 * theta / 3.0) * (coef_1 * r**(5.0/3.0) + coef_2 * r**(2.0/3.0))
        return q_r * torch.cos(theta), q_r * torch.sin(theta)

solution = ExactSolution()

def forcing_f(xy):
    """
    Analytical forcing: f = -sin(2θ/3) * (2r^(2/3) + 14/3 * (r-1) * r^(-1/3))
    Derived from -Δu where u = (r-1)^2 * r^(2/3) * sin(2θ/3).
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    r = torch.sqrt(x**2 + y**2 + 1e-12)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)
    return -torch.sin(2.0 * theta / 3.0) * (2.0 * r**(2.0/3.0) + 14.0/3.0 * (r - 1.0) * r**(-1.0/3.0))

def sector_hard_bc_singular(xy):
    """ 遮罩: 確保 u = 0 on boundary (強制滿足邊界) """
    x, y = xy[:, 0:1], xy[:, 1:2]
    r = torch.sqrt(x**2 + y**2 + 1e-12)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2*np.pi, theta)
    return (1-r) * r**(2.0/3.0) * theta * (1.5 * np.pi - theta)

# ==========================================
# 2. 模型
# ==========================================
class PrimalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 80), nn.Tanh(), nn.Linear(80, 80), nn.Tanh(), nn.Linear(80, 1))
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, xy):
        if not xy.requires_grad: xy.requires_grad_(True)
        return sector_hard_bc_singular(xy) * self.net(xy)

class DualPINNGeneral(nn.Module):
    """
    General dual network: directly outputs q = (q_x, q_y) without
    curl construction. The divergence constraint div(q) + f = 0 is
    NOT built in analytically; instead it is enforced via a penalty
    term in the loss function.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 80), nn.Tanh(), nn.Linear(80, 80), nn.Tanh(), nn.Linear(80, 2))
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, xy):
        if not xy.requires_grad: xy.requires_grad_(True)
        q = self.net(xy)
        return q[:, 0:1], q[:, 1:2]

# ==========================================
# 3. 聯合訓練函數 (Joint Training)
# ==========================================
def train_jointly(primal, dual, xy_train):
    print("\n--- Joint Training (General Dual: Gap + Div Penalty) ---")

    # 將兩個模型的參數合併到一個優化器中
    params = list(primal.parameters()) + list(dual.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    # 訓練記錄
    history = []

    for i in range(20001):
        optimizer.zero_grad()

        # 1. Primal Forward -> 算出位移梯度 grad(u)
        u = primal(xy_train)
        grads_u = torch.autograd.grad(u, xy_train, torch.ones_like(u), create_graph=True)[0]
        grad_u_x, grad_u_y = grads_u[:, 0:1], grads_u[:, 1:2]

        # 2. Dual Forward -> 算出通量 q
        q_x, q_y = dual(xy_train)

        # 3. Divergence of q: div(q) = dq_x/dx + dq_y/dy
        dqx = torch.autograd.grad(
            q_x, xy_train, torch.ones_like(q_x), create_graph=True, retain_graph=True
        )[0]
        dqy = torch.autograd.grad(
            q_y, xy_train, torch.ones_like(q_y), create_graph=True
        )[0]
        div_q = dqx[:, 0:1] + dqy[:, 1:2]

        # 4. Compute forcing f
        f = forcing_f(xy_train)

        # 5. Loss = || grad(u) - q ||^2 + || div(q) + f ||^2
        gap_loss = torch.mean((grad_u_x - q_x)**2 + (grad_u_y - q_y)**2)
        div_loss = torch.mean((div_q + f)**2)
        loss = gap_loss + div_loss

        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i:5d} | Total Loss: {loss.item():.2e} "
                  f"| Gap: {gap_loss.item():.2e} | Div: {div_loss.item():.2e}")
            history.append(loss.item())

    return primal, dual

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    # --- Data ---
    N_samples = 10000

    # 生成數據 (包含角落密集採樣)
    N_unif = int(N_samples)
    r, th = np.sqrt(np.random.rand(N_unif)), np.random.rand(N_unif)*1.5*np.pi
    xy_np = np.stack([r*np.cos(th), r*np.sin(th)], axis=1)
    xy_train = torch.tensor(xy_np, dtype=torch.float64, device=device, requires_grad=True)

    # --- Joint Training ---
    primal_model = PrimalPINN().to(device)
    dual_model = DualPINNGeneral().to(device)

    start_time = time.time()
    primal_model, dual_model = train_jointly(primal_model, dual_model, xy_train)
    print(f"Training Time: {time.time()-start_time:.2f}s")

    # --- Verification ---
    # 建立評估網格
    res = 2000
    xv, yv = np.meshgrid(np.linspace(-1.1, 1.1, res), np.linspace(-1.1, 1.1, res))
    x_flat, y_flat = xv.flatten(), yv.flatten()
    r_flat = np.sqrt(x_flat**2 + y_flat**2)
    th_flat = np.arctan2(y_flat, x_flat)
    th_flat[th_flat < 0] += 2*np.pi
    mask = (r_flat <= 1.0) & (th_flat <= 1.5*np.pi+0.01)
    valid_pts = np.stack([x_flat[mask], y_flat[mask]], axis=1)
    xy_eval = torch.tensor(valid_pts, dtype=torch.float64, device=device, requires_grad=True)

    # 取得結果
    u = primal_model(xy_eval)
    gu = torch.autograd.grad(u, xy_eval, torch.ones_like(u), retain_graph=True)[0].detach().cpu().numpy()
    qx_t, qy_t = dual_model(xy_eval)
    q = np.stack([qx_t.detach().cpu().numpy().flatten(), qy_t.detach().cpu().numpy().flatten()], axis=1)
    gt_qx, gt_qy = solution.exact_gradient(xy_eval)
    gt = np.stack([gt_qx.detach().cpu().numpy().flatten(), gt_qy.detach().cpu().numpy().flatten()], axis=1)

    # Evaluate divergence residual on eval grid
    dqx_eval = torch.autograd.grad(
        qx_t, xy_eval, torch.ones_like(qx_t), retain_graph=True
    )[0]
    dqy_eval = torch.autograd.grad(
        qy_t, xy_eval, torch.ones_like(qy_t)
    )[0]
    div_q_eval = (dqx_eval[:, 0:1] + dqy_eval[:, 1:2]).detach().cpu().numpy().flatten()
    f_eval = forcing_f(xy_eval).detach().cpu().numpy().flatten()
    div_residual = np.mean((div_q_eval + f_eval)**2)

    # 計算各項誤差
    gap_sq = np.mean(np.sum((q - gu)**2, axis=1))
    primal_sq = np.mean(np.sum((gu - gt)**2, axis=1))
    dual_sq = np.mean(np.sum((q - gt)**2, axis=1))

    print("\n" + "="*50)
    print("GENERAL DUAL JOINT TRAINING RESULT")
    print("-" * 50)
    print(f"Loss Used: || grad(u) - q ||^2 + || div(q) + f ||^2")
    print(f"1. Final Gap^2      : {gap_sq:.2e}")
    print(f"2. Primal Error^2   : {primal_sq:.2e}")
    print(f"3. Dual Error^2     : {dual_sq:.2e}")
    print(f"4. Sum (2+3)        : {primal_sq + dual_sq:.2e}")
    print(f"5. Div Residual     : {div_residual:.2e}")
    print(f"Verification Err    : {abs(gap_sq - (primal_sq+dual_sq)):.2e}")
    print("="*50)

    # --- Save Results ---
    def SaveResult(folder, xy, u_vals, qx_vals, qy_vals):
        """Save evaluation coordinates and solution fields to .npy files."""
        out_dir = Path(__file__).resolve().parent / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "xy.npy", xy)
        np.save(out_dir / "u.npy", u_vals)
        np.save(out_dir / "qx.npy", qx_vals)
        np.save(out_dir / "qy.npy", qy_vals)
        print(f"Results saved to: {out_dir}")

    u_np = u.detach().cpu().numpy().flatten()
    SaveResult("result-singular", valid_pts, u_np, q[:, 0], q[:, 1])

    # 繪圖
    import matplotlib.tri as mtri
    triang = mtri.Triangulation(valid_pts[:,0], valid_pts[:,1])
    # 簡單過濾三角形
    x_tri, y_tri = valid_pts[triang.triangles].mean(axis=1)[:,0], valid_pts[triang.triangles].mean(axis=1)[:,1]
    mask_tri = (x_tri**2 + y_tri**2 > 1.01) | ((np.arctan2(y_tri, x_tri)%(2*np.pi)) > 1.5*np.pi+0.1)
    triang.set_mask(mask_tri)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    def plot_t(ax, data, title):
        c = ax.tricontourf(triang, data, levels=60, cmap='viridis')
        ax.set_title(title); ax.axis('equal'); plt.colorbar(c, ax=ax)

    plot_t(axes[0], np.sqrt(np.sum(gu**2, axis=1)), "Primal |grad u|")
    plot_t(axes[1], np.sqrt(np.sum(q**2, axis=1)), "Dual |q|")
    plot_t(axes[2], np.sqrt(np.sum((q-gu)**2, axis=1)), "Gap |q - grad u|")
    plot_t(axes[3], np.abs(div_q_eval + f_eval), "Div Residual |div(q) + f|")

    plt.savefig("hypo_general_dual.png", dpi=300, bbox_inches='tight')
