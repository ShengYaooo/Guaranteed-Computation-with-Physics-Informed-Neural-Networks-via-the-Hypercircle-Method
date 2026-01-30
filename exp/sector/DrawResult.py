"""
DrawResult.py — Load saved results from a specified folder and draw:
  1. 3D surface plot of u
  2. Color map of u
  3. Color map of q_x
  4. Color map of q_y
  5. Vector field of (q_x, q_y)

Usage:
    python DrawResult.py                  # defaults to "result" folder
    python DrawResult.py path/to/folder   # specify folder
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def load_data(folder):
    folder = Path(folder)
    xy = np.load(folder / "xy.npy")
    u = np.load(folder / "u.npy")
    qx = np.load(folder / "qx.npy")
    qy = np.load(folder / "qy.npy")
    return xy, u, qx, qy


def make_triangulation(xy):
    """Create a triangulation with sector-domain masking."""
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1])
    # Filter triangles outside the sector domain
    x_tri = xy[triang.triangles].mean(axis=1)[:, 0]
    y_tri = xy[triang.triangles].mean(axis=1)[:, 1]
    mask_tri = (x_tri**2 + y_tri**2 > 1.01) | (
        (np.arctan2(y_tri, x_tri) % (2 * np.pi)) > 1.5 * np.pi + 0.1
    )
    triang.set_mask(mask_tri)
    return triang


def downsample(xy, *fields, max_points=50000):
    """Subsample data if it exceeds max_points to keep plotting fast."""
    n = xy.shape[0]
    if n <= max_points:
        return (xy, *fields)
    idx = np.random.default_rng(0).choice(n, max_points, replace=False)
    idx.sort()
    return (xy[idx], *(f[idx] for f in fields))


def draw(folder):
    print("Loading data from folder:", folder)
    xy, u, qx, qy = load_data(folder)

    print(f"Loaded {xy.shape[0]} points, downsampling for plotting...")
    xy, u, qx, qy = downsample(xy, u, qx, qy)
    print(f"Using {xy.shape[0]} points for plots.")

    triang = make_triangulation(xy)

    fig = plt.figure(figsize=(30, 12))

    # --- 1. 3D surface of u ---
    # Interpolate onto a regular grid for proper occlusion handling
    print("Drawing 3D surface...")
    interp = mtri.LinearTriInterpolator(triang, u)
    grid_res_3d = 300
    xi = np.linspace(xy[:, 0].min(), xy[:, 0].max(), grid_res_3d)
    yi = np.linspace(xy[:, 1].min(), xy[:, 1].max(), grid_res_3d)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interp(Xi, Yi)  # masked array: NaN outside domain

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot_surface(Xi, Yi, Zi, cmap="viridis", edgecolor="none",
                     antialiased=True, alpha=1.0, rstride=1, cstride=1)
    ax1.set_facecolor("white")
    ax1.xaxis.pane.fill = True
    ax1.yaxis.pane.fill = True
    ax1.zaxis.pane.fill = True
    ax1.xaxis.pane.set_facecolor("white")
    ax1.yaxis.pane.set_facecolor("white")
    ax1.zaxis.pane.set_facecolor("white")
    ax1.view_init(elev=np.degrees(np.arctan2(1, np.sqrt(2))),
                  azim=np.degrees(np.arctan2(-1, 1)))
    ax1.set_title("u (3D Surface)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")

    # --- 2. Color map of u ---
    print("Drawing color map of u...")
    ax2 = fig.add_subplot(2, 3, 2)
    c2 = ax2.tricontourf(triang, u, levels=60, cmap="viridis")
    ax2.set_title("u (Color Map)")
    ax2.axis("equal")
    plt.colorbar(c2, ax=ax2)

    # --- 3. Color map of q_x ---
    print("Drawing color map of q_x...")
    ax3 = fig.add_subplot(2, 3, 3)
    c3 = ax3.tricontourf(triang, qx, levels=60, cmap="viridis")
    ax3.set_title("$q_x$ (Color Map)")
    ax3.axis("equal")
    plt.colorbar(c3, ax=ax3)

    # --- 4. Color map of q_y ---
    print("Drawing color map of q_y...")
    ax4 = fig.add_subplot(2, 3, 4)
    c4 = ax4.tricontourf(triang, qy, levels=60, cmap="viridis")
    ax4.set_title("$q_y$ (Color Map)")
    ax4.axis("equal")
    plt.colorbar(c4, ax=ax4)

    # --- 5. Vector field of (q_x, q_y) ---
    print("Drawing vector field...")
    ax5 = fig.add_subplot(2, 3, 5)
    # Subsample further for readable arrows
    n_arrows = min(1500, xy.shape[0])
    idx_q = np.random.default_rng(1).choice(xy.shape[0], n_arrows, replace=False)
    mag = np.sqrt(qx[idx_q]**2 + qy[idx_q]**2)
    ax5.quiver(xy[idx_q, 0], xy[idx_q, 1], qx[idx_q], qy[idx_q],
               mag, cmap="viridis", scale=None, width=0.002)
    ax5.set_title("Vector Field $(q_x, q_y)$")
    ax5.axis("equal")

    plt.tight_layout()

    out_path = Path(folder) / "DrawResult.png"
    print(f"Saving figure to: {out_path}")  
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")
    # plt.show()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "result-singular")
    draw(folder)
