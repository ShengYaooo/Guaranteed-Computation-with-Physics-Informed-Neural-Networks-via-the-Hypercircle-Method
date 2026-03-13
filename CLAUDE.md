# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis project implementing **strict error control in Physics-Informed Neural Networks (PINNs) via the Hypercircle Method**. It solves Poisson-type PDEs (-Δu = f) on two domains (disk, sector) using two approaches:

- **Hypercircle (Joint Training):** Simultaneously trains primal (u) and dual (flux q) networks, minimizing the gap ||∇u − q||². Provides certified error bounds via the identity: Gap² = Primal Error² + Dual Error².
- **Separate Training:** Independently trains primal (minimize PDE residual) and dual (minimize complementary energy) models for comparison.

## Running Experiments

All scripts should be run from the **repository root**:

```bash
# Setup
python -m venv venv_hypotenuse_loss
source venv_hypotenuse_loss/bin/activate
pip install -r requirements.txt

# Disk domain
python exp/disk/pinn_disk_hypo.py
python exp/disk/pinn_disk_separate.py

# Sector domain (with singularity at origin)
python exp/sector/pinn_sector_hypo.py
python exp/sector/pinn_sector_hypo_singularity.py
python exp/sector/pinn_sector_separate.py
```

Each script is self-contained—no shared modules. Output PNG figures are saved to the current working directory.

## Architecture

**Primal PINN:** Models displacement u(x,y). Hard boundary conditions enforced by multiplying NN output with a distance-to-boundary mask (e.g., `(1 - x² - y²)` for disk). Gradients computed via autograd.

**Dual PINN:** Models flux q(x,y) via a potential ψ. Uses curl construction `q₀ = (∂ψ/∂y, −∂ψ/∂x)` to guarantee divergence-free component, then adds a particular solution q_p satisfying div(q_p) = −f. This analytically enforces the constraint div(q) = −f.

**Key parameters across scripts:**
- float64 precision throughout
- Adam optimizer, lr=0.001
- Disk: 2→50→50→1 networks, 8000 samples, 5000 epochs
- Sector: 2→80→80→1 networks, 10000 samples, 10001 epochs
- Tanh activations

## Domain-Specific Notes

- **Disk:** Exact solution u = (1−r²)/4. Simplest test case.
- **Sector:** Singular solution u = (r−1)² r^(2/3) sin(2θ/3). The singularity at the origin requires larger networks and careful BC masking. The `_singularity` variant uses enhanced corner sampling and high-resolution (2000²) triangulated evaluation.
- **Sector scripts contain Traditional Chinese comments** explaining algorithmic choices.
