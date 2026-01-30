# Thesis Code: Strict Error Control in PINNs via the Hypercircle Method

This repository contains the training scripts used in my thesis.

## Requirements
- Python 3.x

## Project Structure

exp/
disk/
pinn_disk_hypo.py
pinn_disk_separate.py
sector/
pinn_sector_hypo_singularity.py
pinn_sector_hypo.py
pinn_sector_separate.py

bash
Copy code

## Environment Setup (venv)

Create and activate a Python virtual environment, then install dependencies.

### macOS / Linux
```bash
python -m venv venv_hypotenuse_loss
source venv_hypotenuse_loss/bin/activate
pip install -r requirements.txt
Windows (PowerShell)
powershell
Copy code
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
How to Run
All scripts are located under exp/.

Disk
Run from the repository root:

bash
Copy code
python exp/disk/pinn_disk_hypo.py
python exp/disk/pinn_disk_separate.py
Sector
Run from the repository root:

bash
Copy code
python exp/sector/pinn_sector_hypo.py
python exp/sector/pinn_sector_hypo_singularity.py
python exp/sector/pinn_sector_separate.py
Figure Outputs
Some scripts save figures using matplotlib.pyplot.savefig(...).

For example, if a script contains:

python
Copy code
plt.savefig("hypercircle as loss.png", dpi=300, bbox_inches="tight")
then the output image file (hypercircle as loss.png) will be saved to the current working directory (the folder shown by pwd in your terminal when you run the script).

Examples:

If you run the script from the repository root (recommended), e.g.

bash
Copy code
python exp/sector/pinn_sector_hypo.py
then the figure will be saved in the repository root (same level as exp/).

If you change directory first, e.g.

bash
Copy code
cd exp/sector
python pinn_sector_hypo.py
then the figure will be saved in exp/sector/.

Notes
Experiments were originally run on a server using a Python virtual environment (venv).

For exact dependency versions, see requirements.txt.
