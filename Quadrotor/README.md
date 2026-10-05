# Certified Hierarchical Control

Offline synthesis and online implementation of certified hierarchical planning and control for robotic systems.

## Overview

This codebase implements the framework from *Certified Hierarchical Planning and Control*:

- **Planning model** (low-fidelity): generates trajectories via MPC.
- **Tracking model** (high-fidelity): tracked with a learned controller \(\kappa(e, z, v)\).
- **Neural Lyapunov function** \(V(e)\) and ellipsoid bound \(\mathcal{E} = \{e : e^\top E e \le 1\}\).
- **SMT-based certification** (Z3 / dReal) to verify Lyapunov conditions beyond sampling.

## Layout

| Package | Role |
|---------|------|
| `models/` | Tracking, planning, and error dynamics |
| `controllers/` | NN tracker, QP-OSC expert, MPC planner |
| `lyapunov/` | Neural \(V\), constraints, ellipsoid \(E\) |
| `learning/` | Training loop, losses, BC, SMT certification |
| `experiments/` | Affine MVP, manipulator, five-link benchmarks |
| `utils/` | Logging, plotting, reproducibility |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Offline synthesis (train + certify)
python main.py train --experiment affine_mvp

# Online MPC + learned tracker
python main.py run --experiment manipulator --checkpoint results/checkpoint.pt
```

## Experiments

- `experiments/affine_mvp/` — low-dimensional affine sanity check
- `experiments/manipulator/` — manipulator tracking/planning stack
- `experiments/five_link/` — five-link planar arm

Results and checkpoints are written under `results/`; raw datasets under `data/`.
