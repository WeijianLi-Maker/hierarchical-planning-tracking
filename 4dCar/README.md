# Certified Hierarchical Control

Offline synthesis and evaluation of hierarchical planning and tracking control
for a four-dimensional vehicle.

## Overview

- **Planning model**: a two-dimensional single integrator controlled by MPC.
- **Tracking model**: a four-dimensional vehicle controlled by a learned tracker.
- **World-frame tracking error**:
  `e = x - pi(z) = [x-z_x, y-z_y, theta, speed]`, with `phi(x,z) = I`.
- **Certificate models**: a neural Lyapunov function and the ellipsoid
  `{e : e^T E e <= 1}`.
- **Counterexample search**: bounded PyTorch search for Lyapunov, ellipsoid, and
  control-bound violations.

Changing the model or error-coordinate definition invalidates existing expert
datasets and checkpoints. Regenerate the data and retrain after such changes.

## Layout

| Package | Role |
|---------|------|
| `models/` | Tracking, planning, and world-frame error dynamics |
| `controllers/` | Neural tracker, tracking MPC expert, planning MPC |
| `lyapunov/` | Neural Lyapunov function |
| `learning/` | Losses, ellipsoid model, and training loop |
| `SMT/` | Counterexample search and replay buffer |
| `experiments/` | Data generation, joint training, and error-bound reporting |
| `tests/` | Model, controller, training, and certificate checks |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Generate world-frame expert data
python experiments/DataGeneration.py

# Train the Lyapunov function, tracking controller, and ellipsoid
python experiments/JointTraining.py

# Evaluate the learned closed loop and report the error bound
python tests/TrainingTest.py
python experiments/ErrorBound.py
```

Results and checkpoints are written under `results/`; expert data is written
under `data/`.
