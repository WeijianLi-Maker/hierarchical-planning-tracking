# Certified Hierarchical Control for a Dynamic Bicycle

This project implements hierarchical planning and tracking control for a
dynamic bicycle model.

## Models

- Planning model: three-state Dubins vehicle
  - `z = [X_hat, Y_hat, psi_hat]`
  - `v = [speed_hat, omega_hat]`
- Tracking model: six-state dynamic bicycle
  - `x = [X, Y, psi, v_x, v_y, omega]`
  - `u = [delta_f, a_x]`
- Tracking error:
  - `e = diag(R(psi_hat)^-1, I_4) (x - pi(z))`
  - `pi(z) = [X_hat, Y_hat, psi_hat, 0, 0, 0]`
- Certificate error:
  - `r = e - [0, 0, 0, speed_hat, 0, omega_hat]`
  - Lyapunov and ellipsoid conditions are evaluated around this moving reference.
  - Lyapunov decrease is checked locally on the candidate set `0 < V(r) <= 1`
    using `V_dot + lambda V <= 0`.
  - The current certificate covers the dynamic Bicycle high-speed planning
    domain `speed_hat in [4, 10] m/s`; it does not claim low-speed validity.

The dynamic bicycle tire model is undefined at `v_x = 0`. Data generation,
training, and SMT search therefore operate in a positive-speed domain.

## Components

| Package | Role |
|---------|------|
| `models/` | Dubins planning, dynamic bicycle tracking, and error dynamics |
| `controllers/` | Planning MPC, tracking MPC, and neural tracking controller |
| `lyapunov/` | Six-dimensional neural Lyapunov function |
| `learning/` | Joint loss functions, optimizer loop, and learnable ellipsoid |
| `SMT/` | PyTorch-based counterexample search and replay buffer |
| `experiments/` | Data generation, joint training, and error-bound evaluation |
| `tests/` | Model, controller, learning, MPC, and SMT tests |

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

Generate a Bicycle-format expert dataset:

```bash
python experiments/DataGeneration.py
```

Data generation uses closed-loop expert rollouts. Regenerate the dataset after
changing the dynamics, tracking MPC, or certificate coordinates; joint training
rejects older independently sampled datasets.

Train the neural controller, Lyapunov function, and error-bound ellipsoid:

```bash
python experiments/JointTraining.py
```

Joint training performs final counterexample-guided retraining rounds. If
counterexamples remain, it saves a diagnostic checkpoint and exits with an
error instead of marking the result as verified.

The current default configuration is a behavior-cloning-only baseline:
`L_V = L_E = 0`, SMT/CEGIS disabled, and only the neural controller is
optimized. Its checkpoint is saved under `results/BehaviorCloning`.

Evaluate the learned planar position-error bound:

```bash
python experiments/ErrorBound.py
```

Run the automated tests:

```bash
python tests/ModelTest.py
python tests/LyapunovNetworkTest.py
python tests/LossFunctionTest.py
python tests/TrackingControllerTest.py
python tests/TrainingTest.py
python tests/SMTSolverTest.py
python tests/MPCPlanningTest.py
python tests/MPCTrackingTest.py
```

`SMTSolver` currently performs counterexample search rather than a formal SMT
proof. A checkpoint records the final search result and must not be described
as formally certified.

Existing datasets or checkpoints created by the earlier ten-dimensional model
are incompatible and must be regenerated.
