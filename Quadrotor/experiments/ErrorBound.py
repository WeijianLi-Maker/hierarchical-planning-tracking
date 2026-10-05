import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results", "JointTraining", "checkpoint.pt")
POSITION_INDICES = (0, 4, 8)
POSITION_LABELS = ("e_x", "e_y", "e_z")


def load_ellipsoid_matrix(checkpoint_path: str = CHECKPOINT_PATH) -> torch.Tensor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "E" not in checkpoint:
        raise KeyError("Checkpoint does not contain ellipsoid matrix 'E'")

    E = torch.as_tensor(checkpoint["E"], dtype=torch.float64)
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError(f"Expected square E matrix, got shape {tuple(E.shape)}")
    if E.shape[0] <= max(POSITION_INDICES):
        raise ValueError(f"E shape {tuple(E.shape)} does not contain position error indices")
    if not torch.allclose(E, E.T, atol=1e-9, rtol=0.0):
        raise ValueError("Expected E to be symmetric")

    eigvals = torch.linalg.eigvalsh(E)
    if not torch.all(eigvals > 0.0):
        raise ValueError(f"Expected E to be positive definite, got eigenvalues {eigvals}")
    return E


def position_error_bounds(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project {e: e^T E e <= 1} onto the position errors [e_x, e_y, e_z].

    Returns:
        bounds: Axis-aligned maximum absolute errors [Delta_x, Delta_y, Delta_z].
        position_shape: Shape matrix S_p of the projected ellipsoid
            {e_p: e_p^T S_p^{-1} e_p <= 1}.
    """
    E = torch.as_tensor(E)
    position_selector = torch.zeros(3, E.shape[0], dtype=E.dtype, device=E.device)
    position_selector[0, POSITION_INDICES[0]] = 1.0
    position_selector[1, POSITION_INDICES[1]] = 1.0
    position_selector[2, POSITION_INDICES[2]] = 1.0

    position_shape = position_selector @ torch.linalg.solve(E, position_selector.T)
    bounds = torch.sqrt(torch.diagonal(position_shape))
    return bounds, position_shape


def main() -> None:
    E = load_ellipsoid_matrix()
    bounds, _ = position_error_bounds(E)

    print(f"loaded E from: {CHECKPOINT_PATH}")
    print("axis-aligned position error bounds from e^T E e <= 1:")
    for label, bound in zip(POSITION_LABELS, bounds):
        print(f"  |{label}| <= {bound.item():.6f}")


if __name__ == "__main__":
    main()
