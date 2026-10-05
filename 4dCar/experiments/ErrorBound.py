import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.error import ErrorDynamics


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results", "JointTraining", "checkpoint.pt")
ERROR_LABELS = ("e_x", "e_y", "e_theta", "e_v")
POSITION_INDICES = (0, 1)
POSITION_LABELS = ("e_x", "e_y")


def load_ellipsoid_matrix(checkpoint_path: str = CHECKPOINT_PATH) -> torch.Tensor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "E" not in checkpoint:
        raise KeyError("Checkpoint does not contain ellipsoid matrix 'E'")

    error_model = ErrorDynamics()
    expected_dim = error_model.error_dim
    E = torch.as_tensor(checkpoint["E"], dtype=torch.float64)
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError(f"Expected square E matrix, got shape {tuple(E.shape)}")
    if E.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Expected E shape ({expected_dim}, {expected_dim}) for the current Error Model, "
            f"got {tuple(E.shape)}. Retrain JointTraining with the current models."
        )
    model_signature = checkpoint.get("model_signature")
    if model_signature is None:
        raise ValueError(
            "Checkpoint does not identify its model and error-coordinate definition. "
            "Retrain JointTraining with the current ErrorDynamics."
        )
    if model_signature != error_model.model_signature():
        raise ValueError(
            "Checkpoint model signature does not match the current Error Model: "
            f"{model_signature}"
        )
    if not torch.allclose(E, E.T, atol=1e-9, rtol=0.0):
        raise ValueError("Expected E to be symmetric")

    eigvals = torch.linalg.eigvalsh(E)
    if not torch.all(eigvals > 0.0):
        raise ValueError(f"Expected E to be positive definite, got eigenvalues {eigvals}")
    return E


def axis_aligned_error_bounds(E: torch.Tensor) -> torch.Tensor:
    """
    Return maximum absolute values of every error coordinate over
    {e: e^T E e <= 1}.
    """
    E = torch.as_tensor(E)
    covariance_shape = torch.linalg.inv(E)
    return torch.sqrt(torch.diagonal(covariance_shape))


def position_error_bounds(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project {e: e^T E e <= 1} onto the world-frame position errors [e_x, e_y].

    Returns:
        bounds: Axis-aligned maximum absolute errors [Delta_x, Delta_y].
        position_shape: Shape matrix S_p of the projected ellipsoid
            {e_p: e_p^T S_p^{-1} e_p <= 1}.
    """
    E = torch.as_tensor(E)
    position_selector = torch.zeros(
        len(POSITION_INDICES),
        E.shape[0],
        dtype=E.dtype,
        device=E.device,
    )
    for row, index in enumerate(POSITION_INDICES):
        position_selector[row, index] = 1.0

    position_shape = position_selector @ torch.linalg.solve(E, position_selector.T)
    bounds = torch.sqrt(torch.diagonal(position_shape))
    return bounds, position_shape


def main() -> None:
    E = load_ellipsoid_matrix()
    error_bounds = axis_aligned_error_bounds(E)
    bounds, position_shape = position_error_bounds(E)
    position_ellipsoid_matrix = torch.linalg.inv(position_shape)

    print(f"loaded E from: {CHECKPOINT_PATH}")
    print("axis-aligned error bounds from e^T E e <= 1:")
    for label, bound in zip(ERROR_LABELS, error_bounds):
        print(f"  |{label}| <= {bound.item():.6f}")

    print("\naxis-aligned world-frame position error bounds:")
    for label, bound in zip(POSITION_LABELS, bounds):
        print(f"  |{label}| <= {bound.item():.6f}")

    print("\nprojected position shape matrix S_p = P E^{-1} P^T:")
    print(position_shape)
    print("\nprojected position ellipsoid matrix S_p^{-1}:")
    print(position_ellipsoid_matrix)


if __name__ == "__main__":
    main()
