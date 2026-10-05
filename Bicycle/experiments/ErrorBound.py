import os
import sys
import warnings

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results", "JointTraining", "checkpoint.pt")
ERROR_DIM = 6
POSITION_INDICES = (0, 1)
POSITION_LABELS = ("e_X", "e_Y")


def load_ellipsoid_matrix(
    checkpoint_path: str = CHECKPOINT_PATH,
    require_verified: bool = False,
) -> tuple[torch.Tensor, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Run experiments\\JointTraining.py first: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("certificate_coordinates") != "moving_reference_local_exponential_v2":
        raise ValueError(
            "Checkpoint does not use the moving-reference certificate coordinates. "
            "Retrain before reporting an error-bound candidate."
        )
    final_verification = checkpoint.get("final_verification")
    if final_verification is None:
        raise ValueError(
            "Checkpoint predates final counterexample-search validation. Retrain before "
            "reporting an error-bound candidate."
        )
    if not final_verification.get("success", False):
        message = (
            "Final counterexample search found violations. The reported values are "
            "diagnostic ellipsoid projections, not verified error bounds."
        )
        if require_verified:
            raise ValueError(message)
        warnings.warn(message, UserWarning)
    if "E" not in checkpoint:
        raise KeyError("Checkpoint does not contain ellipsoid matrix 'E'")

    E = torch.as_tensor(checkpoint["E"], dtype=torch.float64)
    if E.shape != (ERROR_DIM, ERROR_DIM):
        raise ValueError(
            f"Expected Bicycle ellipsoid matrix shape ({ERROR_DIM}, {ERROR_DIM}), "
            f"got {tuple(E.shape)}. Run the aligned JointTraining.py first."
        )
    if not torch.allclose(E, E.T, atol=1e-9, rtol=0.0):
        raise ValueError("Expected E to be symmetric")

    eigvals = torch.linalg.eigvalsh(E)
    if not torch.all(eigvals > 0.0):
        raise ValueError(f"Expected E to be positive definite, got eigenvalues {eigvals}")
    return E, final_verification


def position_error_bounds(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project {e: e^T E e <= 1} onto body-frame position errors [e_X, e_Y].

    Returns:
        bounds: Axis-aligned maximum absolute errors [Delta_X, Delta_Y].
        position_shape: Shape matrix S_p of the projected ellipsoid
            {e_p: e_p^T S_p^{-1} e_p <= 1}.
    """
    E = torch.as_tensor(E)
    if E.shape != (ERROR_DIM, ERROR_DIM):
        raise ValueError(f"Expected E shape ({ERROR_DIM}, {ERROR_DIM}), got {tuple(E.shape)}")

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


def planar_position_radius(position_shape: torch.Tensor) -> torch.Tensor:
    """
    Return the maximum Euclidean position error over the projected ellipse.

    The Bicycle position error is expressed in a rotating frame. Rotation
    preserves its Euclidean norm, so this is also an inertial-frame bound.
    """
    position_shape = torch.as_tensor(position_shape)
    if position_shape.shape != (2, 2):
        raise ValueError(
            f"Expected position_shape shape (2, 2), got {tuple(position_shape.shape)}"
        )
    return torch.sqrt(torch.linalg.eigvalsh(position_shape).max())


def evaluate_error_bound(
    checkpoint_path: str = CHECKPOINT_PATH,
    require_verified: bool = False,
) -> dict[str, torch.Tensor]:
    E, final_verification = load_ellipsoid_matrix(
        checkpoint_path,
        require_verified=require_verified,
    )
    bounds, position_shape = position_error_bounds(E)
    position_ellipsoid_matrix = torch.linalg.inv(position_shape)
    radius = planar_position_radius(position_shape)
    min_eigenvalue = torch.linalg.eigvalsh(E).min()
    return {
        "E": E,
        "position_bounds": bounds,
        "position_shape": position_shape,
        "position_ellipsoid_matrix": position_ellipsoid_matrix,
        "planar_position_radius": radius,
        "euclidean_reference_error_bound": torch.rsqrt(min_eigenvalue),
        "verified": final_verification.get("success", False),
        "counterexample_count": len(final_verification.get("counterexamples", [])),
    }


def main() -> None:
    result = evaluate_error_bound()

    print(f"loaded E from: {CHECKPOINT_PATH}")
    print(f"final counterexample search verified: {result['verified']}")
    print(f"final counterexamples found: {result['counterexample_count']}")
    print(
        "diagnostic body-frame axis-aligned reference-error projections "
        "from r^T E r <= 1:"
    )
    for label, bound in zip(POSITION_LABELS, result["position_bounds"]):
        print(f"  |{label}| <= {bound.item():.6f}")
    print(
        "diagnostic rotation-invariant planar position projection: "
        f"||(e_X, e_Y)||_2 <= {result['planar_position_radius'].item():.6f}"
    )
    print(
        "diagnostic full moving-reference error projection: "
        f"||r||_2 <= {result['euclidean_reference_error_bound'].item():.6f}"
    )

    print("\nprojected position shape matrix S_p = P E^{-1} P^T:")
    print(result["position_shape"])
    print("\nprojected position ellipsoid matrix S_p^{-1}:")
    print(result["position_ellipsoid_matrix"])


if __name__ == "__main__":
    main()
