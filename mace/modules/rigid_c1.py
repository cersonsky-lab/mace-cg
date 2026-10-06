import torch

from mace.modules.rigid_wigner import (
    full_wigner_features,
    full_wigner_irreps,
)

C1_WIGNER_ELLS = (
    1,
    2,
    3,
)

C1_BODY_IRREPS = full_wigner_irreps(C1_WIGNER_ELLS)


def c1_body_irreducible_features(
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """Return D1(R) + D2(R) + D3(R) for one C1 orientation."""

    features = full_wigner_features(
        rotation_matrices,
        C1_WIGNER_ELLS,
    )

    if features.shape[-1] != C1_BODY_IRREPS.dim:
        raise RuntimeError(
            "Internal C1 feature dimension mismatch: "
            f"got {features.shape[-1]}, "
            f"expected {C1_BODY_IRREPS.dim}"
        )

    return features
