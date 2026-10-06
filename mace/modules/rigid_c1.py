import torch
from e3nn import o3


C1_BODY_IRREPS = o3.Irreps(
    "3x1o + 5x2e + 7x3o"
)


def c1_body_irreducible_features(
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """Return complete C1 body-orientation features through l=3.

    Parameters
    ----------
    rotation_matrices
        Body-to-space rotation matrices with shape ``(..., 3, 3)``.

        For a global rotation G,

            R -> G R

        so the space index of each Wigner matrix transforms under the
        corresponding SO(3) irrep, while the body index labels
        multiplicity copies.

    Returns
    -------
    torch.Tensor
        Features with shape ``(..., 83)`` transforming as

            3x1o + 5x2e + 7x3o.

        For each l, the full Wigner matrix D^l(R) has shape

            (..., 2l+1, 2l+1),

        with the first matrix index carrying the space-frame irrep and
        the second matrix index carrying the body-frame multiplicity.

        The output is packed multiplicity-major: for each fixed body
        index n, the complete space irrep D^l_{m n}(R) is contiguous.
    """

    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrices must have shape (..., 3, 3); "
            f"got {tuple(rotation_matrices.shape)}"
        )

    blocks = []

    for ell, parity in (
        (1, -1),
        (2, +1),
        (3, -1),
    ):
        irrep = o3.Irrep(
            ell,
            parity,
        )

        D = irrep.D_from_matrix(
            rotation_matrices
        )

        block = D.transpose(
            -1,
            -2,
        ).reshape(
            rotation_matrices.shape[:-2]
            + ((2 * ell + 1) ** 2,)
        )

        blocks.append(
            block
        )

    features = torch.cat(
        blocks,
        dim=-1,
    )

    if features.shape[-1] != C1_BODY_IRREPS.dim:
        raise RuntimeError(
            "Internal C1 feature dimension mismatch: "
            f"got {features.shape[-1]}, "
            f"expected {C1_BODY_IRREPS.dim}"
        )

    return features