from __future__ import annotations

from collections.abc import Iterable
from typing import List

import torch
from e3nn import o3


def _parity_for_ell(
    ell: int,
) -> int:
    if ell < 0:
        raise ValueError(f"ell must be nonnegative, got {ell}")

    return -1 if ell % 2 else 1


def full_wigner_irreps(
    ells: Iterable[int],
) -> o3.Irreps:
    """Return irreps for complete Wigner-D matrices.

    For each ell, the body index labels multiplicity copies while the
    space index carries the ell irrep.

    These multiplicity copies are constrained columns of one common
    physical rotation, not independent vectors.
    """

    ells = tuple(int(ell) for ell in ells)

    return o3.Irreps(
        [
            (
                2 * ell + 1,
                o3.Irrep(
                    ell,
                    _parity_for_ell(ell),
                ),
            )
            for ell in ells
        ]
    )


def wigner_matrix(
    rotation: torch.Tensor,
    ell: int,
) -> torch.Tensor:
    """Return D^ell(R) for a proper body-to-space rotation R."""

    if rotation.size(-2) != 3 or rotation.size(-1) != 3:
        raise ValueError("rotation must have shape (..., 3, 3)")

    alpha, beta, gamma = o3.matrix_to_angles(rotation)

    return o3.wigner_D(
        ell,
        alpha,
        beta,
        gamma,
    )


def full_wigner_features(
    rotation: torch.Tensor,
    ells: List[int],
) -> torch.Tensor:
    """Pack complete Wigner-D matrices in multiplicity-major order."""

    blocks = torch.jit.annotate(
        List[torch.Tensor],
        [],
    )

    for ell in ells:
        D = wigner_matrix(
            rotation,
            ell,
        )

        blocks.append(
            D.transpose(
                -1,
                -2,
            ).flatten(
                start_dim=-2,
            )
        )

    return torch.cat(
        blocks,
        dim=-1,
    )


def wigner_body_template(
    rotation: torch.Tensor,
    ell: int,
    body_template: torch.Tensor,
) -> torch.Tensor:
    """Return D^ell(R) times one fixed body-frame irrep template."""

    ell = int(ell)

    expected_dim = 2 * ell + 1

    if body_template.shape != (expected_dim,):
        raise ValueError(
            "body_template must have shape "
            f"({expected_dim},), "
            f"got {tuple(body_template.shape)}"
        )

    template = body_template.to(
        dtype=rotation.dtype,
        device=rotation.device,
    )

    D = wigner_matrix(
        rotation,
        ell,
    )

    return torch.einsum(
        "...mn,n->...m",
        D,
        template,
    )
