"""Compact equivariant rigid-body features for C2 molecules.

The representation is built from one physical body-to-space rotation R.
Fixed body-frame templates are acted on by D1(R) and D2(R), preserving the
historical compact irrep content 1o + 2e exactly.
"""

import math

import torch
from e3nn import o3

from mace.modules.rigid_wigner import (
    wigner_body_template,
)


C2_BODY_IRREPS = o3.Irreps(
    "1o + 2e"
)


def _validate_c2_axis(
    c2_axis: int,
) -> None:
    if c2_axis not in (
        0,
        1,
        2,
    ):
        raise ValueError(
            "c2_axis must be one of 0, 1, or 2; "
            f"got {c2_axis}"
        )


def _c2_body_templates(
    c2_axis: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    _validate_c2_axis(
        c2_axis
    )

    eye = torch.eye(
        3,
        dtype=dtype,
        device=device,
    )

    transverse = [
        i
        for i in range(3)
        if i != c2_axis
    ]

    l1_template = eye[
        :,
        c2_axis,
    ]

    b = eye[
        :,
        transverse[0],
    ]

    c = eye[
        :,
        transverse[1],
    ]

    l2_template = (
        o3.spherical_harmonics(
            2,
            b,
            normalize=True,
            normalization="component",
        )
        - o3.spherical_harmonics(
            2,
            c,
            normalize=True,
            normalization="component",
        )
    ) / math.sqrt(
        2.0
    )

    return (
        l1_template,
        l2_template,
    )


def c2_body_irreducible_features(
    rotation_matrices: torch.Tensor,
    c2_axis: int,
) -> torch.Tensor:
    """Return compact C2-invariant, globally equivariant body features."""

    _validate_c2_axis(
        c2_axis
    )

    if rotation_matrices.shape[-2:] != (
        3,
        3,
    ):
        raise ValueError(
            "rotation_matrices must have shape (..., 3, 3); "
            f"got {tuple(rotation_matrices.shape)}"
        )

    (
        l1_template,
        l2_template,
    ) = _c2_body_templates(
        c2_axis,
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )

    axis = wigner_body_template(
        rotation_matrices,
        ell=1,
        body_template=l1_template,
    )

    transverse_quadrupole = wigner_body_template(
        rotation_matrices,
        ell=2,
        body_template=l2_template,
    )

    return torch.cat(
        (
            axis,
            transverse_quadrupole,
        ),
        dim=-1,
    )
