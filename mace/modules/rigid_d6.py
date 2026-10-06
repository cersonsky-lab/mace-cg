from __future__ import annotations

import math

import torch
from e3nn import o3

from mace.data.rigid_body import (
    quaternion_to_matrix,
)
from mace.modules.rigid_wigner import (
    wigner_body_template,
)

D6_BODY_IRREPS = o3.Irreps("1x2e + 1x6e")


def _hexagon_body_directions(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    angles = torch.arange(
        6,
        dtype=dtype,
        device=device,
    ) * (math.pi / 3.0)

    zeros = torch.zeros_like(angles)

    return torch.stack(
        (
            torch.cos(angles),
            torch.sin(angles),
            zeros,
        ),
        dim=-1,
    )


def _d6_body_templates(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    normal = torch.tensor(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=dtype,
        device=device,
    )

    plane_template = o3.spherical_harmonics(
        2,
        normal,
        normalize=True,
        normalization="component",
    )

    body_hexagon = _hexagon_body_directions(
        dtype=dtype,
        device=device,
    )

    hexatic_template = o3.spherical_harmonics(
        6,
        body_hexagon,
        normalize=True,
        normalization="component",
    ).mean(dim=-2)

    return (
        plane_template,
        hexatic_template,
    )


def d6_body_features_from_matrix(
    rotation: torch.Tensor,
) -> torch.Tensor:
    if rotation.shape[-2:] != (
        3,
        3,
    ):
        raise ValueError("rotation must have shape (..., 3, 3)")

    (
        plane_template,
        hexatic_template,
    ) = _d6_body_templates(
        dtype=rotation.dtype,
        device=rotation.device,
    )

    plane = wigner_body_template(
        rotation,
        ell=2,
        body_template=plane_template,
    )

    hexatic = wigner_body_template(
        rotation,
        ell=6,
        body_template=hexatic_template,
    )

    return torch.cat(
        (
            plane,
            hexatic,
        ),
        dim=-1,
    )


def d6_body_features(
    quaternions: torch.Tensor,
) -> torch.Tensor:
    rotation = quaternion_to_matrix(quaternions)

    return d6_body_features_from_matrix(rotation)
