import math

import torch
from e3nn import o3

from mace.data.rigid_body import (
    quaternion_to_matrix,
)
from mace.modules.rigid_c2 import (
    c2_body_irreducible_features,
)
from mace.modules.rigid_d6 import (
    _hexagon_body_directions,
    d6_body_features_from_matrix,
)
from mace.modules.rigid_wigner import (
    full_wigner_features,
    full_wigner_irreps,
    wigner_body_template,
    wigner_matrix,
)


def _legacy_c2(
    rotation: torch.Tensor,
    c2_axis: int,
) -> torch.Tensor:
    transverse = [i for i in range(3) if i != c2_axis]

    a = rotation[
        ...,
        :,
        c2_axis,
    ]

    b = rotation[
        ...,
        :,
        transverse[0],
    ]

    c = rotation[
        ...,
        :,
        transverse[1],
    ]

    q = (
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
    ) / math.sqrt(2.0)

    return torch.cat(
        (
            a,
            q,
        ),
        dim=-1,
    )


def _legacy_d6(
    rotation: torch.Tensor,
) -> torch.Tensor:
    normal = rotation[
        ...,
        :,
        2,
    ]

    plane = o3.spherical_harmonics(
        2,
        normal,
        normalize=True,
        normalization="component",
    )

    body_hexagon = _hexagon_body_directions(
        dtype=rotation.dtype,
        device=rotation.device,
    )

    space_hexagon = torch.einsum(
        "...ij,kj->...ki",
        rotation,
        body_hexagon,
    )

    hexatic = o3.spherical_harmonics(
        6,
        space_hexagon,
        normalize=True,
        normalization="component",
    ).mean(dim=-2)

    return torch.cat(
        (
            plane,
            hexatic,
        ),
        dim=-1,
    )


def test_e3nn_and_mace_quaternion_conventions_match():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    quaternion = o3.matrix_to_quaternion(rotation)

    reconstructed = quaternion_to_matrix(quaternion)

    torch.testing.assert_close(
        reconstructed,
        rotation,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_wigner_d1_is_body_to_space_rotation_matrix():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    D1 = wigner_matrix(
        rotation,
        ell=1,
    )

    torch.testing.assert_close(
        D1,
        rotation,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_complete_d1_packing_is_one_constrained_rotation():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    assert full_wigner_irreps((1,)) == o3.Irreps("3x1o")

    packed = full_wigner_features(
        rotation,
        ells=[1],
    )

    torch.testing.assert_close(
        packed.reshape(
            3,
            3,
        ),
        rotation.T,
        atol=1.0e-8,
        rtol=1.0e-8,
    )

    frame = packed.reshape(
        3,
        3,
    ).T

    torch.testing.assert_close(
        frame.T @ frame,
        torch.eye(
            3,
            dtype=dtype,
        ),
        atol=1.0e-8,
        rtol=1.0e-8,
    )

    torch.testing.assert_close(
        torch.linalg.det(frame),
        torch.tensor(
            1.0,
            dtype=dtype,
        ),
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_complete_wigner_features_global_equivariance():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    global_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    irreps = full_wigner_irreps(
        (
            1,
            2,
            3,
        )
    )

    before = full_wigner_features(
        rotation,
        ells=[
            1,
            2,
            3,
        ],
    )

    after = full_wigner_features(
        global_rotation @ rotation,
        ells=[
            1,
            2,
            3,
        ],
    )

    expected = irreps.D_from_matrix(global_rotation) @ before

    torch.testing.assert_close(
        after,
        expected,
        atol=2.0e-7,
        rtol=2.0e-7,
    )


def test_wigner_body_template_global_equivariance():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    global_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    template = torch.randn(
        5,
        dtype=dtype,
    )

    before = wigner_body_template(
        rotation,
        ell=2,
        body_template=template,
    )

    after = wigner_body_template(
        global_rotation @ rotation,
        ell=2,
        body_template=template,
    )

    expected = o3.Irrep("2e").D_from_matrix(global_rotation) @ before

    torch.testing.assert_close(
        after,
        expected,
        atol=2.0e-7,
        rtol=2.0e-7,
    )


def test_c2_wigner_form_matches_legacy_axis_formula():
    dtype = torch.float64

    rotations = torch.stack(
        [
            o3.rand_matrix(
                dtype=dtype,
            )
            for _ in range(5)
        ]
    )

    for c2_axis in (
        0,
        1,
        2,
    ):
        new = c2_body_irreducible_features(
            rotations,
            c2_axis=c2_axis,
        )

        old = _legacy_c2(
            rotations,
            c2_axis=c2_axis,
        )

        torch.testing.assert_close(
            new,
            old,
            atol=2.0e-7,
            rtol=2.0e-7,
        )


def test_d6_wigner_form_matches_legacy_geometry_formula():
    dtype = torch.float64

    rotations = torch.stack(
        [
            o3.rand_matrix(
                dtype=dtype,
            )
            for _ in range(5)
        ]
    )

    new = d6_body_features_from_matrix(rotations)

    old = _legacy_d6(rotations)

    torch.testing.assert_close(
        new,
        old,
        atol=5.0e-7,
        rtol=5.0e-7,
    )
