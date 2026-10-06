import torch
from e3nn import o3

from mace.modules.rigid_c1 import (
    C1_BODY_IRREPS,
    c1_body_irreducible_features,
)


def test_c1_body_irreps():
    assert (
        C1_BODY_IRREPS
        == o3.Irreps(
            "3x1o + 5x2e + 7x3o"
        )
    )

    assert C1_BODY_IRREPS.dim == 83


def test_c1_body_feature_shape():
    dtype = torch.float64

    rotations = torch.stack(
        [
            torch.eye(
                3,
                dtype=dtype,
            ),
            o3.rand_matrix(
                dtype=dtype,
            ),
        ]
    )

    features = (
        c1_body_irreducible_features(
            rotations
        )
    )

    assert features.shape == (2, 83)
    assert torch.isfinite(
        features
    ).all()


def test_c1_body_global_equivariance():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    global_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    before = (
        c1_body_irreducible_features(
            rotation
        )
    )

    after = (
        c1_body_irreducible_features(
            global_rotation
            @ rotation
        )
    )

    representation = (
        C1_BODY_IRREPS.D_from_matrix(
            global_rotation
        )
    )

    expected = (
        representation
        @ before
    )

    torch.testing.assert_close(
        after,
        expected,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_c1_l1_block_is_three_vector_copies():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    features = (
        c1_body_irreducible_features(
            rotation
        )
    )

    l1 = features[:9].reshape(
        3,
        3,
    )

    # Multiplicity-major packing:
    # each row is one body-axis copy of the l=1 irrep.
    expected = rotation.T

    torch.testing.assert_close(
        l1,
        expected,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_c1_does_not_quotient_body_pi_rotations():
    dtype = torch.float64

    rotation = o3.rand_matrix(
        dtype=dtype,
    )

    reference = (
        c1_body_irreducible_features(
            rotation
        )
    )

    body_pi_rotations = (
        torch.diag(
            torch.tensor(
                [1.0, -1.0, -1.0],
                dtype=dtype,
            )
        ),
        torch.diag(
            torch.tensor(
                [-1.0, 1.0, -1.0],
                dtype=dtype,
            )
        ),
        torch.diag(
            torch.tensor(
                [-1.0, -1.0, 1.0],
                dtype=dtype,
            )
        ),
    )

    for body_rotation in body_pi_rotations:
        transformed = (
            c1_body_irreducible_features(
                rotation
                @ body_rotation
            )
        )

        assert not torch.allclose(
            reference,
            transformed,
            atol=1.0e-8,
            rtol=1.0e-8,
        )
