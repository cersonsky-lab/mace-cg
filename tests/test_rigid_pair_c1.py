import torch
from e3nn import o3

from mace.modules.rigid_pair_tp import (
    RigidPairC1EdgeEmbedding,
)


def _matrix_to_wxyz(
    matrix: torch.Tensor,
) -> torch.Tensor:
    """e3nn and MACE use scalar-first [w, x, y, z] here."""

    return o3.matrix_to_quaternion(
        matrix
    )


def test_c1_pair_forward_shape_and_finite():
    dtype = torch.float64

    module = (
        RigidPairC1EdgeEmbedding(
            max_ell=2,
            multiplicity=1,
        )
        .to(dtype=dtype)
    )

    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )

    edge_index = torch.tensor(
        [
            [0],
            [1],
        ],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [
            [
                0.8,
                -0.4,
                0.6,
            ]
        ],
        dtype=dtype,
    )

    output = module(
        quaternions,
        edge_index,
        edge_vectors,
    )

    assert output.shape == (
        1,
        module.edge_irreps.dim,
    )

    assert torch.isfinite(
        output
    ).all()


def test_c1_pair_global_equivariance():
    dtype = torch.float64

    module = (
        RigidPairC1EdgeEmbedding(
            max_ell=2,
            multiplicity=1,
        )
        .to(dtype=dtype)
    )

    sender_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    receiver_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    quaternions = torch.stack(
        (
            _matrix_to_wxyz(
                sender_rotation
            ),
            _matrix_to_wxyz(
                receiver_rotation
            ),
        )
    )

    edge_index = torch.tensor(
        [
            [0],
            [1],
        ],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [
            [
                0.8,
                -0.4,
                0.6,
            ]
        ],
        dtype=dtype,
    )

    before = module(
        quaternions,
        edge_index,
        edge_vectors,
    )

    global_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    rotated_quaternions = torch.stack(
        (
            _matrix_to_wxyz(
                global_rotation
                @ sender_rotation
            ),
            _matrix_to_wxyz(
                global_rotation
                @ receiver_rotation
            ),
        )
    )

    rotated_edges = (
        edge_vectors
        @ global_rotation.T
    )

    after = module(
        rotated_quaternions,
        edge_index,
        rotated_edges,
    )

    representation = (
        module.edge_irreps.D_from_matrix(
            global_rotation
        )
    )

    expected = (
        before
        @ representation.T
    )

    torch.testing.assert_close(
        after,
        expected,
        atol=2.0e-7,
        rtol=2.0e-7,
    )
