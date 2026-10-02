import torch
from e3nn import o3

from mace.modules.rigid_pair_tp import (
    C1_WIGNER_L123_BODY_IRREPS,
    RigidPairC1WignerL123EdgeEmbedding,
)


def test_c1_wigner_l123_body_content_and_equivariance():
    dtype = torch.float64

    edge_irreps = o3.Irreps.spherical_harmonics(3)

    embedding = RigidPairC1WignerL123EdgeEmbedding(
        lmax=3,
        edge_irreps=edge_irreps,
        multiplicity=1,
    ).to(dtype=dtype)

    assert (
        embedding.body_irreps
        == C1_WIGNER_L123_BODY_IRREPS
    )

    assert (
        embedding.body_irreps
        == o3.Irreps(
            "3x1o + 5x2e + 7x3o"
        )
    )

    assert embedding.body_irreps.dim == 83
    assert embedding.edge_irreps == edge_irreps

    torch.manual_seed(20260929)

    rotations = o3.rand_matrix(
        4,
        dtype=dtype,
    )

    global_rotation = o3.rand_matrix(
        dtype=dtype,
    )

    before = embedding.body_features(
        rotations
    )

    after = embedding.body_features(
        global_rotation @ rotations
    )

    expected = torch.einsum(
        "ij,nj->ni",
        embedding.body_irreps.D_from_matrix(
            global_rotation
        ),
        before,
    )

    torch.testing.assert_close(
        after,
        expected,
        atol=1.0e-7,
        rtol=1.0e-7,
    )
