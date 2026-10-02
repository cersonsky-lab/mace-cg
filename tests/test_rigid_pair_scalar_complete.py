import torch
from e3nn import o3

from mace.modules.rigid_pair_tp import (
    RigidPairScalarCompleteEdgeEmbedding,
)


torch.set_default_dtype(torch.float64)


def test_scalar_complete_preserves_raw_0e_channels():
    sh = o3.Irreps.spherical_harmonics(3)

    module = RigidPairScalarCompleteEdgeEmbedding(
        lmax=3,
        edge_irreps=sh,
    ).double()

    q = torch.tensor(
        [
            [0.91, 0.11, -0.27, 0.28],
            [0.73, -0.42, 0.31, 0.44],
        ]
    )

    q = q / torch.linalg.norm(
        q,
        dim=-1,
        keepdim=True,
    )

    edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )

    vectors = torch.tensor(
        [
            [1.7, -0.4, 0.8],
            [-1.7, 0.4, -0.8],
        ]
    )

    raw = module.full_pair(
        q,
        edge_index,
        vectors,
    )

    expected_scalar = torch.index_select(
        raw,
        dim=-1,
        index=module.scalar_indices,
    )

    out = module(
        q,
        edge_index,
        vectors,
    )

    n = module.num_raw_scalars

    print("raw scalar multiplicity:", n)
    print("output irreps:", module.edge_irreps)

    torch.testing.assert_close(
        out[:, :n],
        expected_scalar,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    # For lmax=3 generic full-frame C1 this is the multiplicity
    # implicated by the linear-probe diagnostics.
    assert n == 18

    assert (
        module.edge_irreps.count(
            o3.Irrep("0e")
        )
        == 18
    )
