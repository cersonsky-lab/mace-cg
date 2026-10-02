import math

import torch

from mace.modules.rigid_pair_tp import (
    RigidPairOracleMixedScalarEdgeEmbedding,
)


torch.set_default_dtype(torch.float64)


def qmul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack(
        (
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ),
        dim=-1,
    )


def normalized(q):
    return q / torch.linalg.norm(
        q,
        dim=-1,
        keepdim=True,
    )


def test_oracle_is_edge_exchange_symmetric_and_c2_odd():
    model = RigidPairOracleMixedScalarEdgeEmbedding()

    q = normalized(
        torch.tensor(
            [
                [0.91, 0.11, -0.27, 0.28],
                [0.73, -0.42, 0.31, 0.44],
            ]
        )
    )

    edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [
            [1.7, -0.4, 0.8],
            [-1.7, 0.4, -0.8],
        ]
    )

    value = model(
        q,
        edge_index,
        edge_vectors,
    )

    assert value.shape == (2, 1)

    torch.testing.assert_close(
        value[0],
        value[1],
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    half_turn_axis1 = torch.tensor(
        [0.0, 0.0, 1.0, 0.0]
    )

    q_flipped = q.clone()
    q_flipped[0] = normalized(
        qmul(
            q_flipped[0],
            half_turn_axis1,
        )
    )

    flipped = model(
        q_flipped,
        edge_index,
        edge_vectors,
    )

    torch.testing.assert_close(
        flipped,
        -value,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
