import torch

from mace.modules.rigid_pair_tp import (
    RigidPairPoseInvariantEdgeEmbedding,
    validate_rigid_pair_mode,
)


torch.set_default_dtype(
    torch.float64
)


def normalize(q):
    return (
        q
        / torch.linalg.norm(q)
    )


def qmul(a, b):
    aw = a[0]
    av = a[1:]

    bw = b[0]
    bv = b[1:]

    return torch.cat(
        (
            (
                aw*bw
                - torch.dot(av, bv)
            ).reshape(1),
            (
                aw*bv
                + bw*av
                + torch.cross(
                    av,
                    bv,
                    dim=0,
                )
            ),
        )
    )


def qmat(q):
    w, x, y, z = q

    return torch.stack(
        (
            torch.stack(
                (
                    1-2*(y*y+z*z),
                    2*(x*y-w*z),
                    2*(x*z+w*y),
                )
            ),
            torch.stack(
                (
                    2*(x*y+w*z),
                    1-2*(x*x+z*z),
                    2*(y*z-w*x),
                )
            ),
            torch.stack(
                (
                    2*(x*z-w*y),
                    2*(y*z+w*x),
                    1-2*(x*x+y*y),
                )
            ),
        )
    )


def test_pose_invariant_global_rotation_and_exchange():
    module = (
        RigidPairPoseInvariantEdgeEmbedding()
        .double()
    )

    q0 = normalize(
        torch.tensor(
            [0.91, 0.11, -0.27, 0.28]
        )
    )

    q1 = normalize(
        torch.tensor(
            [0.73, -0.42, 0.31, 0.44]
        )
    )

    q = torch.stack(
        (q0, q1)
    )

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    v = torch.tensor(
        [
            [1.7, -0.4, 0.8],
            [-1.7, 0.4, -0.8],
        ]
    )

    base = module(
        q,
        edge_index,
        v,
    )

    # Directed-edge reversal relation.
    A = base[
        0,
        :9,
    ].reshape(
        3,
        3,
    )

    u = base[
        0,
        9:12,
    ]

    A_rev = base[
        1,
        :9,
    ].reshape(
        3,
        3,
    )

    u_rev = base[
        1,
        9:12,
    ]

    torch.testing.assert_close(
        A_rev,
        A.T,
        atol=1e-12,
        rtol=1e-12,
    )

    torch.testing.assert_close(
        u_rev,
        -A.T @ u,
        atol=1e-12,
        rtol=1e-12,
    )

    # Common global proper rotation.
    g = normalize(
        torch.tensor(
            [0.82, -0.21, 0.33, 0.41]
        )
    )

    q_rot = torch.stack(
        (
            normalize(
                qmul(g, q0)
            ),
            normalize(
                qmul(g, q1)
            ),
        )
    )

    G = qmat(g)

    v_rot = torch.einsum(
        "ab,eb->ea",
        G,
        v,
    )

    rotated = module(
        q_rot,
        edge_index,
        v_rot,
    )

    torch.testing.assert_close(
        rotated,
        base,
        atol=2e-12,
        rtol=2e-12,
    )

    assert (
        str(module.edge_irreps)
        == "12x0e"
    )

    assert (
        validate_rigid_pair_mode(
            "pose_invariant_exact"
        )
        == "pose_invariant_exact"
    )
