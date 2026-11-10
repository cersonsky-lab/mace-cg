import math

import torch
from e3nn import o3

from mace.modules.rigid_pair_tp import (
    RigidPairPoseMLPEdgeEmbedding,
    validate_rigid_pair_mode,
)


def qmul(a, b):
    """Hamilton product, scalar-first wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)

    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def q_to_matrix(q):
    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    )

    w, x, y, z = q.unbind(-1)

    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def axis_angle_q(axis, angle):
    axis = torch.as_tensor(
        axis,
        dtype=torch.float64,
    )

    axis = axis / torch.linalg.vector_norm(axis)

    half = 0.5 * angle

    return torch.cat(
        (
            torch.tensor(
                [math.cos(half)],
                dtype=torch.float64,
            ),
            axis * math.sin(half),
        )
    )


def test_pose_mlp_global_so3_invariance():
    torch.manual_seed(101)

    model = RigidPairPoseMLPEdgeEmbedding(
        r_max=8.0,
        num_radial=16,
        hidden_channels=64,
        out_channels=32,
    ).double()

    assert model.edge_irreps == o3.Irreps("32x0e")

    assert validate_rigid_pair_mode("pose_mlp") == "pose_mlp"

    q0 = axis_angle_q(
        [1.0, 0.3, -0.2],
        0.71,
    )

    q1 = axis_angle_q(
        [-0.1, 0.8, 0.4],
        -1.03,
    )

    q = torch.stack(
        [q0, q1],
        dim=0,
    )

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    v = torch.tensor(
        [2.1, -1.3, 0.7],
        dtype=torch.float64,
    )

    edge_vectors = torch.stack(
        [
            v,
            -v,
        ],
        dim=0,
    )

    y = model(
        q,
        edge_index,
        edge_vectors,
    )

    gq = axis_angle_q(
        [0.2, -0.5, 0.9],
        0.83,
    )

    q_rot = qmul(
        gq.expand_as(q),
        q,
    )

    G = q_to_matrix(gq)

    edge_vectors_rot = edge_vectors @ G.T

    y_rot = model(
        q_rot,
        edge_index,
        edge_vectors_rot,
    )

    assert y.shape == (
        2,
        32,
    )

    torch.testing.assert_close(
        y,
        y_rot,
        rtol=1.0e-9,
        atol=1.0e-9,
    )

    loss = y.square().sum()
    loss.backward()

    grads = [p.grad for p in model.mlp.parameters() if p.requires_grad]

    assert grads
    assert all(grad is not None for grad in grads)
