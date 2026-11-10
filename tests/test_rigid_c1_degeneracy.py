import math

import torch
from e3nn import o3

from mace.modules.rigid_pair_tp import RigidPairC1EdgeEmbedding


DTYPE = torch.float64
COLLISION_TOL = 1.0e-10
MIN_ROTATION_SEPARATION = 1.0e-6


def qmul(q1, q2):
    """Scalar-first quaternion product."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def axis_angle_quaternion(axis, angle):
    axis = torch.as_tensor(
        axis,
        dtype=DTYPE,
    )
    axis = axis / torch.linalg.vector_norm(axis)

    half = 0.5 * angle

    return torch.cat(
        (
            torch.tensor(
                [math.cos(half)],
                dtype=DTYPE,
            ),
            axis * math.sin(half),
        )
    )


def rotation_bank():
    """Deterministic mixture of structured and generic SO(3) samples."""
    quaternions = [
        torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=DTYPE,
        )
    ]

    axes = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 2.0, 3.0),
        (-2.0, 1.0, 3.0),
    ]

    angles = [
        math.pi / 12.0,
        math.pi / 6.0,
        math.pi / 4.0,
        math.pi / 3.0,
        math.pi / 2.0,
        2.0 * math.pi / 3.0,
        3.0 * math.pi / 4.0,
        math.pi,
    ]

    for axis in axes:
        for angle in angles:
            quaternions.append(
                axis_angle_quaternion(
                    axis,
                    angle,
                )
            )

    generator = torch.Generator()
    generator.manual_seed(20261006)

    random_q = torch.randn(
        96,
        4,
        generator=generator,
        dtype=DTYPE,
    )

    random_q = random_q / torch.linalg.vector_norm(
        random_q,
        dim=-1,
        keepdim=True,
    )

    quaternions.extend(
        random_q.unbind(0)
    )

    q = torch.stack(quaternions)

    # Canonicalize the quaternion double cover only for reporting.
    q = torch.where(
        q[:, :1] < 0.0,
        -q,
        q,
    )

    return q


def relative_rotation_angle(q1, q2):
    dot = torch.abs(
        torch.sum(q1 * q2)
    ).clamp(
        min=0.0,
        max=1.0,
    )

    return 2.0 * torch.acos(dot)


def relative_rotation_axis_angle(q1, q2):
    q1_conj = q1.clone()
    q1_conj[1:] *= -1.0

    qrel = qmul(
        q2,
        q1_conj,
    )

    if qrel[0] < 0.0:
        qrel = -qrel

    qrel = qrel / torch.linalg.vector_norm(qrel)

    vector_norm = torch.linalg.vector_norm(
        qrel[1:]
    )

    angle = 2.0 * torch.atan2(
        vector_norm,
        qrel[0].clamp(min=0.0),
    )

    if float(vector_norm) < 1.0e-12:
        axis = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=DTYPE,
        )
    else:
        axis = qrel[1:] / vector_norm

    return axis, angle


def make_model():
    return RigidPairC1EdgeEmbedding(
        max_ell=3,
        multiplicity=1,
        edge_irreps=o3.Irreps.spherical_harmonics(3),
    ).to(
        dtype=DTYPE
    )


def stage_features(
    model,
    sender_quaternions,
    receiver_quaternion,
):
    """Evaluate every C1 pair stage while varying the sender rotation."""
    n = sender_quaternions.shape[0]

    receiver_quaternions = receiver_quaternion.expand(
        n,
        -1,
    )

    all_quaternions = torch.cat(
        (
            sender_quaternions,
            receiver_quaternions,
        ),
        dim=0,
    )

    body = model.body_features(
        all_quaternions
    )

    body_sender = body[:n]
    body_receiver = body[n:]

    # Deliberately generic edge direction.
    edge_vector = torch.tensor(
        [0.371, -0.529, 0.763],
        dtype=DTYPE,
    )

    edge_vector = edge_vector / torch.linalg.vector_norm(
        edge_vector
    )

    edge_vectors = edge_vector.expand(
        n,
        -1,
    )

    # Eager diagnostic: use the public e3nn implementation as the
    # canonical edge-SH reference.
    edge_sh = o3.spherical_harmonics(
        model.sh_irreps,
        edge_vectors,
        normalize=True,
        normalization="component",
    )

    edge_sender = model.edge_body_tp(
        edge_sh,
        body_sender,
    )

    pair = model.pair_tp(
        edge_sender,
        body_receiver,
    )

    projected = model.projection(
        pair
    )

    if model.multiplicity > 1:
        projected = projected / (
            model.multiplicity**0.5
        )

    return {
        "body": body_sender,
        "edge_x_sender": edge_sender,
        "pair_filtered": pair,
        "projected": projected,
    }


def normalized_pair_distance(x):
    """Pairwise feature distance normalized by feature norms."""
    distances = torch.cdist(
        x,
        x,
    )

    norms = torch.linalg.vector_norm(
        x,
        dim=-1,
    )

    scale = torch.maximum(
        norms[:, None],
        norms[None, :],
    ).clamp(
        min=1.0,
    )

    return distances / scale


def physical_angle_matrix(quaternions):
    dots = torch.abs(
        quaternions @ quaternions.T
    ).clamp(
        max=1.0,
    )

    return 2.0 * torch.acos(dots)


def report_global_collisions(
    quaternions,
    stages,
):
    angle_matrix = physical_angle_matrix(
        quaternions
    )

    distinct = (
        angle_matrix
        > MIN_ROTATION_SEPARATION
    )

    upper = torch.triu(
        torch.ones_like(
            distinct,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    valid = distinct & upper

    print()
    print("=" * 79)
    print("GLOBAL C1 ROTATION COLLISION SEARCH")
    print("=" * 79)

    first_collision = {}

    for stage_name, features in stages.items():
        d = normalized_pair_distance(
            features
        )

        masked = d.clone()
        masked[~valid] = float("inf")

        min_value, flat_index = masked.reshape(
            -1
        ).min(
            dim=0
        )

        i = int(
            flat_index
            // masked.shape[1]
        )
        j = int(
            flat_index
            % masked.shape[1]
        )

        angle = relative_rotation_angle(
            quaternions[i],
            quaternions[j],
        )

        print()
        print(
            "stage:",
            stage_name,
        )
        print(
            "  dim:",
            features.shape[-1],
        )
        print(
            "  nearest distinct normalized distance:",
            float(min_value),
        )
        print(
            "  corresponding physical separation [deg]:",
            float(angle * 180.0 / math.pi),
        )

        if float(min_value) <= COLLISION_TOL:
            axis, rel_angle = relative_rotation_axis_angle(
                quaternions[i],
                quaternions[j],
            )

            first_collision[stage_name] = (
                i,
                j,
            )

            print(
                "  *** DEGENERATE ROTATIONS FOUND ***"
            )
            print(
                "  rotation indices:",
                i,
                j,
            )
            print(
                "  relative axis:",
                axis.tolist(),
            )
            print(
                "  relative angle [deg]:",
                float(
                    rel_angle
                    * 180.0
                    / math.pi
                ),
            )
        else:
            print(
                "  no sampled collision"
            )

    return first_collision


def infinitesimal_quaternion(axis, epsilon):
    axis = torch.as_tensor(
        axis,
        dtype=DTYPE,
    )

    axis = axis / torch.linalg.vector_norm(
        axis
    )

    return axis_angle_quaternion(
        axis,
        epsilon,
    )


def local_jacobian_rank(
    model,
    base_sender,
    receiver,
    epsilon=1.0e-6,
):
    axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    plus = []
    minus = []

    for axis in axes:
        dq_plus = infinitesimal_quaternion(
            axis,
            epsilon,
        )
        dq_minus = infinitesimal_quaternion(
            axis,
            -epsilon,
        )

        plus.append(
            qmul(
                dq_plus,
                base_sender,
            )
        )

        minus.append(
            qmul(
                dq_minus,
                base_sender,
            )
        )

    plus = torch.stack(plus)
    minus = torch.stack(minus)

    plus_stages = stage_features(
        model,
        plus,
        receiver,
    )

    minus_stages = stage_features(
        model,
        minus,
        receiver,
    )

    results = {}

    print()
    print("=" * 79)
    print("LOCAL C1 ROTATIONAL JACOBIAN")
    print("=" * 79)

    for name in plus_stages:
        columns = []

        for k in range(3):
            derivative = (
                plus_stages[name][k]
                - minus_stages[name][k]
            ) / (
                2.0 * epsilon
            )

            columns.append(
                derivative
            )

        jacobian = torch.stack(
            columns,
            dim=-1,
        )

        singular_values = torch.linalg.svdvals(
            jacobian
        )

        tolerance = (
            singular_values.max()
            * 1.0e-8
        )

        rank = int(
            torch.sum(
                singular_values > tolerance
            )
        )

        results[name] = (
            rank,
            singular_values,
        )

        print()
        print(
            "stage:",
            name,
        )
        print(
            "  rotational Jacobian rank:",
            rank,
            "/ 3",
        )
        print(
            "  singular values:",
            singular_values.tolist(),
        )

        if rank < 3:
            print(
                "  *** LOCAL ROTATIONAL DOF LOST HERE ***"
            )

    return results


def test_c1_rotation_degeneracy_pipeline():
    model = make_model()

    quaternions = rotation_bank()

    receiver = axis_angle_quaternion(
        (1.0, -2.0, 0.7),
        0.731,
    )

    stages = stage_features(
        model,
        quaternions,
        receiver,
    )

    collisions = report_global_collisions(
        quaternions,
        stages,
    )

    # The C1 body representation itself must distinguish every
    # distinct physical SO(3) rotation in the sampled bank.
    assert "body" not in collisions

    # D1 is already injective in R, so loss here would mean the C1
    # representation itself is incorrect rather than merely compressed.
    body_distance = normalized_pair_distance(
        stages["body"]
    )

    angle_matrix = physical_angle_matrix(
        quaternions
    )

    distinct = (
        angle_matrix
        > MIN_ROTATION_SEPARATION
    )

    upper = torch.triu(
        torch.ones_like(
            distinct,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    assert torch.all(
        body_distance[
            distinct & upper
        ]
        > COLLISION_TOL
    )

    # Probe local distinguishability at several unrelated orientations.
    base_indices = (
        0,
        7,
        31,
        73,
        119,
    )

    for index in base_indices:
        ranks = local_jacobian_rank(
            model,
            quaternions[index],
            receiver,
        )

        assert ranks["body"][0] == 3



def tangent_basis(direction):
    direction = direction / torch.linalg.vector_norm(direction)

    reference = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=DTYPE,
    )

    if float(torch.abs(torch.dot(direction, reference))) > 0.9:
        reference = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=DTYPE,
        )

    t1 = reference - torch.dot(
        reference,
        direction,
    ) * direction

    t1 = t1 / torch.linalg.vector_norm(t1)

    t2 = torch.linalg.cross(direction, t1)

    return t1, t2


def normalize_direction(x):
    return x / torch.linalg.vector_norm(
        x,
        dim=-1,
        keepdim=True,
    )


def stage_pair_features(
    model,
    sender_quaternions,
    edge_directions,
):
    """Gauge-fixed C1 pair: receiver orientation is identity."""
    n = sender_quaternions.shape[0]

    if edge_directions.shape != (n, 3):
        raise ValueError(
            "edge_directions must have shape (N, 3)"
        )

    identity = torch.tensor(
        [1.0, 0.0, 0.0, 0.0],
        dtype=DTYPE,
    )

    receiver_quaternions = identity.expand(n, -1)

    all_quaternions = torch.cat(
        (
            sender_quaternions,
            receiver_quaternions,
        ),
        dim=0,
    )

    body = model.body_features(all_quaternions)

    body_sender = body[:n]
    body_receiver = body[n:]

    edge_sh = o3.spherical_harmonics(
        model.sh_irreps,
        edge_directions,
        normalize=True,
        normalization="component",
    )

    edge_sender = model.edge_body_tp(
        edge_sh,
        body_sender,
    )

    pair = model.pair_tp(
        edge_sender,
        body_receiver,
    )

    projected = model.projection(pair)

    if model.multiplicity > 1:
        projected = projected / (model.multiplicity ** 0.5)

    return {
        "body": body_sender,
        "edge_sh": edge_sh,
        "edge_x_sender": edge_sender,
        "pair_filtered": pair,
        "projected": projected,
    }


def pair_local_jacobian_rank(
    model,
    sender,
    direction,
    epsilon=1.0e-6,
):
    """Finite-difference Jacobian on SO(3) x S2.

    Coordinates:
        3 sender-rotation perturbations
        2 edge-direction tangent perturbations

    The receiver orientation is fixed to identity.
    """
    direction = normalize_direction(direction)

    rotation_axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    t1, t2 = tangent_basis(direction)

    plus_q = []
    minus_q = []
    plus_r = []
    minus_r = []

    for axis in rotation_axes:
        dq_plus = infinitesimal_quaternion(axis, epsilon)
        dq_minus = infinitesimal_quaternion(axis, -epsilon)

        plus_q.append(qmul(dq_plus, sender))
        minus_q.append(qmul(dq_minus, sender))

        plus_r.append(direction)
        minus_r.append(direction)

    for tangent in (t1, t2):
        plus_q.append(sender)
        minus_q.append(sender)

        plus_r.append(
            normalize_direction(direction + epsilon * tangent)
        )
        minus_r.append(
            normalize_direction(direction - epsilon * tangent)
        )

    plus_q = torch.stack(plus_q)
    minus_q = torch.stack(minus_q)
    plus_r = torch.stack(plus_r)
    minus_r = torch.stack(minus_r)

    plus = stage_pair_features(
        model,
        plus_q,
        plus_r,
    )

    minus = stage_pair_features(
        model,
        minus_q,
        minus_r,
    )

    results = {}

    print()
    print("=" * 79)
    print("LOCAL GAUGE-FIXED C1 PAIR JACOBIAN")
    print("=" * 79)

    for name in plus:
        columns = []

        for k in range(5):
            derivative = (
                plus[name][k] - minus[name][k]
            ) / (2.0 * epsilon)

            columns.append(derivative)

        jacobian = torch.stack(columns, dim=-1)

        singular_values = torch.linalg.svdvals(jacobian)

        largest = float(singular_values.max().detach())

        tolerance = max(
            largest * 1.0e-8,
            1.0e-10,
        )

        rank = int(
            torch.sum(
                singular_values.detach() > tolerance
            )
        )

        results[name] = (rank, singular_values.detach())

        print()
        print("stage:", name)
        print("  feature dimension:", jacobian.shape[0])
        print("  pair Jacobian rank:", rank, "/ 5")
        print("  singular values:", singular_values.detach().tolist())

        if name == "body":
            print("  expected rank <= 3: no edge dependence")
        elif name == "edge_sh":
            print("  expected rank <= 2: no sender dependence")
        elif rank < 5:
            print("  *** PAIR DOF LOST AT OR BEFORE THIS STAGE ***")

    return results


def test_c1_gauge_fixed_pair_local_completeness():
    model = make_model()

    quaternions = rotation_bank()

    directions = (
        torch.tensor(
            [0.371, -0.529, 0.763],
            dtype=DTYPE,
        ),
        torch.tensor(
            [-0.612, 0.721, 0.326],
            dtype=DTYPE,
        ),
        torch.tensor(
            [0.193, 0.947, -0.257],
            dtype=DTYPE,
        ),
    )

    indices = (7, 31, 73)

    for index, direction in zip(indices, directions):
        direction = normalize_direction(direction)

        ranks = pair_local_jacobian_rank(
            model,
            quaternions[index],
            direction,
        )

        assert ranks["body"][0] == 3
        assert ranks["edge_sh"][0] == 2
        assert ranks["edge_x_sender"][0] == 5

        print()
        print(
            "Final projected pair rank:",
            ranks["projected"][0],
            "/ 5",
        )

if __name__ == "__main__":
    test_c1_rotation_degeneracy_pipeline()
