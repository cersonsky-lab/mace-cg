"""Tests for cubic (T, O, I) body-symmetry edge embeddings.

Properties tested for each group:
  1. Output irreps / dimensionality.
  2. Invariance to right-multiplication by representative group elements.
  3. Global SO(3) equivariance.
  4. Sensitivity to a rotation NOT in the group (sanity check).

Group representations used:
  T  (chiral tetrahedral, |T|=12):  identity + 8 C3 vertex rotations + 3 C2 face rotations
  O  (chiral octahedral,  |O|=24):  identity + 6 C4 + 3 C2(face) + 8 C3 + 6 C2(edge)
  I  (chiral icosahedral, |I|=60):  identity + 12 C5 + 12 C5² + 20 C3 + 15 C2
"""
from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

from mace.modules.rigid_cubic import (
    I_BODY_IRREPS,
    O_BODY_IRREPS,
    T_BODY_IRREPS,
    IBodyFeatures,
    OBodyFeatures,
    TBodyFeatures,
)
from mace.modules.rigid_pair_tp import (
    RigidPairIEdgeEmbedding,
    RigidPairOEdgeEmbedding,
    RigidPairTEdgeEmbedding,
)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _axis_angle_quaternion(axis, angle: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    half = 0.5 * angle
    return torch.cat(
        (torch.tensor([math.cos(half)], dtype=torch.float64), axis * math.sin(half))
    )


def _quaternion_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind()
    bw, bx, by, bz = b.unbind()
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _rotation_matrix(axis, angle: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    zero = torch.zeros((), dtype=torch.float64)
    K = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    eye = torch.eye(3, dtype=torch.float64)
    return eye + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


# ---------------------------------------------------------------------------
# Group element lists
# ---------------------------------------------------------------------------


def _t_quaternions() -> list[torch.Tensor]:
    """12 elements of the chiral tetrahedral group T."""
    sq3 = math.sqrt(3.0)
    body_diagonals = [
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ]
    result = [_axis_angle_quaternion((0.0, 0.0, 1.0), 0.0)]  # identity
    for axis in body_diagonals:
        result.append(_axis_angle_quaternion(axis, 2.0 * math.pi / 3.0))
        result.append(_axis_angle_quaternion(axis, 4.0 * math.pi / 3.0))
    # 3 C2 rotations about x, y, z
    for axis in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        result.append(_axis_angle_quaternion(axis, math.pi))
    return result  # 1 + 8 + 3 = 12


def _o_quaternions() -> list[torch.Tensor]:
    """24 elements of the chiral octahedral group O."""
    result = []

    # Identity
    result.append(_axis_angle_quaternion((0.0, 0.0, 1.0), 0.0))

    # 6 face C4 rotations (±π/2 about x, y, z)
    for axis in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        result.append(_axis_angle_quaternion(axis, math.pi / 2.0))
        result.append(_axis_angle_quaternion(axis, -math.pi / 2.0))

    # 3 face C2 rotations (π about x, y, z)
    for axis in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        result.append(_axis_angle_quaternion(axis, math.pi))

    # 8 vertex C3 rotations (±2π/3 about body diagonals)
    for axis in [
        (1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)
    ]:
        result.append(_axis_angle_quaternion(axis, 2.0 * math.pi / 3.0))
        result.append(_axis_angle_quaternion(axis, 4.0 * math.pi / 3.0))

    # 6 edge C2 rotations (π about face-diagonal axes)
    for axis in [
        (1.0, 1.0, 0.0), (1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0), (1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0), (0.0, 1.0, -1.0),
    ]:
        result.append(_axis_angle_quaternion(axis, math.pi))

    return result  # 1 + 6 + 3 + 8 + 6 = 24


def _i_quaternions() -> list[torch.Tensor]:
    """60 elements of the chiral icosahedral group I.

    Generated from the 12 icosahedral vertices (±1, ±φ, 0) and permutations,
    where φ = (1 + √5) / 2.  We enumerate the 6 C5 axes and their powers.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    # 6 icosahedral axes (12 vertices / 2)
    c5_axes = [
        (0.0, 1.0, phi), (0.0, 1.0, -phi),
        (1.0, phi, 0.0), (1.0, -phi, 0.0),
        (phi, 0.0, 1.0), (phi, 0.0, -1.0),
    ]
    # 10 C3 axes (body diagonals of icosahedron / dodecahedron faces)
    c3_axes = [
        (1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0),
        (0.0, 1.0 / phi, phi), (0.0, 1.0 / phi, -phi),
        (1.0 / phi, phi, 0.0), (1.0 / phi, -phi, 0.0),
        (phi, 0.0, 1.0 / phi), (phi, 0.0, -1.0 / phi),
    ]
    # 15 C2 axes (edge midpoints)
    c2_axes = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.0, phi, 1.0), (0.0, phi, -1.0),
        (phi, 1.0, 0.0), (phi, -1.0, 0.0),
        (1.0, 0.0, phi), (-1.0, 0.0, phi),
        (1.0, phi, 0.0), (1.0, -phi, 0.0),
        (phi, 0.0, 1.0), (phi, 0.0, -1.0),
        (0.0, 1.0, phi), (0.0, -1.0, phi),
    ]

    result = [_axis_angle_quaternion((0.0, 0.0, 1.0), 0.0)]  # identity

    for axis in c5_axes:
        for k in range(1, 5):
            result.append(_axis_angle_quaternion(axis, k * 2.0 * math.pi / 5.0))

    for axis in c3_axes:
        result.append(_axis_angle_quaternion(axis, 2.0 * math.pi / 3.0))
        result.append(_axis_angle_quaternion(axis, 4.0 * math.pi / 3.0))

    for axis in c2_axes:
        result.append(_axis_angle_quaternion(axis, math.pi))

    return result  # 1 + 24 + 20 + 15 = 60


# ---------------------------------------------------------------------------
# Common fixture
# ---------------------------------------------------------------------------


def _example():
    q0 = _axis_angle_quaternion((1.0, 2.0, -1.0), 0.71)
    q1 = _axis_angle_quaternion((-2.0, 1.0, 3.0), -0.93)
    quaternions = torch.stack((q0, q1))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_vectors = torch.tensor(
        [[1.2, -0.7, 0.9], [-1.2, 0.7, -0.9]], dtype=torch.float64
    )
    return quaternions, edge_index, edge_vectors


# ---------------------------------------------------------------------------
# Dimensionality tests
# ---------------------------------------------------------------------------


def test_t_body_irreps_dim():
    assert o3.Irreps(T_BODY_IRREPS).dim == 2 * 3 + 1  # l=3 → dim 7


def test_o_body_irreps_dim():
    assert o3.Irreps(O_BODY_IRREPS).dim == 2 * 4 + 1  # l=4 → dim 9


def test_i_body_irreps_dim():
    assert o3.Irreps(I_BODY_IRREPS).dim == 2 * 6 + 1  # l=6 → dim 13


@pytest.mark.parametrize(
    "cls,expected_dim",
    [(TBodyFeatures, 7), (OBodyFeatures, 9), (IBodyFeatures, 13)],
)
def test_cubic_body_features_output_shape(cls, expected_dim):
    model = cls().double()
    batch = 8
    q = torch.randn(batch, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    out = model(q)
    assert out.shape == (batch, expected_dim), (
        f"{cls.__name__}: expected ({batch}, {expected_dim}), got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("cls", [RigidPairTEdgeEmbedding, RigidPairOEdgeEmbedding, RigidPairIEdgeEmbedding])
def test_cubic_pair_output_irreps(cls):
    edge_irreps = o3.Irreps.spherical_harmonics(3)
    model = cls(lmax=3, edge_irreps=edge_irreps, multiplicity=1).double()
    assert model.irreps_out == edge_irreps


# ---------------------------------------------------------------------------
# Body-symmetry invariance
# ---------------------------------------------------------------------------


def test_t_pair_invariant_to_t_body_rotations():
    torch.manual_seed(7)
    model = RigidPairTEdgeEmbedding(lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()
    reference = model(quaternions, edge_index, edge_vectors)

    for g0 in _t_quaternions():
        for g1 in _t_quaternions():
            transformed = torch.stack(
                (
                    _quaternion_product(quaternions[0], g0),
                    _quaternion_product(quaternions[1], g1),
                )
            )
            actual = model(transformed, edge_index, edge_vectors)
            torch.testing.assert_close(actual, reference, atol=1.0e-9, rtol=1.0e-9,
                                       msg="T: not invariant under T rotation")


def test_o_pair_invariant_to_o_body_rotations():
    torch.manual_seed(7)
    model = RigidPairOEdgeEmbedding(lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()
    reference = model(quaternions, edge_index, edge_vectors)

    for g0 in _o_quaternions():
        for g1 in _o_quaternions():
            transformed = torch.stack(
                (
                    _quaternion_product(quaternions[0], g0),
                    _quaternion_product(quaternions[1], g1),
                )
            )
            actual = model(transformed, edge_index, edge_vectors)
            torch.testing.assert_close(actual, reference, atol=1.0e-9, rtol=1.0e-9,
                                       msg="O: not invariant under O rotation")


def test_i_pair_invariant_to_i_body_rotations():
    torch.manual_seed(7)
    model = RigidPairIEdgeEmbedding(lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()
    reference = model(quaternions, edge_index, edge_vectors)

    # Enumerate a subset: all 12 C5 elements + identity (full enumeration is expensive)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c5_axes = [
        (0.0, 1.0, phi), (0.0, 1.0, -phi),
        (1.0, phi, 0.0), (1.0, -phi, 0.0),
        (phi, 0.0, 1.0), (phi, 0.0, -1.0),
    ]
    subset = [_axis_angle_quaternion((0.0, 0.0, 1.0), 0.0)]
    for axis in c5_axes:
        for k in range(1, 5):
            subset.append(_axis_angle_quaternion(axis, k * 2.0 * math.pi / 5.0))

    for g0 in subset:
        transformed = torch.stack(
            (
                _quaternion_product(quaternions[0], g0),
                quaternions[1],
            )
        )
        actual = model(transformed, edge_index, edge_vectors)
        torch.testing.assert_close(actual, reference, atol=1.0e-9, rtol=1.0e-9,
                                   msg="I: not invariant under C5 rotation")


# ---------------------------------------------------------------------------
# Global SO(3) equivariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [RigidPairTEdgeEmbedding, RigidPairOEdgeEmbedding, RigidPairIEdgeEmbedding],
)
def test_cubic_pair_globally_equivariant(cls):
    torch.manual_seed(11)
    model = cls(lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()

    reference = model(quaternions, edge_index, edge_vectors)

    axis = (1.0, -2.0, 0.5)
    angle = 0.63
    q_global = _axis_angle_quaternion(axis, angle)
    rotation = _rotation_matrix(axis, angle)

    transformed_q = torch.stack(
        [_quaternion_product(q_global, q) for q in quaternions]
    )
    transformed_vectors = edge_vectors @ rotation.T

    actual = model(transformed_q, edge_index, transformed_vectors)
    D = model.irreps_out.D_from_matrix(rotation)
    expected = reference @ D.T

    torch.testing.assert_close(actual, expected, atol=2.0e-7, rtol=1.0e-6,
                                msg=f"{cls.__name__}: global equivariance failed")


# ---------------------------------------------------------------------------
# Sensitivity checks
# ---------------------------------------------------------------------------


def test_t_pair_sensitive_to_non_t_rotation():
    """A C4 rotation (in O but not T) should change the T output."""
    torch.manual_seed(13)
    model = RigidPairTEdgeEmbedding(lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()
    reference = model(quaternions, edge_index, edge_vectors)

    # C4 about z is in O but NOT in T
    g = _axis_angle_quaternion((0.0, 0.0, 1.0), math.pi / 2.0)
    transformed = quaternions.clone()
    transformed[0] = _quaternion_product(transformed[0], g)

    actual = model(transformed, edge_index, edge_vectors)
    delta = torch.linalg.vector_norm(actual - reference)
    assert delta.item() > 1.0e-6, (
        f"T: C4 rotation should change output (delta={delta.item():.2e})"
    )


def test_o_pair_sensitive_to_non_o_rotation():
    """A C5 rotation (in I but not O) should change the O output."""
    torch.manual_seed(13)
    model = RigidPairOEdgeEmbedding(lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()
    reference = model(quaternions, edge_index, edge_vectors)

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    g = _axis_angle_quaternion((0.0, 1.0, phi), 2.0 * math.pi / 5.0)
    transformed = quaternions.clone()
    transformed[0] = _quaternion_product(transformed[0], g)

    actual = model(transformed, edge_index, edge_vectors)
    delta = torch.linalg.vector_norm(actual - reference)
    assert delta.item() > 1.0e-6, (
        f"O: C5 rotation should change output (delta={delta.item():.2e})"
    )
