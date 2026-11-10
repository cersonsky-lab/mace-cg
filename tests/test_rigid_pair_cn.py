"""Tests for Cn body-symmetry edge embeddings.

Mirrors the structure of test_rigid_pair_d6.py:

  1. Output irreps / dimensionality.
  2. Invariance to right-multiplication by any Cn group element.
  3. Global SO(3) equivariance.
  4. Sensitivity to a non-Cn body rotation (sanity check).

Tests are parameterised over representative cyclic orders n = 3, 4, 5, 6.
"""
from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

from mace.modules.rigid_cn import CnBodyFeatures, cn_body_irreps
from mace.modules.rigid_pair_tp import RigidPairCnEdgeEmbedding


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-first convention: [w, x, y, z])
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
# Group element generators
# ---------------------------------------------------------------------------


def _cn_quaternions(n: int) -> list[torch.Tensor]:
    """Return the n quaternions representing Cn rotations about z."""
    return [
        _axis_angle_quaternion((0.0, 0.0, 1.0), k * 2.0 * math.pi / n)
        for k in range(n)
    ]


# ---------------------------------------------------------------------------
# Common test fixture
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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cn_body_features_output_shape(n):
    """forward() output has shape (batch, body_irreps.dim)."""
    model = CnBodyFeatures(n).double()
    batch = 8
    q = torch.randn(batch, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    out = model(q)
    assert out.shape == (batch, model.body_irreps.dim), (
        f"n={n}: expected ({batch}, {model.body_irreps.dim}), got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cn_body_irreps_string(n):
    """cn_body_irreps returns the correct irreps string."""
    irreps = cn_body_irreps(n)
    expected_dim = 3 + (2 * n + 1)  # 1o (dim=3) + l=n irrep (dim=2n+1)
    assert irreps.dim == expected_dim, (
        f"n={n}: expected dim={expected_dim}, got {irreps.dim} from {irreps}"
    )


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cn_pair_output_irreps(n):
    """RigidPairCnEdgeEmbedding.irreps_out matches edge_irreps."""
    edge_irreps = o3.Irreps.spherical_harmonics(3)
    model = RigidPairCnEdgeEmbedding(n=n, lmax=3, edge_irreps=edge_irreps, multiplicity=1).double()
    assert model.irreps_out == edge_irreps


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cn_pair_invariant_to_cn_body_rotations(n):
    """Output is unchanged under right-multiplication of body frames by any Cn element."""
    torch.manual_seed(7)
    model = RigidPairCnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()

    reference = model(quaternions, edge_index, edge_vectors)
    group = _cn_quaternions(n)

    for g0 in group:
        for g1 in group:
            transformed = torch.stack(
                (
                    _quaternion_product(quaternions[0], g0),
                    _quaternion_product(quaternions[1], g1),
                )
            )
            actual = model(transformed, edge_index, edge_vectors)
            torch.testing.assert_close(actual, reference, atol=1.0e-9, rtol=1.0e-9,
                                       msg=f"n={n}: not invariant under Cn rotation")


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cn_pair_globally_equivariant(n):
    """Left-multiplication (global SO(3) rotation) gives equivariant output."""
    torch.manual_seed(11)
    model = RigidPairCnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
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
                                msg=f"n={n}: global equivariance failed")


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cn_pair_sensitive_to_non_cn_rotation(n):
    """A body rotation by 2π/(2n) (half-step) DOES change the output."""
    torch.manual_seed(13)
    model = RigidPairCnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()

    reference = model(quaternions, edge_index, edge_vectors)

    # Rotate by half the fundamental angle — not in Cn for n > 1
    g = _axis_angle_quaternion((0.0, 0.0, 1.0), math.pi / n)
    transformed = quaternions.clone()
    transformed[0] = _quaternion_product(transformed[0], g)

    actual = model(transformed, edge_index, edge_vectors)
    delta = torch.linalg.vector_norm(actual - reference)
    assert delta.item() > 1.0e-6, (
        f"n={n}: output did not change under non-Cn rotation (delta={delta.item():.2e})"
    )


@pytest.mark.parametrize("n", [0, 1])
def test_cn_body_features_raises_for_n_lt_2(n):
    """n < 2 should raise ValueError."""
    with pytest.raises(ValueError, match="n must be >= 2"):
        CnBodyFeatures(n - 1)  # n-1 is < 2 only when n==2 → 1; and n==3 → 2 (valid)


def test_cn_body_features_raises_for_n_1():
    with pytest.raises(ValueError, match="n must be >= 2"):
        CnBodyFeatures(1)
