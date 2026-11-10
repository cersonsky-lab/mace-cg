"""Tests for Dn body-symmetry edge embeddings.

Mirrors test_rigid_pair_d6.py but parameterised over n = 3, 4, 6.

The Dn group has 2n elements:
  - n axial C_n rotations about z  (by k * 2π/n)
  - n C_2 flips about in-plane axes (rotation by π about cos(kπ/n), sin(kπ/n), 0)

Properties tested:
  1. Output irreps / dimensionality.
  2. Invariance to right-multiplication by any Dn element.
  3. Global SO(3) equivariance.
  4. Sensitivity to a non-Dn body rotation.
  5. That D6 (n=6) produces the same body_irreps.dim as the legacy D6 implementation.
"""
from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

from mace.modules.rigid_dn import DnBodyFeatures, dn_body_irreps
from mace.modules.rigid_pair_tp import RigidPairDnEdgeEmbedding


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
# Group element generators
# ---------------------------------------------------------------------------


def _dn_quaternions(n: int) -> list[torch.Tensor]:
    """Return the 2n quaternions representing all Dn group elements."""
    result = []
    # n axial rotations
    for k in range(n):
        result.append(_axis_angle_quaternion((0.0, 0.0, 1.0), k * 2.0 * math.pi / n))
    # n in-plane C2 flips
    for k in range(n):
        phi = k * math.pi / n
        result.append(
            _axis_angle_quaternion((math.cos(phi), math.sin(phi), 0.0), math.pi)
        )
    return result


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


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_body_features_output_shape(n):
    model = DnBodyFeatures(n).double()
    batch = 8
    q = torch.randn(batch, 4, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    out = model(q)
    assert out.shape == (batch, model.body_irreps.dim), (
        f"n={n}: expected ({batch}, {model.body_irreps.dim}), got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_body_irreps_string(n):
    irreps = dn_body_irreps(n)
    expected_dim = 5 + (2 * n + 1)  # 2e (dim=5) + l=n irrep (dim=2n+1)
    assert irreps.dim == expected_dim, (
        f"n={n}: expected dim={expected_dim}, got {irreps.dim} from {irreps}"
    )


def test_d6_body_irreps_matches_legacy():
    """DnBodyFeatures(6).body_irreps.dim == RigidPairD6EdgeEmbedding body dim."""
    from mace.modules.rigid_d6 import D6_BODY_IRREPS  # noqa: PLC0415
    dn6_irreps = dn_body_irreps(6)
    assert dn6_irreps.dim == o3.Irreps(D6_BODY_IRREPS).dim, (
        f"D6 body dim mismatch: new={dn6_irreps.dim}, legacy={o3.Irreps(D6_BODY_IRREPS).dim}"
    )


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_pair_output_irreps(n):
    edge_irreps = o3.Irreps.spherical_harmonics(3)
    model = RigidPairDnEdgeEmbedding(n=n, lmax=3, edge_irreps=edge_irreps, multiplicity=1).double()
    assert model.irreps_out == edge_irreps


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_pair_invariant_to_dn_body_rotations(n):
    """Output is unchanged when each body frame is right-multiplied by any Dn element."""
    torch.manual_seed(7)
    model = RigidPairDnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()

    reference = model(quaternions, edge_index, edge_vectors)
    group = _dn_quaternions(n)

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
                                       msg=f"n={n}: not invariant under Dn rotation")


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_pair_globally_equivariant(n):
    """Left-multiplication (global SO(3) rotation) gives equivariant output."""
    torch.manual_seed(11)
    model = RigidPairDnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
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


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_pair_sensitive_to_non_dn_axial_rotation(n):
    """A body rotation by π/n about z (half-step) DOES change the output."""
    torch.manual_seed(13)
    model = RigidPairDnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()

    reference = model(quaternions, edge_index, edge_vectors)
    g = _axis_angle_quaternion((0.0, 0.0, 1.0), math.pi / n)
    transformed = quaternions.clone()
    transformed[0] = _quaternion_product(transformed[0], g)

    actual = model(transformed, edge_index, edge_vectors)
    delta = torch.linalg.vector_norm(actual - reference)
    assert delta.item() > 1.0e-6, (
        f"n={n}: output unchanged under non-Dn rotation (delta={delta.item():.2e})"
    )


@pytest.mark.parametrize("n", [3, 4, 6])
def test_dn_pair_sensitive_to_z_flip_on_cn_only_frame(n):
    """Cn body embedding IS sensitive to z→-z; Dn is not (key Cn/Dn distinction)."""
    from mace.modules.rigid_pair_tp import RigidPairCnEdgeEmbedding  # noqa: PLC0415

    torch.manual_seed(17)
    cn_model = RigidPairCnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
    dn_model = RigidPairDnEdgeEmbedding(n=n, lmax=3, multiplicity=1).double()
    quaternions, edge_index, edge_vectors = _example()

    # z-flip = π rotation about x-axis
    g_zflip = _axis_angle_quaternion((1.0, 0.0, 0.0), math.pi)

    transformed = quaternions.clone()
    transformed[0] = _quaternion_product(transformed[0], g_zflip)

    # Cn: z-flip IS NOT in Cn → output should change
    cn_ref = cn_model(quaternions, edge_index, edge_vectors)
    cn_flipped = cn_model(transformed, edge_index, edge_vectors)
    cn_delta = torch.linalg.vector_norm(cn_flipped - cn_ref)
    assert cn_delta.item() > 1.0e-6, (
        f"Cn(n={n}): z-flip should change output (delta={cn_delta.item():.2e})"
    )

    # Dn: z-flip IS in Dn → output should be unchanged
    dn_ref = dn_model(quaternions, edge_index, edge_vectors)
    dn_flipped = dn_model(transformed, edge_index, edge_vectors)
    torch.testing.assert_close(dn_flipped, dn_ref, atol=1.0e-9, rtol=1.0e-9,
                                msg=f"Dn(n={n}): z-flip should NOT change output")


def test_dn_body_features_raises_for_n_1():
    with pytest.raises(ValueError, match="n must be >= 2"):
        DnBodyFeatures(1)
