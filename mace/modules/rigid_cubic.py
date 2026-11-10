"""Equivariant rigid-body features for cubic and icosahedral symmetry groups.

Three chiral (proper-rotation) groups are supported.  For each the body
features are the average of ``Y_l`` over the group's vertex orbit, projected
through ``D^l(R)``.  The vertex average is exactly group-invariant by
construction (rotating the body permutes the vertices, leaving the average
unchanged).

Group    Vertices  Invariant l  Body irreps  Dim
-------  --------  -----------  -----------  ---
T  (12)  4 tet.    l = 3        1x3o          7
O  (24)  6 oct.    l = 4        1x4e          9
I  (60)  12 icos.  l = 6        1x6e         13

These are the **chiral** (proper-rotation) groups.  If your molecule has
improper symmetry (Td, Oh, Ih), use the chiral subgroup (T, O, I) since the
body orientation is a proper rotation in SO(3).

Minimum l for full orientation parameterisation
------------------------------------------------
A single l template captures only a 1-D projection of the (2l+1)-D Wigner
matrix column.  For T (12 ops) the fundamental domain has dimension 3 and a
single 7-D template at l=3 may suffice in practice; adding l=4 (9-D) would
give a redundant but more robust description.  For O and I similar reasoning
applies.  These single-template choices follow the same compact philosophy as
the existing C2 / D6 implementations — they are purposely minimal.

Usage
-----
    from mace.modules.rigid_cubic import TBodyFeatures, OBodyFeatures, IBodyFeatures
    t_feats = TBodyFeatures()
    out = t_feats(quaternions)   # shape (N, 7)
"""

from __future__ import annotations

import math

import torch
from e3nn import o3

from mace.data.rigid_body import quaternion_to_matrix
from mace.modules.rigid_wigner import wigner_body_template

T_BODY_IRREPS = o3.Irreps("1x3o")   # 7-dim
O_BODY_IRREPS = o3.Irreps("1x4e")   # 9-dim
I_BODY_IRREPS = o3.Irreps("1x6e")   # 13-dim


# ---------------------------------------------------------------------------
# Vertex sets
# ---------------------------------------------------------------------------


def _tetrahedral_vertices(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Four unit-length vertices of a regular tetrahedron inscribed in the cube."""
    s = 1.0 / math.sqrt(3.0)
    return torch.tensor(
        [
            [ s,  s,  s],
            [ s, -s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
        ],
        dtype=dtype,
        device=device,
    )


def _octahedral_vertices(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Six unit-length vertices of a regular octahedron (±x, ±y, ±z axes)."""
    return torch.tensor(
        [
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0, -1.0],
        ],
        dtype=dtype,
        device=device,
    )


def _icosahedral_vertices(
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Twelve unit-length vertices of a regular icosahedron.

    The raw coordinates use the golden-ratio construction
    (0, ±1, ±φ) and cyclic permutations, normalised to the unit sphere.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = torch.tensor(
        [
            [ 0.0,  1.0,  phi],
            [ 0.0, -1.0,  phi],
            [ 0.0,  1.0, -phi],
            [ 0.0, -1.0, -phi],
            [ 1.0,  phi,  0.0],
            [-1.0,  phi,  0.0],
            [ 1.0, -phi,  0.0],
            [-1.0, -phi,  0.0],
            [ phi,  0.0,  1.0],
            [-phi,  0.0,  1.0],
            [ phi,  0.0, -1.0],
            [-phi,  0.0, -1.0],
        ],
        dtype=dtype,
        device=device,
    )
    return raw / raw.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Template helper
# ---------------------------------------------------------------------------


def _vertex_average_template(
    vertices: torch.Tensor,
    ell: int,
) -> torch.Tensor:
    """Return the average of ``Y_ell`` over a set of group-equivalent vertices.

    Parameters
    ----------
    vertices : (K, 3)
        Unit-length direction vectors forming a group orbit.
    ell : int
        Spherical-harmonic order.

    Returns
    -------
    torch.Tensor
        ``(2*ell+1,)`` template vector.  Non-zero because the group orbit
        is a subset of the ``Y_ell``-invariant subspace.
    """
    return o3.spherical_harmonics(
        ell,
        vertices,
        normalize=True,
        normalization="component",
    ).mean(dim=0)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def t_body_irreducible_features(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Return T-invariant l=3 body features (``3o``, 7-dim)."""
    verts = _tetrahedral_vertices(
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )
    template = _vertex_average_template(verts, ell=3)
    return wigner_body_template(rotation_matrices, ell=3, body_template=template)


def o_body_irreducible_features(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Return O-invariant l=4 body features (``4e``, 9-dim)."""
    verts = _octahedral_vertices(
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )
    template = _vertex_average_template(verts, ell=4)
    return wigner_body_template(rotation_matrices, ell=4, body_template=template)


def i_body_irreducible_features(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Return I-invariant l=6 body features (``6e``, 13-dim)."""
    verts = _icosahedral_vertices(
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )
    template = _vertex_average_template(verts, ell=6)
    return wigner_body_template(rotation_matrices, ell=6, body_template=template)


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------


class TBodyFeatures(torch.nn.Module):
    """T (chiral tetrahedral) rigid-body orientation features.

    Projects the l=3 Wigner matrix onto the unique T-invariant direction
    obtained by averaging ``Y_3`` over the four tetrahedral vertices.

    Output: ``1x3o`` (7-dim).
    """

    __constants__ = ["_ell"]

    def __init__(self) -> None:
        super().__init__()
        self._ell: int = 3
        self.body_irreps = T_BODY_IRREPS

        verts = _tetrahedral_vertices(
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        template = _vertex_average_template(verts, ell=3)
        self.register_buffer("_template", template)

    def forward(self, quaternions: torch.Tensor) -> torch.Tensor:
        rotations = quaternion_to_matrix(quaternions)
        t = self._template.to(dtype=rotations.dtype, device=rotations.device)
        return wigner_body_template(rotations, ell=self._ell, body_template=t)


class OBodyFeatures(torch.nn.Module):
    """O (chiral octahedral) rigid-body orientation features.

    Projects the l=4 Wigner matrix onto the unique O-invariant direction
    obtained by averaging ``Y_4`` over the six octahedral vertices.

    Output: ``1x4e`` (9-dim).
    """

    __constants__ = ["_ell"]

    def __init__(self) -> None:
        super().__init__()
        self._ell: int = 4
        self.body_irreps = O_BODY_IRREPS

        verts = _octahedral_vertices(
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        template = _vertex_average_template(verts, ell=4)
        self.register_buffer("_template", template)

    def forward(self, quaternions: torch.Tensor) -> torch.Tensor:
        rotations = quaternion_to_matrix(quaternions)
        t = self._template.to(dtype=rotations.dtype, device=rotations.device)
        return wigner_body_template(rotations, ell=self._ell, body_template=t)


class IBodyFeatures(torch.nn.Module):
    """I (chiral icosahedral) rigid-body orientation features.

    Projects the l=6 Wigner matrix onto the unique I-invariant direction
    obtained by averaging ``Y_6`` over the twelve icosahedral vertices.

    Output: ``1x6e`` (13-dim).
    """

    __constants__ = ["_ell"]

    def __init__(self) -> None:
        super().__init__()
        self._ell: int = 6
        self.body_irreps = I_BODY_IRREPS

        verts = _icosahedral_vertices(
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        template = _vertex_average_template(verts, ell=6)
        self.register_buffer("_template", template)

    def forward(self, quaternions: torch.Tensor) -> torch.Tensor:
        rotations = quaternion_to_matrix(quaternions)
        t = self._template.to(dtype=rotations.dtype, device=rotations.device)
        return wigner_body_template(rotations, ell=self._ell, body_template=t)
