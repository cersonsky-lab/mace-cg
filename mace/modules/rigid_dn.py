"""Equivariant rigid-body features for Dn-symmetric molecules.

For a molecule with Dn symmetry (n-fold axis **plus** n perpendicular C2 axes
or in-plane mirrors), a perpendicular C2 flips the body z-axis (z → −z), so
the polar l=1 axis vector is **not** Dn-invariant.  Instead two Wigner
contractions that ARE invariant are used:

  - l=2 template: ``Y_2(e_z)`` axial normal (``2e``, 5-dim) — invariant because
    l=2 is even under z → −z
  - l=n template: average of ``Y_n`` over n equispaced xy-plane vertices
    (``ne``/``no``, 2n+1-dim) — Dn-invariant by vertex-permutation argument

Body irreps: ``1x2e + 1xnX`` where ``nX`` is ``ne`` for even n, ``no`` for odd n.

This family includes the existing D6 implementation as the n=6 special case
(``D6_BODY_IRREPS = "1x2e + 1x6e"``), with identical template choices, so
checkpoints trained with ``d6_frame`` are equivalent to this code at n=6.

Cn vs Dn
---------
Use **Cn** when the molecule has a polar axis (a "head" and a "tail", e.g.
ammonia NH3 — C3v proper rotations only).  Use **Dn** when top and bottom are
equivalent (e.g. BF3 — D3h proper subgroup D3).

Usage
-----
    from mace.modules.rigid_dn import DnBodyFeatures, dn_body_irreps
    n = 3  # D3 symmetry (e.g. BF3, D3h)
    feats = DnBodyFeatures(n)
    out = feats(quaternions)       # shape (..., dn_body_irreps(n).dim)
"""

from __future__ import annotations

import math

import torch
from e3nn import o3

from mace.data.rigid_body import quaternion_to_matrix
from mace.modules.rigid_wigner import wigner_body_template


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parity_label(ell: int) -> str:
    return "e" if ell % 2 == 0 else "o"


def dn_body_irreps(n: int) -> o3.Irreps:
    """Return the Wigner-D irreps for Dn body features.

    Parameters
    ----------
    n:
        Dihedral order (n >= 2).

    Returns
    -------
    o3.Irreps
        ``1x2e + 1xnX`` where X is 'e' for even n, 'o' for odd n.
        At n=6 this gives ``"1x2e + 1x6e"``, matching ``D6_BODY_IRREPS``.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for Dn symmetry, got {n}")
    return o3.Irreps(f"1x2e + 1x{n}{_parity_label(n)}")


def _dn_body_templates(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the two fixed body-frame templates for Dn.

    Parameters
    ----------
    n:
        Dihedral order.
    dtype, device:
        Output tensor spec.

    Returns
    -------
    l2_template : (5,)
        ``Y_2(e_z)`` in the l=2 (``2e``) space — axial normal of the symmetry plane.
    ln_template : (2n+1,)
        Average of ``Y_n`` over n equispaced xy-plane vertices in the l=n space.
    """
    # l=2: Y_2(e_z) for the axial / plane-normal direction.
    # Under a perpendicular C2 (which sends z → −z): Y_2 is even so this is
    # invariant.  Under the n-fold axis rotation: Y_2(e_z) is invariant since
    # e_z is on the axis.
    normal = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=dtype,
        device=device,
    )
    l2_template = o3.spherical_harmonics(
        2,
        normal,
        normalize=True,
        normalization="component",
    )  # (5,)

    # l=n: average Y_n over n equispaced xy-plane vertices.
    angles = torch.arange(n, dtype=dtype, device=device) * (2.0 * math.pi / float(n))
    vertices = torch.stack(
        [
            torch.cos(angles),
            torch.sin(angles),
            torch.zeros(n, dtype=dtype, device=device),
        ],
        dim=-1,
    )  # (n, 3)

    ln_template = o3.spherical_harmonics(
        n,
        vertices,
        normalize=True,
        normalization="component",
    ).mean(dim=0)  # (2n+1,)

    return l2_template, ln_template


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def dn_body_irreducible_features(
    rotation_matrices: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Return compact Dn-invariant, globally equivariant body features.

    Parameters
    ----------
    rotation_matrices : (..., 3, 3)
        Body-to-space rotation matrices.
    n : int
        Dihedral order (n >= 2).

    Returns
    -------
    torch.Tensor
        ``(..., dn_body_irreps(n).dim)`` equivariant features.
    """
    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrices must have shape (..., 3, 3); "
            f"got {tuple(rotation_matrices.shape)}"
        )

    l2_t, ln_t = _dn_body_templates(
        n,
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )

    plane = wigner_body_template(rotation_matrices, ell=2, body_template=l2_t)
    nfold = wigner_body_template(rotation_matrices, ell=n, body_template=ln_t)

    return torch.cat([plane, nfold], dim=-1)


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------


class DnBodyFeatures(torch.nn.Module):
    """Dn rigid-body orientation features from scalar-first quaternions.

    For a molecule with Dn symmetry (n-fold axis + perpendicular C2s / mirrors),
    two body-frame templates are projected through the Wigner-D matrices:

        plane   = D2(R) · Y_2(e_z)         → 2e (5-dim)
        n-fold  = Dn(R) · <n-gon avg>      → nX (2n+1-dim)

    Templates are precomputed at double precision in ``__init__`` and stored
    as buffers so forward only does the Wigner contractions.

    Parameters
    ----------
    n : int
        Dihedral order (n >= 2).  n=6 reproduces the existing ``d6_frame``
        checkpoint layout exactly.
    """

    __constants__ = ["n"]

    def __init__(self, n: int) -> None:
        super().__init__()

        if n < 2:
            raise ValueError(f"n must be >= 2 for Dn symmetry, got {n}")

        self.n = int(n)
        self.body_irreps = dn_body_irreps(self.n)

        # Precompute at float64, store as float32 buffers.
        l2_t, ln_t = _dn_body_templates(
            self.n,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.register_buffer("_l2_template", l2_t)
        self.register_buffer("_ln_template", ln_t)

    def forward(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        quaternions : (N, 4)
            Scalar-first ``[w, x, y, z]`` unit quaternions.

        Returns
        -------
        torch.Tensor
            ``(N, body_irreps.dim)`` equivariant features.
        """
        rotations = quaternion_to_matrix(quaternions)

        l2_t = self._l2_template.to(dtype=rotations.dtype, device=rotations.device)
        ln_t = self._ln_template.to(dtype=rotations.dtype, device=rotations.device)

        plane = wigner_body_template(rotations, ell=2, body_template=l2_t)
        nfold = wigner_body_template(rotations, ell=self.n, body_template=ln_t)

        return torch.cat([plane, nfold], dim=-1)
