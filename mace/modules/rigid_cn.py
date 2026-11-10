"""Equivariant rigid-body features for Cn-symmetric molecules.

For a molecule with n-fold **proper rotation** symmetry about its body z-axis
the compact equivariant body features are two Wigner contractions:

  - l=1 template: body z-axis ``e_z`` → space-frame axis direction (``1o``, 3-dim)
  - l=n template: average of ``Y_n`` over n equispaced xy-plane vertices →
    space-frame n-fold texture (``ne``/``no``, 2n+1-dim)

The first captures *where the symmetry axis points* in the lab frame; the
second captures *how the molecule is rotated about that axis* at the
resolution the n-fold symmetry allows.

Body irreps: ``1x1o + 1xnX`` where ``nX`` is ``ne`` for even n, ``no`` for odd n.

Relationship to existing implementations
-----------------------------------------
* n=2: compatible with ``C2_BODY_IRREPS = "1o + 2e"`` in ``rigid_c2.py``.
  The l=2 template here is ``Y_2(e_x)`` (2-gon average collapses to one
  vertex because Y_2 is even); the existing C2 uses ``(Y_2(e_x) - Y_2(e_y)) /
  sqrt(2)``.  Both are valid C2-invariant templates; they emphasise different
  in-plane directions.  For backward-compat use ``c2_frame``.

Usage
-----
    from mace.modules.rigid_cn import CnBodyFeatures, cn_body_irreps
    n = 3  # C3 symmetry (e.g. ammonia)
    feats = CnBodyFeatures(n)
    out = feats(quaternions)       # shape (..., cn_body_irreps(n).dim)
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


def cn_body_irreps(n: int) -> o3.Irreps:
    """Return the Wigner-D irreps for Cn body features.

    Parameters
    ----------
    n:
        Order of the cyclic group (n >= 2).

    Returns
    -------
    o3.Irreps
        ``1x1o + 1xnX`` where X is 'e' for even n, 'o' for odd n.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for Cn symmetry, got {n}")
    return o3.Irreps(f"1x1o + 1x{n}{_parity_label(n)}")


def _cn_body_templates(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the two fixed body-frame templates for Cn.

    Parameters
    ----------
    n:
        Cyclic order.
    dtype, device:
        Output tensor spec.

    Returns
    -------
    l1_template : (3,)
        Body z-axis ``e_z`` in the l=1 (``1o``) space.
    ln_template : (2n+1,)
        Average of ``Y_n`` over n equispaced xy-plane vertices in the l=n space.
    """
    # l=1: Y_1(e_z) in e3nn's m=-1,0,+1 basis.
    # In e3nn's real SH convention, the m=0 component IS the z-axis (not [0,0,1],
    # which is m=+1 / x-axis).  o3.spherical_harmonics gives [0,1,0] for input [0,0,1].
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    l1_template = o3.spherical_harmonics(
        1,
        normal,
        normalize=True,
        normalization="component",
    )

    # l=n: average Y_n over n equispaced vertices in the xy-plane
    # Rotating the body by 2π/n just cyclically permutes the vertices, so
    # their average is exactly Cn-invariant.
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

    return l1_template, ln_template


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def cn_body_irreducible_features(
    rotation_matrices: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Return compact Cn-invariant, globally equivariant body features.

    Parameters
    ----------
    rotation_matrices : (..., 3, 3)
        Body-to-space rotation matrices.
    n : int
        Cyclic order (n >= 2).

    Returns
    -------
    torch.Tensor
        ``(..., cn_body_irreps(n).dim)`` equivariant features.
    """
    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrices must have shape (..., 3, 3); "
            f"got {tuple(rotation_matrices.shape)}"
        )

    l1_t, ln_t = _cn_body_templates(
        n,
        dtype=rotation_matrices.dtype,
        device=rotation_matrices.device,
    )

    axis = wigner_body_template(rotation_matrices, ell=1, body_template=l1_t)
    nfold = wigner_body_template(rotation_matrices, ell=n, body_template=ln_t)

    return torch.cat([axis, nfold], dim=-1)


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------


class CnBodyFeatures(torch.nn.Module):
    """Cn rigid-body orientation features from scalar-first quaternions.

    For a molecule with n-fold proper rotation symmetry about its body z-axis,
    two body-frame templates are projected through the Wigner-D matrices:

        axis    = D1(R) · e_z              → 1o (3-dim)
        n-fold  = Dn(R) · <n-gon avg>     → nX (2n+1-dim)

    Templates are precomputed at double precision in ``__init__`` and stored
    as buffers so forward only does the Wigner contractions.

    Parameters
    ----------
    n : int
        Cyclic order (n >= 2).  Use ``c1_frame`` for C1; use ``c2_frame``
        for C2 to match the existing checkpoint layout.
    """

    __constants__ = ["n"]

    def __init__(self, n: int) -> None:
        super().__init__()

        if n < 2:
            raise ValueError(f"n must be >= 2 for Cn symmetry, got {n}")

        self.n = int(n)
        self.body_irreps = cn_body_irreps(self.n)

        # Precompute at float64, store as float32 buffers.
        l1_t, ln_t = _cn_body_templates(
            self.n,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.register_buffer("_l1_template", l1_t)
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

        l1_t = self._l1_template.to(dtype=rotations.dtype, device=rotations.device)
        ln_t = self._ln_template.to(dtype=rotations.dtype, device=rotations.device)

        axis = wigner_body_template(rotations, ell=1, body_template=l1_t)
        nfold = wigner_body_template(rotations, ell=self.n, body_template=ln_t)

        return torch.cat([axis, nfold], dim=-1)
