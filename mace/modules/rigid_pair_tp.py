"""Equivariant tensor-product features for rigid-body pairs.

A physical orientation is one body-to-space rotation R in SO(3). Its absolute
equivariant features are functions of the Wigner matrices D^ell(R).

The generic full-frame path uses the complete D1(R), stored by e3nn as 3x1o:
the body/Wigner index is multiplicity and the space index is the l=1 irrep.

Those three multiplicity copies are constrained columns of one common
rotation matrix, not independent physical vectors. C1, C2, D6, Cn, Dn, T, O
and I features are likewise derived from one common R.

Supported rigid_pair_mode strings
-----------------------------------
Non-frame modes (no orientation used):
    none, invariant_radial

Full-frame modes (complete SO(3) orientation):
    full_frame, full_frame_compact, full_frame_irrep_complete, full_frame_raw

Body-symmetry modes — compact Wigner projections:
    c1_frame        C1 (no body symmetry), 21-dim
    c2_frame        C2 (legacy), 8-dim
    c{n}_frame      Cn cyclic, n >= 2  (e.g. c3_frame → 3+7=10-dim)
    d6_frame        D6 (legacy), 18-dim
    d{n}_frame      Dn dihedral, n >= 2  (e.g. d3_frame → 5+7=12-dim)
    t_frame         T chiral tetrahedral, 7-dim
    o_frame         O chiral octahedral, 9-dim
    i_frame         I chiral icosahedral, 13-dim
"""

from __future__ import annotations

import math
import re
from functools import partial

import torch
from e3nn import o3
from e3nn.o3._spherical_harmonics import _spherical_harmonics

from mace.data.rigid_body import quaternion_to_matrix
from mace.modules.rigid_c1 import C1_BODY_IRREPS, C1BodyFeatures
from mace.modules.rigid_c2 import C2_BODY_IRREPS, c2_body_irreducible_features
from mace.modules.rigid_cn import CnBodyFeatures, cn_body_irreps
from mace.modules.rigid_cubic import (
    IBodyFeatures,
    I_BODY_IRREPS,
    OBodyFeatures,
    O_BODY_IRREPS,
    TBodyFeatures,
    T_BODY_IRREPS,
)
from mace.modules.rigid_d6 import D6_BODY_IRREPS, d6_body_features
from mace.modules.rigid_dn import DnBodyFeatures, dn_body_irreps
from mace.modules.rigid_wigner import full_wigner_features, full_wigner_irreps

# ---------------------------------------------------------------------------
# Compiled regex patterns for parameterised modes
# ---------------------------------------------------------------------------

_CN_RE = re.compile(r"^c(\d+)_frame$")
_DN_RE = re.compile(r"^d(\d+)_frame$")


class RigidPairPoseMLPEdgeEmbedding(torch.nn.Module):
    """Global-SO(3)-invariant rigid-pair features via an MLP over pose descriptors.

    Invariant descriptors (all invariant under simultaneous global SO(3) rotation):
      - Flattened relative rotation matrix R_i^T R_j  (9 values)
      - Edge vector in frame i: R_i^T r_{ij}          (3 values)
      - Gaussian radial basis over |r_{ij}|            (num_radial values)

    Parameters
    ----------
    r_max : float
        Cutoff radius — sets the centre range of the Gaussian RBF.
    num_radial : int
        Number of Gaussian radial basis functions.
    hidden_channels : int
        Width of the hidden MLP layers.
    out_channels : int
        Number of output scalar features (``edge_irreps = out_channels x 0e``).
    """

    def __init__(
        self,
        r_max: float = 8.0,
        num_radial: int = 16,
        hidden_channels: int = 64,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.r_max = r_max
        self.out_channels = out_channels
        self.edge_irreps = o3.Irreps(f"{out_channels}x0e")
        self.irreps_out = self.edge_irreps

        centres = torch.linspace(0.0, float(r_max), num_radial)
        self.register_buffer("_rbf_centres", centres)
        self._rbf_width: float = float(r_max) / num_radial

        in_dim = 9 + 3 + num_radial
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index

        R_i = quaternion_to_matrix(quaternions[dst])   # (E, 3, 3)
        R_j = quaternion_to_matrix(quaternions[src])

        # Relative rotation: R_i^T @ R_j  (invariant to global SO3)
        R_rel = torch.einsum("eij,ejk->eik", R_i.transpose(-2, -1), R_j)
        R_rel_flat = R_rel.reshape(R_rel.shape[0], 9)  # (E, 9)

        # Edge vector expressed in frame i  (invariant to global SO3)
        r_i = torch.einsum("eij,ej->ei", R_i.transpose(-2, -1), edge_vectors)  # (E, 3)

        # Gaussian radial basis of |r|
        d = edge_vectors.norm(dim=-1, keepdim=True)  # (E, 1)
        centres = self._rbf_centres.to(dtype=d.dtype, device=d.device)
        rbf = torch.exp(-0.5 * ((d - centres) / self._rbf_width) ** 2)  # (E, num_radial)

        x = torch.cat([R_rel_flat, r_i, rbf], dim=-1)
        return self.mlp(x.to(dtype=self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# Mode parsing helper (Python-only; used in MACE.__init__, not in forward)
# ---------------------------------------------------------------------------


def parse_rigid_pair_mode(mode: str) -> tuple[str, int]:
    """Parse a parameterised rigid-pair mode string.

    Returns ``(kind, n)`` where ``kind`` is one of ``"cn"``, ``"dn"``,
    ``"t"``, ``"o"``, ``"i"``, or the original ``mode`` string for
    non-parameterised modes, and ``n`` is the integer order (0 for
    non-parameterised groups).

    Examples
    --------
    >>> parse_rigid_pair_mode("c3_frame")
    ("cn", 3)
    >>> parse_rigid_pair_mode("d4_frame")
    ("dn", 4)
    >>> parse_rigid_pair_mode("t_frame")
    ("t", 0)
    >>> parse_rigid_pair_mode("d6_frame")
    ("d6_frame", 0)   # legacy, handled by its own branch
    """
    m = _CN_RE.match(mode)
    if m:
        return "cn", int(m.group(1))
    m = _DN_RE.match(mode)
    if m:
        return "dn", int(m.group(1))
    if mode == "t_frame":
        return "t", 0
    if mode == "o_frame":
        return "o", 0
    if mode == "i_frame":
        return "i", 0
    return mode, 0


class RigidPairTensorProductFeatures(torch.nn.Module):
    """Full equivariant tensor-product basis for an ordered rigid pair."""

    def __init__(self, lmax: int = 2):
        super().__init__()

        if lmax < 0:
            raise ValueError("lmax must be >= 0")

        self.lmax = lmax

        self.edge_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.frame_irreps = full_wigner_irreps((1,))

        # First couple positional angular information to the center frame.
        self.edge_center_tp = o3.FullTensorProduct(
            self.edge_irreps,
            self.frame_irreps,
        )

        # Then couple the neighbor frame.
        self.pair_tp = o3.FullTensorProduct(
            self.edge_center_tp.irreps_out,
            self.frame_irreps,
        )

        self.irreps_out = self.pair_tp.irreps_out

    @staticmethod
    def _frame_features(
        rotation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Return the complete D1(R) representation of one orientation."""

        return full_wigner_features(
            rotation_matrices,
            ells=[
                1,
            ],
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Construct equivariant rigid-pair features.

        Parameters
        ----------
        quaternions
            ``(num_nodes, 4)`` scalar-first ``[w, x, y, z]`` quaternions.
        edge_index
            ``(2, num_edges)``.
        edge_vectors
            ``(num_edges, 3)``.

        Returns
        -------
        torch.Tensor
            ``(num_edges, irreps_out.dim)`` equivariant pair features.
        """
        if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
            raise ValueError("quaternions must have shape (num_nodes, 4)")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
            raise ValueError("edge_vectors must have shape (num_edges, 3)")

        if edge_vectors.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_vectors and edge_index must contain the same number of edges"
            )

        distances = torch.linalg.vector_norm(
            edge_vectors,
            dim=-1,
        )

        if torch.any(distances <= 1.0e-12):
            raise ValueError(
                "RigidPairTensorProductFeatures does not support zero-length edges"
            )

        directions = edge_vectors / distances.unsqueeze(-1)

        edge_features = o3.spherical_harmonics(
            self.edge_irreps,
            directions,
            normalize=True,
            normalization="component",
        )

        rotations = quaternion_to_matrix(quaternions)

        frame_features = self._frame_features(rotations)

        i = edge_index[0]
        j = edge_index[1]

        frame_i = frame_features[i]
        frame_j = frame_features[j]

        edge_center = self.edge_center_tp(
            edge_features,
            frame_i,
        )

        return self.pair_tp(
            edge_center,
            frame_j,
        )


class RigidPairEdgeEmbedding(torch.nn.Module):
    """Learned compressed rigid-pair edge representation.

    The complete pair tensor product is projected to ``multiplicity``
    learned copies of each ordinary edge spherical-harmonic irrep.

    multiplicity=1 preserves the original projected full-frame model.

    The projected rigid block is scaled by 1/sqrt(multiplicity), keeping
    its aggregate norm approximately comparable as multiplicity grows.
    """

    def __init__(
        self,
        lmax: int,
        edge_irreps: o3.Irreps,
        multiplicity: int = 1,
    ):
        super().__init__()

        if isinstance(multiplicity, bool) or multiplicity < 1:
            raise ValueError(
                "rigid pair multiplicity must be a positive integer, "
                f"got {multiplicity!r}"
            )

        self.multiplicity = int(multiplicity)

        self.full_pair = RigidPairTensorProductFeatures(
            lmax=lmax,
        )

        self.base_edge_irreps = o3.Irreps(edge_irreps)

        # Increase multiplicity without changing which irreps appear.
        #
        # Example:
        #
        #   0e + 1o + 2e + 3o
        #
        # becomes, for multiplicity=4,
        #
        #   4x0e + 4x1o + 4x2e + 4x3o
        #
        self.edge_irreps = o3.Irreps(
            [(mul * self.multiplicity, ir) for mul, ir in self.base_edge_irreps]
        )

        # Do not perturb initialization of the surrounding MACE model.
        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.full_pair.irreps_out,
                self.edge_irreps,
            )

        self.output_scale = 1.0 / math.sqrt(self.multiplicity)

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        full_pair = self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )

        projected = self.projection(full_pair)

        return projected * self.output_scale


class RigidPairIrrepCompleteEdgeEmbedding(torch.nn.Module):
    """Compact projection retaining every raw rigid-pair irrep type.

    Unlike ``RigidPairEdgeEmbedding``, which projects only onto the
    ordinary edge spherical-harmonic irreps, this module retains one
    learned copy of every distinct (L, parity) irrep occurring in the
    complete

        Y_l(r_hat) x frame_i x frame_j

    tensor product.

    Multiplicity inside the raw tensor product is compressed to one
    learned copy per irrep type, but no angular/parity sector is
    discarded.
    """

    def __init__(self, lmax: int):
        super().__init__()

        self.full_pair = RigidPairTensorProductFeatures(
            lmax=lmax,
        )

        by_type = {}

        for _, ir in self.full_pair.irreps_out:
            by_type[(ir.l, ir.p)] = ir

        # Deterministic order:
        #   even parity, increasing L
        #   odd parity, increasing L
        keys = sorted(
            by_type,
            key=lambda key: (
                0 if key[1] == 1 else 1,
                key[0],
            ),
        )

        self.edge_irreps = o3.Irreps([(1, by_type[key]) for key in keys])

        # Do not perturb initialization of the surrounding MACE model.
        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.full_pair.irreps_out,
                self.edge_irreps,
            )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        full_pair = self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )

        return self.projection(full_pair)


class RigidPairRawEdgeEmbedding(torch.nn.Module):
    """Uncompressed rigid-pair tensor-product edge representation.

    This diagnostic module performs no learned projection back to the
    ordinary spherical-harmonic irreps. Its output is exactly the full

        Y_l(r_hat) x frame_i x frame_j

    tensor-product representation.
    """

    def __init__(self, lmax: int):
        super().__init__()

        self.full_pair = RigidPairTensorProductFeatures(lmax=lmax)
        self.edge_irreps = self.full_pair.irreps_out

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )


def validate_rigid_pair_mode(mode: str) -> str:
    """Validate *and* return the rigid-pair edge representation mode.

    Accepted modes
    --------------
    Non-frame modes (no orientation used):
        none, invariant_radial

    Full-frame modes (complete SO(3) orientation):
        full_frame, full_frame_compact, full_frame_irrep_complete, full_frame_raw

    Body-symmetry modes — compact Wigner projections:
        c1_frame        C1 (no body symmetry), 21-dim
        c2_frame        C2 (legacy), 8-dim
        c{n}_frame      Cn cyclic, n >= 2  (e.g. c3_frame)
        d6_frame        D6 (legacy), 18-dim
        d{n}_frame      Dn dihedral, n >= 2  (e.g. d3_frame)
        t_frame         T chiral tetrahedral, 7-dim
        o_frame         O chiral octahedral, 9-dim
        i_frame         I chiral icosahedral, 13-dim
    """
    # Legacy exact-match modes
    if mode in ("c1_frame", "c2_frame", "d6_frame"):
        return mode

    # Non-frame modes
    _OTHER_VALID = {
        "none",
        "full_frame",
        "full_frame_compact",
        "full_frame_irrep_complete",
        "full_frame_raw",
        "invariant_radial",
        "pose_mlp",
    }
    if mode in _OTHER_VALID:
        return mode

    # Cubic / icosahedral
    if mode in ("t_frame", "o_frame", "i_frame"):
        return mode

    # Parameterised cyclic: c{n}_frame (n >= 2)
    m_cn = _CN_RE.match(mode)
    if m_cn:
        n = int(m_cn.group(1))
        if n < 2:
            raise ValueError(
                f"Cn symmetry requires n >= 2; got {mode!r} (n={n})."
            )
        return mode

    # Parameterised dihedral: d{n}_frame (n >= 2)
    m_dn = _DN_RE.match(mode)
    if m_dn:
        n = int(m_dn.group(1))
        if n < 2:
            raise ValueError(
                f"Dn symmetry requires n >= 2; got {mode!r} (n={n})."
            )
        return mode

    raise ValueError(
        f"Unknown rigid_pair_mode={mode!r}. "
        "Valid modes: none, full_frame, full_frame_compact, "
        "full_frame_irrep_complete, full_frame_raw, invariant_radial, "
        "c1_frame, c2_frame, c{n}_frame (n>=2), "
        "d6_frame, d{n}_frame (n>=2), "
        "t_frame, o_frame, i_frame."
    )


class RigidPairSymmetrizedEdgeEmbedding(torch.nn.Module):
    """Projected rigid-pair features for a body-symmetry–constrained molecule."""

    def __init__(
        self,
        body_irreps,
        body_features,
        max_ell=None,
        multiplicity=1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__()
        if max_ell is None:
            max_ell = lmax
        if max_ell is None:
            raise TypeError("max_ell/lmax must be provided")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        if multiplicity < 1:
            raise ValueError("multiplicity must be >= 1")

        self.max_ell = int(max_ell)
        self.multiplicity = int(multiplicity)
        self.body_irreps = body_irreps
        self.body_features = body_features

        if edge_irreps is None:
            self.sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        else:
            self.sh_irreps = o3.Irreps(edge_irreps)

        self.edge_body_tp = o3.FullTensorProduct(
            self.sh_irreps,
            self.body_irreps,
        )
        allowed_irreps = [ir for _, ir in self.sh_irreps]
        self.pair_tp = o3.FullTensorProduct(
            self.edge_body_tp.irreps_out,
            self.body_irreps,
            filter_ir_out=allowed_irreps,
        )
        self.irreps_in = self.pair_tp.irreps_out

        self.edge_irreps = o3.Irreps(
            [(mul * self.multiplicity, ir) for mul, ir in self.sh_irreps]
        )
        self.irreps_out = self.edge_irreps

        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.irreps_in,
                self.irreps_out,
            )

    def forward(
        self,
        quaternions,
        edge_index,
        edge_vectors,
    ):
        body = self.body_features(quaternions)

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vectors,
            normalize=True,
            normalization="component",
        )

        senders = edge_index[0]
        receivers = edge_index[1]

        x = self.edge_body_tp(
            edge_sh,
            body[senders],
        )
        x = self.pair_tp(
            x,
            body[receivers],
        )
        x = self.projection(x)

        if self.multiplicity > 1:
            x = x / self.multiplicity**0.5

        return x


# ---------------------------------------------------------------------------
# Legacy named symmetry classes (unchanged from original)
# ---------------------------------------------------------------------------


class RigidPairD6EdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for a D6-symmetric molecule."""

    def __init__(
        self,
        max_ell=None,
        multiplicity=1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            D6_BODY_IRREPS,
            d6_body_features,
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


def c2_body_features_from_quaternions(
    quaternions, *, c2_axis
):
    """Convert quaternions to matrices for C2 encoding."""
    return c2_body_irreducible_features(
        quaternion_to_matrix(quaternions),
        c2_axis=c2_axis,
    )


class RigidPairC2EdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for a C2-symmetric molecule."""

    def __init__(
        self,
        max_ell=None,
        multiplicity=1,
        c2_axis=1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):

        if c2_axis not in (0, 1, 2):
            raise ValueError("c2_axis must be 0, 1, or 2")

        self.c2_axis = int(c2_axis)

        body_features = partial(c2_body_features_from_quaternions, c2_axis=self.c2_axis)
        super().__init__(
            C2_BODY_IRREPS,
            body_features,
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


class RigidPairC1EdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for a C1-symmetric molecule."""

    def __init__(
        self,
        max_ell=None,
        multiplicity=1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            C1_BODY_IRREPS,
            C1BodyFeatures(),
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# New parameterised symmetry classes
# ---------------------------------------------------------------------------


class RigidPairCnEdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for a Cn-symmetric molecule.

    Parameters
    ----------
    n : int
        Cyclic order (n >= 2).  ``"c3_frame"`` → n=3, etc.

    Note: ``n=2`` uses a different l=2 template than the legacy
    ``c2_frame`` mode; for backward compat use ``c2_frame``.
    """

    def __init__(
        self,
        n: int,
        max_ell=None,
        multiplicity: int = 1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            cn_body_irreps(n),
            CnBodyFeatures(n),
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


class RigidPairDnEdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for a Dn-symmetric molecule.

    Parameters
    ----------
    n : int
        Dihedral order (n >= 2).  ``"d3_frame"`` → n=3, etc.
    """

    def __init__(
        self,
        n: int,
        max_ell=None,
        multiplicity: int = 1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            dn_body_irreps(n),
            DnBodyFeatures(n),
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


class RigidPairTEdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for a T-symmetric (chiral tetrahedral) molecule."""

    def __init__(
        self,
        max_ell=None,
        multiplicity: int = 1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            T_BODY_IRREPS,
            TBodyFeatures(),
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


class RigidPairOEdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for an O-symmetric (chiral octahedral) molecule."""

    def __init__(
        self,
        max_ell=None,
        multiplicity: int = 1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            O_BODY_IRREPS,
            OBodyFeatures(),
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )


class RigidPairIEdgeEmbedding(RigidPairSymmetrizedEdgeEmbedding):
    """Projected rigid-pair features for an I-symmetric (chiral icosahedral) molecule."""

    def __init__(
        self,
        max_ell=None,
        multiplicity: int = 1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        super().__init__(
            I_BODY_IRREPS,
            IBodyFeatures(),
            max_ell,
            multiplicity,
            lmax,
            edge_irreps,
            **kwargs,
        )
