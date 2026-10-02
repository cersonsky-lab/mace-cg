"""Standalone equivariant tensor-product features for rigid-body pairs.

This module is diagnostic only and is not wired into MACE interactions.

A rigid orientation matrix R is represented by its three body axes in the
lab frame. Each axis transforms as an l=1 polar vector under proper global
rotations, giving orientation irreps

    3x1o.

For an edge i -> j we construct

    Y_l(rhat_ij) x frame_i x frame_j

using full e3nn tensor products.

Because FullTensorProduct retains every allowed Clebsch-Gordan path, this
stage does not introduce a learned compression of the angular information.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3

from mace.data.rigid_body import quaternion_to_matrix
from mace.modules.rigid_c2 import C2_BODY_IRREPS, c2_body_irreducible_features
from mace.modules.rigid_d6 import D6_BODY_IRREPS, d6_body_features


class RigidPairTensorProductFeatures(torch.nn.Module):
    """Full equivariant tensor-product basis for an ordered rigid pair."""

    def __init__(self, lmax: int = 2):
        super().__init__()

        if lmax < 0:
            raise ValueError("lmax must be >= 0")

        self.lmax = lmax

        self.edge_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.frame_irreps = o3.Irreps("3x1o")

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
    def _frame_features(rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to 3x1o body-axis features.

        ``rotation_matrices[..., :, a]`` is body axis ``a`` expressed in
        lab coordinates.

        e3nn expects multiplicity-major layout

            [axis_0_xyz, axis_1_xyz, axis_2_xyz],

        hence the transpose before flattening.
        """
        return rotation_matrices.transpose(-1, -2).reshape(
            *rotation_matrices.shape[:-2],
            9,
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


class RigidPairOracleMixedScalarEdgeEmbedding(torch.nn.Module):
    """Diagnostic frozen scalar containing the synthetic mixed C1 teacher.

    This is not a production molecular representation.  It is an
    architecture-localization control: if downstream MACE cannot learn
    the synthetic energy when supplied the exact invariant as a scalar
    edge feature, the failure lies after the generic C1 projection.
    """

    def __init__(self):
        super().__init__()
        self.edge_irreps = o3.Irreps("1x0e")
        self.irreps_out = self.edge_irreps

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
            raise ValueError(
                "quaternions must have shape (num_nodes, 4)"
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape (2, num_edges)"
            )

        if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
            raise ValueError(
                "edge_vectors must have shape (num_edges, 3)"
            )

        distances = torch.linalg.norm(
            edge_vectors,
            dim=-1,
        )

        if torch.any(distances <= 0):
            raise ValueError(
                "edge_vectors must have nonzero length"
            )

        rhat = edge_vectors / distances.unsqueeze(-1)

        rotations = quaternion_to_matrix(quaternions)

        i = edge_index[0]
        j = edge_index[1]

        x_i = rotations[i, :, 0]
        z_i = rotations[i, :, 2]
        x_j = rotations[j, :, 0]
        z_j = rotations[j, :, 2]

        def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.sum(a * b, dim=-1)

        u1 = dot(x_i, x_j)
        u2 = dot(z_i, z_j)

        u3 = 0.5 * (
            dot(x_i, z_j)
            + dot(z_i, x_j)
        )

        x_i_r = dot(x_i, rhat)
        z_i_r = dot(z_i, rhat)
        x_j_r = dot(x_j, rhat)
        z_j_r = dot(z_j, rhat)

        u4 = x_i_r * x_j_r
        u5 = z_i_r * z_j_r

        u6 = 0.5 * (
            x_i_r * z_j_r
            + z_i_r * x_j_r
        )

        value = (
            +0.45 * u1
            -0.35 * u2
            +0.25 * u3
            +0.30 * u4
            -0.20 * u5
            +0.15 * u6
        )

        return value.unsqueeze(-1)


class RigidPairScalarCompleteEdgeEmbedding(torch.nn.Module):
    """Preserve all raw invariant scalars while compressing non-scalars.

    The generic C1 full-frame tensor product contains many independent
    0e channels.  Standard ``full_frame`` immediately learns a map from
    that multiplicity to a single 0e output.  This diagnostic keeps all
    raw 0e channels unchanged while projecting only the non-scalar
    sectors to ordinary spherical-harmonic width.
    """

    def __init__(
        self,
        lmax: int,
        edge_irreps: o3.Irreps,
    ):
        super().__init__()

        self.full_pair = RigidPairTensorProductFeatures(
            lmax=lmax,
        )

        scalar_indices = []

        for (_, ir), sl in zip(
            self.full_pair.irreps_out,
            self.full_pair.irreps_out.slices(),
        ):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(
                    range(sl.start, sl.stop)
                )

        if not scalar_indices:
            raise RuntimeError(
                "raw rigid-pair representation contains no 0e channels"
            )

        self.register_buffer(
            "scalar_indices",
            torch.tensor(
                scalar_indices,
                dtype=torch.long,
            ),
        )

        self.num_raw_scalars = len(scalar_indices)

        self.scalar_irreps = o3.Irreps(
            [
                (
                    self.num_raw_scalars,
                    o3.Irrep("0e"),
                )
            ]
        )

        base_edge_irreps = o3.Irreps(edge_irreps)

        self.non_scalar_irreps = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in base_edge_irreps
                if ir.l != 0
            ]
        )

        with torch.random.fork_rng(devices=[]):
            self.non_scalar_projection = o3.Linear(
                self.full_pair.irreps_out,
                self.non_scalar_irreps,
            )

        self.edge_irreps = (
            self.scalar_irreps
            + self.non_scalar_irreps
        )
        self.irreps_out = self.edge_irreps

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )

        scalar = torch.index_select(
            raw,
            dim=-1,
            index=self.scalar_indices,
        )

        non_scalar = self.non_scalar_projection(
            raw
        )

        return torch.cat(
            (scalar, non_scalar),
            dim=-1,
        )


class RigidPairPoseInvariantEdgeEmbedding(torch.nn.Module):
    """Exact directed internal-pose diagnostic for a rigid dimer.

    For directed edge i -> j, return

        R_i^T R_j
        R_i^T r_hat_ij

    The 3x3 relative rotation is invariant under global SO(3).
    The body-frame separation direction is also invariant under
    global SO(3).

    Under spatial inversion the relative-rotation entries are even
    while the separation-direction entries are odd, hence

        9x0e + 3x0o.

    This is a diagnostic upper bound, not a proposed compressed
    production representation.
    """

    def __init__(self):
        super().__init__()

        # SO(3)-only diagnostic:
        #
        # Both R_i^T R_j and R_i^T r_hat are invariant under
        # global proper rotations.  We deliberately label every
        # component 0e so O(3) parity bookkeeping downstream does
        # not suppress the three body-frame direction coordinates.
        #
        # This is an information/optimization upper-bound control,
        # not an O(3)-equivariant production representation.
        self.edge_irreps = o3.Irreps(
            "12x0e"
        )

        self.irreps_out = (
            self.edge_irreps
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        rotations = quaternion_to_matrix(
            quaternions
        )

        senders = edge_index[0]
        receivers = edge_index[1]

        rnorm = torch.linalg.norm(
            edge_vectors,
            dim=-1,
            keepdim=True,
        )

        eps = torch.finfo(
            edge_vectors.dtype
        ).eps

        rhat = (
            edge_vectors
            / torch.clamp(
                rnorm,
                min=eps,
            )
        )

        Ri = rotations[senders]
        Rj = rotations[receivers]

        RiT = Ri.transpose(
            -1,
            -2,
        )

        relative_rotation = torch.matmul(
            RiT,
            Rj,
        )

        sender_rhat = torch.matmul(
            RiT,
            rhat.unsqueeze(-1),
        ).squeeze(-1)

        return torch.cat(
            (
                relative_rotation.reshape(
                    -1,
                    9,
                ),
                sender_rhat,
            ),
            dim=-1,
        )


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


C1_WIGNER_L123_BODY_IRREPS = o3.Irreps(
    "3x1o + 5x2e + 7x3o"
)


class RigidPairC1WignerL123EdgeEmbedding(torch.nn.Module):
    """C1 Wigner-D body harmonics through l=3.

    For each proper body-to-space rotation R, retain the complete
    body-index multiplicity of D^l(R):

        l=1: 3x1o
        l=2: 5x2e
        l=3: 7x3o

    Pair features are constructed in three same-l branches,

        Y(r_ij) x D^l(R_i) x D^l(R_j),

    and the concatenated complete tensor-product output is projected
    to the same ordinary edge-SH irreps used by full_frame.

    This deliberately omits cross-l body products in this first
    diagnostic.
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
                "multiplicity must be a positive integer"
            )

        self.lmax = int(lmax)
        self.multiplicity = int(multiplicity)

        self.sh_irreps = o3.Irreps(edge_irreps)
        self.body_irreps = C1_WIGNER_L123_BODY_IRREPS

        self.body_irreps_l1 = o3.Irreps("3x1o")
        self.body_irreps_l2 = o3.Irreps("5x2e")
        self.body_irreps_l3 = o3.Irreps("7x3o")

        self.edge_center_tp_l1 = o3.FullTensorProduct(
            self.sh_irreps,
            self.body_irreps_l1,
        )
        self.pair_tp_l1 = o3.FullTensorProduct(
            self.edge_center_tp_l1.irreps_out,
            self.body_irreps_l1,
        )

        self.edge_center_tp_l2 = o3.FullTensorProduct(
            self.sh_irreps,
            self.body_irreps_l2,
        )
        self.pair_tp_l2 = o3.FullTensorProduct(
            self.edge_center_tp_l2.irreps_out,
            self.body_irreps_l2,
        )

        self.edge_center_tp_l3 = o3.FullTensorProduct(
            self.sh_irreps,
            self.body_irreps_l3,
        )
        self.pair_tp_l3 = o3.FullTensorProduct(
            self.edge_center_tp_l3.irreps_out,
            self.body_irreps_l3,
        )

        self.pair_irreps = (
            self.pair_tp_l1.irreps_out
            + self.pair_tp_l2.irreps_out
            + self.pair_tp_l3.irreps_out
        )

        self.base_edge_irreps = o3.Irreps(
            edge_irreps
        )

        self.edge_irreps = o3.Irreps(
            [
                (
                    mul * self.multiplicity,
                    ir,
                )
                for mul, ir in self.base_edge_irreps
            ]
        )

        self.irreps_in = self.pair_irreps
        self.irreps_out = self.edge_irreps

        # Match the existing full_frame convention: the large rigid
        # pair basis is compressed back to ordinary edge-SH irreps.
        # fork_rng avoids changing initialization of the surrounding
        # MACE model merely by enabling this diagnostic.
        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.pair_irreps,
                self.edge_irreps,
            )

        self.output_scale = (
            1.0 / math.sqrt(self.multiplicity)
        )

    @staticmethod
    def _wigner_block(
        rotation_matrices: torch.Tensor,
        ell: int,
    ) -> torch.Tensor:
        """Return complete multiplicity-major D^ell(R).

        e3nn returns D with layout

            [..., m_space, n_body].

        For an equivariant feature representation, n_body labels
        independent copies of the space-frame ell irrep, so transpose
        to

            [..., n_body, m_space]

        before flattening.
        """
        parity = -1 if ell % 2 else 1

        D = o3.Irrep(
            ell,
            parity,
        ).D_from_matrix(
            rotation_matrices
        )

        dim = 2 * ell + 1

        return D.transpose(
            -1,
            -2,
        ).reshape(
            *rotation_matrices.shape[:-2],
            dim * dim,
        )

    def body_features(
        self,
        rotation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """Complete C1 D^1 + D^2 + D^3 body representation."""
        return torch.cat(
            (
                self._wigner_block(
                    rotation_matrices,
                    1,
                ),
                self._wigner_block(
                    rotation_matrices,
                    2,
                ),
                self._wigner_block(
                    rotation_matrices,
                    3,
                ),
            ),
            dim=-1,
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if (
            quaternions.ndim != 2
            or quaternions.shape[-1] != 4
        ):
            raise ValueError(
                "quaternions must have shape "
                "(num_nodes, 4)"
            )

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(
                "edge_index must have shape "
                "(2, num_edges)"
            )

        if (
            edge_vectors.ndim != 2
            or edge_vectors.shape[-1] != 3
        ):
            raise ValueError(
                "edge_vectors must have shape "
                "(num_edges, 3)"
            )

        if (
            edge_vectors.shape[0]
            != edge_index.shape[1]
        ):
            raise ValueError(
                "edge_vectors and edge_index must "
                "contain the same number of edges"
            )

        distances = torch.linalg.vector_norm(
            edge_vectors,
            dim=-1,
        )

        if torch.any(distances <= 1.0e-12):
            raise ValueError(
                "c1_wigner_l123 does not support "
                "zero-length edges"
            )

        directions = (
            edge_vectors
            / distances.unsqueeze(-1)
        )

        edge_features = o3.spherical_harmonics(
            self.sh_irreps,
            directions,
            normalize=True,
            normalization="component",
        )

        rotations = quaternion_to_matrix(
            quaternions
        )

        body = self.body_features(
            rotations
        )

        # Complete C1 blocks:
        #   D1:  9 values  [0:9]
        #   D2: 25 values  [9:34]
        #   D3: 49 values  [34:83]
        body_l1 = body[:, 0:9]
        body_l2 = body[:, 9:34]
        body_l3 = body[:, 34:83]

        centers = edge_index[0]
        neighbors = edge_index[1]

        center_l1 = self.edge_center_tp_l1(
            edge_features,
            body_l1[centers],
        )
        pair_l1 = self.pair_tp_l1(
            center_l1,
            body_l1[neighbors],
        )

        center_l2 = self.edge_center_tp_l2(
            edge_features,
            body_l2[centers],
        )
        pair_l2 = self.pair_tp_l2(
            center_l2,
            body_l2[neighbors],
        )

        center_l3 = self.edge_center_tp_l3(
            edge_features,
            body_l3[centers],
        )
        pair_l3 = self.pair_tp_l3(
            center_l3,
            body_l3[neighbors],
        )

        pair_features = torch.cat(
            (
                pair_l1,
                pair_l2,
                pair_l3,
            ),
            dim=-1,
        )

        return (
            self.projection(pair_features)
            * self.output_scale
        )



def validate_rigid_pair_mode(mode: str) -> str:
    """Validate the rigid-pair edge representation mode."""
    if mode in (
        "c2_frame",
        "d6_frame",
        "d6_frame_compact",
    ):
        return mode
    valid_modes = {
        "none",
        "full_frame",
        "full_frame_compact",
        "full_frame_irrep_complete",
        "full_frame_scalar_complete",
        "pose_invariant_exact",
        "full_frame_raw",
        "oracle_mixed_scalar",
        "invariant_radial",
        "c1_wigner_l123",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown rigid_pair_mode={mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    return mode


class RigidPairD6EdgeEmbedding(torch.nn.Module):
    """Projected rigid-pair features for a D6-symmetric molecule."""

    def __init__(
        self,
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

        if edge_irreps is None:
            self.sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        else:
            self.sh_irreps = o3.Irreps(edge_irreps)

        self.edge_body_tp = o3.FullTensorProduct(
            self.sh_irreps,
            D6_BODY_IRREPS,
        )
        allowed_irreps = [ir for _, ir in self.sh_irreps]
        self.pair_tp = o3.FullTensorProduct(
            self.edge_body_tp.irreps_out,
            D6_BODY_IRREPS,
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
        body = d6_body_features(quaternions)

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


class RigidPairC2EdgeEmbedding(torch.nn.Module):
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
        super().__init__()

        if max_ell is None:
            max_ell = lmax

        if max_ell is None:
            raise TypeError("max_ell/lmax must be provided")

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        if multiplicity < 1:
            raise ValueError("multiplicity must be >= 1")

        if c2_axis not in (0, 1, 2):
            raise ValueError("c2_axis must be 0, 1, or 2")

        self.max_ell = int(max_ell)
        self.multiplicity = int(multiplicity)
        self.c2_axis = int(c2_axis)

        # `edge_irreps` is the ordinary MACE spherical-harmonic
        # basis passed by models.py.  Keep it separate from the
        # projected rigid-pair output irreps.
        if edge_irreps is None:
            self.sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        else:
            self.sh_irreps = o3.Irreps(edge_irreps)

        self.edge_body_tp = o3.FullTensorProduct(
            self.sh_irreps,
            C2_BODY_IRREPS,
        )

        self.pair_tp = o3.FullTensorProduct(
            self.edge_body_tp.irreps_out,
            C2_BODY_IRREPS,
        )

        self.irreps_in = self.pair_tp.irreps_out

        self.edge_irreps = o3.Irreps(
            [(mul * self.multiplicity, ir) for mul, ir in self.sh_irreps]
        )

        self.irreps_out = self.edge_irreps

        # Do not perturb initialization of the ordinary MACE path.
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
        rotations = quaternion_to_matrix(quaternions)

        body = c2_body_irreducible_features(
            rotations,
            c2_axis=self.c2_axis,
        )

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
