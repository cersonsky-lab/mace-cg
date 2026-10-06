import torch
from e3nn import o3
from e3nn.o3._spherical_harmonics import _spherical_harmonics

from mace.modules.rigid_wigner import (
    full_wigner_features,
    full_wigner_irreps,
)

C1_WIGNER_ELLS = (
    1,
    2,
    3,
)

C1_BODY_IRREPS = full_wigner_irreps(C1_WIGNER_ELLS)


class C1WignerFeatures(torch.nn.Module):
    """TorchScript-safe complete C1 orientation features through ell=3.

    For fixed body-frame probes u_k,

        Y_l(R u_k) = D^l(R) Y_l(u_k).

    A fixed pseudoinverse reconstructs D^l(R)^T in the
    multiplicity-major layout used by the C1 representation.

    The spherical harmonics are evaluated with e3nn's internal
    polynomial kernel, avoiding the non-scriptable public wrapper
    annotations while preserving exactly the same basis.
    """

    def __init__(self):
        super().__init__()

        n_probes = 20

        k = torch.arange(
            n_probes,
            dtype=torch.float64,
        )

        z = 1.0 - 2.0 * (k + 0.5) / float(n_probes)

        radius = torch.sqrt(
            torch.clamp(
                1.0 - z * z,
                min=0.0,
            )
        )

        golden_angle = torch.pi * (
            3.0
            - torch.sqrt(
                torch.tensor(
                    5.0,
                    dtype=torch.float64,
                )
            )
        )

        phi = golden_angle * k

        probes = torch.stack(
            (
                radius * torch.cos(phi),
                radius * torch.sin(phi),
                z,
            ),
            dim=-1,
        )

        with torch.no_grad():
            sh_body = _spherical_harmonics(
                3,
                probes[:, 0],
                probes[:, 1],
                probes[:, 2],
            )

            y1_body = sh_body[:, 1:4]
            y2_body = sh_body[:, 4:9]
            y3_body = sh_body[:, 9:16]

            if int(torch.linalg.matrix_rank(y1_body)) != 3:
                raise RuntimeError("C1 ell=1 probe matrix is rank deficient")

            if int(torch.linalg.matrix_rank(y2_body)) != 5:
                raise RuntimeError("C1 ell=2 probe matrix is rank deficient")

            if int(torch.linalg.matrix_rank(y3_body)) != 7:
                raise RuntimeError("C1 ell=3 probe matrix is rank deficient")

            pinv1 = torch.linalg.pinv(y1_body)
            pinv2 = torch.linalg.pinv(y2_body)
            pinv3 = torch.linalg.pinv(y3_body)

        self.register_buffer(
            "probes",
            probes,
        )
        self.register_buffer(
            "pinv1",
            pinv1,
        )
        self.register_buffer(
            "pinv2",
            pinv2,
        )
        self.register_buffer(
            "pinv3",
            pinv3,
        )

    def forward(
        self,
        rotation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        if rotation_matrices.size(-2) != 3 or rotation_matrices.size(-1) != 3:
            raise ValueError("rotation_matrices must have shape (..., 3, 3)")

        space_probes = torch.einsum(
            "...ij,kj->...ki",
            rotation_matrices,
            self.probes,
        )

        sh_space = _spherical_harmonics(
            3,
            space_probes[..., 0],
            space_probes[..., 1],
            space_probes[..., 2],
        )

        y1_space = sh_space[..., 1:4]
        y2_space = sh_space[..., 4:9]
        y3_space = sh_space[..., 9:16]

        d1_t = torch.matmul(
            self.pinv1,
            y1_space,
        )
        d2_t = torch.matmul(
            self.pinv2,
            y2_space,
        )
        d3_t = torch.matmul(
            self.pinv3,
            y3_space,
        )

        return torch.cat(
            (
                d1_t.flatten(start_dim=-2),
                d2_t.flatten(start_dim=-2),
                d3_t.flatten(start_dim=-2),
            ),
            dim=-1,
        )


def c1_body_irreducible_features(
    rotation_matrices: torch.Tensor,
) -> torch.Tensor:
    """Return D1(R) + D2(R) + D3(R) for one C1 orientation."""

    features = full_wigner_features(
        rotation_matrices,
        [
            1,
            2,
            3,
        ],
    )

    if features.shape[-1] != C1_BODY_IRREPS.dim:
        raise RuntimeError(
            "Internal C1 feature dimension mismatch: "
            f"got {features.shape[-1]}, "
            f"expected {C1_BODY_IRREPS.dim}"
        )

    return features
