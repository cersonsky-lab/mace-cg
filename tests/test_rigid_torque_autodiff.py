from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from e3nn import o3
from scipy.spatial.transform import Rotation

from mace import data, modules, tools


DTYPE = torch.float64
TABLE = tools.AtomicNumberTable([0])
CUTOFF = 5.0


def _wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.asarray([w, x, y, z], dtype=float)


def _atoms(body_i: Rotation, body_j: Rotation) -> Atoms:
    atoms = Atoms(
        "XX",
        positions=[
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
    )

    atoms.arrays["quaternions"] = np.stack(
        [_wxyz(body_i), _wxyz(body_j)]
    )

    atoms.arrays["c_diameter[1]"] = np.asarray([2.0, 2.0])
    atoms.arrays["c_diameter[2]"] = np.asarray([3.0, 3.0])
    atoms.arrays["c_diameter[3]"] = np.asarray([4.0, 4.0])

    return atoms


def _batch(atoms: Atoms):
    config = data.config_from_atoms(
        atoms,
        config_type_weights={"Default": 1.0},
    )

    # AtomicData.from_config() uses torch.get_default_dtype() when it
    # constructs the lab-frame rigid tensors and their irreps.  Building
    # them in float32 and casting afterwards is not sufficient for a
    # finite-difference test with ~1e-5-radian perturbations: the small
    # rotational change has already been quantized.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(DTYPE)
        graph = data.AtomicData.from_config(
            config,
            z_table=TABLE,
            cutoff=CUTOFF,
            heads=["Default"],
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    loader = tools.torch_geometric.dataloader.DataLoader(
        dataset=[graph],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(loader)).to_dict()

    for key, value in batch.items():
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(dtype=DTYPE)

    return batch


def _model(
    *,
    rigid_feature_mode: str,
    rigid_pair_mode: str,
):
    torch.manual_seed(1234)

    model = modules.ScaleShiftMACE(
        r_max=CUTOFF,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(
            "8x0e + 8x1o + 8x2e"
        ),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=F.silu,
        atomic_energies=np.asarray([0.0]),
        avg_num_neighbors=1.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        rigid_feature_mode=rigid_feature_mode,
        rigid_pair_mode=rigid_pair_mode,
    )

    return model.to(dtype=DTYPE)


def _finite_difference_neighbor_torque(
    model,
    body_i,
    body_j,
    axis,
    eps=1.0e-5,
):
    # Left multiplication corresponds to a lab-frame rotation.
    plus = Rotation.from_rotvec(axis * eps) * body_j
    minus = Rotation.from_rotvec(-axis * eps) * body_j

    e_plus = model(
        _batch(_atoms(body_i, plus)),
        compute_force=False,
    )["energy"].sum()

    e_minus = model(
        _batch(_atoms(body_i, minus)),
        compute_force=False,
    )["energy"].sum()

    return -(e_plus - e_minus) / (2.0 * eps)


def _check_mode(
    rigid_feature_mode: str,
    rigid_pair_mode: str,
):
    model = _model(
        rigid_feature_mode=rigid_feature_mode,
        rigid_pair_mode=rigid_pair_mode,
    )
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.2, -0.3, 0.1])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.4, 0.1, 0.5])
    )

    axis = np.asarray(
        [0.3, -0.7, 0.2],
        dtype=float,
    )
    axis /= np.linalg.norm(axis)

    reference = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
        compute_torque=False,
    )

    out = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
        compute_torque=True,
    )

    # Turning torque evaluation on must not change the energy itself.
    torch.testing.assert_close(
        out["energy"],
        reference["energy"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    expected = _finite_difference_neighbor_torque(
        model,
        body_i,
        body_j,
        axis,
    )

    actual = torch.dot(
        out["torques"][1],
        torch.tensor(axis, dtype=DTYPE),
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


def test_pair_quaternion_torque_matches_finite_rotation():
    _check_mode(
        rigid_feature_mode="none",
        rigid_pair_mode="full_frame",
    )


def test_rank2_tensor_torque_matches_finite_rotation():
    _check_mode(
        rigid_feature_mode="moi",
        rigid_pair_mode="none",
    )
