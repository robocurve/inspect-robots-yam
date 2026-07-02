"""Canonical 14-D bimanual packing for YAM + MolmoAct2.

The MolmoAct2 bimanual-YAM wire protocol uses a single flat **14-D** vector for
both proprioceptive ``state`` and predicted ``actions``. This module is the *one*
place that defines how those 14 numbers map to the two arms, so the policy
(client) and the embodiment (driver) can never disagree.

Convention (per arm, 7-D): ``[j0, j1, j2, j3, j4, j5, gripper]`` — the six
revolute joints in order, gripper last. The full vector is ``left`` then
``right``: indices ``0..6`` are the left arm, ``7..13`` the right arm.

This module is pure NumPy with no optional/hardware dependencies, so it imports
and tests anywhere.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import StateField, StateSpec

ARM_DOF = 6  # revolute joints per arm
GRIPPER_DOF = 1  # one linear gripper per arm
ARM_WIDTH = ARM_DOF + GRIPPER_DOF  # 7-D per arm
TOTAL_DIM = ARM_WIDTH * 2  # 14-D bimanual

LEFT = slice(0, ARM_WIDTH)  # indices 0..6
RIGHT = slice(ARM_WIDTH, TOTAL_DIM)  # indices 7..13

# The canonical proprioception key MolmoAct2's YAM server expects as a flat 14-D
# ``state``. Joints are radians, the trailing gripper of each arm is normalized;
# we model it as a single field so ``StateSpec.keys == {"joint_pos"}`` stays
# consistent with the ``state_keys`` both components declare for compatibility.
STATE_KEY = "joint_pos"
STATE_SPEC = StateSpec(
    fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit="rad+normalized"),)
)

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike, n: int = TOTAL_DIM) -> Vec:
    """Return ``vec`` as a 1-D float array, raising ``ValueError`` if not length ``n``."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(
            f"expected a {n}-D vector, got shape {np.shape(vec)} ({arr.shape[0]} elems)"
        )
    return arr


def pack(left: npt.ArrayLike, right: npt.ArrayLike) -> Vec:
    """Concatenate a left 7-D and right 7-D arm vector into the flat 14-D vector."""
    lv = validate_dim(left, ARM_WIDTH)
    rv = validate_dim(right, ARM_WIDTH)
    return np.concatenate([lv, rv])


def split(vec: npt.ArrayLike) -> tuple[Vec, Vec]:
    """Split a flat 14-D vector into ``(left 7-D, right 7-D)`` arm vectors."""
    arr = validate_dim(vec, TOTAL_DIM)
    return arr[LEFT].copy(), arr[RIGHT].copy()
