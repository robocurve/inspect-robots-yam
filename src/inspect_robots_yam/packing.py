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

# Human-readable names for the 14 dims, in packing order. Carried on
# ActionSemantics.dim_labels so label-addressed tooling (the LLM agent
# policy, logging, visualization) can name joints instead of indices.
DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(ARM_DOF)), "gripper")
)

# The canonical proprioception key MolmoAct2's YAM server expects as a flat 14-D
# ``state``. Joints are radians, the trailing gripper of each arm is normalized
# (1 = open); we model it as a single field so ``StateSpec.keys == {"joint_pos"}`` stays
# consistent with the ``state_keys`` both components declare for compatibility.
STATE_KEY = "joint_pos"


def state_spec(key: str = STATE_KEY) -> StateSpec:
    """A flat 14-D ``StateSpec`` under ``key`` (rad + normalized grippers, 1 = open).

    Deriving the spec from the key keeps ``state_keys`` and the ``StateField`` key
    consistent for any configured ``state_key`` — ``ObservationSpace`` rejects them
    if they disagree.
    """
    return StateSpec(fields=(StateField(key=key, shape=(TOTAL_DIM,), unit="rad+normalized"),))


STATE_SPEC = state_spec()

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike, n: int = TOTAL_DIM) -> Vec:
    """Return ``vec`` as a 1-D float array of length ``n``, raising ``ValueError`` otherwise.

    Requires ``ndim == 1`` outright — flattening a same-size 2-D array (e.g.
    ``(7, 2)``) would silently interleave-scramble the arm packing.
    """
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"expected a {n}-D vector, got shape {np.shape(vec)}")
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
