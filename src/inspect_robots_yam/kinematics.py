"""Always-importable Cartesian kinematics wrapper for the YAM embodiment.

The raw solver is injected behind a small NumPy-only protocol. This module owns
all safety and stateful command logic, while the optional i2rt adapter only
binds its concrete FK, IK, and model-range APIs.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from inspect_robots_yam.packing import ARM_DOF

Vec = npt.NDArray[np.float64]
Pose = npt.NDArray[np.float64]


class RawKinematics(Protocol):
    """Minimal raw FK/IK/model-range adapter returned by a factory."""

    def get_joint_ranges(self) -> npt.NDArray[np.floating[Any]]:
        """Return model joint ranges as an ``(n, 2)`` array."""
        ...

    def set_joint_ranges(self, ranges: npt.NDArray[np.floating[Any]]) -> None:
        """Replace the wrapper-owned model's live joint ranges."""
        ...

    def fk(self, q: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
        """Return the grasp-site transform for a full model configuration."""
        ...

    def ik(
        self,
        target: npt.NDArray[np.floating[Any]],
        init_q: npt.NDArray[np.floating[Any]],
        max_iters: int,
    ) -> tuple[bool, npt.NDArray[np.floating[Any]]]:
        """Return convergence and the last full-model IK iterate."""
        ...


class _ArmKinematics:
    """Own one arm's model limits, yaw convention, and safe IK command state."""

    def __init__(
        self,
        *,
        side: str,
        raw: RawKinematics,
        config_low: Vec,
        config_high: Vec,
        ik_max_iters: int,
        ik_step_joint_limit: float,
        cmd_resync_threshold: float,
        osc_deadband: float,
        osc_reversals: int,
        osc_window: int,
        osc_hold_steps: int,
    ) -> None:
        self._side = side
        self._raw = raw
        model_ranges = np.asarray(raw.get_joint_ranges(), dtype=np.float64)
        if (
            model_ranges.ndim != 2
            or model_ranges.shape[1:] != (2,)
            or not (ARM_DOF <= model_ranges.shape[0] <= ARM_DOF + 2)
        ):
            raise ValueError(
                f"{side} kinematics model must expose 6 arm joints and 0..2 trailing "
                f"gripper joints, got range shape {model_ranges.shape}"
            )
        cfg_low = np.asarray(config_low, dtype=np.float64)
        cfg_high = np.asarray(config_high, dtype=np.float64)
        effective_low = np.maximum(model_ranges[:ARM_DOF, 0], cfg_low)
        effective_high = np.minimum(model_ranges[:ARM_DOF, 1], cfg_high)
        for joint, (low, high) in enumerate(zip(effective_low, effective_high, strict=True)):
            if low > high:
                raise ValueError(
                    f"empty effective joint range for {side}_j{joint}: "
                    f"intersection is [{low}, {high}]"
                )
        self._effective_ranges = np.column_stack((effective_low, effective_high))
        owned_ranges = model_ranges.copy()
        owned_ranges[:ARM_DOF] = self._effective_ranges
        raw.set_joint_ranges(owned_ranges)
        self._gripper_pin = np.mean(owned_ranges[ARM_DOF:], axis=1)

        self._ik_max_iters = ik_max_iters
        self._ik_step_joint_limit = ik_step_joint_limit
        self._cmd_resync_threshold = cmd_resync_threshold
        self._osc_deadband = osc_deadband
        self._osc_reversals = osc_reversals
        self._osc_window = osc_window
        self._osc_hold_steps = osc_hold_steps

        self._q_cmd_prev: Vec | None = None
        self._hold_counter = 0
        self._reversal_window: deque[npt.NDArray[np.bool_]] = deque(maxlen=osc_window)
        self._last_delta: Vec | None = None
        self._resynced = False
        self._yaw_reference: npt.NDArray[np.float64] | None = None

    @property
    def effective_ranges(self) -> npt.NDArray[np.float64]:
        """Return the six live model/config range intersections."""
        return self._effective_ranges.copy()

    @property
    def hold_counter(self) -> int:
        """Return the remaining whole-arm oscillation hold steps."""
        return self._hold_counter

    @property
    def reversal_window(self) -> tuple[npt.NDArray[np.bool_], ...]:
        """Return copies of the per-step, per-joint reversal records."""
        return tuple(item.copy() for item in self._reversal_window)

    @property
    def resynced(self) -> bool:
        """Report whether the most recent solve re-seeded from measurement."""
        return self._resynced

    def seed(self, q_commanded: npt.ArrayLike) -> None:
        """Seed the six-joint command reference and clear trial-local guard state."""
        q = np.asarray(q_commanded, dtype=np.float64)
        if q.shape != (ARM_DOF,):
            raise ValueError(f"expected {ARM_DOF} commanded arm joints, got {q.shape}")
        self._q_cmd_prev = q.copy()
        self._hold_counter = 0
        self._clear_reversals()
        self._resynced = False

    def clear(self) -> None:
        """Clear all command, guard, resync, and yaw-reference trial state."""
        self._q_cmd_prev = None
        self._hold_counter = 0
        self._clear_reversals()
        self._resynced = False
        self._yaw_reference = None

    def capture_yaw_reference(self, q_measured: npt.ArrayLike) -> None:
        """Pin the trial's reference orientation: zero for yaw, pitch, and roll.

        Targets and observations both work on the relative rotation
        ``R_measured @ reference.T``, which stays near identity inside the
        action box, so extraction never degenerates at any reachable pose
        (the ZYX singularity sits at |pitch| = pi/2, excluded by config
        bounds validation).
        """
        pose = self.fk(q_measured)
        self._yaw_reference = pose[:3, :3].copy()

    def fk(self, q_arm: npt.ArrayLike) -> Pose:
        """Run FK with any model gripper joints pinned at their mid-ranges."""
        return np.asarray(self._raw.fk(self._full_q(q_arm)), dtype=np.float64)

    def observe(self, q_measured: npt.ArrayLike, *, gripper: float) -> Vec:
        """Return ``x, y, z, yaw, pitch, roll, gripper`` from measured arm state.

        Orientation values are relative to the trial's reference orientation,
        matching the command convention in :meth:`solve`.
        """
        reference = self._require_yaw_reference()
        pose = self.fk(q_measured)
        yaw, pitch, roll = self._extract_relative(pose[:3, :3] @ reference.T)
        return np.asarray((*pose[:3, 3], yaw, pitch, roll, gripper), dtype=np.float64)

    def solve(self, target: npt.ArrayLike, q_measured: npt.ArrayLike) -> Vec:
        """Run the normative hold, resync, IK, rate-clamp, and reversal pipeline."""
        previous = self._require_command_reference()
        self._resynced = False

        if self._hold_counter > 0:
            self._hold_counter -= 1
            if self._hold_counter == 0:
                self._clear_reversals()
            return previous.copy()

        measured = np.asarray(q_measured, dtype=np.float64)
        if np.any(np.abs(previous - measured) > self._cmd_resync_threshold):
            previous = self._clip_effective(measured)
            self._q_cmd_prev = previous.copy()
            self._resynced = True

        target_values = np.asarray(target, dtype=np.float64)
        target_pose = np.eye(4, dtype=np.float64)
        target_pose[:3, :3] = (
            self._rotation_z(float(target_values[3]))
            @ self._rotation_y_forward(float(target_values[4]))
            @ self._rotation_x(float(target_values[5]))
            @ self._require_yaw_reference()
        )
        target_pose[:3, 3] = target_values[:3]
        warm_start = self._clip_effective(previous)
        _, q_ik_full = self._raw.ik(
            target_pose,
            self._full_q(warm_start),
            self._ik_max_iters,
        )
        q_ik = np.asarray(q_ik_full, dtype=np.float64)
        if not np.all(np.isfinite(q_ik)):
            return previous.copy()
        q_ik_arm = self._clip_effective(q_ik[:ARM_DOF])
        delta = np.clip(
            q_ik_arm - previous,
            -self._ik_step_joint_limit,
            self._ik_step_joint_limit,
        )
        q_command = previous + delta

        reversal = np.zeros(ARM_DOF, dtype=np.bool_)
        if not self._resynced and self._last_delta is not None:
            active = (np.abs(delta) > self._osc_deadband) & (
                np.abs(self._last_delta) > self._osc_deadband
            )
            reversal = active & (np.signbit(delta) != np.signbit(self._last_delta))
        self._reversal_window.append(reversal)
        if not self._resynced:
            self._last_delta = delta.copy()

        counts = np.sum(np.asarray(self._reversal_window, dtype=np.int64), axis=0)
        if np.any(counts > self._osc_reversals):
            self._hold_counter = self._osc_hold_steps
            self._clear_reversals()
            return previous.copy()
        return q_command

    def update_sent(self, q_sent: npt.ArrayLike) -> None:
        """Track the six-joint value actually sent after the absolute clamp."""
        self._q_cmd_prev = np.asarray(q_sent, dtype=np.float64).copy()

    def _full_q(self, q_arm: npt.ArrayLike) -> Vec:
        q = np.asarray(q_arm, dtype=np.float64)
        return np.concatenate((q, self._gripper_pin))

    def _clip_effective(self, q: npt.ArrayLike) -> Vec:
        values = np.asarray(q, dtype=np.float64)
        return np.clip(values, self._effective_ranges[:, 0], self._effective_ranges[:, 1])

    def _extract_relative(self, relative: npt.NDArray[np.float64]) -> tuple[float, float, float]:
        """Decompose a relative rotation as the yaw/pitch/roll of :meth:`solve`.

        Inverse of ``Rz(yaw) @ Ry'(pitch) @ Rx(roll)`` with ``Ry'`` about -y
        (standard ZYX with the pitch sign flipped, so positive pitch reads as
        tool-forward). Singular only at |pitch| = pi/2, which the config
        bounds exclude.
        """
        pitch = float(np.arcsin(np.clip(relative[2, 0], -1.0, 1.0)))
        yaw = float(np.arctan2(relative[1, 0], relative[0, 0]))
        roll = float(np.arctan2(relative[2, 1], relative[2, 2]))
        return self._wrap_yaw(yaw), pitch, self._wrap_yaw(roll)

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        wrapped = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.pi if wrapped <= -np.pi else wrapped)

    @staticmethod
    def _rotation_z(yaw: float) -> npt.NDArray[np.float64]:
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        return np.asarray(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_y_forward(pitch: float) -> npt.NDArray[np.float64]:
        # Rotation about -y: positive pitch tips a gripper-down tool toward +x.
        cosine = np.cos(pitch)
        sine = np.sin(pitch)
        return np.asarray(
            ((cosine, 0.0, -sine), (0.0, 1.0, 0.0), (sine, 0.0, cosine)),
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_x(roll: float) -> npt.NDArray[np.float64]:
        cosine = np.cos(roll)
        sine = np.sin(roll)
        return np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
            dtype=np.float64,
        )

    def _require_command_reference(self) -> Vec:
        if self._q_cmd_prev is None:
            raise RuntimeError("arm command reference has not been seeded")
        return self._q_cmd_prev

    def _require_yaw_reference(self) -> npt.NDArray[np.float64]:
        if self._yaw_reference is None:
            raise RuntimeError("yaw reference has not been captured")
        return self._yaw_reference

    def _clear_reversals(self) -> None:
        self._reversal_window.clear()
        self._last_delta = None
