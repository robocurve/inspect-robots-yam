"""Predict bimanual YAM collisions for absolute joint-position actions.

The checker maps the package's 14-D action contract to a composed MuJoCo
configuration by joint name:

| Action label | MuJoCo joint |
| --- | --- |
| ``left_j0`` .. ``left_j5`` | ``left_joint1`` .. ``left_joint6`` |
| ``right_j0`` .. ``right_j5`` | ``right_joint1`` .. ``right_joint6`` |
| ``left_gripper``, ``right_gripper`` | ignored; both finger joints stay open |

MuJoCo is loaded only when a checker is constructed. Importing this module
therefore keeps the package's optional-dependency boundary intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from importlib import resources
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.errors import SafetyAbort
from inspect_robots.spaces import Box
from inspect_robots.types import Action

from inspect_robots_yam.config import DEFAULT_JOINT_HOME_POSE, YamConfig
from inspect_robots_yam.packing import ARM_DOF, DIM_LABELS, TOTAL_DIM, validate_dim

Vec = npt.NDArray[np.float64]
ViolationMode = Literal["hold", "abort"]

_INSTALL_COMMAND = 'pip install "inspect-robots-yam[collision]"'
_LAST_SAFE_KEY = "yam_collision:last_safe"
_BLOCKED_COUNT_KEY = "yam_collision:blocked_count"
_SIDES = ("left", "right")
_ACTION_TO_MODEL_JOINT: tuple[tuple[str, str], ...] = tuple(
    (f"{side}_j{index}", f"{side}_joint{index + 1}") for side in _SIDES for index in range(ARM_DOF)
)
_FINGER_JOINTS: tuple[str, ...] = tuple(
    f"{side}_{finger}_finger" for side in _SIDES for finger in ("left", "right")
)
_EXPECTED_JOINTS = frozenset(model_name for _, model_name in _ACTION_TO_MODEL_JOINT) | frozenset(
    _FINGER_JOINTS
)
_ARM_ACTION_INDICES = tuple(
    DIM_LABELS.index(f"{side}_j{index}") for side in _SIDES for index in range(ARM_DOF)
)


def _load_mujoco() -> Any:
    """Load MuJoCo on demand or raise with the collision-extra install command."""
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for the YAM collision guardrail. "
            f"Install it with: {_INSTALL_COMMAND}"
        ) from exc
    return mujoco


@dataclass(frozen=True)
class CollisionConfig:
    """Define the measured rig geometry and conservative collision-query policy.

    The base poses are unverified defaults, not physical facts. Every rig must
    measure and override them before relying on cross-arm results.
    """

    left_base_pos: tuple[float, float, float] = (0.0, 0.3, 0.0)
    right_base_pos: tuple[float, float, float] = (0.0, -0.3, 0.0)
    left_base_yaw: float = 0.0
    right_base_yaw: float = 0.0
    table_height: float | None = 0.0
    penetration_threshold: float = 1e-3
    sweep_resolution: float = 0.05
    gripper_qpos: Literal["open", "command"] = "open"
    hold_limit: int | None = 50

    def __post_init__(self) -> None:
        """Reject geometry or query settings that cannot produce sound checks."""
        for name in ("left_base_pos", "right_base_pos"):
            value = getattr(self, name)
            if len(value) != 3 or not bool(np.all(np.isfinite(value))):
                raise ValueError(f"{name} must contain three finite coordinates")
        for name in ("left_base_yaw", "right_base_yaw"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.table_height is not None and not math.isfinite(self.table_height):
            raise ValueError("table_height must be finite or None")
        if not math.isfinite(self.penetration_threshold) or self.penetration_threshold < 0:
            raise ValueError("penetration_threshold must be finite and >= 0")
        if not math.isfinite(self.sweep_resolution) or self.sweep_resolution <= 0:
            raise ValueError("sweep_resolution must be finite and > 0")
        if self.gripper_qpos != "open":
            raise ValueError(
                "gripper_qpos='command' is not supported by collision guardrail v1; "
                "only 'open' is available"
            )
        if self.hold_limit is not None and (
            not isinstance(self.hold_limit, int)
            or isinstance(self.hold_limit, bool)
            or self.hold_limit < 0
        ):
            raise ValueError("hold_limit must be a non-negative integer or None")


_DEFAULT_COLLISION_CONFIG = CollisionConfig()


@dataclass(frozen=True)
class CollisionReport:
    """Describe whether a query penetrated and name the first offending pair."""

    collided: bool
    geom1: str | None = None
    geom2: str | None = None
    distance: float | None = None


class CollisionChecker:
    """Compile one bimanual scene and answer single-threaded collision queries.

    A checker owns one scratch ``MjData`` and is not thread-safe. ``model_xml``
    exists for validation tests and tooling; normal construction resolves the
    packaged collision-only model with :mod:`importlib.resources`.
    """

    def __init__(
        self,
        config: CollisionConfig = _DEFAULT_COLLISION_CONFIG,
        *,
        model_xml: str | None = None,
    ) -> None:
        self.config = config
        self._mujoco = _load_mujoco()
        xml = model_xml if model_xml is not None else self._read_model_xml()
        try:
            spec = self._compose_spec(xml)
            self._model = spec.compile()
        except Exception as exc:
            raise ValueError(
                "CollisionChecker could not compose and compile the YAM collision model; "
                "the XML may be malformed or incompatible with this MuJoCo version"
            ) from exc

        actual_joints = frozenset(
            self._mujoco.mj_id2name(
                self._model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            for joint_id in range(self._model.njnt)
        )
        if actual_joints != _EXPECTED_JOINTS:
            missing = sorted(_EXPECTED_JOINTS - actual_joints)
            unexpected = sorted(actual_joints - _EXPECTED_JOINTS)
            raise ValueError(
                "CollisionChecker model joint names do not match the 14-D YAM mapping; "
                f"missing={missing}, unexpected={unexpected}"
            )

        self._qpos_addresses = {
            name: int(
                self._model.jnt_qposadr[
                    self._mujoco.mj_name2id(
                        self._model,
                        self._mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                ]
            )
            for name in _EXPECTED_JOINTS
        }
        self._open_finger_qpos = self._resolve_open_finger_qpos()
        self._data = self._mujoco.MjData(self._model)

    @staticmethod
    def _read_model_xml() -> str:
        asset = resources.files("inspect_robots_yam").joinpath("assets/yam_collision.xml")
        return asset.read_text(encoding="utf-8")

    def _compose_spec(self, model_xml: str) -> Any:
        parent = self._mujoco.MjSpec()
        if self.config.table_height is not None:
            parent.worldbody.add_geom(
                name="table",
                type=self._mujoco.mjtGeom.mjGEOM_PLANE,
                pos=(0.0, 0.0, self.config.table_height),
                size=(0.0, 0.0, 0.1),
            )
        for side in _SIDES:
            child = self._mujoco.MjSpec.from_string(model_xml)
            frame = parent.worldbody.add_frame()
            frame.pos = getattr(self.config, f"{side}_base_pos")
            yaw = getattr(self.config, f"{side}_base_yaw")
            frame.quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
            frame.attach_body(child.worldbody.first_body(), f"{side}_", "")
        return parent

    def _resolve_open_finger_qpos(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name in _FINGER_JOINTS:
            joint_id = self._mujoco.mj_name2id(
                self._model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            low, high = self._model.jnt_range[joint_id]
            result[name] = float(high if name.endswith("_left_finger") else low)
        return result

    def check(self, action: npt.ArrayLike) -> CollisionReport:
        """Return the first contact deeper than the configured threshold."""
        values = validate_dim(action, TOTAL_DIM)
        if not bool(np.all(np.isfinite(values))):
            raise SafetyAbort("CollisionChecker: action contains a non-finite value")

        by_label = dict(zip(DIM_LABELS, values, strict=True))
        self._data.qpos[:] = 0.0
        for action_label, joint_name in _ACTION_TO_MODEL_JOINT:
            self._data.qpos[self._qpos_addresses[joint_name]] = by_label[action_label]
        for joint_name, qpos in self._open_finger_qpos.items():
            self._data.qpos[self._qpos_addresses[joint_name]] = qpos

        self._mujoco.mj_kinematics(self._model, self._data)
        self._mujoco.mj_collision(self._model, self._data)
        for contact in self._data.contact[: self._data.ncon]:
            distance = float(contact.dist)
            if distance < -self.config.penetration_threshold:
                return CollisionReport(
                    collided=True,
                    geom1=self._geom_name(int(contact.geom1)),
                    geom2=self._geom_name(int(contact.geom2)),
                    distance=distance,
                )
        return CollisionReport(collided=False)

    def _geom_name(self, geom_id: int) -> str:
        name = self._mujoco.mj_id2name(
            self._model,
            self._mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )
        return str(name) if name is not None else f"geom#{geom_id}"


def _validate_action_space(action_space: Box) -> None:
    semantics = action_space.semantics
    mode = None if semantics is None else semantics.control_mode
    if action_space.dim != TOTAL_DIM or mode != "joint_pos":
        raise ValueError(
            "Plan 0011 collision guardrails require a 14-D action space with "
            f"control_mode='joint_pos'; got dim={action_space.dim}, control_mode={mode!r}"
        )


class CollisionApprover:
    """Sweep absolute targets and hold the last safe pose or abort on a hit."""

    def __init__(
        self,
        checker: CollisionChecker,
        start_pose: npt.ArrayLike,
        *,
        action_space: Box,
        on_violation: ViolationMode = "hold",
        hold_limit: int | None = None,
    ) -> None:
        _validate_action_space(action_space)
        if on_violation not in ("hold", "abort"):
            raise ValueError("on_violation must be 'hold' or 'abort'")
        effective_limit = hold_limit if hold_limit is not None else checker.config.hold_limit
        if effective_limit is not None and (
            not isinstance(effective_limit, int)
            or isinstance(effective_limit, bool)
            or effective_limit < 0
        ):
            raise ValueError("hold_limit must be a non-negative integer or None")
        self._checker = checker
        self._start_pose = validate_dim(start_pose, TOTAL_DIM).copy()
        if not bool(np.all(np.isfinite(self._start_pose))):
            raise ValueError("CollisionApprover start_pose must contain only finite values")
        start_report = checker.check(self._start_pose)
        if start_report.collided:
            raise ValueError(
                "CollisionApprover start_pose is already in collision under the configured "
                f"rig model: {self._report_pair(start_report)}. Set collision_guardrail=false "
                "to opt out, or correct the collision_* geometry fields."
            )
        self._on_violation = on_violation
        self._hold_limit = effective_limit

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        """Approve a safe sweep while preserving identity, or reject it visibly."""
        target = validate_dim(action.data, TOTAL_DIM)
        if not bool(np.all(np.isfinite(target))):
            raise SafetyAbort("CollisionApprover: action contains a non-finite value")
        last = np.asarray(store.get(_LAST_SAFE_KEY, self._start_pose), dtype=np.float64)
        substeps = self._substep_count(last, target)
        for step in range(1, substeps + 1):
            waypoint = last + (target - last) * (step / substeps)
            report = self._checker.check(waypoint)
            if report.collided:
                detail = f"{self._report_pair(report)}@{step}/{substeps}"
                if self._on_violation == "abort":
                    raise SafetyAbort(f"CollisionApprover blocked predicted collision: {detail}")
                consecutive = store.get(_BLOCKED_COUNT_KEY, 0) + 1
                store[_BLOCKED_COUNT_KEY] = consecutive
                if (
                    self._hold_limit is not None
                    and self._hold_limit > 0
                    and consecutive >= self._hold_limit
                ):
                    raise SafetyAbort(
                        "CollisionApprover reached consecutive hold limit "
                        f"({consecutive}/{self._hold_limit}): {detail}"
                    )
                meta = dict(action.meta)
                meta.pop("clamped", None)
                meta.pop("delta_clamped", None)
                meta.update(
                    {
                        "collision_blocked": True,
                        "collision_detail": detail,
                    }
                )
                DeltaLimitApprover.rewind_reference(store, last)
                return replace(action, data=last.copy(), meta=meta)
        store[_LAST_SAFE_KEY] = target.copy()
        store[_BLOCKED_COUNT_KEY] = 0
        return action

    def _substep_count(self, start: Vec, target: Vec) -> int:
        arm_indices = list(_ARM_ACTION_INDICES)
        max_delta = float(np.max(np.abs(target[arm_indices] - start[arm_indices])))
        return max(1, min(64, math.ceil(max_delta / self._checker.config.sweep_resolution)))

    @staticmethod
    def _report_pair(report: CollisionReport) -> str:
        return f"{report.geom1}:{report.geom2}"


def _config_from_yam(yam_config: YamConfig) -> CollisionConfig:
    set_fields: dict[str, Any] = {}
    for yam_name, collision_name in (
        ("collision_left_base_pos", "left_base_pos"),
        ("collision_right_base_pos", "right_base_pos"),
        ("collision_left_base_yaw", "left_base_yaw"),
        ("collision_right_base_yaw", "right_base_yaw"),
        ("collision_penetration_threshold", "penetration_threshold"),
    ):
        value = getattr(yam_config, yam_name)
        if value is not None:
            set_fields[collision_name] = value
    set_fields["hold_limit"] = yam_config.collision_hold_limit
    if not yam_config.collision_table:
        set_fields["table_height"] = None
    elif yam_config.collision_table_height is not None:
        set_fields["table_height"] = yam_config.collision_table_height
    return replace(_DEFAULT_COLLISION_CONFIG, **set_fields)


def _collision_approver(
    yam_config: YamConfig,
    action_space: Box,
    *,
    collision_config: CollisionConfig | None = None,
    on_violation: ViolationMode = "hold",
) -> CollisionApprover:
    configured_home = (
        DEFAULT_JOINT_HOME_POSE if yam_config.home_pose is None else yam_config.home_pose
    )
    start_pose = np.clip(
        np.asarray(configured_home, dtype=np.float64),
        yam_config.low,
        yam_config.high,
    )
    cfg = _config_from_yam(yam_config) if collision_config is None else collision_config
    checker = CollisionChecker(cfg)
    return CollisionApprover(
        checker,
        start_pose,
        action_space=action_space,
        on_violation=on_violation,
        hold_limit=cfg.hold_limit,
    )


def build_yam_guardrails(
    action_space: Box,
    yam_config: YamConfig,
    collision_config: CollisionConfig | None = None,
    *,
    on_violation: ViolationMode = "hold",
) -> ChainApprover:
    """Build clamp, delta-limit, and collision gates in execution order."""
    _validate_action_space(action_space)
    if yam_config.control_interface != "joints":
        raise ValueError("Plan 0011 collision guardrails do not support YamConfig EEF mode")
    if yam_config.joints_are_delta:
        raise ValueError(
            "Plan 0011 collision guardrails do not support YamConfig(joints_are_delta=True)"
        )
    return ChainApprover(
        ClampApprover(action_space),
        DeltaLimitApprover(action_space),
        _collision_approver(
            yam_config,
            action_space,
            collision_config=collision_config,
            on_violation=on_violation,
        ),
    )
