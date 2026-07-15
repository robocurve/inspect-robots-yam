"""Configuration for the YAM embodiment and the generic ``/act`` policy client.

Both configs are frozen dataclasses. The client defaults match MolmoAct2's
first-party bimanual-YAM server, so zero-arg construction "just works". Each
exposes :meth:`from_kwargs` so the adapters can accept flat scalar keyword
arguments. This is what lets ``inspect-robots run -P server_url=...
-E left_channel=...`` configure them, since the Inspect Robots CLI only
forwards scalar ``key=value`` pairs.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

from inspect_robots_yam.packing import ARM_DOF, DIM_LABELS, STATE_KEY, TOTAL_DIM, state_spec

_T = TypeVar("_T", bound="_FromKwargs")

# The i2rt gripper variants this adapter supports. i2rt also defines NO_GRIPPER and
# YAM_TEACHING_HANDLE, but those change the per-arm DOF and would silently break the
# 7-D-per-arm / 14-D packing contract, so they are rejected eagerly. Names match the
# ``i2rt.robots.utils.GripperType`` enum *names* (the enum values are lowercase).
SUPPORTED_GRIPPER_TYPES = frozenset({"CRANK_4310", "LINEAR_3507", "LINEAR_4310", "FLEXIBLE_4310"})

# Conservative default action limits: revolute joints in [-pi, pi], gripper in
# [0, 1]. These are SAFETY limits — override with the real YAM joint limits before
# trusting them on hardware.
_ARM_LOW = (-np.pi,) * ARM_DOF + (0.0,)
_ARM_HIGH = (np.pi,) * ARM_DOF + (1.0,)
_DEFAULT_LOW = _ARM_LOW * 2
_DEFAULT_HIGH = _ARM_HIGH * 2

# Dataset-verified MolmoAct2-BimanualYAM start pose (2,260 episodes,
# 2026-07-14 audit): joints within noise of encoder zero, both grippers open
# in effectively every episode start. Joints match the physically captured
# rest; the gripper slots are commanded open (1.0) rather than the captured
# closed reading so episodes begin in the training distribution.
# Assumes standard upright mounting; exotic mounts override per rig.
_JOINT_HOME_ARM = (0.0,) * ARM_DOF + (1.0,)
DEFAULT_JOINT_HOME_POSE: tuple[float, ...] = _JOINT_HOME_ARM * 2

# Park target == home target: the next episode starts in distribution with
# no gripper re-open transient. (Torque release after parking still lets
# the arms sag slightly, and reset always re-runs the homing ramp.)
DEFAULT_REST_POSE: tuple[float, ...] = DEFAULT_JOINT_HOME_POSE

# Conservative default per-step displacement limits for joints_are_delta mode:
# 0.2 rad per joint per step, a full normalized stroke per gripper per step.
# Symmetric (the declared delta box is +/-step_limits) because opening and
# closing require opposite-signed deltas. Reusing the absolute [0, 1] box here
# would reject gripper motion in one direction.
_STEP_ARM = (0.2,) * ARM_DOF + (1.0,)
_DEFAULT_STEP_LIMITS = _STEP_ARM * 2

EEF_DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{part}" for side in ("left", "right") for part in ("x", "y", "z", "yaw", "gripper")
)
_EEF_ARM_LOW = (0.15, -0.25, 0.03, -np.pi, 0.0)
_EEF_ARM_HIGH = (0.48, 0.25, 0.40, np.pi, 1.0)
DEFAULT_EEF_LOW: tuple[float, ...] = _EEF_ARM_LOW * 2
DEFAULT_EEF_HIGH: tuple[float, ...] = _EEF_ARM_HIGH * 2

# Provisional 2026-07-14 LINEAR_4310 solution for EEF position (0.30, 0, 0.20),
# jaw axis pitched 30 degrees from vertical toward the arm base, and open grippers.
_EEF_HOME_ARM = (-0.024, 0.794, 0.645, -0.375, -0.021, -0.012, 1.0)
DEFAULT_EEF_HOME_POSE: tuple[float, ...] = _EEF_HOME_ARM * 2


class _FromKwargs:
    """Mixin: build a frozen dataclass from flat scalar kwargs (CLI-friendly).

    Fields named in ``_FLOAT_TUPLE_FIELDS`` additionally accept a
    comma-separated string ("0.1,0.2,...") so pose-shaped tuples are
    settable from ``-E key=value`` flags and config.ini, which only carry
    scalars.
    """

    _FLOAT_TUPLE_FIELDS: frozenset[str] = frozenset()

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        for key in cls._FLOAT_TUPLE_FIELDS & set(flat):
            value = flat[key]
            if isinstance(value, str):
                try:
                    flat[key] = tuple(float(part) for part in value.split(","))
                except ValueError:
                    raise ValueError(
                        f"{key} must be a comma-separated list of numbers, got {value!r}"
                    ) from None
        return cls(**flat)


@dataclass(frozen=True)
class YamConfig(_FromKwargs):
    """Static configuration for a bimanual YAM embodiment."""

    _FLOAT_TUPLE_FIELDS = frozenset(
        {
            "joint_low",
            "joint_high",
            "eef_low",
            "eef_high",
            "home_pose",
            "rest_pose",
            "step_limits",
        }
    )

    left_channel: str = "can0"
    right_channel: str = "can1"
    gripper_type: str = "LINEAR_4310"
    control_hz: float = 10.0
    cam_height: int = 224
    cam_width: int = 224
    joint_low: tuple[float, ...] = _DEFAULT_LOW
    joint_high: tuple[float, ...] = _DEFAULT_HIGH
    control_interface: str = "joints"
    # Operator-supplied rig-specific notes appended to the built-in agent docs.
    docs_extra: str = ""
    eef_low: tuple[float, ...] = DEFAULT_EEF_LOW
    eef_high: tuple[float, ...] = DEFAULT_EEF_HIGH
    ik_max_iters: int = 20
    ik_step_joint_limit: float = 0.2
    cmd_resync_threshold: float = 0.35
    osc_deadband: float = 0.005
    osc_reversals: int = 2
    osc_window: int = 6
    osc_hold_steps: int = 10
    # Reset target; None selects the per-mode factory default
    # (DEFAULT_JOINT_HOME_POSE / DEFAULT_EEF_HOME_POSE).
    # Gripper slots are normalized 0-1 (1 = open).
    home_pose: tuple[float, ...] | None = None
    # Pose used to park on close() after reset() captures the initial pose. None
    # opts out of the factory target and parks at that captured pose instead.
    # Gripper slots are normalized 0-1 (1 = open).
    rest_pose: tuple[float, ...] | None = DEFAULT_REST_POSE
    rest_secs: float = 3.0
    # Driver-native positions at the stroke endpoints. Wire gripper 1 maps to
    # gripper_open, and wire gripper 0 maps to gripper_closed.
    gripper_open: float = 1.0
    gripper_closed: float = 0.0
    joints_are_delta: bool = False
    step_limits: tuple[float, ...] = _DEFAULT_STEP_LIMITS
    zero_gravity_mode: bool = True
    unattended: bool = False
    # DEPRECATED fallback: framework-driven runs now supply the real horizon
    # via the embodiment's bind_task hook, so the countdown needs no config.
    # Only consulted when the hook never fires (direct rollout(), or a core
    # that predates it). Display-only; bounds nothing. Removal in a later
    # release.
    max_steps_hint: int | None = None
    # Builtin OpenCV camera reader: set ALL THREE to your rig's V4L2 color
    # nodes (stable udev paths recommended; /dev/videoN reshuffles on replug)
    # and yam_arms works from the CLI/config with no custom camera factory.
    # Plain strings, so `-E top_cam_device=...` and config.ini can carry them.
    # Uses the base opencv-python-headless dependency.
    top_cam_device: str | None = None
    left_cam_device: str | None = None
    right_cam_device: str | None = None

    def __post_init__(self) -> None:
        """Reject values that violate the 14-D packing and hardware invariants.

        Pose and step vectors must span both arms, every step limit must be
        finite and positive, the gripper stroke must be nonzero, and builtin
        camera device paths must be configured all together or not at all.
        """
        if self.gripper_type not in SUPPORTED_GRIPPER_TYPES:
            raise ValueError(
                f"gripper_type {self.gripper_type!r} is not supported; expected one of "
                f"{sorted(SUPPORTED_GRIPPER_TYPES)} (i2rt GripperType enum names)"
            )
        valid_interfaces = {"eef_pos", "joints"}
        if self.control_interface not in valid_interfaces:
            raise ValueError(
                f"control_interface must be one of {sorted(valid_interfaces)}, "
                f"got {self.control_interface!r}"
            )
        if self.control_interface == "eef_pos" and self.joints_are_delta:
            raise ValueError(
                "joints_are_delta=True is incompatible with control_interface='eef_pos'"
            )
        for name in ("joint_low", "joint_high"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        for name in ("eef_low", "eef_high"):
            if len(getattr(self, name)) != len(EEF_DIM_LABELS):
                raise ValueError(f"{name} must have {len(EEF_DIM_LABELS)} entries")
        eef_low = self.eef_low_array
        eef_high = self.eef_high_array
        if not np.all(np.isfinite(eef_low)) or not np.all(np.isfinite(eef_high)):
            raise ValueError("eef_low and eef_high must contain only finite values")
        if np.any(eef_low >= eef_high):
            raise ValueError("eef_low must be below eef_high in every dimension")
        for yaw_index in (3, 8):
            if eef_low[yaw_index] < -np.pi or eef_high[yaw_index] > np.pi:
                raise ValueError("eef yaw bounds must stay within [-pi, pi]")
        if (
            not isinstance(self.ik_max_iters, int)
            or isinstance(self.ik_max_iters, bool)
            or self.ik_max_iters <= 0
        ):
            raise ValueError("ik_max_iters must be a positive integer")
        for name in ("ik_step_joint_limit", "cmd_resync_threshold"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not np.isfinite(self.osc_deadband) or self.osc_deadband < 0:
            raise ValueError("osc_deadband must be finite and >= 0")
        if (
            not isinstance(self.osc_reversals, int)
            or isinstance(self.osc_reversals, bool)
            or self.osc_reversals < 0
        ):
            raise ValueError("osc_reversals must be a non-negative integer")
        if (
            not isinstance(self.osc_window, int)
            or isinstance(self.osc_window, bool)
            or self.osc_window <= 0
        ):
            raise ValueError("osc_window must be a positive integer")
        if self.osc_reversals >= self.osc_window:
            raise ValueError("osc_reversals must be less than osc_window")
        if (
            not isinstance(self.osc_hold_steps, int)
            or isinstance(self.osc_hold_steps, bool)
            or self.osc_hold_steps <= 0
        ):
            raise ValueError("osc_hold_steps must be a positive integer")
        if self.home_pose is not None and len(self.home_pose) != TOTAL_DIM:
            raise ValueError(f"home_pose must have {TOTAL_DIM} entries")
        if self.rest_pose is not None and len(self.rest_pose) != TOTAL_DIM:
            raise ValueError(f"rest_pose must have {TOTAL_DIM} entries")
        if self.rest_secs <= 0:
            raise ValueError("rest_secs must be > 0")
        if self.max_steps_hint is not None and self.max_steps_hint < 1:
            raise ValueError("max_steps_hint must be >= 1")
        devices = (self.top_cam_device, self.left_cam_device, self.right_cam_device)
        if any(d is not None for d in devices) and not all(d is not None for d in devices):
            raise ValueError(
                "camera devices must be set all three or none "
                "(top_cam_device, left_cam_device, right_cam_device)"
            )
        if len(self.step_limits) != TOTAL_DIM or any(
            not (s > 0) or not np.isfinite(s) for s in self.step_limits
        ):
            raise ValueError(f"step_limits must be {TOTAL_DIM} finite positive entries")
        if self.gripper_open == self.gripper_closed:
            raise ValueError(
                "gripper_open and gripper_closed must differ (the gripper stroke "
                "would be zero and observations could not be normalized)"
            )
        # Last, after every validation: an invalid config raises without ever
        # warning. FutureWarning (not DeprecationWarning) so operators running
        # the console script actually see it under Python's default filters.
        if self.max_steps_hint is not None:
            warnings.warn(
                "max_steps_hint is deprecated: framework-driven runs supply the "
                "real horizon via the bind_task hook, so the countdown needs no "
                "config. The hint is only used when the hook never fires "
                "(direct rollout(), or a core that predates it).",
                FutureWarning,
                stacklevel=2,
            )

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return absolute lower bounds in radians and normalized gripper units."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return absolute upper bounds in radians and normalized gripper units."""
        return np.asarray(self.joint_high, dtype=np.float64)

    @property
    def delta_low(self) -> npt.NDArray[np.float64]:
        """Return negative per-step limits in radians and normalized gripper units."""
        return -np.asarray(self.step_limits, dtype=np.float64)

    @property
    def delta_high(self) -> npt.NDArray[np.float64]:
        """Return positive per-step limits in radians and normalized gripper units."""
        return np.asarray(self.step_limits, dtype=np.float64)

    @property
    def eef_low_array(self) -> npt.NDArray[np.float64]:
        """Return Cartesian lower bounds in metres, radians, and gripper units."""
        return np.asarray(self.eef_low, dtype=np.float64)

    @property
    def eef_high_array(self) -> npt.NDArray[np.float64]:
        """Return Cartesian upper bounds in metres, radians, and gripper units."""
        return np.asarray(self.eef_high, dtype=np.float64)


@dataclass(frozen=True)
class ActServerConfig(_FromKwargs):
    """Static configuration for the generic ``/act`` policy client.

    ``num_steps`` is the wire-protocol field of the same name: the number of
    flow-matching **denoising steps** the server runs per inference
    (``predict_action(num_steps=...)`` → ``flow_matching_num_steps``). It does
    NOT control how many actions come back. ``action_horizon`` is the
    checkpoint's advertised chunk length, surfaced as
    :class:`~inspect_robots.policy.PolicyConfig` metadata; the actual length is
    always taken from the server's response. Its 30-step default belongs to
    MolmoAct2's bimanual-YAM checkpoint. ``name`` labels the policy in eval logs.
    """

    server_url: str = "http://127.0.0.1:8202"
    joints_are_delta: bool = False
    endpoint: str = "/act"
    num_steps: int = 10
    action_horizon: int = 30
    timeout_s: float = 30.0
    camera_order: tuple[str, ...] = ("top_cam", "left_cam", "right_cam")
    state_key: str = "joint_pos"
    cam_height: int = 224
    cam_width: int = 224
    name: str = "molmoact2"

    @property
    def url(self) -> str:
        """Join the server and endpoint with exactly one separating slash."""
        return self.server_url.rstrip("/") + "/" + self.endpoint.lstrip("/")


MolmoActConfig = ActServerConfig


DEFAULT_CAMERAS: tuple[str, ...] = ("top_cam", "left_cam", "right_cam")


def action_semantics(
    joints_are_delta: bool = False, *, control_interface: str = "joints"
) -> ActionSemantics:
    """The action *semantics* both the policy and the embodiment declare.

    Compatibility checking compares control_mode + rotation_repr (errors) and
    gripper + frame (warnings); building both sides from this one function,
    with the same ``joints_are_delta``, guarantees a clean check — and a loud
    one when a delta-configured rig is paired with an absolute-declaring
    policy (or vice versa). ``dim_labels`` names the 14 dims so
    label-addressed tooling (e.g. the LLM agent policy) can move joints by
    name.
    """
    if control_interface == "eef_pos":
        return ActionSemantics(
            control_mode="eef_abs_pose",
            rotation_repr="none",
            gripper="continuous",
            frame="base",
            dim_labels=EEF_DIM_LABELS,
        )
    return ActionSemantics(
        control_mode="joint_delta" if joints_are_delta else "joint_pos",
        rotation_repr="none",
        gripper="continuous",
        frame="base",
        dim_labels=DIM_LABELS,
    )


def camera_specs(height: int, width: int, names: tuple[str, ...]) -> tuple[CameraSpec, ...]:
    """Build CameraSpecs for the given names at one resolution (single source of truth)."""
    return tuple(CameraSpec(name=n, height=height, width=width, channels=3) for n in names)


def action_box(
    low: npt.NDArray[np.float64] | None = None,
    high: npt.NDArray[np.float64] | None = None,
    *,
    joints_are_delta: bool = False,
    control_interface: str = "joints",
) -> Box:
    """Build the selected joint or Cartesian action space with optional limits.

    The embodiment supplies bounds while the policy may leave them unset. Joint
    delta mode uses per-step displacement limits; all other modes use absolute
    limits.
    """
    return Box(
        shape=((len(EEF_DIM_LABELS),) if control_interface == "eef_pos" else (TOTAL_DIM,)),
        low=low,
        high=high,
        semantics=action_semantics(joints_are_delta, control_interface=control_interface),
    )


def observation_space(
    height: int,
    width: int,
    names: tuple[str, ...],
    state_key: str = STATE_KEY,
    *,
    control_interface: str = "joints",
) -> ObservationSpace:
    """Build the camera and proprioception contract for the selected interface.

    ``state_key`` drives *both* ``state_keys`` and the ``StateSpec`` field key so
    joint mode stays internally consistent for any configured key. Cartesian
    mode additionally declares its 10-D ``eef_state`` reference.
    """
    state = state_spec(state_key)
    if control_interface == "eef_pos":
        state = StateSpec(
            fields=(
                *state.fields,
                StateField(
                    key="eef_state",
                    shape=(len(EEF_DIM_LABELS),),
                    unit="m+rad+normalized",
                ),
            )
        )
    return ObservationSpace(
        cameras=camera_specs(height, width, names),
        state_keys=state.keys,
        state=state,
    )
