"""Configuration for the YAM embodiment and the MolmoAct2 policy client.

Both configs are frozen dataclasses with defaults that match MolmoAct2's
first-party bimanual-YAM server, so zero-arg construction "just works". Each
exposes :meth:`from_kwargs` so the adapters can accept flat scalar keyword
arguments — this is what lets ``inspect-robots run -P server_url=... -E left_channel=...``
configure them, since the Inspect Robots CLI only forwards scalar ``key=value`` pairs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
)

from inspect_robots_yam.packing import ARM_DOF, STATE_KEY, TOTAL_DIM, state_spec

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


class _FromKwargs:
    """Mixin: build a frozen dataclass from flat scalar kwargs (CLI-friendly)."""

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        return cls(**flat)


@dataclass(frozen=True)
class YamConfig(_FromKwargs):
    """Static configuration for a bimanual YAM embodiment."""

    left_channel: str = "can0"
    right_channel: str = "can1"
    gripper_type: str = "LINEAR_4310"
    control_hz: float = 10.0
    cam_height: int = 224
    cam_width: int = 224
    joint_low: tuple[float, ...] = _DEFAULT_LOW
    joint_high: tuple[float, ...] = _DEFAULT_HIGH
    home_pose: tuple[float, ...] | None = None
    # Pose the arms ramp to on close() BEFORE torque is released, so they don't
    # fall. Same units as home_pose/actions: gripper slots normalized 0-1.
    rest_pose: tuple[float, ...] | None = None
    rest_secs: float = 3.0
    gripper_open: float = 0.0
    gripper_closed: float = 1.0
    joints_are_delta: bool = False
    zero_gravity_mode: bool = True
    unattended: bool = False
    # Display-only hint of the framework's episode horizon (the Task/CLI owns
    # the real max_steps): lets the operator status line show elapsed/total.
    # Bounds nothing.
    max_steps_hint: int | None = None
    # Builtin OpenCV camera reader: set ALL THREE to your rig's V4L2 color
    # nodes (stable udev paths recommended; /dev/videoN reshuffles on replug)
    # and yam_arms works from the CLI/config with no custom camera factory.
    # Plain strings, so `-E top_cam_device=...` and config.ini can carry them.
    # Needs opencv-python-headless (the [cameras] extra).
    top_cam_device: str | None = None
    left_cam_device: str | None = None
    right_cam_device: str | None = None

    def __post_init__(self) -> None:
        if self.gripper_type not in SUPPORTED_GRIPPER_TYPES:
            raise ValueError(
                f"gripper_type {self.gripper_type!r} is not supported; expected one of "
                f"{sorted(SUPPORTED_GRIPPER_TYPES)} (i2rt GripperType enum names)"
            )
        for name in ("joint_low", "joint_high"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
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
        if self.gripper_open == self.gripper_closed:
            raise ValueError(
                "gripper_open and gripper_closed must differ (the gripper stroke "
                "would be zero and observations could not be normalized)"
            )

    @property
    def low(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.joint_high, dtype=np.float64)


@dataclass(frozen=True)
class MolmoActConfig(_FromKwargs):
    """Static configuration for the MolmoAct2 ``/act`` client.

    ``num_steps`` is the wire-protocol field of the same name: the number of
    flow-matching **denoising steps** the server runs per inference
    (``predict_action(num_steps=...)`` → ``flow_matching_num_steps``). It does
    NOT control how many actions come back — the chunk length is fixed by the
    checkpoint's norm stats (``action_horizon``/``n_action_steps``, 30 for
    ``yam_dual_molmoact2``). ``action_horizon`` here is that advertised chunk
    length, surfaced as :class:`~inspect_robots.policy.PolicyConfig` metadata;
    the actual length is always taken from the server's response.
    """

    server_url: str = "http://127.0.0.1:8202"
    endpoint: str = "/act"
    num_steps: int = 10
    action_horizon: int = 30
    timeout_s: float = 30.0
    camera_order: tuple[str, ...] = ("top_cam", "left_cam", "right_cam")
    state_key: str = "joint_pos"
    cam_height: int = 224
    cam_width: int = 224

    @property
    def url(self) -> str:
        return self.server_url.rstrip("/") + "/" + self.endpoint.lstrip("/")


DEFAULT_CAMERAS: tuple[str, ...] = ("top_cam", "left_cam", "right_cam")

# The action *semantics* both the policy and the embodiment declare. Compatibility
# checking compares control_mode + rotation_repr (errors) and gripper + frame
# (warnings); declaring this single constant on both sides guarantees a clean check.
ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
)


def camera_specs(height: int, width: int, names: tuple[str, ...]) -> tuple[CameraSpec, ...]:
    """Build CameraSpecs for the given names at one resolution (single source of truth)."""
    return tuple(CameraSpec(name=n, height=height, width=width, channels=3) for n in names)


def action_box(
    low: npt.NDArray[np.float64] | None = None,
    high: npt.NDArray[np.float64] | None = None,
) -> Box:
    """The shared 14-D joint-position action space. ``low``/``high`` are optional
    safety limits (the embodiment supplies them; the policy leaves them unset)."""
    return Box(shape=(TOTAL_DIM,), low=low, high=high, semantics=ACTION_SEMANTICS)


def observation_space(
    height: int, width: int, names: tuple[str, ...], state_key: str = STATE_KEY
) -> ObservationSpace:
    """The shared observation space: three cameras + the packed 14-D state.

    ``state_key`` drives *both* ``state_keys`` and the ``StateSpec`` field key so
    the space stays internally consistent for any configured key.
    """
    return ObservationSpace(
        cameras=camera_specs(height, width, names),
        state_keys=frozenset({state_key}),
        state=state_spec(state_key),
    )
