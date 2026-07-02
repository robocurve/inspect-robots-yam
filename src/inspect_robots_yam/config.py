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

from inspect_robots_yam.packing import ARM_DOF, STATE_KEY, STATE_SPEC, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

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
    gripper_open: float = 0.0
    gripper_closed: float = 1.0
    joints_are_delta: bool = False

    def __post_init__(self) -> None:
        for name in ("joint_low", "joint_high"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.home_pose is not None and len(self.home_pose) != TOTAL_DIM:
            raise ValueError(f"home_pose must have {TOTAL_DIM} entries")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.joint_high, dtype=np.float64)


@dataclass(frozen=True)
class MolmoActConfig(_FromKwargs):
    """Static configuration for the MolmoAct2 ``/act`` client."""

    server_url: str = "http://127.0.0.1:8202"
    endpoint: str = "/act"
    num_steps: int = 10
    timeout_s: float = 30.0
    camera_order: tuple[str, ...] = ("top_cam", "left_cam", "right_cam")
    state_key: str = "joint_pos"
    cam_height: int = 224
    cam_width: int = 224

    @property
    def url(self) -> str:
        return self.server_url.rstrip("/") + self.endpoint


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


def observation_space(height: int, width: int, names: tuple[str, ...]) -> ObservationSpace:
    """The shared observation space: three cameras + the packed 14-D ``joint_pos`` state."""
    return ObservationSpace(
        cameras=camera_specs(height, width, names),
        state_keys=frozenset({STATE_KEY}),
        state=STATE_SPEC,
    )
