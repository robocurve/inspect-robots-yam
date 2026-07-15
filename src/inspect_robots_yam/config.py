"""Configuration for the YAM embodiment and the MolmoAct2 policy client.

Both configs are frozen dataclasses with defaults that match MolmoAct2's
first-party bimanual-YAM server, so zero-arg construction "just works". Each
exposes :meth:`from_kwargs` so the adapters can accept flat scalar keyword
arguments — this is what lets ``inspect-robots run -P server_url=... -E left_channel=...``
configure them, since the Inspect Robots CLI only forwards scalar ``key=value`` pairs.
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

# Operator-confirmed 2026-07-14 against two physical captures of a YAM pair at
# rest: all joint readings were within 0.09 rad of encoder zero, with both
# grippers reading 0.0 — the closed end of the stroke (wire 0 = closed; the
# original capture note said "open" under the pre-0005 inverted doc convention).
# Assumes standard upright mounting; exotic mounts override this pose per rig.
DEFAULT_REST_POSE: tuple[float, ...] = (0.0,) * TOTAL_DIM

# Conservative default per-step displacement limits for joints_are_delta mode:
# 0.2 rad per joint per step, a full normalized stroke per gripper per step.
# Symmetric (the declared delta box is +/-step_limits) because opening and
# closing require opposite-signed deltas. Reusing the absolute [0, 1] box here
# would reject gripper motion in one direction.
_STEP_ARM = (0.2,) * ARM_DOF + (1.0,)
_DEFAULT_STEP_LIMITS = _STEP_ARM * 2


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
        {"joint_low", "joint_high", "home_pose", "rest_pose", "step_limits"}
    )

    left_channel: str = "can0"
    right_channel: str = "can1"
    gripper_type: str = "LINEAR_4310"
    control_hz: float = 10.0
    cam_height: int = 224
    cam_width: int = 224
    joint_low: tuple[float, ...] = _DEFAULT_LOW
    joint_high: tuple[float, ...] = _DEFAULT_HIGH
    # Optional reset target; gripper slots are normalized 0-1 (1 = open).
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
    joints_are_delta: bool = False
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
        """Join the server and endpoint with exactly one separating slash."""
        return self.server_url.rstrip("/") + "/" + self.endpoint.lstrip("/")


DEFAULT_CAMERAS: tuple[str, ...] = ("top_cam", "left_cam", "right_cam")


def action_semantics(joints_are_delta: bool = False) -> ActionSemantics:
    """The action *semantics* both the policy and the embodiment declare.

    Compatibility checking compares control_mode + rotation_repr (errors) and
    gripper + frame (warnings); building both sides from this one function,
    with the same ``joints_are_delta``, guarantees a clean check — and a loud
    one when a delta-configured rig is paired with an absolute-declaring
    policy (or vice versa). ``dim_labels`` names the 14 dims so
    label-addressed tooling (e.g. the LLM agent policy) can move joints by
    name.
    """
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
) -> Box:
    """The shared 14-D action space. ``low``/``high`` are optional limits
    (the embodiment supplies them; the policy leaves them unset): absolute
    joint limits in absolute mode, per-step displacement limits in delta mode."""
    return Box(
        shape=(TOTAL_DIM,),
        low=low,
        high=high,
        semantics=action_semantics(joints_are_delta),
    )


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
