"""``YAMEmbodiment`` — Inspect Robots embodiment for I2RT YAM bimanual arms.

Wraps the i2rt joint-position driver. Designed for real-robot reality:

* **Safety backstop** — every command is clamped to the configured joint limits
  inside :meth:`step`, *independently* of any Inspect Robots ``Approver`` (so unclamped
  model outputs can never reach the motors).
* **Operator-in-the-loop success** — there is no privileged oracle; when the
  operator signals end-of-episode the embodiment returns
  ``StepResult(terminated=True, termination_reason="success"|"failure")``, which is
  the only path that reaches the scorer.
* **Self-paced** — declares ``SELF_PACED`` and sleeps to the control rate inside
  :meth:`step` (the framework does not pace for us).

Hardware/driver access is injected (``driver_factory``, ``camera_reader``,
``operator``, ``poll_end``, ``sleep_fn``, ``clock``) so the whole embodiment runs
in tests with no CAN bus, no cameras, and no stdin. The real driver/camera seams
are pragma'd defaults that only execute on hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_yam import packing
from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig, action_box, observation_space
from inspect_robots_yam.operator import OperatorIO, default_poll_end

ImageMap = Mapping[str, npt.NDArray[np.uint8]]
Vec = npt.NDArray[np.float64]


@runtime_checkable
class BimanualDriver(Protocol):
    """The minimal 14-D joint-position driver the embodiment needs."""

    def get_joint_pos(self) -> npt.NDArray[np.floating[Any]]: ...

    def command_joint_pos(self, target: npt.NDArray[np.floating[Any]]) -> None: ...

    def close(self) -> None: ...


DriverFactory = Callable[[YamConfig], BimanualDriver]
CameraReader = Callable[[YamConfig], ImageMap]


def _default_driver_factory(cfg: YamConfig) -> BimanualDriver:  # pragma: no cover - real hardware
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    # NAME lookup (GripperType["LINEAR_4310"]) — the enum *values* are lowercase
    # strings, so GripperType(...)/from_string_name would reject the config names.
    # YamConfig.__post_init__ already validated the name against the supported set.
    gripper = GripperType[cfg.gripper_type]
    left = get_yam_robot(
        channel=cfg.left_channel,
        gripper_type=gripper,
        zero_gravity_mode=cfg.zero_gravity_mode,
    )
    right = get_yam_robot(
        channel=cfg.right_channel,
        gripper_type=gripper,
        zero_gravity_mode=cfg.zero_gravity_mode,
    )

    class _Real:
        def get_joint_pos(self) -> npt.NDArray[np.floating[Any]]:
            return packing.pack(left.get_joint_pos(), right.get_joint_pos())

        def command_joint_pos(self, target: npt.NDArray[np.floating[Any]]) -> None:
            lo, ro = packing.split(target)
            left.command_joint_pos(lo)
            right.command_joint_pos(ro)

        def close(self) -> None:
            for arm in (left, right):
                closer = getattr(arm, "close", None)
                if callable(closer):
                    closer()

    return _Real()


def _default_status(line: str | None) -> None:  # pragma: no cover - real TTY output
    """Rewrite one status line in place; ``None`` closes it with a newline."""
    if line is None:
        print(flush=True)
    else:
        print(f"\r  {line}   ", end="", flush=True)


def _opencv_camera_reader(cfg: YamConfig) -> CameraReader:
    """Builtin V4L2 reader for rigs configured via ``*_cam_device`` (YamConfig).

    cv2 is imported lazily on the first frame read, so construction stays inert
    and the package still imports without OpenCV installed. Negotiates YUYV at
    640x480 explicitly (RealSense D435s return empty frames on cv2 defaults)
    and resizes to ``cam_width`` x ``cam_height`` RGB.
    """
    devices = {
        "top_cam": cfg.top_cam_device,
        "left_cam": cfg.left_cam_device,
        "right_cam": cfg.right_cam_device,
    }
    caps: dict[str, Any] = {}

    def reader(cfg: YamConfig) -> ImageMap:  # pragma: no cover - real cameras
        import time as _time

        import cv2

        if not caps:
            for name, dev in devices.items():
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open {name} at {dev}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
                for _ in range(10):  # warm up: first frames can be empty
                    if cap.read()[0]:
                        break
                    _time.sleep(0.1)
                caps[name] = cap
        out: dict[str, npt.NDArray[np.uint8]] = {}
        for name, cap in caps.items():
            frame = None
            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                _time.sleep(0.05)
            if frame is None:
                raise RuntimeError(f"frame read failed for {name} ({devices[name]})")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (cfg.cam_width, cfg.cam_height))
            out[name] = frame.astype(np.uint8)
        return out

    return reader


def _default_camera_reader(cfg: YamConfig) -> ImageMap:
    raise NotImplementedError(
        "provide a camera_reader returning {'top_cam','left_cam','right_cam': HxWx3 uint8}"
    )


class YAMEmbodiment:
    """Inspect Robots embodiment for bimanual YAM arms (joint-position control)."""

    def __init__(
        self,
        config: YamConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        status_fn: Callable[[str | None], None] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else YamConfig.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        if camera_reader is None and self._cfg.top_cam_device is not None:
            # All three device paths are set (YamConfig validates all-or-none):
            # use the builtin OpenCV reader, so config/CLI-only setups work.
            camera_reader = _opencv_camera_reader(self._cfg)
        self._camera_reader: CameraReader = camera_reader or _default_camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._status: Callable[[str | None], None] = status_fn or _default_status

        self._driver: BimanualDriver | None = None
        self._instruction: str | None = None
        self._t_last = 0.0
        self.num_steps = 0

        self.info = EmbodimentInfo(
            name="yam_arms",
            # Delta mode declares the per-step displacement box (symmetric,
            # honest for guardrail derivation); the absolute joint limits stay
            # enforced on the SUMMED command inside _send() either way.
            action_space=(
                action_box(
                    low=self._cfg.delta_low, high=self._cfg.delta_high, joints_are_delta=True
                )
                if self._cfg.joints_are_delta
                else action_box(low=self._cfg.low, high=self._cfg.high)
            ),
            observation_space=observation_space(
                self._cfg.cam_height, self._cfg.cam_width, DEFAULT_CAMERAS
            ),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect (if needed), drive to home, and block on operator readiness."""
        # Fail fast on an unusable camera_reader BEFORE connecting the driver or
        # commanding any motion: this is a pure configuration error. `not callable`
        # also catches a CLI-injected scalar (`-E camera_reader=...` binds a str).
        if self._camera_reader is _default_camera_reader or not callable(self._camera_reader):
            raise ConfigError(
                "yam_arms has no cameras configured. Set the three V4L2 device "
                "paths - top/left/right_cam_device - in YamConfig, config.ini "
                "([embodiment.args]) or the CLI (-E top_cam_device=/dev/video0 "
                "...) to use the builtin OpenCV reader, or provide a custom "
                "camera_reader= via the Python API."
            )
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)

        driver = self._require_driver()
        current = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        target = (
            np.asarray(self._cfg.home_pose, dtype=np.float64)
            if self._cfg.home_pose is not None
            else current
        )
        self._ramp_to(target)
        if not self._cfg.unattended:
            self._operator.wait_ready()
            horizon = self._horizon_secs()
            limit = f" Max {horizon:.0f}s." if horizon is not None else ""
            self._status(f"Running: press any key to end the episode, then y/N to score.{limit}")
        self._instruction = scene.instruction
        self.num_steps = 0
        self._t_last = self._clock()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp + command one action, pace to the control rate, then maybe end."""
        driver = self._require_driver()
        self.num_steps += 1
        cmd = packing.validate_dim(action.data)
        base = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        if self._cfg.joints_are_delta:
            # Normalize the gripper slots of the current position first, so the
            # delta is applied in policy units (a fraction of the gripper stroke)
            # and the sum re-enters _send() in the same units as absolute mode.
            cmd = base + cmd
        self._send(cmd, base=base)
        self._pace()
        self._emit_status()

        obs = self._observe(self._instruction)
        # Unattended runs have no operator: skip the end poll and its success
        # prompt entirely; the episode runs to the framework's max_steps.
        if not self._cfg.unattended and self._poll_end():
            self._status(None)  # close the status line before the y/N prompt
            success = self._operator.confirm_success()
            return StepResult(
                observation=obs,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=obs, terminated=False)

    def close(self) -> None:
        """Park at ``rest_pose`` (if configured), then release the driver handles.

        Releasing the driver zeroes motor torque, so whatever pose the arms are
        in when it happens is the pose they fall from — the rest ramp runs
        first so torque-off is harmless. The release lives in a ``finally`` so
        a driver fault mid-ramp can never leave the handles held. No-op if
        never connected.
        """
        if self._driver is None:
            return
        try:
            if self._cfg.rest_pose is not None:
                self._ramp_to(np.asarray(self._cfg.rest_pose, dtype=np.float64))
        finally:
            self._driver.close()
            self._driver = None

    def _ramp_to(self, target: Vec) -> None:
        """Linearly ramp from the current pose to ``target`` over ``rest_secs``.

        Used for both homing (reset) and parking (close): a single raw jump to
        a distant pose is violent on real arms. Each waypoint goes through
        :meth:`_send`, so the joint-limit clamp and gripper de-normalization
        apply to these motions exactly as they do to policy actions.
        """
        driver = self._require_driver()
        start = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        hz = self._cfg.control_hz if self._cfg.control_hz > 0 else 10.0
        n = max(1, round(self._cfg.rest_secs * hz))
        for i in range(1, n + 1):
            alpha = i / n
            self._send((1.0 - alpha) * start + alpha * target)
            self._sleep(1.0 / hz)

    # -- internals ---------------------------------------------------------

    def _horizon_secs(self) -> float | None:
        """The episode horizon in seconds, if max_steps_hint is configured."""
        hint = self._cfg.max_steps_hint
        hz = self._cfg.control_hz
        if hint is None or not hz or hz <= 0:
            return None
        return hint / hz

    def _emit_status(self) -> None:
        """Once per second (of control time), tell the operator where they are."""
        if self._cfg.unattended:
            return
        hz = self._cfg.control_hz if self._cfg.control_hz > 0 else 10.0
        interval = max(1, round(hz))
        if self.num_steps % interval != 0:
            return
        elapsed = self.num_steps / hz
        horizon = self._horizon_secs()
        span = f"{elapsed:.0f}s / {horizon:.0f}s" if horizon is not None else f"{elapsed:.0f}s"
        self._status(f"t = {span} | any key ends the episode")

    def _require_driver(self) -> BimanualDriver:
        # Reachable: step() before the first reset(), or after close().
        if self._driver is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._driver

    def _send(self, cmd: Vec, base: Vec | None = None) -> None:
        """Clamp to step limits (safety backstop) and de-normalize grippers."""
        driver = self._require_driver()
        current = base if base is not None else self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        clamped = np.clip(cmd, self._cfg.low, self._cfg.high)
        clamped = np.clip(clamped, current + self._cfg.delta_low, current + self._cfg.delta_high)
        physical = self._denorm_grippers(clamped)
        driver.command_joint_pos(physical)

    def _denorm_grippers(self, cmd: Vec) -> Vec:
        out: Vec = cmd.copy()
        span = self._cfg.gripper_closed - self._cfg.gripper_open
        for idx in (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF):  # 6, 13
            out[idx] = self._cfg.gripper_open + cmd[idx] * span
        return out

    def _norm_grippers(self, physical: Vec) -> Vec:
        """Exact inverse of :meth:`_denorm_grippers` (driver units -> normalized 0-1).

        ``YamConfig.__post_init__`` guarantees ``gripper_open != gripper_closed``,
        so the span is never zero.
        """
        out: Vec = physical.copy()
        span = self._cfg.gripper_closed - self._cfg.gripper_open
        for idx in (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF):  # 6, 13
            out[idx] = (physical[idx] - self._cfg.gripper_open) / span
        return out

    def _pace(self) -> None:
        hz = self._cfg.control_hz
        if hz and hz > 0:
            elapsed = self._clock() - self._t_last
            self._sleep(max(0.0, 1.0 / hz - elapsed))
        self._t_last = self._clock()

    def _observe(self, instruction: str | None) -> Observation:
        driver = self._require_driver()
        # Normalize the gripper slots back to 0-1 so the observed state is in the
        # exact units STATE_SPEC declares (and _send() accepts) — the inverse of
        # the de-normalization applied to outgoing commands.
        state = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        return Observation(
            images=dict(self._camera_reader(self._cfg)),
            state={packing.STATE_KEY: state},
            instruction=instruction,
        )
