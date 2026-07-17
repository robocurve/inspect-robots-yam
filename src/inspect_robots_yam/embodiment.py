"""``YAMEmbodiment`` — Inspect Robots embodiment for I2RT YAM bimanual arms.

Wraps the i2rt joint-position driver. Designed for real-robot reality:

* **Safety backstop** — every command is clamped to the configured joint and step limits
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
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_yam import packing
from inspect_robots_yam._i2rt import (
    I2RT_INSTALL_COMMAND,
    _load_i2rt,
    _load_i2rt_kinematics,
    close_robot_safely,
)
from inspect_robots_yam.config import (
    DEFAULT_CAMERAS,
    DEFAULT_EEF_HOME_POSE,
    EEF_DIM_LABELS,
    YamConfig,
    action_box,
    observation_space,
)
from inspect_robots_yam.kinematics import RawKinematics, _ArmKinematics
from inspect_robots_yam.operator import OperatorIO, default_poll_end

ImageMap = Mapping[str, npt.NDArray[np.uint8]]
Vec = npt.NDArray[np.float64]

_DOCS_JOINTS = """Two identical 6-DoF arms, prefixed left_ and right_, each with a parallel-jaw
gripper. Each arm has its own base frame: +x points forward out of the base
(the direction the folded gripper points at all-zero joints), +y left, +z up;
how the two bases are mounted relative to each other depends on the rig.
Joint guide (positive direction, identical for both arms):
- left_j0 / right_j0: base yaw about the vertical axis; positive swings the
  arm counterclockwise seen from above (a forward-pointing gripper moves
  toward +y).
- left_j1 / right_j1: shoulder pitch; 0 points the upper arm horizontally
  backward and is the lower hard stop (it cannot go negative), positive
  raises it (about 1.57 is straight up, about 3.14 is horizontal forward).
- left_j2 / right_j2: elbow; 0 is fully folded with the forearm doubled back
  against the upper arm and is the lower hard stop, positive opens it.
- left_j3 / right_j3: wrist pitch, axis parallel to the elbow; positive tilts
  the gripper up.
- left_j4 / right_j4: wrist yaw; positive swings the gripper toward the arm's
  right seen from above (opposite sign sense of j0).
- left_j5 / right_j5: wrist roll about the gripper's pointing axis; positive
  turns clockwise when viewed from behind the gripper looking out along the
  fingers.
- left_gripper / right_gripper: 0 is fully closed, 1 is fully open (about
  9.5 cm between the jaws).
Proportions: upper arm 0.26 m, forearm 0.25 m, wrist to grasp point 0.25 m
when straight; reach from the shoulder about 0.76 m.
At all-zero joints the arm rests folded low with the gripper pointing
forward. While the arm is folded, a single joint's effect on the gripper
position can be counterintuitive; move deliberately and re-check the
observation after each motion. The joint values above are positions as shown
in the observation; when actions are per-step changes (delta mode), the same
sign conventions apply to each change."""

_DOCS_EEF_POS = """Two identical 6-DoF arms, prefixed left_ and right_, each with a parallel-jaw
gripper, controlled by Cartesian end-effector targets. Each arm's targets are
in that arm's own base frame: +x points forward out of the base, +y left, +z
up; how the two bases are mounted relative to each other depends on the rig.
- left_x / right_x, left_y / right_y, left_z / right_z: grasp-point position
  in meters in the arm's base frame (the grasp point sits between the
  fingertips).
- left_yaw / right_yaw: tool rotation in radians about vertical, relative to
  the trial's start orientation; 0 keeps the start orientation and positive
  turns counterclockwise seen from above.
- left_gripper / right_gripper: 0 is fully closed, 1 is fully open (about
  9.5 cm between the jaws).
Proportions: upper arm 0.26 m, forearm 0.25 m, wrist to grasp point 0.25 m
when straight; reach from the shoulder about 0.76 m.
An inverse-kinematics layer converts targets into joint motion; unreachable
or awkward targets may be tracked slowly or held, so prefer modest steps and
re-check the observation after each motion."""


@runtime_checkable
class TaskEnvelopeLike(Protocol):
    """Structural mirror of ``inspect_robots.task.TaskEnvelope``.

    Read-only property members (not plain attributes) so the frozen core
    dataclass satisfies the protocol under mypy strict. Local rather than
    imported: this package supports cores that predate ``TaskEnvelope``.
    """

    @property
    def name(self) -> str:
        """The task's registry/display name."""
        ...

    @property
    def max_steps(self) -> int:
        """The rollout horizon the framework will enforce."""
        ...


@runtime_checkable
class BimanualDriver(Protocol):
    """The minimal 14-D joint-position driver the embodiment needs."""

    def get_joint_pos(self) -> npt.NDArray[np.floating[Any]]:
        """Read both arm poses in radians and driver-native gripper units."""
        ...

    def command_joint_pos(self, target: npt.NDArray[np.floating[Any]]) -> None:
        """Command both arm poses in radians and driver-native gripper units."""
        ...

    def close(self) -> None:
        """Release both arm handles, allowing their motor torque to drop."""
        ...


DriverFactory = Callable[[YamConfig], BimanualDriver]
KinematicsFactory = Callable[[YamConfig], tuple[RawKinematics, RawKinematics]]
CameraReader = Callable[[YamConfig], ImageMap]


def _default_driver_factory(cfg: YamConfig) -> BimanualDriver:  # pragma: no cover - real hardware
    get_yam_robot, GripperType = _load_i2rt()

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
                    close_robot_safely(arm)

    return _Real()


def _default_kinematics_factory(
    cfg: YamConfig,
) -> tuple[RawKinematics, RawKinematics]:  # pragma: no cover - optional runtime
    Kinematics, ArmType, GripperType, combine_xml, NoSolutionFound = _load_i2rt_kinematics()
    model_path = combine_xml(ArmType.YAM, GripperType[cfg.gripper_type])

    class _Adapter:
        def __init__(self) -> None:
            self._solver = Kinematics(model_path, "grasp_site")

        def get_joint_ranges(self) -> npt.NDArray[np.floating[Any]]:
            return np.asarray(self._solver._configuration.model.jnt_range).copy()

        def set_joint_ranges(self, ranges: npt.NDArray[np.floating[Any]]) -> None:
            self._solver._configuration.model.jnt_range[:] = ranges

        def fk(self, q: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
            return np.asarray(self._solver.fk(q))

        def ik(
            self,
            target: npt.NDArray[np.floating[Any]],
            init_q: npt.NDArray[np.floating[Any]],
            max_iters: int,
        ) -> tuple[bool, npt.NDArray[np.floating[Any]]]:
            try:
                success, q = self._solver.ik(
                    target,
                    "grasp_site",
                    init_q=init_q,
                    max_iters=max_iters,
                )
            except NoSolutionFound as exc:
                raise EmbodimentFault("EEF inverse kinematics QP is infeasible") from exc
            return bool(success), np.asarray(q)

    return _Adapter(), _Adapter()


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
    devices: dict[str, str] = {
        "top_cam": cast(str, cfg.top_cam_device),
        "left_cam": cast(str, cfg.left_cam_device),
        "right_cam": cast(str, cfg.right_cam_device),
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
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"YUYV"))
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
    """Inspect Robots embodiment for bimanual YAM joint or Cartesian control."""

    # cv2 is a base dependency, so its absence indicates a broken package install.
    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "i2rt": I2RT_INSTALL_COMMAND,
        "cv2": "uv pip install inspect-robots-yam",
    }

    # The setup wizard interviews these with real-device probes (issue
    # inspect-robots#61). CAN channels are grouped: a config naming only one
    # arm's channel silently drives the other on the plugin default, the
    # exact failure the interview exists to prevent. Cameras mirror
    # YamConfig's all-three-or-none validation.
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = (
        DeviceSlot(arg="left_channel", kind="can", label="left arm CAN channel", group="arms"),
        DeviceSlot(arg="right_channel", kind="can", label="right arm CAN channel", group="arms"),
        DeviceSlot(arg="top_cam_device", kind="v4l2", label="top camera", group="cameras"),
        DeviceSlot(arg="left_cam_device", kind="v4l2", label="left camera", group="cameras"),
        DeviceSlot(arg="right_cam_device", kind="v4l2", label="right camera", group="cameras"),
    )

    def __init__(
        self,
        config: YamConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        kinematics_factory: KinematicsFactory | None = None,
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
        self._kinematics_factory: KinematicsFactory = (
            kinematics_factory or _default_kinematics_factory
        )
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
        self._left_kinematics: _ArmKinematics | None = None
        self._right_kinematics: _ArmKinematics | None = None
        self._eef_home_validated = False
        self._init_pose: Vec | None = None
        # Set only after the stand-clear prompt returns, so a gate fault
        # (dead stdin) re-prompts on a retried reset instead of ramping
        # unconfirmed; cleared on close() so every connection re-confirms.
        self._home_gate_confirmed = False
        self._instruction: str | None = None
        self._t_last = 0.0
        self.num_steps = 0
        self._bound_max_steps: int | None = None

        docs = _DOCS_EEF_POS if self._cfg.control_interface == "eef_pos" else _DOCS_JOINTS
        docs_extra = self._cfg.docs_extra.strip()
        if docs_extra:
            docs += "\n\n" + docs_extra
        self.info = EmbodimentInfo(
            name="yam_arms",
            # Delta mode declares the per-step displacement box (symmetric,
            # honest for guardrail derivation); the absolute joint limits stay
            # enforced on the SUMMED command inside _send() either way.
            action_space=self._action_space(),
            observation_space=observation_space(
                self._cfg.cam_height,
                self._cfg.cam_width,
                DEFAULT_CAMERAS,
                control_interface=self._cfg.control_interface,
            ),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
        )

    # -- lifecycle ---------------------------------------------------------

    def bind_task(self, envelope: TaskEnvelopeLike) -> None:
        """Store the framework's rollout horizon for the operator countdown.

        Optional-input hook (inspect-robots plan 0013): it never fires on
        direct ``rollout()`` calls or on cores that predate it, in which case
        the countdown falls back to the deprecated ``max_steps_hint`` (or
        elapsed-only). Hardware-free — the framework calls it before
        ``reset()`` ever connects the driver. One call per ``eval()``; the
        latest envelope wins. On a caller-owned instance an aborted eval
        (e.g. a compatibility failure after binding) leaves the envelope in
        place until ``close()`` or the next bind.
        """
        self._bound_max_steps = int(envelope.max_steps)

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
        if self._cfg.control_interface == "eef_pos" and (
            self._left_kinematics is None or self._right_kinematics is None
        ):
            self._construct_kinematics()
        if self._init_pose is None:
            # Capture BEFORE any motion of ours (incl. the home ramp): this is
            # exactly where the operator left the arms — the safest known
            # gravity-stable park target for close(). Later resets keep it;
            # their start pose is just wherever the previous episode ended.
            self._init_pose = self._norm_grippers(
                packing.validate_dim(self._driver.get_joint_pos())
            )
        home_pose = self._home_pose()
        if self._cfg.control_interface == "eef_pos" and not self._eef_home_validated:
            self._validate_eef_home(np.clip(home_pose, self._cfg.low, self._cfg.high))
            self._eef_home_validated = True
        if not self._cfg.unattended and not self._home_gate_confirmed:
            self._operator.wait_ready(
                "Arms will move to the home pose - stand clear, then press Enter..."
            )
            self._home_gate_confirmed = True
        if not self._cfg.unattended:
            self._status("homing: ramping arms to start pose")
        try:
            final_home_command = self._ramp_to(home_pose)
        finally:
            if not self._cfg.unattended:
                self._status(None)
        if self._cfg.control_interface == "eef_pos":
            left_kinematics, right_kinematics = self._require_kinematics()
            left_kinematics.seed(final_home_command[: packing.ARM_DOF])
            right_kinematics.seed(
                final_home_command[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF]
            )
            measured = self._norm_grippers(packing.validate_dim(self._driver.get_joint_pos()))
            left_kinematics.capture_yaw_reference(measured[: packing.ARM_DOF])
            right_kinematics.capture_yaw_reference(
                measured[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF]
            )
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
        if self._cfg.control_interface == "eef_pos":
            cmd = packing.validate_dim(action.data, len(EEF_DIM_LABELS))
            self._step_eef(cmd, driver)
        else:
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
        """Park the arms, then release the driver handles.

        After ``reset()`` captures a pose, parking uses the configured
        ``rest_pose`` or falls back to that captured pose when configured as
        ``None``. A connection that faults before capture is released in place.
        The release lives in a ``finally`` so a driver fault or interrupt
        mid-ramp can never leave the handles held — but the arms may then fall
        from a mid-ramp pose. No-op if never connected.
        """
        # Unconditionally first: a bound-but-never-reset instance (eval() can
        # abort between bind_task and the first reset) must not carry a stale
        # horizon into a later framework-less run.
        self._bound_max_steps = None
        for kinematics in (self._left_kinematics, self._right_kinematics):
            if kinematics is not None:
                kinematics.clear()
        if self._driver is None:
            return
        try:
            if self._init_pose is not None:
                target = (
                    np.asarray(self._cfg.rest_pose, dtype=np.float64)
                    if self._cfg.rest_pose is not None
                    else self._init_pose
                )
                if not self._cfg.unattended:
                    self._status("parking: ramping arms back before torque-off")
                try:
                    self._ramp_to(target)
                finally:
                    # Close the status line even when the ramp faults, so a
                    # traceback never prints appended to it.
                    if not self._cfg.unattended:
                        self._status(None)
        finally:
            try:
                self._driver.close()
            finally:
                # Clear connection state even if the driver's own close()
                # raises, so a later reset() reconnects, re-captures, and
                # re-confirms the stand-clear gate.
                self._driver = None
                self._init_pose = None
                self._home_gate_confirmed = False

    def _ramp_to(self, target: Vec) -> Vec:
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
        sent = start
        for i in range(1, n + 1):
            alpha = i / n
            sent = self._send((1.0 - alpha) * start + alpha * target)
            self._sleep(1.0 / hz)
        return sent

    # -- internals ---------------------------------------------------------

    def _action_space(self) -> Box:
        """Build the declared action contract selected by the configuration."""
        if self._cfg.control_interface == "eef_pos":
            return action_box(
                low=self._cfg.eef_low_array,
                high=self._cfg.eef_high_array,
                control_interface="eef_pos",
            )
        if self._cfg.joints_are_delta:
            return action_box(
                low=self._cfg.delta_low,
                high=self._cfg.delta_high,
                joints_are_delta=True,
            )
        return action_box(low=self._cfg.low, high=self._cfg.high)

    def _home_pose(self) -> Vec:
        """Select the configured joint home, defaulting per control interface."""
        if self._cfg.control_interface == "eef_pos":
            values = self._cfg.home_pose or DEFAULT_EEF_HOME_POSE
            return np.asarray(values, dtype=np.float64)
        else:
            if self._cfg.home_pose is not None:
                return np.asarray(self._cfg.home_pose, dtype=np.float64)
            # If no home_pose is configured, home to the current pose of the joints
            # to prevent jumps, but default the grippers to 1.0 (open) per training
            # distribution.
            driver = self._require_driver()
            current = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
            home = current.copy()
            home[packing.ARM_DOF] = 1.0
            home[packing.ARM_WIDTH + packing.ARM_DOF] = 1.0
            return home

    def _construct_kinematics(self) -> None:
        """Construct per-arm wrappers and apply effective model/config limits."""
        left_raw, right_raw = self._kinematics_factory(self._cfg)
        left_kinematics = _ArmKinematics(
            side="left",
            raw=left_raw,
            config_low=self._cfg.low[: packing.ARM_DOF],
            config_high=self._cfg.high[: packing.ARM_DOF],
            ik_max_iters=self._cfg.ik_max_iters,
            ik_step_joint_limit=self._cfg.ik_step_joint_limit,
            cmd_resync_threshold=self._cfg.cmd_resync_threshold,
            osc_deadband=self._cfg.osc_deadband,
            osc_reversals=self._cfg.osc_reversals,
            osc_window=self._cfg.osc_window,
            osc_hold_steps=self._cfg.osc_hold_steps,
        )
        right_start = packing.ARM_WIDTH
        right_kinematics = _ArmKinematics(
            side="right",
            raw=right_raw,
            config_low=self._cfg.low[right_start : right_start + packing.ARM_DOF],
            config_high=self._cfg.high[right_start : right_start + packing.ARM_DOF],
            ik_max_iters=self._cfg.ik_max_iters,
            ik_step_joint_limit=self._cfg.ik_step_joint_limit,
            cmd_resync_threshold=self._cfg.cmd_resync_threshold,
            osc_deadband=self._cfg.osc_deadband,
            osc_reversals=self._cfg.osc_reversals,
            osc_window=self._cfg.osc_window,
            osc_hold_steps=self._cfg.osc_hold_steps,
        )
        self._left_kinematics = left_kinematics
        self._right_kinematics = right_kinematics

    def _require_kinematics(self) -> tuple[_ArmKinematics, _ArmKinematics]:
        """Return both constructed EEF wrappers."""
        if self._left_kinematics is None or self._right_kinematics is None:
            raise RuntimeError("EEF kinematics are unavailable before reset()")
        return self._left_kinematics, self._right_kinematics

    def _validate_eef_home(self, home: Vec) -> None:
        """Reject a joint home whose grasp sites start outside the EEF box."""
        left_kinematics, right_kinematics = self._require_kinematics()
        arm_values = (
            (
                "left",
                left_kinematics,
                home[: packing.ARM_DOF],
                float(home[packing.ARM_DOF]),
                slice(0, 5),
            ),
            (
                "right",
                right_kinematics,
                home[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF],
                float(home[-1]),
                slice(5, 10),
            ),
        )
        for side, kinematics, joints, gripper, bounds in arm_values:
            position = kinematics.fk(joints)[:3, 3]
            home_state = np.asarray((*position, 0.0, gripper))
            if np.any(home_state < self._cfg.eef_low_array[bounds]) or np.any(
                home_state > self._cfg.eef_high_array[bounds]
            ):
                raise ValueError(
                    f"{side} EEF home state {home_state.tolist()} is outside the "
                    "configured action workspace bounds"
                )

    def _step_eef(self, action: Vec, driver: BimanualDriver) -> None:
        """Convert one 10-D EEF action into the normative two-arm joint command."""
        state = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        left_kinematics, right_kinematics = self._require_kinematics()
        left_command = left_kinematics.solve(
            action[:4],
            state[: packing.ARM_DOF],
        )
        right_command = right_kinematics.solve(
            action[5:9],
            state[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF],
        )
        command = packing.pack(
            np.concatenate((left_command, action[4:5])),
            np.concatenate((right_command, action[9:10])),
        )
        sent = self._send(command, base=state)
        left_kinematics.update_sent(sent[: packing.ARM_DOF])
        right_kinematics.update_sent(sent[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF])

    def _horizon_secs(self) -> float | None:
        """The episode horizon in seconds: the bound envelope, else the hint.

        Dividing by our own ``control_hz`` is honest because this embodiment
        is ``SELF_PACED`` — that rate is the one ``_pace()`` sleeps to.
        """
        steps = (
            self._bound_max_steps if self._bound_max_steps is not None else self._cfg.max_steps_hint
        )
        hz = self._cfg.control_hz
        if steps is None or not hz or hz <= 0:
            return None
        return steps / hz

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

    def _send(self, cmd: Vec, base: Vec | None = None) -> Vec:
        """Clamp to joint and step limits (safety backstop) and de-normalize grippers."""
        driver = self._require_driver()
        current = (
            base
            if base is not None
            else self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
        )
        clamped = np.clip(cmd, self._cfg.low, self._cfg.high)
        clamped = np.clip(clamped, current + self._cfg.delta_low, current + self._cfg.delta_high)
        physical = self._denorm_grippers(clamped)
        driver.command_joint_pos(physical)
        return clamped

    def _denorm_grippers(self, cmd: Vec) -> Vec:
        """Map wire grippers (1 = open, 0 = closed) into driver-native units."""
        out: Vec = cmd.copy()
        span = self._cfg.gripper_open - self._cfg.gripper_closed
        for idx in (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF):  # 6, 13
            out[idx] = self._cfg.gripper_closed + cmd[idx] * span
        return out

    def _norm_grippers(self, physical: Vec) -> Vec:
        """Map driver units to wire grippers (1 = open, 0 = closed).

        ``YamConfig.__post_init__`` guarantees ``gripper_open != gripper_closed``,
        so the span is never zero.
        """
        out: Vec = physical.copy()
        span = self._cfg.gripper_open - self._cfg.gripper_closed
        for idx in (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF):  # 6, 13
            out[idx] = (physical[idx] - self._cfg.gripper_closed) / span
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
        values = {packing.STATE_KEY: state}
        if self._cfg.control_interface == "eef_pos":
            left_kinematics, right_kinematics = self._require_kinematics()
            left = left_kinematics.observe(state[: packing.ARM_DOF], gripper=float(state[6]))
            right = right_kinematics.observe(
                state[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF],
                gripper=float(state[13]),
            )
            values["eef_state"] = np.concatenate((left, right))
        return Observation(
            images=dict(self._camera_reader(self._cfg)),
            state=values,
            instruction=instruction,
        )
