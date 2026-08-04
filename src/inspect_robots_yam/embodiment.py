"""``YAMEmbodiment`` — Inspect Robots embodiment for I2RT YAM bimanual arms.

Wraps the i2rt joint-position driver. Designed for real-robot reality:

* **Safety backstop** — every command is clamped to the configured joint limits
  inside :meth:`step`, *independently* of any Inspect Robots ``Approver`` (so unclamped
  model outputs can never reach the motors).
* **Operator-in-the-loop success** — there is no privileged success oracle; the
  operator's end-of-episode keypress returns
  ``StepResult(terminated=True, termination_reason="operator_end")`` and the
  human verdict (with optional grader notes) is captured afterwards by the
  framework's operator prompt and read by judgement-based scorers.
* **Self-paced** — declares ``SELF_PACED`` and sleeps to the control rate inside
  :meth:`step` (the framework does not pace for us).

Hardware/driver access is injected (``driver_factory``, ``camera_reader``,
``operator``, ``poll_end``, ``sleep_fn``, ``clock``) so the whole embodiment runs
in tests with no CAN bus, no cameras, and no stdin. The real driver/camera seams
are pragma'd defaults that only execute on hardware.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.approver import GuardrailContribution
from inspect_robots.conformance import DeviceSlot, OptionSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.spaces import Box
from inspect_robots.types import OPERATOR_END, Action, Observation, StepResult

from inspect_robots_yam import packing
from inspect_robots_yam._capture_proc import (
    JOIN_TIMEOUT_S,
    MAX_FRAME_AGE_S,
    REALSENSE_CAPTURE_HEIGHT,
    REALSENSE_CAPTURE_WIDTH,
    _CaptureProcess,
    _FrameSnapshot,
)
from inspect_robots_yam._i2rt import (
    I2RT_INSTALL_COMMAND,
    _load_i2rt,
    _load_i2rt_kinematics,
    close_robot_safely,
    start_arms_concurrently,
)
from inspect_robots_yam.config import (
    DEFAULT_CAMERAS,
    DEFAULT_EEF_HOME_POSE,
    DEFAULT_JOINT_HOME_POSE,
    EEF_DIM_LABELS,
    YamConfig,
    action_box,
    observation_space,
)
from inspect_robots_yam.kinematics import RawKinematics, _ArmKinematics
from inspect_robots_yam.operator import (
    OperatorIO,
    _drain_stdin,
    default_poll_end,
    stdin_interactive,
)

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


logger = logging.getLogger(__name__)

#: Spacing between settle polls. A floor, not a guarantee: each get_joint_pos()
#: is two CAN round trips, so on hardware the loop is read-bound.
_SETTLE_POLL_S = 0.01

#: Slots settling checks: both arms' revolute joints, never the grippers. A
#: gripper closing on an object never reaches its commanded position, so
#: including slots 6 and 13 would time out on every grasp. The exclusion is also
#: what makes the comparison sound at all, since get_joint_pos() reports
#: driver-native gripper units while the commanded vector is normalized; the two
#: are only commensurable on these slots.
_ARM_SLOTS = np.array(
    [
        index
        for index in range(packing.TOTAL_DIM)
        if index not in (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF)
    ],
    dtype=np.intp,
)

DriverFactory = Callable[[YamConfig], BimanualDriver]
KinematicsFactory = Callable[[YamConfig], tuple[RawKinematics, RawKinematics]]
CameraReader = Callable[[YamConfig], ImageMap]
DepthReader = Callable[[YamConfig], dict[str, Any]]


class _RealsenseReader(Protocol):
    """Shared callable/metadata/lifecycle surface of both capture modes."""

    def __call__(self, cfg: YamConfig) -> ImageMap:
        """Return the latest colour images."""
        ...

    def extra(self, cfg: YamConfig) -> dict[str, Any]:
        """Return rescaled intrinsics and lazy aligned depth."""
        ...

    def close(self) -> None:
        """Release every camera resource owned by the reader."""
        ...


class _CaptureTransport(Protocol):
    """Parent transport surface consumed by the process-mode reader."""

    @property
    def is_alive(self) -> bool:
        """Whether the child is still running."""
        ...

    @property
    def is_open(self) -> bool:
        """Whether an open cycle currently owns resources."""
        ...

    def open(self, generation: int) -> None:
        """Start one capture generation."""
        ...

    def read(self, name: str) -> _FrameSnapshot | None:
        """Copy the latest coherent shared-memory snapshot."""
        ...

    def close(self) -> None:
        """Stop the child and close its parent mappings."""
        ...


def _default_driver_factory(cfg: YamConfig) -> BimanualDriver:  # pragma: no cover - real hardware
    get_yam_robot, GripperType = _load_i2rt()

    # NAME lookup (GripperType["LINEAR_4310"]) — the enum *values* are lowercase
    # strings, so GripperType(...)/from_string_name would reject the config names.
    # YamConfig.__post_init__ already validated the name against the supported set.
    gripper = GripperType[cfg.gripper_type]

    def _make_arm(channel: str) -> Any:
        return get_yam_robot(
            channel=channel,
            gripper_type=gripper,
            zero_gravity_mode=cfg.zero_gravity_mode,
        )

    # Both arms are independent hardware (own CAN channel, own control
    # thread) and each pays a multi-second gripper calibration on every
    # boot (encoder frame resets at power-off), so bring them up together.
    left, right = start_arms_concurrently(
        lambda: _make_arm(cfg.left_channel),
        lambda: _make_arm(cfg.right_channel),
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


def _import_cv2() -> Any:  # pragma: no cover - real OpenCV
    """Import cv2 on first use, so the package imports without OpenCV."""
    import cv2

    return cv2


def _import_rs() -> Any:  # pragma: no cover - real pyrealsense2
    """Import pyrealsense2 on first use with an actionable error when absent."""
    try:
        import pyrealsense2 as rs  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyrealsense2 is required for RealSense depth streams. "
            "Install it with: uv pip install 'inspect-robots-yam[depth]'",
            name="pyrealsense2",
        ) from exc
    return rs


@dataclass(frozen=True)
class _Published:
    """One captured frame, copied out of the driver's buffer, and when it landed."""

    data: Any
    published_s: float


@dataclass(frozen=True)
class _PipelineBundle:
    """Pipeline, align filter, and depth scale for one RealSense camera."""

    pipeline: Any
    align: Any
    depth_scale: float


@dataclass(frozen=True)
class _PublishedPair:
    """One raw aligned colour/depth pair, its camera matrix K, and when it landed."""

    colour: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.uint16]
    intrinsics: npt.NDArray[np.float32]
    depth_scale: float
    published_s: float


class _OpenCVCameraReader:
    """Builtin V4L2 reader for rigs configured via ``*_cam_device`` (YamConfig).

    One daemon thread per camera reads continuously and publishes the newest
    frame; ``__call__`` converts whatever is in the slot. Without that thread a
    consumer running at ``control_hz`` dequeues from a queue the driver has
    already refilled, so the frame is ``N/control_hz - 1/fps`` old (#63) --
    380 ms on this rig, and worse as the control rate falls. Draining bounds it
    at one frame interval instead, independent of the control rate.

    cv2 is imported on the first frame read and devices open then too, so
    construction stays inert. Negotiates YUYV at 640x480 explicitly (RealSense
    D435s return empty frames on cv2 defaults) and resizes to ``cam_width`` x
    ``cam_height`` RGB.

    ``close()`` is required: the drain threads keep the devices open and, being
    live threads, keep this object reachable. ``YAMEmbodiment.close()`` calls it.
    """

    #: Frames older than this mean the camera has stopped delivering. Matches the
    #: read-retry budget the pre-#63 reader spent before raising.
    MAX_FRAME_AGE_S: ClassVar[float] = 0.5

    #: Longer than CAP_PROP_READ_TIMEOUT_MSEC, so a thread parked in a timing-out
    #: read is still given a chance to notice the stop flag and exit.
    JOIN_TIMEOUT_S: ClassVar[float] = 2.0

    def __init__(
        self,
        devices: Mapping[str, str],
        cv2_module: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._devices = dict(devices)
        self._cv2 = cv2_module
        self._sleep = sleep_fn
        self._clock = clock
        self._caps: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._published: dict[str, _Published] = {}
        self._faults: dict[str, BaseException] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Identifies the open cycle a drain thread belongs to. close() may
        # deliberately leave a thread running (see close), and that zombie must
        # not write into the state of a later cycle: its frames would be stamped
        # fresh on publish though captured before the previous close, which is
        # #63 again by another route, and its faults would poison a healthy
        # camera. Checked under the lock that guards those writes, so there is
        # no window between the check and the write.
        self._generation = 0

    def __call__(self, cfg: YamConfig) -> ImageMap:
        """Return the newest frame from every camera, opening devices on first use."""
        cv2 = self._cv2 if self._cv2 is not None else _import_cv2()
        self._cv2 = cv2
        if not self._caps:
            self._open_all(cv2)
        return {name: self._latest(cv2, name, cfg) for name in self._devices}

    def close(self) -> None:
        """Stop every drain thread, then release the captures it owned.

        Joins before releasing, and skips the release of any capture whose thread
        is still running: a ``release()`` underneath an in-flight ``read()``
        crashes the process, and this process is holding torque-enabled arms. A
        leaked device is the better failure. Idempotent, and a no-op before the
        first read since devices open lazily.
        """
        self._stop.set()
        for thread in self._threads.values():
            thread.join(timeout=self.JOIN_TIMEOUT_S)
        for name, drain in self._threads.items():
            cap = self._caps[name]
            if drain.is_alive():
                logger.warning(
                    "camera %s (%s) is still reading; leaving the device open rather "
                    "than releasing it underneath the read",
                    name,
                    self._devices[name],
                )
                continue
            cap.release()
        self._caps = {}
        self._threads = {}
        with self._lock:
            # Retires any thread this close left running, so it cannot write
            # into the next open cycle.
            self._generation += 1
            self._published = {}
            self._faults = {}

    def _open_all(self, cv2: Any) -> None:
        """Open every camera, or release the ones opened and re-raise.

        All-or-nothing because a half-populated cache would never be retried:
        the ``if not self._caps`` guard would be satisfied by the cameras that
        did open, and the rollout would run on a subset of its declared views.
        """
        with self._lock:
            self._generation += 1
            generation = self._generation
        caps: dict[str, Any] = {}
        try:
            for name, device in self._devices.items():
                caps[name] = self._open_one(cv2, name, device, generation)
        except BaseException:
            for cap in caps.values():
                cap.release()
            raise
        # A fresh stop flag per open cycle: a reader reopened after close() must
        # not hand its new threads an event that is already set.
        self._stop = threading.Event()
        self._caps = caps
        for name, cap in caps.items():
            thread = threading.Thread(
                target=self._drain,
                args=(name, cap, self._stop, generation),
                name=f"yam-camera-{name}",
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()

    def _open_one(self, cv2: Any, name: str, device: str, generation: int) -> Any:
        """Open and configure one camera, seeding its slot from the warm-up read.

        ``BUFFERSIZE`` is set first: OpenCV's V4L2 backend refuses it once
        streaming has begun, and it bounds the queue if a drain thread is ever
        descheduled. Seeding the slot matters because ``reset()`` observes
        immediately after opening, and a first call that found nothing published
        would fail after the arms had already homed.

        A warm-up that never yields a frame is not fatal here, as before #63:
        the drain thread gets its own chance and ``_latest`` waits for it.
        """
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {name} at {device}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
        for _ in range(10):  # warm up: first frames can be empty
            ok, frame = cap.read()
            # `ok` matters as much here as in the drain loop: this read seeds the
            # slot that serves reset()'s first observation.
            if ok and frame is not None:
                self._publish(name, frame, generation)
                break
            self._sleep(0.1)
        return cap

    def _drain(self, name: str, cap: Any, stop: threading.Event, generation: int) -> None:
        """Publish frames until stopped, latching whatever ends the loop.

        Owns the capture exclusively. Nothing else may touch it, including
        property reads: ``VideoCapture`` has no internal locking, so a concurrent
        call races the read in flight.

        An exception here would otherwise be invisible, and an invisible dead
        thread would freeze the slot and serve one frame forever, which is #63
        again in a form nothing reports. ``_latest`` re-raises what is latched.
        """
        while not stop.is_set():
            try:
                ok, frame = cap.read()
            except BaseException as exc:  # latched, then re-raised by _latest
                with self._lock:
                    if generation == self._generation:
                        self._faults[name] = exc
                return
            # `ok` is authoritative: a failed read can still hand back a frame
            # object, and publishing it would put a frame the driver rejected in
            # front of the policy.
            if ok and frame is not None:
                self._publish(name, frame, generation)

    def _publish(self, name: str, frame: Any, generation: int) -> None:
        """Copy a frame out of the driver's buffer into the slot.

        The copy is not optional: ``read()`` can hand back an array viewing the
        capture's own buffer, which the next read overwrites underneath a
        consumer still converting it, and which ``release()`` frees outright.
        """
        # Copied outside the lock: it is a megabyte-scale memcpy and nothing
        # else may touch this frame, so widening the critical section buys
        # nothing.
        copy = frame.copy()
        with self._lock:
            if generation != self._generation:
                return
            self._published[name] = _Published(copy, self._clock())

    def _latest(self, cv2: Any, name: str, cfg: YamConfig) -> npt.NDArray[np.uint8]:
        """Convert the newest published frame, waiting briefly for a fresh one."""
        device = self._devices[name]
        for _ in range(10):
            with self._lock:
                fault = self._faults.get(name)
                published = self._published.get(name)
            if fault is not None:
                raise RuntimeError(f"camera {name} ({device}) stopped reading") from fault
            if published is not None and self._clock() - published.published_s <= (
                self.MAX_FRAME_AGE_S
            ):
                return self._convert(cv2, published.data, cfg)
            self._sleep(0.05)
        raise RuntimeError(f"frame read failed for {name} ({device})")

    def _convert(self, cv2: Any, frame: Any, cfg: YamConfig) -> npt.NDArray[np.uint8]:
        """Turn one captured frame into the RGB uint8 array the contract declares."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (cfg.cam_width, cfg.cam_height))
        out: npt.NDArray[np.uint8] = np.asarray(resized).astype(np.uint8)
        return out


def _opencv_camera_reader(cfg: YamConfig) -> CameraReader:
    """Build the builtin V4L2 reader for device-sourced camera slots."""
    devices = {
        name: device
        for name, device in (
            ("top_cam", cfg.top_cam_device),
            ("left_cam", cfg.left_cam_device),
            ("right_cam", cfg.right_cam_device),
        )
        if device is not None
    }
    return _OpenCVCameraReader(devices)


def _default_camera_reader(cfg: YamConfig) -> ImageMap:
    raise NotImplementedError(
        "provide a camera_reader returning {'top_cam','left_cam','right_cam': HxWx3 uint8}"
    )


class _RealsenseCameraReader:
    """Own RealSense cameras through librealsense; serve colour, depth, and K.

    One pipeline owns each configured camera. Device selection accepts either the
    device serial reported by ``rs-enumerate-devices`` and
    ``rs.camera_info.serial_number``, or the ASIC/USB serial that librealsense calls
    ``asic_serial_number`` and embeds in ``/dev/v4l/by-id`` path names. Colour serves
    ``Observation.images``; aligned depth and intrinsics serve ``Observation.extra``.
    The composite has no cross-reader all-or-nothing guarantee: cv2 slots stay open
    if the RealSense open fails, and ``close()`` recovers them.
    """

    def __init__(
        self,
        serials: Mapping[str, str],
        depth_fps: int = 30,
        rs_module: Any | None = None,
        cv2_module: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._serials = dict(serials)
        self._depth_fps = depth_fps
        self._rs = rs_module
        self._cv2 = cv2_module
        self._sleep = sleep_fn
        self._clock = clock
        self._bundles: dict[str, _PipelineBundle] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._published: dict[str, _PublishedPair] = {}
        self._faults: dict[str, BaseException] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._generation = 0

    def __call__(self, cfg: YamConfig) -> ImageMap:
        """Return the newest RGB frame from every camera, opening them on first use."""
        self._ensure_open()
        cv2 = self._cv2 if self._cv2 is not None else _import_cv2()
        self._cv2 = cv2
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for name in self._serials:
            pair, _ = self._latest(name)
            resized = cv2.resize(pair.colour, (cfg.cam_width, cfg.cam_height))
            images[name] = np.asarray(resized).astype(np.uint8)
        return images

    def extra(self, cfg: YamConfig) -> dict[str, Any]:
        """Return scaled camera matrices and generation-bound lazy depth arrays."""
        self._ensure_open()
        extra: dict[str, Any] = {}
        for name in self._serials:
            pair, generation = self._latest(name)
            intrinsics = pair.intrinsics.copy()
            intrinsics[0, 0] = (
                float(pair.intrinsics[0, 0]) * cfg.cam_width / REALSENSE_CAPTURE_WIDTH
            )
            intrinsics[0, 2] = (
                float(pair.intrinsics[0, 2]) * cfg.cam_width / REALSENSE_CAPTURE_WIDTH
            )
            intrinsics[1, 1] = (
                float(pair.intrinsics[1, 1]) * cfg.cam_height / REALSENSE_CAPTURE_HEIGHT
            )
            intrinsics[1, 2] = (
                float(pair.intrinsics[1, 2]) * cfg.cam_height / REALSENSE_CAPTURE_HEIGHT
            )
            extra[f"{name}_intrinsics"] = intrinsics
            extra[f"{name}_depth"] = self._depth_thunk(
                name, cfg.cam_width, cfg.cam_height, generation
            )
        return extra

    def _ensure_open(self) -> None:
        """Resolve librealsense and open all configured cameras on first use."""
        rs = self._rs if self._rs is not None else _import_rs()
        self._rs = rs
        if not self._bundles:
            self._open_all(rs)

    def close(self) -> None:
        """Stop every drain thread and stop all RealSense pipelines.

        Joins before stopping, and skips the stop of any pipeline whose thread
        is still running: stopping underneath an in-flight hardware read can
        crash the process, and this process is holding torque-enabled arms. A
        leaked pipeline is the better failure. Idempotent, and a no-op before
        the first read since cameras open lazily.
        """
        self._stop.set()
        for thread in self._threads.values():
            thread.join(timeout=JOIN_TIMEOUT_S)
        for name, bundle in self._bundles.items():
            drain = self._threads[name]
            if drain.is_alive():
                logger.warning(
                    "camera %s (%s) is still reading; leaving the RealSense "
                    "pipeline open rather than stopping it underneath the read",
                    name,
                    self._serials[name],
                )
                continue
            try:
                bundle.pipeline.stop()
            except Exception:
                logger.exception("stopping RealSense pipeline for %s failed", name)
        self._bundles = {}
        self._threads = {}
        with self._lock:
            self._generation += 1
            self._published = {}
            self._faults = {}

    def _open_all(self, rs: Any) -> None:
        """Resolve and open every camera, or stop the ones opened and re-raise.

        All-or-nothing: a half-populated state would never be retried (the
        ``if not self._bundles`` guard would be satisfied by opened cameras),
        and the rollout would run with a subset of its declared views.
        """
        with self._lock:
            self._generation += 1
            generation = self._generation
        devices = list(rs.context().query_devices())
        visible: list[tuple[Any, str, str, str | None]] = []
        for device in devices:
            serial = str(device.get_info(rs.camera_info.serial_number))
            asic_info = rs.camera_info.asic_serial_number
            asic_serial = str(device.get_info(asic_info)) if device.supports(asic_info) else None
            device_name = str(device.get_info(rs.camera_info.name))
            visible.append((device, device_name, serial, asic_serial))

        resolved: dict[str, str] = {}
        for name, configured_serial in self._serials.items():
            match = next(
                (
                    serial
                    for _, _, serial, asic_serial in visible
                    if configured_serial in (serial, asic_serial)
                ),
                None,
            )
            if match is None:
                listing = ", ".join(
                    f"{device_name} / {serial} / {asic_serial or '<unavailable>'}"
                    for _, device_name, serial, asic_serial in visible
                )
                raise RuntimeError(
                    f"cannot find RealSense camera {name} ({configured_serial}); "
                    f"visible devices: {listing or 'none'}"
                )
            resolved[name] = match

        resolved_slots: dict[str, tuple[str, str]] = {}
        for name, resolved_serial in resolved.items():
            configured_serial = self._serials[name]
            previous = resolved_slots.get(resolved_serial)
            if previous is not None:
                previous_name, previous_configured_serial = previous
                raise RuntimeError(
                    f"cannot use RealSense cameras {previous_name} "
                    f"({previous_configured_serial}) and {name} ({configured_serial}): "
                    f"both resolve to device serial {resolved_serial}; configure each "
                    f"slot with a different visible device"
                )
            resolved_slots[resolved_serial] = (name, configured_serial)

        bundles: dict[str, _PipelineBundle] = {}
        try:
            for name, serial in resolved.items():
                bundles[name] = self._open_one(rs, name, serial, generation)
        except BaseException:
            for bundle in bundles.values():
                with contextlib.suppress(Exception):
                    bundle.pipeline.stop()
            with self._lock:
                self._published = {}
                self._faults = {}
            raise
        self._stop = threading.Event()
        self._bundles = bundles
        for name, bundle in bundles.items():
            thread = threading.Thread(
                target=self._drain,
                args=(name, bundle, self._stop, generation),
                name=f"yam-camera-{name}",
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()

    def _open_one(self, rs: Any, name: str, serial: str, generation: int) -> _PipelineBundle:
        """Open one pipeline, configure colour+depth streams, and seed from warm-up.

        A warm-up that never yields a frame is not fatal here: the drain thread
        gets its own chance and ``_latest`` waits for it.
        """
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(
            rs.stream.color,
            REALSENSE_CAPTURE_WIDTH,
            REALSENSE_CAPTURE_HEIGHT,
            rs.format.rgb8,
            self._depth_fps,
        )
        rs_cfg.enable_stream(
            rs.stream.depth,
            REALSENSE_CAPTURE_WIDTH,
            REALSENSE_CAPTURE_HEIGHT,
            rs.format.z16,
            self._depth_fps,
        )
        pipeline = rs.pipeline()
        profile = pipeline.start(rs_cfg)
        try:
            depth_scale: float = float(profile.get_device().first_depth_sensor().get_depth_scale())
            align = rs.align(rs.stream.color)
            for _ in range(10):  # warm-up: first frames may arrive after a brief delay
                ok, frames = pipeline.try_wait_for_frames(timeout_ms=1000)
                if not ok:
                    self._sleep(0.1)
                    continue
                aligned = align.process(frames)
                self._publish(name, aligned, depth_scale, generation)
                break
            return _PipelineBundle(pipeline=pipeline, align=align, depth_scale=depth_scale)
        except BaseException:
            with contextlib.suppress(BaseException):
                pipeline.stop()
            raise

    def _drain(
        self,
        name: str,
        bundle: _PipelineBundle,
        stop: threading.Event,
        generation: int,
    ) -> None:
        """Publish frames until stopped, latching whatever ends the loop."""
        while not stop.is_set():
            try:
                ok, frames = bundle.pipeline.try_wait_for_frames(timeout_ms=1000)
                if not ok:
                    continue
                aligned = bundle.align.process(frames)
                self._publish(name, aligned, bundle.depth_scale, generation)
            except BaseException as exc:
                with self._lock:
                    if generation == self._generation:
                        self._faults[name] = exc
                return

    def _publish(self, name: str, aligned: Any, depth_scale: float, generation: int) -> None:
        """Copy one aligned raw frame pair out of librealsense into the slot."""
        depth_frame = aligned.get_depth_frame()
        colour_frame = aligned.get_color_frame()
        if not depth_frame or not colour_frame:
            return
        colour: npt.NDArray[np.uint8] = np.asarray(colour_frame.get_data(), dtype=np.uint8).copy()
        depth: npt.NDArray[np.uint16] = np.asarray(depth_frame.get_data(), dtype=np.uint16).copy()
        intr = colour_frame.profile.as_video_stream_profile().intrinsics
        k_matrix: npt.NDArray[np.float32] = np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        with self._lock:
            if generation != self._generation:
                return
            self._published[name] = _PublishedPair(
                colour=colour,
                depth=depth,
                intrinsics=k_matrix,
                depth_scale=depth_scale,
                published_s=self._clock(),
            )

    def _latest(
        self, name: str, expected_generation: int | None = None
    ) -> tuple[_PublishedPair, int]:
        """Return the newest raw pair, waiting briefly for a fresh one."""
        serial = self._serials[name]
        for _ in range(10):
            with self._lock:
                generation = self._generation
                if expected_generation is not None and expected_generation != generation:
                    raise RuntimeError(f"depth for {name} resolved after camera close or reopen")
                fault = self._faults.get(name)
                published = self._published.get(name)
            if fault is not None:
                raise RuntimeError(f"camera {name} ({serial}) stopped reading") from fault
            if published is not None and self._clock() - published.published_s <= (MAX_FRAME_AGE_S):
                return published, generation
            self._sleep(0.05)
        raise RuntimeError(f"frame read failed for {name} ({serial})")

    def _depth_thunk(
        self, name: str, width: int, height: int, generation: int
    ) -> Callable[[], npt.NDArray[np.float32]]:
        """Build a lazy nearest-neighbour depth conversion for one open cycle."""

        def resolve() -> npt.NDArray[np.float32]:
            pair, _ = self._latest(name, generation)
            depth_m = pair.depth.astype(np.float32) * np.float32(pair.depth_scale)
            rows = np.arange(height) * depth_m.shape[0] // height
            columns = np.arange(width) * depth_m.shape[1] // width
            resized: npt.NDArray[np.float32] = depth_m[rows[:, None], columns]
            return resized.copy()

        return resolve


class _ProcessRealsenseCameraReader:
    """Serve RealSense colour, depth, and K from an isolated capture child."""

    def __init__(
        self,
        serials: Mapping[str, str],
        depth_fps: int = 30,
        *,
        child_entry: Any = None,
        transport: _CaptureTransport | None = None,
        cv2_module: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._serials = dict(serials)
        self._capture: _CaptureTransport = (
            transport
            if transport is not None
            else _CaptureProcess(
                self._serials,
                depth_fps,
                child_entry=child_entry,
            )
        )
        self._cv2 = cv2_module
        self._sleep = sleep_fn
        self._clock = clock
        self._generation = 0
        self._closed = False

    def __call__(self, cfg: YamConfig) -> ImageMap:
        """Return the newest RGB frame from every isolated camera."""
        self._ensure_open()
        cv2 = self._cv2 if self._cv2 is not None else _import_cv2()
        self._cv2 = cv2
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for name in self._serials:
            snapshot = self._latest(name)
            resized = cv2.resize(snapshot.colour, (cfg.cam_width, cfg.cam_height))
            images[name] = np.asarray(resized).astype(np.uint8)
        return images

    def extra(self, cfg: YamConfig) -> dict[str, Any]:
        """Return rescaled camera matrices and generation-bound depth thunks."""
        self._ensure_open()
        extra: dict[str, Any] = {}
        for name in self._serials:
            snapshot = self._latest(name)
            intrinsics = snapshot.intrinsics.copy()
            intrinsics[0, 0] = (
                float(snapshot.intrinsics[0, 0]) * cfg.cam_width / REALSENSE_CAPTURE_WIDTH
            )
            intrinsics[0, 2] = (
                float(snapshot.intrinsics[0, 2]) * cfg.cam_width / REALSENSE_CAPTURE_WIDTH
            )
            intrinsics[1, 1] = (
                float(snapshot.intrinsics[1, 1]) * cfg.cam_height / REALSENSE_CAPTURE_HEIGHT
            )
            intrinsics[1, 2] = (
                float(snapshot.intrinsics[1, 2]) * cfg.cam_height / REALSENSE_CAPTURE_HEIGHT
            )
            extra[f"{name}_intrinsics"] = intrinsics
            extra[f"{name}_depth"] = self._depth_thunk(
                name,
                cfg.cam_width,
                cfg.cam_height,
                self._generation,
            )
        return extra

    def close(self) -> None:
        """Stop the child and retire thunks from its capture generation."""
        if not self._capture.is_open:
            return
        self._closed = True
        self._generation += 1
        self._capture.close()

    def _ensure_open(self) -> None:
        """Spawn the capture child on first use or after a prior close."""
        if self._capture.is_open:
            return
        self._generation += 1
        self._capture.open(self._generation)
        self._closed = False

    def _latest(
        self,
        name: str,
        expected_generation: int | None = None,
    ) -> _FrameSnapshot:
        """Return a fresh coherent slot copy, waiting through brief staleness."""
        serial = self._serials[name]
        if self._closed or (
            expected_generation is not None and expected_generation != self._generation
        ):
            raise RuntimeError(f"depth for {name} resolved after camera close or reopen")
        for _ in range(10):
            if self._closed or (
                expected_generation is not None and expected_generation != self._generation
            ):
                raise RuntimeError(f"depth for {name} resolved after camera close or reopen")
            if not self._capture.is_alive:
                raise RuntimeError(f"frame read failed for {name} ({serial})")
            snapshot = self._capture.read(name)
            # A parent-side frozen test clock can precede the real child's
            # machine-wide monotonic stamp, yielding a large negative age. That
            # is fresh and intentionally satisfies this upper-bound comparison.
            if (
                snapshot is not None
                and snapshot.generation == self._generation
                and self._clock() - snapshot.published_s <= MAX_FRAME_AGE_S
            ):
                return snapshot
            self._sleep(0.05)
        raise RuntimeError(f"frame read failed for {name} ({serial})")

    def _depth_thunk(
        self,
        name: str,
        width: int,
        height: int,
        generation: int,
    ) -> Callable[[], npt.NDArray[np.float32]]:
        """Build a lazy nearest-neighbour depth conversion for one child cycle."""

        def resolve() -> npt.NDArray[np.float32]:
            # Check before _latest constructs any views over shm.buf. A stale
            # thunk must never reach closed shared memory and surface BufferError
            # or ValueError instead of the generation fault.
            if self._closed or generation != self._generation:
                raise RuntimeError(f"depth for {name} resolved after camera close or reopen")
            snapshot = self._latest(name, generation)
            depth_m = snapshot.depth.astype(np.float32) * np.float32(snapshot.depth_scale)
            rows = np.arange(height) * depth_m.shape[0] // height
            columns = np.arange(width) * depth_m.shape[1] // width
            resized: npt.NDArray[np.float32] = depth_m[rows[:, None], columns]
            return resized.copy()

        return resolve


class _CompositeCameraReader:
    """Merge disjoint builtin image readers and release all of their devices."""

    def __init__(self, *readers: CameraReader) -> None:
        self._readers = readers

    def __call__(self, cfg: YamConfig) -> ImageMap:
        """Return the union of every wrapped reader's camera images."""
        images: dict[str, npt.NDArray[np.uint8]] = {}
        for reader in self._readers:
            images.update(reader(cfg))
        return images

    def close(self) -> None:
        """Close every wrapped reader that exposes a duck-typed close method."""
        with contextlib.ExitStack() as stack:
            for reader in self._readers:
                release = getattr(reader, "close", None)
                if callable(release):
                    stack.callback(release)


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
    # exact failure the interview exists to prevent. Camera configs source
    # exactly one reader per slot; the wizard fills all three V4L2 device slots,
    # while RealSense depth serials are outside its scope.
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = (
        DeviceSlot(arg="left_channel", kind="can", label="left arm CAN channel", group="arms"),
        DeviceSlot(arg="right_channel", kind="can", label="right arm CAN channel", group="arms"),
        DeviceSlot(arg="top_cam_device", kind="v4l2", label="top camera", group="cameras"),
        DeviceSlot(arg="left_cam_device", kind="v4l2", label="left camera", group="cameras"),
        DeviceSlot(arg="right_cam_device", kind="v4l2", label="right camera", group="cameras"),
    )

    # The setup wizard offers these as yes/no questions (core OPTION_SLOTS
    # protocol, inspect-robots#222) and writes the answers to config.ini.
    # The behavior contract lives on the matching YamConfig field. The wizard
    # suggestion may diverge from the YamConfig default in either direction:
    # auto_start stays conservative at runtime but the wizard nudges toward
    # zero-touch starts, while collision_guardrail stays protective at
    # runtime but the wizard suggests off, because a fresh setup has no
    # measured collision_*_base_pos geometry and the library-default offsets
    # can false-positive hold until max_steps (#109). An existing config's
    # stored value replaces the suggestion on re-runs.
    OPTION_SLOTS: ClassVar[tuple[OptionSlot, ...]] = (
        OptionSlot(
            arg="auto_start",
            label="Skip the operator start prompts (auto_start)",
            default=True,
        ),
        OptionSlot(
            arg="collision_guardrail",
            label="Block predicted arm collisions before they happen "
            "(collision_guardrail; measure collision_*_base_pos first)",
            default=False,
        ),
    )

    def __init__(
        self,
        config: YamConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        kinematics_factory: KinematicsFactory | None = None,
        camera_reader: CameraReader | None = None,
        depth_reader: DepthReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        status_fn: Callable[[str | None], None] | None = None,
        **flat: Any,
    ) -> None:
        """Construct the embodiment and its configured camera readers.

        An injected ``depth_reader`` is merged after builtin RealSense metadata,
        so injected keys override builtin values when both publish the same key.
        """
        self._cfg = config if config is not None else YamConfig.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._kinematics_factory: KinematicsFactory = (
            kinematics_factory or _default_kinematics_factory
        )
        depth_serials = {
            name: serial
            for name, serial in (
                ("top_cam", self._cfg.top_depth_serial),
                ("left_cam", self._cfg.left_depth_serial),
                ("right_cam", self._cfg.right_depth_serial),
            )
            if serial is not None
        }
        self._builtin_realsense_reader: _RealsenseReader | None = None
        if camera_reader is not None:
            if depth_serials:
                raise ValueError(
                    "custom camera_reader conflicts with configured *_depth_serial "
                    "values: configured depth serials drive the builtin capture path; "
                    "with a custom camera_reader, supply depth via depth_reader instead"
                )
        else:
            builtin_readers: list[CameraReader] = []
            if any(
                device is not None
                for device in (
                    self._cfg.top_cam_device,
                    self._cfg.left_cam_device,
                    self._cfg.right_cam_device,
                )
            ):
                builtin_readers.append(_opencv_camera_reader(self._cfg))
            if depth_serials:
                if self._cfg.realsense_capture == "inline":
                    self._builtin_realsense_reader = _RealsenseCameraReader(
                        depth_serials, self._cfg.depth_fps
                    )
                else:
                    self._builtin_realsense_reader = _ProcessRealsenseCameraReader(
                        depth_serials, self._cfg.depth_fps
                    )
                builtin_readers.append(self._builtin_realsense_reader)
            if len(builtin_readers) == 1:
                camera_reader = builtin_readers[0]
            elif builtin_readers:
                camera_reader = _CompositeCameraReader(*builtin_readers)
        self._camera_reader: CameraReader = (
            camera_reader if camera_reader is not None else _default_camera_reader
        )
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._deferred_operator_end = False
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._status: Callable[[str | None], None] = status_fn or _default_status
        self._depth_reader: DepthReader | None = depth_reader

        self._driver: BimanualDriver | None = None
        self._left_kinematics: _ArmKinematics | None = None
        self._right_kinematics: _ArmKinematics | None = None
        self._eef_home_validated = False
        self._init_pose: Vec | None = None
        # Set only after the stand-clear gate resolves (prompt returned, or
        # the auto_start notice printed), so a gate fault (dead stdin)
        # re-prompts on a retried reset instead of ramping unconfirmed;
        # cleared on close() so every connection re-confirms.
        self._home_gate_confirmed = False
        self._instruction: str | None = None
        self._t_last = 0.0
        self.num_steps = 0
        self.settle_timeouts = 0
        # Set when the per-trial timeout budget is exhausted; suppresses further
        # settling for the rest of the trial. Cleared at reset() entry.
        self._settle_disabled = False
        self._bound_max_steps: int | None = None

        docs = _DOCS_EEF_POS if self._cfg.control_interface == "eef_pos" else _DOCS_JOINTS
        docs_extra = self._cfg.docs_extra.strip()
        if docs_extra:
            docs += "\n\n" + docs_extra
        if self._builtin_realsense_reader is not None:
            depth_cameras = ", ".join(sorted(depth_serials))
            docs += (
                f"\n\nDepth: for each serial-configured camera ({depth_cameras}), "
                '``observation.extra["{cam}_depth"]`` arrives either as an H\u00d7W '
                "float32 array of depth in metres, or as a zero-arg callable returning "
                "that array. If it is callable, resolve it immediately on receipt. The "
                "array has the same resolution as, and is aligned to, that camera's "
                "image in ``observation.images``. Resolving the callable returns a "
                "fresh conversion of the newest captured frames, so delayed resolution "
                "returns depth captured later than the image it accompanies. "
                '``observation.extra["{cam}_intrinsics"]`` is a plain 3\u00d73 float32 '
                "camera matrix K valid at that published resolution. Distortion "
                "coefficients are omitted; the cameras are near-rectilinear at this "
                "resolution."
            )
            if self._depth_reader is not None:
                docs += (
                    " An injected ``depth_reader`` may add or override keys in "
                    "``observation.extra``."
                )
        elif self._depth_reader is not None:
            docs += (
                "\n\nDepth: ``observation.extra`` may contain additional depth data "
                "from the configured depth reader; consult its documentation."
            )
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

    def contribute_guardrails(self, action_space: Box) -> GuardrailContribution:
        """Contribute collision holds when absolute joint checking is available."""
        if not self._cfg.collision_guardrail:
            return GuardrailContribution()
        if self._cfg.control_interface != "joints" or self._cfg.joints_are_delta:
            return GuardrailContribution(
                warnings=("collision guardrail skipped: absolute joints mode only (plan 0011 v1)",)
            )

        from inspect_robots_yam.collision import _INSTALL_COMMAND, _collision_approver

        if importlib.util.find_spec("mujoco") is None:
            return GuardrailContribution(
                warnings=(
                    "collision guardrail skipped: MuJoCo is unavailable; "
                    f"install it with: {_INSTALL_COMMAND}",
                )
            )

        approver = _collision_approver(self._cfg, action_space)
        warnings: tuple[str, ...] = ()
        if self._cfg.collision_left_base_pos is None or self._cfg.collision_right_base_pos is None:
            defaulted = (
                ("collision_left_base_pos", self._cfg.collision_left_base_pos is None),
                ("collision_right_base_pos", self._cfg.collision_right_base_pos is None),
                ("collision_left_base_yaw", self._cfg.collision_left_base_yaw is None),
                ("collision_right_base_yaw", self._cfg.collision_right_base_yaw is None),
                (
                    "collision_table_height",
                    self._cfg.collision_table and self._cfg.collision_table_height is None,
                ),
                (
                    "collision_penetration_threshold",
                    self._cfg.collision_penetration_threshold is None,
                ),
            )
            warnings = (
                "collision guardrail uses library-default geometry fields: "
                + ", ".join(name for name, is_default in defaulted if is_default),
            )
        return GuardrailContribution(
            approvers=(("yam-collision", approver),),
            warnings=warnings,
        )

    # -- lifecycle ---------------------------------------------------------

    def defer_operator_end(self) -> None:
        """Yield stdin and episode termination to the framework console.

        Inspect Robots calls this hook when its operator console owns stdin for
        the run. Afterward this embodiment never reads stdin again: it performs
        no end-of-episode poll or drains, and the framework terminates trials
        with ``operator_end`` itself. The setting persists across resets because
        every trial in the run belongs to the same console.
        """
        self._deferred_operator_end = True

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
        """Connect (if needed), drive to home, and block on operator readiness.

        With ``auto_start`` set, both operator gates are skipped: a printed
        notice replaces the stand-clear prompt and the episode begins right
        after the homing ramp. Requires an interactive stdin
        (faults before any motion otherwise). ``unattended`` skips the gates
        too, along with the rest of the attended flow, and takes precedence.
        """
        # Cleared HERE, at entry, not alongside num_steps further down: the home
        # ramp settles below, and that settle reads _settle_disabled. Clearing
        # after the ramp would let a trial that exhausted its budget suppress
        # the next trial's reset settle, so the yaw reference would again be
        # captured from a possibly mid-motion pose, for every trial thereafter.
        self.settle_timeouts = 0
        self._settle_disabled = False
        # Ahead of the homing settle below, which names the scene if it has to
        # report a budget exhaustion; set later it would report the previous
        # trial's instruction, or None on the first.
        self._instruction = scene.instruction
        # Fail fast on an unusable camera_reader BEFORE connecting the driver or
        # commanding any motion: this is a pure configuration error. `not callable`
        # also catches a CLI-injected scalar (`-E camera_reader=...` binds a str).
        if self._camera_reader is _default_camera_reader or not callable(self._camera_reader):
            raise ConfigError(
                "yam_arms has no cameras configured. Set exactly one source per "
                "camera slot: *_cam_device for V4L2 colour or *_depth_serial for "
                "RealSense colour+depth, in YamConfig, config.ini "
                "([embodiment.args]), or the CLI; or provide a custom "
                "camera_reader= via the Python API."
            )
        # auto_start still needs stdin: the end-episode keypress and the
        # framework's grading prompt both read it. wait_ready() normally
        # fail-fasts a dead stdin before any motion; with the gates skipped,
        # this check keeps that property (off-TTY, default_poll_end() always
        # returns False, so episodes could otherwise only end at max_steps).
        if self._cfg.auto_start and not self._cfg.unattended and not stdin_interactive():
            raise EmbodimentFault(
                "auto_start needs an interactive terminal: the end-episode "
                "keypress and the operator grading prompt both read stdin, "
                "which is not a TTY here. Run from a real TTY, or set "
                "YamConfig(unattended=True) (CLI: -E unattended=true) for "
                "headless runs."
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
            if self._cfg.auto_start:
                # Non-blocking replacement for the stand-clear gate: the
                # operator opted into zero-touch starts, but still gets one
                # line of warning before the first motion of the connection.
                self._operator.output_fn(
                    "auto_start: arms will move to the home pose - stand clear."
                )
            else:
                self._operator.wait_ready(
                    "Arms will move to the home pose - stand clear, then press Enter...",
                    drain=not self._deferred_operator_end,
                    flush_first=self._deferred_operator_end,
                )
            self._home_gate_confirmed = True
        if not self._cfg.unattended:
            self._status("homing: ramping arms to start pose")
        try:
            final_home_command = self._ramp_to(home_pose)
            # Inside the try, so the operator sees a status line instead of up
            # to settle_timeout_s of silence while standing at the e-stop, and
            # so a fault here still closes that line. Ahead of the yaw-reference
            # capture below, which would otherwise pin the whole trial's yaw
            # zero to a mid-motion pose.
            if self._cfg.settle_tolerance is not None and not self._cfg.unattended:
                self._status("settling: waiting for arms to reach the start pose")
            self._settle(final_home_command)
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
            if self._cfg.auto_start:
                # wait_ready() owns the stdin drain; skipping the gate must not
                # skip the drain, or a buffered newline ends the episode on the
                # first default_poll_end() check.
                # When deferred, pending lines belong to the console, which drains
                # after reset returns.
                if not self._deferred_operator_end:
                    _drain_stdin()
            else:
                self._operator.wait_ready(
                    drain=not self._deferred_operator_end,
                    flush_first=self._deferred_operator_end,
                )
            horizon = self._horizon_secs()
            limit = f" Max {horizon:.0f}s." if horizon is not None else ""
            if self._deferred_operator_end:
                self._status(
                    "Running: Enter ends the episode; type a message + Enter to send "
                    f"feedback.{limit}"
                )
            else:
                self._status(f"Running: press any key to end the episode and grade it.{limit}")
        self.num_steps = 0
        self._t_last = self._clock()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp + command one action, pace to the control rate, then maybe end."""
        driver = self._require_driver()
        self.num_steps += 1
        if self._cfg.control_interface == "eef_pos":
            cmd = packing.validate_dim(action.data, len(EEF_DIM_LABELS))
            target = self._step_eef(cmd, driver)
        else:
            cmd = packing.validate_dim(action.data)
            if self._cfg.joints_are_delta:
                # Normalize the gripper slots of the current position first, so
                # the delta is applied in policy units (a fraction of the
                # gripper stroke) and the sum re-enters _send() in the same
                # units as absolute mode.
                base = self._norm_grippers(packing.validate_dim(driver.get_joint_pos()))
                cmd = base + cmd
            target = self._send(cmd)
        # Before _pace(), so a settle that finishes inside the control period
        # costs nothing: the pace simply sleeps out whatever is left.
        settle_info = self._settle_info(self._settle(target))
        self._pace()
        self._emit_status()

        obs = self._observe(self._instruction)
        # Unattended runs have no operator: skip the end poll entirely; the
        # episode runs to the framework's max_steps.
        if not self._cfg.unattended and not self._deferred_operator_end and self._poll_end():
            self._status(None)  # close the status line before control returns
            # The operator only signals *that* the episode is over. The verdict,
            # partial/skip, and grader notes belong to the framework's single
            # operator prompt, which a definitive reason here would suppress —
            # so the reason stays non-definitive (inspect-robots#194).
            return StepResult(
                observation=obs,
                terminated=True,
                termination_reason=OPERATOR_END,
                info=settle_info,
            )
        return StepResult(observation=obs, terminated=False, info=settle_info)

    def close(self) -> None:
        """Park the arms, then release the driver handles.

        After ``reset()`` captures a pose, parking uses the configured
        ``rest_pose`` or falls back to that captured pose when configured as
        ``None``. A connection that faults before capture is released in place.
        The release lives in a ``finally`` so a driver fault or interrupt
        mid-ramp can never leave the handles held — but the arms may then fall
        from a mid-ramp pose. No-op if never connected.

        Cameras are released in an outer ``finally``, so it runs on the
        never-reset path too and, more importantly, can never pre-empt the
        driver teardown. It joins drain threads for up to a few seconds, and a
        ``Thread.join()`` is interruptible: a Ctrl-C there (routine during
        teardown, and often a *second* one, since this method usually already
        runs inside a caller's ``finally``) would otherwise skip torque-off
        entirely. Releasing after the park ramp costs nothing, because nothing
        observes while parking.
        """
        # Unconditionally first: a bound-but-never-reset instance (eval() can
        # abort between bind_task and the first reset) must not carry a stale
        # horizon into a later framework-less run.
        self._bound_max_steps = None
        for kinematics in (self._left_kinematics, self._right_kinematics):
            if kinematics is not None:
                kinematics.clear()
        try:
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
                    # re-runs the stand-clear gate (prompt, or the auto_start
                    # notice).
                    self._driver = None
                    self._init_pose = None
                    self._home_gate_confirmed = False
        finally:
            self._release_cameras()

    def _release_cameras(self) -> None:
        """Release the camera reader's devices and the depth reader's pipelines.

        Duck-typed because ``CameraReader`` and ``DepthReader`` are plain
        callable aliases: every custom reader in tests and user code is a bare
        function with no ``close``. Errors are logged and swallowed: a reader
        that will not let go is not a reason to fail a teardown that has already
        parked the arms.
        """
        release = getattr(self._camera_reader, "close", None)
        if callable(release):
            try:
                release()
            except Exception:
                logger.exception("releasing cameras failed; continuing with teardown")
        depth_release = getattr(self._depth_reader, "close", None)
        if callable(depth_release):
            try:
                depth_release()
            except Exception:
                logger.exception("releasing depth reader failed; continuing with teardown")

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
                gripper_max_step=self._cfg.gripper_max_step,
            )
        if self._cfg.joints_are_delta:
            return action_box(
                low=self._cfg.delta_low,
                high=self._cfg.delta_high,
                joints_are_delta=True,
                gripper_max_step=self._cfg.gripper_max_step,
            )
        return action_box(
            low=self._cfg.low,
            high=self._cfg.high,
            gripper_max_step=self._cfg.gripper_max_step,
        )

    def _home_pose(self) -> Vec:
        """Select the configured joint home, defaulting per control interface."""
        if self._cfg.control_interface == "eef_pos":
            values = self._cfg.home_pose or DEFAULT_EEF_HOME_POSE
        else:
            values = self._cfg.home_pose or DEFAULT_JOINT_HOME_POSE
        return np.asarray(values, dtype=np.float64)

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

    def _step_eef(self, action: Vec, driver: BimanualDriver) -> Vec:
        """Convert one 10-D EEF action into the normative two-arm joint command.

        Returns the clamped vector actually sent, which is what settling waits
        for. In this mode that routinely differs from what the policy asked for:
        an oscillation hold, a non-finite IK solve, or the per-step rate clamp
        all re-send a previous pose.
        """
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
        sent = self._send(command)
        left_kinematics.update_sent(sent[: packing.ARM_DOF])
        right_kinematics.update_sent(sent[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF])
        return sent

    def _horizon_secs(self) -> float | None:
        """The episode horizon in seconds: the bound envelope, else the hint.

        Dividing by our own ``control_hz`` is honest because this embodiment
        is ``SELF_PACED`` — that rate is the one ``_pace()`` sleeps to.

        With ``settle_tolerance`` set, ``control_hz`` becomes a floor on step
        duration rather than the rate, so this is then a lower bound rather
        than an estimate. Issue #64 tracks driving it from the wall clock.
        """
        steps = (
            self._bound_max_steps if self._bound_max_steps is not None else self._cfg.max_steps_hint
        )
        hz = self._cfg.control_hz
        if steps is None or not hz or hz <= 0:
            return None
        return steps / hz

    def _emit_status(self) -> None:
        """Once per second (of control time), tell the operator where they are.

        Elapsed time is counted in steps, so with ``settle_tolerance`` set both
        this counter and the horizon it prints understate real time. Issue #64.
        """
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

    def _send(self, cmd: Vec) -> Vec:
        """Clamp to joint limits (safety backstop) and de-normalize grippers."""
        clamped = np.clip(cmd, self._cfg.low, self._cfg.high)
        physical = self._denorm_grippers(clamped)
        self._require_driver().command_joint_pos(physical)
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

    def _settle(self, target: Vec) -> tuple[bool, float] | None:
        """Wait for the arm joints to reach ``target``; report (settled, residual).

        Returns ``None`` when settling is not running, either because no
        tolerance is configured or because this trial exhausted its timeout
        budget. That single guard lives here so neither call site repeats it.

        Reads before sleeping, so an already-converged step costs no sleep at
        all: that is the common case both in tests and during an oscillation
        hold, where the commanded pose is the one the arm already holds.
        """
        tolerance = self._cfg.settle_tolerance
        if tolerance is None or self._settle_disabled:
            return None
        driver = self._require_driver()
        wanted = target[_ARM_SLOTS]
        started = self._clock()
        # Bounded by poll count as well as elapsed time: the clock is injected
        # and every test fixture but one freezes it, and on hardware a stalled
        # or non-monotonic clock must not be able to wedge a step.
        # settle_timeout_s > 0 is validated, so this is always at least 1.
        max_polls = math.ceil(self._cfg.settle_timeout_s / _SETTLE_POLL_S)
        polls = 0
        while True:
            error = np.abs(packing.validate_dim(driver.get_joint_pos())[_ARM_SLOTS] - wanted)
            residual = float(np.max(error))
            if residual <= tolerance:
                return True, residual
            polls += 1
            if polls >= max_polls:
                break
            if self._clock() - started >= self._cfg.settle_timeout_s:
                break
            self._sleep(_SETTLE_POLL_S)
        self.settle_timeouts += 1
        if self.settle_timeouts >= self._cfg.settle_timeout_budget:
            self._settle_disabled = True
            joint = int(_ARM_SLOTS[int(np.argmax(error))])
            # logging, not warnings.warn: success is graded by the operator at
            # the framework prompt and no human reads StepResult.info, so this
            # is the practical notice that a trial's observations degraded.
            # The warnings registry keys on the message text, and a joint parked
            # against a hard stop repeats its residual, so successive trials
            # would be deduped into silence exactly when the rig is worst.
            logger.warning(
                "settle timeout budget exhausted; observations for the rest of "
                "this trial may precede the commanded pose (scene=%r, "
                "worst joint index=%d, residual=%.4f rad)",
                self._instruction,
                joint,
                residual,
            )
        return False, residual

    def _settle_info(self, settled: tuple[bool, float] | None) -> dict[str, Any]:
        """Per-step settle reporting, empty when no tolerance is configured.

        Keyed off the config rather than off ``settled``, so a trial that gave
        up keeps reporting. Absent keys therefore mean "never enabled" and can
        never be confused with "gave up".
        """
        if self._cfg.settle_tolerance is None:
            return {}
        info: dict[str, Any] = {"settle_timeouts": self.settle_timeouts}
        if settled is not None:
            info["settled"], info["settle_residual"] = settled
        if self._settle_disabled:
            info["settle_disabled"] = True
        return info

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

        images = dict(self._camera_reader(self._cfg))
        expected_shape = (self._cfg.cam_height, self._cfg.cam_width, 3)
        for name, img in images.items():
            if img.shape != expected_shape:
                raise ValueError(
                    f"camera {name!r} returned shape {img.shape}, expected {expected_shape}"
                )

        values = {packing.STATE_KEY: state}
        if self._cfg.control_interface == "eef_pos":
            left_kinematics, right_kinematics = self._require_kinematics()
            left = left_kinematics.observe(state[: packing.ARM_DOF], gripper=float(state[6]))
            right = right_kinematics.observe(
                state[packing.ARM_WIDTH : packing.ARM_WIDTH + packing.ARM_DOF],
                gripper=float(state[13]),
            )
            values["eef_state"] = np.concatenate((left, right))
        if self._builtin_realsense_reader is None and self._depth_reader is None:
            return Observation(images=images, state=values, instruction=instruction)
        extra: dict[str, Any] = {}
        if self._builtin_realsense_reader is not None:
            extra.update(self._builtin_realsense_reader.extra(self._cfg))
        if self._depth_reader is not None:
            extra.update(self._depth_reader(self._cfg))
        return Observation(
            images=images,
            state=values,
            instruction=instruction,
            extra=extra,
        )
