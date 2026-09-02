"""Shared fixtures for settle tests (issue #62).

Two clock behaviors, because the settle loop has two independent bounds and
each must be observable as the *only* reason the loop stopped:

==============  ==========================================  ================
Fixture         Clock behavior                              Sole exit
==============  ==========================================  ================
frozen          constant 0.0 (what every older helper uses)  poll-count bound
read-advancing  ``get_joint_pos()`` adds ``read_advance``    elapsed bound
==============  ==========================================  ================

``read_advance`` must exceed ``_SETTLE_POLL_S`` or both bounds fire on the same
iteration and one of them becomes an unreachable branch, which the 100% gate
turns into a build failure. It is physically justified besides: each
``get_joint_pos()`` is two CAN round trips, so a real poll costs more than the
nominal spacing.

The existing ``_build`` helpers in ``test_embodiment.py`` and
``test_eef_embodiment.py`` keep their hardcoded ``clock=lambda: 0.0``. These
fixtures are additive. Refactoring those helpers onto a read-advancing clock
would inflate ``_pace``'s measured elapsed and break the pacing assertions.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO

#: Comfortably above _SETTLE_POLL_S (0.01), so the elapsed bound trips at 20
#: polls while the count bound would need 100.
READ_ADVANCE_S = 0.05


@pytest.fixture(autouse=True)
def isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests that do not opt into a config isolated from the developer's rig."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INSPECT_ROBOTS_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def frame(fill: int = 7) -> npt.NDArray[np.uint8]:
    """One BGR frame at capture resolution, filled with a recognizable value."""
    return np.full((480, 640, 3), fill, dtype=np.uint8)


class FakeFrame:
    """A RealSense frame with data and optional video intrinsics."""

    def __init__(self, data: npt.NDArray[Any], intrinsics: Any | None = None) -> None:
        self._data = data
        if intrinsics is not None:
            video_profile = types.SimpleNamespace(intrinsics=intrinsics)
            self.profile = types.SimpleNamespace(as_video_stream_profile=lambda: video_profile)

    def get_data(self) -> npt.NDArray[Any]:
        """Return the SDK-owned backing array."""
        return self._data


class FakeFrameset:
    """An aligned colour/depth frameset."""

    def __init__(self, colour: Any, depth: Any) -> None:
        self._colour = colour
        self._depth = depth

    def get_color_frame(self) -> Any:
        """Return the colour frame."""
        return self._colour

    def get_depth_frame(self) -> Any:
        """Return the depth frame."""
        return self._depth


def frameset(
    *,
    colour: npt.NDArray[np.uint8] | None = None,
    depth: npt.NDArray[np.uint16] | None = None,
    falsy_colour: bool = False,
    falsy_depth: bool = False,
) -> FakeFrameset:
    """Build one native-resolution RGB8/z16 pair."""
    if colour is None:
        colour = np.full((480, 640, 3), 7, dtype=np.uint8)
    if depth is None:
        depth = np.full((480, 640), 1000, dtype=np.uint16)
    intrinsics = types.SimpleNamespace(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
    colour_frame: Any = None if falsy_colour else FakeFrame(colour, intrinsics=intrinsics)
    depth_frame: Any = None if falsy_depth else FakeFrame(depth)
    return FakeFrameset(colour_frame, depth_frame)


class FakePipeline:
    """A pipeline with scripted tuple-returning frame waits."""

    def __init__(
        self,
        responses: list[tuple[bool, Any] | BaseException] | None = None,
        *,
        depth_scale: float = 0.001,
        start_error: BaseException | None = None,
        stop_error: Exception | None = None,
        block_after: int | None = None,
    ) -> None:
        self.responses = list(responses or [(True, frameset())])
        self.depth_scale = depth_scale
        self.start_error = start_error
        self.stop_error = stop_error
        self.block_after = block_after
        self.block = threading.Event()
        self.entered = threading.Event()
        self.calls = 0
        self.timeouts: list[int] = []
        self.started = False
        self.stopped = False
        self.config: FakeConfig | None = None
        self._stop_after: tuple[threading.Event, int] | None = None

    def start(self, config: FakeConfig) -> Any:
        """Record the config and return a depth-scale profile."""
        self.config = config
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        sensor = types.SimpleNamespace(get_depth_scale=lambda: self.depth_scale)
        device = types.SimpleNamespace(first_depth_sensor=lambda: sensor)
        return types.SimpleNamespace(get_device=lambda: device)

    def try_wait_for_frames(self, timeout_ms: int) -> tuple[bool, Any]:
        """Return the next scripted result, repeating the last one."""
        self.timeouts.append(timeout_ms)
        self.calls += 1
        if self.block_after is not None and self.calls >= self.block_after:
            self.entered.set()
            self.block.wait(timeout=5.0)
        if self.calls > len(self.responses):
            time.sleep(0.01)
        response = self.responses[min(self.calls, len(self.responses)) - 1]
        if self._stop_after is not None and self.calls >= self._stop_after[1]:
            self._stop_after[0].set()
        if isinstance(response, BaseException):
            raise response
        return response

    def stop_after(self, stop: threading.Event, calls: int) -> None:
        """Set a stop event after a bounded number of waits."""
        self._stop_after = (stop, calls)

    def reset_responses(self, responses: list[tuple[bool, Any] | BaseException]) -> None:
        """Replace the script for a later synchronous drain."""
        self.responses = responses
        self.calls = 0
        self._stop_after = None

    def stop(self) -> None:
        """Record pipeline shutdown, optionally raising after doing so."""
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error


class FakeAlign:
    """An align filter that records processing."""

    def __init__(self) -> None:
        self.calls = 0

    def process(self, source: Any) -> Any:
        """Return an already-aligned frameset."""
        self.calls += 1
        return source


class FakeConfig:
    """A recording librealsense pipeline config."""

    def __init__(self) -> None:
        self.device: str | None = None
        self.streams: list[tuple[Any, ...]] = []

    def enable_device(self, serial: str) -> None:
        """Record the device serial passed to librealsense."""
        self.device = serial

    def enable_stream(self, *args: Any) -> None:
        """Record one stream declaration."""
        self.streams.append(args)


class FakeDevice:
    """A discoverable device with device and optional ASIC serials."""

    def __init__(self, name: str, serial: str, asic_serial: str | None) -> None:
        self.name = name
        self.serial = serial
        self.asic_serial = asic_serial

    def supports(self, info: Any) -> bool:
        """Report whether the optional ASIC serial is available."""
        return info != FakeRs.camera_info.asic_serial_number or self.asic_serial is not None

    def get_info(self, info: Any) -> str:
        """Return the requested device identity field."""
        if info == FakeRs.camera_info.name:
            return self.name
        if info == FakeRs.camera_info.serial_number:
            return self.serial
        if self.asic_serial is None:
            raise AssertionError("unsupported ASIC serial queried")
        return self.asic_serial


class FakeContext:
    """A context whose device enumeration count is assertable."""

    def __init__(self, devices: list[FakeDevice]) -> None:
        self.devices = devices
        self.query_calls = 0

    def query_devices(self) -> list[FakeDevice]:
        """Return all visible devices."""
        self.query_calls += 1
        return self.devices


class FakeRs:
    """The pyrealsense2 namespace used by the reader."""

    stream = types.SimpleNamespace(color="colour", depth="depth")
    format = types.SimpleNamespace(rgb8="rgb8", z16="z16")
    camera_info = types.SimpleNamespace(
        name="name", serial_number="serial", asic_serial_number="asic"
    )

    def __init__(
        self,
        pipelines: list[FakePipeline] | None = None,
        devices: list[FakeDevice] | None = None,
    ) -> None:
        self.pipelines = list(pipelines or [])
        self.context_value = FakeContext(
            devices
            if devices is not None
            else [
                FakeDevice("Top D405", "S1", "A1"),
                FakeDevice("Left D405", "S2", "A2"),
                FakeDevice("Right D405", "S3", "A3"),
            ]
        )
        self.configs: list[FakeConfig] = []
        self.aligns: list[FakeAlign] = []
        self.pipeline_calls = 0

    def context(self) -> FakeContext:
        """Return the one enumeration context."""
        return self.context_value

    def config(self) -> FakeConfig:
        """Return a fresh pipeline config."""
        config = FakeConfig()
        self.configs.append(config)
        return config

    def pipeline(self) -> FakePipeline:
        """Return the next scripted pipeline."""
        if self.pipeline_calls == len(self.pipelines):
            self.pipelines.append(FakePipeline())
        pipeline = self.pipelines[self.pipeline_calls]
        self.pipeline_calls += 1
        return pipeline

    def align(self, stream: Any) -> FakeAlign:
        """Return a fresh colour align filter."""
        assert stream == self.stream.color
        align = FakeAlign()
        self.aligns.append(align)
        return align


class FakeCapture:
    """A ``cv2.VideoCapture`` stand-in that records the calls made against it."""

    def __init__(
        self,
        reads: list[tuple[bool, Any]] | None = None,
        *,
        opened: bool = True,
        raise_at: int | None = None,
        block: threading.Event | None = None,
        idle_from: int | None = 2,
    ) -> None:
        self.calls: list[Any] = []
        self.reads = list(reads if reads is not None else [(True, frame())])
        self.count = 0
        self.released = False
        self._opened = opened
        self._raise_at = raise_at
        self._block = block
        # A background drain thread reads flat out. Idling once the script is
        # spent keeps it from burning a core for the length of the test, while
        # still returning promptly enough for close() to join it.
        self._idle_from = idle_from
        self._stop: threading.Event | None = None
        self._stop_after = 0

    def stop_after(self, stop: threading.Event, reads: int) -> None:
        """Have the Nth read set ``stop``, so a drain loop ends deterministically."""
        self._stop, self._stop_after = stop, reads

    def isOpened(self) -> bool:
        """Whether the device opened, as cv2 reports it."""
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        """Record a property write in call order."""
        self.calls.append(("set", prop, value))
        return True

    def read(self) -> tuple[bool, Any]:
        """Return the next scripted result, repeating the last one forever."""
        self.calls.append(("read",))
        self.count += 1
        if self._block is not None and self.count > 1:
            self._block.wait(timeout=5.0)
        if self._idle_from is not None and self.count >= self._idle_from:
            time.sleep(0.01)
        if self._raise_at is not None and self.count >= self._raise_at:
            raise RuntimeError("device fell off the bus")
        if self._stop is not None and self.count >= self._stop_after:
            self._stop.set()
        return self.reads[min(self.count, len(self.reads)) - 1]

    def release(self) -> None:
        """Mark the capture released."""
        self.released = True


class FakeCv2:
    """The exact cv2 surface used by the camera reader, health montage, and watch.

    Only the constants the production code touches are defined, so a
    misspelled constant raises AttributeError instead of silently passing.
    """

    CAP_V4L2 = 200
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    COLOR_BGR2RGB = 4
    FONT_HERSHEY_SIMPLEX = 0

    class VideoWriter:
        """Namespace for the fourcc helper the reader calls."""

        @staticmethod
        def fourcc(*chars: str) -> float:
            """Return a recognizable code so the recorded `set` is assertable."""
            return 1448695129.0

    def __init__(self, caps: dict[str, FakeCapture] | None = None) -> None:
        self.caps = {} if caps is None else caps
        self.opened: list[str] = []
        self.put_text_calls: list[tuple[str, bool]] = []
        self.writes: list[tuple[str, npt.NDArray[np.uint8]]] = []
        self.encodes: list[tuple[str, npt.NDArray[np.uint8]]] = []
        self.write_ok = True
        self.encode_ok = True

    def VideoCapture(self, device: str, api: int) -> FakeCapture:
        """Hand back the scripted capture for a device path."""
        self.opened.append(device)
        return self.caps[device]

    def cvtColor(self, src: Any, code: int) -> Any:
        """Reverse the channel axis, standing in for a BGR to RGB conversion."""
        return np.asarray(src)[..., ::-1]

    def resize(self, src: Any, size: tuple[int, int]) -> Any:
        """Return an array of the requested size carrying the source's values."""
        width, height = size
        return np.full((height, width, 3), np.asarray(src)[0, 0, :], dtype=np.uint8)

    def putText(
        self,
        image: npt.NDArray[np.uint8],
        text: str,
        origin: tuple[int, int],
        font: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
    ) -> npt.NDArray[np.uint8]:
        """Record that labeling received a contiguous RGB tile, then mark it."""
        self.put_text_calls.append((text, image.flags.c_contiguous))
        image[0, 0] = (10, 20, 30)
        return image

    def imwrite(self, path: str, image: npt.NDArray[np.uint8]) -> bool:
        """Record the exact non-contiguous BGR view handed to the encoder."""
        self.writes.append((path, image.copy()))
        return self.write_ok

    def imencode(
        self, ext: str, image: npt.NDArray[np.uint8]
    ) -> tuple[bool, npt.NDArray[np.uint8]]:
        """Record the encoded image and return recognizable JPEG-like bytes."""
        self.encodes.append((ext, image.copy()))
        encoded = np.frombuffer(b"\xff\xd8fake-jpeg\xff\xd9", dtype=np.uint8)
        return self.encode_ok, encoded


class Clock:
    """A fake monotonic clock advanced explicitly by whatever charges time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward, as a sleep or a slow CAN read would."""
        self.now += seconds


class SettleDriver:
    """Driver that converges toward its commanded target after ``converge_after`` reads.

    ``offset`` is added to the target once converged, so a nonzero arm slot
    models a joint that never arrives (a hard stop, or a tolerance below the
    rig's steady-state error) and a nonzero gripper slot models a gripper
    closed on an object. Reads are counted so a test can assert which of the
    settle loop's two bounds ended it.
    """

    def __init__(
        self,
        *,
        converge_after: int = 1,
        offset: np.ndarray | None = None,
        clock: Clock | None = None,
        read_advance: float = 0.0,
    ) -> None:
        self.state = np.zeros(14)
        self.commands: list[np.ndarray] = []
        self.reads = 0
        self.closed = False
        self._converge_after = converge_after
        self.offset = np.zeros(14) if offset is None else np.asarray(offset, dtype=float)
        self._clock = clock
        self._read_advance = read_advance
        self._target: np.ndarray | None = None
        self._reads_since_command = 0

    def get_joint_pos(self) -> np.ndarray:
        """Report the current pose, charging CAN time and counting the read."""
        self.reads += 1
        if self._clock is not None and self._read_advance:
            self._clock.advance(self._read_advance)
        if self._target is not None:
            self._reads_since_command += 1
            if self._reads_since_command >= self._converge_after:
                self.state = self._target + self.offset
        return self.state.copy()

    def get_joint_eff(self) -> np.ndarray:
        """Report a canned packed effort vector."""
        return np.zeros(14)

    def get_motor_temps(self) -> np.ndarray:
        """Report benign packed motor temperatures."""
        return np.full(14, 30.0)

    def command_joint_pos(self, target: np.ndarray) -> None:
        """Record the target and restart the convergence countdown."""
        self.commands.append(np.asarray(target, dtype=float).copy())
        self._target = np.asarray(target, dtype=float).copy()
        self._reads_since_command = 0

    def close(self) -> None:
        """Mark the handle released."""
        self.closed = True


def _cameras(_cfg: YamConfig) -> dict[str, Any]:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": img, "left_cam": img, "right_cam": img}


def _silent_operator() -> OperatorIO:
    return OperatorIO(input_fn=lambda _p: "", output_fn=lambda _m: None)


@pytest.fixture
def clock() -> Clock:
    """A clock the test advances, for asserting on elapsed-time behavior."""
    return Clock()


@pytest.fixture
def build_settle():  # type: ignore[no-untyped-def]
    """Build an embodiment wired for settle tests; returns (embodiment, sleeps, status)."""

    def _make(
        cfg: YamConfig,
        driver: SettleDriver,
        *,
        clock: Clock | None = None,
        operator: OperatorIO | None = None,
        kinematics_factory: Any = None,
    ):  # type: ignore[no-untyped-def]
        cfg = dataclasses.replace(cfg, cam_height=4, cam_width=4)
        sleeps: list[float] = []
        status: list[str | None] = []

        def _sleep(seconds: float) -> None:
            sleeps.append(seconds)

        extra = {} if kinematics_factory is None else {"kinematics_factory": kinematics_factory}
        emb = YAMEmbodiment(
            cfg,
            driver_factory=lambda _c: driver,
            camera_reader=_cameras,
            **extra,
            operator=operator or _silent_operator(),
            poll_end=lambda: False,
            sleep_fn=_sleep,
            clock=clock if clock is not None else (lambda: 0.0),
            status_fn=status.append,
        )
        return emb, sleeps, status

    return _make
