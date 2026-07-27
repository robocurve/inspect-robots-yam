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


def frame(fill: int = 7) -> npt.NDArray[np.uint8]:
    """One BGR frame at capture resolution, filled with a recognizable value."""
    return np.full((480, 640, 3), fill, dtype=np.uint8)


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
    """The exact cv2 surface used by the camera reader and health montage."""

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
        self.write_ok = True

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
