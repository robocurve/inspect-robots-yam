"""Builtin V4L2 camera reader: queue capping, draining, and lifecycle (issue #63).

Every test drives the drain loop *synchronously* on the test thread, with a fake
capture that sets the stop event after N reads. Real threads appear only where
the thing under test is the threading itself (close). Nothing sleeps on the wall
clock: ``sleep_fn`` and ``clock`` are injected.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment, _OpenCVCameraReader

DEVICES = {"top_cam": "/dev/cam0", "left_cam": "/dev/cam1", "right_cam": "/dev/cam2"}


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
    """The slice of the cv2 module surface the reader touches.

    Only the constants actually used are defined, so a misspelled one raises
    ``AttributeError`` here instead of passing silently, which is the checking
    that injecting the module as ``Any`` gives up.
    """

    CAP_V4L2 = 200
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    COLOR_BGR2RGB = 4

    class VideoWriter:
        """Namespace for the fourcc helper the reader calls."""

        @staticmethod
        def fourcc(*chars: str) -> float:
            """Return a recognizable code so the recorded `set` is assertable."""
            return 1448695129.0

    def __init__(self, caps: dict[str, FakeCapture]) -> None:
        self.caps = caps
        self.opened: list[str] = []

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


class Clock:
    """A monotonic clock advanced only by the test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward, as a stalled camera would."""
        self.now += seconds


def build(
    caps: dict[str, FakeCapture] | None = None,
    clock: Clock | None = None,
) -> tuple[_OpenCVCameraReader, FakeCv2, list[float], Clock]:
    """A reader wired to fakes, plus the recorded sleeps and the clock."""
    caps = caps if caps is not None else {device: FakeCapture() for device in DEVICES.values()}
    cv2 = FakeCv2(caps)
    sleeps: list[float] = []
    clock = clock if clock is not None else Clock()
    reader = _OpenCVCameraReader(DEVICES, cv2_module=cv2, sleep_fn=sleeps.append, clock=clock)
    return reader, cv2, sleeps, clock


def drive(reader: _OpenCVCameraReader, name: str, cap: FakeCapture, iterations: int) -> None:
    """Run the drain loop on this thread for a bounded number of reads."""
    stop = threading.Event()
    cap.stop_after(stop, iterations)
    reader._drain(name, cap, stop)


def test_buffersize_is_capped_first_and_the_rest_of_the_negotiation_survives() -> None:
    # The whole recorded sequence, not just the presence of BUFFERSIZE: this is
    # the regression guard for the fix *and* for the settings it must not
    # disturb. Order is load-bearing -- OpenCV's V4L2 backend refuses
    # BUFFERSIZE once streaming has started.
    reader, cv2, _, _ = build()
    reader(YamConfig())

    cap = cv2.caps["/dev/cam0"]
    assert [call for call in cap.calls if call[0] == "set"] == [
        ("set", FakeCv2.CAP_PROP_BUFFERSIZE, 1),
        ("set", FakeCv2.CAP_PROP_FOURCC, 1448695129.0),
        ("set", FakeCv2.CAP_PROP_FRAME_WIDTH, 640),
        ("set", FakeCv2.CAP_PROP_FRAME_HEIGHT, 480),
        ("set", FakeCv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000),
        ("set", FakeCv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000),
    ]


def test_every_property_is_set_before_the_first_read() -> None:
    # Streaming starts at the first grab, so a property written after it is
    # silently dropped by the driver.
    reader, cv2, _, _ = build()
    reader(YamConfig())

    calls = cv2.caps["/dev/cam0"].calls
    assert calls.index(("read",)) > max(i for i, call in enumerate(calls) if call[0] == "set")


def test_devices_open_once_and_are_reused() -> None:
    reader, cv2, _, _ = build()
    reader(YamConfig())
    reader(YamConfig())

    assert cv2.opened == list(DEVICES.values())


def test_partial_open_failure_releases_what_opened_and_retries_everything() -> None:
    # A half-populated cache would never be retried: the `if not self._caps`
    # guard would be satisfied by the cameras that did open, and the rollout
    # would run on a subset of its declared views without saying so.
    caps = {
        "/dev/cam0": FakeCapture(),
        "/dev/cam1": FakeCapture(opened=False),
        "/dev/cam2": FakeCapture(),
    }
    reader, cv2, _, _ = build(caps)

    with pytest.raises(RuntimeError, match=r"cannot open left_cam at /dev/cam1"):
        reader(YamConfig())

    assert caps["/dev/cam0"].released
    assert not caps["/dev/cam2"].released  # never opened, so nothing to release

    caps["/dev/cam1"] = FakeCapture()
    reader(YamConfig())
    assert cv2.opened == ["/dev/cam0", "/dev/cam1", "/dev/cam0", "/dev/cam1", "/dev/cam2"]


def test_reader_returns_the_newest_drained_frame() -> None:
    # No open, so no background thread competes for the capture: the drain loop
    # is the only writer and the assertion is about it alone.
    reader, cv2, _, _ = build()
    cap = FakeCapture([(True, frame(1)), (True, frame(9))], idle_from=None)

    drive(reader, "top_cam", cap, iterations=2)

    image = reader._latest(cv2, "top_cam", YamConfig())
    assert np.array_equal(np.unique(image), np.array([9], dtype=np.uint8))


def test_drain_ignores_reads_the_driver_rejected() -> None:
    # `ok` is authoritative. A failed read can still hand back a frame object,
    # and publishing it would put a frame the driver rejected in front of the
    # policy -- the latent bug in the pre-#63 loop, which decided on the frame.
    reader, cv2, _, _ = build()
    cap = FakeCapture([(True, frame(1)), (False, frame(9)), (False, None)], idle_from=None)

    drive(reader, "top_cam", cap, iterations=3)

    image = reader._latest(cv2, "top_cam", YamConfig())
    assert np.array_equal(np.unique(image), np.array([1], dtype=np.uint8))


def test_a_dead_drain_thread_is_reported_rather_than_serving_one_frame_forever() -> None:
    # Without the latch this is #63 again, in a form nothing reports: the slot
    # freezes and every later observation is the same stale frame.
    reader, cv2, _, _ = build()
    cap = FakeCapture([(True, frame(1))], raise_at=2, idle_from=None)

    drive(reader, "top_cam", cap, iterations=3)

    with pytest.raises(RuntimeError, match=r"camera top_cam \(/dev/cam0\) stopped reading"):
        reader._latest(cv2, "top_cam", YamConfig())


def test_a_slot_that_stops_advancing_is_reported() -> None:
    # The thread can also wedge without raising (a device returning ok=False
    # forever), so freshness is checked as well as faults.
    clock = Clock()
    reader, cv2, sleeps, _ = build(clock=clock)
    cap = FakeCapture([(True, frame(1))], idle_from=None)
    drive(reader, "top_cam", cap, iterations=1)

    clock.advance(_OpenCVCameraReader.MAX_FRAME_AGE_S + 0.01)

    with pytest.raises(RuntimeError, match=r"frame read failed for top_cam \(/dev/cam0\)"):
        reader._latest(cv2, "top_cam", YamConfig())
    assert sleeps == [0.05] * 10  # the retry budget, spent before giving up


def test_frames_are_converted_to_rgb_uint8_at_the_configured_size() -> None:
    reader, cv2, _, _ = build()
    bgr = np.dstack([np.full((480, 640), value, np.uint8) for value in (1, 2, 3)])
    drive(reader, "top_cam", FakeCapture([(True, bgr)], idle_from=None), iterations=1)

    image = reader._latest(cv2, "top_cam", YamConfig(cam_width=32, cam_height=24))

    assert image.shape == (24, 32, 3)
    assert image.dtype == np.uint8
    assert list(image[0, 0]) == [3, 2, 1]  # channel order reversed by cvtColor


def test_close_stops_the_threads_and_releases_the_devices() -> None:
    caps = {device: FakeCapture() for device in DEVICES.values()}
    reader, _, _, _ = build(caps)
    reader(YamConfig())

    reader.close()

    assert all(cap.released for cap in caps.values())
    assert not any(thread.is_alive() for thread in reader._threads.values())


def test_close_is_idempotent_and_a_no_op_before_the_first_read() -> None:
    reader, cv2, _, _ = build()

    reader.close()  # nothing opened yet
    assert cv2.opened == []

    reader(YamConfig())
    reader.close()
    reader.close()


def test_close_leaks_a_device_rather_than_releasing_it_under_a_live_read() -> None:
    # release() underneath an in-flight read() crashes the process, and this
    # process is holding torque-enabled arms. Leaking the device is the better
    # failure, so the release is skipped and logged.
    block = threading.Event()
    caps = {
        "/dev/cam0": FakeCapture(block=block),
        "/dev/cam1": FakeCapture(),
        "/dev/cam2": FakeCapture(),
    }
    reader, _, _, _ = build(caps)
    reader(YamConfig())
    _OpenCVCameraReader.JOIN_TIMEOUT_S = 0.05
    try:
        reader.close()
    finally:
        _OpenCVCameraReader.JOIN_TIMEOUT_S = 2.0
        block.set()

    assert not caps["/dev/cam0"].released
    assert caps["/dev/cam1"].released


def test_reopening_after_close_starts_threads_that_are_not_already_stopped() -> None:
    # A stop flag reused across open cycles would leave every new thread exiting
    # on its first check, silently restoring the un-drained behavior.
    caps = {device: FakeCapture() for device in DEVICES.values()}
    reader, _, _, _ = build(caps)
    reader(YamConfig())
    reader.close()

    for device in DEVICES.values():
        caps[device].released = False
    reader(YamConfig())
    try:
        assert not reader._stop.is_set()
        assert all(thread.is_alive() for thread in reader._threads.values())
    finally:
        reader.close()


def test_a_camera_that_never_warms_up_is_not_fatal_at_open() -> None:
    # Pre-#63 behavior, deliberately preserved: a warm-up falling through is
    # survivable, because a camera slow to start on a cold USB bus gets another
    # chance from the drain thread. Only an empty slot when frames are wanted is
    # an error, and raising at open would fail reset() after the arms had homed.
    caps = {device: FakeCapture([(False, None)], idle_from=11) for device in DEVICES.values()}
    reader, _, sleeps, _ = build(caps)

    try:
        with pytest.raises(RuntimeError, match=r"frame read failed for top_cam"):
            reader(YamConfig())
    finally:
        reader.close()

    assert sleeps.count(0.1) == 10 * len(DEVICES)  # the warm-up budget, per camera


def test_the_warm_up_seeds_the_slot_so_the_first_observation_needs_no_drain() -> None:
    # reset() observes immediately after opening.
    reader, _, _, _ = build()

    try:
        images = reader(YamConfig())
    finally:
        reader.close()

    assert set(images) == set(DEVICES)


class ClosingReader:
    """A camera reader that owns devices, like the builtin one."""

    def __init__(self, fail: bool = False) -> None:
        self.closed = 0
        self._fail = fail

    def __call__(self, cfg: YamConfig) -> dict[str, npt.NDArray[np.uint8]]:
        """Never called by these tests; present to satisfy the reader contract."""
        raise AssertionError("not used")  # pragma: no cover - contract filler

    def close(self) -> None:
        """Release devices, or fail the way a wedged driver would."""
        self.closed += 1
        if self._fail:
            raise RuntimeError("device busy")


def embodiment(camera_reader: Any) -> YAMEmbodiment:
    """A never-reset embodiment: construction is inert, so no hardware is touched."""
    return YAMEmbodiment(YamConfig(), camera_reader=camera_reader)


def test_embodiment_close_releases_the_cameras() -> None:
    reader = ClosingReader()

    embodiment(reader).close()

    assert reader.closed == 1


def test_embodiment_close_tolerates_a_camera_reader_without_a_close() -> None:
    # CameraReader is a plain callable alias, so every custom reader in tests and
    # in user code is a bare function with no close.
    embodiment(lambda cfg: {}).close()


def test_a_failing_camera_release_never_stops_the_driver_teardown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The release runs ahead of the park ramp and the finally that guarantees the
    # driver release, so an escaping error would strand the arms torque-on.
    reader = ClosingReader(fail=True)

    embodiment(reader).close()

    assert reader.closed == 1
    assert "releasing cameras failed" in caplog.text
