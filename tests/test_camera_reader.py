"""Builtin V4L2 camera reader: queue capping, draining, and lifecycle (issue #63).

The drain loop is driven *synchronously* on the test thread wherever the loop
itself is the subject, with a fake capture that sets the stop event after N
reads, so no assertion depends on a race. Opening a reader necessarily starts
real drain threads, so the tests that open one are closed by an autouse fixture
rather than left running for the session. Nothing sleeps on the wall clock:
``sleep_fn`` and ``clock`` are injected.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from conftest import FakeCapture, FakeCv2, frame
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment, _OpenCVCameraReader

DEVICES = {"top_cam": "/dev/cam0", "left_cam": "/dev/cam1", "right_cam": "/dev/cam2"}

#: Readers handed out by `build`, closed after each test. Opening starts a drain
#: thread per camera, and a test that forgets leaves them reading for the rest of
#: the session and executing Python at interpreter shutdown.
_OPENED: list[_OpenCVCameraReader] = []


@pytest.fixture(autouse=True)
def close_readers() -> Iterator[None]:
    """Close every reader a test built, however the test ended."""
    yield
    while _OPENED:
        _OPENED.pop().close()


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
    _OPENED.append(reader)
    return reader, cv2, sleeps, clock


def drive(reader: _OpenCVCameraReader, name: str, cap: FakeCapture, iterations: int) -> None:
    """Run the drain loop on this thread for a bounded number of reads."""
    stop = threading.Event()
    cap.stop_after(stop, iterations)
    reader._drain(name, cap, stop, reader._generation)


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


class FakeDriver:
    """Just enough driver to observe that teardown reached it."""

    def __init__(self) -> None:
        self.closed = 0

    def get_joint_pos(self) -> npt.NDArray[np.float64]:
        """Never called by these tests."""
        raise AssertionError("not used")  # pragma: no cover - contract filler

    def command_joint_pos(self, target: npt.NDArray[np.float64]) -> None:
        """Never called by these tests."""
        raise AssertionError("not used")  # pragma: no cover - contract filler

    def close(self) -> None:
        """Record that the handles were released."""
        self.closed += 1


def test_a_failing_camera_release_never_stops_the_driver_teardown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Torque-off must happen whatever the cameras do.
    reader = ClosingReader(fail=True)
    emb = embodiment(reader)
    driver = FakeDriver()
    emb._driver = driver

    emb.close()

    assert driver.closed == 1
    assert reader.closed == 1
    assert "releasing cameras failed" in caplog.text


def test_an_interrupt_while_releasing_cameras_still_leaves_the_arms_torque_off() -> None:
    # close() now joins drain threads for up to a few seconds, and a join is
    # interruptible. A Ctrl-C there is routine during teardown -- often a second
    # one, since close() usually already runs inside a caller's finally -- and it
    # must not skip torque-off. Hence the release sits in an outer finally, and
    # BaseException is deliberately not swallowed by _release_cameras.
    class Interrupting(ClosingReader):
        def close(self) -> None:
            self.closed += 1
            raise KeyboardInterrupt

    reader = Interrupting()
    emb = embodiment(reader)
    driver = FakeDriver()
    emb._driver = driver

    with pytest.raises(KeyboardInterrupt):
        emb.close()

    assert driver.closed == 1
    assert emb._driver is None


def test_the_published_frame_survives_the_driver_reusing_its_buffer() -> None:
    # V4L2 hands back an array viewing the capture's own buffer, so the next read
    # overwrites a frame the consumer is still converting and release() frees it
    # outright. The fake returns the same array every read, as the driver does.
    reader, cv2, _, _ = build()
    buffer = frame(1)
    drive(reader, "top_cam", FakeCapture([(True, buffer)], idle_from=None), iterations=1)

    buffer[:] = 9  # the driver filling its buffer with the next capture

    image = reader._latest(cv2, "top_cam", YamConfig())
    assert np.array_equal(np.unique(image), np.array([1], dtype=np.uint8))


def test_the_warm_up_will_not_seed_the_slot_from_a_read_the_driver_rejected() -> None:
    # This read seeds the observation reset() takes first, which is the one the
    # policy plans a whole chunk from.
    caps = {device: FakeCapture([(False, frame(9))], idle_from=11) for device in DEVICES.values()}
    reader, _, _, _ = build(caps)

    with pytest.raises(RuntimeError, match=r"frame read failed for top_cam"):
        reader(YamConfig())


def test_a_thread_close_left_running_cannot_publish_into_the_next_cycle() -> None:
    # close() deliberately leaves a thread alive rather than releasing a capture
    # underneath it. That zombie must not write into a later open cycle: its
    # frame would be stamped fresh on publish though captured before the close,
    # so the freshness check cannot catch it. This is #63 by another route.
    reader, cv2, _, _ = build()
    retired = reader._generation
    reader.close()

    stop = threading.Event()
    zombie = FakeCapture([(True, frame(9))], idle_from=None)
    zombie.stop_after(stop, 1)
    reader._drain("top_cam", zombie, stop, retired)

    with pytest.raises(RuntimeError, match=r"frame read failed for top_cam"):
        reader._latest(cv2, "top_cam", YamConfig())


def test_a_thread_close_left_running_cannot_fault_a_healthy_camera() -> None:
    # The fault latch is sticky by design, so a zombie's exception would poison a
    # working camera until the next close().
    reader, cv2, _, _ = build()
    retired = reader._generation
    reader.close()
    drive(reader, "top_cam", FakeCapture([(True, frame(1))], idle_from=None), iterations=1)

    stop = threading.Event()
    zombie = FakeCapture(raise_at=1, idle_from=None)
    zombie.stop_after(stop, 1)
    reader._drain("top_cam", zombie, stop, retired)

    image = reader._latest(cv2, "top_cam", YamConfig())
    assert np.array_equal(np.unique(image), np.array([1], dtype=np.uint8))
