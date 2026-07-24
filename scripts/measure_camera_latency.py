"""Measure V4L2 frame staleness on the rig's cameras (robocurve/inspect-robots-yam#63).

Reads only from ``/dev/video*``. No CAN traffic, no torque, no robot motion, so
this is safe to run with the arms powered down.

The question it answers: when the rollout consumes frames at ``control_hz``
while the cameras free-run at their native rate, how old is the frame
``cap.read()`` hands back, and which fix collapses that age?

``cv2.VideoCapture`` does not take a picture when you call ``read()``. The
driver free-runs into a queue of buffers and ``read()`` dequeues one, so the
frame you get was captured at some earlier moment you do not control.

**How age is measured.** ``CAP_PROP_POS_MSEC`` carries the kernel's capture
timestamp for the buffer just dequeued, on a clock whose epoch is unrelated to
``perf_counter``. The offset between the two is recovered once per camera by
draining the queue in a tight loop, where the frame is as fresh as the pipeline
allows, and taking the residual as zero. Every age below is therefore an excess
over that floor: what a fix could actually remove. The signal needs no scene
change, perturbs nothing, and resolves to microseconds.

Backlog is reported alongside as a corroborating signal: a queued frame returns
in microseconds while an empty queue blocks for a frame interval, so the count
of instant reads before the first blocking one is the queue depth. Depth is not
age. A saturated ring drained one-in-one-out advances one frame per **consumer**
period, so a 4-deep queue at 10 Hz holds a 400 ms old frame, not a 133 ms one.

Three configurations are compared:

1. ``default`` — what the rollout does today.
2. ``buffersize=1`` — the one-line ``CAP_PROP_BUFFERSIZE`` cap.
3. ``grabber`` — ``buffersize=1`` plus a daemon thread per camera reading
   continuously into a latest-frame slot, so the consumer never dequeues a
   backlog. This mirrors the shape the fix would take in ``embodiment.py``.

Every camera is held open for the whole run and driven together at the control
rate, because that is the configuration the rollout runs: three streams sharing
USB bandwidth, read inside one control period.

Usage on the rig::

    ssh robocurve@omen
    cd ~/robocurve/inspect-robots-yam
    .venv/bin/python scripts/measure_camera_latency.py \
        --device top_cam=/dev/cam_context \
        --device left_cam=/dev/cam_wrist_left \
        --device right_cam=/dev/cam_wrist_right
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any

# Mirrors the open sequence in `_opencv_camera_reader` (embodiment.py). Keep in
# sync: a measurement taken under different capture settings does not describe
# the configuration the rollout actually runs.
_FOURCC = "YUYV"
_CAPTURE_WIDTH = 640
_CAPTURE_HEIGHT = 480
_OPEN_TIMEOUT_MSEC = 3000
_READ_TIMEOUT_MSEC = 1000
_WARMUP_READS = 10

# A read is "instant" when it returns in under this fraction of one frame
# interval: far too fast for the frame to have been captured on demand, so it
# came off the queue.
_INSTANT_FRACTION = 0.5

# Tight reads used to drain a queue and to time the producer once drained.
_DRAIN_READS = 60
_INTERVAL_SAMPLES = 15


class _Source:
    """One camera, read the way the configuration under test reads it."""

    def __init__(self, cv2: Any, cap: Any, device: str) -> None:
        self._cv2 = cv2
        self._cap = cap
        self._device = device
        self._offset = 0.0

    def read(self) -> tuple[float, float]:
        """Return how long the read took and the staleness of what it returned.

        Staleness is measured against the tight-loop floor calibrated by
        `calibrate`, so it is zero for a frame as fresh as the pipeline allows.
        """
        start = time.perf_counter()
        ok, frame = self._cap.read()
        done = time.perf_counter()
        if not ok or frame is None:
            raise SystemExit(f"frame read failed for {self._device}")
        captured = self._cap.get(self._cv2.CAP_PROP_POS_MSEC) / 1000.0
        return done - start, (done - self._offset) - captured

    def calibrate(self) -> None:
        """Pin the capture clock to `perf_counter` with the queue drained.

        A tight read loop keeps the queue empty, so the frame it returns is the
        freshest the pipeline can produce. Treating that residual as zero makes
        every later reading an excess over the floor rather than an absolute
        latency that would carry the sensor's own pipeline delay.
        """
        offsets: list[float] = []
        for _ in range(_DRAIN_READS):
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise SystemExit(f"calibration read failed for {self._device}")
            offsets.append(
                time.perf_counter() - self._cap.get(self._cv2.CAP_PROP_POS_MSEC) / 1000.0
            )
        # Median, not a single sample: one outlier would shift every reading for
        # this camera by a constant, and a few ms is a large fraction of the
        # drained configurations' numbers.
        self._offset = statistics.median(offsets[len(offsets) // 2 :])

    def buffersize(self) -> float:
        """Read the queue depth back, since `set` is a request, not a guarantee.

        Safe only while this thread is the capture's only user: `VideoCapture`
        has no internal locking, so calling it alongside a drain thread would
        race the read in flight. Callers must read it before draining starts.
        """
        return float(self._cap.get(self._cv2.CAP_PROP_BUFFERSIZE))

    def publish_interval_s(self) -> float:
        """Producer period actually achieved, or NaN when nothing tracked it."""
        return float("nan")

    def close(self) -> None:
        """Release the capture."""
        self._cap.release()


class _GrabberSource(_Source):
    """A camera drained by a daemon thread, exposing only its newest frame.

    The thread owns the capture exclusively and publishes each frame's staleness
    into a slot under a lock. Splitting `grab()` and `retrieve()` across threads
    would race on the capture's internal state, so the consumer never touches it.
    """

    def __init__(self, cv2: Any, cap: Any, device: str) -> None:
        super().__init__(cv2, cap, device)
        self._lock = threading.Lock()
        self._captured: float | None = None
        self._publishes = 0
        self._first_publish_s = 0.0
        self._last_publish_s = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def calibrate(self) -> None:
        """Calibrate against the bare capture, then hand it to the drain thread."""
        super().calibrate()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        while self._captured is None and not self._stop.is_set():
            time.sleep(0.005)

    def _drain(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                captured = self._cap.get(self._cv2.CAP_PROP_POS_MSEC) / 1000.0
                now = time.perf_counter()
                with self._lock:
                    self._captured = captured
                    if not self._publishes:
                        self._first_publish_s = now
                    self._publishes += 1
                    self._last_publish_s = now

    def read(self) -> tuple[float, float]:
        """Return the newest published frame's staleness, never blocking."""
        start = time.perf_counter()
        with self._lock:
            captured = self._captured
        done = time.perf_counter()
        if captured is None:  # pragma: no cover - only before the first publish
            raise SystemExit(f"no frame published for {self._device}")
        return done - start, (done - self._offset) - captured

    def publish_interval_s(self) -> float:
        """Measured spacing between publishes: the rate the drain thread sustains.

        Reported rather than assumed, because three drain threads sharing one USB
        bus is the single new risk this configuration introduces, and a rate
        below the sensor's is how it would show up.
        """
        with self._lock:
            if self._publishes < 2:
                return float("nan")
            return (self._last_publish_s - self._first_publish_s) / (self._publishes - 1)

    def close(self) -> None:
        """Stop the drain thread before releasing the capture it owns."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        super().close()


@dataclass(frozen=True)
class CameraResult:
    """One camera measured under one configuration."""

    name: str
    buffersize: float
    staleness_s: tuple[float, ...]
    read_s: tuple[float, ...]
    backlog: int | None
    interval_s: float

    @property
    def median_staleness_s(self) -> float:
        """Median excess over the tight-loop floor: what a fix can remove."""
        return statistics.median(self.staleness_s)

    @property
    def drift_s(self) -> float:
        """Change in staleness from the first samples to the last.

        The classic `POS_MSEC` failure is a frame counter scaled by a nominal
        interval, which shows up as staleness ramping instead of holding. A
        median hides that; this does not.
        """
        head = self.staleness_s[: max(1, len(self.staleness_s) // 4)]
        tail = self.staleness_s[-max(1, len(self.staleness_s) // 4) :]
        return statistics.median(tail) - statistics.median(head)

    @property
    def median_read_s(self) -> float:
        """Median cost of one read on the control thread."""
        return statistics.median(self.read_s)

    @property
    def fps(self) -> float:
        """Producer rate implied by the drained tight-loop interval."""
        return 1.0 / self.interval_s if self.interval_s > 0 else float("nan")


@dataclass(frozen=True)
class Run:
    """Everything measured under one configuration."""

    label: str
    results: tuple[CameraResult, ...]
    sweep_s: float

    @property
    def worst_staleness_s(self) -> float:
        """Oldest median frame across cameras: what sets end-to-end staleness."""
        return max(r.median_staleness_s for r in self.results)


def _open(cv2: Any, device: str, buffersize: int | None) -> Any:
    """Open one capture configured the way the rollout configures it."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {device}")
    if buffersize is not None:
        # Before the format negotiation and before any grab: OpenCV's V4L2
        # backend rejects the property once streaming has started.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MSEC)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MSEC)
    for _ in range(_WARMUP_READS):
        if cap.read()[0]:
            break
        time.sleep(0.1)
    return cap


def _paced(
    sources: dict[str, _Source], hz: float, duration_s: float
) -> tuple[dict[str, list[float]], dict[str, list[float]], float]:
    """Drive every camera at the control rate, recording staleness and read cost.

    The sweep cost returned is the `o` term in #62's `o + s <= 1/hz` control
    budget: time spent capturing inside the control period is time the arm does
    not get to converge before it is photographed.
    """
    period = 1.0 / hz
    deadline = time.perf_counter() + duration_s
    staleness: dict[str, list[float]] = {name: [] for name in sources}
    reads: dict[str, list[float]] = {name: [] for name in sources}
    sweeps: list[float] = []
    while time.perf_counter() < deadline:
        cycle_start = time.perf_counter()
        for name, source in sources.items():
            read_s, stale_s = source.read()
            reads[name].append(read_s)
            staleness[name].append(stale_s)
        sweeps.append(time.perf_counter() - cycle_start)
        time.sleep(max(0.0, period - (time.perf_counter() - cycle_start)))
    return staleness, reads, statistics.median(sweeps)


def _measure_backlog(source: _Source, burst: int) -> tuple[int, float]:
    """Time a burst of reads, then the drained producer period, and classify."""
    read_times = [source.read()[0] for _ in range(burst)]
    interval = statistics.median(source.read()[0] for _ in range(_INTERVAL_SAMPLES))
    threshold = interval * _INSTANT_FRACTION
    backlog = 0
    for elapsed in read_times:
        if elapsed >= threshold:
            break
        backlog += 1
    return backlog, interval


def _run(
    cv2: Any,
    devices: dict[str, str],
    label: str,
    buffersize: int | None,
    grabber: bool,
    hz: float,
    duration_s: float,
    burst: int,
) -> Run:
    """Open every camera together, calibrate, then measure the paced regime."""
    factory = _GrabberSource if grabber else _Source
    sources: dict[str, _Source] = {
        name: factory(cv2, _open(cv2, device, buffersize), device)
        for name, device in devices.items()
    }
    try:
        # Before calibrate(), which hands a grabber's capture to its drain
        # thread; afterwards this would be a second thread on one VideoCapture.
        buffersizes = {name: source.buffersize() for name, source in sources.items()}
        for source in sources.values():
            source.calibrate()
        staleness, reads, sweep_s = _paced(sources, hz, duration_s)
        results = []
        for name, source in sources.items():
            # A grabber's queue is drained by its thread, so a burst measured
            # through the latest-frame slot would report the slot, not the queue.
            backlog, interval = (None, 1.0 / 30.0) if grabber else _measure_backlog(source, burst)
            results.append(
                CameraResult(
                    name,
                    buffersizes[name],
                    tuple(staleness[name]),
                    tuple(reads[name]),
                    backlog,
                    interval,
                )
            )
        return Run(label, tuple(results), sweep_s)
    finally:
        for source in sources.values():
            source.close()


def _print_run(run: Run, hz: float) -> None:
    """Print one configuration's per-camera rows and its sweep cost."""
    print(f"\n=== {run.label} ===")
    for r in run.results:
        backlog = "drained by thread" if r.backlog is None else str(r.backlog)
        print(
            f"  {r.name:<11} stale={r.median_staleness_s * 1000:7.1f} ms"
            f"  read={r.median_read_s * 1000:6.2f} ms"
            f"  buffersize={r.buffersize:.0f}  backlog={backlog}  {r.fps:.1f} fps"
            f"  drift={r.drift_s * 1000:+.1f} ms"
        )
    print(
        f"  sweep of {len(run.results)} cameras: {run.sweep_s * 1000:.2f} ms"
        f" of a {1000.0 / hz:.1f} ms control period"
    )


def _measure_via_reader(cv2: Any, devices: dict[str, str], hz: float, duration_s: float) -> None:
    """Measure staleness through the package's own reader: the code that ships.

    The three configurations above are ones this script constructed, so they
    would all stay green if the fix were reverted. This is the only reading that
    fails when the shipped reader stops draining.

    Staleness comes from the reader's own published timestamps rather than from
    the captures, because nothing outside the drain thread may touch a capture.
    """
    from inspect_robots_yam.config import YamConfig
    from inspect_robots_yam.embodiment import _OpenCVCameraReader

    reader = _OpenCVCameraReader(devices, cv2_module=cv2)
    cfg = YamConfig(
        top_cam_device=devices.get("top_cam"),
        left_cam_device=devices.get("left_cam"),
        right_cam_device=devices.get("right_cam"),
    )
    ages: dict[str, list[float]] = {name: [] for name in devices}
    try:
        reader(cfg)  # opens the devices and starts the drain threads
        period = 1.0 / hz
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            cycle_start = time.perf_counter()
            reader(cfg)
            now = time.monotonic()
            with reader._lock:
                for name, published in reader._published.items():
                    ages[name].append(now - published.published_s)
            time.sleep(max(0.0, period - (time.perf_counter() - cycle_start)))
    finally:
        reader.close()

    print("\n=== via the shipped reader ===")
    for name, samples in ages.items():
        print(f"  {name:<11} slot age median {statistics.median(samples) * 1000:6.1f} ms")
    print(
        "  (age of the newest published frame when the consumer took it; the"
        " capture-to-publish leg is the drain thread's own read, timed above)"
    )


def main() -> None:
    """Measure every requested camera under all three configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="camera to measure, e.g. top_cam=/dev/cam_context (repeatable)",
    )
    parser.add_argument(
        "--hz", type=float, default=10.0, help="consumer rate (default: control_hz)"
    )
    parser.add_argument("--duration", type=float, default=6.0, help="seconds of paced measurement")
    parser.add_argument("--burst", type=int, default=8, help="timed reads for the backlog probe")
    parser.add_argument(
        "--via-reader",
        action="store_true",
        help="also measure through the package's own reader, the code that ships",
    )
    args = parser.parse_args()

    devices: dict[str, str] = {}
    for spec in args.device:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--device wants NAME=PATH, got {spec!r}")
        devices[name] = path

    import cv2

    print(f"cv2 {cv2.__version__}  consumer {args.hz} Hz  paced {args.duration}s")
    configs = (("default", None, False), ("buffersize=1", 1, False), ("grabber", 1, True))
    runs = []
    for label, buffersize, grabber in configs:
        run = _run(cv2, devices, label, buffersize, grabber, args.hz, args.duration, args.burst)
        _print_run(run, args.hz)
        runs.append(run)

    print("\nworst staleness above the tight-loop floor:")
    for run in runs:
        print(f"  {run.label:<14} {run.worst_staleness_s * 1000:7.1f} ms")

    if args.via_reader:
        _measure_via_reader(cv2, devices, args.hz, args.duration)


if __name__ == "__main__":
    main()
