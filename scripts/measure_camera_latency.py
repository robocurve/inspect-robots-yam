"""Measure V4L2 frame staleness on the rig's cameras (robocurve/inspect-robots-yam#63).

Reads from ``/dev/video*`` and toggles a UVC brightness control. No CAN traffic,
no torque, no robot motion, so this is safe to run with the arms powered down.

The question it answers: when the rollout consumes frames at ``control_hz``
while the cameras free-run at their native rate, how old is the frame
``cap.read()`` hands back, and which fix collapses that age?

Two independent signals, because they measure different things and the obvious
one is misleading on its own:

**Age (primary).** A UVC brightness step is a scene change with a known
timestamp. Set it at ``t0``, keep consuming at the control rate, and the wall
time of the first frame that shows the step is the end-to-end latency. This
includes a fixed sensor/ISP delay for applying the control, identical across
configurations, so compare the differences rather than the absolute numbers.
Resolution is one consumer period, which is enough to separate the candidates.

**Backlog (secondary).** A queued frame returns in microseconds; an empty queue
blocks for a full frame interval. Counting instant returns before the first
blocking read gives the queue depth. Depth is *not* age: a saturated ring
drained one-in-one-out advances one frame per **consumer** period, so a 4-deep
queue at 10 Hz holds a 400 ms old frame, not a 133 ms old one.

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

# Tight reads used to time the producer once a queue is drained.
_INTERVAL_SAMPLES = 15

# Brightness endpoints for the step, and the mean-intensity swing that counts as
# having seen it. The probe measured a ~95 level swing, so this is well clear of
# auto-exposure drift without being so tight that a dim scene misses the step.
_BRIGHTNESS_LOW = 0.0
_BRIGHTNESS_HIGH = 64.0
_STEP_THRESHOLD = 30.0

# Give up on one age trial rather than hang if the control never lands.
_AGE_TIMEOUT_S = 3.0

# Trials alternate the step up and down, so between them the pipeline has to be
# flushed of the previous level. Without this the baseline can be read from a
# queued frame the step has already overtaken, and the next read then differs
# from it immediately: a false detection that reports a latency below the
# camera's own control-apply delay.
_SETTLE_S = 0.5


class _Source:
    """One camera, read the way the configuration under test reads it."""

    def __init__(self, cap: Any, device: str, brightness_prop: int) -> None:
        self._cap = cap
        self._device = device
        self._brightness_prop = brightness_prop

    def set_brightness(self, level: float) -> None:
        """Step the brightness control, the scene change the age trial times."""
        self._cap.set(self._brightness_prop, level)

    def read(self) -> tuple[float, Any]:
        """Return how long the read took and the frame it produced."""
        start = time.perf_counter()
        ok, frame = self._cap.read()
        elapsed = time.perf_counter() - start
        if not ok or frame is None:
            raise SystemExit(f"frame read failed for {self._device}")
        return elapsed, frame

    def close(self) -> None:
        """Restore the brightness this measurement stepped, then release."""
        self._cap.set(self._brightness_prop, _BRIGHTNESS_LOW)
        self._cap.release()


class _GrabberSource(_Source):
    """A camera drained by a daemon thread, exposing only its newest frame.

    The thread owns the capture exclusively and publishes each frame into a slot
    under a lock. Splitting `grab()` and `retrieve()` across threads would race
    on the capture's internal state, so the consumer never touches it.
    """

    def __init__(self, cap: Any, device: str, brightness_prop: int) -> None:
        super().__init__(cap, device, brightness_prop)
        self._lock = threading.Lock()
        self._frame: Any = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        while self._frame is None and not self._stop.is_set():
            time.sleep(0.005)

    def _drain(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[float, Any]:
        """Return the newest published frame without ever blocking on capture."""
        start = time.perf_counter()
        with self._lock:
            frame = self._frame
        if frame is None:  # pragma: no cover - only before the first publish
            raise SystemExit(f"no frame published for {self._device}")
        return time.perf_counter() - start, frame

    def close(self) -> None:
        """Stop the drain thread before touching the capture it owns."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        super().close()

    def set_brightness(self, level: float) -> None:
        """Step brightness while the drain thread runs.

        Unsynchronized on purpose: taking the drain lock would stall the step by
        up to a frame interval, which is the same order as the latency being
        measured. The two are different ioctls on one fd, serialized in-kernel.
        """
        self._cap.set(self._brightness_prop, level)


@dataclass(frozen=True)
class CameraResult:
    """One camera measured under one configuration."""

    name: str
    interval_s: float
    backlog: int | None
    ages_s: tuple[float, ...]

    @property
    def fps(self) -> float:
        """Producer rate implied by the drained tight-loop interval."""
        return 1.0 / self.interval_s if self.interval_s > 0 else float("nan")

    @property
    def median_age_s(self) -> float:
        """Median end-to-end age of the frame the consumer receives."""
        return statistics.median(self.ages_s) if self.ages_s else float("nan")


@dataclass(frozen=True)
class Run:
    """Everything measured under one configuration."""

    label: str
    results: tuple[CameraResult, ...]
    sweep_s: float

    @property
    def worst_age_s(self) -> float:
        """Oldest median frame across cameras: what sets end-to-end staleness."""
        return max((r.median_age_s for r in self.results), default=float("nan"))


def _open(cv2: Any, device: str, buffersize: int | None) -> Any:
    """Open one capture configured the way the rollout configures it."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {device}")
    if buffersize is not None:
        # Set before format negotiation, where a driver that honors the property
        # wants it. Requested, not guaranteed: that is part of the measurement.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MSEC)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MSEC)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, _BRIGHTNESS_LOW)
    for _ in range(_WARMUP_READS):
        if cap.read()[0]:
            break
        time.sleep(0.1)
    return cap


def _sweep(sources: dict[str, _Source], hz: float, duration_s: float) -> float:
    """Drive every camera at the control rate until the queues reach steady state.

    Returns the median wall time one full sweep costs. That is the `o` term in
    #62's `o + s <= 1/hz` control budget: capture done inside the control period
    is time the arm does not get to converge before it is photographed.
    """
    period = 1.0 / hz
    deadline = time.perf_counter() + duration_s
    sweeps: list[float] = []
    while time.perf_counter() < deadline:
        cycle_start = time.perf_counter()
        for source in sources.values():
            source.read()
        sweeps.append(time.perf_counter() - cycle_start)
        time.sleep(max(0.0, period - (time.perf_counter() - cycle_start)))
    return statistics.median(sweeps)


def _measure_age(
    sources: dict[str, _Source],
    target: str,
    hz: float,
    trials: int,
) -> tuple[float, ...]:
    """Step the target's brightness and time the first frame that shows it.

    Every camera keeps being read at `hz` throughout, so the target's queue stays
    in the same regime the rollout puts it in. Trials alternate the step up and
    down, which avoids waiting for auto-exposure to settle back between them.
    """
    period = 1.0 / hz
    ages: list[float] = []
    level = _BRIGHTNESS_HIGH
    for _ in range(trials):
        _sweep(sources, hz, _SETTLE_S)
        baseline = float(sources[target].read()[1].mean())
        t0 = time.perf_counter()
        sources[target].set_brightness(level)
        deadline = t0 + _AGE_TIMEOUT_S
        while time.perf_counter() < deadline:
            cycle_start = time.perf_counter()
            seen: float | None = None
            for name, source in sources.items():
                _, frame = source.read()
                if name == target and abs(float(frame.mean()) - baseline) > _STEP_THRESHOLD:
                    seen = time.perf_counter() - t0
            if seen is not None:
                ages.append(seen)
                break
            time.sleep(max(0.0, period - (time.perf_counter() - cycle_start)))
        else:  # pragma: no cover - only if the control never lands
            raise SystemExit(f"brightness step never appeared on {target}")
        level = _BRIGHTNESS_LOW if level == _BRIGHTNESS_HIGH else _BRIGHTNESS_HIGH
    return tuple(ages)


def _measure_floor(cv2: Any, devices: dict[str, str], trials: int) -> dict[str, float]:
    """Age measured by a consumer that never falls behind: the instrument floor.

    A tight read loop keeps the queue drained, so whatever latency remains is not
    staleness. It is the sensor and ISP delay in applying the brightness control,
    plus one frame interval, and it differs per camera model (the context D435
    and the wrist D405s do not respond alike). Every age below sits on top of
    this, so a configuration that measures at the floor has no backlog left to
    remove: the instrument cannot see one.
    """
    floors: dict[str, float] = {}
    for name, device in devices.items():
        source = _Source(_open(cv2, device, 1), device, cv2.CAP_PROP_BRIGHTNESS)
        try:
            ages: list[float] = []
            level = _BRIGHTNESS_HIGH
            for _ in range(trials):
                settle_until = time.perf_counter() + _SETTLE_S
                while time.perf_counter() < settle_until:
                    source.read()
                baseline = float(source.read()[1].mean())
                t0 = time.perf_counter()
                source.set_brightness(level)
                while time.perf_counter() - t0 < _AGE_TIMEOUT_S:
                    if abs(float(source.read()[1].mean()) - baseline) > _STEP_THRESHOLD:
                        ages.append(time.perf_counter() - t0)
                        break
                else:  # pragma: no cover - only if the control never lands
                    raise SystemExit(f"brightness step never appeared on {name}")
                level = _BRIGHTNESS_LOW if level == _BRIGHTNESS_HIGH else _BRIGHTNESS_HIGH
            floors[name] = statistics.median(ages)
        finally:
            source.close()
    return floors


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
    trials: int,
) -> Run:
    """Open every camera together, saturate, then measure age and backlog."""
    factory = _GrabberSource if grabber else _Source
    sources: dict[str, _Source] = {
        name: factory(_open(cv2, device, buffersize), device, cv2.CAP_PROP_BRIGHTNESS)
        for name, device in devices.items()
    }
    try:
        sweep_s = _sweep(sources, hz, duration_s)
        results = []
        for name, source in sources.items():
            ages = _measure_age(sources, name, hz, trials)
            # A grabber's queue is drained by its thread, so a burst measured
            # through the latest-frame slot would report the slot, not the queue.
            backlog, interval = (None, 1.0 / 30.0) if grabber else _measure_backlog(source, burst)
            results.append(CameraResult(name, interval, backlog, ages))
        return Run(label, tuple(results), sweep_s)
    finally:
        for source in sources.values():
            source.close()


def _excess(run: Run, floors: dict[str, float]) -> float:
    """Worst staleness above the instrument floor: the part a fix can remove."""
    return max((r.median_age_s - floors[r.name] for r in run.results), default=float("nan"))


def _print_run(run: Run, floors: dict[str, float], hz: float) -> None:
    """Print one configuration's per-camera rows and its sweep cost."""
    print(f"\n=== {run.label} ===")
    for r in run.results:
        ages = " ".join(f"{a * 1000:.0f}" for a in r.ages_s)
        backlog = "drained by thread" if r.backlog is None else str(r.backlog)
        print(
            f"  {r.name:<11} {r.fps:5.1f} fps  age={r.median_age_s * 1000:6.1f} ms"
            f"  (+{(r.median_age_s - floors[r.name]) * 1000:6.1f} over floor)"
            f"  backlog={backlog}"
        )
        print(f"    per-trial ages (ms): {ages}")
    print(
        f"  sweep of {len(run.results)} cameras: {run.sweep_s * 1000:.1f} ms"
        f" of a {1000.0 / hz:.1f} ms control period"
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
    parser.add_argument("--duration", type=float, default=5.0, help="seconds of steady-state load")
    parser.add_argument("--burst", type=int, default=8, help="timed reads after the load phase")
    parser.add_argument("--trials", type=int, default=5, help="brightness steps per camera")
    args = parser.parse_args()

    devices: dict[str, str] = {}
    for spec in args.device:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--device wants NAME=PATH, got {spec!r}")
        devices[name] = path

    import cv2

    print(
        f"cv2 {cv2.__version__}  consumer {args.hz} Hz  load {args.duration}s"
        f"  burst {args.burst}  trials {args.trials}"
    )
    floors = _measure_floor(cv2, devices, args.trials)
    print("\ninstrument floor (tight-loop consumer, no backlog possible):")
    for name, floor in floors.items():
        print(f"  {name:<11} {floor * 1000:6.1f} ms")

    configs = (("default", None, False), ("buffersize=1", 1, False), ("grabber", 1, True))
    runs = []
    for label, buffersize, grabber in configs:
        run = _run(
            cv2,
            devices,
            label,
            buffersize,
            grabber,
            args.hz,
            args.duration,
            args.burst,
            args.trials,
        )
        _print_run(run, floors, args.hz)
        runs.append(run)

    print("\nworst staleness above the floor (the part a fix can remove):")
    for run in runs:
        print(f"  {run.label:<14} {_excess(run, floors) * 1000:6.1f} ms")


if __name__ == "__main__":
    main()
