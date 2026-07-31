"""RealSense colour/depth camera reader and injected-depth embodiment integration.

The librealsense surface is represented by small recording fakes. Drain loops
are driven synchronously where their generation and timeout behavior is the
subject; readers opened normally are closed by an autouse fixture.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import inspect_robots_yam.embodiment as embodiment_module
from conftest import (
    FakeAlign,
    FakeDevice,
    FakePipeline,
    FakeRs,
    frameset,
)
from inspect_robots_yam._capture_proc import MAX_FRAME_AGE_S
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import (
    YAMEmbodiment,
    _CompositeCameraReader,
    _PipelineBundle,
    _RealsenseCameraReader,
)
from inspect_robots_yam.operator import OperatorIO

SERIALS = {"top_cam": "S1", "left_cam": "S2", "right_cam": "S3"}
DEPTH_SCALE = 0.001

_OPENED: list[_RealsenseCameraReader] = []


@pytest.fixture(autouse=True)
def close_readers() -> Iterator[None]:
    """Close every reader a test built, however the test ended."""
    yield
    while _OPENED:
        _OPENED.pop().close()


class Clock:
    """A monotonic clock advanced only by tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the fake clock."""
        self.now += seconds


Response = tuple[bool, Any] | BaseException


class FakeCapture:
    """An always-readable V4L2 capture for embodiment-level builtin tests."""

    def __init__(self) -> None:
        self.calls = 0
        self.released = False

    def isOpened(self) -> bool:
        """Report a successful device open."""
        return True

    def set(self, _prop: int, _value: float) -> bool:
        """Accept every capture setting."""
        return True

    def read(self) -> tuple[bool, npt.NDArray[np.uint8]]:
        """Return a stable BGR frame without spinning a drain thread."""
        self.calls += 1
        if self.calls > 1:
            time.sleep(0.01)
        return True, np.full((480, 640, 3), 9, dtype=np.uint8)

    def release(self) -> None:
        """Record release by the composite reader."""
        self.released = True


class FakeCv2:
    """The colour resize and V4L2 surfaces used by builtin readers."""

    CAP_V4L2 = 200
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_FOURCC = 6
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_OPEN_TIMEOUT_MSEC = 53
    CAP_PROP_READ_TIMEOUT_MSEC = 54
    COLOR_BGR2RGB = 4

    class VideoWriter:
        """Namespace for the capture fourcc helper."""

        @staticmethod
        def fourcc(*_chars: str) -> float:
            """Return a harmless recognizable code."""
            return 1.0

    def __init__(self) -> None:
        self.captures: dict[str, FakeCapture] = {}

    def VideoCapture(self, device: str, _api: int) -> FakeCapture:
        """Return one persistent fake capture per configured path."""
        capture = FakeCapture()
        self.captures[device] = capture
        return capture

    def cvtColor(self, source: Any, code: int) -> npt.NDArray[np.uint8]:
        """Reverse BGR channels into RGB."""
        assert code == self.COLOR_BGR2RGB
        return np.asarray(source)[..., ::-1]

    def resize(self, source: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.uint8]:
        """Resize by integer nearest-neighbour indexing."""
        width, height = size
        rows = np.arange(height) * source.shape[0] // height
        columns = np.arange(width) * source.shape[1] // width
        return source[rows[:, None], columns].copy()


def build(
    *,
    serials: dict[str, str] | None = None,
    pipelines: list[FakePipeline] | None = None,
    devices: list[FakeDevice] | None = None,
    rs: FakeRs | None = None,
    cv2: FakeCv2 | None = None,
    clock: Clock | None = None,
    sleeps: list[float] | None = None,
    depth_fps: int = 30,
) -> tuple[_RealsenseCameraReader, FakeRs, FakeCv2, Clock, list[float]]:
    """Build a reader and all of its injected recording fakes."""
    rs = rs if rs is not None else FakeRs(pipelines, devices)
    cv2 = cv2 if cv2 is not None else FakeCv2()
    clock = clock if clock is not None else Clock()
    sleeps = sleeps if sleeps is not None else []
    reader = _RealsenseCameraReader(
        serials or SERIALS,
        depth_fps,
        rs_module=rs,
        cv2_module=cv2,
        sleep_fn=sleeps.append,
        clock=clock,
    )
    _OPENED.append(reader)
    return reader, rs, cv2, clock, sleeps


def cfg() -> YamConfig:
    """Use a tiny published size for fast assertions."""
    return YamConfig(cam_height=4, cam_width=4)


def drive(
    reader: _RealsenseCameraReader,
    name: str,
    pipeline: FakePipeline,
    responses: list[Response],
) -> None:
    """Drive one drain synchronously until its scripted final wait."""
    stop = threading.Event()
    pipeline.reset_responses(responses)
    pipeline.stop_after(stop, len(responses))
    bundle = _PipelineBundle(pipeline, FakeAlign(), DEPTH_SCALE)
    reader._drain(name, bundle, stop, reader._generation)


def test_images_depth_thunks_and_intrinsics_are_returned_for_every_camera() -> None:
    colour = np.dstack([np.full((480, 640), value, dtype=np.uint8) for value in (1, 2, 3)])
    depth = np.full((480, 640), 500, dtype=np.uint16)
    pipelines = [FakePipeline([(True, frameset(colour=colour, depth=depth))]) for _ in SERIALS]
    reader, rs, _, _, _ = build(pipelines=pipelines)

    images = reader(cfg())
    extra = reader.extra(cfg())

    assert set(images) == set(SERIALS)
    assert rs.context_value.query_calls == 1
    for name, image in images.items():
        assert image.shape == (4, 4, 3)
        assert image.dtype == np.uint8
        assert list(image[0, 0]) == [1, 2, 3]
        intrinsics = extra[f"{name}_intrinsics"]
        assert intrinsics.shape == (3, 3)
        assert intrinsics.dtype == np.float32
        depth_thunk = extra[f"{name}_depth"]
        assert callable(depth_thunk)
        resolved = depth_thunk()
        assert resolved.shape == (4, 4)
        assert resolved.dtype == np.float32
        assert np.all(resolved == np.float32(0.5))
    assert all(
        pipeline.config is not None
        and pipeline.config.streams
        == [
            ("colour", 640, 480, "rgb8", 30),
            ("depth", 640, 480, "z16", 30),
        ]
        for pipeline in pipelines
    )
    assert all(timeout == 1000 for pipeline in pipelines for timeout in pipeline.timeouts)


def test_inline_reader_threads_configured_depth_fps_to_both_streams() -> None:
    reader, rs, _, _, _ = build(serials={"top_cam": "S1"}, depth_fps=15)

    reader(cfg())

    assert rs.configs[0].streams == [
        ("colour", 640, 480, "rgb8", 15),
        ("depth", 640, 480, "z16", 15),
    ]


def test_rgb8_channel_order_is_not_swapped() -> None:
    colour = np.dstack([np.full((480, 640), value, dtype=np.uint8) for value in (1, 2, 3)])
    pipelines = [FakePipeline([(True, frameset(colour=colour))]) for _ in SERIALS]
    reader, _, _, _, _ = build(pipelines=pipelines)

    assert list(reader(cfg())["top_cam"][0, 0]) == [1, 2, 3]


def test_asic_serial_matches_but_enable_device_receives_device_serial() -> None:
    devices = [
        FakeDevice("Top D405", "DEVICE-T", "ASIC-T"),
        FakeDevice("Left D405", "DEVICE-L", "ASIC-L"),
        FakeDevice("Right D405", "DEVICE-R", "ASIC-R"),
    ]
    serials = {"top_cam": "ASIC-T", "left_cam": "ASIC-L", "right_cam": "ASIC-R"}
    reader, rs, _, _, _ = build(serials=serials, devices=devices)

    reader(cfg())

    assert [config.device for config in rs.configs] == [
        "DEVICE-T",
        "DEVICE-L",
        "DEVICE-R",
    ]


def test_device_and_asic_names_for_one_camera_are_rejected_before_open() -> None:
    devices = [FakeDevice("Top D405", "DEVICE-T", "ASIC-T")]
    pipelines = [FakePipeline(), FakePipeline()]
    reader, rs, _, _, _ = build(
        serials={"top_cam": "DEVICE-T", "left_cam": "ASIC-T"},
        pipelines=pipelines,
        devices=devices,
    )

    with pytest.raises(RuntimeError) as caught:
        reader(cfg())

    message = str(caught.value)
    assert "top_cam (DEVICE-T)" in message
    assert "left_cam (ASIC-T)" in message
    assert "both resolve to device serial DEVICE-T" in message
    assert rs.pipeline_calls == 0
    assert not any(pipeline.started for pipeline in pipelines)


def test_missing_serial_lists_every_visible_identity() -> None:
    devices = [
        FakeDevice("D405 Alpha", "DEVICE-A", "ASIC-A"),
        FakeDevice("D405 Beta", "DEVICE-B", None),
    ]
    reader, _, _, _, _ = build(
        serials={"top_cam": "MISSING"},
        devices=devices,
    )

    with pytest.raises(RuntimeError) as caught:
        reader(cfg())

    message = str(caught.value)
    assert "top_cam" in message and "MISSING" in message
    assert "D405 Alpha / DEVICE-A / ASIC-A" in message
    assert "D405 Beta / DEVICE-B / <unavailable>" in message


def test_missing_serial_with_no_visible_devices_says_none() -> None:
    reader, _, _, _, _ = build(serials={"top_cam": "MISSING"}, devices=[])

    with pytest.raises(RuntimeError, match="visible devices: none"):
        reader(cfg())


def test_empty_configured_serial_does_not_match_missing_asic_serial() -> None:
    devices = [FakeDevice("ASIC-less D405", "DEVICE-A", None)]
    reader, rs, _, _, _ = build(serials={"top_cam": ""}, devices=devices)

    with pytest.raises(RuntimeError, match=r"cannot find RealSense camera top_cam \(\)"):
        reader(cfg())

    assert rs.pipeline_calls == 0


def test_intrinsics_are_scaled_to_the_published_resolution() -> None:
    reader, _, _, _, _ = build()

    intrinsics = reader.extra(cfg())["top_cam_intrinsics"]

    expected = np.array(
        [[600 * 4 / 640, 0, 320 * 4 / 640], [0, 600 * 4 / 480, 240 * 4 / 480], [0, 0, 1]],
        dtype=np.float32,
    )
    assert np.array_equal(intrinsics, expected)


def test_depth_resize_is_nearest_neighbour_without_averaged_values() -> None:
    depth = np.empty((480, 640), dtype=np.uint16)
    depth[:240] = 1000
    depth[240:] = 3000
    pipelines = [FakePipeline([(True, frameset(depth=depth))]) for _ in SERIALS]
    reader, _, _, _, _ = build(pipelines=pipelines)

    resized = reader.extra(cfg())["top_cam_depth"]()

    assert np.allclose(
        resized,
        np.array(
            [[1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 3, 3], [3, 3, 3, 3]],
            dtype=np.float32,
        ),
    )
    assert all(np.isclose(value, 1.0) or np.isclose(value, 3.0) for value in np.unique(resized))


def test_one_depth_thunk_resolves_the_newest_pair_each_time() -> None:
    pipelines = [FakePipeline() for _ in SERIALS]
    reader, _, _, _, _ = build(pipelines=pipelines)
    thunk = reader.extra(cfg())["top_cam_depth"]
    first = thunk()
    reader._stop.set()
    for thread in reader._threads.values():
        thread.join(timeout=1.0)

    newer = frameset(depth=np.full((480, 640), 2000, dtype=np.uint16))
    drive(reader, "top_cam", pipelines[0], [(True, newer)])
    second = thunk()

    assert np.all(first == 1.0)
    assert np.all(second == 2.0)
    assert first is not second


def test_depth_thunk_rejects_resolution_after_close() -> None:
    reader, _, _, _, _ = build()
    thunk = reader.extra(cfg())["top_cam_depth"]

    reader.close()

    with pytest.raises(
        RuntimeError,
        match="depth for top_cam resolved after camera close or reopen",
    ):
        thunk()


def test_old_depth_thunk_stays_retired_after_close_and_reopen() -> None:
    reader, _, _, _, _ = build()
    thunk = reader.extra(cfg())["top_cam_depth"]
    reader.close()

    reader(cfg())

    with pytest.raises(RuntimeError, match="resolved after camera close or reopen"):
        thunk()


def test_warm_up_timeout_retries_then_delivers_images() -> None:
    pipelines = [
        FakePipeline([(False, None), (True, frameset())]),
        FakePipeline(),
        FakePipeline(),
    ]
    reader, _, _, _, sleeps = build(pipelines=pipelines)

    assert set(reader(cfg())) == set(SERIALS)
    assert sleeps == [0.1]
    assert pipelines[0].timeouts[:2] == [1000, 1000]


def test_drain_timeout_continues_and_publishes_the_next_pair() -> None:
    reader, _, _, _, _ = build()
    pipeline = FakePipeline()

    drive(reader, "top_cam", pipeline, [(False, None), (True, frameset())])

    pair, _ = reader._latest("top_cam")
    assert pair.depth[0, 0] == 1000


def test_real_drain_exception_is_latched_and_reraised_by_call() -> None:
    pipelines = [
        FakePipeline([(True, frameset()), RuntimeError("sensor failed")]),
        FakePipeline(),
        FakePipeline(),
    ]
    reader, _, _, _, _ = build(pipelines=pipelines)

    with pytest.raises(RuntimeError, match=r"camera top_cam \(S1\) stopped reading") as caught:
        reader(cfg())

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_retired_drain_cannot_publish_or_fault_the_new_cycle() -> None:
    reader, _, _, _, _ = build()
    reader._generation = 2
    pipeline = FakePipeline()
    stop = threading.Event()
    pipeline.reset_responses([(True, frameset()), KeyboardInterrupt()])
    pipeline.stop_after(stop, 2)
    bundle = _PipelineBundle(pipeline, FakeAlign(), DEPTH_SCALE)

    reader._drain(
        "top_cam",
        bundle,
        stop,
        generation=1,
    )

    assert "top_cam" not in reader._published
    assert "top_cam" not in reader._faults


def test_open_failure_rolls_back_every_pipeline_that_started() -> None:
    pipelines = [
        FakePipeline(),
        FakePipeline(),
        FakePipeline(start_error=RuntimeError("device busy")),
    ]
    reader, _, _, _, _ = build(pipelines=pipelines)

    with pytest.raises(RuntimeError, match="device busy"):
        reader(cfg())

    assert pipelines[0].stopped
    assert pipelines[1].stopped
    assert not pipelines[2].stopped


def test_warm_up_failure_stops_in_flight_and_rolled_back_pipelines() -> None:
    pipelines = [
        FakePipeline(),
        FakePipeline([RuntimeError("device disconnected")]),
    ]
    reader, _, _, _, _ = build(
        serials={"top_cam": "S1", "left_cam": "S2"},
        pipelines=pipelines,
    )

    with pytest.raises(RuntimeError, match="device disconnected"):
        reader(cfg())

    assert all(pipeline.stopped for pipeline in pipelines)
    assert reader._published == {}


def test_close_stops_pipelines_and_is_idempotent_in_both_states() -> None:
    pipelines = [FakePipeline() for _ in SERIALS]
    reader, _, _, _, _ = build(pipelines=pipelines)

    reader.close()
    reader.close()
    assert not any(pipeline.started for pipeline in pipelines)

    reader(cfg())
    reader.close()
    reader.close()
    assert all(pipeline.stopped for pipeline in pipelines)


def test_pipeline_stop_error_is_swallowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipelines = [
        FakePipeline(stop_error=RuntimeError("stop failed")),
        FakePipeline(),
        FakePipeline(),
    ]
    reader, _, _, _, _ = build(pipelines=pipelines)
    reader(cfg())

    reader.close()

    assert "stopping RealSense pipeline for top_cam failed" in caplog.text
    assert all(pipeline.stopped for pipeline in pipelines)


def test_close_leaks_pipeline_whose_drain_is_still_alive() -> None:
    pipelines = [
        FakePipeline(block_after=2),
        FakePipeline(),
        FakePipeline(),
    ]
    reader, _, _, _, _ = build(pipelines=pipelines)
    reader(cfg())
    assert pipelines[0].entered.wait(timeout=1.0)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(embodiment_module, "JOIN_TIMEOUT_S", 0.05)
        reader.close()
        pipelines[0].block.set()

    assert not pipelines[0].stopped
    assert pipelines[1].stopped
    assert pipelines[2].stopped


def test_stale_pair_raises_with_camera_name_and_serial() -> None:
    clock = Clock()
    reader, _, _, _, sleeps = build(clock=clock)
    reader(cfg())
    reader._stop.set()
    for thread in reader._threads.values():
        thread.join(timeout=1.0)
    clock.advance(MAX_FRAME_AGE_S + 0.01)

    with pytest.raises(RuntimeError, match=r"frame read failed for top_cam \(S1\)"):
        reader(cfg())

    assert sleeps == [0.05] * 10


def test_warm_up_falsy_frames_publish_nothing_but_drain_can_recover() -> None:
    pipelines = [
        FakePipeline([(True, frameset(falsy_depth=True)), (True, frameset())]),
        FakePipeline([(True, frameset(falsy_colour=True)), (True, frameset())]),
        FakePipeline(),
    ]
    reader, _, _, _, _ = build(pipelines=pipelines)
    reader._sleep = time.sleep

    assert set(reader(cfg())) == set(SERIALS)


def test_camera_that_never_warms_or_drains_fails_at_consumption() -> None:
    pipelines = [FakePipeline([(False, None)]) for _ in SERIALS]
    reader, _, _, _, sleeps = build(pipelines=pipelines)

    with pytest.raises(RuntimeError, match=r"frame read failed for top_cam \(S1\)"):
        reader(cfg())

    assert sleeps.count(0.1) == 30
    assert sleeps.count(0.05) == 10


def test_raw_sdk_buffers_are_copied_before_publication() -> None:
    colour = np.full((480, 640, 3), 4, dtype=np.uint8)
    depth = np.full((480, 640), 900, dtype=np.uint16)
    reader, _, _, _, _ = build()

    reader._publish("top_cam", frameset(colour=colour, depth=depth), DEPTH_SCALE, 0)
    colour[:] = 8
    depth[:] = 1800
    pair, _ = reader._latest("top_cam")

    assert pair.colour[0, 0, 0] == 4
    assert pair.depth[0, 0] == 900


def test_extra_can_open_first_and_modules_are_resolved_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rs = FakeRs()
    cv2 = FakeCv2()
    monkeypatch.setattr(embodiment_module, "_import_rs", lambda: rs)
    monkeypatch.setattr(embodiment_module, "_import_cv2", lambda: cv2)
    reader = _RealsenseCameraReader(
        SERIALS,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
    )
    _OPENED.append(reader)

    extra = reader.extra(cfg())
    images = reader(cfg())

    assert callable(extra["top_cam_depth"])
    assert images["top_cam"].shape == (4, 4, 3)
    assert reader._rs is rs
    assert reader._cv2 is cv2


def test_yamconfig_all_serials() -> None:
    cfg_value = YamConfig(
        top_depth_serial="S1",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )
    assert cfg_value.top_depth_serial == "S1"


def test_yamconfig_device_and_serial_for_one_slot_are_rejected() -> None:
    message = (
        "top_cam_device and top_depth_serial are mutually exclusive: "
        "a RealSense camera opened through librealsense cannot also be opened "
        "through V4L2 \\(one streamer per node\\)"
    )
    with pytest.raises(ValueError, match=message):
        YamConfig(
            top_cam_device="/dev/video0",
            top_depth_serial="S1",
            left_depth_serial="S2",
            right_depth_serial="S3",
        )


@pytest.mark.parametrize(
    "field",
    (
        "top_cam_device",
        "left_cam_device",
        "right_cam_device",
        "top_depth_serial",
        "left_depth_serial",
        "right_depth_serial",
    ),
)
@pytest.mark.parametrize("value", ("", " \t"))
def test_yamconfig_empty_camera_source_is_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a non-empty string"):
        YamConfig(**{field: value})


def test_yamconfig_none_ok() -> None:
    cfg_value = YamConfig()
    assert cfg_value.top_depth_serial is None
    assert cfg_value.realsense_capture == "process"
    assert cfg_value.depth_fps == 30


@pytest.mark.parametrize("value", ("thread", "", None))
def test_yamconfig_realsense_capture_mode_is_validated(value: Any) -> None:
    with pytest.raises(ValueError, match=r"realsense_capture must be one of.*inline.*process"):
        YamConfig(realsense_capture=value)


@pytest.mark.parametrize("value", (0, 91, 15.0, True))
def test_yamconfig_depth_fps_is_validated(value: Any) -> None:
    with pytest.raises(ValueError, match="depth_fps must be an integer from 1 to 90"):
        YamConfig(depth_fps=value)


@pytest.mark.parametrize("field", ("top_depth_serial", "left_depth_serial", "right_depth_serial"))
def test_from_kwargs_guides_int_depth_serials(field: str) -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"{field} must be a string; quote the serial in config.ini — "
            "numeric values are int-coerced and lose leading zeros"
        ),
    ):
        YamConfig.from_kwargs(**{field: 38212071234})


def cameras(_config: YamConfig) -> dict[str, Any]:
    """Return three tiny injected colour images."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": image, "left_cam": image, "right_cam": image}


def driver_factory(_config: YamConfig) -> Any:
    """Return a minimal injected arm driver."""

    class Driver:
        def get_joint_pos(self) -> np.ndarray:
            return np.zeros(14)

        def command_joint_pos(self, target: np.ndarray) -> None:
            del target

        def close(self) -> None:
            pass

    return Driver()


def silent_operator() -> OperatorIO:
    """Return operator I/O that never blocks."""
    return OperatorIO(input_fn=lambda _: "", output_fn=lambda _: None)


def builtin_embodiment(
    config: YamConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    depth_reader: Any = None,
) -> tuple[YAMEmbodiment, FakeRs, FakeCv2]:
    """Build an embodiment whose lazy builtin imports resolve to camera fakes."""
    rs = FakeRs()
    cv2 = FakeCv2()
    monkeypatch.setattr(embodiment_module, "_import_rs", lambda: rs)
    monkeypatch.setattr(embodiment_module, "_import_cv2", lambda: cv2)
    emb = YAMEmbodiment(
        config,
        driver_factory=driver_factory,
        depth_reader=depth_reader,
        operator=silent_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
        status_fn=lambda _: None,
    )
    return emb, rs, cv2


def observe_once(emb: YAMEmbodiment) -> Any:
    """Observe through builtin cameras, then release their lazy-opened devices."""
    emb._driver = driver_factory(emb._cfg)
    try:
        return emb._observe("task")
    finally:
        emb._driver = None
        emb.close()


def embodiment(depth_reader: Any = None) -> YAMEmbodiment:
    """Build an embodiment with all hardware seams injected."""
    return YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4),
        driver_factory=driver_factory,
        camera_reader=cameras,
        depth_reader=depth_reader,
        operator=silent_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
        status_fn=lambda _: None,
    )


class Action:
    """A zero joint-position action."""

    data = np.zeros(14)


def test_mixed_builtin_rig_merges_images_and_only_serial_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_value = YamConfig(
        cam_height=4,
        cam_width=4,
        top_cam_device="/dev/top",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )
    emb, rs, cv2 = builtin_embodiment(cfg_value, monkeypatch)

    observation = observe_once(emb)

    assert set(observation.images) == {"top_cam", "left_cam", "right_cam"}
    assert all(image.shape == (4, 4, 3) for image in observation.images.values())
    assert observation.extra is not None
    assert callable(observation.extra["left_cam_depth"])
    assert callable(observation.extra["right_cam_depth"])
    assert observation.extra["left_cam_intrinsics"].shape == (3, 3)
    assert observation.extra["right_cam_intrinsics"].shape == (3, 3)
    assert "top_cam_depth" not in observation.extra
    assert "top_cam_intrinsics" not in observation.extra
    assert cv2.captures["/dev/top"].released
    assert all(pipeline.stopped for pipeline in rs.pipelines)


def test_serial_only_rig_constructs_builtin_reader_and_observes_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_value = YamConfig(
        cam_height=4,
        cam_width=4,
        top_depth_serial="S1",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )
    emb, _, _ = builtin_embodiment(cfg_value, monkeypatch)
    assert isinstance(emb._builtin_realsense_reader, _RealsenseCameraReader)

    observation = observe_once(emb)

    assert set(observation.images) == {"top_cam", "left_cam", "right_cam"}
    assert all(image.shape == (4, 4, 3) for image in observation.images.values())
    assert observation.extra is not None
    for name in ("top_cam", "left_cam", "right_cam"):
        assert callable(observation.extra[f"{name}_depth"])
        assert observation.extra[f"{name}_intrinsics"].shape == (3, 3)


def test_serial_only_rig_includes_depth_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_value = YamConfig(
        top_depth_serial="S1",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )
    emb, _, _ = builtin_embodiment(cfg_value, monkeypatch)

    docs = emb.info.docs
    assert "Depth:" in docs
    assert "arrives either as an H\u00d7W float32 array of depth in metres" in docs
    assert "or as a zero-arg callable returning that array" in docs
    assert "If it is callable, resolve it immediately on receipt" in docs
    assert "ZERO-ARG CALLABLE" not in docs
    emb.close()


def test_mixed_builtin_rig_depth_docs_name_only_serial_cameras() -> None:
    emb = YAMEmbodiment(
        YamConfig(
            top_cam_device="/dev/top",
            left_depth_serial="S2",
            right_depth_serial="S3",
        )
    )

    depth_docs = emb.info.docs.split("\n\nDepth:", maxsplit=1)[1]
    assert depth_docs.startswith(" for each serial-configured camera (left_cam, right_cam),")
    assert "top_cam" not in depth_docs
    assert "aligned to" in depth_docs
    assert "pixel-aligned" not in depth_docs
    emb.close()


def test_injected_only_depth_docs_make_no_specific_key_claim() -> None:
    emb = embodiment(depth_reader=lambda _: {})

    depth_docs = emb.info.docs.split("\n\nDepth:", maxsplit=1)[1]
    assert "may contain additional depth data" in depth_docs
    assert "consult its documentation" in depth_docs
    assert "{cam}_depth" not in depth_docs
    assert "{cam}_intrinsics" not in depth_docs
    assert "ZERO-ARG CALLABLE" not in depth_docs
    emb.close()


def test_injected_camera_reader_conflicts_with_configured_serials() -> None:
    cfg_value = YamConfig(
        top_depth_serial="S1",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )

    with pytest.raises(
        ValueError,
        match=(
            "configured depth serials drive the builtin capture path; with "
            "a custom camera_reader, supply depth via depth_reader instead"
        ),
    ):
        YAMEmbodiment(cfg_value, camera_reader=cameras)


def test_injected_depth_reader_merges_with_device_only_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_value = YamConfig(
        cam_height=4,
        cam_width=4,
        top_cam_device="/dev/top",
        left_cam_device="/dev/left",
        right_cam_device="/dev/right",
    )
    injected = {"custom_depth": "device path"}
    emb, _, _ = builtin_embodiment(
        cfg_value,
        monkeypatch,
        depth_reader=lambda _: injected,
    )

    observation = observe_once(emb)

    assert observation.extra == injected


def test_injected_depth_reader_overrides_serial_builtin_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_value = YamConfig(
        cam_height=4,
        cam_width=4,
        top_depth_serial="S1",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )
    replacement = np.eye(3, dtype=np.float32) * 7
    emb, _, _ = builtin_embodiment(
        cfg_value,
        monkeypatch,
        depth_reader=lambda _: {
            "top_cam_intrinsics": replacement,
            "custom_depth": "serial path",
        },
    )
    assert "injected ``depth_reader`` may add or override keys" in emb.info.docs

    observation = observe_once(emb)

    assert observation.extra is not None
    assert observation.extra["top_cam_intrinsics"] is replacement
    assert observation.extra["custom_depth"] == "serial path"
    assert callable(observation.extra["top_cam_depth"])


def test_composite_close_skips_reader_without_close() -> None:
    closed: list[str] = []

    class Closing:
        def __call__(self, _config: YamConfig) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            closed.append("closed")

    composite = _CompositeCameraReader(Closing(), lambda _config: {})
    composite.close()

    assert closed == ["closed"]


def test_observe_no_depth_reader() -> None:
    from inspect_robots.scene import Scene

    emb = embodiment()
    emb.reset(Scene(id="s", instruction="t"))
    result = emb.step(Action())
    emb.close()
    observation = result.observation
    assert not hasattr(observation, "extra") or observation.extra is None or observation.extra == {}


def test_observe_with_injected_depth_reader() -> None:
    from inspect_robots.scene import Scene

    depth_extra: dict[str, Any] = {
        "top_cam_depth": np.zeros((4, 4), dtype=np.float32),
        "top_cam_intrinsics": {"fx": 1.0},
        "left_cam_depth": np.zeros((4, 4), dtype=np.float32),
        "left_cam_intrinsics": {"fx": 1.0},
        "right_cam_depth": np.zeros((4, 4), dtype=np.float32),
        "right_cam_intrinsics": {"fx": 1.0},
    }
    emb = embodiment(depth_reader=lambda _: depth_extra)
    emb.reset(Scene(id="s", instruction="t"))
    result = emb.step(Action())
    emb.close()
    observation = result.observation
    assert observation.extra is not None
    assert "top_cam_depth" in observation.extra


def test_close_releases_injected_depth_reader() -> None:
    closed: list[bool] = []

    class DepthReader:
        def __call__(self, _config: YamConfig) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            closed.append(True)

    embodiment(depth_reader=DepthReader()).close()
    assert closed == [True]


def test_close_depth_reader_release_error_is_swallowed() -> None:
    class FailingDepthReader:
        def __call__(self, _config: YamConfig) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            raise RuntimeError("release failed")

    embodiment(depth_reader=FailingDepthReader()).close()
