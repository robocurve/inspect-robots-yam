"""Shared-memory frame transport tests for RealSense process isolation."""

from __future__ import annotations

import inspect
import multiprocessing
import struct
import threading
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import inspect_robots_yam._capture_proc as capture_proc
from conftest import FakeDevice, FakePipeline, FakeRs, frameset
from inspect_robots_yam._capture_proc import (
    _attach_frame_slot,
    _CaptureSpec,
    _child_main,
    _create_frame_slot,
    _FrameSlotSpec,
    _read_frame,
    _write_frame,
)


def _child_spec(
    slot_spec: _FrameSlotSpec,
    stop: threading.Event,
    *,
    serials: tuple[tuple[str, str], ...] = (("top_cam", "S1"),),
) -> _CaptureSpec:
    """Build one-slot child configuration for in-process entry tests."""
    return _CaptureSpec(
        serials=serials,
        depth_fps=15,
        slots=(("top_cam", slot_spec),),
        generation=4,
        stop_event=stop,
    )


def _record_unregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str]]:
    """Record child unregister calls without mutating this process's tracker."""
    unregisters: list[tuple[str, str]] = []
    monkeypatch.setattr(
        capture_proc.resource_tracker,
        "unregister",
        lambda name, kind: unregisters.append((name, kind)),
    )
    return unregisters


def _payload(
    fill: int = 7,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint16],
    npt.NDArray[np.float32],
]:
    """Build recognizable arrays for one tiny frame publication."""
    colour = np.full((3, 4, 3), fill, dtype=np.uint8)
    depth = np.full((3, 4), fill * 100, dtype=np.uint16)
    intrinsics = np.diag([float(fill), float(fill + 1), 1.0]).astype(np.float32)
    return colour, depth, intrinsics


def _publish(
    shm: shared_memory.SharedMemory,
    spec: _FrameSlotSpec,
    fill: int = 7,
) -> None:
    """Publish one tiny recognizable frame."""
    colour, depth, intrinsics = _payload(fill)
    _write_frame(
        shm,
        spec,
        colour=colour,
        depth=depth,
        intrinsics=intrinsics,
        depth_scale=0.001,
        published_s=12.5,
        generation=3,
    )


def test_frame_layout_round_trip_and_views_do_not_block_close() -> None:
    shm, spec = _create_frame_slot(4, 3)
    try:
        assert spec.layout.nbytes == 32 + 4 * 3 * 3 + 4 * 3 * 2 + 3 * 3 * 4
        assert _read_frame(shm, spec) is None

        _publish(shm, spec)
        snapshot = _read_frame(shm, spec)

        assert snapshot is not None
        colour, depth, intrinsics = _payload()
        np.testing.assert_array_equal(snapshot.colour, colour)
        np.testing.assert_array_equal(snapshot.depth, depth)
        np.testing.assert_array_equal(snapshot.intrinsics, intrinsics)
        assert snapshot.depth_scale == 0.001
        assert snapshot.published_s == 12.5
        assert snapshot.generation == 3
        arrays = (snapshot.colour, snapshot.depth, snapshot.intrinsics)
        assert all(array.flags.owndata for array in arrays)
        shm.close()
    finally:
        shm.unlink()


def test_torn_snapshot_is_retried_until_a_coherent_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, spec = _create_frame_slot(4, 3)
    try:
        _publish(shm, spec)
        original_copy = capture_proc._copy_payload
        copies = 0

        def copy_then_publish(buffer: Any, layout: capture_proc._FrameLayout) -> Any:
            nonlocal copies
            result = original_copy(buffer, layout)
            copies += 1
            if copies == 1:
                _publish(shm, spec, fill=9)
            return result

        monkeypatch.setattr(capture_proc, "_copy_payload", copy_then_publish)

        snapshot = _read_frame(shm, spec)

        assert snapshot is not None
        assert copies == 2
        assert np.all(snapshot.colour == 9)
    finally:
        shm.close()
        shm.unlink()


def test_odd_sequence_exhausts_bounded_retries() -> None:
    shm, spec = _create_frame_slot(4, 3)
    try:
        struct.pack_into("<Q", shm.buf, 0, 1)
        assert _read_frame(shm, spec, retries=2) is None
    finally:
        shm.close()
        shm.unlink()


def test_failed_publication_stays_marked_torn() -> None:
    shm, spec = _create_frame_slot(4, 3)
    colour, depth, intrinsics = _payload()
    try:
        with pytest.raises(ValueError):
            _write_frame(
                shm,
                spec,
                colour=colour[:, :-1],
                depth=depth,
                intrinsics=intrinsics,
                depth_scale=0.001,
                published_s=0.0,
                generation=1,
            )

        assert struct.unpack_from("<Q", shm.buf)[0] % 2 == 1
        assert _read_frame(shm, spec) is None
    finally:
        shm.close()
        shm.unlink()


def test_attach_uses_track_false_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = object()

    def fake_shared_memory(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    signature = inspect.Signature(
        [
            inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("track", inspect.Parameter.KEYWORD_ONLY),
        ]
    )
    monkeypatch.setattr(capture_proc.inspect, "signature", lambda _callable: signature)
    monkeypatch.setattr(capture_proc.shared_memory, "SharedMemory", fake_shared_memory)

    assert _attach_frame_slot(_FrameSlotSpec("slot", 4, 3)) is sentinel
    assert calls == [{"name": "slot", "track": False}]


def test_attach_unregisters_on_python_without_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSharedMemory:
        """Attached shared memory exposing its resource-tracker name."""

        def __init__(self) -> None:
            self._name = "/slot"

    unregisters: list[tuple[str, str]] = []
    signature = inspect.Signature([inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY)])
    monkeypatch.setattr(capture_proc.inspect, "signature", lambda _callable: signature)
    monkeypatch.setattr(
        capture_proc.shared_memory, "SharedMemory", lambda **_kwargs: FakeSharedMemory()
    )
    monkeypatch.setattr(
        capture_proc.resource_tracker,
        "unregister",
        lambda name, kind: unregisters.append((name, kind)),
    )

    attached = _attach_frame_slot(_FrameSlotSpec("slot", 4, 3))

    assert isinstance(attached, FakeSharedMemory)
    assert unregisters == [("/slot", "shared_memory")]


def test_child_opens_warms_publishes_and_stops_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, slot_spec = _create_frame_slot()
    stop = threading.Event()
    pipeline = FakePipeline([(False, None), (True, frameset()), (True, frameset())])
    pipeline.stop_after(stop, 3)
    rs = FakeRs([pipeline])
    parent_conn, child_conn = multiprocessing.Pipe()
    unregisters = _record_unregisters(monkeypatch)
    try:
        _child_main(child_conn, _child_spec(slot_spec, stop), rs_module=rs)

        assert parent_conn.recv() == ("ready", {"slots": ("top_cam",)})
        snapshot = _read_frame(shm, slot_spec)
        assert snapshot is not None
        assert snapshot.generation == 4
        assert snapshot.depth_scale == 0.001
        assert snapshot.colour[0, 0, 0] == 7
        assert rs.configs[0].streams == [
            ("colour", 640, 480, "rgb8", 15),
            ("depth", 640, 480, "z16", 15),
        ]
        assert pipeline.timeouts == [1000, 1000, 1000]
        assert pipeline.stopped
        assert unregisters == [(f"/{slot_spec.name}", "shared_memory")]
    finally:
        parent_conn.close()
        shm.close()
        shm.unlink()


def test_child_resolves_asic_serial_and_uses_lazy_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, slot_spec = _create_frame_slot()
    stop = threading.Event()
    stop.set()
    pipeline = FakePipeline()
    rs = FakeRs(
        [pipeline],
        [FakeDevice("Top D405", "DEVICE-SERIAL", "ASIC-SERIAL")],
    )
    parent_conn, child_conn = multiprocessing.Pipe()
    _record_unregisters(monkeypatch)
    monkeypatch.setattr(capture_proc, "_import_rs", lambda: rs)
    try:
        spec = _child_spec(
            slot_spec,
            stop,
            serials=(("top_cam", "ASIC-SERIAL"),),
        )
        _child_main(child_conn, spec)

        assert parent_conn.recv()[0] == "ready"
        assert rs.configs[0].device == "DEVICE-SERIAL"
        assert pipeline.stopped
    finally:
        parent_conn.close()
        shm.close()
        shm.unlink()


def test_child_accepts_empty_warmup_and_a_drain_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for responses, stop_call in (
        ([(False, None)], None),
        ([(True, frameset()), (False, None)], 2),
    ):
        shm, slot_spec = _create_frame_slot()
        stop = threading.Event()
        if stop_call is None:
            stop.set()
        pipeline = FakePipeline(responses)
        if stop_call is not None:
            pipeline.stop_after(stop, stop_call)
        parent_conn, child_conn = multiprocessing.Pipe()
        _record_unregisters(monkeypatch)
        monkeypatch.setattr(capture_proc.time, "sleep", lambda _seconds: None)
        try:
            _child_main(
                child_conn,
                _child_spec(slot_spec, stop),
                rs_module=FakeRs([pipeline]),
            )

            assert parent_conn.recv()[0] == "ready"
            assert pipeline.stopped
        finally:
            parent_conn.close()
            shm.close()
            shm.unlink()


@pytest.mark.parametrize(
    ("devices", "serials", "message"),
    (
        (
            [FakeDevice("Visible D405", "DEVICE", None)],
            (("top_cam", "MISSING"),),
            "visible devices: Visible D405 / DEVICE / <unavailable>",
        ),
        ([], (("top_cam", "MISSING"),), "visible devices: none"),
        (
            [FakeDevice("Top D405", "DEVICE", "ASIC")],
            (("top_cam", "DEVICE"), ("left_cam", "ASIC")),
            "both resolve to device serial DEVICE",
        ),
    ),
)
def test_child_reports_resolution_errors_before_open(
    monkeypatch: pytest.MonkeyPatch,
    devices: list[FakeDevice],
    serials: tuple[tuple[str, str], ...],
    message: str,
) -> None:
    shm, slot_spec = _create_frame_slot()
    parent_conn, child_conn = multiprocessing.Pipe()
    _record_unregisters(monkeypatch)
    try:
        _child_main(
            child_conn,
            _child_spec(slot_spec, threading.Event(), serials=serials),
            rs_module=FakeRs(devices=devices),
        )

        status, error = parent_conn.recv()
        assert status == "error"
        assert message in error
    finally:
        parent_conn.close()
        shm.close()
        shm.unlink()


def test_child_rolls_back_partial_open_and_swallows_stop_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, slot_spec = _create_frame_slot()
    second_shm, second_slot = _create_frame_slot()
    first = FakePipeline(stop_error=RuntimeError("stop also failed"))
    second = FakePipeline(start_error=RuntimeError("device busy"))
    rs = FakeRs([first, second])
    parent_conn, child_conn = multiprocessing.Pipe()
    _record_unregisters(monkeypatch)
    spec = _CaptureSpec(
        serials=(("top_cam", "S1"), ("left_cam", "S2")),
        depth_fps=30,
        slots=(("top_cam", slot_spec), ("left_cam", second_slot)),
        generation=1,
        stop_event=threading.Event(),
    )
    try:
        _child_main(child_conn, spec, rs_module=rs)

        assert parent_conn.recv() == ("error", "device busy")
        assert first.stopped
        assert not second.stopped
    finally:
        parent_conn.close()
        for handle in (shm, second_shm):
            handle.close()
            handle.unlink()


def test_child_stops_pipeline_when_warmup_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, slot_spec = _create_frame_slot()
    pipeline = FakePipeline(stop_error=RuntimeError("stop failed"))
    rs = FakeRs([pipeline])

    def failing_align(_stream: Any) -> None:
        raise RuntimeError("align failed")

    rs.align = failing_align
    parent_conn, child_conn = multiprocessing.Pipe()
    _record_unregisters(monkeypatch)
    try:
        _child_main(
            child_conn,
            _child_spec(slot_spec, threading.Event()),
            rs_module=rs,
        )

        assert parent_conn.recv() == ("error", "align failed")
        assert pipeline.stopped
    finally:
        parent_conn.close()
        shm.close()
        shm.unlink()


def test_child_sends_nothing_when_error_pipe_is_already_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, slot_spec = _create_frame_slot()
    parent_conn, child_conn = multiprocessing.Pipe()
    _record_unregisters(monkeypatch)
    parent_conn.close()
    try:
        _child_main(
            child_conn,
            _child_spec(slot_spec, threading.Event()),
            rs_module=FakeRs(devices=[]),
        )
    finally:
        shm.close()
        shm.unlink()


def test_child_exits_on_parent_pipe_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, slot_spec = _create_frame_slot()
    parent_conn, child_conn = multiprocessing.Pipe()
    pipeline = FakePipeline()
    _record_unregisters(monkeypatch)
    child = threading.Thread(
        target=_child_main,
        args=(child_conn, _child_spec(slot_spec, threading.Event())),
        kwargs={"rs_module": FakeRs([pipeline])},
    )
    try:
        child.start()
        assert parent_conn.recv()[0] == "ready"
        parent_conn.close()
        child.join(timeout=1.0)

        assert not child.is_alive()
        assert pipeline.stopped
    finally:
        if child.is_alive():
            pipeline.block.set()
            child.join(timeout=1.0)
        shm.close()
        shm.unlink()


def test_child_exits_on_control_message_and_on_drain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for response in ("control", RuntimeError("sensor failed")):
        shm, slot_spec = _create_frame_slot()
        parent_conn, child_conn = multiprocessing.Pipe()
        pipeline = FakePipeline(
            [(True, frameset()), response]
            if isinstance(response, BaseException)
            else [(True, frameset())]
        )
        _record_unregisters(monkeypatch)
        child = threading.Thread(
            target=_child_main,
            args=(child_conn, _child_spec(slot_spec, threading.Event())),
            kwargs={"rs_module": FakeRs([pipeline])},
        )
        try:
            child.start()
            assert parent_conn.recv()[0] == "ready"
            if response == "control":
                parent_conn.send(response)
            child.join(timeout=1.0)
            assert not child.is_alive()
            assert pipeline.stopped
        finally:
            parent_conn.close()
            shm.close()
            shm.unlink()


def test_child_drain_observes_stop_between_pipe_poll_and_camera_read() -> None:
    class StopOnSecondCheck:
        """Become set at the per-camera stop check."""

        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            """Return false for the outer loop and true inside the camera loop."""
            self.checks += 1
            return self.checks == 2

    parent_conn, child_conn = multiprocessing.Pipe()
    pipeline = FakePipeline()
    spec = _CaptureSpec(
        serials=(),
        depth_fps=30,
        slots=(),
        generation=1,
        stop_event=StopOnSecondCheck(),
    )
    try:
        capture_proc._drain_child(
            child_conn,
            spec,
            {"top_cam": capture_proc._ChildPipeline(pipeline, object(), 0.001)},
            {},
        )

        assert pipeline.calls == 0
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.parametrize(
    "aligned",
    (frameset(falsy_depth=True), frameset(falsy_colour=True)),
)
def test_child_ignores_incomplete_framesets(
    aligned: Any,
) -> None:
    shm, slot_spec = _create_frame_slot()
    try:
        capture_proc._publish_frameset(aligned, shm, slot_spec, 0.001, 1)
        assert _read_frame(shm, slot_spec) is None
    finally:
        shm.close()
        shm.unlink()
