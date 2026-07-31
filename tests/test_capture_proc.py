"""Shared-memory frame transport tests for RealSense process isolation."""

from __future__ import annotations

import inspect
import struct
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import inspect_robots_yam._capture_proc as capture_proc
from inspect_robots_yam._capture_proc import (
    _attach_frame_slot,
    _create_frame_slot,
    _FrameSlotSpec,
    _read_frame,
    _write_frame,
)


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
