"""Shared-memory transport for isolated RealSense capture.

The writer uses a single-slot seqlock: an odd sequence means a write is in
progress, and an unchanged even sequence brackets a coherent snapshot. Pure
Python cannot provide memory fences, so this relies on x86-TSO on the deployed
robots. If arm64 robots become a target, replace it with a double buffer and a
published index (or add a payload checksum).

This module intentionally imports only the standard library and NumPy. Spawned
capture children import it without pulling hardware dependencies into their
startup path.
"""

from __future__ import annotations

import inspect
import struct
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Any

import numpy as np
import numpy.typing as npt

REALSENSE_CAPTURE_WIDTH = 640
REALSENSE_CAPTURE_HEIGHT = 480
MAX_FRAME_AGE_S = 0.5
JOIN_TIMEOUT_S = 2.0
SEQLOCK_READ_RETRIES = 3

_HEADER = struct.Struct("<QdQd")
_SEQUENCE = struct.Struct("<Q")


@dataclass(frozen=True)
class _FrameLayout:
    """Byte offsets and shapes for one shared colour/depth frame slot."""

    width: int
    height: int

    @property
    def colour_offset(self) -> int:
        """Return the first byte of the RGB8 payload."""
        return _HEADER.size

    @property
    def depth_offset(self) -> int:
        """Return the first byte of the z16 payload."""
        return self.colour_offset + self.height * self.width * 3

    @property
    def intrinsics_offset(self) -> int:
        """Return the first byte of the float32 camera matrix."""
        return self.depth_offset + self.height * self.width * np.dtype(np.uint16).itemsize

    @property
    def nbytes(self) -> int:
        """Return the total shared-memory allocation size."""
        return self.intrinsics_offset + 9 * np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class _FrameSlotSpec:
    """Pickle-safe identity and dimensions for one shared-memory frame slot."""

    name: str
    width: int
    height: int

    @property
    def layout(self) -> _FrameLayout:
        """Return the derived byte layout."""
        return _FrameLayout(self.width, self.height)


@dataclass(frozen=True)
class _FrameSnapshot:
    """One copied, coherent shared-memory frame publication."""

    colour: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.uint16]
    intrinsics: npt.NDArray[np.float32]
    depth_scale: float
    published_s: float
    generation: int


def _create_frame_slot(
    width: int = REALSENSE_CAPTURE_WIDTH,
    height: int = REALSENSE_CAPTURE_HEIGHT,
) -> tuple[shared_memory.SharedMemory, _FrameSlotSpec]:
    """Create a zeroed, stdlib-named shared-memory frame slot."""
    layout = _FrameLayout(width, height)
    shm = shared_memory.SharedMemory(create=True, size=layout.nbytes)
    buffer = _buffer(shm)
    buffer[:] = b"\0" * layout.nbytes
    del buffer
    return shm, _FrameSlotSpec(shm.name, width, height)


def _attach_frame_slot(spec: _FrameSlotSpec) -> shared_memory.SharedMemory:
    """Attach without letting the child resource tracker own the parent's name."""
    parameters = inspect.signature(shared_memory.SharedMemory).parameters
    if "track" in parameters:
        kwargs: dict[str, Any] = {"name": spec.name, "track": False}
        return shared_memory.SharedMemory(**kwargs)
    shm = shared_memory.SharedMemory(name=spec.name)
    resource_tracker.unregister(vars(shm)["_name"], "shared_memory")
    return shm


def _write_frame(
    shm: shared_memory.SharedMemory,
    spec: _FrameSlotSpec,
    *,
    colour: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.uint16],
    intrinsics: npt.NDArray[np.float32],
    depth_scale: float,
    published_s: float,
    generation: int,
) -> None:
    """Publish one frame, leaving an odd sequence behind if copying fails.

    ``published_s`` is stamped from ``time.monotonic()`` by the child. That
    clock is boot-relative and machine-wide on Linux and macOS, which lets the
    parent compare it with its own monotonic clock.
    """
    layout = spec.layout
    buffer = _buffer(shm)
    sequence = _read_sequence(buffer)
    odd_sequence = sequence + 1 if sequence % 2 == 0 else sequence + 2
    _HEADER.pack_into(
        buffer,
        0,
        odd_sequence,
        published_s,
        generation,
        depth_scale,
    )
    colour_view: npt.NDArray[np.uint8] = np.ndarray(
        (layout.height, layout.width, 3),
        dtype=np.uint8,
        buffer=buffer,
        offset=layout.colour_offset,
    )
    colour_view[...] = colour
    del colour_view
    depth_view: npt.NDArray[np.uint16] = np.ndarray(
        (layout.height, layout.width),
        dtype=np.uint16,
        buffer=buffer,
        offset=layout.depth_offset,
    )
    depth_view[...] = depth
    del depth_view
    intrinsics_view: npt.NDArray[np.float32] = np.ndarray(
        (3, 3),
        dtype=np.float32,
        buffer=buffer,
        offset=layout.intrinsics_offset,
    )
    intrinsics_view[...] = intrinsics
    del intrinsics_view
    _SEQUENCE.pack_into(buffer, 0, odd_sequence + 1)
    del buffer


def _read_frame(
    shm: shared_memory.SharedMemory,
    spec: _FrameSlotSpec,
    *,
    retries: int = SEQLOCK_READ_RETRIES,
) -> _FrameSnapshot | None:
    """Copy a coherent publication, or return ``None`` after bounded retries."""
    buffer = _buffer(shm)
    for _ in range(retries):
        sequence, published_s, generation, depth_scale = _HEADER.unpack_from(buffer)
        if sequence == 0 or sequence % 2:
            continue
        colour, depth, intrinsics = _copy_payload(buffer, spec.layout)
        if _read_sequence(buffer) == sequence:
            del buffer
            return _FrameSnapshot(
                colour=colour,
                depth=depth,
                intrinsics=intrinsics,
                depth_scale=depth_scale,
                published_s=published_s,
                generation=generation,
            )
    del buffer
    return None


def _copy_payload(
    buffer: Any, layout: _FrameLayout
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint16],
    npt.NDArray[np.float32],
]:
    """Copy payload arrays and release every exported buffer view before return."""
    colour_view: npt.NDArray[np.uint8] = np.ndarray(
        (layout.height, layout.width, 3),
        dtype=np.uint8,
        buffer=buffer,
        offset=layout.colour_offset,
    )
    colour = colour_view.copy()
    del colour_view
    depth_view: npt.NDArray[np.uint16] = np.ndarray(
        (layout.height, layout.width),
        dtype=np.uint16,
        buffer=buffer,
        offset=layout.depth_offset,
    )
    depth = depth_view.copy()
    del depth_view
    intrinsics_view: npt.NDArray[np.float32] = np.ndarray(
        (3, 3),
        dtype=np.float32,
        buffer=buffer,
        offset=layout.intrinsics_offset,
    )
    intrinsics = intrinsics_view.copy()
    del intrinsics_view
    return colour, depth, intrinsics


def _read_sequence(buffer: Any) -> int:
    """Read the seqlock counter without retaining a view over shared memory."""
    return int(_SEQUENCE.unpack_from(buffer)[0])


def _buffer(shm: shared_memory.SharedMemory) -> memoryview:
    """Return the live shared-memory buffer with its optional stub narrowed."""
    buffer = shm.buf
    assert buffer is not None
    return buffer
