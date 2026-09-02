"""Pack stored robot frames into H.264 evidence video and lossless FFV1 raw archives.

All encoded artifacts are staged under ``--scratch-dir`` (``/tmp`` by default), so the frame
directory is not written until upload verification and raw-frame deletion are complete. An
unbacked ``--no-upload`` run without ``--allow-unbacked-delete`` is refused and its scratch
artifacts are discarded rather than copied onto a potentially full disk.
"""

import argparse
import contextlib
import fcntl
import hashlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import IO, Any, cast

import numpy as np

TOOL_NAME = "pack_frames"
TOOL_VERSION = "2"
DEFAULT_FPS = 10.0
DEFAULT_TEAM_DRIVE = "0AGNB3pVRo9vkUk9PVA"
CAMERAS = ("left", "top", "right")
FRAME_RE = re.compile(r"^scene-0-e0_(left|top|right)_cam_(\d{6,})\.npy$")
STDERR_TAIL_LINES = 20
RCLONE_RETRY_ARGS = (
    "--contimeout",
    "60s",
    "--timeout",
    "300s",
    "--retries",
    "3",
    "--low-level-retries",
    "10",
)

RunCallable = Callable[..., Any]
PopenCallable = Callable[..., Any]
ClockCallable = Callable[[], float]
SleepCallable = Callable[[float], None]


class PackError(Exception):
    """Report a safe per-run packing failure."""


@dataclass(frozen=True)
class RunInfo:
    """Metadata resolved from one evaluation log."""

    log_path: Path
    run_name: str
    frames_dir: Path | None
    stamp: str
    control_hz: float
    status: str | None
    log_mtime: float
    load_error: str | None = None
    started_at: datetime | None = None
    policy: str | None = None


@dataclass(frozen=True)
class StreamInfo:
    """Ordered files and timeline metadata for one camera stream."""

    camera: str
    frames: tuple[tuple[int, Path], ...]
    first_step: int
    last_step: int
    gaps: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PackOptions:
    """Validated command-line settings used by the packing pipeline."""

    rig: Path
    min_height: int
    verify: bool
    dry_run: bool
    keep: bool
    force: bool
    grace: float
    no_upload: bool
    allow_unbacked_delete: bool
    remote: str
    host_label: str
    ffmpeg: str
    ffprobe: str
    rclone: str
    rclone_extra: tuple[str, ...]
    threads: int
    crf: int
    preset: str
    psnr_min: float
    sample_every: int
    scratch_dir: Path
    raw: str
    keep_raw_local: bool
    since: datetime | None
    policies: tuple[str, ...]


def _tool_default(name: str) -> str:
    """Prefer a user-local tool binary when it exists, otherwise use PATH."""
    local = Path.home() / ".local" / "bin" / name
    return str(local) if local.exists() else name


def _default_fps(data: dict[str, Any]) -> float:
    """Return a finite positive control rate from supported log layouts.

    EvalLog JSON nests the embodiment under ``eval.embodiment_info``; older or
    flattened layouts carry ``embodiment_info``/``embodiment``/``control_hz``
    at the top level. The first rate found wins; anything unusable falls back
    to the 10 Hz default that inspect-robots itself uses.
    """
    rate: Any = None
    scopes = [data.get("eval"), data]
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        embodiment = scope.get("embodiment_info")
        if not isinstance(embodiment, dict):
            embodiment = scope.get("embodiment")
        if isinstance(embodiment, dict):
            rate = embodiment.get("control_hz")
        if rate is None:
            rate = scope.get("control_hz")
        if rate is not None:
            break
    if (
        isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and rate > 0
        and math.isfinite(rate)
    ):
        return float(rate)
    return DEFAULT_FPS


def _stored_stamp(frames_dir: str, run_name: str) -> str:
    """Extract a portable basename from a stored frame-directory string."""
    if not frames_dir:
        return run_name
    return PureWindowsPath(frames_dir).name if "\\" in frames_dir else Path(frames_dir).name


def _frames_dir_candidates(frames_dir: str, rig: Path, log_path: Path) -> tuple[Path, Path]:
    """Return the stored frame path and the log-relative fallback in resolution order."""
    stored = Path(frames_dir)
    first = stored if stored.is_absolute() else rig / stored
    stamp = _stored_stamp(frames_dir, log_path.stem)
    return first, log_path.parent / "frames" / stamp


def _parse_iso8601(value: Any) -> datetime | None:
    """Parse an ISO-8601 instant, accepting ``Z`` and making naive values local-aware."""
    if not isinstance(value, str) or not value:
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.astimezone()
    return parsed


def _since_argument(value: str) -> datetime:
    """Parse a command-line ``--since`` value or raise an argparse usage error."""
    parsed = _parse_iso8601(value)
    if parsed is None:
        raise argparse.ArgumentTypeError("must be a valid ISO-8601 timestamp")
    return parsed


def load_run(log_path: Path, rig: Path) -> RunInfo:
    """Load one run log and resolve its frame directory without raising on bad JSON."""
    log_path = log_path.resolve()
    run_name = log_path.stem
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    try:
        with log_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON value is not an object")
    except FileNotFoundError:
        return RunInfo(
            log_path,
            run_name,
            None,
            run_name,
            DEFAULT_FPS,
            None,
            mtime,
            f"JSON missing: {log_path}",
        )
    except (OSError, ValueError) as exc:
        return RunInfo(
            log_path,
            run_name,
            None,
            run_name,
            DEFAULT_FPS,
            None,
            mtime,
            f"JSON unreadable: {exc}",
        )

    stats = data.get("stats")
    eval_data = data.get("eval")
    started_at = _parse_iso8601(stats.get("started_at")) if isinstance(stats, dict) else None
    if started_at is None and isinstance(eval_data, dict):
        started_at = _parse_iso8601(eval_data.get("created"))
    policy_value = eval_data.get("policy") if isinstance(eval_data, dict) else None
    policy = policy_value if isinstance(policy_value, str) else None
    stored = stats.get("frames_dir") if isinstance(stats, dict) else None
    if not isinstance(stored, str) or not stored:
        return RunInfo(
            log_path,
            run_name,
            None,
            run_name,
            _default_fps(data),
            str(data.get("status")) if data.get("status") is not None else None,
            mtime,
            "log has no stats.frames_dir",
            started_at,
            policy,
        )
    candidates = _frames_dir_candidates(stored, rig.resolve(), log_path)
    frames_dir = next((candidate for candidate in candidates if candidate.is_dir()), candidates[1])
    status = str(data.get("status")) if data.get("status") is not None else None
    return RunInfo(
        log_path,
        run_name,
        frames_dir,
        _stored_stamp(stored, run_name),
        _default_fps(data),
        status,
        mtime,
        started_at=started_at,
        policy=policy,
    )


def _sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pack_manifest(frames_dir: Path) -> dict[str, Any] | None:
    """Load a frame directory's pack manifest when it is a JSON object."""
    manifest_path = frames_dir / "pack_manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _manifest_outputs_match(
    frames_dir: Path,
    manifest: dict[str, Any],
    *,
    verify: bool,
) -> bool:
    """Check local MP4 metadata, hashing only changed metadata unless forced."""
    streams = manifest.get("streams")
    if not isinstance(streams, dict) or not streams:
        return False
    for stream in streams.values():
        if not isinstance(stream, dict):
            return False
        name = stream.get("file")
        expected = stream.get("sha256")
        if not isinstance(name, str) or not isinstance(expected, str):
            return False
        path = frames_dir / name
        try:
            if not path.is_file():
                return False
            stat = path.stat()
            expected_bytes = stream.get("bytes")
            expected_mtime = stream.get("mtime")
            metadata_matches = (
                isinstance(expected_bytes, int)
                and not isinstance(expected_bytes, bool)
                and isinstance(expected_mtime, (int, float))
                and not isinstance(expected_mtime, bool)
                and stat.st_size == expected_bytes
                and stat.st_mtime == float(expected_mtime)
            )
            if (verify or not metadata_matches) and _sha256(path) != expected:
                return False
        except OSError:
            return False
    return True


def _manifest_is_packed(
    frames_dir: Path,
    *,
    verify: bool = False,
    logger: logging.Logger | None = None,
) -> bool:
    """Treat final manifests as authoritative even when local media are missing or changed.

    Once state is ``packed`` or ``packed-kept``, raw frames may already be partly or wholly gone,
    so an output mismatch is warned about but never triggers re-encoding. Metadata/hash mismatch
    permits re-encoding only for pre-deletion ``encoded``/``uploaded`` manifests while their full
    NumPy set is still present; :func:`check_eligible` enforces that second condition.
    """
    manifest = _load_pack_manifest(frames_dir)
    if manifest is None or manifest.get("state") not in {"packed", "packed-kept"}:
        return False
    if not _manifest_outputs_match(frames_dir, manifest, verify=verify):
        target = logger if logger is not None else logging.getLogger(TOOL_NAME)
        target.warning(
            "packed manifest in %s does not match local MP4s; refusing unsafe re-encode",
            frames_dir,
        )
    return True


def _npy_set_matches_manifest(frames_dir: Path, manifest: dict[str, Any]) -> bool:
    """Return whether every stream timeline recorded before deletion is still present."""
    recorded = manifest.get("streams")
    if not isinstance(recorded, dict) or not recorded:
        return False
    try:
        current = discover_streams(frames_dir)
    except PackError:
        return False
    if set(current) != set(recorded):
        return False
    for camera, stream in current.items():
        expected = recorded.get(camera)
        if not isinstance(expected, dict):
            return False
        if (
            len(stream.frames) != expected.get("frames")
            or stream.first_step != expected.get("first_step")
            or stream.last_step != expected.get("last_step")
            or [list(gap) for gap in stream.gaps] != expected.get("gaps")
        ):
            return False
    return True


def _npy_shape(path: Path) -> tuple[int, ...]:
    """Read only a NumPy file header and return its declared shape."""
    try:
        with path.open("rb") as handle:
            major, _minor = np.lib.format.read_magic(handle)
            if major == 1:
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(handle)
    except (OSError, ValueError, EOFError) as exc:
        raise PackError(f"unreadable frame header {path.name}: {exc}") from exc
    return tuple(int(value) for value in shape)


def check_eligible(
    info: RunInfo,
    min_height: int = 360,
    *,
    verify: bool = False,
    logger: logging.Logger | None = None,
    since: datetime | None = None,
    policies: Sequence[str] = (),
) -> str | None:
    """Return a skip reason while preventing re-encode after deletion has begun."""
    if info.load_error is not None:
        return info.load_error
    if since is not None:
        if info.started_at is None:
            return "no start timestamp for --since"
        if info.started_at < since:
            started = info.started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")
            return f"started {started} before --since"
    if policies and info.policy not in policies:
        policy = info.policy if info.policy is not None else "<none>"
        return f"policy {policy} not in ({', '.join(policies)})"
    if info.status == "started":
        return "status is started"
    live_path = info.log_path.with_name(f"{info.run_name}.live.json")
    if live_path.exists():
        return f"live log exists: {live_path.name}"
    frames_dir = info.frames_dir
    if frames_dir is None or not frames_dir.is_dir():
        return "frames directory missing"
    if _manifest_is_packed(frames_dir, verify=verify, logger=logger):
        return "already packed"
    frame_paths = sorted(frames_dir.glob("*.npy"))
    if not frame_paths:
        return "frames directory has no .npy files"
    manifest = _load_pack_manifest(frames_dir)
    if (
        manifest is not None
        and manifest.get("state") in {"encoded", "uploaded"}
        and not _npy_set_matches_manifest(frames_dir, manifest)
    ):
        return "partial .npy set after interrupted packing; refusing re-encode"
    try:
        shape = _npy_shape(frame_paths[0])
    except PackError as exc:
        return str(exc)
    if not shape:
        return f"frame has invalid shape {shape}"
    height = shape[0]
    if height < min_height:
        return f"frame height {height} < {min_height}"
    return None


def _step_gaps(steps: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return inclusive missing-step ranges in an ordered step sequence."""
    gaps: list[tuple[int, int]] = []
    for previous, current in itertools.pairwise(steps):
        if current > previous + 1:
            gaps.append((previous + 1, current - 1))
    return tuple(gaps)


def discover_streams(
    frames_dir: Path,
    paths: Iterable[Path] | None = None,
) -> dict[str, StreamInfo]:
    """Validate and group frame files into camera streams ordered by numeric step."""
    candidates = list(frames_dir.glob("*.npy")) if paths is None else list(paths)
    grouped: dict[str, list[tuple[int, Path]]] = {}
    strays: list[str] = []
    for path in candidates:
        match = FRAME_RE.fullmatch(path.name)
        if match is None:
            strays.append(path.name)
            continue
        grouped.setdefault(match.group(1), []).append((int(match.group(2)), path))
    if strays:
        raise PackError(f"stray .npy files: {', '.join(sorted(strays))}")
    if not grouped:
        raise PackError("no camera frame streams found")

    streams: dict[str, StreamInfo] = {}
    for camera in CAMERAS:
        entries = grouped.get(camera)
        if entries is None:
            continue
        ordered = sorted(entries, key=lambda item: (item[0], item[1].name))
        steps = [step for step, _path in ordered]
        for previous, current in itertools.pairwise(steps):
            if current <= previous:
                raise PackError(f"{camera} frame steps are duplicate or non-increasing")
        streams[camera] = StreamInfo(
            camera=camera,
            frames=tuple(ordered),
            first_step=steps[0],
            last_step=steps[-1],
            gaps=_step_gaps(steps),
        )
    return streams


def _stream_bytes(streams: dict[str, StreamInfo]) -> int:
    """Return the total size of all enumerated NumPy frame files."""
    total = 0
    for stream in streams.values():
        for _step, path in stream.frames:
            try:
                total += path.stat().st_size
            except OSError as exc:
                raise PackError(f"cannot stat frame {path.name}: {exc}") from exc
    return total


def _ffmpeg_argv(
    options: PackOptions,
    width: int,
    height: int,
    fps: float,
    output: Path,
) -> list[str]:
    """Build the pinned ffmpeg command for a raw RGB camera stream."""
    return [
        options.ffmpeg,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(options.threads),
        "-fps_mode",
        "passthrough",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output),
    ]


def _raw_ffmpeg_argv(
    options: PackOptions,
    width: int,
    height: int,
    fps: float,
    output: Path,
) -> list[str]:
    """Build the pinned lossless FFV1 command for one exact-dimension RGB stream."""
    return [
        options.ffmpeg,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "-",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gbrp",
        "-g",
        "1",
        "-slices",
        "4",
        "-slicecrc",
        "1",
        "-threads",
        str(options.threads),
        "-fps_mode",
        "passthrough",
        "-f",
        "matroska",
        str(output),
    ]


def _stderr_tail(fd: int, name: str) -> str:
    """Close and read the tail of a temporary subprocess stderr file."""
    os.close(fd)
    try:
        with open(name, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().strip().splitlines()
    finally:
        Path(name).unlink(missing_ok=True)
    return "\n".join(lines[-STDERR_TAIL_LINES:])


def _load_frame(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Memory-map and validate one RGB uint8 frame."""
    try:
        frame = np.load(path, mmap_mode="r")
    except (OSError, ValueError, EOFError) as exc:
        raise PackError(f"unreadable frame {path.name}: {exc}") from exc
    if frame.dtype != np.uint8:
        raise PackError(f"unsupported dtype {frame.dtype} in {path.name}; expected uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise PackError(f"unsupported shape {frame.shape} in {path.name}; expected (H, W, 3)")
    return frame


def encode_stream(
    stream: StreamInfo,
    output: Path,
    fps: float,
    options: PackOptions,
    *,
    popen: PopenCallable = subprocess.Popen,
) -> tuple[list[str], int, int]:
    """Pipe one validated camera stream to ffmpeg and return argv and source dimensions."""
    first = _load_frame(stream.frames[0][1])
    height, width = int(first.shape[0]), int(first.shape[1])
    expected_shape = first.shape
    argv = _ffmpeg_argv(options, width, height, fps, output)
    stderr_fd, stderr_name = tempfile.mkstemp(suffix=".ffmpeg.log")
    try:
        process = popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fd,
        )
    except OSError as exc:
        os.close(stderr_fd)
        Path(stderr_name).unlink(missing_ok=True)
        raise PackError(f"could not launch ffmpeg ({options.ffmpeg}): {exc}") from exc

    stream_error: str | None = None
    broken_pipe = False
    stdin = cast("IO[bytes]", process.stdin)
    try:
        try:
            for _step, path in stream.frames:
                frame = _load_frame(path)
                if frame.shape != expected_shape:
                    raise PackError(
                        f"frame shape changed from {expected_shape} to {frame.shape} at {path.name}"
                    )
                stdin.write(np.ascontiguousarray(frame).tobytes())
        except PackError as exc:
            stream_error = str(exc)
            process.kill()
        except (BrokenPipeError, OSError):
            broken_pipe = True
        try:
            stdin.close()
        except (BrokenPipeError, OSError):
            broken_pipe = True
        returncode = int(process.wait())
    finally:
        tail = _stderr_tail(stderr_fd, stderr_name)
    if stream_error is None and (broken_pipe or returncode != 0):
        stream_error = tail or f"ffmpeg exited with code {returncode}"
    if stream_error is not None:
        output.unlink(missing_ok=True)
        raise PackError(f"{stream.camera} encode failed: {stream_error}")
    return argv, width, height


def encode_raw_stream(
    stream: StreamInfo,
    output: Path,
    fps: float,
    options: PackOptions,
    *,
    popen: PopenCallable = subprocess.Popen,
) -> tuple[list[str], int, int]:
    """Pipe one camera stream to lossless FFV1 and return argv and exact dimensions."""
    first = _load_frame(stream.frames[0][1])
    height, width = int(first.shape[0]), int(first.shape[1])
    expected_shape = first.shape
    argv = _raw_ffmpeg_argv(options, width, height, fps, output)
    stderr_fd, stderr_name = tempfile.mkstemp(suffix=".ffmpeg.log")
    try:
        process = popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fd,
        )
    except OSError as exc:
        os.close(stderr_fd)
        Path(stderr_name).unlink(missing_ok=True)
        raise PackError(f"could not launch ffmpeg ({options.ffmpeg}): {exc}") from exc

    stream_error: str | None = None
    broken_pipe = False
    stdin = cast("IO[bytes]", process.stdin)
    try:
        try:
            for _step, path in stream.frames:
                frame = _load_frame(path)
                if frame.shape != expected_shape:
                    raise PackError(
                        f"frame shape changed from {expected_shape} to {frame.shape} at {path.name}"
                    )
                stdin.write(np.ascontiguousarray(frame).tobytes())
        except PackError as exc:
            stream_error = str(exc)
            process.kill()
        except (BrokenPipeError, OSError):
            broken_pipe = True
        try:
            stdin.close()
        except (BrokenPipeError, OSError):
            broken_pipe = True
        returncode = int(process.wait())
    finally:
        tail = _stderr_tail(stderr_fd, stderr_name)
    if stream_error is None and (broken_pipe or returncode != 0):
        stream_error = tail or f"ffmpeg exited with code {returncode}"
    if stream_error is not None:
        output.unlink(missing_ok=True)
        raise PackError(f"{stream.camera} FFV1 encode failed: {stream_error}")
    return argv, width, height


def _completed_text(value: Any) -> str:
    """Convert a subprocess output field to readable text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def probe_frame_count(
    mp4: Path,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    ffprobe: str,
    *,
    require_nb_frames: bool = True,
    run: RunCallable = subprocess.run,
) -> None:
    """Require ffprobe packet, optional frame, and dimension counts to match the input."""
    argv = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_frames,nb_read_packets,width,height",
        "-of",
        "json",
        str(mp4),
    ]
    try:
        completed = run(argv, capture_output=True, check=False)
    except OSError as exc:
        raise PackError(f"could not launch ffprobe ({ffprobe}): {exc}") from exc
    if completed.returncode != 0:
        error = _completed_text(completed.stderr).strip()
        raise PackError(f"ffprobe failed for {mp4.name}: {error or completed.returncode}")
    try:
        payload = json.loads(_completed_text(completed.stdout))
        stream = payload["streams"][0]
        packets = int(stream["nb_read_packets"])
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PackError(f"invalid ffprobe output for {mp4.name}: {exc}") from exc
    raw_frames = stream.get("nb_frames")
    frames: int | None = None
    if raw_frames not in {None, "N/A"}:
        try:
            frames = int(raw_frames)
        except (TypeError, ValueError) as exc:
            raise PackError(f"invalid ffprobe frame count for {mp4.name}: {raw_frames}") from exc
    if packets != expected_frames or (require_nb_frames and frames != expected_frames):
        raise PackError(
            f"{mp4.name}: expected {expected_frames} frames, ffprobe reported "
            f"{frames} frames and {packets} packets"
        )
    if (width, height) != (expected_width, expected_height):
        raise PackError(
            f"{mp4.name}: expected {expected_width}x{expected_height}, got {width}x{height}"
        )


def verify_psnr(
    mp4: Path,
    stream: StreamInfo,
    width: int,
    height: int,
    ffmpeg: str,
    psnr_min: float,
    sample_every: int,
    *,
    run: RunCallable = subprocess.run,
) -> list[list[float | int]]:
    """Decode selected output frames and require each cropped frame to meet a PSNR floor."""
    count = len(stream.frames)
    indices = sorted({0, *range(0, count, sample_every), count - 1})
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4),
        "-vf",
        f"select='{expression}'",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        completed = run(argv, capture_output=True, check=False)
    except OSError as exc:
        raise PackError(f"could not launch ffmpeg ({ffmpeg}): {exc}") from exc
    if completed.returncode != 0:
        error = _completed_text(completed.stderr).strip()
        raise PackError(f"PSNR decode failed for {mp4.name}: {error or completed.returncode}")
    decoded = completed.stdout
    if not isinstance(decoded, bytes):
        raise PackError(f"PSNR decode for {mp4.name} did not return bytes")
    padded_width = width + width % 2
    padded_height = height + height % 2
    expected_bytes = len(indices) * padded_height * padded_width * 3
    if len(decoded) != expected_bytes:
        raise PackError(
            f"PSNR decode for {mp4.name} returned {len(decoded)} bytes; expected {expected_bytes}"
        )
    decoded_frames = np.frombuffer(decoded, dtype=np.uint8).reshape(
        len(indices),
        padded_height,
        padded_width,
        3,
    )
    samples: list[list[float | int]] = []
    for decoded_frame, index in zip(decoded_frames, indices, strict=True):
        step, source_path = stream.frames[index]
        source = _load_frame(source_path)
        difference = decoded_frame[:height, :width].astype(np.float64) - source.astype(np.float64)
        mse = float(np.mean(difference * difference))
        psnr = math.inf if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse)
        if psnr < psnr_min:
            raise PackError(f"{mp4.name}: step {step} PSNR {psnr:.2f} dB is below {psnr_min:g} dB")
        samples.append([step, psnr])
    return samples


def _read_exact(handle: IO[bytes], size: int) -> bytes:
    """Read up to exactly ``size`` bytes, retrying when a pipe returns a short chunk."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def verify_raw_bit_exact(
    raw_path: Path,
    stream: StreamInfo,
    width: int,
    height: int,
    ffmpeg: str,
    *,
    popen: PopenCallable = subprocess.Popen,
) -> None:
    """Stream-decode a complete FFV1 archive and compare every RGB byte with its NumPy frame."""
    argv = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(raw_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    stderr_fd, stderr_name = tempfile.mkstemp(suffix=".ffmpeg.log")
    try:
        process = popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_fd,
        )
    except OSError as exc:
        os.close(stderr_fd)
        Path(stderr_name).unlink(missing_ok=True)
        raise PackError(f"could not launch ffmpeg ({ffmpeg}): {exc}") from exc

    frame_bytes = height * width * 3
    decoded = cast("IO[bytes]", process.stdout)
    error: str | None = None
    try:
        try:
            for step, source_path in stream.frames:
                actual = _read_exact(decoded, frame_bytes)
                if len(actual) != frame_bytes:
                    error = (
                        f"{raw_path.name}: short decoded frame at step {step}: "
                        f"{len(actual)} of {frame_bytes} bytes"
                    )
                    break
                expected = np.ascontiguousarray(_load_frame(source_path)).tobytes()
                if actual != expected:
                    error = f"{raw_path.name}: decoded bytes differ at step {step}"
                    break
            if error is None and decoded.read(1):
                error = f"{raw_path.name}: decoded output contains extra frame data"
        except (OSError, PackError) as exc:
            error = str(exc)
        if error is not None:
            process.kill()
        decoded.close()
        returncode = int(process.wait())
    finally:
        tail = _stderr_tail(stderr_fd, stderr_name)
    if error is not None:
        raise PackError(error)
    if returncode != 0:
        raise PackError(
            f"{raw_path.name}: lossless verification decode failed: "
            f"{tail or f'ffmpeg exited with code {returncode}'}"
        )


def _iso_now() -> str:
    """Return a UTC ISO-8601 timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically replace a pack manifest in its frame directory."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PackError(f"cannot write manifest {path}: {exc}") from exc


def _ffmpeg_version(ffmpeg: str, run: RunCallable) -> str:
    """Return the encoder's first version line, or ``unknown`` if it cannot be queried."""
    try:
        completed = run([ffmpeg, "-version"], capture_output=True, check=False)
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    lines = _completed_text(completed.stdout).splitlines()
    return lines[0] if lines else "unknown"


def _remote_path(options: PackOptions, info: RunInfo) -> str:
    """Build the host/rig/stamp destination directory for one run."""
    return f"{options.remote.rstrip('/')}/{options.host_label}/{options.rig.name}/{info.stamp}/"


def _rclone_media_includes(*, include_raw: bool) -> list[str]:
    """Return rclone include arguments for immutable packed media outputs."""
    includes: list[str] = []
    for camera in CAMERAS:
        includes.extend(["--include", f"scene-0-e0_{camera}_cam.mp4"])
    if include_raw:
        includes.extend(["--include", "scene-0-e0_*_cam.ffv1.mkv"])
    return includes


def _rclone_copy_includes(*, include_raw: bool) -> list[str]:
    """Return rclone include arguments for camera outputs plus the initial manifest."""
    includes = _rclone_media_includes(include_raw=include_raw)
    includes.extend(["--include", "pack_manifest.json"])
    return includes


def upload(
    frames_dir: Path,
    remote_path: str,
    options: PackOptions,
    *,
    run: RunCallable = subprocess.run,
) -> None:
    """Copy staged artifacts and verify all immutable media at the destination."""
    include_raw = options.raw == "ffv1"
    copy_includes = _rclone_copy_includes(include_raw=include_raw)
    check_includes = _rclone_media_includes(include_raw=include_raw)
    copy_argv = [
        options.rclone,
        "copy",
        *RCLONE_RETRY_ARGS,
        "--checksum",
        "--transfers",
        "4",
        "--drive-chunk-size",
        "64M",
        *options.rclone_extra,
        *copy_includes,
        str(frames_dir),
        remote_path,
    ]
    check_argv = [
        options.rclone,
        "check",
        *RCLONE_RETRY_ARGS,
        "--one-way",
        *options.rclone_extra,
        *check_includes,
        str(frames_dir),
        remote_path,
    ]
    calls = (("copy", copy_argv, 6 * 60 * 60), ("check", check_argv, 30 * 60))
    for action, argv, timeout in calls:
        try:
            completed = run(argv, capture_output=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise PackError(f"rclone {action} timed out after {timeout}s") from exc
        except OSError as exc:
            raise PackError(f"could not launch rclone ({options.rclone}): {exc}") from exc
        if completed.returncode != 0:
            error = _completed_text(completed.stderr).strip()
            raise PackError(f"rclone {action} failed: {error or completed.returncode}")


def upload_final_manifest(
    manifest_path: Path,
    remote_path: str,
    options: PackOptions,
    *,
    run: RunCallable = subprocess.run,
) -> None:
    """Upload the final-state manifest, raising a pack error for callers to downgrade."""
    argv = [
        options.rclone,
        "copyto",
        *RCLONE_RETRY_ARGS,
        *options.rclone_extra,
        str(manifest_path),
        f"{remote_path.rstrip('/')}/pack_manifest.json",
    ]
    try:
        completed = run(argv, capture_output=True, check=False, timeout=10 * 60)
    except subprocess.TimeoutExpired as exc:
        raise PackError("rclone copyto timed out after 600s") from exc
    except OSError as exc:
        raise PackError(f"could not launch rclone ({options.rclone}): {exc}") from exc
    if completed.returncode != 0:
        error = _completed_text(completed.stderr).strip()
        raise PackError(f"rclone copyto failed: {error or completed.returncode}")


def _logger_for(path: Path) -> logging.Logger:
    """Create a timestamped logger, falling back to stderr when its file cannot open."""
    logger = logging.getLogger(f"{TOOL_NAME}.{os.getpid()}.{time.monotonic_ns()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(path)
    except OSError as exc:
        logger.warning("cannot open pack log %s: %s; continuing with stderr only", path, exc)
    else:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """Flush, close, and detach every handler from a per-run logger."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _manifest_base(
    info: RunInfo,
    options: PackOptions,
    stream_manifests: dict[str, dict[str, Any]],
    raw_manifests: dict[str, dict[str, Any]],
    argv_by_camera: dict[str, list[str]],
    remote_path: str,
    created: str,
    run: RunCallable,
) -> dict[str, Any]:
    """Build the encoded-state manifest common to later state transitions."""
    try:
        relative_log = str(info.log_path.relative_to(options.rig))
    except ValueError:
        relative_log = str(info.log_path)
    manifest: dict[str, Any] = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "host": options.host_label,
        "rig": options.rig.name,
        "run": info.run_name,
        "log": relative_log,
        "stamp": info.stamp,
        "control_hz": info.control_hz,
        "created": created,
        "updated": created,
        "ffmpeg": {
            "version": _ffmpeg_version(options.ffmpeg, run),
            "argv": argv_by_camera,
        },
        "streams": stream_manifests,
        "state": "encoded",
        "remote": remote_path,
        "npy_bytes_freed": 0,
        "npy_unlink_failures": 0,
    }
    if raw_manifests:
        manifest["raw"] = raw_manifests
        manifest["raw_verified"] = "bit-exact"
    return manifest


def _set_manifest_state(
    path: Path,
    manifest: dict[str, Any],
    state: str,
    *,
    npy_bytes_freed: int | None = None,
) -> None:
    """Update state, timestamp, and optional freed-byte count, then persist atomically."""
    manifest["state"] = state
    manifest["updated"] = _iso_now()
    if npy_bytes_freed is not None:
        manifest["npy_bytes_freed"] = npy_bytes_freed
    write_manifest(path, manifest)


def _cleanup_stale_temporary_outputs(directory: Path) -> None:
    """Remove legacy and ffmpeg-created temporary media files from a directory."""
    for stale in directory.glob("scene-0-e0_*_cam.mp4.tmp*"):
        stale.unlink(missing_ok=True)
    for stale in directory.glob("scene-0-e0_*_cam.ffv1.mkv.tmp*"):
        stale.unlink(missing_ok=True)


def _verify_staged_hashes(scratch: Path, manifest: dict[str, Any]) -> None:
    """Re-hash every staged media file and require agreement with its manifest digest."""
    streams = manifest.get("streams")
    if not isinstance(streams, dict) or not streams:
        raise PackError("staged manifest has no streams")
    groups = [("stream", streams)]
    raw = manifest.get("raw")
    if isinstance(raw, dict) and raw:
        groups.append(("raw stream", raw))
    for label, group in groups:
        for camera, stream in group.items():
            if not isinstance(stream, dict):
                raise PackError(f"staged manifest {label} {camera} is invalid")
            name = stream.get("file")
            expected = stream.get("sha256")
            if not isinstance(name, str) or not isinstance(expected, str):
                raise PackError(f"staged manifest {label} {camera} has no file hash")
            path = scratch / name
            try:
                actual = _sha256(path)
            except OSError as exc:
                raise PackError(f"cannot re-hash staged media {name}: {exc}") from exc
            if actual != expected:
                raise PackError(f"staged media hash mismatch: {name}")


def _manifest_file_names(path: Path) -> set[str]:
    """Return MP4 names recorded by an existing local manifest, if readable."""
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return set()
    streams = manifest.get("streams") if isinstance(manifest, dict) else None
    if not isinstance(streams, dict):
        return set()
    return {
        name
        for stream in streams.values()
        if isinstance(stream, dict) and isinstance((name := stream.get("file")), str)
    }


def _warn_foreign_outputs(
    frames_dir: Path,
    output_names: Iterable[str],
    logger: logging.Logger,
) -> None:
    """Warn when verified staged outputs will replace MP4s not owned by a manifest."""
    recorded = _manifest_file_names(frames_dir / "pack_manifest.json")
    for name in output_names:
        destination = frames_dir / name
        if destination.is_file() and name not in recorded:
            logger.warning(
                "foreign MP4 %s has no matching manifest; verified staged file will overwrite it",
                destination,
            )


def _move_staged_media(
    scratch: Path,
    frames_dir: Path,
    manifest: dict[str, Any],
    *,
    keep_raw_local: bool,
) -> Path:
    """Move verified media after the safety manifest exists locally and refresh metadata."""
    streams = cast("dict[str, dict[str, Any]]", manifest["streams"])
    groups = [streams]
    raw = manifest.get("raw")
    if keep_raw_local and isinstance(raw, dict):
        groups.append(raw)
    for group in groups:
        for stream in group.values():
            name = cast("str", stream["file"])
            shutil.move(str(scratch / name), str(frames_dir / name))
    destination_manifest = frames_dir / "pack_manifest.json"
    for group in groups:
        for stream in group.values():
            output = frames_dir / cast("str", stream["file"])
            stat = output.stat()
            stream["bytes"] = stat.st_size
            stream["mtime"] = stat.st_mtime
    write_manifest(destination_manifest, manifest)
    return destination_manifest


def _pack_locked(
    info: RunInfo,
    options: PackOptions,
    logger: logging.Logger,
    *,
    run: RunCallable,
    popen: PopenCallable,
    clock: ClockCallable,
    sleep: SleepCallable,
) -> int:
    """Run the complete pack state machine while the rig lock is held."""
    reason = check_eligible(
        info,
        options.min_height,
        verify=options.verify,
        logger=logger,
        since=options.since,
        policies=options.policies,
    )
    if reason is not None:
        logger.info("skipped: %s", reason)
        return 3
    frames_dir = cast("Path", info.frames_dir)
    try:
        streams = discover_streams(frames_dir)
        npy_bytes = _stream_bytes(streams)
    except PackError as exc:
        logger.error("failed: %s", exc)
        return 1
    logger.info(
        "eligible: %d streams, %d frames, %.3f GB",
        len(streams),
        sum(len(stream.frames) for stream in streams.values()),
        npy_bytes / 1_000_000_000,
    )
    if options.dry_run:
        final_action = "keep" if options.keep else "delete"
        logger.info(
            "dry-run: would encode H.264%s, verify, upload, and %s",
            " plus FFV1" if options.raw == "ffv1" else "",
            final_action,
        )
        return 0

    try:
        _cleanup_stale_temporary_outputs(frames_dir)
        scratch = Path(tempfile.mkdtemp(prefix="pack_frames-", dir=str(options.scratch_dir)))
    except OSError as exc:
        logger.error("failed to create scratch directory: %s", exc)
        return 1
    initial_files = {path for stream in streams.values() for _step, path in stream.frames}
    deleted = False
    completed = False
    local_manifest_written = False
    unlink_failures = 0
    freed_bytes = 0
    manifest: dict[str, Any] | None = None
    try:
        _cleanup_stale_temporary_outputs(scratch)
        final_outputs: dict[str, Path] = {}
        raw_outputs: dict[str, Path] = {}
        argv_by_camera: dict[str, list[str]] = {}
        source_dimensions: dict[str, tuple[int, int]] = {}
        psnr_by_camera: dict[str, list[list[float | int]]] = {}
        for camera, stream in streams.items():
            temporary = scratch / f"scene-0-e0_{camera}_cam.mp4.tmp"
            argv, width, height = encode_stream(
                stream,
                temporary,
                info.control_hz,
                options,
                popen=popen,
            )
            argv_by_camera[camera] = argv
            source_dimensions[camera] = (width, height)
            padded_width = width + width % 2
            padded_height = height + height % 2
            probe_frame_count(
                temporary,
                len(stream.frames),
                padded_width,
                padded_height,
                options.ffprobe,
                run=run,
            )
            psnr_by_camera[camera] = verify_psnr(
                temporary,
                stream,
                width,
                height,
                options.ffmpeg,
                options.psnr_min,
                options.sample_every,
                run=run,
            )
            final = scratch / f"scene-0-e0_{camera}_cam.mp4"
            os.replace(temporary, final)
            final_outputs[camera] = final
            if options.raw == "ffv1":
                raw_temporary = scratch / f"scene-0-e0_{camera}_cam.ffv1.mkv.tmp"
                _raw_argv, raw_width, raw_height = encode_raw_stream(
                    stream,
                    raw_temporary,
                    info.control_hz,
                    options,
                    popen=popen,
                )
                if (raw_width, raw_height) != (width, height):
                    raise PackError(f"{camera} raw encoder dimensions changed unexpectedly")
                probe_frame_count(
                    raw_temporary,
                    len(stream.frames),
                    width,
                    height,
                    options.ffprobe,
                    require_nb_frames=False,
                    run=run,
                )
                verify_raw_bit_exact(
                    raw_temporary,
                    stream,
                    width,
                    height,
                    options.ffmpeg,
                    popen=popen,
                )
                raw_final = scratch / f"scene-0-e0_{camera}_cam.ffv1.mkv"
                os.replace(raw_temporary, raw_final)
                raw_outputs[camera] = raw_final
        stream_manifests: dict[str, dict[str, Any]] = {}
        raw_manifests: dict[str, dict[str, Any]] = {}
        for camera, stream in streams.items():
            final = final_outputs[camera]
            width, height = source_dimensions[camera]
            stat = final.stat()
            stream_manifests[camera] = {
                "file": final.name,
                "sha256": _sha256(final),
                "bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "frames": len(stream.frames),
                "width": width,
                "height": height,
                "first_step": stream.first_step,
                "last_step": stream.last_step,
                "gaps": [list(gap) for gap in stream.gaps],
                "psnr_samples": psnr_by_camera[camera],
            }
            raw_final = raw_outputs.get(camera)
            if raw_final is not None:
                raw_stat = raw_final.stat()
                raw_manifests[camera] = {
                    "file": raw_final.name,
                    "codec": "ffv1",
                    "sha256": _sha256(raw_final),
                    "bytes": raw_stat.st_size,
                    "mtime": raw_stat.st_mtime,
                    "frames": len(stream.frames),
                    "width": width,
                    "height": height,
                }
        remote_path = _remote_path(options, info)
        created = _iso_now()
        manifest = _manifest_base(
            info,
            options,
            stream_manifests,
            raw_manifests,
            argv_by_camera,
            remote_path,
            created,
            run,
        )
        manifest_path = scratch / "pack_manifest.json"
        write_manifest(manifest_path, manifest)

        if not options.no_upload:
            try:
                upload(scratch, remote_path, options, run=run)
            except PackError as exc:
                logger.error("failed: %s; staged artifacts discarded and .npy files retained", exc)
                return 1
            _set_manifest_state(manifest_path, manifest, "uploaded")
            logger.info("uploaded and verified: %s", remote_path)
        elif not options.allow_unbacked_delete:
            logger.error(
                "refusing to delete .npy without a verified upload "
                "(pass --allow-unbacked-delete to override); staged artifacts discarded"
            )
            return 1

        if not options.force and not options.keep:
            while True:
                remaining = options.grace - (clock() - info.log_mtime)
                if remaining <= 0:
                    break
                wait = min(60.0, remaining)
                logger.info("grace: waiting %.0fs before deleting .npy", wait)
                sleep(wait)

        _verify_staged_hashes(scratch, manifest)
        output_names = [cast("str", stream["file"]) for stream in stream_manifests.values()]
        _warn_foreign_outputs(frames_dir, output_names, logger)

        if options.keep:
            _set_manifest_state(manifest_path, manifest, "packed-kept")
            write_manifest(frames_dir / "pack_manifest.json", manifest)
            local_manifest_written = True
        else:
            current_streams = discover_streams(frames_dir)
            current_files = {
                path for stream in current_streams.values() for _step, path in stream.frames
            }
            if current_files != initial_files:
                raise PackError("frame file set changed during packing; refusing deletion")
            deleted = True
            for path in sorted(initial_files):
                size = 0
                with contextlib.suppress(OSError):
                    size = path.stat().st_size
                try:
                    path.unlink()
                    freed_bytes += size
                except OSError as exc:
                    unlink_failures += 1
                    logger.error("failed to unlink %s: %s", path, exc)
            manifest["state"] = "packed"
            manifest["updated"] = _iso_now()
            manifest["npy_bytes_freed"] = freed_bytes
            manifest["npy_unlink_failures"] = unlink_failures
            write_manifest(frames_dir / "pack_manifest.json", manifest)
            local_manifest_written = True
            write_manifest(manifest_path, manifest)
            logger.info(
                "packed: deleted %d of %d .npy files, freed %d bytes, %d unlink failures",
                len(initial_files) - unlink_failures,
                len(initial_files),
                freed_bytes,
                unlink_failures,
            )

        final_manifest = _move_staged_media(
            scratch,
            frames_dir,
            manifest,
            keep_raw_local=options.keep_raw_local,
        )
        if not options.no_upload:
            try:
                upload_final_manifest(
                    final_manifest,
                    remote_path,
                    options,
                    run=run,
                )
            except PackError as exc:
                logger.warning("final manifest upload failed after backup: %s", exc)
        if options.keep:
            logger.info("packed-kept: retained %d .npy files", len(initial_files))
        completed = True
        return 0
    except Exception as exc:
        logger.error("failed: %s", exc)
        return 1
    finally:
        if deleted and not local_manifest_written and manifest is not None:
            manifest["state"] = "packed"
            manifest["updated"] = _iso_now()
            manifest["npy_bytes_freed"] = freed_bytes
            manifest["npy_unlink_failures"] = unlink_failures
            try:
                write_manifest(frames_dir / "pack_manifest.json", manifest)
            except PackError as manifest_exc:
                logger.error("failed to write post-deletion safety manifest: %s", manifest_exc)
        if deleted and not completed:
            logger.error("staged outputs preserved at %s", scratch)
        else:
            shutil.rmtree(scratch, ignore_errors=True)


def pack_one(
    log_path: Path,
    options: PackOptions,
    *,
    run: RunCallable = subprocess.run,
    popen: PopenCallable = subprocess.Popen,
    clock: ClockCallable = time.time,
    sleep: SleepCallable = time.sleep,
) -> int:
    """Pack one run while holding the blocking per-rig advisory lock."""
    log_path = log_path.resolve()
    pack_dir = log_path.parent / "pack"
    try:
        pack_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"pack_frames: cannot create {pack_dir}: {exc}", file=sys.stderr)
        return 1
    lock_path = pack_dir / ".lock"
    try:
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            info = load_run(log_path, options.rig)
            logger = _logger_for(pack_dir / f"{info.stamp}.log")
            try:
                return _pack_locked(
                    info,
                    options,
                    logger,
                    run=run,
                    popen=popen,
                    clock=clock,
                    sleep=sleep,
                )
            finally:
                _close_logger(logger)
    except OSError as exc:
        print(f"pack_frames: lock/log failure: {exc}", file=sys.stderr)
        return 1


def _resolve_run_argument(value: str, rig: Path) -> Path:
    """Resolve a bare run name or a rig-relative/absolute JSON log path."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.parent != Path("."):
        return rig / candidate
    name = candidate.name if candidate.suffix == ".json" else f"{candidate.name}.json"
    return rig / "logs" / name


def _all_logs(rig: Path) -> list[Path]:
    """List final run JSON files directly under the rig's log directory."""
    return [
        path
        for path in sorted((rig / "logs").glob("*.json"))
        if not path.name.endswith(".live.json")
    ]


def _run_all(
    options: PackOptions,
    limit: int | None,
    *,
    run: RunCallable,
    popen: PopenCallable,
    clock: ClockCallable,
    sleep: SleepCallable,
) -> int:
    """Pack all eligible runs smallest-first and continue after per-run failures."""
    skipped = 0
    candidates: list[tuple[int, Path]] = []
    for log_path in _all_logs(options.rig):
        info = load_run(log_path, options.rig)
        reason = check_eligible(
            info,
            options.min_height,
            verify=options.verify,
            since=options.since,
            policies=options.policies,
        )
        if reason is not None:
            skipped += 1
            print(f"skipped {log_path.name}: {reason}")
            continue
        try:
            streams = discover_streams(cast("Path", info.frames_dir))
            candidates.append((_stream_bytes(streams), log_path))
        except PackError as exc:
            skipped += 1
            print(f"skipped {log_path.name}: {exc}")
    candidates.sort(key=lambda item: (item[0], item[1].name))
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        print(f"nothing to pack (skipped={skipped})")
        return 0

    packed = 0
    failed = 0
    for _size, log_path in candidates:
        try:
            result = pack_one(
                log_path,
                options,
                run=run,
                popen=popen,
                clock=clock,
                sleep=sleep,
            )
        except Exception as exc:
            print(f"failed {log_path.name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        if result == 0:
            packed += 1
        elif result == 3:
            skipped += 1
        else:
            failed += 1
    print(f"summary: packed={packed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def _directory_npy_bytes(path: Path) -> tuple[int, int]:
    """Return NumPy byte and file counts for a frame directory."""
    total = 0
    count = 0
    for frame in path.glob("*.npy"):
        count += 1
        with contextlib.suppress(OSError):
            total += frame.stat().st_size
    return total, count


def _print_status(options: PackOptions) -> int:
    """Print packability, skip reasons, orphan directories, and remaining bytes."""
    packable_count = 0
    packable_bytes = 0
    packed_count = 0
    packed_with_raw = 0
    packed_without_raw = 0
    remaining_bytes = 0
    reasons: Counter[str] = Counter()
    referenced: set[Path] = set()
    for log_path in _all_logs(options.rig):
        info = load_run(log_path, options.rig)
        if info.frames_dir is not None:
            referenced.add(info.frames_dir.resolve())
        reason = check_eligible(
            info,
            options.min_height,
            verify=options.verify,
            since=options.since,
            policies=options.policies,
        )
        if reason == "already packed":
            packed_count += 1
            manifest = _load_pack_manifest(info.frames_dir) if info.frames_dir is not None else None
            if manifest is not None and manifest.get("raw_verified") == "bit-exact":
                packed_with_raw += 1
            else:
                packed_without_raw += 1
            continue
        if reason is not None:
            reasons[reason] += 1
            continue
        try:
            streams = discover_streams(cast("Path", info.frames_dir))
            size = _stream_bytes(streams)
        except PackError as exc:
            reasons[str(exc)] += 1
            continue
        packable_count += 1
        packable_bytes += size
        remaining_bytes += size

    live_dirs: set[Path] = set()
    for live_path in sorted((options.rig / "logs").glob("*.live.json")):
        live_info = load_run(live_path, options.rig)
        if live_info.frames_dir is not None and live_info.frames_dir.is_dir():
            resolved = live_info.frames_dir.resolve()
            referenced.add(resolved)
            live_dirs.add(resolved)
    live_bytes = sum(_directory_npy_bytes(directory)[0] for directory in live_dirs)

    orphan_count = 0
    orphan_bytes = 0
    orphan_empty = 0
    frames_root = options.rig / "logs" / "frames"
    if frames_root.is_dir():
        for directory in sorted(path for path in frames_root.iterdir() if path.is_dir()):
            if directory.resolve() in referenced:
                continue
            orphan_count += 1
            size, count = _directory_npy_bytes(directory)
            orphan_bytes += size
            if count == 0:
                orphan_empty += 1

    print("frame pack status")
    print(f"packable        {packable_count:6d}  {packable_bytes / 1_000_000_000:9.3f} GB")
    print(f"packed          {packed_count:6d}")
    print(f"packed raw      {packed_with_raw:6d}")
    print(f"packed no raw   {packed_without_raw:6d}")
    print(f"live dirs       {len(live_dirs):6d}  {live_bytes / 1_000_000_000:9.3f} GB")
    for reason, count in sorted(reasons.items()):
        print(f"skipped         {count:6d}  {reason}")
    print(
        f"orphan dirs     {orphan_count:6d}  {orphan_bytes / 1_000_000_000:9.3f} GB  "
        f"({orphan_empty} empty)"
    )
    print(f"GB remaining at >= {options.min_height}px: {remaining_bytes / 1_000_000_000:.3f}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the frame packer's command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", metavar="LOG")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--rig", type=Path, default=Path.cwd())
    parser.add_argument("--min-height", type=int, default=360)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--since", type=_since_argument)
    parser.add_argument("--policy", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--grace", type=float, default=600.0)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--allow-unbacked-delete", action="store_true")
    parser.add_argument("--remote", default="gdrive-rc:rig-video")
    parser.add_argument("--host-label", default=socket.gethostname().split(".")[0])
    parser.add_argument("--ffmpeg", default=_tool_default("ffmpeg"))
    parser.add_argument("--ffprobe", default=_tool_default("ffprobe"))
    parser.add_argument("--rclone", default=_tool_default("rclone"))
    parser.add_argument("--rclone-extra", action="append", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--psnr-min", type=float, default=35.0)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--raw", choices=("ffv1", "none"), default="ffv1")
    parser.add_argument("--keep-raw-local", action="store_true")
    return parser


def _options(namespace: argparse.Namespace, parser: argparse.ArgumentParser) -> PackOptions:
    """Validate parsed arguments and convert them to immutable pipeline options."""
    rig = namespace.rig.resolve()
    if not (rig / "config.ini").is_file() or not (rig / "logs").is_dir():
        parser.error(f"rig must contain config.ini and logs/: {rig}")
    if namespace.min_height < 0:
        parser.error("--min-height must be >= 0")
    if namespace.limit is not None and namespace.limit < 1:
        parser.error("--limit must be >= 1")
    if namespace.grace < 0:
        parser.error("--grace must be >= 0")
    if namespace.threads < 1:
        parser.error("--threads must be >= 1")
    if namespace.sample_every < 1:
        parser.error("--sample-every must be >= 1")
    scratch_dir = namespace.scratch_dir.resolve()
    if not scratch_dir.is_dir():
        parser.error(f"--scratch-dir must be an existing directory: {scratch_dir}")
    if not math.isfinite(namespace.psnr_min):
        parser.error("--psnr-min must be finite")
    if namespace.allow_unbacked_delete and not namespace.no_upload:
        parser.error("--allow-unbacked-delete requires --no-upload")
    extras = namespace.rclone_extra
    if extras is None:
        extras = ["--drive-team-drive", DEFAULT_TEAM_DRIVE]
    return PackOptions(
        rig=rig,
        min_height=namespace.min_height,
        verify=namespace.verify,
        dry_run=namespace.dry_run,
        keep=namespace.keep,
        force=namespace.force,
        grace=namespace.grace,
        no_upload=namespace.no_upload,
        allow_unbacked_delete=namespace.allow_unbacked_delete,
        remote=namespace.remote,
        host_label=namespace.host_label,
        ffmpeg=namespace.ffmpeg,
        ffprobe=namespace.ffprobe,
        rclone=namespace.rclone,
        rclone_extra=tuple(extras),
        threads=namespace.threads,
        crf=namespace.crf,
        preset=namespace.preset,
        psnr_min=namespace.psnr_min,
        sample_every=namespace.sample_every,
        scratch_dir=scratch_dir,
        raw=namespace.raw,
        keep_raw_local=namespace.keep_raw_local,
        since=namespace.since,
        policies=tuple(namespace.policy or ()),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    run: RunCallable = subprocess.run,
    popen: PopenCallable = subprocess.Popen,
    clock: ClockCallable = time.time,
    sleep: SleepCallable = time.sleep,
) -> int:
    """Parse arguments, execute the selected mode, and return a documented exit code."""
    parser = _parser()
    try:
        namespace = parser.parse_args(argv)
        options = _options(namespace, parser)
    except SystemExit as exc:
        return int(exc.code)
    if namespace.status:
        return _print_status(options)
    if namespace.all:
        return _run_all(
            options,
            namespace.limit,
            run=run,
            popen=popen,
            clock=clock,
            sleep=sleep,
        )
    log_path = _resolve_run_argument(namespace.run, options.rig)
    return pack_one(
        log_path,
        options,
        run=run,
        popen=popen,
        clock=clock,
        sleep=sleep,
    )


if __name__ == "__main__":
    sys.exit(main())
