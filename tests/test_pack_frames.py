import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_module() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "pack_frames.py"
    spec = importlib.util.spec_from_file_location("pack_frames", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pack_frames = _load_module()


def _frame(height: int, width: int, value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _make_rig(
    root: Path,
    name: str = "run-a",
    *,
    height: int = 360,
    width: int = 4,
    count: int = 2,
    status: str = "success",
    started_at: str | None = None,
    created: str | None = None,
    policy: str | None = None,
) -> tuple[Path, Path, Path]:
    rig = root
    (rig / "config.ini").write_text("[defaults]\n", encoding="utf-8")
    frames_dir = rig / "logs" / "frames" / f"stamp-{name}"
    frames_dir.mkdir(parents=True)
    for camera_index, camera in enumerate(("left", "top", "right")):
        for step in range(count):
            np.save(
                frames_dir / f"scene-0-e0_{camera}_cam_{step:06d}.npy",
                _frame(height, width, camera_index * 20 + step),
            )
    log_path = rig / "logs" / f"{name}.json"
    stats = {"frames_dir": str(frames_dir.relative_to(rig))}
    if started_at is not None:
        stats["started_at"] = started_at
    payload: dict[str, Any] = {
        "status": status,
        "stats": stats,
        "embodiment_info": {"control_hz": 30},
    }
    if created is not None or policy is not None:
        payload["eval"] = {}
        if created is not None:
            payload["eval"]["created"] = created
        if policy is not None:
            payload["eval"]["policy"] = policy
    log_path.write_text(json.dumps(payload), encoding="utf-8")
    return rig, log_path, frames_dir


def _write_log(log_path: Path, frames_dir: Path, rig: Path, status: str = "success") -> None:
    log_path.write_text(
        json.dumps(
            {
                "status": status,
                "stats": {"frames_dir": str(frames_dir.relative_to(rig))},
                "control_hz": 20,
            }
        ),
        encoding="utf-8",
    )


def _completed(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> Any:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeStdin:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.data = bytearray()

    def write(self, data: bytes) -> int:
        self.data.extend(data)
        return len(data)

    def close(self) -> None:
        marker = b"fake-ffv1" if ".ffv1.mkv" in self.output.name else b"fake-mp4"
        self.output.write_bytes(marker)


class _FakeProcess:
    def __init__(self, argv: list[str], *, decoded: bytes | None = None) -> None:
        self.argv = argv
        self.stdin = _FakeStdin(Path(argv[-1])) if decoded is None else None
        self.stdout = io.BytesIO(decoded) if decoded is not None else None

    def kill(self) -> None:
        pass

    def wait(self) -> int:
        return 0


class FakeTools:
    def __init__(
        self,
        frames_dir: Path,
        *,
        rclone_failure: bool = False,
        corrupt_after_check: bool = False,
        copyto_failure: bool = False,
        corrupt_raw_decode: bool = False,
        rclone_timeout: bool = False,
    ) -> None:
        self.frames_dir = frames_dir
        self.rclone_failure = rclone_failure
        self.corrupt_after_check = corrupt_after_check
        self.copyto_failure = copyto_failure
        self.corrupt_raw_decode = corrupt_raw_decode
        self.rclone_timeout = rclone_timeout
        self.commands: list[list[str]] = []
        self.run_timeouts: list[tuple[list[str], float | None]] = []
        self.encode_outputs: list[Path] = []
        self.frames_dir_at_encode: list[set[str]] = []
        self.scratch_at_encode: list[set[str]] = []

    def popen(self, argv: list[str], **_kwargs: Any) -> _FakeProcess:
        self.commands.append(argv)
        if argv[-1] == "-" and any(".ffv1.mkv" in value for value in argv):
            camera = re.search(r"_(left|top|right)_cam", " ".join(argv))
            assert camera is not None
            files = sorted(self.frames_dir.glob(f"scene-0-e0_{camera.group(1)}_cam_*.npy"))
            decoded = bytearray(b"".join(np.load(path).tobytes() for path in files))
            if self.corrupt_raw_decode and decoded:
                decoded[len(decoded) // 2] ^= 1
            return _FakeProcess(argv, decoded=bytes(decoded))
        output = Path(argv[-1])
        self.encode_outputs.append(output)
        self.frames_dir_at_encode.append({path.name for path in self.frames_dir.iterdir()})
        self.scratch_at_encode.append({path.name for path in output.parent.iterdir()})
        return _FakeProcess(argv)

    def run(self, argv: list[str], **_kwargs: Any) -> Any:
        self.commands.append(argv)
        self.run_timeouts.append((argv, _kwargs.get("timeout")))
        if argv[1:2] == ["-version"]:
            return _completed(stdout=b"ffmpeg fake 1.0\n")
        if "-count_packets" in argv:
            camera = re.search(r"_(left|top|right)_cam", Path(argv[-1]).name)
            assert camera is not None
            files = sorted(self.frames_dir.glob(f"scene-0-e0_{camera.group(1)}_cam_*.npy"))
            first = np.load(files[0], mmap_mode="r")
            height, width = first.shape[:2]
            is_raw = ".ffv1.mkv" in Path(argv[-1]).name
            payload = {
                "streams": [
                    {
                        "nb_frames": "N/A" if is_raw else str(len(files)),
                        "nb_read_packets": str(len(files)),
                        "width": width if is_raw else width + width % 2,
                        "height": height if is_raw else height + height % 2,
                    }
                ]
            }
            return _completed(stdout=json.dumps(payload).encode())
        if "rawvideo" in argv and "select=" in " ".join(argv):
            camera = re.search(r"_(left|top|right)_cam", " ".join(argv))
            assert camera is not None
            files = sorted(self.frames_dir.glob(f"scene-0-e0_{camera.group(1)}_cam_*.npy"))
            select = next(value for value in argv if value.startswith("select="))
            indices = [int(value) for value in re.findall(r"eq\(n\\,(\d+)\)", select)]
            decoded = bytearray()
            for index in indices:
                frame = np.load(files[index])
                pad_height = frame.shape[0] % 2
                pad_width = frame.shape[1] % 2
                padded = np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)))
                decoded.extend(padded.tobytes())
            return _completed(stdout=bytes(decoded))
        if argv[0] == "rclone" or argv[0].endswith("/rclone"):
            if self.rclone_timeout and argv[1] == "copy":
                raise subprocess.TimeoutExpired(argv, _kwargs.get("timeout"))
            if self.rclone_failure and argv[1] == "copy":
                return _completed(returncode=1, stderr=b"upload failed")
            if self.copyto_failure and argv[1] == "copyto":
                return _completed(returncode=1, stderr=b"final manifest failed")
            if self.corrupt_after_check and argv[1] == "check":
                source = Path(argv[-2])
                (source / "scene-0-e0_left_cam.mp4").write_bytes(b"corrupt")
            return _completed()
        raise AssertionError(f"unexpected command: {argv}")


def _main_args(rig: Path, log_path: Path, *extra: str) -> list[str]:
    scratch_dir = rig / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    return [
        "--run",
        str(log_path),
        "--rig",
        str(rig),
        "--min-height",
        "0",
        "--ffmpeg",
        "ffmpeg",
        "--ffprobe",
        "ffprobe",
        "--rclone",
        "rclone",
        "--scratch-dir",
        str(scratch_dir),
        *extra,
    ]


def test_eligibility_status_live_missing_and_empty(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    info = pack_frames.load_run(log_path, rig)
    assert pack_frames.check_eligible(info, 360) is None

    _write_log(log_path, frames_dir, rig, "started")
    assert "started" in pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 360)
    _write_log(log_path, frames_dir, rig)
    live = log_path.with_name("run-a.live.json")
    live.write_text("{}", encoding="utf-8")
    assert "live log" in pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 360)
    live.unlink()

    missing = frames_dir.with_name("missing")
    _write_log(log_path, missing, rig)
    assert "missing" in pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 360)
    _write_log(log_path, frames_dir, rig)
    for path in frames_dir.glob("*.npy"):
        path.unlink()
    assert "no .npy" in pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 360)


def test_eligibility_packed_hash_and_height(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path, height=224)
    reason = pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 360)
    assert reason == "frame height 224 < 360"

    mp4 = frames_dir / "scene-0-e0_left_cam.mp4"
    mp4.write_bytes(b"video")
    digest = hashlib.sha256(b"video").hexdigest()
    manifest = {
        "state": "packed",
        "streams": {"left": {"file": mp4.name, "sha256": digest}},
    }
    (frames_dir / "pack_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 360) == "already packed"
    mp4.write_bytes(b"changed")
    assert pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 0) == "already packed"


def test_since_filter_uses_started_at_and_created_fallback(tmp_path: Path) -> None:
    rig, old_log, _old_frames = _make_rig(
        tmp_path,
        "old",
        started_at="2026-08-13T03:27:59+00:00",
    )
    _rig, new_log, _new_frames = _make_rig(
        tmp_path,
        "new",
        started_at="2026-08-13T03:29:00+00:00",
    )
    _rig, fallback_log, _fallback_frames = _make_rig(
        tmp_path,
        "fallback",
        created="2026-08-13T03:30:00Z",
    )
    since = pack_frames._since_argument("2026-08-13T03:28:00+00:00")

    assert (
        pack_frames.check_eligible(
            pack_frames.load_run(old_log, rig),
            since=since,
        )
        == "started 2026-08-13T03:27 before --since"
    )
    assert pack_frames.check_eligible(pack_frames.load_run(new_log, rig), since=since) is None
    assert pack_frames.check_eligible(pack_frames.load_run(fallback_log, rig), since=since) is None
    assert (
        pack_frames.main(
            [
                "--run",
                str(fallback_log),
                "--rig",
                str(rig),
                "--since",
                "2026-08-13T03:28:00Z",
                "--dry-run",
            ]
        )
        == 0
    )

    _rig, missing_log, _missing_frames = _make_rig(tmp_path, "missing-time")
    assert (
        pack_frames.check_eligible(
            pack_frames.load_run(missing_log, rig),
            since=since,
        )
        == "no start timestamp for --since"
    )
    assert pack_frames._since_argument("2026-08-13T03:28:00").utcoffset() is not None


def test_policy_filter_matches_eval_policy(tmp_path: Path) -> None:
    rig, molmo_log, _molmo_frames = _make_rig(tmp_path, "molmo", policy="molmoact2")
    _rig, pi05_log, _pi05_frames = _make_rig(tmp_path, "pi05", policy="pi05")
    _rig, missing_log, _missing_frames = _make_rig(tmp_path, "missing-policy")

    assert (
        pack_frames.check_eligible(
            pack_frames.load_run(molmo_log, rig),
            policies=("pi05",),
        )
        == "policy molmoact2 not in (pi05)"
    )
    assert (
        pack_frames.check_eligible(
            pack_frames.load_run(pi05_log, rig),
            policies=("pi05",),
        )
        is None
    )
    assert (
        pack_frames.check_eligible(
            pack_frames.load_run(missing_log, rig),
            policies=("pi05",),
        )
        == "policy <none> not in (pi05)"
    )


def test_status_counts_since_and_policy_skip_reasons(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, _old_log, _old_frames = _make_rig(
        tmp_path,
        "old",
        started_at="2026-08-13T03:27:00+00:00",
        policy="pi05",
    )
    _rig, _molmo_log, _molmo_frames = _make_rig(
        tmp_path,
        "molmo",
        started_at="2026-08-13T03:29:00+00:00",
        policy="molmoact2",
    )
    _rig, _pi05_log, _pi05_frames = _make_rig(
        tmp_path,
        "pi05",
        started_at="2026-08-13T03:29:00+00:00",
        policy="pi05",
    )

    assert (
        pack_frames.main(
            [
                "--status",
                "--rig",
                str(rig),
                "--since",
                "2026-08-13T03:28:00+00:00",
                "--policy",
                "pi05",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "1  started 2026-08-13T03:27 before --since" in output
    assert "1  policy molmoact2 not in (pi05)" in output


def test_packed_detection_uses_metadata_fast_path_and_verify_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    mp4 = frames_dir / "scene-0-e0_left_cam.mp4"
    mp4.write_bytes(b"video")
    stat = mp4.stat()
    digest = hashlib.sha256(b"video").hexdigest()
    manifest = {
        "state": "packed",
        "streams": {
            "left": {
                "file": mp4.name,
                "sha256": digest,
                "bytes": stat.st_size,
                "mtime": stat.st_mtime,
            }
        },
    }
    (frames_dir / "pack_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes: list[Path] = []

    def recording_hash(path: Path) -> str:
        hashes.append(path)
        return digest

    monkeypatch.setattr(pack_frames, "_sha256", recording_hash)
    info = pack_frames.load_run(log_path, rig)
    assert pack_frames.check_eligible(info, 0) == "already packed"
    assert hashes == []
    assert pack_frames.check_eligible(info, 0, verify=True) == "already packed"
    assert hashes == [mp4]

    hashes.clear()
    os.utime(mp4, (stat.st_atime, stat.st_mtime + 1))
    assert pack_frames.check_eligible(info, 0) == "already packed"
    assert hashes == [mp4]


@pytest.mark.parametrize("control_hz", [True, 0, -1.0, float("inf"), float("nan"), "30"])
def test_load_run_rejects_invalid_control_hz(tmp_path: Path, control_hz: Any) -> None:
    rig, log_path, _frames_dir = _make_rig(tmp_path)
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    payload["embodiment_info"]["control_hz"] = control_hz
    log_path.write_text(json.dumps(payload), encoding="utf-8")
    assert pack_frames.load_run(log_path, rig).control_hz == 10.0


def test_discover_streams_rejects_stray_duplicate_and_records_gaps(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    first = frames_dir / "scene-0-e0_left_cam_000000.npy"
    third = frames_dir / "scene-0-e0_left_cam_000003.npy"
    np.save(first, _frame(2, 2, 0))
    np.save(third, _frame(2, 2, 1))
    streams = pack_frames.discover_streams(frames_dir)
    assert streams["left"].gaps == ((1, 2),)

    stray = frames_dir / "surprise.npy"
    np.save(stray, _frame(2, 2, 2))
    with pytest.raises(pack_frames.PackError, match="stray"):
        pack_frames.discover_streams(frames_dir)
    stray.unlink()
    with pytest.raises(pack_frames.PackError, match="duplicate or non-increasing"):
        pack_frames.discover_streams(frames_dir, [first, first, third])


def test_all_orders_smallest_first_filters_height_and_honours_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, _small_log, _small_frames = _make_rig(tmp_path, "small", height=360, count=1)
    _rig, _large_log, _large_frames = _make_rig(tmp_path, "large", height=360, count=3)
    _rig, _short_log, _short_frames = _make_rig(tmp_path, "short", height=224, count=1)
    order: list[str] = []

    def fake_pack(log_path: Path, _options: Any, **_kwargs: Any) -> int:
        order.append(log_path.stem)
        return 0

    monkeypatch.setattr(pack_frames, "pack_one", fake_pack)
    assert pack_frames.main(["--all", "--rig", str(rig), "--no-upload"]) == 0
    assert order == ["small", "large"]
    assert "frame height 224 < 360" in capsys.readouterr().out

    order.clear()
    assert pack_frames.main(["--all", "--rig", str(rig), "--limit", "1"]) == 0
    assert order == ["small"]


@pytest.mark.parametrize(
    (
        "extra",
        "rclone_failure",
        "expected_code",
        "expected_state",
        "npy_remain",
        "artifacts_remain",
    ),
    [
        ((), True, 1, None, True, False),
        (("--no-upload",), False, 1, None, True, False),
        (
            ("--no-upload", "--allow-unbacked-delete", "--force"),
            False,
            0,
            "packed",
            False,
            True,
        ),
        (
            ("--no-upload", "--allow-unbacked-delete", "--keep", "--force"),
            False,
            0,
            "packed-kept",
            True,
            True,
        ),
        (("--force",), False, 0, "packed", False, True),
    ],
)
def test_state_machine(
    tmp_path: Path,
    extra: tuple[str, ...],
    rclone_failure: bool,
    expected_code: int,
    expected_state: str | None,
    npy_remain: bool,
    artifacts_remain: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir, rclone_failure=rclone_failure)
    result = pack_frames.main(
        _main_args(rig, log_path, *extra),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == expected_code
    manifest_path = frames_dir / "pack_manifest.json"
    assert manifest_path.exists() is artifacts_remain
    if expected_state is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["state"] == expected_state
    assert bool(list(frames_dir.glob("*.npy"))) is npy_remain
    assert (
        all(
            (frames_dir / f"scene-0-e0_{camera}_cam.mp4").is_file()
            for camera in ("left", "top", "right")
        )
        is artifacts_remain
    )
    assert all(output.parent.is_relative_to(rig / "scratch") for output in tools.encode_outputs)
    assert all("pack_manifest.json" not in listing for listing in tools.frames_dir_at_encode)
    assert all(
        not any(name.endswith(".mp4") for name in listing) for listing in tools.frames_dir_at_encode
    )
    assert not list((rig / "scratch").glob("pack_frames-*"))
    if extra == ("--no-upload",):
        assert "refusing to delete .npy without a verified upload" in capsys.readouterr().err


def test_upload_success_checks_only_mp4s_and_copyto_is_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir)
    states: list[str] = []
    real_write_manifest = pack_frames.write_manifest

    def recording_write(path: Path, manifest: dict[str, Any]) -> None:
        states.append(manifest["state"])
        real_write_manifest(path, manifest)

    monkeypatch.setattr(pack_frames, "write_manifest", recording_write)
    result = pack_frames.main(
        _main_args(rig, log_path, "--force"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert states == ["encoded", "uploaded", "packed", "packed", "packed"]
    rclone_commands = [command for command in tools.commands if command[0] == "rclone"]
    assert [command[1] for command in rclone_commands] == ["copy", "check", "copyto"]
    rclone_timeouts = [timeout for command, timeout in tools.run_timeouts if command[0] == "rclone"]
    assert rclone_timeouts == [6 * 60 * 60, 30 * 60, 10 * 60]
    for command in rclone_commands:
        assert "--contimeout" in command
        assert "--timeout" in command
        assert "--retries" in command
        assert "--low-level-retries" in command
    copy_command, check_command, copyto_command = rclone_commands
    assert "pack_manifest.json" in copy_command
    assert "pack_manifest.json" not in check_command
    assert "scene-0-e0_*_cam.ffv1.mkv" in copy_command
    assert "scene-0-e0_*_cam.ffv1.mkv" in check_command
    for camera in ("left", "top", "right"):
        name = f"scene-0-e0_{camera}_cam.mp4"
        assert name in check_command
    assert Path(copyto_command[-2]) == frames_dir / "pack_manifest.json"
    assert copyto_command[-1].endswith("/pack_manifest.json")
    final = json.loads((frames_dir / "pack_manifest.json").read_text(encoding="utf-8"))
    assert final["state"] == "packed"
    assert final["raw_verified"] == "bit-exact"
    assert set(final["raw"]) == {"left", "top", "right"}
    assert not list(frames_dir.glob("*.ffv1.mkv"))


def test_stale_temporary_outputs_are_removed_in_frames_and_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    scratch_root = rig / "scratch"
    scratch_root.mkdir()
    scratch = scratch_root / "pack_frames-injected"
    scratch.mkdir()
    frame_stale = frames_dir / "scene-0-e0_left_cam.mp4.tmp.tmp"
    scratch_stale = scratch / "scene-0-e0_top_cam.mp4.tmp-faststart"
    frame_stale.write_bytes(b"stale")
    scratch_stale.write_bytes(b"stale")
    tools = FakeTools(frames_dir)

    def fake_mkdtemp(*, prefix: str, dir: str) -> str:
        assert prefix == "pack_frames-"
        assert Path(dir) == scratch_root
        return str(scratch)

    monkeypatch.setattr(pack_frames.tempfile, "mkdtemp", fake_mkdtemp)
    result = pack_frames.main(
        _main_args(
            rig,
            log_path,
            "--no-upload",
            "--allow-unbacked-delete",
            "--keep",
            "--force",
        ),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert not frame_stale.exists()
    assert not scratch.exists()
    assert all(
        not any(".mp4.tmp" in name for name in listing) for listing in tools.scratch_at_encode
    )


def test_final_manifest_copyto_failure_is_only_a_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir, copyto_failure=True)
    result = pack_frames.main(
        _main_args(rig, log_path, "--force"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert not list(frames_dir.glob("*.npy"))
    assert (frames_dir / "pack_manifest.json").is_file()
    assert "final manifest upload failed after backup" in capsys.readouterr().err


def test_staged_hash_mismatch_aborts_before_deletion(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir, corrupt_after_check=True)
    result = pack_frames.main(
        _main_args(rig, log_path, "--force"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 1
    assert list(frames_dir.glob("*.npy"))
    assert not list(frames_dir.glob("*.mp4"))
    assert not (frames_dir / "pack_manifest.json").exists()
    assert not list((rig / "scratch").glob("pack_frames-*"))


def test_foreign_mp4_is_warned_and_overwritten(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    foreign = frames_dir / "scene-0-e0_left_cam.mp4"
    foreign.write_bytes(b"inspect-robots-video")
    tools = FakeTools(frames_dir)
    result = pack_frames.main(
        _main_args(
            rig,
            log_path,
            "--no-upload",
            "--allow-unbacked-delete",
            "--keep",
            "--force",
        ),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert foreign.read_bytes() == b"fake-mp4"
    assert "foreign MP4" in capsys.readouterr().err


def test_pack_lock_and_log_follow_selected_json_parent(tmp_path: Path) -> None:
    rig, original_log, frames_dir = _make_rig(tmp_path)
    custom_logs = rig / "alternate-logs"
    custom_logs.mkdir()
    log_path = custom_logs / original_log.name
    original_log.replace(log_path)
    tools = FakeTools(frames_dir)
    result = pack_frames.main(
        _main_args(rig, log_path, "--no-upload"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 1
    assert (custom_logs / "pack" / ".lock").is_file()
    assert (custom_logs / "pack" / "stamp-run-a.log").is_file()
    assert not (rig / "logs" / "pack").exists()


def test_grace_uses_remaining_time_then_deletes(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    os.utime(log_path, (50.0, 50.0))
    tools = FakeTools(frames_dir)
    now = [100.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    result = pack_frames.main(
        _main_args(
            rig,
            log_path,
            "--no-upload",
            "--allow-unbacked-delete",
            "--grace",
            "100",
        ),
        run=tools.run,
        popen=tools.popen,
        clock=clock,
        sleep=sleep,
    )
    assert result == 0
    assert sleeps == [50.0]
    assert not list(frames_dir.glob("*.npy"))


def test_force_skips_sleep(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir)

    def forbidden_sleep(_seconds: float) -> None:
        raise AssertionError("force must skip grace sleep")

    result = pack_frames.main(
        _main_args(
            rig,
            log_path,
            "--no-upload",
            "--allow-unbacked-delete",
            "--force",
        ),
        run=tools.run,
        popen=tools.popen,
        sleep=forbidden_sleep,
    )
    assert result == 0


def test_raw_none_skips_ffv1_and_manifest_keys(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir)
    result = pack_frames.main(
        _main_args(
            rig,
            log_path,
            "--raw",
            "none",
            "--no-upload",
            "--allow-unbacked-delete",
            "--keep",
            "--force",
        ),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert not any(".ffv1.mkv" in output.name for output in tools.encode_outputs)
    manifest = json.loads((frames_dir / "pack_manifest.json").read_text(encoding="utf-8"))
    assert "raw" not in manifest
    assert "raw_verified" not in manifest


def test_raw_bit_exact_verification_rejects_corrupted_byte(tmp_path: Path) -> None:
    _rig, _log_path, frames_dir = _make_rig(tmp_path)
    stream = pack_frames.discover_streams(frames_dir)["left"]
    raw_path = tmp_path / "scene-0-e0_left_cam.ffv1.mkv"
    raw_path.write_bytes(b"fake-ffv1")
    tools = FakeTools(frames_dir, corrupt_raw_decode=True)
    with pytest.raises(pack_frames.PackError, match="decoded bytes differ"):
        pack_frames.verify_raw_bit_exact(
            raw_path,
            stream,
            4,
            360,
            "ffmpeg",
            popen=tools.popen,
        )


def test_post_deletion_move_failure_preserves_scratch_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir)

    def failed_move(_source: str, _destination: str) -> None:
        raise OSError("move failed")

    monkeypatch.setattr(pack_frames.shutil, "move", failed_move)
    result = pack_frames.main(
        _main_args(rig, log_path, "--force"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 1
    scratch_dirs = list((rig / "scratch").glob("pack_frames-*"))
    assert len(scratch_dirs) == 1
    assert "staged outputs preserved at" in capsys.readouterr().err
    manifest = json.loads((frames_dir / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "packed"
    assert not list(frames_dir.glob("*.npy"))
    info = pack_frames.load_run(log_path, rig)
    assert pack_frames.check_eligible(info, 0) == "already packed"


def test_unlink_failure_is_counted_and_remaining_frame_is_not_reencoded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    failed_frame = sorted(frames_dir.glob("*.npy"))[0]
    real_unlink = Path.unlink

    def flaky_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == failed_frame:
            raise OSError("busy")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    tools = FakeTools(frames_dir)
    result = pack_frames.main(
        _main_args(rig, log_path, "--force"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert failed_frame.is_file()
    manifest = json.loads((frames_dir / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["npy_unlink_failures"] == 1
    assert pack_frames.check_eligible(pack_frames.load_run(log_path, rig), 0) == "already packed"


def test_rclone_timeout_keeps_raw_frames_and_discards_scratch(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir, rclone_timeout=True)
    result = pack_frames.main(
        _main_args(rig, log_path, "--force"),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 1
    assert len(list(frames_dir.glob("*.npy"))) == 6
    assert not (frames_dir / "pack_manifest.json").exists()
    assert not list((rig / "scratch").glob("pack_frames-*"))


def test_keep_raw_local_moves_lossless_archives(tmp_path: Path) -> None:
    rig, log_path, frames_dir = _make_rig(tmp_path)
    tools = FakeTools(frames_dir)
    result = pack_frames.main(
        _main_args(
            rig,
            log_path,
            "--no-upload",
            "--allow-unbacked-delete",
            "--keep-raw-local",
            "--force",
        ),
        run=tools.run,
        popen=tools.popen,
    )
    assert result == 0
    assert len(list(frames_dir.glob("*.ffv1.mkv"))) == 3


def test_status_reports_raw_coverage_and_live_dirs_not_orphans(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rig, _raw_log, raw_frames = _make_rig(tmp_path, "raw-packed")
    _rig, _old_log, old_frames = _make_rig(tmp_path, "old-packed")
    for frames_dir, raw_verified in ((raw_frames, "bit-exact"), (old_frames, None)):
        manifest: dict[str, Any] = {"state": "packed", "streams": {}}
        if raw_verified is not None:
            manifest["raw_verified"] = raw_verified
        (frames_dir / "pack_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    live_frames = rig / "logs" / "frames" / "live-stamp"
    live_frames.mkdir()
    np.save(live_frames / "scene-0-e0_left_cam_000000.npy", _frame(360, 4, 1))
    (rig / "logs" / "active.live.json").write_text(
        json.dumps({"stats": {"frames_dir": str(live_frames.relative_to(rig))}}),
        encoding="utf-8",
    )
    orphan = rig / "logs" / "frames" / "orphan-stamp"
    orphan.mkdir()
    np.save(orphan / "scene-0-e0_left_cam_000000.npy", _frame(360, 4, 2))

    assert pack_frames.main(["--status", "--rig", str(rig)]) == 0
    output = capsys.readouterr().out
    assert "packed raw           1" in output
    assert "packed no raw        1" in output
    assert "live dirs            1" in output
    assert "orphan dirs          1" in output


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for the integration test",
)
def test_ffmpeg_integration(tmp_path: Path) -> None:
    rig = tmp_path
    (rig / "config.ini").write_text("[defaults]\n", encoding="utf-8")
    frames_dir = rig / "logs" / "frames" / "integration-stamp"
    frames_dir.mkdir(parents=True)
    yy, xx = np.mgrid[:36, :64]
    for camera_index, camera in enumerate(("left", "top", "right")):
        for step in range(6):
            red = (xx * 3 + step * 2 + camera_index * 7) % 256
            green = (yy * 5 + step * 3 + camera_index * 11) % 256
            blue = (xx + yy + step * 4 + camera_index * 13) % 256
            frame = np.stack((red, green, blue), axis=2).astype(np.uint8)
            np.save(frames_dir / f"scene-0-e0_{camera}_cam_{step:06d}.npy", frame)
    log_path = rig / "logs" / "integration.json"
    _write_log(log_path, frames_dir, rig)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    scratch_dir = rig / "scratch"
    scratch_dir.mkdir()

    result = pack_frames.main(
        [
            "--run",
            str(log_path),
            "--rig",
            str(rig),
            "--min-height",
            "0",
            "--no-upload",
            "--allow-unbacked-delete",
            "--keep",
            "--force",
            "--ffmpeg",
            ffmpeg,
            "--ffprobe",
            ffprobe,
            "--scratch-dir",
            str(scratch_dir),
        ]
    )
    assert result == 0
    manifest = json.loads((frames_dir / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "packed-kept"
    assert manifest["raw_verified"] == "bit-exact"
    assert set(manifest["raw"]) == {"left", "top", "right"}
    assert not list(frames_dir.glob("*.ffv1.mkv"))
    assert not list(scratch_dir.glob("pack_frames-*"))
    for camera in ("left", "top", "right"):
        mp4 = frames_dir / f"scene-0-e0_{camera}_cam.mp4"
        assert mp4.is_file()
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_packets",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(mp4),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert int(probe.stdout.strip()) == 6
        assert all(sample[1] >= 35 for sample in manifest["streams"][camera]["psnr_samples"])


def test_default_fps_reads_eval_embodiment_info() -> None:
    data = {"eval": {"embodiment_info": {"control_hz": 30, "is_simulated": False}}, "stats": {}}
    assert pack_frames._default_fps(data) == 30.0
    assert pack_frames._default_fps({"embodiment_info": {"control_hz": 25}}) == 25.0
    assert pack_frames._default_fps({"eval": {"embodiment_info": {"control_hz": True}}}) == 10.0
    assert pack_frames._default_fps({}) == 10.0
