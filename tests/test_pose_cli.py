"""Tests for the injected named-pose operator CLI."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots_yam import pose_cli, poses
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.operator import OperatorIO


class FakeDriver:
    def __init__(self, state: np.ndarray | None = None) -> None:
        self.state = np.zeros(14) if state is None else state.copy()
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def get_joint_eff(self) -> np.ndarray:
        return np.zeros(14)

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.commands.append(np.asarray(target, dtype=float).copy())
        self.state = np.asarray(target, dtype=float).copy()

    def close(self) -> None:
        self.closed = True


class ScriptedConsole:
    def __init__(self, responses: list[str | BaseException] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[str] = []
        self.lines: list[str] = []
        self.io = OperatorIO(input_fn=self.input, output_fn=self.lines.append)

    def input(self, prompt: str) -> str:
        self.prompts.append(prompt)
        response = self.responses.pop(0) if self.responses else ""
        if isinstance(response, BaseException):
            raise response
        return response


def _main_args(tmp_path: Path, *command: str) -> list[str]:
    return ["--no-config", "--pose-dir", str(tmp_path), *command]


def _now() -> datetime:
    return datetime(2026, 8, 19, 12, 30, tzinfo=timezone.utc)


def _save(tmp_path: Path, name: str = "ready", joints: tuple[float, ...] | None = None) -> None:
    poses.save_pose(
        tmp_path,
        poses.StartPose(
            name=name,
            joints=joints or (0.2,) * 14,
            created_at="2026-08-19T12:00:00+00:00",
            notes="verify",
            rig="rig-a",
        ),
    )


def test_capture_writes_wire_shape_and_pins_zero_gravity(tmp_path: Path) -> None:
    state = np.arange(14, dtype=float) / 10
    state[:6] = 0.1
    state[7:13] = 0.2
    state[6] = 17.0
    state[13] = 12.0
    driver = FakeDriver(state)
    configs: list[YamConfig] = []

    def factory(cfg: YamConfig) -> FakeDriver:
        configs.append(cfg)
        return driver

    console = ScriptedConsole(["", ""])
    code = pose_cli.main(
        [
            "--no-config",
            "--pose-dir",
            str(tmp_path),
            "-E",
            "gripper_open=10",
            "-E",
            "gripper_closed=20",
            "capture",
            "ready",
            "--notes",
            "table setup",
        ],
        driver_factory=factory,
        io=console.io,
        sleep_fn=lambda _delay: None,
        now_fn=_now,
        hostname_fn=lambda: "rig-1",
    )
    saved = poses.load_pose(tmp_path, "ready")
    assert code == 0
    assert configs[0].zero_gravity_mode is True
    assert saved.joints[6] == pytest.approx(0.3)
    assert saved.joints[13] == pytest.approx(0.8)
    assert saved.created_at == "2026-08-19T12:30:00+00:00"
    assert saved.notes == "table setup" and saved.rig == "rig-1"
    assert console.prompts[-1] == "support both arms, then press Enter to release torque"
    assert any("-E start_pose=ready" in line for line in console.lines)
    assert driver.closed is True


def test_capture_silently_clamps_gripper_slots(tmp_path: Path) -> None:
    state = np.zeros(14)
    state[6] = 1.2
    state[13] = -0.2
    driver = FakeDriver(state)
    console = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "grips"),
            driver_factory=lambda _cfg: driver,
            io=console.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 0
    )
    stored = poses.load_pose(tmp_path, "grips")
    assert stored.joints[6] == 1.0 and stored.joints[13] == 0.0
    assert not any("clamped" in line for line in console.lines)


def test_capture_joint_range_error_and_clamp_diff(tmp_path: Path) -> None:
    state = np.zeros(14)
    state[0] = 4.0
    error_driver = FakeDriver(state)
    error_console = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "unsafe"),
            driver_factory=lambda _cfg: error_driver,
            io=error_console.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 1
    )
    assert not (tmp_path / "unsafe.json").exists()
    assert any("left_j0 (packed index 0)=4.0 outside" in line for line in error_console.lines)

    clamp_driver = FakeDriver(state)
    clamp_console = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "safe", "--clamp"),
            driver_factory=lambda _cfg: clamp_driver,
            io=clamp_console.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 0
    )
    assert poses.load_pose(tmp_path, "safe").joints[0] == pytest.approx(np.pi)
    assert any("clamped left_j0 (packed index 0): 4.0 ->" in line for line in clamp_console.lines)


def test_capture_refuses_existing_before_hardware_and_force_replaces(tmp_path: Path) -> None:
    _save(tmp_path)
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        return FakeDriver(np.full(14, 0.4))

    refused = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "ready"),
            driver_factory=factory,
            io=refused.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 1
    )
    assert calls == 0
    assert "pass --force" in refused.lines[0]

    forced = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "ready", "--force"),
            driver_factory=factory,
            io=forced.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 0
    )
    assert calls == 1
    assert poses.load_pose(tmp_path, "ready").joints == pytest.approx((0.4,) * 14)


def test_capture_ctrl_c_writes_nothing_and_still_gates_torque_off(tmp_path: Path) -> None:
    driver = FakeDriver()
    console = ScriptedConsole([KeyboardInterrupt(), ""])
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "aborted"),
            driver_factory=lambda _cfg: driver,
            io=console.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 1
    )
    assert poses.pose_names(tmp_path) == ()
    assert console.prompts == [
        "Press Enter to snapshot...",
        "support both arms, then press Enter to release torque",
    ]
    assert driver.closed is True


def test_capture_park_ramps_then_closes_with_two_safe_exit_prompts(tmp_path: Path) -> None:
    driver = FakeDriver(np.full(14, 0.2))
    console = ScriptedConsole()
    sleeps: list[float] = []
    assert (
        pose_cli.main(
            [
                "--no-config",
                "--pose-dir",
                str(tmp_path),
                "-E",
                "rest_secs=0.2",
                "capture",
                "parked",
                "--park",
            ],
            driver_factory=lambda _cfg: driver,
            io=console.io,
            sleep_fn=sleeps.append,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 0
    )
    assert len(driver.commands) == 2
    assert sleeps == pytest.approx([0.1, 0.1])
    assert console.prompts[-2:] == [
        "Arms will move to the rest pose - stand clear, then press Enter...",
        "support both arms, then press Enter to release torque",
    ]
    assert driver.closed is True


@pytest.mark.parametrize("command", [("capture", "p"), ("goto", "p")])
def test_park_without_rest_pose_is_usage_error(tmp_path: Path, command: tuple[str, str]) -> None:
    if command[0] == "goto":
        _save(tmp_path, "p")
    with pytest.raises(SystemExit) as caught:
        pose_cli.main(
            [
                "--no-config",
                "--pose-dir",
                str(tmp_path),
                "-E",
                "rest_pose=none",
                *command,
                "--park",
            ],
            driver_factory=lambda _cfg: pytest.fail("factory called"),
            io=ScriptedConsole().io,
        )
    assert caught.value.code == 2


def test_goto_ramp_holds_reports_and_releases(tmp_path: Path) -> None:
    target = (0.4,) * 14
    _save(tmp_path, joints=target)
    driver = FakeDriver()
    console = ScriptedConsole()
    sleeps: list[float] = []
    assert (
        pose_cli.main(
            [
                "--no-config",
                "--pose-dir",
                str(tmp_path),
                "-E",
                "rest_secs=0.3",
                "goto",
                "ready",
            ],
            driver_factory=lambda _cfg: driver,
            io=console.io,
            sleep_fn=sleeps.append,
        )
        == 0
    )
    assert len(driver.commands) == 3
    assert driver.commands[-1] == pytest.approx(target)
    assert console.prompts == [
        "Arms will move to start pose 'ready' - stand clear, then press Enter...",
        "Press Enter to finish inspecting this pose...",
        "support both arms, then press Enter to release torque",
    ]
    assert any("final measured pose" in line for line in console.lines)
    assert driver.closed is True


def test_goto_park_gate_precedes_park_motion(tmp_path: Path) -> None:
    _save(tmp_path, joints=(0.3,) * 14)
    driver = FakeDriver()
    prompt_command_counts: list[tuple[str, int]] = []

    def respond(prompt: str) -> str:
        prompt_command_counts.append((prompt, len(driver.commands)))
        return ""

    io = OperatorIO(input_fn=respond, output_fn=lambda _line: None)
    assert (
        pose_cli.main(
            [
                "--no-config",
                "--pose-dir",
                str(tmp_path),
                "-E",
                "rest_secs=0.2",
                "goto",
                "ready",
                "--park",
            ],
            driver_factory=lambda _cfg: driver,
            io=io,
            sleep_fn=lambda _delay: None,
        )
        == 0
    )
    assert prompt_command_counts[0][1] == 0
    assert prompt_command_counts[2] == (
        "Arms will move to the rest pose - stand clear, then press Enter...",
        2,
    )
    assert len(driver.commands) == 4


def test_goto_rejects_out_of_range_before_hardware(tmp_path: Path) -> None:
    values = [0.0] * 14
    values[1] = 4.0
    _save(tmp_path, joints=tuple(values))
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        return FakeDriver()

    console = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "goto", "ready"),
            driver_factory=factory,
            io=console.io,
        )
        == 1
    )
    assert calls == 0
    assert "left_j1" in console.lines[0]


@pytest.mark.parametrize("command", [("capture", "p"), ("goto", "p")])
def test_noninteractive_hardware_commands_exit_two_before_factory(
    tmp_path: Path,
    command: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if command[0] == "goto":
        _save(tmp_path, "p")
    monkeypatch.setattr(pose_cli, "stdin_interactive", lambda: False)
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        return FakeDriver()

    with pytest.raises(SystemExit) as caught:
        pose_cli.main(_main_args(tmp_path, *command), driver_factory=factory)
    assert caught.value.code == 2
    assert calls == 0


def test_injected_io_bypasses_noninteractive_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pose_cli, "stdin_interactive", lambda: False)
    driver = FakeDriver()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "p"),
            driver_factory=lambda _cfg: driver,
            io=ScriptedConsole().io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 0
    )


def test_pose_dir_flag_and_extra_conflict_is_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        pose_cli.main(
            ["--no-config", "--pose-dir", str(tmp_path), "-E", "pose_dir=other", "list"],
            io=ScriptedConsole().io,
        )
    assert caught.value.code == 2


def test_store_commands_list_show_delete_and_rename(tmp_path: Path) -> None:
    _save(tmp_path, "beta")
    _save(tmp_path, "alpha")
    listed = ScriptedConsole()
    assert pose_cli.main(_main_args(tmp_path, "list"), io=listed.io) == 0
    assert listed.lines[0] == "name\tcreated_at\trig\tnotes"
    assert [line.split("\t")[0] for line in listed.lines[1:]] == ["alpha", "beta"]

    shown = ScriptedConsole()
    assert pose_cli.main(_main_args(tmp_path, "show", "alpha"), io=shown.io) == 0
    assert any(line == "left: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]" for line in shown.lines)
    assert any(str(tmp_path / "alpha.json") in line for line in shown.lines)

    renamed = ScriptedConsole()
    assert pose_cli.main(_main_args(tmp_path, "rename", "alpha", "gamma"), io=renamed.io) == 0
    assert poses.load_pose(tmp_path, "gamma").name == "gamma"

    deleted = ScriptedConsole()
    assert pose_cli.main(_main_args(tmp_path, "delete", "gamma"), io=deleted.io) == 0
    assert not (tmp_path / "gamma.json").exists()
    missing = ScriptedConsole()
    assert pose_cli.main(_main_args(tmp_path, "delete", "missing"), io=missing.io) == 1
    assert "available poses: beta" in missing.lines[0]


def test_store_failure_and_invalid_name_exit_codes(tmp_path: Path) -> None:
    broken = ScriptedConsole()
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert pose_cli.main(_main_args(tmp_path, "list"), io=broken.io) == 1
    assert "broken.json" in broken.lines[1]
    with pytest.raises(SystemExit) as caught:
        pose_cli.main(_main_args(tmp_path, "show", "../bad"), io=ScriptedConsole().io)
    assert caught.value.code == 2


@pytest.mark.parametrize("extra", ["missing-equals", "=value", "unknown=1"])
def test_bad_extras_are_usage_errors(tmp_path: Path, extra: str) -> None:
    with pytest.raises(SystemExit) as caught:
        pose_cli.main(
            ["--no-config", "--pose-dir", str(tmp_path), "-E", extra, "list"],
            io=ScriptedConsole().io,
        )
    assert caught.value.code == 2


def test_digit_only_pose_extras_stay_strings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[YamConfig] = []

    def record(_args: Any, cfg: YamConfig, _io: OperatorIO) -> int:
        seen.append(cfg)
        return 0

    monkeypatch.setattr(pose_cli, "_store_command", record)
    assert (
        pose_cli.main(
            ["--no-config", "-E", "pose_dir=007", "-E", "start_pose=42", "list"],
            io=ScriptedConsole().io,
        )
        == 0
    )
    assert seen[0].pose_dir == "007"
    assert seen[0].start_pose == "42"


def test_wizard_config_and_no_config_pose_directory(tmp_path: Path) -> None:
    config = tmp_path / "inspect-robots" / "config.ini"
    config.parent.mkdir()
    configured_dir = tmp_path / "configured"
    config.write_text(
        f"[defaults]\nembodiment = yam_arms\n[embodiment.args]\npose_dir = {configured_dir}\n",
        encoding="utf-8",
    )
    _save(configured_dir, "configured")
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    configured = ScriptedConsole()
    assert pose_cli.main(["list"], env=env, io=configured.io) == 0
    assert any(line.startswith("configured\t") for line in configured.lines)
    bypassed = ScriptedConsole()
    assert pose_cli.main(["--no-config", "list"], env=env, io=bypassed.io) == 0
    assert bypassed.lines == ["name\tcreated_at\trig\tnotes"]


def test_malformed_config_is_bypassed_only_with_no_config(tmp_path: Path) -> None:
    config = tmp_path / "inspect-robots" / "config.ini"
    config.parent.mkdir()
    config.write_text("[defaults\n", encoding="utf-8")
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    with pytest.raises(SystemExit, match="error in"):
        pose_cli.main(["list"], env=env, io=ScriptedConsole().io)
    assert pose_cli.main(["--no-config", "list"], env=env, io=ScriptedConsole().io) == 0


def test_cwd_dotenv_pins_pose_config_for_real_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pose_dir = tmp_path / "pinned-poses"
    _save(pose_dir, "pinned")
    config = tmp_path / "pinned" / "config.ini"
    config.parent.mkdir()
    config.write_text(
        f"[defaults]\nembodiment = yam_arms\n[embodiment.args]\npose_dir = {pose_dir}\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        f"INSPECT_ROBOTS_CONFIG={config}\n",
        encoding="utf-8",
    )
    copied_environ = dict(os.environ)
    copied_environ.pop("INSPECT_ROBOTS_CONFIG", None)
    monkeypatch.setattr(os, "environ", copied_environ)
    console = ScriptedConsole()

    assert pose_cli.main(["list"], io=console.io) == 0
    assert any(line.startswith("pinned\t") for line in console.lines)


def test_driver_and_teardown_failures_return_one(tmp_path: Path) -> None:
    console = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "p"),
            driver_factory=lambda _cfg: (_ for _ in ()).throw(RuntimeError("connect fault")),
            io=console.io,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 1
    )
    assert "connect fault" in console.lines[0]

    class CloseFault(FakeDriver):
        def close(self) -> None:
            raise RuntimeError("close fault")

    close_console = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "capture", "q"),
            driver_factory=lambda _cfg: CloseFault(),
            io=close_console.io,
            sleep_fn=lambda _delay: None,
            now_fn=_now,
            hostname_fn=lambda: "rig",
        )
        == 1
    )
    assert any("torque-off teardown" in line for line in close_console.lines)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("true", True), ("FALSE", False), ("2.5", 2.5), ("word", "word")],
)
def test_parse_scalar_forms(text: str, expected: object) -> None:
    assert pose_cli._parse_scalar(text) == expected


def test_goto_missing_interrupt_driver_and_teardown_failures(tmp_path: Path) -> None:
    missing = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "goto", "missing"),
            driver_factory=lambda _cfg: pytest.fail("factory called"),
            io=missing.io,
        )
        == 1
    )
    assert "available poses" in missing.lines[0]

    _save(tmp_path)
    interrupted_driver = FakeDriver()
    interrupted = ScriptedConsole([KeyboardInterrupt(), ""])
    assert (
        pose_cli.main(
            _main_args(tmp_path, "goto", "ready"),
            driver_factory=lambda _cfg: interrupted_driver,
            io=interrupted.io,
            sleep_fn=lambda _delay: None,
        )
        == 1
    )
    assert "goto interrupted" in interrupted.lines

    connect = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "goto", "ready"),
            driver_factory=lambda _cfg: (_ for _ in ()).throw(RuntimeError("connect fault")),
            io=connect.io,
        )
        == 1
    )
    assert "connect fault" in connect.lines[0]

    class CloseFault(FakeDriver):
        def close(self) -> None:
            raise RuntimeError("close fault")

    teardown = ScriptedConsole()
    assert (
        pose_cli.main(
            _main_args(tmp_path, "goto", "ready"),
            driver_factory=lambda _cfg: CloseFault(),
            io=teardown.io,
            sleep_fn=lambda _delay: None,
        )
        == 1
    )
    assert any("torque-off teardown" in line for line in teardown.lines)


def test_unexpected_store_command_is_reported() -> None:
    args = argparse.Namespace(command="unexpected")
    console = ScriptedConsole()
    assert pose_cli._store_command(args, YamConfig(), console.io) == 1
    assert "unexpected store command" in console.lines[0]
