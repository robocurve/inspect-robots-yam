"""hold_check: the 6.4 hold-behavior verification (plan 0008 §6.4)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from inspect_robots_yam.hold_check import HoldResult, main, run_hold_check


def _write_config_file(path: Path, values: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    args = "\n".join(f"{key} = {value}" for key, value in values.items())
    path.write_text(
        f"[defaults]\nembodiment = yam_arms\n[embodiment.args]\n{args}\n",
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path: Path, values: dict[str, str]) -> dict[str, str]:
    _write_config_file(tmp_path / "inspect-robots" / "config.ini", values)
    return {"XDG_CONFIG_HOME": str(tmp_path)}


class _FakeArm:
    """A single 7-D arm whose pose drifts by `drift_per_read` each get."""

    def __init__(self, drift_per_read: float = 0.0):
        self.pose = np.zeros(7)
        self.drift = drift_per_read
        self.commands: list[np.ndarray] = []
        self.reads = 0

    def get_joint_pos(self) -> np.ndarray:
        self.reads += 1
        self.pose = self.pose + self.drift
        return self.pose.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.commands.append(np.asarray(target).copy())


def _run(arm: _FakeArm, **kwargs: object) -> HoldResult:
    sleeps: list[float] = []
    result = run_hold_check(
        robot=arm,
        duration_s=20.0,
        interval_s=5.0,
        sleep_fn=sleeps.append,
        emit=lambda _line: None,
        **kwargs,  # type: ignore[arg-type]
    )
    assert sleeps == [5.0] * 4
    return result


def test_holding_arm_passes() -> None:
    arm = _FakeArm(drift_per_read=0.0)
    result = _run(arm)
    assert result.max_drift == pytest.approx(0.0)
    assert result.passed is True
    assert len(arm.commands) == 1  # exactly one command: the current pose


def test_drifting_arm_fails_and_reports_worst_joint() -> None:
    arm = _FakeArm(drift_per_read=0.01)
    result = _run(arm)
    assert result.passed is False
    assert result.max_drift > 0.01
    assert result.samples  # per-interval history retained for the report


def test_thresholds_are_configurable() -> None:
    arm = _FakeArm(drift_per_read=0.001)
    assert _run(arm, settle_rad=1.0, trend_rad=1.0).passed is True


def test_settle_and_trend_are_judged_separately() -> None:
    class _SettlingArm(_FakeArm):
        """Settles 0.03 on the first read after command, then holds flat."""

        def get_joint_pos(self):  # type: ignore[no-untyped-def]
            self.reads += 1
            if self.reads > 1:
                self.pose = np.full(7, 0.03)
            return self.pose.copy()

    settled = _run(_SettlingArm())
    assert settled.settle == pytest.approx(0.03)
    assert settled.trend == pytest.approx(0.0)
    assert settled.passed is True  # one-time settle within the generous limit

    drifting = _run(_FakeArm(drift_per_read=0.02))
    assert drifting.trend > 0.01
    assert drifting.passed is False  # growth after the first sample = sag


def test_main_wires_argv_and_exit_codes() -> None:
    lines: list[str] = []
    holding = _FakeArm()

    def factory(channel: str, zero_gravity_mode: bool) -> _FakeArm:
        assert channel == "can0" and zero_gravity_mode is True
        return holding

    rc = main(
        ["can0", "--zero-gravity", "true", "--duration-s", "10", "--interval-s", "5"],
        robot_factory=factory,
        sleep_fn=lambda _s: None,
        emit=lines.append,
    )
    assert rc == 0
    assert any("PASS" in line for line in lines)

    drifting = _FakeArm(drift_per_read=0.05)
    rc = main(
        ["can1", "--zero-gravity", "false"],
        robot_factory=lambda channel, zero_gravity_mode: drifting,
        sleep_fn=lambda _s: None,
        emit=lines.append,
    )
    assert rc == 1
    assert any("FAIL" in line for line in lines)


@pytest.mark.parametrize(
    ("side", "key", "resolved"),
    [
        ("left", "left_channel", "can_left"),
        ("right", "right_channel", "can_right"),
    ],
)
def test_main_resolves_wizard_side_and_prints_resolved_channel(
    side: str,
    key: str,
    resolved: str,
    tmp_path: Path,
) -> None:
    env = _write_config(tmp_path, {key: resolved})
    lines: list[str] = []
    calls: list[tuple[str, bool]] = []

    def factory(channel: str, zero_gravity: bool) -> _FakeArm:
        calls.append((channel, zero_gravity))
        return _FakeArm()

    assert (
        main(
            [side, "--zero-gravity", "false", "--duration-s", "0"],
            env=env,
            robot_factory=factory,
            sleep_fn=lambda _seconds: None,
            emit=lines.append,
        )
        == 0
    )
    assert calls == [(resolved, False)]
    assert lines[0] == f"{resolved} ({side}) zero_gravity=false: watching for 0s"


def test_main_loads_cwd_dotenv_config_pin_for_wizard_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned_path = _write_config_file(
        tmp_path / "pinned" / "config.ini",
        {"left_channel": "can_pinned_left"},
    )
    (tmp_path / ".env").write_text(
        f"INSPECT_ROBOTS_CONFIG={pinned_path}\n",
        encoding="utf-8",
    )
    copied_environ = dict(os.environ)
    assert "INSPECT_ROBOTS_CONFIG" not in copied_environ
    monkeypatch.setattr(os, "environ", copied_environ)
    calls: list[tuple[str, bool]] = []

    def factory(channel: str, zero_gravity: bool) -> _FakeArm:
        calls.append((channel, zero_gravity))
        return _FakeArm()

    assert (
        main(
            ["left", "--zero-gravity", "false", "--duration-s", "0"],
            robot_factory=factory,
            sleep_fn=lambda _seconds: None,
            emit=lambda _line: None,
        )
        == 0
    )
    assert calls == [("can_pinned_left", False)]


def test_raw_channel_never_loads_a_malformed_config(tmp_path: Path) -> None:
    path = tmp_path / "inspect-robots" / "config.ini"
    path.parent.mkdir()
    path.write_text("[defaults\n", encoding="utf-8")
    channels: list[str] = []

    assert (
        main(
            ["can0", "--zero-gravity", "true", "--duration-s", "0"],
            env={"XDG_CONFIG_HOME": str(tmp_path)},
            robot_factory=lambda channel, _mode: channels.append(channel) or _FakeArm(),
            sleep_fn=lambda _seconds: None,
            emit=lambda _line: None,
        )
        == 0
    )
    assert channels == ["can0"]


def test_side_without_config_has_guided_parser_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            ["left", "--zero-gravity", "true"],
            env={"XDG_CONFIG_HOME": str(tmp_path)},
        )

    assert exc_info.value.code == 2
    assert (
        "left requires left_channel in the wizard config; run inspect-robots setup, "
        "or pass the CAN channel name directly"
    ) in capsys.readouterr().err


def test_no_config_forces_side_resolution_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _write_config(tmp_path, {"right_channel": "can_right"})

    with pytest.raises(SystemExit) as exc_info:
        main(["right", "--zero-gravity", "true", "--no-config"], env=env)

    assert exc_info.value.code == 2
    assert "right requires right_channel in the wizard config" in capsys.readouterr().err


def test_channel_help_explains_claimed_side_literals(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "interface literally named left or right must be renamed" in output


def test_main_rejects_bad_zero_gravity_value() -> None:
    with pytest.raises(SystemExit):
        main(["can0", "--zero-gravity", "maybe"], sleep_fn=lambda _s: None, emit=lambda _l: None)


def test_main_closes_the_robot_even_on_failure() -> None:
    class _ClosableArm(_FakeArm):
        closed = 0

        def close(self) -> None:
            _ClosableArm.closed += 1

    arm = _ClosableArm(drift_per_read=0.5)  # guaranteed FAIL verdict
    rc = main(
        ["can0", "--zero-gravity", "true"],
        robot_factory=lambda channel, zero_gravity_mode: arm,
        sleep_fn=lambda _s: None,
        emit=lambda _l: None,
    )
    assert rc == 1
    assert _ClosableArm.closed == 1  # released regardless of the verdict


def test_default_emit_flushes(capsys: pytest.CaptureFixture[str]) -> None:
    from inspect_robots_yam.hold_check import _print_flushed

    _print_flushed("hello")
    assert capsys.readouterr().out == "hello\n"
