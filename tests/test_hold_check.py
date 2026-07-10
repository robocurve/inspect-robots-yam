"""hold_check: the 6.4 hold-behavior verification (plan 0008 §6.4)."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_yam.hold_check import HoldResult, main, run_hold_check


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
