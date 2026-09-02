"""Tests for YAMEmbodiment (all hardware/IO seams injected — no CAN, cameras, stdin)."""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import pytest
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.types import Action

import inspect_robots_yam.embodiment as embodiment_module
from inspect_robots_yam import poses
from inspect_robots_yam.config import (
    DEFAULT_JOINT_HOME_POSE,
    DEFAULT_REST_POSE,
    YamConfig,
)
from inspect_robots_yam.embodiment import BimanualDriver, YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO


class FakeDriver:
    def __init__(
        self,
        state: np.ndarray | None = None,
        effort: np.ndarray | None = None,
        temps: np.ndarray | None = None,
        temps_seq: list[np.ndarray] | None = None,
    ) -> None:
        self.state = np.zeros(14) if state is None else state
        self.effort = np.zeros(14) if effort is None else effort
        self.temps = np.full(14, 30.0) if temps is None else temps
        self.temps_seq = list(temps_seq or [])
        self.temp_reads = 0
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def get_joint_eff(self) -> np.ndarray:
        return self.effort.copy()

    def get_motor_temps(self) -> np.ndarray:
        self.temp_reads += 1
        if self.temps_seq:
            return self.temps_seq.pop(0).copy()
        return self.temps.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.commands.append(np.asarray(target, dtype=float).copy())

    def close(self) -> None:
        self.closed = True


class EchoDriver(FakeDriver):
    """A driver whose reported position echoes the last commanded target."""

    def command_joint_pos(self, target: np.ndarray) -> None:
        super().command_joint_pos(target)
        self.state = np.asarray(target, dtype=float).copy()


def _cameras(_cfg):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": img, "left_cam": img, "right_cam": img}


def _operator(*, prompts: list[str] | None = None) -> OperatorIO:
    def _input(prompt: str) -> str:
        if prompts is not None:
            prompts.append(prompt)
        return ""

    return OperatorIO(input_fn=_input, output_fn=lambda _m: None)


class _RecordingOperator(OperatorIO):
    """Record readiness ownership controls without touching stdin."""

    def __init__(self) -> None:
        super().__init__(input_fn=lambda _prompt: "", output_fn=lambda _message: None)
        self.wait_calls: list[tuple[str, bool, bool]] = []

    def wait_ready(
        self,
        prompt: str = "Position the scene, then press Enter to start...",
        *,
        drain: bool = True,
        flush_first: bool = False,
    ) -> None:
        """Record the prompt and stdin ownership controls."""
        self.wait_calls.append((prompt, drain, flush_first))


class _RecordingSession:
    """Record the terminal operations owned by a framework operator session."""

    def __init__(self, gate_error: Exception | None = None) -> None:
        self.statuses: list[str | None] = []
        self.lines: list[str] = []
        self.gates: list[tuple[str, str | None]] = []
        self._gate_error = gate_error

    def status(self, line: str | None) -> None:
        """Record a replaceable status line or its closing marker."""
        self.statuses.append(line)

    def write_line(self, text: str) -> None:
        """Record one durable scrollback line."""
        self.lines.append(text)

    def gate(self, prompt: str, *, hint: str | None = None) -> None:
        """Record one readiness gate and raise its configured fault."""
        self.gates.append((prompt, hint))
        if self._gate_error is not None:
            raise self._gate_error


class _PacedClock:
    """Fake clock that only moves when someone sleeps, plus optional overrun.

    A frozen clock cannot tell a step-count counter apart from a wall-clock
    one, which is how #64 stayed invisible. This advances on the paced sleep
    the way real time does, and ``overrun`` adds time the pacing never
    accounts for, standing in for a settle or a slow camera read.
    """

    def __init__(self, overrun: float = 0.0) -> None:
        self.now = 0.0
        self.overrun = overrun

    def __call__(self) -> float:
        """Read the current fake time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance time by the slept interval, then by any configured overrun."""
        self.now += seconds + self.overrun


def _build(
    cfg: YamConfig | None = None,
    *,
    driver: FakeDriver | None = None,
    poll_end_seq: list[bool] | None = None,
    operator: OperatorIO | None = None,
):
    import dataclasses

    cfg = cfg or YamConfig()
    cfg = dataclasses.replace(cfg, cam_height=4, cam_width=4)
    drv = driver or FakeDriver()
    polls = list(poll_end_seq or [False])
    sleeps: list[float] = []
    emb = YAMEmbodiment(
        cfg or YamConfig(),
        driver_factory=lambda _c: drv,
        camera_reader=_cameras,
        operator=operator or _operator(),
        poll_end=lambda: polls.pop(0) if polls else False,
        sleep_fn=sleeps.append,
        clock=lambda: 0.0,
    )
    return emb, drv, sleeps


def test_zero_arg_info_no_hardware() -> None:
    emb = YAMEmbodiment()  # nothing mocked: construction must not touch hardware
    assert emb.info.name == "yam_arms"
    assert emb.info.action_space.dim == 14
    assert emb.info.action_space.low is not None and emb.info.action_space.high is not None
    assert emb.info.control_hz == 10.0
    assert SELF_PACED in emb.info.capabilities
    assert emb.info.observation_space.camera_names == frozenset(
        {"top_cam", "left_cam", "right_cam"}
    )
    assert emb.info.observation_space.state_keys == frozenset({"joint_pos"})


def test_reset_returns_observation_and_homes() -> None:
    # Homing is a smooth ramp (like the rest-pose motion), NOT a single jump:
    # rest_secs=2.0 at 10 Hz -> 20 interpolated commands ending at home.
    cfg = YamConfig(home_pose=(0.1,) * 14, rest_secs=2.0, gripper_open=10.0, gripper_closed=20.0)
    drv = EchoDriver()
    emb, _, _ = _build(cfg, driver=drv)
    obs = emb.reset(Scene(id="s", instruction="pour"))
    assert set(obs.images) == {"top_cam", "left_cam", "right_cam"}
    assert obs.state["joint_pos"].shape == (14,)
    assert obs.instruction == "pour"
    # The home pose is in policy units and goes through the same clamp+denorm
    # path as actions: joints pass through, gripper slots are de-normalized.
    assert len(drv.commands) == 20  # interpolated homing ramp
    j0 = [c[0] for c in drv.commands]
    assert all(b >= a for a, b in itertools.pairwise(j0))  # monotonic, no jump
    home_cmd = drv.commands[-1]
    assert home_cmd[0] == pytest.approx(0.1)
    assert home_cmd[6] == pytest.approx(19.0)  # 20 + 0.1 * (10 - 20)
    assert home_cmd[13] == pytest.approx(19.0)


def _save_start_pose(directory: Path, name: str, values: tuple[float, ...]) -> None:
    poses.save_pose(
        directory,
        poses.StartPose(
            name=name,
            joints=values,
            created_at="2026-08-19T12:00:00+00:00",
        ),
        overwrite=True,
    )


def test_named_start_pose_resolves_and_homing_ramps_to_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = (0.2,) * 6 + (0.4,) + (0.3,) * 6 + (0.6,)
    _save_start_pose(tmp_path, "ready", target)
    cfg = YamConfig(start_pose="ready", pose_dir=str(tmp_path), rest_secs=0.2)
    driver = EchoDriver()
    emb, _, _ = _build(cfg, driver=driver)

    with caplog.at_level("INFO", logger="inspect_robots_yam.embodiment"):
        emb.reset(Scene(id="s", instruction="x"))

    assert len(driver.commands) == 2
    assert driver.commands[-1] == pytest.approx(target)
    assert "resolved start pose 'ready'" in caplog.text
    assert str(tmp_path / "ready.json") in caplog.text


def test_named_start_pose_resolution_fails_before_driver_factory(tmp_path: Path) -> None:
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        return FakeDriver()

    emb = YAMEmbodiment(
        YamConfig(start_pose="missing", pose_dir=str(tmp_path)),
        driver_factory=factory,
        camera_reader=_cameras,
        operator=_operator(),
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(poses.PoseStoreError, match="available poses"):
        emb.reset(Scene(id="s", instruction="x"))
    assert calls == 0


def test_named_start_pose_out_of_bounds_names_indices_before_connect(tmp_path: Path) -> None:
    values = [0.0] * 14
    values[0] = 4.0
    values[8] = -4.0
    _save_start_pose(tmp_path, "unsafe", tuple(values))
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        return FakeDriver()

    emb = YAMEmbodiment(
        YamConfig(start_pose="unsafe", pose_dir=str(tmp_path)),
        driver_factory=factory,
        camera_reader=_cameras,
        operator=_operator(),
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(ValueError, match=r"unsafe.*packed indices \[0, 8\].*4\.0"):
        emb.reset(Scene(id="s", instruction="x"))
    assert calls == 0


def test_named_start_pose_is_cached_across_failed_factory_retry(tmp_path: Path) -> None:
    old = (0.1,) * 14
    new = (0.2,) * 14
    _save_start_pose(tmp_path, "ready", old)
    driver = EchoDriver()
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("connect fault")
        return driver

    emb = YAMEmbodiment(
        YamConfig(
            start_pose="ready",
            pose_dir=str(tmp_path),
            rest_secs=0.1,
            cam_height=4,
            cam_width=4,
        ),
        driver_factory=factory,
        camera_reader=_cameras,
        operator=_operator(),
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(RuntimeError, match="connect fault"):
        emb.reset(Scene(id="s", instruction="x"))
    _save_start_pose(tmp_path, "ready", new)
    emb.reset(Scene(id="s", instruction="x"))
    assert driver.commands[-1] == pytest.approx(old)


def test_close_clears_named_pose_cache_even_without_connection(tmp_path: Path) -> None:
    old = (0.1,) * 14
    new = (0.2,) * 14
    _save_start_pose(tmp_path, "ready", old)
    driver = EchoDriver()
    calls = 0

    def factory(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("connect fault")
        return driver

    emb = YAMEmbodiment(
        YamConfig(
            start_pose="ready",
            pose_dir=str(tmp_path),
            rest_secs=0.1,
            cam_height=4,
            cam_width=4,
        ),
        driver_factory=factory,
        camera_reader=_cameras,
        operator=_operator(),
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(RuntimeError, match="connect fault"):
        emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    _save_start_pose(tmp_path, "ready", new)
    emb.reset(Scene(id="s", instruction="x"))
    assert driver.commands[-1] == pytest.approx(new)


def test_named_start_pose_status_includes_name(tmp_path: Path) -> None:
    _save_start_pose(tmp_path, "ready", (0.1,) * 14)
    status: list[str | None] = []
    emb = YAMEmbodiment(
        YamConfig(
            start_pose="ready",
            pose_dir=str(tmp_path),
            rest_secs=0.1,
            cam_height=4,
            cam_width=4,
        ),
        driver_factory=lambda _cfg: EchoDriver(),
        camera_reader=_cameras,
        operator=_operator(),
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
        status_fn=status.append,
    )
    emb.reset(Scene(id="s", instruction="x"))
    assert status[0] == "homing: ramping arms to start pose 'ready'"


def test_joint_eff_is_absent_by_default() -> None:
    emb, _, _ = _build()

    observation = emb.reset(Scene(id="s", instruction="inspect"))

    assert "joint_eff" not in observation.state


def test_joint_eff_passes_through_raw_driver_values() -> None:
    effort = np.asarray([1, 2, 3, 4, 5, 6, 73, -1, -2, -3, -4, -5, -6, -91])
    driver = FakeDriver(effort=effort)
    emb, _, _ = _build(YamConfig(report_joint_eff=True), driver=driver)

    observation = emb.reset(Scene(id="s", instruction="inspect"))
    reported = observation.state["joint_eff"]

    assert reported.shape == (14,)
    assert reported.dtype == np.float64
    assert reported == pytest.approx(effort)
    assert reported[[6, 13]] == pytest.approx((73, -91))
    assert "joint_eff" not in emb.info.observation_space.state_keys


def test_joint_eff_survives_full_reset_step_cycle_without_warnings() -> None:
    emb, _, _ = _build(YamConfig(report_joint_eff=True))

    with warnings.catch_warnings(record=True) as caught:
        observation = emb.reset(Scene(id="s", instruction="inspect"))
        result = emb.step(Action(data=np.zeros(14)))

    assert observation.state["joint_eff"].shape == (14,)
    assert result.observation.state["joint_eff"].shape == (14,)
    assert caught == []


def test_joint_eff_requires_updated_injected_driver() -> None:
    class LegacyDriver:
        def get_joint_pos(self) -> np.ndarray:
            return np.zeros(14)

        def command_joint_pos(self, target: np.ndarray) -> None:
            del target

        def close(self) -> None:
            pass

    driver = LegacyDriver()
    emb = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4, report_joint_eff=True),
        driver_factory=lambda _cfg: cast(BimanualDriver, driver),
        camera_reader=_cameras,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )

    with pytest.raises(
        RuntimeError,
        match=r"report_joint_eff=true.*get_joint_eff\(\)",
    ):
        emb.reset(Scene(id="s", instruction="inspect"))


def test_motor_temp_guardrail_default_off_never_reads_or_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    emb, driver, _ = _build()

    with caplog.at_level("WARNING", logger="inspect_robots_yam.embodiment"):
        emb.reset(Scene(id="s", instruction="inspect"))
        emb.step(Action(data=np.zeros(14)))

    assert driver.temp_reads == 0
    assert "thermal guardrail" not in caplog.text


def test_motor_temp_mid_run_trip_uses_session_notice_and_skips_motion() -> None:
    temperatures = np.full(14, 30.0)
    temperatures[8] = 80.0
    driver = FakeDriver(
        temps_seq=[np.full(14, 30.0), temperatures, temperatures],
    )
    emb, _, _ = _build(YamConfig(motor_temp_limit=80.0), driver=driver)
    session = _RecordingSession()
    emb.connect_operator_session(session)
    emb.reset(Scene(id="s", instruction="inspect"))
    command_count = len(driver.commands)

    result = emb.step(Action(data=np.ones(14)))

    assert result.terminated
    assert result.termination_reason == "overheat"
    assert len(driver.commands) == command_count
    assert result.info["overheat"] == {
        "slot": 8,
        "label": "right_j1",
        "motor_id": 2,
        "channel": "can1",
        "temp": 80.0,
    }
    assert session.statuses[-1] is None
    assert len(session.lines) == 1
    assert all(text in session.lines[0] for text in ("right_j1", "motor id 2", "can1", "80"))


def test_motor_temp_mid_run_trip_uses_unconnected_operator_notice() -> None:
    lines: list[str] = []
    statuses: list[str | None] = []
    temperatures = np.full(14, 30.0)
    temperatures[0] = 81.0
    driver = FakeDriver(temps_seq=[np.full(14, 30.0), temperatures, temperatures])
    cfg = YamConfig(motor_temp_limit=80.0, cam_height=4, cam_width=4)
    emb = YAMEmbodiment(
        cfg,
        driver_factory=lambda _cfg: driver,
        camera_reader=_cameras,
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=lines.append),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        status_fn=statuses.append,
    )
    emb.reset(Scene(id="s", instruction="inspect"))

    result = emb.step(Action(data=np.zeros(14)))

    assert result.terminated
    assert statuses[-1] is None
    assert len(lines) == 1
    assert all(text in lines[0] for text in ("left_j0", "motor id 1", "can0", "81"))


def test_motor_temp_confirmation_rejects_glitch_and_uses_fallback_sleep() -> None:
    hot = np.full(14, 30.0)
    hot[3] = 80.0
    driver = FakeDriver(temps_seq=[np.full(14, 30.0), hot, np.full(14, 30.0)])
    emb, _, sleeps = _build(
        YamConfig(motor_temp_limit=80.0, control_hz=0.0, rest_secs=0.1),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="inspect"))
    sleeps.clear()
    command_count = len(driver.commands)

    result = emb.step(Action(data=np.zeros(14)))

    assert not result.terminated
    assert len(driver.commands) == command_count + 1
    assert sleeps == [0.1]


def test_motor_temp_nonpositive_sentinels_never_trip() -> None:
    driver = FakeDriver(temps=np.full(14, -1.0))
    emb, _, _ = _build(
        YamConfig(motor_temp_limit=0.1, motor_temp_warn_margin=0.01),
        driver=driver,
    )

    emb.reset(Scene(id="s", instruction="inspect"))
    result = emb.step(Action(data=np.zeros(14)))

    assert not result.terminated


def test_motor_temp_no_data_warns_once_per_trial_and_resets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver = FakeDriver(temps=np.full(14, 30.0))
    emb, _, _ = _build(YamConfig(motor_temp_limit=80.0), driver=driver)
    emb.reset(Scene(id="s", instruction="inspect"))
    driver.temps[:] = -1.0

    with caplog.at_level("WARNING", logger="inspect_robots_yam.embodiment"):
        emb.step(Action(data=np.zeros(14)))
        emb.step(Action(data=np.zeros(14)))
        emb.reset(Scene(id="s2", instruction="inspect again"))

    warnings = [
        record
        for record in caplog.records
        if "thermal guardrail got no valid temperature data" in record.message
    ]
    assert len(warnings) == 2


def test_motor_temp_right_gripper_slot_trips() -> None:
    hot = np.full(14, 30.0)
    hot[13] = 85.0
    driver = FakeDriver(temps_seq=[np.full(14, 30.0), hot, hot])
    emb, _, _ = _build(YamConfig(motor_temp_limit=80.0), driver=driver)
    emb.reset(Scene(id="s", instruction="inspect"))

    result = emb.step(Action(data=np.zeros(14)))

    assert result.termination_reason == "overheat"
    assert result.info["overheat"] == {
        "slot": 13,
        "label": "right_gripper",
        "motor_id": 7,
        "channel": "can1",
        "temp": 85.0,
    }


def test_motor_temp_warns_once_per_trial_and_resets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    driver = FakeDriver(temps=np.full(14, 30.0))
    emb, _, _ = _build(YamConfig(motor_temp_limit=80.0), driver=driver)
    emb.reset(Scene(id="s", instruction="inspect"))
    driver.temps[4] = 72.0

    with caplog.at_level("WARNING", logger="inspect_robots_yam.embodiment"):
        emb.step(Action(data=np.zeros(14)))
        emb.step(Action(data=np.zeros(14)))
        emb.reset(Scene(id="s2", instruction="inspect again"))

    warnings = [
        record for record in caplog.records if "thermal guardrail warning" in record.message
    ]
    assert len(warnings) == 2
    assert all("left_j4" in record.message for record in warnings)


def test_motor_temp_reset_read_warns_inside_margin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    temperatures = np.full(14, 30.0)
    temperatures[7] = 75.0
    emb, _, _ = _build(
        YamConfig(motor_temp_limit=80.0),
        driver=FakeDriver(temps=temperatures),
    )

    with caplog.at_level("WARNING", logger="inspect_robots_yam.embodiment"):
        emb.reset(Scene(id="s", instruction="inspect"))

    assert "thermal guardrail warning: right_j0" in caplog.text


def test_motor_temp_pre_run_gate_faults_before_motion_and_close_does_not_ramp() -> None:
    hot = np.full(14, 30.0)
    hot[2] = 82.0
    driver = FakeDriver(temps_seq=[hot, hot])
    emb, _, sleeps = _build(YamConfig(motor_temp_limit=80.0), driver=driver)

    with pytest.raises(EmbodimentFault, match="thermal guardrail") as caught:
        emb.reset(Scene(id="s", instruction="inspect"))

    message = str(caught.value)
    assert all(text in message for text in ("left_j2", "motor id 3", "can0", "82"))
    assert driver.commands == []
    assert sleeps == [0.1]
    emb.close()
    assert driver.commands == []
    assert driver.closed


def test_motor_temp_pre_run_confirmation_rejects_glitch() -> None:
    hot = np.full(14, 30.0)
    hot[1] = 80.0
    driver = FakeDriver(temps_seq=[hot, np.full(14, 30.0)])
    emb, _, _ = _build(YamConfig(motor_temp_limit=80.0), driver=driver)

    observation = emb.reset(Scene(id="s", instruction="inspect"))

    assert observation.instruction == "inspect"
    assert driver.commands


def test_motor_temp_warm_second_reset_faults_and_close_still_ramps() -> None:
    cold = np.full(14, 30.0)
    hot = cold.copy()
    hot[9] = 83.0
    driver = FakeDriver(temps_seq=[cold, hot, hot])
    emb, _, _ = _build(
        YamConfig(motor_temp_limit=80.0, rest_secs=0.1),
        driver=driver,
    )
    emb.reset(Scene(id="s", instruction="inspect"))
    first_trial_commands = len(driver.commands)

    with pytest.raises(EmbodimentFault, match="thermal guardrail"):
        emb.reset(Scene(id="s2", instruction="inspect again"))

    assert len(driver.commands) == first_trial_commands
    emb.close()
    assert len(driver.commands) > first_trial_commands
    assert driver.closed


def test_motor_temp_limit_requires_updated_injected_driver() -> None:
    class LegacyDriver:
        def get_joint_pos(self) -> np.ndarray:
            return np.zeros(14)

        def command_joint_pos(self, target: np.ndarray) -> None:
            del target

        def close(self) -> None:
            pass

    driver = LegacyDriver()
    emb = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4, motor_temp_limit=80.0),
        driver_factory=lambda _cfg: cast(BimanualDriver, driver),
        camera_reader=_cameras,
        operator=_operator(),
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )

    with pytest.raises(RuntimeError, match=r"motor_temp_limit.*get_motor_temps"):
        emb.reset(Scene(id="s", instruction="inspect"))


def test_reset_without_home_pose_ramps_to_factory_joint_home() -> None:
    state = np.zeros(14)
    state[[6, 13]] = 20.0
    cfg = YamConfig(rest_secs=0.1, gripper_open=10.0, gripper_closed=20.0)
    emb, drv, _ = _build(cfg, driver=FakeDriver(state=state))
    emb.reset(Scene(id="s", instruction="x"))
    expected = np.asarray(DEFAULT_JOINT_HOME_POSE).copy()
    expected[[6, 13]] = cfg.gripper_open
    assert drv.commands[-1] == pytest.approx(expected)


def test_step_clamps_to_limits() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    # Way out of bounds; joints clip to +/-pi, gripper to [0,1].
    emb.step(Action(data=np.full(14, 100.0)))
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(np.pi)  # joint clamped
    # Wire gripper 1 is open and stays driver 1.0 under the default identity calibration.
    assert cmd[6] == pytest.approx(1.0)


def test_step_gripper_denormalization() -> None:
    cfg = YamConfig(gripper_open=10.0, gripper_closed=20.0)
    emb, drv, _ = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(14)))  # normalized gripper 0 -> closed value
    cmd = drv.commands[-1]
    assert cmd[6] == pytest.approx(20.0)
    assert cmd[13] == pytest.approx(20.0)
    emb.step(Action(data=np.concatenate([np.zeros(6), [1.0], np.zeros(6), [1.0]])))
    cmd = drv.commands[-1]
    assert cmd[6] == pytest.approx(10.0)  # normalized gripper 1 -> open value


def test_gripper_wire_endpoints_map_one_to_open_and_zero_to_closed() -> None:
    state = np.zeros(14)
    state[6] = state[13] = 0.72  # driver starts at the open endpoint
    cfg = YamConfig(gripper_open=0.72, gripper_closed=0.04)
    emb, drv, _ = _build(cfg, driver=FakeDriver(state=state))
    emb.reset(Scene(id="s", instruction="x"))

    open_action = np.zeros(14)
    open_action[6] = open_action[13] = 1.0
    emb.step(Action(data=open_action))
    assert drv.commands[-1][6] == pytest.approx(0.72)
    assert drv.commands[-1][13] == pytest.approx(0.72)

    emb.step(Action(data=np.zeros(14)))
    assert drv.commands[-1][6] == pytest.approx(0.04)
    assert drv.commands[-1][13] == pytest.approx(0.04)


def test_gripper_driver_open_endpoint_observes_as_wire_one() -> None:
    state = np.zeros(14)
    state[6] = state[13] = 0.72
    cfg = YamConfig(gripper_open=0.72, gripper_closed=0.04)
    emb, _, _ = _build(cfg, driver=FakeDriver(state=state))

    observation = emb.reset(Scene(id="s", instruction="x"))

    assert observation.state["joint_pos"][6] == pytest.approx(1.0)
    assert observation.state["joint_pos"][13] == pytest.approx(1.0)


def test_gripper_default_calibration_is_identity_both_directions() -> None:
    state = np.zeros(14)
    state[6] = state[13] = 0.35
    home = (0.0,) * 6 + (0.35,) + (0.0,) * 6 + (0.35,)
    drv = EchoDriver(state=state)
    emb, _, _ = _build(YamConfig(home_pose=home, rest_secs=0.1), driver=drv)

    observation = emb.reset(Scene(id="s", instruction="x"))
    assert observation.state["joint_pos"][6] == pytest.approx(0.35)
    assert observation.state["joint_pos"][13] == pytest.approx(0.35)

    action = np.zeros(14)
    action[6] = action[13] = 0.35
    result = emb.step(Action(data=action))
    assert drv.commands[-1][6] == pytest.approx(0.35)
    assert drv.commands[-1][13] == pytest.approx(0.35)
    assert result.observation.state["joint_pos"][6] == pytest.approx(0.35)
    assert result.observation.state["joint_pos"][13] == pytest.approx(0.35)


def test_step_delta_mode_adds_current() -> None:
    drv = FakeDriver(state=np.full(14, 0.5))
    cfg = YamConfig(joints_are_delta=True)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.1)))
    # current 0.5 + delta 0.1 = 0.6 (within +/-pi), gripper slots de-normalized below
    assert drv.commands[-1][0] == pytest.approx(0.6)


def test_gripper_absolute_round_trip_non_identity() -> None:
    cfg = YamConfig(gripper_open=10.0, gripper_closed=20.0)
    drv = EchoDriver()
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    action = np.zeros(14)
    action[6] = action[13] = 0.3
    result = emb.step(Action(data=action))
    # Outgoing: normalized 0.3 de-normalizes to 20 + 0.3 * (10 - 20) = 17 hw units.
    assert drv.commands[-1][6] == pytest.approx(17.0)
    assert drv.commands[-1][13] == pytest.approx(17.0)
    # Incoming: the observed state re-normalizes 17 hw back to exactly 0.3.
    state = result.observation.state["joint_pos"]
    assert state[6] == pytest.approx(0.3)
    assert state[13] == pytest.approx(0.3)


def test_gripper_positive_span_round_trip() -> None:
    cfg = YamConfig(gripper_open=20.0, gripper_closed=10.0)
    drv = EchoDriver()
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    action = np.zeros(14)
    action[6] = action[13] = 0.3
    result = emb.step(Action(data=action))
    # Open 20 and closed 10 is a normal calibration with a positive span:
    # outgoing wire 0.3 maps to 10 + 0.3 * (20 - 10) = 13 hw units.
    assert drv.commands[-1][6] == pytest.approx(13.0)
    assert drv.commands[-1][13] == pytest.approx(13.0)
    # Incoming: (13 - 10) / (20 - 10) = 0.3, so the bijection holds.
    state = result.observation.state["joint_pos"]
    assert state[6] == pytest.approx(0.3)
    assert state[13] == pytest.approx(0.3)


def test_step_delta_mode_gripper_uses_normalized_base() -> None:
    state = np.full(14, 0.5)
    state[6] = state[13] = 15.0  # hardware units: mid-stroke for open=10, closed=20
    drv = FakeDriver(state=state)
    cfg = YamConfig(joints_are_delta=True, gripper_open=10.0, gripper_closed=20.0)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.1)))
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(0.6)  # joints: plain radian addition
    # Gripper delta means fraction-of-stroke: 15 hw -> base 0.5 normalized,
    # +0.1 -> 0.6, de-normalized back out to 14 hw (NOT 15.1 or denorm(0.51)).
    assert cmd[6] == pytest.approx(14.0)
    assert cmd[13] == pytest.approx(14.0)


def test_reset_twice_reuses_driver() -> None:
    calls = {"n": 0}

    def _factory(_c):
        calls["n"] += 1
        return FakeDriver()

    emb = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4),
        driver_factory=_factory,
        camera_reader=_cameras,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    emb.reset(Scene(id="s", instruction="x"))
    emb.reset(Scene(id="s", instruction="x"))
    assert calls["n"] == 1  # driver built once, reused on the second reset


def test_step_terminates_operator_end_without_prompting() -> None:
    prompts: list[str] = []
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(prompts=prompts))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert result.termination_reason == "operator_end"
    assert "operator_confirmed" not in result.info
    # Grading is the framework prompt's job; the embodiment asks nothing.
    assert all("succeed" not in prompt for prompt in prompts)


def test_step_continues_when_no_end_signal() -> None:
    emb, _, _ = _build(poll_end_seq=[False])
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is False
    assert emb.num_steps == 1


def test_defer_operator_end_is_duck_typed_and_skips_poll() -> None:
    emb, _, _ = _build()
    polls: list[bool] = []

    def _poll_end() -> bool:
        polls.append(True)
        return True

    emb._poll_end = _poll_end
    hook = getattr(emb, "defer_operator_end", None)
    assert callable(hook)
    hook()
    emb.reset(Scene(id="s", instruction="x"))

    result = emb.step(Action(data=np.zeros(14)))

    assert result.terminated is False
    assert polls == []


def test_defer_operator_end_survives_resets() -> None:
    emb, _, _ = _build()
    polls: list[bool] = []
    emb._poll_end = lambda: polls.append(True) or True
    emb.defer_operator_end()
    scene = Scene(id="s", instruction="x")

    for _ in range(2):
        emb.reset(scene)
        assert emb.step(Action(data=np.zeros(14))).terminated is False

    assert polls == []


def test_connect_operator_session_owns_status_and_episode_end() -> None:
    session = _RecordingSession()
    constructor_status: list[str | None] = []
    polls: list[bool] = []
    driver = FakeDriver()
    clock = _PacedClock()
    emb = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4, control_hz=1.0),
        driver_factory=lambda _cfg: driver,
        camera_reader=_cameras,
        operator=_operator(),
        poll_end=lambda: polls.append(True) or True,
        sleep_fn=clock.sleep,
        clock=clock,
        status_fn=constructor_status.append,
    )

    emb.connect_operator_session(session)

    assert emb._deferred_operator_end is True
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    emb.close()

    assert result.terminated is False
    assert polls == []
    assert constructor_status == []
    assert session.statuses[0:3] == [
        "homing: ramping arms to start pose",
        None,
        "Running.",
    ]
    assert session.statuses[3] == "t = 1s | wall 1s"
    assert session.statuses[4:] == [
        "parking: ramping arms back before torque-off",
        None,
    ]


def test_pacing_sleeps_to_control_rate() -> None:
    emb, _, sleeps = _build()  # control_hz=10 -> period 0.1, clock constant 0 -> sleep ~0.1
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(14)))
    assert sleeps and sleeps[-1] == pytest.approx(0.1)


def test_pacing_skipped_when_hz_zero() -> None:
    cfg = YamConfig(control_hz=0.0)
    emb, _, sleeps = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    reset_sleeps = sleeps.copy()
    assert reset_sleeps
    emb.step(Action(data=np.zeros(14)))
    assert sleeps == reset_sleeps  # the step adds no pacing sleep at hz=0


def test_close_idempotent_and_releases() -> None:
    emb, drv, _ = _build()
    emb.close()  # before connect: no error
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert drv.closed is True
    emb.close()  # second close: no error


def test_step_before_reset_raises() -> None:
    emb, _, _ = _build()
    with pytest.raises(RuntimeError, match="before reset"):
        emb.step(Action(data=np.zeros(14)))


def test_reset_default_camera_reader_fails_fast_before_connect() -> None:
    calls = {"n": 0}

    def _factory(_c):
        calls["n"] += 1
        return FakeDriver()

    emb = YAMEmbodiment(
        YamConfig(home_pose=(0.0,) * 14),
        driver_factory=_factory,  # no camera_reader: the unusable default remains
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(ConfigError, match="camera_reader"):
        emb.reset(Scene(id="s", instruction="x"))
    assert calls["n"] == 0  # raised BEFORE any driver connect / homing motion


def test_reset_non_callable_camera_reader_fails_fast() -> None:
    # The CLI can only bind scalars, so `-E camera_reader=...` would arrive as a str.
    emb = YAMEmbodiment(
        YamConfig(),
        driver_factory=lambda _c: FakeDriver(),
        camera_reader="my_cams",  # type: ignore[arg-type]
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(
        ConfigError,
        match=r"\*_cam_device.*\*_depth_serial.*camera_reader",
    ):
        emb.reset(Scene(id="s", instruction="x"))


def test_unattended_skips_operator_prompts() -> None:
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return "y"

    op = OperatorIO(input_fn=_input, output_fn=lambda _m: None)
    emb, _, _ = _build(YamConfig(unattended=True), poll_end_seq=[True], operator=op)
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert prompts == []  # neither wait_ready nor the end poll ran
    assert result.terminated is False  # the end poll is skipped entirely


def test_auto_start_skips_gates_but_keeps_attended_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    prompts: list[str] = []
    lines: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    op = OperatorIO(input_fn=_input, output_fn=lines.append)
    emb, _, _ = _build(YamConfig(auto_start=True), poll_end_seq=[True], operator=op)
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert prompts == []  # neither Enter gate ran
    assert any("stand clear" in line for line in lines)  # notice replaces the home gate
    assert result.terminated is True  # end-episode keypress still active
    assert result.termination_reason == "operator_end"


def test_auto_start_keeps_running_status_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    emb, status = _build_with_status(YamConfig(auto_start=True))
    emb.reset(Scene(id="s", instruction="x"))
    assert any(line is not None and line.startswith("Running:") for line in status)


def test_auto_start_notice_prints_once_per_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    lines: list[str] = []
    op = OperatorIO(input_fn=lambda _p: "", output_fn=lines.append)
    emb, _, _ = _build(YamConfig(auto_start=True), operator=op)
    emb.reset(Scene(id="s1", instruction="x"))
    emb.reset(Scene(id="s2", instruction="x"))
    assert sum("stand clear" in line for line in lines) == 1
    emb.close()
    emb.reset(Scene(id="s3", instruction="x"))  # new connection: notice again
    assert sum("stand clear" in line for line in lines) == 2


def test_auto_start_drains_stdin_before_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    drains: list[bool] = []
    monkeypatch.setattr(embodiment_module, "_drain_stdin", lambda: drains.append(True))
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    op = OperatorIO(input_fn=_input, output_fn=lambda _m: None)
    emb, _, _ = _build(YamConfig(auto_start=True), operator=op)
    emb.reset(Scene(id="s", instruction="x"))
    assert drains == [True]  # wait_ready's drain is replaced, not dropped
    assert prompts == []


def test_deferred_auto_start_leaves_stdin_for_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    drains: list[bool] = []
    monkeypatch.setattr(embodiment_module, "_drain_stdin", lambda: drains.append(True))
    emb, _, _ = _build(YamConfig(auto_start=True))
    emb.defer_operator_end()

    emb.reset(Scene(id="s", instruction="x"))

    assert drains == []


def test_connected_auto_start_routes_notice_and_leaves_stdin_for_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: True)
    drains: list[bool] = []
    monkeypatch.setattr(embodiment_module, "_drain_stdin", lambda: drains.append(True))
    output_lines: list[str] = []
    operator = OperatorIO(input_fn=lambda _prompt: "", output_fn=output_lines.append)
    session = _RecordingSession()
    emb, _, _ = _build(YamConfig(auto_start=True), operator=operator)
    emb.connect_operator_session(session)

    emb.reset(Scene(id="s", instruction="x"))

    assert drains == []
    assert session.lines == ["auto_start: arms will move to the home pose - stand clear."]
    assert output_lines == []


@pytest.mark.parametrize(
    ("deferred", "expected_controls"),
    [
        (False, [(True, False), (True, False)]),
        (True, [(False, True), (False, True)]),
    ],
)
def test_reset_gates_follow_stdin_ownership(
    deferred: bool, expected_controls: list[tuple[bool, bool]]
) -> None:
    operator = _RecordingOperator()
    emb, _, _ = _build(operator=operator)
    if deferred:
        emb.defer_operator_end()

    emb.reset(Scene(id="s", instruction="x"))

    assert [(drain, flush) for _, drain, flush in operator.wait_calls] == expected_controls


def test_connected_reset_routes_both_exact_gates_through_session() -> None:
    operator = _RecordingOperator()
    session = _RecordingSession()
    emb, _, _ = _build(operator=operator)
    emb.connect_operator_session(session)

    emb.reset(Scene(id="s", instruction="x"))

    hint = "Set YamConfig(unattended=True) (CLI: -E unattended=true) to skip operator prompts."
    assert session.gates == [
        (
            "Arms will move to the home pose - stand clear, then press Enter...",
            hint,
        ),
        ("Position the scene, then press Enter to start...", hint),
    ]
    assert operator.wait_calls == []


def test_connected_gate_fault_propagates_unwrapped() -> None:
    fault = EmbodimentFault("session gate failed")
    session = _RecordingSession(gate_error=fault)
    operator = _RecordingOperator()
    emb, driver, _ = _build(operator=operator)
    emb.connect_operator_session(session)

    with pytest.raises(EmbodimentFault) as caught:
        emb.reset(Scene(id="s", instruction="x"))

    assert caught.value is fault
    assert operator.wait_calls == []
    assert driver.commands == []


def test_auto_start_requires_interactive_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embodiment_module, "stdin_interactive", lambda: False)
    emb, drv, _ = _build(YamConfig(auto_start=True))
    with pytest.raises(EmbodimentFault, match="auto_start"):
        emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []  # faulted before any motion


def test_unattended_precedes_auto_start() -> None:
    prompts: list[str] = []
    lines: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    op = OperatorIO(input_fn=_input, output_fn=lines.append)
    emb, _, _ = _build(
        YamConfig(unattended=True, auto_start=True), poll_end_seq=[True], operator=op
    )
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert prompts == []
    assert lines == []  # no stand-clear notice, no TTY fault: unattended wins outright
    assert result.terminated is False  # unattended still disables the end poll


def test_first_attended_reset_gates_home_motion_once_per_connection() -> None:
    drv = EchoDriver()
    prompt_calls: list[tuple[str, int]] = []

    def _input(prompt: str) -> str:
        prompt_calls.append((prompt, len(drv.commands)))
        return ""

    emb, _, _ = _build(
        YamConfig(rest_secs=0.1),
        driver=drv,
        operator=OperatorIO(input_fn=_input, output_fn=lambda _message: None),
    )
    scene = Scene(id="s", instruction="x")
    emb.reset(scene)
    emb.reset(scene)

    stand_clear_calls = [call for call in prompt_calls if "stand clear" in call[0]]
    assert stand_clear_calls == [
        ("Arms will move to the home pose - stand clear, then press Enter...", 0)
    ]
    assert len(drv.commands) == 2


def test_gate_fault_reprompts_before_motion_on_retried_reset() -> None:
    drv = EchoDriver()
    prompts: list[str] = []

    def _input(prompt: str) -> str:
        prompts.append(prompt)
        if "stand clear" in prompt and sum("stand clear" in p for p in prompts) == 1:
            raise EOFError
        return ""

    emb, _, _ = _build(
        YamConfig(rest_secs=0.1),
        driver=drv,
        operator=OperatorIO(input_fn=_input, output_fn=lambda _message: None),
    )
    scene = Scene(id="s", instruction="x")
    with pytest.raises(EmbodimentFault):
        emb.reset(scene)
    assert drv.commands == []  # the gate fault preceded any motion
    emb.reset(scene)
    assert sum("stand clear" in p for p in prompts) == 2  # retry re-confirmed
    assert drv.commands  # and only then homed


def test_close_then_reset_reprompts_stand_clear_per_connection() -> None:
    drv = EchoDriver()
    prompt_calls: list[tuple[str, int]] = []

    def _input(prompt: str) -> str:
        prompt_calls.append((prompt, len(drv.commands)))
        return ""

    emb, _, _ = _build(
        YamConfig(rest_secs=0.1),
        driver=drv,
        operator=OperatorIO(input_fn=_input, output_fn=lambda _message: None),
    )
    emb.reset(Scene(id="a", instruction="x"))
    emb.close()
    commands_after_park = len(drv.commands)
    emb.reset(Scene(id="b", instruction="x"))
    stand_clear_counts = [count for prompt, count in prompt_calls if "stand clear" in prompt]
    # One prompt per connection, each before that connection's first motion.
    assert stand_clear_counts == [0, commands_after_park]


def test_default_camera_reader_not_implemented() -> None:
    from inspect_robots_yam.embodiment import _default_camera_reader

    with pytest.raises(NotImplementedError, match="camera_reader"):
        _default_camera_reader(YamConfig())


def test_observe_parked_disabled_skips_driver_camera_and_ramp() -> None:
    class CountingDriver(EchoDriver):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        def get_joint_pos(self) -> np.ndarray:
            self.reads += 1
            return super().get_joint_pos()

    camera_calls: list[bool] = []

    def _recording_cameras(cfg: YamConfig):
        camera_calls.append(True)
        return _cameras(cfg)

    driver = CountingDriver()
    emb = YAMEmbodiment(
        YamConfig(
            cam_height=4,
            cam_width=4,
            park_before_grade=False,
            rest_secs=0.1,
        ),
        driver_factory=lambda _cfg: driver,
        camera_reader=_recording_cameras,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )
    emb.reset(Scene(id="s", instruction="x"))
    reads = driver.reads
    commands = len(driver.commands)
    camera_calls.clear()

    assert emb.observe_parked() is None
    assert driver.reads == reads
    assert len(driver.commands) == commands
    assert camera_calls == []


def test_observe_parked_declines_before_connect_or_pose_capture() -> None:
    emb, driver, _ = _build()
    assert emb.observe_parked() is None
    assert driver.commands == []

    class CaptureFault(FakeDriver):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        def get_joint_pos(self) -> np.ndarray:
            self.reads += 1
            raise RuntimeError("encoder read fault")

    faulty_driver = CaptureFault()
    emb, _, _ = _build(driver=faulty_driver)
    with pytest.raises(RuntimeError, match="encoder read fault"):
        emb.reset(Scene(id="s", instruction="x"))
    assert faulty_driver.reads == 1
    assert emb.observe_parked() is None
    assert faulty_driver.reads == 1


def test_observe_parked_ramps_settles_observes_and_drops_extra() -> None:
    events: list[str] = []

    class RecordingDriver(EchoDriver):
        def get_joint_pos(self) -> np.ndarray:
            events.append("read")
            return super().get_joint_pos()

        def command_joint_pos(self, target: np.ndarray) -> None:
            events.append("command")
            super().command_joint_pos(target)

    images = {
        name: np.full((4, 4, 3), fill, dtype=np.uint8)
        for name, fill in (("top_cam", 1), ("left_cam", 2), ("right_cam", 3))
    }

    def _recording_cameras(_cfg: YamConfig):
        events.append("camera")
        return images

    produced_extra = {"lazy_depth": lambda: np.ones((4, 4), dtype=np.float32)}

    def _recording_extra(_cfg: YamConfig):
        events.append("extra")
        return produced_extra

    status: list[str | None] = []
    driver = RecordingDriver(state=np.full(14, 0.2))
    emb = YAMEmbodiment(
        YamConfig(
            cam_height=4,
            cam_width=4,
            rest_pose=(0.6,) * 14,
            rest_secs=0.1,
            settle_tolerance=0.01,
            zero_gravity_mode=False,
        ),
        driver_factory=lambda _cfg: driver,
        camera_reader=_recording_cameras,
        depth_reader=_recording_extra,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        status_fn=status.append,
    )
    emb.reset(Scene(id="s", instruction="policy instruction"))
    events.clear()
    driver.commands.clear()
    status.clear()

    observation = emb.observe_parked()

    assert observation is not None
    assert events == ["read", "command", "read", "read", "camera", "extra"]
    assert len(driver.commands) == 1
    assert driver.commands[0] == pytest.approx(np.full(14, 0.6))
    assert observation.state["joint_pos"] == pytest.approx(np.full(14, 0.6))
    assert all(observation.images[name] is image for name, image in images.items())
    assert not observation.extra
    assert observation.extra is not produced_extra
    assert observation.instruction is None
    assert status == ["parking for grading: ramping arms clear", None]


def test_observe_parked_uses_captured_pose_and_is_silent_unattended() -> None:
    init_pose = np.full(14, 0.2)
    driver = EchoDriver(state=init_pose.copy())
    status: list[str | None] = []
    emb = YAMEmbodiment(
        YamConfig(
            cam_height=4,
            cam_width=4,
            rest_pose=None,
            rest_secs=0.1,
            unattended=True,
        ),
        driver_factory=lambda _cfg: driver,
        camera_reader=_cameras,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        status_fn=status.append,
    )
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    driver.commands.clear()

    observation = emb.observe_parked()

    assert observation is not None
    assert driver.commands[-1] == pytest.approx(init_pose)
    assert observation.state["joint_pos"] == pytest.approx(init_pose)
    assert status == []


def test_observe_parked_ramp_fault_propagates_and_closes_status() -> None:
    class FaultyDriver(EchoDriver):
        fail_commands = False

        def command_joint_pos(self, target: np.ndarray) -> None:
            if self.fail_commands:
                raise RuntimeError("grading park fault")
            super().command_joint_pos(target)

    driver = FaultyDriver()
    status: list[str | None] = []
    emb = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4, rest_secs=0.1),
        driver_factory=lambda _cfg: driver,
        camera_reader=_cameras,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        status_fn=status.append,
    )
    emb.reset(Scene(id="s", instruction="x"))
    status.clear()
    driver.fail_commands = True

    with pytest.raises(RuntimeError, match="grading park fault"):
        emb.observe_parked()

    assert status == ["parking for grading: ramping arms clear", None]


def test_close_ramps_to_rest_pose_then_releases() -> None:
    # Reset and close each issue 20 waypoints at 10 Hz.
    cfg = YamConfig(rest_pose=(0.5,) * 14, rest_secs=2.0)
    drv = EchoDriver()
    emb, _, sleeps = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert len(drv.commands) == 40
    park_commands = drv.commands[20:]
    assert park_commands[-1] == pytest.approx(np.full(14, 0.5))
    j0 = [c[0] for c in park_commands]
    assert all(b >= a for a, b in itertools.pairwise(j0))  # monotonic ramp, no jump
    assert park_commands[0][0] == pytest.approx(0.5 / 20)  # first step is 1/n of the way
    assert drv.closed is True
    assert sleeps[-1] == pytest.approx(0.1)  # paced at 1/control_hz


def test_close_rest_pose_goes_through_clamp_and_denorm() -> None:
    # Out-of-range joints clamp to +/-pi; gripper slots de-normalize like actions.
    cfg = YamConfig(
        rest_pose=(100.0,) * 6 + (0.3,) + (100.0,) * 6 + (0.3,),
        rest_secs=0.1,  # 1 waypoint
        gripper_open=10.0,
        gripper_closed=20.0,
    )
    emb, drv, _ = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(np.pi)
    assert cmd[6] == pytest.approx(17.0)  # 20 + 0.3 * (10 - 20)


def test_close_without_rest_pose_ramps_to_captured_init_pose() -> None:
    init_pose = np.full(14, 0.2)
    drv = EchoDriver(state=init_pose.copy())
    cfg = YamConfig.from_kwargs(rest_pose=None, rest_secs=0.3)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    command_count = len(drv.commands)
    emb.close()

    park_commands = drv.commands[command_count:]
    assert len(park_commands) > 1
    assert park_commands[-1] == pytest.approx(init_pose)
    j0 = [command[0] for command in park_commands]
    assert all(b <= a for a, b in itertools.pairwise(j0))
    assert drv.closed is True


def test_close_default_rest_pose_wins_over_captured_init_pose() -> None:
    init_pose = np.full(14, 0.2)
    drv = EchoDriver(state=init_pose.copy())
    emb, _, _ = _build(YamConfig(), driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    command_count = len(drv.commands)
    emb.close()

    park_commands = drv.commands[command_count:]
    assert len(park_commands) > 1
    assert park_commands[-1] == pytest.approx(DEFAULT_REST_POSE)
    assert park_commands[-1] != pytest.approx(init_pose)
    assert drv.closed is True


def test_close_explicit_rest_pose_wins_over_captured_init_pose() -> None:
    init_pose = np.full(14, 0.2)
    rest_pose = np.full(14, 0.6)
    drv = EchoDriver(state=init_pose.copy())
    emb, _, _ = _build(YamConfig(rest_pose=(0.6,) * 14, rest_secs=0.2), driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    emb.close()

    assert drv.commands[-1] == pytest.approx(rest_pose)
    assert drv.commands[-1] != pytest.approx(init_pose)
    assert drv.closed is True


def test_close_init_pose_grippers_round_trip_through_normalized_units() -> None:
    init_pose = np.full(14, 0.2)
    init_pose[6] = init_pose[13] = 17.0
    cfg = YamConfig(
        rest_pose=None,
        rest_secs=0.2,
        gripper_open=10.0,
        gripper_closed=20.0,
    )
    drv = EchoDriver(state=init_pose.copy())
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    emb.close()

    assert drv.commands[-1][6] == pytest.approx(17.0)
    assert drv.commands[-1][13] == pytest.approx(17.0)
    assert drv.closed is True


def test_close_parks_at_first_reset_pose_across_episodes() -> None:
    # Later resets start wherever the previous episode ended; parking must
    # return to where the operator left the arms when the run began.
    init_pose = np.full(14, 0.2)
    drv = EchoDriver(state=init_pose.copy())
    cfg = YamConfig(rest_pose=None, rest_secs=0.2)
    emb, _, _ = _build(cfg, driver=drv, operator=_operator())
    emb.reset(Scene(id="a", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    emb.reset(Scene(id="b", instruction="x"))  # starts at 0.8, must not re-capture
    emb.close()

    assert drv.commands[-1] == pytest.approx(init_pose)
    assert drv.closed is True


def test_close_parks_at_pre_home_pose_when_home_pose_configured() -> None:
    # The operator-left pose, not the raised home pose, is the park target:
    # torque is released after parking, so the target must be gravity-stable.
    operator_pose = np.full(14, 0.1)
    drv = EchoDriver(state=operator_pose.copy())
    cfg = YamConfig(rest_pose=None, home_pose=(0.5,) * 14, rest_secs=0.2)
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    emb.close()

    assert drv.commands[-1] == pytest.approx(operator_pose)
    assert drv.closed is True


@pytest.mark.parametrize(
    ("rest_pose", "expected_park"),
    [
        (None, np.full(14, 0.2)),
        (DEFAULT_REST_POSE, np.asarray(DEFAULT_REST_POSE)),
    ],
    ids=["opt-out-captured-init", "factory-default"],
)
def test_close_after_mid_reset_fault_parks(
    rest_pose: tuple[float, ...] | None, expected_park: np.ndarray
) -> None:
    def _camera_fault(_cfg: YamConfig) -> NoReturn:
        raise RuntimeError("camera open fault")

    init_pose = np.full(14, 0.2)
    drv = EchoDriver(state=init_pose.copy())
    cfg = YamConfig(rest_pose=rest_pose, home_pose=(0.6,) * 14, rest_secs=0.2)
    emb = YAMEmbodiment(
        cfg,
        driver_factory=lambda _cfg: drv,
        camera_reader=_camera_fault,
        operator=_operator(),
        poll_end=lambda: False,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    with pytest.raises(RuntimeError, match="camera open fault"):
        emb.reset(Scene(id="s", instruction="x"))
    command_count = len(drv.commands)
    emb.close()

    park_commands = drv.commands[command_count:]
    assert park_commands[-1] == pytest.approx(expected_park)
    assert drv.closed is True


def test_failed_driver_close_still_clears_connection_state() -> None:
    class FaultyClose(EchoDriver):
        fail = True

        def close(self) -> None:
            if self.fail:
                raise RuntimeError("CAN teardown fault")
            super().close()

    pose_a = np.full(14, 0.2)
    pose_b = np.full(14, 0.4)
    drv = FaultyClose(state=pose_a.copy())
    cfg = YamConfig(rest_pose=None, rest_secs=0.2)
    emb, _, _ = _build(cfg, driver=drv, operator=_operator())
    emb.reset(Scene(id="s", instruction="x"))
    with pytest.raises(RuntimeError, match="teardown"):
        emb.close()
    emb.close()  # connection state was cleared: the second close is a clean no-op
    with pytest.raises(RuntimeError, match="before reset"):
        emb.step(Action(data=np.zeros(14)))
    # The captured pose was cleared too: a reconnect re-captures at the new
    # pose, so the next park cannot ramp to the stale pre-fault target.
    drv.fail = False
    drv.state = pose_b.copy()
    emb.reset(Scene(id="s2", instruction="x"))
    emb.step(Action(data=np.full(14, 0.8)))
    emb.close()
    assert drv.commands[-1] == pytest.approx(pose_b)
    assert drv.closed is True


def test_reconnect_after_close_recaptures_init_pose() -> None:
    pose_a = np.full(14, 0.2)
    pose_b = np.full(14, 0.4)
    drv = EchoDriver(state=pose_a.copy())
    cfg = YamConfig(rest_pose=None, rest_secs=0.2)
    emb, _, _ = _build(cfg, driver=drv, operator=_operator())
    emb.reset(Scene(id="a", instruction="x"))
    emb.close()
    drv.state = pose_b.copy()
    drv.closed = False
    emb.reset(Scene(id="b", instruction="x"))  # fresh connection: capture anew
    emb.step(Action(data=np.full(14, 0.8)))
    emb.close()

    assert drv.commands[-1] == pytest.approx(pose_b)
    assert drv.closed is True


def test_close_before_connect_skips_rest_motion() -> None:
    emb, drv, _ = _build(YamConfig(rest_pose=(0.0,) * 14))
    emb.close()  # never connected: no motion, no close
    assert drv.commands == []
    assert drv.closed is False


@pytest.mark.parametrize(
    "cfg",
    [YamConfig(), YamConfig(rest_pose=(0.5,) * 14)],
    ids=["factory-default", "explicit-override"],
)
def test_close_connected_before_pose_capture_only_releases(cfg: YamConfig) -> None:
    class CaptureFault(FakeDriver):
        def get_joint_pos(self) -> np.ndarray:
            raise RuntimeError("encoder read fault")

    drv = CaptureFault()
    emb, _, _ = _build(cfg, driver=drv)
    with pytest.raises(RuntimeError, match="encoder read fault"):
        emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert drv.commands == []
    assert drv.closed is True


def test_close_rest_fault_still_releases_driver() -> None:
    class FaultyDriver(FakeDriver):
        fail_commands = False

        def command_joint_pos(self, target: np.ndarray) -> None:
            if self.fail_commands:
                raise RuntimeError("CAN fault")
            super().command_joint_pos(target)

    drv = FaultyDriver()
    emb, _, _ = _build(YamConfig(rest_pose=(0.0,) * 14), driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    drv.fail_commands = True
    with pytest.raises(RuntimeError, match="CAN fault"):
        emb.close()
    assert drv.closed is True  # handles released despite the fault
    emb.close()  # and close() stays idempotent afterwards


def test_close_rest_pose_zero_hz_falls_back_to_10hz() -> None:
    cfg = YamConfig(rest_pose=(0.1,) * 14, rest_secs=1.0, control_hz=0.0)
    emb, drv, _ = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert len(drv.commands) == 20  # reset and close each use the 10 Hz fallback


def _build_with_status(
    cfg: YamConfig | None = None,
    poll_end_seq: list[bool] | None = None,
    *,
    clock: _PacedClock | None = None,
):
    import dataclasses

    cfg = cfg or YamConfig()
    cfg = dataclasses.replace(cfg, cam_height=4, cam_width=4)
    drv = FakeDriver()
    polls = list(poll_end_seq or [False])
    status: list[str | None] = []
    clock = clock or _PacedClock()
    emb = YAMEmbodiment(
        cfg,
        driver_factory=lambda _c: drv,
        camera_reader=_cameras,
        operator=_operator(),
        poll_end=lambda: polls.pop(0) if polls else False,
        sleep_fn=clock.sleep,
        clock=clock,
        status_fn=status.append,
    )
    return emb, status


def test_reset_announces_run_instructions() -> None:
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(max_steps_hint=1200)
    emb, status = _build_with_status(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    assert len(status) == 3
    assert status[:2] == ["homing: ramping arms to start pose", None]
    msg = status[-1]
    assert msg is not None
    assert "any key" in msg and "grade" in msg  # how to end + how scoring works
    assert "120s" in msg  # horizon from max_steps_hint / control_hz


def test_status_line_updates_once_per_second_with_horizon() -> None:
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(max_steps_hint=1200)
    emb, status = _build_with_status(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)
    for _ in range(25):  # 2.5 s at 10 Hz
        emb.step(Action(data=np.zeros(14)))
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates == [
        "t = 1s / ~120s | wall 1s | any key ends the episode",
        "t = 2s / ~120s | wall 2s | any key ends the episode",
    ]


@pytest.mark.parametrize("connected", [False, True])
def test_ticker_gesture_prose_belongs_to_the_session_when_connected(connected: bool) -> None:
    emb, status = _build_with_status(YamConfig(control_hz=1.0))
    if connected:
        session = _RecordingSession()
        emb.connect_operator_session(session)
        status = session.statuses
    else:
        emb.defer_operator_end()

    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)
    emb.step(Action(data=np.zeros(14)))

    # Connected: rig state only, the session composes the end-gesture hint.
    # Defer-only: the session never sees our status, so we keep our own hint.
    expected = "t = 1s | wall 1s" if connected else "t = 1s | wall 1s | Esc ends the episode"
    assert status[reset_entries:] == [expected]


def test_status_line_without_hint_shows_elapsed_only() -> None:
    emb, status = _build_with_status()
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)
    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates and "1s" in updates[0] and "/" not in updates[0].split("|")[0]


def test_elapsed_follows_the_wall_clock_when_steps_overrun_the_period() -> None:
    # 10 Hz, but every step burns another period beyond the pace (a settle, or
    # a slow camera read). Counting steps would report 1s at step 10; the
    # operator has actually been standing there for 2s.
    clock = _PacedClock(overrun=0.1)
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(max_steps_hint=1200)
    emb, status = _build_with_status(cfg, clock=clock)
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)
    started = clock.now

    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))

    # 10 steps that a step-count counter would call 1s, and the clock agrees
    # they took 2s. The reported elapsed follows the clock.
    assert clock.now - started == pytest.approx(2.0)
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates == ["t = 1s / ~120s | wall 2s | any key ends the episode"]


def test_status_labels_large_wall_time_from_slow_policy_shape() -> None:
    clock = _PacedClock(overrun=198.8)
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(max_steps_hint=1200)
    emb, status = _build_with_status(cfg, clock=clock)
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)
    started = clock.now

    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))

    assert clock.now - started == pytest.approx(1989.0)
    assert status[reset_entries:] == ["t = 1s / ~120s | wall 1989s | any key ends the episode"]


@pytest.mark.parametrize("control_hz", [0.0, -1.0])
def test_status_motion_uses_fallback_when_control_hz_is_nonpositive(control_hz: float) -> None:
    clock = _PacedClock()
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(control_hz=control_hz, max_steps_hint=1200)
    emb, status = _build_with_status(cfg, clock=clock)
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)

    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))

    assert status[reset_entries:] == ["t = 1s | wall 0s | any key ends the episode"]


def test_status_without_horizon_uses_motion_and_labeled_wall_format() -> None:
    clock = _PacedClock(overrun=3.0)
    emb, _ = _build_with_status(YamConfig(control_hz=1.0), clock=clock)
    session = _RecordingSession()
    emb.connect_operator_session(session)
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(session.statuses)

    emb.step(Action(data=np.zeros(14)))

    assert session.statuses[reset_entries:] == ["t = 1s | wall 4s"]


def test_homing_time_is_not_charged_to_the_episode() -> None:
    # reset() ramps the arms home before handing over, and that ramp sleeps.
    # The operator's counter starts when the episode does, not at reset entry.
    clock = _PacedClock()
    emb, status = _build_with_status(YamConfig(control_hz=1.0), clock=clock)
    emb.reset(Scene(id="s", instruction="x"))
    homing_elapsed = clock.now
    reset_entries = len(status)

    emb.step(Action(data=np.zeros(14)))

    assert homing_elapsed > 0.0  # the ramp really did consume fake time
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates == ["t = 1s | wall 1s | any key ends the episode"]


def test_status_finishes_with_none_when_operator_ends_episode() -> None:
    emb, status = _build_with_status(poll_end_seq=[True])
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert status[-1] is None  # line closed before control returns for grading


def test_unattended_runs_emit_no_status() -> None:
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(unattended=True, max_steps_hint=100)
    emb, status = _build_with_status(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    for _ in range(15):
        emb.step(Action(data=np.zeros(14)))
    assert status == []


@dataclass(frozen=True)
class _Envelope:
    """Local stand-in for the core TaskEnvelope (the hook protocol is structural)."""

    name: str
    max_steps: int


def _running_status(status: list[str | None]) -> str:
    matches = [
        message for message in status if message is not None and message.startswith("Running:")
    ]
    assert len(matches) == 1
    return matches[0]


def test_deferred_status_explains_console_feedback_with_horizon() -> None:
    emb, status = _build_with_status()
    emb.bind_task(_Envelope(name="adhoc", max_steps=1200))
    emb.defer_operator_end()

    emb.reset(Scene(id="s", instruction="x"))

    assert _running_status(status) == (
        "Running: Esc (or /stop) ends the episode; type a message + Enter to "
        "send feedback. Max ~120s."
    )


def test_connected_banner_carries_rig_facts_only() -> None:
    emb, _ = _build_with_status()
    emb.bind_task(_Envelope(name="adhoc", max_steps=1200))
    session = _RecordingSession()
    emb.connect_operator_session(session)

    emb.reset(Scene(id="s", instruction="x"))

    # No console prose: the session owns the end gesture and knows per policy
    # whether typed messages are delivered, so yam claims neither.
    assert "Running. Max ~120s." in session.statuses
    assert not any(s is not None and "ends the episode" in s for s in session.statuses)


def test_bind_task_drives_the_countdown_horizon() -> None:
    emb, status = _build_with_status()
    emb.bind_task(_Envelope(name="adhoc", max_steps=1200))
    emb.reset(Scene(id="s", instruction="x"))
    assert "Max ~120s." in _running_status(status)
    reset_entries = len(status)
    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates and "1s / ~120s" in updates[0]


def test_bound_horizon_wins_over_deprecated_hint() -> None:
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(max_steps_hint=100)  # would show "Max ~10s."
    emb, status = _build_with_status(cfg)
    emb.bind_task(_Envelope(name="adhoc", max_steps=1200))
    emb.reset(Scene(id="s", instruction="x"))
    running = _running_status(status)
    assert "Max ~120s." in running
    assert "Max ~10s." not in running


def test_rebind_latest_envelope_wins() -> None:
    emb, status = _build_with_status()
    emb.bind_task(_Envelope(name="first", max_steps=100))
    emb.bind_task(_Envelope(name="second", max_steps=1200))
    emb.reset(Scene(id="s", instruction="x"))
    assert "Max ~120s." in _running_status(status)


def test_close_clears_the_bound_horizon() -> None:
    # close() before any reset: the clear must not depend on a connected driver,
    # and the next (framework-less) run must fall back, not show stale data.
    emb, status = _build_with_status()
    emb.bind_task(_Envelope(name="stale", max_steps=1200))
    emb.close()
    emb.reset(Scene(id="s", instruction="x"))
    assert "Max" not in _running_status(status)


def test_real_envelope_shape_satisfies_the_protocol() -> None:
    from inspect_robots_yam.embodiment import TaskEnvelopeLike

    assert isinstance(_Envelope(name="t", max_steps=1), TaskEnvelopeLike)


def test_camera_devices_select_the_builtin_opencv_reader() -> None:
    from inspect_robots_yam.embodiment import _default_camera_reader

    emb = YAMEmbodiment(
        YamConfig(
            top_cam_device="/dev/video0",
            left_cam_device="/dev/video2",
            right_cam_device="/dev/video4",
        )
    )
    # Construction stays inert (no cv2 import, no device open), but the
    # embodiment must have picked the builtin reader over the config-error stub.
    assert emb._camera_reader is not _default_camera_reader


def test_injected_camera_reader_suppresses_configured_opencv_builtin() -> None:
    emb = YAMEmbodiment(
        YamConfig(
            top_cam_device="/dev/video0",
            left_cam_device="/dev/video2",
            right_cam_device="/dev/video4",
        ),
        camera_reader=_cameras,
    )

    assert emb._camera_reader is _cameras
    assert emb._builtin_realsense_reader is None


def test_no_cameras_configured_keeps_fail_fast_reader_with_device_hint() -> None:
    emb, drv, _ = _build()
    emb._camera_reader = __import__(
        "inspect_robots_yam.embodiment", fromlist=["_default_camera_reader"]
    )._default_camera_reader
    with pytest.raises(
        ConfigError,
        match=r"\*_cam_device.*\*_depth_serial.*camera_reader",
    ):
        emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []  # fail-fast happened before any driver connect


def test_delta_mode_declares_joint_delta_and_per_step_box() -> None:
    import numpy as np

    cfg = YamConfig(joints_are_delta=True)
    emb, _, _ = _build(cfg)
    sem = emb.info.action_space.semantics
    assert sem is not None and sem.control_mode == "joint_delta"
    # The declared box is the per-step displacement limits, NOT the absolute
    # joint limits: symmetric, so the gripper can move in either direction.
    assert np.allclose(emb.info.action_space.low, cfg.delta_low)
    assert np.allclose(emb.info.action_space.high, cfg.delta_high)
    # The absolute-limit backstop still applies to the summed command in _send.


def test_absolute_mode_declares_joint_pos_with_labels() -> None:
    from inspect_robots_yam.packing import DIM_LABELS

    emb, _, _ = _build(YamConfig())
    sem = emb.info.action_space.semantics
    assert sem is not None and sem.control_mode == "joint_pos"
    assert sem.dim_labels == DIM_LABELS


@pytest.mark.parametrize(
    ("cfg", "gripper_indices"),
    [
        (YamConfig(), (6, 13)),
        (YamConfig(control_interface="eef_pos"), (6, 13)),
    ],
    ids=["joint-pos", "eef-abs-pose"],
)
def test_absolute_action_spaces_wire_derived_gripper_max_step(
    cfg: YamConfig, gripper_indices: tuple[int, int]
) -> None:
    emb, _, _ = _build(cfg)
    sem = emb.info.action_space.semantics
    assert sem is not None and sem.max_step is not None
    assert len(sem.max_step) == emb.info.action_space.dim
    assert all(
        step == pytest.approx(0.1) if index in gripper_indices else step is None
        for index, step in enumerate(sem.max_step)
    )


def test_delta_action_space_ignores_derived_gripper_max_step() -> None:
    emb, _, _ = _build(YamConfig(joints_are_delta=True))
    sem = emb.info.action_space.semantics
    assert sem is not None and sem.max_step is None


def test_observe_validates_camera_shape() -> None:
    def _bad_cameras(_cfg):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        return {"top_cam": img, "left_cam": img, "right_cam": img}

    emb, _, _ = _build()
    emb._camera_reader = _bad_cameras
    with pytest.raises(ValueError, match="camera 'top_cam' returned shape"):
        emb.reset(Scene(id="s", instruction="x"))


def test_observe_names_the_camera_that_dropped_a_frame() -> None:
    def _dropped_frame(cfg):
        img = np.zeros((cfg.cam_height, cfg.cam_width, 3), dtype=np.uint8)
        return {"top_cam": img, "left_cam": None, "right_cam": img}

    emb, _, _ = _build()
    emb._camera_reader = _dropped_frame
    with pytest.raises(ValueError, match="camera 'left_cam' returned no frame"):
        emb.reset(Scene(id="s", instruction="x"))
