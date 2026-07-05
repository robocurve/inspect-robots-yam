"""Tests for YAMEmbodiment (all hardware/IO seams injected — no CAN, cameras, stdin)."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO


class FakeDriver:
    def __init__(self, state: np.ndarray | None = None) -> None:
        self.state = np.zeros(14) if state is None else state
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

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


def _operator(answers: list[str] | None = None) -> OperatorIO:
    seq = list(answers or [""])
    return OperatorIO(input_fn=lambda _p: seq.pop(0), output_fn=lambda _m: None)


def _build(
    cfg: YamConfig | None = None,
    *,
    driver: FakeDriver | None = None,
    poll_end_seq: list[bool] | None = None,
    operator: OperatorIO | None = None,
):
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
    cfg = YamConfig(home_pose=(0.1,) * 14, gripper_open=10.0, gripper_closed=20.0)
    emb, drv, _ = _build(cfg)
    obs = emb.reset(Scene(id="s", instruction="pour"))
    assert set(obs.images) == {"top_cam", "left_cam", "right_cam"}
    assert obs.state["joint_pos"].shape == (14,)
    assert obs.instruction == "pour"
    # The home pose is in policy units and goes through the same clamp+denorm
    # path as actions: joints pass through, gripper slots are de-normalized.
    assert len(drv.commands) == 1  # homing command issued
    home_cmd = drv.commands[0]
    assert home_cmd[0] == pytest.approx(0.1)
    assert home_cmd[6] == pytest.approx(11.0)  # 10 + 0.1 * (20 - 10)
    assert home_cmd[13] == pytest.approx(11.0)


def test_reset_without_home_pose_issues_no_command() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    assert drv.commands == []


def test_step_clamps_to_limits() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    # Way out of bounds; joints clip to +/-pi, gripper to [0,1].
    emb.step(Action(data=np.full(14, 100.0)))
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(np.pi)  # joint clamped
    # gripper slot clamped to 1.0 then de-normalized with default identity (0..1) -> 1.0
    assert cmd[6] == pytest.approx(1.0)


def test_step_gripper_denormalization() -> None:
    cfg = YamConfig(gripper_open=10.0, gripper_closed=20.0)
    emb, drv, _ = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(14)))  # normalized gripper 0 -> open value
    cmd = drv.commands[-1]
    assert cmd[6] == pytest.approx(10.0)
    assert cmd[13] == pytest.approx(10.0)
    emb.step(Action(data=np.concatenate([np.zeros(6), [1.0], np.zeros(6), [1.0]])))
    cmd = drv.commands[-1]
    assert cmd[6] == pytest.approx(20.0)  # normalized 1 -> closed value


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
    # Outgoing: normalized 0.3 de-normalizes to 10 + 0.3 * (20 - 10) = 13 hw units.
    assert drv.commands[-1][6] == pytest.approx(13.0)
    assert drv.commands[-1][13] == pytest.approx(13.0)
    # Incoming: the observed state re-normalizes 13 hw back to exactly 0.3.
    state = result.observation.state["joint_pos"]
    assert state[6] == pytest.approx(0.3)
    assert state[13] == pytest.approx(0.3)


def test_gripper_inverted_polarity_round_trip() -> None:
    cfg = YamConfig(gripper_open=20.0, gripper_closed=10.0)  # negative span
    drv = EchoDriver()
    emb, _, _ = _build(cfg, driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    action = np.zeros(14)
    action[6] = action[13] = 0.3
    result = emb.step(Action(data=action))
    # Outgoing: 20 + 0.3 * (10 - 20) = 17 hw units.
    assert drv.commands[-1][6] == pytest.approx(17.0)
    assert drv.commands[-1][13] == pytest.approx(17.0)
    # Incoming: (17 - 20) / (10 - 20) = 0.3 — bijection holds under inversion.
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
    # +0.1 -> 0.6, de-normalized back out to 16 hw (NOT 15.1 or denorm(0.51)).
    assert cmd[6] == pytest.approx(16.0)
    assert cmd[13] == pytest.approx(16.0)


def test_reset_twice_reuses_driver() -> None:
    calls = {"n": 0}

    def _factory(_c):
        calls["n"] += 1
        return FakeDriver()

    emb = YAMEmbodiment(
        YamConfig(),
        driver_factory=_factory,
        camera_reader=_cameras,
        operator=_operator(["", ""]),
        poll_end=lambda: False,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )
    emb.reset(Scene(id="s", instruction="x"))
    emb.reset(Scene(id="s", instruction="x"))
    assert calls["n"] == 1  # driver built once, reused on the second reset


def test_step_terminates_success_on_operator_yes() -> None:
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(["", "y"]))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert result.termination_reason == "success"
    assert result.info["operator_confirmed"] is True


def test_step_terminates_failure_on_operator_no() -> None:
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(["", "n"]))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert result.termination_reason == "failure"


def test_step_continues_when_no_end_signal() -> None:
    emb, _, _ = _build(poll_end_seq=[False])
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is False
    assert emb.num_steps == 1


def test_pacing_sleeps_to_control_rate() -> None:
    emb, _, sleeps = _build()  # control_hz=10 -> period 0.1, clock constant 0 -> sleep ~0.1
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(14)))
    assert sleeps and sleeps[-1] == pytest.approx(0.1)


def test_pacing_skipped_when_hz_zero() -> None:
    cfg = YamConfig(control_hz=0.0)
    emb, _, sleeps = _build(cfg)
    emb.reset(Scene(id="s", instruction="x"))
    emb.step(Action(data=np.zeros(14)))
    assert sleeps == []  # no sleep attempted at hz=0


def test_close_idempotent_and_releases() -> None:
    emb, drv, _ = _build()
    emb.close()  # before connect: no error
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    assert drv.closed is True
    emb.close()  # second close: no error


def test_default_camera_reader_not_implemented() -> None:
    from inspect_robots_yam.embodiment import _default_camera_reader

    with pytest.raises(NotImplementedError, match="camera_reader"):
        _default_camera_reader(YamConfig())
