"""Tests for YAMEmbodiment (all hardware/IO seams injected — no CAN, cameras, stdin)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import pytest
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.errors import ConfigError, EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_yam.config import (
    DEFAULT_JOINT_HOME_POSE,
    DEFAULT_REST_POSE,
    YamConfig,
)
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


def _operator(answers: list[str] | None = None, *, prompts: list[str] | None = None) -> OperatorIO:
    seq = list(answers or [])

    def _input(prompt: str) -> str:
        if prompts is not None:
            prompts.append(prompt)
        if "succeed" not in prompt:
            return ""
        return seq.pop(0)

    return OperatorIO(input_fn=_input, output_fn=lambda _m: None)


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


def test_reset_without_home_pose_ramps_to_current_pose() -> None:
    state = np.zeros(14)
    state[0] = 0.5
    state[[6, 13]] = 20.0
    cfg = YamConfig(rest_secs=0.1, gripper_open=10.0, gripper_closed=20.0)
    emb, drv, _ = _build(cfg, driver=FakeDriver(state=state))
    emb.reset(Scene(id="s", instruction="x"))
    expected = state.copy()
    expected[[6, 13]] = cfg.gripper_open
    assert drv.commands[-1] == pytest.approx(expected)




def test_step_clamps_to_limits() -> None:
    emb, drv, _ = _build()
    emb.reset(Scene(id="s", instruction="x"))
    # Way out of bounds; joints clip to step limit (0.2), gripper to [0,1].
    emb.step(Action(data=np.full(14, 100.0)))
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(0.2)  # joint clamped to step limit
    # gripper slot clamped to 1.0 then de-normalized with default identity (0..1) -> 1.0
    assert cmd[6] == pytest.approx(1.0)


def test_step_gripper_denormalization() -> None:
    cfg = YamConfig(gripper_open=10.0, gripper_closed=20.0)
    state = np.zeros(14)
    state[6] = 10.0; state[13] = 10.0
    emb, drv, _ = _build(cfg, driver=EchoDriver(state=state))
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
        YamConfig(),
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


def test_step_terminates_success_on_operator_yes() -> None:
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(["y"]))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert result.termination_reason == "success"
    assert result.info["operator_confirmed"] is True


def test_step_terminates_failure_on_operator_no() -> None:
    prompts: list[str] = []
    emb, _, _ = _build(poll_end_seq=[True], operator=_operator(["n"], prompts=prompts))
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert result.termination_reason == "failure"
    assert result.info["operator_confirmed"] is False
    assert sum("succeed" in prompt for prompt in prompts) == 1


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
    with pytest.raises(ConfigError, match="cam_device"):
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
    assert prompts == []  # neither wait_ready nor confirm_success ran
    assert result.terminated is False  # the end poll is skipped entirely


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
    # Out-of-range joints clamp to step limits (0.2); gripper slots de-normalize like actions.
    cfg = YamConfig(
        rest_pose=(100.0,) * 6 + (0.5,) + (100.0,) * 6 + (0.5,),
        rest_secs=0.1,  # 1 waypoint
        gripper_open=10.0,
        gripper_closed=20.0,
    )
    state = np.zeros(14)
    state[6] = 10.0; state[13] = 10.0
    emb, drv, _ = _build(cfg, driver=EchoDriver(state=state))
    emb.reset(Scene(id="s", instruction="x"))
    emb.close()
    cmd = drv.commands[-1]
    assert cmd[0] == pytest.approx(0.2)  # step limit from 0
    assert cmd[6] == pytest.approx(15.0)  # 10 + 0.5 * (20 - 10)


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
    cfg = YamConfig(rest_pose=None, home_pose=(0.5,) * 14, rest_secs=0.4)
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
    cfg = YamConfig(rest_pose=rest_pose, home_pose=(0.6,) * 14, rest_secs=0.4)
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


def _build_with_status(cfg: YamConfig | None = None, poll_end_seq: list[bool] | None = None):
    drv = FakeDriver()
    polls = list(poll_end_seq or [False])
    status: list[str | None] = []
    emb = YAMEmbodiment(
        cfg or YamConfig(),
        driver_factory=lambda _c: drv,
        camera_reader=_cameras,
        operator=_operator(["y"]),
        poll_end=lambda: polls.pop(0) if polls else False,
        sleep_fn=lambda _s: None,
        clock=lambda: 0.0,
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
    assert "any key" in msg and "y/N" in msg  # how to end + how scoring works
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
    assert len(updates) == 2  # at steps 10 and 20
    assert "1s / 120s" in updates[0]
    assert "2s / 120s" in updates[1]
    assert "any key" in updates[0]  # instructions ride along


def test_status_line_without_hint_shows_elapsed_only() -> None:
    emb, status = _build_with_status()
    emb.reset(Scene(id="s", instruction="x"))
    reset_entries = len(status)
    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates and "1s" in updates[0] and "/" not in updates[0].split("|")[0]


def test_status_finishes_with_none_when_operator_ends_episode() -> None:
    emb, status = _build_with_status(poll_end_seq=[True])
    emb.reset(Scene(id="s", instruction="x"))
    result = emb.step(Action(data=np.zeros(14)))
    assert result.terminated is True
    assert status[-1] is None  # line closed before the y/N prompt


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


def test_bind_task_drives_the_countdown_horizon() -> None:
    emb, status = _build_with_status()
    emb.bind_task(_Envelope(name="adhoc", max_steps=1200))
    emb.reset(Scene(id="s", instruction="x"))
    assert "Max 120s." in _running_status(status)
    reset_entries = len(status)
    for _ in range(10):
        emb.step(Action(data=np.zeros(14)))
    updates = [m for m in status[reset_entries:] if m is not None]
    assert updates and "1s / 120s" in updates[0]


def test_bound_horizon_wins_over_deprecated_hint() -> None:
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        cfg = YamConfig(max_steps_hint=100)  # would show "Max 10s."
    emb, status = _build_with_status(cfg)
    emb.bind_task(_Envelope(name="adhoc", max_steps=1200))
    emb.reset(Scene(id="s", instruction="x"))
    running = _running_status(status)
    assert "Max 120s." in running
    assert "Max 10s." not in running


def test_rebind_latest_envelope_wins() -> None:
    emb, status = _build_with_status()
    emb.bind_task(_Envelope(name="first", max_steps=100))
    emb.bind_task(_Envelope(name="second", max_steps=1200))
    emb.reset(Scene(id="s", instruction="x"))
    assert "Max 120s." in _running_status(status)


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


def test_no_cameras_configured_keeps_fail_fast_reader_with_device_hint() -> None:
    emb, drv, _ = _build()
    emb._camera_reader = __import__(
        "inspect_robots_yam.embodiment", fromlist=["_default_camera_reader"]
    )._default_camera_reader
    with pytest.raises(ConfigError, match="cam_device"):
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


def test_step_tracks_far_target_at_step_limits() -> None:
    # EchoDriver means driver.get_joint_pos() follows exactly what was last commanded.
    drv = EchoDriver()
    emb, _, _ = _build(driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    drv.commands.clear()
    
    # Try to jump to 1.0 (step limit is 0.2)
    emb.step(Action(data=np.full(14, 1.0)))
    cmd1 = drv.commands[-1]
    assert cmd1[0] == pytest.approx(0.2)
    
    # Try again, it should walk another 0.2 to 0.4
    emb.step(Action(data=np.full(14, 1.0)))
    cmd2 = drv.commands[-1]
    assert cmd2[0] == pytest.approx(0.4)


def test_reset_zero_g_first_command_equals_current_pose() -> None:
    # If the arm is sitting at 0.7 in zero-g, reset should send 0.7 as the first command.
    drv = FakeDriver(state=np.full(14, 0.7))
    emb, _, _ = _build(YamConfig(home_pose=None), driver=drv)
    emb.reset(Scene(id="s", instruction="x"))
    assert len(drv.commands) > 0
    # The homing ramp should have sent 0.7 across the board (clamped properly)
    assert drv.commands[0][0] == pytest.approx(0.7)
    assert drv.commands[-1][0] == pytest.approx(0.7)
    # Grippers should be set to 1.0 (fully open)
    assert drv.commands[-1][6] == pytest.approx(1.0)
    assert drv.commands[-1][13] == pytest.approx(1.0)
