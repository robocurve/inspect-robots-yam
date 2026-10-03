"""Post-IK protection at the EEF motor-command boundary; no hardware access."""

from types import SimpleNamespace

import numpy as np
import pytest
from inspect_robots.errors import SafetyAbort
from inspect_robots.types import Action

from inspect_robots_yam.collision import CollisionChecker, CollisionConfig, CollisionReport
from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, DEFAULT_JOINT_HOME_POSE, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment


class Driver:
    def __init__(self, state):
        self.state = np.array(state, dtype=float)
        self.commands = []

    def get_joint_pos(self):
        return self.state.copy()

    def command_joint_pos(self, command):
        self.commands.append(command.copy())


def rig():
    cfg = YamConfig(control_interface="eef_pos", collision_table_height=-0.1)
    emb = YAMEmbodiment(cfg)
    drv = Driver(DEFAULT_JOINT_HOME_POSE)
    emb._driver = drv
    return emb, drv


def test_safe_eef_home_and_reverse_sweeps():
    checker = CollisionChecker(CollisionConfig(table_height=-0.1))
    checker.check_motion(DEFAULT_JOINT_HOME_POSE, DEFAULT_EEF_HOME_POSE)
    checker.check_motion(DEFAULT_EEF_HOME_POSE, DEFAULT_JOINT_HOME_POSE)


def test_real_table_collision_never_reaches_driver():
    emb, drv = rig()
    target = np.array(DEFAULT_JOINT_HOME_POSE)
    target[[1, 8]] = 1.8
    target[[2, 9]] = 0.3
    with pytest.raises(SafetyAbort, match="blocked"):
        emb._send(target)
    assert not drv.commands


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_command_never_reaches_driver(value):
    emb, drv = rig()
    target = np.array(DEFAULT_JOINT_HOME_POSE)
    target[0] = value
    with pytest.raises(SafetyAbort, match="non-finite"):
        emb._send(target)
    assert not drv.commands


def test_measurement_not_last_command_is_sweep_start():
    emb, drv = rig()
    calls = []
    emb._eef_collision_checker = SimpleNamespace(
        check_motion=lambda a, b: calls.append((a.copy(), b.copy()))
    )
    target = np.array(DEFAULT_JOINT_HOME_POSE)
    emb._send(target)
    drv.state[0] = 0.2
    emb._send(target)
    assert calls[-1][0][0] == 0.2
    assert calls[-1][1][0] == 0


def test_intermediate_collision_rejected_with_safe_endpoints(monkeypatch):
    checker = CollisionChecker()

    def report(q):
        return CollisionReport(collided=0.4 < q[0] < 0.6, geom1="left", geom2="right")

    monkeypatch.setattr(checker, "check", report)
    target = np.array(DEFAULT_JOINT_HOME_POSE)
    target[0] = 1
    with pytest.raises(SafetyAbort, match="left:right"):
        checker.check_motion(DEFAULT_JOINT_HOME_POSE, target)


def test_current_collision_and_invalid_measurement_rejected():
    emb, drv = rig()
    drv.state[[1, 8]] = 1.8
    drv.state[[2, 9]] = 0.3
    with pytest.raises(SafetyAbort):
        emb._send(np.array(DEFAULT_JOINT_HOME_POSE))
    drv.state[0] = np.nan
    with pytest.raises(SafetyAbort, match="non-finite"):
        emb._send(np.array(DEFAULT_JOINT_HOME_POSE))
    assert not drv.commands


def test_post_ik_unsafe_command_rejected():
    emb, drv = rig()
    unsafe = np.array([0, 1.8, 0.3, 0, 0, 0])
    kin = SimpleNamespace(solve=lambda *args: unsafe, update_sent=lambda q: None)
    emb._left_kinematics = kin
    emb._right_kinematics = kin
    with pytest.raises(SafetyAbort):
        emb._step_eef(np.zeros(14), drv)
    assert not drv.commands


def test_missing_checker_dependency_fails_closed(monkeypatch):
    from inspect_robots_yam import collision

    emb, drv = rig()

    def fail(*args):
        raise RuntimeError("MuJoCo unavailable")

    monkeypatch.setattr(collision, "CollisionChecker", fail)
    with pytest.raises(RuntimeError, match="MuJoCo unavailable"):
        emb._send(np.array(DEFAULT_JOINT_HOME_POSE))
    assert not drv.commands


def test_real_ik_small_cartesian_moves_and_home_park():
    from inspect_robots.scene import Scene

    from inspect_robots_yam.embodiment import _default_kinematics_factory
    from test_eef_embodiment import _build

    cfg = YamConfig(
        control_interface="eef_pos",
        eef_orientation=True,
        collision_table_height=-0.1,
        zero_gravity_mode=False,
        unattended=True,
        rest_secs=0.5,
    )
    emb, drv, _, _ = _build(cfg)
    emb._kinematics_factory = _default_kinematics_factory
    observation = emb.reset(Scene(id="eef-smoke", instruction="small safe motion"))
    target = observation.state["eef_state"].copy()
    target[[0, 7]] += 0.02
    target[[2, 9]] += 0.02
    target[[3, 10]] += 0.1
    target[[4, 11]] += 0.05
    target[[5, 12]] += 0.05
    for _ in range(30):
        result = emb.step(Action(target))
    assert np.max(np.abs(result.observation.state["eef_state"] - target)) < 0.015
    assert len(drv.commands) >= 30
    emb.close()
    assert drv.closed
