"""Tier-2 validation against i2rt's bundled YAM + LINEAR_4310 model."""

from __future__ import annotations

import importlib.util
import itertools
import time

import numpy as np
import pytest

from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, YamConfig
from inspect_robots_yam.embodiment import _default_kinematics_factory
from inspect_robots_yam.kinematics import RawKinematics, _ArmKinematics

_RUNTIME_MODULES = ("i2rt", "mink", "mujoco")
_HAS_REAL_KINEMATICS = all(importlib.util.find_spec(name) is not None for name in _RUNTIME_MODULES)

pytestmark = [
    pytest.mark.real_kinematics,
    pytest.mark.skipif(
        not _HAS_REAL_KINEMATICS,
        reason="i2rt, mink, and mujoco are required for real-kinematics tests",
    ),
]


@pytest.fixture(scope="module")
def real_kinematics() -> tuple[YamConfig, RawKinematics, _ArmKinematics, np.ndarray]:
    """Build one real raw adapter and its plugin-owned safety wrapper."""
    cfg = YamConfig(control_interface="eef_pos")
    raw, _ = _default_kinematics_factory(cfg)
    wrapper = _ArmKinematics(
        side="left",
        raw=raw,
        config_low=cfg.low[:6],
        config_high=cfg.high[:6],
        ik_max_iters=cfg.ik_max_iters,
        ik_step_joint_limit=cfg.ik_step_joint_limit,
        cmd_resync_threshold=cfg.cmd_resync_threshold,
        osc_deadband=cfg.osc_deadband,
        osc_reversals=cfg.osc_reversals,
        osc_window=cfg.osc_window,
        osc_hold_steps=cfg.osc_hold_steps,
    )
    home = np.asarray(DEFAULT_EEF_HOME_POSE[:6])
    wrapper.capture_yaw_reference(home)
    return cfg, raw, wrapper, home


def _full_q(raw: RawKinematics, arm_q: np.ndarray) -> np.ndarray:
    ranges = np.asarray(raw.get_joint_ranges())
    pins = np.mean(ranges[6:], axis=1)
    return np.concatenate((arm_q, pins))


def _target(reference: np.ndarray, position: np.ndarray, relative_yaw: float) -> np.ndarray:
    cosine = np.cos(relative_yaw)
    sine = np.sin(relative_yaw)
    rotation_z = np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    target = np.eye(4)
    target[:3, :3] = rotation_z @ reference
    target[:3, 3] = position
    return target


def test_real_fk_ik_round_trip_position_and_relative_yaw(
    real_kinematics: tuple[YamConfig, RawKinematics, _ArmKinematics, np.ndarray],
) -> None:
    _, raw, wrapper, home = real_kinematics
    home_full = _full_q(raw, home)
    pose = np.asarray(raw.fk(home_full))
    assert pose[:3, 3] == pytest.approx((0.30, 0.0, 0.20), abs=0.005)
    assert pose[:3, 0] == pytest.approx((-0.5, 0.0, -np.sqrt(3.0) / 2.0), abs=0.01)
    # z-axis = x-axis x y-axis = (cos30, 0, -sin30): its z-component is -0.5.
    assert pose[2, 2] == pytest.approx(-0.5, abs=0.01)
    init = home_full.copy()
    init[:6] += 0.01
    _, solution = raw.ik(pose, init, 500)
    reconstructed = np.asarray(raw.fk(solution))
    assert np.linalg.norm(reconstructed[:3, 3] - pose[:3, 3]) < 0.001
    state = wrapper.observe(np.asarray(solution)[:6], gripper=1.0)
    assert abs(state[3]) < 0.01


def test_all_fifteen_default_workspace_probes_are_within_five_millimetres(
    real_kinematics: tuple[YamConfig, RawKinematics, _ArmKinematics, np.ndarray],
) -> None:
    cfg, raw, wrapper, home = real_kinematics
    reference = wrapper.fk(home)[:3, :3]
    low = cfg.eef_low_array[:3]
    high = cfg.eef_high_array[:3]
    middle = (low + high) / 2.0
    corners = [np.asarray(point) for point in itertools.product(*zip(low, high, strict=True))]
    faces = []
    for axis in range(3):
        for edge in (low[axis], high[axis]):
            point = middle.copy()
            point[axis] = edge
            faces.append(point)
    probes = (*corners, *faces, middle)
    assert len(probes) == 15

    init = _full_q(raw, home)
    for position in probes:
        radial_yaw = float(np.arctan2(position[1], position[0]))
        target = _target(reference, position, radial_yaw)
        _, solution = raw.ik(target, init, 500)
        achieved = np.asarray(raw.fk(solution))[:3, 3]
        assert np.linalg.norm(achieved - position) < 0.005, position


def test_real_warm_start_converges_along_centimetre_sequence(
    real_kinematics: tuple[YamConfig, RawKinematics, _ArmKinematics, np.ndarray],
) -> None:
    _, raw, wrapper, home = real_kinematics
    reference = wrapper.fk(home)[:3, :3]
    init = _full_q(raw, home)
    for x in np.arange(0.30, 0.36, 0.01):
        position = np.asarray((x, 0.0, 0.20))
        success, init = raw.ik(_target(reference, position, 0.0), init, 100)
        assert success
        achieved = np.asarray(raw.fk(init))[:3, 3]
        assert np.linalg.norm(achieved - position) < 0.001


@pytest.mark.perf
def test_saturated_nonconvergent_solve_stays_within_generous_budget(
    real_kinematics: tuple[YamConfig, RawKinematics, _ArmKinematics, np.ndarray],
) -> None:
    cfg, raw, wrapper, home = real_kinematics
    reference = wrapper.fk(home)[:3, :3]
    start = time.perf_counter()
    raw.ik(
        _target(reference, np.asarray((2.0, 0.0, 2.0)), 0.0), _full_q(raw, home), cfg.ik_max_iters
    )
    assert time.perf_counter() - start < 5.0


def test_real_signed_yaw_fk_is_positive_for_positive_command(
    real_kinematics: tuple[YamConfig, RawKinematics, _ArmKinematics, np.ndarray],
) -> None:
    _, raw, wrapper, home = real_kinematics
    home_pose = wrapper.fk(home)
    target = _target(home_pose[:3, :3], home_pose[:3, 3], 0.2)
    _, solution = raw.ik(target, _full_q(raw, home), 500)
    reported = wrapper.observe(np.asarray(solution)[:6], gripper=1.0)[3]
    assert reported == pytest.approx(0.2, abs=0.01)
