"""Tier-1 EEF embodiment tests with injected driver, cameras, and raw kinematics."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.errors import EmbodimentFault
from inspect_robots.scene import Scene
from inspect_robots.types import Action

import inspect_robots_yam.embodiment as embodiment_module
from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, EEF_DIM_LABELS, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment, _default_kinematics_factory
from inspect_robots_yam.operator import OperatorIO


def _pose(position: Sequence[float] = (0.3, 0.0, 0.2)) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = position
    return value


class FakeRawKinematics:
    """Raw seam that records full-model FK/IK vectors."""

    def __init__(
        self,
        *,
        pose: np.ndarray | None = None,
        solutions: Sequence[np.ndarray] | None = None,
    ) -> None:
        self.ranges = np.asarray([[-2.0, 2.0]] * 6 + [[0.0, 0.04], [0.0, 0.04]])
        self.pose = _pose() if pose is None else pose
        self.solutions = list(solutions or [np.zeros(8)])
        self.fk_inputs: list[np.ndarray] = []
        self.ik_calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def get_joint_ranges(self) -> np.ndarray:
        return self.ranges.copy()

    def set_joint_ranges(self, ranges: np.ndarray) -> None:
        self.ranges = np.asarray(ranges).copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.fk_inputs.append(q.copy())
        return self.pose.copy()

    def ik(self, target: np.ndarray, init_q: np.ndarray, max_iters: int) -> tuple[bool, np.ndarray]:
        self.ik_calls.append((target.copy(), init_q.copy(), max_iters))
        index = min(len(self.ik_calls) - 1, len(self.solutions) - 1)
        return True, self.solutions[index].copy()


class EchoDriver:
    """Driver that echoes arm commands into measured state."""

    def __init__(self, state: np.ndarray | None = None, *, echo_grippers: bool = True) -> None:
        self.state = np.zeros(14) if state is None else state.copy()
        self.echo_grippers = echo_grippers
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        command = np.asarray(target, dtype=float).copy()
        self.commands.append(command)
        grippers = self.state[[6, 13]].copy()
        self.state = command.copy()
        if not self.echo_grippers:
            self.state[[6, 13]] = grippers

    def close(self) -> None:
        self.closed = True


def _cameras(_cfg: YamConfig):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": image, "left_cam": image, "right_cam": image}


def _build(
    cfg: YamConfig | None = None,
    *,
    driver: EchoDriver | None = None,
    left: FakeRawKinematics | None = None,
    right: FakeRawKinematics | None = None,
) -> tuple[YAMEmbodiment, EchoDriver, FakeRawKinematics, FakeRawKinematics]:
    config = cfg or YamConfig(control_interface="eef_pos", rest_secs=0.1, unattended=True)
    drv = driver or EchoDriver()
    left_raw = left or FakeRawKinematics()
    right_raw = right or FakeRawKinematics()
    emb = YAMEmbodiment(
        config,
        driver_factory=lambda _cfg: drv,
        kinematics_factory=lambda _cfg: (left_raw, right_raw),
        camera_reader=_cameras,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )
    return emb, drv, left_raw, right_raw


def test_eef_info_uses_10d_space_and_dual_state_observation() -> None:
    emb, _, _, _ = _build()
    assert emb.info.action_space.shape == (10,)
    assert emb.info.action_space.semantics.dim_labels == EEF_DIM_LABELS
    assert emb.info.observation_space.state_keys == frozenset({"joint_pos", "eef_state"})


def test_default_eef_home_is_mandatory_when_home_pose_is_none() -> None:
    emb, driver, _, _ = _build()
    observation = emb.reset(Scene(id="eef", instruction="move"))
    assert driver.commands[-1] == pytest.approx(DEFAULT_EEF_HOME_POSE)
    assert observation.state["joint_pos"] == pytest.approx(DEFAULT_EEF_HOME_POSE)
    assert observation.state["eef_state"] == pytest.approx(
        (0.3, 0.0, 0.2, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0, 1.0)
    )


def test_eef_step_commands_both_arms_and_passes_grippers_outside_ik() -> None:
    home = np.asarray(DEFAULT_EEF_HOME_POSE)
    left_solution = np.concatenate((home[:6] + 0.05, (0.0, 0.0)))
    right_solution = np.concatenate((home[7:13] - 0.05, (0.0, 0.0)))
    left = FakeRawKinematics(solutions=[left_solution])
    right = FakeRawKinematics(solutions=[right_solution])
    emb, driver, _, _ = _build(left=left, right=right)
    emb.reset(Scene(id="eef", instruction="move"))

    action = np.asarray((0.35, 0.1, 0.25, 0.2, 0.25, 0.4, -0.1, 0.3, -0.2, 0.75))
    emb.step(Action(data=action))
    command = driver.commands[-1]
    assert command[:6] == pytest.approx(left_solution[:6])
    assert command[7:13] == pytest.approx(right_solution[:6])
    assert command[6] == pytest.approx(0.25)
    assert command[13] == pytest.approx(0.75)
    assert left.ik_calls[-1][1].shape == (8,)
    assert right.ik_calls[-1][1].shape == (8,)


def test_nonfinite_solution_resends_previous_arm_command_but_updates_gripper() -> None:
    solution = np.zeros(8)
    solution[0] = np.nan
    left = FakeRawKinematics(solutions=[solution])
    emb, driver, _, _ = _build(left=left)
    emb.reset(Scene(id="eef", instruction="move"))
    prior = driver.commands[-1].copy()
    action = np.asarray((0.3, 0.0, 0.2, 0.0, 0.1, 0.3, 0.0, 0.2, 0.0, 0.9))
    emb.step(Action(data=action))
    assert np.all(np.isfinite(driver.commands[-1]))
    assert driver.commands[-1][:6] == pytest.approx(prior[:6])
    assert driver.commands[-1][6] == pytest.approx(0.1)


def test_home_fk_must_start_inside_workspace_box() -> None:
    left = FakeRawKinematics(pose=_pose((0.1, 0.0, 0.2)))
    emb, driver, _, _ = _build(left=left)
    with pytest.raises(ValueError, match=r"left EEF home state.*workspace"):
        emb.reset(Scene(id="eef", instruction="move"))
    assert driver.commands == []


def _build_attended(
    left: FakeRawKinematics | None = None,
) -> tuple[YAMEmbodiment, EchoDriver, list[tuple[str, int]]]:
    """Attended EEF embodiment whose prompts are recorded with command counts."""
    drv = EchoDriver()
    prompt_calls: list[tuple[str, int]] = []

    def _input(prompt: str) -> str:
        prompt_calls.append((prompt, len(drv.commands)))
        return ""

    emb = YAMEmbodiment(
        YamConfig(control_interface="eef_pos", rest_secs=0.1),
        driver_factory=lambda _cfg: drv,
        kinematics_factory=lambda _cfg: (left or FakeRawKinematics(), FakeRawKinematics()),
        camera_reader=_cameras,
        operator=OperatorIO(input_fn=_input, output_fn=lambda _message: None),
        poll_end=lambda: False,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
        status_fn=lambda _message: None,
    )
    return emb, drv, prompt_calls


def test_attended_eef_reset_gates_home_motion() -> None:
    emb, drv, prompt_calls = _build_attended()
    emb.reset(Scene(id="eef", instruction="move"))
    stand_clear = [call for call in prompt_calls if "stand clear" in call[0]]
    assert stand_clear == [
        ("Arms will move to the home pose - stand clear, then press Enter...", 0)
    ]
    assert drv.commands  # homing ramp ran only after the gate


def test_home_fk_failure_raises_before_stand_clear_prompt() -> None:
    emb, drv, prompt_calls = _build_attended(left=FakeRawKinematics(pose=_pose((0.1, 0.0, 0.2))))
    with pytest.raises(ValueError, match=r"left EEF home state.*workspace"):
        emb.reset(Scene(id="eef", instruction="move"))
    assert prompt_calls == []  # configuration errors fail fast, before any prompt
    assert drv.commands == []


@pytest.mark.parametrize(
    "config",
    [
        YamConfig(
            control_interface="eef_pos",
            rest_secs=0.1,
            unattended=True,
            eef_low=(0.15, -0.25, 0.03, 0.1, 0.0) * 2,
        ),
        YamConfig(
            control_interface="eef_pos",
            rest_secs=0.1,
            unattended=True,
            eef_high=(0.48, 0.25, 0.40, np.pi, 0.9) * 2,
        ),
    ],
)
def test_home_relative_yaw_and_gripper_must_also_start_inside_action_box(
    config: YamConfig,
) -> None:
    emb, driver, _, _ = _build(config)
    with pytest.raises(ValueError, match=r"left EEF home state.*workspace"):
        emb.reset(Scene(id="eef", instruction="move"))
    assert driver.commands == []


def test_configured_joint_home_and_parking_remain_joint_space_mechanisms() -> None:
    home = tuple([0.1] * 6 + [1.0] + [0.1] * 6 + [1.0])
    rest = tuple([0.2] * 6 + [0.0] + [0.2] * 6 + [0.0])
    cfg = YamConfig(
        control_interface="eef_pos",
        home_pose=home,
        rest_pose=rest,
        rest_secs=0.1,
        unattended=True,
    )
    emb, driver, _, _ = _build(cfg)
    emb.reset(Scene(id="eef", instruction="move"))
    assert driver.commands[-1] == pytest.approx(home)
    emb.close()
    assert driver.commands[-1] == pytest.approx(rest)
    assert driver.closed is True


def test_yaw_references_are_captured_from_post_homing_measurement() -> None:
    emb, _, left, right = _build()
    emb.reset(Scene(id="eef", instruction="move"))
    assert left.fk_inputs[-1][:6] == pytest.approx(DEFAULT_EEF_HOME_POSE[:6])
    assert right.fk_inputs[-1][:6] == pytest.approx(DEFAULT_EEF_HOME_POSE[7:13])


def test_reset_clears_active_hold_before_next_trial_opening_step() -> None:
    home = np.asarray(DEFAULT_EEF_HOME_POSE)
    alternating = [
        np.concatenate((home[:6] + value, (0.0, 0.0))) for value in (1.0, -1.0, 1.0, -1.0, 0.1)
    ]
    left = FakeRawKinematics(solutions=alternating)
    cfg = YamConfig(
        control_interface="eef_pos",
        rest_secs=0.1,
        unattended=True,
        osc_hold_steps=3,
    )
    emb, _, _, _ = _build(cfg, left=left)
    scene = Scene(id="eef", instruction="move")
    emb.reset(scene)
    action = Action(data=np.asarray((0.3, 0.0, 0.2, 0.0, 1.0) * 2))
    for _ in range(4):
        emb.step(action)
    assert len(left.ik_calls) == 4
    emb.reset(scene)
    assert emb._left_kinematics is not None
    assert emb._left_kinematics.hold_counter == 0
    assert emb._left_kinematics.reversal_window == ()
    assert emb._left_kinematics.resynced is False
    emb.step(action)
    assert len(left.ik_calls) == 5


def test_gripper_measurement_gap_never_resyncs_or_disables_arm_reversal_counting() -> None:
    home = np.asarray(DEFAULT_EEF_HOME_POSE)
    alternating = [
        np.concatenate((home[:6] + value, (0.0, 0.0))) for value in (1.0, -1.0, 1.0, -1.0)
    ]
    left = FakeRawKinematics(solutions=alternating)
    driver = EchoDriver(echo_grippers=False)
    emb, _, _, _ = _build(driver=driver, left=left)
    emb.reset(Scene(id="eef", instruction="move"))
    action = Action(data=np.asarray((0.3, 0.0, 0.2, 0.0, 1.0) * 2))
    for _ in range(4):
        emb.step(action)
    assert len(left.ik_calls) == 4
    assert emb._left_kinematics is not None
    assert emb._left_kinematics.hold_counter == 10
    assert emb._left_kinematics.resynced is False


def test_eef_action_requires_exactly_ten_dimensions() -> None:
    emb, _, _, _ = _build()
    emb.reset(Scene(id="eef", instruction="move"))
    with pytest.raises(ValueError, match="expected a 10-D vector"):
        emb.step(Action(data=np.zeros(14)))


def test_eef_kinematics_are_unavailable_before_first_reset() -> None:
    emb, _, _, _ = _build()
    with pytest.raises(RuntimeError, match="unavailable before reset"):
        emb._require_kinematics()


def test_injected_kinematics_factory_is_lazy_and_reused_across_resets() -> None:
    driver = EchoDriver()
    left = FakeRawKinematics()
    right = FakeRawKinematics()
    calls = 0

    def factory(_cfg: YamConfig):
        nonlocal calls
        calls += 1
        return left, right

    emb = YAMEmbodiment(
        YamConfig(control_interface="eef_pos", rest_secs=0.1, unattended=True),
        driver_factory=lambda _cfg: driver,
        kinematics_factory=factory,
        camera_reader=_cameras,
        sleep_fn=lambda _seconds: None,
        clock=lambda: 0.0,
    )
    assert calls == 0
    scene = Scene(id="eef", instruction="move")
    emb.reset(scene)
    emb.reset(scene)
    assert calls == 1


def test_default_factory_binds_grasp_site_and_translates_real_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[Any] = []
    combined: list[tuple[object, object]] = []

    class NoSolutionFound(Exception):
        pass

    class Solver:
        def __init__(self, path: str, site_name: str) -> None:
            self.path = path
            self.site_name = site_name
            self._configuration = SimpleNamespace(
                model=SimpleNamespace(jnt_range=np.asarray([[-2.0, 2.0]] * 8))
            )
            self.raise_infeasible = False
            self.ik_site: str | None = None
            created.append(self)

        def fk(self, q: np.ndarray) -> np.ndarray:
            return _pose()

        def ik(
            self,
            target: np.ndarray,
            site_name: str,
            *,
            init_q: np.ndarray,
            max_iters: int,
        ) -> tuple[bool, np.ndarray]:
            self.ik_site = site_name
            if self.raise_infeasible:
                raise NoSolutionFound
            return True, init_q

    arm_type = SimpleNamespace(YAM=object())
    gripper = object()
    gripper_type = {"LINEAR_4310": gripper}

    def combine(arm: object, selected_gripper: object) -> str:
        combined.append((arm, selected_gripper))
        return "combined.xml"

    monkeypatch.setattr(
        embodiment_module,
        "_load_i2rt_kinematics",
        lambda: (Solver, arm_type, gripper_type, combine, NoSolutionFound),
    )
    left, right = _default_kinematics_factory(YamConfig(control_interface="eef_pos"))
    assert combined == [(arm_type.YAM, gripper)]
    assert len(created) == 2
    assert all(solver.site_name == "grasp_site" for solver in created)
    ranges = left.get_joint_ranges()
    ranges[0] = (-1.0, 1.0)
    left.set_joint_ranges(ranges)
    assert created[0]._configuration.model.jnt_range[0] == pytest.approx((-1.0, 1.0))
    _, solution = right.ik(np.eye(4), np.zeros(8), 17)
    assert solution == pytest.approx(np.zeros(8))
    assert created[1].ik_site == "grasp_site"
    created[1].raise_infeasible = True
    with pytest.raises(EmbodimentFault, match="infeasible"):
        right.ik(np.eye(4), np.zeros(8), 17)
