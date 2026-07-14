"""End-to-end: full eval() rollouts of a KitchenBench task on mocked YAM +
MolmoAct2 — proving the termination_reason -> scorer wiring and chunk replay
compose (the static compat test cannot show this), and that a trial ending at
the task horizon still parks the arms at the captured init pose on close."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_robots import eval as rl_eval

from inspect_robots_yam.config import MolmoActConfig, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO
from inspect_robots_yam.policy import MolmoAct2Policy


class _FakeDriver:
    def __init__(self) -> None:
        self.state = np.zeros(14)
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.commands.append(np.asarray(target, dtype=float).copy())
        self.state = np.asarray(target, dtype=float)

    def close(self) -> None:
        self.closed = True


def _cameras(_cfg):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": img, "left_cam": img, "right_cam": img}


def _post(url, payload, timeout_s):
    # One-action chunk of zeros; dt_ms => chunk control rate.
    return {"actions": np.zeros((1, 14), dtype=np.float32), "dt_ms": 100.0}


def _always_yes_operator() -> OperatorIO:
    return OperatorIO(input_fn=lambda _p: "y", output_fn=lambda _m: None)


def _post_away(url, payload, timeout_s):
    # Push every joint away from the start pose so the park motion is visible.
    return {"actions": np.full((1, 14), 0.8, dtype=np.float32), "dt_ms": 100.0}


def test_timer_runout_parks_at_init_pose_on_close() -> None:
    init_pose = np.full(14, 0.2)
    drv = _FakeDriver()
    drv.state = init_pose.copy()
    policy = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4, num_steps=1), post_fn=_post_away)
    embodiment = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4, rest_pose=None, unattended=True, rest_secs=0.4),
        driver_factory=lambda _c: drv,
        camera_reader=_cameras,
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )

    logs = rl_eval("kitchenbench/stack", policy, embodiment, sinks=[], seed=0)

    assert logs[0].status == "success"
    assert embodiment.num_steps == 80  # the task horizon ended the trial, not the operator
    assert drv.closed is False  # caller-owned: eval() must NOT have closed it for us
    embodiment.close()  # eval() itself closes registry-resolved embodiments (the CLI path)
    assert drv.commands[-1] == pytest.approx(init_pose)
    assert drv.closed is True


def test_eval_scores_success_end_to_end() -> None:
    policy = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4, num_steps=1), post_fn=_post)
    embodiment = YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4),
        driver_factory=lambda _c: _FakeDriver(),
        camera_reader=_cameras,
        operator=_always_yes_operator(),
        poll_end=lambda: True,  # operator ends every episode immediately
        sleep_fn=lambda _d: None,
        clock=lambda: 0.0,
    )

    logs = rl_eval("kitchenbench/stack", policy, embodiment, sinks=[], seed=0)

    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["task_success"] == 1.0
