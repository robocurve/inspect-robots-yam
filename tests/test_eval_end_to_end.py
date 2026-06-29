"""End-to-end: a full eval() rollout of a KitchenBench task on mocked YAM +
MolmoAct2 actually scores success — proving the termination_reason -> scorer
wiring and chunk replay compose (the static compat test cannot show this)."""

from __future__ import annotations

import numpy as np
from robolens import eval as rl_eval

from robolens_yam.config import MolmoActConfig, YamConfig
from robolens_yam.embodiment import YAMEmbodiment
from robolens_yam.operator import OperatorIO
from robolens_yam.policy import MolmoAct2Policy


class _FakeDriver:
    def __init__(self) -> None:
        self.state = np.zeros(14)

    def get_joint_pos(self) -> np.ndarray:
        return self.state.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        self.state = np.asarray(target, dtype=float)

    def close(self) -> None: ...


def _cameras(_cfg):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": img, "left_cam": img, "right_cam": img}


def _post(url, payload, timeout_s):
    # One-action chunk of zeros; dt_ms => chunk control rate.
    return {"actions": np.zeros((1, 14), dtype=np.float32), "dt_ms": 100.0}


def _always_yes_operator() -> OperatorIO:
    return OperatorIO(input_fn=lambda _p: "y", output_fn=lambda _m: None)


def test_eval_scores_success_end_to_end() -> None:
    policy = MolmoAct2Policy(MolmoActConfig(num_steps=1), post_fn=_post)
    embodiment = YAMEmbodiment(
        YamConfig(),
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
