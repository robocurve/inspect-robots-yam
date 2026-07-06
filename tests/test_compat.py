"""Goal-closing test: Inspect Robots + KitchenBench are provably compatible with
YAM arms + MolmoAct2 — zero errors, zero warnings, all 10 tasks realizable."""

from __future__ import annotations

import pytest
from inspect_robots.compat import check_compatibility
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_yam.config import action_box, observation_space
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.policy import MolmoAct2Policy

KITCHENBENCH_TASKS = [
    "kitchenbench/place_cutlery",
    "kitchenbench/stack",
    "kitchenbench/place_in_rack",
    "kitchenbench/pour_pasta",
    "kitchenbench/open_container",
    "kitchenbench/fold_cloth",
    "kitchenbench/seal_container",
    "kitchenbench/handoff",
    "kitchenbench/sort_cutlery",
    "kitchenbench/scoop_pasta",
]


def test_molmoact2_yam_compatible_no_errors_no_warnings() -> None:
    report = check_compatibility(MolmoAct2Policy(), YAMEmbodiment())
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


@pytest.mark.parametrize("task_name", KITCHENBENCH_TASKS)
def test_every_kitchenbench_task_is_realizable(task_name: str) -> None:
    task = resolve("task", task_name)
    report = check_compatibility(MolmoAct2Policy(), YAMEmbodiment(), task)
    assert report.ok is True, [i.message for i in report.errors]
    assert report.errors == []


def test_state_key_mismatch_trips_state_error() -> None:
    # A non-default state_key must surface in compat (the policy would KeyError in
    # act() otherwise), and disappear again when both sides agree.
    from inspect_robots_yam.config import MolmoActConfig

    mismatched = MolmoAct2Policy(MolmoActConfig(state_key="proprio"))
    report = check_compatibility(mismatched, YAMEmbodiment())
    assert report.ok is False
    assert any(i.code == "missing_state" for i in report.errors)

    matched = MolmoAct2Policy(MolmoActConfig(state_key="joint_pos"))
    ok_report = check_compatibility(matched, YAMEmbodiment())
    assert ok_report.ok is True and ok_report.errors == []


def test_negative_wrong_dim_policy_trips_action_dim_error() -> None:
    bad = PolicyInfo(
        name="bad",
        action_space=Box(shape=(8,), semantics=ActionSemantics(control_mode="joint_pos")),
    )

    class _Bad:
        info = bad
        config = PolicyConfig()

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Bad(), YAMEmbodiment())  # type: ignore[arg-type]
    assert report.ok is False
    assert any(i.code == "action_dim" for i in report.errors)


def test_negative_policy_advertising_rate_trips_control_rate_warning() -> None:
    # Locks the load-bearing detail: leaving PolicyInfo.control_hz=None is the only
    # reason the real pairing is warning-free. A policy that advertises a rate warns.
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(224, 224, ("top_cam", "left_cam", "right_cam")),
        control_hz=100.0,
    )

    class _Rated:
        config = PolicyConfig()

        def __init__(self) -> None:
            self.info = info

        def reset(self, scene: object) -> None: ...
        def act(self, obs: object) -> object: ...  # pragma: no cover

    report = check_compatibility(_Rated(), YAMEmbodiment())  # type: ignore[arg-type]
    assert report.ok is True  # only a warning, not an error
    assert any(i.code == "control_rate" for i in report.warnings)
