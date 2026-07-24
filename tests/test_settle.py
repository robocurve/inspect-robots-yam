"""Settle-before-observe tests (issue #62, plans/0009-settle-before-observe.md).

The property under test: when a tolerance is configured, the observation a step
returns reflects the pose the embodiment last commanded, rather than whatever
the arm happened to be doing one control period later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from conftest import READ_ADVANCE_S, SettleDriver
from inspect_robots_yam.config import YamConfig

# Defaults the assertions below are pinned to: control_hz 10 gives a 0.1 s pace,
# and settle_timeout_s 1.0 over a 0.01 s poll gives a 100-poll count bound.
PACE_S = 0.1
POLL_S = 0.01
MAX_POLLS = 100

TARGET = np.full(14, 0.1)
FAR = np.zeros(14)
FAR[0] = 0.5  # an arm joint that never arrives: a hard stop, or too tight a tolerance


def _cfg(**kwargs: Any) -> YamConfig:
    kwargs.setdefault("unattended", True)
    kwargs.setdefault("rest_secs", 0.5)
    return YamConfig(**kwargs)


def _settled_cfg(**kwargs: Any) -> YamConfig:
    return _cfg(settle_tolerance=0.05, **kwargs)


def _step(emb: Any, sleeps: list[float]) -> Any:
    """Run one step with the sleep log cleared, so only that step's sleeps show."""
    sleeps.clear()
    return emb.step(Action(data=TARGET.copy()))


def test_disabled_by_default_leaves_timing_untouched(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))
    reads_before = driver.reads

    result = _step(emb, sleeps)

    assert sleeps == [PACE_S]  # the pace, and nothing else
    assert driver.reads - reads_before == 1  # _observe only; no settle polls
    assert result.info == {}


def test_settles_by_polling_until_within_tolerance(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=3)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))
    reads_before = driver.reads

    _step(emb, sleeps)

    assert sleeps == [POLL_S, POLL_S, PACE_S]
    assert driver.reads - reads_before == 4  # three settle polls, then _observe


def test_already_converged_costs_no_poll_sleep(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))

    _step(emb, sleeps)

    # Reads before sleeping, so the common case (and every oscillation hold)
    # adds no latency at all.
    assert sleeps == [PACE_S]


def test_step_info_reports_settled_residual_and_count(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    assert result.info["settled"] is True
    assert result.info["settle_residual"] == pytest.approx(0.0)
    assert result.info["settle_timeouts"] == 0
    assert "settle_disabled" not in result.info


def test_timeout_bounded_by_poll_count_when_the_clock_is_frozen(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))
    # Diverge only after homing, so this asserts on the step's settle alone.
    driver.offset = FAR
    reads_before = driver.reads

    result = _step(emb, sleeps)

    # The frozen clock can never satisfy the elapsed bound, so the poll count is
    # the only thing that ends the loop. Without it this test would hang.
    assert driver.reads - reads_before == MAX_POLLS + 1  # + _observe
    assert sleeps.count(POLL_S) == MAX_POLLS - 1
    assert result.info["settled"] is False
    assert result.info["settle_timeouts"] == 1


def test_timeout_bounded_by_elapsed_time_when_reads_cost_time(
    build_settle: Any, clock: Any
) -> None:

    driver = SettleDriver(converge_after=1, clock=clock, read_advance=READ_ADVANCE_S)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver, clock=clock)
    emb.reset(Scene(id="s", instruction="go"))
    driver.offset = FAR
    reads_before = driver.reads

    result = _step(emb, sleeps)

    # Each read charges 0.05 s, so 1.0 s of budget is gone after 20 polls, well
    # short of the 100-poll cap: elapsed is the only reason the loop stopped.
    expected = int(1.0 / READ_ADVANCE_S)
    assert driver.reads - reads_before == expected + 1  # + _observe
    assert expected < MAX_POLLS
    assert result.info["settled"] is False


def test_budget_exhaustion_disables_settling_and_marks_every_later_step(
    build_settle: Any,
) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(settle_timeout_budget=2), driver)
    emb.reset(Scene(id="s", instruction="go"))

    driver.offset = FAR
    first = _step(emb, sleeps)
    assert first.info["settle_timeouts"] == 1

    # A success in between must not replenish the budget: this is why the budget
    # is absolute rather than a consecutive-timeout streak, which an oscillation
    # hold's instant settle would reset forever.
    driver.offset = np.zeros(14)
    good = _step(emb, sleeps)
    assert good.info["settled"] is True
    assert good.info["settle_timeouts"] == 1

    driver.offset = FAR
    with pytest.warns(UserWarning, match="settle timeout budget exhausted"):
        tripped = _step(emb, sleeps)
    assert tripped.info["settle_timeouts"] == 2
    assert tripped.info["settled"] is False

    reads_before = driver.reads
    after = _step(emb, sleeps)
    assert after.info["settle_disabled"] is True
    assert "settled" not in after.info  # no settle ran
    assert driver.reads - reads_before == 1  # _observe only
    assert sleeps == [PACE_S]


def test_gripper_divergence_alone_does_not_block_settling(build_settle: Any) -> None:
    stuck_gripper = np.zeros(14)
    stuck_gripper[[6, 13]] = 0.5  # closed on an object, never reaching its target
    driver = SettleDriver(converge_after=1, offset=stuck_gripper)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    assert result.info["settled"] is True
    assert sleeps == [PACE_S]


def test_reset_settles_after_homing(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1, offset=FAR)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)

    emb.reset(Scene(id="s", instruction="go"))

    # A non-arriving arm makes the reset settle visible: without it, homing
    # would flow straight into capture_yaw_reference and the first observation
    # the policy ever sees, both taken from a possibly mid-motion pose.
    assert sleeps.count(POLL_S) == MAX_POLLS - 1
    assert emb.settle_timeouts == 1


def test_close_does_not_settle(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1, offset=FAR)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))
    sleeps.clear()

    emb.close()

    # Waiting for convergence during teardown would delay releasing the handles
    # for nothing, since torque drops immediately afterwards.
    assert sleeps.count(POLL_S) == 0


def test_reset_clears_settle_state_at_entry_so_the_next_trial_still_settles(
    build_settle: Any,
) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(settle_timeout_budget=1), driver)
    emb.reset(Scene(id="s", instruction="go"))

    driver.offset = FAR
    with pytest.warns(UserWarning, match="settle timeout budget exhausted"):
        _step(emb, sleeps)
    assert emb._settle_disabled is True

    # Clearing alongside num_steps would run *after* the homing ramp, leaving
    # this reset's settle suppressed and the yaw reference pinned mid-motion for
    # every remaining trial of the eval.
    driver.offset = np.zeros(14)
    sleeps.clear()
    emb.reset(Scene(id="s2", instruction="again"))

    assert emb.settle_timeouts == 0
    assert emb._settle_disabled is False
    result = _step(emb, sleeps)
    assert result.info["settled"] is True


def test_unattended_settling_emits_no_status(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, status = build_settle(_settled_cfg(unattended=True), driver)

    emb.reset(Scene(id="s", instruction="go"))
    _step(emb, sleeps)

    assert status == []


def test_attended_settling_announces_itself(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=2)
    emb, _sleeps, status = build_settle(_settled_cfg(unattended=False), driver)

    emb.reset(Scene(id="s", instruction="go"))

    assert any(line is not None and "settling" in line for line in status)


def test_eef_settles_against_the_joint_vector_actually_sent(build_settle: Any) -> None:
    solution = np.full(8, 0.2)
    left, right = _FakeRawKinematics(solution), _FakeRawKinematics(solution)
    driver = SettleDriver(converge_after=2)
    emb, sleeps, _ = build_settle(
        _settled_cfg(control_interface="eef_pos"),
        driver,
        kinematics_factory=lambda _c: (left, right),
    )
    emb.reset(Scene(id="s", instruction="go"))
    sleeps.clear()

    result = emb.step(Action(data=np.zeros(10)))

    # The settle target is the clamped joint command _step_eef sent, not the
    # 10-D end-effector request the policy made.
    assert result.info["settled"] is True
    assert np.allclose(driver.state[:6], driver.commands[-1][:6])


class _FakeRawKinematics:
    """Minimal raw-kinematics stand-in: fixed FK pose, fixed IK solution."""

    def __init__(self, solution: np.ndarray) -> None:
        self.ranges = np.asarray([[-2.0, 2.0]] * 6 + [[0.0, 0.04], [0.0, 0.04]])
        self.solution = solution

    def get_joint_ranges(self) -> np.ndarray:
        """Report the fake joint ranges."""
        return self.ranges.copy()

    def set_joint_ranges(self, ranges: np.ndarray) -> None:
        """Accept a range override."""
        self.ranges = np.asarray(ranges).copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return a fixed homogeneous pose regardless of joint angles."""
        pose = np.eye(4)
        pose[:3, 3] = (0.3, 0.0, 0.2)
        return pose

    def ik(self, target: np.ndarray, init_q: np.ndarray, max_iters: int) -> tuple[bool, np.ndarray]:
        """Return the fixed solution, always converged."""
        return True, self.solution.copy()
