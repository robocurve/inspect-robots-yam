"""Settle-before-observe tests (issue #62, plans/0009-settle-before-observe.md).

The property under test: when a tolerance is configured, the observation a step
returns reflects the pose the embodiment last commanded, rather than whatever
the arm happened to be doing one control period later.
"""

from __future__ import annotations

import logging
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


def _far(index: int = 0) -> np.ndarray:
    """A residual on one arm joint that never arrives: a hard stop, or too tight a tolerance."""
    offset = np.zeros(14)
    offset[index] = 0.5
    return offset


FAR = _far()
#: Left arm joint 0 and right arm joint 0. Perturbing only the left would let a
#: mask that silently drops the right arm pass every divergence test.
ARM_INDICES = (0, 7)


def _cfg(**kwargs: Any) -> YamConfig:
    kwargs.setdefault("unattended", True)
    kwargs.setdefault("rest_secs", 0.5)
    return YamConfig(**kwargs)


def _settled_cfg(**kwargs: Any) -> YamConfig:
    # zero_gravity_mode off: settling presumes a position-holding servo, and the
    # config warns about the compliant pairing.
    kwargs.setdefault("zero_gravity_mode", False)
    kwargs.setdefault("settle_tolerance", 0.05)
    return _cfg(**kwargs)


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


@pytest.mark.parametrize("index", ARM_INDICES)
def test_timeout_bounded_by_poll_count_when_the_clock_is_frozen(
    build_settle: Any, index: int
) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(), driver)
    emb.reset(Scene(id="s", instruction="go"))
    # Diverge only after homing, so this asserts on the step's settle alone.
    driver.offset = _far(index)
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
    build_settle: Any, caplog: pytest.LogCaptureFixture
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
    with caplog.at_level(logging.WARNING, logger="inspect_robots_yam.embodiment"):
        tripped = _step(emb, sleeps)
    assert "settle timeout budget exhausted" in caplog.text
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


def test_budget_warning_repeats_across_trials(
    build_settle: Any, caplog: pytest.LogCaptureFixture
) -> None:
    driver = SettleDriver(converge_after=1, offset=FAR)
    emb, _sleeps, _ = build_settle(_settled_cfg(settle_timeout_budget=1), driver)

    with caplog.at_level(logging.WARNING, logger="inspect_robots_yam.embodiment"):
        for scene in ("one", "two"):
            emb.reset(Scene(id=scene, instruction=scene))

    # A joint parked against a hard stop reports the same residual every trial.
    # Routed through warnings.warn, the registry keys on the message text and
    # the second trial's notice would vanish, exactly when the rig is worst.
    exhausted = [r for r in caplog.records if "budget exhausted" in r.getMessage()]
    assert len(exhausted) == 2
    # Named per trial: _instruction is set at reset entry, so the homing settle
    # cannot report the previous trial's scene.
    assert "'one'" in exhausted[0].getMessage()
    assert "'two'" in exhausted[1].getMessage()


def test_reset_clears_settle_state_at_entry_so_the_next_trial_still_settles(
    build_settle: Any,
) -> None:
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(_settled_cfg(settle_timeout_budget=1), driver)
    emb.reset(Scene(id="s", instruction="go"))

    driver.offset = FAR
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


def test_eef_settles_against_the_commanded_pose_not_the_request(build_settle: Any) -> None:
    # A non-finite IK solve makes solve() re-send the previous pose. Settling
    # then succeeds instantly while the arm sits nowhere near the requested
    # end-effector target, which is exactly what the guarantee does not cover:
    # "reached what was commanded", never "reached what the policy asked for".
    left = _FakeRawKinematics(np.full(8, np.nan))
    right = _FakeRawKinematics(np.full(8, np.nan))
    driver = SettleDriver(converge_after=1)
    emb, sleeps, _ = build_settle(
        _settled_cfg(control_interface="eef_pos"),
        driver,
        kinematics_factory=lambda _c: (left, right),
    )
    emb.reset(Scene(id="s", instruction="go"))
    sleeps.clear()

    result = emb.step(Action(data=np.full(14, 0.3)))

    assert result.info["settled"] is True
    assert sleeps == [PACE_S]  # nothing to wait for: the arm was already there


def test_reset_settles_before_capturing_the_yaw_reference(build_settle: Any) -> None:
    solution = np.zeros(8)
    left, right = _FakeRawKinematics(solution), _FakeRawKinematics(solution)
    driver = SettleDriver(converge_after=1, offset=FAR)
    emb, _sleeps, _ = build_settle(
        _settled_cfg(control_interface="eef_pos"),
        driver,
        kinematics_factory=lambda _c: (left, right),
    )
    left.watch(driver)

    emb.reset(Scene(id="s", instruction="go"))

    # This arm's fk calls are, in order: the home validation, capture_yaw_
    # reference, then the first observation. The homing settle burns MAX_POLLS
    # reads, so with it running first the capture has already seen them. Move the
    # settle after the capture and only the trailing observation call has.
    first_after_settle = next(i for i, reads in enumerate(left.fk_reads) if reads >= MAX_POLLS)
    assert first_after_settle <= len(left.fk_reads) - 2


def test_arm_mask_covers_both_arms_and_excludes_only_the_grippers() -> None:
    from inspect_robots_yam.embodiment import _ARM_SLOTS

    assert set(_ARM_SLOTS.tolist()) == set(range(14)) - {6, 13}


def test_terminal_step_result_keeps_the_settle_keys(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=1)
    cfg = _settled_cfg(unattended=False)
    emb, sleeps, _ = build_settle(cfg, driver)
    emb._poll_end = lambda: True  # operator ends the episode on the next step
    emb.reset(Scene(id="s", instruction="go"))

    result = _step(emb, sleeps)

    # The terminated step is the final state a scorer judges, which is exactly
    # where settle_disabled would matter most.
    assert result.terminated is True
    assert result.termination_reason == "operator_end"
    assert result.info["settled"] is True
    assert result.info["settle_timeouts"] == 0


def test_residual_exactly_at_the_tolerance_counts_as_settled(build_settle: Any) -> None:
    # Powers of two, so target + offset - target is exactly the tolerance rather
    # than a float hair above it: this is the boundary "within that many
    # radians" promises, and it is what separates <= from <.
    tolerance = 0.0625
    at_limit = np.zeros(14)
    at_limit[0] = tolerance
    driver = SettleDriver(converge_after=1, offset=at_limit)
    emb, sleeps, _ = build_settle(_settled_cfg(settle_tolerance=tolerance), driver)
    emb.reset(Scene(id="s", instruction="go"))
    sleeps.clear()

    result = emb.step(Action(data=np.full(14, 0.125)))

    assert result.info["settle_residual"] == tolerance
    assert result.info["settled"] is True


def test_delta_mode_settles_against_the_absolute_target(build_settle: Any) -> None:
    driver = SettleDriver(converge_after=2)
    emb, sleeps, _ = build_settle(_settled_cfg(joints_are_delta=True), driver)
    emb.reset(Scene(id="s", instruction="go"))
    sleeps.clear()

    result = emb.step(Action(data=np.full(14, 0.05)))

    # The delta is resolved against the measured base inside step(); settling
    # then makes that base a converged pose on the following step.
    assert result.info["settled"] is True
    assert np.allclose(driver.state[:6], driver.commands[-1][:6])


class _FakeRawKinematics:
    """Minimal raw-kinematics stand-in: fixed FK pose, fixed IK solution."""

    def __init__(self, solution: np.ndarray) -> None:
        self.ranges = np.asarray([[-6.0, 6.0]] * 6 + [[0.0, 0.04], [0.0, 0.04]])
        self.solution = solution
        self.fk_reads: list[int] = []
        self._driver: Any = None

    def watch(self, driver: Any) -> None:
        """Record the driver's read count at each fk call, to pin down call ordering."""
        self._driver = driver

    def get_joint_ranges(self) -> np.ndarray:
        """Report the fake joint ranges."""
        return self.ranges.copy()

    def set_joint_ranges(self, ranges: np.ndarray) -> None:
        """Accept a range override."""
        self.ranges = np.asarray(ranges).copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return a fixed homogeneous pose regardless of joint angles."""
        if self._driver is not None:
            self.fk_reads.append(self._driver.reads)
        pose = np.eye(4)
        pose[:3, 3] = (0.3, 0.0, 0.2)
        return pose

    def ik(self, target: np.ndarray, init_q: np.ndarray, max_iters: int) -> tuple[bool, np.ndarray]:
        """Return the fixed solution, always converged."""
        return True, self.solution.copy()
