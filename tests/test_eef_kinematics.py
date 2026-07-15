"""Tier-1 tests for the always-importable Cartesian kinematics wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from inspect_robots_yam.kinematics import _ArmKinematics


def _pose(rotation: np.ndarray | None = None, position: Sequence[float] = (0.3, 0.0, 0.2)):
    pose = np.eye(4)
    pose[:3, :3] = np.eye(3) if rotation is None else rotation
    pose[:3, 3] = position
    return pose


def _rz(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


class FakeRawKinematics:
    """Deterministic raw seam with configurable FK poses and IK outcomes."""

    def __init__(
        self,
        *,
        ranges: np.ndarray | None = None,
        fk_poses: Sequence[np.ndarray] | None = None,
        ik_results: Sequence[tuple[bool, np.ndarray] | BaseException] | None = None,
    ) -> None:
        self.ranges = (
            np.asarray(ranges, dtype=float).copy()
            if ranges is not None
            else np.asarray([[-2.0, 2.0]] * 6 + [[0.0, 0.04], [0.0, 0.04]])
        )
        self.fk_poses = list(fk_poses or [_pose()])
        self.ik_results = list(ik_results or [(True, np.zeros(len(self.ranges)))])
        self.fk_inputs: list[np.ndarray] = []
        self.ik_calls: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.range_writes: list[np.ndarray] = []

    def get_joint_ranges(self) -> np.ndarray:
        return self.ranges.copy()

    def set_joint_ranges(self, ranges: np.ndarray) -> None:
        self.ranges = np.asarray(ranges, dtype=float).copy()
        self.range_writes.append(self.ranges.copy())

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.fk_inputs.append(np.asarray(q).copy())
        index = min(len(self.fk_inputs) - 1, len(self.fk_poses) - 1)
        return self.fk_poses[index].copy()

    def ik(self, target: np.ndarray, init_q: np.ndarray, max_iters: int) -> tuple[bool, np.ndarray]:
        self.ik_calls.append((target.copy(), init_q.copy(), max_iters))
        result = self.ik_results[min(len(self.ik_calls) - 1, len(self.ik_results) - 1)]
        if isinstance(result, BaseException):
            raise result
        success, q = result
        return success, q.copy()


def _wrapper(raw: FakeRawKinematics, **overrides: object) -> _ArmKinematics:
    values: dict[str, object] = {
        "side": "left",
        "raw": raw,
        "config_low": np.full(6, -1.0),
        "config_high": np.full(6, 1.0),
        "ik_max_iters": 20,
        "ik_step_joint_limit": 0.2,
        "cmd_resync_threshold": 0.35,
        "osc_deadband": 0.005,
        "osc_reversals": 2,
        "osc_window": 6,
        "osc_hold_steps": 10,
    }
    values.update(overrides)
    return _ArmKinematics(**values)  # type: ignore[arg-type]


def test_limit_intersection_is_per_side_and_forwarded_to_raw_model() -> None:
    ranges = np.asarray(
        [
            [-2.0, 2.0],
            [0.0, 3.65],
            [0.0, 3.66],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-2.1, 2.1],
            [0.0, 0.04],
            [0.01, 0.03],
        ]
    )
    raw = FakeRawKinematics(ranges=ranges)
    kin = _wrapper(
        raw,
        config_low=np.asarray([-1.0, -1.0, 0.2, -1.0, -2.0, -3.0]),
        config_high=np.asarray([1.0, 2.0, 3.0, 1.0, 1.0, 2.0]),
    )

    expected = np.asarray(
        [[-1.0, 1.0], [0.0, 2.0], [0.2, 3.0], [-1.0, 1.0], [-1.57, 1.0], [-2.1, 2.0]]
    )
    assert np.array_equal(kin.effective_ranges, expected)
    assert np.array_equal(raw.range_writes[-1][:6], expected)
    assert np.array_equal(raw.range_writes[-1][6:], ranges[6:])


def test_empty_limit_intersection_names_the_offending_joint() -> None:
    ranges = np.asarray([[-2.0, 2.0], [0.0, 3.65]] + [[-2.0, 2.0]] * 4)
    raw = FakeRawKinematics(ranges=ranges)
    with pytest.raises(ValueError, match="left_j1"):
        _wrapper(raw, config_low=np.full(6, -2.0), config_high=np.asarray([2, -0.1] + [2] * 4))


@pytest.mark.parametrize("count", [5, 9])
def test_raw_model_must_have_six_arm_joints_and_at_most_two_gripper_joints(count: int) -> None:
    raw = FakeRawKinematics(ranges=np.asarray([[-1.0, 1.0]] * count))
    with pytest.raises(ValueError, match=r"6 arm joints and 0\.\.2 trailing gripper joints"):
        _wrapper(raw)


def test_gripper_joints_are_midrange_pinned_for_fk_and_ik_then_stripped() -> None:
    ranges = np.asarray([[-2.0, 2.0]] * 6 + [[0.0, 0.04], [0.02, 0.06]])
    solution = np.asarray([0.5] * 6 + [0.0, 0.0])
    raw = FakeRawKinematics(ranges=ranges, ik_results=[(True, solution)])
    kin = _wrapper(raw, ik_step_joint_limit=1.0)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))

    assert raw.fk_inputs[-1][6:] == pytest.approx((0.02, 0.04))
    command = kin.solve(np.asarray((0.3, 0.0, 0.2, 0.0)), np.zeros(6))
    assert raw.ik_calls[-1][1][6:] == pytest.approx((0.02, 0.04))
    assert command.shape == (6,)
    assert command == pytest.approx(np.full(6, 0.5))


def test_rate_backstop_clamps_elbow_flip_and_plumbs_iteration_cap() -> None:
    raw = FakeRawKinematics(ik_results=[(True, np.full(8, 1.5))])
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))

    command = kin.solve(np.asarray((0.35, 0.1, 0.25, 0.0)), np.zeros(6))
    assert command == pytest.approx(np.full(6, 0.2))
    assert raw.ik_calls[-1][2] == 20


def test_nonconverged_finite_last_iterate_is_commanded_best_effort() -> None:
    raw = FakeRawKinematics(ik_results=[(False, np.full(8, 0.1))])
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    assert kin.solve(np.asarray((0.3, 0.0, 0.2, 0.0)), np.zeros(6)) == pytest.approx(
        np.full(6, 0.1)
    )


def test_nonfinite_ik_output_degrades_step_to_previous_command() -> None:
    solution = np.zeros(8)
    solution[2] = np.nan
    raw = FakeRawKinematics(ik_results=[(False, solution)])
    kin = _wrapper(raw)
    kin.seed(np.full(6, 0.1))
    kin.capture_yaw_reference(np.zeros(6))
    assert kin.solve(np.asarray((0.3, 0.0, 0.2, 0.0)), np.zeros(6)) == pytest.approx(
        np.full(6, 0.1)
    )


def test_solver_infeasibility_exception_propagates() -> None:
    error = RuntimeError("NoSolutionFound")
    raw = FakeRawKinematics(ik_results=[error])
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    with pytest.raises(RuntimeError, match="NoSolutionFound"):
        kin.solve(np.asarray((0.3, 0.0, 0.2, 0.0)), np.zeros(6))


def test_resync_reseeds_from_measured_effective_range_and_records_no_reversal() -> None:
    raw = FakeRawKinematics(ik_results=[(True, np.full(8, 1.0))])
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    measured = np.asarray((2.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    command = kin.solve(np.asarray((0.3, 0.0, 0.2, 0.0)), measured)
    assert kin.resynced is True
    assert raw.ik_calls[-1][1][0] == pytest.approx(1.0)
    assert command[0] == pytest.approx(1.0)
    assert not kin.reversal_window[-1].any()


def test_signed_relative_yaw_and_home_reads_exactly_zero() -> None:
    delta = 0.3
    raw = FakeRawKinematics(fk_poses=[_pose(), _pose(), _pose(_rz(delta))])
    kin = _wrapper(raw)
    kin.capture_yaw_reference(np.zeros(6))
    home = kin.observe(np.zeros(6), gripper=1.0)
    moved = kin.observe(np.zeros(6), gripper=0.5)
    assert home == pytest.approx((0.3, 0.0, 0.2, 0.0, 1.0))
    assert moved[3] == pytest.approx(delta)
    assert moved[4] == pytest.approx(0.5)


def test_vertical_axis_fallback_branch_is_pinned_across_threshold_crossing() -> None:
    reference = np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))
    crossed = reference.copy()
    crossed[:, 0] = (0.1, -0.1, 0.99)
    crossed[:, 1] = (np.cos(0.2), np.sin(0.2), 0.0)
    raw = FakeRawKinematics(fk_poses=[_pose(reference), _pose(crossed)])
    kin = _wrapper(raw)
    kin.capture_yaw_reference(np.zeros(6))
    state = kin.observe(np.zeros(6), gripper=1.0)
    assert kin.yaw_axis == 1
    assert state[3] == pytest.approx(0.2)


def test_reset_reference_is_never_reread_when_building_targets() -> None:
    reference = _rz(0.4)
    drifted = _rz(-0.7)
    raw = FakeRawKinematics(
        fk_poses=[_pose(reference), _pose(drifted)], ik_results=[(True, np.zeros(8))]
    )
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    kin.observe(np.zeros(6), gripper=1.0)
    kin.solve(np.asarray((0.3, 0.0, 0.2, 0.25)), np.zeros(6))
    assert raw.ik_calls[-1][0][:3, :3] == pytest.approx(_rz(0.25) @ reference)


def test_alternating_branches_trip_hold_then_re_evaluate_after_hold_steps() -> None:
    results = [(True, np.full(8, value)) for value in (1.0, -1.0, 1.0, -1.0, 0.5)]
    raw = FakeRawKinematics(ik_results=results)
    kin = _wrapper(raw, osc_hold_steps=2)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    target = np.asarray((0.3, 0.0, 0.2, 0.0))

    sent = np.zeros(6)
    for _ in range(3):
        sent = kin.solve(target, sent)
        kin.update_sent(sent)
    before_trip = sent.copy()
    trip = kin.solve(target, sent)
    assert trip == pytest.approx(before_trip)
    assert kin.hold_counter == 2
    assert len(raw.ik_calls) == 4
    for _ in range(2):
        held = kin.solve(target, sent)
        assert held == pytest.approx(before_trip)
    assert len(raw.ik_calls) == 4
    resumed = kin.solve(target, sent)
    assert len(raw.ik_calls) == 5
    assert resumed != pytest.approx(before_trip)


def test_monotone_approach_and_single_overshoot_never_trigger_hold() -> None:
    monotone = FakeRawKinematics(ik_results=[(True, np.full(8, 1.0))])
    monotone_kin = _wrapper(monotone, osc_reversals=0)
    monotone_kin.seed(np.zeros(6))
    monotone_kin.capture_yaw_reference(np.zeros(6))
    target = np.asarray((0.3, 0.0, 0.2, 0.0))
    sent = np.zeros(6)
    for _ in range(4):
        sent = monotone_kin.solve(target, sent)
        monotone_kin.update_sent(sent)
    assert monotone_kin.hold_counter == 0

    overshoot = FakeRawKinematics(
        ik_results=[(True, np.full(8, value)) for value in (1.0, -1.0, -1.0)]
    )
    overshoot_kin = _wrapper(overshoot, osc_reversals=1)
    overshoot_kin.seed(np.zeros(6))
    overshoot_kin.capture_yaw_reference(np.zeros(6))
    sent = np.zeros(6)
    for _ in range(3):
        sent = overshoot_kin.solve(target, sent)
        overshoot_kin.update_sent(sent)
    assert overshoot_kin.hold_counter == 0


def test_pure_yaw_motion_is_unaffected_by_position_progress() -> None:
    raw = FakeRawKinematics(ik_results=[(True, np.full(8, 0.1))])
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    command = kin.solve(np.asarray((0.3, 0.0, 0.2, 0.7)), np.zeros(6))
    assert command == pytest.approx(np.full(6, 0.1))
    assert raw.ik_calls[-1][0][:3, 3] == pytest.approx((0.3, 0.0, 0.2))
    assert kin.hold_counter == 0


def test_clear_resets_hold_window_counters_resync_and_reference() -> None:
    raw = FakeRawKinematics()
    kin = _wrapper(raw)
    kin.seed(np.zeros(6))
    kin.capture_yaw_reference(np.zeros(6))
    kin.clear()
    assert kin.hold_counter == 0
    assert kin.reversal_window == ()
    assert kin.resynced is False
    with pytest.raises(RuntimeError, match="yaw reference"):
        kin.observe(np.zeros(6), gripper=1.0)


def test_seed_validates_shape_and_solve_requires_a_seed() -> None:
    raw = FakeRawKinematics()
    kin = _wrapper(raw)
    kin.capture_yaw_reference(np.zeros(6))
    with pytest.raises(ValueError, match="expected 6 commanded arm joints"):
        kin.seed(np.zeros(7))
    with pytest.raises(RuntimeError, match="command reference"):
        kin.solve(np.asarray((0.3, 0.0, 0.2, 0.0)), np.zeros(6))


def test_wrap_yaw_pi_boundary_maps_to_positive_pi() -> None:
    """A raw difference of exactly -pi must report +pi, keeping (-pi, pi]."""
    assert _ArmKinematics._wrap_yaw(-np.pi) == pytest.approx(np.pi)
    assert _ArmKinematics._wrap_yaw(np.pi) == pytest.approx(np.pi)
    assert _ArmKinematics._wrap_yaw(3 * np.pi) == pytest.approx(np.pi)
