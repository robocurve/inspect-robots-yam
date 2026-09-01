from __future__ import annotations

import builtins
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib import resources
from typing import Any

import numpy as np
import pytest
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.errors import SafetyAbort
from inspect_robots.spaces import ActionSemantics, Box
from inspect_robots.types import Action

from inspect_robots_yam import collision
from inspect_robots_yam import embodiment as embodiment_module
from inspect_robots_yam.collision import (
    CollisionApprover,
    CollisionChecker,
    CollisionConfig,
    build_yam_guardrails,
)
from inspect_robots_yam.config import (
    DEFAULT_EEF_HIGH,
    DEFAULT_EEF_LOW,
    DEFAULT_JOINT_HOME_POSE,
    YamConfig,
    action_box,
)
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.packing import DIM_LABELS, TOTAL_DIM


def _pose(**values: float) -> np.ndarray:
    pose = np.zeros(TOTAL_DIM, dtype=np.float64)
    pose[DIM_LABELS.index("left_gripper")] = 1.0
    pose[DIM_LABELS.index("right_gripper")] = 1.0
    for label, value in values.items():
        pose[DIM_LABELS.index(label)] = value
    return pose


HOME = np.asarray(DEFAULT_JOINT_HOME_POSE, dtype=np.float64)
REACH_DOWN = _pose(left_j1=1.8, left_j2=0.3, right_j1=1.8, right_j2=0.3)
_EEF_SKIP_WARNING = "collision guardrail skipped: absolute joints mode only (plan 0011 v1)"
_EEF_DISABLED_WARNING = (
    "collision guardrail disabled by config; set collision_guardrail=true "
    "after measuring collision_*_base_pos"
)
_EEF_ORIENTATION_WARNING = (
    "eef_orientation=true: pitch/roll bounds written as 0,0 are widened "
    "to +/-0.6 / +/-pi/2; set eef_orientation=false to re-pin"
)
_EEF_PINNED_WARNING = (
    "eef_pos: action dims left_pitch, left_roll, right_pitch, right_roll are pinned "
    "(low == high) and not commandable; widen eef_low/eef_high "
    "(eef_orientation=true opens only zero-pinned pitch/roll)"
)
_EEF_Z_FLOOR_WARNING = (
    "eef pitch/roll are open but eef_low z is at or below the fingertips-down default; "
    "knuckles or the wrist camera can reach the table first; raise the z floor"
)


@pytest.fixture(scope="module")
def checker() -> CollisionChecker:
    return CollisionChecker()


@pytest.fixture(scope="module")
def joint_space() -> Box:
    config = YamConfig()
    return action_box(config.low, config.high)


def test_importing_package_and_collision_submodule_does_not_import_mujoco() -> None:
    code = (
        "import sys; "
        "import inspect_robots_yam; "
        "assert 'mujoco' not in sys.modules; "
        "import inspect_robots_yam.collision; "
        "assert 'mujoco' not in sys.modules"
    )
    subprocess.run((sys.executable, "-c", code), check=True)


def test_lazy_loader_has_guided_install_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def broken_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mujoco":
            raise ImportError("injected missing mujoco")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(RuntimeError, match=r'pip install "inspect-robots-yam\[collision\]"'):
        collision._load_mujoco()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"left_base_pos": (0.0, 0.0)}, "left_base_pos"),
        ({"right_base_pos": (0.0, np.inf, 0.0)}, "right_base_pos"),
        ({"left_base_yaw": np.nan}, "left_base_yaw"),
        ({"right_base_yaw": np.inf}, "right_base_yaw"),
        ({"table_height": np.nan}, "table_height"),
        ({"penetration_threshold": -1.0}, "penetration_threshold"),
        ({"penetration_threshold": np.inf}, "penetration_threshold"),
        ({"sweep_resolution": 0.0}, "sweep_resolution"),
        ({"sweep_resolution": np.nan}, "sweep_resolution"),
        ({"gripper_qpos": "command"}, "not supported"),
    ],
)
def test_collision_config_rejects_unsound_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CollisionConfig(**kwargs)


def test_collision_config_accepts_no_table_and_measured_finite_geometry() -> None:
    config = CollisionConfig(
        left_base_pos=(0.1, 0.2, 0.3),
        right_base_pos=(-0.1, -0.2, -0.3),
        left_base_yaw=0.2,
        right_base_yaw=-0.2,
        table_height=None,
        penetration_threshold=0.0,
        sweep_resolution=0.01,
    )
    assert config.table_height is None


def test_packaged_asset_has_provenance_and_parses_via_resources() -> None:
    asset = resources.files("inspect_robots_yam").joinpath("assets/yam_collision.xml")
    xml = asset.read_text(encoding="utf-8")
    assert "Menagerie commit: 71f066ad0be9cd271f7ed58c030243ef157af9f4" in xml
    assert "Source: i2rt_yam/yam.xml" in xml
    root = ET.fromstring(xml)
    assert root.tag == "mujoco"
    for tag in (
        "compiler",
        "option",
        "asset",
        "mesh",
        "material",
        "texture",
        "light",
        "camera",
        "actuator",
        "general",
        "keyframe",
        "equality",
    ):
        assert root.find(f".//{tag}") is None


def test_composed_model_has_expected_named_joints_and_only_primitive_geoms(
    checker: CollisionChecker,
) -> None:
    mujoco = checker._mujoco
    names = {
        mujoco.mj_id2name(checker._model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(checker._model.njnt)
    }
    expected = {
        *(f"{side}_joint{index}" for side in ("left", "right") for index in range(1, 7)),
        *(f"{side}_{finger}_finger" for side in ("left", "right") for finger in ("left", "right")),
    }
    assert names == expected
    assert checker._model.nq == 16
    assert not np.any(checker._model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    joint1 = mujoco.mj_name2id(
        checker._model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "left_joint1",
    )
    assert checker._model.jnt_range[joint1] == pytest.approx((-2.61799, 3.05433))


def test_bad_xml_and_missing_joint_names_fail_with_actionable_errors() -> None:
    with pytest.raises(ValueError, match="malformed or incompatible"):
        CollisionChecker(model_xml="<mujoco><broken>")

    missing_joint_xml = """
    <mujoco>
      <worldbody>
        <body name="arm">
          <joint name="wrong"/>
          <geom name="mass" type="sphere" size="0.01"/>
        </body>
      </worldbody>
    </mujoco>
    """
    with pytest.raises(ValueError, match=r"joint names do not match.*missing=.*unexpected="):
        CollisionChecker(CollisionConfig(table_height=None), model_xml=missing_joint_xml)


def test_home_and_folded_zero_are_clear_with_no_base_table_contacts(
    checker: CollisionChecker,
) -> None:
    assert not checker.check(HOME).collided
    assert not checker.check(np.zeros(TOTAL_DIM)).collided
    contacts = checker._data.contact[: checker._data.ncon]
    contact_names = {
        checker._geom_name(int(geom_id))
        for contact in contacts
        for geom_id in (contact.geom1, contact.geom2)
    }
    assert "table" not in contact_names


def test_reach_down_hits_named_table_geom_and_no_table_scene_removes_it(
    checker: CollisionChecker,
) -> None:
    report = checker.check(REACH_DOWN)
    assert report.collided
    assert report.distance is not None and report.distance < -checker.config.penetration_threshold
    assert "table" in (report.geom1, report.geom2)
    assert report.geom1 is not None and report.geom2 is not None

    no_table = CollisionChecker(CollisionConfig(table_height=None))
    no_table_report = no_table.check(REACH_DOWN)
    assert "table" not in (no_table_report.geom1, no_table_report.geom2)
    geom_names = {
        no_table._mujoco.mj_id2name(
            no_table._model,
            no_table._mujoco.mjtObj.mjOBJ_GEOM,
            index,
        )
        for index in range(no_table._model.ngeom)
    }
    assert "table" not in geom_names


def test_cross_arm_collision_names_both_sides() -> None:
    overlapping = CollisionChecker(
        CollisionConfig(
            left_base_pos=(0.0, 0.0, 0.0),
            right_base_pos=(0.0, 0.0, 0.0),
            table_height=None,
        )
    )
    report = overlapping.check(HOME)
    assert report.collided
    assert report.geom1 is not None and report.geom2 is not None
    assert {report.geom1.split("_", 1)[0], report.geom2.split("_", 1)[0]} == {
        "left",
        "right",
    }


def test_penetration_threshold_is_strict_at_the_deepest_contact() -> None:
    raw = CollisionChecker(CollisionConfig(penetration_threshold=0.0))
    assert raw.check(REACH_DOWN).collided
    deepest = max(
        -float(contact.dist)
        for contact in raw._data.contact[: raw._data.ncon]
        if contact.dist < 0.0
    )
    at_boundary = CollisionChecker(CollisionConfig(penetration_threshold=deepest))
    assert not at_boundary.check(REACH_DOWN).collided
    below_boundary = CollisionChecker(
        CollisionConfig(penetration_threshold=float(np.nextafter(deepest, 0.0)))
    )
    assert below_boundary.check(REACH_DOWN).collided


def test_fingers_use_sign_correct_open_extremes_and_ignore_action_grippers(
    checker: CollisionChecker,
) -> None:
    checker.check(_pose(left_gripper=0.0, right_gripper=0.0))
    first = checker._data.qpos.copy()
    checker.check(_pose(left_gripper=1.0, right_gripper=1.0))
    assert checker._data.qpos == pytest.approx(first)
    for side in ("left", "right"):
        left_name = f"{side}_left_finger"
        right_name = f"{side}_right_finger"
        left_id = checker._mujoco.mj_name2id(
            checker._model, checker._mujoco.mjtObj.mjOBJ_JOINT, left_name
        )
        right_id = checker._mujoco.mj_name2id(
            checker._model, checker._mujoco.mjtObj.mjOBJ_JOINT, right_name
        )
        assert checker._data.qpos[checker._qpos_addresses[left_name]] == pytest.approx(
            checker._model.jnt_range[left_id, 1]
        )
        assert checker._data.qpos[checker._qpos_addresses[right_name]] == pytest.approx(
            checker._model.jnt_range[right_id, 0]
        )


def test_checker_rejects_wrong_shape_and_nonfinite_values(checker: CollisionChecker) -> None:
    with pytest.raises(ValueError, match="expected a 14-D vector"):
        checker.check(np.zeros((2, 7)))
    for value in (np.nan, np.inf, -np.inf):
        action = HOME.copy()
        action[0] = value
        with pytest.raises(SafetyAbort, match="non-finite"):
            checker.check(action)


def test_unnamed_geom_has_stable_fallback_name(checker: CollisionChecker) -> None:
    assert checker._geom_name(-1) == "geom#-1"


@pytest.mark.parametrize(
    "space",
    [
        action_box(
            np.full(14, -1.0),
            np.full(14, 1.0),
            control_interface="eef_pos",
        ),
        action_box(
            np.full(14, -1.0),
            np.full(14, 1.0),
            joints_are_delta=True,
        ),
        Box(shape=(14,), semantics=None),
        Box(
            shape=(13,),
            semantics=ActionSemantics(control_mode="joint_pos"),
        ),
    ],
)
def test_approver_rejects_non_absolute_fourteen_dimensional_spaces(
    checker: CollisionChecker,
    space: Box,
) -> None:
    with pytest.raises(ValueError, match=r"Plan 0011.*14-D.*joint_pos"):
        CollisionApprover(checker, HOME, action_space=space)


def test_approver_rejects_bad_mode_start_shape_nonfinite_and_collision(
    checker: CollisionChecker,
    joint_space: Box,
) -> None:
    with pytest.raises(ValueError, match="on_violation"):
        CollisionApprover(checker, HOME, action_space=joint_space, on_violation="stop")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected a 14-D vector"):
        CollisionApprover(checker, np.zeros(13), action_space=joint_space)
    bad = HOME.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CollisionApprover(checker, bad, action_space=joint_space)
    with pytest.raises(
        ValueError,
        match=r"already in collision.*collision_guardrail=false.*collision_\* geometry",
    ):
        CollisionApprover(checker, REACH_DOWN, action_space=joint_space)


def test_safe_pass_through_preserves_identity_and_updates_last_safe(
    checker: CollisionChecker,
    joint_space: Box,
) -> None:
    approver = CollisionApprover(checker, HOME, action_space=joint_space)
    target = _pose(left_j0=0.01)
    action = Action(target, meta={"source": "policy"})
    store: dict[str, Any] = {}
    assert approver.review(action, store) is action
    assert store["yam_collision:last_safe"] == pytest.approx(target)

    next_action = Action(_pose(left_j0=0.02))
    assert approver.review(next_action, store) is next_action
    assert store["yam_collision:last_safe"] == pytest.approx(next_action.data)


def test_hold_uses_last_safe_meta_and_reanchors_delta_limiter(
    checker: CollisionChecker,
    joint_space: Box,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approver = CollisionApprover(checker, HOME, action_space=joint_space)
    store: dict[str, Any] = {"yam_collision:last_safe": HOME.copy()}
    DeltaLimitApprover(joint_space).review(Action(REACH_DOWN), store)
    rewinds: list[np.ndarray] = []
    rewind_reference = DeltaLimitApprover.rewind_reference

    def record_rewind(target_store: dict[str, Any], pose: np.ndarray) -> None:
        rewinds.append(pose.copy())
        rewind_reference(target_store, pose)

    monkeypatch.setattr(
        DeltaLimitApprover,
        "rewind_reference",
        staticmethod(record_rewind),
    )
    action = Action(
        REACH_DOWN,
        meta={
            "source": "policy",
            "clamped": True,
            "delta_clamped": True,
        },
    )
    held = approver.review(action, store)
    assert held is not action
    assert held.data == pytest.approx(HOME)
    assert held.meta["source"] == "policy"
    assert held.meta["collision_blocked"] is True
    assert held.meta["collision_detail"].endswith(tuple(f"@{step}/36" for step in range(1, 37)))
    assert ":" in held.meta["collision_detail"]
    assert "clamped" not in held.meta
    assert "delta_clamped" not in held.meta
    assert len(rewinds) == 1
    assert rewinds[0] == pytest.approx(HOME)
    assert store["yam_collision:last_safe"] == pytest.approx(HOME)

    no_delta_store: dict[str, Any] = {"yam_collision:last_safe": HOME.copy()}
    approver.review(Action(REACH_DOWN), no_delta_store)
    assert len(rewinds) == 2
    assert rewinds[1] == pytest.approx(HOME)


def test_strict_mode_and_nonfinite_action_abort(
    checker: CollisionChecker,
    joint_space: Box,
) -> None:
    strict = CollisionApprover(
        checker,
        HOME,
        action_space=joint_space,
        on_violation="abort",
    )
    with pytest.raises(SafetyAbort, match=r"blocked predicted collision: .*@\d+/\d+"):
        strict.review(Action(REACH_DOWN), {})
    bad = HOME.copy()
    bad[1] = np.inf
    with pytest.raises(SafetyAbort, match="non-finite"):
        strict.review(Action(bad), {})


def test_resolution_derived_substeps_clamp_at_one_and_sixty_four(
    checker: CollisionChecker,
    joint_space: Box,
) -> None:
    approver = CollisionApprover(checker, HOME, action_space=joint_space)
    assert approver._substep_count(HOME, HOME) == 1
    gripper_only = HOME.copy()
    gripper_only[DIM_LABELS.index("left_gripper")] = 0.0
    assert approver._substep_count(HOME, gripper_only) == 1
    far = HOME.copy()
    far[DIM_LABELS.index("left_j0")] = 100.0
    assert approver._substep_count(HOME, far) == 64


def test_first_action_sweep_blocks_cross_arm_collision_between_safe_endpoints() -> None:
    config = CollisionConfig(table_height=None, sweep_resolution=0.1)
    checker = CollisionChecker(config)
    rng = np.random.default_rng(11)
    arm_low = np.asarray((-2.6, 0.0, 0.0, -1.5, -1.5, -2.0))
    arm_high = np.asarray((3.0, 3.6, 3.6, 1.5, 1.5, 2.0))
    safe: list[np.ndarray] = []
    for _ in range(600):
        candidate = _pose()
        for side in ("left", "right"):
            for index, value in enumerate(rng.uniform(arm_low, arm_high)):
                candidate[DIM_LABELS.index(f"{side}_j{index}")] = value
        if not checker.check(candidate).collided:
            safe.append(candidate)
    assert len(safe) >= 20

    crossing: tuple[np.ndarray, np.ndarray] | None = None
    arm_indices = [index for index, label in enumerate(DIM_LABELS) if "gripper" not in label]
    for _ in range(4000):
        start, target = (safe[index] for index in rng.integers(0, len(safe), size=2))
        max_delta = float(np.max(np.abs(target[arm_indices] - start[arm_indices])))
        substeps = max(1, min(64, int(np.ceil(max_delta / config.sweep_resolution))))
        for step in range(1, substeps):
            report = checker.check(start + (target - start) * (step / substeps))
            if (
                report.collided
                and report.geom1 is not None
                and report.geom2 is not None
                and {report.geom1.split("_", 1)[0], report.geom2.split("_", 1)[0]}
                == {"left", "right"}
            ):
                crossing = start.copy(), target.copy()
                break
        if crossing is not None:
            break
    assert crossing is not None
    start, target = crossing
    assert not checker.check(start).collided
    assert not checker.check(target).collided

    configured_low = np.asarray(YamConfig().joint_low)
    configured_high = np.asarray(YamConfig().joint_high)
    for side in ("left", "right"):
        for index, (low, high) in enumerate(zip(arm_low, arm_high, strict=True)):
            configured_low[DIM_LABELS.index(f"{side}_j{index}")] = low
            configured_high[DIM_LABELS.index(f"{side}_j{index}")] = high
    yam_config = YamConfig(
        joint_low=tuple(configured_low),
        joint_high=tuple(configured_high),
        home_pose=tuple(start),
        collision_table=False,
    )
    embodiment = YAMEmbodiment(yam_config)
    contribution = embodiment.contribute_guardrails(embodiment.info.action_space)
    name, approver = contribution.approvers[0]
    assert name == "yam-collision"
    held = approver.review(Action(target), {})
    assert held.data == pytest.approx(start)
    assert held.meta["collision_blocked"] is True
    detail = held.meta["collision_detail"]
    pair = detail.split("@", 1)[0]
    assert {name.split("_", 1)[0] for name in pair.split(":")} == {"left", "right"}


def test_build_guardrails_rejects_inconsistent_yam_modes(joint_space: Box) -> None:
    with pytest.raises(ValueError, match="EEF mode"):
        build_yam_guardrails(joint_space, YamConfig(control_interface="eef_pos"))
    with pytest.raises(ValueError, match="joints_are_delta=True"):
        build_yam_guardrails(joint_space, YamConfig(joints_are_delta=True))


def test_build_guardrails_order_and_clamped_default_home(joint_space: Box) -> None:
    high = list(YamConfig().joint_high)
    high[DIM_LABELS.index("left_gripper")] = 0.5
    high[DIM_LABELS.index("right_gripper")] = 0.5
    config = YamConfig(joint_high=tuple(high), home_pose=None)
    chain = build_yam_guardrails(joint_space, config)
    clamp, delta, collision_approver = chain._approvers
    assert isinstance(chain, ChainApprover)
    assert isinstance(clamp, ClampApprover)
    assert isinstance(delta, DeltaLimitApprover)
    assert isinstance(collision_approver, CollisionApprover)
    expected = np.clip(HOME, config.low, config.high)
    assert collision_approver._start_pose == pytest.approx(expected)


def test_build_guardrails_uses_custom_home_and_integrates_chain(joint_space: Box) -> None:
    home = tuple(_pose(left_j0=0.01))
    config = YamConfig(home_pose=home)
    collision_config = CollisionConfig(table_height=None)
    chain = build_yam_guardrails(
        joint_space,
        config,
        collision_config,
        on_violation="hold",
    )
    collision_approver = chain._approvers[-1]
    assert collision_approver._start_pose == pytest.approx(home)
    assert collision_approver._checker.config is collision_config
    action = Action(_pose(left_j0=0.02))
    assert chain.review(action, {}) is action


def test_contribution_ladder_off_warns_to_measure_geometry(joint_space: Box) -> None:
    embodiment = YAMEmbodiment(YamConfig(collision_guardrail=False))

    contribution = embodiment.contribute_guardrails(joint_space)

    assert contribution.approvers == ()
    assert contribution.warnings == (
        "collision guardrail disabled by config; set collision_guardrail=true "
        "after measuring collision_*_base_pos",
    )


@pytest.mark.parametrize(
    ("config", "expected_warnings"),
    [
        (
            YamConfig(control_interface="eef_pos"),
            (_EEF_SKIP_WARNING, _EEF_PINNED_WARNING),
        ),
        (YamConfig(joints_are_delta=True), (_EEF_SKIP_WARNING,)),
    ],
    ids=["eef", "delta-joints"],
)
def test_contribution_ladder_skips_non_absolute_joint_modes(
    config: YamConfig,
    expected_warnings: tuple[str, ...],
) -> None:
    embodiment = YAMEmbodiment(config)

    contribution = embodiment.contribute_guardrails(embodiment.info.action_space)

    assert contribution.approvers == ()
    assert contribution.warnings == expected_warnings


def test_eef_pinned_warning_follows_guardrail_disabled_path_warning() -> None:
    embodiment = YAMEmbodiment(YamConfig(control_interface="eef_pos", collision_guardrail=False))

    contribution = embodiment.contribute_guardrails(embodiment.info.action_space)

    assert contribution.approvers == ()
    assert contribution.warnings == (_EEF_DISABLED_WARNING, _EEF_PINNED_WARNING)


@pytest.mark.parametrize(
    ("collision_guardrail", "path_warning"),
    [(True, _EEF_SKIP_WARNING), (False, _EEF_DISABLED_WARNING)],
)
def test_eef_orientation_notice_precedes_z_floor_warning_without_pin_warning(
    collision_guardrail: bool,
    path_warning: str,
) -> None:
    embodiment = YAMEmbodiment(
        YamConfig(
            control_interface="eef_pos",
            eef_orientation=True,
            collision_guardrail=collision_guardrail,
        )
    )

    contribution = embodiment.contribute_guardrails(embodiment.info.action_space)

    assert contribution.warnings == (
        path_warning,
        _EEF_ORIENTATION_WARNING,
        _EEF_Z_FLOOR_WARNING,
    )


def test_eef_warning_order_includes_notice_pin_and_z_floor_after_path() -> None:
    low = list(DEFAULT_EEF_LOW)
    high = list(DEFAULT_EEF_HIGH)
    low[4] = high[4] = 0.1
    config = YamConfig(
        control_interface="eef_pos",
        eef_orientation=True,
        eef_low=tuple(low),
        eef_high=tuple(high),
    )

    contribution = YAMEmbodiment(config).contribute_guardrails(action_box())

    assert contribution.warnings == (
        _EEF_SKIP_WARNING,
        _EEF_ORIENTATION_WARNING,
        "eef_pos: action dims left_pitch are pinned (low == high) and not commandable; "
        "widen eef_low/eef_high (eef_orientation=true opens only zero-pinned pitch/roll)",
        _EEF_Z_FLOOR_WARNING,
    )


def test_eef_orientation_notice_is_absent_when_flag_is_off_or_mode_is_joints() -> None:
    eef = YAMEmbodiment(YamConfig(control_interface="eef_pos"))
    joints = YAMEmbodiment(YamConfig(eef_orientation=True, collision_guardrail=False))

    assert eef.contribute_guardrails(eef.info.action_space).warnings == (
        _EEF_SKIP_WARNING,
        _EEF_PINNED_WARNING,
    )
    assert joints.contribute_guardrails(joints.info.action_space).warnings == (
        _EEF_DISABLED_WARNING,
    )


def test_eef_z_floor_warning_clears_after_both_arms_raise_their_floor() -> None:
    low = list(DEFAULT_EEF_LOW)
    low[2] += 0.01
    low[9] += 0.01
    config = YamConfig(
        control_interface="eef_pos",
        eef_orientation=True,
        eef_low=tuple(low),
    )

    contribution = YAMEmbodiment(config).contribute_guardrails(action_box())

    assert contribution.warnings == (_EEF_SKIP_WARNING, _EEF_ORIENTATION_WARNING)


def test_eef_z_floor_warning_couples_open_tilt_to_the_same_arm() -> None:
    low = list(DEFAULT_EEF_LOW)
    high = list(DEFAULT_EEF_HIGH)
    low[2] += 0.01
    low[4], high[4] = -0.2, 0.2
    config = YamConfig(control_interface="eef_pos", eef_low=tuple(low), eef_high=tuple(high))

    contribution = YAMEmbodiment(config).contribute_guardrails(action_box())

    assert contribution.warnings == (
        _EEF_SKIP_WARNING,
        "eef_pos: action dims left_roll, right_pitch, right_roll are pinned "
        "(low == high) and not commandable; widen eef_low/eef_high "
        "(eef_orientation=true opens only zero-pinned pitch/roll)",
    )


@pytest.mark.parametrize("right_z", [DEFAULT_EEF_LOW[9], DEFAULT_EEF_LOW[9] - 0.01])
def test_eef_z_floor_warning_fires_for_open_right_tilt_at_default_or_lowered_z(
    right_z: float,
) -> None:
    low = list(DEFAULT_EEF_LOW)
    high = list(DEFAULT_EEF_HIGH)
    low[9] = right_z
    low[12], high[12] = -0.2, 0.2
    config = YamConfig(control_interface="eef_pos", eef_low=tuple(low), eef_high=tuple(high))

    contribution = YAMEmbodiment(config).contribute_guardrails(action_box())

    assert contribution.warnings[-1] == _EEF_Z_FLOOR_WARNING


def test_joints_mode_has_no_eef_warnings_even_with_open_orientation_tuples() -> None:
    config = YamConfig(eef_orientation=True, collision_guardrail=False)

    contribution = YAMEmbodiment(config).contribute_guardrails(action_box())

    assert contribution.warnings == (_EEF_DISABLED_WARNING,)


def test_contribution_ladder_reports_missing_mujoco_before_construction(
    joint_space: Box,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = False

    def fail_if_constructed(*args: Any, **kwargs: Any) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("collision construction must not run")

    monkeypatch.setattr(embodiment_module.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(collision, "_collision_approver", fail_if_constructed)

    contribution = YAMEmbodiment().contribute_guardrails(joint_space)

    assert not constructed
    assert contribution.approvers == ()
    assert contribution.warnings == (
        "collision guardrail skipped: MuJoCo is unavailable; install it with: "
        'pip install "inspect-robots-yam[collision]"',
    )


def test_contribution_happy_path_is_named_hold_with_default_geometry_warning(
    joint_space: Box,
) -> None:
    contribution = YAMEmbodiment().contribute_guardrails(joint_space)

    assert len(contribution.approvers) == 1
    name, approver = contribution.approvers[0]
    assert name == "yam-collision"
    assert isinstance(approver, CollisionApprover)
    assert approver._on_violation == "hold"
    assert contribution.warnings == (
        "collision guardrail uses library-default geometry fields: "
        "collision_left_base_pos, collision_right_base_pos, "
        "collision_left_base_yaw, collision_right_base_yaw, "
        "collision_table_height, collision_penetration_threshold",
    )


def test_contribution_propagates_malformed_model(
    joint_space: Box,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CollisionChecker,
        "_read_model_xml",
        staticmethod(lambda: "<mujoco><broken>"),
    )

    with pytest.raises(ValueError, match="malformed or incompatible"):
        YAMEmbodiment().contribute_guardrails(joint_space)


def test_contribution_propagates_colliding_home_with_remedies(joint_space: Box) -> None:
    embodiment = YAMEmbodiment(
        YamConfig(
            collision_left_base_pos=(0.0, 0.0, 0.0),
            collision_right_base_pos=(0.0, 0.0, 0.0),
            collision_table=False,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"start_pose is already in collision.*collision_guardrail=false.*collision_\*",
    ):
        embodiment.contribute_guardrails(joint_space)


def test_contribution_maps_configured_geometry_to_checker(joint_space: Box) -> None:
    config = YamConfig(
        collision_left_base_pos=(0.0, 0.31, 0.02),
        collision_right_base_pos=(0.0, -0.32, 0.03),
        collision_left_base_yaw=0.1,
        collision_right_base_yaw=-0.2,
        collision_table_height=-0.01,
        collision_penetration_threshold=0.004,
    )

    contribution = YAMEmbodiment(config).contribute_guardrails(joint_space)
    _, approver = contribution.approvers[0]
    assert isinstance(approver, CollisionApprover)
    assert approver._checker.config == CollisionConfig(
        left_base_pos=(0.0, 0.31, 0.02),
        right_base_pos=(0.0, -0.32, 0.03),
        left_base_yaw=0.1,
        right_base_yaw=-0.2,
        table_height=-0.01,
        penetration_threshold=0.004,
    )
    assert contribution.warnings == ()


@pytest.mark.parametrize("position", ["0.0,0.3", "0.0,nan,0.0"])
def test_contribution_delegates_geometry_validation_to_collision_config(
    joint_space: Box,
    position: str,
) -> None:
    config = YamConfig.from_kwargs(collision_left_base_pos=position)

    with pytest.raises(ValueError, match="left_base_pos must contain three finite coordinates"):
        YAMEmbodiment(config).contribute_guardrails(joint_space)


def test_collision_table_false_removes_table_from_contributed_checker(
    joint_space: Box,
) -> None:
    contribution = YAMEmbodiment(YamConfig(collision_table=False)).contribute_guardrails(
        joint_space
    )
    _, approver = contribution.approvers[0]
    assert isinstance(approver, CollisionApprover)
    assert approver._checker.config.table_height is None
    geom_names = {
        approver._checker._mujoco.mj_id2name(
            approver._checker._model,
            approver._checker._mujoco.mjtObj.mjOBJ_GEOM,
            index,
        )
        for index in range(approver._checker._model.ngeom)
    }
    assert "table" not in geom_names
    assert "collision_table_height" not in contribution.warnings[0]


def test_geometry_warning_stands_with_one_base_position_and_lifts_with_both(
    joint_space: Box,
) -> None:
    one_position = YAMEmbodiment(
        YamConfig(
            collision_left_base_pos=(0.0, 0.3, 0.0),
            collision_table_height=0.0,
        )
    ).contribute_guardrails(joint_space)
    assert one_position.warnings == (
        "collision guardrail uses library-default geometry fields: "
        "collision_right_base_pos, collision_left_base_yaw, "
        "collision_right_base_yaw, collision_penetration_threshold",
    )

    one_position_default_table = YAMEmbodiment(
        YamConfig(collision_left_base_pos=(0.0, 0.3, 0.0))
    ).contribute_guardrails(joint_space)
    assert one_position_default_table.warnings == (
        "collision guardrail uses library-default geometry fields: "
        "collision_right_base_pos, collision_left_base_yaw, "
        "collision_right_base_yaw, collision_table_height, "
        "collision_penetration_threshold",
    )

    both_positions = YAMEmbodiment(
        YamConfig(
            collision_left_base_pos=(0.0, 0.3, 0.0),
            collision_right_base_pos=(0.0, -0.3, 0.0),
        )
    ).contribute_guardrails(joint_space)
    assert both_positions.warnings == ()


def test_build_and_contribution_collision_approvers_are_equivalent(joint_space: Box) -> None:
    config = YamConfig(
        home_pose=tuple(_pose(left_j0=0.01)),
        collision_left_base_pos=(0.0, 0.3, 0.0),
        collision_right_base_pos=(0.0, -0.3, 0.0),
        collision_table=False,
    )

    built = build_yam_guardrails(joint_space, config)._approvers[-1]
    _, contributed = YAMEmbodiment(config).contribute_guardrails(joint_space).approvers[0]

    assert isinstance(built, CollisionApprover)
    assert isinstance(contributed, CollisionApprover)
    assert built._start_pose == pytest.approx(contributed._start_pose)
    assert built._checker.config == contributed._checker.config
    assert built._on_violation == contributed._on_violation == "hold"


@pytest.mark.perf
def test_bimanual_check_stays_under_one_millisecond_per_configuration(
    checker: CollisionChecker,
) -> None:
    iterations = 500
    start = time.perf_counter()
    for _ in range(iterations):
        checker.check(HOME)
    elapsed = time.perf_counter() - start
    assert elapsed / iterations < 0.001
