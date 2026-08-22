"""Tests for YamConfig and ActServerConfig."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from inspect_robots.spaces import CameraSpec

from inspect_robots_yam.config import (
    _DEFAULT_HIGH,
    _DEFAULT_LOW,
    DEFAULT_CAMERAS,
    DEFAULT_EEF_HIGH,
    DEFAULT_EEF_HOME_POSE,
    DEFAULT_EEF_LOW,
    DEFAULT_JOINT_HOME_POSE,
    DEFAULT_REST_POSE,
    EEF_DIM_LABELS,
    ActServerConfig,
    MolmoActConfig,
    YamConfig,
    action_box,
    camera_specs,
    observation_space,
)


def test_yam_defaults() -> None:
    cfg = YamConfig()
    assert cfg.left_channel == "can0"
    assert cfg.right_channel == "can1"
    assert cfg.control_hz == 10.0
    assert cfg.low.shape == (14,)
    assert cfg.high.shape == (14,)
    # gripper slot (index 6) bounded [0, 1]; joints bounded by +/-pi.
    assert cfg.low[6] == 0.0 and cfg.high[6] == 1.0
    assert cfg.low[0] == pytest.approx(-np.pi)
    assert cfg.control_interface == "joints"


def test_pose_config_defaults() -> None:
    cfg = YamConfig()
    assert cfg.start_pose is None
    assert cfg.pose_dir == "poses"


def test_molmo_defaults_and_url() -> None:
    cfg = ActServerConfig()
    assert cfg.num_steps == 10
    assert cfg.action_horizon == 30
    assert cfg.state_key == "joint_pos"
    assert cfg.camera_order == DEFAULT_CAMERAS
    assert cfg.name == "molmoact2"
    assert cfg.url == "http://127.0.0.1:8202/act"


def test_molmo_url_strips_trailing_slash() -> None:
    cfg = MolmoActConfig(server_url="http://host:9000/")
    assert cfg.url == "http://host:9000/act"


def test_molmo_url_adds_missing_endpoint_slash() -> None:
    cfg = MolmoActConfig(server_url="http://host:9000", endpoint="act")
    assert cfg.url == "http://host:9000/act"  # not "http://host:9000act"


def test_from_kwargs_populates_scalars() -> None:
    cfg = ActServerConfig.from_kwargs(
        server_url="http://gpu:8202", num_steps=20, name="remote-model"
    )
    assert cfg.server_url == "http://gpu:8202"
    assert cfg.num_steps == 20
    assert cfg.name == "remote-model"


def test_yam_from_kwargs() -> None:
    cfg = YamConfig.from_kwargs(left_channel="canA", control_hz=25.0)
    assert cfg.left_channel == "canA"
    assert cfg.control_hz == 25.0


def test_start_pose_and_home_pose_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="start_pose and home_pose"):
        YamConfig(start_pose="ready", home_pose=(0.0,) * 14)


def test_start_pose_accepted_with_eef_control() -> None:
    cfg = YamConfig(start_pose="ready", control_interface="eef_pos")
    assert cfg.start_pose == "ready"


@pytest.mark.parametrize("name", ["", "   ", ".hidden", "../evil", "x" * 65])
def test_start_pose_rejects_invalid_names_with_rule(name: str) -> None:
    with pytest.raises(ValueError, match=r"must match \^\[A-Za-z0-9\]"):
        YamConfig(start_pose=name)


@pytest.mark.parametrize("pose_dir", ["", "   "])
def test_pose_dir_must_be_nonempty(pose_dir: str) -> None:
    with pytest.raises(ValueError, match="pose_dir must be a non-empty string"):
        YamConfig(pose_dir=pose_dir)


@pytest.mark.parametrize("field", ["start_pose", "pose_dir"])
def test_pose_strings_reject_scalar_coercion_with_config_hint(field: str) -> None:
    with pytest.raises(ValueError, match=rf"{field}.*quote.*config.ini"):
        YamConfig.from_kwargs(**{field: 42})


def test_pose_fields_land_through_from_kwargs() -> None:
    cfg = YamConfig.from_kwargs(start_pose="007", pose_dir="42")
    assert cfg.start_pose == "007"
    assert cfg.pose_dir == "42"


@pytest.mark.parametrize("field", ["start_pose", "pose_dir"])
def test_pose_strings_treat_none_as_unset(field: str) -> None:
    cfg = YamConfig.from_kwargs(**{field: None})
    assert getattr(cfg, field) == getattr(YamConfig(), field)


def test_gripper_stroke_defaults_to_one_second_and_point_one_step() -> None:
    cfg = YamConfig()
    assert cfg.gripper_stroke_s == 1.0
    assert cfg.gripper_max_step == pytest.approx(0.1)


@pytest.mark.parametrize("gripper_stroke_s", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_gripper_stroke_validation(gripper_stroke_s: float) -> None:
    with pytest.raises(ValueError, match="gripper_stroke_s must be finite and > 0"):
        YamConfig(gripper_stroke_s=gripper_stroke_s)


def test_gripper_max_step_caps_at_one_step() -> None:
    assert YamConfig(gripper_stroke_s=0.01).gripper_max_step == pytest.approx(1.0)


@pytest.mark.parametrize("control_hz", [0.0, -5.0])
def test_gripper_max_step_nonpositive_hz_falls_back(control_hz: float) -> None:
    assert YamConfig(control_hz=control_hz).gripper_max_step == pytest.approx(0.1)


@pytest.mark.parametrize("control_hz", [np.nan, np.inf, -np.inf])
def test_gripper_max_step_nonfinite_hz_declines_to_declare(control_hz: float) -> None:
    assert YamConfig(control_hz=control_hz).gripper_max_step is None


def test_gripper_max_step_nonfinite_product_or_result_declines_to_declare() -> None:
    assert YamConfig(gripper_stroke_s=1e308, control_hz=1e308).gripper_max_step is None
    smallest_positive = float.fromhex("0x0.0000000000001p-1022")
    assert YamConfig(gripper_stroke_s=smallest_positive, control_hz=1.0).gripper_max_step is None


def test_from_kwargs_rejects_unknown() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        MolmoActConfig.from_kwargs(nope=1)


def test_yam_rejects_bad_joint_limits() -> None:
    with pytest.raises(ValueError, match="joint_low must have 14 entries"):
        YamConfig(joint_low=(0.0,) * 13)


def test_yam_rejects_bad_home_pose() -> None:
    with pytest.raises(ValueError, match="home_pose must have 14 entries"):
        YamConfig(home_pose=(0.0,) * 10)


def test_yam_control_interface_validation() -> None:
    with pytest.raises(ValueError, match=r"control_interface.*eef_pos.*joints"):
        YamConfig(control_interface="cartesian")
    with pytest.raises(ValueError, match="joints_are_delta"):
        YamConfig(control_interface="eef_pos", joints_are_delta=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eef_low": (0.0,) * 13}, "eef_low must have 14"),
        ({"eef_high": (1.0,) * 13}, "eef_high must have 14"),
        (
            {"eef_high": (*DEFAULT_EEF_HIGH[:2], np.nan, *DEFAULT_EEF_HIGH[3:])},
            "only finite values",
        ),
        (
            {"eef_low": (0.5, *DEFAULT_EEF_LOW[1:])},
            "eef_low must not exceed eef_high",
        ),
        (
            {"eef_low": (*DEFAULT_EEF_LOW[:3], -np.pi - 0.01, *DEFAULT_EEF_LOW[4:])},
            "yaw and roll bounds must stay within",
        ),
        (
            {"eef_high": (*DEFAULT_EEF_HIGH[:10], np.pi + 0.01, *DEFAULT_EEF_HIGH[11:])},
            "yaw and roll bounds must stay within",
        ),
        (
            {"eef_high": (*DEFAULT_EEF_HIGH[:12], np.pi + 0.01, *DEFAULT_EEF_HIGH[13:])},
            "yaw and roll bounds must stay within",
        ),
        (
            {"eef_low": (*DEFAULT_EEF_LOW[:4], -np.pi / 2, *DEFAULT_EEF_LOW[5:])},
            "pitch bounds must stay strictly inside",
        ),
        (
            {"eef_high": (*DEFAULT_EEF_HIGH[:11], np.pi / 2, *DEFAULT_EEF_HIGH[12:])},
            "pitch bounds must stay strictly inside",
        ),
        ({"ik_max_iters": 0}, "ik_max_iters must be a positive integer"),
        ({"ik_max_iters": 1.5}, "ik_max_iters must be a positive integer"),
        ({"ik_max_iters": True}, "ik_max_iters must be a positive integer"),
        ({"ik_step_joint_limit": 0.0}, "ik_step_joint_limit must be finite and > 0"),
        ({"cmd_resync_threshold": np.inf}, "cmd_resync_threshold must be finite and > 0"),
        ({"osc_deadband": -0.1}, "osc_deadband must be finite and >= 0"),
        ({"osc_reversals": -1}, "osc_reversals must be a non-negative integer"),
        ({"osc_reversals": 1.5}, "osc_reversals must be a non-negative integer"),
        ({"osc_reversals": False}, "osc_reversals must be a non-negative integer"),
        ({"osc_window": 0}, "osc_window must be a positive integer"),
        ({"osc_window": 1.5}, "osc_window must be a positive integer"),
        ({"osc_window": True}, "osc_window must be a positive integer"),
        ({"osc_reversals": 6}, "osc_reversals must be less than osc_window"),
        ({"osc_hold_steps": 0}, "osc_hold_steps must be a positive integer"),
        ({"osc_hold_steps": False}, "osc_hold_steps must be a positive integer"),
    ],
)
def test_eef_config_knob_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        YamConfig(control_interface="eef_pos", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"settle_tolerance": 0.0}, "settle_tolerance must be finite and > 0"),
        ({"settle_tolerance": -0.01}, "settle_tolerance must be finite and > 0"),
        ({"settle_tolerance": np.inf}, "settle_tolerance must be finite and > 0"),
        ({"settle_timeout_s": 0.0}, "settle_timeout_s must be finite and > 0"),
        ({"settle_timeout_s": np.nan}, "settle_timeout_s must be finite and > 0"),
        ({"settle_timeout_budget": 0}, "settle_timeout_budget must be a positive integer"),
        ({"settle_timeout_budget": 1.5}, "settle_timeout_budget must be a positive integer"),
        ({"settle_timeout_budget": True}, "settle_timeout_budget must be a positive integer"),
    ],
)
def test_settle_config_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        YamConfig(**kwargs)


def test_settle_defaults_are_off() -> None:
    cfg = YamConfig()
    assert cfg.settle_tolerance is None  # opt-in, so VLA cadence is untouched
    assert cfg.settle_timeout_s == pytest.approx(1.0)
    assert cfg.settle_timeout_budget == 20
    # Scalars arrive already typed (the CLI parses them); from_kwargs only
    # coerces the comma-separated tuple fields.
    assert YamConfig.from_kwargs(settle_tolerance=0.05).settle_tolerance == pytest.approx(0.05)


def test_eef_config_defaults_and_cli_tuple_overrides() -> None:
    cfg = YamConfig.from_kwargs(
        control_interface="eef_pos",
        eef_low=",".join(str(value) for value in DEFAULT_EEF_LOW),
        eef_high=",".join(str(value) for value in DEFAULT_EEF_HIGH),
    )
    assert cfg.eef_low_array.shape == (14,)
    assert cfg.eef_high_array.shape == (14,)
    assert cfg.ik_max_iters == 20
    assert cfg.ik_step_joint_limit == pytest.approx(0.2)
    assert cfg.cmd_resync_threshold == pytest.approx(0.35)
    assert cfg.osc_deadband == pytest.approx(0.005)
    assert cfg.osc_reversals == 2
    assert cfg.osc_window == 6
    assert cfg.osc_hold_steps == 10


def test_default_eef_home_pose_has_provisional_joint_values_and_open_grippers() -> None:
    assert len(DEFAULT_EEF_HOME_POSE) == 14
    assert DEFAULT_EEF_HOME_POSE[:6] == pytest.approx(
        (-0.024, 0.794, 0.645, -0.375, -0.021, -0.012)
    )
    assert DEFAULT_EEF_HOME_POSE[6] == DEFAULT_EEF_HOME_POSE[13] == 1.0
    assert DEFAULT_EEF_HOME_POSE[7:13] == pytest.approx(DEFAULT_EEF_HOME_POSE[:6])


def test_default_joint_home_pose_is_zero_joints_open_grippers() -> None:
    assert len(DEFAULT_JOINT_HOME_POSE) == 14
    assert DEFAULT_JOINT_HOME_POSE[:6] == pytest.approx((0.0,) * 6)
    assert DEFAULT_JOINT_HOME_POSE[6] == DEFAULT_JOINT_HOME_POSE[13] == 1.0
    assert DEFAULT_JOINT_HOME_POSE[7:13] == pytest.approx(DEFAULT_JOINT_HOME_POSE[:6])
    assert DEFAULT_REST_POSE == DEFAULT_JOINT_HOME_POSE


def test_eef_action_space_shape_labels_bounds_and_semantics() -> None:
    space = action_box(
        low=np.asarray(DEFAULT_EEF_LOW),
        high=np.asarray(DEFAULT_EEF_HIGH),
        control_interface="eef_pos",
    )
    assert space.shape == (14,)
    assert space.low is not None and np.array_equal(space.low, DEFAULT_EEF_LOW)
    assert space.high is not None and np.array_equal(space.high, DEFAULT_EEF_HIGH)
    assert space.semantics.control_mode == "eef_abs_pose"
    assert space.semantics.rotation_repr == "none"
    assert space.semantics.gripper == "continuous"
    assert space.semantics.frame == "base"
    assert space.semantics.dim_labels == EEF_DIM_LABELS


def test_absolute_action_spaces_declare_gripper_max_step_only_at_grippers() -> None:
    joint_space = action_box(gripper_max_step=0.25)
    assert joint_space.semantics is not None
    assert joint_space.semantics.max_step == tuple(
        0.25 if index in (6, 13) else None for index in range(14)
    )

    eef_space = action_box(control_interface="eef_pos", gripper_max_step=0.25)
    assert eef_space.semantics is not None
    assert eef_space.semantics.max_step == tuple(
        0.25 if index in (6, 13) else None for index in range(len(EEF_DIM_LABELS))
    )


def test_pinned_gripper_bounds_decline_the_declaration_per_dim() -> None:
    # A gripper pinned via custom bounds (low == high) constructs today; the
    # declaration must decline on exactly that dim instead of tripping core's
    # pinned-dim rejection at Box construction.
    low = np.array(YamConfig().low)
    high = np.array(YamConfig().high)
    high[6] = low[6]  # pin the left gripper only
    space = action_box(low, high, gripper_max_step=0.25)
    assert space.semantics is not None
    assert space.semantics.max_step == tuple(0.25 if index == 13 else None for index in range(14))

    high[13] = low[13]  # pin both grippers: nothing left to declare
    space = action_box(low, high, gripper_max_step=0.25)
    assert space.semantics is not None
    assert space.semantics.max_step is None


def test_delta_and_unspecified_action_spaces_declare_no_max_step() -> None:
    delta_space = action_box(joints_are_delta=True, gripper_max_step=0.25)
    assert delta_space.semantics is not None
    assert delta_space.semantics.max_step is None

    unspecified_space = action_box()
    assert unspecified_space.semantics is not None
    assert unspecified_space.semantics.max_step is None


def test_eef_pitch_and_roll_default_to_pinned_zero_bounds() -> None:
    # Equality means a pinned axis: the default layout carries pitch/roll but
    # they are not commandable until an operator widens their bounds.
    cfg = YamConfig(control_interface="eef_pos")
    for index in (4, 5, 11, 12):
        assert cfg.eef_low_array[index] == cfg.eef_high_array[index] == 0.0
    opened = YamConfig(
        control_interface="eef_pos",
        eef_high=(*DEFAULT_EEF_HIGH[:4], 0.8, 0.5, *DEFAULT_EEF_HIGH[6:]),
        eef_low=(*DEFAULT_EEF_LOW[:4], -0.8, -0.5, *DEFAULT_EEF_LOW[6:]),
    )
    assert opened.eef_high_array[4] == pytest.approx(0.8)
    assert opened.eef_low_array[5] == pytest.approx(-0.5)


def test_eef_observation_space_declares_joint_and_eef_state_once() -> None:
    space = observation_space(224, 224, DEFAULT_CAMERAS, control_interface="eef_pos")
    assert space.state_keys == frozenset({"joint_pos", "eef_state"})
    assert space.state is not None
    fields = {field.key: field.shape for field in space.state.fields}
    assert fields == {"joint_pos": (14,), "eef_state": (14,)}


def test_yam_operational_defaults() -> None:
    cfg = YamConfig()
    assert cfg.gripper_type == "LINEAR_4310"  # i2rt GripperType enum *name*
    assert cfg.zero_gravity_mode is True
    assert cfg.unattended is False


def test_yam_rejects_unsupported_gripper_type() -> None:
    # NO_GRIPPER / YAM_TEACHING_HANDLE would break the 7-D-per-arm packing contract.
    with pytest.raises(ValueError, match="gripper_type 'NO_GRIPPER' is not supported"):
        YamConfig(gripper_type="NO_GRIPPER")


def test_yam_rejects_gripper_type_enum_value_spelling() -> None:
    # The seam does a GripperType[...] NAME lookup; lowercase enum *values* must
    # be rejected here rather than exploding at driver-connect time.
    with pytest.raises(ValueError, match="not supported"):
        YamConfig(gripper_type="linear_4310")


def test_yam_rejects_equal_gripper_calibration() -> None:
    with pytest.raises(ValueError, match="gripper_open and gripper_closed must differ"):
        YamConfig(gripper_open=0.5, gripper_closed=0.5)


def test_yam_accepts_valid_home_pose() -> None:
    cfg = YamConfig(home_pose=(0.0,) * 14)
    assert cfg.home_pose is not None and len(cfg.home_pose) == 14


def test_camera_specs() -> None:
    specs = camera_specs(224, 224, DEFAULT_CAMERAS)
    assert len(specs) == 3
    assert all(isinstance(s, CameraSpec) for s in specs)
    assert specs[0].name == "top_cam"
    assert specs[0].height == 224 and specs[0].width == 224


def test_yam_rest_defaults() -> None:
    cfg = YamConfig()
    assert cfg.rest_pose == DEFAULT_REST_POSE
    assert cfg.rest_secs == 3.0


def test_default_rest_pose_is_valid_for_default_limits() -> None:
    assert len(DEFAULT_REST_POSE) == 14
    assert all(0.0 <= DEFAULT_REST_POSE[index] <= 1.0 for index in (6, 13))
    arm_indices = set(range(14)) - {6, 13}
    assert all(
        _DEFAULT_LOW[index] <= DEFAULT_REST_POSE[index] <= _DEFAULT_HIGH[index]
        for index in arm_indices
    )
    assert YamConfig(rest_pose=DEFAULT_REST_POSE).rest_pose == DEFAULT_REST_POSE


def test_yam_rejects_bad_rest_pose() -> None:
    with pytest.raises(ValueError, match="rest_pose must have 14 entries"):
        YamConfig(rest_pose=(0.0,) * 3)


def test_yam_rejects_nonpositive_rest_secs() -> None:
    with pytest.raises(ValueError, match="rest_secs must be > 0"):
        YamConfig(rest_secs=0.0)


def test_yam_max_steps_hint_default_and_validation() -> None:
    assert YamConfig().max_steps_hint is None
    with pytest.warns(FutureWarning, match="max_steps_hint"):
        assert YamConfig(max_steps_hint=1200).max_steps_hint == 1200
    # Invalid values raise without ever warning (validation precedes the
    # deprecation warning).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="max_steps_hint must be >= 1"):
            YamConfig(max_steps_hint=0)


def test_yam_camera_devices_default_none() -> None:
    cfg = YamConfig()
    assert cfg.top_cam_device is None
    assert cfg.left_cam_device is None
    assert cfg.right_cam_device is None


def test_yam_camera_sources_all_slots_or_none() -> None:
    with pytest.raises(ValueError, match="unsourced slots: left, right"):
        YamConfig(top_cam_device="/dev/video0")
    cfg = YamConfig(
        top_cam_device="/dev/video0",
        left_cam_device="/dev/video2",
        right_cam_device="/dev/video4",
    )
    assert cfg.left_cam_device == "/dev/video2"


def test_yam_duplicate_camera_devices_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "top_cam_device and left_cam_device must be different; "
            "duplicate camera device '/dev/video0'"
        ),
    ):
        YamConfig(
            top_cam_device="/dev/video0",
            left_cam_device="/dev/video0",
            right_cam_device="/dev/video4",
        )


def test_yam_duplicate_depth_serials_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "top_depth_serial and left_depth_serial must be different; "
            "duplicate depth serial 'SAME'"
        ),
    ):
        YamConfig(
            top_depth_serial="SAME",
            left_depth_serial="SAME",
            right_depth_serial="S3",
        )


def test_action_semantics_is_config_dependent_with_labels() -> None:
    from inspect_robots_yam.config import action_semantics
    from inspect_robots_yam.packing import DIM_LABELS

    absolute = action_semantics(joints_are_delta=False)
    assert absolute.control_mode == "joint_pos"
    assert absolute.dim_labels == DIM_LABELS
    delta = action_semantics(joints_are_delta=True)
    assert delta.control_mode == "joint_delta"
    assert delta.dim_labels == DIM_LABELS


def test_dim_labels_shape_and_order() -> None:
    from inspect_robots_yam.packing import DIM_LABELS, TOTAL_DIM

    assert len(DIM_LABELS) == TOTAL_DIM
    assert DIM_LABELS[0] == "left_j0"
    assert DIM_LABELS[6] == "left_gripper"
    assert DIM_LABELS[7] == "right_j0"
    assert DIM_LABELS[13] == "right_gripper"


def test_step_limits_default_and_validation() -> None:
    import numpy as np

    cfg = YamConfig()
    assert len(cfg.step_limits) == 14
    assert cfg.step_limits[0] == pytest.approx(0.2)  # rad per step, per joint
    assert cfg.step_limits[6] == pytest.approx(1.0)  # normalized gripper stroke
    assert np.allclose(cfg.delta_low, -np.asarray(cfg.step_limits))
    assert np.allclose(cfg.delta_high, np.asarray(cfg.step_limits))
    with pytest.raises(ValueError, match="step_limits"):
        YamConfig(step_limits=(0.1,) * 13)
    with pytest.raises(ValueError, match="step_limits"):
        YamConfig(step_limits=(0.1,) * 13 + (-0.1,))


def test_molmoact_config_gains_delta_flag() -> None:
    from inspect_robots_yam.config import MolmoActConfig

    assert MolmoActConfig().joints_are_delta is False
    assert MolmoActConfig(joints_are_delta=True).joints_are_delta is True


def test_pose_fields_parse_comma_strings_from_flat_kwargs() -> None:
    import numpy as np

    csv = ",".join(["0.1"] * 6 + ["1.0"] + ["0.2"] * 6 + ["0.9"])
    cfg = YamConfig.from_kwargs(rest_pose=csv)
    assert isinstance(cfg.rest_pose, tuple) and len(cfg.rest_pose) == 14
    assert cfg.rest_pose[0] == pytest.approx(0.1)
    assert cfg.rest_pose[13] == pytest.approx(0.9)
    # Spaces tolerated; other pose-shaped fields parse the same way.
    cfg = YamConfig.from_kwargs(home_pose=" 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1 ")
    assert cfg.home_pose is not None and cfg.home_pose[6] == pytest.approx(1.0)
    cfg = YamConfig.from_kwargs(step_limits=csv)
    assert np.allclose(cfg.delta_high[0], 0.1)


def test_pose_string_parse_errors_are_guided() -> None:
    with pytest.raises(ValueError, match="rest_pose"):
        YamConfig.from_kwargs(rest_pose="0.1,zoom,0.3")
    with pytest.raises(ValueError, match="rest_pose"):
        YamConfig.from_kwargs(rest_pose="")
    with pytest.raises(ValueError, match="rest_pose must have 14"):
        YamConfig.from_kwargs(rest_pose="0.1,0.2")


def test_pose_fields_still_accept_real_tuples() -> None:
    cfg = YamConfig.from_kwargs(rest_pose=(0.0,) * 14)
    assert cfg.rest_pose == (0.0,) * 14


def test_auto_start_defaults_off_and_binds_via_kwargs() -> None:
    assert YamConfig().auto_start is False
    assert YamConfig.from_kwargs(auto_start=True).auto_start is True


def test_report_joint_eff_defaults_off_and_binds_via_kwargs() -> None:
    assert YamConfig().report_joint_eff is False
    assert YamConfig.from_kwargs(report_joint_eff=True).report_joint_eff is True


@pytest.mark.parametrize("value", [None, "yes", 1])
def test_report_joint_eff_rejects_non_bool_values(value: object) -> None:
    with pytest.raises(ValueError, match="report_joint_eff must be true or false"):
        YamConfig.from_kwargs(report_joint_eff=value)


@pytest.mark.parametrize("value", [True, False])
def test_park_before_grade_accepts_explicit_bool(value: bool) -> None:
    assert YamConfig.from_kwargs(park_before_grade=value).park_before_grade is value


@pytest.mark.parametrize("value", [None, "yes", 1])
def test_park_before_grade_rejects_non_bool_values(value: object) -> None:
    with pytest.raises(ValueError, match="park_before_grade must be true or false"):
        YamConfig.from_kwargs(park_before_grade=value)


def test_collision_guardrail_defaults_on_and_binds_via_kwargs() -> None:
    assert YamConfig().collision_guardrail is True
    assert YamConfig.from_kwargs(collision_guardrail=False).collision_guardrail is False
    assert YamConfig.from_kwargs(collision_table=False).collision_table is False


def test_collision_geometry_parses_comma_strings_and_scalars() -> None:
    cfg = YamConfig.from_kwargs(
        collision_left_base_pos="0.1, 0.2, 0.3",
        collision_right_base_pos="-0.1,-0.2,-0.3",
        collision_left_base_yaw=0.25,
        collision_right_base_yaw=-0.25,
        collision_table_height=0.02,
        collision_penetration_threshold=0.004,
    )

    assert cfg.collision_left_base_pos == pytest.approx((0.1, 0.2, 0.3))
    assert cfg.collision_right_base_pos == pytest.approx((-0.1, -0.2, -0.3))
    assert cfg.collision_left_base_yaw == pytest.approx(0.25)
    assert cfg.collision_right_base_yaw == pytest.approx(-0.25)
    assert cfg.collision_table is True
    assert cfg.collision_table_height == pytest.approx(0.02)
    assert cfg.collision_penetration_threshold == pytest.approx(0.004)


def test_collision_geometry_string_parse_errors_are_guided() -> None:
    with pytest.raises(ValueError, match="collision_left_base_pos"):
        YamConfig.from_kwargs(collision_left_base_pos="0.1,sideways,0.3")


def test_collision_table_height_none_names_explicit_table_switch() -> None:
    with pytest.raises(ValueError, match="collision_table=false"):
        YamConfig.from_kwargs(collision_table_height=None)


def test_collision_table_height_rejects_disabled_table_contradiction() -> None:
    with pytest.raises(ValueError, match="contradicts collision_table=false"):
        YamConfig.from_kwargs(collision_table=False, collision_table_height=0.1)


@pytest.mark.parametrize("flag", ["collision_guardrail", "collision_table"])
@pytest.mark.parametrize("value", [None, "yes", 1])
def test_collision_bool_flags_reject_non_bool_values(flag: str, value: object) -> None:
    # The CLI parses the literal `none` to Python None; unvalidated it would
    # falsy-disable a safety gate (or drop the table plane) instead of meaning
    # "library default" as it does for every other collision_* field.
    with pytest.raises(ValueError, match=f"{flag} must be true or false"):
        YamConfig.from_kwargs(**{flag: value})


def test_default_collision_flag_does_not_reject_unsupported_contribution_modes() -> None:
    assert YamConfig(control_interface="eef_pos").collision_guardrail is True
    assert YamConfig(joints_are_delta=True).collision_guardrail is True


def test_collision_hold_limit_default_and_validation() -> None:
    assert YamConfig().collision_hold_limit == 50
    assert YamConfig(collision_hold_limit=None).collision_hold_limit is None
    assert YamConfig(collision_hold_limit=0).collision_hold_limit == 0
    assert YamConfig(collision_hold_limit=100).collision_hold_limit == 100

    msg = "collision_hold_limit must be a non-negative integer or None"
    for invalid in (-1, -50, "50", 50.0, True):
        with pytest.raises(ValueError, match=msg):
            YamConfig(collision_hold_limit=invalid)  # type: ignore[arg-type]
