"""Tests for YamConfig / MolmoActConfig."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_robots.spaces import CameraSpec

from inspect_robots_yam.config import (
    DEFAULT_CAMERAS,
    MolmoActConfig,
    YamConfig,
    camera_specs,
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


def test_molmo_defaults_and_url() -> None:
    cfg = MolmoActConfig()
    assert cfg.num_steps == 10
    assert cfg.action_horizon == 30
    assert cfg.state_key == "joint_pos"
    assert cfg.camera_order == DEFAULT_CAMERAS
    assert cfg.url == "http://127.0.0.1:8202/act"


def test_molmo_url_strips_trailing_slash() -> None:
    cfg = MolmoActConfig(server_url="http://host:9000/")
    assert cfg.url == "http://host:9000/act"


def test_molmo_url_adds_missing_endpoint_slash() -> None:
    cfg = MolmoActConfig(server_url="http://host:9000", endpoint="act")
    assert cfg.url == "http://host:9000/act"  # not "http://host:9000act"


def test_from_kwargs_populates_scalars() -> None:
    cfg = MolmoActConfig.from_kwargs(server_url="http://gpu:8202", num_steps=20)
    assert cfg.server_url == "http://gpu:8202"
    assert cfg.num_steps == 20


def test_yam_from_kwargs() -> None:
    cfg = YamConfig.from_kwargs(left_channel="canA", control_hz=25.0)
    assert cfg.left_channel == "canA"
    assert cfg.control_hz == 25.0


def test_from_kwargs_rejects_unknown() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        MolmoActConfig.from_kwargs(nope=1)


def test_yam_rejects_bad_joint_limits() -> None:
    with pytest.raises(ValueError, match="joint_low must have 14 entries"):
        YamConfig(joint_low=(0.0,) * 13)


def test_yam_rejects_bad_home_pose() -> None:
    with pytest.raises(ValueError, match="home_pose must have 14 entries"):
        YamConfig(home_pose=(0.0,) * 10)


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
