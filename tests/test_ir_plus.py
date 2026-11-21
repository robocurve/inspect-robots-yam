"""IR+'s YAM arm models: MuJoCo forward kinematics at the grasp site."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inspect_robots_yam import ir_plus

# Six hinges in a chain (z, y, y, y, z, x) with 0.1 m and 0.2 m links, the grasp site
# 0.1 m past the last joint, pointing down.
_JOINTS = [
    ("0 0 0", "0 0 1", "-2.6 3.0"),
    ("0 0 0.1", "0 1 0", "0 3.6"),
    ("0.2 0 0", "0 1 0", "0 3.6"),
    ("0.2 0 0", "0 1 0", "-1.5 1.5"),
    ("0 0 0", "0 0 1", "-1.5 1.5"),
    ("0 0 0", "1 0 0", "-2.0 2.0"),
]
_BODIES = "".join(
    f'<body name="b{i}" pos="{pos}"><joint name="j{i}" axis="{axis}" range="{rng}"/>'
    '<geom type="sphere" size="0.01"/>'
    for i, (pos, axis, rng) in enumerate(_JOINTS)
)
_XML = (
    '<mujoco><compiler angle="radian"/><worldbody>'
    + _BODIES
    + '<site name="grasp_site" pos="0 0 -0.1" quat="0 1 0 0"/>'
    + "</body>" * len(_JOINTS)
    + "</worldbody></mujoco>"
)


@pytest.fixture
def xml(tmp_path: Path) -> str:
    path = tmp_path / "arm.xml"
    path.write_text(_XML)
    return str(path)


def test_fk_at_the_grasp_site(xml: str) -> None:
    arm = ir_plus.MujocoArm(xml)
    position, rotation = arm.fk(np.zeros(6))
    assert position == pytest.approx([0.4, 0.0, 0.0])  # base 0.1 up, two 0.2 links, tool 0.1 down
    assert rotation == pytest.approx(np.diag([1.0, -1.0, -1.0]))  # the tool points down
    turned, _ = arm.fk(np.array([np.pi / 2, 0, 0, 0, 0, 0, 0.5]))  # a gripper slot is ignored
    assert turned == pytest.approx([0.0, 0.4, 0.0])
    assert arm.joint_low == pytest.approx([-2.6, 0.0, 0.0, -1.5, -1.5, -2.0])
    assert arm.joint_high == pytest.approx([3.0, 3.6, 3.6, 1.5, 1.5, 2.0])
    assert arm.xml_path == xml


def test_ik_seeds_face_the_target(xml: str) -> None:
    seeds = ir_plus.MujocoArm(xml).ik_seeds(np.array([0.0, 0.3, 0.1]))
    assert len(seeds) == 4
    assert seeds[0][0] == pytest.approx(np.pi / 2) and seeds[1][0] == 0.0


def test_i2rt_kinematics_gives_each_arm_the_model(
    xml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def fake_path(gripper_type: str = "LINEAR_4310") -> str:
        asked.append(gripper_type)
        return xml

    monkeypatch.setattr(ir_plus, "i2rt_model_path", fake_path)
    arms = ir_plus.i2rt_kinematics(None)
    assert set(arms) == {"left", "right"} and arms["left"] is not arms["right"]
    assert arms["left"].xml_path == xml and asked == ["LINEAR_4310"]
    assert arms["left"].down == pytest.approx(ir_plus.I2RT_DOWN)
    assert ir_plus.MujocoArm(xml).down is None
