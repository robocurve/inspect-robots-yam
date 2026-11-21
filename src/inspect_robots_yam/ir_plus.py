"""Arm models for IR+ (inspect-robots-agent ``-P toolset=plus``) on YAM arms.

IR+ moves the arms the way GPT-6.2 Astra's own controllers did when it drove the
rigs directly through Codex: inverse kinematics once at the goal on the arm's
MuJoCo model, then all joints eased there together. Those controllers loaded
i2rt's model of the arm and gripper and solved for its ``grasp_site``;
``i2rt_kinematics`` gives IR+ the same model:

    inspect-robots run --embodiment yam_arms -E control_interface=joints -E control_hz=40 \\
        --policy agent -P toolset=plus -P kinematics=inspect_robots_yam.ir_plus:i2rt_kinematics ...

Each arm's model is in its own base frame (+x forward, +y left, +z up), as the
controllers used it. ``MujocoArm`` wraps any MuJoCo model the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

#: The site the controllers solved for: between the fingertips.
GRASP_SITE = "grasp_site"
#: The YAM's revolute joints per arm (the model's first joints).
ARM_JOINTS = 6
#: Orientation zero on i2rt's model: the tool straight down, jaws as the arm faces at home.
#: The controllers' ``down(np.pi)``, Rz(pi) diag(1, -1, -1), which nearly all of their calls
#: passed (pointing down, only yaw near pi is in reach of the wrist).
I2RT_DOWN = np.diag([-1.0, 1.0, -1.0])


class MujocoArm:
    """IR+'s ``ArmModel`` for one arm: forward kinematics of a MuJoCo model at ``site``.

    ``fk(q)`` sets the model's first ``ARM_JOINTS`` joints and returns the site's
    position (m) and rotation matrix; ``joint_low``/``joint_high`` are those
    joints' ranges in the model. ``down``, when given, is the tool rotation IR+
    calls orientation zero.
    """

    def __init__(self, xml_path: str, site: str = GRASP_SITE, down: Vec | None = None) -> None:
        import mujoco

        self._mujoco = mujoco
        self.xml_path = str(xml_path)
        self._model = mujoco.MjModel.from_xml_path(self.xml_path)
        self._data = mujoco.MjData(self._model)
        self._site = self._model.site(site).id
        ranges = np.asarray(self._model.jnt_range[:ARM_JOINTS], dtype=np.float64)
        self.joint_low: Vec = ranges[:, 0].copy()
        self.joint_high: Vec = ranges[:, 1].copy()
        self.down: Vec | None = None if down is None else np.asarray(down, dtype=np.float64)

    def fk(self, q: Vec) -> tuple[Vec, Vec]:
        """Site position and rotation for the arm's joint angles ``q`` (radians)."""
        self._data.qpos[:ARM_JOINTS] = np.asarray(q, dtype=np.float64)[:ARM_JOINTS]
        self._mujoco.mj_kinematics(self._model, self._data)
        position = np.asarray(self._data.site_xpos[self._site], dtype=np.float64).copy()
        rotation = np.asarray(self._data.site_xmat[self._site], dtype=np.float64).reshape(3, 3)
        return position, rotation.copy()

    def ik_seeds(self, position: Vec) -> list[Vec]:
        """The canonical YAM poses facing ``position`` that the controllers also started IK from."""
        a = float(np.arctan2(position[1], position[0]))
        return [
            np.array([a, 1.2, 1.5, -0.5, 0.0, 0.0]),
            np.array([0.0, 1.3, 1.6, 0.0, 0.0, 0.0]),
            np.array([a, 1.3, 1.5, -0.5, 0.0, 1.5]),
            np.array([a, 1.3, 1.5, -0.5, 0.0, -1.5]),
        ]


def i2rt_model_path(gripper_type: str = "LINEAR_4310") -> str:  # pragma: no cover - needs i2rt
    """The MuJoCo model i2rt assembles for a YAM arm with this gripper (as ``yam_arms`` does)."""
    from inspect_robots_yam._i2rt import _load_i2rt_kinematics

    _, arm_type, gripper, combine_xml, _ = _load_i2rt_kinematics()
    return str(combine_xml(arm_type.YAM, gripper[gripper_type]))


def i2rt_kinematics(info: Any = None, gripper_type: str = "LINEAR_4310") -> Mapping[str, MujocoArm]:
    """IR+'s arm models from i2rt's MuJoCo model (YAM + ``gripper_type``), one per arm.

    ``-P kinematics=inspect_robots_yam.ir_plus:i2rt_kinematics``. The rigs'
    gripper is LINEAR_4310, ``yam_arms``' default.
    """
    path = i2rt_model_path(gripper_type)
    return {"left": MujocoArm(path, down=I2RT_DOWN), "right": MujocoArm(path, down=I2RT_DOWN)}
