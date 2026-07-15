"""Re-derive the provisional Cartesian home joints with real i2rt kinematics."""

from __future__ import annotations

import numpy as np

from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import _default_kinematics_factory


def main() -> None:
    """Solve the documented working pose and print one seven-slot arm home."""
    cfg = YamConfig(control_interface="eef_pos")
    raw, _ = _default_kinematics_factory(cfg)
    ranges = np.asarray(raw.get_joint_ranges(), dtype=float)
    initial_arm = np.asarray((0.0, 0.8, 0.65, -0.4, 0.0, 0.0))
    initial = np.concatenate((initial_arm, np.mean(ranges[6:], axis=1)))

    sine_30 = 0.5
    cosine_30 = np.sqrt(3.0) / 2.0
    target = np.eye(4)
    target[:3, :3] = np.asarray(
        (
            (-sine_30, 0.0, -cosine_30),
            (0.0, -1.0, 0.0),
            (-cosine_30, 0.0, sine_30),
        )
    )
    target[:3, 3] = (0.30, 0.0, 0.20)
    success, solution = raw.ik(target, initial, 500)
    achieved = np.asarray(raw.fk(solution))
    print("success:", success)
    print("arm joints:", np.asarray(solution)[:6].tolist())
    print("position error m:", float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])))
    print("14-D home: append gripper 1.0 and repeat for the right arm")


if __name__ == "__main__":
    main()
