# YAM viewer model

This is the YAM arm with the LINEAR_4310 parallel gripper from
[i2rt ac096928d6899ddf852a71c5e8fbaa6055cd9745](https://github.com/i2rt-robotics/i2rt/tree/ac096928d6899ddf852a71c5e8fbaa6055cd9745),
the revision pinned by `src/inspect_robots_yam/_i2rt.py`. The XML and nine STL
meshes are covered by the adjacent MIT [LICENSE](LICENSE).

Source files at that revision:

- `i2rt/robot_models/arm/yam/yam.xml` and its `assets/*.stl`
- `i2rt/robot_models/gripper/linear_4310/linear_4310.xml` and its `assets/*.stl`
- `i2rt/robots/config/linear_4310.yml`

`yam.xml` combines those two source XML files using the same geometry operations
as `i2rt.robots.utils.combine_arm_and_gripper_xml`: set `link6`'s position to
`2.39858e-07 -0.0419481 0.0404996`, quaternion to
`0.499998 -0.5 -0.5 -0.500002`, and `joint6` axis to `0 0 -1` from the YAML's
`last_joint_mount.yam`; append the gripper body beneath `link6`; merge assets,
equality, and contact sections. Mesh paths stay relative to `assets/`.

No joint ranges, grasp-site transforms, inertials, or mesh geometry are changed.
The viewer composes two copies at runtime, changes their display colors, and
disables contact participation. It only calls forward kinematics, never physics
integration. The standalone IK adapter uses the QP settings from the same
revision's `i2rt/robots/kinematics.py`, under the same MIT license.

When updating the runtime driver pin, update this model and rerun
`tests/test_mujoco_viewer.py` as well as the full suite. The model provenance test
intentionally fails if the driver pin and this snapshot diverge.
