# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- RealSense RGB-D capture now runs all configured pipelines in one lazy spawn
  child by default, isolating librealsense and frame-copy work from the
  motor-control interpreter (#95). Shared-memory seqlock slots preserve the
  existing colour, aligned depth, intrinsics, staleness, and lazy-thunk
  contracts. `realsense_capture=inline` restores the previous in-process path,
  and `depth_fps` configures both colour and depth stream rates. All-digit depth
  serials passed as integers now fail fast with a config.ini quoting hint
  because integer parsing can discard leading zeros.

- The predictive MuJoCo collision guardrail is now contributed by
  `YAMEmbodiment` for absolute joint control and defaults on. This is a
  results-affecting upgrade: table-press grasps can hold when demonstration
  targets penetrate the modeled table, while bimanual close-quarters work can
  livelock when the open-finger model blocks a repeatedly commanded target.
  Raise `collision_penetration_threshold` or lower `collision_table_height` for
  table presses, set `collision_table=false` on tableless rigs, configure
  measured `collision_left_base_pos` and `collision_right_base_pos` for
  cross-arm work, or set `collision_guardrail=false` for a per-rig opt-out. A
  home pose that collides under the effective geometry now refuses to start;
  correct the `collision_*` geometry or use the opt-out. EEF and delta-joints
  modes keep running with a skip warning, and installs without MuJoCo get the
  collision-extra install command instead of a construction failure. Requires
  inspect-robots 0.31.

- The `inspect-robots setup` wizard now suggests yes for the `auto_start`
  question (`[Y/n]`). The `YamConfig.auto_start` default is unchanged
  (`False`): runs configured outside the wizard keep the prompt-gated flow,
  and the wizard still writes an explicit true/false answer.

### Added
- `ActServerPolicy.server_url` and `remedy` connection-failure hint attributes,
  plus the CLI-settable `ActServerConfig.remedy` recovery instruction (#97,
  robocurve/inspect-robots#219).
- `YamConfig.gripper_stroke_s` and per-gripper `max_step` declarations for
  absolute joint and Cartesian action spaces, pacing a full normalized gripper
  stroke at approximately one second by default. Requires inspect-robots 0.30
  (the new dependency floor) (#90).
- `OPTION_SLOTS` declaration on `YAMEmbodiment` (#87): `inspect-robots setup`
  now offers `auto_start` as a yes/no question and writes the answer to
  config.ini. This feature requires inspect-robots 0.29.
- `YamConfig.auto_start` (plan 0015, #87): opt-in zero-touch attended starts.
  Skips both operator Enter gates in `reset()` (a printed stand-clear notice
  replaces the home gate; the scene-ready gate is dropped and stdin is drained
  in its place) while keeping status lines, the end-episode keypress, and
  operator grading. Faults before any motion when stdin is not an interactive
  TTY; `unattended=True` takes precedence.
- Collision guardrail (plan 0011, #85): `inspect_robots_yam.collision` with
  `CollisionChecker` (bimanual MuJoCo scene composed from a vendored
  collision-only Menagerie model), `CollisionApprover` (predictive sweep;
  blocked targets hold at the last safe pose instead of aborting the eval,
  strict mode raises `SafetyAbort`), and `build_yam_guardrails` (clamp,
  delta limit, collision chain). New optional `collision` extra pulling in
  `mujoco>=3.3.1`; absolute 14-D `joint_pos` mode only.

### Changed

- Bare `inspect-robots-yam-health` now checks the wizard-configured rig and
  announces contributed configuration with a stderr attribution line;
  `--no-config` restores flag-only behavior.
- `inspect-robots-yam-holdcheck` accepts `left` and `right`, resolving their CAN
  channels through the wizard config.
- Depth-configured camera slots are outside health-check and `--watch` scope and
  are reported as unchecked.
- Arm drivers now initialize concurrently, reducing driver initialization
  wall-clock to approximately the slower arm's time instead of their sum.
- The two arms' i2rt bring-up logs can now interleave; new
  `left/right arm bring-up starting/complete` markers bracket each arm's output.
- Ctrl-C during arm bring-up now takes effect only after both arms finish
  initializing and cleans up whatever was built. A factory wedged on
  unresponsive hardware makes Ctrl-C ineffective where sequential bring-up
  aborted immediately.

### Fixed

- An initialization failure on one arm no longer leaks the other arm's control
  thread, CAN socket, and torque-enabled motors.
