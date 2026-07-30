# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `OPTION_SLOTS` declaration on `YAMEmbodiment` (#87): `inspect-robots setup`
  now offers `auto_start` as a yes/no question and writes the answer to
  config.ini. Requires inspect-robots 0.29 (the new dependency floor).
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
