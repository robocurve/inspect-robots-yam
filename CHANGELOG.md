# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `scripts/run_batch.sh`: repeat one task N times from a rig directory with a
  human in the loop. Each trial is its own `./run` process with `--epochs 1`
  forced, so the framework's grading prompt pauses for the operator and the
  arms are parked with torque released before the script asks for a scene
  reset; the next trial starts on Enter, after discarding keystrokes typed
  during the park ramp. A trial that exits uncleanly gets a check-the-arms
  warning instead of the torque-off claim. Per-trial verdicts are pulled from
  the eval logs into `<log-dir>/batches/<stamp>.tsv`, echoed per trial, and
  tallied at the end. Before each trial launches it saves a top-camera JPEG of
  the reset scene as `batch_<stamp>/trial_NN_<run-id>_start.jpg`
  (`--no-snapshots` to skip). Any other `--epochs` value is rejected because
  within-process epochs hold torque at home between trials.

### Fixed

- The operator's elapsed counter now reads the wall clock instead of the step
  count. It was `num_steps / control_hz`, which is only the truth while every
  step fits inside its control period. A step can already overrun that on a slow
  camera read, and `settle_tolerance` makes overrun routine, at which point an
  operator watching a hardware run could see 30s reported after 90 real seconds
  had passed. Homing is excluded: the clock starts when `reset()` hands the
  episode over. The horizon is unchanged but now renders as `Max ~120s` and
  `t = 30s / ~120s`, because remaining step duration is not knowable in advance
  and dividing the step budget by `control_hz` is an estimate rather than a
  deadline (#64).

### Changed

- **Breaking (EEF layout):** the Cartesian interface grows from 5 to 7 slots
  per arm — `x, y, z, yaw, pitch, roll, gripper`, 14 total. Old→new per-arm
  slot mapping: x/y/z/yaw keep slots 0–3, gripper moves 4→6, and pitch/roll
  occupy the new slots 4/5 (right arm starts at slot 7, was 5). Affects
  `eef_low`/`eef_high` config strings, `eef_state` observations, logged EEF
  actions, and `EEF_DIM_LABELS`. Pitch and roll ship **pinned at (0, 0)** —
  behavior at default bounds is identical to the yaw-only interface; open an
  axis by widening its bounds (pitch strictly inside (-π/2, π/2), roll within
  [-π, π]). Orientation slots are relative to the reset orientation; positive
  pitch tips the tool forward, positive roll toward the arm's left. Equality
  in `eef_low`/`eef_high` now means a pinned axis (previously rejected). The
  relative-rotation extraction supersedes the near-vertical yaw fallback
  (`_ArmKinematics.yaw_axis` is gone) and reports identical yaw inside the
  yaw-only family (#133).

### Added

- The opt-in `YamConfig.eef_orientation` field and setup-wizard option widen
  zero-pinned EEF pitch and roll to conservative ranges. EEF-mode CLI runs now
  warn when that rewrite is active, name orientation axes that remain pinned,
  and flag open tilt axes whose arm still uses the fingertips-down default z
  floor. Direct `rollout()` and `eval()` API runs and CLI runs with
  `--disable-guardrails` do not emit these run-header warnings (#140).
- Named start poses now work in EEF mode: the config-time veto is gone, the
  resolved pose is validated against the EEF action box (FK grasp-point
  position, gripper aperture, and relative yaw 0) before the homing ramp,
  the box error names the offending pose, and a reconnect revalidates the
  re-read pose file (#131).
- Named joint-space start poses with the `inspect-robots-yam-pose` capture,
  goto, list, show, delete, and rename workflows. Pose files use a versioned,
  shareable JSON format with normalized grippers, while `YamConfig.start_pose`
  resolves and validates a named pose before the arm driver connects (#128).

### Changed

- Session-connected runs no longer author console prose: the running banner
  becomes "Running." plus the horizon and the per-second ticker sends bare rig
  state ("t = 4s / ~120s"). The framework session appends its own
  "Esc ends the episode" hint and replaces stale gesture clauses (inspect-robots
  plan 0062), so this text can never drift again when the framework gesture
  changes. Defer-only and never-connected modes keep their own text: the
  session never sees those status lines. The `inspect-robots` floor rises to
  0.51, the release carrying the session-owned hint (#122).

- The documented `i2rt` pin now matches the commit the rigs run
  (`ac096928`, was `db582eaa`), in both the README and
  `I2RT_INSTALL_COMMAND`, the remedy string a missing driver surfaces. The newer
  commit defaults `enable_auto_recovery=False`, so a motor error fails the
  episode fast instead of the driver cleaning and re-enabling the motor inside
  the control loop. A rig installed from the old pin ran the self-healing
  behavior no rig has actually been operated with (#118).

- Session-connected runs now delegate episode end to the framework console
  gesture (Esc, or `/stop`, on inspect-robots 0.47+). Other typed lines become
  policy feedback or logged notes, while never-connected runs retain the
  legacy any-key ending (plan 0022, #114).
- Deferred-mode status text now names the current end gesture: the running
  banner says "Esc (or /stop) ends the episode" and the per-second ticker says
  "Esc ends the episode" (both previously said Enter, which stopped being true
  when inspect-robots 0.47 moved episode end off the Enter key). The
  `inspect-robots` floor rises to 0.47 so the text matches the console behind
  it (#120).
- The setup wizard now suggests **no** for its `collision_guardrail` question
  (previously yes). A fresh setup has no measured `collision_*_base_pos`
  geometry, and the guardrail's library-default offsets can false-positive
  hold until `max_steps` — a silent livelock that scores the episode 0. The
  `YamConfig` runtime default is unchanged (`True`), and a config that
  already sets the key keeps its stored value as the wizard suggestion.
  Caveat for hand-written configs that set the geometry keys while relying
  on the runtime default: write `collision_guardrail = true` explicitly
  before re-running setup, or the wizard will suggest off. Disabling the
  guardrail by config now emits a startup warning naming the re-enable path
  (#109).

### Fixed

- `inspect-robots-yam-health` and `inspect-robots-yam-holdcheck` now honor the
  working directory's `.env` before resolving wizard configuration, including
  `INSPECT_ROBOTS_CONFIG` pins, with exported values retaining precedence and
  `--no-config` retaining the bypass. Requires inspect-robots 0.38 (the new
  dependency floor) (#107).

### Changed

- Operator console support via `YAMEmbodiment.defer_operator_end()` lets the
  framework own stdin, deliver typed feedback, and terminate trials without
  competing with YAM's legacy keypress reader (#102).
- The connection-failure `remedy` now defaults to the policy entry's canonical
  server launch command plus a docs link — `host_server_yam.py` for
  `molmoact2`, `serve_gr00t_act.py` for `gr00t` — instead of an empty string,
  so the hint tells a new user exactly how to start the server. `-P
  remedy=...` still replaces it; an empty string omits the line (#99).
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

### Fixed

- A camera reader that drops a frame and returns `None` for one key now raises
  the same camera-named `ValueError` as any other malformed frame, in both
  `YAMEmbodiment._observe()` and `MolmoAct2Policy.act()`, instead of a bare
  `AttributeError: 'NoneType' object has no attribute 'shape'` (#61).

### Added
- `YAMEmbodiment.connect_operator_session()` gives the framework session sole
  terminal ownership, including readiness gates, durable notices, and the
  replaceable running ticker (plan 0022, #114).
- Opt-in `YamConfig.report_joint_eff` reporting of sign-corrected estimated
  torque under `observation.state["joint_eff"]`, using the same packed 14-slot
  arm and gripper layout as `joint_pos` with raw N·m gripper values (#112).
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
