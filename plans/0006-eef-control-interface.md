# 0006 — Cartesian EEF control interface (the LLM-agent-native API)

Two-repo feature. This plan owns the yam side; §8 scopes the small
inspect-robots-agent companion change. The current joint-vector API stays
fully supported behind the same switch (`control_interface="joints"`).

## 1. Why: research summary

The agent policy today exposes `move_joints` with 14 named joint targets.
Research on LLM/VLM robot control says this is the weakest possible surface
for a frontier model:

- Anthropic's robotics interface study found models "mostly fail" at
  joint/motor-level control while high-level command primitives succeed
  substantially, and that semantic state feedback (pose, heading) beats raw
  vectors and even images ("it is still more helpful to tell the model which
  way it is facing than to show it a picture of itself").
- FAEA (arXiv 2601.20334) reached 70.6% success across 120 tasks
  demonstration-free with exactly one control abstraction: absolute Cartesian
  end-effector position commands plus a gripper primitive, in a closed loop
  with structured error feedback.
- CaP-X (arXiv 2603.22435) benchmarked API abstraction levels explicitly:
  success rises monotonically with abstraction for every model tested, and
  low-level primitives only catch up when wrapped in multi-turn feedback.
  Their design maxim: the agent should "reason in a task-oriented Cartesian
  space while delegating execution feasibility to the controller".
- Demystifying Action Space Design (arXiv 2602.23408): EEF pose is favored
  for semantic simplicity; the recent joint-space trend in *learned* policies
  exists to bypass IK numerical issues — a rationale that does not apply to
  an LLM agent, which cannot do IK in its head at all.
- MOKA / RoboPoint / PIVOT: VLMs are weak at metric 3D from pixels but
  strong at 2D image pointing. A camera-frame pointing surface is a natural
  *future* interface (needs depth deprojection + extrinsics calibration);
  out of scope here, but the switch architecture must leave room for it.

Conclusion: the LLM-native API for YAM is absolute Cartesian EEF position
per arm plus gripper, with pose-level state feedback, IK delegated to the
embodiment, and every existing safety layer unchanged underneath.

## 2. Feasibility

i2rt ships `i2rt.robots.kinematics.Kinematics`: FK and mink differential IK
over the bundled YAM MuJoCo models (arm xml combined with the gripper xml
via `combine_arm_and_gripper_xml`; the `grasp_site` EEF site lives in the
*gripper* XMLs, present in all four supported gripper types). mink, mujoco,
and the QP solver come through i2rt's own dependencies.

Dependency reality: i2rt is NOT a pyproject dependency of this plugin — it
is a git-installed, lazily-imported runtime requirement (`_i2rt.py`), and
the repo's gates (100% coverage, import-hygiene with only
inspect-robots+numpy) must keep passing without it. §5c defines the
injection seam and §7 the two test tiers this forces. No new top-level
dependencies are added by this plan.

## 3. The switch: `control_interface`

`YamConfig` gains `control_interface: str = "joints"`:

- `"joints"` (default, unchanged): 14-D joint_pos (or joint_delta with the
  existing `joints_are_delta=True`), exactly today's behavior. The factory
  default stays joints because VLA policies (molmoact2) declare joint
  semantics and the compat check must keep passing for them.
- `"eef_pos"`: the new 10-D Cartesian interface below.
- Anything else: `ValueError` at config construction, naming the valid set.
- `joints_are_delta=True` together with `control_interface="eef_pos"` is a
  config error (delta-EEF is a possible future interface, not this one).

Selection is per run: `[embodiment.args] control_interface = eef_pos` in
config.ini, or the CLI's embodiment-args passthrough. The agent policy needs
no flag of its own — it already adapts to whatever space the embodiment
declares at bind time. Future interfaces (camera-frame pointing, delta-EEF,
per-arm primitives) join as new `control_interface` values.

## 4. The `eef_pos` action space

10-D, absolute, per-dimension labeled — chosen so the agent plugin's
existing absolute-mode machinery (named partial targets, speed-limited
straight-line interpolation, bounds/delta guardrails) applies unchanged:

| dims | labels | unit | default bounds |
|---|---|---|---|
| 0-2 | `left_x, left_y, left_z` | m | x [0.15, 0.48], y [-0.25, 0.25], z [0.03, 0.40] |
| 3 | `left_yaw` | rad | [-np.pi, np.pi] (exactly, so an observed yaw of π echoes back in-bounds) |
| 4 | `left_gripper` | — | [0, 1] (1 = open, plan 0005 polarity) |
| 5-9 | `right_*` | same | same |

Default bounds are **empirically validated on the bundled YAM +
LINEAR_4310 model** (2026-07-14, mink IK, 500 iters): with the tool held at
the §5b working orientation (pitched 30° from vertical toward the base) and
radially-pointing yaw, all 8 corners, 6 face centers, and the box center
converge with < 5 mm position error. Two envelope facts drove this box, both
measured: a *strictly vertical* tool is wrist-limited (±1.57 rad wrist, one-
sided elbow) and unreachable below z ≈ 0.15 anywhere near the base, and the
prior draft's box failed 8/14 probes. Reachability inside the box still
depends on commanded yaw (the validation uses radial yaw); off-radial yaws
at box extremes may fall back to best-effort IK, which the observation
reports honestly. `z_min = 0.03` keeps table clearance against best-effort
position error. The tier-2 test (§7) re-validates all 15 probe points on
every run so a model update cannot silently invalidate the defaults.

- Coordinates are in **each arm's own base frame** (+x forward from the arm
  base, +z up — a per-mount convention software cannot verify; on mirrored
  bimanual mounts the two arms' +y axes point opposite ways in the world,
  and the README must say so). No cross-arm calibration exists on rigs yet;
  a future `arm_base_poses` config knob can lift both arms into one shared
  frame without changing this interface. The frame convention is stated in the
  space's `ActionSemantics.frame` ("base") and must be spelled out in the
  agent-facing bounds text via dim naming alone (the plugin already
  enumerates labels and bounds into the tool description).
- `yaw` is rotation about base +z, absolute, bounded — NOT wrapped: a
  3.1 → −3.1 command linearly sweeps ~2π through zero rather than taking
  the short way. The tool description must say so and advise intermediate
  yaws for near-±π regrasps. Full orientation (rot6d) is a possible later
  interface value; position+yaw covers tabletop regrasp/alignment (fork
  rotation) without asking a VLM to emit 6-D rotation matrices, which the
  research says they are bad at.
- Yaw convention, pinned: at the end of `reset()` each arm captures its FK
  orientation `R0` and a reference `R_ref := Rz(−yaw0) · R0`, where
  `yaw0 = atan2(a_y, a_x)` for `a` the horizontal projection of the gripper
  frame's x-axis. If that projection has norm < sin(5°) ≈ 0.087 (tool axis
  near vertical), the gripper y-axis is used instead — and whichever axis
  is chosen at reset is **pinned for the whole trial**, for both the
  reference and every reported `eef_state` yaw, so measurement noise near
  the threshold cannot flip the extraction branch mid-trial (a per-step
  re-evaluation would jump the reported yaw by ~π/2 across the flip). At
  the §5b default home orientation the x-axis projection is 0.5 — an order
  of magnitude above the threshold — so the default path never sits near
  the degeneracy. The commanded target orientation is
  `R_target(yaw) = Rz(yaw) · R_ref`: no per-step extraction of the target,
  and roll/pitch are held at their reset-captured values for the whole
  trial — the reference is captured once, never re-read from measurements,
  so tracking error cannot integrate into orientation drift. A signed
  FK-based test pins the sign convention.
- Semantics: `control_mode="eef_abs_pose"`, `rotation_repr="none"`,
  `gripper="continuous"`, `frame="base"`, the labels above. `"none"` is the
  honest declared repr for per-dim clamping purposes: no packed 3-D rotation
  encoding is present; yaw rides as an ordinary bounded scalar dimension.
  (The core's per-dim-safe set is {none, rot6d}; this stays inside it.)
- Workspace bounds are config knobs (`eef_low`/`eef_high`, defaults above,
  same override pattern as `joint_low/high`). They are a conservative box,
  not the exact reachable set — reachability failures are handled per step
  (§5), and the bounds exist mainly so the toolset can derive speeds and the
  guardrails can clamp.

## 5. Execution: IK inside the embodiment

Per-arm `Kinematics` instances are constructed lazily at embodiment init
(combined arm+gripper model, `grasp_site`).

- `step(action)` in eef mode: split the 10-D action per arm; build the
  target as (position, `R_target(yaw)` per §4); differential IK
  warm-started from the arm's **last commanded** joints (`q_cmd_prev`, not
  the measured joints — path-continuity of the warm start is what keeps
  differential IK on one solution branch; warm-starting from measurements
  reintroduces branch flips through tracking noise), iteration-capped
  (`ik_max_iters`, default 20 — warm starts make centimeter steps converge
  in a few iterations, and the cap bounds the non-convergent worst case
  well inside the 100 ms budget); take mink's best iterate — convergence
  failure is NOT an exception: the best-effort joints move toward the
  target and the truth shows up in the next observation's FK state,
  closing the loop.
- No-progress guard (anti-livelock): if the clamped command reduces the
  Cartesian target error by less than 1 mm for 5 consecutive steps, the arm
  holds its last command for the remainder of the chunk. Without this, a
  branch-flip clamp cycle could oscillate ±`ik_step_joint_limit` at 10 Hz
  for a full 10 s chunk; with it, the stall becomes visible in the next
  observation and the agent replans.
- **Joint-space rate backstop (safety-critical).** The Cartesian
  `DeltaLimitApprover` cannot see joint deltas, and an IK branch flip
  (elbow reconfiguration near a singularity or limit) can return a solution
  radians away from the current pose. Before commanding, eef-mode `step()`
  clamps the per-joint displacement: `q_cmd = q_now + clip(q_ik − q_now,
  ±ik_step_joint_limit)` with `ik_step_joint_limit` a config knob
  defaulting to 0.2 rad per joint per step (the same scale as delta mode's
  `_STEP_ARM`). This restores in eef mode the "no wild swings" guarantee
  joint mode gets from the core guardrails — in *joint space*. Stated
  honestly: it bounds Cartesian motion only loosely (6 joints × 0.2 rad ×
  ~0.9 m lever can traverse tens of centimeters of EEF translation in one
  step during a multi-step branch transit, ~20× the Cartesian delta limit);
  the residual exposure is bounded, brief, and visible in the observation,
  and joint mode's own step limits carry the same order of exposure. The
  absolute joint limits additionally stay enforced in `_send()` as today.
- Config joint limits and IK: `YamConfig.joint_low/high`, where tighter
  than the MuJoCo model ranges, are applied by tightening the loaded
  model's `jnt_range` for the six arm joints at wrapper construction (the
  wrapper owns its private `MjModel`; mink's `ConfigurationLimit` reads
  ranges from the model, so this is the supported mechanism — no XML
  rewriting). The solver then respects operator-tightened limits instead of
  having `_send()`'s clamp silently distort the Cartesian solution
  afterward.
- Gripper joints in the IK model: the gripper joint count varies by gripper
  type — LINEAR_* grippers contribute two equality-coupled, meter-ranged
  slide joints; CRANK_4310 contributes none (its XML has no joints). The
  wrapper addresses the six arm joints positionally as the first six and
  handles 0..2 trailing gripper joints: whatever gripper joints exist are
  pinned to their model mid-range in `init_q` and stripped from the
  returned solution. The commanded normalized gripper value flows through
  the existing driver path untouched and never enters IK.
- The agent-side chunk interpolates *in Cartesian space* (the plugin's
  existing linspace over the action dims), so motions are straight lines in
  the workspace with per-step Cartesian displacement capped by the
  toolset's speed limit and the core `DeltaLimitApprover` (5% of workspace
  range per step by default) — the same two-layer story as joints, in
  better units.
- **No collision awareness in v1** — stated loudly, not implied: neither
  arm-table nor arm-arm collision checking exists (the two arms' default
  y-ranges overlap in a bimanual workspace and mink is given no collision
  limits). The workspace box, the Cartesian and joint-space rate limits,
  and the absolute joint limits are the only geometric protections.
  Operator attendance is assumed, as it already is for joint mode; the
  README must flag eef mode + unattended operation as operator discretion.

### 5b. reset(), home pose, and close() in eef mode

Homing, `rest_pose`, and parking remain joint-space mechanisms — those
poses are 14-D joint vectors regardless of `control_interface`. But eef
mode changes the *default* home: `DEFAULT_EEF_HOME_POSE`, a 14-D joint
vector placing each arm's EEF at (0.30, 0, 0.20) in its base frame with
the tool pitched 30° from vertical toward the base and yaw 0 (per-arm
joints ≈ [-0.024, 0.794, 0.645, -0.375, -0.021, -0.012], grippers open —
provisional values from the 2026-07-14 probe; implementation re-derives
them with the tier-2 test and records the derivation script). This
orientation is what §4's box validation holds, and it is what makes the
yaw extraction robust (§4). The joint-space zero rest pose is NOT a valid
eef-mode start: its EEF sits outside the default box and its vertical tool
axis is yaw-degenerate.

At embodiment init in eef mode, the configured (or default) home pose is
validated: FK(home) must lie inside the workspace box, erroring out
otherwise. This kills the out-of-box-start failure mode where the agent's
first partial `move_to` would clip *held* dimensions to the box boundary
and command unintended motion (the toolset clips every interpolant into
the box, including unnamed held dims).

`reset()` captures the per-arm yaw references (§4) after homing completes,
and its returned observation already carries `eef_state`.

### 5c. The kinematics seam

`YAMEmbodiment.__init__` gains a `kinematics_factory` keyword argument —
the same seam pattern as the existing `driver_factory` constructor kwarg
(NOT a `YamConfig` field: the config is a frozen, CLI-constructable
dataclass of scalars/tuples reachable via `-E key=value`, and a callable
would break that contract). The factory returns per-arm objects satisfying
a small `fk(q) -> 4x4` / `ik(target, init_q) -> (bool, q)` protocol. The
default factory lazily imports i2rt (`_i2rt.py` conventions) and wraps
`i2rt.robots.kinematics.Kinematics`; tests inject a deterministic fake.
This keeps the import-hygiene gate green (no mink/mujoco at import time)
and makes 100% coverage achievable without i2rt installed.

## 6. Observation in eef mode

The observation keeps the existing camera images and 14-D `joint_pos` state
field, and adds a 10-D `eef_state` field using the same labels as the
action dims: position and yaw come from FK of the measured arm joints (yaw
via the §4 pinned extraction), gripper dims from the normalized driver
state (FK cannot produce them). The `StateSpec` in eef mode declares `eef_state` as the
field whose shape matches the action dim (10,), which is exactly what the
agent plugin's absolute mode requires for its proprioceptive reference —
so the agent reads and re-checks the same quantities it commands
(semantic-state-feedback finding, §1). `joint_pos` (14,) stays for logging
and debugging; its shape cannot collide with (10,).

## 7. yam tests (TDD, two tiers)

**Tier 1 — gated (100% coverage, no i2rt):** everything runs against the
injected fake kinematics (§5c) and the existing fake driver.

- Config: `control_interface` validation (unknown value, delta+eef
  combination), default unchanged; new knobs (`eef_low/high`,
  `ik_max_iters`, `ik_step_joint_limit`, `kinematics_factory`) validated.
- Space: eef space shape/labels/bounds/semantics; observation StateSpec
  declares `eef_state` at (10,) exactly once alongside `joint_pos` (14,).
- step(): 10-D action → both arms commanded; joint-space rate backstop
  clamps a fake-IK solution placed radians away (the elbow-flip case) to
  `ik_step_joint_limit`; gripper passthrough never enters IK and gripper
  slide joints are pinned/stripped; best-effort on fake non-convergence
  (finite, in-limit joints commanded); `ik_max_iters` plumbed through;
  config joint limits forwarded to the IK seam.
- Yaw: sign convention against a hand-built rotation; vertical-tool-axis
  fallback extraction; references captured at reset and never re-read
  (feed a drifting fake measurement, assert the target orientation is
  unaffected).
- reset()/close(): joint-space homing/parking unchanged in eef mode;
  reset observation carries `eef_state`; yaw references captured
  post-homing.
- Integration with the agent plugin: `build_toolset` on the eef space
  produces tooling with the 10 labels and correct bounds text; a scripted
  conversation drives a straight-line move and the emitted chunk passes
  `ChainApprover(Clamp, DeltaLimit)` untouched. Dependency mechanism,
  spelled out: this repo is not a uv workspace — add
  `inspect-robots-agent>=0.2.2` (PyPI) to the `dev` extra and re-run
  `uv lock`.
- Housekeeping the tiers force: the repo currently has zero skipif tests
  and `--strict-markers`, so the tier-2 module registers its markers
  (`real_kinematics`, `perf`) in pyproject and skips module-wide when i2rt
  or mink is not importable.
- Existing joint-mode tests untouched and passing (the default path).

**Tier 2 — real-kinematics (skipif i2rt/mink/mujoco not importable; runs
on rig machines and any environment with the runtime deps, never gates
CI):**

- FK/IK round trip on the bundled YAM model: position within 1 mm, yaw
  within 0.01 rad, via the pinned extraction rule.
- All 15 default-box probe points (8 corners, 6 face centers, center)
  reach IK convergence within 5 mm at the §5b held orientation with
  radially-pointing yaw — the exact validation that produced the defaults.
- Warm-start convergence over a straight-line sequence of centimeter
  steps; saturated non-convergent step (unreachable target at
  `ik_max_iters`) stays within a generous wall-clock budget (marked perf
  test, not a tight CI assertion).
- Signed yaw FK test on the real model.

## 8. Companion change: inspect-robots-agent (separate PR, core repo)

Small, mode-aware polish — no behavior change for joint-mode users:

- When the bound space's control mode is in `_POSE_MODES`, the move tool is
  named `move_to` (not `move_joints`) and its description says Cartesian
  end-effector targets with units (meters/radians per the labels), still
  enumerating per-dim bounds. The description also states the frame
  contract in embodiment-agnostic terms: "coordinates are absolute in the
  embodiment's declared frame; on multi-arm embodiments each arm uses its
  own base frame and axes may differ between arms depending on mounting."
  This is the only agent-visible surface (the LLM never reads READMEs), so
  the mirrored-mount caveat must live here, not only in yam docs. A richer
  per-embodiment hint channel (free-text surface notes on ActionSemantics)
  is deliberately out of scope. (Precisely: today's tool-name branch is
  absolute-vs-displacement and `_POSE_MODES` only gates rotation repr — this
  adds a pose-mode case to the naming/description selection. Small either
  way.) The system prompt template stays shared.
- Tests: schema name/description under an eef-mode space; joint-mode
  schemas unchanged.
- Plugin version bump per its policy.

## 9. Out of scope (future `control_interface` values)

- Camera-frame 2D pointing with depth deprojection (needs extrinsics
  calibration per rig) — the research-favored next surface for VLMs.
- Delta-EEF, full-orientation (rot6d) EEF, discrete primitive skills
  (`pick(x,y)`), shared cross-arm world frame via `arm_base_poses`.
- Any change to VLA policy paths or the joints default.

## 10. Rollout

yam PR first (self-contained; agent plugin works against it unchanged, with
the tool merely still named `move_joints`), core PR second (rename polish),
then a yam release and a core release per the usual one-click flows. The
omen rig opts in via `[embodiment.args] control_interface = eef_pos`.
