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

i2rt (already a dependency) ships `i2rt.robots.kinematics.Kinematics`: FK
and mink differential IK over the bundled YAM MuJoCo models
(`robot_models/arm/yam/yam.xml` + gripper xml via
`combine_arm_and_gripper_xml`, EEF site `grasp_site`). mink and mujoco come
in through i2rt's own dependencies; this plan adds no new top-level deps.
Implementation must verify the exact site/model names against the pinned
i2rt version at build time, not hardcode blindly.

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
| 0-2 | `left_x, left_y, left_z` | m | x [0.05, 0.65], y [-0.45, 0.45], z [0.01, 0.60] |
| 3 | `left_yaw` | rad | [-3.14159, 3.14159] |
| 4 | `left_gripper` | — | [0, 1] (1 = open, plan 0005 polarity) |
| 5-9 | `right_*` | same | same |

- Coordinates are in **each arm's own base frame** (+x forward from the arm
  base, +z up). No cross-arm calibration exists on rigs yet; a future
  `arm_base_poses` config knob can lift both arms into one shared frame
  without changing this interface. The frame convention is stated in the
  space's `ActionSemantics.frame` ("base") and must be spelled out in the
  agent-facing bounds text via dim naming alone (the plugin already
  enumerates labels and bounds into the tool description).
- `yaw` is rotation about base +z, absolute, bounded (not wrapped): the IK
  target orientation holds the current wrist roll/pitch and tracks
  commanded yaw. Full orientation (rot6d) is a possible later interface
  value; position+yaw covers tabletop regrasp/alignment (fork rotation)
  without asking a VLM to emit 6-D rotation matrices, which the research
  says they are bad at.
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
  target as (position, yaw ∘ current roll/pitch); differential IK warm-
  started from the arm's current joints (`init_q`), joint-limit constraints
  on; take mink's best iterate — convergence failure is NOT an exception:
  the best-effort joints get commanded (they move toward the target), and
  the truth shows up in the next observation's FK state, closing the loop.
  The absolute joint limits stay enforced in `_send()` as today, so IK
  output cannot exceed them no matter what.
- The agent-side chunk interpolates *in Cartesian space* (the plugin's
  existing linspace over the action dims), so motions are straight lines in
  the workspace with per-step Cartesian displacement capped by the toolset's
  speed limit and the core `DeltaLimitApprover` (5% of workspace range per
  step by default) — the same two-layer story as joints, in better units.
- Gripper dims pass through to the existing gripper path untouched.
- Per-step IK cost at 10 Hz: mink warm-started from the previous solution
  converges in a few iterations for centimeter-scale steps; implementation
  must include a latency check in tests (a step must fit well inside the
  100 ms budget on CPU).

## 6. Observation in eef mode

The observation keeps the existing camera images and 14-D `joint_pos` state
field, and adds an FK-derived 10-D `eef_state` field using the same labels
as the action dims. The `StateSpec` in eef mode declares `eef_state` as the
field whose shape matches the action dim (10,), which is exactly what the
agent plugin's absolute mode requires for its proprioceptive reference —
so the agent reads and re-checks the same quantities it commands
(semantic-state-feedback finding, §1). `joint_pos` (14,) stays for logging
and debugging; its shape cannot collide with (10,).

## 7. yam tests (TDD)

- Config: `control_interface` validation (unknown value, delta+eef
  combination), default unchanged.
- Space: eef space shape/labels/bounds/semantics; observation StateSpec
  declares `eef_state` at (10,) exactly once.
- Kinematics wrapper: FK/IK round trip on the real bundled model (command
  a pose, IK, FK the result, assert position within 1 mm and yaw within
  0.01 rad); warm-start convergence over a straight-line sequence of
  centimeter steps; per-step wall-clock budget.
- step(): 10-D action → both arms commanded with IK joints within absolute
  limits; gripper passthrough; best-effort behavior on an unreachable
  target (commanded joints finite, within limits, observation FK reflects
  reality); eef_state present and consistent with FK of commanded joints on
  the fake driver.
- Integration with the agent plugin (dev-dep already available in the
  workspace): `build_toolset` on the eef space produces `move_joints`-class
  tooling with the 10 labels and correct bounds text; a scripted
  conversation drives a straight-line move and the emitted chunk passes
  `ChainApprover(Clamp, DeltaLimit)` untouched.
- Existing joint-mode tests untouched and passing (the default path).

## 8. Companion change: inspect-robots-agent (separate PR, core repo)

Small, mode-aware polish — no behavior change for joint-mode users:

- When the bound space's control mode is in `_POSE_MODES`, the move tool is
  named `move_to` (not `move_joints`) and its description says Cartesian
  end-effector targets with units (meters/radians per the labels), still
  enumerating per-dim bounds. The toolset already branches on mode; this
  extends the existing branch, and the system prompt template stays shared.
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
