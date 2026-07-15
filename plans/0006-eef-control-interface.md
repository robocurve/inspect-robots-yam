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
| 4 | `left_gripper` | — | [0, 1] (1 = open, plan 0005 polarity; a full 0→1 stroke alone computes 101 steps and trips the toolset's split-the-move error at default speed — inherited joint-mode behavior, recoverable by the agent) |
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
reports honestly. On `z_min = 0.03`, measured on the model: the fingertip
meshes extend ~11 mm *below* `grasp_site` at the held orientation, so true
table clearance at z_min is ~19 mm, less up to ~5 mm of best-effort IK
error — adequate over a table at the arm-base plane, but a raised work
surface needs a raised `z_min` (config-overridable like the rest of the
box). The tier-2 test (§7) re-validates all 15 probe points on every run
so a model update cannot silently invalidate the defaults.
`eef_low`/`eef_high` overrides are validated: yaw bounds outside
[−π, π] are rejected (reported yaw always wraps into (−π, π], so a wider
bound would break echo-back).

- Coordinates are in **each arm's own base frame** (+x forward from the arm
  base, +z up — a per-mount convention software cannot verify; on mirrored
  bimanual mounts the two arms' +y axes point opposite ways in the world,
  and the README must say so). No cross-arm calibration exists on rigs yet;
  a future `arm_base_poses` config knob can lift both arms into one shared
  frame without changing this interface. The frame convention is stated in the
  space's `ActionSemantics.frame` ("base") and must be spelled out in the
  agent-facing bounds text via dim naming alone (the plugin already
  enumerates labels and bounds into the tool description).
- `yaw` is rotation about base +z, absolute-not-delta (a target, not a
  displacement; its zero point is relative to the reset orientation, per
  the convention below), bounded — NOT wrapped: a
  3.1 → −3.1 command linearly sweeps ~2π through zero rather than taking
  the short way. The tool description must say so and advise intermediate
  yaws for near-±π regrasps. Full orientation (rot6d) is a possible later
  interface value; position+yaw covers tabletop regrasp/alignment (fork
  rotation) without asking a VLM to emit 6-D rotation matrices, which the
  research says they are bad at.
- Yaw convention, pinned — and **relative to the reset orientation**: at
  the end of `reset()` each arm captures its FK orientation `R0`, extracts
  `yaw0 = atan2(a_y, a_x)` for `a` the horizontal projection of the gripper
  frame's x-axis (fallback to the y-axis when that projection has norm
  < sin(5°) ≈ 0.087 — tool axis near vertical — with the chosen branch
  **pinned for the whole trial** for both reference and reporting, so
  noise cannot flip it mid-trial), and stores `R0` itself as the
  reference. Commanded yaw `c` targets `R_target(c) = Rz(c) · R0`, and the
  reported `eef_state` yaw is `wrap(yaw_raw − yaw0)` via the pinned
  branch. The two halves are consistent: at orientation `Rz(d) · R0` the
  report reads `d`, and commanding `c = d` targets the current
  orientation — echo-back holds still, and home reads and commands
  **exactly 0** by construction. (Do NOT pair relative reporting with a
  `Rz(−yaw0) · R0` reference — commanding the reported 0 would then
  target a 180°-flipped orientation at this home, since `yaw0 = π`.) — this matters: the §5b home orientation's
  raw x-axis projection points along −x (`atan2(0, −0.5) = π`), so an
  absolute convention would park the default pose *on* the ±π wrap
  discontinuity, where noise flips the reported sign and an agent echoing
  the observed yaw back would command a full 360° wrist sweep through
  zero. Under the relative convention the discontinuity sits at 180° from
  home — a pose the bounded, non-wrapped interpolation already discourages
  — and echo-back near home is stable (±ε reads as ±ε). Roll/pitch are
  held at their reset-captured values for the whole trial; the reference
  is captured once, never re-read from measurements, so tracking error
  cannot integrate into orientation drift. A signed FK-based test pins the
  sign convention, and a test pins home-reads-zero. (The x-axis projection
  at home has norm 0.5 ≈ 5.7× the fallback threshold — comfortably clear,
  not "an order of magnitude".)
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

Per-arm `Kinematics` instances are constructed at **first `reset()`**
(combined arm+gripper model, `grasp_site`) — the same lazy point as the
drivers and cameras, keeping heavy imports out of `__init__` per the
repo's established pattern. A bad kinematics config therefore errors at
reset, exactly like a bad CAN channel does today.

- `step(action)` in eef mode runs this exact per-arm pipeline — the order
  is normative, not illustrative:

  1. **Hold check.** If the arm's `hold_counter > 0`: decrement, re-send
     `q_cmd_prev` unchanged, and return for this arm — no IK, no resync,
     no counter updates; frozen means frozen. (The gripper dimension is
     NOT frozen: it never enters IK, cannot oscillate, and freezing an
     in-flight grasp is strictly worse — it continues through the driver
     path during a hold.)
  2. **Resync check.** If any joint's `|q_cmd_prev − q_measured|` exceeds
     `cmd_resync_threshold` (config, default 0.35 rad), re-seed
     `q_cmd_prev := clip(q_measured, effective per-side intersection)`
     (the same range the IK model enforces — NOT the raw config limits,
     which can be wider than a one-sided model range) and mark the step
     `resynced` — bounding command windup ahead of a stalled or
     obstructed arm.
  3. **IK.** Build the target (position, `R_target(yaw)` per §4);
     differential IK warm-started from `q_cmd_prev` clipped into the
     model's joint ranges (mink's limit check logs epsilon-outside starts
     at DEBUG level in the pinned version — clipping is still required:
     it keeps the solve well-posed and immune to `check_limits`'
     construction-time-snapshotted ranges after the wrapper mutates
     `jnt_range`), iteration-capped
     (`ik_max_iters`, default 20 — warm starts make centimeter steps
     converge in a few iterations, and the cap bounds the non-convergent
     worst case well inside the 100 ms budget). Convergence failure is
     NOT an exception: the solver's **last** iterate (that is what
     `Kinematics.ik` returns on failure — not a best-of pick) still moves
     toward the target, and the truth shows up in the next observation's
     FK state, closing the loop. Two edges, pinned: a solve returning any
     **non-finite** value degrades this step to a hold (re-send
     `q_cmd_prev`) — commands are generated here, *downstream* of the
     approvers whose NaN abort is the repo's "must never reach hardware"
     bar, so the gate has to exist locally; and a QP-infeasibility
     exception from mink (`NoSolutionFound`) is NOT caught — it
     propagates as an `EmbodimentFault` and halts the eval, loudly. Only
     iteration-cap non-convergence is best-effort.
  4. **Rate clamp** (below) against `q_cmd_prev`.
  5. **Reversal counting** on `q_cmd − q_cmd_prev`, skipped entirely on a
     `resynced` step: the resync snap-back is corrective, not
     oscillatory, and counting it would put an obstructed arm into a
     perpetual hold/resync limit cycle. (The obstructed-arm steady state
     is therefore a bounded press/resync cycle — at most two rate-limited
     steps of advance before each re-seed, force bounded by the motor
     drivers' own current limits, fully visible in the observation.) A
     trip starts the whole-arm hold and clears the window.
  6. **Send** through `_send()` (absolute config clamp, unchanged), then
     `q_cmd_prev := the value actually sent` (post-clamp — after a resync
     from out-of-limit measured joints this differs from the pipeline
     value, and the reference must track reality).

  State lifecycle and scope, normative: `q_cmd_prev`, the resync test, the
  rate clamp, and the reversal machinery all cover the **six arm joints
  only** — the gripper dimension never participates (it never enters IK,
  and a gripper closed on an object holds a commanded-vs-measured gap of
  ~0.4 normalized *by design*, which would otherwise mark every step
  `resynced` and permanently disable reversal counting exactly during
  contact-rich phases). `q_cmd_prev` (6-D per arm) is seeded from the
  final home-ramp command at the end of `reset()`; `hold_counter`, the
  reversal window, and the `resynced` flag are all cleared by `reset()`
  (a hold must not leak across trials and freeze the opening steps of the
  next one) and by `close()`. On the step that trips the guard, the hold
  replaces that step's command — the reversing `q_cmd` is not sent. A
  `resynced` step occupies a normal slot in the sliding window (recorded
  as no-reversal); the window does not freeze. The resync re-seed clips
  the measured joints into the same *effective* range the IK model
  enforces (the §5 per-side intersection), keeping the warm-start and
  clamp references aligned by construction.
- Oscillation damping (anti-livelock), step-observable — `step()` has no
  chunk concept, so the guard cannot reference chunk boundaries, and a
  position-progress rule would falsely freeze pure-yaw regrasps (position
  error is static during rotation). Per arm joint, a commanded-direction
  *reversal* is counted only when both deltas exceed a deadband
  (`osc_deadband`, default 0.005 rad — converged solves alternate delta
  signs at floating-point scale and must not count). If any joint of an
  arm accumulates more than `osc_reversals = 2` reversals within a sliding
  `osc_window = 6`-step window, **the whole arm** holds its last command
  for `osc_hold_steps = 10` steps (1 s at 10 Hz) — holding only the
  tripped joint would execute a mixture of branch-A and held joints, a
  configuration on neither IK branch with its own uncontrolled EEF
  excursion. After a hold expires, that arm's window and counters clear.
  All four are config knobs. A branch-flip clamp cycle — the failure this
  kills — reverses direction every step and gets duty-cycled to ≤ 3
  reversals per 1.6 s instead of oscillating ±`ik_step_joint_limit` at
  10 Hz for a full 10 s chunk; monotone approaches and single overshoots
  never trigger it; a legitimate fine-positioning dither at worst pauses
  the arm for 1 s and resumes. The observation meanwhile reports true
  state, so a persistent stall is visible to the agent, which replans.
- Stated plainly for the safety record: rate-clamped intermediate
  configurations are not IK solutions, so the EEF path during a clamped
  branch transit is uncontrolled (bounded per step by the joint clamp,
  bounded overall by joint limits and the hold guard) — this is the
  mechanism behind the tens-of-centimeters residual excursion documented
  below.
- **Joint-space rate backstop (safety-critical; pipeline step 4).** The
  Cartesian `DeltaLimitApprover` cannot see joint deltas, and an IK branch
  flip (elbow reconfiguration near a singularity or limit) can return a
  solution radians away from the current pose. The clamp is
  `q_cmd = q_cmd_prev + clip(q_ik − q_cmd_prev, ±ik_step_joint_limit)` —
  the same reference as the warm start, keeping the command path
  deterministic and encoder-noise-free (clamping against measured joints
  would feed noise into saturated deltas and the reversal counter); the
  windup this invites is what pipeline step 2's resync bounds.
  `ik_step_joint_limit` is a config knob
  defaulting to 0.2 rad per joint per step (the same scale as delta mode's
  `_STEP_ARM`). This restores in eef mode the "no wild swings" guarantee
  joint mode gets from the core guardrails — in *joint space*. Stated
  honestly: it bounds Cartesian motion only loosely (6 joints × 0.2 rad ×
  ~0.9 m lever can traverse tens of centimeters of EEF translation in one
  step during a multi-step branch transit, ~20× the Cartesian delta limit);
  the residual exposure is bounded, brief, and visible in the observation,
  and joint mode's own step limits carry the same order of exposure. The
  absolute joint limits additionally stay enforced in `_send()` as today.
- Config joint limits and IK: the effective range for each of the six arm
  joints is the **per-side intersection**
  `[max(model_lo, cfg_lo), min(model_hi, cfg_hi)]` (the default config
  ±π is *wider* than the model's one-sided [0, 3.65] shoulder/elbow ranges
  below and tighter above — a one-sided "where tighter" rule would get
  this wrong). Mechanism, chosen explicitly: the `_ArmKinematics` wrapper
  (§5c) applies the intersection through the seam's read/write-ranges
  affordance, mutating the wrapper-owned model
  (`Kinematics._configuration.model`) right after construction — the
  wrapper wholly owns that model instance, mink's `ConfigurationLimit`
  reads ranges live from it on every solve, and `Kinematics` exposes no
  limits seam of its own, so owned-model mutation is the least-invasive
  supported path (no XML rewriting). Placement matters: the intersection
  and its non-empty validation are wrapper logic (tier-1 testable against
  fakes), never factory logic — §5c's rule that nothing spec'd lives only
  inside the lazily-importing factory applies here too. The intersection is validated **non-empty per joint** at
  kinematics construction — an operator config whose range misses a
  one-sided model range entirely (e.g. `cfg_hi < 0` on a [0, 3.65]
  shoulder) would otherwise write an inverted `jnt_range` with undefined
  solver behavior; it errors with the offending joint named instead. The
  solver then respects operator-tightened limits instead of having
  `_send()`'s clamp silently distort the Cartesian solution afterward.
- Gripper joints in the IK model: the gripper joint count varies by gripper
  type — LINEAR_* grippers contribute two equality-coupled, meter-ranged
  slide joints; CRANK_4310 contributes none (its XML has no joints). The
  wrapper addresses the six arm joints positionally as the first six and
  handles 0..2 trailing gripper joints: whatever gripper joints exist are
  pinned to their model mid-range in `init_q` — and in every `fk()` call
  (the observation's FK needs full-length q vectors too; `grasp_site`
  lives in a static body, so the gripper joints have zero influence on
  the site pose and the pin value is immaterial there) — and stripped
  from the returned solution. The commanded normalized gripper value flows through
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
the tool pitched 30° from vertical toward the base — precisely: the
*gripper-frame x-axis* (jaw axis) is 30° from straight-down; the physical
approach axis (site z) is 60° from vertical — and yaw 0 (per-arm
joints ≈ [-0.024, 0.794, 0.645, -0.375, -0.021, -0.012], grippers open —
provisional values from the 2026-07-14 probe; implementation re-derives
them with the tier-2 test and records the derivation script). This
orientation is what §4's box validation holds, and it is what makes the
yaw extraction robust (§4). The joint-space zero rest pose is NOT a valid
eef-mode start: its EEF sits outside the default box and its vertical tool
axis is yaw-degenerate. Accordingly, `home_pose=None` — which in joint
mode means "skip homing" — is redefined in eef mode to select
`DEFAULT_EEF_HOME_POSE`: an un-homed eef-mode start would bypass the
FK-in-box validation and begin yaw-degenerate, so skipping is not offered
in this mode.

> **Amended by plan 0007 (2026-07-15):** the statement above that
> `home_pose=None` skips homing in joint mode is superseded. Joint mode now
> selects the mandatory `DEFAULT_JOINT_HOME_POSE` and always homes.

At first `reset()` in eef mode (once kinematics exist, §5), the configured
(or default) home pose is validated: FK(home) must lie inside the
workspace box, erroring out otherwise. This kills the out-of-box-*start*
failure mode where the agent's first partial `move_to` would clip *held*
dimensions to the box boundary and command unintended motion (the toolset
clips every interpolant into the box, including unnamed held dims).
Mid-trial best-effort drift can still leave a held dimension epsilon
outside the box, and the toolset then clips it back in-box — a small
motion in the safe direction; documented, not prevented.

`reset()` captures the per-arm yaw references (§4) after homing completes,
and its returned observation already carries `eef_state`.

### 5c. The kinematics seam

`YAMEmbodiment.__init__` gains a `kinematics_factory` keyword argument —
the same seam pattern as the existing `driver_factory` constructor kwarg
(NOT a `YamConfig` field: the config is a frozen, CLI-constructable
dataclass of scalars/tuples reachable via `-E key=value`, and a callable
would break that contract). The seam is placed so that every behavior this
plan specifies is *plugin* code, tier-1 testable against fakes:

- The factory returns per-arm **raw** solver objects satisfying the
  minimal protocol `fk(q) -> 4x4` / `ik(target, init_q, max_iters) ->
  (bool, q)`, plus a way to read/write the model joint ranges
  (`ik_max_iters` must be expressible through the seam — §5 requires it
  plumbed to the solver). The default factory lazily imports i2rt
  (`_i2rt.py` conventions) and returns a thin *adapter* binding
  `site_name="grasp_site"` over `i2rt.robots.kinematics.Kinematics`
  (whose `ik()` takes `site_name` as a required positional, so raw
  instances do not satisfy the protocol); tests inject deterministic
  fakes.
- A plugin-level `_ArmKinematics` wrapper — ordinary, always-importable
  plugin code — owns everything else: first-six positional arm-joint
  addressing, gripper-joint pinning/stripping (0..2 trailing joints),
  the per-side `jnt_range` intersection with config limits, warm-start
  clipping, the rate backstop, and the oscillation guard. Tier-1 tests
  exercise all of it through injected raw fakes; nothing spec'd lives
  only inside the lazily-importing factory.

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
and debugging — and, acknowledged deliberately: the agent plugin renders
*every* state key into the observation text, so the LLM sees the raw joint
vector alongside `eef_state` each turn. That is accepted, not fought: the
extra 14 numbers are redundant context the model may ignore, and hiding
them would take a core-plugin filtering feature this plan does not need.
Its shape cannot collide with (10,).

## 7. yam tests (TDD, two tiers)

**Tier 1 — gated (100% coverage, no i2rt):** everything runs against the
injected fake kinematics (§5c) and the existing fake driver.

- Config: `control_interface` validation (unknown value, delta+eef
  combination), default unchanged; new config knobs (`eef_low/high`
  including the yaw-bounds-within-[−π, π] rule, `ik_max_iters`,
  `ik_step_joint_limit`, `cmd_resync_threshold`, `osc_deadband`,
  `osc_reversals`, `osc_window`, `osc_hold_steps`) validated.
  (`kinematics_factory` is a constructor kwarg, not a config field — its
  tests live with the embodiment, not the config.)
- Space: eef space shape/labels/bounds/semantics; observation StateSpec
  declares `eef_state` at (10,) exactly once alongside `joint_pos` (14,).
- step(): 10-D action → both arms commanded; joint-space rate backstop
  clamps a fake-IK solution placed radians away (the elbow-flip case) to
  `ik_step_joint_limit`; gripper passthrough never enters IK and gripper
  slide joints are pinned/stripped; best-effort on fake non-convergence
  (finite, in-limit joints commanded); `ik_max_iters` plumbed through;
  config joint limits forwarded to the IK seam.
- Yaw: sign convention against a hand-built rotation; home reads exactly
  0 under the relative convention; vertical-tool-axis fallback extraction
  with the branch pinned across a threshold-crossing fake measurement;
  references captured at reset and never re-read (feed a drifting fake
  measurement, assert the target orientation is unaffected).
- Oscillation damping: a fake IK alternating between two branches trips
  the reversal counter and the joint holds for `osc_hold_steps`, then
  re-evaluates; a monotone approach and a single overshoot never trigger
  it; a pure-yaw motion is unaffected (named test — this is spec'd
  safety behavior, not incidental coverage).
- reset()/close(): joint-space homing/parking unchanged in eef mode;
  reset observation carries `eef_state`; yaw references captured
  post-homing; a hold active at trial end is cleared by the next
  `reset()` (window, counters, and flag too) and does not freeze the next
  trial's opening steps.
- Gripper exclusion: a fake trial with the gripper commanded closed on an
  "object" (large commanded-vs-measured gripper gap) never marks steps
  resynced and keeps reversal counting live on the arm joints.
- Empty limit intersection (config range disjoint from a one-sided model
  range) errors at kinematics construction naming the joint.
- Non-finite IK output (fake solver returning NaN) degrades the step to a
  hold — `q_cmd_prev` re-sent, nothing non-finite reaches the driver; a
  fake raising the solver's infeasibility exception propagates (halts),
  not caught.
- Integration with the agent plugin, through its *public* surface only
  (`LLMAgentPolicy` with an injected `httpx` transport — the private
  `_tools.build_toolset` is not in the plugin's `__all__`, and a private
  cross-package import would let any agent-plugin refactor break this
  repo under the open-ended version pin): a scripted conversation binds
  the policy to the eef space, asserts the advertised tool carries the 10
  labels and correct bounds text, drives a straight-line move, and the
  emitted chunk passes `ChainApprover(Clamp, DeltaLimit)` untouched. (Clean-case only, noted:
  after a best-effort divergence, `DeltaLimitApprover`'s store still
  references the last *approved* action, so the next chunk's first steps
  can clamp toward the stale target for a few steps — bounded, safe
  direction, documented rather than tested.) Dependency mechanism,
  spelled out: this repo is not a uv workspace — add
  `inspect-robots-agent>=0.2.2` (PyPI) to the `dev` extra and re-run
  `uv lock`. The integration test must not assert the move tool's *name*
  (the §8 companion PR renames it in pose modes; fresh locks would pull
  the renamed version and a name assertion would couple the repos).
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
  reach **< 5 mm position error** at the §5b held orientation with
  radially-pointing yaw (commanded in the §4 relative convention) — the
  exact validation that produced the defaults. Assert position error, NOT
  mink's success flag: the solver reports failure on its orientation
  threshold at probe points that sit at sub-millimeter position error
  (measured: (0.15, 0, 0.215) returns success=False at 0.75 mm).
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
  enumerating per-dim bounds. The description also states, in
  embodiment-agnostic terms, the two contracts the agent cannot guess:
  the frame contract ("coordinates are absolute in the embodiment's
  declared frame; on multi-arm embodiments each arm uses its own base
  frame and axes may differ between arms depending on mounting") and the
  rotation contract ("rotation dimensions are absolute targets measured
  relative to the trial's start orientation — 0 means the start
  orientation — and interpolate linearly without wrapping, so prefer
  intermediate values for large rotations"). Without the latter the agent
  sees only `left_yaw: [-3.14, 3.14]` and §4's conventions never reach
  the model that must obey them. Scope, stated normatively: this
  relative-to-start convention becomes the *tool surface's contract* for
  scalar rotation dimensions in pose modes, documented as such in the
  plugin — any embodiment declaring `eef_abs_pose` with
  `rotation_repr="none"` scalar rotation dims must implement rotation
  relative to the trial start (as this plan's yam mode does), or not use
  this mode. A per-embodiment free-text hint channel that would let
  embodiments override such wording is explicitly out of scope (§9).
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
- A per-embodiment free-text hint channel on `ActionSemantics` that would
  let embodiments inject their own tool-description wording (referenced
  from §8).
- Any change to VLA policy paths or the joints default.

## 10. Rollout

yam PR first (self-contained; agent plugin works against it unchanged, with
the tool merely still named `move_joints`), core PR second (rename polish),
then a yam release and a core release per the usual one-click flows. The
omen rig opts in via `[embodiment.args] control_interface = eef_pos`.
