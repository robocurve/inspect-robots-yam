# 0024: Named start poses (zero-g capture CLI + `start_pose` config)

Issue: #128. Branch: `feat/start-poses`.

## Goal

Evals begin from custom, named starting poses, and operators author those
poses by hand: a new `inspect-robots-yam-pose` CLI puts both arms in
gravity-comp idle ("zero-g"), the operator poses them by hand, and the
snapshot is saved as a named pose file. Poses are plain JSON files, one per
name, so they are swappable (change one config string) and shareable (copy or
commit the file).

## What already exists (do not rebuild)

- **Zero-g is the default bring-up mode.** `_default_driver_factory` passes
  `cfg.zero_gravity_mode` (default `True`) to `get_yam_robot`; i2rt's
  `MotorChainRobot` then idles in gravity compensation until the first
  `command_joint_pos`. Capture needs no new driver mode, only "connect and
  read". Commanding later (park, goto ramp) works from zero-g bring-up: evals
  already connect this way and then home.
- **Custom start poses already work mechanically** via
  `YamConfig.home_pose` feeding `_home_pose()` and the homing ramp in
  `reset()`. This plan adds naming/storage/resolution on top; the motion path
  is untouched.
- **Wire shape convention:** `home_pose`/`rest_pose` are 14-D with gripper
  slots normalized 0-1 (1 = open); the driver speaks native gripper units.
  `_norm_grippers`/`_denorm_grippers` convert. Captured poses are stored
  wire-shaped (radians + normalized grippers) so they are portable across
  rigs with different `gripper_open`/`gripper_closed` calibrations.
- **CLI conventions:** `health.py` is the worked example: bare invocation
  honors the working directory's `.env` including an
  `INSPECT_ROBOTS_CONFIG` pin, wizard config supplies device args via
  `_user_config.load_yam_defaults`, `-E key=value` extras override, and
  `--no-config` restores flag-only behavior. Exit 0 ok / 1 hardware or
  operation failure / 2 usage via `parser.error`. All hardware and stdin
  behind injected seams; only real-I/O defaults are `# pragma: no cover`.

## New module: `poses.py` (pure store)

Stdlib + numpy only, no optional deps, no hardware.

- `POSE_SCHEMA = 1`.
- `@dataclass(frozen=True) class StartPose`: `name: str`,
  `joints: tuple[float, ...]` (14, wire-shaped), `created_at: str` (ISO
  8601), `notes: str = ""`, `rig: str | None = None`.
- File layout: `<pose_dir>/<name>.json` holding
  `{"schema": 1, "name", "joints", "created_at", "notes", "rig"}`.
- Functions: `pose_path(pose_dir, name) -> Path`,
  `save_pose(pose_dir, pose, *, overwrite: bool = False)`,
  `load_pose(pose_dir, name) -> StartPose`,
  `list_poses(pose_dir) -> tuple[StartPose, ...]` (sorted by name; a file
  that fails to load raises with the file named),
  `delete_pose(pose_dir, name)`, `rename_pose(pose_dir, old, new)`.
- Name rule (single source of truth, used by every entry point):
  `^[A-Za-z0-9][A-Za-z0-9._-]*$` and no more than 64 chars. Rejects path
  separators by construction. `load_pose` also rejects a file whose embedded
  `name` does not match its filename stem.
- `pose_names(pose_dir) -> tuple[str, ...]`: sorted filename stems of
  `*.json` in the dir, WITHOUT loading the files. Every "available names"
  error message enumerates via this helper, so one corrupt file cannot
  poison missing-pose or delete error paths (`list_poses` alone loads
  fully and raises naming the corrupt file).
- `rename_pose` rewrites the embedded `name` to the new stem (it must,
  or the store would reject its own file on the next load) and preserves
  `created_at`, `notes`, and `rig`. A load-after-rename round-trip is a
  required test.
- Load validation: known `schema`, exactly 14 entries, all finite floats.
  Joint-LIMIT validation deliberately does NOT live here: limits are
  config-dependent, so use sites check against `cfg.low`/`cfg.high`.
- Errors: raise `PoseStoreError(ValueError)` with actionable messages
  (missing file lists the pose_dir and available names; overwrite without
  `overwrite=True` says to pass `--force`).

## Shared gripper-norm helpers: `packing.py`

Move the conversion math into pure functions next to the packing contract so
the CLI and the embodiment cannot drift:

- `norm_grippers(vec, *, gripper_open, gripper_closed) -> Vec` and
  `denorm_grippers(vec, *, gripper_open, gripper_closed) -> Vec`, operating
  on copies, touching only indices `ARM_DOF` and `ARM_WIDTH + ARM_DOF`.
- `YAMEmbodiment._norm_grippers`/`_denorm_grippers` become one-line
  delegations passing `self._cfg.gripper_open`/`gripper_closed`. The `_send`
  clamp-then-denorm order and the step() clamp backstop are retained
  unchanged; this refactor moves math, not behavior, and the existing
  embodiment tests must keep passing without assertion changes.

## Config: `YamConfig` additions

- `start_pose: str | None = None` — name of a stored pose to home to.
- `pose_dir: str = "poses"` — pose store directory; a relative path resolves
  against the working directory (the rig dir, matching `.env` conventions).
- `__post_init__` validation (alongside the existing home_pose length
  check, which is retained):
  - `start_pose` and `home_pose` both set -> `ValueError` naming both keys.
  - `start_pose` with `control_interface="eef_pos"` -> `ValueError`
    (poses are joint-space; eef conversion is out of scope, say so).
  - `start_pose` set but empty/whitespace, or failing the pose-name rule ->
    `ValueError` quoting the rule.
  - `pose_dir` empty/whitespace -> `ValueError`.
- Both are plain strings: no `_FLOAT_TUPLE_FIELDS` change. But the core
  scalar parser coerces digit-only `-E`/config.ini values to `int` (the
  name rule permits `42`), so `from_kwargs` needs the same guided
  string-coercion guard the depth serials already have (config.py's
  existing pattern): a non-string `start_pose`/`pose_dir` is str()-coerced
  with the quoting hint, never a bare TypeError.

## Embodiment: resolve `start_pose` at first reset

In `reset()`, immediately BEFORE `self._driver = self._driver_factory(...)`
(fail fast, before arms power): if `cfg.start_pose` is set and not yet
resolved, `poses.load_pose(cfg.pose_dir, cfg.start_pose)`, then validate
against this config: 14-D (store guarantees it) and within
`cfg.low`/`cfg.high` per element; out of range -> `ValueError` naming the
pose, the offending packed indices, and the values vs bounds. Cache the
resolved vector; a retried reset does not re-read the file. `close()`
clears the cache alongside `_driver`/`_init_pose`/`_home_gate_confirmed`,
so a reconnected instance re-reads the (possibly edited) file.

`_home_pose()` returns the cached resolved pose when present; otherwise its
existing branches run unchanged (the eef default branch is retained, and is
unreachable with `start_pose` set because config validation already rejected
that pairing). The homing status line becomes
`homing: ramping arms to start pose '<name>'` when a named pose is in use so
eval logs record which pose ran. Everything downstream (stand-clear gate,
ramp through `_send` with clamp + denorm, settle, park-on-close semantics,
`rest_pose` behavior) is retained byte-for-byte.

Provenance: the homing status line becomes
`homing: ramping arms to start pose '<name>'` when a named pose is in use;
note this reaches the terminal/status ticker only when `not unattended`, so
a `logger.info` with the pose name and file path fires unconditionally at
resolution time. That log line is the provenance record; the eval log
proper is unchanged.

Collision honesty: the homing ramp goes `_ramp_to` -> `_send` ->
`command_joint_pos` directly and is joint-limit-clamped ONLY. The collision
guardrail (`CollisionApprover`) wraps policy actions during `step()` via
`contribute_guardrails` and never sees homing, so a straight-line joint
interpolation from rest to an arbitrary captured pose is collision-unchecked.
This is exactly the exposure a hand-set `home_pose` already has today; the
feature widens usage, not the mechanism. The README section must tell
operators to verify a new pose with `goto` while ready on the e-stop before
wiring it into unattended evals.

## New CLI: `pose_cli.py` -> `inspect-robots-yam-pose`

`[project.scripts]` gains
`inspect-robots-yam-pose = "inspect_robots_yam.pose_cli:main"`.

Follows `health.py`'s main() conventions (env/config pin, wizard defaults
via `load_yam_defaults`, `-E` extras, `--no-config`, exit codes 0/1/2).
Injected seams: `driver_factory` (default `embodiment._default_driver_factory`),
`io: OperatorIO` (prompts and prints), `sleep_fn`, `now_fn` (for
`created_at`), `hostname_fn` (for `rig` provenance). Only the real-default
wiring in `main()` is pragma'd.

Config plumbing decisions (they differ between the eval path and the CLIs,
so they are explicit here):

- `load_yam_defaults` filters wizard `[embodiment.args]` to
  `_YAM_DEVICE_KEYS`, while the eval path passes ALL embodiment args
  through unfiltered. So that a config-pinned `pose_dir`/`start_pose` is
  honored consistently, `load_yam_defaults` gains an optional
  `extra_keys: frozenset[str]` parameter (default empty: health/preflight
  behavior unchanged) and the pose CLI requests
  `{"pose_dir", "start_pose"}`. Values are str()-coerced by the existing
  loader path, which is correct for both keys.
- `-E start_pose=...`/`-E pose_dir=...` join the CLI's raw-string key set
  (health.py's `_RAW_STRING_KEYS` analogue) so digit-only values are not
  int-coerced before `from_kwargs`.
- `--pose-dir` and `-E pose_dir` both set -> usage error (exit 2),
  mirroring health's flag-vs-`-E` duplicate rule.
- Every subcommand, including store-only ones, reads the wizard config the
  same way (uniformity: a config-pinned `pose_dir` applies to `list` too).
  Consequence, stated in the CLI docstring: a malformed config.ini breaks
  `pose list` without `--no-config`, exactly as it does for health.

Interactivity guard: `capture` and `goto` block on Enter prompts, so with a
non-interactive stdin the EOF would otherwise land AFTER hardware connect
and the teardown would drop torque on hand-posed arms with nobody at the
rig. Both subcommands therefore fail fast (usage error, exit 2, before
`driver_factory` is called) when `operator.stdin_interactive()` is false,
unless a non-default `io` seam is injected (tests). Same precedent as
`auto_start`'s pre-motion TTY check in `reset()`.

Global flags: `--pose-dir` (overrides `cfg.pose_dir`), `-E`, `--no-config`.

Subcommands:

- `capture <name> [--notes TEXT] [--force] [--clamp] [--park]`
  1. Build config; pin `zero_gravity_mode=True` via `dataclasses.replace`
     regardless of config so the arms come up hand-movable.
  2. Refuse an existing pose name up front (before touching hardware)
     unless `--force`.
  3. Connect (driver factory), print: arms are in gravity-comp idle, pose
     them by hand (including gripper apertures), press Enter to snapshot;
     Ctrl-C aborts without writing (teardown still prompts to support the
     arms before torque-off).
  4. On Enter: `packing.validate_dim(driver.get_joint_pos())`, then
     `packing.norm_grippers(...)`.
  5. Range handling: gripper slots are silently clamped to [0, 1]
     (hand-set apertures overshoot by measurement noise). Arm joints outside
     `cfg.low`/`cfg.high` are an error (exit 1) listing each offending
     joint, unless `--clamp`, which writes the clamped values and prints
     exactly what changed. Rationale: an out-of-range pose would be
     silently distorted by the eval-time clamp backstop, which is retained
     and NOT relaxed by this feature.
  6. Write the pose file; print the path and a ready-to-use
     `-E start_pose=<name>` hint.
  7. Safe exit. Two prompts with OPPOSITE operator postures; never blur
     them:
     - Torque-off prompt (default): "support both arms, then press Enter
       to release torque", then close (`close_robot_safely` path). i2rt
       `close()` drops torque, so unsupported arms fall; this wording is
       the only place "support the arms" appears.
     - `--park`: a stand-clear Enter gate FIRST ("arms will move to the
       rest pose - stand clear, then press Enter", the embodiment homing
       gate wording), because the operator has just been hand-posing and
       plausibly still has hands on the arms; only after that gate does
       the ramp to `cfg.rest_pose` run (same interpolation semantics as
       the embodiment ramp: `rest_secs`, `control_hz` with the same
       `hz <= 0 -> 10.0` fallback, `sleep(1/hz)` per waypoint, clamp +
       denorm each waypoint), then close. `--park` with `rest_pose=None`
       is a usage error (exit 2).
     The Ctrl-C abort path (step 3) writes nothing but still runs the
     torque-off prompt via the same teardown.
- `goto <name> [--park]`
  1. Resolve the pose from the store; validate against `cfg.low`/`cfg.high`
     (error, no `--clamp` here: fix the pose or the config instead of
     moving somewhere else than asked).
  2. Connect, stand-clear Enter gate (same wording convention as the
     embodiment homing gate), ramp to the pose, report the final measured
     pose, then hold. Exit follows the same two-prompt discipline as
     capture: default is the support-the-arms torque-off prompt; `--park`
     interposes its own stand-clear gate, ramps back to `rest_pose`, then
     torque-off (with `rest_pose=None`, `--park` is a usage error, exit 2,
     same as capture).
- `list` — table of name, created_at, rig, notes (store only, no hardware).
- `show <name>` — joints printed per arm (6 joint radians + gripper 0-1),
  plus metadata and the file path.
- `delete <name>` / `rename <old> <new>` — store ops; delete of a missing
  pose exits 1 with the available names.

The capture/goto ramp reuses one shared interpolation helper: extract the
waypoint loop of `_ramp_to` into a module-level pure function in
`embodiment.py` (`ramp_waypoints(start, target, n) -> Iterator[Vec]` or
equivalent) used by both `_ramp_to` (behavior unchanged, existing ramp
tests must pass without assertion changes) and the CLI (which applies
`np.clip(cfg.low, cfg.high)` + `denorm_grippers` per waypoint, mirroring
`_send`).

## Docs

- README: a "Named start poses" section under the operator tooling docs:
  capture workflow, file format example, `-E start_pose=name` usage,
  sharing (commit/copy `poses/`). Follow the repo writing-style rules (no
  em dashes, minimal bold).
- CHANGELOG: minor-release entry.
- `src/inspect_robots_yam/CLAUDE.md` module table: add `poses.py` and
  `pose_cli.py` rows.
- Root `CLAUDE.md`: no changes needed (safety invariants unchanged).

## Tests (gates: ruff, mypy --strict, pytest --cov at 100%)

- `tests/test_poses.py`: round-trip save/load; overwrite refusal + `--force`
  path via `overwrite=True`; name rule (rejects `../evil`, empty, >64,
  leading dot); schema/dim/nonfinite rejection; stem-vs-name mismatch;
  list ordering + corrupt-file error naming the file; `pose_names` returns
  stems and is unaffected by a corrupt sibling (missing-pose errors stay
  usable); rename rewrites the embedded name, preserves created_at, and
  round-trips through load; delete/rename incl. missing-source and
  existing-target errors.
- `tests/test_config.py` additions: mutual exclusion, eef rejection,
  bad-name rejection, empty `pose_dir` rejection, digit-only
  `start_pose`/`pose_dir` int values str()-coerced by `from_kwargs` with
  the guided hint, defaults land in `from_kwargs`.
- `tests/test_user_config.py` additions: `extra_keys` admits
  `pose_dir`/`start_pose` for the pose CLI and default calls stay
  device-keys-only.
- `tests/test_embodiment.py` additions: `start_pose` resolves and homing
  ramps to it (fake driver records commands); resolution failure raises
  BEFORE `driver_factory` is called (assert factory not invoked);
  out-of-limits pose error names indices; retried reset does not re-read
  (count `load_pose` calls via tmp file mutation or monkeypatch); status
  line carries the pose name; gripper norm delegation unchanged
  (existing tests keep passing).
- `tests/test_pose_cli.py`: fake driver + scripted `OperatorIO`; capture
  happy path writes wire-shaped JSON (gripper normalization asserted
  numerically); gripper-slot silent clamp; joint out-of-range error and
  `--clamp` diff output; `--force`; Ctrl-C abort writes nothing and still
  prompts before torque-off; `--park` ramps then closes; `--park` without
  rest_pose exits 2; goto ramp waypoint count and hold/release flow;
  goto's `--park` stand-clear gate precedes any motion; non-interactive
  stdin exits 2 before `driver_factory` for capture/goto (assert factory
  not called) while injected `io` bypasses the guard; `--pose-dir` +
  `-E pose_dir` duplicate exits 2; list/show/delete/rename output;
  `-E`/config/env handling mirroring the health CLI tests; every usage
  error exits 2.
- `tests/test_api_snapshot.py`: the package root `__init__.__all__` is
  deliberately unchanged (the pose store's public surface is the
  `inspect_robots_yam.poses` module itself plus the CLI); the snapshot
  test should need no edits, which is itself the assertion.

## Out of scope (say no)

- eef-mode start poses (joint-space only; config rejects the pairing).
- Per-pose collision validation beyond joint limits. Homing and goto ramps
  are collision-unchecked today (see "Collision honesty" above) and stay
  that way; sweeping the ramp through `CollisionChecker` is a separate
  follow-up issue to file at implementation time, not silent scope creep
  here.
- Core `inspect-robots` changes: none needed, this is yam-only.
- Pose "libraries"/search paths beyond one `pose_dir` (a later plan can add
  it; the file format carries `schema` for that reason).

## Delivery

Implementation lands as one PR (`Closes #128`) from this branch. On-rig
validation happens after merge on rig-1 once it is re-zeroed (left arm is
currently outside joint limits); the suite itself needs no hardware.
