# `inspect_robots_yam` package — module map

Three Inspect Robots components + the glue to make them an honest, testable, safe pair.
The package is `mypy --strict` clean, ships `py.typed`, and is 100%-covered.

## Modules

| Module | Responsibility |
|--------|----------------|
| `packing.py` | **Pure** 14-D bimanual packing — the single source of truth for how the flat vector maps to two arms (`[j0..j5, gripper]` per arm, left then right). `pack`/`split`/`validate_dim`, `STATE_KEY`, `STATE_SPEC`. No optional deps. |
| `config.py` | `YamConfig` / `ActServerConfig` (with the `MolmoActConfig` alias; frozen, `from_kwargs` for CLI scalars) + joint and EEF action/observation-space builders. |
| `operator.py` | `OperatorIO` (injectable stdin/stdout) for the readiness prompt; `default_poll_end` (real TTY poll, `# pragma: no cover`); `stdin_interactive`, the TTY probe behind `auto_start`'s pre-motion fail-fast. Verdicts + grader notes are the framework prompt's job. |
| `_i2rt.py` | Lazy i2rt loader + `I2RT_INSTALL_COMMAND`, the single source of truth for the git-only driver remedy. Also `close_robot_safely`, which joins i2rt's discarded control thread before the CAN socket closes (#28 — i2rt's own `close()` races the two, crashing every teardown), and `start_arms_concurrently`, which brings up both arms concurrently and tears down the survivor on failure (#83). |
| `_capture_proc.py` | Spawn-only RealSense capture child and parent lifecycle. One daemon child owns all configured pipelines and publishes copied RGB8, z16, intrinsics, scale, generation, and monotonic timestamps through unlinked shared-memory seqlock slots. Top-level imports stay limited to the standard library and NumPy. |
| `policy.py` | Generic `ActServerPolicy` `/act` client (with the `MolmoAct2Policy` alias) and `gr00t_policy` factory. `act()` packs cameras+instruction+state, POSTs via the injectable `post_fn`, and returns an `ActionChunk`. Real transport is the pragma'd `_default_post`. |
| `kinematics.py` | Always-importable `_ArmKinematics` wrapper. Owns model/config range intersections, gripper-joint pinning, relative yaw, warm starts, resync, rate clamp, and oscillation holds behind a raw NumPy protocol. |
| `collision.py` | Optional MuJoCo collision guardrail (plans 0011/0017): `CollisionChecker` composes the bimanual scene from `assets/yam_collision.xml` at configured base offsets, `CollisionApprover` sweeps commanded targets and holds at the last safe pose (or `SafetyAbort` in strict mode), and shared assembly maps `YamConfig.collision_*` geometry into both contributed and programmatic guardrails. Lazily imports mujoco; absolute `joint_pos` mode only. |
| `embodiment.py` | `YAMEmbodiment`: i2rt driver with joint and EEF control. Clamp backstop, optional delta→abs, lazy kinematics, default-on collision-guardrail contribution with explicit skip warnings, gripper de-norm, `SELF_PACED` pacing, operator-keypress episode end (`operator_end`), and joint-space homing/parking. Its `defer_operator_end()` hook yields stdin and trial termination to the framework console for the run. Hardware seams are injected/pragma'd. |
| `preflight.py` | `build` / `run_preflight` + the `inspect-robots-yam-preflight` CLI: run the compat check, print, exit non-zero on errors. |
| `health.py` | One-shot camera freshness and packed motor-position health gate, with JSON/human reports, montage output, and injected hardware seams. |
| `watch.py` | Cameras-only live browser view with per-camera recovery supervision, MJPEG streams, and injected HTTP, clock, and OpenCV seams. |
| `__init__.py` | Public API fenced by `__all__` (guarded by `tests/test_api_snapshot.py`). |

## Key invariants

- **Contract symmetry:** policy and embodiment build their `action_space` /
  `observation_space` from the *same* `config.py` helpers. If you change the dim,
  semantics, camera names, or state key, change them there once — not in two
  places — or compat breaks.
- **Construction is inert:** `__init__` touches no hardware/network/stdin (only
  `.info`). The driver connects lazily on the first `reset()`. This is what lets
  the registry (`factories[name]()`) and preflight construct components freely.
- **Coverage discipline:** the only uncoverable code is hardware/TTY I/O, isolated
  in `# pragma: no cover` seams (`_default_post`, `_default_driver_factory`,
  `default_poll_end`, `_import_cv2`, `_capture_proc._import_rs`, the
  `_require_driver` pre-reset guard, `__main__`). Keep new hardware access inside
  such seams so the 100% gate stays meaningful. `_OpenCVCameraReader` is the
  worked example: the cv2 module, `sleep_fn`, and `clock` are constructor
  arguments, so queue configuration, draining, conversion, and teardown are all
  tested against fakes and only the `import cv2` itself is pragma'd.
- **RealSense capture stays isolated:** `realsense_capture="process"` lazily
  starts one daemon spawn child for all serial-configured cameras. The child
  unregisters parent-created shared-memory names from its resource tracker; the
  parent unlinks them immediately after the ready handshake, while both
  mappings remain usable. Pipe EOF and the stop event end the drain loop.
  Parent close escalates from join to terminate to kill. Shared-memory NumPy
  views are per-read copies and must be discarded before close. Keep
  `realsense_capture="inline"` behaviorally equivalent as the debugging escape
  hatch.
- **Cameras must be released:** `_OpenCVCameraReader` runs a daemon drain thread
  per camera (#63), while the default RealSense reader owns a daemon capture
  child (#95). Both keep devices open until released. `YAMEmbodiment.close()`
  calls `close()` on any reader that has one, after park + driver teardown
  (deliberately — arms reach a safe state before any camera work) and guarded
  so a camera error cannot strand the arms.
- **Safety lives in `step()`**, not in an optional Approver — see the root
  `CLAUDE.md`.
