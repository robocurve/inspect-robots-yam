# 0001 — `robolens-yam`: YAM arms + MolmoAct2 adapters

> **Superseded naming.** This document predates the framework rename and is kept
> verbatim as the approved design record. Read the old names as follows:
> package `robolens-yam` → **`inspect-robots-yam`** (module `inspect_robots_yam`);
> framework `robolens`/`RoboInspect`/`roboinspect` → **Inspect Robots**
> (`inspect-robots`, module `inspect_robots`); entry-point groups
> `roboinspect.*` → **`inspect_robots.*`**; CLI `roboinspect run` →
> **`inspect-robots run`**; `robolens-yam-preflight` →
> **`inspect-robots-yam-preflight`**. The design itself is unchanged and current.

**Status:** design + implementation plan (rev 2 — addresses critique round 1)
**Goal:** make RoboInspect + KitchenBench runnable on **I2RT YAM bimanual arms** driven by
**MolmoAct2**, with `(policy, embodiment)` compatibility guaranteed — and motion made
*safe* — *before any arm moves*.

The deliverable is a standalone plugin package `robolens-yam` →
`github.com/robocurve/robolens-yam`, registered via entry points so:

```bash
roboinspect run --task kitchenbench/pour_pasta --policy molmoact2 --embodiment yam_arms
```

resolves the two new components and runs the closed-loop rollout, scored by
KitchenBench's existing `task_success` scorer.

---

## 1. Grounded contract (researched, not guessed)

Sources: `allenai/MolmoAct2-BimanualYAM` (HF + `examples/yam/host_server_yam.py`),
`i2rt-robotics/i2rt` driver docs.

MolmoAct2 ships a **first-party bimanual-YAM integration**: the model runs as a
FastAPI server exposing `/act` over a `json_numpy`-encoded wire protocol.

**`/act` request:** `top_cam`,`left_cam`,`right_cam` `ndarray(H,W,3)` uint8 RGB
(**order matters**); `instruction` str; `state` `ndarray(14,)` float32 (per-arm 7-D ×2);
`num_steps` int (default **10**); `timestamp` float (optional).
**`/act` response:** `actions` `ndarray(num_steps, 14)` float32; `dt_ms` float.

**YAM arm:** 6 revolute joints + 1 linear gripper per arm → 7-D/arm, **14-D bimanual**.
Commanded via i2rt (`get_yam_robot(channel, gripper_type)`,
`robot.command_joint_pos(target)`, `robot.get_joint_pos()`) → **joint-position** control.

**Resulting RoboInspect spaces (identical on both sides → zero remap, zero compat errors):**

| field | value |
|---|---|
| action `Box.shape` | `(14,)` with `low`/`high` joint limits set (see §3) |
| `ActionSemantics.control_mode` | `"joint_pos"` (a legal `ControlMode` literal) |
| `ActionSemantics.rotation_repr` | `"none"` |
| `ActionSemantics.gripper` | `"continuous"` |
| `ActionSemantics.frame` | `"base"` (joint space; frame is informational) |
| cameras | `top_cam`, `left_cam`, `right_cam` (`CameraSpec`, resolution configurable) |
| state key | `"joint_pos"` (the packed 14-D vector; richer `StateSpec` attached, see §2.1) |
| policy `action_horizon` | `num_steps` (default 10) |

`compat._check_action_spaces` errors on **dim / control_mode / rotation_repr**
mismatch and warns on gripper/frame; `_resolve_keys` errors on missing
**camera/state** keys; a `control_rate` **warning** fires only if
`policy.info.control_hz` is non-`None` and differs from the embodiment's.
Declaring both components against this single contract yields **zero errors and
zero warnings** (see §3 for the load-bearing `control_hz=None` detail).

---

## 2. Architecture

Two components; **neither imports torch** (the model lives in MolmoAct2's own
server process). Heavy/hardware deps (`requests`, `json-numpy`, `i2rt`) are
**lazily imported inside methods** so the package imports — and is fully testable —
on a laptop with no hardware and no server.

```
robolens-yam/
  pyproject.toml          # name=robolens-yam; roboinspect+kitchenbench via tool.uv.sources tags
  README.md               # install, launch the MolmoAct2 server, run, preflight, safety, risks
  CLAUDE.md  +  src/robolens_yam/CLAUDE.md
  LICENSE                 # MIT  (done)
  .gitignore              # (done)
  src/robolens_yam/
    __init__.py           # public API + __all__
    packing.py            # PURE: 14-D <-> (left 7-D, right 7-D); validation. No deps.
    config.py             # YamConfig, MolmoActConfig (frozen dataclasses + defaults)
    operator.py           # operator confirmation (injectable I/O for tests)
    policy.py             # MolmoAct2Policy — /act HTTP client -> ActionChunk
    embodiment.py         # YAMEmbodiment — i2rt driver; clamps; operator success; SELF_PACED
    preflight.py          # build both, run check_compatibility, print report; CLI
    py.typed
  tests/                  # i2rt + /act server + stdin + clock all mocked; 100% branch cov
  .github/workflows/ci.yml
  .pre-commit-config.yaml
```

### 2.1 `packing.py` (pure, no deps)
- Constants `ARM_DOF=6`, `GRIPPER_DOF=1`, `ARM_WIDTH=7`, `TOTAL_DIM=14`,
  `LEFT=slice(0,7)`, `RIGHT=slice(7,14)`.
- `pack(left, right) -> ndarray(14)`; `split(vec) -> (left7, right7)`;
  `validate_dim(vec, n=14) -> ndarray` (raises `ValueError` with a clear message).
- One documented convention: each arm 7-D = `[j0..j5, gripper]`, gripper last; left then right.
- Also exports a `STATE_SPEC: StateSpec` documenting mixed units (j0..j5 `rad`,
  gripper `normalized`) so the embodiment can attach it (addresses the units smell;
  compat only set-compares keys so this is documentation, not enforcement).

### 2.2 `config.py`
- `YamConfig` (frozen): `left_channel="can0"`, `right_channel="can1"`,
  `gripper_type="LINEAR_4310"`, `control_hz=10.0`,
  `cam_height=224`, `cam_width=224`,
  `joint_low: tuple[float,...]` / `joint_high: tuple[float,...]` — **14-D action
  limits** (conservative YAM defaults: joints ±π, gripper 0..1) used for the action
  `Box` and the `step()` clamp backstop,
  `home_pose: tuple[float,...] | None`,
  `gripper_open=0.0`, `gripper_closed=1.0` (calibration; superseded by plan
  0005 — wire 1 now maps to open, defaults `gripper_open=1.0`, `gripper_closed=0.0`),
  `joints_are_delta=False` (**see §4 Risk #1 — does NOT change declared semantics**).
- `MolmoActConfig` (frozen): `server_url="http://127.0.0.1:8202"`, `endpoint="/act"`,
  `num_steps=10`, `timeout_s=30.0`,
  `camera_order=("top_cam","left_cam","right_cam")`, `state_key="joint_pos"`,
  `cam_height=224`, `cam_width=224`.
- Both have classmethod `from_kwargs(**flat)` so the constructors accept **flat
  scalar kwargs** (`server_url=`, `num_steps=`, `left_channel=`, `control_hz=`, …)
  — required so `roboinspect run -P server_url=... -E left_channel=...` works (the CLI
  only passes scalars; see §6).

### 2.3 `operator.py`
- `OperatorIO(input_fn=input, output_fn=print)` — all stdin/stdout injected.
- `wait_ready(prompt) -> None` (blocks for Enter; pre-episode).
- `confirm_success(prompt) -> bool` (y/n loop; episode-end verdict).
- `default_poll_end() -> bool` — the real non-blocking "operator pressed end" check;
  TTY/platform-specific, marked **`# pragma: no cover`** (can't run without a TTY).
- Tests inject scripted `input_fn` and a scripted `poll_end`, so every *coverable*
  branch is hit with no real stdin.

### 2.4 `policy.py` — `MolmoAct2Policy`
- `__init__(self, config: MolmoActConfig | None = None, *, post_fn=None, **flat)`:
  build `config = config or MolmoActConfig.from_kwargs(**flat)`; `post_fn` is the
  injectable HTTP transport (default `_default_post` — lazily imports
  `requests`+`json_numpy`, **`# pragma: no cover`**). **No network at construction.**
- `info = PolicyInfo(name="molmoact2", action_space=<14-D joint_pos/continuous Box
  with limits>, observation_space=ObservationSpace(cameras=3×CameraSpec,
  state_keys={"joint_pos"}))`. **`control_hz` intentionally left `None`** (comment
  in code) — this is what keeps compat warning-free; the trained rate rides on the
  *chunk* instead.
- `config_rl = PolicyConfig(action_horizon=num_steps)` (attr name `config` per
  protocol).
- `reset(scene)`: stash `scene.instruction`; reset counters.
- `act(obs) -> ActionChunk`:
  1. build request: cameras in `camera_order` (uint8, dim-agnostic), `instruction`,
     `state = validate_dim(obs.state[state_key])` (float32), `num_steps`;
  2. `resp = post_fn(url, payload)` (default transport encodes/decodes `json_numpy`);
     wrap transport/parse failures in a clear message (rollout maps to `PolicyError`);
  3. parse `actions (N,14)`, `dt_ms`;
  4. return `ActionChunk([Action(row.copy()) for row in actions],
     control_hz=(1000.0/dt_ms if dt_ms else None),
     inference_latency_s=measured)`. **Both `dt_ms` truthy and falsy/None branches
     tested.** `ActionChunk` requires ≥1 action → guard empty `actions` with a
     clear error.

### 2.5 `embodiment.py` — `YAMEmbodiment`
- `__init__(self, config: YamConfig | None = None, *, driver_factory=None,
  camera_reader=None, operator=None, poll_end=None, sleep_fn=None, **flat)`:
  `config = config or YamConfig.from_kwargs(**flat)`; seams default to **real,
  pragma-excluded** module helpers:
  - `_default_driver_factory(cfg)` — lazily imports i2rt, opens both CAN channels
    (`# pragma: no cover`);
  - `_default_camera_reader(cfg)` — reads the 3 cameras (`# pragma: no cover`);
  - `operator = operator or OperatorIO()`, `poll_end = poll_end or default_poll_end`
    (pragma'd), `sleep_fn = sleep_fn or time.sleep`.
  **No hardware/network/stdin touched at construction** — only `.info` is built, so
  registry/compat/preflight construct it freely. Driver connects lazily on first
  `reset()`.
- `info = EmbodimentInfo(name="yam_arms", action_space=<same 14-D joint_pos Box with
  limits>, observation_space=ObservationSpace(cameras=3×CameraSpec,
  state_keys={"joint_pos"}, state=STATE_SPEC), control_hz=cfg.control_hz,
  is_simulated=False, capabilities={SELF_PACED})`. `supported_target_kinds` left
  **empty** → all 10 KitchenBench target kinds realizable.
- `reset(scene, seed=None) -> Observation`: connect driver if needed; drive to
  `home_pose` (clamped); `operator.wait_ready(...)`; return first `Observation`
  (`images=camera_reader()`, `state={"joint_pos": get_joint_pos_14()}`,
  `instruction=scene.instruction`).
- `step(action) -> StepResult`:
  1. `cmd = validate_dim(action.data)`; if `joints_are_delta`, `cmd = current + cmd`
     (delta→abs **inside** the embodiment, so the *declared* `joint_pos` stays
     honest);
  2. **clamp** `cmd` to `[joint_low, joint_high]` — a hard safety backstop
     independent of any Approver (addresses Blocker #4);
  3. gripper de-normalization via calibration;
  4. `driver.command_joint_pos(left/right)`;
  5. **pace**: `sleep_fn(max(0, 1/control_hz - elapsed))` (SELF_PACED is advisory —
     the framework does *not* sleep for us; we pace ourselves);
  6. build next `Observation`;
  7. **episode end**: if `poll_end()` returns `True`, ask `operator.confirm_success`
     → `StepResult(terminated=True, termination_reason="success" if ok else
     "failure", observation=obs)`; else `StepResult(terminated=False, ...)`.
     Success reaches the scorer **exclusively** via `termination_reason="success"`
     (the only wired path — `operator_judgement` is never set by stock `rollout`).
     If the operator never ends, the task's `max_steps` truncates →
     `"max_steps"` → scored failure (so unattended/CI runs score failure — by design).
- `close()`: release driver handles (no-op if never connected).

### 2.6 `preflight.py`
- `build(yam_cfg=None, molmo_cfg=None) -> (MolmoAct2Policy, YAMEmbodiment)` — pure
  `.info` construction, no connect.
- `run_preflight(task_name=None, *, policy=None, embodiment=None,
  check=check_compatibility) -> CompatibilityReport` — uses injected `policy`/
  `embodiment` when given (lets a test pass a deliberately **incompatible** pair to
  exercise the error path), else `build()`; resolves the KitchenBench task **inside
  the function** (never import kitchenbench at module top).
- `main(argv=None, *, run=run_preflight) -> int` — CLI
  `robolens-yam-preflight [--task ...] [--json] [--dry-run]`: prints errors/warnings
  (both `--json` and human branches tested), returns non-zero on any error
  (tested via injected incompatible pair). `--dry-run` is documented as the
  pre-motion gate.

---

## 3. Compatibility verification (the heart of the goal)

`tests/test_compat.py`:
- `MolmoAct2Policy()` + `YAMEmbodiment()` → `check_compatibility(...).ok is True`,
  `errors == []` **and** `warnings == []` (dim=14, control_mode=joint_pos,
  rotation_repr=none, gripper=continuous, frame match, 3 cameras + `joint_pos`
  provided, **policy `control_hz` is None** so no `control_rate` warning).
- For **each** of the 10 KitchenBench tasks (resolved via entry points), still `ok`
  with no `scene_target`/`scene_setup` errors.
- **Negative controls** (prove the check bites): an 8-D policy → `action_dim` error;
  a policy advertising `control_hz=100` → `control_rate` **warning** appears (locks
  finding #1 so a future regression is caught).

`tests/test_eval_end_to_end.py` (addresses #14): run the real `eval()`/`rollout()`
with a mocked `post_fn` (canned `actions (N,14)`, `dt_ms`), mocked driver+cameras,
`sleep_fn=lambda *_: None`, and a scripted `poll_end`/`operator` that ends in
success → assert `task_success == True` and an `EvalLog` with `status="success"`.
This proves the `termination_reason="success"` → scorer wiring and chunk replay
actually compose (the static compat test cannot).

This pair of tests is what *closes the goal*: roboinspect + kitchenbench are provably
compatible with — and actually score a rollout on — YAM + MolmoAct2.

---

## 4. Risks RoboInspect cannot catch (correctness/safety, not shape) — surfaced

1. **Absolute vs. delta joints.** `ControlMode` has **no `joint_delta` literal**, so
   compat *cannot* represent the delta case — green preflight does **not** prove
   abs/delta correctness. We keep the embodiment's command interface absolute
   (delta→abs done inside `step()`), so the declared `joint_pos` stays honest.
   README flags this as the **#1 thing to verify on hardware**, gated behind
   `preflight --dry-run` + a single-joint jog before any task run.
2. **Safety / clamping (was Blocker #4).** `eval()` defaults to `AutoApprover`
   (no clamp). We mitigate three ways: (a) the action `Box` carries `low`/`high`
   joint limits; (b) `YAMEmbodiment.step()` **always clamps** to those limits as a
   hard backstop regardless of Approver; (c) README documents driving hardware via
   the Python API with `approver=ClampApprover(...)` for defense in depth. "Safe to
   move" = preflight green **and** clamp backstop active.
3. **Gripper calibration.** `gripper_open`/`gripper_closed` map MolmoAct2's gripper
   value → YAM range; documented; identity default + README warning.
   Superseded by plan 0005, which defines wire 1 as open and wire 0 as closed.
4. **State/action packing order.** Centralized in `packing.py`, one convention
   (`[j0..j5, gripper]` per arm, left then right) matching the YAM server's 14-D.
5. **Camera resolution/order.** Compat checks camera *names* only, **not**
   resolution. `camera_order` + shared `cam_height`/`cam_width` in both configs keep
   them in sync by construction; README notes resolution is a config invariant, not
   a checked one.
6. **`dt_ms` vs `control_hz`.** Chunk carries `control_hz=1000/dt_ms`; the embodiment
   paces to its own `control_hz`. `SELF_PACED` is advisory (framework doesn't sleep).

---

## 5. Quality gates (mirror roboinspect/kitchenbench)

- `ruff check .`, `ruff format --check .`, `mypy --strict` on `src/robolens_yam`.
- `pytest --cov` at **100% branch** (`fail_under=100`). Uncoverable hardware/TTY
  seams isolated in tiny `# pragma: no cover` module helpers (the repo's
  `exclude_also` honors `pragma: no cover`): `_default_post`, `_default_driver_factory`,
  `_default_camera_reader`, `default_poll_end`, `if __name__=="__main__"`.
- Pre-commit (ruff+mypy on commit; coverage on push) via `uv run`.
- CI: Linux+macOS / py3.11–3.12, all gates blocking.
- Static version `0.1.0`. Public API fenced by `robolens_yam.__all__`.
- **Dependencies (addresses Blocker #9):**
  ```toml
  dependencies = ["roboinspect>=0.1", "numpy>=1.24"]
  [project.optional-dependencies]
  client = ["requests>=2.31", "json-numpy>=2.1"]   # real /act transport
  yam    = ["i2rt"]                                 # real arm driver
  dev    = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6", "mypy>=1.11",
            "pre-commit>=3.5", "kitchenbench"]      # kitchenbench needed by compat test
  [tool.uv.sources]
  roboinspect     = { git = "https://github.com/robocurve/roboinspect",     tag = "v0.1.0" }
  kitchenbench = { git = "https://github.com/robocurve/kitchenbench", tag = "v0.1.0" }
  ```
  (Tag `kitchenbench` v0.1.0 first if it isn't tagged; else use `branch="main"`.)
  `requests`/`json-numpy`/`i2rt` are **not** test deps — those paths are mocked.

## 6. Entry points & CLI knobs

```toml
[project.entry-points."roboinspect.embodiments"]
yam_arms = "robolens_yam.embodiment:YAMEmbodiment"
[project.entry-points."roboinspect.policies"]
molmoact2 = "robolens_yam.policy:MolmoAct2Policy"
[project.scripts]
robolens-yam-preflight = "robolens_yam.preflight:main"
```
Registry resolves with `factories[name](**kwargs)` and `roboinspect run` (no `-E/-P`)
passes `kwargs={}` → **both classes must be zero-arg constructible** (defaults
build configs; nothing connects). `**flat` scalar kwargs let
`roboinspect run -P server_url=http://host:8202 -E left_channel=can0` work despite the
CLI only producing scalars (addresses #13). A test constructs `YAMEmbodiment()` and
`MolmoAct2Policy()` with nothing mocked and asserts `.info`/`.config`.

## 7. Out of scope
- Hosting/launching the MolmoAct2 server (lives in `allenai/molmoact2`; README links it
  + the `huggingface-cli download allenai/MolmoAct2-BimanualYAM` step that needs the HF token on the 5090 box).
- Single-arm / non-YAM I2RT robots; fine-tuning MolmoAct2 (we consume the published checkpoint).

## 8. Build order (TDD, commit per step)
1. scaffolding: pyproject, ruff/mypy/pytest cfg, pre-commit, CI, `__init__`. (LICENSE/.gitignore done.)
2. `packing.py` (+ `STATE_SPEC`) + tests.
3. `config.py` (+ `from_kwargs`) + tests.
4. `operator.py` + tests.
5. `policy.py` + tests (mock `post_fn`; cover `dt_ms` both branches, empty-actions guard, lazy-import pragma).
6. `embodiment.py` + tests (mock driver/cameras/clock/operator/poll_end; cover clamp, delta→abs, success/failure/continue, close-before-connect).
7. `preflight.py` + CLI + tests (injected incompatible pair → error path + non-zero exit; `--json` + human branches).
8. `test_compat.py` + `test_eval_end_to_end.py` (goal-closing).
9. `test_api_snapshot.py` (fence `__all__`); zero-arg construction test.
10. README + CLAUDE.md ×2. Verify 100% cov + all gates green. Push; create
    `robocurve/robolens-yam`; add a one-line "adapters" mention to WorldEvals
    README (it's an adapter package, not a benchmark — no catalog entry).
