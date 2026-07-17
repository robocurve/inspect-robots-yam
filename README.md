<div align="center">

# inspect-robots-yam

Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on real
[I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) bimanual arms driven by
[MolmoAct2](https://github.com/allenai/molmoact2).

![Status: alpha](https://img.shields.io/badge/status-alpha-blue)
[![CI](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inspect-robots-yam)](https://pypi.org/project/inspect-robots-yam/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![Docs coverage](https://img.shields.io/badge/public%20docstrings-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

> [!NOTE]
> This project is in early development. The API may change between releases, so pin a version before depending on it.

Inspect Robots has two swappable inputs: a `Policy` (the VLA brain) and an
`Embodiment` (the robot body + world). This package provides both for the
YAM + MolmoAct2 stack, so any embodiment-agnostic Inspect Robots task (e.g. all of
[KitchenBench](https://github.com/robocurve/kitchenbench)) runs on real arms:

- **`molmoact2` policy**: a thin client for MolmoAct2's first-party bimanual-YAM
  `/act` server (the model owns the GPU + weights in its own process).
- **`yam_arms` embodiment**: the I2RT driver with joint-position control by
  default and an opt-in Cartesian end-effector interface, plus a hard safety
  clamp, operator-in-the-loop success, and self-paced control.

Both declare the same 14-D joint-position contract (2 arms × [6 joints +
gripper], cameras `top/left/right`, packed `joint_pos` state), so Inspect Robots's
compatibility check passes with zero errors and zero warnings. This is
verifiable before any motion.

```bash
inspect-robots run --task kitchenbench/pour_pasta --policy molmoact2 --embodiment yam_arms
```

> **Note:** cameras are configured with three plain device paths
> (`top/left/right_cam_device`), so the whole rig is drivable from config.ini
> or `-E key=value` flags with no custom code. A Python `camera_reader` remains
> available for exotic camera stacks. With neither configured, `yam_arms` fails
> fast with a `ConfigError` at `reset()`, before any driver connect or motion.

## Install (on the robot/GPU machine)

```bash
uv venv && source .venv/bin/activate
uv pip install inspect-robots-yam
# The i2rt driver is git-only and not on PyPI. Install it directly.
# The build-constraints file works around a build failure in i2rt's ruckig
# dependency (source-only releases that no longer build under scikit-build-core
# 1.0; the pin below 0.10 matches i2rt's own in-repo workaround):
echo 'scikit-build-core<0.10' > build-constraints.txt
uv pip install --build-constraints build-constraints.txt "i2rt @ git+https://github.com/i2rt-robotics/i2rt@db582eaa70b6a057a1e2981da6219dfa6c29422a"
```

The base package includes the `/act` transport and builtin OpenCV camera reader.
Only `i2rt`, the I2RT YAM arm driver required for real hardware, needs the
separate git install. The `scikit-build-core` build constraint can be dropped
once ruckig ships a release with the fix from
[pantor/ruckig#261](https://github.com/pantor/ruckig/issues/261) and i2rt
moves off `ruckig==0.15.3`. The camera reader depends on
`opencv-python-headless`; if your environment also carries `opencv-python`,
the two share the `cv2` module and the last one installed wins.

Then download the model weights (needs a Hugging Face token) and start the server,
from the [MolmoAct2 repo](https://github.com/allenai/molmoact2):

```bash
huggingface-cli download allenai/MolmoAct2-BimanualYAM
python examples/yam/host_server_yam.py          # serves /act on :8202
```

### Serving a GR00T fine-tune

Run the shim from an [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
environment with a CUDA, PyTorch, and flash-attn stack that supports the GPU.
Blackwell GPUs (`sm_120`) require a matching PyTorch build. Download the YAM
fine-tune and start its `/act` server on the default port 8203:

```bash
hf download robocurve/gr00t-n1.7-yam-molmoact2
python scripts/serve_gr00t_act.py \
    --model robocurve/gr00t-n1.7-yam-molmoact2
```

Then run it through the distinct `gr00t` policy entry point so eval logs carry
the correct model family:

```bash
inspect-robots "stack the red block on the blue block" \
    --policy gr00t --embodiment yam_arms
```

The client defaults to `http://127.0.0.1:8203`. Override a remote or alternate
server with `-P server_url=http://gpu:8203`. The config key is `server_url`;
`url` is a read-only property, and `ActServerConfig.from_kwargs` rejects it.
For another GR00T fine-tune, pass `-P action_horizon=<its chunk length>` so the
recorded policy metadata matches that checkpoint.

> [!WARNING]
> The shim's startup checks validate the packed layout and units ranges, but
> joint polarity and absolute-vs-delta semantics cannot be detected from
> dataset statistics. For the first runs with a new checkpoint family, run
> `inspect-robots-yam-preflight`, leave guardrails on, and keep an operator at
> the e-stop.

## Preflight: prove compatibility before any motion

Check dims, semantics, cameras, and state keys:

```bash
inspect-robots-yam-preflight
```

Also check a specific task's scenes are realizable:

```bash
inspect-robots-yam-preflight --task kitchenbench/pour_pasta
```

Affirm that no motion will occur:

```bash
inspect-robots-yam-preflight --dry-run
```

A green preflight means action dim (14), control mode (`joint_pos`), cameras, and
state keys all line up. It does not prove the joint values are interpreted the
same way. See *Safety* below.

## Run on hardware

Write your defaults once. The interactive wizard interviews this plugin's
declared devices (three cameras and both arms' CAN channels) with live
probes, including unplug-to-identify:

```bash
inspect-robots setup
```

Or write the file yourself, replacing the three camera paths with your rig's
V4L2 color nodes (use stable `/dev/v4l/by-id/...` or udev-symlink paths;
bare `/dev/videoN` numbers reshuffle on every replug):

```bash
mkdir -p ~/.config/inspect-robots && cat > ~/.config/inspect-robots/config.ini <<'EOF'
[defaults]
policy = molmoact2
embodiment = yam_arms
scorer = success_at_end    # scores the operator's y/N answer at episode end
max_steps = 1200           # 120 s at 10 Hz
rerun = true               # live viewer of cams/state/actions (inspect-robots[rerun])
store_frames = true        # keep the policy's camera frames per run

[embodiment.args]
top_cam_device = /dev/v4l/by-id/YOUR-TOP-CAM
left_cam_device = /dev/v4l/by-id/YOUR-LEFT-CAM
right_cam_device = /dev/v4l/by-id/YOUR-RIGHT-CAM
EOF
```

Then tell the robot what to do:

```bash
inspect-robots "place the fork on the plate"
```

The attended flow: position the scene, press Enter to start, press any key to
end the episode, answer y/N to score. The status line counts up against the
run's real step limit (`t = 42s / 120s`) with no configuration needed
(requires inspect-robots newer than 0.8.1; on older cores set
`max_steps_hint`).

For exotic camera stacks (or full programmatic control), the Python API takes
a custom `camera_reader` returning
`{"top_cam", "left_cam", "right_cam": HxWx3 uint8}`:

```python
from inspect_robots import eval
from inspect_robots.approver import ClampApprover
from inspect_robots_yam import MolmoAct2Policy, YAMEmbodiment, YamConfig

emb = YAMEmbodiment(YamConfig(left_channel="can0", right_channel="can1"),
                    camera_reader=my_camera_reader)
pol = MolmoAct2Policy(server_url="http://127.0.0.1:8202")

(log,) = eval("kitchenbench/pour_pasta", pol, emb,
              approver=ClampApprover(emb.info.action_space))  # defense in depth
print(log.status, log.results.metrics)
```

At each episode end the embodiment asks the operator (y/N); a `yes` records
`termination_reason="success"`, which KitchenBench's `task_success` scorer reads.
The operator prompts need an interactive terminal: a dead stdin raises
`EmbodimentFault` (the framework's always-halt path). For runs with no operator,
set `YamConfig(unattended=True)` (CLI: `-E unattended=true`): all operator
prompts are skipped and every episode runs to `max_steps`, scoring as a failure.

## Drive the arms with an LLM (agent mode)

With the [inspect-robots-agent](https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-agent)
plugin installed, a frontier LLM can drive the arms directly: it sees the
cameras and the labeled 14-D state, and moves joints by name
(`left_j0`..`left_gripper`, `right_j0`..`right_gripper`) through smooth,
approver-checked motions.

Put a `.env` with your API key in the working directory, reusing one you already have or copying the [.env.example](.env.example) template (the CLI loads it automatically; real environment variables take precedence over its values):

```ini
ANTHROPIC_API_KEY=sk-ant-...
```

Install the add-on:

```bash
uv pip install inspect-robots-agent inspect-robots-yam
inspect-robots config set embodiment yam_arms     # once, per machine
```

Cameras come from the builtin reader: set the three `*_cam_device` paths in
`~/.config/inspect-robots/config.ini` (see Run on hardware above) or pass them as
`-E` flags per run. Then run the LLM on the robot:

```bash
inspect-robots "place the fork on the plate" --policy agent \
    -P model=anthropic/claude-fable-5
```

> [!NOTE]
> Invoke the CLI as plain `inspect-robots`, not `uv run inspect-robots`.
> Inside a uv project, `uv run` first re-syncs the environment to the
> project's lockfile, downgrading whatever the `uv pip install` commands
> above just added back to the locked versions; the only trace is an
> easy-to-miss "Uninstalled N / Installed N packages" line. To use
> `uv run` anyway, pass `--no-sync`, or declare everything as real
> dependencies with `uv add inspect-robots-yam` plus your plugins.

Safety guardrails (a bounds clamp plus a per-step delta limit derived from the
declared action space) are wired in by default for every CLI run; turning them
off requires an explicit `--disable-guardrails`.

### Cartesian EEF mode

For LLM-agent runs, opt into the 10-D absolute Cartesian interface:

```ini
[embodiment.args]
control_interface = eef_pos
```

Each arm is controlled as `x, y, z, yaw, gripper`. Positions are metres in
that arm's own base frame, with +x forward from the base and +z up. The two
base frames are independent. On common mirrored bimanual mounts, the arms'
+y axes point in opposite world directions, so equal signed y targets do not
mean equal world directions.

Yaw is an absolute target relative to the orientation captured at reset:
`0` means the reset orientation. It rotates about base +z while preserving the
captured roll and pitch. Yaw interpolation does not wrap. A move from `3.1` to
`-3.1` sweeps through zero instead of taking the short path, so use
intermediate yaw targets for near-±π regrasps.

The default workspace per arm is x `[0.15, 0.48]`, y `[-0.25, 0.25]`, and z
`[0.03, 0.40]`, with yaw `[-π, π]` and gripper `[0, 1]`. These bounds were
validated against the bundled YAM + LINEAR_4310 model at the default working
orientation, but they are a conservative box rather than an exact reachable
set. `eef_low` and `eef_high` override all ten bounds. The observation keeps
the 14-D `joint_pos` field for logging and adds the command-aligned 10-D
`eef_state` field.

In both control interfaces, `home_pose=None` selects a mandatory per-mode
factory default instead of skipping homing. Joint mode uses the
dataset-verified `DEFAULT_JOINT_HOME_POSE`, with every joint at encoder zero
and both grippers open. EEF mode uses `DEFAULT_EEF_HOME_POSE`; its provisional
per-arm joints are `[-0.024, 0.794, 0.645, -0.375, -0.021, -0.012]`, with both
grippers open. The first EEF reset validates that the configured home FK lies
in the workspace box before moving, then captures each arm's yaw reference
after homing.

> [!WARNING]
> EEF mode has no arm-table or arm-arm collision checking. The workspace box,
> Cartesian guardrails, joint-space IK rate limit, oscillation hold, and joint
> limits are the only geometric protections. The two default y ranges overlap.
> Keep an operator at the e-stop; using EEF mode unattended is operator
> discretion and requires rig-specific validation.

> [!WARNING]
> Before any unattended agent run, verify on your rig that the arms hold
> position while the LLM thinks (seconds between action chunks). Run the
> bundled check per arm and per mode, arms mid-workspace, e-stop in hand:
>
> ```bash
> inspect-robots-yam-holdcheck can_left --zero-gravity true
> inspect-robots-yam-holdcheck can_right --zero-gravity true
> ```
>
> (Channel names match your rig's CAN interfaces; `can0`/`can1` on default
> setups.) PASS in the mode you run agents in closes the verification. The
> default `zero_gravity_mode=true` puts the i2rt driver in a
> gravity-compensated, compliant mode; if it drifts but `--zero-gravity
> false` holds, run agents with `-E zero_gravity_mode=false`. If both drift,
> file an issue with the numbers. Keep a hand on the e-stop for the first
> runs.

YAM ships with a factory resting pose at encoder zero for every joint and 1.0
(open) for both grippers. It equals the joint-mode factory home, so standard
upright rigs end with a gentle 3-second park and the next episode begins with
open grippers. Override it per rig when needed. Pose fields accept
comma-separated values from the CLI and config.ini. For example, a per-rig
rest target can retain measured joint offsets while parking open:

```ini
[embodiment.args]
rest_pose = -0.002,0.002,0.002,-0.089,0.007,-0.026,1.0,-0.006,0.002,0.001,-0.087,-0.007,-0.019,1.0
```

Set `rest_pose = none` to opt out of the factory target and park at the pose
captured before the first commanded motion instead.

In delta mode (`-E joints_are_delta=true`) the declared action space is the
per-step displacement box (`YamConfig.step_limits`, default 0.2 rad per joint
and a full gripper stroke per step); the absolute joint limits still clamp the
summed command inside the embodiment as a backstop. A delta-configured rig
must be paired with a delta-declaring policy (`-P joints_are_delta=true` for
`molmoact2`); a mismatch fails the compatibility check before any motion.

## Safety

- **Hard clamp backstop.** Every command is clipped to `YamConfig.joint_low/high`
  *inside* `step()`, independent of any Inspect Robots `Approver`: unclamped model
  outputs can never reach the motors. **Set the arm slots to your real YAM joint
  limits** (the defaults are conservative placeholders: joints ±π, gripper 0–1).
  But note the limits are in *policy units* per the table below: gripper slots 6
  and 13 stay normalized 0–1, only slots 0–5 and 7–12 are radians.
- **Use `ClampApprover`** on hardware for a second layer.
- **Zero-gravity handoff jump.** The arms connect in zero-gravity mode by default
  (`YamConfig(zero_gravity_mode=True)`, passed through to the i2rt driver).
  Homing and rest-pose motions ramp at `control_hz`, but the first *policy*
  action in joint mode is still a stiff PD command that can jump from wherever
  the arm ended up. Nothing bounds the per-step joint delta in absolute joint
  mode yet (tracked as a known issue). EEF mode applies a 0.2-rad-per-joint
  per-step IK backstop, but a six-joint branch transit can still move the EEF
  tens of centimetres because rate-clamped intermediate configurations are not
  IK solutions. Reset always moves the arms through the full homing ramp, and
  every mode has a factory home. Attended runs issue a stand-clear prompt
  before the first homing ramp of each connection. Stand clear when the
  episode starts, and use `home_pose` as the per-rig override when the factory
  start is not validated for your setup.
- **EEF reachability and collision limits.** Iteration-cap non-convergence uses
  the solver's finite last iterate as best effort, and the next `eef_state`
  reports the true result. IK branch flips are joint-rate-clamped and repeated
  reversals hold the whole affected arm temporarily. These controls do not
  check collisions or guarantee a Cartesian path during a clamped branch
  transit. Raised work surfaces also need a raised EEF z minimum: the default
  `z_min=0.03` leaves only about 19 mm nominal fingertip clearance over a table
  at the arm-base plane, less up to 5 mm of IK error.
- **Park pose must rest under gravity.** On close, the arms ramp back to an
  explicit per-rig `rest_pose` or the factory zero-joint, open-gripper target,
  and torque is released once the ramp finishes. Set `rest_pose=none` to opt
  out and fall back to the pose captured at the first reset. Verify that the
  factory target is a supported resting pose on your rig, or start runs (or set
  `rest_pose`) with the arms in one, not held mid-air: whatever pose the park
  ends in is the pose the arms go limp from. The park path is not
  collision-checked, so keep the workspace clear at episode end. The default
  parks with both grippers open (wire 1), so parking releases anything still
  held during the ramp, wherever the arms happen to be. Rigs that must keep an
  object gripped at park should override `rest_pose` with gripper slots 0.0.
  Override both `home_pose` and `rest_pose` on rigs whose joint limits exclude
  zero, since both targets are clamped through the same per-joint box as every
  command.
- **Absolute vs. delta joints: verify first.** MolmoAct2's YAM `actions` are
  treated as *absolute* joint targets by default. If your checkpoint emits
  deltas, set `YamConfig(joints_are_delta=True)` (the embodiment converts to
  absolute internally so the declared `joint_pos` stays honest). Inspect Robots's
  compat check *cannot* tell these apart: confirm with `--dry-run` and a single
  slow jog before running a task.
- **Gripper polarity/trim.** The wire convention is normalized 0–1, with 1 open
  and 0 closed. The defaults (`gripper_open=1.0`, `gripper_closed=0.0`) preserve
  an identity map for the standard i2rt driver. These fields are the measured
  driver-native positions at the open and closed ends of the stroke. Configure
  an inverted or offset gripper with its actual endpoints, for example
  `gripper_open=0.72, gripper_closed=0.04`. Commands are de-normalized on the way
  out and observations are re-normalized on the way back, so the model always
  sees the wire convention. **Warning:** values outside [0, 1] are forwarded on
  a path i2rt does *not* clip. Avoid them unless you have verified your firmware's
  behavior.

  Compatibility (pre-1.0): earlier releases interpreted these fields with the
  opposite endpoint mapping. A config that explicitly copied the old defaults
  (`gripper_open=0.0`, `gripper_closed=1.0`) now inverts its gripper. A config
  that followed the old inversion recipe (`gripper_open=1.0`,
  `gripper_closed=0.0`) no longer inverts because those values are now the
  identity defaults. On identity-calibrated rigs, `home_pose`, `rest_pose`, and
  custom `joint_low`/`joint_high` retain their numeric behavior, but their
  gripper-slot meaning is now 1 open and 0 closed.

## Configuration

### Joint-space vectors

`joint_low`/`joint_high`, `home_pose`, `rest_pose`, actions, and the observed
`joint_pos` state all use *policy units*:

| Slots | Meaning | Unit |
|-------|---------|------|
| 0–5, 7–12 | left / right arm revolute joints | radians |
| 6, 13 | left / right gripper | normalized 0–1 (1 = open, 0 = closed) |

Hardware gripper units (via `gripper_open`/`gripper_closed`) exist only at the
driver boundary; pose and limit vectors never use driver-native gripper units.

In `control_interface="eef_pos"`, actions and `eef_low`/`eef_high` are 10-D:

| Slots | Meaning | Unit |
|-------|---------|------|
| 0–2, 5–7 | left / right EEF x, y, z in each arm's base frame | metres |
| 3, 8 | left / right yaw relative to reset orientation | radians |
| 4, 9 | left / right gripper | normalized 0–1 (1 = open, 0 = closed) |

`home_pose`, `rest_pose`, joint limits, and parking remain 14-D joint-space
vectors in both control interfaces.

`YamConfig`: `left_channel`, `right_channel`, `gripper_type` (i2rt `GripperType`
enum *name*, e.g. `LINEAR_4310`; grippers only: `NO_GRIPPER`/`YAM_TEACHING_HANDLE`
would break the 14-D packing and are rejected), `control_hz`, `cam_height/width`,
`joint_low/high`, `control_interface` (`joints` by default or `eef_pos`),
`docs_extra` (rig-specific notes appended to the built-in agent documentation),
`eef_low/high`, `ik_max_iters`, `ik_step_joint_limit`,
`cmd_resync_threshold`, `osc_deadband`, `osc_reversals`, `osc_window`,
`osc_hold_steps`, `home_pose` (reset always ramps here smoothly over
`rest_secs`; `none` selects `DEFAULT_JOINT_HOME_POSE` in joint mode or
`DEFAULT_EEF_HOME_POSE` in EEF mode), `rest_pose` (close park target; defaults
to the factory zero-joint, open-gripper pose equal to the joint factory home,
accepts a per-rig override, and accepts `none` to fall back to the pose captured
at the first reset before torque is released),
`rest_secs` (ramp duration, default 3.0), `gripper_open/closed`,
`joints_are_delta`, `zero_gravity_mode` (default `True`; see *Safety*),
`unattended` (default `False`; skip operator prompts),
`top/left/right_cam_device` (V4L2 paths for the builtin camera reader; all
three or none), `max_steps_hint` (deprecated: on inspect-robots newer than
0.8.1, framework runs feed the status line the real horizon automatically;
the hint is only a fallback for direct `rollout()` calls or older cores;
bounds nothing).
The current factory value is available for inspection as
`inspect_robots_yam.config.DEFAULT_REST_POSE`; this is an informational constant,
not a stable import.
`ActServerConfig`: `server_url`, `endpoint`, `num_steps` (the wire field: the
server's flow-matching denoising steps, *not* the chunk length),
`action_horizon` (the checkpoint's advertised chunk length, 30 for the bimanual
YAM tag; metadata only), `timeout_s`, `camera_order`, `state_key`,
`cam_height/width`, `name` (the policy label recorded in eval logs).

Scalar knobs are settable from the CLI:
`inspect-robots run -P server_url=http://gpu:8202 -E left_channel=can0 ...`.

## Development

> **Dependency changes:** after editing dependencies in `pyproject.toml`, run
> `uv lock` and commit the updated lockfile. CI installs with
> `uv sync --locked` and fails with "the lockfile needs to be updated" if you
> forget. Day-to-day conventions (PR-only `main`, the required `ci-ok` check,
> one-click releases) are documented in [`CLAUDE.md`](CLAUDE.md).

```bash
uv venv && uv pip install -e ".[dev]"     # inspect_robots + kitchenbench from PyPI
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

Every public module, class, and function needs a docstring, enforced by Ruff D1;
state the contract instead of restating the symbol name.

The whole suite runs with no hardware, no server, and no stdin: the i2rt
driver, cameras, the `/act` transport, the clock, and operator I/O are all
injected. The default hardware seams are excluded from coverage (`# pragma: no
cover`).

## Citation

If you use Inspect Robots YAM in your research, please cite it:

```bibtex
@software{inspect-robots-yam,
  author  = {Robocurve},
  title   = {Inspect Robots YAM: Adapters for I2RT YAM bimanual arms},
  year    = {2026},
  url     = {https://github.com/robocurve/inspect-robots-yam},
  version = {0.3.0},
  license = {MIT}
}
```

## License

[MIT](LICENSE)
