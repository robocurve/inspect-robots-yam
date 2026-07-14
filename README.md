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
- **`yam_arms` embodiment**: the I2RT joint-position driver, with a hard safety
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
uv pip install inspect-robots-yam
# The i2rt driver is git-only and not on PyPI. Install it directly:
uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"
```

The base package includes the `/act` transport and builtin OpenCV camera reader.
Only `i2rt`, the I2RT YAM arm driver required for real hardware, needs the
separate git install. The camera reader depends on `opencv-python-headless`;
if your environment also carries `opencv-python`, the two share the `cv2`
module and the last one installed wins.

Then download the model weights (needs a Hugging Face token) and start the server,
from the [MolmoAct2 repo](https://github.com/allenai/molmoact2):

```bash
huggingface-cli download allenai/MolmoAct2-BimanualYAM
python examples/yam/host_server_yam.py          # serves /act on :8202
```

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
uv run inspect-robots setup
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
uv run inspect-robots "place the fork on the plate"
```

The attended flow: position the scene, press Enter to start, press any key to
end the episode, answer y/N to score.

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

```bash
uv pip install inspect-robots-agent inspect-robots-yam
inspect-robots config set embodiment yam_arms     # once, per machine
export ANTHROPIC_API_KEY=sk-ant-...

# Cameras come from the builtin reader: set the three *_cam_device paths in
# ~/.config/inspect-robots/config.ini (see Quickstart above) or pass them as
# -E flags per run.
inspect-robots "place the fork on the plate" --policy agent \
    -P model=anthropic/claude-fable-5
```

Safety guardrails (a bounds clamp plus a per-step delta limit derived from the
declared action space) are wired in by default for every CLI run; turning them
off requires an explicit `--disable-guardrails`.

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

Set a resting pose so runs end with a gentle 3-second park instead of the
arms going limp mid-air (pose fields accept comma-separated values from the
CLI and config.ini):

```ini
[embodiment.args]
rest_pose = -0.002,0.002,0.002,-0.089,0.007,-0.026,1.0,-0.006,0.002,0.001,-0.087,-0.007,-0.019,1.0
```

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
  action is still a stiff PD command that can jump from wherever the arm ended
  up. Nothing bounds the per-step joint delta yet (tracked as a known issue);
  stand clear when the episode starts, and set `home_pose` so episodes begin
  from your checkpoint's trained start state.
- **Absolute vs. delta joints: verify first.** MolmoAct2's YAM `actions` are
  treated as *absolute* joint targets by default. If your checkpoint emits
  deltas, set `YamConfig(joints_are_delta=True)` (the embodiment converts to
  absolute internally so the declared `joint_pos` stays honest). Inspect Robots's
  compat check *cannot* tell these apart: confirm with `--dry-run` and a single
  slow jog before running a task.
- **Gripper polarity/trim.** The i2rt driver already exposes the YAM gripper as
  normalized 0–1 in both directions, so the defaults (`gripper_open=0.0`,
  `gripper_closed=1.0`) are an identity map and correct for standard grippers.
  `YamConfig(gripper_open=..., gripper_closed=...)` is a polarity/trim remap over
  that already-normalized range. Its main use is a gripper wired with inverted
  polarity (`gripper_open=1.0, gripper_closed=0.0`). The remap is a bijection:
  commands are de-normalized on the way out and observations are re-normalized on
  the way back, so the model always sees 0–1. **Warning:** values outside [0, 1]
  are forwarded on a path i2rt does *not* clip. Avoid them unless you have
  verified your firmware's behavior.

## Configuration

### Units: every 14-D vector uses the same layout

`joint_low`/`joint_high`, `home_pose`, `rest_pose`, actions, and the observed
`joint_pos` state all use *policy units*:

| Slots | Meaning | Unit |
|-------|---------|------|
| 0–5, 7–12 | left / right arm revolute joints | radians |
| 6, 13 | left / right gripper | normalized 0–1 (0 = open, 1 = closed) |

Hardware gripper units (via `gripper_open`/`gripper_closed`) exist only at the
driver boundary; nothing you configure here is in hardware gripper units.

`YamConfig`: `left_channel`, `right_channel`, `gripper_type` (i2rt `GripperType`
enum *name*, e.g. `LINEAR_4310`; grippers only: `NO_GRIPPER`/`YAM_TEACHING_HANDLE`
would break the 14-D packing and are rejected), `control_hz`, `cam_height/width`,
`joint_low/high`, `home_pose` (reset ramps here smoothly over `rest_secs` rather
than jumping), `rest_pose` (explicit close park override; by default, close
ramps back to the pose the arms were in at the first reset before torque is
released),
`rest_secs` (ramp duration, default 3.0), `gripper_open/closed`,
`joints_are_delta`, `zero_gravity_mode` (default `True`; see *Safety*),
`unattended` (default `False`; skip operator prompts),
`top/left/right_cam_device` (V4L2 paths for the builtin camera reader; all
three or none), `max_steps_hint` (display-only horizon for the operator
status line; bounds nothing).
`MolmoActConfig`: `server_url`, `endpoint`, `num_steps` (the wire field: the
server's flow-matching denoising steps, *not* the chunk length),
`action_horizon` (the checkpoint's advertised chunk length, 30 for the bimanual
YAM tag; metadata only), `timeout_s`, `camera_order`, `state_key`,
`cam_height/width`.

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
