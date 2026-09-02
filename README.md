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

> **Note:** cameras are configured with one plain source per slot
> (`*_cam_device` or `*_depth_serial`), so the whole rig is drivable from
> config.ini or `-E key=value` flags with no custom code. A Python
> `camera_reader` remains available for exotic camera stacks. With no sources
> configured, `yam_arms` fails fast with a `ConfigError` at `reset()`, before
> any driver connect or motion.

The builtin reader drains each camera continuously on its own thread, so an
observation carries a frame about one camera frame interval old: 33 ms at
30 fps. Without that, a V4L2 queue read at `control_hz` hands back a frame
`N/control_hz` old, measured at 380 ms on a 10 Hz rig and worse as the control
rate falls. Two limits worth knowing: freshness is bounded by the camera's own
frame rate, which no setting here changes (a 5 fps camera means 200 ms whatever
the control rate), and a camera that stops delivering for half a second raises
rather than serving a stale frame. A custom `camera_reader` that owns devices
should expose a `close()`, which the embodiment calls during teardown.

## Install (on the robot/GPU machine)

```bash
uv venv && source .venv/bin/activate
uv pip install inspect-robots-yam
# The i2rt driver is git-only and not on PyPI. Install it directly. The commit is
# the one the rigs run: it defaults enable_auto_recovery=False, so a motor error
# fails fast rather than being cleaned and re-enabled inside the control loop.
# The build-constraints file works around a build failure in i2rt's ruckig
# dependency (source-only releases that no longer build under scikit-build-core
# 1.0; the pin below 0.10 matches i2rt's own in-repo workaround):
echo 'scikit-build-core<0.10' > build-constraints.txt
uv pip install --build-constraints build-constraints.txt "i2rt @ git+https://github.com/i2rt-robotics/i2rt@ac096928d6899ddf852a71c5e8fbaa6055cd9745"
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
The policy's `server_url` and `remedy` attributes feed core's
connection-failure hint with the configured address and a recovery
instruction; each policy entry defaults `remedy` to its own canonical server
launch command, and `-P remedy=...` replaces it (empty string omits the line).
For another GR00T fine-tune, pass `-P action_horizon=<its chunk length>` so the
recorded policy metadata matches that checkpoint.

> [!WARNING]
> The shim's startup checks validate the packed layout and units ranges, but
> joint polarity and absolute-vs-delta semantics cannot be detected from
> dataset statistics. For the first runs with a new checkpoint family, run
> `inspect-robots-yam-preflight`, leave guardrails on (the bounds clamp and
> per-step delta limit are always active; add the collision guardrail once
> the rig's `collision_*_base_pos` geometry is measured), and keep an
> operator at the e-stop.

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

## Health check: verify the idle rig

Check that all three cameras deliver fresh, non-uniform frames and that both
arms report finite joint positions within the configured limits:

```bash
# Uses the devices and CAN channels saved by `inspect-robots setup`.
inspect-robots-yam-health

# Or override the configured camera slots explicitly.
inspect-robots-yam-health \
  --top-cam /dev/v4l/by-id/...-top \
  --left-cam /dev/v4l/by-id/...-left \
  --right-cam /dev/v4l/by-id/...-right
```

Like `inspect-robots` itself, both `inspect-robots-yam-health` and
`inspect-robots-yam-holdcheck` honor the working directory's `.env` before
resolving wizard configuration, including an `INSPECT_ROBOTS_CONFIG` pin for
selecting the current rig. Pass `--no-config` to either command to bypass the
wizard configuration; holdcheck then requires a raw CAN channel instead of
`left` or `right`.

The command writes a labeled montage to `health.jpg`. Use `--out PATH` to
change the destination, `--json` for a machine-readable report, or
`--skip-cameras` and `--skip-motors` to run one section. Camera devices can
also be supplied with `-E top_cam_device=...`, `-E left_cam_device=...`, and
`-E right_cam_device=...`. Explicit flags and `-E` values override the wizard
config one camera slot at a time. For health, the `--no-config` bypass also
applies when the wizard file is malformed and restores flag-only behavior.

> [!NOTE]
> The health tool can check and watch only V4L2 `*_cam_device` sources. It
> reports configured `*_depth_serial` slots as unchecked; they do not pass or
> fail camera health. On an all-depth rig, camera checks are skipped while
> motors are still checked, and `--watch` errors because there are no streams
> this tool can serve. On a mixed rig, the montage and watch page contain only
> the V4L2 slots.

> [!WARNING]
> Run the health check only while the rig is idle, with both arms at rest or
> supported and an e-stop in hand. Connecting and then closing the motor driver
> drops motor torque. Do not use the mid-workspace holdcheck setup, and do not
> run this command concurrently with an eval.

### Live view: aim the cameras

Stream the configured V4L2 cameras while positioning them:

```bash
inspect-robots-yam-health --watch

# Flags remain available when no wizard config should be used.
inspect-robots-yam-health --watch --no-config \
  --top-cam /dev/v4l/by-id/...-top \
  --left-cam /dev/v4l/by-id/...-left \
  --right-cam /dev/v4l/by-id/...-right
```

Open `http://<host>:8807/` and press Ctrl-C to stop. Watch never touches the
motors, so the torque warning above does not apply.

The stream is unauthenticated, and the default `0.0.0.0` bind listens on all
interfaces. Use `--bind <tailscale-ip>` to limit it to the rig's tailnet
address.

## Named start poses: capture and reuse a rig setup

Capture a joint-space start pose by bringing the arms up in gravity-compensation
mode, moving both arms and grippers by hand, and pressing Enter:

```bash
inspect-robots-yam-pose capture table-ready --notes "bowls placed for pouring"
```

The command writes `poses/table-ready.json` by default. It prompts you to
support both arms before closing the driver and releasing torque. Pass `--park`
to gate and ramp to the configured rest pose first, or `--clamp` to explicitly
clamp arm joints that are outside the configured limits. Gripper readings are
always normalized to the portable 0 to 1 range.

Each file is plain JSON with the packed left-then-right 14-slot joint layout:

```json
{
  "schema": 1,
  "name": "table-ready",
  "joints": [0.12, 0.65, 1.04, -0.31, 0.08, 0.0, 0.72, -0.1, 0.7, 0.98, -0.28, -0.05, 0.02, 0.68],
  "created_at": "2026-08-19T19:30:00+00:00",
  "notes": "bowls placed for pouring",
  "rig": "rig-1"
}
```

Use the pose for an eval by setting its name on the embodiment:

```bash
inspect-robots "pour the pasta into the bowl" \
    --embodiment yam_arms -E start_pose=table-ready
```

Set `pose_dir` in `[embodiment.args]`, pass `-E pose_dir=...`, or use the pose
tool's `--pose-dir` flag to select another store. Commit `poses/` with a rig
configuration or copy the directory between compatible rigs to share poses.
The normalized gripper slots remain portable across different native gripper
calibrations.

> [!WARNING]
> Before using a new pose in an unattended eval, run
> `inspect-robots-yam-pose goto table-ready` and verify the full ramp while
> ready on the e-stop. The straight-line joint interpolation checks configured
> joint limits, but it does not perform collision checking.

## Run on hardware

Write your defaults once. The interactive wizard interviews this plugin's
declared devices (three cameras and both arms' CAN channels) with live
probes, including unplug-to-identify. It also offers the `auto_start`,
`collision_guardrail`, `report_joint_eff`, and `eef_orientation` boolean
options. The `eef_orientation` option applies to `eef_pos` rigs and reminds
you to raise the `eef_low` z floor after opening tilt axes:

```bash
inspect-robots setup
```

Or write the file yourself. This example uses the primary mixed RealSense rig:
the D435 top camera stays on V4L2 while librealsense owns both D405 wrists.
Use stable `/dev/v4l/by-id/...` or udev-symlink paths for V4L2 sources; bare
`/dev/videoN` numbers reshuffle on every replug.

```bash
mkdir -p ~/.config/inspect-robots && cat > ~/.config/inspect-robots/config.ini <<'EOF'
[defaults]
policy = molmoact2
embodiment = yam_arms
scorer = operator          # scores the verdict you type at the end-of-episode prompt
max_steps = 1200           # 120 s at 10 Hz
rerun = true               # live viewer of cams/state/actions (inspect-robots[rerun])
store_frames = true        # keep the policy's camera frames per run

[embodiment.args]
top_cam_device = /dev/v4l/by-id/YOUR-TOP-CAM
# A depth serial replaces, rather than augments, that slot's *_cam_device.
# Do not also set left_cam_device or right_cam_device in this mixed rig.
left_depth_serial = YOUR-LEFT-D405-SERIAL
right_depth_serial = YOUR-RIGHT-D405-SERIAL
EOF
```

### Repeat one task N times with a human in the loop

`scripts/run_batch.sh` runs the same task N times from a rig directory (a
directory holding a `./run` wrapper and its `config.ini`). Type the prompt,
policy, and effort once; everything except `-n` is forwarded to `./run`:

```bash
cd ~/robocurve/rig-1
../inspect-robots-yam/scripts/run_batch.sh -n 20 \
    --instruction "Place the fork on the plate" -P model=claude-opus-5 -P effort=medium
```

Each trial is a separate `./run` process with `--epochs 1` forced. That
process asks the operator for a verdict after the episode (the grading pause),
then parks the arms and releases torque on exit. Only then does the script ask
you to reset the scene; the next trial, which powers the arms back on and
ramps to the start pose, begins when you press Enter (`q` stops the batch).
Keystrokes typed while the arms were parking are discarded before that prompt.
A trial that does not exit cleanly gets a warning instead of the torque-off
claim and asks whether to continue: check the arms are limp before reaching in.
Ctrl-C cancels the running trial (the framework writes a cancelled log and
parks) and ends the batch. Per-trial verdicts are read from the eval logs into
`<log-dir>/batches/<stamp>.tsv`, echoed after each trial, and tallied at the end.

Right before each trial launches, after you confirm the reset and before the
arms power on, the script saves one top-camera JPEG of the scene to
`<log-dir>/batches/batch_<stamp>/trial_NN_<run-id>_start.jpg` (the run id is
appended once the eval log exists). It reads `top_cam_device` from
`config.ini` (or `-E top_cam_device=...`) and opens it with OpenCV from the
shared venv the way the plugin's V4L2 reader does. A camera failure warns and
never blocks the trial; `--no-snapshots` turns it off.

Any other `--epochs` value is rejected on purpose: within one process the arms
stay connected and torque-held at the home pose between epochs while you reach
into the scene.

### Packing stored frames

`scripts/pack_frames` stages each camera's H.264 CRF 16 evidence MP4 and a
bit-exact FFV1 Matroska raw archive in `/tmp` at the run's `control_hz`, then
backs both up with rclone to
`gdrive-rc:rig-video/<host>/<rig>/<stamp>/` on the `pi05_kaedim_tasks` shared
drive. After the upload is verified and a 600-second grace period has elapsed,
it deletes the much larger `.npy` inputs and only then moves the MP4s and final
manifest into the frames directory. The FFV1 files remain in backup and leave
with scratch unless `--keep-raw-local` is passed. Use `--raw none` to disable
the lossless archive. Upload/verification failures leave `.npy` untouched;
post-deletion handoff failures preserve scratch for recovery. The default
`--min-height 360` keeps 224x224 runs as `.npy`; pass `--min-height 0` to
include them. Use `--since <ISO8601>` to restrict work to runs starting at or
after an instant, and repeat `--policy <name>` to allow only selected policies;
the same filters apply to `--run`, `--all`, and `--status`.

`run_batch.sh` starts this packer in the background after each trial; use
`--no-pack` to disable that hook. Runs made directly with `./run` must be packed
manually, and status can be inspected from the rig directory:

```bash
./pack-frames --run logs/<run-name>.json
./pack-frames --status
./pack-frames --status --verify
```

To restore one stream after downloading its FFV1 archive, decode it and split
the RGB frames using `height`, `width`, and `first_step` from
`pack_manifest.json`:

```bash
ffmpeg -i scene-0-e0_top_cam.ffv1.mkv -f rawvideo -pix_fmt rgb24 top.raw
```

```python
import json, numpy as np
m = json.load(open("pack_manifest.json"))
H, W = m["raw"]["top"]["height"], m["raw"]["top"]["width"]
first_step = m["streams"]["top"]["first_step"]
arr = np.fromfile("top.raw", np.uint8).reshape(-1, H, W, 3)
for i, frame in enumerate(arr): np.save(f"scene-0-e0_top_cam_{first_step + i:06d}.npy", frame)
```

`--status`, `--all`, and repeat `--run` checks normally trust matching MP4 size
and modification time from the manifest, avoiding multi-gigabyte hashes;
`--verify` forces SHA-256 checks. The per-run lock and log live beside the
selected JSON under `<log-dir>/pack/`.

Exit status 0 means packed/status/dry-run success, 1 means a failure, 2 means
invalid usage, and 3 means the selected run was skipped or not eligible.

### RealSense depth

Install the optional librealsense dependency on the robot machine:

```bash
uv pip install 'inspect-robots-yam[depth]'
```

A slot configured with `*_depth_serial` is owned by librealsense, which serves
both its colour image and aligned depth plus intrinsics. Find the device serial
with `rs-enumerate-devices`, or reuse the ASIC serial embedded in the
`/dev/v4l/by-id/...` name used for `*_cam_device`; either namespace is accepted.
Quote all-digit serials in `config.ini` (`top_depth_serial = "0385..."`) —
unquoted numeric values are int-coerced and rejected with a hint.

RealSense capture runs in an isolated child process by default
(`realsense_capture = process`), keeping librealsense and frame-copy work away
from the motor-control interpreter; `realsense_capture = inline` restores the
in-process reader as a debugging escape hatch. `depth_fps` (default 30) sets
both stream rates — devices accept only their discrete rates (D435/D405:
6/15/30/60/90).

Cameras open lazily, so the first `reset()` has a one-time warm-up cost while
the pipelines start and deliver their first frames. A RealSense opened through
librealsense cannot also be opened through V4L2—there can be only one streamer
per device node—so `*_cam_device` and `*_depth_serial` are mutually exclusive
for each slot.

Make sure the plugin is installed and the MolmoAct2 server is up. The
`molmoact2` policy is only a client: nothing moves until the server is
listening, and it does not start itself or survive a reboot. A connection
failure names the configured server address in the policy error (full setup in
[Install](#install-on-the-robotgpu-machine)):

```bash
uv pip install inspect-robots-yam   # provides the molmoact2 policy + yam_arms rig
# On the GPU machine, from the MolmoAct2 repo. Leave it running, e.g. in tmux:
python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202
curl http://127.0.0.1:8202/act      # 200 means the server is ready
```

Then tell the robot what to do:

```bash
inspect-robots "place the fork on the plate"
```

The attended flow has two terminal modes. When the framework connects its
operator session, press Enter at either readiness gate, then press Esc (or
type `/stop`) to end the episode; typed lines become policy feedback or
logged notes.
On the never-connected legacy path, press Enter at the gates and press any key
to end the episode. In both modes the status shows motion-budget consumption
against the run's real step limit, with separately labeled wall time
(`t = 42s / ~120s | wall 75s`), and needs no configuration (requires
inspect-robots newer than 0.8.1; on older cores set `max_steps_hint`).

To skip both Enter gates, set `auto_start=true` (CLI: `-E auto_start=true`,
persistently via `[embodiment.args]` in config.ini, or accept the suggested
yes when `inspect-robots setup` offers the toggle). The arms home immediately
after a one-line stand-clear notice and the episode starts right after the
homing ramp, so stage the scene before launching the run. The same holds
between episodes of a multi-episode run: the next episode starts as soon as
the arms re-home, so restage while answering the grading prompt, not after.
Everything else about the attended flow stays: the status line, the
active mode's end control, and operator grading, which is also why `auto_start`
refuses to run without an interactive terminal.

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

Pressing the end-episode key terminates the episode with
`termination_reason="operator_end"` — the embodiment itself asks nothing.
On CLI runs (inspect-robots ≥ 0.25), the framework then asks once per trial:
`did the robot succeed? [y/n/partial/skip]` plus an optional grader note.
The bare `eval()` call above never prompts: pass
`before_scoring=` a callable that sets `record.operator_judgement` (grade
live, or from your own UI) when driving the Python API directly.
Score attended runs with the `operator` scorer (reads the recorded judgement);
KitchenBench's `task_success` reads it too. `success_at_end` only counts
embodiment-detected success terminations, so it scores operator-graded runs as
failures — don't pair it with attended yam runs.
The operator prompts need an interactive terminal: a dead stdin raises
`EmbodimentFault` (the framework's always-halt path). For runs with no operator,
set `YamConfig(unattended=True)` (CLI: `-E unattended=true`): all operator
prompts are skipped and every episode runs to `max_steps`, scoring as a failure.
For attended runs that only want to drop the Enter gates, use `auto_start=true`
instead; `unattended` wins when both are set.

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
uv pip install -U inspect-robots-agent inspect-robots-yam
inspect-robots config set embodiment yam_arms     # once, per machine
```

The `-U` matters if you installed the agent plugin before: the run below needs
its native Anthropic wire, added in `inspect-robots-agent` 0.13.0.

Cameras come from the builtin reader: configure one `*_cam_device` or
`*_depth_serial` source per slot in `~/.config/inspect-robots/config.ini` (see
Run on hardware above), or pass them as `-E` flags per run. Then run the LLM on
the robot:

```bash
inspect-robots "place the fork on the plate" --policy agent \
    -P model=anthropic/claude-opus-5 \
    -P wire=anthropic -P speed=fast -P effort=high \
    -P max_output_tokens=32000
```

`-P wire=anthropic` drives Claude through its native Messages API, which is what
`-P speed=fast` needs: the same model served at up to 2.5x higher output tokens
per second, for roughly double the standard price. That trade is worth more here
than in sim, because the arms hold their pose while the model thinks, so serving
latency is time the fork spends waiting. Fast mode covers Claude Opus 5 and Opus
4.8 on the Claude API. `-P effort=high` buys deeper reasoning for a contact-rich
task, and since thinking bills against the same budget as the reply, the output
cap goes up with it.

Drop those four flags to run any OpenAI-compatible model instead, such as
`-P model=openai/gpt-5.6` or `-P model=anthropic/claude-fable-5`.

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

For LLM-agent runs, opt into the 14-D absolute Cartesian interface:

```ini
[embodiment.args]
control_interface = eef_pos
```

Each arm is controlled as `x, y, z, yaw, pitch, roll, gripper`. Positions are
metres in that arm's own base frame, with +x forward from the base and +z up.
The two base frames are independent. On common mirrored bimanual mounts, the
arms' +y axes point in opposite world directions, so equal signed y targets do
not mean equal world directions.

All three orientation slots are absolute targets relative to the orientation
captured at reset: `0, 0, 0` means the reset orientation. Yaw rotates about
base +z (positive counterclockwise from above); positive pitch tips the tool
forward (+x at yaw 0); positive roll tips it toward the arm's left (+y at
yaw 0). Orientation interpolation does not wrap. A yaw move from `3.1` to
`-3.1` sweeps through zero instead of taking the short path, so use
intermediate yaw targets for near-±π regrasps.

The default workspace per arm is x `[0.15, 0.48]`, y `[-0.25, 0.25]`, and z
`[0.03, 0.40]`, with yaw `[-π, π]`, pitch and roll pinned at `[0, 0]`,
and gripper `[0, 1]`. Pinned axes are declared but not commandable. The
default behaves exactly like the historical yaw-only interface. Opening
pitch and roll is supported through `eef_orientation=true`, which widens each
exactly `0,0` pitch pin to `[-0.6, 0.6]` and roll pin to `[-π/2, π/2]` in the
effective `eef_low`/`eef_high` bounds. The rewrite also applies when the tuples
contain tuned position bounds. Custom orientation bounds remain supported:
pitch must stay strictly inside `(-π/2, π/2)` and roll within `[-π, π]`.
With `eef_orientation=true`, a `0,0` pitch or roll pin is widened. To re-pin
one, set `eef_orientation=false` or pin it at a nonzero epsilon. See the
z-floor WARNING below before opening either axis. These bounds were validated
against the bundled YAM + LINEAR_4310 model at the default working orientation,
but they are a conservative box rather than an exact reachable set. `eef_low`
and `eef_high` override all fourteen bounds. The observation keeps the 14-D
`joint_pos` field for logging and adds the command-aligned 14-D `eef_state`
field.

> [!WARNING]
> The z floor (`z >= 0.03`) protects *fingertips* assuming a gripper-down
> tool. A pitched or rolled gripper can reach the table with its knuckles or
> wrist camera at a legal fingertip z. When opening pitch or roll, raise the
> z lower bound to cover the tilted gripper body. The run warning checks open
> axes only. A deliberate nonzero tilt pin has the same knuckles-first hazard
> but does not trigger that warning, so its operator must still raise z.

In both control interfaces, `home_pose=None` selects a mandatory per-mode
factory default instead of skipping homing. Joint mode uses the
dataset-verified `DEFAULT_JOINT_HOME_POSE`, with every joint at encoder zero
and both grippers open. EEF mode uses `DEFAULT_EEF_HOME_POSE`; its provisional
per-arm joints are `[-0.024, 0.794, 0.645, -0.375, -0.021, -0.012]`, with both
grippers open. The first EEF reset validates that the configured home FK lies
in the workspace box before moving, then captures each arm's yaw reference
after homing. Named `start_pose` poses work in EEF mode too: the resolved
joint-space pose must start inside the EEF action box (grasp-point position,
gripper aperture, and relative yaw/pitch/roll 0), and a reconnect revalidates
the re-read pose file.

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
> inspect-robots-yam-holdcheck left --zero-gravity true
> inspect-robots-yam-holdcheck right --zero-gravity true
> ```
>
> (`left` and `right` resolve through the wizard config; a raw interface such
> as `can0` or `can_left` still passes through unchanged.) PASS in the mode you
> run agents in closes the verification. The
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

## Collision guardrail

`YamConfig.collision_guardrail` defaults to `True`. In absolute joint mode,
`YAMEmbodiment` contributes a predictive MuJoCo guardrail automatically. A
blocked target becomes a hold at the last safe commanded pose and is marked in
the recorded action metadata. The setup wizard suggests answering **no** to
its `collision_guardrail` question until the rig's `collision_*_base_pos`
geometry below is measured: on unmeasured geometry the guardrail can
false-positive hold, and a policy that repeats the blocked target livelocks
to `max_steps`. Answer yes (or set the key to `true`) once the base positions
are measured; a config that already sets the key keeps its value as the
wizard suggestion. If your config sets the geometry keys but relies on the
runtime default instead of writing `collision_guardrail = true`, set the key
explicitly before re-running setup — otherwise the wizard suggests off and an
Enter-accept would disable an already-measured guardrail. If MuJoCo is
unavailable, the run continues with
a warning that includes this install command:

```bash
pip install "inspect-robots-yam[collision]"
```

Configure measured rig geometry in `config.ini` or with `-E` arguments:

```ini
[embodiment.args]
collision_left_base_pos = 0.0,0.3,0.0
collision_right_base_pos = 0.0,-0.3,0.0
collision_left_base_yaw = 0.0
collision_right_base_yaw = 0.0
collision_table_height = 0.0
```

> [!WARNING]
> The default base offsets, `(0.0, 0.3, 0.0)` and `(0.0, -0.3, 0.0)`, are
> unverified placeholders. Measure the mounting position and yaw of both bases
> on every rig, then override them. Incorrect offsets can silently miss real
> cross-arm collisions or block safe motions.

Two known false-positive classes can change eval results:

- Table-press grasps can hold when demonstration-derived targets press slightly
  through the modeled table. Raise `collision_penetration_threshold` or lower
  `collision_table_height`. On a tableless rig, set `collision_table=false`.
- Bimanual close-quarters work such as handovers or clapping can hold because
  both finger joints are modeled at their open extremes. A policy that repeats
  the blocked target can livelock until `max_steps`. Configure both measured
  base positions to reduce cross-arm geometry error, or set
  `collision_guardrail=false` for that rig.

The run refuses to start when the configured home pose is already in collision
under the effective geometry. Correct the `collision_*` geometry fields or set
`collision_guardrail=false` after verifying that an opt-out is appropriate.

The guardrail supports only 14-D absolute `joint_pos` actions. It refuses EEF
and `joint_delta` spaces because an approver sees Cartesian targets or deltas
before the embodiment converts them. Those modes continue with a skip warning.

The checker models commanded poses, not measured arm motion. Physical motion can
lag or sag away from checked waypoints, including with the default
`zero_gravity_mode=true`, so do not reduce clearance margins to zero.
`build_yam_guardrails` remains available for programmatic chains and strict
abort behavior.

This guardrail reduces collision risk. It does not model props, certify a
continuous path, observe the measured arm trajectory, check reset or park
motions, or replace the operator and physical e-stop.

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
  With `park_before_grade=true`, the arms also make the same motion as the
  `close()` park at episode end, before grading. This is a new time for that
  motion and there is no stand-clear gate. Tasks whose success state is the
  gripper holding an object must set `park_before_grade=false` so the grader
  uses the last step's frames instead.
  Override both `home_pose` and `rest_pose` on rigs whose joint limits exclude
  zero, since both targets are clamped through the same per-joint box as every
  command.
- **Absolute vs. delta joints: verify first.** MolmoAct2's YAM `actions` are
  treated as *absolute* joint targets by default. If your checkpoint emits
  deltas, set `joints_are_delta=True` on both the policy and embodiment. They
  then declare `joint_delta`, so compatibility checking rejects a mode
  mismatch before motion. Confirm a new checkpoint's value scale and joint
  mapping with `--dry-run` and a single slow jog before running a task.
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

In `control_interface="eef_pos"`, actions and `eef_low`/`eef_high` are 14-D:

| Slots | Meaning | Unit |
|-------|---------|------|
| 0–2, 7–9 | left / right EEF x, y, z in each arm's base frame | metres |
| 3, 10 | left / right yaw relative to reset orientation | radians |
| 4, 11 | left / right pitch relative to reset orientation (pinned at 0 by default) | radians |
| 5, 12 | left / right roll relative to reset orientation (pinned at 0 by default) | radians |
| 6, 13 | left / right gripper | normalized 0–1 (1 = open, 0 = closed) |

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
`auto_start` (default `False`; skip both operator Enter gates but keep the
attended episode flow; needs a TTY; `unattended` takes precedence),
`report_joint_eff` (default `False`; add the optional `joint_eff` observation
state with sign-corrected estimated torque in raw N·m, including the gripper
slots),
`park_before_grade` (default `True`; park for an unobstructed final grader view;
set `False` for tasks whose success state is the gripper holding an object so
grading uses the last step's frames),
`eef_orientation` (default `False`; widen exactly zero-pinned EEF pitch and
roll bounds to conservative ranges; set it back to `False` or use a nonzero
epsilon to re-pin, and raise the EEF z floor as described above),
`collision_guardrail` (default `True`; predictive holds in absolute joint
mode; the setup wizard suggests `false` until the base positions below are
measured),
`collision_left_base_pos`, `collision_right_base_pos`,
`collision_left_base_yaw`, `collision_right_base_yaw` (optional measured rig
geometry), `collision_table` (default `True`; set `False` for no table plane),
`collision_table_height`, `collision_penetration_threshold` (optional collision
model overrides),
`motor_temp_limit` (degrees C; `none` by default, which disables the thermal
guardrail), `motor_temp_warn_margin` (degrees C below the limit; default `10.0`),
`settle_tolerance` (radians; `none` by default, which disables settling; see
*Settling before observing*), `settle_timeout_s` (default `1.0`),
`settle_timeout_budget` (default `20`),
`top/left/right_cam_device` (V4L2 camera sources; each slot needs either its
device path or its depth serial), `top/left/right_depth_serial` (RealSense
sources owned by librealsense, serving both colour and depth; mutually exclusive
with that slot's `*_cam_device`; either device or ASIC serial namespace is
accepted; all slots must be sourced or none), `max_steps_hint`
(deprecated: on inspect-robots newer than 0.8.1, framework runs feed the status
line the real horizon automatically; the hint is only a fallback for direct
`rollout()` calls or older cores; bounds nothing).
The current factory value is available for inspection as
`inspect_robots_yam.config.DEFAULT_REST_POSE`; this is an informational constant,
not a stable import.

The thermal guardrail compares `motor_temp_limit` with the hotter of the MOS
and rotor readings for every arm and gripper motor. At episode start, a motor
at the limit refuses the reset before the arms move. During an episode, a
confirmed over-limit reading ends the trial with `overheat` while the motors
still have torque, allowing the normal grading flow to run. Before returning,
the trip parks to rest immediately in every mode, including ungraded and
unattended runs and when `park_before_grade=false`. Thermal safety outranks
that flag's scene-preservation preference. The setup wizard offers
`motor_temp_limit` (suggested `70`; answer `none` to leave it off). Run
`inspect-robots-yam-health` after a long episode to see the hottest motor, then
choose a limit comfortably below the temperature where the firmware has
faulted on that rig.

`ActServerConfig`: `server_url`, `remedy` (connection-failure recovery
instruction; defaults to the policy entry's canonical server launch command
plus a docs link), `endpoint`, `num_steps` (the wire field: the server's
flow-matching denoising steps, *not* the chunk length),
`action_horizon` (the checkpoint's advertised chunk length, 30 for the bimanual
YAM tag; metadata only), `timeout_s`, `camera_order`, `state_key`,
`cam_height/width`, `name` (the policy label recorded in eval logs).

Scalar knobs, including the free-text remedy, are settable from the CLI:

```bash
inspect-robots run -P server_url=http://gpu:8202 \
    -P remedy='run ~/robocurve/molmoact2/run_yam.sh' -E left_channel=can0 ...
```

### Settling before observing

By default `step()` commands a pose, paces out the control period, and observes,
without checking that the arm arrived. A VLA running closed loop at `control_hz`
is fine with that, since its next observation is 100 ms away either way.

Chunked policies are not. The `agent` policy interpolates one tool call into up
to 100 actions and only looks at the observation from the last of them, so it
plans its next motion from a pose the arm may not have reached.

Setting `settle_tolerance` makes `step()` and `reset()` wait for every arm joint
to come within that many radians of the commanded pose first:

```bash
inspect-robots "place the fork on the plate" --policy agent \
  -P model=anthropic/claude-opus-5 \
  -E settle_tolerance=0.05 -E zero_gravity_mode=false
```

Three things to know before turning it on.

**Pick the tolerance from your rig, not from this example.** Run
`inspect-robots-yam-holdcheck` and use a value comfortably above the settle
figure it reports. A tolerance at or below your rig's steady-state control offset
can never be met, so the first `settle_timeout_budget` steps each burn
`settle_timeout_s` before settling disables itself for that trial.

**Take that figure in the mode you will run, and expect
`zero_gravity_mode=false`.** Settling presumes a servo that holds position. The
default gravity-compensated mode is compliant and may drift instead of holding.

**It guarantees the arm reached what was commanded, not what the policy asked
for.** In `eef_pos` mode an oscillation hold, a failed IK solve, or the per-step
rate clamp all re-send the previous pose, and settling against that succeeds
immediately. Commands are also clamped to `joint_low/high`, which can sit outside
the reachable range.

Timeouts are not failures: the step observes anyway and records
`settled`/`settle_residual`/`settle_timeouts` in `StepResult.info`. After
`settle_timeout_budget` timeouts in a trial, settling switches off for the rest
of that trial, warns, and marks every later step with `settle_disabled`. A scorer
that judges the final state should check for it. Those per-step values reach
scorers and custom sinks; they are not written to the JSON eval log.

> [!NOTE]
> Settling fixes *when* the frame is asked for; the builtin reader's drain
> threads fix *which* frame comes back (#63). Both are needed, and both are in
> place from v0.14.0. A custom `camera_reader` that reads a V4L2 device on
> demand still hands back whatever the driver queued earlier, so a settled arm
> can be photographed mid-motion however tight the tolerance.

The operator status line reads its elapsed time from the wall clock, so it stays
true with settling on. The `Max ~...s` horizon is still a step budget divided by
`control_hz`, which is why it is printed with a leading tilde: remaining step
duration is not knowable in advance, and settling makes steps run long (#64).

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
