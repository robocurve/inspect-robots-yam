# Standalone YAM MuJoCo viewer

Preview LLM `move_to` arguments using this repo's Cartesian action contract.
The viewer opens two YAM arms with parallel grippers. It needs no robot, CAN
adapter, cameras, LLM credentials, or `/act` server.

From this checkout on macOS:

```bash
uv sync --locked --extra viewer
uv run --extra viewer python scripts/mujoco_viewer.py
```

The same command works on Linux. On macOS the script automatically re-launches
through MuJoCo's `mjpython` launcher and resolves the Python library path for uv
virtual environments. This handles the upstream
[`mjpython`/uv library lookup issue](https://github.com/google-deepmind/mujoco/issues/1923).

Paste one JSON object per terminal line:

```json
{"targets":{"left_x":0.34,"left_z":0.25,"right_gripper":0.2}}
```

This is the agent's `move_to` argument format. Omitted dimensions hold their
current observed value. The optional `note` field is accepted, and full tool
calls with `function.name = "move_to"` and JSON `function.arguments` are accepted
too. Bounds violations, unknown labels, and commands to pinned axes are rejected
by the same agent tool code used in an eval. Wait for a move to finish before
pasting another, or use `reset` to cancel it.

## Coordinates

Each arm takes `x, y, z, yaw, pitch, roll, gripper`, with the `left_` or
`right_` prefix. The 14-D ordering is left first, then right.

| Values | Meaning |
| --- | --- |
| x, y, z | Absolute grasp-point position in metres in that arm's own base frame |
| yaw, pitch, roll | Radians relative to that arm's orientation captured at reset |
| gripper | Normalized opening: 0 closed, 1 open |

Base axes are drawn with X red, Y green, and Z blue. +x is forward out of the
base, +y is the arm's left, and +z is up. Colored target markers show the final
requested grasp poses; the arms show the motion toward them.

The orientation convention is the repo's
`Rz(yaw) @ Ry(-pitch) @ Rx(roll) @ R_reset`, expressed in the arm base frame.
Positive pitch tips the tool forward; positive roll tips it left at yaw zero.
Yaw interpolates numerically without wrapping across ±π.

Pitch and roll stay pinned to zero by default, just like the repo. To preview
the supported expanded orientation bounds:

```bash
uv run --extra viewer python scripts/mujoco_viewer.py --eef-orientation
```

## Playback

Playback starts at 0.25x. Terminal commands are:

| Command | Effect |
| --- | --- |
| `pause` / `play` | Pause or resume playback |
| `next` | Pause and show the next control-command endpoint |
| `speed 0.1` | Display at one tenth speed |
| `reset` | Cancel the current move, restore start joints, recapture orientation zero |
| `state` | Print achieved EEF state, joints, and position error |
| `quit` | Close the viewer |

With the window focused, Space toggles pause, N steps, R resets, and Q quits.
Use the mouse to orbit, zoom, and pan with the native MuJoCo viewer controls.

`--speed` changes only the display clock. It does not change `control_hz`,
agent interpolation, or the per-step IK clamp. Smooth frames interpolate joint
commands for display; intermediate display frames are not extra policy actions.
The printed state reflects the latest computed command endpoint, which may be
ahead of the display while that command is animating. A completion report means
the action chunk finished; inspect the achieved state and residual to see how
closely the requested target was reached. Unreachable targets are not retried
indefinitely.

Replay the supplied sequence, or make your own JSONL file:

```bash
uv run --extra viewer python scripts/mujoco_viewer.py --commands scripts/yam_viewer_demo.jsonl
uv run --extra viewer python scripts/mujoco_viewer.py --headless --commands scripts/yam_viewer_demo.jsonl
```

Headless replay executes the same commands without wall-clock delays or rendering.

## Match a rig

Load the same Inspect Robots config used by the LLM run:

```bash
uv run --extra viewer python scripts/mujoco_viewer.py --config /path/to/config.ini
```

The config must select `embodiment = yam_arms` in `[defaults]` and
`control_interface = eef_pos` in `[embodiment.args]`. The viewer preserves motion
settings, including workspace bounds, IK limits, gripper calibration, home or
named start pose, and the agent's `max_speed_frac`. Pass an absolute `pose_dir`
when replaying a rig's named start poses from a different working directory.
Camera and hardware I/O, temperature checks, operator prompts, and physical
settling are replaced by an ideal in-memory driver.

Display placement uses the repo's `collision_left_base_pos`,
`collision_right_base_pos`, `collision_left_base_yaw`,
`collision_right_base_yaw`, and table settings. For example, a right base yaw
of π rotates that displayed arm and its local axes by 180 degrees. It does not
change what a right-arm target means. Default placement is illustrative: two
parallel bases 0.6 m apart. Load measured transforms to match your rig.

## What the preview models

The viewer uses the YAM + LINEAR_4310 model from the repo's pinned i2rt revision,
Mink with the same raw QP settings, the existing `_ArmKinematics` guards, and
`YAMEmbodiment.step()` with an injected ideal driver. The agent package generates
the action chunk, so partial targets and interpolation use its installed version.
Its private tool API is covered by a parity test.

This is a kinematic preview of commanded motion. It does not simulate motor
lag, gravity, friction, grasping, or contact avoidance. The current repo's
collision approver only handles absolute joint mode, so it is not applied to
EEF moves here either. LINEAR_4310 is the only bundled gripper; other gripper
types are rejected instead of displayed with mismatched geometry.

The [model provenance and license](assets/yam_viewer/README.md) document the
vendored visual assets. No `yam-ik` checkout or hardware driver installation is
required.
