<div align="center">

# 🦾 inspect-robots-yam

**Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on real
[I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) bimanual arms driven by
[MolmoAct2](https://github.com/allenai/molmoact2).**

[![CI](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inspect-robots-yam)](https://pypi.org/project/inspect-robots-yam/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect%20Robots-indigo)](https://github.com/robocurve/inspect-robots)

</div>

Inspect Robots has **two** swappable inputs: a `Policy` (the VLA brain) and an
`Embodiment` (the robot body + world). This package provides both for the
YAM + MolmoAct2 stack, so any embodiment-agnostic Inspect Robots task — e.g. all of
[KitchenBench](https://github.com/robocurve/kitchenbench) — runs on real arms:

- **`molmoact2` policy** — a thin client for MolmoAct2's first-party bimanual-YAM
  `/act` server (the model owns the GPU + weights in its own process).
- **`yam_arms` embodiment** — the I2RT joint-position driver, with a hard safety
  clamp, operator-in-the-loop success, and self-paced control.

Both declare the **same 14-D joint-position contract** (2 arms × [6 joints +
gripper], cameras `top/left/right`, packed `joint_pos` state), so Inspect Robots's
compatibility check passes with **zero errors and zero warnings** — verifiable
before any motion.

```bash
inspect-robots run --task kitchenbench/pour_pasta --policy molmoact2 --embodiment yam_arms
```

> **Note:** the CLI forwards scalar `key=value` knobs only — it cannot inject a
> `camera_reader`, which hardware runs require. Launch from Python (see *Run on
> hardware*) or register your own entry-point factory that bundles the cameras;
> otherwise `yam_arms` fails fast with a `ConfigError` at `reset()`, before any
> driver connect or motion.

## Install (on the robot/GPU machine)

```bash
uv pip install "inspect-robots-yam[client]"
# The i2rt driver is GitHub-only (not on PyPI), so the [yam] hardware extra
# can't resolve from PyPI — install the driver directly instead:
uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"
```

- `client` → `requests` + `json-numpy` (the `/act` transport).
- `i2rt` → the I2RT YAM arm driver, required for real hardware (the `[yam]`
  extra declares it, but only resolves in a git/dev install where
  `[tool.uv.sources]` applies — from PyPI, install it directly as above).

Then download the model weights (needs a Hugging Face token) and start the server,
from the [MolmoAct2 repo](https://github.com/allenai/molmoact2):

```bash
huggingface-cli download allenai/MolmoAct2-BimanualYAM
python examples/yam/host_server_yam.py          # serves /act on :8202
```

## Preflight — *prove compatibility before any motion*

```bash
inspect-robots-yam-preflight                                  # dims/semantics/cameras/state
inspect-robots-yam-preflight --task kitchenbench/pour_pasta   # + scene realizability
inspect-robots-yam-preflight --dry-run                        # affirm no motion
```

A green preflight means action dim (14), control mode (`joint_pos`), cameras, and
state keys all line up. **It does not prove the joint values are interpreted the
same way** — see *Safety* below.

## Run on hardware

You must provide a `camera_reader` (there is no universal camera API). It is a
`Callable[[YamConfig], dict]` — called once per step with the config — returning
`{"top_cam", "left_cam", "right_cam": HxWx3 uint8}` at the config's
`cam_height`×`cam_width`. Here is a concrete reader for Intel RealSense cameras
over V4L2/OpenCV that opens each device once and reuses the handle:

```python
import cv2
import numpy as np
from inspect_robots_yam import YamConfig

def make_realsense_reader(devices: dict[str, str]):
    """devices maps cam name -> V4L2 path, e.g.
    {"top_cam": "/dev/video0", "left_cam": "/dev/video6", "right_cam": "/dev/video12"}."""
    caps: dict[str, cv2.VideoCapture] = {}

    def reader(cfg: YamConfig) -> dict[str, np.ndarray]:
        for name, dev in devices.items():
            caps.setdefault(name, cv2.VideoCapture(dev, cv2.CAP_V4L2))
        frames: dict[str, np.ndarray] = {}
        for name, cap in caps.items():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"camera read failed: {name} ({devices[name]})")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames[name] = cv2.resize(rgb, (cfg.cam_width, cfg.cam_height)).astype(np.uint8)
        return frames

    return reader
```

> RealSense `/dev/videoN` numbers reshuffle on replug; pin each camera with a
> `udev` rule keyed on its firmware serial so the paths above stay stable.

Then wire it up and run:

```python
from inspect_robots import eval
from inspect_robots.approver import ClampApprover
from inspect_robots_yam import MolmoAct2Policy, YAMEmbodiment, YamConfig

emb = YAMEmbodiment(YamConfig(left_channel="can0", right_channel="can1"),
                    camera_reader=make_realsense_reader({...}))
pol = MolmoAct2Policy(server_url="http://127.0.0.1:8202")

(log,) = eval("kitchenbench/pour_pasta", pol, emb,
              approver=ClampApprover(emb.info.action_space))  # defense in depth
print(log.status, log.results.metrics)
```

> **Validate before you move.** A run with a dead `/act` server or a mis-wired
> camera should fail at setup, not mid-motion. Before the first real run, confirm
> the reader returns three `(cam_height, cam_width, 3)` `uint8` frames and that
> the server answers at `server_url` — then keep [preflight](#preflight--prove-compatibility-before-any-motion)
> green and clear the workspace.

At each episode end the embodiment asks the operator (y/N); a `yes` records
`termination_reason="success"`, which KitchenBench's `task_success` scorer reads.
The operator prompts need an interactive terminal — a dead stdin raises
`EmbodimentFault` (the framework's always-halt path). For runs with no operator,
set `YamConfig(unattended=True)` (CLI: `-E unattended=true`): all operator
prompts are skipped and every episode runs to `max_steps`, scoring as a failure.

## Safety

- **Hard clamp backstop.** Every command is clipped to `YamConfig.joint_low/high`
  *inside* `step()`, independent of any Inspect Robots `Approver` — unclamped model
  outputs can never reach the motors. **Set the arm slots to your real YAM joint
  limits** (the defaults are conservative placeholders: joints ±π, gripper 0–1) —
  but note the limits are in *policy units* per the table below: gripper slots 6
  and 13 stay normalized 0–1, only slots 0–5 and 7–12 are radians.
- **Use `ClampApprover`** on hardware for a second layer.
- **Zero-gravity handoff jump.** The arms connect in zero-gravity mode by default
  (`YamConfig(zero_gravity_mode=True)`, passed through to the i2rt driver), so the
  first stiff PD command — homing or the first action — can jump from wherever the
  arm was idling. Nothing bounds the per-step joint delta yet (tracked as a known
  issue); stand clear at `reset()` and prefer a `home_pose` near the resting pose.
- **Absolute vs. delta joints — verify first.** MolmoAct2's YAM `actions` are
  treated as **absolute** joint targets by default. If your checkpoint emits
  deltas, set `YamConfig(joints_are_delta=True)` (the embodiment converts to
  absolute internally so the declared `joint_pos` stays honest). Inspect Robots's
  compat check *cannot* tell these apart — confirm with `--dry-run` and a single
  slow jog before running a task.
- **Gripper polarity/trim.** The i2rt driver already exposes the YAM gripper as
  normalized 0–1 in both directions, so the defaults (`gripper_open=0.0`,
  `gripper_closed=1.0`) are an identity map and correct for standard grippers.
  `YamConfig(gripper_open=..., gripper_closed=...)` is a polarity/trim remap over
  that already-normalized range — its main use is a gripper wired with inverted
  polarity (`gripper_open=1.0, gripper_closed=0.0`). The remap is a bijection:
  commands are de-normalized on the way out and observations are re-normalized on
  the way back, so the model always sees 0–1. **Warning:** values outside [0, 1]
  are forwarded on a path i2rt does *not* clip — avoid them unless you have
  verified your firmware's behavior.

## Configuration

### Units — every 14-D vector uses the same layout

`joint_low`/`joint_high`, `home_pose`, actions, and the observed `joint_pos`
state all use *policy units*:

| Slots | Meaning | Unit |
|-------|---------|------|
| 0–5, 7–12 | left / right arm revolute joints | radians |
| 6, 13 | left / right gripper | normalized 0–1 (0 = open, 1 = closed) |

Hardware gripper units (via `gripper_open`/`gripper_closed`) exist only at the
driver boundary; nothing you configure here is in hardware gripper units.

`YamConfig`: `left_channel`, `right_channel`, `gripper_type` (i2rt `GripperType`
enum *name*, e.g. `LINEAR_4310`; grippers only — `NO_GRIPPER`/`YAM_TEACHING_HANDLE`
would break the 14-D packing and are rejected), `control_hz`, `cam_height/width`,
`joint_low/high`, `home_pose`, `gripper_open/closed`, `joints_are_delta`,
`zero_gravity_mode` (default `True`; see *Safety*), `unattended` (default `False`;
skip operator prompts).
`MolmoActConfig`: `server_url`, `endpoint`, `num_steps`, `timeout_s`,
`camera_order`, `state_key`, `cam_height/width`.

Scalar knobs are settable from the CLI:
`inspect-robots run -P server_url=http://gpu:8202 -E left_channel=can0 ...`.

## Development

> **Dependency changes:** after editing dependencies in `pyproject.toml`, run
> `uv lock` and commit the updated lockfile — CI installs with
> `uv sync --locked` and fails with "the lockfile needs to be updated" if you
> forget. Day-to-day conventions (PR-only `main`, the required `ci-ok` check,
> one-click releases) are documented in [`CLAUDE.md`](CLAUDE.md).

```bash
uv venv && uv pip install -e ".[dev]"     # inspect_robots + kitchenbench from PyPI
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

The whole suite runs with **no hardware, no server, and no stdin** — the i2rt
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
