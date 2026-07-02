<div align="center">

# 🦾 inspect-robots-yam

**Run [Inspect Robots](https://github.com/robocurve/inspect-robots) evals on real
[I2RT YAM](https://i2rt.com/products/yam-6-dof-arm) bimanual arms driven by
[MolmoAct2](https://github.com/allenai/molmoact2).**

[![CI](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/robocurve/inspect-robots-yam/actions/workflows/ci.yml)
[![Built on Inspect Robots](https://img.shields.io/badge/built%20on-Inspect Robots-indigo)](https://github.com/robocurve/inspect-robots)

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

## Install (on the robot/GPU machine)

```bash
# Inspect Robots isn't on PyPI yet; uv resolves it (and the optional i2rt driver) from git.
uv pip install "inspect-robots-yam[client,yam] @ git+https://github.com/robocurve/inspect-robots-yam"
```

- `client` → `requests` + `json-numpy` (the `/act` transport).
- `yam` → the I2RT `i2rt` driver (GitHub-only).

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

You must provide a `camera_reader` (there is no universal camera API) returning
`{"top_cam", "left_cam", "right_cam": HxWx3 uint8}`. From Python:

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
Unattended runs simply run to `max_steps` and score as failures.

## Safety

- **Hard clamp backstop.** Every command is clipped to `YamConfig.joint_low/high`
  *inside* `step()`, independent of any Inspect Robots `Approver` — unclamped model
  outputs can never reach the motors. **Set these to your real YAM joint limits**
  (the defaults are conservative placeholders: joints ±π, gripper 0–1).
- **Use `ClampApprover`** on hardware for a second layer.
- **Absolute vs. delta joints — verify first.** MolmoAct2's YAM `actions` are
  treated as **absolute** joint targets by default. If your checkpoint emits
  deltas, set `YamConfig(joints_are_delta=True)` (the embodiment converts to
  absolute internally so the declared `joint_pos` stays honest). Inspect Robots's
  compat check *cannot* tell these apart — confirm with `--dry-run` and a single
  slow jog before running a task.
- **Gripper calibration.** Map MolmoAct2's normalized gripper to your hardware via
  `YamConfig(gripper_open=..., gripper_closed=...)`.

## Configuration

`YamConfig`: `left_channel`, `right_channel`, `gripper_type`, `control_hz`,
`cam_height/width`, `joint_low/high`, `home_pose`, `gripper_open/closed`,
`joints_are_delta`.
`MolmoActConfig`: `server_url`, `endpoint`, `num_steps`, `timeout_s`,
`camera_order`, `state_key`, `cam_height/width`.

Scalar knobs are settable from the CLI:
`inspect-robots run -P server_url=http://gpu:8202 -E left_channel=can0 ...`.

## Development

```bash
uv venv && uv pip install -e ".[dev]"     # inspect_robots + kitchenbench from git tags
uv run pre-commit install
uv run pytest --cov                        # 100% coverage required
uv run ruff check . && uv run mypy
```

The whole suite runs with **no hardware, no server, and no stdin** — the i2rt
driver, cameras, the `/act` transport, the clock, and operator I/O are all
injected. The default hardware seams are excluded from coverage (`# pragma: no
cover`).

## License

[MIT](LICENSE)
