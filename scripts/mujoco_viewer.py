"""Preview this repo's EEF move_to commands on two simulated YAM arms.

uv run --extra viewer python scripts/mujoco_viewer.py

On macOS this re-launches through mjpython with uv's Python library resolved.

Paste one JSON object per line: {"targets": {"left_x": 0.34, "right_gripper": 0.2}}.
The terminal accepts pause, play, next, reset, state, speed 0.1, and quit.
Window shortcuts: Space pause/play, N next command, R reset, Q quit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from inspect_robots.defaults import load_defaults
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from inspect_robots_yam import packing
from inspect_robots_yam.collision import _config_from_yam
from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, EEF_DIM_LABELS, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.kinematics import _ArmKinematics

MODEL_PATH = Path(__file__).parent / "assets" / "yam_viewer" / "yam.xml"
_SIDES = ("left", "right")


class PreviewKinematics:
    """Use the pinned i2rt model and its Mink QP settings without importing CAN code.

    Matches i2rt ac096928's Kinematics.ik defaults: FrameTask at grasp_site,
    unit position/orientation costs, lm_damping=1, quadprog, dt=.01,
    damping=1e-4, and 1e-4 position/orientation convergence thresholds.
    """

    def __init__(self) -> None:
        import mink
        import mujoco

        self.mink = mink
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.configuration = mink.Configuration(self.model)

    def get_joint_ranges(self) -> np.ndarray:
        """Return the model's six arm and two finger ranges."""
        return self.model.jnt_range.copy()

    def set_joint_ranges(self, ranges: np.ndarray) -> None:
        """Apply the embodiment's model/config range intersection."""
        self.model.jnt_range[:] = ranges

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Return the grasp point in this arm's base frame."""
        self.configuration.update(q)
        return self.configuration.get_transform_frame_to_world("grasp_site", "site").as_matrix()

    def ik(self, target: np.ndarray, init_q: np.ndarray, max_iters: int) -> tuple[bool, np.ndarray]:
        """Return the finite last iterate, leaving command guards to the embodiment."""
        self.configuration.update(init_q)
        task = self.mink.FrameTask(
            frame_name="grasp_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        task.set_target(self.mink.SE3.from_matrix(target))
        for _ in range(max_iters):
            velocity = self.mink.solve_ik(
                self.configuration,
                [task],
                0.01,
                "quadprog",
                damping=1e-4,
            )
            self.configuration.integrate_inplace(velocity, 0.01)
            error = task.compute_error(self.configuration)
            if np.linalg.norm(error[:3]) <= 1e-4 and np.linalg.norm(error[3:]) <= 1e-4:
                return True, self.configuration.q.copy()
        return False, self.configuration.q.copy()


class PreviewDriver:
    """Instantly track commanded joints in memory, including native gripper units."""

    def __init__(self, config: YamConfig) -> None:
        self.state = packing.denorm_grippers(
            DEFAULT_EEF_HOME_POSE,
            gripper_open=config.gripper_open,
            gripper_closed=config.gripper_closed,
        )

    def get_joint_pos(self) -> np.ndarray:
        """Return an independent copy of the ideal measured state."""
        return self.state.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        """Record a driver-native command without opening any hardware."""
        self.state = target.copy()

    def close(self) -> None:
        """Release the in-memory driver; there are no external resources."""


def preview_config(config: YamConfig) -> YamConfig:
    """Keep motion settings while removing hardware, camera, and operator I/O."""
    if config.control_interface != "eef_pos" or config.gripper_type != "LINEAR_4310":
        raise ValueError("viewer requires control_interface=eef_pos and gripper_type=LINEAR_4310")
    if config.control_hz <= 0:
        raise ValueError("viewer requires a positive control_hz")
    return replace(
        config,
        unattended=True,
        auto_start=False,
        rest_secs=1.0 / config.control_hz,
        settle_tolerance=None,
        motor_temp_limit=None,
        report_joint_eff=False,
        top_cam_device=None,
        left_cam_device=None,
        right_cam_device=None,
        top_depth_serial=None,
        left_depth_serial=None,
        right_depth_serial=None,
    )


class Preview:
    """Run the real agent interpolation and YAM step pipeline with an ideal driver."""

    def __init__(self, config: YamConfig, max_speed_frac: float = 0.1) -> None:
        from inspect_robots_agent._tools import build_toolset

        self.config = preview_config(config)
        self.raw = (PreviewKinematics(), PreviewKinematics())
        self.driver = PreviewDriver(self.config)
        self.embodiment = YAMEmbodiment(
            self.config,
            driver_factory=lambda _: self.driver,
            kinematics_factory=lambda _: self.raw,
            camera_reader=lambda _: {},
            sleep_fn=lambda _: None,
            clock=lambda: 0.0,
            status_fn=lambda _: None,
        )
        info = self.embodiment.info
        self.tools = build_toolset(
            info.action_space,
            info.observation_space,
            self.config.control_hz,
            max_speed_frac=max_speed_frac,
        )
        self.pending: deque[Action] = deque()
        self.reset()

    def reset(self) -> None:
        """Reset both arms to the configured start pose and recapture orientation zero."""
        self.pending.clear()
        self.observation = self.embodiment.reset(
            Scene(id="mujoco-preview", instruction="Preview EEF targets")
        )
        self.target = self.eef.copy()
        self.references = [
            raw.fk(np.r_[self.joints[offset : offset + 6], [0.02375, 0.02375]])[:3, :3]
            for raw, offset in zip(self.raw, (0, 7), strict=True)
        ]

    @property
    def joints(self) -> np.ndarray:
        """Return the current command in the repo's normalized 14-D joint contract."""
        return np.asarray(self.observation.state["joint_pos"]).copy()

    @property
    def eef(self) -> np.ndarray:
        """Return observed grasp positions and reset-relative orientations."""
        return np.asarray(self.observation.state["eef_state"]).copy()

    def submit(self, payload: Any) -> int:
        """Accept move_to arguments or its full tool call; reject overlapping moves."""
        from inspect_robots_agent._llm import ToolCall

        if self.pending:
            raise ValueError("motion in progress; wait for completion or reset first")
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object with targets")
        name = "move_to"
        arguments = payload
        if "function" in payload:
            function = payload["function"]
            if not isinstance(function, dict):
                raise ValueError("function must be an object")
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
        if name != "move_to" or not isinstance(arguments, dict):
            raise ValueError("expected move_to arguments with targets")
        arguments = {"note": "Manual MuJoCo pose preview", **arguments}
        result = self.tools.execute(
            ToolCall(id="preview", name=name, arguments=json.dumps(arguments)),
            self.observation,
        )
        if result.error:
            raise ValueError(result.error)
        assert result.chunk is not None and result.target is not None
        self.pending.extend(result.chunk.actions)
        self.target = result.target.copy()
        return len(self.pending)

    def step(self) -> None:
        """Execute exactly one interpolated agent action through YAMEmbodiment.step."""
        self.observation = self.embodiment.step(self.pending.popleft()).observation

    def report(self) -> dict[str, Any]:
        """Report achieved state and errors without claiming an unreachable target arrived."""
        return {
            "eef_state": dict(zip(EEF_DIM_LABELS, self.eef.tolist(), strict=True)),
            "joint_pos": self.joints.tolist(),
            "position_error_m": {
                side: float(
                    np.linalg.norm(self.target[start : start + 3] - self.eef[start : start + 3])
                )
                for side, start in zip(_SIDES, (0, 7), strict=True)
            },
            "remaining_actions": len(self.pending),
        }


class ViewerScene:
    """Compose two visual models; base transforms affect drawing, never IK inputs."""

    def __init__(self, preview: Preview) -> None:
        import mujoco

        self.mj = mujoco
        self.preview = preview
        geometry = _config_from_yam(preview.config)
        parent = mujoco.MjSpec()
        parent.worldbody.add_light(pos=(0.3, 0, 2), dir=(0, 0, -1))
        if geometry.table_height is not None:
            parent.worldbody.add_geom(
                name="table",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=(1.5, 1.5, 0.01),
                pos=(0, 0, geometry.table_height),
                rgba=(0.19, 0.21, 0.24, 1),
            )
        self.bases = []
        colors = ((0.27, 0.59, 0.88, 1), (0.93, 0.65, 0.24, 1))
        for side, color in zip(_SIDES, colors, strict=True):
            position = getattr(geometry, f"{side}_base_pos")
            yaw = getattr(geometry, f"{side}_base_yaw")
            rotation = _ArmKinematics._rotation_z(yaw)
            self.bases.append((np.asarray(position), rotation))
            frame = parent.worldbody.add_frame(
                pos=position,
                quat=(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)),
            )
            child = mujoco.MjSpec.from_file(str(MODEL_PATH))
            for geom in child.geoms:
                geom.rgba = color
                geom.contype = geom.conaffinity = 0
            parent.attach(child, frame=frame, prefix=f"{side}_")
            # Axes are expressed in each base, including a mirrored mounting yaw.
            for axis, axis_color in enumerate(
                ((1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.2, 0.4, 1, 1))
            ):
                end = np.asarray(position) + rotation[:, axis] * 0.13
                parent.worldbody.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=(0.003, 0, 0),
                    fromto=(*position, *end),
                    rgba=axis_color,
                    contype=0,
                    conaffinity=0,
                )
            marker = parent.worldbody.add_body(name=f"{side}_target", mocap=True)
            marker.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=(0.009, 0, 0),
                rgba=color,
                contype=0,
                conaffinity=0,
            )
            for axis, axis_color in enumerate(
                ((1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.2, 0.4, 1, 1))
            ):
                marker.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=(0.002, 0, 0),
                    fromto=(0, 0, 0, *(np.eye(3)[axis] * 0.06)),
                    rgba=axis_color,
                    contype=0,
                    conaffinity=0,
                )
        self.model = parent.compile()
        self.data = mujoco.MjData(self.model)
        self.draw(preview.joints)

    def draw(self, joints: np.ndarray) -> None:
        """Set a visual pose and final goal markers, then run forward kinematics only."""
        for arm, (side, offset) in enumerate(zip(_SIDES, (0, 7), strict=True)):
            for index in range(6):
                joint = self.model.joint(f"{side}_joint{index + 1}")
                self.data.qpos[joint.qposadr[0]] = joints[offset + index]
            # In the pinned LINEAR_4310 model q=0 is open; both slides close
            # toward +0.0475 m. The public wire convention runs the other way.
            for index in (7, 8):
                joint = self.model.joint(f"{side}_joint{index}")
                self.data.qpos[joint.qposadr[0]] = (1 - joints[offset + 6]) * 0.0475
            target = self.preview.target[offset : offset + 7]
            position, base_rotation = self.bases[arm]
            rotation = (
                _ArmKinematics._rotation_z(target[3])
                @ _ArmKinematics._rotation_y_forward(target[4])
                @ _ArmKinematics._rotation_x(target[5])
                @ self.preview.references[arm]
            )
            mocap = self.model.body(f"{side}_target").mocapid[0]
            self.data.mocap_pos[mocap] = position + base_rotation @ target[:3]
            self.mj.mju_mat2Quat(self.data.mocap_quat[mocap], (base_rotation @ rotation).ravel())
        self.mj.mj_forward(self.model, self.data)


class Playback:
    """Slow the display clock without changing the agent or IK control rate."""

    def __init__(self, preview: Preview, speed: float = 0.25) -> None:
        self.preview = preview
        self.set_speed(speed)
        self.paused = False
        self.reset_display()

    def set_speed(self, speed: float) -> None:
        """Set a finite playback multiplier from 0.01 through 4."""
        if not math.isfinite(speed) or not 0.01 <= speed <= 4:
            raise ValueError("speed must be between 0.01 and 4")
        self.speed = speed

    def reset_display(self) -> None:
        """Discard visual interpolation after resetting the simulated robot."""
        self.start = self.end = self.shown = self.preview.joints
        self.elapsed = 0.0
        self.animating = False

    @property
    def busy(self) -> bool:
        """Include the final command's unfinished display interpolation."""
        return self.animating or bool(self.preview.pending)

    def advance(self, wall_dt: float, *, single: bool = False) -> np.ndarray:
        """Animate command endpoints, or expose exactly the next endpoint when stepping."""
        if self.paused and not single:
            return self.shown
        if not self.animating and self.preview.pending:
            self.start = self.shown.copy()
            self.preview.step()
            self.end = self.preview.joints
            self.elapsed = 0.0
            self.animating = True
        if self.animating:
            self.elapsed += wall_dt * self.speed
            fraction = 1.0 if single else min(1.0, self.elapsed * self.preview.config.control_hz)
            self.shown = self.start + fraction * (self.end - self.start)
            if fraction == 1.0:
                self.animating = False
        return self.shown


def read_config(path: Path | None, orientation: bool) -> tuple[YamConfig, float]:
    """Load an explicit rig config; default to the repo's EEF settings otherwise."""
    values: dict[str, Any] = {"control_interface": "eef_pos"}
    speed_frac = 0.1
    if path is not None:
        if not path.is_file():
            raise ValueError(f"config does not exist: {path}")
        defaults = load_defaults({"INSPECT_ROBOTS_CONFIG": str(path.resolve())})
        if defaults.embodiment_args_owner != "yam_arms":
            raise ValueError("config [defaults] must set embodiment = yam_arms")
        values.update(defaults.embodiment_args)
        if defaults.policy_args_owner == "agent":
            speed_frac = float(defaults.policy_args.get("max_speed_frac", 0.1))
    if orientation:
        values["eef_orientation"] = True
    return YamConfig.from_kwargs(**values), speed_frac


def handle_command(line: str, player: Playback) -> str:
    """Apply one terminal command; invalid targets leave the queued motion unchanged."""
    command = line.strip()
    preview = player.preview
    if command in ("pause", "play", "toggle"):
        player.paused = not player.paused if command == "toggle" else command == "pause"
    elif command == "next":
        player.paused = True
        player.advance(0, single=True)
    elif command == "reset":
        preview.reset()
        player.reset_display()
    elif command.startswith("speed "):
        player.set_speed(float(command.split(maxsplit=1)[1]))
    elif command == "state":
        return json.dumps(preview.report())
    elif command == "quit":
        return "quit"
    elif command:
        if player.busy:
            raise ValueError("motion in progress; wait for completion or reset first")
        steps = preview.submit(json.loads(command))
        return f"Queued {steps} actions ({steps / preview.config.control_hz:.2f}s at 1x)."
    return f"{'Paused' if player.paused else 'Playing'} at {player.speed:g}x."


def run_window(
    preview: Preview,
    player: Playback,
    commands: queue.Queue[str],
    moves: deque[Any],
) -> None:
    """Keep the native viewer responsive while terminal input arrives independently."""
    import mujoco.viewer

    scene = ViewerScene(preview)
    keys = {32: "toggle", ord("N"): "next", ord("R"): "reset", ord("Q"): "quit"}
    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        key_callback=lambda key: commands.put(keys[key]) if key in keys else None,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (0.22, 0, 0.18)
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -28
        last = time.monotonic()
        was_busy = player.busy
        while viewer.is_running():
            now = time.monotonic()
            dt, last = now - last, now
            if moves and not player.busy:
                commands.put(json.dumps(moves.popleft()))
            while not commands.empty():
                try:
                    message = handle_command(commands.get_nowait(), player)
                    if message == "quit":
                        return
                    print(message, flush=True)
                except (ValueError, TypeError) as exc:
                    print(f"Invalid command: {exc}", flush=True)
            was_busy = was_busy or player.busy
            player.advance(min(dt, 0.1))
            with viewer.lock():
                scene.draw(player.shown)
            viewer.sync()
            if was_busy and not player.busy:
                print(json.dumps(preview.report()), flush=True)
                was_busy = False
            time.sleep(1 / 60)


def main(argv: list[str] | None = None) -> None:
    """Launch an interactive viewer, or replay JSONL commands without a window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="explicit Inspect Robots rig config.ini")
    parser.add_argument("--eef-orientation", action="store_true", help="open pitch/roll bounds")
    parser.add_argument("--speed", type=float, default=0.25, help="display speed, default 0.25x")
    parser.add_argument("--commands", type=Path, help="JSONL move_to arguments, one move per line")
    parser.add_argument(
        "--headless", action="store_true", help="replay JSONL and print final states"
    )
    args = parser.parse_args(argv)
    if args.headless and args.commands is None:
        parser.error("--headless requires --commands")
    if not args.headless:
        launch_macos(sys.argv[1:] if argv is None else argv)
    config, max_speed_frac = read_config(args.config, args.eef_orientation)
    preview = Preview(config, max_speed_frac)
    player = Playback(preview, args.speed)
    print("YAM EEF preview: blue left, amber right. Base axes: X red, Y green, Z blue.")
    print("Absolute base-frame metres; reset-relative radians; gripper 0 closed / 1 open.")
    print("Ideal command tracking, no dynamics/contact avoidance. Base placement uses collision_*.")
    if config.collision_left_base_pos is None or config.collision_right_base_pos is None:
        print("Base placement is illustrative: 0.6m apart, parallel. Load measured rig geometry.")
    print("Pinned axes:", ", ".join(config.pinned_orientation_labels()) or "none")
    print(json.dumps(preview.report()), flush=True)
    moves = (
        []
        if args.commands is None
        else [json.loads(line) for line in args.commands.read_text().splitlines() if line.strip()]
    )
    if args.headless:
        for payload in moves:
            preview.submit(payload)
            while preview.pending:
                preview.step()
            print(json.dumps(preview.report()), flush=True)
        return
    commands: queue.Queue[str] = queue.Queue()
    print('Paste {"targets": {"left_x": 0.34, "right_gripper": 0.2}}')
    print("Commands: pause, play, next, reset, state, speed 0.1, quit. Keys: Space / N / R / Q.")

    def read_stdin() -> None:
        for line in sys.stdin:
            commands.put(line)

    threading.Thread(target=read_stdin, daemon=True).start()
    with suppress(KeyboardInterrupt):
        run_window(preview, player, commands, deque(moves))


def launch_macos(arguments: list[str]) -> None:
    """Re-exec via MuJoCo's Cocoa launcher, resolving uv's symlinked libpython."""
    if sys.platform != "darwin":
        return
    import mujoco.viewer

    if mujoco.viewer._MJPYTHON is not None:
        return
    launcher = Path(sys.executable).with_name("mjpython")
    env = os.environ.copy()
    # mjpython resolves @executable_path relative to the venv symlink rather
    # than uv's real Python binary (MuJoCo issue #1923). Supply its library
    # directory to dyld, retaining any user-provided fallback search paths.
    existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
    env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{Path(sys.base_prefix) / 'lib'}:{existing}"
    os.execve(str(launcher), [str(launcher), str(Path(__file__).resolve()), *arguments], env)


if __name__ == "__main__":
    main()
