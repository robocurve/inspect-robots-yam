"""``YAMEmbodiment`` — Inspect Robots embodiment for I2RT YAM bimanual arms.

Wraps the i2rt joint-position driver. Designed for real-robot reality:

* **Safety backstop** — every command is clamped to the configured joint limits
  inside :meth:`step`, *independently* of any Inspect Robots ``Approver`` (so unclamped
  model outputs can never reach the motors).
* **Operator-in-the-loop success** — there is no privileged oracle; when the
  operator signals end-of-episode the embodiment returns
  ``StepResult(terminated=True, termination_reason="success"|"failure")``, which is
  the only path that reaches the scorer.
* **Self-paced** — declares ``SELF_PACED`` and sleeps to the control rate inside
  :meth:`step` (the framework does not pace for us).

Hardware/driver access is injected (``driver_factory``, ``camera_reader``,
``operator``, ``poll_end``, ``sleep_fn``, ``clock``) so the whole embodiment runs
in tests with no CAN bus, no cameras, and no stdin. The real driver/camera seams
are pragma'd defaults that only execute on hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_yam import packing
from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig, action_box, observation_space
from inspect_robots_yam.operator import OperatorIO, default_poll_end

ImageMap = Mapping[str, npt.NDArray[np.uint8]]
Vec = npt.NDArray[np.float64]


@runtime_checkable
class BimanualDriver(Protocol):
    """The minimal 14-D joint-position driver the embodiment needs."""

    def get_joint_pos(self) -> npt.NDArray[np.floating[Any]]: ...

    def command_joint_pos(self, target: npt.NDArray[np.floating[Any]]) -> None: ...

    def close(self) -> None: ...


DriverFactory = Callable[[YamConfig], BimanualDriver]
CameraReader = Callable[[YamConfig], ImageMap]


def _default_driver_factory(cfg: YamConfig) -> BimanualDriver:  # pragma: no cover - real hardware
    from i2rt.robots.get_robot import get_yam_robot

    left = get_yam_robot(channel=cfg.left_channel)
    right = get_yam_robot(channel=cfg.right_channel)

    class _Real:
        def get_joint_pos(self) -> npt.NDArray[np.floating[Any]]:
            return packing.pack(left.get_joint_pos(), right.get_joint_pos())

        def command_joint_pos(self, target: npt.NDArray[np.floating[Any]]) -> None:
            lo, ro = packing.split(target)
            left.command_joint_pos(lo)
            right.command_joint_pos(ro)

        def close(self) -> None:
            for arm in (left, right):
                closer = getattr(arm, "close", None)
                if callable(closer):
                    closer()

    return _Real()


def _default_camera_reader(cfg: YamConfig) -> ImageMap:
    raise NotImplementedError(
        "provide a camera_reader returning {'top_cam','left_cam','right_cam': HxWx3 uint8}"
    )


class YAMEmbodiment:
    """Inspect Robots embodiment for bimanual YAM arms (joint-position control)."""

    def __init__(
        self,
        config: YamConfig | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        camera_reader: CameraReader | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else YamConfig.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._camera_reader: CameraReader = camera_reader or _default_camera_reader
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._clock: Callable[[], float] = clock or time.perf_counter

        self._driver: BimanualDriver | None = None
        self._instruction: str | None = None
        self._t_last = 0.0
        self.num_steps = 0

        self.info = EmbodimentInfo(
            name="yam_arms",
            action_space=action_box(low=self._cfg.low, high=self._cfg.high),
            observation_space=observation_space(
                self._cfg.cam_height, self._cfg.cam_width, DEFAULT_CAMERAS
            ),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
        )

    # -- lifecycle ---------------------------------------------------------

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect (if needed), drive to home, and block on operator readiness."""
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)
        if self._cfg.home_pose is not None:
            self._send(np.asarray(self._cfg.home_pose, dtype=np.float64))
        self._operator.wait_ready()
        self._instruction = scene.instruction
        self.num_steps = 0
        self._t_last = self._clock()
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp + command one action, pace to the control rate, then maybe end."""
        driver = self._require_driver()
        self.num_steps += 1
        cmd = packing.validate_dim(action.data)
        if self._cfg.joints_are_delta:
            cmd = packing.validate_dim(driver.get_joint_pos()) + cmd
        self._send(cmd)
        self._pace()

        obs = self._observe(self._instruction)
        if self._poll_end():
            success = self._operator.confirm_success()
            return StepResult(
                observation=obs,
                terminated=True,
                termination_reason="success" if success else "failure",
                info={"operator_confirmed": success},
            )
        return StepResult(observation=obs, terminated=False)

    def close(self) -> None:
        """Release the driver handles (no-op if never connected)."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # -- internals ---------------------------------------------------------

    def _require_driver(self) -> BimanualDriver:
        if self._driver is None:  # pragma: no cover - reset() always connects first
            raise RuntimeError("step() called before reset()")
        return self._driver

    def _send(self, cmd: Vec) -> None:
        """Clamp to joint limits (safety backstop) and de-normalize grippers."""
        clamped = np.clip(cmd, self._cfg.low, self._cfg.high)
        physical = self._denorm_grippers(clamped)
        self._require_driver().command_joint_pos(physical)

    def _denorm_grippers(self, cmd: Vec) -> Vec:
        out: Vec = cmd.copy()
        span = self._cfg.gripper_closed - self._cfg.gripper_open
        for idx in (packing.ARM_DOF, packing.ARM_WIDTH + packing.ARM_DOF):  # 6, 13
            out[idx] = self._cfg.gripper_open + cmd[idx] * span
        return out

    def _pace(self) -> None:
        hz = self._cfg.control_hz
        if hz and hz > 0:
            elapsed = self._clock() - self._t_last
            self._sleep(max(0.0, 1.0 / hz - elapsed))
        self._t_last = self._clock()

    def _observe(self, instruction: str | None) -> Observation:
        driver = self._require_driver()
        state = packing.validate_dim(driver.get_joint_pos())
        return Observation(
            images=dict(self._camera_reader(self._cfg)),
            state={packing.STATE_KEY: state},
            instruction=instruction,
        )
