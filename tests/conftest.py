"""Shared fixtures for settle tests (issue #62).

Three clock behaviors, because the settle loop has two independent bounds and
each must be observable as the *only* reason the loop stopped:

===============  ==========================================  =================
Fixture          Clock behavior                              Sole exit
===============  ==========================================  =================
frozen           constant 0.0 (what every older helper uses)  poll-count bound
sleep-advancing  ``sleep_fn`` adds its argument               neither alone
read-advancing   ``get_joint_pos()`` adds ``read_advance``    elapsed bound
===============  ==========================================  =================

``read_advance`` must exceed ``_SETTLE_POLL_S`` or both bounds fire on the same
iteration and one of them becomes an unreachable branch, which the 100% gate
turns into a build failure. It is physically justified besides: each
``get_joint_pos()`` is two CAN round trips, so a real poll costs more than the
nominal spacing.

The existing ``_build`` helpers in ``test_embodiment.py`` and
``test_eef_embodiment.py`` keep their hardcoded ``clock=lambda: 0.0``. These
fixtures are additive. Refactoring those helpers onto a read-advancing clock
would inflate ``_pace``'s measured elapsed and break the pacing assertions.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO

#: Comfortably above _SETTLE_POLL_S (0.01), so the elapsed bound trips at 20
#: polls while the count bound would need 100.
READ_ADVANCE_S = 0.05


class Clock:
    """A fake monotonic clock advanced explicitly by whatever charges time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward, as a sleep or a slow CAN read would."""
        self.now += seconds


class SettleDriver:
    """Driver that converges toward its commanded target after ``converge_after`` reads.

    ``offset`` is added to the target once converged, so a nonzero arm slot
    models a joint that never arrives (a hard stop, or a tolerance below the
    rig's steady-state error) and a nonzero gripper slot models a gripper
    closed on an object. Reads are counted so a test can assert which of the
    settle loop's two bounds ended it.
    """

    def __init__(
        self,
        *,
        converge_after: int = 1,
        offset: np.ndarray | None = None,
        clock: Clock | None = None,
        read_advance: float = 0.0,
        state: np.ndarray | None = None,
    ) -> None:
        self.state = np.zeros(14) if state is None else np.asarray(state, dtype=float)
        self.commands: list[np.ndarray] = []
        self.reads = 0
        self.closed = False
        self._converge_after = converge_after
        self.offset = np.zeros(14) if offset is None else np.asarray(offset, dtype=float)
        self._clock = clock
        self._read_advance = read_advance
        self._target: np.ndarray | None = None
        self._reads_since_command = 0

    def get_joint_pos(self) -> np.ndarray:
        """Report the current pose, charging CAN time and counting the read."""
        self.reads += 1
        if self._clock is not None and self._read_advance:
            self._clock.advance(self._read_advance)
        if self._target is not None:
            self._reads_since_command += 1
            if self._reads_since_command >= self._converge_after:
                self.state = self._target + self.offset
        return self.state.copy()

    def command_joint_pos(self, target: np.ndarray) -> None:
        """Record the target and restart the convergence countdown."""
        self.commands.append(np.asarray(target, dtype=float).copy())
        self._target = np.asarray(target, dtype=float).copy()
        self._reads_since_command = 0

    def close(self) -> None:
        """Mark the handle released."""
        self.closed = True


def _cameras(_cfg: YamConfig) -> dict[str, Any]:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": img, "left_cam": img, "right_cam": img}


def _silent_operator() -> OperatorIO:
    return OperatorIO(input_fn=lambda _p: "", output_fn=lambda _m: None)


@pytest.fixture
def clock() -> Clock:
    """A clock the test advances, for asserting on elapsed-time behavior."""
    return Clock()


@pytest.fixture
def build_settle():  # type: ignore[no-untyped-def]
    """Build an embodiment wired for settle tests; returns (embodiment, sleeps, status)."""

    def _make(
        cfg: YamConfig,
        driver: SettleDriver,
        *,
        clock: Clock | None = None,
        sleep_advances: bool = False,
        operator: OperatorIO | None = None,
        kinematics_factory: Any = None,
    ):  # type: ignore[no-untyped-def]
        cfg = dataclasses.replace(cfg, cam_height=4, cam_width=4)
        sleeps: list[float] = []
        status: list[str | None] = []

        def _sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if sleep_advances and clock is not None:
                clock.advance(seconds)

        extra = {} if kinematics_factory is None else {"kinematics_factory": kinematics_factory}
        emb = YAMEmbodiment(
            cfg,
            driver_factory=lambda _c: driver,
            camera_reader=_cameras,
            **extra,
            operator=operator or _silent_operator(),
            poll_end=lambda: False,
            sleep_fn=_sleep,
            clock=clock if clock is not None else (lambda: 0.0),
            status_fn=status.append,
        )
        return emb, sleeps, status

    return _make
