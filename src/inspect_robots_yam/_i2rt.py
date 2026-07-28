"""Load the optional I2RT driver lazily with actionable installation guidance."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# The build constraint is required while every published ruckig (i2rt's pinned
# dependency) is a source-only release that no longer builds under
# scikit-build-core 1.0; drop it once ruckig ships the fix from pantor/ruckig#261
# and i2rt moves off ruckig==0.15.3 (#47).
I2RT_INSTALL_COMMAND = (
    "echo 'scikit-build-core<0.10' > build-constraints.txt && "
    "uv pip install --build-constraints build-constraints.txt "
    '"i2rt @ git+https://github.com/i2rt-robotics/'
    'i2rt@db582eaa70b6a057a1e2981da6219dfa6c29422a"'
)
_CONTROL_THREAD_JOIN_TIMEOUT = 5.0
_CONTROL_THREAD_GRACE_PERIOD = 0.05

logger = logging.getLogger(__name__)


def close_robot_safely(robot: Any) -> None:
    """Close an I2RT robot without racing its control loop against the CAN socket.

    I2RT discards its control-thread handle and closes the CAN socket without joining
    that thread, so the loop crashes with ``fd=-1`` during every teardown. This helper
    works around robocurve/inspect-robots-yam#28 by discovering the discarded thread
    and interposing its join between I2RT setting ``running = False`` and closing the
    socket.
    """
    chain = getattr(robot, "motor_chain", None)
    if chain is None or getattr(chain, "motor_interface", None) is None:
        # No single-chain interface to guard (unknown driver shape, or a
        # multi-chain aggregate) — fall back to the driver's own teardown.
        robot.close()
        return

    control_threads = [
        thread
        for thread in threading.enumerate()
        if getattr(getattr(thread, "_target", None), "__self__", None) is chain
    ]
    original_close = chain.motor_interface.close
    close_lock = threading.Lock()
    closed = False

    def close_motor_interface_safely() -> None:
        nonlocal closed
        with close_lock:
            if closed:
                return

            if control_threads:
                for thread in control_threads:
                    thread.join(timeout=_CONTROL_THREAD_JOIN_TIMEOUT)
                    if thread.is_alive():
                        logger.warning(
                            "I2RT control thread %s did not stop within %.1f seconds",
                            thread.name,
                            _CONTROL_THREAD_JOIN_TIMEOUT,
                        )
            else:
                logger.debug(
                    "No I2RT control thread was discoverable; waiting %.2f seconds "
                    "before closing the motor interface",
                    _CONTROL_THREAD_GRACE_PERIOD,
                )
                time.sleep(_CONTROL_THREAD_GRACE_PERIOD)

            original_close()
            closed = True

    chain.motor_interface.close = close_motor_interface_safely
    robot.close()


def start_arms_concurrently(
    left_factory: Callable[[], Any],
    right_factory: Callable[[], Any],
) -> tuple[Any, Any]:
    """Bring up both arms concurrently, tearing down the survivor on failure.

    Each arm's driver init includes a mandatory gripper hard-stop calibration
    (multiple seconds), and the arms are fully independent hardware (own CAN
    channel, own control thread), so running the two factories on worker
    threads cuts bring-up wall-clock to max(left, right). If either factory
    raises, any arm that did come up is closed via ``close_robot_safely`` —
    the sequential code leaked it (robocurve/inspect-robots-yam#83) — and the
    init error is re-raised (left's first when both fail).
    """
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="yam-arm-init") as pool:
        futures = [pool.submit(left_factory), pool.submit(right_factory)]
        results: list[Any] = []
        errors: list[tuple[str, BaseException]] = []
        for side, future in zip(("left", "right"), futures, strict=True):
            try:
                results.append(future.result())
            except BaseException as exc:  # re-raised below
                errors.append((side, exc))

    if not errors:
        return results[0], results[1]

    for robot in results:
        try:
            close_robot_safely(robot)
        except Exception:
            logger.warning("closing the surviving arm after an init failure failed", exc_info=True)
    for side, error in errors[1:]:
        logger.error("%s arm bring-up also failed: %s", side, error, exc_info=error)
    raise errors[0][1]


def _load_i2rt() -> tuple[Any, Any]:
    """Load the git-only YAM driver symbols with actionable installation guidance."""
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
    except ModuleNotFoundError as exc:
        if exc.name != "i2rt" and not (exc.name or "").startswith("i2rt."):
            raise
        raise ModuleNotFoundError(
            "i2rt is the I2RT YAM arm driver. It is git-only and not on PyPI. "
            f"Install or update it with: {I2RT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc
    return get_yam_robot, GripperType


def _load_i2rt_kinematics() -> tuple[Any, Any, Any, Any, Any]:
    """Load optional i2rt kinematics symbols without affecting package imports."""
    try:
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        from mink import NoSolutionFound
    except ModuleNotFoundError as exc:
        if exc.name != "i2rt" and not (exc.name or "").startswith("i2rt."):
            raise
        raise ModuleNotFoundError(
            "i2rt is the I2RT YAM arm driver. It is git-only and not on PyPI. "
            f"Install or update it with: {I2RT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc
    return Kinematics, ArmType, GripperType, combine_arm_and_gripper_xml, NoSolutionFound
