"""Load the optional I2RT driver lazily with actionable installation guidance."""

from __future__ import annotations

from typing import Any

I2RT_INSTALL_COMMAND = 'uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"'


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
