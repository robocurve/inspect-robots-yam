"""Load the wizard's device defaults when their owner is a YAM embodiment.

The owner gate recognizes YAM embodiments only when the registered entry-point
factory is the embodiment class itself. Function factories are deliberately
treated as non-YAM because checking them would require constructing foreign
embodiments. Also, the core scalar parser turns an unquoted numeric serial into
an integer; quote a serial with a leading zero in config.ini so the zero is
preserved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from inspect_robots.defaults import config_path, load_defaults
from inspect_robots.registry import registered

from inspect_robots_yam.embodiment import YAMEmbodiment

_YAM_DEVICE_KEYS = frozenset(
    {
        "top_cam_device",
        "left_cam_device",
        "right_cam_device",
        "top_depth_serial",
        "left_depth_serial",
        "right_depth_serial",
        "left_channel",
        "right_channel",
    }
)


@dataclass(frozen=True)
class YamDefaults:
    """Filtered YAM device arguments and their wizard-config attribution."""

    args: dict[str, Any]
    source: str | None
    owner: str | None


def load_yam_defaults(env: Mapping[str, str]) -> YamDefaults:
    """Return YAM-owned device defaults, ignoring stale or foreign owners."""
    defaults = load_defaults(env)
    owner = defaults.embodiment_args_owner
    if owner is None or not defaults.embodiment_args:
        return YamDefaults(args={}, source=None, owner=None)

    factory = registered("embodiment").get(owner)
    if not isinstance(factory, type) or not issubclass(factory, YAMEmbodiment):
        return YamDefaults(args={}, source=None, owner=None)

    args = {
        key: str(value)
        for key, value in defaults.embodiment_args.items()
        if key in _YAM_DEVICE_KEYS and value is not None
    }
    if not args:
        return YamDefaults(args={}, source=None, owner=None)

    path = config_path(env)
    assert path is not None
    return YamDefaults(args=args, source=str(path), owner=owner)
