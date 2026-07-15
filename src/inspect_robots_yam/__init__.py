"""Inspect Robots adapters for I2RT YAM bimanual arms and ``/act`` policies.

Registers three Inspect Robots components via entry points:

* embodiment ``yam_arms`` — :class:`~inspect_robots_yam.embodiment.YAMEmbodiment`
* policy ``molmoact2`` — :class:`~inspect_robots_yam.policy.MolmoAct2Policy`
* policy ``gr00t`` — :func:`~inspect_robots_yam.policy.gr00t_policy`

so ``inspect-robots run --task kitchenbench/pour_pasta --policy molmoact2
--embodiment yam_arms`` works once both packages are installed. Use
:func:`~inspect_robots_yam.preflight.run_preflight` (or the ``inspect-robots-yam-preflight``
CLI) to verify compatibility before any motion.
"""

from __future__ import annotations

from inspect_robots_yam.config import ActServerConfig, MolmoActConfig, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO
from inspect_robots_yam.packing import STATE_KEY, TOTAL_DIM, pack, split
from inspect_robots_yam.policy import ActServerPolicy, MolmoAct2Policy, gr00t_policy
from inspect_robots_yam.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-yam")
except PackageNotFoundError:  # pragma: no cover - only hit in a non-installed tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "STATE_KEY",
    "TOTAL_DIM",
    "ActServerConfig",
    "ActServerPolicy",
    "MolmoAct2Policy",
    "MolmoActConfig",
    "OperatorIO",
    "YAMEmbodiment",
    "YamConfig",
    "build",
    "gr00t_policy",
    "pack",
    "run_preflight",
    "split",
]
