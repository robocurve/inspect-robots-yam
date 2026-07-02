"""inspect-robots-yam — Inspect Robots adapters for I2RT YAM bimanual arms + MolmoAct2.

Registers two Inspect Robots components via entry points:

* embodiment ``yam_arms`` — :class:`~inspect_robots_yam.embodiment.YAMEmbodiment`
* policy ``molmoact2`` — :class:`~inspect_robots_yam.policy.MolmoAct2Policy`

so ``inspect-robots run --task kitchenbench/pour_pasta --policy molmoact2
--embodiment yam_arms`` works once both packages are installed. Use
:func:`~inspect_robots_yam.preflight.run_preflight` (or the ``inspect-robots-yam-preflight``
CLI) to verify compatibility before any motion.
"""

from __future__ import annotations

from inspect_robots_yam.config import MolmoActConfig, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO
from inspect_robots_yam.packing import STATE_KEY, TOTAL_DIM, pack, split
from inspect_robots_yam.policy import MolmoAct2Policy
from inspect_robots_yam.preflight import build, run_preflight

__version__ = "0.3.0"

__all__ = [
    "STATE_KEY",
    "TOTAL_DIM",
    "MolmoAct2Policy",
    "MolmoActConfig",
    "OperatorIO",
    "YAMEmbodiment",
    "YamConfig",
    "build",
    "pack",
    "run_preflight",
    "split",
]
