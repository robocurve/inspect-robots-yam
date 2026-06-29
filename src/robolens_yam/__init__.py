"""robolens-yam — RoboLens adapters for I2RT YAM bimanual arms + MolmoAct2.

Registers two RoboLens components via entry points:

* embodiment ``yam_arms`` — :class:`~robolens_yam.embodiment.YAMEmbodiment`
* policy ``molmoact2`` — :class:`~robolens_yam.policy.MolmoAct2Policy`

so ``robolens run --task kitchenbench/pour_pasta --policy molmoact2
--embodiment yam_arms`` works once both packages are installed. Use
:func:`~robolens_yam.preflight.run_preflight` (or the ``robolens-yam-preflight``
CLI) to verify compatibility before any motion.
"""

from __future__ import annotations

from robolens_yam.config import MolmoActConfig, YamConfig
from robolens_yam.embodiment import YAMEmbodiment
from robolens_yam.operator import OperatorIO
from robolens_yam.packing import STATE_KEY, TOTAL_DIM, pack, split
from robolens_yam.policy import MolmoAct2Policy
from robolens_yam.preflight import build, run_preflight

__version__ = "0.1.0"

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
