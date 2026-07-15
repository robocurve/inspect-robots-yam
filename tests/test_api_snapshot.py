"""Guard the public API surface so changes to __all__ are deliberate."""

from __future__ import annotations

import re

import inspect_robots_yam

EXPECTED_API = {
    "STATE_KEY",
    "TOTAL_DIM",
    "ActServerPolicy",
    "ActServerConfig",
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
}


def test_public_api_matches_all() -> None:
    assert set(inspect_robots_yam.__all__) == EXPECTED_API


def test_all_names_are_importable() -> None:
    for name in inspect_robots_yam.__all__:
        assert hasattr(inspect_robots_yam, name), name


def test_version() -> None:
    # Tag-derived via hatch-vcs; 0.0.0 fallback in non-installed trees.
    assert re.match(r"\d+\.\d+", inspect_robots_yam.__version__)


def test_entry_points_resolve_via_registry() -> None:
    # The installed entry points must resolve to our classes.
    from inspect_robots.registry import resolve

    pol = resolve("policy", "molmoact2")
    gr00t = resolve("policy", "gr00t")
    emb = resolve("embodiment", "yam_arms")
    assert pol.info.name == "molmoact2"
    assert gr00t.info.name == "gr00t"
    assert emb.info.name == "yam_arms"
