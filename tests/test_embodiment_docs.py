"""Verify the mode-specific operating notes advertised to agent policies."""

from __future__ import annotations

import re

import pytest

from inspect_robots_yam.config import EEF_DIM_LABELS, YamConfig
from inspect_robots_yam.embodiment import _DOCS_EEF_POS, _DOCS_JOINTS, YAMEmbodiment
from inspect_robots_yam.packing import DIM_LABELS

_GRIPPER_POLARITY = "0 is fully closed, 1 is fully open"


def _bullet_lines(docs: str) -> str:
    """Return only top-level bullet lines from an embodiment docs string."""
    return "\n".join(line for line in docs.splitlines() if line.startswith("- "))


def test_joint_docs_cover_each_dimension_once_and_state_gripper_polarity() -> None:
    """Joint notes name every action dimension once and state gripper polarity."""
    docs = YAMEmbodiment().info.docs
    assert docs
    assert "left_j0" in docs
    bullet_lines = _bullet_lines(docs)
    assert all(bullet_lines.count(label) == 1 for label in DIM_LABELS)
    assert _GRIPPER_POLARITY in docs


def test_eef_docs_cover_dimensions_without_leaking_workspace_bounds() -> None:
    """EEF notes name all dimensions while leaving numeric bounds to the tool."""
    docs = YAMEmbodiment(control_interface="eef_pos").info.docs
    assert docs
    assert "left_x" in docs
    bullet_lines = _bullet_lines(docs)
    assert all(label in bullet_lines for label in EEF_DIM_LABELS)
    assert _GRIPPER_POLARITY in docs
    assert all(
        re.search(rf"\b{re.escape(token)}\b", docs) is None for token in ("0.48", "0.15", "0.03")
    )


@pytest.mark.parametrize(
    ("control_interface", "built_in"),
    (("joints", _DOCS_JOINTS), ("eef_pos", _DOCS_EEF_POS)),
)
def test_default_docs_equal_mode_builtin(control_interface: str, built_in: str) -> None:
    """Default configuration publishes the byte-exact builtin text only."""
    info = YAMEmbodiment(control_interface=control_interface).info
    assert info.docs == built_in


def test_docs_extra_is_stripped_and_appended_verbatim_after_one_blank_line() -> None:
    """Scalar kwargs append brace-containing rig notes with one blank separator."""
    info = YAMEmbodiment(docs_extra="  rig note {with braces}\n").info
    assert info.docs == _DOCS_JOINTS + "\n\nrig note {with braces}"
    assert info.docs.endswith("rig note {with braces}")


@pytest.mark.parametrize("control_interface", ("joints", "eef_pos"))
def test_whitespace_only_docs_extra_is_empty(control_interface: str) -> None:
    """Whitespace-only operator notes leave the selected builtin unchanged."""
    cfg = YamConfig(control_interface=control_interface, docs_extra="  \n ")
    built_in = _DOCS_EEF_POS if control_interface == "eef_pos" else _DOCS_JOINTS
    assert YAMEmbodiment(cfg).info.docs == built_in
