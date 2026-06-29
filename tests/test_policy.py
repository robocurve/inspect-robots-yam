"""Tests for MolmoAct2Policy (mocked /act transport — no server, no network)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from robolens.scene import Scene
from robolens.types import Observation

from robolens_yam import packing
from robolens_yam.config import MolmoActConfig
from robolens_yam.policy import MolmoAct2Policy


def _obs(instruction: str | None = "do it") -> Observation:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return Observation(
        images={"top_cam": img, "left_cam": img, "right_cam": img},
        state={"joint_pos": np.zeros(14)},
        instruction=instruction,
    )


def _fake_post(actions: np.ndarray, dt_ms: Any = 100.0):
    captured: dict[str, Any] = {}

    def _post(url: str, payload: Any, timeout_s: float):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return {"actions": actions, "dt_ms": dt_ms}

    return _post, captured


def test_info_and_config_zero_arg() -> None:
    pol = MolmoAct2Policy()
    assert pol.info.name == "molmoact2"
    assert pol.info.action_space.dim == 14
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.control_mode == "joint_pos"
    assert pol.info.control_hz is None  # load-bearing: keeps compat warning-free
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert pol.config.action_horizon == 10


def test_act_builds_request_and_chunk() -> None:
    actions = np.arange(2 * 14, dtype=float).reshape(2, 14)
    post, captured = _fake_post(actions, dt_ms=50.0)
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction="pour the pasta"))
    chunk = pol.act(_obs())

    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[0].data, actions[0])
    assert chunk.control_hz == pytest.approx(1000.0 / 50.0)
    assert chunk.inference_latency_s is not None
    # Request payload carries cameras (in order), instruction, float32 state, num_steps.
    payload = captured["payload"]
    assert list(payload)[:3] == ["top_cam", "left_cam", "right_cam"]
    assert payload["instruction"] == "pour the pasta"
    assert payload["state"].dtype == np.float32
    assert payload["num_steps"] == 10
    assert captured["url"].endswith("/act")
    assert pol.num_inferences == 1


def test_act_dt_ms_none_gives_no_chunk_hz() -> None:
    post, _ = _fake_post(np.zeros((1, 14)), dt_ms=None)
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction=None))
    chunk = pol.act(_obs(instruction=None))
    assert chunk.control_hz is None  # falsy dt_ms branch


def test_act_dt_ms_zero_gives_no_chunk_hz() -> None:
    post, _ = _fake_post(np.zeros((1, 14)), dt_ms=0.0)
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    assert pol.act(_obs()).control_hz is None


def test_act_empty_actions_raises() -> None:
    post, _ = _fake_post(np.zeros((0, 14)))
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="empty action chunk"):
        pol.act(_obs())


def test_act_wrong_action_width_raises() -> None:
    post, _ = _fake_post(np.zeros((2, 8)))
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"expected \(N, 14\)"):
        pol.act(_obs())


def test_act_missing_actions_key_raises() -> None:
    def _post(url: str, payload: Any, timeout_s: float):
        return {"dt_ms": 100.0}

    pol = MolmoAct2Policy(post_fn=_post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="missing 'actions'"):
        pol.act(_obs())


def test_act_missing_camera_raises() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(
        images={"top_cam": np.zeros((4, 4, 3), np.uint8)}, state={"joint_pos": np.zeros(14)}
    )
    with pytest.raises(ValueError, match="missing camera"):
        pol.act(obs)


def test_act_missing_state_raises() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = MolmoAct2Policy(post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    img = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(images={"top_cam": img, "left_cam": img, "right_cam": img}, state={})
    with pytest.raises(ValueError, match="missing state key"):
        pol.act(obs)


def test_config_object_overrides_flat() -> None:
    pol = MolmoAct2Policy(MolmoActConfig(num_steps=3))
    assert pol.config.action_horizon == 3
    assert packing.TOTAL_DIM == 14  # sanity
