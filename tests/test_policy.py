"""Tests for the generic /act policy client (mocked transport, no network)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_yam import packing
from inspect_robots_yam.config import ActServerConfig, MolmoActConfig
from inspect_robots_yam.policy import ActServerPolicy, MolmoAct2Policy, gr00t_policy


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
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4))
    assert pol.info.name == "molmoact2"
    assert pol.info.action_space.dim == 14
    assert pol.info.action_space.semantics is not None
    assert pol.info.action_space.semantics.control_mode == "joint_pos"
    assert pol.info.control_hz is None  # load-bearing: keeps compat warning-free
    assert pol.info.observation_space.state_keys == frozenset({"joint_pos"})
    assert pol.config.action_horizon == 30


def test_server_url_property_follows_config_default_and_flat_kwargs() -> None:
    assert MolmoAct2Policy().server_url == ActServerConfig().server_url
    assert MolmoAct2Policy(server_url="http://gpu:9000").server_url == "http://gpu:9000"


def test_remedy_property_defaults_and_follows_construction_paths() -> None:
    # The default must be actionable on its own: a runnable command plus a
    # docs URL, since it renders in an error message with no other guidance.
    default = MolmoAct2Policy().remedy
    assert "host_server_yam.py" in default
    assert "https://github.com/robocurve/inspect-robots-yam#" in default
    configured = MolmoAct2Policy(ActServerConfig(remedy="start the configured server"))
    assert configured.remedy == "start the configured server"
    assert MolmoAct2Policy(remedy="run server.sh").remedy == "run server.sh"
    # Explicit empty string opts out of the hint line (falsy for core).
    assert MolmoAct2Policy(remedy="").remedy == ""


def test_remedy_docs_anchors_resolve_to_readme_headings() -> None:
    # The remedy renders inside an error message; a README heading rename must
    # not silently 404 its deep link. Mirrors GitHub's heading slugger.
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    slugs = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in readme.splitlines()
        if line.startswith("#")
    }
    for remedy in (MolmoAct2Policy().remedy, gr00t_policy().remedy):
        fragment = remedy.rsplit("#", 1)[1].rstrip(")")
        assert fragment in slugs


def test_gr00t_remedy_default_names_the_gr00t_server() -> None:
    default = gr00t_policy().remedy
    assert "serve_gr00t_act.py" in default
    assert "host_server_yam.py" not in default
    assert "https://github.com/robocurve/inspect-robots-yam#" in default
    assert gr00t_policy(remedy="run gr00t.sh").remedy == "run gr00t.sh"


def test_connection_failure_hint_getattr_contract() -> None:
    pol = MolmoAct2Policy(
        server_url="http://robot-gpu:8202",
        remedy="run ~/robocurve/molmoact2/run_yam.sh",
    )
    assert getattr(pol, "server_url", None) == "http://robot-gpu:8202"
    assert getattr(pol, "remedy", None) == "run ~/robocurve/molmoact2/run_yam.sh"


def test_connection_failure_hint_properties_are_read_only() -> None:
    pol = MolmoAct2Policy()
    with pytest.raises(AttributeError):
        pol.server_url = "http://other:8202"
    with pytest.raises(AttributeError):
        pol.remedy = "restart it"


def test_policy_action_space_declares_no_gripper_max_step() -> None:
    sem = MolmoAct2Policy().info.action_space.semantics
    assert sem is not None and sem.max_step is None


def test_gr00t_info_and_config_zero_arg() -> None:
    pol = gr00t_policy()
    assert pol.info.name == "gr00t"
    assert pol._cfg.url == "http://127.0.0.1:8203/act"
    assert pol.config.action_horizon == 16
    assert pol.info.action_space.dim == MolmoAct2Policy().info.action_space.dim == 14
    assert pol.info.control_hz is MolmoAct2Policy().info.control_hz is None


def test_gr00t_flat_overrides_and_explicit_config() -> None:
    remote = gr00t_policy(server_url="http://gpu:9000")
    assert remote.info.name == "gr00t"
    assert remote._cfg.url == "http://gpu:9000/act"
    assert gr00t_policy(name="custom").info.name == "custom"

    cfg = ActServerConfig(
        name="configured",
        server_url="http://configured:9999",
        action_horizon=7,
    )
    pol = gr00t_policy(config=cfg, name="ignored", server_url="http://ignored")
    assert pol.info.name == "configured"
    assert pol._cfg.url == "http://configured:9999/act"
    assert pol.config.action_horizon == 7


def test_gr00t_post_fn_passthrough() -> None:
    post, captured = _fake_post(np.zeros((2, 14)), dt_ms=0.0)
    pol = gr00t_policy(cam_height=4, cam_width=4, post_fn=post)
    pol.reset(Scene(id="s", instruction="move"))
    chunk = pol.act(_obs())
    assert len(chunk) == 2
    assert captured["url"] == "http://127.0.0.1:8203/act"


def test_act_builds_request_and_chunk() -> None:
    actions = np.arange(2 * 14, dtype=float).reshape(2, 14)
    post, captured = _fake_post(actions, dt_ms=50.0)
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
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
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction=None))
    chunk = pol.act(_obs(instruction=None))
    assert chunk.control_hz is None  # falsy dt_ms branch


def test_act_dt_ms_zero_gives_no_chunk_hz() -> None:
    post, _ = _fake_post(np.zeros((1, 14)), dt_ms=0.0)
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    assert pol.act(_obs()).control_hz is None


def test_act_negative_dt_ms_raises() -> None:
    post, _ = _fake_post(np.zeros((1, 14)), dt_ms=-5.0)
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="negative dt_ms"):
        pol.act(_obs())


def test_state_key_drives_observation_space() -> None:
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4, state_key="proprio"))
    space = pol.info.observation_space
    assert space.state_keys == frozenset({"proprio"})
    assert space.state is not None
    assert space.state.keys == frozenset({"proprio"})  # StateSpec field key too


def test_act_empty_actions_raises() -> None:
    post, _ = _fake_post(np.zeros((0, 14)))
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="empty action chunk"):
        pol.act(_obs())


def test_act_wrong_action_width_raises() -> None:
    post, _ = _fake_post(np.zeros((2, 8)))
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match=r"expected \(N, 14\)"):
        pol.act(_obs())


def test_act_non_finite_actions_raise() -> None:
    actions = np.zeros((1, 14))
    actions[0, 3] = np.nan
    post, _ = _fake_post(actions)
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="non-finite"):
        pol.act(_obs())


def test_act_missing_actions_key_raises() -> None:
    def _post(url: str, payload: Any, timeout_s: float):
        return {"dt_ms": 100.0}

    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=_post)
    pol.reset(Scene(id="s", instruction="x"))
    with pytest.raises(ValueError, match="missing 'actions'"):
        pol.act(_obs())


def test_act_missing_camera_raises() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    obs = Observation(
        images={"top_cam": np.zeros((4, 4, 3), np.uint8)}, state={"joint_pos": np.zeros(14)}
    )
    with pytest.raises(ValueError, match="missing camera"):
        pol.act(obs)


def test_act_missing_camera_error_uses_configured_name() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = ActServerPolicy(ActServerConfig(name="gr00t"), post_fn=post)
    obs = Observation(
        images={"top_cam": np.zeros((4, 4, 3), np.uint8)},
        state={"joint_pos": np.zeros(14)},
    )
    with pytest.raises(ValueError, match="required by gr00t"):
        pol.act(obs)


def test_act_missing_state_raises() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))
    img = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(images={"top_cam": img, "left_cam": img, "right_cam": img}, state={})
    with pytest.raises(ValueError, match="missing state key"):
        pol.act(obs)


def test_config_object_overrides_flat() -> None:
    # num_steps is the denoising-step count; it must NOT leak into action_horizon.
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4, num_steps=3, action_horizon=5))
    assert pol.config.action_horizon == 5
    assert packing.TOTAL_DIM == 14  # sanity


def test_back_compat_aliases_are_identical() -> None:
    assert MolmoAct2Policy is ActServerPolicy
    assert MolmoActConfig is ActServerConfig


def test_act_validates_camera_shape() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))

    img = np.zeros((2, 2, 3), np.uint8)
    obs = Observation(
        images={"top_cam": img, "left_cam": img, "right_cam": img},
        state={"joint_pos": np.zeros(14)},
    )
    with pytest.raises(ValueError, match="has shape \\(2, 2, 3\\)"):
        pol.act(obs)


def test_act_names_the_camera_that_dropped_a_frame() -> None:
    post, _ = _fake_post(np.zeros((1, 14)))
    pol = MolmoAct2Policy(MolmoActConfig(cam_height=4, cam_width=4), post_fn=post)
    pol.reset(Scene(id="s", instruction="x"))

    img = np.zeros((4, 4, 3), np.uint8)
    obs = Observation(
        images={"top_cam": img, "left_cam": None, "right_cam": img},
        state={"joint_pos": np.zeros(14)},
    )
    with pytest.raises(ValueError, match="camera 'left_cam' has no frame"):
        pol.act(obs)
