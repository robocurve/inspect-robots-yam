"""Generic client for bimanual-YAM policies served through an ``/act`` endpoint.

The model runs as a separate process that owns the GPU and weights. This
stateless client packs three cameras, the language instruction, and the packed
14-D state into each request, then converts the returned ``(N, 14)`` array into
an Inspect Robots :class:`~inspect_robots.types.ActionChunk`. MolmoAct2's
``examples/yam/host_server_yam.py`` is the canonical server implementation;
``scripts/serve_gr00t_act.py`` serves GR00T fine-tunes through the same wire
contract.

The HTTP transport is injected (``post_fn``) so the whole policy is testable with
no server and no network; the real transport (`requests` + `json_numpy`) is a
pragma'd default that only runs on hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_yam import packing
from inspect_robots_yam.config import ActServerConfig, action_box, observation_space

# (url, payload, timeout_s) -> response mapping with keys "actions" and "dt_ms".
PostFn = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def _default_post(  # pragma: no cover - real network transport, only vs a live server
    url: str, payload: Mapping[str, Any], timeout_s: float
) -> Mapping[str, Any]:
    import json_numpy
    import requests

    resp = requests.post(url, data=json_numpy.dumps(payload), timeout=timeout_s)
    resp.raise_for_status()
    decoded: Mapping[str, Any] = json_numpy.loads(resp.content)
    return decoded


class ActServerPolicy:
    """Inspect Robots policy wrapping a bimanual-YAM ``/act`` endpoint."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "requests": "uv pip install inspect-robots-yam",
        "json_numpy": "uv pip install inspect-robots-yam",
    }

    def __init__(
        self,
        config: ActServerConfig | None = None,
        *,
        post_fn: PostFn | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else ActServerConfig.from_kwargs(**flat)
        self._post_fn: PostFn = post_fn if post_fn is not None else _default_post
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            # Semantics only; the embodiment owns limits. joints_are_delta must
            # mirror the rig's YamConfig or compat fails loudly (by design).
            action_space=action_box(joints_are_delta=self._cfg.joints_are_delta),
            observation_space=observation_space(
                self._cfg.cam_height,
                self._cfg.cam_width,
                self._cfg.camera_order,
                state_key=self._cfg.state_key,
            ),
            # Intentionally None: advertising a rate would trip a (harmless) compat
            # control_rate warning. The trained rate rides on the returned chunk.
            control_hz=None,
        )
        self.config = PolicyConfig(action_horizon=self._cfg.action_horizon)

    def reset(self, scene: Scene) -> None:
        """Stash the scene's instruction (fed to the VLA verbatim)."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Query the ``/act`` server and return the predicted action chunk."""
        cfg = self._cfg
        try:
            images = {cam: observation.images[cam] for cam in cfg.camera_order}
        except KeyError as exc:
            raise ValueError(f"observation missing camera {exc} required by {cfg.name}") from exc

        expected_shape = (cfg.cam_height, cfg.cam_width, 3)
        for cam, img in images.items():
            if img.shape != expected_shape:
                raise ValueError(f"camera {cam!r} has shape {img.shape}, expected {expected_shape}")

        if cfg.state_key not in observation.state:
            raise ValueError(f"observation missing state key {cfg.state_key!r}")
        state = packing.validate_dim(observation.state[cfg.state_key]).astype(np.float32)

        payload: dict[str, Any] = {
            **images,
            "instruction": self._instruction or "",
            "state": state,
            "num_steps": cfg.num_steps,
        }

        t0 = time.perf_counter()
        resp = self._post_fn(cfg.url, payload, cfg.timeout_s)
        elapsed = time.perf_counter() - t0

        if "actions" not in resp:
            raise ValueError("/act response missing 'actions'")
        actions = np.asarray(resp["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != packing.TOTAL_DIM:
            raise ValueError(
                f"/act returned actions of shape {actions.shape}; expected (N, {packing.TOTAL_DIM})"
            )
        if not np.isfinite(actions).all():
            raise ValueError("/act returned non-finite actions")
        if actions.shape[0] == 0:
            raise ValueError("/act returned an empty action chunk")

        dt_ms = resp.get("dt_ms")
        if dt_ms is not None and dt_ms < 0:
            raise ValueError(f"/act returned negative dt_ms: {dt_ms!r}")
        # 0/None deliberately mean "no advertised rate" (falsy), not an error.
        chunk_hz = 1000.0 / dt_ms if dt_ms else None
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=chunk_hz,
            inference_latency_s=elapsed,
        )


MolmoAct2Policy = ActServerPolicy

GR00T_DEFAULTS: Mapping[str, Any] = {
    "name": "gr00t",
    "server_url": "http://127.0.0.1:8203",
    # robocurve/gr00t-n1.7-yam-molmoact2 predicts 16-step chunks
    # (experiment_cfg delta_indices 0..15). Like every field, this is
    # PolicyConfig *metadata* recorded in the eval log — rollout always uses
    # the returned chunk — but leaving MolmoAct2's 30 here would record a
    # false horizon for GR00T runs, the exact dishonesty this plan removes.
    # Operators evaluating a different GR00T fine-tune pass
    # -P action_horizon=<its chunk length>.
    "action_horizon": 16,
}


def gr00t_policy(
    config: ActServerConfig | None = None,
    *,
    post_fn: PostFn | None = None,
    **flat: Any,
) -> ActServerPolicy:
    """Registry factory for the ``gr00t`` policy: the /act client with GR00T defaults.

    Same wire protocol and 14-D contract as ``molmoact2``; only the advertised
    policy name and the default port differ, so a GR00T /act server (see
    ``scripts/serve_gr00t_act.py``) and the MolmoAct2 server can run side by
    side and eval logs record which model actually ran. Explicit ``flat``
    kwargs override the defaults; an explicit ``config`` wins outright,
    including its ``name`` — a caller passing ``config=ActServerConfig()``
    gets the molmoact2 defaults and owns the resulting log label.
    """
    if config is None:
        config = ActServerConfig.from_kwargs(**{**GR00T_DEFAULTS, **flat})
    return ActServerPolicy(config, post_fn=post_fn)
