"""Public-surface integration with the inspect-robots-agent EEF tool flow."""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_agent import LLMAgentPolicy

from inspect_robots_yam.config import EEF_DIM_LABELS, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment


def test_public_agent_policy_advertises_and_executes_guardrail_clean_eef_move() -> None:
    captured: dict[str, Any] = {}

    def scripted_response(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured["tools"] = payload["tools"]
        move_schema = next(
            tool
            for tool in payload["tools"]
            if "targets" in tool["function"]["parameters"]["properties"]
        )
        move_name = move_schema["function"]["name"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "move-1",
                                    "type": "function",
                                    "function": {
                                        "name": move_name,
                                        "arguments": json.dumps(
                                            {"targets": {"left_x": 0.31, "right_y": 0.01}}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    embodiment = YAMEmbodiment(YamConfig(control_interface="eef_pos"))
    policy = LLMAgentPolicy(
        model="test-model",
        base_url="http://agent.test/v1",
        transport=httpx.MockTransport(scripted_response),
        env={},
    )
    policy.bind(embodiment.info)
    policy.reset(Scene(id="eef-agent", instruction="move both arms"))
    current = np.asarray((0.30, 0.0, 0.20, 0.0, 1.0, 0.30, 0.0, 0.20, 0.0, 1.0))
    observation = Observation(
        images={},
        state={"joint_pos": np.zeros(14), "eef_state": current},
        instruction="move both arms",
    )

    chunk = policy.act(observation)

    advertised = json.dumps(captured["tools"])
    assert all(label in advertised for label in EEF_DIM_LABELS)
    assert "left_x: [0.15, 0.48]" in advertised
    assert "left_yaw: [-3.142, 3.142]" in advertised
    assert "right_gripper: [0, 1]" in advertised
    assert len(chunk.actions) > 1
    assert chunk.actions[-1].data[0] == pytest.approx(0.31)
    assert chunk.actions[-1].data[6] == pytest.approx(0.01)
    x_values = np.asarray([action.data[0] for action in chunk.actions])
    assert np.all(np.diff(x_values) > 0)

    approver = ChainApprover(
        ClampApprover(embodiment.info.action_space),
        DeltaLimitApprover(embodiment.info.action_space),
    )
    store: dict[str, Any] = {}
    for action in chunk.actions:
        assert approver.review(action, store) is action
