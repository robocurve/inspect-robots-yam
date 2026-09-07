"""Verify offline LLM-command parity, frames, and playback against real MuJoCo."""

from __future__ import annotations

import builtins
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from inspect_robots_agent._llm import ToolCall

from inspect_robots_yam._i2rt import I2RT_INSTALL_COMMAND
from inspect_robots_yam.config import DEFAULT_EEF_HOME_POSE, YamConfig

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mujoco_viewer.py"
_SPEC = importlib.util.spec_from_file_location("mujoco_viewer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
viewer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(viewer)


@pytest.fixture
def preview():
    return viewer.Preview(YamConfig(control_interface="eef_pos"))


def finish(preview):
    while preview.pending:
        preview.step()


def test_model_snapshot_tracks_the_runtime_driver_pin():
    revision = re.search(r"i2rt@([0-9a-f]{40})", I2RT_INSTALL_COMMAND)
    assert revision is not None
    assert revision[1] in viewer.MODEL_PATH.read_text()
    assert (viewer.MODEL_PATH.parent / "LICENSE").is_file()


def test_pinned_model_home_matches_hardware_contract(preview):
    assert preview.joints == pytest.approx(DEFAULT_EEF_HOME_POSE)
    for start in (0, 7):
        assert preview.eef[start : start + 3] == pytest.approx((0.30, 0, 0.20), abs=0.001)
        assert preview.eef[start + 3 : start + 6] == pytest.approx((0, 0, 0), abs=1e-12)
        assert preview.eef[start + 6] == 1
    reference = preview.references[0]
    assert reference[:, 0] == pytest.approx((-0.5, 0, -np.sqrt(3) / 2), abs=0.001)


def test_preview_never_imports_the_hardware_driver(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith("i2rt")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    preview = viewer.Preview(YamConfig(control_interface="eef_pos"))
    preview.submit({"targets": {"left_x": 0.32}})
    finish(preview)


def test_macos_launcher_preserves_arguments_and_library_paths(monkeypatch):
    import mujoco.viewer

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", None)
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/custom/lib")
    calls = []
    monkeypatch.setattr(viewer.os, "execve", lambda *args: calls.append(args))
    viewer.launch_macos(["--speed", "0.1"])
    path, argv, env = calls[0]
    assert path.endswith("/mjpython")
    assert argv[1:] == [str(_SCRIPT), "--speed", "0.1"]
    assert env["DYLD_FALLBACK_LIBRARY_PATH"] == f"{Path(sys.base_prefix) / 'lib'}:/custom/lib"
    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", object())
    viewer.launch_macos([])
    monkeypatch.setattr(sys, "platform", "linux")
    viewer.launch_macos([])
    assert len(calls) == 1


def test_identical_agent_actions_and_ik_step_clamp(preview):
    payload = {"targets": {"left_x": 0.34, "right_y": 0.025, "right_gripper": 0.2}, "note": "Test"}
    expected = preview.tools.execute(
        ToolCall(id="test", name="move_to", arguments=json.dumps(payload)),
        preview.observation,
    )
    assert expected.chunk is not None
    steps = preview.submit(payload)
    assert steps == len(expected.chunk.actions)
    for actual, expected_action in zip(preview.pending, expected.chunk.actions, strict=True):
        np.testing.assert_array_equal(actual.data, expected_action.data)
    while preview.pending:
        before = preview.joints
        preview.step()
        arm_slots = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        assert np.max(np.abs((preview.joints - before)[arm_slots])) <= 0.2 + 1e-12
    assert preview.eef[0] == pytest.approx(0.34, abs=0.001)
    assert preview.eef[8] == pytest.approx(0.025, abs=0.001)
    assert preview.eef[13] == pytest.approx(0.2)
    assert max(preview.report()["position_error_m"].values()) < 0.001


def test_partial_command_leaves_other_arm_at_its_observed_pose(preview):
    initial = preview.eef.copy()
    preview.submit({"function": {"name": "move_to", "arguments": '{"targets":{"left_z":0.23}}'}})
    finish(preview)
    np.testing.assert_allclose(preview.eef[7:], initial[7:], atol=1e-4)
    assert preview.eef[2] == pytest.approx(0.23, abs=0.001)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"targets": {"left_pitch": 0.2}}, "fixed"),
        ({"targets": {"left_x": 2}}, "outside"),
        ({"targets": {"left_y": float("nan")}}, "finite"),
        ({"targets": {"right_gripper": True}}, "finite"),
        ({"targets": {"left_j0": 1}}, "unknown"),
        ({"targets": {}}, "non-empty"),
        ([], "JSON object"),
        ({"function": 1}, "function"),
        ({"function": {"name": "move_by", "arguments": {}}}, "move_to"),
    ],
)
def test_invalid_commands_do_not_move_or_mutate_state(preview, payload, message):
    before = preview.joints
    with pytest.raises(ValueError, match=message):
        preview.submit(payload)
    np.testing.assert_array_equal(preview.joints, before)
    assert not preview.pending


def test_explicit_orientation_uses_repo_pitch_sign_and_reset_reference():
    preview = viewer.Preview(YamConfig(control_interface="eef_pos", eef_orientation=True))
    reference = preview.references[0].copy()
    preview.submit({"targets": {"left_yaw": 0.15, "left_pitch": 0.12, "left_roll": -0.08}})
    finish(preview)
    assert preview.eef[3:6] == pytest.approx((0.15, 0.12, -0.08), abs=0.003)
    preview.reset()
    np.testing.assert_allclose(preview.references[0], reference)
    assert preview.eef[3:6] == pytest.approx((0, 0, 0), abs=1e-12)


def test_world_mount_does_not_change_base_frame_ik_or_targets():
    config = YamConfig(
        control_interface="eef_pos",
        collision_left_base_pos=(0.1, 0.5, 0.2),
        collision_right_base_pos=(0.1, -0.5, 0.2),
        collision_right_base_yaw=np.pi,
    )
    preview = viewer.Preview(config)
    scene = viewer.ViewerScene(preview)
    for index, side in enumerate(("left", "right")):
        offset = 7 * index
        base, rotation = scene.bases[index]
        achieved = scene.data.site_xpos[scene.model.site(f"{side}_grasp_site").id]
        expected = base + rotation @ preview.eef[offset : offset + 3]
        np.testing.assert_allclose(achieved, expected, atol=1e-10)
        marker = scene.model.body(f"{side}_target").mocapid[0]
        np.testing.assert_allclose(scene.data.mocap_pos[marker], expected, atol=1e-10)
    preview.submit({"targets": {"left_x": 0.34, "right_x": 0.34}})
    finish(preview)
    np.testing.assert_allclose(preview.eef[:7], preview.eef[7:], atol=1e-10)


def test_normalized_grippers_animate_with_correct_polarity(preview):
    scene = viewer.ViewerScene(preview)
    distances = []
    for opening in (0, 0.5, 1):
        joints = preview.joints
        joints[6] = joints[13] = opening
        scene.draw(joints)
        positions = []
        for tip in ("left", "right"):
            positions.append(scene.data.xpos[scene.model.body(f"left_tip_{tip}").id].copy())
        distances.append(np.linalg.norm(positions[0] - positions[1]))
        for side in ("left", "right"):
            for index in (7, 8):
                adr = scene.model.joint(f"{side}_joint{index}").qposadr[0]
                assert scene.data.qpos[adr] == pytest.approx((1 - opening) * 0.0475)
    assert distances[0] < distances[1] < distances[2]


def test_native_gripper_calibration_is_preserved():
    config = YamConfig(control_interface="eef_pos", gripper_open=-2, gripper_closed=3)
    preview = viewer.Preview(config)
    preview.submit({"targets": {"left_gripper": 0.25}})
    finish(preview)
    assert preview.driver.state[6] == pytest.approx(1.75)
    assert preview.eef[6] == pytest.approx(0.25)


def test_playback_speed_changes_time_only_and_pause_does_not_solve():
    endpoints = []
    frame_counts = []
    for speed in (1, 0.1):
        preview = viewer.Preview(YamConfig(control_interface="eef_pos"))
        player = viewer.Playback(preview, speed)
        viewer.handle_command('{"targets":{"left_x":0.32}}', player)
        viewer.handle_command("pause", player)
        count = len(preview.pending)
        initial = player.shown.copy()
        for _ in range(3):
            np.testing.assert_array_equal(player.advance(0.1), initial)
        assert len(preview.pending) == count
        viewer.handle_command("next", player)
        assert len(preview.pending) == count - 1
        assert player.paused
        viewer.handle_command("play", player)
        frames = 0
        while player.busy:
            player.advance(0.02)
            frames += 1
        endpoints.append(preview.joints)
        frame_counts.append(frames)
    np.testing.assert_array_equal(*endpoints)
    assert frame_counts[1] > 8 * frame_counts[0]


def test_busy_includes_last_frame_and_reset_clears_it(preview):
    player = viewer.Playback(preview)
    viewer.handle_command('{"targets":{"left_x":0.301}}', player)
    with pytest.raises(ValueError, match="progress"):
        preview.submit({"targets": {"left_x": 0.32}})
    while preview.pending:
        player.advance(0.01)
    assert player.animating
    with pytest.raises(ValueError, match="progress"):
        viewer.handle_command('{"targets":{"left_x":0.32}}', player)
    viewer.handle_command("reset", player)
    assert not player.busy
    assert json.loads(viewer.handle_command("state", player))["remaining_actions"] == 0
    assert viewer.handle_command("quit", player) == "quit"
    viewer.handle_command("toggle", player)
    assert player.paused
    viewer.handle_command("speed 0.5", player)
    assert player.speed == 0.5
    for value in (0, -1, float("nan"), float("inf"), 5):
        with pytest.raises(ValueError, match="speed"):
            player.set_speed(value)


def test_explicit_config_preserves_motion_and_strips_hardware(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text(
        "[defaults]\nembodiment=yam_arms\npolicy=agent\n[embodiment.args]\n"
        "control_interface=eef_pos\ncontrol_hz=20\nik_step_joint_limit=0.08\n"
        "collision_right_base_yaw=3.141592653589793\n"
        "top_depth_serial=A\nleft_depth_serial=B\nright_depth_serial=C\n"
        "[policy.args]\nmax_speed_frac=0.05\n"
    )
    config, speed = viewer.read_config(path, True)
    assert speed == 0.05
    clean = viewer.preview_config(config)
    assert clean.control_hz == 20
    assert clean.ik_step_joint_limit == 0.08
    assert clean.collision_right_base_yaw == np.pi
    assert clean.eef_orientation
    assert clean.top_depth_serial is None
    assert config.top_depth_serial == "A"
    with pytest.raises(ValueError, match="does not exist"):
        viewer.read_config(tmp_path / "missing", False)
    path.write_text("[defaults]\nembodiment=another_robot\n")
    with pytest.raises(ValueError, match="yam_arms"):
        viewer.read_config(path, False)
    with pytest.raises(ValueError, match="eef_pos"):
        viewer.preview_config(YamConfig())


def test_headless_replays_sequence_and_reports_achieved_states(tmp_path, capsys):
    path = tmp_path / "moves.jsonl"
    path.write_text('{"targets":{"left_x":0.32}}\n\n{"targets":{"right_gripper":0}}\n')
    viewer.main(["--headless", "--commands", str(path)])
    reports = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert len(reports) == 3
    assert reports[-1]["eef_state"]["left_x"] == pytest.approx(0.32, abs=0.001)
    assert reports[-1]["eef_state"]["right_gripper"] == 0
    with pytest.raises(SystemExit):
        viewer.main(["--headless"])
