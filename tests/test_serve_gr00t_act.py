"""Exercise the dependency-light validation and packing logic in the GR00T shim."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import serve_gr00t_act  # type: ignore[import-not-found]


@dataclass(frozen=True)
class _ModalityConfig:
    """Provide the modality fields consumed by the server's pure helpers."""

    modality_keys: list[str]
    delta_indices: list[int]


CANONICAL_KEYS = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
CAMERA_MAP = {
    "top_cam": "base_view",
    "left_cam": "left_wrist_view",
    "right_cam": "right_wrist_view",
}


def _modality_configs() -> dict[str, _ModalityConfig]:
    """Return a valid single-frame modality configuration."""
    return {
        "video": _ModalityConfig(list(CAMERA_MAP.values()), [0]),
        "state": _ModalityConfig(CANONICAL_KEYS.copy(), [0]),
        "action": _ModalityConfig(CANONICAL_KEYS.copy(), [0]),
        "language": _ModalityConfig(["annotation.human.action.task_description"], [0]),
    }


def _statistics() -> dict[str, Any]:
    """Return realistic radians and normalized-gripper checkpoint statistics."""
    widths = {"left_arm": 6, "left_gripper": 1, "right_arm": 6, "right_gripper": 1}
    modality_stats: dict[str, dict[str, list[float]]] = {}
    for key, width in widths.items():
        if key.endswith("_arm"):
            minimum = [-3.0] * width
            maximum = [3.150] * width
        else:
            minimum = [0.0] * width
            maximum = [1.0] * width
        modality_stats[key] = {
            "mean": [0.0] * width,
            "min": minimum,
            "max": maximum,
        }
    return {
        serve_gr00t_act.EMBODIMENT_TAG: {
            "state": modality_stats,
            "action": {
                key: {name: values.copy() for name, values in key_stats.items()}
                for key, key_stats in modality_stats.items()
            },
        }
    }


def _valid_payload() -> dict[str, Any]:
    """Return one complete request with distinct RGB camera values."""
    return {
        "top_cam": np.full((2, 3, 3), 10, dtype=np.uint8),
        "left_cam": np.full((2, 3, 3), 20, dtype=np.uint8),
        "right_cam": np.full((2, 3, 3), 30, dtype=np.uint8),
        "state": [float(index) for index in range(14)],
        "instruction": "put the cup in the sink",
    }


def _action_array(chunk_len: int, width: int, value: float) -> npt.NDArray[np.float32]:
    """Build one batched action part with a distinctive constant value."""
    return np.full((1, chunk_len, width), value, dtype=np.float32)


def test_parse_default_camera_map() -> None:
    """The advertised default maps every client camera to its checkpoint key."""
    assert serve_gr00t_act._parse_camera_map(serve_gr00t_act.DEFAULT_CAMERA_MAP) == CAMERA_MAP


def test_parse_camera_map_rejects_typoed_source() -> None:
    """A source typo reports all supported client-side names."""
    raw = "top_cam:base_view,left_cam:left_wrist_view,right_camera:right_wrist_view"
    with pytest.raises(ValueError, match=r"left_cam.*right_cam.*top_cam"):
        serve_gr00t_act._parse_camera_map(raw)


def test_parse_camera_map_rejects_wrong_pair_count() -> None:
    """Exactly three camera pairs are required."""
    with pytest.raises(ValueError, match="source names"):
        serve_gr00t_act._parse_camera_map("top_cam:base_view,left_cam:left_wrist_view")


def test_parse_camera_map_rejects_duplicate_source() -> None:
    """Every client camera source must occur exactly once."""
    raw = "top_cam:base_view,top_cam:left_wrist_view,right_cam:right_wrist_view"
    with pytest.raises(ValueError, match="source names"):
        serve_gr00t_act._parse_camera_map(raw)


def test_parse_camera_map_rejects_duplicate_target() -> None:
    """Checkpoint camera targets must be distinct."""
    raw = "top_cam:base_view,left_cam:base_view,right_cam:right_wrist_view"
    with pytest.raises(ValueError, match="target names must be distinct"):
        serve_gr00t_act._parse_camera_map(raw)


def test_validate_modality_configs_accepts_canonical_single_frame_config() -> None:
    """Canonical state/action keys and mapped single-frame videos are accepted."""
    serve_gr00t_act._validate_modality_configs(_modality_configs(), CAMERA_MAP)


@pytest.mark.parametrize("modality", ["state", "action"])
def test_validate_modality_configs_rejects_missing_canonical_key(modality: str) -> None:
    """Both state and action configs must contain every canonical body part."""
    configs = _modality_configs()
    configs[modality] = _ModalityConfig(CANONICAL_KEYS[:-1], [0])
    with pytest.raises(ValueError, match=rf"checkpoint {modality} keys must equal"):
        serve_gr00t_act._validate_modality_configs(configs, CAMERA_MAP)


@pytest.mark.parametrize("modality", ["state", "action"])
def test_validate_modality_configs_rejects_unknown_key(modality: str) -> None:
    """Unknown state and action body parts cannot enter the canonical partition."""
    configs = _modality_configs()
    configs[modality] = _ModalityConfig([*CANONICAL_KEYS, "torso"], [0])
    with pytest.raises(ValueError, match="torso"):
        serve_gr00t_act._validate_modality_configs(configs, CAMERA_MAP)


@pytest.mark.parametrize("modality", ["video", "state"])
def test_validate_modality_configs_rejects_multiple_frames(modality: str) -> None:
    """The stateless shim rejects video and state frame history."""
    configs = _modality_configs()
    configs[modality] = _ModalityConfig(configs[modality].modality_keys, [-1, 0])
    with pytest.raises(ValueError, match=rf"{modality} delta_indices"):
        serve_gr00t_act._validate_modality_configs(configs, CAMERA_MAP)


def test_validate_modality_configs_rejects_camera_target_mismatch() -> None:
    """Mapped targets must exactly match the checkpoint's video keys."""
    configs = _modality_configs()
    configs["video"] = _ModalityConfig(["base_view", "left_wrist_view", "overhead"], [0])
    with pytest.raises(ValueError, match="target names must equal checkpoint video keys"):
        serve_gr00t_act._validate_modality_configs(configs, CAMERA_MAP)


def test_validate_statistics_accepts_realistic_ranges_and_widths() -> None:
    """Radians up to 3.150 and normalized grippers satisfy the safety bounds."""
    serve_gr00t_act._validate_statistics(_statistics(), _modality_configs())


def test_validate_statistics_rejects_wrong_width() -> None:
    """A statistic vector must match its canonical state/action part width."""
    stats = _statistics()
    stats[serve_gr00t_act.EMBODIMENT_TAG]["state"]["left_arm"]["mean"] = [0.0] * 5
    with pytest.raises(ValueError, match="has width 5; expected 6"):
        serve_gr00t_act._validate_statistics(stats, _modality_configs())


def test_validate_statistics_rejects_degree_scale_arms() -> None:
    """Degree-scale arm statistics are rejected as incompatible with radians."""
    stats = _statistics()
    stats[serve_gr00t_act.EMBODIMENT_TAG]["action"]["right_arm"]["max"][0] = 90.0
    with pytest.raises(ValueError, match="exceed radians limit"):
        serve_gr00t_act._validate_statistics(stats, _modality_configs())


def test_validate_statistics_arm_tolerance_boundary() -> None:
    """The arm limit is pinned at pi + 0.05: 3.19 rad passes, 3.20 rad fails."""
    stats = _statistics()
    stats[serve_gr00t_act.EMBODIMENT_TAG]["action"]["right_arm"]["max"][0] = 3.19
    serve_gr00t_act._validate_statistics(stats, _modality_configs())
    stats[serve_gr00t_act.EMBODIMENT_TAG]["action"]["right_arm"]["max"][0] = 3.20
    with pytest.raises(ValueError, match="exceed radians limit"):
        serve_gr00t_act._validate_statistics(stats, _modality_configs())


def test_validate_statistics_rejects_gripper_outside_normalized_range() -> None:
    """Gripper statistics must stay within the tolerated normalized range."""
    stats = _statistics()
    stats[serve_gr00t_act.EMBODIMENT_TAG]["state"]["left_gripper"]["min"][0] = -0.06
    with pytest.raises(ValueError, match="fall outside"):
        serve_gr00t_act._validate_statistics(stats, _modality_configs())


@pytest.mark.parametrize("non_finite", [float("inf"), float("nan")])
def test_validate_statistics_rejects_non_finite_bounds(non_finite: float) -> None:
    """Infinite and NaN range endpoints cannot pass checkpoint validation."""
    stats = _statistics()
    stats[serve_gr00t_act.EMBODIMENT_TAG]["action"]["right_gripper"]["max"][0] = non_finite
    with pytest.raises(ValueError, match="non-finite"):
        serve_gr00t_act._validate_statistics(stats, _modality_configs())


def test_build_observation_batches_and_slices_request_fields() -> None:
    """Request cameras, state, and language are shaped for one GR00T batch."""
    payload = _valid_payload()
    observation = serve_gr00t_act._build_observation(payload, _modality_configs(), CAMERA_MAP)

    for source, target in CAMERA_MAP.items():
        video = observation["video"][target]
        assert video.shape == (1, 1, 2, 3, 3)
        assert video.dtype == np.uint8
        np.testing.assert_array_equal(video[0, 0], payload[source])

    expected_state = np.asarray(payload["state"], dtype=np.float32)
    for key, part_slice in serve_gr00t_act.CANONICAL_SLICES.items():
        state_part = observation["state"][key]
        assert state_part.shape == (1, 1, part_slice.stop - part_slice.start)
        assert state_part.dtype == np.float32
        np.testing.assert_array_equal(state_part[0, 0], expected_state[part_slice])

    language_key = _modality_configs()["language"].modality_keys[0]
    assert observation["language"] == {language_key: [[payload["instruction"]]]}


def test_build_observation_rejects_wrong_state_shape() -> None:
    """The wire state must be exactly one packed 14-D vector."""
    payload = _valid_payload()
    payload["state"] = [0.0] * 13
    with pytest.raises(ValueError, match=r"state must have shape \(14,\)"):
        serve_gr00t_act._build_observation(payload, _modality_configs(), CAMERA_MAP)


@pytest.mark.parametrize("field", ["state", "top_cam", "instruction"])
def test_build_observation_names_missing_request_field(field: str) -> None:
    """Missing state, camera, and instruction errors identify the wire field."""
    payload = _valid_payload()
    del payload[field]
    with pytest.raises(ValueError, match=rf"missing field '{field}'"):
        serve_gr00t_act._build_observation(payload, _modality_configs(), CAMERA_MAP)


def test_pack_actions_scatters_scrambled_keys_by_name() -> None:
    """Returned dictionary order cannot change the canonical packed joint order."""
    actions = {
        "right_gripper": _action_array(2, 1, 40.0),
        "left_arm": _action_array(2, 6, 10.0),
        "right_arm": _action_array(2, 6, 30.0),
        "left_gripper": _action_array(2, 1, 20.0),
    }
    packed = serve_gr00t_act._pack_actions(actions, CANONICAL_KEYS)

    np.testing.assert_array_equal(packed[:, 0:6], np.full((2, 6), 10.0))
    np.testing.assert_array_equal(packed[:, 6:7], np.full((2, 1), 20.0))
    np.testing.assert_array_equal(packed[:, 7:13], np.full((2, 6), 30.0))
    np.testing.assert_array_equal(packed[:, 13:14], np.full((2, 1), 40.0))


def test_pack_actions_rejects_inconsistent_chunk_lengths() -> None:
    """Every returned body part must contain the same action horizon."""
    actions = {
        "left_arm": _action_array(2, 6, 10.0),
        "left_gripper": _action_array(2, 1, 20.0),
        "right_arm": _action_array(3, 6, 30.0),
        "right_gripper": _action_array(2, 1, 40.0),
    }
    with pytest.raises(ValueError, match="inconsistent chunk lengths"):
        serve_gr00t_act._pack_actions(actions, CANONICAL_KEYS)


def test_pack_actions_rejects_missing_key() -> None:
    """A response missing any configured action key is rejected."""
    actions = {
        "left_arm": _action_array(2, 6, 10.0),
        "left_gripper": _action_array(2, 1, 20.0),
        "right_arm": _action_array(2, 6, 30.0),
    }
    with pytest.raises(ValueError, match="returned action keys must equal"):
        serve_gr00t_act._pack_actions(actions, CANONICAL_KEYS)


def test_pack_actions_rejects_extra_key() -> None:
    """A response containing an unconfigured action key is rejected."""
    actions = {
        "left_arm": _action_array(2, 6, 10.0),
        "left_gripper": _action_array(2, 1, 20.0),
        "right_arm": _action_array(2, 6, 30.0),
        "right_gripper": _action_array(2, 1, 40.0),
        "torso": _action_array(2, 1, 50.0),
    }
    with pytest.raises(ValueError, match="returned action keys must equal"):
        serve_gr00t_act._pack_actions(actions, CANONICAL_KEYS)
