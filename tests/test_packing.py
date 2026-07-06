"""Tests for the pure 14-D bimanual packing module."""

from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_yam import packing


def test_constants() -> None:
    assert packing.ARM_WIDTH == 7
    assert packing.TOTAL_DIM == 14
    assert slice(0, 7) == packing.LEFT
    assert slice(7, 14) == packing.RIGHT
    assert packing.STATE_KEY == "joint_pos"


def test_state_spec_keys_match_state_key() -> None:
    # Single field keeps StateSpec.keys consistent with the declared state_keys.
    assert packing.STATE_SPEC.keys == frozenset({"joint_pos"})


def test_state_spec_derives_field_key_from_argument() -> None:
    spec = packing.state_spec("proprio")
    assert spec.keys == frozenset({"proprio"})
    assert spec.fields[0].shape == (packing.TOTAL_DIM,)


def test_validate_dim_accepts_correct_length() -> None:
    out = packing.validate_dim(list(range(14)))
    assert out.shape == (14,)
    assert out.dtype == np.float64


def test_validate_dim_accepts_custom_length() -> None:
    out = packing.validate_dim(np.zeros((7,)), n=7)
    assert out.shape == (7,)


def test_validate_dim_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected a 14-D vector"):
        packing.validate_dim(np.zeros(8))


def test_validate_dim_rejects_2d_same_size() -> None:
    # A (7, 2) array has 14 elements but flattening it would interleave-scramble
    # the left/right arm packing — it must be rejected, not reshaped.
    with pytest.raises(ValueError, match=r"got shape \(7, 2\)"):
        packing.validate_dim(np.zeros((7, 2)))


def test_pack_concatenates() -> None:
    left = np.arange(7, dtype=float)
    right = np.arange(7, 14, dtype=float)
    out = packing.pack(left, right)
    assert np.array_equal(out, np.arange(14, dtype=float))


def test_pack_rejects_wrong_arm_dim() -> None:
    with pytest.raises(ValueError, match="expected a 7-D vector"):
        packing.pack(np.zeros(6), np.zeros(7))


def test_split_roundtrips_pack() -> None:
    vec = np.arange(14, dtype=float)
    left, right = packing.split(vec)
    assert np.array_equal(left, np.arange(7, dtype=float))
    assert np.array_equal(right, np.arange(7, 14, dtype=float))
    assert np.array_equal(packing.pack(left, right), vec)


def test_split_returns_copies() -> None:
    vec = np.zeros(14)
    left, right = packing.split(vec)
    left[0] = 99.0
    right[0] = 88.0
    assert vec[0] == 0.0 and vec[7] == 0.0  # originals untouched


def test_split_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected a 14-D vector"):
        packing.split(np.zeros(13))
