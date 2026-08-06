"""Wizard-config filtering and YAM owner-gate behavior."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from inspect_robots_yam import _user_config
from inspect_robots_yam._user_config import YamDefaults, load_yam_defaults
from inspect_robots_yam.embodiment import YAMEmbodiment


def _env(tmp_path: Path) -> dict[str, str]:
    return {"XDG_CONFIG_HOME": str(tmp_path)}


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "inspect-robots" / "config.ini"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_yam_defaults(_env(tmp_path)) == YamDefaults({}, None, None)


def test_missing_args_section_is_empty(tmp_path: Path) -> None:
    _write_config(tmp_path, "[defaults]\nembodiment = yam_arms\n")

    assert load_yam_defaults(_env(tmp_path)) == YamDefaults({}, None, None)


def test_args_without_an_owner_are_empty(tmp_path: Path) -> None:
    _write_config(tmp_path, "[embodiment.args]\nleft_channel = can_left\n")

    assert load_yam_defaults(_env(tmp_path)) == YamDefaults({}, None, None)


@pytest.mark.parametrize(
    "factory",
    [
        None,
        object,
        lambda: object(),
    ],
    ids=["unknown", "non-yam-class", "function-factory"],
)
def test_unknown_and_non_yam_owners_are_empty(
    factory: Callable[..., Any] | type[Any] | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        "[defaults]\nembodiment = foreign\n[embodiment.args]\nleft_channel = can_left\n",
    )
    mapping = {} if factory is None else {"foreign": factory}
    monkeypatch.setattr(_user_config, "registered", lambda _kind: mapping)

    assert load_yam_defaults(_env(tmp_path)) == YamDefaults({}, None, None)


def test_yam_arms_filters_and_coerces_device_values(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "[defaults]\nembodiment = yam_arms\n"
        "[embodiment.args]\n"
        "top_cam_device = 0\n"
        "left_depth_serial = 838212071234\n"
        "left_channel = can_left\n"
        "right_channel = none\n"
        "control_hz = 20\n",
    )

    result = load_yam_defaults(_env(tmp_path))

    assert result == YamDefaults(
        args={
            "top_cam_device": "0",
            "left_depth_serial": "838212071234",
            "left_channel": "can_left",
        },
        source=str(path),
        owner="yam_arms",
    )


def test_only_keys_outside_the_device_filter_yield_an_empty_result(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "[defaults]\nembodiment = yam_arms\n[embodiment.args]\ncontrol_hz = 20\n",
    )

    assert load_yam_defaults(_env(tmp_path)) == YamDefaults({}, None, None)


def test_yam_subclass_registered_under_another_name_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlternateYam(YAMEmbodiment):
        pass

    path = _write_config(
        tmp_path,
        "[defaults]\nembodiment = custom_yam\n[embodiment.args]\nright_channel = can_right\n",
    )
    monkeypatch.setattr(
        _user_config,
        "registered",
        lambda _kind: {"custom_yam": AlternateYam},
    )

    assert load_yam_defaults(_env(tmp_path)) == YamDefaults(
        {"right_channel": "can_right"},
        str(path),
        "custom_yam",
    )


def test_malformed_config_propagates_system_exit(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "[defaults\nembodiment = yam_arms\n")

    with pytest.raises(SystemExit, match=f"error in {re.escape(str(path))}"):
        load_yam_defaults(_env(tmp_path))
