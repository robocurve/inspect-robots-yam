"""Tests for lazy I2RT loading and declared runtime requirements."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

from inspect_robots_yam._i2rt import I2RT_INSTALL_COMMAND, _load_i2rt
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.policy import MolmoAct2Policy


def test_load_i2rt_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_get_robot = object()
    fake_gripper_type = object()
    modules = {
        "i2rt": ModuleType("i2rt"),
        "i2rt.robots": ModuleType("i2rt.robots"),
        "i2rt.robots.get_robot": ModuleType("i2rt.robots.get_robot"),
        "i2rt.robots.utils": ModuleType("i2rt.robots.utils"),
    }
    modules["i2rt"].__path__ = []
    modules["i2rt.robots"].__path__ = []
    modules["i2rt.robots.get_robot"].get_yam_robot = fake_get_robot
    modules["i2rt.robots.utils"].GripperType = fake_gripper_type
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert _load_i2rt() == (fake_get_robot, fake_gripper_type)


@pytest.mark.parametrize("missing_name", ["i2rt", "i2rt.robots"])
def test_load_i2rt_guides_missing_driver(
    monkeypatch: pytest.MonkeyPatch, missing_name: str
) -> None:
    real_import = builtins.__import__

    def missing_i2rt(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("i2rt"):
            raise ModuleNotFoundError(f"No module named {missing_name!r}", name=missing_name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_i2rt)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_i2rt()
    assert I2RT_INSTALL_COMMAND in str(exc_info.value)


def test_load_i2rt_preserves_nameless_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__
    original = ModuleNotFoundError("import machinery gave no module name", name=None)

    def nameless_failure(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("i2rt"):
            raise original
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", nameless_failure)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_i2rt()
    assert exc_info.value is original


def test_load_i2rt_preserves_other_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__
    original = ModuleNotFoundError("No module named 'broken_driver_dep'", name="broken_driver_dep")

    def broken_i2rt(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("i2rt"):
            raise original
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_i2rt)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_i2rt()
    assert exc_info.value is original


@pytest.mark.parametrize("component", [YAMEmbodiment, MolmoAct2Policy])
def test_runtime_requirements_use_top_level_modules_and_nonempty_commands(component: type) -> None:
    requirements = component.RUNTIME_REQUIREMENTS

    assert requirements
    assert all(isinstance(module, str) and "." not in module for module in requirements)
    assert all(isinstance(command, str) and command for command in requirements.values())


def test_yam_i2rt_runtime_requirement_uses_install_command() -> None:
    assert YAMEmbodiment.RUNTIME_REQUIREMENTS["i2rt"] == I2RT_INSTALL_COMMAND
