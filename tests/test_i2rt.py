"""Tests for lazy I2RT loading and declared conformance (runtime requirements, device slots)."""

from __future__ import annotations

import builtins
import logging
import sys
import threading
import time
from types import ModuleType

import pytest

import inspect_robots_yam._i2rt as i2rt_module
from inspect_robots_yam._i2rt import (
    I2RT_INSTALL_COMMAND,
    _load_i2rt,
    _load_i2rt_kinematics,
    close_robot_safely,
)
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.policy import MolmoAct2Policy


class FakeMotorInterface:
    def __init__(self, events: list[str]) -> None:
        self.closed = False
        self.events = events
        self.send_after_close: list[ValueError] = []

    def send(self) -> None:
        if self.closed:
            error = ValueError("file descriptor cannot be a negative integer (-1)")
            self.send_after_close.append(error)
            raise error

    def close(self) -> None:
        self.events.append("iface_close")
        self.closed = True


class FakeChain:
    def __init__(self, *, bound_target: bool = True, ignores_running: bool = False) -> None:
        self.running = True
        self.events: list[str] = []
        self.errors: list[ValueError] = []
        self.motor_interface = FakeMotorInterface(self.events)
        self._started = threading.Event()
        self._release = threading.Event()
        self._ignores_running = ignores_running

        if bound_target:
            target = self._control_loop
        else:

            def target() -> None:
                self._control_loop()

        self.control_thread = threading.Thread(
            target=target,
            name="fake-i2rt-control",
            daemon=ignores_running,
        )
        self.control_thread.start()
        assert self._started.wait(timeout=1.0)

    def _control_loop(self) -> None:
        self._started.set()
        try:
            while self.running or (self._ignores_running and not self._release.is_set()):
                try:
                    self.motor_interface.send()
                except ValueError as exc:
                    self.errors.append(exc)
                    if not self._ignores_running:
                        break
                time.sleep(0.001)
        finally:
            self.events.append("loop_exit")

    def release(self) -> None:
        self._release.set()

    def stop(self) -> None:
        self.running = False
        self.release()
        self.control_thread.join(timeout=0.2)


class FakeRobot:
    def __init__(self, chain: FakeChain) -> None:
        self.motor_chain = chain

    def close(self) -> None:
        self.motor_chain.running = False
        self.motor_chain.motor_interface.close()


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


def test_load_i2rt_kinematics_success(monkeypatch: pytest.MonkeyPatch) -> None:
    symbols = [object() for _ in range(5)]
    modules = {
        "i2rt": ModuleType("i2rt"),
        "i2rt.robots": ModuleType("i2rt.robots"),
        "i2rt.robots.kinematics": ModuleType("i2rt.robots.kinematics"),
        "i2rt.robots.utils": ModuleType("i2rt.robots.utils"),
        "mink": ModuleType("mink"),
    }
    modules["i2rt"].__path__ = []
    modules["i2rt.robots"].__path__ = []
    modules["i2rt.robots.kinematics"].Kinematics = symbols[0]
    modules["i2rt.robots.utils"].ArmType = symbols[1]
    modules["i2rt.robots.utils"].GripperType = symbols[2]
    modules["i2rt.robots.utils"].combine_arm_and_gripper_xml = symbols[3]
    modules["mink"].NoSolutionFound = symbols[4]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert _load_i2rt_kinematics() == tuple(symbols)


@pytest.mark.parametrize("missing_name", ["i2rt", "i2rt.robots"])
def test_load_i2rt_kinematics_guides_missing_driver(
    monkeypatch: pytest.MonkeyPatch, missing_name: str
) -> None:
    real_import = builtins.__import__

    def missing_i2rt(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("i2rt"):
            raise ModuleNotFoundError(f"No module named {missing_name!r}", name=missing_name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_i2rt)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_i2rt_kinematics()
    assert I2RT_INSTALL_COMMAND in str(exc_info.value)


@pytest.mark.parametrize("missing_name", [None, "mink"])
def test_load_i2rt_kinematics_preserves_non_i2rt_import_failures(
    monkeypatch: pytest.MonkeyPatch, missing_name: str | None
) -> None:
    real_import = builtins.__import__
    original = ModuleNotFoundError("optional solver dependency is broken", name=missing_name)

    def broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("i2rt"):
            modules = sys.modules
            if "i2rt" not in modules:
                modules["i2rt"] = ModuleType("i2rt")
                modules["i2rt"].__path__ = []
                modules["i2rt.robots"] = ModuleType("i2rt.robots")
                modules["i2rt.robots"].__path__ = []
            raise original
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_i2rt_kinematics()
    assert exc_info.value is original


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


def test_i2rt_install_command_carries_ruckig_build_constraint() -> None:
    """The remedy must pre-pin scikit-build-core, or ruckig's sdist fails to build (#47)."""
    assert "scikit-build-core<0.10" in I2RT_INSTALL_COMMAND
    assert "--build-constraints" in I2RT_INSTALL_COMMAND
    assert '"i2rt @ git+https://github.com/i2rt-robotics/i2rt@' in I2RT_INSTALL_COMMAND


def test_close_robot_safely_joins_control_thread_before_socket_close() -> None:
    chain = FakeChain()
    robot = FakeRobot(chain)

    try:
        close_robot_safely(robot)

        assert not chain.control_thread.is_alive()
        assert chain.motor_interface.closed
        assert not chain.motor_interface.send_after_close
        assert not chain.errors

        robot.close()
        assert chain.events.count("iface_close") == 1
    finally:
        chain.stop()


def test_close_robot_safely_orders_loop_exit_before_interface_close() -> None:
    chain = FakeChain()

    try:
        close_robot_safely(FakeRobot(chain))

        assert chain.events.index("loop_exit") < chain.events.index("iface_close")
    finally:
        chain.stop()


def test_close_robot_safely_without_motor_chain_calls_robot_close() -> None:
    class RobotWithoutChain:
        motor_chain = None

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    robot = RobotWithoutChain()

    close_robot_safely(robot)

    assert robot.closed


def test_close_robot_safely_uses_grace_period_without_discoverable_thread() -> None:
    chain = FakeChain(bound_target=False)

    try:
        close_robot_safely(FakeRobot(chain))
        chain.control_thread.join(timeout=0.2)

        assert chain.motor_interface.closed
        assert not chain.control_thread.is_alive()
        assert not chain.motor_interface.send_after_close
    finally:
        chain.stop()


def test_close_robot_safely_warns_and_closes_interface_after_join_timeout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    chain = FakeChain(ignores_running=True)
    monkeypatch.setattr(i2rt_module, "_CONTROL_THREAD_JOIN_TIMEOUT", 0.01)

    try:
        with caplog.at_level(logging.WARNING, logger=i2rt_module.__name__):
            close_robot_safely(FakeRobot(chain))

        assert chain.motor_interface.closed
        assert "fake-i2rt-control" in caplog.text
    finally:
        chain.stop()


def test_device_slots_cover_channels_and_cameras_with_valid_config_args() -> None:
    """Every declared slot writes a real YamConfig field, grouped per hardware constraint."""
    from dataclasses import fields

    from inspect_robots.conformance import DEVICE_KINDS, DeviceSlot, device_slots

    slots = device_slots(YAMEmbodiment)

    assert slots == YAMEmbodiment.DEVICE_SLOTS  # defensive reader keeps every entry
    config_fields = {f.name for f in fields(YamConfig)}
    assert all(isinstance(slot, DeviceSlot) for slot in slots)
    assert all(slot.arg in config_fields for slot in slots)
    assert all(slot.kind in DEVICE_KINDS for slot in slots)
    assert {slot.arg for slot in slots if slot.group == "arms"} == {
        "left_channel",
        "right_channel",
    }
    assert {slot.arg for slot in slots if slot.group == "cameras"} == {
        "top_cam_device",
        "left_cam_device",
        "right_cam_device",
    }
