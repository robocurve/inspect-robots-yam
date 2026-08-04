"""Tests for operator-in-the-loop confirmation."""

from __future__ import annotations

import sys

import pytest
from inspect_robots.errors import EmbodimentFault

import inspect_robots_yam.operator as operator_module
from inspect_robots_yam.operator import OperatorIO, default_poll_end, stdin_interactive


def _scripted(answers: list[str]):
    seen: list[str] = []

    def _input(prompt: str) -> str:
        seen.append(prompt)
        return answers.pop(0)

    return _input, seen


def test_wait_ready_calls_input() -> None:
    inp, seen = _scripted([""])
    io = OperatorIO(input_fn=inp, output_fn=lambda _m: None)
    io.wait_ready("ready?")
    assert seen == ["ready?"]


@pytest.mark.parametrize("drain", [None, True])
def test_wait_ready_drains_by_default_and_when_requested(
    monkeypatch: pytest.MonkeyPatch, drain: bool | None
) -> None:
    drains: list[bool] = []
    monkeypatch.setattr(operator_module, "_drain_stdin", lambda: drains.append(True))
    io = OperatorIO(input_fn=lambda _prompt: "", output_fn=lambda _message: None)

    if drain is None:
        io.wait_ready()
    else:
        io.wait_ready(drain=drain)

    assert drains == [True]


def test_wait_ready_can_leave_stdin_for_console(monkeypatch: pytest.MonkeyPatch) -> None:
    drains: list[bool] = []
    monkeypatch.setattr(operator_module, "_drain_stdin", lambda: drains.append(True))
    io = OperatorIO(input_fn=lambda _prompt: "", output_fn=lambda _message: None)

    io.wait_ready(drain=False)

    assert drains == []


def test_wait_ready_flushes_before_input_when_requested() -> None:
    calls: list[str] = []
    io = OperatorIO(
        input_fn=lambda _prompt: calls.append("input") or "",
        output_fn=lambda _message: None,
        flush_fn=lambda: calls.append("flush"),
    )

    io.wait_ready(flush_first=True)

    assert calls == ["flush", "input"]


def test_wait_ready_does_not_flush_by_default() -> None:
    flushes: list[bool] = []
    io = OperatorIO(
        input_fn=lambda _prompt: "",
        output_fn=lambda _message: None,
        flush_fn=lambda: flushes.append(True),
    )

    io.wait_ready()

    assert flushes == []


@pytest.mark.parametrize("exc_type", [EOFError, OSError])
def test_wait_ready_dead_stdin_raises_embodiment_fault(exc_type: type[Exception]) -> None:
    def _dead_stdin(_prompt: str) -> str:
        raise exc_type("stdin closed")

    io = OperatorIO(input_fn=_dead_stdin, output_fn=lambda _m: None)
    with pytest.raises(EmbodimentFault, match="unattended=True"):
        io.wait_ready()


@pytest.mark.parametrize("exc_type", [EOFError, OSError])
def test_wait_ready_dead_stdin_with_console_controls_raises_embodiment_fault(
    exc_type: type[Exception],
) -> None:
    def _dead_stdin(_prompt: str) -> str:
        raise exc_type("stdin closed")

    io = OperatorIO(input_fn=_dead_stdin, output_fn=lambda _m: None)
    with pytest.raises(EmbodimentFault, match="unattended=True"):
        io.wait_ready(drain=False, flush_first=True)


def test_flush_stdin_fd_is_noop_off_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", _Stub())
    operator_module._flush_stdin_fd()


def test_default_poll_end_is_callable() -> None:
    # The body is TTY-bound (pragma: no cover); just assert it's wired and callable.
    assert callable(default_poll_end)


def test_stdin_interactive_reports_tty_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stub:
        def __init__(self, tty: bool) -> None:
            self._tty = tty

        def isatty(self) -> bool:
            return self._tty

    monkeypatch.setattr(sys, "stdin", _Stub(True))
    assert stdin_interactive() is True
    monkeypatch.setattr(sys, "stdin", _Stub(False))
    assert stdin_interactive() is False
