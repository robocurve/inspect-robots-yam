"""Tests for operator-in-the-loop confirmation."""

from __future__ import annotations

import sys

import pytest
from inspect_robots.errors import EmbodimentFault

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


@pytest.mark.parametrize("exc_type", [EOFError, OSError])
def test_wait_ready_dead_stdin_raises_embodiment_fault(exc_type: type[Exception]) -> None:
    def _dead_stdin(_prompt: str) -> str:
        raise exc_type("stdin closed")

    io = OperatorIO(input_fn=_dead_stdin, output_fn=lambda _m: None)
    with pytest.raises(EmbodimentFault, match="unattended=True"):
        io.wait_ready()


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
