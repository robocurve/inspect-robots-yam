"""Tests for operator-in-the-loop confirmation."""

from __future__ import annotations

from inspect_robots_yam.operator import OperatorIO, default_poll_end


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


def test_confirm_success_affirmative() -> None:
    for ans in ("y", "Yes", "1", "TRUE", "success", "pass"):
        inp, _ = _scripted([ans])
        io = OperatorIO(input_fn=inp)
        assert io.confirm_success() is True


def test_confirm_success_negative() -> None:
    for ans in ("n", "no", "", "nope"):
        inp, _ = _scripted([ans])
        io = OperatorIO(input_fn=inp)
        assert io.confirm_success() is False


def test_default_poll_end_is_callable() -> None:
    # The body is TTY-bound (pragma: no cover); just assert it's wired and callable.
    assert callable(default_poll_end)
