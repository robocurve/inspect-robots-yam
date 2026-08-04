"""Operator-in-the-loop I/O for real hardware runs.

The operator readies scenes and signals end-of-episode; the success verdict
itself (with optional grader notes) is collected afterwards by the framework's
operator prompt, so this module owns no grading UI. All stdin/stdout goes
through injectable ``input_fn`` / ``output_fn`` so tests drive these paths
without a real terminal. The genuinely TTY-bound pieces are isolated in
module-private seams or :func:`default_poll_end` and excluded from coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from inspect_robots.errors import EmbodimentFault


def _flush_stdin_fd() -> None:
    """Discard pending TTY input at the file-descriptor level before a gate."""
    import sys

    if not sys.stdin.isatty():
        return
    import os  # pragma: no cover - TTY-bound
    import select  # pragma: no cover - TTY-bound

    while select.select([sys.stdin], [], [], 0)[0]:  # pragma: no cover - TTY-bound
        if not os.read(sys.stdin.fileno(), 65536):  # pragma: no cover - TTY-bound
            break  # pragma: no cover - TTY-bound


@dataclass
class OperatorIO:
    """Console I/O for operator prompts, with injectable functions for testing."""

    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print
    flush_fn: Callable[[], None] = _flush_stdin_fd

    def wait_ready(
        self,
        prompt: str = "Position the scene, then press Enter to start...",
        *,
        drain: bool = True,
        flush_first: bool = False,
    ) -> None:
        """Block until the operator confirms the scene is set up.

        A dead stdin (no TTY: nohup, CI, a closed pipe) surfaces as
        :class:`~inspect_robots.errors.EmbodimentFault` — the framework's always-halt
        path — with instructions, instead of a bare ``EOFError`` mid-eval.

        The trailing drain protects the legacy keypress path from a stray
        buffered newline. A framework console that owns stdin passes
        ``drain=False`` because pending lines belong to the console and
        ``flush_first=True`` so stale feedback cannot stand in for a safety-gate
        confirmation. Exactly one reader discipline must consume stdin: mixing
        this buffered reader with the console's file-descriptor reads is fragile.
        """
        if flush_first:
            self.flush_fn()
        try:
            self.input_fn(prompt)
        except (EOFError, OSError) as exc:
            raise EmbodimentFault(
                "operator readiness prompt could not read stdin (no interactive "
                "terminal?). Run from a real TTY, inject an OperatorIO with a "
                "working input_fn, or set YamConfig(unattended=True) "
                "(CLI: -E unattended=true) to skip operator prompts."
            ) from exc
        # Discard anything still buffered on stdin (e.g. an extra newline from the
        # start prompt). Otherwise default_poll_end's select() reads it as an
        # "end episode" keypress on the first step and terminates immediately.
        if drain:
            _drain_stdin()


def _drain_stdin() -> None:
    """Discard buffered stdin so a stale newline (e.g. from the start prompt)
    doesn't immediately trip :func:`default_poll_end`. No-op off a real TTY."""
    import sys

    if not sys.stdin.isatty():
        return
    import select  # pragma: no cover - TTY-bound

    while select.select([sys.stdin], [], [], 0)[0]:  # pragma: no cover - TTY-bound
        sys.stdin.readline()  # pragma: no cover - TTY-bound


def stdin_interactive() -> bool:
    """True when stdin is a real TTY that can deliver operator keypresses."""
    import sys

    return sys.stdin.isatty()


def default_poll_end() -> bool:  # pragma: no cover - requires a real TTY
    """Real non-blocking check for an operator "end episode" keypress.

    Platform/TTY-specific; replaced by a scripted callable in tests. The default
    returns ``False`` so an unattended run simply runs to ``max_steps``.
    """
    import select
    import sys

    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True
