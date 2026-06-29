"""Operator-in-the-loop confirmation for real hardware runs.

A real kitchen has no privileged success oracle, so the human operator decides.
All stdin/stdout goes through injectable ``input_fn`` / ``output_fn`` so tests
drive these paths without a real terminal. The one genuinely TTY-bound piece —
the non-blocking "operator pressed end" poll — is isolated in
:func:`default_poll_end`, which is excluded from coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Affirmative answers (case-insensitive) for the end-of-episode success prompt.
_AFFIRMATIVE = frozenset({"y", "yes", "1", "true", "success", "pass"})


@dataclass
class OperatorIO:
    """Console I/O for operator prompts, with injectable functions for testing."""

    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print

    def wait_ready(self, prompt: str = "Position the scene, then press Enter to start...") -> None:
        """Block until the operator confirms the scene is set up."""
        self.input_fn(prompt)

    def confirm_success(self, prompt: str = "Did the robot succeed? [y/N]: ") -> bool:
        """Return the operator's success verdict (affirmative answers → True)."""
        answer = self.input_fn(prompt)
        return answer.strip().lower() in _AFFIRMATIVE


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
