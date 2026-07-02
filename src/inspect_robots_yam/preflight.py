"""Preflight: prove a YAM + MolmoAct2 pairing is compatible *before any motion*.

``inspect-robots-yam-preflight`` constructs the policy and embodiment (no hardware
connection, no server call — only their declared ``.info``), runs Inspect Robots's
compatibility check, prints the report, and exits non-zero on any error. Pass
``--task kitchenbench/pour_pasta`` to also verify every scene is realizable.

This is the "is it safe to move?" one-liner: a green preflight means dims,
control mode, cameras, and state keys all line up. (Absolute-vs-delta joint
behavior is *not* checkable here — see the README; use ``--dry-run`` plus a single
jog to confirm on hardware.)
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from inspect_robots.compat import CompatibilityReport, check_compatibility
from inspect_robots.registry import resolve
from inspect_robots.task import Task

from inspect_robots_yam.config import MolmoActConfig, YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.policy import MolmoAct2Policy

CheckFn = Callable[..., CompatibilityReport]


def build(
    yam_cfg: YamConfig | None = None,
    molmo_cfg: MolmoActConfig | None = None,
) -> tuple[MolmoAct2Policy, YAMEmbodiment]:
    """Construct the (policy, embodiment) pair without connecting to anything."""
    return MolmoAct2Policy(molmo_cfg), YAMEmbodiment(yam_cfg)


def run_preflight(
    task_name: str | None = None,
    *,
    policy: MolmoAct2Policy | None = None,
    embodiment: YAMEmbodiment | None = None,
    check: CheckFn = check_compatibility,
) -> CompatibilityReport:
    """Return the compatibility report for the YAM + MolmoAct2 pairing."""
    pol = policy if policy is not None else MolmoAct2Policy()
    emb = embodiment if embodiment is not None else YAMEmbodiment()
    task: Task | None = resolve("task", task_name) if task_name else None
    return check(pol, emb, task)


def _format_human(report: CompatibilityReport, *, dry_run: bool) -> str:
    lines = []
    if report.ok:
        lines.append("OK: policy and embodiment are compatible.")
    else:
        lines.append("INCOMPATIBLE:")
    for issue in report.errors:
        lines.append(f"  ERROR   [{issue.code}] {issue.message}")
    for issue in report.warnings:
        lines.append(f"  WARNING [{issue.code}] {issue.message}")
    if dry_run:
        lines.append("(dry-run) No motion will be commanded.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, run: CheckFn | None = None) -> int:
    """CLI entry point. Returns a process exit code (non-zero on compat errors)."""
    parser = argparse.ArgumentParser(prog="inspect-robots-yam-preflight")
    parser.add_argument(
        "--task", default=None, help="optional task name to check scene realizability"
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument("--dry-run", action="store_true", help="affirm no motion will be commanded")
    args = parser.parse_args(argv)

    run_fn: Callable[..., CompatibilityReport] = run if run is not None else run_preflight
    report = run_fn(args.task)

    if args.json:
        payload = {
            "ok": report.ok,
            "errors": [{"code": i.code, "message": i.message} for i in report.errors],
            "warnings": [{"code": i.code, "message": i.message} for i in report.warnings],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_human(report, dry_run=args.dry_run))
    return 1 if report.errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
