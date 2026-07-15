# 0002 — Rest pose on close (don't drop the arms)

**Problem.** `YAMEmbodiment.close()` releases the i2rt driver handles, which
zeroes motor torque. Wherever the arms happen to be, they fall. Operators want
the arms parked somewhere safe *before* torque-off.

**Decision (approved 2026-07-07).** Explicit config: `YamConfig.rest_pose`
(14 values, gripper slots normalized 0–1 — the same units every `_send()`
accepts and `home_pose` uses) plus `YamConfig.rest_secs` (default 3.0, > 0).

On `close()`, when a driver is connected **and** `rest_pose` is set:

1. Read the current joints (gripper slots normalized, as in `_observe`).
2. Linearly interpolate current → `rest_pose` over `rest_secs` at
   `control_hz` (fallback 10 Hz if the configured rate is 0), sending each
   waypoint through `_send()` — so the joint-limit clamp backstop and gripper
   de-normalization apply to the rest motion exactly as they do to policy
   actions — and pacing with the injected `sleep_fn`.
3. Release the driver in a `finally:` — a driver fault mid-motion must never
   leave the handles (and CAN torque state) dangling. The exception still
   propagates after cleanup.

`rest_pose=None` (default) keeps today's behavior bit-for-bit. `close()`
without a connected driver stays a no-op.

> **Amended by PR #36 (2026-07-14):** the `rest_pose=None` resolution above no
> longer holds. Without a configured `rest_pose`, `close()` now parks at the
> pose captured at the first `reset()` after connecting (before any commanded
> motion); `rest_pose` remains an explicit override. Release-in-place only
> happens when no pose was ever captured.

> **Amended by PR #44 (2026-07-14):** `rest_pose` now defaults to the factory
> all-zero pose. Setting `rest_pose=None` opts out and retains the captured-init
> fallback from PR #36. Parking, whether factory, explicit, or captured-init,
> only happens after a pose was captured; a connection fault before capture
> releases in place.

> **Amended by plan 0007 (2026-07-15):** the factory rest tuple's gripper
> slots are now open (1.0) and the park target equals the joint factory home.

**Out of scope.** Capture tooling lives in the operator's run harness
(`~/run_molmoact_yam.py --capture-rest-pose` writes `~/.yam_rest_pose.json`),
not in this package: the package has no camera/TTY/file conventions, and the
harness already owns operator UX.

**Testing.** All via the injected fake driver / `sleep_fn` / `clock` (no
hardware): waypoint count = `rest_secs * control_hz`; the sequence is
monotonic and ends exactly at the de-normalized clamped `rest_pose`; the
driver is closed afterwards; no rest motion without `rest_pose`; close before
connect is a no-op; a command fault mid-motion still closes the driver;
config validation (length, positivity). Gates stay at ruff / mypy-strict /
100 % coverage.
