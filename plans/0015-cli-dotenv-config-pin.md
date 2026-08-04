# CLI dotenv config-pin Implementation Plan

> **For agentic workers:** Implement task-by-task in order; each task is
> test-first and ends in its own commit. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** `inspect-robots-yam-health` and `inspect-robots-yam-holdcheck`
must honor the working directory's `.env` the same way the core
`inspect-robots` CLI does, so a per-rig directory whose `.env` pins
`INSPECT_ROBOTS_CONFIG` gets that rig's wizard-configured devices instead of
silently falling through to the XDG config (another rig's cameras and CAN
channels). Closes #107.

**Architecture:** mirror core `cli.py:main`, which calls
`init_dotenv(os.environ)` before parsing. Both yam entry points resolve
config through `load_yam_defaults(os.environ if env is None else env)`;
the dotenv load applies exactly when `env is None` (a real invocation), so
the injected-`env` test seam keeps working untouched. `init_dotenv` uses
`setdefault` semantics, so an exported `INSPECT_ROBOTS_CONFIG` still beats
the `.env` line, identical to core CLI precedence. `preflight` is out of
scope: it never resolves config. The import is
`from inspect_robots._dotenv import init_dotenv`; the module is private to
core but the plugin already tracks core minor-for-minor, and duplicating
the parser would drift (state the rationale in a comment at the import).

**Dependency floor (load-bearing):** `INSPECT_ROBOTS_CONFIG` does not exist
in core before 0.37 (`config_path` in 0.31-0.36 derives only from
`XDG_CONFIG_HOME`/`HOME`), and the current pin is `inspect-robots>=0.31`
with `uv.lock` resolving exactly 0.31.0 — against which this fix would load
a variable core silently ignores. Task 0 bumps the floor to
`inspect-robots>=0.38` (0.38 rather than 0.37 for `--config` parity with
the per-rig workflow in #107) and re-locks. `init_dotenv` itself is
signature-identical from 0.31 through 0.38, so the import needs no guard.

**Tech stack:** stdlib only. pytest with `tmp_path` + `monkeypatch.chdir`
(new to this suite: no existing test changes cwd, which is exactly the
hermeticity gap Task 0 closes).

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict), `uv run pytest --cov` at **100% coverage**
  (branch coverage on).
- Repo root is the `wt-yam-health-dotenv` worktree at
  `~/robocurve/wt-yam-health-dotenv`; run everything via `uv run ...` there.
- Existing tests pass untouched, in particular every test that passes an
  explicit `env=` to `main` must be unaffected by design, not by fixture
  surgery. (The autouse isolation fixture is NOT covered by this
  constraint: Task 0 extends it deliberately, because ~18 existing tests
  call `main` with `env=None` and currently pass only by environmental
  luck — they run from the repo root, where a developer-local `.env` is an
  invited pattern per `.env.example`.)
- Never mutate the developer's real environment from tests. Beware that
  `monkeypatch.delenv(..., raising=False)` on an absent key records
  nothing to restore, after which `init_dotenv`'s `setdefault` writes into
  the live `os.environ` and leaks past the test. New tests must swap in a
  copied mapping (e.g. `monkeypatch.setattr(os, "environ",
  dict(os.environ))`-style, or monkeypatch.setenv on keys that exist)
  before invoking the `env=None` path.
- Commit messages: imperative, scoped; reference #107.

## Reference: current wiring (main @ a82f1e8)

- `health.py:352` — `def main(argv=None, *, env=None, run=...)`; config
  resolution at 406-410: `load_yam_defaults(os.environ if env is None else
  env)` guarded by `--no-config`. The stderr line
  `devices: from <source> (embodiment <owner>)` at 429-433 names the config
  file that won (printed whenever the config contributed keys the flags did
  not override), which makes the fix observable end-to-end.
- `tests/conftest.py:47-50` — autouse `isolate_user_config` fixture sets
  only `XDG_CONFIG_HOME`, which `INSPECT_ROBOTS_CONFIG` outranks in core
  0.37+; no existing test changes cwd.
- Existing hardware-free invocation pattern: `main([...,"--skip-motors"],
  env=..., run=capture)` (`tests/test_health.py:564`) — the `run=` stub is
  what keeps config-driven tests off real devices.
- `hold_check.py:160` — `def main(...)`; same resolution shape at 212.
- core `src/inspect_robots/_dotenv.py:45` — `init_dotenv(environ, path=None)`:
  reads CWD `.env`, `environ.setdefault` per key. Core `cli.py:main` calls it
  first, before argument parsing.
- The plugin has no dotenv usage today (`grep -rn dotenv src/` is empty).

## Task 0: dependency floor and test hermeticity

- [ ] **Step 1: bump the core floor.** `pyproject.toml`: `inspect-robots>=
  0.38`; run `uv lock` and `uv sync` so the worktree venv actually carries
  a core that honors `INSPECT_ROBOTS_CONFIG`. Confirm
  `python -c "from inspect_robots.defaults import _ENV_CONFIG"` succeeds.
- [ ] **Step 2: harden the autouse fixture.** Extend `isolate_user_config`
  (`tests/conftest.py:47-50`) with `monkeypatch.chdir(tmp_path)` so no
  test — existing `env=None` callers included — can see a developer-local
  repo-root `.env`. Run the full suite; fix any test that turns out to
  depend on the repo-root cwd (none are known; if one appears, it was
  latently broken and gets a `tmp_path`-relative fixture).
- [ ] **Step 3: gates green, commit** (message: floor bump + hermeticity,
  reference #107).

## Task 1: health CLI honors the CWD .env

- [ ] **Step 1: failing tests.** In a `tmp_path` cwd containing `.env` with
  `INSPECT_ROBOTS_CONFIG=<tmp config.ini>` (write a minimal wizard-style
  config whose `[embodiment.args]`-equivalent section carries a camera
  device; copy the shape from existing `load_yam_defaults` tests), invoke
  `health.main` with `env=None`, a `run=` capture stub, and
  `--skip-motors` (mirroring `tests/test_health.py:564` so no hardware is
  touched) and assert the pinned config's devices are used (the
  `devices: from <tmp config.ini>` stderr line). The `env=None` path must
  run against a swapped-in copy of `os.environ` per the Global Constraint.
  Second test: `INSPECT_ROBOTS_CONFIG` already present in the (copied)
  environment beats a conflicting `.env` line. Third test: explicit `env=`
  mapping bypasses dotenv entirely (no `.env` read even when one exists in
  cwd).
- [ ] **Step 2: implement.** At the top of `health.main`, when `env is
  None`, call `init_dotenv(os.environ)` (import with the rationale
  comment). Update the module docstring's config paragraph to say the CWD
  `.env` is honored like the core CLI.
- [ ] **Step 3: gates green, commit.**

## Task 2: holdcheck CLI honors the CWD .env

- [ ] **Step 1: failing test.** Same `.env`-pin shape as Task 1 against
  `hold_check.main`'s `env is None` path, asserting the pinned config's
  values are consulted.
- [ ] **Step 2: implement.** Same two lines, same docstring touch.
- [ ] **Step 3: gates green, commit.**

## Task 3: docs and changelog sweep

- [ ] **Step 1:** README/docs mention of the health or holdcheck CLIs:
  state that both honor the working directory's `.env`
  (`INSPECT_ROBOTS_CONFIG` pin included) exactly like `inspect-robots`
  itself, and that `--no-config` remains the bypass.
- [ ] **Step 2:** `CHANGELOG.md` Unreleased entries (Keep a Changelog
  style, matching the existing #102/#99/#95 entries): the `.env` fix
  referencing #107, and the core floor bump to >=0.38 as its own
  user-visible line.
- [ ] **Step 3: gates green, commit.**
