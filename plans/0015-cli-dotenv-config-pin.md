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

**Tech stack:** stdlib only. pytest with `tmp_path` + `monkeypatch.chdir`,
mirroring the existing health CLI tests.

## Global Constraints

- Gates (all blocking): `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run mypy` (strict), `uv run pytest --cov` at **100% coverage**
  (branch coverage on).
- Repo root is the `wt-yam-health-dotenv` worktree at
  `~/robocurve/wt-yam-health-dotenv`; run everything via `uv run ...` there.
- Existing tests pass untouched, in particular every test that passes an
  explicit `env=` to `main` must be unaffected by design, not by fixture
  surgery.
- Never mutate the developer's real environment from tests: monkeypatch
  `os.environ` around the `env=None` paths.
- Commit messages: imperative, scoped; reference #107.

## Reference: current wiring (main @ a82f1e8)

- `health.py:352` — `def main(argv=None, *, env=None, ...)`; config
  resolution at 407-410: `load_yam_defaults(os.environ if env is None else
  env)` guarded by `--no-config`. The stderr line
  `devices: from <source> (embodiment <owner>)` at 429-433 names the config
  file that won, which makes the fix observable end-to-end.
- `hold_check.py:160` — `def main(...)`; same resolution shape at 212.
- core `src/inspect_robots/_dotenv.py:45` — `init_dotenv(environ, path=None)`:
  reads CWD `.env`, `environ.setdefault` per key. Core `cli.py:main` calls it
  first, before argument parsing.
- The plugin has no dotenv usage today (`grep -rn dotenv src/` is empty).

## Task 1: health CLI honors the CWD .env

- [ ] **Step 1: failing tests.** In a `tmp_path` cwd containing `.env` with
  `INSPECT_ROBOTS_CONFIG=<tmp config.ini>` (write a minimal wizard-style
  config whose `[embodiment.args]`-equivalent section carries a camera
  device; copy the shape from existing `load_yam_defaults` tests), invoke
  `health.main` with `env=None` (monkeypatching `os.environ` to lack the
  var) and assert the pinned config's devices are used (the
  `devices: from <tmp config.ini>` stderr line). Second test: exported
  `INSPECT_ROBOTS_CONFIG` in the (monkeypatched) environment beats a
  conflicting `.env` line. Third test: explicit `env=` mapping bypasses
  dotenv entirely (no `.env` read even when one exists in cwd).
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

## Task 3: docs sweep

- [ ] **Step 1:** README/docs mention of the health or holdcheck CLIs:
  state that both honor the working directory's `.env`
  (`INSPECT_ROBOTS_CONFIG` pin included) exactly like `inspect-robots`
  itself, and that `--no-config` remains the bypass.
- [ ] **Step 2: gates green, commit.**
