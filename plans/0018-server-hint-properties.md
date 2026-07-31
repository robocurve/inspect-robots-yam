# 0018 — Expose `server_url`/`remedy` so core's connection hint names the endpoint

Yam side of inspect-robots #219 (core PR #221, released in core v0.29.0).
Closes #97.

## Problem

Since core v0.29.0, a connection failure inside a policy gets an actionable
hint appended to the `PolicyError`. Core reads two optional duck-typed
attributes off the policy instance via `getattr`: `server_url` (truthy → the
hint names the exact endpoint and says to start the server) and `remedy`
(truthy → appended verbatim as an extra `hint:` line). `ActServerPolicy`
keeps its config private (`self._cfg`), so both `getattr`s return `None` and
molmoact2 users see only the generic fallback — "a backend it depends on may
be down or unreachable" — with no URL and no recovery step, even though the
config knows both.

## Changes

- `ActServerConfig.remedy: str = ""` — free-text recovery instruction shown
  as its own `hint:` line on connection failures (e.g. the rig's launch
  script: `-P remedy='run ~/robocurve/molmoact2/run_yam.sh'`). Plain
  dataclass field, so `from_kwargs` accepts it like any other; the empty
  default is falsy, so out-of-the-box behavior gains only the URL line. No
  validation: any string is a valid remedy, and `""` means "no remedy line".
- `ActServerPolicy.server_url` (read-only property) → `self._cfg.server_url`.
  Deliberately the config field, not the joined `url`: connection refusal is
  a host:port-level condition, the hint should name what the operator checks,
  and the name mirrors the config key users already set (`-P server_url=…`).
- `ActServerPolicy.remedy` (read-only property) → `self._cfg.remedy`.
- Properties never raise (attribute reads on a frozen dataclass); core guards
  against raising properties anyway, but we don't rely on it.
- No `pyproject.toml` floor bump: on cores older than v0.29.0 the properties
  are inert extra attributes, and nothing here imports new core API.
- README: in the `server_url` config paragraph, note the two attributes feed
  core's connection-failure hint; in the "nothing moves until the server is
  up" paragraph, mention the error now names the endpoint. CHANGELOG under
  `Unreleased`/`Added`, referencing #97 and core #219.

## Tests

- `server_url` property equals the config default and follows a custom
  `-P`-style flat kwarg (`from_kwargs` path).
- `remedy` defaults to `""` (falsy → core skips the line) and follows both
  the explicit-config and flat-kwargs construction paths.
- Contract test mirroring core's exact read:
  `getattr(policy, "server_url", None)` / `getattr(policy, "remedy", None)`
  return the configured values (guards against ever renaming the properties
  out from under core's duck-typing).
- Properties are read-only: assignment raises `AttributeError`.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
`uv run pytest --cov` (100%, branch).
