# `inspect_robots_yam` package — module map

Two Inspect Robots components + the glue to make them an honest, testable, safe pair.
The package is `mypy --strict` clean, ships `py.typed`, and is 100%-covered.

## Modules

| Module | Responsibility |
|--------|----------------|
| `packing.py` | **Pure** 14-D bimanual packing — the single source of truth for how the flat vector maps to two arms (`[j0..j5, gripper]` per arm, left then right). `pack`/`split`/`validate_dim`, `STATE_KEY`, `STATE_SPEC`. No optional deps. |
| `config.py` | `YamConfig` / `MolmoActConfig` (frozen, `from_kwargs` for CLI scalars) + shared `action_box()` / `observation_space()` / `ACTION_SEMANTICS` so both components declare an **identical** contract. |
| `operator.py` | `OperatorIO` (injectable stdin/stdout) for readiness + success prompts; `default_poll_end` (real TTY poll, `# pragma: no cover`). |
| `_i2rt.py` | Lazy i2rt loader + `I2RT_INSTALL_COMMAND`, the single source of truth for the git-only driver remedy. |
| `policy.py` | `MolmoAct2Policy` — `/act` client with `RUNTIME_REQUIREMENTS` for transport dependencies. `act()` packs cameras+instruction+state, POSTs via the injectable `post_fn`, returns an `ActionChunk`. Real transport is the pragma'd `_default_post`. |
| `embodiment.py` | `YAMEmbodiment` — i2rt driver with `RUNTIME_REQUIREMENTS`. Clamp backstop, optional delta→abs, gripper de-norm, `SELF_PACED` pacing, operator-keypress success. Hardware seams (`_default_driver_factory`, `_default_camera_reader`) are injected/pragma'd. |
| `preflight.py` | `build` / `run_preflight` + the `inspect-robots-yam-preflight` CLI: run the compat check, print, exit non-zero on errors. |
| `__init__.py` | Public API fenced by `__all__` (guarded by `tests/test_api_snapshot.py`). |

## Key invariants

- **Contract symmetry:** policy and embodiment build their `action_space` /
  `observation_space` from the *same* `config.py` helpers. If you change the dim,
  semantics, camera names, or state key, change them there once — not in two
  places — or compat breaks.
- **Construction is inert:** `__init__` touches no hardware/network/stdin (only
  `.info`). The driver connects lazily on the first `reset()`. This is what lets
  the registry (`factories[name]()`) and preflight construct components freely.
- **Coverage discipline:** the only uncoverable code is hardware/TTY I/O, isolated
  in `# pragma: no cover` seams (`_default_post`, `_default_driver_factory`,
  `default_poll_end`, the `_require_driver` pre-reset guard, `__main__`). Keep new
  hardware access inside such seams so the 100% gate stays meaningful.
- **Safety lives in `step()`**, not in an optional Approver — see the root
  `CLAUDE.md`.
