# 0013 — maintenance CLIs default devices from the wizard config

Issue: #80. Depends on: robocurve/inspect-robots#197 (public
`inspect_robots.defaults` API, core plan 0029; requires
`inspect-robots >= 0.24`). Status: revised after critique rounds 1 and 2
(round 1: owner gate rebuilt on `registered()`, preflight dropped as a
no-op, depth keys admitted; round 2: health taught to check only
cam-device slots — a mixed RGB/depth config must not fail a healthy rig —
raw-string `-E` handling extended to depth serials, attribution moved
before validation; rounds 3-4: montage writer and watch index page brought
into the depth-aware change, key-set constant split three ways so flag
collection survives).

## Problem

`inspect-robots setup` writes the rig's camera devices and CAN channels to
`~/.config/inspect-robots/config.ini` under `[embodiment.args]`. The run CLI
reads them; the yam maintenance CLIs do not. On a fully configured rig:

```
$ inspect-robots-yam-health --watch
inspect-robots-yam-health: error: --watch requires configured camera devices
```

The same gap hits motors: health's motor check runs on `YamConfig`'s builtin
`can0`/`can1` channel names unless the user re-types `-E left_channel=...`
that the wizard already recorded, and holdcheck demands a raw channel name
even though the config file knows which channel is which arm.

Scope: **health and holdcheck.** Preflight is explicitly out: its report is
derived from `EmbodimentInfo` (`embodiment.py:1013-1029`), which depends on
`cam_height`/`cam_width`/`control_hz`/limits — none of which the wizard
writes device values for. Wiring the config into preflight would change no
byte of its output; issue #80's mention of preflight is narrowed by this
plan, with this paragraph as the recorded reason.

## Design

One new module, `inspect_robots_yam._user_config`, consumed by the two
CLIs. It wraps the core's public API (core plan 0029):

```python
from inspect_robots.defaults import config_path, load_defaults
```

### The loader

`load_yam_defaults(env: Mapping[str, str]) -> YamDefaults` where
`YamDefaults` is a small frozen dataclass: `args: dict[str, Any]` (empty when
nothing applies), `source: str | None` (the config file path when `args` is
non-empty), `owner: str | None` (the resolved owner name, for attribution).

Rules, in order:

1. `load_defaults(env)` — the core applies file/env precedence and raises
   `SystemExit` naming the file when it is malformed. That propagates —
   deliberately. Two failure classes get opposite treatment throughout this
   plan: a *stale* config (unknown owner, non-YAM owner, missing plugin) is
   silently ignored, because it must not break a hand-flagged invocation; a
   *malformed* config (unparseable file) is a loud guided exit, because
   every later `inspect-robots run` will hit the same wall and the right
   move is to fix the file now. `--no-config` (below) is the escape hatch,
   and it bypasses `load_defaults` entirely — with it, a malformed file
   cannot exit anything.
2. If `embodiment_args_owner` is `None` or `embodiment_args` is empty →
   empty result.
3. Owner gate: the args apply only when the configured embodiment *is a
   YAM rig*. A name allowlist is wrong — `yam_arms_omen` ships in the
   separate `omen-rig` package — so ask the registry for the **uncalled
   factory** and check it:

   ```python
   from inspect_robots.registry import registered
   factory = registered("embodiment").get(owner)   # never constructs
   ok = isinstance(factory, type) and issubclass(factory, YAMEmbodiment)
   ```

   `registered()` returns a plain name→factory dict without calling
   anything (`registry.py:103-108`), so no foreign embodiment is
   constructed — and no hardware is touched — at CLI startup. Limitation,
   stated in the module docstring: the gate recognizes an embodiment only
   when its entry point is the class itself (both known YAM rigs register
   classes). A function-factory registration is treated as non-YAM and
   ignored, the safe direction. Unknown owner (plugin not installed, stale
   config) or non-YAM owner → empty result, never an error.
4. Key filter: the camera-slot keys and channel keys —
   `{top,left,right}_cam_device`, `{top,left,right}_depth_serial`,
   `left_channel`, `right_channel`. Depth serials are included **not**
   because the wizard writes them (it does not — `_setup.py` writes only
   the `DEVICE_SLOTS` the embodiment declares, which are the three
   `cam_device` slots plus the channels) but because they are documented
   config.ini-carriable keys (`config.py:191-197`, reachable via manual
   edit or `inspect-robots config set`, and preserved across wizard
   re-runs) and `YamConfig.__post_init__` makes them inseparable from the
   cam-device keys: all-or-none camera sourcing (`config.py:333-343`) plus
   per-slot `cam_device` XOR `depth_serial` (`config.py:324-332`) mean
   filtering the depth half out would hand `from_kwargs` a partial camera
   set and turn a *valid* mixed config into exit 2 on a bare `-health`
   run. Everything else in `[embodiment.args]` (poses, limits, checkpoint
   paths) is the run CLI's business and is not forwarded.
5. String coercion: the core coerces every config value through its scalar
   parser before plugins see it, so `top_cam_device = 0` arrives as int
   `0` and a numeric depth serial arrives as int. The loader applies
   `str()` to every non-`None` forwarded value — all eight keys are
   string-typed in `YamConfig`, and an int reaching `__post_init__`'s
   `.strip()` is an uncaught `AttributeError` traceback
   (`config.py:296-299`). Values that are `None` after the core's parse
   (a literal `none`/`null` in config.ini) are **dropped**, not
   forwarded — an explicit `None` is validation-harmless but would count
   as a contributed key for attribution and slot supersession, which
   would be misleading. Docstring caveat (inherited from the core
   parser, unfixable in the loader): a serial with a leading zero is
   int-coerced upstream and `str()` cannot restore the zero — quote the
   value in config.ini (the core parser's quoted-string escape hatch) to
   keep it verbatim.

### Precedence and slot supersession (health)

explicit flags > `-E` extras > wizard config > `YamConfig` builtins — with
the caveat that flag-vs-`-E` on the same camera key is already an exit-2
conflict (health.py:333-335), i.e. conflict-checked, not silently
overridden. The config layer sits strictly below both.

Same-key merge is a plain dict merge:
`{**yam_defaults.args, **extras, **flag_values}`.

Cross-key, the camera slots need one extra rule. `cam_device` and
`depth_serial` are XOR *per slot*, so config-sourced `top_cam_device` plus a
user's `-E top_depth_serial=...` would merge into an XOR violation the user
never wrote. Rule: **a slot the user touches is entirely the user's.** For
each slot in {top, left, right}, if flags or extras contain either of that
slot's keys, both of the config layer's keys for that slot are dropped
before the merge. The all-or-none sourcing check then runs on the merged
result exactly as it does today (a genuinely partial result remains the
existing exit-2, current behavior for hand-flagged partial sets).

Channels have no cross-key partner; plain merge applies.

Known residual conflict, by design: a user flag can collide with a
config-sourced *other* slot (config `left_cam_device=/dev/video1`, user
`--top-cam /dev/video1` → duplicate-device exit 2, `config.py:300-311`).
The attribution line (below) prints **before** `from_kwargs` runs, so the
error message is preceded by the line naming the file that contributed
`left_cam_device` — the user is never left wondering where a key they
didn't type came from.

### health.py: depth-aware camera checks

Health's camera section can only exercise V4L2 colour devices (cv2
readers). Today `_camera_devices` (health.py:130-134) yields all three
slots and `cameras_configured` keys off `top_cam_device` alone
(health.py:215,351) — fine while cam-device was the only sourcing that
could reach the tool, wrong the moment a valid mixed or all-depth config
flows in (a reader constructed on device `None` raises, the broad
`except` at health.py:162 converts it to a FAULT, and a healthy rig exits
1). Adjusted, minimally:

- The camera check iterates **only slots with a non-None `cam_device`**.
- Skipped depth slots are **absent from the camera results**, never
  encoded as pass or fail: `HealthReport.cameras` contains only checked
  slots, `report.ok` is unaffected by skipped slots, and the `--json`
  payload gains an `unchecked_cameras` list naming them with the reason
  string (named to avoid confusion with the existing `cameras_skipped`
  boolean, health.py:79). On stderr each one gets a note:
  `top_cam: skipped (depth-configured; not checked by this tool)` — and
  the summary line for an all-depth rig says the camera checks were
  skipped *because the configured slots are depth*, resolving the
  otherwise-contradictory "no camera devices configured" message next to
  an attribution line that just said the config configured cameras.
- The montage writer iterates the **checked** slots — `DEFAULT_CAMERAS`
  filtered to names present in frames or faulted, preserving the
  canonical tile order its docstring promises — not all of
  `DEFAULT_CAMERAS`: today's
  `_default_write_montage` (health.py:103-127) requires every one of the
  three names to be in `frames` or `faulted` and KeyErrors otherwise,
  which the montage `except` at health.py:234-238 would convert into a
  spurious `montage` FAULT (exit 1) on a healthy mixed rig.
- `cameras_configured` becomes "at least one slot has a `cam_device`";
  the `--watch` gate uses the same predicate and its `parser.error`
  message gains the depth-aware wording too (an all-depth rig is told
  the slots are depth-configured, not "no camera devices configured").
- `--watch` serves exactly the cam-device slots, **including the index
  page**: `_serve_index` (watch.py:139-143) iterates `DEFAULT_CAMERAS`
  today and must iterate the supervisors' keys instead, or a mixed rig's
  index renders `<img>` tiles whose `/stream/` endpoints 404
  (watch.py:129). The stream side needs no change — `serve` already
  builds supervisors from `health._camera_devices` (watch.py:230).
- Honest limit, stated in README and the exit-code docstring: depth slots
  are out of this tool's scope; on an all-depth rig `--watch` still
  errors (nothing servable) and camera checks are skipped. Unchanged from
  today in effect, but now it happens for a *stated* reason instead of by
  accident of the `top_cam_device` predicate.

### Per-CLI behavior

**health** (`health.py`):

- `main(argv, *, env=None)` — `env` defaults to `os.environ`; the loader is
  called with it. This is the test seam (see Tests).
- Merge as above, then everything downstream (all-or-none validation,
  `--watch` gating, watch key injection) operates on the merged dict.
- Key-set surgery, three distinct sets (today one constant serves all
  three jobs, and naively growing it breaks flag collection — there are
  no `--*-depth` argparse dests, so `getattr(args, "top_depth_serial")`
  at health.py:331-332 would AttributeError on every invocation):
  - `_CAMERA_FLAG_KEYS` — the three `cam_device` keys, used **only** for
    collecting `--top-cam`/`--left-cam`/`--right-cam` flag values
    (health.py:331-332) and the flag-vs-`-E` conflict check (333-335).
    Unchanged contents, renamed for clarity.
  - `_RAW_STRING_KEYS` — cam-device **plus** the three `depth_serial`
    keys; the `-E` values that must never be scalar-coerced. A numeric
    `-E top_depth_serial=...` (RealSense serials are typically
    all-numeric) must stay a string or it tracebacks in `__post_init__`
    exactly like a numeric cam device would.
  - `_CAMERA_SLOT_KEYS` — all six, used by the skip-conflict check and
    the slot-supersession rule.
- Skips beat config: `--skip-cameras` removes all six camera-slot keys
  from the *config layer* before the merge; `--skip-motors` removes the
  channel keys. The existing exit-2 for `--skip-cameras` combined with an
  explicit user camera key extends to `_CAMERA_SLOT_KEYS` — an explicit
  design change, not a rename side effect (today it checks only the three
  cam-device keys, so `--skip-cameras -E top_depth_serial=X` would dodge
  the guided message and die later in an opaque all-or-none error).
  Config-sourced cameras must never turn a skip into an error.
- Attribution, on **stderr** (stdout carries the `--json` payload,
  health.py:391-394), printed **before** `from_kwargs`/validation (see
  the residual-conflict note above): when the config layer contributed at
  least one key that survived the merge:
  `devices: from <config_path> (embodiment <resolved owner>)` — the owner
  printed is `YamDefaults.owner`, whatever it resolved to.
- `--no-config`: skip the wizard-config layer entirely (never calls
  `load_defaults`). The escape hatch for "test this exact flag set", and
  the back-compat path for scripts that relied on "no flags = cameras
  skipped": with a config present, bare `inspect-robots-yam-health` now
  checks cameras too. Intended — the tool checks the rig as configured —
  but the old behavior stays reachable.
- `--watch` requires ≥1 cam-device slot *after* the merge, so
  `inspect-robots-yam-health --watch` works on a wizard-configured RGB
  rig with zero flags.
- Module-docstring exit-code contract (health.py:6-17) is updated for
  every behavior this changes: bare invocation on a configured rig, the
  `--watch` gate wording, skip semantics vs config, depth-slot skipping,
  and `--no-config`.

**holdcheck** (`hold_check.py`):

- `main(argv, *, env=None)` seam, as health.
- The positional `channel` additionally accepts the literals `left` and
  `right`, resolved through the config layer's `left_channel` /
  `right_channel`. Anything else passes through as a raw channel name,
  untouched — today's behavior. Help text notes that an interface
  literally named `left`/`right` must be passed another way (socketcan
  names are arbitrary; the literals are claimed) — in practice: rename
  the interface, there is no escape syntax.
- The startup emit line (hold_check.py:195) prints the **resolved**
  channel, e.g. `can_left (left) zero_gravity=false: ...`, not the bare
  literal.
- Lazy load: the config is loaded **only** when the positional is `left`
  or `right`. `inspect-robots-yam-holdcheck can0 ...` never reads the
  file, so a malformed config cannot break a raw-name invocation.
- `left`/`right` with no resolvable config value is a `parser.error`
  naming the fix: `left requires left_channel in the wizard config; run
  inspect-robots setup, or pass the CAN channel name directly`.
- `--no-config` accepted for symmetry (makes `left`/`right` always error).

### Dependency and packaging

- `pyproject.toml`: `inspect-robots>=0.24` (first release with
  `inspect_robots.defaults`).
- `uv.lock` regeneration **requires core v0.24.0 to exist on PyPI** — this
  PR is sequenced strictly after the core release, and the lock bump is
  what makes CI's locked sync install the new core. For the record: the
  ci.yml job at lines ~82-89 is the *import-hygiene* job (minimal
  dependency count, versions pinned from the lock — it installs the
  latest locked version, not the declared floor; nothing in CI exercises
  the `>=0.24` floor itself). No CI edits needed.
- No entry-point changes; no new console scripts.

## Tests

Isolation first: every existing CLI test calls `main()` directly, and with
this change a developer machine's real `~/.config/inspect-robots/config.ini`
would leak into them (bare `main([])` starting camera checks against real
device paths). Two layers:

- The `env` keyword on both mains is the explicit seam; new tests pass a
  constructed env (`{"XDG_CONFIG_HOME": str(tmp_path)}` — XDG takes strict
  precedence over HOME in the core's `config_path`, so this alone
  isolates).
- An autouse fixture in `tests/conftest.py` monkeypatches
  `XDG_CONFIG_HOME` to an empty tmp dir so every test that *doesn't* opt
  in runs config-free. Existing assertions then hold unmodified.

Coverage:

- `test_user_config.py`: empty when no file / no `[embodiment.args]` /
  owner `None` / owner unknown / owner resolves to a non-YAM class /
  factory is a function (not a class); owner `yam_arms` flows args
  through; a `YAMEmbodiment` subclass registered under another name is
  accepted (monkeypatched `registered` mapping); keys outside the filter
  dropped; int-coerced values (`top_cam_device = 0`, numeric depth
  serial) arrive as `str`; malformed config propagates `SystemExit`.
- health: bare invocation with an RGB config runs camera checks on the
  config devices; flag and `-E` each override a config key; slot
  supersession (config slot fully dropped when the user sets either key
  of that slot — config `top_cam_device` + `-E top_depth_serial=838212071234`
  builds a valid config, as a **string** serial, instead of an XOR error
  or a traceback); mixed cam-device/depth config: cam-device slots
  checked, depth slots absent from `cameras` and listed in
  `unchecked_cameras`, the montage written for checked slots only, exit 0
  on a healthy rig; watch index page on a mixed config lists only the
  served slots;
  all-depth config: camera checks skipped with the depth-specific
  message, motors still checked; cross-layer duplicate-device conflict
  exits 2 *after* the attribution line names the config file;
  `--skip-cameras` with config-only cameras exits 0 (no exit-2) while
  `--skip-cameras` + explicit `-E top_cam_device=...` **or**
  `-E top_depth_serial=...` stays exit 2; `--watch` passes the gate from
  config and serves only cam-device slots; `--no-config` restores today's
  error; attribution goes to stderr exactly when config contributed and
  names the resolved owner.
- holdcheck: `left`/`right` resolve and the emit line shows the resolved
  name; raw names never load the config (malformed file + `can0` still
  runs); `left` without config → guided parser error; `--no-config`
  forces that error even with config.
- Coverage stays 100% (`fail_under` is enforced repo-wide).

## Risks

- **Behavior change on configured rigs**: bare `inspect-robots-yam-health`
  goes from "motors only, builtin channels" to "cameras + motors on the
  wizard's devices". This is the point, is announced on stderr, and is
  escapable via `--no-config`/skips. README + CHANGELOG entries say so.
- **Registry import cost and side effects**: `registered("embodiment")`
  triggers the registry's full entry-point load — builtins plus entry
  points for **all five kinds** (tasks, policies, embodiments, scorers,
  sinks; `registry.py:79-100`) — once, at maintenance-CLI startup, only
  when `[embodiment.args]` is non-empty. A broken plugin of any kind
  surfaces as the registry's existing `RuntimeWarning`, not an error.
  Interactive tools; acceptable.
- **Core API drift**: pinned by `inspect-robots>=0.24`; the core façade's
  `__all__` is covered by core tests; this plan touches only `Defaults`
  fields that mirror the config file format.
