# 0008 — GR00T fine-tunes behind /act: generic client + `gr00t` entry point + serving shim

Status: draft
Issue: robocurve/inspect-robots-yam#55

## 1. Problem

We have a GR00T N1.7 YAM fine-tune
([robocurve/gr00t-n1.7-yam-molmoact2](https://huggingface.co/robocurve/gr00t-n1.7-yam-molmoact2))
to evaluate on the rig. The natural transport is the /act HTTP protocol the
`molmoact2` policy already speaks: the client is checkpoint-agnostic in every
way that matters (packed 14-D joint layout, three cameras, any `(N, 14)` chunk
length, `dt_ms`-advertised rate). Two gaps:

1. `MolmoAct2Policy` hardcodes `PolicyInfo(name="molmoact2")` and the only
   registered policy entry point is `molmoact2`, so running GR00T behind it
   records a misleading policy name in every `EvalLog`.
2. Nothing bridges an Isaac-GR00T checkpoint to the /act wire format.
   Isaac-GR00T's own `run_gr00t_server.py` speaks its `PolicyServer` protocol
   (zmq-style, port 5555), not /act.

## 2. Design

### 2.1 Generic client: rename with back-compat aliases

`policy.py`: rename class `MolmoAct2Policy` → `ActServerPolicy`; keep
`MolmoAct2Policy = ActServerPolicy` as a module-level alias (still exported,
still the `molmoact2` entry-point target, so existing imports, isinstance
checks, and the installed entry point all keep working).

`config.py`: rename `MolmoActConfig` → `ActServerConfig`; keep
`MolmoActConfig = ActServerConfig` alias. Add one field, **appended after the
existing fields** (all fields have defaults, but inserting earlier would
silently change positional construction like `MolmoActConfig("http://...")`):

```python
name: str = "molmoact2"
```

Defaults deliberately stay the MolmoAct2 first-party server's (port 8202,
name `molmoact2`): the wire format's canonical reference implementation is
MolmoAct2's `examples/yam/host_server_yam.py`, and a zero-arg
`MolmoAct2Policy()` must behave exactly as today.

In `ActServerPolicy.__init__`, `PolicyInfo(name=self._cfg.name)` replaces the
hardcoded string. The missing-camera error message
(`"required by molmoact2"`) becomes `f"required by {cfg.name}"`.

Module docstring: rewrite to describe the generic /act client, naming
MolmoAct2's `host_server_yam.py` as the canonical server and
`scripts/serve_gr00t_act.py` as the GR00T one.

Stale prose to update in the same commit (repo convention: CLAUDE.md stays
current):

- `config.py:1` module docstring ("the MolmoAct2 policy client" → the generic
  /act client).
- `src/inspect_robots_yam/__init__.py` docstring: "Registers two Inspect
  Robots components" → three; add the `gr00t` policy line.
- `src/inspect_robots_yam/CLAUDE.md` module table (`policy.py` row) and the
  root `CLAUDE.md` wherever it says the package ships exactly the `molmoact2`
  policy.

### 2.2 `gr00t` entry point

The registry resolves entry points as plain callables
(`inspect_robots/registry.py: resolve()` calls `factory(**kwargs)`), so a
factory function carries the GR00T defaults:

```python
GR00T_DEFAULTS: Mapping[str, Any] = {
    "name": "gr00t",
    "server_url": "http://127.0.0.1:8203",
    # robocurve/gr00t-n1.7-yam-molmoact2 predicts 16-step chunks
    # (experiment_cfg delta_indices 0..15). Like every field, this is
    # PolicyConfig *metadata* recorded in the eval log — rollout always uses
    # the returned chunk — but leaving MolmoAct2's 30 here would record a
    # false horizon for GR00T runs, the exact dishonesty this plan removes.
    # Operators evaluating a different GR00T fine-tune pass
    # -P action_horizon=<its chunk length>.
    "action_horizon": 16,
}


def gr00t_policy(
    config: ActServerConfig | None = None,
    *,
    post_fn: PostFn | None = None,
    **flat: Any,
) -> ActServerPolicy:
    """Registry factory for the ``gr00t`` policy: the /act client with GR00T defaults.

    Same wire protocol and 14-D contract as ``molmoact2``; only the advertised
    policy name and the default port differ, so a GR00T /act server (see
    ``scripts/serve_gr00t_act.py``) and the MolmoAct2 server can run side by
    side and eval logs record which model actually ran. Explicit ``flat``
    kwargs override the defaults; an explicit ``config`` wins outright.
    """
    if config is None:
        config = ActServerConfig.from_kwargs(**{**GR00T_DEFAULTS, **flat})
    return ActServerPolicy(config, post_fn=post_fn)
```

Port 8203 (not 8202) so both servers can run side by side.

pyproject:

```toml
[project.entry-points."inspect_robots.policies"]
molmoact2 = "inspect_robots_yam.policy:MolmoAct2Policy"
gr00t = "inspect_robots_yam.policy:gr00t_policy"
```

Note for the implementer: entry points are read from the installed dist's
metadata; after editing pyproject, re-run `uv pip install -e .` in the
worktree venv or the registry test for `gr00t` fails with `KeyError`.

### 2.3 Public API

`__init__.py` re-exports and `__all__` gain `ActServerPolicy`,
`ActServerConfig`, and `gr00t_policy`; `MolmoAct2Policy` and `MolmoActConfig`
stay. `tests/test_api_snapshot.py: EXPECTED_API` updated to match
(deliberate-change guard, same PR).

### 2.4 Serving shim: `scripts/serve_gr00t_act.py`

A standalone script (not packaged; wheel ships `src/` only) that runs in an
Isaac-GR00T environment on the GPU machine, mirroring MolmoAct2's
`host_server_yam.py` wire contract:

```
GET  /act  -> {"status": "ok", "model": <path-or-repo-id>}
POST /act  -> request (json_numpy):
    {"top_cam": (H,W,3) uint8, "left_cam": ..., "right_cam": ...,
     "instruction": str, "state": (14,) float32,
     "num_steps": int (accepted, ignored — logged once),
     "timestamp": float (optional, ignored)}
  response (json_numpy):
    {"actions": (N, 14) float32, "dt_ms": float}
```

Implementation contract (verified against the cached Isaac-GR00T checkout,
`gr00t/policy/gr00t_policy.py`):

- Load: `Gr00tPolicy(embodiment_tag="new_embodiment", model_path=<local dir>,
  device="cuda")`. `--model` accepts an HF repo id or a local path; repo ids
  go through `huggingface_hub.snapshot_download` first (matches
  `run_gr00t_server.py`, which requires a local dir).
- Observation is nested and batched:
  `{"video": {key: (1, T, H, W, 3) uint8}, "state": {key: (1, T, D) float32},
  "language": {key: [[instruction]]}}` where `T` is
  `len(modality_configs[m].delta_indices)` per modality (1 for this
  checkpoint) and the key sets come from
  `policy.get_modality_config()` — do not hardcode them.
- Camera mapping: client order `top_cam, left_cam, right_cam` → checkpoint
  video keys, default `base_view, left_wrist_view, right_wrist_view`,
  overridable via `--camera-map top_cam:base_view,...` for future
  checkpoints.
- State and action mapping is **name-keyed, never order-keyed**. A
  width-only check would let a checkpoint whose modality-config lists keys as
  `right_arm, right_gripper, left_arm, left_gripper` pass validation and
  silently swap the arms on real hardware. The shim owns one canonical map,
  `{"left_arm": slice(0, 6), "left_gripper": slice(6, 7),
  "right_arm": slice(7, 13), "right_gripper": slice(13, 14)}` (this repo's
  packed layout, `packing.py` DIM_LABELS). At startup: every state and
  action modality key from `get_modality_config()` must be present in the
  map (hard-fail on unrecognized names) and its width from the checkpoint's
  stats must equal the slice width. Per request: fill each state key from
  its named slice; scatter each returned action key into its named slice of
  a `(T, 14)` buffer. Modality-config order is never load-bearing.
- Action return: `get_action` returns a **tuple** — `actions, _info =
  policy.get_action(observation)` with `actions: {key: (1, T, D) float32}`
  (`BasePolicy.get_action` → `return action, info`).
- Fail at startup, not per-request: also assert
  `len(delta_indices) == 1` for the video and state modalities (a
  frame-history checkpoint is unservable by this stateless shim and must be
  rejected at load, not as opaque per-request 500s), and assert the
  camera-map target set equals `get_modality_config()["video"].modality_keys`
  (a `--camera-map` typo must fail at startup with the valid key list, not
  500 on every request with a GR00T-internal message).
- `dt_ms`: `--dt-ms` flag, default `0.0` (client maps falsy → "no advertised
  rate", so the embodiment paces itself). Do not guess the trained rate.
- FastAPI + `json_numpy.patch()`, single-request lock around inference (same
  as the MolmoAct2 server), errors return 500 with a JSON message.
- Ruff runs on the whole tree: module/function docstrings (D1) required.
  The script is exempt from mypy (files = src) and coverage (source =
  inspect_robots_yam) — no test job runs it; it is validated on the rig.

### 2.5 README

New subsection after the MolmoAct2 server instructions: "Serving a GR00T
fine-tune". Content: `hf download robocurve/gr00t-n1.7-yam-molmoact2`, the
shim command, and the run command

```bash
inspect-robots "stack the red block on the blue block" \
    --policy gr00t --embodiment yam_arms
```

with notes that `-P server_url=...` overrides the default
`http://127.0.0.1:8203` (the config key is `server_url`; `url` is a read-only
property and `from_kwargs` rejects it) and that a different GR00T fine-tune
should pass `-P action_horizon=<its chunk length>` so the logged metadata is
accurate.
Follow the repo writing-style rules (worldevals model-cards.md): no em dashes
in prose, bold only for definition lead-ins, no decorative emoji.

## 3. Tests

All in existing files; mocked transport, no network, 100% coverage holds.

- `tests/test_policy.py`:
  - `gr00t_policy()` zero-arg: `info.name == "gr00t"`, URL
    `http://127.0.0.1:8203/act`, `config.action_horizon == 16`, chunk/space
    contract identical to `molmoact2` (14-D, `control_hz is None`).
  - `gr00t_policy(server_url=...)` keeps name `gr00t`;
    `gr00t_policy(name="x")` overrides; `gr00t_policy(config=cfg)` bypasses
    defaults entirely; `post_fn` passthrough exercises `act()` once.
  - Missing-camera error message carries the configured name (construct with
    `name="gr00t"` and match `"required by gr00t"`).
  - Alias identity: `MolmoAct2Policy is ActServerPolicy`,
    `MolmoActConfig is ActServerConfig`.
- `tests/test_config.py`: `ActServerConfig().name == "molmoact2"`; `name`
  reachable via `from_kwargs`.
- `tests/test_api_snapshot.py`: extend `EXPECTED_API`; extend
  `test_entry_points_resolve_via_registry` with
  `resolve("policy", "gr00t").info.name == "gr00t"`.

## 4. Out of scope

- No new packages, no dependency changes, no floor bumps (the current
  `inspect-robots>=0.12` floor suffices; nothing here uses newer core APIs).
- No launcher/lifecycle management for the shim (same stance as the
  XPolicyLab plugin: we connect to a URL).
- No closed-loop success-rate reporting; that is a rig session, not code.
- Isaac-GR00T environment setup (Blackwell/sm_120 PyTorch, flash-attn) is
  documented in the README section as pointers, not automated.

## 5. Release

Cut a yam minor release after merge (0.13.0): new public API surface
(`gr00t` entry point). No inspect-robots core changes needed.
