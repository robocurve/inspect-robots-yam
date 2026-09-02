# 0028: Thermal guardrail (end the trial while the motors still have torque)

Issue: #144. Branch: `feat/thermal-guardrail`.

## Goal

A motor that reaches the DM firmware's internal over-temperature cutoff fails
hard: the CAN parse raises from the driver's control thread
(`i2rt/motor_drivers/dm_driver.py:313-315`; the over-temperature error codes
`0xB`/`0xC` are defined in `i2rt/motor_drivers/utils.py:97-98`), the motor
chain exits, and torque is gone wherever the arm happens to be. Observed on a
real run 2026-09-01 (rig on `can_can2`, motor id 2, DM4340, log
`adhoc_de62db89.json`): the trial ended cleanly only because a human was
watching. Unattended, the arm goes limp mid-pose and recovery under power is
impossible.

This plan adds a soft, plugin-owned thermal limit with two gates:

1. **Pre-run:** a motor already at or over the limit at `reset()` refuses the
   episode before any motion is commanded (`EmbodimentFault`, the existing
   fail-fast reset-gate pattern).
2. **Mid-run:** a motor crossing the limit during the episode ends the trial
   gracefully from `step()` with `terminated=True,
   termination_reason="overheat"` so the existing park-for-grading ramp runs
   while torque is still available and the operator verdict prompt fires.

## Load-bearing decisions (verified before implementation)

- **The only graceful embodiment-initiated end is a `StepResult`.** Raising
  anything from `step()`/`reset()` becomes `EmbodimentFault`
  (`inspect_robots/rollout.py:288-294`, `:430-436`), and an errored record is
  never parked or graded (`inspect_robots/eval.py:560-568`; the
  `observe_parked()` + grader block lives in the `status == "success"` branch,
  `eval.py:569-602`). So the mid-run gate must **return, never raise**.
- **The reason string must be non-definitive.** `"success"`/`"failure"`
  (`_DEFINITIVE_REASONS`, `inspect_robots/session.py:40`) are adopted as the
  verdict and suppress both the park and the prompt. A custom reason
  (`"overheat"`) passes the gate at `eval.py:574-576`: `observe_parked()`
  ramps home, then `prompt_verdict` asks the operator. `StepResult`'s own
  docstring sanctions custom embodiment reasons ("collision", "fault",
  `inspect_robots/types.py:99-102`). The framework's reason table
  (`inspect_robots/cli.py:123-132`) prints unknown reasons raw; acceptable,
  noted in Risks.
- **The pre-run gate cannot use the grading screen.** A trial only reaches the
  grader with `status == "success"`, which requires at least one policy
  inference plus one step; burning an LLM call to grade a trial that never
  started is wrong. Issue #144 asked for the grading screen in both cases;
  this plan deliberately deviates for the start case and documents that in the
  PR. Refusing before the homing ramp commands no **arm** motion (the driver
  factory's gripper calibration, `embodiment.py:284-290`, has already run by
  the time temps are readable; unavoidable and low-energy), and `close()`
  releases torque in place because its park is guarded by
  `if self._init_pose is not None:` (`embodiment.py:1771`) — see the placement
  invariant in Changes step 4.
- **Temperature source:** `MotorInfo.temp_mos`/`temp_rotor` are parsed from
  every DM feedback frame (`dm_driver.py:320-328`; `MotorInfo` construction at
  `:717-720`) and
  ride the same cached snapshot `joint_eff` comes from, refreshed >100 Hz by
  i2rt's own control thread. `MotorChainRobot.get_observations()` hides them
  behind `temp_record_flag`, which `get_yam_robot` never sets, so the `_Real`
  adapter reads `robot.motor_chain.read_states()` per arm instead: public,
  takes the DM `state_lock` (`dm_driver.py:691-719`), needs no remapping
  (the vel/eff sign correction lives inside `read_states` itself,
  `dm_driver.py:707-708`, and does not touch temps), and includes the gripper
  motor slot (the gripper motor `0x07` joins the same chain,
  `get_robot.py:210`, so each arm always yields 7 slots for `packing.pack`).
- **Dependency floor raised to `inspect-robots>=0.57`** (amendment, found by the
  post-implementation review's e2e test): the park-before-grading lifecycle
  (`observe_parked`) first shipped in framework 0.57, while the old floor was
  0.51 — on floor installs the plugin's own `observe_parked` hook (plan 0023)
  was already a silent no-op and this feature's central promise would not
  hold. Deployed rigs run 0.57.1. The e2e overheat lifecycle test in
  `tests/test_eval_end_to_end.py` pins the promise against the real installed
  framework so a future floor/behavior drift fails loudly.
- **No upstream threshold exists to inherit.** Neither i2rt nor the DM
  constants carry any temperature limit (only the error codes); the sim's
  idle values are 35/40 C. Any default would be invented, so the limit is
  **opt-in**: `motor_temp_limit = none` (the default) disables the guardrail
  entirely. Deployments choose a measured value; `health` gains a temperature
  readout (below) to give operators data to choose from. Opt-in also keeps
  injected drivers that predate `get_motor_temps()` working by default, like
  `report_joint_eff` did (plan 0021).

## Changes

1. **`config.py`** — two fields after the collision block (`config.py:197-206`
   area), same comment discipline:
   - `motor_temp_limit: float | None = None`: degrees C, compared against the
     max of MOS and rotor temperature per motor, all 14 slots (grippers
     included). `None` disables both gates. Validated in `__post_init__`:
     finite and > 0 when set.
   - `motor_temp_warn_margin: float = 10.0`: a `logger.warning` fires once per
     trial when any motor reaches `limit - margin` (only meaningful when the
     limit is set). Validated finite and >= 0. `from_kwargs` additionally
     rejects `motor_temp_warn_margin=none` with a curated `ValueError` (the
     `collision_table_height` precedent, `config.py:304-308`); an unguarded
     `None` would reach the finiteness check as a raw `TypeError`.
     `__post_init__` also rejects `margin >= limit` when the limit is set (the
     warn threshold would be <= 0, warning on every trial's first read).
   No bool flag: a single optional float is the whole surface, so the
   `from_kwargs` bool guard (`config.py:288-302`) gains no entry, and the CLI
   literal `none` for the **limit** maps to None = off, which is exactly the
   intended meaning.

2. **`embodiment.py` — driver protocol.** `BimanualDriver` gains
   `get_motor_temps(self) -> npt.NDArray[np.floating[Any]]: ...`
   (`embodiment.py:173-192`): shape `(14,)`, degrees C, per motor the max of
   MOS and rotor so one number per slot is compared; values <= 0 mean "no
   data" (i2rt's `-1` sentinel, `i2rt/motor_drivers/utils.py:67-68`) and are
   never treated as cold. The `_Real` adapter (inside the existing
   `# pragma: no cover - real hardware` seam starting at `embodiment.py:269`)
   implements it via `read_states()` per arm + `packing.pack`, mirroring
   `get_joint_eff` (`:296-301`).

3. **`embodiment.py` — access guard.** A `_read_motor_temps()` helper mirrors
   the `report_joint_eff` lookup (`embodiment.py:2138-2146`): `getattr` on the
   driver, raising `RuntimeError` with a fix hint ("motor_temp_limit is set
   but the injected driver lacks get_motor_temps()") when the feature is on
   and the driver predates it. Reads happen only when
   `motor_temp_limit is not None`: the default config performs zero extra
   driver calls, keeping the existing contract byte-identical.

4. **`embodiment.py` — pre-run gate in `reset()`.** Placement invariant, in
   one sentence: **the gate sits at `reset()` top level, after the
   `if self._driver is None:` connect block (`embodiment.py:1549`) and before
   the `if self._init_pose is None:` capture block (`:1554-1561`) — so it runs
   on every reset including a warm second trial, and a first-reset trip leaves
   `_init_pose` None so `close()` releases torque in place instead of ramping
   (`embodiment.py:1771`).** The gate uses the same confirmation re-read as
   the mid-run gate (finding: its consequence, a whole-eval halt, is harsher
   than a trial end, so it must be at least as glitch-immune). The earlier
   fail-fast gates (no-cameras
   `ConfigError`, `auto_start` stdin check, `embodiment.py:1506-1548`) stay
   first and unchanged, as do the stand-clear gate and the
   `_home_gate_confirmed` retry semantics (`:1278-1282`). The gate: read
   temps; any valid reading >= limit raises
   `EmbodimentFault("thermal guardrail: <label> (<channel>) at <T> C >= "
   "limit <L> C at episode start; let the rig cool or raise "
   "motor_temp_limit")` where `<label>` is `packing.DIM_LABELS[slot]` and
   `<channel>` is `cfg.left_channel`/`right_channel` by slot half. The
   warn-margin warning (step 5) also fires from this read when applicable,
   since reset is when cooling is cheapest.

5. **`embodiment.py` — mid-run gate in `step()`.** At `step()` entry, before
   the motion is played (the pose-hold during a slow policy's thinking time is
   exactly when heat soaks in, and a hot motor should not be given more work):
   read temps; if any valid reading >= limit, re-read once after
   `sleep_fn(1 / (control_hz if control_hz > 0 else 10.0))` (the repo-wide
   fallback for self-paced rigs, e.g. `embodiment.py:2124`; a bare
   `1 / control_hz` is a `ZeroDivisionError` on a legal config) and trip only
   if the same motor is still hot (one glitched frame never ends a trial).
   On trip: close the status line (`self._status(None)`, the operator-end
   precedent at `embodiment.py:1683-1695`) and emit one notice line through
   the **session-aware notice pattern** (`self._session.write_line(...)` when
   connected, else `self._operator.output_fn(...)` — the auto_start notice at
   `embodiment.py:1565-1573`), naming the **hottest** motor's
   `packing.DIM_LABELS[slot]`, its CAN motor id (slot-within-arm + 1; the
   gripper slot is motor `0x07`, `get_robot.py:210` — so the line correlates
   with the firmware's own "motor id: N" fault logs), the arm's CAN channel
   (`cfg.left_channel`/`right_channel` by slot half), and the temperature.
   This line is the only explanation the grading operator sees (the
   "ended early" note fires for truncated trials only,
   `session.py:712-718`). Then skip the motion entirely and return
   `StepResult(observation=self._observe(self._instruction), terminated=True,
   termination_reason="overheat", info={"overheat": {"slot": i,
   "label": ..., "motor_id": ..., "channel": ..., "temp": t}})` — no settle
   info on this path (no settle ran; absent keys mean "never enabled", per
   `_settle_info`'s docstring), and a code comment noting the recorded policy
   action was never executed (`rollout.py:447-455` stores it regardless).
   Durability of the details: `StepResult.info` is in-memory only, so the
   durable trace of a trip is `termination_reason="overheat"` in the eval log
   plus the visible console line and the `logger` record; issue #144's
   "motor id/channel/temperature" ask is met by putting all three in that
   visible line and in the pre-run fault message (this is a documented
   weakening — no stock sink persists per-step info).
   The step-side hard clamp on every commanded target (`_send`,
   `embodiment.py:2028-2033`; root CLAUDE.md "Safety lives in step()") is
   retained untouched: the thermal gate adds an early return before the
   command path, it does not modify it. The legacy keypress-poll early return
   (`embodiment.py:1683-1694`) is also retained; operator end keeps priority
   (checked by the rollout before the policy runs, `rollout.py:342-356`).
   The warn-margin `logger.warning` (once per trial, flag cleared at
   `reset()` entry beside `settle_timeouts`, `embodiment.py:1497-1498`) fires
   from the same read.

6. **`health.py`** — `_run_motors` (`health.py:214-244`) additionally reads
   temps when the built driver provides `get_motor_temps` and reports them as
   a **separate, always-`ok=True` `CheckResult` row** (`temps: max <T> C @
   <label>`), read before the `driver.close()` in the `finally`. It must NOT
   be appended to the existing per-joint detail strings: `_run_motors`
   computes `ok=not detail` (`health.py:210`, `:243`), so appending would turn
   passing rows into FAULT. No pass/fail change: health has no config to
   compare against.

## Retained invariants (restated at the point of modification)

- `step()` always clamps to `joint_low/high` before commanding; the thermal
  early-return sits before the command path and the clamp is unmodified.
- The termination reason is non-definitive, so the park-for-grading ramp and
  the operator verdict prompt both still run (the plan-0013 rationale for
  `operator_end` being non-definitive applies verbatim to `"overheat"`).
  Caveat, stated in the README too: `observe_parked()` runs only in graded
  runs with `park_before_grade=true` (`eval.py:570-576`). In an ungraded or
  unattended run a mid-run trip terminates the trial without an immediate
  park; the arms keep holding under power (position hold, or gravity-comp
  under the default `zero_gravity_mode=true`) until `close()` parks them —
  still strictly better than the firmware cutoff, which drops torque
  entirely.
- Multi-trial semantics: within one eval, a mid-run trip grades that trial,
  and the **next** trial's pre-run gate then raises `EmbodimentFault`, which
  halts the whole eval (`eval.py:535-543`) — there is no per-trial cooldown
  retry inside a run. The halt path's `close()` ramps to rest pose (its
  `_init_pose` was captured by the earlier successful reset), which is
  intended: parking a still-warm rig is exactly the safe end state. Per-trial
  retry semantics exist only across separate processes (`run_batch`).
- Construction stays inert: no temperature is read in `__init__`; the first
  read happens inside `reset()` after the lazy driver connect.
- With `motor_temp_limit` unset (the default), no new driver method is called
  and no observation/space/docs change occurs: the policy-facing contract is
  untouched, exactly like `report_joint_eff=false`.

## Tests (CI enforces 100% line+branch coverage)

New, in `tests/test_embodiment.py` (helper: extend `FakeDriver.__init__` with
`temps=None` defaulting to `np.full(14, 30.0)`, plus a `temps_seq` list the
fake pops per read, mirroring `SettleDriver`'s read counting):

- default off: `motor_temp_limit=None` performs zero `get_motor_temps` reads
  across a full reset+step cycle (fake counts reads) and raises no warnings.
- mid-run trip: temps at limit on both reads -> `terminated=True`,
  `termination_reason == "overheat"`, no command appended for that step,
  `info["overheat"]` carries slot/label/motor_id/channel/temp, the notice
  line (captured via a recording session's `write_line`, and via
  `operator.output_fn` in the unconnected variant) names the joint label,
  motor id, channel, and temperature, and the status line was closed.
- glitch immunity: first read hot, confirmation read cool -> motion plays,
  no termination; the confirmation sleep used `sleep_fn` (captured).
- sentinel: temps of `-1` never trip even with a tiny limit.
- gripper slot 13 trips like an arm slot.
- warn margin: `limit - margin <= t < limit` -> `logger.warning` once
  (caplog), not twice across two steps, cleared by the next `reset()`; the
  same warning also fires from the reset-time read (separate test branch).
- pre-run gate: hot at reset -> `pytest.raises(EmbodimentFault,
  match="thermal guardrail")` and `drv.commands == []` (no motion commanded),
  message names the joint label, motor id, and channel; glitch immunity at
  reset too (hot then cool -> reset proceeds).
- placement invariant: after a first-reset gate fault, `emb.close()` appends
  no commands (no ramp) and still closes the driver (`drv.closed`); after a
  successful trial, a second `reset()` that trips the gate raises even though
  the driver is already connected, and `close()` then DOES ramp (the
  `_init_pose` from the first reset is retained by design).
- legacy driver: `motor_temp_limit` set + driver without `get_motor_temps` ->
  `RuntimeError` matching `motor_temp_limit.*get_motor_temps`.

`tests/test_config.py`: default None; `from_kwargs` binding; rejection of
zero/negative/non-finite limit, negative/non-finite margin, and
`motor_temp_warn_margin=None` (curated `ValueError`, not `TypeError`).
`tests/test_health.py`: a separate always-ok temps row appears when the fake
driver reports temps (and every pre-existing motor row keeps its verdict);
the row is absent when the driver lacks the method.

Sanctioned updates to pre-existing tests (mechanical, the plan-0021 pattern):
add `get_motor_temps` returning benign temps to every fake driver —
`tests/test_embodiment.py:29` (`FakeDriver`), `tests/conftest.py:444`
(`SettleDriver`), `tests/test_eef_embodiment.py:72`,
`tests/test_depth_reader.py:889`, `tests/test_eval_end_to_end.py:28`,
`tests/test_camera_reader.py:338`, `tests/test_pose_cli.py:28`. `LegacyDriver`
(`tests/test_embodiment.py:404`) intentionally stays without it — and so does
the shared `tests/test_health.py:139` fake: giving it temps would add the
always-ok temps row to `report.joints` and break the three pre-existing
row-set pins (`test_health.py:221`, `:529`, `:884`), which must stay
untouched and which keep covering the legacy-driver health path. The new
health tests construct their own temps-enabled fake locally. No other
pre-existing assertion may change; in particular the `OPTION_SLOTS` pin
(`tests/test_i2rt.py:373-394`) is untouched because a float option adds no
wizard slot, and `tests/test_api_snapshot.py` is untouched because nothing new
is exported.

## Docs/meta

- README: option table entry for `motor_temp_limit` / `motor_temp_warn_margin`
  with the measurement guidance (run `inspect-robots-yam-health` after a long
  episode; pick a limit comfortably below where the firmware has faulted).
- CHANGELOG `## Unreleased` / `### Added`: prose paragraph with the why
  (firmware cutoff strands the arm limp; the guardrail ends the trial onto the
  grading screen while torque remains) and the ref `(#144)`.
- Root `CLAUDE.md`: one line in the safety-invariants section (the thermal
  gate ends trials with a non-definitive reason; never make it definitive).
- `src/inspect_robots_yam/CLAUDE.md`: add the new flags to the `config.py` and
  `embodiment.py` rows of the module table.

## Non-goals

- No per-motor-type limits (DM4340 vs the gripper's 4310-class motor share one
  limit; conservative by construction).
- No temperature key in observations and no agent-facing docs paragraph: the
  policy cannot act on motor temperature, and undeclared runtime keys were a
  forced exception for effort (plan 0021), not a pattern to extend.
- No upstream i2rt change (`temp_record_flag` stays untouched; `read_states()`
  suffices).
- No cooldown/auto-resume logic: once tripped, the trial is over, and the
  next reset's gate fault ends the eval (see Retained invariants for the
  exact multi-trial semantics).

## Risks

- **Threshold provenance.** There is no documented DM4340 cutoff to derive the
  limit from; a value chosen too low ends healthy runs. Mitigated by opt-in
  default, the health readout for measurement, and the warn margin announcing
  the approach before the trip.
- **Unknown reason string.** The framework summary table does not know
  `"overheat"` and prints it raw with `has_unmapped` set
  (`inspect_robots/cli.py:123-132`). Cosmetic; a framework-side mapping is a
  one-line follow-up there, not here.
- **Confirmation re-read adds up to one control period of latency** to the
  trip (100 ms at 10 Hz). Irrelevant against thermal time constants.
- **Skew:** the temps read is a third lock acquisition beside pos/eff, so it
  can be one control tick stale; also irrelevant at thermal timescales.
