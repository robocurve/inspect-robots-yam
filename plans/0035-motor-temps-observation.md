# 0035: Motor temperatures in the observation

Status: implemented.

## Problem

The thermal guardrail (plans 0028, 0029) reads every motor's temperature each
step and ends the trial when one reaches `motor_temp_limit`. The policy never
sees those readings, so the first it learns of a hot motor is the trial ending
with outcome `overheat`. A typical case: an agent spends its last few dozen
calls holding one arm at full reach against a stiff part, a shoulder motor
climbs from about 30 C to past a 60 C limit, and the guardrail stops the run.
A policy that could see the temperature rising would have had a minute of
warning in which to rest the arm.

## Change

When `motor_temp_limit` is set, every observation carries
`extra["motor_temps"]`:

```
{"limit": 60.0, "warn_margin": 10.0,
 "arms": {"left":  {"hottest": "left_j1", "celsius": 58.0, "temps": [...7]},
          "right": {"hottest": "right_j2", "celsius": 40.0, "temps": [...7]}},
 "lines": ["motor temperatures: left arm hottest left_j1 58 C, right arm hottest right_j2 40 C; the trial ends when any motor reaches 60 C.",
           "left_j1 within 10 C of the limit: move that arm back to its start pose and sleep there before the guardrail ends the trial."]}
```

`lines` follows the motion-report convention (plan 0034): plain language
written for the policy, which the agent plugin renders verbatim. The second
line appears only inside `motor_temp_warn_margin` of the limit. Readings at or
below 0 C are the driver's no-data sentinel, as in the guardrail, and are
skipped; an arm with no valid reading says so.

The reading is the one the guardrail already took at the start of the step
(or at reset), so no extra CAN traffic is added. With the guardrail off there
is no reading and no entry.

The built-in agent docs gain a paragraph, present only when the limit is set,
telling the policy what heats motors (holding an extended or loaded pose,
pressing against contact), what cools them (resting near the start pose) and
to treat the warning line as urgent: move the arm back to the start pose it
had at the first observation and sleep there. The agent plugin's plain
`sleep` tool (inspect-robots plan 0087) is the matching action; the harness
deliberately adds no parking logic of its own.

## Not changed

The guardrail's trip behaviour, the warning log, and `motor_temp_limit`
semantics are untouched. The limit itself is a per-installation decision (the
wizard suggests 70 C).
