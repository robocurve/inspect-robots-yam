# Rig facts

Agent-facing docs for agent-policy runs, split in three so each part can
be included or dropped per run:

- **Per-rig fact sheets** ([rig-2.md](rig-2.md), ...) — the measured and
  derived truths about one physical rig: kinematic constants, camera
  geometry, controller behavior, joint-effort signatures.
- **[formulas.md](formulas.md)** — planar FK/IK for the YAM arm and the
  approximation's measured biases. Shared by all YAM rigs.
- **[advice.md](advice.md)** — working practice distilled from episode
  hindsight. Task-agnostic; shared by all YAM rigs.

## Using the docs on your rig

`YamConfig.docs_extra` appends arbitrary text to the embodiment docs that
agent policies inject into their system prompt (as "Embodiment notes:").
Feed the docs to a run with:

```bash
inspect-robots run --policy agent \
  -E "docs_extra=$(cat rig-2.md formulas.md advice.md)" \
  ...
```

or bake it into a small per-rig run wrapper so every run gets it — an env
var in the wrapper makes the selection per-run switchable, which is also
how you A/B what each part is worth. Policies that don't read embodiment
docs (e.g. VLAs) ignore it.

Start by copying [rig-2.md](rig-2.md) and replacing every number you
haven't verified on your own rig — wrong facts are worse than no facts,
because agents trust this text over their own exploration. In particular
the overhead-camera scale is known to differ between identically built
rigs; the fact sheets state ranges and agents are told to calibrate.
`formulas.md` and `advice.md` should apply to any YAM rig as-is.

Facts should be rig-specific but task-agnostic: nothing about particular
objects or scenes, no internal provenance notes. If a fact only helps one
task, it belongs in that task's metadata instead.

- [rig-2](rig-2.md)
- [formulas](formulas.md) (all YAM rigs)
- [advice](advice.md) (all YAM rigs)
