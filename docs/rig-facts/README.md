# Rig facts

Per-rig fact sheets for agent-policy runs. Each file collects the measured
and derived facts about one physical rig — kinematic constants, camera
geometry, controller behavior — so episode agents don't burn steps
re-deriving them.

## Using a fact sheet on your rig

`YamConfig.docs_extra` appends arbitrary text to the embodiment docs that
agent policies inject into their system prompt (as "Embodiment notes:").
Feed your rig's sheet to a run with:

```bash
inspect-robots run --policy agent \
  -E "docs_extra=$(cat rig-facts.md)" \
  ...
```

or bake it into a small per-rig run wrapper so every run gets it. Policies
that don't read embodiment docs (e.g. VLAs) ignore it. Start by copying
[rig-2.md](rig-2.md) and replacing every number you haven't verified on
your own rig — wrong facts are worse than no facts, because agents trust
this text over their own exploration.

Facts should be rig-specific but task-agnostic: nothing about particular
objects or scenes, no internal provenance notes. If a fact only helps one
task, it belongs in that task's metadata instead.

- [rig-2](rig-2.md)
