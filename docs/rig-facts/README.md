# Rig facts

Per-rig fact sheets injected into agent-policy system prompts via
`YamConfig.docs_extra` (the run wrapper passes
`-E "docs_extra=$(cat rig-facts.md)"`). Each file collects the measured and
derived facts about one physical rig — kinematic constants, camera
geometry, controller behavior — so episode agents don't burn steps
re-deriving them.

Facts should be rig-specific but task-agnostic: nothing about particular
objects or scenes, no internal provenance notes. If a fact only helps one
task, it belongs in that task's metadata instead.

- [rig-2](rig-2.md)
