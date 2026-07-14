# inspect-robots-yam — agent guide

Inspect Robots adapters that let evals (e.g. [KitchenBench](https://github.com/robocurve/kitchenbench))
run on real **I2RT YAM bimanual arms** driven by **MolmoAct2**. This is a
**plugin package** in the Inspect Robots ecosystem — the framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots); benchmarks are separate repos
indexed by [WorldEvals](https://github.com/robocurve/worldevals).

## The one big idea

Inspect Robots evals swap two inputs: a `Policy` (VLA brain) and an `Embodiment` (robot
body + world). We ship both for one real stack:

- **`molmoact2` policy** — a thin HTTP client for MolmoAct2's first-party
  bimanual-YAM `/act` server. The model runs in its own process (GPU + weights);
  we never import torch here.
- **`yam_arms` embodiment** — the i2rt joint-position driver.

Both declare the **same 14-D `joint_pos` contract** (2 arms × [6 joints +
gripper], cameras `top/left/right`, packed `joint_pos` state). That makes
`inspect_robots.compat.check_compatibility` pass with zero errors **and** zero warnings
— the property `tests/test_compat.py` locks down.

## Layout

- `src/inspect_robots_yam/` — the package (see `src/inspect_robots_yam/CLAUDE.md`).
- `tests/` — pytest; everything (driver, cameras, `/act`, clock, operator stdin)
  is injected, so the suite needs **no hardware, no server, no stdin**.
- `plans/0001-yam-molmoact2-design.md` — the design + plan (approved after two
  adversarial subagent critique rounds). Read it before changing the contract.

## Working here

- Dev loop: `uv venv && uv pip install -e ".[dev]"`, `uv run pre-commit install`,
  then `uv run pytest --cov`.
- **Local install gotcha:** `uv pip install -e ".[dev]"` resolves inspect-robots +
  kitchenbench from git tags. To work against sibling checkouts instead:
  `uv pip install -e ../inspect-robots && uv pip install --no-deps -e ../kitchenbench`
  (the `--no-deps` avoids an inspect-robots URL conflict with kitchenbench's own
  `tool.uv.sources`).
- Gates (all blocking in CI): `ruff check .`, `ruff format --check .`,
  `mypy` (strict), `pytest --cov` at **100%**.
  Every public module, class, and function needs a docstring, enforced by Ruff
  D1; state the contract instead of restating the symbol name.
- **mypy + numpy:** numpy 2.5's stubs use 3.12-only syntax that mypy (py3.10
  target) rejects; the dev extra pins `numpy<2.5` and CI runs mypy on 3.11.
- **No torch.** The model lives in the MolmoAct2 server. The base dependencies
  include `requests`, `json_numpy`, and OpenCV, but those modules remain lazily
  imported behind seams. The git-only `i2rt` driver is also loaded lazily and
  produces guided installation help when absent. The `import-hygiene` CI job
  enforces that `import inspect_robots_yam` works with only inspect_robots + numpy.

## Safety invariants (do not weaken)

- `YAMEmbodiment.step()` **always clamps** to `YamConfig.joint_low/high` before
  commanding, independent of any `Approver`. This is the last line of defense.
- The declared `control_mode` is `joint_pos` (absolute). Delta checkpoints are
  converted to absolute *inside* `step()` (`joints_are_delta=True`) so the
  declared semantics stay honest. There is no `joint_delta` control mode in
  Inspect Robots, so compat cannot verify abs-vs-delta — that's a hardware check.
- Success reaches the scorer **only** via `StepResult.termination_reason="success"`
  (stock `rollout` never sets `operator_judgement`).

## Out of scope

Launching/serving MolmoAct2 (that's the `allenai/molmoact2` repo), single-arm or
non-YAM I2RT robots, and model fine-tuning.

## CI, merging, and releases

- **main is PR-only** — a branch ruleset (admins included) blocks direct pushes,
  force pushes, and deletion. Merging requires the `ci-ok` check green and the
  branch up to date with main.
- **`ci-ok` is the single required status check** — an aggregate job at the end
  of `ci.yml`. When adding a CI job, add it to `ci-ok`'s `needs` list, or it
  will not gate merges.
- **Red main is stop-the-line**: if CI fails on a push to main, the
  `alert-red-main` job opens an issue. Fix forward or revert before merging
  anything else; if the failure was transient, re-run the failed jobs and close
  the issue.
- **CI installs from `uv.lock`** (`uv sync --locked`). After changing
  dependencies in `pyproject.toml`, run `uv lock` and commit the lockfile —
  otherwise CI fails with "the lockfile needs to be updated".
- A weekly **canary** (`canary.yml`) does the opposite: it installs the latest
  dependency versions the pyproject ranges allow (ignoring the lockfile), runs
  the tests, and opens an issue on failure — catching ecosystem breakage that
  locked CI can't see. A green canary means `uv lock --upgrade` is safe.
- The `i2rt` driver is git-only and intentionally absent from `uv.lock`. Install
  it with `uv pip install "i2rt @ git+https://github.com/i2rt-robotics/i2rt"`.
- **Releases are one-click**: Actions → Release → Run workflow → pick
  patch/minor/major. The version is derived from the git tag by hatch-vcs —
  never add a static `version =` back to pyproject (`__version__` comes from importlib.metadata). The same
  run publishes to PyPI via trusted publishing; nothing is pushed to main.
- **PyPI readme is transformed at build time** — `hatch-fancy-pypi-readme`
  rewrites GitHub-only alert syntax (`> [!NOTE]` etc.) in README.md into bold
  blockquotes (`> **Note:**`) that PyPI renders; keep using alert syntax in the
  README itself. Config lives at the bottom of pyproject.toml.

## Writing style (public-facing text)

READMEs, docs pages, repo/collection descriptions, and HF model cards must
avoid AI-writing tells. The full rule with the gating checklist lives in
[worldevals docs/model-cards.md, "Writing style"](https://github.com/robocurve/worldevals/blob/main/docs/model-cards.md);
short version:

- No em dashes in prose. Use periods, colons, commas, or parentheses (`—` is
  fine as an empty table cell and inside code blocks).
- Bold only for definition-list lead-ins (`**term:**`) and at most one critical
  imperative per safety bullet. Never mid-sentence for emphasis.
- No decorative emoji (functional ✅/⚠️ marks and 🤗 for Hugging Face are fine),
  no slogans or chiasmus, no "not just X, but Y".
- Headers use colons, never em dashes or italics.

Style-only edits must never touch YAML frontmatter, code blocks, numbers,
links, or safety qualifiers.
