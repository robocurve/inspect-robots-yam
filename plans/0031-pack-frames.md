# 0031 — Pack camera frames to H.264 evidence video (+ Drive backup)

Status: backlog executed 2026-09-02 (65 runs, 0 failures, 250 GiB on Drive). Issue #145, PR #146. Companion: rig-dashboard `feat/adopt-packed-clips`.
Where a step below conflicts with the "Amendments" section, the amendments win.


## Context

omen's root disk (1.9 TB) is 100% full (333 MB free). Cause: `store_frames = true` in every rig's
`config.ini` makes inspect-robots save every control step's three camera images as raw `.npy`
(`<rig>/logs/frames/<stamp>/scene-0-e0_{left,top,right}_cam_NNNNNN.npy`, uint8 HxWx3, 691 KB each at
360x640). Today's pi0.5 batches run at 30 Hz with 360x640 frames for 3600-12800 steps, so one trial
writes 7.5-26.5 GB; three rigs ran `run-batch -n 20` in parallel and wrote ~840 GB in a day. Two
batch trials (rig-1, rig-2) are hung mid-write since the disk filled at 19:09.

Inventory (verified 2026-09-01): 486 frames dirs, 462 populated, exactly 3 streams each, contiguous
steps from 0, no gaps, no non-.npy files. 360x640 (pi0.5): 83 dirs, 854 GB, 1.24 M files. 224x224
(LLM-agent runs): 379 dirs, 86 GB. 24 empty dirs. Per-step commanded actions live separately in
`logs/actions/<stamp>/scene-0-e0.jsonl` (2 MB/run, untouched by this plan).

User decisions (fixed):
- Frames are kept as **evidence video**, not as reconstructible arrays. Format: libx264, yuv420p,
  crf 16, preset slow, one MP4 per camera per run, fps = run's `control_hz`. Measured 85-245x
  (940 GB -> ~5-10 GB), per-frame PSNR >= 37.9 dB on real frames.
- Initial backlog: **only 360x640 runs**. 224x224 dirs stay as `.npy` (their transcript stills
  depend on it).
- Backup: **Google Drive** via rclone, into the `robocurve` shared drive (org-owned), before any
  deletion. Existing remote `gdrive-rc` is jay@robocurve.org's token (verified robocurve workspace).
- Steady state: `run_batch.sh` packs each trial after grading. Single `./run` runs: manual command.
- The rig-dashboard must keep showing clips for packed runs. No daemons/systemd.

Consumers of the `.npy` layout (verified): rig-dashboard `clips.py` renders clips by running
`inspect-robots video` and returns False -> permanent `<name>.noclips` when the frames dir has no
`.npy`; `/thumb` reads `.npy` only for **live** runs (safe); `inspect-robots view` degrades silently
to no stills; `inspect-robots video` exits "no frames found". None of the 30 existing `.noclips`
markers belong to a populated 360x640 run (16 are 224x224, 14 are empty aborted dirs).

## Deliverables and file tree

```
inspect-robots-yam/                         (branch feat/pack-frames from origin/main, after #143 merges)
  scripts/pack_frames.py                    NEW  packer (numpy + ffmpeg/ffprobe/rclone subprocesses)
  scripts/pack_frames                       NEW  bash wrapper: exec ../shared/.venv/bin/python pack_frames.py
  scripts/run_batch.sh                      MOD  post-trial detached pack hook, --no-pack flag
  tests/test_pack_frames.py                 NEW
  README.md, CHANGELOG.md                   MOD
rig-dashboard/                              (branch feat/adopt-packed-clips)
  src/rig_dashboard/clips.py                MOD  adopt packed MP4s; .noclips retry (local hosts only)
  tests/test_clips.py                       MOD  adoption tests
  CLAUDE.md, README.md                      MOD
rig-{1,2,6}/pack-frames -> ../inspect-robots-yam/.claude/worktrees/pack-frames/scripts/pack_frames   NEW symlinks
rig-{1,2,6}/CLAUDE.md                       MOD  gotcha + usage
~/.config/rclone/rclone.conf                MOD  add [gdrive-robocurve] remote (same token, team_drive=0ACXvkf_ip4sPUk9PVA)
memory: frames-pack.md (+ MEMORY.md line); update run-batch-script.md, rig-dashboard.md
```

## Step 0: git setup (Fable, no Codex)

1. `gh pr merge 143 --squash -R robocurve/inspect-robots-yam` (keep branch; three live `run-batch`
   bash processes still execute the worktree copy of `run_batch.sh`, never edit that file in place).
2. `git -C ~/robocurve/robocurve/inspect-robots-yam fetch origin && git worktree add
   .claude/worktrees/pack-frames -b feat/pack-frames origin/main`.
3. Issue "Pack camera frames to H.264 + Drive backup (disk full)", self-assign, draft PR `Closes #N`.
   Plan-critique loop with a fresh subagent before dispatching Codex. Codex writes files only, no git.
4. rig-dashboard: branch `feat/adopt-packed-clips`, draft PR.

## Step 1: `scripts/pack_frames.py`

Stdlib + numpy, Python >= 3.10 syntax (ruff `target-version py310`, `D1` docstrings, `UP`), type
hints. Every function that shells out takes an injectable `run=subprocess.run`. Mirror core's
`frames_dir_candidates` and `default_fps` logic from `inspect_robots/_video.py` (reference only,
never patch site-packages).

CLI: `--run LOG.json | --all | --status`; `--rig DIR` (default cwd, must contain `config.ini` and
`logs/`); `--min-height 360` (applies to `--run` too, so the batch hook never packs 224x224 runs;
pass `--min-height 0` to override); `--limit N`; `--dry-run`; `--keep` (never unlink); `--force`
(skip grace); `--grace 600`; `--no-upload`; `--allow-unbacked-delete` (refused unless `--no-upload`
also given); `--remote gdrive-rc:rig-video`; `--host-label` (default `hostname -s`);
`--ffmpeg/--ffprobe/--rclone` (default `~/.local/bin/*` if present else PATH); `--threads 8`;
`--crf 16`; `--preset slow`; `--psnr-min 35`; `--sample-every 200`. Exit codes: 0 packed,
3 skipped/not eligible, 1 failure, 2 usage.

Pipeline per run (`pack_one`), under a blocking per-rig `fcntl.flock` on `logs/pack/.lock`, logging to
`logs/pack/<stamp>.log` and stderr:

1. `load_run`: parse JSON; `frames_dir = rig/stats.frames_dir` with fallback
   `rig/logs/frames/<basename>`; `control_hz` with core's guards (numeric, not bool, > 0, finite;
   else 10); `status`; log mtime.
2. `check_eligible` -> skip reason or None: JSON missing; `status == "started"`; `<name>.live.json`
   exists; no `.npy`; `pack_manifest.json` says packed **and** per-stream sha256 match on-disk MP4s
   (MP4 presence alone is NOT "packed": core's `video --out` default writes identically named
   `scene-0-e0_{cam}_cam.mp4` into the frames dir); frame height < `--min-height`
   (header-only read via `np.lib.format`). Grace is not checked here.
3. `discover_streams`: regex `^scene-0-e0_(left|top|right)_cam_(\d{6,})\.npy$`; fail on strays or
   non-increasing steps; record first/last step and gaps in the manifest.
4. Remove stale `*.mp4.tmp`. `encode_stream` per camera: `Popen` ffmpeg with `stdin=PIPE`, stderr
   to a tmpfs temp file (`/tmp` is a 31 GB tmpfs, unaffected by the full disk); stream each `.npy`
   via `np.load(mmap_mode="r")`, assert uint8 and constant (h,w,3), write contiguous bytes.
   ```
   ffmpeg -hide_banner -nostats -loglevel error -y \
     -f rawvideo -pix_fmt rgb24 -s WxH -framerate HZ -i - \
     -vf pad=ceil(iw/2)*2:ceil(ih/2)*2 -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
     -threads 8 -fps_mode passthrough -movflags +faststart -f mp4 \
     <frames_dir>/scene-0-e0_{cam}_cam.mp4.tmp
   ```
   (`-f mp4` is required because `.tmp` hides the container from ffmpeg.)
5. `probe_frame_count`: `ffprobe -v error -select_streams v:0 -count_packets -show_entries
   stream=nb_frames,nb_read_packets,width,height -of json`; require `nb_frames == nb_read_packets ==
   npy count` and matching (padded) dimensions. Verified reliable on faststart output.
6. `verify_psnr`: decode only sampled frames (`0`, every 200th, last) with
   `ffmpeg -i mp4 -vf "select='not(mod(n\,200))+eq(n\,LAST)'" -fps_mode passthrough -f rawvideo
   -pix_fmt rgb24 -`, crop padding, compare to the same `.npy`; fail if any frame < `--psnr-min`
   or byte count mismatches.
7. `write_manifest` (atomic tmp + `os.replace`) `pack_manifest.json`: tool + version, host, rig,
   run name, rig-relative log path, stamp, control_hz, timestamps, ffmpeg version + argv, per
   stream {file, sha256, bytes, frames, width, height, first_step, last_step, gaps, psnr_samples},
   `state` in encoded | uploaded | packed | packed-kept, remote path, `npy_bytes_freed`.
   Rename `.mp4.tmp` -> `.mp4`.
8. `upload`: `rclone copy --checksum --transfers 4 --drive-chunk-size 64M --include
   'scene-0-e0_*_cam.mp4' --include pack_manifest.json <frames_dir>
   <remote>/<host>/<rig>/<stamp>/` then `rclone check --one-way` with the same includes (Drive
   exposes md5, so this is real verification). Any non-zero exit: state stays `encoded`, MP4s stay,
   `.npy` untouched, exit 1. Never delete on an rclone failure (the shared client_id retires in
   2026; a silent auth expiry must not cause unbacked deletion).
9. `delete_npy`: requires state `uploaded` (or `--no-upload --allow-unbacked-delete`); requires
   `now - log_mtime >= grace` else **sleep until it is** (the hook is detached, waiting is fine; log
   it). Grace protects the dashboard's up-to-300 s `inspect-robots video` render which reads `.npy`.
   Recount the stream file set immediately before unlinking; abort if it changed. Unlink only
   enumerated `.npy`; keep the directory (so `stats.frames_dir` still resolves), MP4s and manifest.
   `--keep` -> state `packed-kept`, nothing unlinked. Else state `packed`.

`--all`: glob `logs/*.json` excluding `*.live.json`, filter eligible, sort by total `.npy` bytes
ascending (smallest first so the first packs free space on a full disk; smallest 360 dirs are 3, 6,
201 frames). `--status`: counts of packable / packed / skipped by reason / orphan frames dirs (no
JSON references them: the 24 empties, the two hung trials' dirs, ~5 orphan 360 dirs ~28 GB) / GB
remaining at >= min-height. Orphan dirs are reported, not packed (no run metadata); decide later.

## Step 2: wrapper and symlinks

`scripts/pack_frames` (chmod +x): `[[ -f ./config.ini && -d ./logs ]] || exit 2`; `py=../shared/.venv/bin/python`
(fallback `python3`); `exec "$py" "$(dirname "$(readlink -f "$0")")/pack_frames.py" "$@"`.
Symlink `rig-{1,2,6}/pack-frames` at the worktree copy (same pattern as `run-batch`); re-point to
`../inspect-robots-yam/scripts/pack_frames` after merge.

## Step 3: `run_batch.sh` hook

Add `--no-pack` (default pack on) to usage/parsing. Insert right after the TSV row is appended and
`echo "log: $log_path"`, before the interrupt check and the scene-reset prompt:
```bash
if [[ $pack -eq 1 && -f "$log_path" ]]; then
  if [[ -x ./pack-frames ]]; then
    mkdir -p "$log_dir/pack"
    setsid nohup nice -n 19 ./pack-frames --run "$log_path" >>"$log_dir/pack/batch.log" 2>&1 </dev/null &
    echo "packing frames in the background (log: $log_dir/pack/batch.log)"
  else
    echo "run_batch: warning: ./pack-frames not found; frames left as .npy" >&2
  fi
fi
```
`setsid` so it survives the tmux window; `</dev/null` so it never touches the operator tty. The
packer's own grace wait handles the dashboard race; the flock serializes overlapping packs.

## Step 4: rig-dashboard `clips.py` adoption

Files: `src/rig_dashboard/clips.py` (`poll_once` ~175-233, `_already_handled` 153-165,
`_render_local` 245-274, `_render_remote` 300-355, `clips_for` memo 128-132).

- `_frames_path(host, run)` helper extracted from `_render_local`.
- `_adoptable_local(host, run)`: local host only; returns present non-empty
  `scene-0-e0_{cam}_cam.mp4` paths in the frames dir, else None.
- `_adopt_local(sources, key)`: copy (not hardlink; `--media-dir` may be another fs) to
  `media/<host>/<rig>/<name>_{cam}.mp4` via tmp + replace (reuse `_store_local_outputs` shape);
  unlink `.noclips` if present; `self._memo.pop(key)`; `self._handled.add(key)`.
- `poll_once` per candidate: skip if in `_handled` or any clip file exists; validate frames_dir as
  today; if local and adoptable -> adopt and continue; if `.noclips` exists: remote host -> add to
  `_handled` (permanent, as before; an ssh per poll is unacceptable), local host -> just continue
  (re-checked next poll, 4 stats); else render as today.
- `_render_remote`: first `ssh ls -- <frames_dir>`; if it lists `scene-0-e0_*_cam.mp4`, `cat` those
  with **timeout 120** (current 15 s is too short for 100-300 MB files) and skip rendering; else
  existing flow, also with the 120 s fetch timeout.
- Docs: CLAUDE.md gotchas replace "never transfer frame directories" with "adopt packed
  `scene-0-e0_{cam}_cam.mp4` from the frames dir; `.noclips` permanent for remote hosts,
  re-checked for local hosts"; README "Video clip cache" paragraph likewise, plus a note that packed
  runs show clips from the packer's MP4s and transcripts render without stills.

## Step 5: tests

rig-dashboard `tests/test_clips.py` (existing `FakeCollector`, `run_summary`, monkeypatched
`subprocess.run`): local adopts MP4s without calling ffmpeg; adoption clears `.noclips` and the
`clips_for` memo; local `.noclips` without MP4s is not re-rendered (regression); remote `.noclips`
stays permanent with no ssh; remote adopts listed MP4s with timeout 120 and no `video` command;
existing render-from-npy test still passes. Run `uv run pytest -q` in the repo.

inspect-robots-yam `tests/test_pack_frames.py` (import via `importlib.util.spec_from_file_location`,
tools injected): eligibility matrix (started, live.json, missing dir, packed with matching sha,
mismatched sha -> re-pack, height < min); `discover_streams` rejects strays/non-monotonic; `--all`
ordering smallest-first and height filter with synthetic headers; state machine (upload failure keeps
`.npy` and state `encoded`; `--no-upload` alone refuses; `--keep` never unlinks); grace wait with
mocked clock and `--force`; ffmpeg integration test `skipif(shutil.which("ffmpeg") is None)` on six
synthetic 36x64 gradient frames (count 6, PSNR >= 35). CI runs on ubuntu + macOS without ffmpeg.
Gates: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest --cov`.

Review Codex's diff with `git diff -- tests/` separately; no pre-existing test may change.

## Step 6: end-to-end on one small real run before trusting deletion

1. Add the rclone remote (Step 7.1) and `rclone mkdir gdrive-rc:rig-video-test`.
2. `cd ~/robocurve/robocurve/rig-1 && ./pack-frames --status`; pick the 201-frame run
   (`logs/frames/20260901_193641_5847f8ba`).
3. `./pack-frames --run logs/<name>.json --keep --remote gdrive-rc:rig-video-test`: check
   three MP4s + manifest in the dir, ffprobe counts 67 per stream, `rclone ls` shows them, dashboard
   row (after Step 4 deploy) shows/plays clips.
4. Re-run with `--force` and no `--keep`: `.npy` gone, manifest `packed`, dashboard clips still
   present, `inspect-robots view <log> -o /tmp/x.html --no-video --frames-budget 12` succeeds,
   `inspect-robots video <log>` exits "no frames found" (expected).
5. Re-run again -> exit 3 "already packed". `rclone purge gdrive-rc:rig-video-test`.

## Step 7: backlog runbook

1. rclone: append to `~/.config/rclone/rclone.conf` a `[gdrive-robocurve]` section, `type = drive`,
   same `token` as `gdrive-rc`, `team_drive = 0ACXvkf_ip4sPUk9PVA`. Verify `rclone lsd
   gdrive-rc:`; `rclone mkdir gdrive-rc:rig-video`.
2. Deploy dashboard first: merge PR, `cd ~/robocurve/robocurve/rig-dashboard && ./stop && git pull
   && ./start`; confirm no tracebacks in the `dashboard` tmux session.
3. Launch per rig in tmux, rig-2 first (smallest dirs), at most 2 concurrent while trials are live:
   ```
   tmux new -d -s pack-rig2 'cd ~/robocurve/robocurve/rig-2 && ./pack-frames --all --min-height 360 2>&1 | tee -a logs/pack/all_$(date +%F_%H%M).log'
   ```
   then `pack-rig6`, `pack-rig1`. Monitor: `tail -f ~/robocurve/robocurve/rig-*/logs/pack/all_*.log`,
   `df -h /`, `./pack-frames --status`, `rclone size gdrive-rc:rig-video`.
   Estimate: 1.24 M frames at ~200 fps per worker, 8 threads, nice 19 -> about 2-3 h with 2-3
   workers; upload 5-10 GB ~10 min. Expect ~845 GB freed.
4. Hung trials (rig-1 tty pts/19, rig-2 pts/0): their dirs have no JSON so `--all` skips them.
   Operator Ctrl-C's them (no `kill -9`, arms torque state) after the first packs free space, so the
   framework can write the cancelled JSON; those dirs then become packable. rig-6's `run-batch` is
   idle at its reset prompt.
5. Post-pack: `./pack-frames --status` shows 0 unpacked >= 360 on each rig; `rclone check --one-way`
   per rig against `gdrive-rc:rig-video/omen/<rig>`; open three packed rows on the dashboard
   and play clips; `df -h /`.
6. After the three `run-batch` processes exit: `git pull` main checkout, re-point `run-batch` and
   `pack-frames` symlinks to `../inspect-robots-yam/scripts/...`, remove both worktrees.

## Step 8: docs and memory

- `rig-{1,2,6}/CLAUDE.md` gotcha: store_frames at 30 Hz x 360x640 writes 7.5-26.5 GB per trial;
  `run-batch` packs each trial in the background (`./pack-frames`, logs in `logs/pack/`); single
  `./run` runs need `./pack-frames --run logs/<name>.json`; packed dirs keep MP4s + manifest, lose
  `.npy` (transcript stills vanish, dashboard clips come from the MP4s); backup path
  `gdrive-rc:rig-video/omen/<rig>/<stamp>/`; 224x224 runs stay `.npy`.
- inspect-robots-yam README section + CHANGELOG entry; rig-dashboard docs per Step 4.
- Memory `frames-pack.md` (+ MEMORY.md line); update `run-batch-script.md` (merged, re-pointed) and
  `rig-dashboard.md` (adoption).
- Follow-ups to file: own OAuth client_id for rclone Drive (shared id retires 2026); upstream
  inspect-robots issue to log observed joint state per step and to store frames compressed at
  capture time.

## Amendments from the independent critique (applied to the Codex briefs)

- Drive target is the `pi05_kaedim_tasks` shared drive (id 0AGNB3pVRo9vkUk9PVA), reached via the
  existing `gdrive-rc` remote plus `--drive-team-drive` (no new remote section: copying the OAuth
  token is classifier-blocked). Every `--remote` is `gdrive-rc:rig-video`.
- Stage on tmpfs: encode, verify and upload from a per-run scratch dir under `/tmp` (31 GB tmpfs);
  delete `.npy` first, then move MP4s + manifest into the frames dir, then `rclone copyto` the final
  manifest. Nothing is written to the full root disk before the `.npy` are gone.
- Stale-tmp glob `scene-0-e0_*_cam.mp4.tmp*` (faststart writes a second scratch file).
- `rclone check` covers the three MP4s only; the manifest is re-uploaded after its state change.
- Re-hash staged MP4s against the manifest immediately before unlinking `.npy`.
- Packed detection compares size+mtime first; sha256 only on mismatch or `--verify`.
- Lock and logs live under `<logs-dir>/pack/` where logs-dir is the run JSON's parent, so the hook
  and the packer agree; ENOSPC on the log file falls back to stderr.
- Hook uses `nice -n 19 ionice -c3`.
- Dashboard `_adopt_local` hardlinks first and falls back to copy on OSError.
- Dashboard `_mark_no_clips` no longer adds LOCAL keys to `_handled`, so an in-process render failure
  is retried by adoption without a restart. Sanctioned pre-existing test edit: the remote render test
  now expects 4 ssh calls (ls 15 s, video 300 s, two cats 120 s).
- PR #146 is stacked on `feat/run-batch` (base branch) because `gh pr merge` is classifier-blocked
  for Fable; the user merges #143 and GitHub retargets #146 to main.
- Runbook order: Ctrl-C the two hung trials BEFORE launching the backlog so their writers cannot
  race the first packs for the freed space.

## Failure handling summary

| Failure | Result |
|---|---|
| ENOSPC on `.mp4.tmp` or log | ffmpeg non-zero -> tmp removed, exit 1, `.npy` intact; `--all` continues |
| count mismatch or PSNR < 35 | run fails, `.npy` kept, tmp removed, logged for investigation |
| rclone copy/check fails (auth, network) | state `encoded`, MP4s kept, `.npy` kept, exit 1 |
| dashboard mid-render when `.npy` vanish | prevented by grace >= 600 s; if it still writes `.noclips`, local retry adopts MP4s next poll |
| packer interrupted | next run removes `*.mp4.tmp`, re-encodes streams whose sha mismatches, resumes from manifest state |
| someone ran `inspect-robots video` into the dir | same-named crf-23 MP4s; no manifest match -> packer re-encodes and replaces |
| two packers on one rig | flock serializes |
| `control_hz` missing | 10, as core does; recorded in manifest |

## Explicitly out of scope

224x224 frames dirs; `.rrd` files (11.6 GB), `logs/wire` (8 GB), stray `~/robocurve/DSCF0083.MOV`
(15 GB) and other non-frame disk consumers; turning `store_frames` off; upstream core changes.
