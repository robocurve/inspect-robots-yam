#!/usr/bin/env bash
# run_batch.sh — repeat one rig task N times with a human in the loop.
#
# Run it from a rig directory (the one holding ./run and config.ini):
#
#     ../inspect-robots-yam/scripts/run_batch.sh -n 20 \
#         --instruction "Place the fork on the plate" -P model=claude-opus-5
#
# Every argument except -n/--trials passes straight through to ./run, so the
# prompt, policy, and effort are typed once and reused for every trial.
#
# Each trial is its own ./run process, on purpose. That process already does
# the two things the operator needs between trials:
#   1. After the episode it asks "did the robot succeed? [y/n/partial/skip]"
#      and waits — that is the human grading pause.
#   2. On exit it parks the arms and releases motor torque (YAMEmbodiment.close).
# So when ./run returns cleanly, the arms are off. Only then does this script
# ask you to reset the scene, and the next trial (which powers the arms back on
# and ramps them to the start pose) starts only after you press Enter. A run
# that did NOT exit cleanly gets a warning instead of the torque-off claim:
# check the arms are limp yourself before reaching in.
#
# `--epochs 1` is always forced. Within-process epochs would NOT do this:
# between epochs the arms stay connected and torque-held at the home pose while
# you reach into the scene. Any other --epochs value is rejected.
#
# Per-trial verdicts are collected from each run's eval log and written to
# <log-dir>/batches/<stamp>.tsv, with a tally printed at the end. Ctrl-C
# cancels the running trial (the framework writes a cancelled log and parks
# the arms) and ends the batch; the tally still prints.
#
# Linux/GNU assumptions (true on the rigs): GNU find -printf, python3 on PATH.
# RIG_RUN_DRY=1 is honoured by ./run (prints the command, no hardware), which
# makes this loop exercisable without a rig.

set -uo pipefail

usage() {
  cat <<'EOF'
usage: run_batch.sh [-n N] [--] <./run arguments...>

  -n, --trials N   number of trials (default 20)
  -h, --help       this text

Everything else goes to ./run unchanged (plus a forced --epochs 1), e.g.
  run_batch.sh -n 20 --instruction "Stack the blocks" -P model=claude-opus-5 -P effort=medium

Must be run from a rig directory (cwd holds ./run and config.ini).
EOF
}

die() { echo "run_batch: $*" >&2; exit 2; }

trials=20
run_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--trials)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      trials="$2"; shift 2 ;;
    -n=*|--trials=*) trials="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; run_args+=("$@"); break ;;
    *) run_args+=("$1"); shift ;;
  esac
done

# One pass over the forwarded list (including anything after `--`): find the
# task flag, the log dir, and any --epochs the operator typed.
have_task=0
log_dir=logs
epochs_seen=""
filtered=()
i=0
while [[ $i -lt ${#run_args[@]} ]]; do
  a="${run_args[$i]}"
  case "$a" in
    --instruction|--instruction=*|--task|--task=*|--auto-task) have_task=1; filtered+=("$a") ;;
    --log-dir)
      [[ $((i+1)) -lt ${#run_args[@]} ]] || die "--log-dir needs a value"
      log_dir="${run_args[$((i+1))]}"; filtered+=("$a" "$log_dir"); i=$((i+1)) ;;
    --log-dir=*) log_dir="${a#*=}"; filtered+=("$a") ;;
    --epochs)
      [[ $((i+1)) -lt ${#run_args[@]} ]] || die "--epochs needs a value"
      epochs_seen="${run_args[$((i+1))]}"; i=$((i+1)) ;;
    --epochs=*) epochs_seen="${a#*=}" ;;
    *) filtered+=("$a") ;;
  esac
  i=$((i+1))
done
if [[ -n "$epochs_seen" && "$epochs_seen" != "1" ]]; then
  die "--epochs $epochs_seen is incompatible: this script runs one epoch per process so the arms are torque-off while you reset (see header)"
fi
run_args=("${filtered[@]}" --epochs 1)

[[ "$trials" =~ ^[1-9][0-9]*$ ]] || die "trials must be a positive integer, got '$trials'"
[[ -x ./run ]] || die "no executable ./run here — cd into a rig directory first"
[[ -f ./config.ini ]] || die "no config.ini here — cd into a rig directory first"
[[ $have_task -eq 1 ]] || die "specify the task once: --instruction \"...\" (or --task NAME / --auto-task)"
if [[ ! -t 0 ]]; then
  die "stdin is not a terminal: the grading and scene-reset prompts need one (run in tmux, not with redirected stdin)"
fi
command -v python3 >/dev/null || die "python3 not found on PATH (needed to read verdicts from the eval logs)"

stamp="$(date +%Y-%m-%d_%H%M%S)"
batch_dir="$log_dir/batches"
mkdir -p "$batch_dir" || die "cannot create $batch_dir"
batch_file="$batch_dir/batch_$stamp.tsv"
# Comment line first, header second: keeps the file loadable with csv/pandas
# after skipping the leading '#' line.
{ printf '# args:'; printf ' %q' "${run_args[@]}"; printf '\n'; } > "$batch_file"
printf 'trial\texit\tstatus\tjudgement\ttermination\tduration_s\tlog\n' >> "$batch_file"

marker="$(mktemp "${TMPDIR:-/tmp}/run_batch_marker.XXXXXX")"
trap 'rm -f "$marker"' EXIT

interrupted=0
trap 'interrupted=1' INT

# Throw away anything typed while the previous run was parking. The framework
# drains before its own prompts for the same reason: a buffered Enter here
# would start the next trial and power the arms on while you are in the scene.
drain_tty() {
  while read -r -t 0.05 -n 1000 _ </dev/tty; do :; done
}

# Pull the operator verdict out of the eval log that this trial wrote.
# Prints: status<TAB>judgement<TAB>termination<TAB>duration_s
read_verdict() {
  python3 - "$1" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:  # unreadable/partial log: still report the trial
    print("unreadable\t-\t-\t-")
    sys.exit(0)
s = (d.get("samples") or [{}])[0]
def first(key):
    v = s.get(key) or []
    return str(v[0]) if v and v[0] is not None else "-"
dur = (d.get("stats") or {}).get("duration_s")
print("\t".join([
    str(d.get("status", "-")),
    first("operator_judgements"),
    first("termination_reasons"),
    f"{dur:.0f}" if isinstance(dur, (int, float)) else "-",
]))
PY
}

declare -a rows=()
completed=0
yes=0; no=0; partial=0; other=0
bold=$'\e[1m'; dim=$'\e[2m'; red=$'\e[31m'; reset=$'\e[0m'

tally_add() {
  case "$1" in
    y|yes) yes=$((yes+1)) ;;
    n|no) no=$((no+1)) ;;
    partial) partial=$((partial+1)) ;;
    *) other=$((other+1)) ;;
  esac
}

summary() {
  echo
  echo "${bold}batch summary${reset}  ($completed/$trials trials run)"
  printf '%-6s %-8s %-10s %-14s %s\n' trial judged status termination log
  for r in "${rows[@]}"; do
    IFS=$'\t' read -r t _ex st jd tr _du lg <<<"$r"
    printf '%-6s %-8s %-10s %-14s %s\n' "$t" "$jd" "$st" "$tr" "$lg"
  done
  echo
  echo "success: $yes   failure: $no   partial: $partial   ungraded/other: $other"
  echo "${dim}batch record: $batch_file${reset}"
}

for ((i = 1; i <= trials; i++)); do
  echo
  echo "${bold}=== trial $i/$trials ===${reset}"
  touch "$marker"
  ./run "${run_args[@]}"
  rc=$?
  completed=$((completed+1))

  # The run's eval log is the newest final JSON written to the log dir since
  # the marker. *.live.json is the transient snapshot the framework unlinks on
  # a clean exit; after a crash it survives and is not the eval log.
  log_path="$(find "$log_dir" -maxdepth 1 -name '*.json' ! -name '*.live.json' -newer "$marker" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -n "$log_path" ]]; then
    verdict="$(read_verdict "$log_path")"
    [[ -n "$verdict" ]] || verdict=$'?\t?\t?\t?'
  else
    verdict=$'-\t-\t-\t-'
    log_path="(no log written)"
  fi
  row="$(printf '%s\t%s\t%s\t%s' "$i" "$rc" "$verdict" "$log_path")"
  rows+=("$row")
  printf '%s\n' "$row" >> "$batch_file"
  IFS=$'\t' read -r _ _ st jd tr du _ <<<"$row"
  tally_add "$jd"
  echo
  echo "trial $i/$trials: judged ${bold}$jd${reset} ($st, $tr, ${du}s) — so far $yes success / $no failure / $partial partial"
  echo "${dim}log: $log_path${reset}"

  if [[ $interrupted -eq 1 ]]; then
    echo "run_batch: interrupted — stopping after trial $i"
    break
  fi

  # 0 = normal end; 130 = the framework's clean Ctrl-C cancel (still parks and
  # releases). Anything else means close() may not have run: no torque claim.
  clean_exit=0
  [[ $rc -eq 0 || $rc -eq 130 ]] && clean_exit=1
  if [[ $clean_exit -eq 0 ]]; then
    stty sane </dev/tty 2>/dev/null   # a killed child can leave the tty in no-echo cbreak mode
    echo "${red}run_batch: trial $i exited with status $rc — the run did not finish cleanly.${reset}"
    echo "${red}Confirm both arms are limp (torque off) before reaching into the scene.${reset}"
    drain_tty
    read -r -p "continue with the remaining trials? [y/N] " ans </dev/tty || ans=n
    [[ "$ans" =~ ^[Yy] ]] || break
  fi

  if [[ $i -lt $trials ]]; then
    echo
    if [[ $clean_exit -eq 1 ]]; then
      echo "${bold}Arms are parked and torque is released.${reset}"
    fi
    echo "Reset the scene for trial $((i+1))/$trials, then stand clear."
    echo "${dim}(Enter starts the next trial — the arms power on and ramp to the start pose immediately; q quits)${reset}"
    drain_tty
    read -r -p "> " ans </dev/tty || ans=q
    if [[ "$ans" =~ ^[Qq] ]]; then
      echo "run_batch: stopped by operator after trial $i"
      break
    fi
    if [[ $interrupted -eq 1 ]]; then
      break
    fi
  fi
done

summary
