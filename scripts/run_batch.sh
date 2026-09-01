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
# So when ./run returns, the arms are off. Only then does this script ask you
# to reset the scene, and the next trial (which powers the arms back on and
# ramps them to the start pose) starts only after you press Enter.
#
# `--epochs N` in a single run would NOT do this: between epochs the arms stay
# connected and torque-held at the home pose while you reach into the scene.
#
# Per-trial verdicts are collected from each run's eval log and written to
# <log-dir>/batches/<stamp>.tsv, with a tally printed at the end. Ctrl-C ends
# the batch after the current trial and still prints the tally.
#
# Env: RIG_RUN_DRY=1 is honoured by ./run (prints the command, no hardware),
# which makes this loop exercisable without a rig.

set -uo pipefail

usage() {
  cat <<'EOF'
usage: run_batch.sh [-n N] [--] <./run arguments...>

  -n, --trials N   number of trials (default 20)
  -h, --help       this text

Everything else goes to ./run unchanged, e.g.
  run_batch.sh -n 20 --instruction "Stack the blocks" -P model=claude-opus-5 -P effort=medium

Must be run from a rig directory (cwd holds ./run and config.ini).
EOF
}

die() { echo "run_batch: $*" >&2; exit 2; }

trials=20
run_args=()
log_dir=logs
have_task=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--trials)
      [[ $# -ge 2 ]] || die "$1 needs a value"
      trials="$2"; shift 2 ;;
    -n=*|--trials=*) trials="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; run_args+=("$@"); break ;;
    --epochs|--epochs=*)
      die "--epochs is incompatible: this script runs one epoch per process so the arms are off while you reset (see header)" ;;
    --instruction|--instruction=*|--task|--task=*|--auto-task)
      have_task=1; run_args+=("$1")
      if [[ "$1" == "--instruction" || "$1" == "--task" ]]; then
        [[ $# -ge 2 ]] || die "$1 needs a value"
        run_args+=("$2"); shift
      fi
      shift ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "--log-dir needs a value"
      log_dir="$2"; run_args+=("$1" "$2"); shift 2 ;;
    --log-dir=*) log_dir="${1#*=}"; run_args+=("$1"); shift ;;
    *) run_args+=("$1"); shift ;;
  esac
done
# Anything after `--` may still carry a task flag or an --epochs the loop above
# never saw; check the whole forwarded list once.
for a in "${run_args[@]}"; do
  case "$a" in
    --instruction|--instruction=*|--task|--task=*|--auto-task) have_task=1 ;;
    --epochs|--epochs=*) die "--epochs is incompatible (see header)" ;;
  esac
done

[[ "$trials" =~ ^[1-9][0-9]*$ ]] || die "trials must be a positive integer, got '$trials'"
[[ -x ./run ]] || die "no executable ./run here — cd into a rig directory first"
[[ -f ./config.ini ]] || die "no config.ini here — cd into a rig directory first"
[[ $have_task -eq 1 ]] || die "specify the task once: --instruction \"...\" (or --task NAME / --auto-task)"
if [[ ! -t 0 ]]; then
  die "stdin is not a terminal: the grading and scene-reset prompts need one (run in tmux, not with redirected stdin)"
fi

stamp="$(date +%Y-%m-%d_%H%M%S)"
batch_dir="$log_dir/batches"
mkdir -p "$batch_dir" || die "cannot create $batch_dir"
batch_file="$batch_dir/batch_$stamp.tsv"
printf 'trial\texit\tstatus\tjudgement\ttermination\tduration_s\tlog\n' > "$batch_file"
{ printf '# args:'; printf ' %q' "${run_args[@]}"; printf '\n'; } >> "$batch_file"

marker="$(mktemp "${TMPDIR:-/tmp}/run_batch_marker.XXXXXX")"
trap 'rm -f "$marker"' EXIT

interrupted=0
trap 'interrupted=1' INT

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
bold=$'\e[1m'; dim=$'\e[2m'; reset=$'\e[0m'

summary() {
  echo
  echo "${bold}batch summary${reset}  ($completed/$trials trials run)"
  printf '%-6s %-8s %-10s %-14s %s\n' trial judged status termination log
  local yes=0 no=0 partial=0 other=0
  for r in "${rows[@]}"; do
    IFS=$'\t' read -r t ex st jd tr du lg <<<"$r"
    printf '%-6s %-8s %-10s %-14s %s\n' "$t" "$jd" "$st" "$tr" "$lg"
    case "$jd" in
      y|yes) yes=$((yes+1)) ;;
      n|no) no=$((no+1)) ;;
      partial) partial=$((partial+1)) ;;
      *) other=$((other+1)) ;;
    esac
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

  # The run's eval log is the JSON written to the log dir since the marker.
  log_path="$(find "$log_dir" -maxdepth 1 -name '*.json' -newer "$marker" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -n "$log_path" ]]; then
    verdict="$(read_verdict "$log_path")"
  else
    verdict=$'-\t-\t-\t-'
    log_path="(no log written)"
  fi
  row="$(printf '%s\t%s\t%s\t%s' "$i" "$rc" "$verdict" "$log_path")"
  rows+=("$row")
  printf '%s\n' "$row" >> "$batch_file"

  if [[ $interrupted -eq 1 ]]; then
    echo "run_batch: interrupted — stopping after trial $i"
    break
  fi
  if [[ $rc -ne 0 ]]; then
    echo "run_batch: trial $i exited with status $rc"
    read -r -p "continue with the remaining trials? [y/N] " ans </dev/tty || ans=n
    [[ "$ans" =~ ^[Yy] ]] || break
  fi
  if [[ $i -lt $trials ]]; then
    echo
    echo "${bold}Arms are parked and torque is off.${reset}"
    echo "Reset the scene for trial $((i+1))/$trials, then stand clear."
    echo "${dim}(Enter starts the next trial — the arms power on and ramp to the start pose immediately; q quits)${reset}"
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
