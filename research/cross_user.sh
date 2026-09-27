#!/bin/sh
# Cross-user validation on public Claude Code sessions (trace-commons/agent-traces, CC-BY-4.0).
# Everything happens in an isolated work dir; your own ~/.claude/agent-eta is never touched.
#   sh cross_user.sh <workdir> [plugin scripts dir]
set -e
W=${1:?workdir}; R=$(cd "$(dirname "$0")" && pwd); P=${2:-$R/../plugins/agent-eta/scripts}
mkdir -p "$W/raw"
curl -s "https://huggingface.co/api/datasets/trace-commons/agent-traces/tree/main/sessions/claude_code?recursive=true" |
  python3 -c "import sys,json; [print(e['path']) for e in json.load(sys.stdin) if e['path'].endswith('.jsonl')]" |
  while read p; do [ -f "$W/raw/$(basename "$p")" ] || curl -sL "https://huggingface.co/datasets/trace-commons/agent-traces/resolve/main/$p" -o "$W/raw/$(basename "$p")"; done
# one "user" per contributor, told apart by the project path of each session
python3 "$R/split_contributors.py" "$W/raw" "$W/users"
for u in "$W"/users/*; do
  AGENT_ETA_HOME="$u/home" python3 "$P/eta.py" backfill --dir "$u/projects" >/dev/null
  AGENT_ETA_HOME="$u/home" python3 "$P/eta.py" tune >/dev/null
  echo "== $(basename "$u")"; AGENT_ETA_HOME="$u/home" python3 "$P/eta.py" eval 2>/dev/null | sed -n '4,7p' || true
done
echo "leave windows through time (with the ledger feedback loop):"
PLUGIN="$P" python3 "$R/timetravel3.py" "$W"/users/*/home
