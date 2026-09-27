"""Attach each finished run's prompt text (from the transcripts) to its database row, for research only.
Uses the plugin's own transcript segmentation (backfill.events_from_records)."""
import glob
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "agent-eta", "scripts"))
from agent_eta import backfill  # noqa: E402

import json

# Histories to analyse: research/histories.json (or $HISTORIES), see histories.example.json.
_CONFIG = os.environ.get("HISTORIES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "histories.json")
_H = json.load(open(_CONFIG))
HISTORIES = {name: (h["home"], h["transcripts"]) for name, h in _H.items() if h.get("transcripts")}  # name: (home, transcript glob)


def load(name):
    home, pattern = HISTORIES[name]
    starts = {}
    for path in glob.glob(pattern):
        recs = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
        sid, events = backfill.events_from_records(recs)
        for t, ev in events:
            if ev["hook_event_name"] == "UserPromptSubmit" and not ev.get("_bash"):
                starts.setdefault(sid, []).append((t, ev.get("prompt") or ""))
    conn = sqlite3.connect(os.path.join(home, "eta.db"))
    conn.row_factory = sqlite3.Row
    out = []
    for r in conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s > 0 ORDER BY started_at"):
        cands = starts.get(r["session_id"], [])
        best = min(cands, key=lambda c: abs(c[0] - r["started_at"]), default=None)
        if best is None or abs(best[0] - r["started_at"]) > 5.0 or not best[1].strip():
            continue
        first_attn = r["first_attn_at"]
        attn = (first_attn - r["started_at"]) if first_attn is not None else None
        out.append({"id": r["id"], "t": r["started_at"], "end": r["ended_at"], "dur": r["active_s"],
                    "stop": r["end_reason"] == "stop", "attn": attn, "prompt": best[1], "category": r["category"],
                    "effort": r["effort"], "prompt_len": r["prompt_len"] or 0, "model": r["model"]})
    return out


if __name__ == "__main__":
    for name in HISTORIES:
        rows = load(name)
        n_db = sqlite3.connect(os.path.join(HISTORIES[name][0], "eta.db")).execute(
            "SELECT COUNT(*) FROM runs WHERE ended_at IS NOT NULL AND active_s > 0").fetchone()[0]
        print("%-11s %4d/%4d runs matched to a prompt; e.g. %r" % (name, len(rows), n_db, rows[len(rows) // 2]["prompt"][:60] if rows else None))
