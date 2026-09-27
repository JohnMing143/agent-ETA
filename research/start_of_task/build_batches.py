"""Batches of tasks for the LLM judge: each run's request and the tail of the assistant's previous reply."""
import glob
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts as P  # noqa: E402

BATCH = 40
INSTR = """You estimate how long an AI coding agent (Claude Code, working autonomously with tools in the user's repository or server) will take on each request below, counted from when the request is sent until the agent stops and waits for the user. Judge the scope of the work from the request and the agent's previous reply: a direct answer, a yes/no or a tiny change is quick; reading code, editing several files, running builds/tests, deploying or debugging takes longer.
For each item return: id, size (one of XS = under 30 s, S = 30 s-2 min, M = 2-8 min, L = 8-30 min, XL = over 30 min), minutes (your best single estimate, a number), and ask (true if the agent will probably stop to ask the user something before finishing).
Reply with only a JSON array, no prose.

"""


def ts(v):
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def assistant_texts(pattern):
    """session -> sorted [(t, text)] of assistant text blocks."""
    out = {}
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"assistant"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") != "assistant" or r.get("isSidechain"):
                    continue
                c = (r.get("message") or {}).get("content")
                text = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text") \
                    if isinstance(c, list) else ""
                t = ts(r.get("timestamp"))
                if text.strip() and t:
                    out.setdefault(r.get("sessionId"), []).append((t, text.strip()))
    for v in out.values():
        v.sort()
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "batches"), exist_ok=True)
    n_batches = 0
    for name, (home, pattern) in P.HISTORIES.items():
        rows = P.load(name)
        texts = assistant_texts(pattern)
        import sqlite3
        sess = dict(sqlite3.connect(os.path.join(home, "eta.db")).execute("SELECT id, session_id FROM runs").fetchall())
        items = []
        for r in rows:
            prev = [x for x in texts.get(sess.get(r["id"]), []) if x[0] < r["t"] - 0.5]
            items.append({"id": "%s:%d" % (name, r["id"]), "prev": prev[-1][1][-300:] if prev else "",
                          "request": r["prompt"][:600]})
        for i in range(0, len(items), BATCH):
            chunk = items[i:i + BATCH]
            with open(os.path.join(HERE, "batches", "%s_%03d.txt" % (name, i // BATCH)), "w") as f:
                f.write(INSTR + json.dumps(chunk, ensure_ascii=False, indent=0))
            n_batches += 1
        print("%-11s %4d tasks" % (name, len(items)))
    print("%d batches" % n_batches)
