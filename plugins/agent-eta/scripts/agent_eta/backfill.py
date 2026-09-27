"""Import past Claude Code transcripts (~/.claude/projects/*/*.jsonl) as history.

The transcript already holds everything the estimator learns from: prompt timestamps, every
tool_use / tool_result with timestamps, the model and effort, and (on recent versions) a
`turn_duration` system record marking exactly when each turn ended. Replaying it through the same
hook handlers gives useful estimates on day one instead of after a month.

Transcript format is internal to Claude Code, so parsing is defensive: unknown records are skipped.
Not recoverable from transcripts: permission prompts (they are learned from live use only).
"""
import glob
import json
import os

from . import features as F
from . import ingest, store

_NOT_PROMPT_PREFIXES = ("<local-command", "<command-message>", "<bash-input>", "<bash-stdout>",
                        "<bash-stderr>", "<system-reminder>", "Caveat:", "[Request interrupted")


def _prompt_text(rec):
    """The user's prompt if this record is a human turn start, else None."""
    if rec.get("isMeta") or rec.get("isSidechain"):
        return None
    origin = rec.get("origin")
    if isinstance(origin, dict) and origin.get("kind") not in (None, "human", "task-notification"):
        return None
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return None
    text = F.content_text(content).strip()
    if F.is_system_prompt(text) or (isinstance(origin, dict) and origin.get("kind") == "task-notification"):
        return text  # background task finished: replayed so ingest continues the waiting run
    if not text or text.startswith(_NOT_PROMPT_PREFIXES):
        return None
    if text.startswith("<command-name>"):
        # A slash command / skill invocation: keep just the command name as the prompt.
        name = text[len("<command-name>"):].split("<", 1)[0].strip()
        return name or "/command"
    return text


def _bash_part(rec):
    if rec.get("isMeta") or rec.get("isSidechain"):
        return None
    return F.bash_part(F.content_text((rec.get("message") or {}).get("content")))


def _is_bash_input(rec):
    return rec.get("type") == "user" and _bash_part(rec) == "input"


def events_from_records(records):
    """Turn transcript records into (timestamp, hook-event) pairs, in order."""
    recs = [r for r in records if isinstance(r, dict) and not r.get("isSidechain")]
    sid = next((r.get("sessionId") for r in recs if r.get("sessionId")), None)
    if not sid:
        return None, []
    # Segment into turns: each human prompt opens one; keep only turns Claude actually answered.
    segments, cur = [], None
    for r in recs:
        prompt = _prompt_text(r) if r.get("type") == "user" else None
        bash = prompt is None and _is_bash_input(r)
        if bash:
            prompt = ""
        if prompt is not None and cur is not None and r.get("promptId") and \
                r.get("promptId") == cur["rec"].get("promptId"):
            prompt = None  # a nudge Claude Code injected into the same turn, not a new prompt
        if prompt is not None and F.iso_ts(r.get("timestamp")) is not None:
            cur = {"prompt": prompt, "rec": r, "body": [], "bash": bash}
            segments.append(cur)
        elif cur is not None:
            cur["body"].append(r)
    events = []
    first = next((r for r in recs if F.iso_ts(r.get("timestamp"))), None)
    if first is None:
        return sid, []
    cwd = next((r.get("cwd") for r in recs if r.get("cwd")), None)
    base = {"session_id": sid, "cwd": cwd, "transcript_path": None}
    events.append((F.iso_ts(first["timestamp"]), dict(base, hook_event_name="SessionStart", source="startup")))
    for seg in segments:
        assistants = [b for b in seg["body"] if b.get("type") == "assistant"]
        if not assistants:
            continue
        r = seg["rec"]
        t0 = F.iso_ts(r["timestamp"])
        if seg["bash"]:
            # Timed from when your command finished: its runtime is yours, not Claude's work.
            before = seg["body"][:seg["body"].index(assistants[0])]
            done = [F.iso_ts(b.get("timestamp")) for b in before if b.get("type") == "user" and _bash_part(b) == "output"]
            t0 = max([t0] + [t for t in done if t is not None])
        model = next(((a.get("message") or {}).get("model") for a in assistants
                      if (a.get("message") or {}).get("model") not in (None, "<synthetic>")), None)
        effort = next((a.get("effort") for a in assistants if a.get("effort")), None)
        ev = dict(base, hook_event_name="UserPromptSubmit", prompt=seg["prompt"], prompt_id=r.get("promptId"),
                  permission_mode=r.get("permissionMode"), cwd=r.get("cwd") or base["cwd"], _model=model,
                  _bash=seg["bash"])
        if effort:
            ev["effort"] = {"level": effort}
        events.append((t0, ev))
        pending = {}
        stopped = interrupted = False
        last_assistant = None
        for b in seg["body"]:
            t = F.iso_ts(b.get("timestamp"))
            if t is None:
                continue
            common = dict(base, prompt_id=b.get("promptId") or r.get("promptId"), cwd=b.get("cwd") or base["cwd"])
            if b.get("type") == "assistant":
                last_assistant = b
                for blk in (b.get("message") or {}).get("content") or []:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        pending[blk.get("id")] = (t, blk.get("name"), blk.get("input") or {})
                        events.append((t, dict(common, hook_event_name="PreToolUse", tool_name=blk.get("name"),
                                               tool_input=blk.get("input") or {}, tool_use_id=blk.get("id"))))
            elif b.get("type") == "user":
                content = (b.get("message") or {}).get("content")
                if isinstance(content, list):
                    for blk in content:
                        if not (isinstance(blk, dict) and blk.get("type") == "tool_result"):
                            continue
                        start = pending.pop(blk.get("tool_use_id"), None)
                        if start is None:
                            continue
                        failed = blk.get("is_error") in (True, "true", "True")
                        events.append((t, dict(
                            common, hook_event_name="PostToolUseFailure" if failed else "PostToolUse",
                            tool_name=start[1], tool_input=start[2], tool_use_id=blk.get("tool_use_id"),
                            duration_ms=max(0.0, (t - start[0]) * 1000.0))))
                if "[Request interrupted" in F.content_text(content)[:60]:
                    interrupted = True
            elif b.get("type") == "system" and b.get("subtype") == "turn_duration" and not stopped:
                events.append((t, dict(common, hook_event_name="Stop", background_tasks=[])))
                stopped = True
        if not stopped and not interrupted and last_assistant is not None:
            if (last_assistant.get("message") or {}).get("stop_reason") == "end_turn":
                events.append((F.iso_ts(last_assistant["timestamp"]) + 0.001,
                               dict(base, hook_event_name="Stop", background_tasks=[])))
    last = next((r for r in reversed(recs) if F.iso_ts(r.get("timestamp"))), None)
    events.append((F.iso_ts(last["timestamp"]) + 0.002, dict(base, hook_event_name="SessionEnd")))
    events.sort(key=lambda e: e[0])  # stable: keeps file order for equal timestamps
    return sid, events


def import_file(conn, path, force=False):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return 0, "missing"
    if not force and conn.execute("SELECT 1 FROM imported WHERE path=?", (path,)).fetchone():
        return 0, "already imported"
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    sid, events = events_from_records(records)
    if not sid:
        return 0, "no session"
    if conn.execute("SELECT source FROM sessions WHERE session_id=?", (sid,)).fetchone():
        return 0, "session already recorded"
    before = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    ingest.LIVE["enabled"] = False
    try:
        with store.tx(conn):
            for t, ev in events:
                ev["transcript_path"] = path
                handler = ingest.HANDLERS.get(ev["hook_event_name"])
                if handler:
                    handler(conn, ev, t)
            n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] - before
            conn.execute("INSERT OR REPLACE INTO imported(path, session_id, mtime, runs, at) VALUES (?,?,?,?,?)",
                         (path, sid, mtime, n, events[-1][0] if events else None))
    finally:
        ingest.LIVE["enabled"] = True
    return n, "ok"


def transcript_files(projects_dir=None, skip_session=None):
    root = projects_dir or os.path.join(os.path.expanduser("~"), ".claude", "projects")
    files = sorted(glob.glob(os.path.join(root, "*", "*.jsonl")), key=os.path.getmtime)
    if skip_session:
        files = [f for f in files if os.path.basename(f) != skip_session + ".jsonl"]
    return files


def run(projects_dir=None, skip_session=None, force=False, out=None):
    conn = store.connect()
    total_runs = total_files = 0
    for path in transcript_files(projects_dir, skip_session):
        n, why = import_file(conn, path, force)
        if n:
            total_files += 1
            total_runs += n
        if out is not None:
            out.append((path, n, why))
    return total_files, total_runs
