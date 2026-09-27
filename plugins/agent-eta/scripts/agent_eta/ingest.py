"""Hook event -> database. Pure bookkeeping: no model calls, never prints anything.

Run lifecycle
  UserPromptSubmit            -> new run (any still-open run of the session is closed as interrupted,
                                 because Stop does not fire when you press Esc)
  `!` shell command           -> no hook fires; Claude's first event answering it (new prompt_id) opens a
                                 run of its own, back-dated to when the command finished
  Pre/PostToolUse(+Failure)   -> tool_calls rows, counters, a trajectory snapshot per finished step
  PermissionRequest,
  AskUserQuestion/ExitPlanMode,
  Elicitation                 -> attention events; the wait is excluded from agent-active time
  Stop                        -> run finished (unless it is only pausing for background agents)
  StopFailure / SessionEnd    -> run closed, censored
"""
import json
import os

from . import config
from . import features as F

LIVE = {"enabled": True}  # backfill flips this so env vars of the importing process are ignored

REOPEN_WINDOW_S = 600      # a Stop-hook continuation with the same prompt_id re-opens the run
# Only the main agent working again is a continuation. Claude Code's own post-turn helpers (e.g. the
# "while you were away" summary) fire subagent/task events carrying the finished turn's prompt_id;
# re-opening on those left a finished run "running" (and issuing leave windows) until the session ended.
REOPEN_EVENTS = ("PreToolUse", "PostToolUse", "PostToolUseFailure", "PermissionRequest")
BASH_TURN_EVENTS = REOPEN_EVENTS + ("Stop",)  # main-thread events that can be the first of a `!` answer
TRANSCRIPT_TAIL_BYTES = 1 << 20
BACKGROUND_STALE_S = 1800  # a run "waiting on background agents" with no events for this long is closed
MAX_DENSE_SNAPSHOTS = 150


# ---------------------------------------------------------------- helpers

def get_run(conn, run_id):
    return conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()


def open_runs(conn, session_id):
    return conn.execute(
        "SELECT * FROM runs WHERE session_id=? AND ended_at IS NULL ORDER BY id", (session_id,)).fetchall()


def active_at(run, t):
    """Agent-active seconds of `run` at wall time t: elapsed minus time spent waiting on the human."""
    a = (t - run["started_at"]) - (run["human_wait_s"] or 0.0)
    if run["attn_since"] is not None and t > run["attn_since"]:
        a -= t - run["attn_since"]
    return max(0.0, a)


def _session(conn, sid):
    return conn.execute("SELECT * FROM sessions WHERE session_id=?", (sid,)).fetchone()


def upsert_session(conn, ev, now, **fields):
    sid = ev["session_id"]
    cwd = ev.get("cwd")
    if _session(conn, sid) is None:
        conn.execute(
            "INSERT INTO sessions(session_id, source, cwd, repo, transcript_path, started_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (sid, "live" if LIVE["enabled"] else "backfill", cwd, F.repo_root(cwd),
             ev.get("transcript_path"), now, now))
    else:
        conn.execute(
            "UPDATE sessions SET updated_at=?, cwd=COALESCE(?, cwd), transcript_path=COALESCE(?, transcript_path)"
            " WHERE session_id=?", (now, cwd, ev.get("transcript_path"), sid))
    for key, value in fields.items():
        if value is not None:
            conn.execute("UPDATE sessions SET %s=? WHERE session_id=?" % key, (value, sid))


def snapshot(conn, run, t):
    n = run["n_snaps"] or 0
    conn.execute("UPDATE runs SET n_snaps=n_snaps+1 WHERE id=?", (run["id"],))
    if n >= MAX_DENSE_SNAPSHOTS and n % 5:
        return  # thin out very long runs
    conn.execute(
        "INSERT INTO snapshots(run_id, t, active_s, n_tools, n_edit, n_verify, n_fail, n_verify_fail,"
        " phase, plan_total, plan_done, agents_running) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run["id"], t, active_at(run, t), run["n_tools"], run["n_edit"], run["n_verify"], run["n_fail"],
         run["n_verify_fail"], F.phase_of(run), run["plan_total"], run["plan_done"], run["agents_running"]))


NEW_STEP_GAP_S = 1.0  # tool calls of one parallel batch start within milliseconds of each other


def begin_attention(conn, run, kind, detail, t, agent_id=None, tool_use_id=None):
    """Open one attention request. Several can be open at once; the waiting episode runs from the first
    opening to the last resolution, so overlapping waits are not counted twice."""
    conn.execute(
        "INSERT INTO attention(run_id, session_id, agent_id, kind, detail, at, active_s, tool_use_id)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (run["id"], run["session_id"], agent_id, kind, detail, t, active_at(run, t), tool_use_id))
    conn.execute("UPDATE runs SET n_attn=n_attn+1, n_perm=n_perm+? WHERE id=?",
                 (1 if kind == "permission" else 0, run["id"]))
    if run["attn_since"] is None:
        conn.execute(
            "UPDATE runs SET attn_kind=?, attn_since=?, attn_detail=?, attn_agent=?, attn_tool=?,"
            " status='attention' WHERE id=?", (kind, t, detail, agent_id, tool_use_id, run["id"]))
    return get_run(conn, run["id"])


def resolve_attention(conn, run, t, tool_use_id=None, agent_id=None, kind=None, everything=False):
    """Close the matching open requests; end the waiting episode once none remain.

    An unrelated tool finishing never resolves a request (a read can finish while a Bash call still
    waits for approval). Matching is by tool_use_id, by kind, or - for a new model step - by agent.
    """
    if run["attn_since"] is None:
        return run
    sql, args = "UPDATE attention SET resolved_at=? WHERE run_id=? AND resolved_at IS NULL", [t, run["id"]]
    if not everything:
        if tool_use_id is not None:
            sql += " AND tool_use_id=?"
            args.append(tool_use_id)
        elif kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        else:
            sql += " AND COALESCE(agent_id, '') = COALESCE(?, '')"
            args.append(agent_id)
    conn.execute(sql, args)
    left = conn.execute("SELECT * FROM attention WHERE run_id=? AND resolved_at IS NULL ORDER BY at LIMIT 1",
                        (run["id"],)).fetchone()
    if left is None:
        conn.execute(
            "UPDATE runs SET human_wait_s=human_wait_s+?, attn_kind=NULL, attn_since=NULL, attn_detail=NULL,"
            " attn_agent=NULL, attn_tool=NULL, status=CASE WHEN status='attention' THEN 'running' ELSE status END"
            " WHERE id=?", (max(0.0, t - run["attn_since"]), run["id"]))
    else:
        conn.execute("UPDATE runs SET attn_kind=?, attn_detail=?, attn_agent=?, attn_tool=? WHERE id=?",
                     (left["kind"], left["detail"], left["agent_id"], left["tool_use_id"], run["id"]))
    return get_run(conn, run["id"])


def end_attention(conn, run, t):
    return resolve_attention(conn, run, t, everything=True)


def touch(conn, run, ev, t, new_step=False):
    """Record activity. A *new* tool call by an agent means everything it waited on was answered:
    the model can only start its next step after every result of the previous batch came back."""
    if (new_step and run["attn_since"] is not None and t > run["attn_since"] + NEW_STEP_GAP_S):
        run = resolve_attention(conn, run, t, agent_id=ev.get("agent_id"))
    if new_step:
        # For the same reason every earlier tool call of that agent is over, even one whose PostToolUse
        # never came (a denied prompt): it must not stay "running". Its duration stays unknown, and a
        # PostToolUse that is merely late still fills it in.
        conn.execute(
            "UPDATE tool_calls SET ended_at=? WHERE run_id=? AND ended_at IS NULL AND started_at < ?"
            " AND COALESCE(agent_id, '') = COALESCE(?, '')", (t, run["id"], t - NEW_STEP_GAP_S, ev.get("agent_id")))
    conn.execute(
        "UPDATE runs SET last_event_at=MAX(COALESCE(last_event_at, 0), ?),"
        " status=CASE WHEN status='background' THEN 'running' ELSE status END WHERE id=?", (t, run["id"]))
    return get_run(conn, run["id"])


def start_run(conn, ev, now, prompt, category=None):
    sid = ev["session_id"]
    sess = _session(conn, sid)
    cwd = ev.get("cwd") or (sess["cwd"] if sess else None)
    pf = F.prompt_features(prompt)
    model = ev.get("_model") or (sess["model"] if sess else None)
    effort = F.effort_level(ev, allow_env=LIVE["enabled"]) or (sess["effort"] if sess else None)
    mode = ev.get("permission_mode") or (sess["permission_mode"] if sess else None)
    cur = conn.execute(
        "INSERT INTO runs(session_id, prompt_id, source, started_at, last_event_at, status, repo, cwd, model,"
        " effort, permission_mode, prompt_len, prompt_lines, prompt_code, category, prompt_preview)"
        " VALUES (?,?,?,?,?,'running',?,?,?,?,?,?,?,?,?,?)",
        (sid, ev.get("prompt_id"), "live" if LIVE["enabled"] else "backfill", now, now,
         (sess["repo"] if sess and sess["cwd"] == cwd else F.repo_root(cwd)), cwd, model, effort, mode,
         pf["len"], pf["lines"], pf["code"], category or pf["category"],
         pf["preview"] if config.load().get("store_prompt_preview") else None))
    run = get_run(conn, cur.lastrowid)
    snapshot(conn, run, now)
    return get_run(conn, run["id"])


def finalize(conn, run, reason, t_end):
    """Close a run and write the supervised labels onto every snapshot of it."""
    t_end = max(t_end, run["started_at"])
    run = end_attention(conn, run, t_end)
    active_total = active_at(run, t_end)
    attn = conn.execute(
        "SELECT at, active_s, kind FROM attention WHERE run_id=? ORDER BY at", (run["id"],)).fetchall()
    first_attn = attn[0]["at"] if attn else (t_end if reason == "stop" else None)
    conn.execute(
        "UPDATE runs SET ended_at=?, end_reason=?, status='done', active_s=?, first_attn_at=?, bg_count=0,"
        " agents_running=0 WHERE id=?", (t_end, reason, active_total, first_attn, run["id"]))
    conn.execute(
        "UPDATE tool_calls SET ended_at=? WHERE run_id=? AND ended_at IS NULL", (t_end, run["id"]))
    censored = 0 if reason == "stop" else 1
    # Permission prompts depend on your allow-rules, not on the task; they are modelled separately
    # (see estimator.permission_hazard), so the task-attention label ignores them.
    task_attn = [(a["at"], a["active_s"]) for a in attn if a["kind"] != "permission"]
    labels = []
    for s in conn.execute("SELECT id, t, active_s FROM snapshots WHERE run_id=?", (run["id"],)).fetchall():
        rem_done = max(0.0, active_total - s["active_s"])
        nxt = next((a for at, a in task_attn if at >= s["t"] - 1e-6), None)
        if nxt is not None:
            rem_attn, attn_event = max(0.0, nxt - s["active_s"]), 1
        else:
            rem_attn, attn_event = rem_done, 1 - censored
        labels.append((rem_done, censored, rem_attn, attn_event, s["id"]))
    conn.executemany(
        "UPDATE snapshots SET rem_done_s=?, censored=?, rem_attn_s=?, attn_event=? WHERE id=?", labels)
    return get_run(conn, run["id"])


def close_open_runs(conn, session_id, reason, keep_id=None):
    for r in open_runs(conn, session_id):
        if r["id"] != keep_id:
            why = "superseded" if (reason == "interrupted" and r["status"] == "background") else reason
            finalize(conn, r, why, r["last_event_at"] or r["started_at"])


def ensure_run(conn, ev, now, create=True):
    """The run an event belongs to. Handles async hooks that arrive after Stop, and Stop-hook continuations."""
    sid = ev["session_id"]
    runs = open_runs(conn, sid)
    if runs:
        return runs[-1]
    last = conn.execute(
        "SELECT * FROM runs WHERE session_id=? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
    if last is not None and last["ended_at"] is not None:
        if now <= last["ended_at"] + 0.5:
            return last  # a late async event from before the Stop: attach, don't reopen
        pid = ev.get("prompt_id")
        if (pid and pid == last["prompt_id"] and last["end_reason"] == "stop"
                and now - last["ended_at"] < REOPEN_WINDOW_S
                and ev.get("hook_event_name") in REOPEN_EVENTS and not ev.get("agent_id")):
            return _reopen(conn, last)
    if not create:
        return None
    upsert_session(conn, ev, now)
    return start_run(conn, ev, now, "", category="wake")


def _start_bash_run(conn, ev, t):
    close_open_runs(conn, ev["session_id"], "interrupted")
    return start_run(conn, ev, t, "", category="bash")


def _bash_turn(conn, ev, now):
    """Open the run of Claude answering a `!` shell command (see F.BASH_INPUT_PREFIX): no hook marks its
    start, so the first main-thread event carrying a prompt_id no run has yet is checked against the
    transcript. Checked once per prompt_id: afterwards a run carries it either way."""
    pid = ev.get("prompt_id")
    if not (LIVE["enabled"] and pid and not ev.get("agent_id")
            and ev.get("hook_event_name") in BASH_TURN_EVENTS):
        return
    last = conn.execute("SELECT prompt_id FROM runs WHERE session_id=? ORDER BY id DESC LIMIT 1",
                        (ev["session_id"],)).fetchone()
    if last is not None and last["prompt_id"] == pid:
        return
    found, done_at = _bash_command(ev.get("transcript_path"), pid)
    if found:
        _start_bash_run(conn, ev, min(now, done_at or now))
        return
    runs = open_runs(conn, ev["session_id"])
    if runs:
        # Same work under a new prompt_id (e.g. after a wake-up): follow it, so this is not re-checked.
        conn.execute("UPDATE runs SET prompt_id=? WHERE id=?", (pid, runs[-1]["id"]))


def _reopen(conn, run):
    conn.execute("UPDATE runs SET ended_at=NULL, end_reason=NULL, status='running', active_s=NULL,"
                 " first_attn_at=NULL WHERE id=?", (run["id"],))
    conn.execute("UPDATE snapshots SET rem_done_s=NULL, censored=NULL, rem_attn_s=NULL, attn_event=NULL"
                 " WHERE run_id=?", (run["id"],))
    return get_run(conn, run["id"])


def _is_late(run):
    return run["ended_at"] is not None


# ---------------------------------------------------------------- handlers

def on_session_start(conn, ev, now):
    upsert_session(conn, ev, now, model=ev.get("model"), permission_mode=ev.get("permission_mode"))
    if ev.get("source") in ("startup", "resume", "clear"):
        close_open_runs(conn, ev["session_id"], "abandoned")
    root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if LIVE["enabled"] and root:
        _write_plugin_root(root)


def _write_plugin_root(root):
    target = config.path("plugin_root")
    try:
        with open(target, encoding="utf-8") as f:
            if f.read().strip() == root:
                return
    except OSError:
        pass
    os.makedirs(config.home(), exist_ok=True)
    config.write_atomic(target, root + "\n")


def on_prompt(conn, ev, now):
    upsert_session(conn, ev, now, permission_mode=ev.get("permission_mode"),
                   effort=F.effort_level(ev, allow_env=LIVE["enabled"]))
    if ev.get("_bash"):  # imported history: a `!` command's answer, already back-dated
        _start_bash_run(conn, ev, now)
        return
    if F.is_system_prompt(ev.get("prompt")):
        # A background task finished and Claude was woken up: same piece of work, not a new request.
        runs = open_runs(conn, ev["session_id"])
        if runs:
            _follow_prompt(conn, touch(conn, runs[-1], ev, now), ev)
            return
        last = conn.execute("SELECT * FROM runs WHERE session_id=? ORDER BY id DESC LIMIT 1",
                            (ev["session_id"],)).fetchone()
        if last is not None and last["end_reason"] == "stop" and now - last["ended_at"] < BACKGROUND_STALE_S:
            _follow_prompt(conn, touch(conn, _reopen(conn, last), ev, now), ev)
            return
        start_run(conn, ev, now, "", category="wake")
        return
    close_open_runs(conn, ev["session_id"], "interrupted")
    start_run(conn, ev, now, ev.get("prompt") or "")


def _follow_prompt(conn, run, ev):
    """The run continues under the wake-up's prompt_id: that is what Claude's next events carry."""
    if ev.get("prompt_id"):
        conn.execute("UPDATE runs SET prompt_id=? WHERE id=?", (ev["prompt_id"], run["id"]))


def on_pre_tool(conn, ev, now):
    run = ensure_run(conn, ev, now, create=not ev.get("agent_id"))
    if run is None:
        return  # a background subagent still working after its run was closed
    name = ev.get("tool_name") or "?"
    inp = ev.get("tool_input") if isinstance(ev.get("tool_input"), dict) else {}
    cmd_key, cmd_kind = F.tool_key(name, inp)
    summary = F.safe_summary(name, inp)
    tuid = ev.get("tool_use_id")
    if not _is_late(run):
        run = touch(conn, run, ev, now, new_step=True)
    exists = tuid and conn.execute("SELECT 1 FROM tool_calls WHERE tool_use_id=?", (tuid,)).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO tool_calls(run_id, session_id, tool_use_id, agent_id, tool, kind, cmd_key, cmd_kind,"
            " repo, detail, started_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run["id"], ev["session_id"], tuid, ev.get("agent_id"), name, F.tool_kind(name), cmd_key, cmd_kind,
             run["repo"], summary, now))
    if _is_late(run):
        return
    if tuid:
        # async hooks may record the PermissionRequest before this PreToolUse: link it now
        conn.execute(
            "UPDATE attention SET tool_use_id=? WHERE id=(SELECT id FROM attention WHERE run_id=?"
            " AND resolved_at IS NULL AND tool_use_id IS NULL AND kind='permission' AND detail=?"
            " AND COALESCE(agent_id, '') = COALESCE(?, '') AND at >= ? ORDER BY at LIMIT 1)",
            (tuid, run["id"], summary, ev.get("agent_id"), now - 0.5))
        if run["attn_since"] is not None and run["attn_tool"] is None:
            conn.execute("UPDATE runs SET attn_tool=(SELECT tool_use_id FROM attention WHERE run_id=?"
                         " AND resolved_at IS NULL ORDER BY at LIMIT 1) WHERE id=?", (run["id"], run["id"]))
    if name in F.ATTENTION_TOOLS:
        begin_attention(conn, run, F.ATTENTION_TOOLS[name], summary, now, ev.get("agent_id"), tuid)


def on_post_tool(conn, ev, now, ok):
    run = ensure_run(conn, ev, now, create=not ev.get("agent_id"))
    if run is None:
        return
    name = ev.get("tool_name") or "?"
    inp = ev.get("tool_input") if isinstance(ev.get("tool_input"), dict) else {}
    cmd_key, cmd_kind = F.tool_key(name, inp)
    kind = F.tool_kind(name)
    dur = ev.get("duration_ms")
    dur = float(dur) if isinstance(dur, (int, float)) else None
    tuid = ev.get("tool_use_id")
    row = tuid and conn.execute(
        "SELECT id, started_at, duration_ms FROM tool_calls WHERE tool_use_id=?", (tuid,)).fetchone()
    if row and row["duration_ms"] is not None:
        return  # the same tool call reported twice: count it once
    if not _is_late(run) and run["attn_since"] is not None:
        summary = F.safe_summary(name, inp)
        mine = conn.execute(
            "SELECT id, kind, tool_use_id FROM attention WHERE run_id=? AND resolved_at IS NULL AND"
            " (tool_use_id=? OR (tool_use_id IS NULL AND kind='permission' AND detail=?"
            " AND COALESCE(agent_id, '') = COALESCE(?, '')))",
            (run["id"], tuid, summary, ev.get("agent_id"))).fetchall()
        if mine:
            kinds = {m["kind"] for m in mine}
            t_res = now
            if kinds == {"permission"} and dur is not None:
                # duration_ms excludes the permission prompt, so approval happened when execution began
                t_res = max(run["attn_since"], now - dur / 1000.0)
            for m in mine:
                conn.execute("UPDATE attention SET tool_use_id=? WHERE id=? AND tool_use_id IS NULL", (tuid, m["id"]))
            run = resolve_attention(conn, run, t_res, tool_use_id=tuid)
    if row:
        if dur is None:
            dur = max(0.0, (now - row["started_at"]) * 1000.0)
        conn.execute("UPDATE tool_calls SET ended_at=?, duration_ms=?, ok=? WHERE id=?",
                     (now, dur, 1 if ok else 0, row["id"]))
    else:
        conn.execute(
            "INSERT INTO tool_calls(run_id, session_id, tool_use_id, agent_id, tool, kind, cmd_key, cmd_kind,"
            " repo, detail, started_at, ended_at, duration_ms, ok) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run["id"], ev["session_id"], tuid, ev.get("agent_id"), name, kind, cmd_key, cmd_kind, run["repo"],
             F.safe_summary(name, inp), now - (dur or 0) / 1000.0, now, dur, 1 if ok else 0))

    if ev.get("agent_id"):
        # Inside a subagent: keep the duration sample, but count only main-thread steps so live data
        # matches imported transcripts (whose subagent calls live in separate files).
        if not _is_late(run):
            touch(conn, run, ev, now)
        return
    sets, args = ["n_tools=n_tools+1"], []
    if kind == "explore":
        sets.append("n_explore=n_explore+1")
    elif kind == "edit" and ok:
        sets.append("n_edit=n_edit+1")
        sets.append("last_edit_at=?")
        args.append(now)
    elif kind == "shell":
        sets.append("n_shell=n_shell+1")
        if cmd_kind in F.VERIFY_KINDS:
            sets += ["n_verify=n_verify+1", "last_verify_at=?", "last_verify_ok=?"]
            args += [now, 1 if ok else 0]
            if not ok:
                sets.append("n_verify_fail=n_verify_fail+1")
    if not ok and not ev.get("is_interrupt"):
        sets.append("n_fail=n_fail+1")
    if name == "TodoWrite" and ok and isinstance(inp.get("todos"), list):
        todos = [t for t in inp["todos"] if isinstance(t, dict)]
        done = sum(1 for t in todos if t.get("status") == "completed")
        sets += ["plan_total=?", "plan_first_total=COALESCE(plan_first_total, ?)"]
        args += [len(todos), len(todos)]
        if done > (run["plan_done"] or 0):
            sets += ["plan_done=?"]
            args += [done]
            _plan_steps_done(conn, run, now, done - (run["plan_done"] or 0))
    conn.execute("UPDATE runs SET %s WHERE id=?" % ", ".join(sets), args + [run["id"]])
    if _is_late(run):
        return
    run = touch(conn, get_run(conn, run["id"]), ev, now)
    snapshot(conn, run, now)


def on_permission(conn, ev, now):
    run = ensure_run(conn, ev, now)
    if _is_late(run):
        return
    name = ev.get("tool_name") or "?"
    summary = F.safe_summary(name, ev.get("tool_input"))
    pending = conn.execute(
        "SELECT tool_use_id FROM tool_calls WHERE run_id=? AND ended_at IS NULL AND tool=? AND detail=?"
        " AND COALESCE(agent_id, '') = COALESCE(?, '') ORDER BY started_at DESC LIMIT 1",
        (run["id"], name, summary, ev.get("agent_id"))).fetchone()
    begin_attention(conn, run, "permission", summary, now, ev.get("agent_id"),
                    pending["tool_use_id"] if pending else None)


def on_permission_denied(conn, ev, now):
    run = ensure_run(conn, ev, now, create=False)
    if run is None or _is_late(run):
        return
    tuid = ev.get("tool_use_id")
    if tuid:
        conn.execute("UPDATE tool_calls SET ended_at=?, ok=0 WHERE tool_use_id=? AND ended_at IS NULL", (now, tuid))
    conn.execute("UPDATE runs SET n_fail=n_fail+1 WHERE id=?", (run["id"],))
    touch(conn, run, ev, now)


def on_notification(conn, ev, now):
    kind = ev.get("notification_type")
    if kind not in ("permission_prompt", "elicitation_dialog", "elicitation_url_dialog", "agent_needs_input"):
        return
    run = ensure_run(conn, ev, now, create=False)
    if run is None or _is_late(run) or run["attn_since"] is not None:
        return
    # These fire ~6 s after the dialog appeared (and only if you were idle): back-date the start.
    begin_attention(conn, run, "permission" if kind == "permission_prompt" else "elicitation",
                    kind, max(run["last_event_at"] or now, now - 6.0))


def on_elicitation(conn, ev, now):
    run = ensure_run(conn, ev, now, create=False)
    if run is None or _is_late(run):
        return
    begin_attention(conn, run, "elicitation", ev.get("server_name") or "mcp", now, ev.get("agent_id"))


def on_elicitation_result(conn, ev, now):
    run = ensure_run(conn, ev, now, create=False)
    if run is not None and not _is_late(run):
        resolve_attention(conn, run, now, kind="elicitation")


def on_subagent(conn, ev, now, start):
    run = ensure_run(conn, ev, now, create=False)
    if run is None or _is_late(run):
        return
    if start:
        conn.execute("UPDATE runs SET n_agents=n_agents+1, agents_running=agents_running+1 WHERE id=?", (run["id"],))
    else:
        conn.execute("UPDATE runs SET agents_running=MAX(0, agents_running-1) WHERE id=?", (run["id"],))
    run = get_run(conn, run["id"])
    conn.execute("UPDATE runs SET last_event_at=MAX(COALESCE(last_event_at,0), ?) WHERE id=?", (now, run["id"]))
    if not start:
        snapshot(conn, run, now)


def on_task(conn, ev, now, created):
    run = ensure_run(conn, ev, now, create=False)
    if run is None or _is_late(run):
        return
    if created:
        conn.execute(
            "UPDATE runs SET plan_total=plan_total+1 WHERE id=?", (run["id"],))
        if (run["plan_done"] or 0) == 0:
            conn.execute("UPDATE runs SET plan_first_total=plan_total WHERE id=?", (run["id"],))
    else:
        conn.execute("UPDATE runs SET plan_done=MIN(plan_total, plan_done+1) WHERE id=?", (run["id"],))
        _plan_steps_done(conn, run, now, 1)
        snapshot(conn, get_run(conn, run["id"]), now)


def _plan_steps_done(conn, run, now, steps):
    """EMA of step-completion intervals (Codex design B1). Several steps ticked off at once carry no
    usable per-step interval, so a batched update moves the anchor but leaves the EMA alone."""
    a = active_at(run, now)
    if steps == 1:
        interval = a - (run["active_at_last_done"] or 0.0)
        e = interval if run["plan_ema"] is None else 0.3 * interval + 0.7 * run["plan_ema"]
        conn.execute("UPDATE runs SET plan_ema=?, active_at_last_done=? WHERE id=?", (e, a, run["id"]))
    else:
        conn.execute("UPDATE runs SET active_at_last_done=? WHERE id=?", (a, run["id"]))


def on_model_switch(conn, ev, now):
    upsert_session(conn, ev, now, model=ev.get("to_model"))


def _tail_lines(path, max_bytes):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except (OSError, TypeError):
        return []


def _bash_command(path, pid):
    """(is prompt `pid` a `!` shell command, when its output was recorded) from the transcript tail."""
    found, done_at = False, None
    for line in _tail_lines(path, TRANSCRIPT_TAIL_BYTES):
        if pid not in line or "<bash-" not in line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict) or rec.get("promptId") != pid or rec.get("type") != "user":
            continue
        part = F.bash_part(F.content_text((rec.get("message") or {}).get("content")))
        if part is None:
            continue
        found = True
        t = F.iso_ts(rec.get("timestamp"))
        if part == "output" and t is not None:
            done_at = max(done_at or t, t)
    return found, done_at


def _model_from_transcript(path, max_bytes=262144):
    """Hooks never receive the model reliably (SessionStart may omit it); the transcript has it."""
    lines = _tail_lines(path, max_bytes)
    for line in reversed(lines):
        if '"assistant"' not in line or '"model"' not in line:
            continue
        try:
            model = (json.loads(line).get("message") or {}).get("model")
        except ValueError:
            continue
        if model and model != "<synthetic>":
            return model
    return None


def on_stop(conn, ev, now):
    runs = open_runs(conn, ev["session_id"])
    if not runs:
        return
    run = touch(conn, runs[-1], ev, now)
    if LIVE["enabled"] and not run["model"] and ev.get("transcript_path"):
        model = _model_from_transcript(ev["transcript_path"])
        if model:
            conn.execute("UPDATE runs SET model=? WHERE id=?", (model, run["id"]))
            conn.execute("UPDATE sessions SET model=COALESCE(model, ?) WHERE session_id=?", (model, ev["session_id"]))
            run = get_run(conn, run["id"])
    waking = [t for t in (ev.get("background_tasks") or [])
              if isinstance(t, dict) and t.get("type") in F.WAKING_BG_TYPES
              and str(t.get("status") or "running").lower() not in F.BG_DONE_STATUSES]
    if waking:
        conn.execute("UPDATE runs SET status='background', bg_count=? WHERE id=?", (len(waking), run["id"]))
        return
    finalize(conn, run, "stop", now)


def on_stop_failure(conn, ev, now):
    for run in open_runs(conn, ev["session_id"]):
        finalize(conn, run, "stop_failure", now)


def on_session_end(conn, ev, now):
    close_open_runs(conn, ev["session_id"], "session_end")
    conn.execute("UPDATE sessions SET ended_at=? WHERE session_id=?", (now, ev["session_id"]))


HANDLERS = {
    "SessionStart": on_session_start,
    "UserPromptSubmit": on_prompt,
    "PreToolUse": on_pre_tool,
    "PostToolUse": lambda c, e, t: on_post_tool(c, e, t, True),
    "PostToolUseFailure": lambda c, e, t: on_post_tool(c, e, t, False),
    "PermissionRequest": on_permission,
    "PermissionDenied": on_permission_denied,
    "Notification": on_notification,
    "Elicitation": on_elicitation,
    "ElicitationResult": on_elicitation_result,
    "SubagentStart": lambda c, e, t: on_subagent(c, e, t, True),
    "SubagentStop": lambda c, e, t: on_subagent(c, e, t, False),
    "TaskCreated": lambda c, e, t: on_task(c, e, t, True),
    "TaskCompleted": lambda c, e, t: on_task(c, e, t, False),
    "PostModelSwitch": on_model_switch,
    "Stop": on_stop,
    "StopFailure": on_stop_failure,
    "SessionEnd": on_session_end,
}


def handle(conn, ev, now):
    if not isinstance(ev, dict) or not ev.get("session_id"):
        return
    fn = HANDLERS.get(ev.get("hook_event_name"))
    if fn is None:
        return
    from .store import tx
    with tx(conn):
        _bash_turn(conn, ev, now)
        fn(conn, ev, now)


def expire_stale(conn, now):
    """Close runs that will never see another event (crashed session, forgotten background task)."""
    stale = conn.execute(
        "SELECT * FROM runs WHERE ended_at IS NULL AND ("
        " (status='background' AND COALESCE(last_event_at, started_at) < ?) OR"
        " COALESCE(last_event_at, started_at) < ?)",
        (now - BACKGROUND_STALE_S, now - 6 * 3600)).fetchall()
    for r in stale:
        finalize(conn, r, "abandoned", r["last_event_at"] or r["started_at"])
    return len(stale)
