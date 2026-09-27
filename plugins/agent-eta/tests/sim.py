"""Synthetic agent trajectories with known structure, for validating the estimator.

Structure the estimator should discover on its own:
  * task type sets the amount of work (question << fix << feature)
  * repo /work/web has a slow test suite (90 s) vs /work/api (25 s), and its fixes take longer
  * a failed test run triggers a fix loop (more edits + another test run)
  * once tests pass, the run ends shortly after (summary)
"""
import random

REPOS = ["/work/api", "/work/web"]
TEST_S = {"/work/api": 25.0, "/work/web": 90.0}
PROMPTS = {
    "question": "why does the cache return stale values here?",
    "fix": "fix the bug where login fails after a password reset, the error is in the auth module",
    "feature": "add a feature: implement CSV export for the reports page with filters, pagination and "
               "a download button; write tests and update the docs accordingly",
}


def _gap(rng, med):
    return rng.lognormvariate(0, 0.5) * med


def gen_run(rng, sid, t, repo, n):
    """Return (events, end_time). Events are (timestamp, hook-event dict)."""
    cat = rng.choices(["question", "fix", "feature"], [0.3, 0.4, 0.3])[0]
    base = {"session_id": sid, "cwd": repo, "permission_mode": "acceptEdits"}
    ev = [(t, dict(base, hook_event_name="UserPromptSubmit", prompt=PROMPTS[cat], prompt_id="p%d" % n,
                   effort={"level": "high"}, _model="claude-opus-5"))]
    tid = [0]
    think = {"question": 6.0, "fix": 10.0, "feature": 14.0}[cat]

    def tool(name, inp, dur, ok=True):
        nonlocal t
        t += _gap(rng, think)
        tid[0] += 1
        u = "toolu_%d_%d" % (n, tid[0])
        ev.append((t, dict(base, hook_event_name="PreToolUse", tool_name=name, tool_input=inp, tool_use_id=u)))
        t += dur
        ev.append((t, dict(base, hook_event_name="PostToolUse" if ok else "PostToolUseFailure", tool_name=name,
                           tool_input=inp, tool_use_id=u, duration_ms=dur * 1000.0)))

    n_explore = {"question": rng.randint(1, 3), "fix": rng.randint(3, 7), "feature": rng.randint(5, 10)}[cat]
    for _ in range(n_explore):
        tool(rng.choice(["Read", "Grep", "Glob"]), {"file_path": "/x"}, rng.uniform(0.2, 2))
    if cat != "question":
        n_edit = {"fix": rng.randint(1, 3), "feature": rng.randint(4, 9)}[cat]
        for _ in range(n_edit):
            tool("Edit", {"file_path": "/x.py"}, rng.uniform(0.2, 1))
        slow_fix = 2.0 if (repo == "/work/web" and cat == "fix") else 1.0
        for attempt in range(4):
            dur = TEST_S[repo] * rng.lognormvariate(0, 0.15)
            ok = attempt == 3 or rng.random() > 0.35
            tool("Bash", {"command": "npm test"}, dur, ok)
            if ok:
                break
            for _ in range(int(rng.randint(1, 3) * slow_fix)):
                tool("Edit", {"file_path": "/x.py"}, rng.uniform(0.2, 1))
    t += _gap(rng, think * 1.5)  # final summary
    ev.append((t, dict(base, hook_event_name="Stop", background_tasks=[])))
    return ev, t


def populate(conn, n_runs=240, seed=7, start=1.7e9):
    from agent_eta import ingest, store
    rng = random.Random(seed)
    t = start
    ingest.LIVE["enabled"] = False
    try:
        with store.tx(conn):
            for i in range(n_runs):
                if i % 6 == 0:
                    sid = "sim-%d" % i
                    repo = rng.choice(REPOS)
                    ingest.HANDLERS["SessionStart"](conn, {"session_id": sid, "cwd": repo,
                                                           "hook_event_name": "SessionStart",
                                                           "source": "startup"}, t)
                events, t = gen_run(rng, sid, t, repo, i)
                for ts, e in events:
                    ingest.HANDLERS[e["hook_event_name"]](conn, e, ts)
                t += rng.uniform(60, 1800)
    finally:
        ingest.LIVE["enabled"] = True
    return t
