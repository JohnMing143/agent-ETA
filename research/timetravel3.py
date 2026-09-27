"""Faithful replay of the live system through time, including its feedback loop: every promise the
simulation shows is written to the windows ledger, so later re-tunes see real outcomes (and step
SAFE_K up when windows were breached too often). Policies:
  current      no gate (v0.5.0 behaviour)
  after_tune   no promise before the first tune (fewer than 10 finished runs), then as current
  medium       after_tune, and only while the replayed promises are >= 10 with breach rate <= 20 %"""
import math
import os
import sqlite3
import sys

PLUGIN = os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts")
sys.path.insert(0, PLUGIN)
from agent_eta import estimator as E, render, store, tuning  # noqa: E402

ALPHA, PER_RUN = 0.2, 6


def truncated(src, T, prev, windows):
    mem = sqlite3.connect(":memory:", isolation_level=None)
    src.backup(mem)
    mem.row_factory = sqlite3.Row
    mem.execute("DELETE FROM runs WHERE ended_at IS NULL OR ended_at > ? OR started_at > ?", (T, T))
    for tb in ("snapshots", "tool_calls", "attention", "predictions", "windows"):
        mem.execute("DELETE FROM %s WHERE run_id NOT IN (SELECT id FROM runs)" % tb)
    mem.execute("DELETE FROM windows")  # the ledger = the promises this simulated policy showed
    for run_id, t, b in windows:
        if t < T:
            mem.execute("INSERT INTO windows(run_id, issued_at, horizon_s, expires_at, safe_s) VALUES (?,?,?,?,?)",
                        (run_id, t, b, t + b, b))
    mem.execute("DELETE FROM windows WHERE run_id NOT IN (SELECT id FROM runs)")
    mem.execute("DELETE FROM tuning")
    if prev is not None:
        mem.execute("INSERT INTO tuning(key, value, n_runs) VALUES ('safe_k', ?, 0)", (prev,))
    return mem


def simulate(conn, pool, runs, marks, policy):
    windows, prev, cur = [], None, None
    shown = br = 0
    mins = 0.0
    mi = 0
    for r in sorted(runs, key=lambda r: r["started_at"]):
        while mi < len(marks) and marks[mi] <= r["started_at"]:
            mem = truncated(conn, marks[mi], prev, windows)
            res = tuning.tune(mem, now=marks[mi])
            prev = res["safe_k"]
            n, b = res["safe_stats"][res["safe_k"]]
            cur = {"k0": res["k0"], "hl": res["half_life"], "recal": E.tuned_recal(mem), "safe_k": res["safe_k"],
                   "medium": n >= 10 and b <= ALPHA * n}
            mem.close()
            mi += 1
        if cur is None and policy != "current":
            continue
        if policy == "medium" and not cur["medium"]:
            continue
        snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t",
                             (r["id"],)).fetchall()
        step = max(1, len(snaps) // PER_RUN)
        for sn in snaps[::step][:PER_RUN]:
            attn = sn["rem_attn_s"] if sn["attn_event"] else None
            if attn is None:
                continue
            st = E.state_from_snapshot(r, sn)
            if cur is None:
                est = E.estimate(conn, st, pool=pool, with_permission=False, k0=E.K0_DEFAULT,
                                 half_life=E.RECENCY_HALF_LIFE_D, recal=False, safe_k=E.SAFE_K_DEFAULT)
            else:
                est = E.estimate(conn, st, pool=pool, with_permission=False, k0=cur["k0"], half_life=cur["hl"],
                                 recal=cur["recal"] or False, safe_k=cur["safe_k"])
            b = render.safe_bucket(E.safe_seconds(est, ALPHA))
            if b:
                shown += 1
                br += attn < b
                mins += b / 60.0
                windows.append((r["id"], sn["t"], b))
    return shown, br, mins


def main(home):
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    pool = E.Pool.load(conn, 1e12, limit=100000)
    runs = [r for r in conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL"
                                    " ORDER BY ended_at").fetchall() if r["id"] in pool.runs]
    marks, last = [], None
    for i, r in enumerate(runs, 1):
        if (last is None and i >= tuning.FIRST_TUNE_RUNS) or (last is not None and i - last >= max(5, tuning.RETUNE_GROWTH * last)):
            marks.append(r["ended_at"])
            last = i
    name = os.path.basename(os.path.dirname(home.rstrip("/")))
    print("-- %s：%d 次运行，%d 次调参" % (name, len(runs), len(marks)), flush=True)
    for policy in os.environ.get("POLICIES", "current,after_tune,medium").split(","):
        n, br, m = simulate(conn, pool, runs, marks, policy)
        ub = E.clopper_pearson_upper(n, br, 0.05) if n else None
        print("   %-11s %4d 次承诺  违约 %2d 次 %3.0f%%（95%%上界 %4s）  %5.0f 分钟" % (
            policy, n, br, 100.0 * br / n if n else 0, "–" if ub is None else "%.0f%%" % (100 * ub), m), flush=True)


if __name__ == "__main__":
    for h in sys.argv[1:]:
        main(h)
