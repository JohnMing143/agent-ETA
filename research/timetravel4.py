"""The live system replayed through time (as timetravel3.py, policy: no promise before the first tune), also
recording every point's loss for the tuned estimator and for the built-in prior. Writes JSON per history
so two plugin versions can be compared point by point.  PLUGIN=<scripts dir> timetravel4.py OUT_DIR HOME..."""
import json
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402


def history_name(home):
    """A label for a history directory: the parent's name for .../<name>/home, else its own name."""
    p = os.path.abspath(home.rstrip("/"))
    base = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if base == "home" else base

ALPHA, PER_RUN = 0.2, 6


def truncated(src, T, prev, windows):
    mem = sqlite3.connect(":memory:", isolation_level=None)
    src.backup(mem)
    mem.row_factory = sqlite3.Row
    mem.execute("DELETE FROM runs WHERE ended_at IS NULL OR ended_at > ? OR started_at > ?", (T, T))
    for tb in ("snapshots", "tool_calls", "attention", "predictions", "windows"):
        mem.execute("DELETE FROM %s WHERE run_id NOT IN (SELECT id FROM runs)" % tb)
    mem.execute("DELETE FROM windows")
    for run_id, t, b in windows:
        if t < T:
            mem.execute("INSERT INTO windows(run_id, issued_at, horizon_s, expires_at, safe_s) VALUES (?,?,?,?,?)",
                        (run_id, t, b, t + b, b))
    mem.execute("DELETE FROM windows WHERE run_id NOT IN (SELECT id FROM runs)")
    mem.execute("DELETE FROM tuning")
    if prev is not None:
        mem.execute("INSERT INTO tuning(key, value, n_runs) VALUES ('safe_k', ?, 0)", (prev,))
    return mem


def main(out_dir, home):
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    pool = E.Pool.load(conn, 1e12, limit=100000)
    empty = E.Pool()
    runs = [r for r in conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL"
                                    " ORDER BY ended_at").fetchall() if r["id"] in pool.runs]
    marks, last = [], None
    for i, r in enumerate(runs, 1):
        if (last is None and i >= tuning.FIRST_TUNE_RUNS) or (last is not None and i - last >= max(5, tuning.RETUNE_GROWTH * last)):
            marks.append(r["ended_at"])
            last = i
    windows, prev, cur, mi = [], None, None, 0
    recs = []
    for r in sorted(runs, key=lambda r: r["started_at"]):
        while mi < len(marks) and marks[mi] <= r["started_at"]:
            mem = truncated(conn, marks[mi], prev, windows)
            res = tuning.tune(mem, now=marks[mi])
            prev = res["safe_k"]
            cur = {"k0": res["k0"], "hl": res["half_life"], "recal": E.tuned_recal(mem), "safe_k": res["safe_k"]}
            mem.close()
            mi += 1
        snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t",
                             (r["id"],)).fetchall()
        step = max(1, len(snaps) // PER_RUN)
        for sn in snaps[::step][:PER_RUN]:
            st = E.state_from_snapshot(r, sn)
            pt = {"done": None if sn["censored"] else sn["rem_done_s"], "attn": sn["rem_attn_s"] if sn["attn_event"] else None}
            if cur is None:
                est = E.estimate(conn, st, pool=pool, with_permission=False, k0=E.K0_DEFAULT,
                                 half_life=E.RECENCY_HALF_LIFE_D, recal=False, safe_k=E.SAFE_K_DEFAULT)
            else:
                est = E.estimate(conn, st, pool=pool, with_permission=False, k0=cur["k0"], half_life=cur["hl"],
                                 recal=cur["recal"] or False, safe_k=cur["safe_k"])
            pri = E.estimate(conn, st, pool=empty, with_permission=False, k0=math.inf,
                             half_life=E.RECENCY_HALF_LIFE_D, recal=False, safe_k=E.SAFE_K_DEFAULT)
            summ = lambda e: {"p50": e["done_p50"], "p80": e["done_p80"], "a20": e["attn_p20"]}  # noqa: E731
            b = None
            if cur is not None and pt["attn"] is not None:
                b = render.safe_bucket(E.safe_seconds(est, ALPHA))
                if b:
                    windows.append((r["id"], sn["t"], b))
            recs.append({"run": r["id"], "t": sn["t"], "loss": tuning.point_loss(pt, summ(est)),
                         "loss_prior": tuning.point_loss(pt, summ(pri)), "promise": b,
                         "breach": (pt["attn"] < b) if b else None,
                         "hit50": (pt["done"] <= est["done_p50"]) if pt["done"] is not None and est["done_p50"] else None,
                         "cov80": (pt["done"] <= est["done_p80"]) if pt["done"] is not None and est["done_p80"] else None})
    name = history_name(home)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(recs, f)
    prom = [x for x in recs if x["promise"]]
    ls = [x["loss"] for x in recs if x["loss"] is not None]
    print("%-40s %4d 次承诺 违约 %3.0f%% %5.0f 分钟 | loss %.3f（先验 %.3f）" % (
        name, len(prom), 100.0 * sum(x["breach"] for x in prom) / len(prom) if prom else 0,
        sum(x["promise"] for x in prom) / 60.0, sum(ls) / len(ls),
        sum(x["loss_prior"] for x in recs if x["loss_prior"] is not None) / max(1, sum(1 for x in recs if x["loss_prior"] is not None))),
        flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    for h in sys.argv[2:]:
        main(sys.argv[1], h)
