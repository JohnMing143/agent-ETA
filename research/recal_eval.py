"""Hypothesis: the estimates run long because of the model's shape, not a lack of data. Test: a
quantile recalibration learned prequentially from your own past PIT values (where the truth fell in
the predicted distribution), using only runs that had finished before each point."""
import bisect
import math
import random
import os
import sys
import time

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E  # noqa: E402
from agent_eta import render, store, tuning  # noqa: E402

PER_RUN = 6
rng = random.Random(7)
conn = store.connect(create=False)
FULL = E.Pool.load(conn, time.time() + 1, limit=100000)
EMPTY = E.Pool()
RUNS = [r for r in conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL"
                                " ORDER BY started_at").fetchall() if r["id"] in FULL.runs]

pts = []
for r in RUNS:
    snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t",
                         (r["id"],)).fetchall()
    step = max(1, len(snaps) // PER_RUN)
    for s in snaps[::step][:PER_RUN]:
        st = E.state_from_snapshot(r, s)
        p = {"run": r["id"], "ended": r["ended_at"], "as_of": st["as_of"],
             "done": None if s["censored"] else s["rem_done_s"], "attn": s["rem_attn_s"] if s["attn_event"] else None}
        for name, pool, k0 in (("full", FULL, 50.0), ("prior", EMPTY, math.inf)):
            est = E.estimate(conn, st, pool=pool, with_permission=False, k0=k0, safe_k=6.0)
            p[name] = est
            p[name + "_u_done"] = 1 - est["S_done"](p["done"]) if p["done"] is not None else None
            p[name + "_u_attn"] = 1 - est["S_attn"](p["attn"]) if p["attn"] is not None else None
        pts.append(p)
pts.sort(key=lambda p: p["as_of"])


def recal_q(hist, q, m0):
    """Level to read off the curve so that, on past points, a fraction q fell below it (shrunk to q)."""
    if not hist:
        return q
    h = sorted(hist)
    emp = h[min(len(h) - 1, max(0, int(math.ceil(q * len(h))) - 1))]
    m = len(h)
    return min(0.995, max(0.005, (m * emp + m0 * q) / (m + m0)))


def run(name, m0):
    rows_raw, rows_cal = [], []
    for p in pts:
        past = [x for x in pts if x["ended"] < p["as_of"]]
        hd = [x[name + "_u_done"] for x in past if x[name + "_u_done"] is not None]
        ha = [x[name + "_u_attn"] for x in past if x[name + "_u_attn"] is not None]
        est = p[name]
        raw = {"p50": est["done_p50"], "p80": est["done_p80"], "a20": est["attn_p20"]}
        cal = {"p50": E.quantile(est["S_done"], recal_q(hd, 0.5, m0)),
               "p80": E.quantile(est["S_done"], recal_q(hd, 0.8, m0)),
               "a20": E.quantile(est["S_attn"], recal_q(ha, 0.2, m0))}
        n = est.get("ess_global") or 0.0
        q_safe = 0.2 * (n + 1) / (n + 1 + 6.0)
        raw["safe"] = E.quantile(est["S_attn"], q_safe)
        cal["safe"] = E.quantile(est["S_attn"], recal_q(ha, q_safe, m0))
        rows_raw.append((p, raw))
        rows_cal.append((p, cal))
    return rows_raw, rows_cal


def summary(rows):
    ls = [tuning.point_loss(p, e) for p, e in rows]
    ls = [x for x in ls if x is not None]
    done = [(p, e) for p, e in rows if p["done"] is not None and e["p50"]]
    prom = [(p, render.safe_bucket(e["safe"])) for p, e in rows if p["attn"] is not None]
    prom = [(p, b) for p, b in prom if b]
    return (sum(ls) / len(ls), sum(p["done"] <= e["p50"] for p, e in done) / len(done),
            sum(p["done"] <= e["p80"] for p, e in done) / len(done),
            math.exp(sorted(abs(math.log(max(e["p50"], 1) / max(p["done"], 1))) for p, e in done)[len(done) // 2]),
            len(prom), sum(p["attn"] < b for p, b in prom), sum(b for _, b in prom) / 60)


def boot_diff(ra, rb, base):
    per = {}
    for (p, a), (_, b) in zip(ra, rb):
        la, lb = tuning.point_loss(p, a), tuning.point_loss(p, b)
        if la is not None and lb is not None:
            per.setdefault(p["run"], []).append(la - lb)
    ids = list(per)
    est = sum(v for i in ids for v in per[i]) / sum(len(per[i]) for i in ids)
    bs = []
    for _ in range(3000):
        s = [rng.choice(ids) for _ in ids]
        bs.append(sum(v for i in s for v in per[i]) / sum(len(per[i]) for i in s))
    bs.sort()
    return -est / base, -bs[2924] / base, -bs[75] / base


fmtr = "loss %.3f  P50命中 %2.0f%%  P80覆盖 %2.0f%%  误差×%.2f  可离开 %d次/违约%d 共%.0f分钟"
prior_raw, _ = run("prior", 30)
base = summary(prior_raw)[0]
print("内置先验（不学习）        " + fmtr % tuple(summary(prior_raw)[:3]) % () if False else
      "内置先验（不学习）        " + fmtr % (summary(prior_raw)[0], 100 * summary(prior_raw)[1], 100 * summary(prior_raw)[2],
                                         *summary(prior_raw)[3:]))
for name in ("full", "prior"):
    for m0 in (10, 30, 60):
        raw, cal = run(name, m0)
        sr, sc = summary(raw), summary(cal)
        if m0 == 10:
            print("%-5s 原样                 " % name + fmtr % (sr[0], 100 * sr[1], 100 * sr[2], *sr[3:]))
        d = boot_diff(cal, prior_raw, base)
        print("%-5s + 重校准 m0=%-3d        " % (name, m0) + fmtr % (sc[0], 100 * sc[1], 100 * sc[2], *sc[3:]) +
              "   相对内置先验 %+.1f%% [95%% CI %+.1f, %+.1f]" % (100 * d[0], 100 * d[1], 100 * d[2]))
