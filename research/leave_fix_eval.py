"""Candidate fixes for leave windows, per history (replay points, tuned K0 / half-life / SAFE_K):
  now      as v0.5.1 (attention recalibration only when the tuner turned recalibration on)
  F1       the leave window always read through a prequential attention recalibration (the lower tail)
  F1+F2    ... and no promise at all when the replay shows every SAFE_K candidate breaching > 20 %
           (at least MIN_PROMISES promises), evaluated with the SAFE_K stats known before each point"""
import math
import os
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402


def history_name(home):
    """A label for a history directory: the parent's name for .../<name>/home, else its own name."""
    p = os.path.abspath(home.rstrip("/"))
    base = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if base == "home" else base

TOT = {k: [0, 0, 0.0] for k in ("now", "F1", "F1+F2")}
for home in sys.argv[1:]:
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    cfg = tuning.current_config(conn)
    tuned = E.tuned_values(conn)
    k = tuned.get("safe_k", 6.0)
    recal_on = bool(tuned.get("recal"))
    pts = tuning.replay(conn, max_runs=100000, configs=[cfg], keep=True, with_baselines=False, pool_limit=100000)
    pre = tuning.prequential_recal(pts, cfg, 0.2, k)  # windows read through a recalibration learned from the past
    order = sorted(range(len(pts)), key=lambda i: pts[i]["as_of"])
    by_end = sorted(range(len(pts)), key=lambda i: pts[i]["ended"])
    res = {x: [0, 0, 0.0] for x in TOT}
    known, j = [], 0
    for i in order:
        p = pts[i]
        while j < len(by_end) and pts[by_end[j]]["ended"] < p["as_of"]:
            known.append(pts[by_end[j]])
            j += 1
        if p["attn"] is None:
            continue
        est = p["est"][cfg]["est"]
        w_now = E.safe_seconds(dict(est, recal=E.tuned_recal(conn) if recal_on else None), 0.2, k) if recal_on else \
            E.safe_seconds(dict(est, recal=None), 0.2, k)
        w_f1 = pre[i]["safe"]
        # F2: would every SAFE_K candidate have breached > 20 % on what was known then (read through F1)?
        fail = False
        if len(known) >= 30:
            rec = tuning.fit_recal(known, cfg)
            rates = []
            for kk in tuning.SAFE_GRID:
                n = b = 0
                for q in known:
                    if q["attn"] is None:
                        continue
                    bk = render.safe_bucket(E.safe_seconds(dict(q["est"][cfg]["est"], recal=rec), 0.2, kk))
                    if bk:
                        n += 1
                        b += q["attn"] < bk
                if n >= tuning.MIN_PROMISES:
                    rates.append(b / n)
            fail = bool(rates) and min(rates) > 0.2
        for key, w in (("now", w_now), ("F1", w_f1), ("F1+F2", None if fail else w_f1)):
            bk = render.safe_bucket(w)
            if bk:
                res[key][0] += 1
                res[key][1] += p["attn"] < bk
                res[key][2] += bk / 60.0
    name = history_name(home)[:38]
    print("%-38s " % name + "  ".join("%s %3d次 %3.0f%%" % (x, n, 100.0 * b / n if n else 0) for x, (n, b, m) in res.items()),
          flush=True)
    for x in TOT:
        for t in range(3):
            TOT[x][t] += res[x][t]
print("合计  " + "  ".join("%s %d 次承诺 违约 %.1f%%（95%%上界 %.1f%%）%.0f 分钟" % (
    x, n, 100.0 * b / n, 100 * E.clopper_pearson_upper(n, b, 0.05), m) for x, (n, b, m) in TOT.items()))
