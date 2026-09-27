"""Drift check: does weighting recent runs more (shorter recency half-life) help?"""
import math, random, sys, time
sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, store, tuning
rng = random.Random(3)
conn = store.connect(create=False)
FULL = E.Pool.load(conn, time.time() + 1, limit=100000)
EMPTY = E.Pool()
RUNS = [r for r in conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL ORDER BY started_at").fetchall() if r["id"] in FULL.runs]
pts = []
for r in RUNS:
    snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t", (r["id"],)).fetchall()
    step = max(1, len(snaps) // 6)
    for s in snaps[::step][:6]:
        pts.append({"run": r["id"], "start": r["started_at"], "state": E.state_from_snapshot(r, s),
                    "done": None if s["censored"] else s["rem_done_s"], "attn": s["rem_attn_s"] if s["attn_event"] else None})
def summ(e): return {"p50": e["done_p50"], "p80": e["done_p80"], "a20": e["attn_p20"]}
prior = [summ(E.estimate(conn, p["state"], pool=EMPTY, with_permission=False, k0=math.inf, safe_k=6.0)) for p in pts]
cut = sorted(r["started_at"] for r in RUNS)[-20]
def skill(sel, ests):
    per = {}
    for i in sel:
        a, b = tuning.point_loss(pts[i], ests[i]), tuning.point_loss(pts[i], prior[i])
        if a is not None and b is not None: per.setdefault(pts[i]["run"], []).append((a, b))
    ids = list(per)
    def s(ids_):
        la = sum(a for i in ids_ for a, _ in per[i]); lb = sum(b for i in ids_ for _, b in per[i]); return 1 - la / lb
    bs = sorted(s([rng.choice(ids) for _ in ids]) for _ in range(2000))
    return s(ids), bs[50], bs[1949]
allp = range(len(pts)); recent = [i for i, p in enumerate(pts) if p["start"] >= cut]
for hl in (60.0, 7.0, 2.0, 0.5):
    E.RECENCY_HALF_LIFE_D = hl
    row = []
    for k0 in (50.0, 20.0, math.inf):
        ests = [summ(E.estimate(conn, p["state"], pool=FULL, with_permission=False, k0=k0, safe_k=6.0)) for p in pts]
        a, r_ = skill(allp, ests), skill(recent, ests)
        row.append("K0=%s 全部 %+.1f%% [%+.1f,%+.1f] 最近20次 %+.1f%% [%+.1f,%+.1f]" % ("∞" if k0 == math.inf else "%g" % k0, 100*a[0], 100*a[1], 100*a[2], 100*r_[0], 100*r_[1], 100*r_[2]))
    print("半衰期 %4.1f 天:\n   " % hl + "\n   ".join(row))
