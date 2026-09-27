"""Does Agent ETA's learning pay off, and is there still marginal value in more data?

Scores the plugin's own estimator (same code, same loss as tuning.py: log-scale pinball of done P50/P80
and attention P20) under controlled amounts of history. Run on a copy of the database.

  E1 prequential   every point predicted with only the runs finished before it (what really happened),
                   grouped by how much history existed at that moment
  E2 fixed test    the 20 most recent runs; history = n random runs from before them (n = 0 .. all)
  E3 exchangeable  every run in turn is the test; history = n random other runs (lower variance, but
                   ignores drift over time)

Uncertainty: cluster bootstrap over runs (points of one run are not independent).
"""
import json
import math
import random
import statistics
import os
import sys
import time

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E  # noqa: E402
from agent_eta import render, store, tuning  # noqa: E402

INF = math.inf
PER_RUN = 6
SAFE_Q, SAFE_K = 0.2, 6.0
rng = random.Random(20260925)

conn = store.connect(create=False)
FULL = E.Pool.load(conn, time.time() + 1, limit=100000)
RUNS = [r for r in conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL"
                                " ORDER BY started_at").fetchall() if r["id"] in FULL.runs]
BY_ID = {r["id"]: r for r in RUNS}


def points_of(run):
    snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t",
                         (run["id"],)).fetchall()
    step = max(1, len(snaps) // PER_RUN)
    return [{"run": run["id"], "state": E.state_from_snapshot(run, s),
             "done": None if s["censored"] else s["rem_done_s"],
             "attn": s["rem_attn_s"] if s["attn_event"] else None} for s in snaps[::step][:PER_RUN]]


PTS = {r["id"]: points_of(r) for r in RUNS}
EMPTY = E.Pool()


def sub(ids):
    p = E.Pool()
    p.runs = {i: FULL.runs[i] for i in ids}
    return p


def summ(est):
    return {"p50": est["done_p50"], "p80": est["done_p80"], "a20": est["attn_p20"],
            "safe": E.safe_seconds(est, SAFE_Q, SAFE_K)}


def estimates(pt, pool, k0s, as_of=None):
    st = pt["state"] if as_of is None else dict(pt["state"], as_of=as_of)
    cands = pool.candidates(st)
    return {k: summ(E.estimate(conn, st, pool=pool, with_permission=False, k0=k, safe_k=SAFE_K, cands=cands))
            for k in k0s}


PRIOR = {}  # built-in prior: no history at all, independent of the pool


def prior_of(pt):
    key = (pt["run"], pt["state"]["as_of"])
    if key not in PRIOR:
        PRIOR[key] = estimates(pt, EMPTY, (INF,))[INF]
    return PRIOR[key]


def done_loss(pt, est):
    if pt["done"] is None or est["p50"] is None or est["p80"] is None:
        return None
    return (tuning._log_pinball(0.5, pt["done"], est["p50"]) + tuning._log_pinball(0.8, pt["done"], est["p80"])) / 2


def metrics(rows):
    """rows: (point, est). Loss, calibration and the leave-window outcome."""
    losses = [tuning.point_loss(p, e) for p, e in rows]
    losses = [x for x in losses if x is not None]
    done = [(p, e) for p, e in rows if p["done"] is not None and e["p50"] is not None]
    promised = [(p, render.safe_bucket(e["safe"])) for p, e in rows if p["attn"] is not None]
    promised = [(p, b) for p, b in promised if b]
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        "hit50": sum(p["done"] <= e["p50"] for p, e in done) / len(done) if done else None,
        "cov80": sum(p["done"] <= e["p80"] for p, e in done) / len(done) if done else None,
        "x_err": math.exp(statistics.median(abs(math.log(max(e["p50"], 1) / max(p["done"], 1))) for p, e in done))
        if done else None,
        "promises": len(promised), "breach": sum(p["attn"] < b for p, b in promised),
        "promised_min": sum(b for _, b in promised) / 60.0,
        "n_attn": sum(1 for p, _ in rows if p["attn"] is not None),
    }


def cluster_boot(per_run, stat=lambda xs: sum(xs) / len(xs), b=2000):
    """per_run: {run: [values]} -> (estimate, lo95, hi95) of the mean over points, resampling runs."""
    ids = [i for i in per_run if per_run[i]]
    est = stat([v for i in ids for v in per_run[i]])
    boots = []
    for _ in range(b):
        vals = [v for i in (rng.choice(ids) for _ in ids) for v in per_run[i]]
        boots.append(stat(vals))
    boots.sort()
    return est, boots[int(0.025 * b)], boots[int(0.975 * b) - 1]


def diff_by_run(rows_a, rows_b):
    """Paired per-point loss differences a - b, grouped by run (only points both can score)."""
    out = {}
    for (p, ea), (_, eb) in zip(rows_a, rows_b):
        la, lb = tuning.point_loss(p, ea), tuning.point_loss(p, eb)
        if la is not None and lb is not None:
            out.setdefault(p["run"], []).append(la - lb)
    return out


def fmt(m):
    return ("loss %.3f  P50命中 %3.0f%%  P80覆盖 %3.0f%%  误差×%.2f  可离开 %d次/违约%d (%.0f%%) 共%.0f分钟" % (
        m["loss"], 100 * m["hit50"], 100 * m["cov80"], m["x_err"], m["promises"], m["breach"],
        100.0 * m["breach"] / m["promises"] if m["promises"] else 0, m["promised_min"]))


RESULT = {}
t0 = time.time()

# ------------------------------------------------------------------ E1 prequential
print("== E1 按时间顺序回放（每个点只用当时已结束的运行）  runs=%d points=%d" % (len(RUNS), sum(map(len, PTS.values()))))
rows = {"prior": [], "cal": [], "full": [], "b0": []}
hist = []
for r in RUNS:
    for pt in PTS[r["id"]]:
        e = estimates(pt, FULL, (50.0, INF))
        rows["prior"].append((pt, prior_of(pt)))
        rows["cal"].append((pt, e[INF]))
        rows["full"].append((pt, e[50.0]))
        q = E.b0_estimate(FULL, pt["state"])
        rows["b0"].append((pt, {"p50": q[0.5], "p80": q[0.8], "a20": None, "safe": None}))
        hist.append(sum(1 for m in FULL.runs.values() if m[0]["ended_at"] < pt["state"]["as_of"]))
buckets = [(0, 15), (15, 30), (30, 45), (45, 70)]
RESULT["E1"] = []
for lo, hi in buckets:
    idx = [i for i, h in enumerate(hist) if lo <= h < hi]
    sel = {k: [v[i] for i in idx] for k, v in rows.items()}
    d = cluster_boot(diff_by_run(sel["full"], sel["prior"]))
    base = metrics(sel["prior"])["loss"]
    runs_in = len({rows["prior"][i][0]["run"] for i in idx})
    line = {"hist": [lo, hi], "runs": runs_in, "prior": metrics(sel["prior"]), "cal": metrics(sel["cal"]),
            "full": metrics(sel["full"]), "skill": -d[0] / base, "skill_ci": [-d[2] / base, -d[1] / base]}
    RESULT["E1"].append(line)
    print("  历史 %2d-%2d 次（%d 次运行）  学习后相对内置先验 %+.1f%%  [95%% CI %+.1f%%, %+.1f%%]" % (
        lo, hi - 1, runs_in, 100 * line["skill"], 100 * line["skill_ci"][0], 100 * line["skill_ci"][1]))
    for k in ("prior", "cal", "full"):
        print("     %-5s %s" % (k, fmt(line[k])))
all_d = cluster_boot(diff_by_run(rows["full"], rows["prior"]))
all_base = metrics(rows["prior"])["loss"]
RESULT["E1_all"] = {"skill": -all_d[0] / all_base, "ci": [-all_d[2] / all_base, -all_d[1] / all_base],
                    "prior": metrics(rows["prior"]), "cal": metrics(rows["cal"]), "full": metrics(rows["full"])}
print("  全部：学习后相对内置先验 %+.1f%%  [95%% CI %+.1f%%, %+.1f%%]   (%.0fs)" % (
    100 * RESULT["E1_all"]["skill"], 100 * RESULT["E1_all"]["ci"][0], 100 * RESULT["E1_all"]["ci"][1], time.time() - t0))

# ------------------------------------------------------------------ E2 fixed temporal test set
TEST = RUNS[-20:]
cut = min(r["started_at"] for r in TEST)
TRAIN = [r["id"] for r in RUNS if r["ended_at"] < cut]
print("\n== E2 固定测试集：最近 %d 次运行；历史从它们之前的 %d 次里随机抽 n 次" % (len(TEST), len(TRAIN)))
test_pts = [pt for r in TEST for pt in PTS[r["id"]]]
prior_rows = [(pt, prior_of(pt)) for pt in test_pts]
RESULT["E2"] = []
grid = [0, 5, 10, 15, 20, 30, len(TRAIN)]
for n in grid:
    draws = 1 if n in (0, len(TRAIN)) else 30
    per_draw = {50.0: [], INF: []}
    per_run = {50.0: {}, INF: {}}
    pooled = {50.0: [], INF: []}
    for _ in range(draws):
        pool = sub(rng.sample(TRAIN, n))
        rows_k = {50.0: [], INF: []}
        for pt in test_pts:
            e = estimates(pt, pool, (50.0, INF))
            for k in (50.0, INF):
                rows_k[k].append((pt, e[k]))
        for k in (50.0, INF):
            per_draw[k].append(metrics(rows_k[k])["loss"])
            pooled[k] += rows_k[k]
            for rid, ds in diff_by_run(rows_k[k], prior_rows).items():
                per_run[k].setdefault(rid, []).append(sum(ds) / len(ds))
    base = metrics(prior_rows)["loss"]
    line = {"n": n}
    for k in (50.0, INF):
        d = cluster_boot(per_run[k])
        line[str(k)] = {"loss": sum(per_draw[k]) / len(per_draw[k]), "skill": -d[0] / base,
                        "ci": [-d[2] / base, -d[1] / base], "metrics": metrics(pooled[k])}
    RESULT["E2"].append(line)
    print("  n=%2d  完整估计器 %+.1f%% [%+.1f, %+.1f]   只校准 %+.1f%% [%+.1f, %+.1f]   %s" % (
        n, 100 * line["50.0"]["skill"], 100 * line["50.0"]["ci"][0], 100 * line["50.0"]["ci"][1],
        100 * line["inf"]["skill"], 100 * line["inf"]["ci"][0], 100 * line["inf"]["ci"][1],
        fmt(line["50.0"]["metrics"])))
print("  (%.0fs)" % (time.time() - t0))

# ------------------------------------------------------------------ E3 exchangeable learning curve
AS_OF = max(r["ended_at"] for r in RUNS) + 1.0
K0S = (8.0, 20.0, 50.0, INF)
all_ids = [r["id"] for r in RUNS]
grid3 = [0, 5, 10, 20, 30, 40, 50, len(RUNS) - 1]
DRAWS3 = 6
print("\n== E3 可交换学习曲线：每次运行轮流当测试，历史 = 其余运行里随机 n 次（每个 n 抽 %d 次）" % DRAWS3)
prior3 = {}
for r in RUNS:
    for pt in PTS[r["id"]]:
        prior3[id(pt)] = estimates(pt, EMPTY, (INF,), AS_OF)[INF]
RESULT["E3"] = []
for n in grid3:
    per_run = {k: {} for k in K0S}
    for r in RUNS:
        others = [i for i in all_ids if i != r["id"]]
        draws = 1 if n in (0, len(others)) else DRAWS3
        for _ in range(draws):
            pool = sub(rng.sample(others, n))
            for pt in PTS[r["id"]]:
                lp = tuning.point_loss(pt, prior3[id(pt)])
                if lp is None:
                    continue
                e = estimates(pt, pool, K0S, AS_OF)
                for k in K0S:
                    per_run[k].setdefault(r["id"], []).append(tuning.point_loss(pt, e[k]) - lp)
    base = sum(tuning.point_loss(pt, prior3[id(pt)]) for r in RUNS for pt in PTS[r["id"]]
               if tuning.point_loss(pt, prior3[id(pt)]) is not None) / sum(
        1 for r in RUNS for pt in PTS[r["id"]] if tuning.point_loss(pt, prior3[id(pt)]) is not None)
    line = {"n": n, "base": base}
    for k in K0S:
        d = cluster_boot(per_run[k], b=1000)
        line[str(k)] = {"skill": -d[0] / base, "ci": [-d[2] / base, -d[1] / base], "per_run": per_run[k]}
    best = max(K0S, key=lambda k: line[str(k)]["skill"])
    line["best_k0"] = best
    RESULT["E3"].append(line)
    print("  n=%2d  " % n + "  ".join("K0=%s %+.1f%%" % ("∞" if k == INF else "%g" % k, 100 * line[str(k)]["skill"])
                                      for k in K0S) + "   最优 K0=%s" % ("∞" if best == INF else "%g" % best))
print("  (%.0fs)" % (time.time() - t0))

with open(sys.argv[1] if len(sys.argv) > 1 else "result.json", "w") as f:
    json.dump(RESULT, f, default=lambda o: None if isinstance(o, float) and math.isinf(o) else str(o))
