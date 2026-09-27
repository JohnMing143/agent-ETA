"""Does a per-model pace (on top of the per-category one) help histories that mix models? Prequential,
cluster bootstrap over runs; each home keeps its own tuned K0 / half-life (and K0=inf isolates the prior)."""
import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, store, tuning  # noqa: E402


def history_name(home):
    """A label for a history directory: the parent's name for .../<name>/home, else its own name."""
    p = os.path.abspath(home.rstrip("/"))
    base = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if base == "home" else base

ORIG = E.calibration


def cal_model(pool, state, half_life=E.RECENCY_HALF_LIFE_D):
    mu_cat, mu, sigma, n = ORIG(pool, state, half_life)
    model = state.get("model")
    if not n or not model:
        return mu_cat, mu, sigma, n
    num = den = 0.0
    for run_id, entry in pool.runs.items():
        meta = entry[0]
        if run_id == state["run_id"] or meta["ended_at"] >= state["as_of"] or meta["end_reason"] != "stop":
            continue
        if meta["model"] != model:
            continue
        cat = meta["category"] or "other"
        med = E.prior_median({"category": cat, "effort": meta["effort"], "prompt_len": meta["prompt_len"] or 0})
        w = 0.5 ** (max(0.0, (state["as_of"] - meta["ended_at"]) / 86400.0) / half_life)
        num += w * (math.log(max(meta["run_active_s"] or 0.0, 1.0) / med) - mu_cat.get(cat, mu))
        den += w
    delta = num / (den + E.CAL_N1)
    return {c: v + delta for c, v in mu_cat.items()}, mu + delta, sigma, n


def run(home):
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    cfg = tuning.current_config(conn)
    out = {}
    for name, fn in (("base", ORIG), ("model", cal_model)):
        E.calibration = fn
        pts = tuning.replay(conn, max_runs=400, configs=[cfg, (math.inf, cfg[1])], safe_q=0.2, with_baselines=False,
                            pool_limit=100000)
        out[name] = pts
    E.calibration = ORIG
    rng = random.Random(5)
    res = []
    for c in (cfg, (math.inf, cfg[1])):
        per = {}
        for a, b in zip(out["base"], out["model"]):
            la, lb = tuning.point_loss(a, a["est"][c]), tuning.point_loss(b, b["est"][c])
            if la is not None and lb is not None:
                acc = per.setdefault(a["run"], [0.0, 0.0])
                acc[0] += la
                acc[1] += lb
        ids = sorted(per)
        g = lambda s: 1 - sum(per[i][1] for i in s) / sum(per[i][0] for i in s)  # noqa: E731
        bs = sorted(g([rng.choice(ids) for _ in ids]) for _ in range(2000))
        res.append((c, g(ids), bs[50], bs[1949], len(ids)))
    models = conn.execute("SELECT model, COUNT(*) FROM runs WHERE end_reason='stop' GROUP BY model ORDER BY 2 DESC").fetchall()
    return res, [(m[0], m[1]) for m in models]


if __name__ == "__main__":
    for home in sys.argv[1:]:
        res, models = run(home)
        print("-- %s  模型：%s" % (history_name(home), ", ".join("%s×%d" % (str(m).split("-2026")[0].replace("claude-", ""), n) for m, n in models)))
        for c, gv, lo, hi, n in res:
            print("   %-22s 加上按模型的节奏：%+.1f%%  [95%% %+.1f%%, %+.1f%%]（%d 次运行）" % (
                "K0=%s 半衰期 %g 天" % ("∞" if math.isinf(c[0]) else "%g" % c[0], c[1]), 100 * gv, 100 * lo, 100 * hi, n))
