"""Model-agnostic leave windows? Adaptive conformal inference (Gibbs & Candes 2021) on the attention
quantile, prequential with delayed feedback (an outcome counts once its run has finished), compared with
the plugin's SAFE_K shrinkage, for the built-in prior (no learning) and the learned estimator."""
import math
import os
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402

conn = store.connect(create=False)
cfg = tuning.current_config(conn)
pts = tuning.replay(conn, max_runs=500, configs=[cfg], safe_q=0.2, keep=True, pool_limit=100000)
pts = [p for p in pts if p["attn"] is not None]
prior = {}
for p in pts:
    pass
order = sorted(range(len(pts)), key=lambda i: pts[i]["as_of"])
by_end = sorted(range(len(pts)), key=lambda i: pts[i]["ended"])
print("有“需要你”结果的回放点 %d 个，%d 次运行\n" % (len(pts), len({p["run"] for p in pts})))


def curves(p, which):
    if which == "prior":
        return p["_prior_S"]
    return p["est"][cfg]["est"]["S_attn"]


# built-in prior curves for the same points (no history at all)
empty = E.Pool()
for p in pts:
    run = conn.execute("SELECT * FROM runs WHERE id=?", (p["run"],)).fetchone()
    snap = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND ABS(t-?)<1e-6", (p["run"], p["as_of"])).fetchone()
    st = E.state_from_snapshot(run, snap)
    p["_prior_S"] = E.estimate(conn, st, pool=empty, with_permission=False, k0=math.inf, recal=False, safe_k=6.0)["S_attn"]
    p["_ess"] = p["est"][cfg]["est"]["ess_global"] or 0.0


def evaluate(which, method, alpha=0.2, gamma=0.03, safe_k=6.0):
    q = alpha
    j = 0
    seen = []
    shown = breached = 0
    minutes = 0.0
    errs = []
    pending = {}
    for i in order:
        p = pts[i]
        # outcomes of runs finished before this moment arrive now (delayed feedback)
        while j < len(by_end) and pts[by_end[j]]["ended"] < p["as_of"]:
            k = by_end[j]
            if k in pending:
                err = pending.pop(k)
                if method == "aci":
                    q = min(0.95, max(0.001, q + gamma * (alpha - err)))
            j += 1
        S = curves(p, which)
        if method == "aci":
            level = q
        elif method == "safe_k":
            n = p["_ess"] if which == "learned" else 0.0
            level = alpha * (n + 1) / (n + 1 + safe_k)
        else:
            level = alpha
        w = E.quantile(S, level) or 0.0
        pending[i] = 1.0 if p["attn"] < w else 0.0
        errs.append(pending[i])
        b = render.safe_bucket(w)
        if b:
            shown += 1
            breached += p["attn"] < b
            minutes += b / 60.0
    return shown, breached, minutes, sum(errs) / len(errs)


print("%-26s %6s %6s %8s %10s" % ("", "承诺次数", "违约", "违约率", "承诺总分钟"))
for which, name in (("prior", "内置先验（不学习）"), ("learned", "学习后的估计器")):
    for method, mname in (("raw", "直接取 20% 分位"), ("safe_k", "SAFE_K=6 收缩"), ("aci", "ACI 在线校准 α=20%")):
        if which == "prior" and method == "safe_k":
            mname = "SAFE_K=6 收缩（无历史）"
        s, b, m, _ = evaluate(which, method)
        print("%-10s %-18s %6d %6d %7.0f%% %9.0f" % (name, mname, s, b, 100.0 * b / s if s else 0, m))
