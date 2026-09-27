"""Leave windows across users: raw 20 % quantile, SAFE_K shrinkage, and ACI with several step sizes,
for the built-in prior (no learning) and the tuned estimator. Prequential, delayed feedback."""
import math
import os
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402

ALPHA = 0.2
users = sys.argv[1:]
print("%-10s %-24s %6s %6s %7s %8s %8s" % ("用户", "方法", "承诺", "违约", "违约率", "承诺分钟", "逐点误差"))
for home in users:
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    cfg = tuning.current_config(conn)
    tuned = E.tuned_values(conn)
    pts = tuning.replay(conn, max_runs=100000, configs=[cfg], safe_q=ALPHA, keep=True, pool_limit=100000)
    pts = [p for p in pts if p["attn"] is not None]
    empty = E.Pool()
    for p in pts:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (p["run"],)).fetchone()
        snap = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND ABS(t-?)<1e-6", (p["run"], p["as_of"])).fetchone()
        p["S_prior"] = E.estimate(conn, E.state_from_snapshot(run, snap), pool=empty, with_permission=False,
                                  k0=math.inf, recal=False, safe_k=6.0)["S_attn"]
    order = sorted(range(len(pts)), key=lambda i: pts[i]["as_of"])
    by_end = sorted(range(len(pts)), key=lambda i: pts[i]["ended"])

    def run(which, method, gamma=0.03):
        q, j, pending = ALPHA, 0, {}
        shown = br = 0
        mins = 0.0
        errs = []
        for i in order:
            p = pts[i]
            while j < len(by_end) and pts[by_end[j]]["ended"] < p["as_of"]:
                k = by_end[j]
                if k in pending and method == "aci":
                    q = min(0.95, q + gamma * (ALPHA - pending[k]))
                pending.pop(k, None)
                j += 1
            est = p["est"][cfg]["est"]
            S = p["S_prior"] if which == "prior" else est["S_attn"]
            if method == "aci":
                w = E.quantile(S, q) if q > 0 else 0.0
            elif method == "safe_k":
                w = E.safe_seconds(dict(est, S_attn=S, ess_global=0.0 if which == "prior" else est["ess_global"]),
                                   ALPHA, 6.0 if which == "prior" else tuned.get("safe_k", 6.0))
            else:
                w = E.quantile(S, ALPHA)
            w = w or 0.0
            pending[i] = 1.0 if p["attn"] < w else 0.0
            errs.append(pending[i])
            b = render.safe_bucket(w)
            if b:
                shown += 1
                br += p["attn"] < b
                mins += b / 60.0
        return shown, br, mins, sum(errs) / len(errs)

    name = os.path.basename(os.path.dirname(home.rstrip("/")))
    print("-- %s: %d 个点，%d 次运行，调参 K0=%s 半衰期=%g 重校准=%s SAFE_K=%g" % (
        name, len(pts), len({p["run"] for p in pts}), cfg[0], cfg[1], bool(tuned.get("recal")), tuned.get("safe_k", 6.0)))
    for which in ("prior", "learned"):
        for method, gamma, label in (("raw", 0, "直接取20%分位"), ("safe_k", 0, "SAFE_K"), ("aci", 0.01, "ACI γ=0.01"),
                                     ("aci", 0.03, "ACI γ=0.03"), ("aci", 0.1, "ACI γ=0.1")):
            s, b, m, e = run(which, method, gamma)
            print("%-10s %-24s %6d %6d %6.0f%% %8.0f %7.0f%%" % (
                "先验" if which == "prior" else "学习后", label, s, b, 100.0 * b / s if s else 0, m, 100 * e))
