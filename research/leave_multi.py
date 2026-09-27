"""Leave windows calibrated on what the user actually sees (shown promises >= 1 min), across users.
Prequential with delayed feedback: a moment's outcome is known once its run has finished.

  safe_k       the plugin's current rule (ESS shrinkage, tuned SAFE_K)
  aci_pt       textbook ACI on every moment (per-point error -> 20 %)
  aci_shown    ACI updated only by promises that were shown (at issue time)
  aci_cf       ACI updated at resolution time by the promise the *current* level would have shown
  src          selective risk control: on the most recent resolved moments, walk the level grid from
               cautious to generous and keep the last level whose shown-promise breach rate has a 75 %
               upper bound <= 20 % (Clopper-Pearson); too little evidence at a level stops the walk
"""
import math
import os
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402

ALPHA = 0.2
LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]
RECENT = 300
MIN_N = 10
GAMMA = float(os.environ.get("GAMMA", "0.03"))
AIM = float(os.environ.get("AIM", "0.2"))  # ACI's internal target


def load(home):
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
    return conn, cfg, tuned, pts


def windows(S):
    return {lv: E.quantile(S, lv) or 0.0 for lv in LEVELS}


def simulate(pts, which, method, tuned):
    order = sorted(range(len(pts)), key=lambda i: pts[i]["as_of"])
    by_end = sorted(range(len(pts)), key=lambda i: pts[i]["ended"])
    wcache = {}
    for i, p in enumerate(pts):
        S = p["S_prior"] if which == "prior" else p["est"][p["_cfg"]]["est"]["S_attn"]
        wcache[i] = S
    q, j = AIM, 0
    issued = {}
    resolved = []
    shown = br = 0
    mins = 0.0
    log = []
    for i in order:
        p = pts[i]
        while j < len(by_end) and pts[by_end[j]]["ended"] < p["as_of"]:
            k = by_end[j]
            j += 1
            if k not in issued:
                continue
            w_issued, was_shown = issued.pop(k)
            A = pts[k]["attn"]
            resolved.append(k)
            if method == "aci_pt":
                q = min(0.95, q + GAMMA * (ALPHA - (1.0 if A < w_issued else 0.0)))
            elif method == "aci_shown" and was_shown:
                q = min(0.95, max(0.001, q + GAMMA * (AIM - (1.0 if A < was_shown else 0.0))))
            elif method == "aci_cf":
                b = render.safe_bucket(E.quantile(wcache[k], q) if q > 0 else 0.0)
                if b:
                    q = min(0.95, max(0.001, q + GAMMA * (AIM - (1.0 if A < b else 0.0))))
        S = wcache[i]
        if method == "safe_k":
            est = p["est"][p["_cfg"]]["est"]
            w = E.safe_seconds(dict(est, S_attn=S, ess_global=0.0 if which == "prior" else est["ess_global"]),
                               ALPHA, 6.0 if which == "prior" else tuned.get("safe_k", 6.0)) or 0.0
        elif method == "raw":
            w = E.quantile(S, ALPHA) or 0.0
        elif method == "src":
            recent = resolved[-RECENT:]
            lv_ok = None
            for lv in LEVELS:
                n = b_ = 0
                for k in recent:
                    bk = render.safe_bucket(E.quantile(wcache[k], lv) or 0.0)
                    if bk:
                        n += 1
                        b_ += pts[k]["attn"] < bk
                if n == 0:
                    continue  # nothing would be shown at this level: vacuously safe
                if n < MIN_N or E.clopper_pearson_upper(n, b_, 0.25) > ALPHA:
                    break
                lv_ok = lv
            w = (E.quantile(S, lv_ok) or 0.0) if lv_ok else 0.0
        else:
            w = (E.quantile(S, q) or 0.0) if q > 0 else 0.0
        b = render.safe_bucket(w)
        issued[i] = (w, b)
        if b:
            shown += 1
            br += p["attn"] < b
            mins += b / 60.0
        log.append((b, p["attn"] < b if b else False))
    half = log[len(log) // 2:]
    h_s = sum(1 for b, _ in half if b)
    h_b = sum(1 for b, x in half if b and x)
    h_m = sum(b for b, _ in half if b) / 60.0
    return shown, br, mins, h_s, h_b, h_m


if __name__ == "__main__":
    print("%-12s %-8s %-10s %6s %6s %7s %8s" % ("用户", "预测器", "方法", "承诺", "违约", "违约率", "承诺分钟"))
    for home in sys.argv[1:]:
        conn, cfg, tuned, pts = load(home)
        for p in pts:
            p["_cfg"] = cfg
        name = os.path.basename(os.path.dirname(home.rstrip("/")))
        print("-- %s（%d 个点，%d 次运行）" % (name, len(pts), len({p["run"] for p in pts})))
        for which in ("prior", "learned"):
            for method in os.environ.get("METHODS", "safe_k,aci_shown,aci_cf,src").split(","):
                s, b, m, hs, hb, hm = simulate(pts, which, method, tuned)
                print("%-12s %-8s %-10s 全程 %4d 次 违约 %3.0f%% %5.0f 分钟 | 后半段 %4d 次 违约 %3.0f%% %5.0f 分钟" % (
                    "", "先验" if which == "prior" else "学习后", method, s, 100.0 * b / s if s else 0, m,
                    hs, 100.0 * hb / hs if hs else 0, hm))
