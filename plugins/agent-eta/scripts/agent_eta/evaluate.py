"""`eta.py eval`: is the learning paying off, and is more history still worth having?

Scored the way tuning.py scores (log-scale pinball loss of done P50/P80 and attention P20) and always
prequentially: every point is predicted only from runs that had finished before it, the quantile
recalibration included. The tuned settings themselves were chosen on this same history, which makes
the numbers slightly optimistic; the comparison with the built-in prior (no learning at all) is not
affected by that choice.

Uncertainty: a cluster bootstrap over runs (the points of one run are not independent). A gain counts
as shown only when its 95 % interval excludes zero.

  overall   current estimator vs the built-in prior, the calibrated prior alone, and the Codex B0
  history   the same comparison, split by how much history existed at each point
  curve     (optional) the most recent runs predicted from n random earlier runs for growing n; the
            gain of the full history over half of it is the marginal value of more data
"""
import bisect
import math
import random
import time
import unicodedata

from . import estimator as E
from . import render, tuning

BOOT = 2000


def _loss(p, est):
    return tuning.point_loss(p, est) if est is not None else None


def gain(points, rows, base, rng, boot=BOOT):
    """1 - loss(rows) / loss(base) on the points both can score, with a 95 % cluster-bootstrap interval:
    {"gain", "lo", "hi", "runs"} or None."""
    per = {}
    for p, a, b in zip(points, rows, base):
        la, lb = _loss(p, a), _loss(p, b)
        if la is not None and lb is not None:
            acc = per.setdefault(p["run"], [0.0, 0.0])
            acc[0] += la
            acc[1] += lb
    ids = sorted(per)
    if len(ids) < 3:
        return None

    def g(sample):
        lb = sum(per[i][1] for i in sample)
        return 1.0 - sum(per[i][0] for i in sample) / lb if lb > 0 else 0.0

    bs = sorted(g([rng.choice(ids) for _ in ids]) for _ in range(boot))
    return {"gain": g(ids), "lo": bs[int(0.025 * boot)], "hi": bs[int(0.975 * boot) - 1], "runs": len(ids)}


def metrics(points, rows):
    losses = [x for x in (_loss(p, r) for p, r in zip(points, rows)) if x is not None]
    done = [(p, r) for p, r in zip(points, rows) if p["done"] is not None and r is not None and r["p50"] is not None]
    err = sorted(abs(math.log(max(r["p50"], 5.0) / max(p["done"], 5.0))) for p, r in done)
    prom = [(p, render.safe_bucket(r.get("safe"))) for p, r in zip(points, rows) if p["attn"] is not None and r]
    prom = [(p, b) for p, b in prom if b]
    return {
        "loss": sum(losses) / len(losses) if losses else None, "n": len(losses),
        "hit50": sum(p["done"] <= r["p50"] for p, r in done) / len(done) if done else None,
        "cov80": sum(p["done"] <= r["p80"] for p, r in done if r["p80"] is not None) / len(done) if done else None,
        "x_err": math.exp(err[len(err) // 2]) if err else None,
        "promises": len(prom), "breached": sum(p["attn"] < b for p, b in prom),
        "promised_min": sum(b for _, b in prom) / 60.0,
    }


def _history_counts(conn, points):
    ends = sorted(r[0] for r in conn.execute("SELECT ended_at FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL"))
    return [bisect.bisect_left(ends, p["as_of"]) for p in points]


def learning_curve(conn, cfg, safe_q, rng, test_frac=0.3, min_test=15, draws=4, per_run=6):
    """Fixed exam: the most recent runs. History: n random earlier runs (nested prefixes of one shuffle per
    draw, so larger n always contains smaller n). Raw curves with the current K0 / half-life."""
    pool = E.Pool.load(conn, time.time() + 1, limit=100000)
    runs = conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL"
                        " ORDER BY started_at").fetchall()
    runs = [r for r in runs if r["id"] in pool.runs]
    n_test = max(min_test, int(round(test_frac * len(runs))))
    if len(runs) - n_test < 10:
        return None
    test, cut = runs[-n_test:], runs[-n_test]["started_at"]
    train = [r["id"] for r in runs if r["ended_at"] < cut]
    m = len(train)
    sizes = sorted({0, m // 4, m // 2, m})
    safe_k = E.tuned_values(conn).get("safe_k", E.SAFE_K_DEFAULT)
    pts = []
    for r in test:
        snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t",
                             (r["id"],)).fetchall()
        step = max(1, len(snaps) // per_run)
        for s in snaps[::step][:per_run]:
            pts.append({"run": r["id"], "state": E.state_from_snapshot(r, s),
                        "done": None if s["censored"] else s["rem_done_s"],
                        "attn": s["rem_attn_s"] if s["attn_event"] else None})
    rows = {n: [] for n in sizes}  # per draw, aligned with pts: list of (point, summary)
    for _ in range(draws):
        order = train[:]
        rng.shuffle(order)
        for n in sizes:
            sub = E.Pool()
            sub.runs = {i: pool.runs[i] for i in order[:n]}
            for p in pts:
                est = E.estimate(conn, p["state"], pool=sub, with_permission=False, k0=cfg[0], half_life=cfg[1],
                                 recal=False, safe_k=safe_k)
                rows[n].append((p, tuning._summary(est, safe_q)))
    base = rows[0]
    out = {"test_runs": len(test), "train_runs": m, "draws": draws, "sizes": []}
    for n in sizes:
        pp = [p for p, _ in rows[n]]
        out["sizes"].append({"n": n, "gain": gain(pp, [r for _, r in rows[n]], [r for _, r in base], rng)})
    half, full = rows[m // 2], rows[m]
    out["marginal"] = gain([p for p, _ in full], [r for _, r in full], [r for _, r in half], rng)
    return out


def evaluate(conn, safe_q=0.2, curve=False, max_runs=200, seed=1):
    rng = random.Random(seed)
    cfg = tuning.current_config(conn)
    tuned = E.tuned_values(conn)
    recal_on = bool(tuned.get("recal"))
    cal_cfg = (math.inf, cfg[1])
    points = tuning.replay(conn, max_runs=max_runs, configs=[cfg, cal_cfg], safe_q=safe_q, keep=recal_on,
                           pool_limit=100000)
    if not points:
        return {"points": 0}
    current = tuning.prequential_recal(points, cfg, safe_q) if recal_on else [dict(p["est"][cfg]) for p in points]
    if not E.leave_supported(conn):
        current = [dict(c, safe=None) for c in current]  # the gate is closed: nothing would be promised
    rows = {"prior": [p["prior"] for p in points], "calibrated": [p["est"][cal_cfg] for p in points],
            "current": current, "b0": [dict(p["b0"], safe=None) for p in points]}
    res = {"points": len(points), "runs": len({p["run"] for p in points}), "config": cfg, "recal": recal_on,
           "safe_k": tuned.get("safe_k", E.SAFE_K_DEFAULT), "tuned": "k0" in tuned, "leave_ok": E.leave_supported(conn),
           "metrics": {k: metrics(points, v) for k, v in rows.items() if k != "b0"},
           "gain": {k: gain(points, rows[k], rows["prior"], rng) for k in ("calibrated", "current")}}
    # B0 predicts the finish only: compare it on that part alone
    res["metrics"]["b0"] = metrics(points, rows["b0"])
    res["gain"]["b0"] = gain(points, rows["b0"], [dict(r, a20=None) for r in rows["prior"]], rng)
    hist = _history_counts(conn, points)
    cuts = sorted(hist)
    edges = [cuts[int(len(cuts) * f)] for f in (1 / 3.0, 2 / 3.0)]
    groups = []
    for lo, hi in ((0, edges[0]), (edges[0], edges[1]), (edges[1], max(cuts) + 1)):
        idx = [i for i, h in enumerate(hist) if lo <= h < hi]
        if not idx:
            continue
        sel = lambda key: [rows[key][i] for i in idx]  # noqa: E731
        groups.append({"from": lo, "to": hi - 1, "points": len(idx),
                       "gain": gain([points[i] for i in idx], sel("current"), sel("prior"), rng)})
    res["history"] = groups
    if curve:
        res["curve"] = learning_curve(conn, cfg, safe_q, rng)
    return res


def to_json(res):
    """JSON-safe copy (infinite K0 as "inf")."""
    import json

    def clean(o):
        if isinstance(o, float) and math.isinf(o):
            return "inf"
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return json.dumps(clean(res), ensure_ascii=False, indent=1)


def _pad(text, width):
    """Left-justify by display width (CJK characters take two columns)."""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(0, width - w)


def _pct(x):
    return "%+.1f%%" % (100 * x)


def _gain_text(g, t):
    if g is None:
        return t("数据不足", "not enough data")
    return "%s  [95%% %s %s, %s]" % (_pct(g["gain"]), t("区间", "interval"), _pct(g["lo"]), _pct(g["hi"]))


def _verdict(g, t):
    if g is None:
        return t("数据不足，先用一段时间", "not enough data yet")
    if g["lo"] > 0:
        return t("已证实：学习让预测更准", "shown: learning makes the estimates better")
    if g["hi"] < 0:
        return t("学习让预测变差：检查历史里的异常运行，或运行 eta.py tune 重新调参",
                 "learning makes the estimates worse: look for odd runs in the history, or re-run eta.py tune")
    return t("尚未证实：区间包含 0（运行次数还不够，或者剩下的误差主要是任务本身的随机性）",
             "not shown yet: the interval includes 0 (too few runs, or the remaining error is mostly the tasks' own randomness)")


def describe(res, lang="zh"):
    def t(zh, en):
        return zh if lang == "zh" else en

    if not res.get("points"):
        return t("还没有可评估的历史（需要已结束的运行）。", "No finished runs to evaluate yet.")
    k0, hl = res["config"]
    out = [t("Agent ETA 学习效果评估", "Agent ETA learning evaluation"), "=" * 44,
           t("按时间前瞻回放 %d 次运行、%d 个预测点：每个点只用当时已结束的运行（重校准也一样）。",
             "Prequential replay of %d runs, %d points: each point uses only runs finished before it (recalibration too).")
           % (res["runs"], res["points"]),
           t("当前设置  K0=%s · 半衰期 %g 天 · 重校准%s · SAFE_K=%g · “可离开”门槛%s%s",
             "Current settings  K0=%s · half-life %g days · recalibration %s · SAFE_K=%g · leave gate %s%s")
           % ("∞" if math.isinf(k0) else "%g" % k0, hl, t("开", " on") if res["recal"] else t("关", " off"),
              res["safe_k"], t("开放", "open") if res.get("leave_ok") else t("未开放", "closed"),
              "" if res["tuned"] else t("（默认值，尚未调参）", " (defaults, not tuned yet)")),
           ""]
    g = res["gain"]["current"]
    out.append(t("结论  %s", "Verdict  %s") % _verdict(g, t))
    out.append(t("  学习后相对内置先验（完全不学习）  %s", "  learned vs built-in prior (no learning)  %s") % _gain_text(g, t))
    out.append(t("  其中只靠节奏校准（K0=∞）         %s", "  of which pace calibration alone (K0=∞)  %s")
               % _gain_text(res["gain"]["calibrated"], t))
    out.append(t("  Codex 的 B0 基线（只比完成时间）   %s", "  Codex B0 baseline (finish time only)    %s")
               % _gain_text(res["gain"]["b0"], t))
    out.append("")
    out.append("  " + _pad("", 16) + t("  损失   P50命中  P80覆盖  典型误差  “可离开”承诺",
                                        "  loss   P50 hit  P80 cov  typ. err  leave promises"))
    names = (("prior", t("内置先验", "built-in prior")), ("calibrated", t("只用校准", "calibrated only")),
             ("current", t("当前（学习后）", "current (learned)")), ("b0", "B0"))
    for key, name in names:
        m = dict(res["metrics"][key])
        if key == "b0":
            m["loss"] = None  # finish-time part only: not comparable with the rows above
        f = lambda x, fmt: "–" if x is None else fmt % x  # noqa: E731
        prom = (t("%d 次，违约 %d，共 %.0f 分钟", "%d, %d breached, %.0f min total") % (m["promises"], m["breached"], m["promised_min"])
                if key != "b0" else "–")
        out.append("  %s %6s   %6s   %6s   %7s   %s" % (
            _pad(name, 16), f(m["loss"], "%.3f"), f(m["hit50"] and 100 * m["hit50"], "%.0f%%"),
            f(m["cov80"] and 100 * m["cov80"], "%.0f%%"), f(m["x_err"], "×%.2f"), prom))
    out.append(t("  （理想：P50 命中约 50%%、P80 覆盖约 80%%；“可离开”违约率不超过 %d%%）",
                 "  (ideal: P50 hit ~50%%, P80 coverage ~80%%; leave promises breached at most %d%% of the time)")
               % 20)
    out.append("")
    out.append(t("随历史增长的进步（按每个点当时已有的历史次数分组，相对内置先验）",
                 "Progress as history grew (grouped by runs finished before each point; vs the built-in prior)"))
    for grp in res["history"]:
        out.append(t("  历史 %3d-%-3d 次  %s", "  history %3d-%-3d runs  %s") % (grp["from"], grp["to"], _gain_text(grp["gain"], t)))
    c = res.get("curve")
    if "curve" in res:
        out.append("")
        if c is None:
            out.append(t("学习曲线：历史太少，至少需要约 25 次已结束的运行。", "Learning curve: needs about 25 finished runs."))
        else:
            out.append(t("学习曲线（考题固定为最近 %d 次运行；历史从更早的 %d 次里随机取 n 次，每个 n 抽 %d 次；当前 K0/半衰期，不含重校准）",
                         "Learning curve (exam: the latest %d runs; history: n random runs of the %d before them, %d draws each;"
                         " current K0 / half-life, no recalibration)") % (c["test_runs"], c["train_runs"], c["draws"]))
            for row in c["sizes"]:
                out.append("  n=%-4d %s" % (row["n"], _gain_text(row["gain"], t) if row["n"] else t("0（内置先验）", "0 (built-in prior)")))
            mg = c["marginal"]
            out.append(t("  后一半历史带来的额外提升  %s", "  extra gain from the second half of the history  %s") % _gain_text(mg, t))
            if mg is None:
                pass
            elif mg["lo"] > 0:
                out.append(t("  → 边际收益仍然显著：继续积累历史值得", "  -> the marginal value is still clear: more history pays"))
            elif mg["hi"] < 0:
                out.append(t("  → 旧历史在拖后腿：习惯变了，更短的半衰期可能更好（eta.py tune 会自动尝试）",
                             "  -> older history hurts: habits changed; a shorter half-life may help (eta.py tune tries it)"))
            else:
                out.append(t("  → 边际收益尚不明显：单靠更多历史，短期内难有可测的提升",
                             "  -> no clear marginal value: more history alone is unlikely to show a measurable gain soon"))
    return "\n".join(out)
