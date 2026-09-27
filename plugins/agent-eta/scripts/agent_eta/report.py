"""History summary and accuracy report ("is the ETA any good?").

Two kinds of evidence:
  live      predictions the status line actually showed, checked against what then happened
  backtest  replay every past run: at several points in it, predict using only runs that had
            finished before that moment, and compare with the truth. Works from day one (with
            backfilled history) and compares the estimator with the built-in prior and B0.
"""
import json
import math
import time

from . import estimator as E
from . import render, tuning


def _pct(values, q):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _rate(flags):
    flags = [f for f in flags if f is not None]
    return (sum(1 for f in flags if f) / len(flags)) if flags else None


def _fmt_rate(r):
    return "–" if r is None else "%d%%" % round(r * 100)


def _fmt_factor(errs):
    m = _pct(errs, 0.5)
    return "–" if m is None else "×%.1f" % math.exp(m)


def _log_err(pred, actual):
    if pred is None or actual is None:
        return None
    return abs(math.log(max(pred, 5.0) / max(actual, 5.0)))


def _metrics(rows):
    """rows: dicts with p50, p80, safe, actual_done (None if censored), actual_attn (None if unknown).

    "safe" is judged as displayed: the status line rounds it down to a bucket and never promises
    less than a minute (it says "may need you any moment" instead), so only real promises count.
    """
    done = [r for r in rows if r["actual_done"] is not None]
    promised = [(r["actual_attn"], render.safe_bucket(r["safe"])) for r in rows if r["actual_attn"] is not None]
    promised = [(a, b) for a, b in promised if b]
    return {
        "n": len(rows),
        "hit50": _rate([r["actual_done"] <= r["p50"] for r in done if r["p50"] is not None]),
        "hit80": _rate([r["actual_done"] <= r["p80"] for r in done if r["p80"] is not None]),
        "safe_viol": _rate([a < b for a, b in promised]),
        "safe_n": len(promised),
        "err": [_log_err(r["p50"], r["actual_done"]) for r in done],
    }


def live_rows(conn, since):
    rows = []
    for p in conn.execute(
            "SELECT p.*, r.active_s AS run_active, r.end_reason FROM predictions p JOIN runs r ON r.id = p.run_id"
            " WHERE r.ended_at IS NOT NULL AND p.t > ?", (since,)).fetchall():
        finished = p["end_reason"] == "stop"
        actual_done = (p["run_active"] - p["active_s"]) if finished else None
        att = conn.execute("SELECT active_s FROM attention WHERE run_id=? AND at >= ? ORDER BY at LIMIT 1",
                           (p["run_id"], p["t"])).fetchone()
        if att is not None:
            actual_attn = att["active_s"] - p["active_s"]
        else:
            actual_attn = actual_done
        rows.append({"p50": p["done_p50"], "p80": p["done_p80"], "safe": p["safe_s"],
                     "actual_done": actual_done, "actual_attn": actual_attn})
    return rows


def _rows(points, pick):
    return [dict(pick(p), actual_done=p["done"], actual_attn=p["attn"]) for p in points]


def split_rows(points, cfg, current=None):
    """Replay points -> (current estimator, built-in prior, B0) rows for _metrics / pinball_comparison.
    `current`: per-point summaries to use instead of the raw p["est"][cfg] (the recalibrated reading)."""
    b0 = [dict(p["b0"], actual_done=p["done"], actual_attn=None) for p in points]
    cur = current if current is not None else [p["est"][cfg] for p in points]
    return ([dict(c, actual_done=p["done"], actual_attn=p["attn"]) for p, c in zip(points, cur)],
            _rows(points, lambda p: p["prior"]), b0)


def backtest_rows(conn, since, max_runs=80, per_run=6, safe_q=0.2):
    """Backtest of the estimator as the status line runs it (the recalibration applied prequentially)."""
    cfg = tuning.current_config(conn)
    recal_on = bool(E.tuned_values(conn).get("recal"))
    points = tuning.replay(conn, since, max_runs, per_run, configs=[cfg], safe_q=safe_q, keep=recal_on)
    return split_rows(points, cfg, tuning.prequential_recal(points, cfg, safe_q) if recal_on else None)


def pinball(q, y, p):
    return q * (y - p) if y >= p else (1.0 - q) * (p - y)


def pinball_comparison(learned, prior, b0):
    """Mean of the P50 and P80 pinball losses on points every method can be scored on (the Codex design's
    release criterion: at least 10 % below B0). B0 abstains where the history's tail is unidentified."""
    idx = [i for i, r in enumerate(b0) if r["actual_done"] is not None]
    answered = [i for i in idx if b0[i]["p50"] is not None and b0[i]["p80"] is not None
                and learned[i]["p50"] is not None and learned[i]["p80"] is not None
                and prior[i]["p50"] is not None and prior[i]["p80"] is not None]

    def loss(rows):
        if not answered:
            return None
        return sum((pinball(0.5, rows[i]["actual_done"], rows[i]["p50"])
                    + pinball(0.8, rows[i]["actual_done"], rows[i]["p80"])) / 2.0 for i in answered) / len(answered)

    return {"n": len(answered), "scorable": len(idx), "learned": loss(learned), "prior": loss(prior),
            "b0": loss(b0), "b0_abstain": (1.0 - len(answered) / len(idx)) if idx else None}


def build(conn, lang, days=30, backtest=True, safe_q=0.2):
    zh = lang == "zh"

    def t(a, b):
        return a if zh else b

    now = time.time()
    since = now - days * 86400
    runs = conn.execute("SELECT * FROM runs WHERE ended_at IS NOT NULL AND started_at > ?", (since,)).fetchall()
    out = []
    out.append(t("Agent ETA 报告（最近 %d 天）" % days, "Agent ETA report (last %d days)" % days))
    out.append("=" * 44)
    if not runs:
        out.append(t("还没有任何记录。运行 /agent-eta:setup 导入历史，或正常使用一段时间。",
                     "No history yet. Run /agent-eta:setup to import transcripts, or just keep working."))
        return "\n".join(out)
    done = [r for r in runs if r["end_reason"] == "stop"]
    live = sum(1 for r in runs if r["source"] == "live")
    act = [r["active_s"] for r in done if r["active_s"] is not None]
    out.append(t("运行次数      %d（实时 %d · 导入 %d）· 正常结束 %d · 中断/未完成 %d",
                 "Runs          %d (live %d · imported %d) · finished %d · interrupted/other %d")
               % (len(runs), live, len(runs) - live, len(done), len(runs) - len(done)))
    if act:
        out.append(t("单次耗时      中位 %s · P80 %s · 最长 %s", "Duration      median %s · P80 %s · max %s")
                   % (render.dur(_pct(act, .5)), render.dur(_pct(act, .8)), render.dur(max(act))))
    ttas = [r["first_attn_at"] - r["started_at"] for r in done if r["first_attn_at"]]
    if ttas:
        out.append(t("到需要你为止  中位 %s（这段时间你本可以离开）",
                     "Until needed  median %s (time you could have walked away)") % render.dur(_pct(ttas, .5)))
    waits = sum((r["human_wait_s"] or 0) for r in runs)
    perms = sum((r["n_perm"] or 0) for r in runs)
    if perms or waits:
        out.append(t("等你处理      %d 次授权请求，共让 Claude 等了 %s",
                     "Waiting on you %d permission prompts, Claude waited %s in total")
                   % (perms, render.dur(waits)))

    out.append("")
    out.append(t("按任务类型", "By task type"))
    cats = {}
    for r in done:
        cats.setdefault(r["category"] or "other", []).append(r["active_s"] or 0)
    for cat, vals in sorted(cats.items(), key=lambda kv: -len(kv[1]))[:8]:
        out.append("  %-10s %3d × %s %s" % (cat, len(vals), t("中位", "median"), render.dur(_pct(vals, .5))))

    repos = {}
    for r in done:
        repos.setdefault(r["repo"] or "?", []).append(r["active_s"] or 0)
    if len(repos) > 1:
        out.append("")
        out.append(t("按仓库", "By repository"))
        for repo, vals in sorted(repos.items(), key=lambda kv: -len(kv[1]))[:6]:
            out.append("  %-34s %3d × %s %s" % (repo[-34:], len(vals), t("中位", "median"), render.dur(_pct(vals, .5))))

    cmds = conn.execute(
        "SELECT cmd_key, COUNT(*) AS n, SUM(duration_ms)/1000.0 AS total FROM tool_calls WHERE kind='shell'"
        " AND duration_ms IS NOT NULL AND started_at > ? AND cmd_key IS NOT NULL GROUP BY cmd_key HAVING n >= 2"
        " ORDER BY total DESC LIMIT 6", (since,)).fetchall()
    if cmds:
        out.append("")
        out.append(t("最耗时的命令", "Slowest commands (total time)"))
        for c in cmds:
            durs = [x[0] / 1000.0 for x in conn.execute(
                "SELECT duration_ms FROM tool_calls WHERE cmd_key=? AND duration_ms IS NOT NULL AND started_at > ?",
                (c["cmd_key"], since)).fetchall()]
            out.append("  %-28s %3d × %s %-6s %s %s" % (c["cmd_key"][:28], c["n"], t("中位", "median"),
                                                     render.dur(_pct(durs, .5)), t("合计", "total"),
                                                     render.dur(c["total"])))

    def metric_lines(title, m, note=None):
        out.append("")
        out.append(title + ("  " + note if note else ""))
        if not m["n"]:
            out.append(t("  （暂无数据）", "  (no data yet)"))
            return
        out.append(t("  P50 命中率    %s（理想 50%%）", "  P50 hit rate   %s (ideal 50%%)") % _fmt_rate(m["hit50"]))
        out.append(t("  P80 覆盖率    %s（理想 80%%）", "  P80 coverage   %s (ideal 80%%)") % _fmt_rate(m["hit80"]))
        out.append(t("  “可离开”违约  %s（理想 ≤%d%%；%d 次给出 ≥1 分钟的承诺）",
                     "  'safe' misses  %s (ideal <=%d%%; %d promises of >=1 min)")
                   % (_fmt_rate(m["safe_viol"]), round(safe_q * 100), m["safe_n"]))
        out.append(t("  典型误差      %s（P50 与实际的倍数差，中位）", "  typical error  %s (median ratio P50 vs actual)")
                   % _fmt_factor(m["err"]))

    lm = _metrics(live_rows(conn, since))
    metric_lines(t("实时预测准确度（状态栏当时显示的）", "Live accuracy (what the status line showed)"), lm,
                 t("%d 个预测点" % lm["n"], "%d predictions" % lm["n"]))
    from . import live
    ls = live.ledger_stats(conn, now, {"safe_quantile": safe_q}, since)
    out.append("")
    out.append(t("“可离开”窗口账本（状态栏真实给出的承诺，固定到期，撤销也计分）",
                 "Leave-window ledger (promises actually shown; fixed expiry, revoked ones still scored)"))
    if not ls["n"]:
        out.append(t("  （还没有已到期的窗口）", "  (no expired windows yet)"))
    else:
        out.append(t("  已发 %d 个 · 守住 %d · 违约 %d · 结果未知 %d（按不利计）",
                     "  issued %d · held %d · breached %d · unknown %d (counted as adverse)")
                   % (ls["n"], ls["held"], ls["breach"], ls["unknown"]))
        rate = (ls["breach"] + ls["unknown"]) / float(ls["n"])
        verdict = t("已验证 ✓", "VALIDATED ✓") if ls["validated"] else t(
            "尚未验证（需要 ≥%d 个窗口且上界 ≤%d%%）" % (live.LEDGER_MIN_TRIALS, round(safe_q * 100)),
            "not validated yet (needs >=%d windows and upper bound <=%d%%)" % (live.LEDGER_MIN_TRIALS, round(safe_q * 100)))
        out.append(t("  不利率 %s，95%% 单侧上界 %s → %s", "  adverse rate %s, one-sided 95%% upper bound %s -> %s")
                   % (_fmt_rate(rate), _fmt_rate(ls["upper"]), verdict))
    if backtest:
        k0 = E.tuned_k0(conn)
        learned, prior, b0 = backtest_rows(conn, since, safe_q=safe_q)
        ml, mp = _metrics(learned), _metrics(prior)
        metric_lines(t("回测：当前估计器（只用当时已有的数据）", "Backtest: current estimator (only data available at the time)"),
                     ml, t("%d 个预测点" % ml["n"], "%d predictions" % ml["n"]))
        metric_lines(t("回测：仅内置先验（对照组）", "Backtest: built-in prior only (baseline)"), mp)
        _k0_lines(out, t, conn, k0)
        pc = pinball_comparison(learned, prior, b0)
        out.append("")
        out.append(t("回测：对照 Codex 设计的 B0 基线（历史总时长的条件生存，不用轨迹特征）",
                     "Backtest vs the Codex design's B0 baseline (conditional survival of run length, no trajectory)"))
        if not pc["n"]:
            out.append(t("  （B0 在所有点上都弃权：历史尾部不足）", "  (B0 abstains everywhere: history tail unidentified)"))
        else:
            gain = 1.0 - pc["learned"] / pc["b0"] if pc["b0"] else 0.0
            out.append(t("  平均 pinball loss  当前 %.1f · 仅内置先验 %.1f · B0 %.1f（%d 个共同可评估点）",
                         "  mean pinball loss  current %.1f · built-in prior %.1f · B0 %.1f (%d common points)")
                       % (pc["learned"], pc["prior"], pc["b0"], pc["n"]))
            out.append(t("  当前比 B0 %s %d%%（Codex 的发布门槛：至少低 10%%）",
                         "  current vs B0: %s %d%% (Codex release criterion: at least 10%% lower)")
                       % (t("低", "lower by") if gain >= 0 else t("高", "higher by"), round(abs(gain) * 100)))
            out.append(t("  B0 弃权率 %s（尾部无法识别时不给数）", "  B0 abstains on %s of points (unidentified tail)")
                       % _fmt_rate(pc["b0_abstain"]))
    return "\n".join(out)


def _k0_label(k):
    return "∞" if math.isinf(k) else "%g" % k


def _k0_lines(out, t, conn, k0):
    """How strongly the estimator trusts similar past runs over its calibrated prior, and why."""
    rec = tuning.record(conn)
    out.append("")
    out.append(t("收缩强度 K0（越大越依赖校准先验，∞ = 只用校准先验）",
                 "Shrinkage K0 (larger = lean on the calibrated prior; ∞ = calibrated prior only)"))
    if rec is None or rec["value"] is None:
        out.append(t("  当前 K0=%s（默认值；累计 %d 次运行后会按你的历史自动选择）",
                     "  current K0=%s (default; chosen from your history once %d runs have finished)")
                   % (_k0_label(k0), tuning.FIRST_TUNE_RUNS))
        out.append(t("  在那之前状态栏不给“可离开”承诺（显示“离开：待验证”）：未调参时的承诺不可靠",
                     "  Until then the status line makes no leave promise ('leave: unverified'): untuned promises are unreliable"))
        return
    out.append(t("  当前 K0=%s：%s 回测 %d 次运行（%d 个点）后自动选择；运行数再增长 10%% 会在后台重选",
                 "  current K0=%s: chosen %s by replaying %d runs (%d points); re-chosen in the background as history grows 10%%")
               % (_k0_label(k0), time.strftime("%m-%d %H:%M", time.localtime(rec["at"])), rec["n_runs"] or 0,
                  rec["n_points"] or 0))
    try:
        losses = json.loads(rec["detail"] or "{}")
    except ValueError:
        losses = {}
    if losses:
        out.append(t("  各候选的对数 pinball loss（越低越好）  ", "  log pinball loss per candidate (lower is better)  ")
                   + " · ".join("%s%s %.3f" % ("▶" if key == _k0_label(k0).replace("∞", "inf") else "",
                                               key.replace("inf", "∞"), v)
                                for key, v in losses.items() if v is not None))
    hl = conn.execute("SELECT * FROM tuning WHERE key='half_life' AND value IS NOT NULL").fetchone()
    if hl is not None:
        try:
            by_hl = json.loads(hl["detail"] or "{}")
        except ValueError:
            by_hl = {}
        out.append(t("  近期权重半衰期 %g 天（越短越看重最近的运行）：各候选的最佳损失  ",
                     "  recency half-life %g days (shorter = recent runs count more): best loss per candidate  ")
                   % hl["value"] + " · ".join("%s%s %.3f" % ("▶" if float(k) == hl["value"] else "", k, v)
                                              for k, v in by_hl.items() if v is not None))
    rc = conn.execute("SELECT * FROM tuning WHERE key='recal' AND value IS NOT NULL").fetchone()
    if rc is not None:
        try:
            info = json.loads(rc["detail"] or "{}")
        except ValueError:
            info = {}
        raw, rec = info.get("loss_raw"), info.get("loss_recal")
        if raw is not None and rec is not None:
            out.append(t("  分位数重校准：%s（按时间前瞻回放的损失 原样 %.3f → 重校准 %.3f；开 = 按你的实际结果修正系统性偏长/偏短）",
                         "  quantile recalibration: %s (prequential loss raw %.3f -> recalibrated %.3f; on = systematic"
                         " over/under-estimation corrected from your outcomes)")
                       % (t("开", "on") if rc["value"] else t("关", "off"), raw, rec))
    safe = conn.execute("SELECT * FROM tuning WHERE key='safe_k' AND value IS NOT NULL").fetchone()
    if safe is None:
        return
    try:
        info = json.loads(safe["detail"] or "{}")
    except ValueError:
        info = {}
    n, br = (info.get("replay") or {}).get("%g" % safe["value"], (0, 0))
    ln, lbr = info.get("ledger") or (0, 0)
    out.append(t("  “可离开”保守度 SAFE_K=%g（越大越保守）：回测中给出 %d 次承诺、违约 %d 次；最近 %d 天真实窗口 %d 个、违约 %d 个",
                 "  leave-window caution SAFE_K=%g (larger = more cautious): %d replayed promises, %d breached;"
                 " real windows in the last %d days: %d, %d breached")
               % (safe["value"], n, br, tuning.LEDGER_DAYS, ln, lbr))

