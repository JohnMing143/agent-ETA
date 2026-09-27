"""Text rendering: status line, detailed status, durations, i18n."""
import math
import os
import time

STRINGS = {
    "zh": {
        "done": "完成", "safe": "可离开", "until": "至", "soon": "随时可能需要你", "needs_you": "需要你",
        "withheld": "离开：待验证",
        "waited": "已等", "conf": "置信", "levels": {"low": "低", "medium": "中", "high": "高"},
        "learning": "学习中", "last": "上次", "pred": "预测", "learned": "已学习 {n} 次",
        "prior_tag": "先验", "insufficient": "历史不足",
        "background": "等后台", "running": "运行中",
        "attn": {"permission": "权限确认", "question": "回答问题", "plan_approval": "审批计划",
                 "elicitation": "MCP 输入"},
        "phase": {"start": "刚开始", "explore": "阅读/搜索", "implement": "修改中", "fixing": "修复失败",
                  "verify": "验证中"},
        "reasons": {
            "history": "参考 {n} 次相似历史（有效样本 {ess:.1f}，同仓库 {same} 次）",
            "prior": "历史数据不足（已有 {n} 次），暂用先验估计",
            "pace": "按你 {n} 次历史校准：这类任务约为默认估计的 ×{x:.1f}",
            "phase_verify": "已进入验证阶段，通常接近尾声",
            "fails": "验证失败 {n} 次，修复循环会拉长时间",
            "inflight": "正在运行 {what}（已 {running}{typical}）",
            "typical": "，历史中位 {t}",
            "plan": "任务清单 {done}/{total}",
            "plan_grew": "计划扩张 {first}→{total} 步",
            "agents": "{n} 个子代理在并行运行",
            "background": "在等 {n} 个后台任务，完成后会自动继续",
            "permission": "当前权限模式下约每 {every} 次工具调用需要一次授权",
            "trend_up": "预计比刚才更久", "trend_down": "预计比刚才更快",
            "new_fails": "新增 {n} 次失败",
        },
        "detail": {
            "title": "Agent ETA", "finish": "完成", "need_you": "需要你", "confidence": "置信度",
            "current": "当前", "stage": "阶段", "why": "依据", "leave_until": "你可以离开到",
            "unlikely_for": "至少 {x} 内不太可能", "waiting_now": "现在就需要你：{what}（已等 {w}）",
            "stats": "工具 {tools} 次 · 编辑 {edits} · 失败 {fails}", "none": "没有正在运行的任务。",
            "method": {"learned": "历史学习", "blended": "历史+校准先验", "calibrated": "校准先验", "prior": "先验"},
            "session": "会话", "elapsed": "已运行",
            "window_fixed": "（窗口已固定，不会自动顺延）", "validated": "已通过真实窗口验证",
            "unvalidated": "尚未验证",
        },
    },
    "en": {
        "done": "done", "safe": "safe", "until": "until", "soon": "may need you any moment", "needs_you": "NEEDS YOU",
        "withheld": "leave: unverified",
        "waited": "waiting", "conf": "conf", "levels": {"low": "low", "medium": "med", "high": "high"},
        "learning": "learning", "last": "last", "pred": "pred", "learned": "{n} runs learned",
        "prior_tag": "prior", "insufficient": "not enough history",
        "background": "bg wait", "running": "running",
        "attn": {"permission": "permission", "question": "question", "plan_approval": "plan approval",
                 "elicitation": "MCP input"},
        "phase": {"start": "starting", "explore": "exploring", "implement": "editing", "fixing": "fixing failures",
                  "verify": "verifying"},
        "reasons": {
            "history": "{n} similar past runs (effective {ess:.1f}, {same} in this repo)",
            "prior": "not enough history yet ({n} runs), using priors",
            "pace": "calibrated on {n} of your runs: this kind of task takes ~x{x:.1f} the default estimate",
            "phase_verify": "verification started, usually near the end",
            "fails": "{n} failed verification(s): fix loops take longer",
            "inflight": "running {what} ({running} so far{typical})",
            "typical": ", typically {t}",
            "plan": "task list {done}/{total}",
            "plan_grew": "plan grew {first} -> {total} steps",
            "agents": "{n} subagent(s) running in parallel",
            "background": "waiting on {n} background task(s); will resume on its own",
            "permission": "in this permission mode ~1 in {every} tool calls asks for approval",
            "trend_up": "estimate went up", "trend_down": "estimate went down",
            "new_fails": "{n} new failure(s)",
        },
        "detail": {
            "title": "Agent ETA", "finish": "Finish", "need_you": "Need you", "confidence": "Confidence",
            "current": "Current", "stage": "Stage", "why": "Why", "leave_until": "You can leave until",
            "unlikely_for": "unlikely for ~{x}", "waiting_now": "needs you NOW: {what} (waiting {w})",
            "stats": "{tools} tool calls · {edits} edits · {fails} failures", "none": "No active runs.",
            "method": {"learned": "learned", "blended": "learned+calibrated prior", "calibrated": "calibrated prior",
                       "prior": "prior"},
            "session": "session", "elapsed": "elapsed",
            "window_fixed": "(fixed window, never silently extended)", "validated": "validated on real windows",
            "unvalidated": "not yet validated",
        },
    },
}

SAFE_BUCKETS = [60, 120, 180, 300, 480, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200, 10800]


def S(lang):
    return STRINGS.get(lang, STRINGS["en"])


class Paint:
    def __init__(self, on=True):
        self.on = on and not os.environ.get("NO_COLOR")

    def __call__(self, code, text):
        return "\033[%sm%s\033[0m" % (code, text) if self.on else text

    def green(self, t):
        return self("32", t)

    def yellow(self, t):
        return self("33", t)

    def red(self, t):
        return self("31", t)

    def dim(self, t):
        return self("2", t)

    def bold(self, t):
        return self("1", t)

    def cyan(self, t):
        return self("36", t)


# ---------------------------------------------------------------- durations

def clock(s):
    s = int(max(0, s or 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, sec) if h else "%d:%02d" % (m, sec)


def dur(s, up=False):
    """Human duration; `up` rounds up (for upper bounds), otherwise down."""
    if s is None:
        return "?"
    s = max(0.0, s)
    rnd = math.ceil if up else math.floor
    if s < 60:
        v = int(rnd(s / 5.0) * 5) if s >= 10 else int(rnd(s))
        return "%ds" % max(v, 1 if up else 0)
    if s < 3600:
        return "%dm" % max(1, int(rnd(s / 60.0)))
    h, m = divmod(int(rnd(s / 60.0)), 60)
    return "%dh%02dm" % (h, m) if m else "%dh" % h


def span(a, b):
    if a is None and b is None:
        return "?"
    if b is None:
        return ">" + dur(a)
    lo, hi = dur(a), dur(b, up=True)
    if lo == hi:
        return lo
    if lo[-1] == hi[-1] and lo[-1] in "sm" and "h" not in lo + hi:
        return "%s–%s" % (lo[:-1], hi)
    return "%s–%s" % (lo, hi)


def safe_bucket(s):
    if s is None or s < SAFE_BUCKETS[0]:
        return None
    best = SAFE_BUCKETS[0]
    for b in SAFE_BUCKETS:
        if b <= s:
            best = b
    return best


def hhmm(ts):
    return time.strftime("%H:%M", time.localtime(ts))


def reason_text(lang, code, params):
    tpl = S(lang)["reasons"].get(code)
    if not tpl:
        return code
    p = dict(params)
    if code == "inflight":
        p["running"] = dur(p.get("running"))
        p["typical"] = S(lang)["reasons"]["typical"].format(t=dur(p["typical"])) if p.get("typical") else ""
    try:
        return tpl.format(**p)
    except (KeyError, ValueError, IndexError):
        return tpl


# ---------------------------------------------------------------- status line

def statusline(state, est, lang, now, safe_q, paint, trend=None):
    L = S(lang)
    sep = paint.dim(" │ ")
    if state["attn"]:
        a = state["attn"]
        what = L["attn"].get(a["kind"], a["kind"])
        detail = a.get("detail") or ""
        text = "⚠ %s: %s%s · %s %s" % (L["needs_you"], what, (" " + detail) if detail else "", L["waited"],
                                        clock(a["waiting_s"]))
        return paint.red(paint.bold(text))
    parts = [paint.bold("⏱ " + clock(state["active_s"]))]
    if state["status"] == "background":
        parts.append(paint.cyan("⏳ %s ×%d" % (L["background"], state["bg_count"] or 1)))
    arrow = {"up": " ↑", "down": " ↓"}.get(trend or "", "")
    if est.get("cold_unknown"):
        parts.append("%s ? (%s)" % (L["done"], L["insufficient"]))
    else:
        parts.append("%s ~%s%s" % (L["done"], span(est["done_p50"], est["done_p80"]),
                                   paint.yellow(arrow) if arrow else ""))
        parts.append(_leave_segment(est, L, now, paint))
    inf = state.get("in_flight")
    if inf and inf["running_s"] >= 3:
        what = inf["cmd_key"] if inf["kind"] == "shell" and inf.get("cmd_key") else inf["tool"]
        parts.append(paint.dim("▶ %s %s" % (what, clock(inf["running_s"]))))
    if state["plan_total"]:
        parts.append(paint.dim("☰ %d/%d" % (state["plan_done"], state["plan_total"])))
    conf = "%s %s" % (L["conf"], L["levels"][est["confidence"]])
    if est["method"] == "prior":
        conf += "·" + L["prior_tag"]
    parts.append(paint.dim(conf))
    return sep.join(parts)


def _leave_segment(est, L, now, paint):
    w = est.get("window")
    if w is not None:  # an issued window: fixed expiry, counting down
        left = w["expires_at"] - now
        txt = "%s ~%s → %s%s" % (L["safe"], dur(left), hhmm(w["expires_at"]), " ✓" if est.get("validated") else "")
        return paint.green(txt) if left >= 300 else paint.yellow(txt)
    if est.get("leave_withheld"):
        return paint.dim(L["withheld"])
    bucket = safe_bucket(est.get("safe_s"))
    if bucket is None:
        return paint.red(L["soon"])
    txt = "%s ~%s → %s" % (L["safe"], dur(bucket), hhmm(now + bucket))
    return paint.green(txt) if bucket >= 300 else paint.yellow(txt)


def idle_line(last, n_learned, lang, paint):
    L = S(lang)
    if n_learned < 5:
        return paint.dim("⏱ ETA %s %d/5" % (L["learning"], n_learned))
    bits = ["⏱ ETA"]
    if last is not None and last["active_s"] is not None:
        txt = "%s %s" % (L["last"], clock(last["active_s"]))
        if last["p50"] is not None:
            ok = last["p80"] is not None and last["actual_from_pred"] <= last["p80"]
            txt += " (%s %s %s)" % (L["pred"], span(last["p50"], last["p80"]), "✓" if ok else "✗")
        bits.append(txt)
    bits.append(L["learned"].format(n=n_learned))
    return paint.dim(" · ".join(bits))


# ---------------------------------------------------------------- detailed view

def detail(state, est, lang, now, safe_q, paint, session_label=None, trend=None):
    L = S(lang)
    D = L["detail"]
    lines = []
    head = "%s · %s %s" % (D["title"], D["session"], session_label or state["session_id"][:8])
    if state.get("preview"):
        head += " · “%s”" % state["preview"][:60]
    lines.append(paint.bold(head))
    prog = est.get("progress")
    if prog is not None:
        k = max(0, min(10, int(round(prog * 10))))
        lines.append("%s%s  ~%d%%   %s %s" % ("█" * k, "░" * (10 - k), int(prog * 100), D["elapsed"],
                                              clock(state["active_s"])))
    lines.append("")
    arrow = {"up": " ↑", "down": " ↓"}.get(trend or "", "")
    lines.append("%-14s ~%s  (P50 %s · P80 %s)%s" % (
        D["finish"], span(est["done_p50"], est["done_p80"]), dur(est["done_p50"]),
        dur(est["done_p80"], up=True), arrow))
    if state["attn"]:
        a = state["attn"]
        what = "%s %s" % (L["attn"].get(a["kind"], a["kind"]), a.get("detail") or "")
        lines.append("%-14s %s" % (D["need_you"], paint.red(D["waiting_now"].format(what=what.strip(),
                                                                                    w=clock(a["waiting_s"])))))
    else:
        w = est.get("window")
        if w is not None:
            need = D["unlikely_for"].format(x=dur(w["expires_at"] - now)) + " " + D["window_fixed"]
        else:
            bucket = safe_bucket(est.get("safe_s"))
            if est.get("leave_withheld"):
                need = paint.dim(L["withheld"])
            else:
                need = paint.red(L["soon"]) if bucket is None else D["unlikely_for"].format(x=dur(bucket))
        lines.append("%-14s %s" % (D["need_you"], need))
    lines.append("%-14s %s · ESS %.1f · %s" % (D["confidence"], L["levels"][est["confidence"]], est["ess"],
                                              D["method"][est["method"]]))
    lines.append("")
    inf = state.get("in_flight")
    if inf:
        lines.append("%-14s ▶ %s  %s" % (D["current"], inf["detail"] or inf["tool"], clock(inf["running_s"])))
    lines.append("%-14s %s · %s" % (D["stage"], L["phase"].get(state["phase"], state["phase"]),
                                    D["stats"].format(tools=state["n_tools"], edits=state["n_edit"],
                                                      fails=state["n_fail"])))
    lines.append(D["why"])
    for code, params in est["reasons"]:
        lines.append("  · " + reason_text(lang, code, params))
    w = est.get("window")
    until = w["expires_at"] if w is not None else (now + safe_bucket(est["safe_s"]) if safe_bucket(est.get("safe_s")) else None)
    if until and not state["attn"] and not est.get("cold_unknown"):
        lines.append("")
        lines.append(paint.green("%s ~%s" % (D["leave_until"], hhmm(until)))
                     + paint.dim("  (%s)" % (D["validated"] if est.get("validated") else D["unvalidated"])))
    return "\n".join(lines)
