"""Per-model turn statistics across all imported histories (each home = one user or one dataset)."""
import math
import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E  # noqa: E402

from common import HOMES  # noqa: E402


def short(m):
    m = (m or "?").replace("moonshotai/", "").replace("minimax/", "")
    for cut in ("-2026", "-2025"):
        m = m.split(cut)[0]
    return m.replace("claude-", "")


rows = []
for label, (home, kind) in HOMES.items():
    c = sqlite3.connect(os.path.join(home, "eta.db"))
    c.row_factory = sqlite3.Row
    for r in c.execute("SELECT r.*, (SELECT COUNT(*) FROM attention a WHERE a.run_id=r.id) AS n_att FROM runs r"
                       " WHERE end_reason='stop' AND active_s > 1"):
        med = E.prior_median({"category": r["category"] or "other", "effort": r["effort"], "prompt_len": r["prompt_len"] or 0})
        steps = conn_steps = r["n_tools"] or 0
        rows.append({"src": label, "kind": kind, "model": short(r["model"]), "dur": r["active_s"],
                     "lr": math.log(r["active_s"] / med), "tools": steps, "cat": r["category"] or "other",
                     "att": r["n_att"], "effort": r["effort"]})


def summarize(rs):
    d = sorted(x["dur"] for x in rs)
    lr = [x["lr"] for x in rs]
    mu = statistics.mean(lr)
    # spread after removing each source's and category's own pace (what calibration can learn)
    by = defaultdict(list)
    for x in rs:
        by[(x["src"], x["cat"])].append(x["lr"])
    resid = [v - statistics.mean(g) for g in by.values() if len(g) >= 3 for v in g]
    per_step = [x["dur"] / (x["tools"] + 1) for x in rs]
    return (len(rs), statistics.median(d), d[int(0.8 * (len(d) - 1))], math.exp(mu),
            math.sqrt(statistics.pvariance(resid)) if len(resid) > 5 else float("nan"),
            statistics.median(x["tools"] for x in rs), statistics.median(per_step),
            sum(1 for x in rs if x["att"]) / len(rs))


print("%-22s %-10s %5s %7s %7s %8s %7s %6s %8s %6s" % ("模型", "来源", "轮数", "中位", "P80", "节奏×先验", "残差σ", "工具数", "秒/步", "问你%"))
by_model = defaultdict(list)
for x in rows:
    by_model[(x["model"], x["kind"])].append(x)
for (m, kind), rs in sorted(by_model.items(), key=lambda kv: (kv[0][1], -len(kv[1]))):
    if len(rs) < 8:
        continue
    n, med, p80, pace, sig, tools, per, att = summarize(rs)
    print("%-22s %-10s %5d %6.0fs %6.0fs %8.2f %7.2f %6.0f %7.1fs %5.0f%%" % (m, kind, n, med, p80, pace, sig, tools, per, 100 * att))

print("\n同一用户内部（wisp，一台机器、同一个人）按模型：")
for m in sorted({x["model"] for x in rows if x["src"] == "wisp"}):
    rs = [x for x in rows if x["src"] == "wisp" and x["model"] == m]
    if len(rs) >= 5:
        n, med, p80, pace, sig, tools, per, att = summarize(rs)
        print("  %-18s %4d 轮  中位 %5.0fs  P80 %5.0fs  节奏×%.2f  工具 %2.0f 次  %.1f 秒/步" % (m, n, med, p80, pace, tools, per))
print("\n按来源（每个来源内部的模型）：")
for src in HOMES:
    ms = defaultdict(int)
    for x in rows:
        if x["src"] == src:
            ms[x["model"]] += 1
    print("  %-14s %s" % (src, dict(sorted(ms.items(), key=lambda kv: -kv[1]))))
