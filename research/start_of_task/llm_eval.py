"""Start-of-task prediction with an LLM's read of the request, calibrated to each user's own history.
Prequential; conformal leave windows as in models/start_eval.py."""
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
import prompts as P  # noqa: E402
from start_eval import leave_stats, pace  # noqa: E402

J = {}
for path in glob.glob(os.path.join(HERE, "judged", "*.json")):
    for x in json.load(open(path)):
        J[str(x.get("id"))] = x
SIZE_S = {"XS": 15, "S": 60, "M": 240, "L": 900, "XL": 2700}
N_CAL = 3.0  # pseudo-runs behind "the judge's scale is right"


def corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else float("nan")


METHODS = ("current", "llm_minutes", "llm_size", "blend")
tot = {k: [[], 0, 0, 0.0, 0, 0] for k in METHODS}
print("%-11s %4s  %s" % ("历史", "任务", "   ".join("%s: 误差×/相关r/≥3分窗口/违约" % k for k in METHODS)))
for name in P.HISTORIES:
    rows = [r for r in P.load(name) if r["stop"] and ("%s:%d" % (name, r["id"])) in J]
    rows.sort(key=lambda r: r["t"])
    res = {k: [] for k in METHODS}
    done = []
    for r in rows:
        j = J["%s:%d" % (name, r["id"])]
        try:
            m = math.log(max(float(j.get("minutes") or 0) * 60.0, 5.0))
        except (TypeError, ValueError):
            m = math.log(SIZE_S.get(j.get("size"), 240))
        size = j.get("size") or "M"
        past = [x for x in done if x["end"] < r["t"]]
        base = pace(past, r)
        # the judge's scale, learned from your past: shrunk mean of (actual - judged) in log space
        bias = sum(x["_y"] - x["_m"] for x in past) / (len(past) + N_CAL)
        lm = m + bias
        same = [x["_y"] - lm_ for x, lm_ in ((x, x["_m"] + bias) for x in past) if x["_size"] == size]
        ls = lm + (sum(same) / (len(same) + N_CAL) if same else 0.0)
        need = min(r["attn"], r["dur"]) if r["attn"] is not None else r["dur"]
        y, a = math.log(max(r["dur"], 1.0)), math.log(max(need, 1.0))
        for k, p in (("current", base), ("llm_minutes", lm), ("llm_size", ls), ("blend", 0.5 * base + 0.5 * lm)):
            res[k].append((p, y, a))
        r["_y"], r["_m"], r["_size"] = y, m, size
        done.append(r)
    cells = []
    for k in METHODS:
        trip = res[k]
        err = [y - p for p, y, _ in trip]
        s, b, mins, u = leave_stats(trip)
        n = len(trip)
        cells.append("×%.2f/%.2f/%3.0f%%/%3.0f%%" % (math.exp(statistics.median(abs(e) for e in err)),
                                                    corr([p for p, _, _ in trip], [y for _, y, _ in trip]),
                                                    100.0 * u / n, 100.0 * b / s if s else 0))
        T = tot[k]
        T[0] += [(p, y) for p, y, _ in trip]
        T[1] += s
        T[2] += b
        T[3] += mins
        T[4] += u
        T[5] += n
    print("%-11s %4d  %s" % (name, len(rows), "   ".join("%-28s" % c for c in cells)), flush=True)
print()
for k in METHODS:
    pairs, s, b, mins, u, n = tot[k]
    err = [y - p for p, y in pairs]
    print("%-12s 开局误差中位 ×%.2f  σ %.2f  | 开局给出窗口 %2.0f%%，≥3 分钟 %2.0f%%，违约 %.0f%%，平均 %.1f 分钟" % (
        k, math.exp(statistics.median(abs(e) for e in err)), statistics.pstdev(err), 100.0 * s / n, 100.0 * u / n,
        100.0 * b / s if s else 0, mins / s if s else 0))
