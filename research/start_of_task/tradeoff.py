"""How long could the leave window be at each confidence level, and how often would it be broken?
Replay points with each history's tuned settings; the window is the q-quantile of time-to-attention
(q = 1 - confidence), rounded down to the status line's buckets; below one minute nothing is shown."""
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5)
tot = {q: [0, 0, 0.0, 0] for q in LEVELS}
free = []
print("%-12s %5s  %s" % ("历史", "点数", "   ".join("把握%d%%: 给出率/中位时长/违约" % round(100 * (1 - q)) for q in LEVELS)))
for home in sys.argv[1:]:
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    cfg = tuning.current_config(conn)
    tuned = E.tuned_values(conn)
    recal = E.tuned_recal(conn) if tuned.get("recal") else None
    pts = [p for p in tuning.replay(conn, max_runs=100000, configs=[cfg], keep=True, with_baselines=False,
                                    pool_limit=100000) if p["attn"] is not None]
    cells = []
    for q in LEVELS:
        shown = br = 0
        lens = []
        for p in pts:
            S = p["est"][cfg]["est"]["S_attn"]
            b = render.safe_bucket(E.quantile(S, E.recal_level(recal, "attn", q)))
            if b:
                shown += 1
                br += p["attn"] < b
                lens.append(b / 60.0)
        cells.append("%3.0f%% / %4.1f 分 / %3.0f%%" % (100.0 * shown / len(pts), statistics.median(lens) if lens else 0,
                                                     100.0 * br / shown if shown else 0))
        t = tot[q]
        t[0] += shown
        t[1] += br
        t[2] += sum(lens)
        t[3] += len(pts)
    free += [p["attn"] / 60.0 for p in pts]
    name = home.rstrip("/").split("/")[-2]
    print("%-12s %5d  %s" % (name[:12], len(pts), "   ".join(cells)), flush=True)
print("\n合计：")
for q in LEVELS:
    n, br, mins, npts = tot[q]
    print("  把握 %d%%：%3.0f%% 的时刻给出承诺，平均 %.1f 分钟，违约 %.0f%%" % (
        round(100 * (1 - q)), 100.0 * n / npts, mins / n if n else 0, 100.0 * br / n if n else 0))
fs = sorted(free)
print("实际离下次需要你还有多久（全部时刻）：中位 %.1f 分，P25 %.1f 分，P75 %.1f 分；≥3 分钟的时刻 %.0f%%，≥5 分钟 %.0f%%" % (
    statistics.median(fs), fs[len(fs) // 4], fs[3 * len(fs) // 4], 100.0 * sum(x >= 3 for x in fs) / len(fs),
    100.0 * sum(x >= 5 for x in fs) / len(fs)))
