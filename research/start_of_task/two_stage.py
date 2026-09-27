"""A two-stage plan given at the start of every task, from your own history only (no model, no prompt):
  "wait tau seconds; if Claude is still working then, you can leave for W minutes".
W = 20 % quantile of (time-to-need - tau) among your earlier tasks that were still running at tau
(prequential, only tasks finished before this one started). Reports, per checkpoint tau: how many tasks
got past it, the window W, and how often Claude needed you inside W anyway."""
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402

TAUS = (60, 90, 120, 180)
MIN_PAST = 10


def need_of(r):
    return min(r["attn"], r["dur"]) if r["attn"] is not None else r["dur"]


tot = {t: [0, 0, 0, 0.0, 0] for t in TAUS}  # tasks, past tau, promised, minutes, breached
print("%-11s %4s  %s" % ("历史", "任务", "   ".join("τ=%3ds: 过检查点/窗口中位/违约" % t for t in TAUS)))
for name in P.HISTORIES:
    rows = sorted([r for r in P.load(name) if r["stop"]], key=lambda r: r["t"])
    cells = []
    for tau in TAUS:
        n = past_tau = prom = br = 0
        wins = []
        for i, r in enumerate(rows):
            past = [need_of(x) - tau for x in rows[:i] if x["end"] < r["t"] and need_of(x) > tau]
            n += 1
            if need_of(r) <= tau:
                continue  # done (or needed you) before the checkpoint: the plan said "wait", and that was right
            past_tau += 1
            if len(past) < MIN_PAST:
                continue
            past.sort()
            w = past[int(0.2 * (len(past) - 1))]
            if w < 60:
                continue
            b = 60 * math.floor(w / 60)
            prom += 1
            wins.append(b / 60.0)
            br += (need_of(r) - tau) < b
        T = tot[tau]
        T[0] += n
        T[1] += past_tau
        T[2] += prom
        T[3] += sum(wins)
        T[4] += br
        cells.append("%3.0f%% / %4.1f分 / %3.0f%%" % (100.0 * past_tau / n, statistics.median(wins) if wins else 0,
                                                    100.0 * br / prom if prom else 0))
    print("%-11s %4d  %s" % (name, len(rows), "   ".join("%-26s" % c for c in cells)), flush=True)
print()
for tau in TAUS:
    n, pt, prom, mins, br = tot[tau]
    print("检查点 %3d 秒：%2.0f%% 的任务跑过检查点；其中 %2.0f%% 拿到窗口，平均 %.1f 分钟，违约 %.0f%%；"
          "折算到全部任务，有 %.0f%% 在开局就知道“若 %d 秒后还在跑，可离开 %.1f 分钟”" % (
              tau, 100.0 * pt / n, 100.0 * prom / pt if pt else 0, mins / prom if prom else 0,
              100.0 * br / prom if prom else 0, 100.0 * prom / n, tau, mins / prom if prom else 0))
