"""How much do Claude's first moves reveal? At elapsed 0 / 15 / 30 / 60 s into each task (the last snapshot at
or before that moment), predict the remaining time with the plugin's tuned estimator and give a conformal
leave window (q20 of earlier residuals of log time-to-need vs the P50, per checkpoint). Prequential."""
import math
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, store, tuning  # noqa: E402


def history_name(home):
    """A label for a history directory: the parent's name for .../<name>/home, else its own name."""
    p = os.path.abspath(home.rstrip("/"))
    base = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if base == "home" else base

CHECKS = (0, 15, 30, 60)
tot = {c: [[], 0, 0, 0.0, 0, 0, 0] for c in CHECKS}
for home in sys.argv[1:]:
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    k0, hl = tuning.current_config(conn)
    tuned = E.tuned_values(conn)
    recal = E.tuned_recal(conn) if tuned.get("recal") else False
    pool = E.Pool.load(conn, 1e12, limit=100000)
    runs = conn.execute("SELECT * FROM runs WHERE end_reason='stop' AND active_s > 0 ORDER BY started_at").fetchall()
    per = {c: [] for c in CHECKS}  # (pred log remaining, log remaining-to-need, run end time, start time)
    for r in runs:
        snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY active_s",
                             (r["id"],)).fetchall()
        if not snaps:
            continue
        need_total = (r["first_attn_at"] - r["started_at"]) if r["first_attn_at"] else r["active_s"]
        need_total = min(need_total, r["active_s"])
        for c in CHECKS:
            if r["active_s"] <= c:
                continue  # already over: nothing to predict at this checkpoint
            s = [x for x in snaps if x["active_s"] <= c + 1e-6][-1:] or snaps[:1]
            st = E.state_from_snapshot(r, s[0])
            st["active_s"] = float(c)
            st["as_of"] = r["started_at"] + c
            est = E.estimate(conn, st, pool=pool, with_permission=False, k0=k0, half_life=hl, recal=recal,
                             safe_k=6.0)
            rem_need = max(need_total - c, 1.0)
            per[c].append((math.log(max(est["done_p50"] or 1.0, 1.0)), math.log(rem_need),
                           math.log(max(r["active_s"] - c, 1.0)), r["ended_at"], st["as_of"]))
    name = history_name(home)[:12]
    line = "%-12s" % name
    for c in CHECKS:
        rows = sorted(per[c], key=lambda x: x[4])
        shown = br = useful = 0
        mins = 0.0
        for i, (p, a, y, end, t) in enumerate(rows):
            past = sorted(x[1] - x[0] for x in rows[:i] if x[3] < t)
            if len(past) < 10:
                continue
            w = math.exp(p + past[int(0.2 * (len(past) - 1))])
            if w >= 60:
                b = 60 * math.floor(w / 60) if w < 600 else 300 * math.floor(w / 300)
                shown += 1
                br += a < math.log(b)
                mins += b / 60.0
                useful += b >= 180
        err = [y - p for p, a, y, _, _ in rows]
        T = tot[c]
        T[0] += err
        T[1] += shown
        T[2] += br
        T[3] += mins
        T[4] += useful
        T[5] += len(rows)
        line += " | %2ds 误差×%.2f 窗口%3.0f%% ≥3分%3.0f%% 违约%3.0f%%" % (
            c, math.exp(statistics.median(abs(x) for x in err)), 100.0 * shown / len(rows), 100.0 * useful / len(rows),
            100.0 * br / shown if shown else 0)
    print(line, flush=True)
print()
for c in CHECKS:
    e, s, b, m, u, n = tot[c][:6]
    print("开局第 %2d 秒：%4d 个仍在进行的任务，剩余时间误差中位 ×%.2f（σ %.2f）；给出离开窗口 %2.0f%%，其中 ≥3 分钟 %2.0f%%，违约 %.0f%%，平均 %.1f 分钟" % (
        c, n, math.exp(statistics.median(abs(x) for x in e)), statistics.pstdev(e), 100.0 * s / n, 100.0 * u / n,
        100.0 * b / s if s else 0, m / s if s else 0))
