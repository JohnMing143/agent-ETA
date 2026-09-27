"""Universal 'about to finish' signals? For every snapshot (taken as a main-thread tool call finishes),
group by what just happened and compare the real remaining time with the current estimator's P50."""
import math
import statistics
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, store, tuning  # noqa: E402

conn = store.connect(create=False)

if conn is None:

    sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
cfg = tuning.current_config(conn)
recal = E.tuned_recal(conn)
pool = E.Pool.load(conn, time.time() + 1, limit=100000)
runs = conn.execute("SELECT * FROM runs WHERE end_reason='stop' AND active_s IS NOT NULL").fetchall()


def signal(run, snap):
    if snap["plan_total"] and snap["plan_done"] == snap["plan_total"]:
        return "计划清单全部完成"
    tc = conn.execute("SELECT * FROM tool_calls WHERE run_id=? AND COALESCE(agent_id,'')='' AND ended_at<=?"
                      " ORDER BY ended_at DESC LIMIT 1", (run["id"], snap["t"] + 0.05)).fetchone()
    if tc is None:
        return "刚开始（还没调用工具）"
    key = tc["cmd_key"] or ""
    if key.startswith("git") and any(w in key for w in ("commit", "push", "tag")):
        return "git commit/push"
    if tc["cmd_kind"] in ("test", "build", "lint"):
        return "验证通过" if tc["ok"] else "验证失败"
    if tc["kind"] == "edit":
        return "编辑文件"
    if tc["kind"] == "explore":
        return "查看/搜索"
    if tc["tool"] in ("TodoWrite", "TaskUpdate", "TaskCreate"):
        return "更新计划"
    if tc["kind"] == "shell":
        return "其他命令" + ("" if tc["ok"] else "（失败）")
    return "其他工具"


groups = defaultdict(list)
for run in runs:
    snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL AND censored=0 ORDER BY t",
                         (run["id"],)).fetchall()
    for s in snaps:
        st = E.state_from_snapshot(run, s)
        est = E.estimate(conn, st, pool=pool, with_permission=False, k0=cfg[0], half_life=cfg[1],
                         recal=recal or False, safe_k=6.0)
        groups[signal(run, s)].append((run["id"], s["rem_done_s"], est["done_p50"], s["active_s"]))

allrows = [r for g in groups.values() for r in g]
print("全部快照 %d 个（%d 次正常结束的运行）：实际剩余中位 %.0fs，<30s 占 %.0f%%\n" % (
    len(allrows), len(runs), statistics.median(r[1] for r in allrows), 100 * sum(r[1] < 30 for r in allrows) / len(allrows)))
print("%-20s %5s %5s  %9s  %7s  %9s  %s" % ("刚发生的事", "点数", "运行", "实际剩余中位", "<30s占比", "预测P50中位", "实际/预测（中位）"))
for name, rows in sorted(groups.items(), key=lambda kv: statistics.median(r[1] for r in kv[1])):
    if len(rows) < 8:
        continue
    ratio = statistics.median(math.log(max(r[1], 1) / max(r[2] or 1, 1)) for r in rows)
    print("%-20s %5d %5d  %8.0fs  %6.0f%%  %8.0fs   ×%.2f" % (
        name, len(rows), len({r[0] for r in rows}), statistics.median(r[1] for r in rows),
        100 * sum(r[1] < 30 for r in rows) / len(rows), statistics.median(r[2] or 0 for r in rows), math.exp(ratio)))
