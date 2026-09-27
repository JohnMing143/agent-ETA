"""Compare two time-travel runs (old vs new plugin) point by point."""
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E  # noqa: E402

old_dir, new_dir = sys.argv[1], sys.argv[2]
rng = random.Random(9)


def gain(pairs):
    """pairs: {run: [(loss_old, loss_new)]} -> 1 - new/old with 95 % cluster-bootstrap interval."""
    ids = [i for i in pairs if pairs[i]]
    if len(ids) < 3:
        return None
    g = lambda s: 1 - sum(b for i in s for _, b in pairs[i]) / sum(a for i in s for a, _ in pairs[i])  # noqa: E731
    bs = sorted(g([rng.choice(ids) for _ in ids]) for _ in range(2000))
    return g(ids), bs[50], bs[1949]


def fmt_g(x):
    return "–" if x is None else "%+.1f%% [%+.1f, %+.1f]" % (100 * x[0], 100 * x[1], 100 * x[2])


def rate(xs):
    xs = [x for x in xs if x is not None]
    return "%3.0f%%" % (100.0 * sum(xs) / len(xs)) if xs else "  –"


tot = {"old": [0, 0, 0.0], "new": [0, 0, 0.0]}
all_cur, all_pri = {}, {}
print("%-38s %-26s %-26s %-24s %-24s %s" % ("历史", "旧：承诺/违约/分钟", "新：承诺/违约/分钟", "学习后预测 新vs旧",
                                           "内置先验 新vs旧", "P50命中/P80覆盖 旧→新"))
for path in sorted(glob.glob(os.path.join(old_dir, "*.json"))):
    name = os.path.basename(path)[:-5]
    a, b = json.load(open(path)), json.load(open(os.path.join(new_dir, name + ".json")))
    assert len(a) == len(b)
    cur, pri = {}, {}
    for x, y in zip(a, b):
        key = (name, x["run"])
        if x["loss"] is not None and y["loss"] is not None:
            cur.setdefault(key, []).append((x["loss"], y["loss"]))
        if x["loss_prior"] is not None and y["loss_prior"] is not None:
            pri.setdefault(key, []).append((x["loss_prior"], y["loss_prior"]))
    all_cur.update(cur)
    all_pri.update(pri)
    cols = []
    for tag, recs in (("old", a), ("new", b)):
        prom = [x for x in recs if x["promise"]]
        br = sum(x["breach"] for x in prom)
        mins = sum(x["promise"] for x in prom) / 60.0
        tot[tag][0] += len(prom)
        tot[tag][1] += br
        tot[tag][2] += mins
        cols.append("%3d 次 %3.0f%% %4.0f 分" % (len(prom), 100.0 * br / len(prom) if prom else 0, mins))
    print("%-38s %-26s %-26s %-24s %-24s %s/%s → %s/%s" % (
        name[:38], cols[0], cols[1], fmt_g(gain(cur)), fmt_g(gain(pri)),
        rate(x["hit50"] for x in a), rate(x["cov80"] for x in a), rate(x["hit50"] for x in b), rate(x["cov80"] for x in b)))
print()
for tag in ("old", "new"):
    n, br, m = tot[tag]
    print("合计 %s：%d 次承诺，违约 %d 次 %.1f%%（95%%上界 %.1f%%），承诺 %.0f 分钟" % (
        "旧版 v0.5.2" if tag == "old" else "新版（双峰）", n, br, 100.0 * br / n, 100 * E.clopper_pearson_upper(n, br, 0.05), m))
print("学习后预测（全系统按时间前瞻）新 vs 旧：%s" % fmt_g(gain(all_cur)))
print("内置先验（不学习，新用户第一天）新 vs 旧：%s" % fmt_g(gain(all_pri)))
