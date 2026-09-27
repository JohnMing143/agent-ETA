"""At the moment a task starts: how long will it take, and how long can you safely leave?
Prequential (each task predicted only from earlier ones). Methods:
  current   built-in prior median x your pace (per category), as the plugin does at elapsed 0
  similar   median of similar past prompts (char n-gram TF-IDF cosine), shrunk toward `current`
Leave window at start = predicted median x exp(q20 of past log(time-to-need / predicted median)) - a
conformal lower bound from your own history, so ~80 % hold whatever the predictor; its length shows how
well the predictor tells long tasks from short ones."""
import math
import os
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E  # noqa: E402
import prompts as P  # noqa: E402

TOPK = int(os.environ.get("TOPK", "10"))
SHRINK = float(os.environ.get("SHRINK", "1.0"))
MIN_SIM = float(os.environ.get("MIN_SIM", "0.15"))
MAXLEN = 400


def grams(text):
    t = re.sub(r"\s+", " ", text.lower())[:MAXLEN]
    out = Counter()
    for n in (2, 3):
        for i in range(len(t) - n + 1):
            g = t[i:i + n]
            if g.strip():
                out[g] += 1
    return out


class Index:
    def __init__(self):
        self.docs, self.df, self.n = [], Counter(), 0

    def vec(self, g):
        v = {k: (1 + math.log(c)) * math.log((1 + self.n) / (1 + self.df[k])) for k, c in g.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}

    def add(self, g, payload):
        self.docs.append((g, payload))
        self.n += 1
        for k in g:
            self.df[k] += 1

    def query(self, g):
        q = self.vec(g)
        sims = []
        for dg, payload in self.docs:
            d = self.vec(dg)
            s = sum(x * d.get(k, 0.0) for k, x in q.items())
            sims.append((s, payload))
        sims.sort(key=lambda x: -x[0])
        return [x for x in sims[:TOPK] if x[0] >= MIN_SIM]


def pace(past, row):
    """The plugin's calibrated prior median at elapsed 0 (per-category pace, shrunk), from past runs."""
    med = lambda r: E.prior_median({"category": r["category"] or "other", "effort": r["effort"], "prompt_len": r["prompt_len"]})  # noqa: E731
    rs = [(r["category"] or "other", math.log(r["dur"] / med(r))) for r in past]
    if not rs:
        return math.log(med(row))
    mu = sum(x for _, x in rs) / (len(rs) + E.CAL_N0)
    cat = row["category"] or "other"
    sc = [x for c, x in rs if c == cat]
    mc = (sum(sc) + E.CAL_N1 * mu) / (len(sc) + E.CAL_N1)
    return math.log(med(row)) + mc


def run(name):
    rows = [r for r in P.load(name) if r["stop"]]
    rows.sort(key=lambda r: r["t"])
    idx = Index()
    res = {"current": [], "similar": []}  # (log pred, log dur, log need)
    finished = []
    for r in rows:
        past = [x for x in finished if x["end"] < r["t"]]
        base = pace(past, r)
        g = grams(r["prompt"])
        nb = idx.query(g) if idx.n else []
        nb = [(s, p) for s, p in nb if p["end"] < r["t"]]
        if nb:
            w = [s * s for s, _ in nb]
            knn = sum(wi * math.log(p["dur"]) for wi, (_, p) in zip(w, nb)) / sum(w)
            neff = sum(w)
            sim = (neff * knn + SHRINK * base) / (neff + SHRINK)
        else:
            sim = base
        need = min(r["attn"], r["dur"]) if r["attn"] is not None else r["dur"]
        y, a = math.log(max(r["dur"], 1.0)), math.log(max(need, 1.0))
        res["current"].append((base, y, a))
        res["similar"].append((sim, y, a))
        finished.append(r)
        idx.add(g, r)
    return res


def leave_stats(trip):
    """Conformal start-of-task windows: q20 of earlier (need - pred) residuals, once 10 are known."""
    shown = br = 0
    mins, useful = 0.0, 0
    for i, (p, _, a) in enumerate(trip):
        past = sorted(x[2] - x[0] for x in trip[:i])
        if len(past) < 10:
            continue
        w = math.exp(p + past[int(0.2 * (len(past) - 1))])
        if w >= 60:
            b = 60 * math.floor(w / 60) if w < 600 else 300 * math.floor(w / 300)
            shown += 1
            br += a < math.log(b)
            mins += b / 60.0
            useful += b >= 180
    return shown, br, mins, useful


if __name__ == "__main__":
    tot = {k: [[], 0, 0, 0.0, 0, 0] for k in ("current", "similar")}
    print("%-11s %4s | %-44s | %-44s" % ("历史", "任务", "现在：误差×/开局给窗口/≥3分钟/违约/分钟", "相似任务：同上"))
    for name in P.HISTORIES:
        res = run(name)
        cells = []
        for k in ("current", "similar"):
            trip = res[k]
            err = [abs(y - p) for p, y, _ in trip]
            s, b, m, u = leave_stats(trip)
            n = len(trip)
            cells.append("×%.2f / %3.0f%% / %3.0f%% / %3.0f%% / %4.0f" % (
                math.exp(statistics.median(err)), 100.0 * s / n, 100.0 * u / n, 100.0 * b / s if s else 0, m))
            t = tot[k]
            t[0] += [(y - p) for p, y, _ in trip]
            t[1] += s
            t[2] += b
            t[3] += m
            t[4] += u
            t[5] += n
        print("%-11s %4d | %-44s | %-44s" % (name, len(res["current"]), cells[0], cells[1]), flush=True)
    for k in ("current", "similar"):
        e, s, b, m, u, n = tot[k]
        print("合计 %-8s 开局误差中位 ×%.2f，残差σ %.2f；开局给出窗口 %2.0f%% 的任务，其中 ≥3 分钟的占全部任务 %2.0f%%；违约 %.0f%%；共 %.0f 分钟" % (
            k, math.exp(statistics.median(abs(x) for x in e)), statistics.pstdev(e), 100.0 * s / n, 100.0 * u / n,
            100.0 * b / s if s else 0, m))
