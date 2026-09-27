"""Choose, per user, how far the estimator trusts neighbouring runs over its calibrated prior, and
how cautious the "safe to leave" promise is.

The right balance depends on the history. A few dozen varied runs are predicted best by the
calibrated prior alone; hundreds of similar runs by the neighbour curves; a history whose habits
drift needs the neighbours sooner. So each candidate strength K0 is scored on a replay of your own
finished runs, where every prediction may only use runs that had finished before it, and the best
one is kept. The score is the log-scale pinball loss of the three quantiles the status line shows
(done P50 and P80, attention P20), so a few very long runs cannot dominate it.

The same replay picks how long the estimator remembers (the recency half-life of the weights on past
runs: habits drift, and for some histories the last day predicts better than the last month), and
whether to read the curves through a quantile recalibration learned from your own outcomes (when the
estimates run systematically long or short). The recalibration is judged prequentially - each point
read through a mapping learned only from runs that had finished before it - and kept only if that
lowers the loss.

The leave-window caution SAFE_K is chosen on the same replay: the most generous candidate whose
breach rate (Claude needed you before the promised time) has a 75 % upper bound under the target.
The replay cannot see permission prompts in imported history, so the windows the status line really
showed are the final check: if they were credibly breached too often (the breach rate's 75 % lower
bound is over the target), SAFE_K moves one step more cautious than it was, on every re-tune until
the ledger recovers. Merely uncertain evidence does not count: stepping up whenever the upper bound
was over the target ratcheted SAFE_K to its maximum on an honest 15 % record, and the few promises
left were the overconfident ones (replayed through time: 101 promises / 15 % -> 27 / 26 %).

No leave window is shown before the first tune (fewer than FIRST_TUNE_RUNS finished runs, counting
imported transcripts). Replayed through time on six histories (405 runs, five of them public), the
28 promises an untuned prior would have made were breached 21 % of the time - 3 of 5 and 2 of 2 on
two of them; after the first tune, 21 of 132 (16 %). It costs good promises on histories close to
the built-in pace (the author's: 19 of 20 held), which is the price of not guessing.

Replaying the whole candidate grid costs a few seconds on a large history, so live re-tuning runs in
a detached background process after enough new runs have finished.
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
import time

from . import config, render, store
from . import estimator as E

GRID = (3.0, 8.0, 20.0, 50.0, math.inf)   # K0; inf: calibrated prior only
HL_GRID = (0.5, 2.0, 7.0, E.RECENCY_HALF_LIFE_D)  # recency half-life of past runs, days
HL_TOLERANCE = 0.01                        # a shorter memory must beat the longest by more than this
RECAL_MIN_POINTS = 30                      # scored outcomes before a recalibration is fitted
RECAL_MIN_GAIN = 0.005                     # ... and it must lower the prequential loss by this much
MIN_POINTS = 20                            # fewer scored points than this: keep the default
FIRST_TUNE_RUNS = 10
RETUNE_GROWTH = 0.10                       # re-tune once finished runs grew 10 % (and by >= 5)
PENDING_TTL_S = 900
SAFE_GRID = (0.0, 2.0, 6.0, 15.0, 40.0)
MIN_PROMISES = 20
SAFE_ALPHA = 0.25                          # one-sided 75 % upper bound on the breach rate
LEDGER_DAYS = 30


def _summary(est, safe_q, keep=False):
    out = {"p50": est["done_p50"], "p80": est["done_p80"], "a20": est["attn_p20"],
           "safe": E.safe_seconds(est, safe_q), "method": est["method"]}
    if keep:
        out["est"] = est  # the curves, to try other leave-window settings later
    return out


def grid(k0s=GRID, half_lives=HL_GRID):
    return [(k, h) for h in half_lives for k in k0s]


def current_config(conn):
    """(K0, half-life) the live estimator uses."""
    tuned = E.tuned_values(conn)
    return tuned.get("k0", E.K0_DEFAULT), tuned.get("half_life", E.RECENCY_HALF_LIFE_D)


def replay(conn, since=0.0, max_runs=80, per_run=6, configs=None, safe_q=0.2, with_baselines=True, keep=False,
           pool_limit=None):
    """Backtest points of the most recent finished runs.

    Each point: its run and moment, the truth (remaining done / attention time; None when censored), the
    raw estimate (no recalibration) for every (K0, half-life) in `configs` (default: the current one),
    and optionally the built-in prior alone and the Codex design's B0 baseline.
    """
    configs = configs or [current_config(conn)]
    safe_k = E.tuned_values(conn).get("safe_k", E.SAFE_K_DEFAULT)
    # the evaluated runs are the most recent ones; their past lies within the newest 2 x POOL_RUNS runs,
    # which keeps the neighbour pool the size the live estimator uses
    pool = E.Pool.load(conn, time.time() + 1, limit=pool_limit or 2 * E.POOL_RUNS)
    empty = E.Pool()
    runs = conn.execute(
        "SELECT * FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL AND started_at > ?"
        " ORDER BY started_at DESC LIMIT ?", (since, max_runs)).fetchall()
    points = []
    for run in runs:
        snaps = conn.execute("SELECT * FROM snapshots WHERE run_id=? AND rem_done_s IS NOT NULL ORDER BY t",
                             (run["id"],)).fetchall()
        if not snaps:
            continue
        step = max(1, len(snaps) // per_run)
        for s in snaps[::step][:per_run]:
            state = E.state_from_snapshot(run, s)
            cands = pool.candidates(state)
            point = {"run": run["id"], "as_of": state["as_of"], "ended": run["ended_at"],
                     "done": None if s["censored"] else s["rem_done_s"],
                     "attn": s["rem_attn_s"] if s["attn_event"] else None,
                     "est": {c: _summary(E.estimate(conn, state, pool=pool, with_permission=False, k0=c[0],
                                                    half_life=c[1], recal=False, safe_k=safe_k, cands=cands),
                                         safe_q, keep) for c in configs}}
            if with_baselines:
                point["prior"] = _summary(E.estimate(conn, state, pool=empty, with_permission=False, k0=math.inf,
                                                     half_life=E.RECENCY_HALF_LIFE_D, recal=False, safe_k=safe_k),
                                          safe_q)
                q = E.b0_estimate(pool, state)
                point["b0"] = {"p50": q[0.5], "p80": q[0.8], "a20": None, "safe": None}
            points.append(point)
    return points


def _log_pinball(q, y, p):
    y, p = math.log(max(y, 1.0)), math.log(max(p, 1.0))
    return q * (y - p) if y >= p else (1.0 - q) * (p - y)


def point_loss(point, est):
    """Mean log-scale pinball loss of the quantiles shown; None if nothing is scorable."""
    parts = []
    if point["done"] is not None:
        for q, key in ((0.5, "p50"), (0.8, "p80")):
            if est[key] is not None:
                parts.append(_log_pinball(q, point["done"], est[key]))
    if point["attn"] is not None and est["a20"] is not None:
        parts.append(_log_pinball(0.2, point["attn"], est["a20"]))
    return sum(parts) / len(parts) if parts else None


def mean_loss(items, pick, point=lambda p: p):
    """(mean point_loss, n scored) of pick(item) against point(item)."""
    losses = [point_loss(point(x), pick(x)) for x in items]
    losses = [x for x in losses if x is not None]
    return (sum(losses) / len(losses) if losses else None), len(losses)


def choose(points, configs):
    """((K0, half-life), {config: mean loss}, scored points). Too few points: the defaults.

    The half-life is the longest one whose best loss is within HL_TOLERANCE of the overall best (a
    shorter memory has more ways to fit noise); K0 is then the best at that half-life, ties to the
    more cautious."""
    scored = [p for p in points if all(point_loss(p, p["est"][c]) is not None for c in configs)]
    losses = {c: (sum(point_loss(p, p["est"][c]) for p in scored) / len(scored)) if scored else None
              for c in configs}
    if len(scored) < MIN_POINTS:
        return (E.K0_DEFAULT, E.RECENCY_HALF_LIFE_D), losses, len(scored)
    best_by_hl = {}
    for (k, h), v in losses.items():
        best_by_hl[h] = min(v, best_by_hl.get(h, v))
    best = min(best_by_hl.values())
    hl = max(h for h, v in best_by_hl.items() if v <= best * (1.0 + HL_TOLERANCE))
    k0 = min((c[0] for c in configs if c[1] == hl), key=lambda k: (losses[(k, hl)], -k))
    return (k0, hl), losses, len(scored)


def _pit(S, y):
    return min(1.0, max(0.0, 1.0 - S(y)))


def fit_recal(points, cfg):
    """Quantile recalibration from the PIT values of replayed outcomes (curves kept by replay(keep=True))."""
    done = [_pit(p["est"][cfg]["est"]["S_done"], p["done"]) for p in points if p["done"] is not None]
    attn = [_pit(p["est"][cfg]["est"]["S_attn"], p["attn"]) for p in points if p["attn"] is not None]
    if len(done) < RECAL_MIN_POINTS:
        return None
    return {"done": E.pit_knots(done), "n_done": len(done),
            "attn": E.pit_knots(attn) if len(attn) >= RECAL_MIN_POINTS else None, "n_attn": len(attn)}


def read_through(est, recal, safe_q, safe_k=None):
    """A summary of raw curves read at recalibrated levels."""
    lv = lambda part, q: E.recal_level(recal, part, q)  # noqa: E731
    return {"p50": E.quantile(est["S_done"], lv("done", 0.5)), "p80": E.quantile(est["S_done"], lv("done", 0.8)),
            "a20": E.quantile(est["S_attn"], lv("attn", 0.2)),
            "safe": E.safe_seconds(dict(est, recal=recal), safe_q, safe_k), "method": est["method"]}


def prequential_recal(points, cfg, safe_q, safe_k=None):
    """Each point read through a recalibration learned only from runs that had finished before it."""
    by_end = sorted(points, key=lambda p: p["ended"])
    out, j, known = [None] * len(points), 0, []
    for i in sorted(range(len(points)), key=lambda i: points[i]["as_of"]):
        p = points[i]
        while j < len(by_end) and by_end[j]["ended"] < p["as_of"]:
            known.append(by_end[j])
            j += 1
        out[i] = read_through(p["est"][cfg]["est"], fit_recal(known, cfg), safe_q, safe_k)
    return out


def promise_stats(points, cfg, safe_q, k, recal=None):
    """(promises, breaches) of the leave windows the replay would have shown with SAFE_K = k."""
    n = breaches = 0
    for p in points:
        if p["attn"] is None:
            continue
        bucket = render.safe_bucket(E.safe_seconds(dict(p["est"][cfg]["est"], recal=recal), safe_q, k))
        if bucket:
            n += 1
            breaches += p["attn"] < bucket
    return n, breaches


def ledger_breaches(conn, now, safe_q):
    """(scored, breached) windows the status line really showed lately; unknown outcomes (you
    interrupted, the session ended) say nothing about the promise and are left out here."""
    from . import live
    ls = live.ledger_stats(conn, now, {"safe_quantile": safe_q}, since=now - LEDGER_DAYS * 86400)
    return ls["held"] + ls["breach"], ls["breach"]


def _step_up(k):
    return next((g for g in SAFE_GRID if g > k), SAFE_GRID[-1])


def choose_safe_k(stats, safe_q, previous=None, ledger=(0, 0)):
    """stats: {k: (promises, breaches)}. In order of preference: the most generous k whose breach rate
    is credibly under the target; else the k with the lowest observed rate if that is under the
    target; else the default, one step more cautious if its own rate is over the target. Real windows
    breached too often move it one step past `previous`. Small samples are noisy and need not get
    better with more caution, so the replay never jumps straight to the most cautious setting."""
    chosen = None
    for k in SAFE_GRID:
        n, br = stats[k]
        if n >= MIN_PROMISES and E.clopper_pearson_upper(n, br, SAFE_ALPHA) <= safe_q:
            chosen = k
            break
    if chosen is None:
        enough = [(br / n, -k, k) for k, (n, br) in stats.items() if n >= MIN_PROMISES]
        if enough and min(enough)[0] <= safe_q:
            chosen = min(enough)[2]
        else:
            n, br = stats[E.SAFE_K_DEFAULT]
            bad = n >= MIN_PROMISES and br > safe_q * n
            chosen = _step_up(E.SAFE_K_DEFAULT) if bad else E.SAFE_K_DEFAULT
    n, br = ledger
    if n >= MIN_PROMISES and E.clopper_pearson_lower(n, br, SAFE_ALPHA) > safe_q:
        chosen = max(chosen, _step_up(previous if previous is not None else chosen))
    return chosen


def finished_runs(conn):
    return conn.execute("SELECT COUNT(*) FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL").fetchone()[0]


def _key(k):
    return "inf" if math.isinf(k) else "%g" % k


def tune(conn, now=None):
    """Replay the history; choose K0, the half-life, the recalibration and SAFE_K and store them."""
    now = now or time.time()
    safe_q = config.load()["safe_quantile"]
    n_runs = finished_runs(conn)
    configs = grid()
    points = replay(conn, configs=configs, with_baselines=False, safe_q=safe_q, keep=True)
    (k0, hl), losses, n_points = choose(points, configs)
    cfg = (k0, hl)

    raw_loss, n_scored = mean_loss(points, lambda p: p["est"][cfg])
    pre = prequential_recal(points, cfg, safe_q)
    recal_loss, _ = mean_loss(list(zip(points, pre)), lambda pr: pr[1], lambda pr: pr[0])
    recal = fit_recal(points, cfg)
    use_recal = (recal is not None and n_scored >= MIN_POINTS and raw_loss is not None and recal_loss is not None
                 and recal_loss < raw_loss * (1.0 - RECAL_MIN_GAIN))

    stats = {k: promise_stats(points, cfg, safe_q, k, recal if use_recal else None) for k in SAFE_GRID}
    previous = E.tuned_values(conn).get("safe_k")
    ledger = ledger_breaches(conn, now, safe_q)
    safe_k = choose_safe_k(stats, safe_q, previous, ledger)

    k0_detail = json.dumps({_key(k): v for (k, h), v in losses.items() if h == hl})
    hl_best = {}
    for (k, h), v in losses.items():
        if v is not None:
            hl_best[h] = min(v, hl_best.get(h, v))
    hl_detail = json.dumps({"%g" % h: v for h, v in sorted(hl_best.items())})
    recal_detail = json.dumps(dict(recal or {}, loss_raw=raw_loss, loss_recal=recal_loss))
    safe_detail = json.dumps({"replay": {"%g" % k: v for k, v in stats.items()}, "ledger": ledger})
    rows = (("k0", k0, k0_detail), ("half_life", hl, hl_detail), ("recal", 1.0 if use_recal else 0.0, recal_detail),
            ("safe_k", safe_k, safe_detail))
    with store.tx(conn):
        for key, value, detail in rows:
            conn.execute("INSERT OR REPLACE INTO tuning(key, value, n_runs, n_points, at, pending_at, detail)"
                         " VALUES (?, ?, ?, ?, ?, NULL, ?)", (key, value, n_runs, n_points, now, detail))
    return {"k0": k0, "half_life": hl, "losses": losses, "hl_losses": hl_best, "n_points": n_points,
            "n_runs": n_runs, "recal": use_recal, "recal_losses": (raw_loss, recal_loss), "safe_k": safe_k,
            "safe_stats": stats, "ledger": ledger}


def describe(res):
    hl = res["half_life"]
    losses = " · ".join("%s %.3f" % ("∞" if math.isinf(k) else "%g" % k, v)
                        for (k, h), v in res["losses"].items() if v is not None and h == hl)
    hls = " · ".join("%g %.3f" % (h, v) for h, v in sorted(res["hl_losses"].items()))
    raw, rec = res["recal_losses"]
    n, br = res["safe_stats"][res["safe_k"]]
    return ("K0 = %s  (replayed %d runs, %d points; log pinball loss %s)\n"
            "half-life = %g days  (best loss per half-life %s)\n"
            "recalibration %s  (prequential loss %s raw, %s recalibrated)\n"
            "SAFE_K = %g  (replay: %d leave windows, %d breached; real windows lately: %d scored, %d breached)") % (
        "∞" if math.isinf(res["k0"]) else "%g" % res["k0"], res["n_runs"], res["n_points"], losses or "-",
        hl, hls or "-", "on" if res["recal"] else "off", "-" if raw is None else "%.3f" % raw,
        "-" if rec is None else "%.3f" % rec, res["safe_k"], n, br, res["ledger"][0], res["ledger"][1])


def record(conn):
    try:
        return conn.execute("SELECT * FROM tuning WHERE key='k0'").fetchone()
    except sqlite3.OperationalError:
        return None


def due(conn, now):
    n = finished_runs(conn)
    row = record(conn)
    if row is None:
        return n >= FIRST_TUNE_RUNS
    if row["pending_at"] is not None and now - row["pending_at"] < PENDING_TTL_S:
        return False
    return n - (row["n_runs"] or 0) >= max(5, RETUNE_GROWTH * (row["n_runs"] or 0))


def maybe_spawn(conn, now, spawn=None):
    """Called after a run finished: start a background re-tune when the history has grown enough.
    The claim (pending_at) is taken inside a write transaction, so concurrent hooks start one."""
    with store.tx(conn):
        if not due(conn, now):
            return False
        conn.execute("INSERT OR IGNORE INTO tuning(key, value, n_runs) VALUES ('k0', NULL, 0)")
        conn.execute("UPDATE tuning SET pending_at=? WHERE key='k0'", (now,))
    (spawn or _spawn_detached)()
    return True


def _spawn_detached():
    eta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eta.py")
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
              "close_fds": True}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # outlives the hook; never holds Claude Code's pipes
    subprocess.Popen([sys.executable, eta, "tune", "--quiet"], **kwargs)
