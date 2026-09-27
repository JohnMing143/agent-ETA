"""Remaining-time estimator for a running agent turn.

Two targets, as in the original discussion:
  T_c  time to completion  - when Claude will hand control back
  T_a  time to attention   - when Claude will next need you (permission, question, plan approval, or done)
"Safe to leave for X" is a *low* quantile of T_a (default P20: 80 % chance nothing needs you before X).

Every component is a survival curve S(t) = P(remaining > t), evaluable exactly at any t, so the
pieces combine coherently instead of averaging point guesses:

  1. prior      lognormal by task category / prompt length / effort, conditioned on elapsed time, and
                calibrated to your pace: overall and per-category speed and the spread, learned from
                your finished runs with shrinkage (few parameters, so it helps from ~5 runs on)
  2. learned    weighted Kaplan-Meier over the most similar snapshot of each past run, at two levels:
                  global  all your runs, matched only on the current stage (elapsed, tool count,
                          phase, failures, plan, subagents)
                  local   additionally matched on repo, model, effort, task type, prompt length
                kernels use an adaptive bandwidth (k-th nearest, k ~ sqrt(N)); interrupted runs enter
                as censored observations instead of being thrown away
  3. shrinkage  S_global = mix(KM_global, prior), S = mix(KM_local, S_global), each weighted
                ESS / (ESS + K0): sparse evidence falls back to the broader level. K0 is chosen per
                user by replaying their own history (tuning.py): a few dozen varied runs are
                predicted best by the calibrated prior, hundreds of similar ones by the neighbours
  4. plan       when Claude keeps a task list, mix in a per-step extrapolation
  5. in-flight  a long tool call (tests, build, subagent) puts a floor under both curves, using the
                duration history of that exact command
  6. permission T_a also carries a hazard of permission prompts: S_a *= exp(-rate * p_prompt * t)
No model calls, no self-reported progress.
"""
import bisect
import collections
import json
import math
import sqlite3

from . import features as F

K0_DEFAULT = 8.0  # until tuning.py has replayed enough of your history to choose
SIGMA = 1.0
POOL_RUNS = 400
RECENCY_HALF_LIFE_D = 60.0  # default memory; tuning.py picks a shorter one when your recent runs predict better
RECAL_M0 = 30.0             # pseudo-points behind "the curve is already calibrated" (see recal_level)
CAL_N0 = 5.0        # pseudo-runs behind the built-in pace (log ratio 0) and spread (SIGMA)
CAL_N1 = 5.0        # pseudo-runs behind the overall pace, for each task category
CAL_MIN_RUNS = 5    # below this the prior is still "built-in" (shown as such)

# Cold-start priors (seconds). Replaced by your own history as it accumulates.
BASE_MEDIAN_S = {"question": 40, "followup": 60, "command": 60, "wake": 60, "bash": 60, "review": 150, "fix": 180,
                 "test": 240, "ops": 240, "feature": 300, "refactor": 360, "other": 120}
EFFORT_MULT = {"low": 0.6, "medium": 0.8, "high": 1.0, "xhigh": 1.3, "max": 1.6}
PHASE_MULT = {"start": 1.0, "explore": 1.0, "implement": 0.85, "fixing": 1.0, "verify": 0.35}
PERM_P0 = {"default": 0.12, "plan": 0.05, "acceptEdits": 0.06, "auto": 0.01}
TOOL_PRIOR_S = {"test": 60, "build": 60, "install": 45, "lint": 20, "vcs": 3, "agent": 150, "explore": 3,
                "edit": 1, "mcp": 8, "plan": 1, "other": 10}


# ---------------------------------------------------------------- exact Kaplan-Meier (pure math)

class KaplanMeier:
    """Exact weighted Kaplan-Meier step function.

    Conventions (identical to the Codex handoff's fixtures/math-golden.json):
      * ties: events and censorings at the same instant share the risk set; the event jump comes first
      * S is right-continuous; Q(p) = first jump time t with F(t) >= p, else None
      * beyond the last observation S is unidentified (None) unless it has already reached 0
      * conditional remaining: S(e + r) / S(e)
    """

    def __init__(self, samples):
        pts = sorted((max(0.0, float(t)), float(w), bool(e)) for t, w, e in samples if w and w > 0 and t is not None)
        self.n = len(pts)
        self.times, self.values = [], []
        at_risk = sum(w for _, w, _ in pts)
        s_val, i = 1.0, 0
        while i < self.n:
            t = pts[i][0]
            d = removed = 0.0
            while i < self.n and pts[i][0] == t:
                removed += pts[i][1]
                if pts[i][2]:
                    d += pts[i][1]
                i += 1
            if d > 0 and at_risk > 0:
                s_val *= max(0.0, 1.0 - d / at_risk)
                self.times.append(t)
                self.values.append(s_val)
            at_risk -= removed
        self.t_max = pts[-1][0] if pts else 0.0
        self.s_end = s_val

    def survival(self, t):
        if t < 0:
            return 1.0
        if t > self.t_max and self.s_end > 0:
            return None
        i = bisect.bisect_right(self.times, t) - 1
        return 1.0 if i < 0 else self.values[i]

    def quantile(self, q):
        target = 1.0 - q
        for t, v in zip(self.times, self.values):
            if v <= target + 1e-12:
                return t
        return None

    def conditional_quantile(self, elapsed, q):
        base = self.survival(elapsed)
        if not base:
            return None
        target = (1.0 - q) * base
        for t, v in zip(self.times, self.values):
            if t > elapsed and v <= target + 1e-12:
                return t - elapsed
        return None


def ess(weights):
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    return s1 * s1 / s2 if s2 > 0 else 0.0


_ess = ess


def ema(intervals, alpha=0.3):
    """m_1 = first interval, m_k = alpha * latest + (1 - alpha) * m_(k-1)  (task-progress-bar baseline)."""
    states, m = [], None
    for x in intervals:
        m = x if m is None else alpha * x + (1.0 - alpha) * m
        states.append(m)
    return states


def clopper_pearson_lower(n, v, alpha):
    """One-sided Clopper-Pearson lower bound for a binomial rate with v events in n trials."""
    return 0.0 if n <= 0 or v <= 0 else 1.0 - clopper_pearson_upper(n, n - v, alpha)


def clopper_pearson_upper(n, v, alpha):
    """One-sided Clopper-Pearson upper bound for a binomial rate with v events in n trials."""
    if n <= 0:
        return 1.0
    if v >= n:
        return 1.0
    if v == 0:
        return 1.0 - alpha ** (1.0 / n)

    def cdf(p):  # P(X <= v | n, p), via log-sum-exp
        logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                + k * math.log(p) + (n - k) * math.log1p(-p) for k in range(v + 1)]
        top = max(logs)
        return math.exp(top) * sum(math.exp(x - top) for x in logs)

    lo, hi = v / n, 1.0 - 1e-15
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------- composable survival curves

HORIZON_S = 12 * 3600.0
_BASE_POINTS = [0.5 * 1.08 ** i for i in range(int(math.log(HORIZON_S / 0.5) / math.log(1.08)) + 2)]  # only brackets crossings; bisection and exact jumps do the rest


class Curve:
    """S(t) = P(remaining > t), evaluable exactly at any t; `jumps` lists its step times."""
    __slots__ = ("f", "jumps")

    def __init__(self, f, jumps=()):
        self.f = f
        self.jumps = tuple(jumps)

    def __call__(self, t):
        return self.f(t)


_SQRT2 = math.sqrt(2.0)


def _norm_cdf(z):
    return 0.5 * math.erfc(-z / _SQRT2)


def _median(values):
    v = sorted(values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2.0


def lognormal_survival(med, sigma=SIGMA, elapsed=0.0):
    """P(T - e > t | T > e) for T ~ LogNormal(ln med, sigma)."""
    mu = math.log(max(med, 0.5))

    def cdf(x):
        return _norm_cdf((math.log(x) - mu) / sigma) if x > 0 else 0.0

    base = 1.0 - cdf(elapsed) if elapsed > 0 else 1.0
    if base < 1e-6:
        # Far past the prior's mass: a run this old is likely to last about as long again (Lindy).
        return lognormal_survival(max(elapsed, 1.0) * 0.6, sigma)
    return Curve(lambda t: 1.0 if t <= 0 else min(1.0, max(0.0, (1.0 - cdf(elapsed + t)) / base)))


def km_survival(samples, tail=None):
    """Kaplan-Meier curve for the estimator. Beyond the data the leftover mass follows `tail`
    (normally the prior), so a handful of short past runs cannot claim the current one ends soon.
    This extrapolation is an estimator policy; KaplanMeier itself stays exact and returns None there."""
    km = KaplanMeier(samples)
    if km.n == 0:
        return None
    ref = tail(km.t_max) if tail is not None else 0.0
    times, values, t_max, s_end = km.times, km.values, km.t_max, km.s_end

    def f(t):
        if t < 0:
            return 1.0
        if t <= t_max or s_end <= 0:
            i = bisect.bisect_right(times, t) - 1
            return 1.0 if i < 0 else values[i]
        if tail is not None and ref > 1e-12:
            return s_end * min(1.0, tail(t) / ref)
        return s_end * math.exp(-(t - t_max) / max(t_max, 30.0))

    return Curve(f, times)


def mix(a, b, w):
    return Curve(lambda t: w * a(t) + (1.0 - w) * b(t), a.jumps + b.jumps)


def cmax(a, b):
    return Curve(lambda t: max(a(t), b(t)), a.jumps + b.jumps)


def cmin(a, b):
    return Curve(lambda t: min(a(t), b(t)), a.jumps + b.jumps)


def stretch(S, k):
    """Survival of k * T given survival of T."""
    if abs(k - 1.0) < 1e-9:
        return S
    return Curve(lambda t: S(t / k), [j * k for j in S.jumps])


def with_hazard(S, rate, delay=0.0):
    return Curve(lambda t: S(t) * math.exp(-rate * max(0.0, t - delay)), S.jumps)


def quantiles(S, qs):
    """Exact quantiles of a Curve: every jump is a candidate point, continuous parts are bisected.

    Every curve here is non-increasing (survival functions and their mixtures, min/max, stretches and
    hazard products), so the first candidate point at or below the target is found by binary search:
    ~log2(N) evaluations instead of one per jump.
    """
    pts = sorted(set(_BASE_POINTS).union(j for j in S.jumps if 0 < j <= HORIZON_S))
    jumps = set(S.jumps)
    vals = {}

    def at(i):
        v = vals.get(i)
        if v is None:
            v = vals[i] = S(pts[i])
        return v

    out = {}
    for q in qs:
        target = 1.0 - q + 1e-12
        lo, hi = 0, len(pts)
        while lo < hi:
            mid = (lo + hi) // 2
            if at(mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        res = None
        if lo < len(pts):
            t, prev = pts[lo], (pts[lo - 1] if lo else 0.0)
            if t in jumps and S(t * (1 - 1e-12)) > target:
                res = t  # the drop happens exactly at this jump
            else:
                a, b = prev, t
                for _ in range(60):
                    m = (a + b) / 2.0
                    if S(m) <= target:
                        b = m
                    else:
                        a = m
                    if b - a <= 1e-4 * b:
                        break
                res = b
        out[q] = res
    return out


def quantile(S, q):
    return quantiles(S, [q])[q]


# ---------------------------------------------------------------- neighbours

def _lage(a):
    return math.log1p(max(a or 0.0, 0.0) / 15.0)


def static_distance(state, r):
    d = 0.0
    if state["repo"] != r["repo"]:
        d += 1.0
    if state["model"] and r["model"] and state["model"] != r["model"]:
        d += 0.4
    if state["effort"] and r["effort"] and state["effort"] != r["effort"]:
        d += 0.3
    if state["category"] != r["category"]:
        d += 1.0 if {state["category"], r["category"]} & set(F.NON_PROMPT_CATEGORIES) else 0.5
    d += 0.5 * (math.log(state["prompt_len"] + 30.0) - math.log((r["prompt_len"] or 0) + 30.0)) ** 2
    if F.perm_class(state["mode"]) != F.perm_class(r["permission_mode"]):
        d += 0.2
    return d


def _features(active_s, n_tools, n_verify_fail, phase, plan_total, plan_done, agents_running):
    """Trajectory coordinates compared by dynamic_distance."""
    return (_lage(active_s), math.log1p(n_tools or 0), phase, F.PHASES.get(phase, 1.0), min(n_verify_fail or 0, 3),
            (plan_done or 0) / plan_total if (plan_total or 0) >= 2 else None, min(agents_running or 0, 2))


def state_features(st):
    return _features(st["active_s"], st["n_tools"], st["n_verify_fail"], st["phase"], st["plan_total"],
                     st["plan_done"], st["agents_running"])


def dynamic_distance(a, b):
    """Distance between two feature tuples. Its first term, 1.5 * (elapsed gap)^2, is also a lower
    bound of the whole distance, which lets the nearest-snapshot search stop early."""
    d = 1.5 * (a[0] - b[0]) ** 2 + 0.6 * (a[1] - b[1]) ** 2
    if a[2] != b[2]:
        d += 0.6 + 0.4 * abs(a[3] - b[3])
    d += 0.4 * abs(a[4] - b[4])
    pa, pb = a[5], b[5]
    if pa is not None and pb is not None:
        d += 1.5 * abs(pa - pb)
    elif (pa is None) != (pb is None):
        d += 0.3
    return d + 0.5 * abs(a[6] - b[6])


# A labelled snapshot of a finished run; fields 1..7 are the arguments of _features, in order.
Snap = collections.namedtuple("Snap", "run_id active_s n_tools n_verify_fail phase plan_total plan_done"
                                      " agents_running rem_done_s censored rem_attn_s attn_event")


class Pool:
    """Labelled snapshots of finished runs, grouped per run and sorted by elapsed time.

    It is rebuilt on every status line refresh, so rows stay plain tuples and a snapshot's features are
    only computed when the nearest-snapshot search actually looks at it.
    """

    _RECENT = ("SELECT id FROM runs WHERE ended_at IS NOT NULL AND active_s IS NOT NULL AND ended_at < ?"
               " ORDER BY ended_at DESC LIMIT ?")
    RUNS_SQL = ("SELECT id AS run_id, repo, model, effort, permission_mode, category, prompt_len, ended_at,"
                " end_reason, active_s AS run_active_s"
                " FROM runs WHERE id IN (%s)" % _RECENT)
    SNAPS_SQL = ("SELECT run_id, active_s, n_tools, n_verify_fail, phase, plan_total, plan_done, agents_running,"
                 " rem_done_s, censored, rem_attn_s, attn_event FROM snapshots"
                 " WHERE run_id IN (%s) AND rem_done_s IS NOT NULL ORDER BY run_id, active_s, id" % _RECENT)

    def __init__(self, metas=(), snaps=()):
        by_run = {}
        for row in snaps:
            entry = by_run.get(row[0])
            if entry is None:
                entry = by_run[row[0]] = ([], [])
            entry[0].append(row)
            entry[1].append(row[1] or 0.0)
        # run_id -> (run meta, snapshots, their elapsed times, feature cache)
        self.runs = {m["run_id"]: (m, by_run[m["run_id"]][0], by_run[m["run_id"]][1],
                                   [None] * len(by_run[m["run_id"]][0]))
                     for m in metas if m["run_id"] in by_run}

    @classmethod
    def load(cls, conn, before, limit=POOL_RUNS):
        metas = conn.execute(cls.RUNS_SQL, (before, limit)).fetchall()
        cur = conn.cursor()
        cur.row_factory = None  # plain tuples in Snap field order: thousands of rows, fetched every refresh
        return cls(metas, cur.execute(cls.SNAPS_SQL, (before, limit)).fetchall())

    def candidates(self, state):
        """For every earlier run: its snapshot most similar to the current state.

        Returns (static_distance, dynamic_distance, age_days, snapshot, run_meta) tuples; the recency
        weight is applied by estimate(), so one search serves every half-life a replay tries.
        """
        as_of = state["as_of"]
        q = state_features(state)
        active = state["active_s"] or 0.0
        out = []
        for run_id, (meta, snaps, actives, keys) in self.runs.items():
            if run_id == state["run_id"] or meta["ended_at"] >= as_of:
                continue
            i, dd = _nearest(q, active, snaps, actives, keys)
            age_days = max(0.0, (as_of - meta["ended_at"]) / 86400.0)
            out.append((static_distance(state, meta), dd, age_days, Snap._make(snaps[i]), meta))
        return out


def _nearest(q, active, snaps, actives, keys):
    """Exact nearest snapshot (index, distance). Snapshots are sorted by elapsed time, so the search
    walks outwards from the current elapsed time and stops once the elapsed-time term alone exceeds
    the best distance found."""
    start = bisect.bisect_left(actives, active)
    best_i, best_d = -1, math.inf
    for rng in (range(start, len(snaps)), range(start - 1, -1, -1)):
        for i in rng:
            k = keys[i]
            if k is None:
                k = keys[i] = _features(*snaps[i][1:8])
            if 1.5 * (k[0] - q[0]) ** 2 > best_d:
                break
            d = dynamic_distance(q, k)
            if d < best_d or (d == best_d and i < best_i):
                best_i, best_d = i, d
    return best_i, best_d


def adaptive_weights(distances, recency, k_min, k_max):
    """Gaussian kernel weights exp(-(d/h)^2), h = distance of the k-th nearest point, k ~ sqrt(N).

    Sparse history -> wide kernel (borrow from everything); rich history -> only close matches count.
    """
    n = len(distances)
    if not n:
        return []
    k = int(min(max(round(math.sqrt(n) * 1.2), k_min), k_max, n))
    h = max(sorted(distances)[k - 1], 0.25)
    return [math.exp(-(d / h) ** 2) * r for d, r in zip(distances, recency)]


# ---------------------------------------------------------------- state

def state_from_run(conn, run, now):
    from .ingest import active_at
    last_resolved = conn.execute(
        "SELECT MAX(resolved_at) FROM attention WHERE run_id=?", (run["id"],)).fetchone()[0] or 0.0
    in_flight = None
    for r in conn.execute(
            "SELECT tool, kind, cmd_key, cmd_kind, detail, started_at FROM tool_calls WHERE run_id=?"
            " AND ended_at IS NULL AND started_at > ? ORDER BY (agent_id IS NULL) DESC, started_at",
            (run["id"], now - 7200)).fetchall():
        if r["tool"] in F.ATTENTION_TOOLS:
            continue
        if run["attn_since"] is not None and r["started_at"] <= run["attn_since"]:
            continue  # it is waiting for your approval, not running
        in_flight = {"tool": r["tool"], "kind": r["kind"], "cmd_key": r["cmd_key"] or r["tool"],
                     "cmd_kind": r["cmd_kind"], "detail": r["detail"],
                     "running_s": max(0.0, now - max(r["started_at"], last_resolved))}
        break
    attn = None
    if run["attn_since"] is not None:
        attn = {"kind": run["attn_kind"], "detail": run["attn_detail"], "waiting_s": max(0.0, now - run["attn_since"])}
    return _state(run, active_at(run, now), now, F.phase_of(run), in_flight, attn)


def state_from_snapshot(run, snap):
    """State as it was known at the snapshot (for backtests). Run-level fields that change during the
    run (plan anchors, EMA, background count) hold their *final* values, so they are blanked rather
    than leaked from the future."""
    s = _state(run, snap["active_s"], snap["t"], snap["phase"], None, None)
    for k in ("n_tools", "n_edit", "n_verify", "n_fail", "n_verify_fail", "plan_total", "plan_done",
              "agents_running"):
        s[k] = snap[k] or 0
    s.update(status="running", bg_count=0, plan_first_total=None, active_at_last_done=None, plan_ema=None)
    return s


def _state(run, active_s, as_of, phase, in_flight, attn):
    return {
        "run_id": run["id"], "session_id": run["session_id"], "as_of": as_of,
        "started_at": run["started_at"], "repo": run["repo"], "model": run["model"], "effort": run["effort"],
        "mode": run["permission_mode"], "category": run["category"] or "other",
        "prompt_len": run["prompt_len"] or 0, "preview": run["prompt_preview"],
        "active_s": active_s, "phase": phase,
        "n_tools": run["n_tools"] or 0, "n_edit": run["n_edit"] or 0, "n_verify": run["n_verify"] or 0,
        "n_fail": run["n_fail"] or 0, "n_verify_fail": run["n_verify_fail"] or 0,
        "plan_total": run["plan_total"] or 0, "plan_done": run["plan_done"] or 0,
        "plan_first_total": run["plan_first_total"], "active_at_last_done": run["active_at_last_done"],
        "plan_ema": run["plan_ema"],
        "agents_running": run["agents_running"] or 0, "bg_count": run["bg_count"] or 0,
        "status": run["status"], "in_flight": in_flight, "attn": attn,
    }


# ---------------------------------------------------------------- components

def prior_median(state):
    base = BASE_MEDIAN_S.get(state["category"], BASE_MEDIAN_S["other"])
    return base * EFFORT_MULT.get(state["effort"], 1.0) * (1.0 + min(state["prompt_len"], 4000) / 2000.0)


NO_CALIBRATION = ({}, 0.0, SIGMA, 0)


def calibration(pool, state, half_life=RECENCY_HALF_LIFE_D):
    """Your pace relative to the built-in prior: (log-ratio per category, overall log-ratio, sigma, runs).

    Learned from runs that finished before the state's moment, in log space, each part shrunk toward
    the built-in value by a few pseudo-runs: the overall pace, a per-category pace (shrunk toward the
    overall one) and the spread. Far fewer degrees of freedom than the neighbour curves, so it is
    useful from a handful of runs on. Censored runs are left out (a mild bias toward shorter runs).
    """
    as_of = state["as_of"]
    rows = []
    for run_id, entry in pool.runs.items():
        meta = entry[0]
        if run_id == state["run_id"] or meta["ended_at"] >= as_of or meta["end_reason"] != "stop":
            continue
        cat = meta["category"] or "other"
        med = prior_median({"category": cat, "effort": meta["effort"], "prompt_len": meta["prompt_len"] or 0})
        age_days = max(0.0, (as_of - meta["ended_at"]) / 86400.0)
        rows.append((cat, math.log(max(meta["run_active_s"] or 0.0, 1.0) / med), 0.5 ** (age_days / half_life)))
    if not rows:
        return NO_CALIBRATION
    mu = sum(w * r for _, r, w in rows) / (sum(w for _, _, w in rows) + CAL_N0)
    by_cat = {}
    for cat, r, w in rows:
        acc = by_cat.setdefault(cat, [0.0, 0.0])
        acc[0] += w * r
        acc[1] += w
    mu_cat = {c: (sr + CAL_N1 * mu) / (sw + CAL_N1) for c, (sr, sw) in by_cat.items()}
    ss = sum(w * (r - mu_cat[c]) ** 2 for c, r, w in rows)
    sigma = math.sqrt((ss + CAL_N0 * SIGMA ** 2) / (sum(w for _, _, w in rows) + CAL_N0))
    return mu_cat, mu, sigma, len(rows)


def prior_survival(state, cal=NO_CALIBRATION):
    mu_cat, mu, sigma, _ = cal
    med = prior_median(state) * math.exp(mu_cat.get(state["category"], mu))
    S = lognormal_survival(med, sigma, state["active_s"])
    k = PHASE_MULT.get(state["phase"], 1.0) * (1.0 + 0.15 * min(state["n_verify_fail"], 4))
    return stretch(S, k)


def tool_survival(conn, state):
    inf = state.get("in_flight")
    if not inf:
        return None, None
    rows = conn.execute(
        "SELECT duration_ms, repo FROM tool_calls WHERE cmd_key=? AND duration_ms IS NOT NULL"
        " AND ended_at < ? ORDER BY id DESC LIMIT 80", (inf["cmd_key"], state["as_of"])).fetchall()
    same = [r["duration_ms"] / 1000.0 for r in rows if r["repo"] == state["repo"]]
    durs = same if len(same) >= 3 else [r["duration_ms"] / 1000.0 for r in rows]
    typical = _median(durs) if durs else None
    running = inf["running_s"]
    prior_med = typical if (typical and len(durs) >= 3) else TOOL_PRIOR_S.get(
        inf["cmd_kind"], TOOL_PRIOR_S.get(inf["kind"], 10))
    prior = lognormal_survival(prior_med, 1.0, running)
    samples = [(d - running, 1.0, True) for d in durs if d > running]
    if len(samples) >= 2:
        S = mix(km_survival(samples, tail=prior), prior, len(samples) / (len(samples) + 2.0))
    else:
        S = prior
    return S, typical


def permission_hazard(conn, state):
    """(hazard per second, p_prompt per tool call) for the current permission mode."""
    mode = state["mode"] or "default"
    if mode in ("bypassPermissions", "dontAsk"):
        return 0.0, 0.0
    p0 = PERM_P0.get(mode, 0.08)
    since = state["as_of"] - 30 * 86400
    n_perm = n_tools = 0
    for repo_filter in (True, False):
        sql = ("SELECT COALESCE(SUM(n_perm),0), COALESCE(SUM(n_tools),0) FROM runs WHERE source='live'"
               " AND permission_mode=? AND ended_at IS NOT NULL AND ended_at > ? AND ended_at < ?")
        args = [mode, since, state["as_of"]]
        if repo_filter:
            sql += " AND repo=?"
            args.append(state["repo"])
        n_perm, n_tools = conn.execute(sql, args).fetchone()
        if n_tools >= 40:
            break
    p = (n_perm + 20.0 * p0) / (n_tools + 20.0)
    if state["n_tools"] >= 3 and state["active_s"] > 20:
        rate = state["n_tools"] / state["active_s"]
    else:
        row = conn.execute(
            "SELECT COALESCE(SUM(n_tools),0), COALESCE(SUM(active_s),0) FROM runs WHERE ended_at > ?"
            " AND ended_at < ? AND active_s > 0", (since, state["as_of"])).fetchone()
        rate = row[0] / row[1] if row[1] > 60 else 1.0 / 15.0
    return rate * p, p


# ---------------------------------------------------------------- main entry

def tuned_values(conn):
    """Settings chosen for this history by tuning.py: {"k0", "safe_k", "half_life", "recal"} (missing until
    tuned; "recal" is 1 when the quantile recalibration is on)."""
    try:
        return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM tuning WHERE value IS NOT NULL")}
    except sqlite3.OperationalError:  # a database opened read-only by an older version
        return {}


def leave_supported(conn):
    """"Safe to leave" is only promised once the estimator has been tuned on your history (tuning.py)."""
    return "k0" in tuned_values(conn)


def tuned_recal(conn):
    """The quantile recalibration learned by tuning.py, or None when it is off."""
    try:
        row = conn.execute("SELECT detail FROM tuning WHERE key='recal' AND value=1").fetchone()
        return json.loads(row[0]) if row and row[0] else None
    except (sqlite3.OperationalError, ValueError):
        return None


def recal_level(recal, part, q):
    """Level at which to read the `part` curve ("done" / "attn") so that, on your past runs, a fraction q
    of outcomes fell below it. `recal` holds the empirical quantiles ("knots", evenly spaced levels) of
    PIT values (1 - S(actual)) from replayed history, shrunk toward q by RECAL_M0 pseudo-points. A
    calibrated model has PIT uniform, and then the level stays q."""
    if not recal or not recal.get(part):
        return q
    knots = recal[part]
    x = q * (len(knots) - 1)
    i = min(int(x), len(knots) - 2)
    emp = knots[i] + (knots[i + 1] - knots[i]) * (x - i)
    m = recal.get("n_" + part) or 0
    return min(0.995, max(0.005, (m * emp + RECAL_M0 * q) / (m + RECAL_M0)))


def pit_knots(values, k=21):
    """Empirical quantiles of PIT values at k evenly spaced levels (the stored form of a recalibration)."""
    v = sorted(values)
    return [round(v[int(round(i * (len(v) - 1) / float(k - 1)))], 4) for i in range(k)]


def tuned_k0(conn):
    """The neighbour-vs-prior strength chosen for this history (default until then)."""
    return tuned_values(conn).get("k0", K0_DEFAULT)


def estimate(conn, state, pool=None, with_permission=True, k0=None, cands=None, safe_k=None, half_life=None,
             recal=None):
    """`k0` / `safe_k` / `half_life` / `recal` override the tuned settings (`recal=False`: raw curves) and
    `cands` reuses pool.candidates(state) (replays)."""
    if pool is None:
        pool = Pool.load(conn, state["as_of"])
    if k0 is None or safe_k is None or half_life is None or recal is None:
        tuned = tuned_values(conn)
        k0 = tuned.get("k0", K0_DEFAULT) if k0 is None else k0
        safe_k = tuned.get("safe_k", SAFE_K_DEFAULT) if safe_k is None else safe_k
        half_life = tuned.get("half_life", RECENCY_HALF_LIFE_D) if half_life is None else half_life
        if recal is None:
            recal = tuned_recal(conn) if tuned.get("recal") else False
    cal = calibration(pool, state, half_life)
    prior_done = prior_survival(state, cal)
    prior_attn = with_hazard(prior_done, 1.0 / 3600.0)
    if cands is None:
        cands = pool.candidates(state)
    reasons = []

    # Three-level shrinkage: similar runs -> all your runs at a similar stage -> calibrated prior.
    S_done, S_attn = prior_done, prior_attn
    ess_local = ess_global = 0.0
    history = None
    if cands:
        rec = [0.5 ** (c[2] / half_life) for c in cands]
        w_glob = adaptive_weights([c[1] for c in cands], rec, 5, 40)
        w_loc = adaptive_weights([c[0] + c[1] for c in cands], rec, 3, 25)
        ess_global, ess_local = _ess(w_glob), _ess(w_loc)
        for ws, e in ((w_glob, ess_global), (w_loc, ess_local)):
            w_layer = e / (e + k0)
            if w_layer <= 0.0:
                continue
            done_km = km_survival([(c[3].rem_done_s, w, not c[3].censored) for w, c in zip(ws, cands)],
                                  tail=S_done)
            attn_km = km_survival([(c[3].rem_attn_s, w, c[3].attn_event) for w, c in zip(ws, cands)],
                                  tail=S_attn)
            if done_km is not None:
                S_done = mix(done_km, S_done, w_layer)
                S_attn = mix(attn_km, S_attn, w_layer)
        top = max(w_loc) if w_loc else 0.0
        close = [c for w, c in zip(w_loc, cands) if w >= 0.2 * top]
        history = {"n": len(close), "ess": ess_local, "same": sum(1 for c in close if c[4]["repo"] == state["repo"])}
    w_learn = ess_global / (ess_global + k0)
    if history is not None and w_learn >= 0.1:
        reasons.append(("history", history))
    if cal[3] >= CAL_MIN_RUNS:
        reasons.append(("pace", {"x": math.exp(cal[0].get(state["category"], cal[1])), "n": cal[3]}))
    elif not reasons:
        reasons.append(("prior", {"n": len(pool.runs)}))
    S_attn = cmin(S_attn, S_done)  # you are needed at the latest when it is done

    if state["phase"] == "verify":
        reasons.append(("phase_verify", {}))
    if state["n_verify_fail"]:
        reasons.append(("fails", {"n": state["n_verify_fail"]}))

    total, done = state["plan_total"], state["plan_done"]
    if total >= 2 and 1 <= done < total and state["active_at_last_done"]:
        # B1 of the Codex design / task-progress-bar: EMA of step-completion intervals x remaining steps
        step = state.get("plan_ema") or state["active_at_last_done"] / done
        since = state["active_s"] - state["active_at_last_done"]
        rem = max(step * 0.3, (total - done) * step - since)
        S_done = mix(lognormal_survival(rem, 0.6), S_done, 0.5 * done / total)
        S_attn = cmin(S_attn, S_done)
    if total:
        reasons.append(("plan", {"done": done, "total": total}))
        first = state["plan_first_total"]
        if first and total > first:
            reasons.append(("plan_grew", {"first": first, "total": total}))

    tool_med_rem = 0.0
    S_tool, typical = tool_survival(conn, state)
    if S_tool is not None:
        S_done = cmax(S_done, S_tool)
        S_attn = cmax(S_attn, S_tool)
        tool_med_rem = quantile(S_tool, 0.5) or 0.0
        inf = state["in_flight"]
        reasons.append(("inflight", {"what": inf["detail"] or inf["tool"], "running": inf["running_s"],
                                     "typical": typical}))

    if state["agents_running"]:
        reasons.append(("agents", {"n": state["agents_running"]}))
    if state["status"] == "background":
        reasons.append(("background", {"n": state["bg_count"]}))

    p_prompt = 0.0
    if with_permission:
        h, p_prompt = permission_hazard(conn, state)
        if h > 0:
            S_attn = with_hazard(S_attn, h, tool_med_rem)
            if p_prompt >= 0.03:
                reasons.append(("permission", {"every": max(1, int(round(1.0 / p_prompt)))}))

    lv = {(part, q): recal_level(recal, part, q) for part, q in (("done", 0.5), ("done", 0.8), ("attn", 0.2),
                                                                 ("attn", 0.5))}
    qd = quantiles(S_done, [lv["done", 0.5], lv["done", 0.8]])
    qa = quantiles(S_attn, [lv["attn", 0.2], lv["attn", 0.5]])
    done50, done80 = qd[lv["done", 0.5]], qd[lv["done", 0.8]]
    if w_learn >= 0.75:
        method = "learned"
    elif w_learn >= 0.25:
        method = "blended"
    else:
        method = "calibrated" if cal[3] >= CAL_MIN_RUNS else "prior"
    ratio = (done80 or 1e9) / max(done50 or 1.0, 1.0)
    if method == "prior":
        confidence = "low"
    elif method == "calibrated":
        confidence = "medium" if cal[3] >= 20 and ratio <= 3.0 else "low"
    elif ess_local >= 10 and ratio <= 2.5:
        confidence = "high"
    elif ess_local >= 3:
        confidence = "medium"
    else:
        confidence = "low"
    active = state["active_s"]
    return {
        "done_p50": done50, "done_p80": done80,
        "attn_p20": qa[lv["attn", 0.2]], "attn_p50": qa[lv["attn", 0.5]],
        "S_attn": S_attn, "S_done": S_done,
        "progress": active / (active + done50) if done50 is not None and active + done50 > 0 else None,
        "ess": ess_local, "ess_global": ess_global, "n_neighbours": len(cands), "method": method, "confidence": confidence,
        "p_prompt": p_prompt, "reasons": reasons, "k0": k0, "safe_k": safe_k, "cal_runs": cal[3],
        "half_life": half_life, "recal": recal or None,
    }


SAFE_K_DEFAULT = 6.0


def safe_seconds(est, q, k=None):
    """How long you can step away: a low quantile of time-to-attention.

    Promising too much is worse than promising too little (you walk away and Claude sits waiting),
    and small or shifting histories overstate low quantiles. So the quantile is shrunk toward 0 while
    evidence is thin: q_eff = q * (ESS + 1) / (ESS + 1 + k). k (SAFE_K) is chosen per user by
    tuning.py: the most generous value whose replayed breach rate stays under q, tightened further
    when the leave windows actually shown were breached too often.
    """
    k = est.get("safe_k", SAFE_K_DEFAULT) if k is None else k
    n = est.get("ess_global") or 0.0
    return quantile(est["S_attn"], recal_level(est.get("recal"), "attn", q * (n + 1.0) / (n + 1.0 + k)))


def b0_estimate(pool, state, qs=(0.5, 0.8)):
    """Codex design's B0: conditional survival of total run length, S(e + r) / S(e), over past runs with
    the same model and permission class. No trajectory features, no prior, no tail extrapolation:
    quantiles the history cannot identify stay None (the baseline abstains)."""
    samples = []
    for run_id, (meta, snaps, _, _) in pool.runs.items():
        if run_id == state["run_id"] or meta["ended_at"] >= state["as_of"]:
            continue
        if state["model"] and meta["model"] and meta["model"] != state["model"]:
            continue
        if F.perm_class(meta["permission_mode"]) != F.perm_class(state["mode"]):
            continue
        first = Snap._make(snaps[0])  # the earliest snapshot: run length = its elapsed + its remaining time
        samples.append((first.active_s + first.rem_done_s, 1.0, not first.censored))
    km = KaplanMeier(samples)
    return {q: km.conditional_quantile(state["active_s"], q) for q in qs}
