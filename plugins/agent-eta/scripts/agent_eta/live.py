"""Glue for the live views: status line, `status`, `watch`."""
import json
import sys
import time

from . import config, ingest, render, store
from . import estimator as E


def compute(conn, run, now, cfg, log=True):
    """Estimate for a running turn. `log` (status line only) records the prediction and may issue a
    leave window; the read-only views (`status`, `watch`) never create windows."""
    state = E.state_from_run(conn, run, now)
    est = E.estimate(conn, state)
    est["safe_s"] = E.safe_seconds(est, cfg["safe_quantile"])
    if not E.leave_supported(conn):
        est["safe_s"], est["leave_withheld"] = None, True  # no promise rather than an unreliable one
    est["cold_unknown"] = cfg.get("cold_start") == "unknown" and est["method"] == "prior"
    trend = _trend(conn, run, state, est, now)
    if log:
        _log_prediction(conn, run, state, est, now, cfg)
    est["window"] = _window(conn, run, state, est, now, issue=log and not est["cold_unknown"])
    est["validated"] = ledger_validated(conn, cfg, now) if est["window"] else False
    return state, est, trend


def _window(conn, run, state, est, now, issue):
    """The "safe to leave" promise actually shown (Codex design INV-13).

    Once issued its expiry is fixed: refreshing never silently pushes it later. A new window is only
    issued after the old one expired or was revoked. Revoking (you are needed, or the estimate has
    collapsed) never deletes the record: every issued window is scored against what happened.
    """
    w = conn.execute("SELECT * FROM windows WHERE run_id=? AND revoked_at IS NULL AND expires_at > ?"
                     " ORDER BY id DESC LIMIT 1", (run["id"], now)).fetchone()
    bucket = render.safe_bucket(est["safe_s"])
    if w is not None:
        reason = None
        if state["attn"]:
            reason = "attention"
        elif bucket is None or (est["safe_s"] or 0) < 0.5 * (w["expires_at"] - now):
            reason = "estimate_dropped"
        if reason is None:
            return w
        if issue:
            with store.tx(conn):
                conn.execute("UPDATE windows SET revoked_at=?, revoke_reason=? WHERE id=?", (now, reason, w["id"]))
        return None
    if not issue or state["attn"] or bucket is None:
        return None
    with store.tx(conn):
        cur = conn.execute(
            "INSERT INTO windows(run_id, issued_at, horizon_s, expires_at, safe_s, ess) VALUES (?,?,?,?,?,?)",
            (run["id"], now, bucket, now + bucket, est["safe_s"], est.get("ess_global")))
    return conn.execute("SELECT * FROM windows WHERE id=?", (cur.lastrowid,)).fetchone()


def window_outcomes(conn, now, since=0.0):
    """Score every issued window: breach if you were needed (attention or Claude finishing) before its
    expiry; unknown if the run ended some other way first (counted as adverse, never dropped)."""
    out = []
    for w in conn.execute(
            "SELECT w.*, r.ended_at, r.end_reason, (SELECT MIN(a.at) FROM attention a WHERE a.run_id = w.run_id"
            " AND a.at >= w.issued_at) AS first_attn FROM windows w JOIN runs r ON r.id = w.run_id"
            " WHERE w.issued_at > ?", (since,)).fetchall():
        first = w["first_attn"]
        needed = [x for x in (first, w["ended_at"] if w["end_reason"] == "stop" else None)
                  if x is not None and x >= w["issued_at"]]
        hit = min(needed) if needed else None
        if hit is not None and hit < w["expires_at"]:
            out.append((w, "breach"))
        elif w["ended_at"] is not None and w["ended_at"] < w["expires_at"]:
            out.append((w, "unknown"))
        elif w["ended_at"] is None and now < w["expires_at"]:
            out.append((w, "pending"))
        else:
            out.append((w, "held"))
    return out


LEDGER_MIN_TRIALS = 100


def ledger_stats(conn, now, cfg, since=0.0):
    scored = [o for _, o in window_outcomes(conn, now, since) if o != "pending"]
    n = len(scored)
    adverse = sum(1 for o in scored if o in ("breach", "unknown"))
    upper = E.clopper_pearson_upper(n, adverse, 0.05) if n else None
    return {"n": n, "breach": scored.count("breach"), "unknown": scored.count("unknown"),
            "held": scored.count("held"), "upper": upper,
            "validated": n >= LEDGER_MIN_TRIALS and upper is not None and upper <= cfg["safe_quantile"]}


def ledger_validated(conn, cfg, now):
    return ledger_stats(conn, now, cfg)["validated"]


def _trend(conn, run, state, est, now):
    prev = conn.execute(
        "SELECT active_s, done_p50, n_fail, plan_total FROM predictions WHERE run_id=? AND t < ?"
        " ORDER BY t DESC LIMIT 1", (run["id"], now - 60)).fetchone()
    if prev is None or prev["done_p50"] is None or est["done_p50"] is None:
        return None
    before = prev["active_s"] + prev["done_p50"]
    after = state["active_s"] + est["done_p50"]
    trend = None
    if after > before * 1.25 and after - before > 30:
        trend = "up"
    elif after < before * 0.8 and before - after > 30:
        trend = "down"
    if trend:
        extra = [("trend_" + trend, {})]
        new_fails = state["n_fail"] - (prev["n_fail"] or 0)
        if trend == "up" and new_fails > 0:
            extra.append(("new_fails", {"n": new_fails}))
        est["reasons"] = extra + est["reasons"]
    return trend


def _log_prediction(conn, run, state, est, now, cfg):
    last = conn.execute("SELECT MAX(t) FROM predictions WHERE run_id=?", (run["id"],)).fetchone()[0]
    if last is not None and now - last < cfg["prediction_log_interval_s"]:
        return
    if state["attn"]:
        return
    with store.tx(conn):
        conn.execute(
            "INSERT INTO predictions(run_id, t, active_s, done_p50, done_p80, safe_s, attn_p50, ess, method,"
            " n_fail, plan_total) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run["id"], now, state["active_s"], est["done_p50"], est["done_p80"], est["safe_s"],
             est["attn_p50"], est["ess"], est["method"], state["n_fail"], state["plan_total"]))


def _idle(conn, session_id, lang, paint):
    n = conn.execute("SELECT COUNT(*) FROM runs WHERE end_reason='stop' AND active_s IS NOT NULL").fetchone()[0]
    last = None
    row = conn.execute(
        "SELECT id, active_s FROM runs WHERE session_id=? AND ended_at IS NOT NULL AND category != 'wake'"
        " ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
    if row is not None:
        pred = conn.execute(
            "SELECT active_s, done_p50, done_p80 FROM predictions WHERE run_id=? ORDER BY t LIMIT 1",
            (row["id"],)).fetchone()
        last = {"active_s": row["active_s"], "p50": pred["done_p50"] if pred else None,
                "p80": pred["done_p80"] if pred else None,
                "actual_from_pred": (row["active_s"] or 0) - (pred["active_s"] if pred else 0)}
    return render.idle_line(last, n, lang, paint)


def _start_wrapped(cfg, stdin_text):
    """Start the status line we wrap; it runs while our own segment is computed."""
    cmd = cfg.get("wrap_command")
    if not cmd:
        return None
    import subprocess
    import threading
    result = {"out": ""}

    def run():
        try:
            out = subprocess.run(cmd, shell=True, input=stdin_text, capture_output=True, text=True, timeout=4)
            result["out"] = out.stdout.rstrip("\n")
        except Exception:
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def _finish_wrapped(job):
    if job is None:
        return ""
    thread, result = job
    thread.join(5)
    return result["out"]


def statusline_main(stdin_text, now):
    cfg = config.load()
    lang = config.lang(cfg)
    paint = render.Paint(True)
    try:
        data = json.loads(stdin_text) if stdin_text.strip() else {}
    except ValueError:
        data = {}
    job = _start_wrapped(cfg, stdin_text)
    ours = ""
    try:
        ours = _ours(data, now, cfg, lang, paint)
    except Exception:
        store.log_error("statusline")
    wrapped = _finish_wrapped(job)
    if wrapped and ours:
        if cfg.get("compose") == "inline":
            first, _, rest = wrapped.partition("\n")
            out = first + paint.dim(" │ ") + ours + ("\n" + rest if rest else "")
        else:
            out = wrapped + "\n" + ours
    else:
        out = wrapped or ours
    if out:
        sys.stdout.write(out + "\n")


def _ours(data, now, cfg, lang, paint):
    conn = store.connect(create=False)
    if conn is None:
        return paint.dim("⏱ ETA: no data yet")
    sid = data.get("session_id")
    model = (data.get("model") or {}).get("id") if isinstance(data.get("model"), dict) else None
    effort = (data.get("effort") or {}).get("level") if isinstance(data.get("effort"), dict) else None
    with store.tx(conn):
        if sid and (model or effort):
            conn.execute("UPDATE sessions SET model=COALESCE(?, model), effort=COALESCE(?, effort) WHERE session_id=?",
                         (model, effort, sid))
            conn.execute("UPDATE runs SET model=COALESCE(model, ?), effort=COALESCE(effort, ?)"
                         " WHERE session_id=? AND ended_at IS NULL", (model, effort, sid))
        ingest.expire_stale(conn, now)
    run = conn.execute("SELECT * FROM runs WHERE session_id=? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
                       (sid,)).fetchone() if sid else None
    if run is None:
        return _idle(conn, sid, lang, paint) if cfg.get("show_idle", True) and sid else ""
    state, est, trend = compute(conn, run, now, cfg)
    return render.statusline(state, est, lang, now, cfg["safe_quantile"], paint, trend)


def active_runs(conn, exclude_session=None, session=None):
    sql = "SELECT * FROM runs WHERE ended_at IS NULL"
    args = []
    if session:
        sql += " AND session_id=?"
        args.append(session)
    if exclude_session:
        sql += " AND session_id != ?"
        args.append(exclude_session)
    return conn.execute(sql + " ORDER BY started_at", args).fetchall()


def status_text(now, session=None, exclude_session=None, color=True):
    cfg = config.load()
    lang = config.lang(cfg)
    paint = render.Paint(color)
    conn = store.connect()
    with store.tx(conn):
        ingest.expire_stale(conn, now)
    runs = active_runs(conn, exclude_session, session)
    if not runs:
        return render.S(lang)["detail"]["none"]
    blocks = []
    for run in runs:
        state, est, trend = compute(conn, run, now, cfg, log=False)
        label = run["session_id"][:8]
        cwd = run["cwd"] or ""
        if cwd:
            label += " · " + cwd
        blocks.append(render.detail(state, est, lang, now, cfg["safe_quantile"], paint, label, trend))
    return ("\n\n" + paint.dim("─" * 60) + "\n\n").join(blocks)


def watch(interval, exclude_session=None):
    try:
        while True:
            text = status_text(time.time(), exclude_session=exclude_session, color=sys.stdout.isatty())
            sys.stdout.write("\033[2J\033[H" + text + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0
