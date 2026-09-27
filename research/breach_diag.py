"""Why are leave promises breached? For each history (tuned settings, replay points), classify promises:
held / breached because the run finished / breached because Claude asked for input, and how short."""
import math
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("PLUGIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plugins", "agent-eta", "scripts"))
from agent_eta import estimator as E, render, store, tuning  # noqa: E402


def history_name(home):
    """A label for a history directory: the parent's name for .../<name>/home, else its own name."""
    p = os.path.abspath(home.rstrip("/"))
    base = os.path.basename(p)
    return os.path.basename(os.path.dirname(p)) if base == "home" else base

for home in sys.argv[1:]:
    os.environ["AGENT_ETA_HOME"] = home
    conn = store.connect(create=False)
    if conn is None:
        sys.exit("no eta.db found: point AGENT_ETA_HOME (or the HOME argument) at an imported history, see research/README.md")
    cfg = tuning.current_config(conn)
    tuned = E.tuned_values(conn)
    recal = E.tuned_recal(conn) if tuned.get("recal") else None
    pts = tuning.replay(conn, max_runs=100000, configs=[cfg], keep=True, with_baselines=False, pool_limit=100000)
    rows = []
    for p in pts:
        if p["attn"] is None:
            continue
        est = dict(p["est"][cfg]["est"], recal=recal)
        b = render.safe_bucket(E.safe_seconds(est, 0.2, tuned.get("safe_k", 6.0)))
        if not b:
            continue
        run = conn.execute("SELECT * FROM runs WHERE id=?", (p["run"],)).fetchone()
        asked = conn.execute("SELECT MIN(at) FROM attention WHERE run_id=? AND at>=?", (p["run"], p["as_of"])).fetchone()[0]
        breach = p["attn"] < b
        cause = "held" if not breach else ("asked" if asked is not None and asked - p["as_of"] < b + 1 else "finished")
        elapsed = run["active_s"] - (p["done"] or 0) if p["done"] is not None else None
        rows.append((cause, b, p["attn"], p["done"], elapsed, run["category"], run["active_s"]))
    name = history_name(home)
    if not rows:
        print("-- %s: no promises" % name)
        continue
    br = [r for r in rows if r[0] != "held"]
    print("-- %-40s promises %3d  breached %3d (%2.0f%%): finished early %d, asked you %d" % (
        name, len(rows), len(br), 100.0 * len(br) / len(rows), sum(r[0] == "finished" for r in br), sum(r[0] == "asked" for r in br)))
    if br:
        ratio = [r[2] / r[1] for r in br]
        print("     breached: needed after %.0f%% of the promised time (median); promised %s; elapsed at promise %s" % (
            100 * statistics.median(ratio), statistics.median(r[1] for r in br),
            statistics.median(r[4] for r in br if r[4] is not None) if any(r[4] is not None for r in br) else "-"))
    held = [r for r in rows if r[0] == "held"]
    if held:
        print("     held:     promised %s median; elapsed at promise %s; run length %s" % (
            statistics.median(r[1] for r in held),
            statistics.median(r[4] for r in held if r[4] is not None) if any(r[4] is not None for r in held) else "-",
            statistics.median(r[6] for r in held)))
    # how the breach rate depends on how long the run had already gone
    for lo, hi in ((0, 60), (60, 300), (300, 1e9)):
        sel = [r for r in rows if r[4] is not None and lo <= r[4] < hi]
        if sel:
            print("     elapsed %4.0f-%-6s: %3d promises, %2.0f%% breached" % (
                lo, "∞" if hi > 1e8 else "%.0fs" % hi, len(sel), 100.0 * sum(r[0] != "held" for r in sel) / len(sel)))
