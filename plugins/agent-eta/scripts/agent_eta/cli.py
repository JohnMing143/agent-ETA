"""Command line: `eta.py <command>`. Hook and statusline modes never fail loudly.

Hooks and the status line run many times a minute, so they import only what they use.
"""
import json
import sys
import time

from . import config, store


def _hook(t0):
    try:
        raw = sys.stdin.read()
        ev = json.loads(raw) if raw.strip() else None
        if isinstance(ev, dict):
            from . import ingest
            conn = store.connect()
            ingest.handle(conn, ev, t0)
            if ev.get("hook_event_name") == "Stop":
                from . import tuning
                tuning.maybe_spawn(conn, t0)  # a run finished: re-tune in the background when due
    except Exception:
        store.log_error("hook")
    return 0  # print nothing: stdout of some events is injected into Claude's context


def _statusline(t0):
    try:
        text = sys.stdin.read()
    except Exception:
        text = ""
    try:
        from . import live
        live.statusline_main(text, t0)
    except Exception:
        store.log_error("statusline")
    return 0


def main(argv, t0=None):
    t0 = t0 or time.time()
    if argv and argv[0] == "hook":
        return _hook(t0)
    if argv and argv[0] == "statusline":
        return _statusline(t0)

    import argparse
    ap = argparse.ArgumentParser(prog="eta.py", description="Agent ETA for Claude Code")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("status", help="detailed view of running turns")
    p.add_argument("--session")
    p.add_argument("--exclude-session")
    p.add_argument("--no-color", action="store_true")
    p = sub.add_parser("watch", help="live dashboard of all running sessions")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--exclude-session")
    p = sub.add_parser("report", help="history and accuracy report")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--no-backtest", action="store_true")
    p = sub.add_parser("backfill", help="import past transcripts")
    p.add_argument("--dir")
    p.add_argument("--skip-session")
    p.add_argument("--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("setup", help="install the status line (and import history)")
    p.add_argument("--lang", choices=["zh", "en", "auto"])
    p.add_argument("--refresh", type=int, default=5)
    p.add_argument("--no-backfill", action="store_true")
    p.add_argument("--skip-session")
    p.add_argument("--remove", action="store_true")
    p = sub.add_parser("tune", help="choose the estimator's shrinkage by replaying your history")
    p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("eval", help="is the learning paying off? (prequential, with 95%% intervals)")
    p.add_argument("--curve", action="store_true", help="also the learning curve: marginal value of more history")
    p.add_argument("--json", action="store_true")
    sub.add_parser("doctor", help="check that everything is wired up")
    args = ap.parse_args(argv)

    if args.cmd == "status":
        from . import live
        print(live.status_text(time.time(), args.session, args.exclude_session,
                               color=not args.no_color and sys.stdout.isatty()))
        return 0
    if args.cmd == "watch":
        from . import live
        return live.watch(args.interval, args.exclude_session)
    if args.cmd == "report":
        from . import report
        cfg = config.load()
        print(report.build(store.connect(), config.lang(cfg), args.days, not args.no_backtest,
                           cfg["safe_quantile"]))
        return 0
    if args.cmd == "backfill":
        from . import backfill
        details = []
        files, runs = backfill.run(args.dir, args.skip_session, args.force, details)
        if args.verbose:
            for path, n, why in details:
                print("%4d  %-26s %s" % (n, why, path))
        print("imported %d runs from %d transcripts" % (runs, files))
        return 0
    if args.cmd == "setup":
        from . import setup
        if args.remove:
            return setup.remove()
        return setup.install(args.lang, args.refresh, not args.no_backfill, args.skip_session)
    if args.cmd == "tune":
        from . import tuning
        try:
            res = tuning.tune(store.connect())
        except Exception:
            store.log_error("tune")
            raise
        if not args.quiet:
            print(tuning.describe(res))
        return 0
    if args.cmd == "eval":
        from . import evaluate
        cfg = config.load()
        res = evaluate.evaluate(store.connect(), cfg["safe_quantile"], curve=args.curve)
        print(evaluate.to_json(res) if args.json else evaluate.describe(res, config.lang(cfg)))
        return 0
    if args.cmd == "doctor":
        from . import setup
        return setup.doctor()
    ap.print_help()
    return 0
