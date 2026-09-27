"""SQLite storage. One small DB, WAL mode, safe for concurrent async hook processes."""
import contextlib
import os
import sqlite3
import time

from . import config

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY,
  source TEXT,                 -- live | backfill
  cwd TEXT, repo TEXT, model TEXT, effort TEXT, permission_mode TEXT, transcript_path TEXT,
  started_at REAL, updated_at REAL, ended_at REAL
);

-- One run = one user prompt until Claude hands control back (Stop), or is interrupted.
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  prompt_id TEXT,
  source TEXT,                 -- live | backfill
  started_at REAL NOT NULL,
  ended_at REAL,
  end_reason TEXT,             -- stop | stop_failure | interrupted | superseded | session_end | abandoned
  status TEXT,                 -- running | attention | background | done
  repo TEXT, cwd TEXT, model TEXT, effort TEXT, permission_mode TEXT,
  prompt_len INTEGER, prompt_lines INTEGER, prompt_code INTEGER, category TEXT, prompt_preview TEXT,
  n_tools INTEGER DEFAULT 0, n_explore INTEGER DEFAULT 0, n_edit INTEGER DEFAULT 0,
  n_shell INTEGER DEFAULT 0, n_verify INTEGER DEFAULT 0, n_fail INTEGER DEFAULT 0,
  n_verify_fail INTEGER DEFAULT 0, n_agents INTEGER DEFAULT 0, agents_running INTEGER DEFAULT 0,
  n_attn INTEGER DEFAULT 0, n_perm INTEGER DEFAULT 0, n_snaps INTEGER DEFAULT 0,
  plan_total INTEGER DEFAULT 0, plan_done INTEGER DEFAULT 0, plan_first_total INTEGER,
  active_at_last_done REAL,
  bg_count INTEGER DEFAULT 0,
  last_edit_at REAL, last_verify_at REAL, last_verify_ok INTEGER, last_event_at REAL,
  human_wait_s REAL DEFAULT 0,
  attn_kind TEXT, attn_since REAL, attn_detail TEXT, attn_agent TEXT, attn_tool TEXT,
  plan_ema REAL,
  first_attn_at REAL,
  active_s REAL                -- label: agent-active seconds (wall time minus time waiting on you)
);
CREATE INDEX IF NOT EXISTS runs_session ON runs(session_id, ended_at);
CREATE INDEX IF NOT EXISTS runs_open ON runs(ended_at);

CREATE TABLE IF NOT EXISTS tool_calls(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, session_id TEXT, tool_use_id TEXT, agent_id TEXT,
  tool TEXT, kind TEXT, cmd_key TEXT, cmd_kind TEXT, repo TEXT, detail TEXT,
  started_at REAL, ended_at REAL, duration_ms REAL, ok INTEGER
);
CREATE INDEX IF NOT EXISTS tool_calls_run ON tool_calls(run_id, ended_at);
CREATE INDEX IF NOT EXISTS tool_calls_tuid ON tool_calls(tool_use_id);
CREATE INDEX IF NOT EXISTS tool_calls_key ON tool_calls(cmd_key, repo);

-- Trajectory state after each step; labels (remaining time) are filled when the run ends.
CREATE TABLE IF NOT EXISTS snapshots(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL, t REAL, active_s REAL,
  n_tools INTEGER, n_edit INTEGER, n_verify INTEGER, n_fail INTEGER, n_verify_fail INTEGER,
  phase TEXT, plan_total INTEGER, plan_done INTEGER, agents_running INTEGER,
  rem_done_s REAL, censored INTEGER, rem_attn_s REAL, attn_event INTEGER
);
CREATE INDEX IF NOT EXISTS snapshots_run ON snapshots(run_id);

CREATE TABLE IF NOT EXISTS attention(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, session_id TEXT, agent_id TEXT, kind TEXT, detail TEXT,
  at REAL, active_s REAL, resolved_at REAL, tool_use_id TEXT
);
CREATE INDEX IF NOT EXISTS attention_run ON attention(run_id, at);

-- What we predicted, so accuracy can be checked against what really happened.
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, t REAL, active_s REAL,
  done_p50 REAL, done_p80 REAL, safe_s REAL, attn_p50 REAL, ess REAL, method TEXT,
  n_fail INTEGER, plan_total INTEGER
);
CREATE INDEX IF NOT EXISTS predictions_run ON predictions(run_id, t);

-- Every "safe to leave" window actually shown: fixed expiry, never silently extended, never deleted.
CREATE TABLE IF NOT EXISTS windows(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER, issued_at REAL, horizon_s REAL, expires_at REAL, safe_s REAL, ess REAL,
  revoked_at REAL, revoke_reason TEXT
);
CREATE INDEX IF NOT EXISTS windows_run ON windows(run_id, issued_at);

CREATE TABLE IF NOT EXISTS imported(
  path TEXT PRIMARY KEY, session_id TEXT, mtime REAL, runs INTEGER, at REAL
);

-- Estimator settings chosen by replaying your own history (tuning.py).
CREATE TABLE IF NOT EXISTS tuning(
  key TEXT PRIMARY KEY, value REAL, n_runs INTEGER, n_points INTEGER, at REAL, pending_at REAL, detail TEXT
);
"""


def db_path():
    return config.path("eta.db")


def connect(create=True):
    path = db_path()
    if not create and not os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)  # CREATE ... IF NOT EXISTS: creates what is missing, keeps data
        _migrate(conn)
        conn.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
        try:
            os.chmod(config.home(), 0o700)
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.chmod(path + suffix, 0o600)
        except OSError:
            pass
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


# Columns added after v1; ALTER only what an older database lacks (idempotent under concurrent opens).
_ADDED_COLUMNS = [("runs", "attn_tool", "TEXT"), ("runs", "plan_ema", "REAL"), ("attention", "tool_use_id", "TEXT")]


def _migrate(conn):
    for table, column, kind in _ADDED_COLUMNS:
        have = {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}
        if column not in have:
            try:
                conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, kind))
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e):
                    raise


@contextlib.contextmanager
def tx(conn):
    """Serialize writers: every hook handler runs inside one IMMEDIATE transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def log_error(where):
    """Never let a failure reach Claude Code; keep a small local log instead."""
    try:
        import traceback
        os.makedirs(config.home(), exist_ok=True)
        p = config.path("error.log")
        if os.path.exists(p) and os.path.getsize(p) > 512 * 1024:
            os.replace(p, p + ".1")
        with open(p, "a", encoding="utf-8") as f:
            f.write("---- %s %s\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), where, traceback.format_exc()))
    except Exception:
        pass
