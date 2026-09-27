"""Tests for Agent ETA. Run: python3 -m unittest discover -s tests  (from the plugin root)."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "scripts"), HERE]

from agent_eta import backfill, config, estimator as E, evaluate, features as F, ingest, live, render, report, setup, store, tuning  # noqa: E402
import sim  # noqa: E402

ETA = os.path.join(ROOT, "scripts", "eta.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agent-eta-test-")
        self.env_backup = {k: os.environ.get(k) for k in ("AGENT_ETA_HOME", "HOME", "CLAUDE_EFFORT")}
        os.environ["AGENT_ETA_HOME"] = os.path.join(self.tmp, "eta")
        os.environ["HOME"] = os.path.join(self.tmp, "home")
        os.environ.pop("CLAUDE_EFFORT", None)
        os.makedirs(os.environ["HOME"])
        self.conn = store.connect()

    def tearDown(self):
        self.conn.close()
        for k, v in self.env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ev(self, t, name, **kw):
        e = {"session_id": "s1", "cwd": "/work/api", "hook_event_name": name, "permission_mode": "default"}
        e.update(kw)
        ingest.handle(self.conn, e, t)

    def run_row(self, n=-1):
        return self.conn.execute("SELECT * FROM runs ORDER BY id").fetchall()[n]

    def bash_transcript(self, pid, typed_at, done_at, command="gh auth login"):
        """A transcript holding a `!` shell command, as Claude Code writes it once the command is done."""
        iso = lambda t: "1970-01-01T00:%02d:%06.3fZ" % divmod(t, 60)
        path = os.path.join(self.tmp, "transcript.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for t, text in ((typed_at, "<bash-input>%s</bash-input>" % command),
                            (done_at, "<bash-stdout>done</bash-stdout><bash-stderr></bash-stderr>")):
                f.write(json.dumps({"type": "user", "promptId": pid, "timestamp": iso(t),
                                    "message": {"role": "user", "content": text}}) + "\n")
        return path


class FeatureTests(unittest.TestCase):
    def test_shell_command_info(self):
        cases = {
            "cd web && npm run test -- --watch=false": ("npm run test", "test"),
            "FOO=1 pytest -q tests/": ("pytest", "test"),
            "timeout 60 cargo test": ("cargo test", "test"),
            "tsc --noEmit": ("tsc", "lint"),
            "for f in *.conf; do curl -s http://x/$f; done": ("curl", "other"),
            "python3 - <<'PY'\nprint(1)\nPY": ("python3", "other"),
            "git status && git diff": ("git status", "vcs"),
            "npm ci": ("npm ci", "install"),
            "make": ("make", "build"),
            # here-document bodies are data, not commands
            "cat > /tmp/x.py <<'EOF'\nimport os\nnpm test\nEOF\npython3 /tmp/x.py": ("python3", "other"),
            "cat <<-EOF > f\n\tcontent\n\tEOF\nmake build": ("make build", "build"),
            "sed -n 1p f <<< \"x\"; npm test": ("sed", "test"),
            # the kind comes from the command itself, not from words in its arguments
            "grep -rn pytest src": ("grep", "other"),
            "cat jest.config.js": ("cat", "other"),
            "uv run pytest -x": ("uv run pytest", "test"),
            "npx -y vitest run": ("npx vitest", "test"),
            "python -m pytest tests": ("python -m pytest", "test"),
            "bash -c 'cd a && npm test'": ("npm test", "test"),
            "npm run dev & sleep 3; curl localhost:3000": ("npm run dev", "other"),
            "echo x &> /dev/null; pytest 2>&1 | tail": ("pytest", "test"),
        }
        for cmd, want in cases.items():
            self.assertEqual(F.shell_command_info(cmd), want, cmd)

    def test_command_keys_never_carry_arguments(self):
        risky = [
            "curl -H 'Authorization: Bearer SEKRET' https://api.example.com",
            'bash -c "curl -H \'Authorization: Bearer SEKRET\' https://a/b"',
            "DEPLOY_TOKEN=SEKRET ./deploy.sh", "./deploy.sh SEKRET", "mysql -uroot -pSEKRET prod",
            "git push https://user:SEKRET@github.com/x.git", "python -m 'SEKRET module'",
            "python -c \"print('SEKRET')\"", "echo SEKRET | docker login --password-stdin",
            "export TOKEN=SEKRET && npm publish", "sh -c 'echo SEKRET; aws s3 ls'",
        ]
        for cmd in risky:
            key, _ = F.shell_command_info(cmd)
            self.assertNotIn("SEKRET", key or "", cmd)
            self.assertNotIn("SEKRET", F.safe_summary("Bash", {"command": cmd}), cmd)

    def test_classify_prompt(self):
        self.assertEqual(F.classify_prompt("帮我修一下登录报错"), "fix")
        self.assertEqual(F.classify_prompt("为什么这里会返回旧数据？"), "question")
        self.assertEqual(F.classify_prompt("add a CSV export feature"), "feature")
        self.assertEqual(F.classify_prompt("refactor the auth module"), "refactor")
        self.assertEqual(F.classify_prompt("继续"), "followup")
        self.assertEqual(F.classify_prompt("/agent-eta:setup"), "command")
        self.assertEqual(F.classify_prompt(""), "wake")


class LifecycleTests(Base):
    def test_basic_run(self):
        self.ev(0, "SessionStart", source="startup")
        self.ev(10, "UserPromptSubmit", prompt="fix the failing login test", prompt_id="p1")
        self.ev(15, "PreToolUse", tool_name="Read", tool_input={"file_path": "/a"}, tool_use_id="t1")
        self.ev(16, "PostToolUse", tool_name="Read", tool_input={"file_path": "/a"}, tool_use_id="t1",
                duration_ms=1000)
        self.ev(20, "PreToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="t2")
        self.ev(50, "PostToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="t2",
                duration_ms=30000)
        self.ev(60, "Stop", background_tasks=[])
        r = self.run_row()
        self.assertEqual(r["end_reason"], "stop")
        self.assertAlmostEqual(r["active_s"], 50.0)
        self.assertEqual(r["n_tools"], 2)
        self.assertEqual(r["n_verify"], 1)
        self.assertEqual(r["category"], "fix")
        snaps = self.conn.execute("SELECT * FROM snapshots WHERE run_id=? ORDER BY t", (r["id"],)).fetchall()
        self.assertEqual(len(snaps), 3)
        self.assertAlmostEqual(snaps[0]["rem_done_s"], 50.0)
        self.assertAlmostEqual(snaps[-1]["rem_done_s"], 10.0)
        self.assertEqual(snaps[-1]["phase"], "verify")
        self.assertTrue(all(s["censored"] == 0 for s in snaps))

    def test_permission_wait_excluded(self):
        self.ev(0, "UserPromptSubmit", prompt="deploy", prompt_id="p1")
        self.ev(10, "PreToolUse", tool_name="Bash", tool_input={"command": "rm -rf build"}, tool_use_id="t1")
        self.ev(10.2, "PermissionRequest", tool_name="Bash", tool_input={"command": "rm -rf build"})
        self.assertEqual(self.run_row()["status"], "attention")
        # approved at t=65, command ran 5 s
        self.ev(70, "PostToolUse", tool_name="Bash", tool_input={"command": "rm -rf build"}, tool_use_id="t1",
                duration_ms=5000)
        self.ev(80, "Stop")
        r = self.run_row()
        self.assertAlmostEqual(r["human_wait_s"], 54.8, places=3)
        self.assertAlmostEqual(r["active_s"], 80 - 54.8, places=3)
        self.assertEqual(r["n_perm"], 1)

    def test_ask_user_question_is_attention(self):
        self.ev(0, "UserPromptSubmit", prompt="build a feature", prompt_id="p1")
        self.ev(20, "PreToolUse", tool_name="AskUserQuestion", tool_input={"questions": [{"question": "A or B?"}]},
                tool_use_id="q")
        self.ev(80, "PostToolUse", tool_name="AskUserQuestion", tool_input={}, tool_use_id="q", duration_ms=60000)
        self.ev(100, "Stop")
        r = self.run_row()
        self.assertAlmostEqual(r["active_s"], 40.0)
        first = self.conn.execute("SELECT * FROM snapshots WHERE run_id=? ORDER BY t LIMIT 1", (r["id"],)).fetchone()
        self.assertAlmostEqual(first["rem_attn_s"], 20.0)  # the question came 20 s in
        self.assertAlmostEqual(first["rem_done_s"], 40.0)

    def test_interrupt_is_censored(self):
        self.ev(0, "UserPromptSubmit", prompt="long task", prompt_id="p1")
        self.ev(5, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1")
        self.ev(6, "PostToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", duration_ms=1000)
        # user hits Esc (no Stop), then types a new prompt
        self.ev(40, "UserPromptSubmit", prompt="never mind", prompt_id="p2")
        first, second = self.conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        self.assertEqual(first["end_reason"], "interrupted")
        self.assertAlmostEqual(first["active_s"], 6.0)
        self.assertIsNone(second["ended_at"])
        censored = self.conn.execute("SELECT DISTINCT censored FROM snapshots WHERE run_id=?", (first["id"],)).fetchall()
        self.assertEqual([c[0] for c in censored], [1])

    def test_late_async_event_does_not_reopen(self):
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1")
        self.ev(1, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1")
        self.ev(50, "Stop")
        self.ev(49.9, "PostToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", duration_ms=48900)
        runs = self.conn.execute("SELECT * FROM runs").fetchall()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["end_reason"], "stop")

    def test_stop_hook_continuation_reopens(self):
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1")
        self.ev(30, "Stop")
        self.ev(35, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", prompt_id="p1")
        self.ev(36, "PostToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", prompt_id="p1", duration_ms=1000)
        self.ev(60, "Stop")
        runs = self.conn.execute("SELECT * FROM runs").fetchall()
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0]["active_s"], 60.0)

    def test_post_turn_helper_agents_do_not_reopen(self):
        # Claude Code's own helpers (e.g. the "while you were away" summary) run as subagents after the
        # turn and carry its prompt_id. Seen live in v0.4.0: runs stayed open until the session ended.
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1")
        self.ev(40, "Stop")
        self.ev(41, "SubagentStart", agent_id="h1", agent_type="helper", prompt_id="p1")
        self.ev(43, "SubagentStop", agent_id="h1", agent_type="helper", prompt_id="p1")
        self.ev(44, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="h1t", agent_id="h1", prompt_id="p1")
        self.ev(45, "TaskCompleted", prompt_id="p1")
        self.ev(575, "SubagentStop", agent_id="h2", agent_type="helper", prompt_id="p1")
        runs = self.conn.execute("SELECT * FROM runs").fetchall()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["end_reason"], "stop")
        self.assertAlmostEqual(runs[0]["active_s"], 40.0)
        self.assertEqual(self.conn.execute("SELECT MAX(t) FROM snapshots").fetchone()[0], 0)

    def test_bash_command_answer_is_its_own_run(self):
        # `!gh auth login` typed at 20 fires no UserPromptSubmit; it finishes at 80 and Claude answers
        # under the command's prompt_id. The command's runtime is yours: the new run starts at 80.
        tp = self.bash_transcript("p2", 20, 80)
        self.ev(0, "UserPromptSubmit", prompt="create the repo", prompt_id="p1", transcript_path=tp)
        self.ev(10, "Stop", prompt_id="p1", transcript_path=tp)
        self.ev(86, "PreToolUse", tool_name="Bash", tool_input={"command": "gh repo create x"}, tool_use_id="b1",
                prompt_id="p2", transcript_path=tp)
        self.ev(88, "PostToolUse", tool_name="Bash", tool_input={"command": "gh repo create x"}, tool_use_id="b1",
                prompt_id="p2", transcript_path=tp, duration_ms=2000)
        self.ev(95, "Stop", prompt_id="p2", transcript_path=tp)
        first, second = self.conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        self.assertEqual((first["end_reason"], first["active_s"]), ("stop", 10.0))
        self.assertEqual((second["category"], second["end_reason"], second["n_tools"]), ("bash", "stop", 1))
        self.assertAlmostEqual(second["started_at"], 80.0)
        self.assertAlmostEqual(second["active_s"], 15.0)

    def test_bash_command_text_only_answer(self):
        tp = self.bash_transcript("p2", 20, 30)
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1", transcript_path=tp)
        self.ev(10, "Stop", prompt_id="p1", transcript_path=tp)
        self.ev(34, "Stop", prompt_id="p2", transcript_path=tp)
        r = self.run_row()
        self.assertEqual((r["category"], r["end_reason"]), ("bash", "stop"))
        self.assertAlmostEqual(r["active_s"], 4.0)

    def test_bash_command_after_esc_closes_the_interrupted_run(self):
        tp = self.bash_transcript("p2", 20, 30)
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1", transcript_path=tp)
        self.ev(5, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", prompt_id="p1")
        # Esc: no Stop. Then `!` at 20, done at 30, Claude answers.
        self.ev(33, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t2", prompt_id="p2", transcript_path=tp)
        first, second = self.conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        self.assertEqual((first["end_reason"], first["active_s"]), ("interrupted", 5.0))
        self.assertEqual(second["category"], "bash")
        self.assertIsNone(second["ended_at"])
        self.assertAlmostEqual(second["started_at"], 30.0)

    def test_new_prompt_id_without_bash_command_is_unchanged(self):
        tp = self.bash_transcript("other", 20, 30)
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1", transcript_path=tp)
        self.ev(10, "Stop", prompt_id="p1", transcript_path=tp)
        self.ev(50, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", prompt_id="p9", transcript_path=tp)
        r = self.run_row()
        self.assertEqual((r["category"], r["started_at"]), ("wake", 50))
        # A wake-up continuing under its own prompt_id stays one run.
        self.ev(60, "UserPromptSubmit", prompt="<task-notification> done", prompt_id="p10", transcript_path=tp)
        self.ev(61, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t2", prompt_id="p10", transcript_path=tp)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 2)

    def test_model_switch_is_read_from_the_transcript(self):
        path = os.path.join(self.tmp, "switch.jsonl")

        def answer(model):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "assistant", "message": {"model": model, "content": []}}) + "\n")

        answer("claude-opus-5")
        self.ev(0, "UserPromptSubmit", prompt="x", prompt_id="p1", transcript_path=path)
        self.ev(10, "Stop", prompt_id="p1", transcript_path=path)
        self.assertEqual(self.run_row()["model"], "claude-opus-5")
        answer("claude-sonnet-4-6")  # /model sonnet, then the next answer
        self.ev(20, "UserPromptSubmit", prompt="y", prompt_id="p2", transcript_path=path)
        self.ev(30, "Stop", prompt_id="p2", transcript_path=path)
        self.assertEqual(self.run_row()["model"], "claude-sonnet-4-6")
        self.ev(40, "UserPromptSubmit", prompt="z", prompt_id="p3", transcript_path=path)
        self.assertEqual(self.run_row()["model"], "claude-sonnet-4-6")  # the next run starts on it

    def test_background_agents_keep_run_open(self):
        self.ev(0, "UserPromptSubmit", prompt="research", prompt_id="p1")
        self.ev(20, "Stop", background_tasks=[{"id": "a", "type": "subagent", "status": "running"}])
        self.assertEqual(self.run_row()["status"], "background")
        self.assertIsNone(self.run_row()["ended_at"])
        self.ev(200, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", prompt_id="p1")
        self.ev(201, "PostToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", prompt_id="p1", duration_ms=500)
        self.ev(230, "Stop", background_tasks=[{"id": "a", "type": "subagent", "status": "completed"}])
        r = self.run_row()
        self.assertEqual(r["end_reason"], "stop")
        self.assertAlmostEqual(r["active_s"], 230.0)

    def test_task_notification_continues_background_run(self):
        self.ev(0, "UserPromptSubmit", prompt="research this with an agent", prompt_id="p1")
        self.ev(5, "PreToolUse", tool_name="Agent", tool_input={"subagent_type": "general-purpose"}, tool_use_id="a")
        self.ev(5.1, "PostToolUse", tool_name="Agent", tool_input={}, tool_use_id="a", duration_ms=100)
        self.ev(9, "Stop", background_tasks=[{"id": "x", "type": "subagent", "status": "running"}])
        self.ev(90, "UserPromptSubmit", prompt="<task-notification> <task-id>x</task-id> done", prompt_id="p2")
        self.ev(95, "Stop", background_tasks=[])
        runs = self.conn.execute("SELECT * FROM runs").fetchall()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["end_reason"], "stop")
        self.assertAlmostEqual(runs[0]["active_s"], 95.0)
        self.assertEqual(runs[0]["category"], "other")

    def test_background_shell_does_not_block(self):
        self.ev(0, "UserPromptSubmit", prompt="start dev server", prompt_id="p1")
        self.ev(20, "Stop", background_tasks=[{"id": "b", "type": "shell", "status": "running"}])
        self.assertEqual(self.run_row()["end_reason"], "stop")


class CodexScenarioTests(Base):
    """Behavioural invariants from the Codex handoff (fixtures/scenarios.json, INV-xx), replayed as
    Claude Code hook events. Codex-only mechanics (notify authority, supervisor heartbeat) are skipped."""

    def start(self):
        self.ev(0, "UserPromptSubmit", prompt="fix the failing test", prompt_id="p1")

    def test_s02_duplicate_tool_events_count_once(self):  # INV-06
        self.start()
        self.ev(1, "PreToolUse", tool_name="Bash", tool_input={"command": "pytest"}, tool_use_id="t1")
        for t in (5, 5.2, 6):
            self.ev(t, "PostToolUse", tool_name="Bash", tool_input={"command": "pytest"}, tool_use_id="t1",
                    duration_ms=4000)
        r = self.run_row()
        self.assertEqual((r["n_tools"], r["n_verify"]), (1, 1))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 2)

    def test_s03_post_before_pre(self):
        self.start()
        self.ev(5, "PostToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", duration_ms=1000)
        self.ev(4, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1")
        rows = self.conn.execute("SELECT started_at, duration_ms FROM tool_calls").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.run_row()["n_tools"], 1)

    def test_s05_unrelated_tool_does_not_clear_permission(self):  # INV-08
        self.start()
        self.ev(10, "PreToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="a")
        self.ev(10.01, "PreToolUse", tool_name="Read", tool_input={"file_path": "/x"}, tool_use_id="b")
        self.ev(10.2, "PermissionRequest", tool_name="Bash", tool_input={"command": "npm test"})
        self.ev(11, "PostToolUse", tool_name="Read", tool_input={"file_path": "/x"}, tool_use_id="b", duration_ms=990)
        r = self.run_row()
        self.assertEqual(r["status"], "attention")  # still waiting for the Bash approval
        self.assertEqual(r["attn_tool"], "a")
        self.ev(70, "PostToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="a",
                duration_ms=20000)
        r = self.run_row()
        self.assertIsNone(r["attn_since"])
        self.assertAlmostEqual(r["human_wait_s"], 50 - 10.2)  # approved when execution began (70 - 20)

    def test_permission_recorded_before_its_pretooluse_is_linked(self):
        self.start()
        self.ev(10.2, "PermissionRequest", tool_name="Bash", tool_input={"command": "rm -rf build"})
        self.ev(10.0, "PreToolUse", tool_name="Bash", tool_input={"command": "rm -rf build"}, tool_use_id="a")
        self.assertEqual(self.run_row()["attn_tool"], "a")

    def test_new_step_after_denial_resolves(self):
        self.start()
        self.ev(10, "PreToolUse", tool_name="Bash", tool_input={"command": "rm -rf /tmp/x"}, tool_use_id="a")
        self.ev(10.2, "PermissionRequest", tool_name="Bash", tool_input={"command": "rm -rf /tmp/x"})
        # denied: no PostToolUse; the model's next tool call proves the request is over
        self.ev(40, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="b")
        r = self.run_row()
        self.assertIsNone(r["attn_since"])
        self.assertAlmostEqual(r["human_wait_s"], 29.8)
        # the denied Bash call is over too: the running tool is the Read, not "rm" ticking on forever
        self.assertEqual(E.state_from_run(self.conn, r, 41)["in_flight"]["tool"], "Read")

    def test_late_post_tool_still_records_duration_after_next_step(self):
        self.start()
        self.ev(1, "PreToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="a")
        self.ev(40, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="b")  # processed first
        self.ev(39, "PostToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="a",
                duration_ms=38000)
        row = self.conn.execute("SELECT ended_at, duration_ms FROM tool_calls WHERE tool_use_id='a'").fetchone()
        self.assertEqual((row["ended_at"], row["duration_ms"]), (39, 38000))
        self.assertEqual(self.run_row()["n_verify"], 1)

    def test_s07_nonzero_search_is_not_a_test_failure(self):
        self.start()
        self.ev(1, "PreToolUse", tool_name="Bash", tool_input={"command": "rg TODO src"}, tool_use_id="t1")
        self.ev(2, "PostToolUseFailure", tool_name="Bash", tool_input={"command": "rg TODO src"}, tool_use_id="t1",
                error="Exit code 1", duration_ms=500)
        self.assertEqual(self.run_row()["n_verify_fail"], 0)

    def test_s04_subagent_stop_does_not_end_parent(self):  # INV-03
        self.start()
        self.ev(2, "SubagentStart", agent_id="ag", agent_type="Explore")
        self.ev(9, "SubagentStop", agent_id="ag", agent_type="Explore")
        self.assertIsNone(self.run_row()["ended_at"])

    def test_s12_late_tool_after_stop_does_not_reopen(self):  # INV-07
        self.start()
        self.ev(1, "PreToolUse", tool_name="Read", tool_input={}, tool_use_id="t1")
        self.ev(20, "Stop")
        self.ev(19.9, "PostToolUse", tool_name="Read", tool_input={}, tool_use_id="t1", duration_ms=18900)
        self.assertEqual(self.run_row()["end_reason"], "stop")

    def test_s20_two_requests_need_two_resolutions(self):
        self.start()
        self.ev(10, "PreToolUse", tool_name="AskUserQuestion", tool_input={}, tool_use_id="q")
        self.ev(10.001, "PreToolUse", tool_name="ExitPlanMode", tool_input={}, tool_use_id="p")
        self.ev(30, "PostToolUse", tool_name="AskUserQuestion", tool_input={}, tool_use_id="q", duration_ms=20000)
        r = self.run_row()
        self.assertEqual((r["status"], r["attn_kind"]), ("attention", "plan_approval"))
        self.ev(50, "PostToolUse", tool_name="ExitPlanMode", tool_input={}, tool_use_id="p", duration_ms=40000)
        r = self.run_row()
        self.assertIsNone(r["attn_since"])
        self.assertAlmostEqual(r["human_wait_s"], 40.0)  # union of overlapping waits, not 20 + 40

    def test_s24_duplicate_stop_counts_once(self):
        self.start()
        self.ev(10, "Stop")
        self.ev(10.5, "Stop")
        runs = self.conn.execute("SELECT * FROM runs").fetchall()
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0]["active_s"], 10.0)

    def test_s06_failed_test_loop(self):
        self.start()
        self.ev(1, "PreToolUse", tool_name="Bash", tool_input={"command": "pytest -q"}, tool_use_id="t1")
        self.ev(9, "PostToolUseFailure", tool_name="Bash", tool_input={"command": "pytest -q"}, tool_use_id="t1",
                error="Exit code 1", duration_ms=8000)
        self.assertEqual(F.phase_of(self.run_row()), "fixing")
        self.ev(10, "PreToolUse", tool_name="Edit", tool_input={"file_path": "/a.py"}, tool_use_id="t2")
        self.ev(11, "PostToolUse", tool_name="Edit", tool_input={"file_path": "/a.py"}, tool_use_id="t2", duration_ms=50)
        self.ev(12, "PreToolUse", tool_name="Bash", tool_input={"command": "pytest -q"}, tool_use_id="t3")
        self.ev(20, "PostToolUse", tool_name="Bash", tool_input={"command": "pytest -q"}, tool_use_id="t3", duration_ms=8000)
        r = self.run_row()
        self.assertEqual((r["n_verify"], r["n_verify_fail"], F.phase_of(r)), (2, 1, "verify"))
        self.assertIsNone(r["ended_at"])  # a failed tool never closes the run

    def test_s10_s15_plan_expansion_and_completion(self):
        self.start()
        for i in range(5):
            self.ev(1 + i * 0.1, "TaskCreated", task_id="k%d" % i, task_subject="s")
        self.ev(20, "TaskCompleted", task_id="k0", task_subject="s")
        for i in range(5, 7):
            self.ev(21, "TaskCreated", task_id="k%d" % i, task_subject="s")
        r = self.run_row()
        self.assertEqual((r["plan_total"], r["plan_first_total"]), (7, 5))
        est = E.estimate(self.conn, E.state_from_run(self.conn, r, 22))
        self.assertIn(("plan_grew", {"first": 5, "total": 7}), est["reasons"])
        for i in range(1, 7):
            self.ev(30 + i, "TaskCompleted", task_id="k%d" % i, task_subject="s")
        r = self.run_row()
        self.assertEqual(r["plan_done"], 7)
        self.assertIsNone(r["ended_at"])  # INV-09: a finished plan is not a finished turn

    def test_s19_api_failure_is_not_success(self):
        self.start()
        self.ev(15, "StopFailure", error_type="overloaded")
        r = self.run_row()
        self.assertEqual(r["end_reason"], "stop_failure")
        self.assertEqual({c[0] for c in self.conn.execute("SELECT censored FROM snapshots")}, {1})

    def test_s21_same_cwd_sessions_stay_separate(self):
        self.ev(0, "UserPromptSubmit", prompt="a", prompt_id="pa", session_id="A")
        self.ev(1, "UserPromptSubmit", prompt="b", prompt_id="pb", session_id="B")
        self.ev(5, "Stop", session_id="A")
        runs = {r["session_id"]: r for r in self.conn.execute("SELECT * FROM runs")}
        self.assertEqual(runs["A"]["end_reason"], "stop")
        self.assertIsNone(runs["B"]["ended_at"])

    def test_inv12_backtest_state_has_no_future_plan_info(self):
        self.start()
        for i, t in enumerate((10, 20, 30)):
            self.ev(t, "TaskCreated", task_id="k%d" % i, task_subject="s")
        self.ev(40, "TaskCompleted", task_id="k0", task_subject="s")
        self.ev(90, "Stop")
        run = self.run_row()
        first = self.conn.execute("SELECT * FROM snapshots WHERE run_id=? ORDER BY t", (run["id"],)).fetchone()
        st = E.state_from_snapshot(run, first)
        self.assertEqual((st["plan_done"], st["active_at_last_done"], st["plan_ema"]), (0, None, None))

    def test_privacy_no_command_arguments_or_prompt_stored(self):
        secret = "sk-live-0123456789abcdef"
        self.ev(0, "UserPromptSubmit", prompt="deploy with token " + secret, prompt_id="p1")
        cmd = 'curl -H "Authorization: Bearer %s" https://api.example.com/v1/deploy' % secret
        self.ev(1, "PreToolUse", tool_name="Bash", tool_input={"command": cmd}, tool_use_id="t1")
        self.ev(1.1, "PermissionRequest", tool_name="Bash", tool_input={"command": cmd})
        self.ev(9, "PostToolUse", tool_name="Bash", tool_input={"command": cmd}, tool_use_id="t1", duration_ms=2000)
        self.ev(10, "Stop")
        self.conn.execute("PRAGMA wal_checkpoint(FULL)")
        for name in os.listdir(config.home()):
            with open(os.path.join(config.home(), name), "rb") as f:
                data = f.read()
            self.assertNotIn(secret.encode(), data, name)
            self.assertNotIn(b"api.example.com", data, name)
        self.assertEqual(self.conn.execute("SELECT detail FROM attention").fetchone()[0], "Bash: curl")


class EstimatorTests(Base):
    def test_survival_helpers(self):
        S = E.km_survival([(10, 1, True), (20, 1, True), (30, 1, True), (40, 1, True)])
        self.assertEqual(E.quantile(S, 0.5), 20)  # exact jump, no grid error
        cond = E.lognormal_survival(60, 1.0, elapsed=300)
        fresh = E.lognormal_survival(60, 1.0, elapsed=0)
        self.assertGreater(E.quantile(cond, 0.5), E.quantile(fresh, 0.5))  # heavy tail: older runs last longer
        censored = E.km_survival([(10, 1, True), (20, 1, False), (30, 1, False)])
        self.assertAlmostEqual(censored(25), 2 / 3)
        mixed = E.mix(E.km_survival([(100, 1, True)]), E.lognormal_survival(100, 0.5), 0.5)
        self.assertEqual(E.quantile(mixed, 0.5), 100)  # a jump inside a mixture is still found exactly

    def test_learned_estimator_is_calibrated_and_beats_prior(self):
        sim.populate(self.conn, 150, seed=11)
        learned, prior, b0 = report.backtest_rows(self.conn, 0, max_runs=60, per_run=5)
        ml, mp = report._metrics(learned), report._metrics(prior)
        self.assertGreater(ml["n"], 200)
        self.assertTrue(0.35 <= ml["hit50"] <= 0.65, ml["hit50"])
        self.assertTrue(0.70 <= ml["hit80"] <= 0.95, ml["hit80"])
        self.assertLessEqual(ml["safe_viol"], 0.30)
        med = lambda v: sorted(v)[len(v) // 2]
        self.assertLessEqual(med(ml["err"]), med(mp["err"]))

    def test_live_estimate_with_inflight_tool(self):
        sim.populate(self.conn, 60, seed=3)
        t = 1.9e9
        self.ev(t, "UserPromptSubmit", prompt=sim.PROMPTS["fix"], prompt_id="live", cwd="/work/web")
        self.ev(t + 40, "PreToolUse", tool_name="Bash", tool_input={"command": "npm test"}, tool_use_id="x",
                cwd="/work/web")
        run = self.run_row()
        state = E.state_from_run(self.conn, run, t + 50)
        self.assertEqual(state["in_flight"]["cmd_key"], "npm test")
        est = E.estimate(self.conn, state)
        # web tests take ~90 s and 10 s have passed: completion cannot be predicted sooner than that
        self.assertGreater(est["done_p50"], 60)
        self.assertTrue(any(code == "inflight" for code, _ in est["reasons"]))


class CalibrationAndTuningTests(Base):
    def slow_runs(self, n, factor, t0=1000.0):
        """n finished 'fix' runs that each take `factor` times the built-in prior median."""
        state = {"category": "fix", "effort": None, "prompt_len": len("fix the failing login test")}
        dur = E.prior_median(state) * factor
        t = t0
        for i in range(n):
            self.ev(t, "UserPromptSubmit", prompt="fix the failing login test", prompt_id="c%d" % i)
            self.ev(t + dur, "Stop")
            t += dur + 60
        return t

    def test_calibration_learns_your_pace(self):
        t = self.slow_runs(20, 4.0)
        self.ev(t, "UserPromptSubmit", prompt="fix the failing login test", prompt_id="now")
        state = E.state_from_run(self.conn, self.run_row(), t + 1)
        pool = E.Pool.load(self.conn, state["as_of"])
        mu_cat, mu, sigma, n = E.calibration(pool, state)
        self.assertEqual(n, 20)
        # shrunk toward the built-in pace, but most of the way to x4 with 20 runs
        self.assertTrue(3.0 < math.exp(mu_cat["fix"]) < 4.0, math.exp(mu_cat["fix"]))
        self.assertLess(sigma, 1.0)  # all runs alike: tighter than the built-in spread
        est = E.estimate(self.conn, state, k0=math.inf)
        self.assertEqual(est["method"], "calibrated")
        self.assertTrue(any(code == "pace" for code, _ in est["reasons"]))
        built_in = E.quantile(E.prior_survival(state), 0.5)
        self.assertGreater(est["done_p50"], 2.5 * built_in)

    def test_short_half_life_follows_recent_pace(self):
        t = self.slow_runs(10, 4.0)                       # a month ago: slow
        t = self.slow_runs(10, 0.25, t0=t + 30 * 86400)   # lately: fast
        self.ev(t, "UserPromptSubmit", prompt="fix the failing login test", prompt_id="now")
        state = E.state_from_run(self.conn, self.run_row(), t + 1)
        pool = E.Pool.load(self.conn, state["as_of"])
        long_mu = E.calibration(pool, state, 60.0)[0]["fix"]
        short_mu = E.calibration(pool, state, 0.5)[0]["fix"]
        self.assertLess(short_mu, long_mu)
        self.assertLess(math.exp(short_mu), 0.5)
        fast = E.estimate(self.conn, state, k0=math.inf, half_life=0.5, recal=False)
        slow = E.estimate(self.conn, state, k0=math.inf, half_life=60.0, recal=False)
        self.assertLess(fast["done_p50"], slow["done_p50"])

    def test_recalibration_reads_earlier_when_estimates_run_long(self):
        self.assertEqual(E.recal_level(None, "done", 0.5), 0.5)
        S = E.lognormal_survival(100.0, 1.0)
        cfg = (math.inf, 60.0)
        # outcomes far earlier than predicted: PIT values pile up near 0
        pts = [{"done": 20.0 + i, "attn": None, "est": {cfg: {"est": {"S_done": S, "S_attn": S, "method": "x",
                                                                       "safe_k": 6.0, "ess_global": 0.0}}}}
               for i in range(40)]
        recal = tuning.fit_recal(pts, cfg)
        self.assertEqual(recal["n_done"], 40)
        lv50, lv80 = E.recal_level(recal, "done", 0.5), E.recal_level(recal, "done", 0.8)
        self.assertLess(lv50, 0.5)
        self.assertLess(lv50, lv80)
        self.assertEqual(E.recal_level(recal, "attn", 0.2), 0.2)  # too few attention outcomes: unchanged
        raw = E.quantile(S, 0.5)
        self.assertLess(tuning.read_through(pts[0]["est"][cfg]["est"], recal, 0.2)["p50"], raw)
        # shrinkage: few outcomes move the level less than many
        few = tuning.fit_recal(pts[:30], cfg)
        self.assertGreater(E.recal_level(few, "done", 0.5), lv50)

    def test_prequential_recal_uses_only_the_past(self):
        S = E.lognormal_survival(100.0, 1.0)
        cfg = (math.inf, 60.0)
        pts = [{"run": i, "as_of": float(i), "ended": float(i) + 0.5, "done": 10.0, "attn": None,
                "est": {cfg: {"est": {"S_done": S, "S_attn": S, "method": "x", "safe_k": 6.0, "ess_global": 0.0}}}}
               for i in range(60)]
        out = tuning.prequential_recal(pts, cfg, 0.2)
        raw = E.quantile(S, 0.5)
        self.assertAlmostEqual(out[0]["p50"], raw, delta=raw * 1e-3)   # nothing known yet
        self.assertAlmostEqual(out[29]["p50"], raw, delta=raw * 1e-3)  # 29 outcomes: below RECAL_MIN_POINTS
        self.assertLess(out[59]["p50"], 0.5 * raw)

    def test_calibration_ignores_the_future(self):
        self.slow_runs(10, 4.0)
        run = self.conn.execute("SELECT * FROM runs ORDER BY id").fetchone()
        snap = self.conn.execute("SELECT * FROM snapshots WHERE run_id=? ORDER BY t", (run["id"],)).fetchone()
        state = E.state_from_snapshot(run, snap)
        self.assertEqual(E.calibration(E.Pool.load(self.conn, 1e12), state)[3], 0)

    def test_tune_chooses_and_persists(self):
        sim.populate(self.conn, 40, seed=2)
        self.assertEqual(E.tuned_k0(self.conn), E.K0_DEFAULT)
        res = tuning.tune(self.conn)
        self.assertIn(res["k0"], tuning.GRID)
        self.assertIn(res["half_life"], tuning.HL_GRID)
        self.assertGreaterEqual(res["n_points"], tuning.MIN_POINTS)
        chosen = res["losses"][(res["k0"], res["half_life"])]
        # the half-life is within tolerance of the best, and K0 is the best at that half-life
        self.assertLessEqual(chosen, min(res["losses"].values()) * (1 + tuning.HL_TOLERANCE) + 1e-12)
        self.assertEqual(chosen, min(v for (k, h), v in res["losses"].items() if h == res["half_life"]))
        tuned = E.tuned_values(self.conn)
        self.assertEqual((E.tuned_k0(self.conn), tuned["half_life"]), (res["k0"], res["half_life"]))
        self.assertEqual(bool(tuned["recal"]), res["recal"])
        self.assertEqual(E.tuned_recal(self.conn) is not None, res["recal"])
        text = report.build(self.conn, "en", 3650, True, 0.2)
        self.assertIn("chosen", text)
        self.assertIn("log pinball loss per candidate", text)
        self.assertIn("recency half-life", text)
        self.assertIn("quantile recalibration", text)
        self.assertIn("half-life", tuning.describe(res))

    def test_safe_k_choice(self):
        stats = {0.0: (100, 30), 2.0: (90, 12), 6.0: (80, 8), 15.0: (60, 3), 40.0: (40, 0)}
        self.assertEqual(tuning.choose_safe_k(stats, 0.2), 2.0)  # most generous credibly under 20 %
        few = {k: (5, 0) for k in tuning.SAFE_GRID}
        self.assertEqual(tuning.choose_safe_k(few, 0.2), E.SAFE_K_DEFAULT)  # replay cannot tell
        bad = {k: (50, 25) for k in tuning.SAFE_GRID}
        self.assertEqual(tuning.choose_safe_k(bad, 0.2), 15.0)  # one step past the default, not the maximum
        noisy = {0.0: (63, 13), 2.0: (35, 6), 6.0: (20, 5), 15.0: (5, 2), 40.0: (0, 0)}
        self.assertEqual(tuning.choose_safe_k(noisy, 0.2), 2.0)  # lowest observed rate with enough promises
        # real windows breached too often: one step more cautious than before, whatever the replay says
        self.assertEqual(tuning.choose_safe_k(stats, 0.2, previous=6.0, ledger=(30, 12)), 15.0)
        self.assertEqual(tuning.choose_safe_k(stats, 0.2, previous=6.0, ledger=(30, 2)), 2.0)

    def test_tuned_safe_k_is_used_live(self):
        sim.populate(self.conn, 40, seed=2)
        res = tuning.tune(self.conn)
        self.assertIn(res["safe_k"], tuning.SAFE_GRID)
        self.ev(1.9e9, "UserPromptSubmit", prompt=sim.PROMPTS["fix"], prompt_id="now", cwd="/work/web")
        est = E.estimate(self.conn, E.state_from_run(self.conn, self.run_row(), 1.9e9 + 5))
        self.assertEqual((est["k0"], est["safe_k"]), (res["k0"], res["safe_k"]))
        self.assertIn("SAFE_K=", report.build(self.conn, "en", 3650, True, 0.2))

    def test_retune_only_when_history_grew(self):
        spawned = []
        spawn = lambda: spawned.append(1)
        self.slow_runs(9, 1.0)
        self.assertFalse(tuning.maybe_spawn(self.conn, 1e6, spawn))  # 9 runs: not yet
        t = self.slow_runs(1, 1.0, t0=5e5)
        self.assertTrue(tuning.maybe_spawn(self.conn, 1e6, spawn))
        self.assertFalse(tuning.maybe_spawn(self.conn, 1e6 + 60, spawn))  # claimed, still running
        tuning.tune(self.conn)
        self.slow_runs(4, 1.0, t0=t + 100)
        self.assertFalse(tuning.maybe_spawn(self.conn, 2e6, spawn))  # 14 runs vs 10 tuned: < 5 new
        self.slow_runs(1, 1.0, t0=8e5)
        self.assertTrue(tuning.maybe_spawn(self.conn, 2e6, spawn))
        self.assertEqual(len(spawned), 2)


class CodexMathGoldenTests(unittest.TestCase):
    """fixtures/math-golden.json from the Codex handoff package (agent-eta-codex-handoff.zip)."""

    @classmethod
    def setUpClass(cls):
        cls.g = json.loads(_read(os.path.join(HERE, "fixtures", "codex-math-golden.json")))
        cls.tol = cls.g["absolute_tolerance"]

    def close(self, got, want, msg):
        if want is None:
            self.assertIsNone(got, msg)
        else:
            self.assertIsNotNone(got, msg)
            self.assertAlmostEqual(got, want, delta=self.tol, msg=msg)

    def test_kaplan_meier(self):
        for case in self.g["kaplan_meier"]:
            km = E.KaplanMeier([(s["remaining_seconds"], s["weight"], s["observed"]) for s in case["samples"]])
            for t, want in case["survival_at"].items():
                self.close(km.survival(float(t)), want, "%s S(%s)" % (case["id"], t))
            for q, want in case["quantiles"].items():
                self.close(km.quantile(float(q)), want, "%s Q(%s)" % (case["id"], q))
            for q, want in case.get("conditional_remaining_quantiles", {}).items():
                self.close(km.conditional_quantile(case["conditional_on_surviving_seconds"], float(q)), want,
                           "%s conditional Q(%s)" % (case["id"], q))
            # the estimator's curve agrees with the exact KM wherever the KM is identified
            curve = E.km_survival([(s["remaining_seconds"], s["weight"], s["observed"]) for s in case["samples"]])
            for q, want in case["quantiles"].items():
                if want is not None:
                    self.close(E.quantile(curve, float(q)), want, "%s curve Q(%s)" % (case["id"], q))

    def test_ema_ess_clopper_pearson(self):
        for case in self.g["ema"]:
            states = E.ema(case["intervals_seconds"], case["alpha"])
            for got, want in zip(states, case["expected_states"]):
                self.close(got, want, case["id"])
            self.close(states[-1] * case["remaining_steps"], case["expected_eta_seconds"], case["id"] + " eta")
        for case in self.g["effective_sample_size"]:
            self.close(E.ess(case["weights"]), case["expected"], case["id"])
        for case in self.g["clopper_pearson"]:
            u = E.clopper_pearson_upper(case["n"], case["violations"], case["alpha"])
            self.close(u, case["expected_upper"], case["id"])
            self.assertEqual(case["n"] >= 100 and u <= 0.05, case["passes_numeric_gate"], case["id"])


class WindowLedgerTests(Base):
    """Codex design section 13.5 / INV-13 / POL05: fixed expiry, revocation keeps the record."""

    def live_run(self, t0, tune=True):
        sim.populate(self.conn, 80, seed=4)
        if tune:  # as live use does from 10 finished runs on: no leave promise before the first tune
            tuning.tune(self.conn)
        # bypassPermissions: no approval prompts, so the permission hazard does not veto the window
        self.ev(t0, "UserPromptSubmit", prompt=sim.PROMPTS["feature"], prompt_id="w1", cwd="/work/web",
                permission_mode="bypassPermissions")
        return self.run_row()

    def test_window_has_fixed_expiry_and_is_scored(self):
        cfg = config.load()
        t0 = 1.95e9
        run = self.live_run(t0)
        _, est, _ = live.compute(self.conn, run, t0 + 5, cfg)
        w = est["window"]
        self.assertIsNotNone(w, "a feature run with rich history should get a leave window")
        self.assertEqual(w["expires_at"], t0 + 5 + w["horizon_s"])
        _, est2, _ = live.compute(self.conn, self.run_row(), t0 + 65, cfg)
        self.assertEqual(est2["window"]["id"], w["id"])            # same window, not re-issued
        self.assertEqual(est2["window"]["expires_at"], w["expires_at"])  # never pushed later
        # Claude asks a question inside the window: revoked on screen, but scored as a breach
        self.ev(t0 + 70, "PreToolUse", tool_name="AskUserQuestion", tool_input={}, tool_use_id="q",
                cwd="/work/web")
        state3, est3, _ = live.compute(self.conn, self.run_row(), t0 + 71, cfg)
        self.assertIsNone(est3["window"])
        row = self.conn.execute("SELECT revoked_at, revoke_reason FROM windows WHERE id=?", (w["id"],)).fetchone()
        self.assertEqual(row["revoke_reason"], "attention")
        outcomes = dict((x["id"], o) for x, o in live.window_outcomes(self.conn, t0 + 3600))
        self.assertEqual(outcomes[w["id"]], "breach")
        stats = live.ledger_stats(self.conn, t0 + 3600, cfg)
        self.assertEqual((stats["n"], stats["breach"], stats["validated"]), (1, 1, False))

    def test_no_promise_until_the_history_supports_it(self):
        cfg = config.load()
        t0 = 1.95e9
        run = self.live_run(t0, tune=False)
        _, est, _ = live.compute(self.conn, run, t0 + 5, cfg)
        self.assertIsNone(est["window"])
        self.assertIsNone(est["safe_s"])
        line = render.statusline(live.E.state_from_run(self.conn, run, t0 + 5), est, "en", t0 + 5, 0.2,
                                 render.Paint(False), None)
        self.assertIn("leave: unverified", line)
        self.assertNotIn("any moment", line)

    def test_ledger_steps_up_only_when_credibly_over_target(self):
        stats = {0.0: (100, 10), 2.0: (90, 8), 6.0: (80, 6), 15.0: (60, 3), 40.0: (40, 0)}
        self.assertEqual(tuning.choose_safe_k(stats, 0.2, previous=0.0, ledger=(40, 6)), 0.0)    # 15 %: fine
        self.assertEqual(tuning.choose_safe_k(stats, 0.2, previous=0.0, ledger=(40, 9)), 0.0)    # 22 %: not credibly over
        self.assertEqual(tuning.choose_safe_k(stats, 0.2, previous=0.0, ledger=(40, 14)), 2.0)   # 35 %: step up
        self.assertLess(E.clopper_pearson_lower(40, 9, 0.25), 0.2)
        self.assertGreater(E.clopper_pearson_lower(40, 14, 0.25), 0.2)
        self.assertEqual(E.clopper_pearson_lower(40, 0, 0.25), 0.0)

    def test_status_view_does_not_issue_windows(self):
        t0 = 1.95e9
        run = self.live_run(t0)
        live.compute(self.conn, run, t0 + 5, config.load(), log=False)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM windows").fetchone()[0], 0)

    def test_cold_start_unknown_shows_no_numbers(self):
        cfg = config.load()
        cfg["cold_start"] = "unknown"
        self.ev(10, "UserPromptSubmit", prompt="fix it", prompt_id="c1")
        state, est, _ = live.compute(self.conn, self.run_row(), 20, cfg)
        self.assertTrue(est["cold_unknown"])
        self.assertIsNone(est["window"])
        from agent_eta import render
        line = render.statusline(state, est, "en", 20, 0.2, render.Paint(False))
        self.assertIn("done ? (not enough history)", line)
        self.assertNotIn("safe", line)


class EvaluateTests(Base):
    def test_eval_reports_gain_with_interval(self):
        sim.populate(self.conn, 60, seed=4)
        tuning.tune(self.conn)
        res = evaluate.evaluate(self.conn, 0.2, curve=True)
        self.assertGreater(res["points"], 100)
        for key in ("calibrated", "current", "b0"):
            g = res["gain"][key]
            self.assertLessEqual(g["lo"], g["gain"])
            self.assertLessEqual(g["gain"], g["hi"])
        # the simulated history has real structure: learning must be shown to help
        self.assertGreater(res["gain"]["current"]["lo"], 0)
        self.assertEqual(len(res["history"]), 3)
        self.assertEqual(res["curve"]["sizes"][0]["n"], 0)
        self.assertIsNotNone(res["curve"]["marginal"])
        text = evaluate.describe(res, "en")
        self.assertIn("shown: learning makes the estimates better", text)
        self.assertIn("Learning curve", text)
        self.assertIn("结论", evaluate.describe(res, "zh"))
        json.loads(evaluate.to_json(res))

    def test_eval_without_history(self):
        self.assertIn("No finished runs", evaluate.describe(evaluate.evaluate(self.conn, 0.2), "en"))


class CliTests(Base):
    def _run(self, args, stdin=""):
        return subprocess.run([sys.executable, ETA] + args, input=stdin, capture_output=True, text=True,
                              env=dict(os.environ), timeout=60)

    def test_hook_is_silent_and_never_fails(self):
        for payload in ("", "not json", "[]", json.dumps({"hook_event_name": "UserPromptSubmit"}),
                        json.dumps({"session_id": "z", "hook_event_name": "UserPromptSubmit", "prompt": "hi"})):
            out = self._run(["hook"], payload)
            self.assertEqual(out.returncode, 0)
            self.assertEqual(out.stdout, "")
        n = store.connect().execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        self.assertEqual(n, 1)

    def test_statusline_renders(self):
        self._run(["hook"], json.dumps({"session_id": "z", "hook_event_name": "UserPromptSubmit",
                                        "prompt": "fix the bug", "cwd": "/tmp"}))
        out = self._run(["statusline"], json.dumps({"session_id": "z", "model": {"id": "claude-opus-5"}}))
        self.assertEqual(out.returncode, 0)
        self.assertIn("⏱", out.stdout)
        self.assertEqual(store.connect().execute("SELECT model FROM runs").fetchone()[0], "claude-opus-5")
        idle = self._run(["statusline"], json.dumps({"session_id": "other"}))
        self.assertIn("ETA", idle.stdout)
        garbage = self._run(["statusline"], "{{{")
        self.assertEqual(garbage.returncode, 0)


class BackfillTests(Base):
    def test_transcript_import(self):
        recs = [
            {"type": "user", "sessionId": "old", "cwd": "/work/api", "timestamp": "2026-09-01T10:00:00.000Z",
             "promptId": "p", "permissionMode": "default", "origin": {"kind": "human"},
             "message": {"role": "user", "content": "fix the flaky test"}},
            {"type": "assistant", "sessionId": "old", "timestamp": "2026-09-01T10:00:05.000Z", "effort": "high",
             "message": {"model": "claude-opus-5", "stop_reason": "tool_use", "content": [
                 {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "pytest -q"}}]}},
            {"type": "user", "sessionId": "old", "timestamp": "2026-09-01T10:00:35.000Z",
             "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "is_error": True}]}},
            {"type": "assistant", "sessionId": "old", "timestamp": "2026-09-01T10:00:50.000Z",
             "message": {"model": "claude-opus-5", "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]}},
            {"type": "system", "subtype": "turn_duration", "durationMs": 50000, "sessionId": "old",
             "timestamp": "2026-09-01T10:00:50.100Z"},
            {"type": "user", "sessionId": "old", "timestamp": "2026-09-01T10:01:00.000Z", "isMeta": True,
             "message": {"role": "user", "content": "<local-command-caveat>Caveat: ...</local-command-caveat>"}},
        ]
        d = os.path.join(self.tmp, "projects", "-work-api")
        os.makedirs(d)
        with open(os.path.join(d, "old.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        files, runs = backfill.run(os.path.join(self.tmp, "projects"))
        self.assertEqual((files, runs), (1, 1))
        r = self.run_row()
        self.assertEqual(r["source"], "backfill")
        self.assertEqual(r["end_reason"], "stop")
        self.assertEqual(r["model"], "claude-opus-5")
        self.assertEqual(r["effort"], "high")
        self.assertEqual(r["n_verify_fail"], 1)
        self.assertAlmostEqual(r["active_s"], 50.1, places=2)
        self.assertEqual(backfill.run(os.path.join(self.tmp, "projects")), (0, 0))  # idempotent

    def test_transcript_background_agent_is_one_run(self):
        ts = lambda s: "2026-09-02T10:00:%06.3fZ" % s
        user = lambda s, text, pid, **kw: dict({"type": "user", "sessionId": "bg", "cwd": "/w", "timestamp": ts(s),
                                                "promptId": pid, "message": {"role": "user", "content": text}}, **kw)
        asst = lambda s, content, stop="tool_use": {"type": "assistant", "sessionId": "bg", "timestamp": ts(s),
                                                    "message": {"model": "m", "stop_reason": stop, "content": content}}
        recs = [
            user(0, "use an agent to check uname", "p1", origin={"kind": "human"}),
            asst(2, [{"type": "tool_use", "id": "a1", "name": "Agent", "input": {"subagent_type": "general-purpose"}}]),
            {"type": "user", "sessionId": "bg", "timestamp": ts(2.1), "promptId": "p1",
             "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a1", "content": "launched"}]}},
            asst(6, [{"type": "text", "text": "waiting"}], stop="end_turn"),
            user(20, "<task-notification> <task-id>x</task-id> </task-notification>", "p2",
                 origin={"kind": "task-notification"}),
            user(22, "[Your previous response had no visible output. Please continue.]", "p2"),
            asst(25, [{"type": "text", "text": "Linux"}], stop="end_turn"),
        ]
        d = os.path.join(self.tmp, "projects", "-w")
        os.makedirs(d)
        with open(os.path.join(d, "bg.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        self.assertEqual(backfill.run(os.path.join(self.tmp, "projects")), (1, 1))
        r = self.run_row()
        self.assertEqual(r["end_reason"], "stop")
        self.assertAlmostEqual(r["active_s"], 25.001, places=2)

    def test_transcript_bash_command_answer_is_its_own_run(self):
        ts = lambda s: "2026-09-03T10:%02d:%06.3fZ" % divmod(s, 60)
        user = lambda s, text, pid, **kw: dict({"type": "user", "sessionId": "sh", "cwd": "/w", "timestamp": ts(s),
                                                "promptId": pid, "message": {"role": "user", "content": text}}, **kw)
        asst = lambda s, content, stop="tool_use": {"type": "assistant", "sessionId": "sh", "timestamp": ts(s),
                                                    "message": {"model": "m", "stop_reason": stop, "content": content}}
        done = lambda s: {"type": "system", "subtype": "turn_duration", "sessionId": "sh", "timestamp": ts(s)}
        recs = [
            user(0, "create the repo", "p1", origin={"kind": "human"}),
            asst(8, [{"type": "text", "text": "log in first: gh auth login"}], stop="end_turn"),
            done(8.1),
            user(20, "<bash-input>gh auth login</bash-input>", "p2"),
            user(80, "<bash-stdout>Logged in</bash-stdout><bash-stderr></bash-stderr>", "p2"),
            asst(84, [{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "gh repo create x"}}]),
            {"type": "user", "sessionId": "sh", "timestamp": ts(88), "promptId": "p2",
             "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b1", "content": "ok"}]}},
            asst(95, [{"type": "text", "text": "created"}], stop="end_turn"),
            done(95.1),
            user(120, "<bash-input>ls</bash-input>", "p3"),  # not answered: no run
            user(121, "<bash-stdout>a b</bash-stdout>", "p3"),
        ]
        d = os.path.join(self.tmp, "projects", "-w")
        os.makedirs(d)
        with open(os.path.join(d, "sh.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(r) for r in recs) + "\n")
        self.assertEqual(backfill.run(os.path.join(self.tmp, "projects")), (1, 2))
        first, second = self.conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        self.assertEqual((first["end_reason"], first["category"]), ("stop", "feature"))
        self.assertAlmostEqual(first["active_s"], 8.1, places=2)
        self.assertEqual((second["end_reason"], second["category"], second["n_tools"]), ("stop", "bash", 1))
        self.assertAlmostEqual(second["active_s"], 15.1, places=2)


class SetupTests(Base):
    def test_install_keeps_and_restores_existing_statusline(self):
        spath = setup.settings_file()
        os.makedirs(os.path.dirname(spath))
        with open(spath, "w") as f:
            json.dump({"model": "opus", "statusLine": {"type": "command", "command": "echo mine", "padding": 1}}, f)
        setup.install("zh", 5, do_backfill=False, out=lambda *_: None)
        s = json.loads(_read(spath))
        self.assertEqual(s["statusLine"]["command"], setup.launcher_file())
        self.assertEqual(s["statusLine"]["refreshInterval"], 5)
        self.assertEqual(s["statusLine"]["padding"], 1)
        self.assertEqual(s["model"], "opus")
        self.assertTrue(os.path.exists(spath + ".agent-eta-backup"))
        self.assertEqual(config.load()["wrap_command"], "echo mine")
        out = subprocess.run(["sh", setup.launcher_file()], input=json.dumps({"session_id": "q"}),
                             capture_output=True, text=True, env=dict(os.environ), timeout=30)
        self.assertTrue(out.stdout.startswith("mine\n"), out.stdout)
        setup.remove(out=lambda *_: None)
        s = json.loads(_read(spath))
        self.assertEqual(s["statusLine"], {"type": "command", "command": "echo mine", "padding": 1})
        self.assertFalse(os.path.exists(setup.launcher_file()))

    def test_refuses_unparseable_settings(self):
        spath = setup.settings_file()
        os.makedirs(os.path.dirname(spath))
        with open(spath, "w") as f:
            f.write("{ not json")
        with self.assertRaises(ValueError):
            setup.install("en", 5, do_backfill=False, out=lambda *_: None)
        self.assertEqual(_read(spath), "{ not json")


if __name__ == "__main__":
    unittest.main()
