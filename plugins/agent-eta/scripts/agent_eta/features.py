"""Pure feature extraction: tool / command / prompt classification. No I/O except repo_root()."""
import functools
import os
import re
import shlex
from datetime import datetime

EXPLORE_TOOLS = {"Read", "Grep", "Glob", "LS", "WebFetch", "WebSearch", "NotebookRead", "ToolSearch"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
AGENT_TOOLS = {"Agent", "Task"}  # "Task" is the pre-2.1 name of the subagent tool
PLAN_TOOLS = {"TaskCreate", "TaskUpdate", "TodoWrite"}
# Tools whose *call* means Claude is now waiting on the human.
ATTENTION_TOOLS = {"AskUserQuestion": "question", "ExitPlanMode": "plan_approval"}

# Background task types that will wake Claude up again (so a Stop is not the end of the run).
WAKING_BG_TYPES = {"subagent", "workflow", "teammate", "cloud session"}
BG_DONE_STATUSES = {"completed", "complete", "done", "failed", "killed", "stopped", "cancelled", "canceled", "error"}

VERIFY_KINDS = {"test", "build", "lint"}

PHASES = {"start": 0.0, "explore": 1.0, "implement": 2.0, "fixing": 2.5, "verify": 3.0}


def tool_kind(name):
    if name in EXPLORE_TOOLS:
        return "explore"
    if name in EDIT_TOOLS:
        return "edit"
    if name in SHELL_TOOLS:
        return "shell"
    if name in AGENT_TOOLS:
        return "agent"
    if name in PLAN_TOOLS:
        return "plan"
    if name in ATTENTION_TOOLS:
        return "attention"
    if name and name.startswith("mcp__"):
        return "mcp"
    return "other"


# ---------------------------------------------------------------- shell commands

_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._:+-]{1,32}")
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}


_HEREDOC_WORD_END = set(" \t\n;&|<>()")


def _heredoc_delimiter(command, i):
    """At a `<<` (not `<<<`): return (delimiter, index after it), or (None, i)."""
    j = i + 2
    if j < len(command) and command[j] == "-":
        j += 1
    while j < len(command) and command[j] in " \t":
        j += 1
    if j < len(command) and command[j] in "'\"":
        end = command.find(command[j], j + 1)
        if end < 0:
            return None, i
        return command[j + 1:end], end + 1
    k = j
    while k < len(command) and command[k] not in _HEREDOC_WORD_END:
        k += 1
    word = command[j:k].replace("\\", "")
    return (word, k) if word else (None, i)


def _skip_heredoc_bodies(command, i, delims):
    """`i` is just past a newline: skip each pending here-document body, through its closing line."""
    for word in delims:
        while i < len(command):
            nl = command.find("\n", i)
            line, i = (command[i:], len(command)) if nl < 0 else (command[i:nl], nl + 1)
            if line.strip() == word:
                break
    return i


def _split_segments(command):
    """Split on ; && || | & and newlines, but not inside quotes. Here-document bodies are dropped:
    they are data (a script piped into python, a file being written), not commands."""
    segs, cur, quote, i, heredocs = [], [], None, 0, []
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < n:
                cur.append(command[i + 1])
                i += 1
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif command.startswith("<<<", i):  # here-string: an ordinary argument
            cur.append("<<<")
            i += 3
            continue
        elif command.startswith("<<", i):
            word, end = _heredoc_delimiter(command, i)
            if word is None:
                cur.append(ch)
            else:
                heredocs.append(word)
                cur.append("<<" + word)
                i = end
                continue
        elif ch == "\n":
            segs.append("".join(cur))
            cur = []
            if heredocs:
                i = _skip_heredoc_bodies(command, i + 1, heredocs)
                heredocs = []
                continue
        elif ch == ";" or ch == "|" or (ch == "&" and not command.startswith("&>", i)
                                        and (i == 0 or command[i - 1] not in "<>")):
            segs.append("".join(cur))
            cur = []
            if command.startswith(("&&", "||"), i):
                i += 1
        else:
            cur.append(ch)
        i += 1
    segs.append("".join(cur))
    return [x.strip() for x in segs if x.strip()]


def _safe(tok):
    """Only short plain words may enter a stored key: no quotes, '=', '/', '@', URLs or long blobs."""
    return tok if _SAFE_TOKEN.fullmatch(tok) else None


_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPERS = {"sudo", "time", "nice", "nohup", "env", "command", "exec", "stdbuf", "xvfb-run"}
_NOISE = {"cd", "pushd", "popd", "export", "source", ".", "set", "echo", "printf", "true", "false",
          "sleep", "clear", "unset", "mkdir", "ls", "pwd", "cat", "head", "tail", "wc", "sort", "grep"}
_TWO_WORD = {"go", "cargo", "dotnet", "docker", "docker-compose", "git", "kubectl", "terraform", "mvn",
             "gradle", "./gradlew", "gradlew", "make", "systemctl", "helm", "swift", "flutter", "dart",
             "mix", "rake", "bundle", "composer", "pip", "pip3", "brew", "apt", "apt-get", "nginx",
             "deno", "pnpm", "yarn", "bun", "npm", "uv", "poetry", "pipenv", "hatch", "rye", "just"}
_RUNNERS = {"npx", "bunx", "uvx", "pnpx"}

# Matched against the start of each command (after env assignments, wrappers and runners such as
# `npx`, `uv run`, `python -m` are stripped), so `grep pytest` or `cat jest.config.js` are not tests.
_KIND_PATTERNS = [
    ("test",
     r"(pytest|py\.test|jest|vitest|mocha|ava|rspec|phpunit|ctest|tox|nox|unittest|playwright test|"
     r"cypress run|go test|cargo (?:test|nextest)|dotnet test|deno test|bun test|swift test|mix test|"
     r"flutter test|(?:python\S* )?\S*manage\.py test|(?:npm|pnpm|yarn|bun) (?:run )?test\S*|make (?:test|check)\S*|"
     r"(?:mvn|gradlew?)\b.*\btest)\b"),
    ("lint",
     r"(eslint|ruff|flake8|pylint|mypy|pyright|golangci-lint|shellcheck|stylelint|cargo clippy|"
     r"tsc\b.*--noEmit|prettier\b.*--check|black\b.*--check|biome (?:check|lint)|"
     r"(?:npm|pnpm|yarn|bun) (?:run )?(?:lint|typecheck|type-check|check)\S*|nginx -t)\b"),
    ("build",
     r"(tsc|cargo build|go build|cmake --build|ninja|webpack|esbuild|rollup|next build|vite build|"
     r"docker build|docker compose build|dotnet build|swift build|"
     r"(?:npm|pnpm|yarn|bun) (?:run )?build\S*|mvn\b.*\b(?:package|install|compile)|"
     r"gradlew?\b.*\b(?:build|assemble)|make)\b"),
    ("install",
     r"(npm (?:install|ci|i)|pnpm (?:install|i|add)|yarn(?: install| add)?$|bun (?:install|add)|"
     r"pip3? install|uv (?:sync|add|pip install)|poetry (?:install|add)|pipenv install|"
     r"cargo (?:fetch|add|install)|go (?:mod (?:download|tidy)|get)|apt(?:-get)? install|"
     r"brew install|bundle install|composer install)\b"),
    ("vcs", r"(?:git|gh|hg|svn)\b"),
]
_RUNNER_PREFIXES = [("uv", "run"), ("poetry", "run"), ("pipenv", "run"), ("hatch", "run"), ("rye", "run"),
                    ("pdm", "run"), ("bundle", "exec"), ("npm", "exec"), ("pnpm", "exec"), ("yarn", "exec")]


@functools.lru_cache(maxsize=None)
def _kind_patterns():
    # compiled on first use: most hook and status line calls never classify a shell command
    return [(kind, re.compile(rx)) for kind, rx in _KIND_PATTERNS]


def _tokens(segment):
    try:
        toks = shlex.split(segment, posix=True)
    except ValueError:
        toks = segment.split()
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if not out and (_ENV_ASSIGN.match(t) or t in _WRAPPERS):
            i += 1
            continue
        if not out and t == "timeout" and i + 1 < len(toks):
            i += 2
            continue
        out.append(t)
        i += 1
    return out


_SHELL_KEYWORDS = {"do", "then", "else", "elif", "{", "(", "!"}
_SHELL_HEADERS = {"for", "while", "until", "if", "case", "done", "fi", "esac", "}", ")", "function"}


def _significant(toks):
    """Strip shell control keywords and redirections/heredocs from a segment's tokens."""
    while toks and toks[0] in _SHELL_KEYWORDS:
        toks = toks[1:]
    if not toks or toks[0] in _SHELL_HEADERS:
        return []
    out = []
    for t in toks:
        if t.startswith(("<", ">", "2>", "&>", "1>")) or "<<" in t:
            break
        out.append(t)
    return out


def _key_from_tokens(toks, depth=0):
    toks = _significant(toks)
    if not toks:
        return None
    head = _safe(os.path.basename(toks[0]))
    if head is None:
        return None
    if head in _SHELLS and len(toks) > 2 and toks[1] == "-c" and depth < 2:
        return _key_of_command(toks[2], depth + 1) or head
    rest = [t for t in (_safe(x) for x in toks[1:] if not x.startswith("-")) if t]
    if head in ("python", "python3", "python3.11", "python3.12", "node", "ruby", "perl", "bash", "sh", "php"):
        if len(toks) > 2 and toks[1] == "-m":
            mod = _safe(toks[2])
            return "%s -m %s" % (head.rstrip("0123456789."), mod) if mod else head
        if rest:
            return "%s %s" % (head.rstrip("0123456789."), os.path.basename(rest[0]))
        return head
    if head in _RUNNERS and rest:
        return "%s %s" % (head, rest[0])
    if head in ("npm", "pnpm", "yarn", "bun") and rest:
        if rest[0] == "run" and len(rest) > 1:
            return "%s run %s" % (head, rest[1])
        return "%s %s" % (head, rest[0])
    if head in ("uv", "poetry", "pipenv", "hatch", "rye") and len(rest) > 1 and rest[0] == "run":
        return "%s run %s" % (head, os.path.basename(rest[1]))
    if head in _TWO_WORD and rest:
        return "%s %s" % (head, rest[0])
    return head


def _command_lines(command, depth=0):
    """Each simple command as it would be matched against _KIND_PATTERNS: env assignments, wrappers,
    runners (`npx`, `uv run`, `python -m`, ...) and redirections removed; `sh -c "..."` expanded."""
    out = []
    for seg in _split_segments(command):
        toks = _significant(_tokens(seg))
        if not toks:
            continue
        head = os.path.basename(toks[0])
        if head in _SHELLS and len(toks) > 2 and toks[1] == "-c" and depth < 2:
            out += _command_lines(toks[2], depth + 1)
            continue
        while True:
            if head in _RUNNERS and len(toks) > 1:
                toks = [t for t in toks[1:] if not t.startswith("-")] or toks[:1]
            elif (head, toks[1] if len(toks) > 1 else None) in _RUNNER_PREFIXES and len(toks) > 2:
                toks = toks[2:]
            elif head.startswith("python") and len(toks) > 2 and toks[1] == "-m":
                toks = toks[2:]
            else:
                break
            head = os.path.basename(toks[0])
        out.append(" ".join([head] + toks[1:]))
    return out


def shell_command_info(command):
    """Return (cmd_key, cmd_kind) for a shell command line.

    cmd_key groups repeated commands ("npm test", "pytest", "cargo build") so their historical
    durations can predict how long an in-flight run of the same command will take.
    """
    if not command or not isinstance(command, str):
        return None, "other"
    command = command.strip()
    lines = _command_lines(command)
    kind = "other"
    for want, rx in _kind_patterns():
        if any(rx.match(line) for line in lines):
            kind = want
            break
    return _key_of_command(command), kind


def _key_of_command(command, depth=0):
    segments = _split_segments(command)
    for seg in segments:
        toks = _significant(_tokens(seg))
        if toks and os.path.basename(toks[0]) not in _NOISE:
            key = _key_from_tokens(toks, depth)
            if key:
                return key[:48]
    if segments:
        key = _key_from_tokens(_tokens(segments[0]), depth)
        return key[:48] if key else None
    return None


def tool_key(tool_name, tool_input):
    """(cmd_key, cmd_kind) for any tool call; cmd_key is the duration-history bucket."""
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    if tool_name in SHELL_TOOLS:
        return shell_command_info(tool_input.get("command"))
    if tool_name in AGENT_TOOLS:
        return "%s:%s" % (tool_name, tool_input.get("subagent_type") or "general"), "agent"
    return tool_name, tool_kind(tool_name)


def safe_summary(tool_name, tool_input):
    """What may be stored and shown about a tool call: its kind, never its arguments.

    Shell commands keep only the grouping key ("npm test", "curl"): flags, URLs, paths, quoted
    strings and env assignments (where tokens and passwords live) are never persisted.
    """
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    if tool_name in SHELL_TOOLS:
        key, _ = shell_command_info(tool_input.get("command"))
        return "%s: %s" % (tool_name, key) if key else tool_name
    if tool_name in AGENT_TOOLS:
        return "%s: %s" % (tool_name, tool_input.get("subagent_type") or "general")
    return tool_name or "?"


# ---------------------------------------------------------------- prompts

_CATEGORIES = [
    # (category, english word regex, chinese substrings)
    ("fix", r"fix|bug|bugs|error|errors|broken|crash|crashes|failing|fails|failed|issue|debug|wrong",
     ["修", "报错", "错误", "异常", "崩", "不工作", "不行", "失败", "问题", "怪", "排查", "不对", "坏了"]),
    ("refactor", r"refactor|cleanup|clean up|rename|restructure|simplify|reorganize",
     ["重构", "优化", "整理", "简化", "拆分", "改名"]),
    ("test", r"tests?|coverage|unittest", ["测试", "单测", "用例"]),
    ("review", r"review|audit|inspect|check", ["检查", "审查", "审核", "看看", "看一下", "评审"]),
    ("ops", r"deploy|install|configure|config|server|nginx|docker|systemd|ssl|cert|dns|migrate|backup",
     ["部署", "安装", "配置", "服务器", "证书", "域名", "迁移", "备份", "上线"]),
    ("feature", r"add|implement|create|build|write|make|generate|design|support|scaffold",
     ["做", "写", "实现", "添加", "加上", "新增", "设计", "生成", "开发", "创建", "搭建", "打包", "制作"]),
    ("question", r"what|why|how|explain|which|where|when|is there|can you tell",
     ["什么", "为什么", "怎么", "如何", "解释", "是否", "吗", "呢", "哪"]),
]


@functools.lru_cache(maxsize=None)
def _category_patterns():
    return [(c, re.compile(r"\b(?:%s)\b" % en, re.I), zh) for c, en, zh in _CATEGORIES]


SYSTEM_PROMPT_PREFIXES = ("<task-notification>",)


def is_system_prompt(text):
    """Prompts Claude Code submits by itself (e.g. a background agent finished), not the user."""
    return (text or "").lstrip().startswith(SYSTEM_PROMPT_PREFIXES)


# A `!` shell command you type fires no UserPromptSubmit. Claude Code records it as <bash-input>, then
# (once it has finished) its output as <bash-stdout>/<bash-stderr>, both under a new prompt_id, and
# Claude answers the output. That answer is a run of its own ("bash"), timed from the output.
BASH_INPUT_PREFIX = "<bash-input>"
BASH_OUTPUT_PREFIXES = ("<bash-stdout>", "<bash-stderr>")
NON_PROMPT_CATEGORIES = ("wake", "bash")  # runs not opened by a prompt you typed


def bash_part(text):
    """'input' / 'output' if a user message is part of a `!` shell command, else None."""
    t = (text or "").lstrip()
    if t.startswith(BASH_INPUT_PREFIX):
        return "input"
    if t.startswith(BASH_OUTPUT_PREFIXES):
        return "output"
    return None


def content_text(content):
    """Text of a transcript message's content (a plain string or a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def iso_ts(value):
    """Epoch seconds of a transcript ISO timestamp, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def classify_prompt(text):
    t = (text or "").strip()
    if not t:
        return "wake"
    if t.startswith("/"):
        return "command"
    scores = {}
    for cat, rx, zh in _category_patterns():
        s = len(rx.findall(t)) + sum(t.count(w) for w in zh)
        if s:
            scores[cat] = s
    if not scores:
        return "followup" if len(t) <= 16 else "other"
    if set(scores) == {"question"} or (t.rstrip().endswith(("?", "？")) and len(scores) == 1):
        return "question"
    scores.pop("question", None)
    order = [c for c, _, _ in _CATEGORIES]
    return max(scores, key=lambda c: (scores[c], -order.index(c)))


def prompt_features(text):
    t = text or ""
    return {
        "len": len(t),
        "lines": t.count("\n") + 1 if t else 0,
        "code": 1 if "```" in t else 0,
        "category": classify_prompt(t),
        "preview": " ".join(t.split())[:80],
    }


# ---------------------------------------------------------------- misc

def repo_root(cwd):
    if not cwd:
        return None
    p = os.path.abspath(cwd)
    probe = p
    for _ in range(64):
        if os.path.exists(os.path.join(probe, ".git")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return p


def effort_level(ev, allow_env=True):
    e = ev.get("effort") if isinstance(ev, dict) else None
    if isinstance(e, dict):
        e = e.get("level")
    if isinstance(e, str) and e:
        return e
    if allow_env:
        return os.environ.get("CLAUDE_EFFORT") or None
    return None


def perm_class(mode):
    if mode in ("bypassPermissions", "dontAsk", "auto"):
        return "autonomous"
    return "interactive"


def phase_of(run):
    """Coarse stage of a run from its counters (a Row or dict)."""
    n_tools = run["n_tools"] or 0
    if n_tools == 0:
        return "start"
    last_edit = run["last_edit_at"] or 0
    last_verify = run["last_verify_at"] or 0
    if last_verify and last_verify >= last_edit:
        # "verify" = the latest test/build/lint passed and nothing changed since: usually the tail end
        return "verify" if run["last_verify_ok"] else "fixing"
    if (run["n_verify_fail"] or 0) > 0:
        return "fixing"
    if (run["n_edit"] or 0) > 0:
        return "implement"
    return "explore"
