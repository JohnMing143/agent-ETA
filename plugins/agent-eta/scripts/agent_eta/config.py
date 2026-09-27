"""User config and on-disk locations.

Everything lives in one stable directory (default ~/.claude/agent-eta, override with AGENT_ETA_HOME)
rather than ${CLAUDE_PLUGIN_DATA}: the status line command needs a path that survives plugin
updates, and the learned history should survive an uninstall/reinstall.
"""
import json
import os

DEFAULTS = {
    "lang": "auto",            # auto | zh | en
    "safe_quantile": 0.2,      # "safe to leave for X" = this quantile of time-to-attention
    "wrap_command": None,      # a pre-existing statusLine command whose output we keep showing
    "previous_statusline": None,  # the full statusLine setting we replaced, restored on remove
    "compose": "newline",      # newline | inline  (how our segment joins the wrapped output)
    "show_idle": True,         # show a one-line summary when no run is active
    "prediction_log_interval_s": 20,
    "store_prompt_preview": False,  # keep the first 80 chars of each prompt (off: only length/category)
    "cold_start": "estimate",  # estimate: show prior-based numbers, marked; unknown: show nothing
}


def home():
    return os.environ.get("AGENT_ETA_HOME") or os.path.join(os.path.expanduser("~"), ".claude", "agent-eta")


def path(*parts):
    return os.path.join(home(), *parts)


def load():
    cfg = dict(DEFAULTS)
    try:
        with open(path("config.json"), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cfg.update(data)
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg):
    os.makedirs(home(), exist_ok=True)
    keep = {k: v for k, v in cfg.items() if k in DEFAULTS}
    write_atomic(path("config.json"), json.dumps(keep, indent=2, ensure_ascii=False) + "\n")


def write_atomic(target, text, mode=None):
    tmp = "%s.tmp-%d" % (target, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, target)


def lang(cfg):
    value = (cfg or {}).get("lang") or "auto"
    if value in ("zh", "en"):
        return value
    env = os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG") or ""
    return "zh" if env.lower().startswith("zh") else "en"
