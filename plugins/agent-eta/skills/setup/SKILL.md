---
name: setup
description: Install the Agent ETA status line (time-to-completion and "safe to leave" countdown) and import past Claude Code transcripts as history. Arguments "remove" uninstalls the status line, "doctor" checks the wiring.
argument-hint: "[remove | doctor] [--lang zh|en] [--refresh SECONDS]"
disable-model-invocation: true
allowed-tools: Bash(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py *)
---

# Agent ETA setup

Arguments: `$ARGUMENTS`

Run exactly one of these, based on the arguments (keep the command text exactly as written so it
matches the pre-approved permission):

- contains `remove` →
  `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py setup --remove`
- contains `doctor` →
  `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py doctor`
- otherwise (install) →
  `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py setup --lang LANG --skip-session ${CLAUDE_SESSION_ID}`
  where LANG is the `--lang` argument if given, else `zh` if the user writes to you in Chinese,
  else `en`. Pass `--refresh N` through if the user gave one.

What install does (tell the user if they ask): it points `statusLine` in `~/.claude/settings.json`
at `~/.claude/agent-eta/statusline.sh` with a 5 s refresh (a backup is written to
`settings.json.agent-eta-backup`; an existing status line is kept and shown above the ETA), and it
imports past transcripts from `~/.claude/projects` so estimates are personalised from day one.

After a successful install, also run
`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py report --days 90`
and reply briefly, in the user's language:

1. What was installed and how many past runs were imported.
2. How to read the status line, e.g.
   `⏱ 3:12 │ 完成 ~4–9m │ 可离开 ~3m → 14:52 │ ▶ npm test 0:41 │ ☰ 3/5 │ 置信 中`
   - `⏱` agent-active time of the current turn (time spent waiting on the user is excluded)
   - `完成 / done` likely (P50) to pessimistic (P80) remaining time until Claude hands back control
   - `可离开 / safe` how long you can step away: 80 % chance Claude will not need you before then
     (permission prompt, question, plan approval, or finishing), with the clock time
   - `▶` a long-running tool call right now and how long it has run; `☰` task list progress
   - `⚠ 需要你 / NEEDS YOU` replaces the line while Claude is waiting on the user
3. Two or three numbers from the report (typical run length, the slowest commands). If fewer than
   about 20 runs exist, say the estimates will be rough at first and sharpen with use.
4. Undo: `/agent-eta:setup remove`. Health check: `/agent-eta:setup doctor`.

If the command fails, show the error and suggest `/agent-eta:setup doctor`. Do not edit
`settings.json` by hand.
