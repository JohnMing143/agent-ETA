# Agent ETA — "how long will Claude take, and can I walk away?"

[中文说明](plugin.zh-CN.md)

> **Read this first.** This page documents the plugin. What we learned building it — including why
> "safe to leave" windows are short by nature (about 1–2 minutes at 80 % confidence) and how reliable
> they turned out across users — is in the [research write-up](../README.md). In short: good for
> seeing where your agent's time goes and for honest ranges; it does not free large blocks of time.


A Claude Code plugin that puts two numbers in your status line: **how long until Claude hands control
back**, and **how long you can safely step away** before it needs you. Both are learned from your own
history and keep improving as you use Claude Code. No model calls, nothing injected into the context,
zero tokens.

```
⏱ 3:12 │ done ~4–9m │ safe ~3m → 14:52 │ ▶ npm test 0:41 │ ☰ 3/5 │ conf med
⚠ NEEDS YOU: permission Bash: rm · waiting 0:37              ← replaces the line while Claude waits on you
⏱ ETA · last 6:12 (pred 5–9m ✓) · 57 runs learned             ← between turns
```

## Why

Once you hand a task to an agent you don't know whether it will be back in 2 minutes or 20, so you
can't start anything else. Agent ETA answers one question: **can I leave for a while right now?**

- **done (T_c)**: time until Claude returns control, as a range (P50–P80), not a fake-precise point.
- **safe (T_a)**: time until Claude next needs you: a permission prompt, a question, plan approval, or
  finishing. The number is deliberately cautious (80 % sure nothing needs you before then), and it
  promises less while your history is thin.
- **Behaviour, not self-report.** The plugin only watches hook events. It never asks the model how far
  along it is.

## It keeps learning

- **Day one:** setup imports your past transcripts (`~/.claude/projects`), so estimates are personal
  from the start.
- **Every turn** becomes a labelled example: the trajectory at each step, how long was left, when you
  were needed. Interrupted turns count as censored data rather than being thrown away.
- **Your pace:** the built-in prior is calibrated to how fast *your* tasks run (overall, per task type,
  and how variable they are). This works from about 5 runs on.
- **How much to trust history:** once 10 runs have finished, and again each time your history grows by
  10 %, a background process replays your history and picks the estimator settings that would have
  predicted it best:
  - `K0`: how much similar past runs outweigh the calibrated prior. A few dozen varied runs are
    predicted best by the prior alone; hundreds of similar runs by the neighbours.
  - Recency half-life (0.5 / 2 / 7 / 60 days): how fast older runs fade. Habits drift; for some
    histories today's runs predict better than last month's. A shorter memory has to beat the
    longest one by more than 1 % to be taken.
  - Quantile recalibration: if the estimates run systematically long or short, the curves are read
    at corrected levels learned from your own outcomes (where the truth fell in each prediction). It
    is judged prequentially and switched on only if it lowers the loss.
  - `SAFE_K`: how cautious the leave window is. It picks the most generous setting whose replayed
    breach rate stays under 20 % (credibly so when there is enough data). No leave window is shown
    before the first tune: until then the status line says `leave: unverified`.
- **Real-world check:** every "safe to leave" window the status line actually shows is recorded with
  its fixed expiry and scored afterwards. If real windows are credibly breached too often (the breach
  rate's lower bound is over 20 %), the next re-tune makes them more cautious. After 100+ windows under the target, the status line adds `✓`.

`eta.py report` shows all of this: accuracy against baselines, the current settings, and why they were
chosen. `eta.py eval` answers "is the learning actually paying off?" with 95 % intervals (see below).

## Install

Requires Claude Code 2.1.2xx or newer (tested on 2.1.280) and `python3` ≥ 3.8 (tested on 3.8 and
3.13). Standard library only, nothing to pip install.

```bash
claude plugin marketplace add JohnMing143/agent-ETA   # or a local path to this repository
claude plugin install agent-eta@agent-eta
```

Restart Claude Code, then run this in a session:

```
/agent-eta:setup
```

Setup does four things:

1. It points `statusLine` in `~/.claude/settings.json` at `~/.claude/agent-eta/statusline.sh`
   (5 s refresh). It writes a backup to `settings.json.agent-eta-backup` first. If you already have a
   status line, it is kept and shown above the ETA line.
2. It imports past transcripts as history.
3. With at least 10 runs, it tunes the estimator on that history.
4. It prints a report.

Plugins cannot set the status line themselves, so this one-time setup step is needed. Plugin updates
move the plugin directory, but you don't need to run setup again: every session start refreshes the
pointer the launcher follows.

Try it without installing: `claude --plugin-dir path/to/agent-eta/plugins/agent-eta`.

## Use

| You want | Do |
|---|---|
| How long the current turn will take | Look at the status line |
| History, accuracy, what other sessions are doing | `/agent-eta:eta`, or just ask Claude "can I step away?" |
| A live board of all sessions in another terminal | `python3 <plugin>/scripts/eta.py watch` (`/agent-eta:eta watch` prints the path) |
| Check the wiring | `/agent-eta:setup doctor` |
| Remove the status line | `/agent-eta:setup remove` |

| Field | Meaning |
|---|---|
| `⏱ 3:12` | Agent-active time of this turn (time spent waiting on you is excluded) |
| `done ~4–9m` | Likely (P50) to pessimistic (P80) time until Claude hands back control. `↑`/`↓`: the estimate just moved a lot |
| `safe ~3m → 14:52` | How long you can leave, and until what time. Once shown, the expiry is fixed; it is revoked (never silently extended) if Claude needs you or the estimate collapses |
| `▶ npm test 0:41` | A long tool call in flight, predicted from that command's own history in this repo |
| `☰ 3/5` | Claude's task list progress (only when the model uses one) |
| `⏳ bg wait ×2` | Claude is waiting for background agents and will resume on its own |
| `conf low/med/high` | Evidence behind the estimate; `conf low·prior` = built-in prior only |

Command line (`scripts/eta.py`):

```
eta.py status [--session ID]     # detailed view: progress bar, reasons, leave-until time
eta.py watch  [--interval 2]     # live board of all sessions
eta.py report [--days 30]        # history, window ledger, backtest vs built-in prior and B0, tuning
eta.py tune                      # re-tune now (normally happens in the background)
eta.py eval [--curve] [--json]   # is the learning paying off? gains vs no learning, with 95 % intervals
eta.py backfill [-v] [--force]   # re-import transcripts
eta.py doctor                    # self-check
eta.py setup [--lang zh|en] [--refresh 5] [--no-backfill] | --remove
```

## Privacy

- Everything stays in `~/.claude/agent-eta/eta.db` (SQLite, file mode 0600, directory 0700). The plugin
  makes no network requests.
- What is stored: the length and category of each prompt, tool names, a command's *grouping key*
  (`npm test`, `curl`, `git push`), and timings. Not stored: prompt text, command arguments (URLs,
  tokens, passwords live there), file contents, command output.
- Each word of a grouping key must pass an allow-list (short plain words, no quotes, `=`, `/`, `@`).
  `bash -c "…"` is resolved to the inner command, and here-document bodies are skipped. A test runs a
  `curl` with a Bearer token and then searches the database files for it.
- To erase everything, delete `~/.claude/agent-eta/`.

Config (`~/.claude/agent-eta/config.json`):

| Key | Default | Meaning |
|---|---|---|
| `lang` | `auto` | `zh` / `en` / `auto` |
| `safe_quantile` | `0.2` | Risk level of "safe": 0.2 = 80 % sure |
| `compose` | `newline` | How to join a pre-existing status line: `newline` or `inline` |
| `show_idle` | `true` | Show a summary line between turns |
| `store_prompt_preview` | `false` | Keep the first 80 characters of each prompt to tell tasks apart in `status` |
| `cold_start` | `estimate` | `unknown`: show no numbers until there is personal history |

## Accuracy, honestly

Every report shows the current estimator next to the built-in prior and the B0 baseline from the Codex
design (the history of run lengths, with no trajectory features). Numbers below are mean pinball loss
of done P50/P80 (lower is better). Each prediction uses only runs that had finished before it.

| History | built-in prior | B0 | Agent ETA |
|---|---|---|---|
| Simulated, 240 runs | 46.5 | 47.7 | **26.3** |
| Simulated, 40 runs (3 seeds) | 44–46 | 51–55 | **29–32** |
| One real user, 38 very mixed runs | 113.4 | 149.1 | **109.3** |

With 38 very mixed real runs, estimates are still rough (median P50 error ×2.3). That is what
"day one" looks like; the report shows when learning starts to pay off.

`eta.py eval` checks the learning itself, on your own history. Every point is predicted only from runs
that had finished before it (the recalibration too), and the gain over the built-in prior (no learning
at all) gets a 95 % interval from a cluster bootstrap over runs: a gain counts as shown only when the
interval excludes zero. It also splits the gain by how much history existed, and `--curve` measures
the marginal value of more history (the most recent runs predicted from n random earlier ones).
On the author's 67 real runs: +7.4 % [−0.3 %, +13.7 %] overall (P50 hit 67 % → 54 %, typical error
×2.43 → ×2.12), +14.0 % [+3.4 %, +22.0 %] on the most recent third; older history adds nothing
measurable. The gain comes from recent context and recalibration, not from a long memory.

### Across users

Checked on six histories: the author's plus five contributors to the public
[trace-commons/agent-traces](https://huggingface.co/datasets/trace-commons/agent-traces) dataset
(raw Claude Code sessions, CC-BY-4.0), 405 runs in all, each contributor tuned as a separate user.
[DESIGN.zh-CN.md](DESIGN.zh-CN.md) section 7 has the full experiments.

- **Learning pays where the default pace is wrong.** One contributor (267 runs, median turn 52 s,
  twice as fast as the built-in pace): +27.7 % [+19.2 %, +35.6 %] over the built-in prior, typical
  error ×4.05 → ×2.52, almost all of it from pace calibration. Histories near the default pace
  gain little (−3 % to +7 %, intervals include 0) - the prior already fits them, and learning
  does not hurt.
- **Models differ, but the work matters more.** Five more public datasets of Claude Code sessions
  (Fable 5 ×2, a human-plus-automation history on Fable 5 / Opus 4.8 / Sonnet 4.6, and automated runs
  with Kimi K2.6, MiniMax M3 and MiMo V2.5 Pro; 1,676 turns). Model time per step (tool time
  excluded) is 7–16 s across Claude models and 17 s for MiniMax M3; Kimi K2.6 takes ~100 s but runs
  far fewer steps. MiniMax M3 stopped to ask the user in a third of its turns (Claude models 1–10 %).
  Yet the built-in prior fits interactive coding on all of them (P50 hit 51–57 %); what breaks it is
  the workload - a stream of few-second tasks, or a user whose turns mostly take under a minute -
  and there learning helps a lot (+38 % to +83 %). A per-model pace on top of the per-category one
  helped only a history mixing vendors (+1.4 % [+0.6, +2.3]) and slightly hurt within the Claude
  family (−1.5 %), so it is not used; the short recency half-life already follows model switches.
- **A turn's length is hard to predict from its start, for everyone:** after calibration the
  remaining spread is σ≈1.3–1.4 in log space (a 50 % interval about ×2.3 wide).
- **"Safe to leave" windows, replayed through time** (every re-tune sees only its past, windows feed
  the ledger as in live use). On these six histories: 132 promises, 16 % breached. On five more public
  datasets (below): 268 promises, 29 %. **All eleven: 400 promises, 24.5 % breached (95 % upper bound
  28 %) - above the 20 % target.** Some histories stay within it (8 %, 17 %, 19 %), others do not
  (21 %–45 %). Nearly every breach is Claude *finishing early* against a one-minute promise made early
  in a turn: turn lengths are bimodal (quick answers vs. longer tasks) and the log-normal model
  underrates the quick mode. Until that is fixed, trust a window once your own ledger shows `✓`
  (100+ real windows within the target). No promise is made before the estimator has been tuned on
  at least 10 runs: untuned promises were breached 21 % of the time, up to 3 of 5 on one history. The design, the algorithms and
all validation are in [DESIGN.zh-CN.md](DESIGN.zh-CN.md) (Chinese).

## Known limitations

- **Early estimates are coarse**, as above. The leave window is labelled unvalidated until the real
  window ledger passes (100+ windows under the target).
- **Permission prompts are learned from live use only.** Transcripts do not record them.
- **Task lists are rare.** Claude 5 models don't offer TaskCreate/TodoWrite by default, so `☰` rarely
  shows. Predictions don't depend on it.
- **Windows is untested.** The hooks call `python3`, which may not exist there, and CI runs Windows as
  experimental. Reports and fixes are welcome.

## Roadmap and contributing

The goal is a tool that works for any developer running coding agents, not one machine. Contributions
most wanted:

1. **Cross-platform hooks:** Windows and unusual Python setups.
2. **Opt-in shared priors:** an export that contains only aggregate statistics (per task type, effort
   and model: counts and log-duration mean/spread; no paths, repos, prompts or commands). You review the
   file and share it by hand, and it is merged into the built-in prior in releases, so new users start
   better calibrated. Nothing leaves your machine automatically.
3. **Codex adapter:** map Codex hooks/notify onto the same event contract so one estimator serves both
   agents (see [CODEX_COMPARISON.zh-CN.md](CODEX_COMPARISON.zh-CN.md)).
4. **Notifications:** push to your phone when Claude needs you, paired with the "safe until 14:52"
   promise.
5. **Retention:** optional expiry of old detail data.

Development:

```bash
cd plugins/agent-eta
python3 -m unittest discover -s tests        # 63 tests: simulation calibration, tuning, Codex math golden values, 16 Codex scenarios, privacy
claude plugin validate . --strict
```

When you change the estimator, backtest it on your own history (`AGENT_ETA_HOME=/tmp/x eta.py backfill`
followed by `eta.py report`) and on several simulated histories (`tests/sim.py`), not just one.

## Uninstall

```
/agent-eta:setup remove                      # restores your previous status line
claude plugin uninstall agent-eta@agent-eta
rm -rf ~/.claude/agent-eta                   # also delete the history
```

MIT licensed.
