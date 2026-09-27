# agent-ETA — can a coding agent tell you when you're free to walk away?

We built a Claude Code plugin to answer two questions **at the moment you hand a task to the agent**:
*how long will this take*, and *how long can I safely step away before it needs me*.
We then measured it as honestly as we could, on our own history and on public Claude Code session
datasets from other people and other models (≈2,000 turns, 11 histories).

**Short version:** an honest time estimate is possible, but only a rough one (×2.3–2.6 median error at
the start of a task). A *reliable and useful* "safe to leave" time is not, for a structural reason:
after its first minute, a coding agent hands control back at a nearly constant **12–19 % per minute**,
however long it has already been working. With no extra information, an 80 %-reliable window is
therefore 1–2 minutes at any moment. Most of what we tried to break that ceiling did not work; this
write-up records what we tried, what we measured, and what we would try next.

The plugin still works and is installable (see [docs/plugin.md](docs/plugin.md)); this README is
about what we learned.

[中文版](README.zh-CN.md)

---

## Contents

1. [What we built](#1-what-we-built)
2. [Data](#2-data)
3. [How we evaluated](#3-how-we-evaluated)
4. [Findings](#4-findings)
5. [Things that did not work](#5-things-that-did-not-work)
6. [Where the free time actually is](#6-where-the-free-time-actually-is)
7. [Pros and cons of the plugin as it stands](#7-pros-and-cons-of-the-plugin-as-it-stands)
8. [Claude Code hook pitfalls we hit](#8-claude-code-hook-pitfalls-we-hit)
9. [Methodology lessons](#9-methodology-lessons)
10. [Open questions](#10-open-questions)
11. [Reproduce](#11-reproduce)
12. [Data sources and licenses](#12-data-sources-and-licenses)

---

## 1. What we built

A passive Claude Code plugin (hooks + status line, pure Python standard library, no model calls, nothing
injected into the agent's context). Every turn becomes a labelled example: the trajectory at each tool
step, how long was left, and when you were needed (finished, asked a question, or asked for permission).
The status line shows something like:

```
⏱ 0:05 │ done ~2–8m │ safe ~1m → 14:32 │ confidence medium
```

The estimator:

- a built-in prior per task type (question / fix / feature / …, from keywords and prompt length, effort level),
  calibrated to **your pace** (overall and per task type, shrunk toward the default);
- Kaplan–Meier curves from **similar past moments** of your own runs (static features + trajectory:
  tool count, failures, phase, plan progress), blended with the prior by a learned shrinkage;
- hazards for permission prompts and for known slow commands in flight;
- a "safe to leave" window: a low quantile of time-to-attention, recorded with a fixed expiry in a
  **ledger** and scored afterwards, so the promise is always checked against what really happened.

Everything is tuned per user by replaying their own history (see §3). Design details, in Chinese:
[docs/DESIGN.zh-CN.md](docs/DESIGN.zh-CN.md).

## 2. Data

| Source | Agent / models | Turns | Kind |
|---|---|---|---|
| The author's own history | Claude Code, Opus 5 / Opus 5.5 | 68 | real use |
| [trace-commons/agent-traces](https://huggingface.co/datasets/trace-commons/agent-traces), 5 contributors ("A"–"E") | Sonnet 4.6, Opus 4.8 | 337 (267 / 34 / 15 / 11 / 10) | real use, donated |
| [AlinCiocan/fable-5-claude-code-traces](https://huggingface.co/datasets/AlinCiocan/fable-5-claude-code-traces) | Fable 5 | 68 | real use |
| [armand0e/claude-fable-5-claude-code](https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code) | Fable 5 | 267 | pooled real traces |
| [crispwisp/wisp-claude-code-sessions](https://huggingface.co/datasets/crispwisp/wisp-claude-code-sessions) | Fable 5, Opus 4.8, Sonnet 4.6 | 187 | one human + automated benchmark runs |
| [armand0e/kimi-k2.6-claude-code-traces](https://huggingface.co/datasets/armand0e/kimi-k2.6-claude-code-traces) | Kimi K2.6 in Claude Code | 116 | automated |
| [armand0e/minimax-m3-claude-code-traces](https://huggingface.co/datasets/armand0e/minimax-m3-claude-code-traces) | MiniMax M3 in Claude Code | 64 | automated |
| [choucsan/mimo-claude-code-traces-1k](https://huggingface.co/datasets/choucsan/mimo-claude-code-traces-1k) | MiMo V2.5 Pro | 974 | automated few-second tasks |

All are native Claude Code transcript JSONL, imported with the plugin's own importer. Each history was
tuned and evaluated as a separate user. Transcripts do not record permission prompts, so imported
histories under-count one kind of "needs you" event. No raw session content from any source is
published here; only aggregate numbers.

## 3. How we evaluated

We learned early that it is very easy to fool yourself here, so every number below follows these rules:

- **Prequential.** Every prediction uses only runs that had finished before it. Nothing sees the future,
  including the recalibration and the tuning.
- **Replay the whole live system through time.** For the leave-window results we re-ran the plugin as if
  it had been installed from day one: at every moment it would re-tune (after 10 runs, then every +10 %),
  we tuned on a copy of the database truncated to that moment, used those settings until the next
  re-tune, and fed every promise it showed into the ledger, so its feedback loop acted as in real use.
- **Score what the user sees.** A leave promise counts only if it would have been displayed (≥ 1 min,
  rounded down to the displayed bucket). Per-moment coverage can look perfect while displayed promises
  fail badly (see ACI in §5).
- **Uncertainty by cluster bootstrap over runs.** Points from one run are not independent. A gain is
  called "shown" only when its 95 % interval excludes zero.
- **Loss:** log-scale pinball loss of the displayed quantiles (done P50 and P80, attention P20).

The plugin ships this as `eta.py eval` so anyone can check the claims on their own history.

## 4. Findings

### 4.1 At the start of a task, duration is a ×2.5 guess — for everyone

After calibrating each user's pace per task type, the remaining spread of log turn length is
σ ≈ 1.27 (author) to 1.43 (pooled public data): a 50 % interval about ×2.3 wide. We tried every source
of information available at the start:

| Information at the start of a task | Median error | Correlation with actual (log) |
|---|---|---|
| prompt length + keyword task type + your pace (plugin) | ×2.60 | 0.1–0.4 |
| + your most similar past prompts (char n-gram TF-IDF, local) | ×2.51 | – |
| waiting 15 / 30 / 60 s to see the agent's first moves | ×2.4–2.7 | – |
| an LLM (Claude Haiku) reading the request and the previous reply, calibrated to you | ×2.46–2.59 | 0.3–0.5 |
| blend of plugin + LLM | **×2.29** | – |
| the agent's own plan (TaskCreate / TodoWrite), when it makes one | ×1.62 on one user (21 tasks), worse on another | 0.77 / 0.18 |

The request's text alone does not carry the information: "fix this bug" can be one minute or thirty
depending on what the agent finds. The only signal that clearly broke through was the agent's own plan
after it had looked at the code — and most agents in our data almost never make one.

### 4.2 The agent hands control back at a nearly constant rate

Probability that the agent needs you (finishes, asks, or requests permission) in the next minute,
918 finished tasks across 8 histories:

| Already running | Tasks still running | Needs you in the next minute | 80 %-reliable window implied |
|---|---|---|---|
| 0 min | 100 % | 39 % | 0.4 min |
| 1 min | 61 % | 19 % | 1.0 min |
| 2 min | 49 % | 18 % | 1.1 min |
| 3 min | 40 % | 13 % | 1.6 min |
| 5 min | 30 % | 14 % | 1.5 min |
| 8 min | 19 % | 12 % | 1.8 min |
| 12 min | 12 % | 11 % | 1.9 min |

After the first minute the hazard barely falls: how long a task has run tells you little about how long
it has left. This is a property of interactive coding agents — they can decide they are done after
almost any step — not of the estimator. It explains everything in §5: without task-specific information,
an 80 % promise is 1–2 minutes long, whatever method you use. A "wait 90 s; if it is still running you
can leave for W minutes" plan gives W ≈ 1–1.5 min for the same reason.

Relaxing the confidence does not help much either (pooled, replay):

| Confidence | Moments with a promise | Average promise | Breached |
|---|---|---|---|
| 90 % | 3 % | 1.2 min | 21 % |
| 80 % | 17 % | 1.3 min | 21 % |
| 70 % | 38 % | 1.4 min | 27 % |
| 60 % | 58 % | 1.7 min | 35 % |
| 50 % | 70 % | 2.3 min | 46 % |

### 4.3 "Safe to leave" reliability varies a lot between users

Replaying the final design through time on all 11 histories: **400 displayed promises, 24.5 % breached
(95 % upper bound 28.3 %)** — above the 20 % target. Across histories with at least 20 promises it
ranged from 8 % to 45 %. Nearly every
breach was the agent *finishing early* against a one-minute promise made early in a turn. Turn lengths
are bimodal (quick answers of tens of seconds vs. multi-minute work), and histories with many quick
answers break short promises most.

In the author's real use (not a replay): of 281 minutes of agent work, 219 minutes (78 %) sat in gaps of
3 minutes or more where one could have left. The plugin's windows that held add up to 138 minutes —
but they were **121 one-minute windows, 7 two-minute windows and 1 three-minute window; none of 5 minutes
or more**. On paper that covers 63 % of the free time; in practice it frees almost none.

### 4.4 What learning is good for

`eta.py eval`, learned estimator vs. the built-in prior (no learning):

| History | Gain | Why |
|---|---|---|
| automated few-second tasks (MiMo) | +82.8 % [+80.8, +84.5] | the default pace is off by ×36 |
| human + automation (wisp) | +38.2 % [+29.6, +46.1] | many quick turns |
| contributor A (267 turns, median 52 s) | +27.7 % [+19.2, +35.6] | twice as fast as the default |
| pooled Fable 5 traces | +7.0 % [+1.4, +12.0] | |
| the author (68 turns) | +7.4 % [−0.3, +13.7]; most recent third +14.0 % [+3.4, +22.0] | recency + recalibration |
| histories close to the default pace | −3 % to +5 %, intervals include 0 | the prior already fits |

Learning mostly pays when a user's pace is far from the default. Beyond calibration, longer memory
adds little: the marginal value of older history was +1.0 % [−2.4, +4.3] for the author; what helped
was weighting recent runs (a 0.5-day half-life won for two histories, 60 days for others — the plugin
picks it per user) and a PIT quantile recalibration that fixed systematically long estimates
(P50 hit 67 % → 54 %). A two-mode (quick answer + work) prior improved day-one accuracy for new users by
**+8.1 % [+6.9, +9.2]** across 11 histories, but not the leave windows (§5); it is kept as a patch in
[research/](research/).

### 4.5 Models differ, but the workload matters more

| Model | Model time per step (tool time excluded) | Stops to ask the user | Built-in prior P50 hit |
|---|---|---|---|
| Sonnet 4.6 | 7.2 s | 3 % | – |
| Opus 5.5 / Opus 5 | 9.9 s / 11.3 s | 6 % / 3 % | – |
| Fable 5 | 13.1 s | 1–4 % | 55–57 % |
| Opus 4.8 | 15.5 s | 10 % | – |
| MiniMax M3 | 17.1 s | **33 %** | 53 % |
| Kimi K2.6 | ~100 s (tool time missing in that dataset) | 6 % | 51 % |

Per-step speed differs up to an order of magnitude and some models stop to ask far more often, yet for
interactive coding the same built-in prior fits them all (P50 hit 51–57 %). What breaks it is the kind
of work. A per-model pace on top of the per-task-type one slightly hurt within the Claude family
(−1.5 % [−2.4, −0.7]) and helped only a history mixing vendors (+1.4 % [+0.6, +2.3]).

## 5. Things that did not work

Kept here because they are the most useful part for anyone trying the same thing.

| Idea | Result |
|---|---|
| **Adaptive conformal inference (ACI)** on the leave quantile | Drives the *per-moment* error to 20 % as promised, but most moments give windows under a minute that are never shown; the displayed long windows were breached 34–51 % on one user. Variants updated only by shown promises were no better than the plugin's tuned shrinkage. |
| Gate: show promises only when the replay credibly supports them | Fewer promises, no gain in reliability (e.g. 17 % → 18 %), and useless on the user who needed it most (33 %). |
| Always read the leave window through the recalibration | 17.4 % → 21.4 % breached. |
| Two-mode (quick + work) prior | Day-one accuracy +8 %, leave reliability unchanged in aggregate (24.5 % → 24.7 %): better for some users (45 % → 15 %), worse for others (17 % → 28 %). |
| Similar past prompts | ×2.60 → ×2.51. |
| Waiting 15–60 s for the first moves | no improvement. |
| LLM reading the request | discrimination up (correlation 0.1–0.4 → 0.3–0.5), recall of leave-worthy tasks 4 % → 17 %, but breaches up to 28 %. |
| Agent-declared scope (inject "state your plan first") | not tried live: the signal would depend on the model, and often the model does not know either. |
| Push notification when done | not a solution to the stated goal (knowing at the start), and every major agent already has it. |

And one bug worth knowing about: our ledger feedback made the leave window more cautious whenever the
*upper* confidence bound of the real breach rate exceeded the target. With a few dozen windows that is
almost always true, so it ratcheted caution to the maximum on an honest 15 % record — and the few
promises left were the over-confident ones (101 promises at 15 % → 27 at 26 %). Stepping up only when the
*lower* bound exceeds the target fixed it.

## 6. Where the free time actually is

The free time exists — 78 % of the author's agent working time was in gaps of 3+ minutes — it just
cannot be *predicted* one task at a time. It can be *aggregated*: the total of several tasks run back to
back is far more predictable than any one of them (author's history, resampled):

| Tasks queued | Median total | 80 % guaranteed at least |
|---|---|---|
| 1 | 3.5 min | 0.6 min |
| 2 | 7.7 min | 3.7 min |
| 3 | 12.5 min | 7.0 min |
| 5 | 22.6 min | 14.1 min |
| 8 | 39.8 min | 25.7 min |

But keeping an agent from handing back control is exactly what autonomy modes such as Codex's `/goal`
and "Ralph loops" do, and their trade-offs are well documented: runaway quota use (one 12-hour goal
used 70 % of a weekly limit), drift and scope creep, runs that never declare themselves finished (a
65-hour run, 36 % of plans restarting from zero), and agents glossing over problems to keep going.
The frequent hand-backs are also where the human steers. We concluded this is the same trade-off under
another name, and did not build it.

## 7. Pros and cons of the plugin as it stands

**Good at**

- Accurate turn accounting from hooks (see §8), including permission waits and background agents.
- Seeing where the agent's time goes: turn lengths by task type and repo, the slowest commands.
- An honest duration range, calibrated to you, that improves a lot if your pace is unusual.
- Reliable leave windows *while a known-slow operation runs* (a test suite, a build) — then the agent
  genuinely cannot need you.
- Checking its own claims: `eta.py eval` and the window ledger.
- Private: local SQLite, no prompt text stored, no network, no model calls.

**Not good at**

- Freeing meaningful blocks of time: leave windows are 1–2 minutes by nature (§4.2).
- Reliability of those windows varies by user (8–45 % breached in replay).
- The start-of-task estimate stays rough (×2.3–2.6).

## 8. Claude Code hook pitfalls we hit

Useful for anyone building on Claude Code hooks (analytics, cost tracking, notifications):

- **Post-turn helper agents re-open finished turns.** After `Stop`, Claude Code's own helpers (e.g. the
  "while you were away" summary) emit `SubagentStop` / task events carrying the finished turn's
  `prompt_id`. Treating "same prompt_id after Stop" as a continuation left 15 of 27 turns open until the
  session ended; 119 of 186 leave windows were issued for turns already over. Only main-thread tool
  events (no `agent_id`) indicate a real continuation.
- **`!` shell commands fire no `UserPromptSubmit`.** The command and its output are written to the
  transcript only after it finishes, under a new `prompt_id`, and the model then answers. Nothing is
  observable while it runs.
- **Esc fires no `Stop`**; the turn only closes at the next prompt or session end.
- **Background tasks wake the agent with a `<task-notification>` prompt**: same piece of work, new
  prompt_id.
- **Subagent tool calls carry `agent_id`**; transcripts keep them in separate files.
- **Transcripts lack permission prompts**, so imported history under-counts "needs you".

## 9. Methodology lessons

- Replay the *whole* system through time before believing a reliability number. In-sample tuning said
  17 %; the time-travel replay said 24.5 %.
- Check a claim on more users before writing it down. We published "16 %" from six histories and had to
  retract it when five more gave 29 %.
- Score what users see, not what the math guarantees (ACI's per-moment coverage vs. displayed promises).
- Beware feedback loops keyed on uncertain bounds (the ratchet in §5).
- A flat hazard is the first thing to measure: it tells you immediately how long a model-free promise
  can be.
- Negative results are results. Most of the value of this project is in §5.
- A tooling trap: `pgrep -f` / `pkill -f` also match the shell that runs them when the pattern is part of
  the same command line; one of our "long-running" experiments had finished three hours earlier.

## 10. Open questions

- Can an agent expose a *robust* scope signal (a plan, a step count) early, without changing how it
  works, and does it transfer across models? The one user whose agent planned showed r = 0.77.
- Is "the cost of leaving" a better interface than a promise? E.g. for the author: leave 5 minutes and
  the agent is already waiting when you return 67 % of the time (for 3 minutes on average), still
  running 33 % of the time. These probabilities can be calibrated from your own history.
- Parallel sessions: with several agents, the question becomes *which one needs me next* — a ranking
  problem, far more forgiving than a guarantee.
- Larger consented datasets with timing *and* attention events (permission prompts, questions), across
  agents (Codex, Cursor, …), would settle how general the flat hazard is.

## 11. Reproduce

Install the plugin from this repository (details in [docs/plugin.md](docs/plugin.md)):

```bash
claude plugin marketplace add JohnMing143/agent-ETA
claude plugin install agent-eta@agent-eta
# then in Claude Code: /agent-eta:setup   (imports your past transcripts, tunes, installs the status line)
python3 ~/.claude/plugins/…/agent-eta/scripts/eta.py eval --curve   # is learning paying off for you?
```

The experiments in [research/](research/) run on any set of histories imported with the plugin; the
public datasets above can be fetched and imported with `research/cross_user.sh`.

## 12. Data sources and licenses

- trace-commons/agent-traces — CC-BY-4.0.
- AlinCiocan/fable-5-claude-code-traces — CC-BY-4.0.
- armand0e/claude-fable-5-claude-code, crispwisp/wisp-claude-code-sessions,
  choucsan/mimo-claude-code-traces-1k — MIT.
- armand0e/kimi-k2.6-claude-code-traces, armand0e/minimax-m3-claude-code-traces — no license stated;
  we report aggregate statistics only and redistribute nothing.

Only aggregate statistics derived from these datasets appear in this repository. Codex `/goal` sources:
[OpenAI cookbook](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex),
[a hands-on review](https://www.jdhodges.com/blog/codex-goal-feature-review/),
[a 65-hour run](https://note.com/hataraiku/n/n89ed4070a929?hl=en),
[openai/codex #28688](https://github.com/openai/codex/issues/28688),
[#47293](https://github.com/openai/codex/issues/47293), [#45864](https://github.com/openai/codex/issues/45864).

The plugin and the research code are MIT licensed (see [LICENSE](LICENSE)).
