---
name: eta
description: Agent ETA report - how long your Claude Code turns usually take, how accurate the time-to-completion and "safe to leave" predictions have been, which commands eat the most time, and what other running Claude Code sessions are doing and when they will finish or need you. Use when the user asks about agent ETAs, how long runs take, whether they can step away, or the progress of other sessions.
argument-hint: "[days] | watch"
allowed-tools: Bash(python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py *)
---

# Agent ETA

## Other Claude Code sessions running right now
!`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py status --exclude-session ${CLAUDE_SESSION_ID} --no-color`

## History and prediction accuracy (last 30 days)
!`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py report --days 30`

## How to answer

Arguments: `$ARGUMENTS`

- If the arguments contain `watch`: do not run it yourself (it is an endless live view). Tell the
  user to run it in another terminal or tmux pane:
  `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py watch`
- If the arguments are a number of days other than 30, rerun the report with `--days N`.
- If the user asks whether the learning works, is improving, or is worth continuing, run
  `python3 ${CLAUDE_PLUGIN_ROOT}/scripts/eta.py eval --curve` and answer from it: the verdict line
  (a gain counts as shown only when its 95 % interval excludes 0), the gain split by history size,
  and the learning curve's "extra gain from the second half of the history" (the marginal value of
  more data). Never call a gain real when its interval includes 0.

Answer in the user's language, briefly:

1. Other sessions: for each one, what it is doing, when it will likely finish, and how long the
   user can stay away. Lead with any session marked as needing the user right now.
2. History: typical turn length, median time until Claude needed the user, and the slowest
   commands (if one dominates, e.g. a slow test suite, say so; it inflates every ETA).
3. Accuracy, in plain words. Targets: P50 hit rate about 50 %, P80 coverage about 80 %,
   "safe" misses at most 20 %. Higher hit/coverage means the estimates run long (too cautious);
   lower means they run short. Compare the backtest's current-estimator numbers with the built-in
   prior and B0 baselines (B0 = run-length history without trajectory features; the estimator
   should beat it by 10 %+ in pinball loss). The shrinkage K0 line says how the estimator currently
   learns: a large K0 or ∞ means it relies on the prior calibrated to the user's pace (normal for a
   small or varied history), a small K0 means similar past runs dominate. It is re-chosen
   automatically from the user's own history. With fewer than about 20 finished runs, say it is
   still learning rather than drawing conclusions.
4. The leave-window ledger is the real-world check of "safe to leave": windows actually shown,
   with fixed expiry. Report held / breached / unknown and the upper bound; it is only "validated"
   (✓ in the status line) after 100+ windows with the upper bound at or under the target.

Do not invent numbers that are not in the output above. If the output says there is no data,
suggest `/agent-eta:setup`.
