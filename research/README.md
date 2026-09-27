# Research scripts

The experiments behind the [write-up](../README.md). They analyse histories imported with the plugin
(`eta.py backfill`) — your own, or public datasets — and print aggregate numbers. No data is included.

## Setup

1. Import each history into its own directory (one per user or dataset), e.g.

   ```bash
   AGENT_ETA_HOME=/work/users/public_a/home python3 ../plugins/agent-eta/scripts/eta.py backfill --dir /work/users/public_a/projects
   AGENT_ETA_HOME=/work/users/public_a/home python3 ../plugins/agent-eta/scripts/eta.py tune
   ```

   `cross_user.sh <workdir>` does this for [trace-commons/agent-traces](https://huggingface.co/datasets/trace-commons/agent-traces):
   downloads the Claude Code sessions, splits them into one pseudo-user per contributor
   (`split_contributors.py`, by recorded project path), imports, tunes, evaluates, and runs the
   time-travel replay. Never point `AGENT_ETA_HOME` at your real `~/.claude/agent-eta` for this.

2. Copy `histories.example.json` to `histories.json` and list the homes (and, for the start-of-task
   experiments, the transcript globs). Scripts that take homes as arguments do not need it.

Scripts listed without `HOME` read the history from `AGENT_ETA_HOME`; the rest take history directories
as arguments. `learning_eval.py` needs at least 30 finished runs and `recency_eval.py` at least 25.
Everything these scripts write (batches with prompt text, judgments, databases) is ignored by
`.gitignore` — keep it that way.

The scripts find the plugin relative to this directory; set `PLUGIN=<scripts dir>` to test another
version (e.g. with `bimodal-prior.patch` applied). Output labels are partly in Chinese.

## Experiments

| Script | Question | Finding in the write-up |
|---|---|---|
| `learning_eval.py HOME [OUT.json]` (≥ 30 finished runs) | Does learning pay off? prequential / fixed exam / exchangeable learning curves | §4.4 |
| `recal_eval.py`, `recency_eval.py` | PIT quantile recalibration; recency half-life | §4.4 |
| `signals_eval.py` | Remaining time after "what just happened" (tests passed, git push, …) | a built-in "tests passed → nearly done" rule was wrong on the author's data |
| `aci_eval.py`, `aci_multi.py HOME…`, `leave_multi.py HOME…` | Adaptive conformal inference and selective risk control for leave windows | §5 |
| `timetravel3.py HOME…` | The live system replayed through time, promises fed into the ledger | §4.3 |
| `timetravel4.py OUT HOME…`, `compare_tt.py OLD NEW` | Same, recording every point, to compare two plugin versions point by point | §4.4, §5 (two-mode prior) |
| `breach_diag.py HOME…` | Why promises are breached (finished early vs. asked you) | §4.3 |
| `leave_fix_eval.py HOME…` | Candidate fixes (always recalibrate; gate on replay failure) | §5 |
| `model_stats.py`, `model_cal_eval.py HOME…` | Per-model speed and attention; per-model pace calibration | §4.5 |
| `start_of_task/start_eval.py` | Start-of-task prediction: plugin vs. similar past prompts | §4.1 |
| `start_of_task/early_eval.py HOME…` | Prediction at 0 / 15 / 30 / 60 s into a task | §4.1 |
| `start_of_task/build_batches.py`, `judge.py`, `llm_eval.py` | An LLM (Claude Haiku via `claude -p`, isolated, no tools) reading each request; calibrated per user | §4.1 |
| `start_of_task/tradeoff.py HOME…` | Window length vs. confidence | §4.2 |
| `start_of_task/two_stage.py` | "Wait τ; if still running, leave W" | §4.2 |
| `bimodal-prior.patch` | The two-mode (quick answer + work) prior; `git apply` at the repository root | §4.4, §5 |

`judge.py` sends each request (truncated) and the tail of the previous assistant reply to a model
through your Claude Code login. Run it only on data you are allowed to send, and note that it counts
against your plan.

## Notes

- Everything is prequential: a prediction uses only runs that had finished before it.
- Leave windows are scored as displayed (≥ 1 min, rounded down to the status line's buckets).
- Intervals are cluster bootstraps over runs.
- Imported transcripts do not contain permission prompts; live-recorded histories do.
