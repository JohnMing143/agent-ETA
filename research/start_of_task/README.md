# Start-of-task prediction (2026-09-26)

Question: at the moment a task starts, how long will it take and how long can you safely leave?
All prequential, conformal leave windows (q20 of earlier residuals), 8 histories (~900 finished tasks).

| information at the start | median error | ≥3-min window given | breach | recall of leave-worthy tasks |
|---|---|---|---|---|
| prompt length + keyword category + your pace (plugin today) | ×2.60 | 2 % | 23 % | 4 % |
| + similar past prompts (TF-IDF, local) | ×2.51 | 2 % | 20 % | – |
| waiting 15/30/60 s for the agent's first moves | ×2.4–2.7 | 2–9 % | 19–25 % | – |
| Claude Haiku reads request + previous reply (minutes) | ×2.59 | 10 % | 28 % | 17 % |
| Haiku size bucket, calibrated to you | ×2.46 | 6 % | 20 % | – |
| blend of plugin + Haiku | ×2.29 | 3 % | 19 % | 6 % |
| agent's own plan (TaskCreate/TodoWrite), when present | ×1.62 (contributor A, 21 tasks) / ×2.62 (kimi) | – | – | – |

Correlation of prediction with actual log duration: plugin 0.1–0.4, Haiku 0.3–0.5 (kimi ~0), agent plan 0.77 (contributor A).
40 % of tasks actually leave ≥3 min before the user is needed; median time-to-need from any moment is 1.5 min.

Scripts: prompts.py (runs ↔ prompt text), start_eval.py, early_eval.py, tradeoff.py, build_batches.py + judge.py
(Haiku via `claude -p`, isolated AGENT_ETA_HOME, no tools, no saved session), llm_eval.py.
Histories come from ../histories.json.

## Why model-free windows stay short (2026-09-27)

two_stage.py: "wait τ; if still running, leave W" gives W≈1–1.5 min for τ = 60–180 s.
Hazard of being needed in the next minute, 918 tasks: 39 % in minute 0, then 19 %, 18 %, 13 %, 14 %, 12 %, 11 % at
1/2/3/5/8/12 min elapsed — nearly flat (close to memoryless). An 80 % window is −ln 0.8 / hazard ≈ 1–2 min whatever
the elapsed time. Longer windows need task-specific information (scope, or a known-slow operation in flight).
