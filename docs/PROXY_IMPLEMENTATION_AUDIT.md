# Proxy implementation audit (2026-09-08)

Scope: formula, score direction, masks, aggregation, and released upstream configuration. This audit does not address SFT FLOP control and did not launch Slurm jobs, model inference, judging, or training.

## Corrections

| proxy / layer | finding | correction | result status |
|---|---|---|---|
| evaluator | Ordinary scores were always ranked descending; direction metadata was ignored. GRACE alone had a hard-coded sign flip. | `PerTask` now carries an audited native direction. One normalization path is used by point rankings, task winners, bootstrap, ablations, and paired comparisons. Reports retain both native raw scores and a higher-is-better utility. Unknown directions fail loudly. | Covered by CPU unit tests. Existing higher-is-better proxies are unchanged. |
| RSR | The official metric is lower-is-better, but rows declared `higher_better`, reversing every downstream RSR comparison. Token ranks, NLL, and ratio-of-means aggregation were otherwise numerically equivalent to pinned upstream commit `59a7c4c`. | Rows now declare `lower_better`; the evaluator also migrates historical RSR files with stale metadata. | Corrected n=200 point order: `Claude > DeepSeek > GLM > Qwen3.5`; τ-b `-0.18`, pairwise `0.40`, wrong top-1. Exact bootstrap and min-RSR task-winner shares require the original score JSONL. |
| ASLEC | The implementation used `skip_tokens=1`. The released experiment driver defaults to 2 and released selection paths use `*_skip2`. Both formulas themselves match upstream for skip 1 and 2. | Primary configuration changed to `skip_tokens=2`; skip 1 remains available as a sensitivity setting in `_aslec_components`. | Published n=200 rows are legacy skip-1 results and must not be presented as the primary released configuration. A student forward pass is required for skip-2 scores. |
| SCAS | The old mask used all assistant-content spans. It differs from the released final-answer mask (observed maximum relative score difference 10.4% in the prior n=8 check), so calling it official was incorrect. | One forward now emits two explicit views: `official_final` and `agent_all_assistant`. The equivalence check compares upstream only to `official_final`. | Published n=200 SCAS row is the legacy agent adaptation. Both revised views require a student forward pass. |
| cmd_error | The code already reconstructed one segment per executed command, but its docstring and metadata still said turn-level and the count field was named `failed_turns`. | Metadata now says executed-command granularity and rows add `failed_commands`; the old field remains as a compatibility alias. | No numeric change. |

The scorer now rejects legacy ASLEC/SCAS caches whose configuration does not match the corrected implementation. It never deletes them automatically; replacement requires the caller to preserve/rename the old file or explicitly pass `--force`.

## Checked without a formula correction

- Global NLL: higher mean assistant-token log-probability is better; task, system, and observation tokens are conditioning only.
- Local NLL: assistant-turn means are averaged equally, with the previous k action-observation pairs as the declared agentic adaptation.
- GRACE: the official teacher-level raw score is lower-is-better. Direction handling is now generic; projected gradients and `grace()` aggregation are unchanged.
- TOR/EGS, trajectory length, teacher benchmark, error-retry, and SCRF: declared higher-is-better views are internally consistent. Their agentic operationalizations and judge dependence remain methodological limitations, not sign/aggregation bugs found by this audit.
- Ground-truth ties, Kendall tau-b, pairwise accuracy excluding tied GT pairs, and task-level resampling were checked and retained.

## Data boundary

The standalone repository does not contain `runs/terminal_lego-n200-s42/proxy_scores` or `ranking_report.json`; those files were on the Capella workspace. Therefore only the RSR point result can be corrected exactly from the published ordering. Copying/mounting the existing score directory is sufficient to regenerate its bootstrap and diagnostics on CPU—no model forward pass is needed. ASLEC skip-2 and revised SCAS genuinely require new forward scores because the necessary token-level components were not stored in the public artifact.
