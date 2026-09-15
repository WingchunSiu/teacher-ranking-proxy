# Proxy implementation audit (2026-09-08)

Scope: formula, score direction, masks, aggregation, and released upstream configuration. This audit does not address SFT FLOP control and did not launch Slurm jobs, model inference, judging, or training.

## Corrections

| proxy / layer | finding | correction | result status |
|---|---|---|---|
| evaluator | Ordinary scores were always ranked descending; direction metadata was ignored. GRACE alone had a hard-coded sign flip. | `PerTask` now carries an audited native direction. One normalization path is used by point rankings, task winners, bootstrap, ablations, and paired comparisons. Reports retain both native raw scores and a higher-is-better utility. Unknown directions fail loudly. | Covered by CPU unit tests. Existing higher-is-better proxies are unchanged. |
| RSR | The official metric is lower-is-better, but rows declared `higher_better`, reversing every downstream comparison. A second audit found that the pinned upstream assistant-header scan misses Qwen response spans when response-leading whitespace merges with the template newline. This affected 517/724 Claude turns in the Terminal-Lego n=200 sample. | Rows declare `lower_better`. The scorer now finds the stable role prefix and drops one boundary-whitespace token; this intentionally differs from upstream only on the broken boundary case. | Boundary-safe n=200 order: `DeepSeek > GLM > Claude > Qwen3.5`; τ-b `+0.55`, pairwise `0.80`, correct top-1 with 99.8% bootstrap support. See `RSR_BOUNDARY_AUDIT.md`. |
| shared assistant mask | Global/local NLL, ASLEC, GRACE, and agent-all-assistant SCAS used the same brittle Qwen header scan, so their old Claude rows are also incomplete. | All all-assistant views now use the boundary-safe helper. | Historical Terminal-Lego values for these views are stale and require forward rescoring; only RSR has been regenerated so far. |
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

The standalone repository does not contain the original Capella score directory,
but the exact task manifest and public Terminal-Lego release were sufficient to
reconstruct and rescore the n=200 RSR sample.  The boundary audit summary is in
`artifacts/terminal_lego_rsr_boundary_audit_n200.json`; per-trajectory score
tables remain on Jupiter scratch.  ASLEC skip-2 and revised SCAS still require
new forward scores because the necessary token-level components were not stored
in the public artifact.
