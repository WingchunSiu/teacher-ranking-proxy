# OT-Agent proxy-transfer audit (2026-09-10)

## Scope

This retrospective case study asks whether trajectory proxies that were proposed
or tested in reasoning-style settings transfer to public multi-turn agent traces.
It uses four teacher arms from the Qwen3-8B SFT ablation reported in OpenThoughts-
Agent Table 6:

- `DCAgent/b1_top4_seq` (GLM-4.7), revision
  `01924f6f86b8e836e06754caadf99b88aa4cbcb4`
- `DCAgent/c1_kimi_k2.5_fixed` (Kimi K2.5), revision
  `5807137b49d0d1d27e7b100da3e8d4156ddb94e3`
- `DCAgent/c1_top4_seq_glm46_traces` (GLM-4.6), revision
  `6e70e5cd05b2eb737cf2fe7c0b9cc2aab8f35e31`
- `DCAgent/c1_gpt53_codex_fixed` (GPT-5.3-Codex), revision
  `38a6f93a475416e79a04e373ed2b1ff2d1d7c45a`

GLM-5 is omitted because no matching public Top-4 trajectory dataset was found.
The downstream ground truth is the paper's SFT-student result, not the teachers'
own rollout pass rate: GLM-4.7 `+0.73`, Kimi K2.5 `+0.66`, GLM-4.6 `+0.08`, and
GPT-5.3-Codex `-1.47` after the paper's per-benchmark normalization and averaging.
The public trace `result` fields are too incomplete and source-asymmetric to form
a four-teacher rollout-reward ranking.

## Pairing and scoring controls

The four full datasets have similar Top-4 source proportions but do not contain
the same realized task sample. We therefore retained only the 445 normalized
instructions shared by all teachers. Every retained group also agrees on
normalized task ID, source, and initial working directory. The realized source
mix is Tezos 178, IssueTasks 132, SuperUser 108, and SWE-Smith 27. One
deterministic trajectory per instruction and teacher receives equal weight.

Instruction normalization removes the Terminus-2 wrapper and collapses
whitespace before hashing. Task ID alone is not safe because some `b1`/`c1` rows
reuse an ID for different instruction text. Public rows do not expose the exact
environment image digest, repository commit, task-config revision, or verifier
version, so identical sandbox state remains unproven.

RSR was scored with `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, a 32,768-token limit, rank clipping
at 100, the Qwen chat template, and the official ratio-of-trajectory-means
aggregation. Lower RSR is better.

## Results

| teacher | mean RSR | paired-bootstrap 95% interval | downstream GT rank |
|---|---:|---:|---:|
| GLM-4.7 | 2.4245 | [2.3931, 2.4566] | 1 |
| GLM-4.6 | 2.4545 | [2.4237, 2.4857] | 3 |
| Kimi K2.5 | 2.5039 | [2.4767, 2.5320] | 2 |
| GPT-5.3-Codex | 2.9520 | [2.9007, 3.0022] | 4 |

RSR recovers the top, bottom, and 5/6 pairwise relations (Spearman `0.80`,
Kendall `0.67`) but reverses Kimi and GLM-4.6. The displayed RSR order occurs in
94.5% of 20,000 paired task bootstraps and GLM-4.7 is top in 94.9%.

That apparent middle-order confidence is conditional on the realized source mix.
Equal weighting of the four sources restores the GT point order, but Kimi beats
GLM-4.6 in only 61.9% of stratified bootstraps. Excluding every task with a
timeout in any teacher moves this probability to 51.4%. Thus the robust result is
top/bottom separation plus source sensitivity, not a stable full ranking.

The trajectory-only alternatives do not provide a robust replacement. Corrected
aligned TOR ranks Kimi > GLM-4.7 > GPT > GLM-4.6 (Kendall `0.33`). Corrected
inspect-act-verify/EGS-loop reaches Kendall `0.67` only on 228 complete tasks and
has split top-1 support. Assistant-token length, command count, NLL components,
and Error-Retry do not robustly recover the downstream order. Proxy definitions
and missing-trajectory handling materially change several conclusions.

## GPT no-op confound

The full GPT dataset contains 328,666 assistant turns. Of these, 301,284 have no
executable command and 286,555 both have no command and do not request task
completion. These are not empty strings: they usually contain analysis/plan text
inside a valid Terminus-2 JSON response.

The turn-weighted 91.7% command-empty headline must not be interpreted as 91.7%
of trajectories being broken. The behavior is concentrated:

- 1,295/9,868 rows contain any non-completing no-op;
- 908 rows contain at least 50;
- 890 rows (9.0%) contain at least 100;
- 846 of those 890 rows are IssueTasks.

A heuristic classification of the first no-op assigns 671 affected rows,
contributing 209,872 no-op turns, to missing repository/files. Another 395 rows
first use an empty command batch as a wait/poll action. Direct inspection shows
two common model/harness interactions:

1. GPT searches an empty workspace, declares the required repository unavailable,
   then repeatedly waits rather than constructing or installing a working tree.
2. GPT launches a long command such as `pytest`, receives partial output, and
   returns `commands: []` intending to poll.

Terminus-2 has no explicit wait or blocked action. An empty command list causes it
to read incremental terminal output immediately and request another model turn;
the public default episode cap is effectively unbounded. The scaffold therefore
amplifies a single conservative/unsupported action choice into hundreds of model
turns. Completion confirmation is not the main cause: only 192/286,555
non-completing no-op turns immediately follow that prompt.

The pattern is teacher-specific on the paired slice. Across the 132 matched
IssueTasks, GLM-4.7, Kimi, and GLM-4.6 produce 20, 17, and 0 non-completing
no-op turns, while GPT produces 26,522 across 85 rows; 83 GPT rows reach at least
100. This controls instruction and initial cwd, but not the unpublished image
digest.

The anomaly also exposes an RSR failure mode. GPT IssueTasks timeouts receive a
better mean RSR (`1.9917`) than non-timeouts (`2.4940`) despite averaging far more
assistant tokens. Within GPT IssueTasks, no-op fraction and RSR have Spearman
`-0.826`: more repetitive waiting looks better to the lower-is-better proxy. The
ratio can improve when repetitive low-state-change text shifts clipped rank and
NLL together, even though the trajectory is not useful agent behavior.

## Interpretation

This is not evidence that serialization corrupted most GPT trajectories. The raw
dataset contains valid model responses, and approximately 91% of rows do not have
a 100-turn no-op loop. Nor is the downstream result a clean measurement of native
GPT-5.3-Codex capability. The most defensible causal description is a concentrated
model-policy x Terminus-2 scaffold x task/environment interaction that the harness
amplifies and the public SFT preprocessing appears to retain.

Consequences for proxy evaluation:

- call the arm "GPT-5.3-Codex via the Terminus-2 teacher recipe," not native model
  quality;
- report task-paired, source-stratified results and artifact filters;
- report native proxy direction separately from higher-is-better ranking utility;
- test no-op/timeout sensitivity because likelihood ratios can reward degenerate
  loops;
- keep published proxy semantics separate from agent-specific adaptations;
- do not infer causal contamination of the downstream SFT checkpoint without a
  filtered-and-retrained counterfactual.

No full-dataset proxy rescore or SFT retraining was performed for this audit.
