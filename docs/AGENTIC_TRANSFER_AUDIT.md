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

## Exact 445-task results

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

## Source-balanced 1K per teacher

To reduce the small and highly uneven per-source sample without claiming that
nonmatching tasks are paired, a second deterministic sample contains 1,000
unique instructions per teacher:

- 251 SWE-Smith, 251 SuperUser, 247 Tezos, and 251 IssueTasks in every arm;
- all 445 four-teacher exact overlaps retained;
- pairwise selected overlap ranges from 655 to 920 tasks;
- shared hashes are prioritized before teacher-only fill;
- response length, proxy score, result, and timeout status are not used in
  selection.

Tezos is set to 247 because Kimi and GPT each expose only 247 unique Tezos
instructions; the other sources receive one additional task so every arm remains
exactly 1,000 without duplicate sampling. This is a source-matched distribution
comparison with partial overlap, not a fully paired experiment. The reported
bootstrap resamples independently within each teacher/source cell and therefore
does not pretend that all 1,000 rows are paired.

| teacher | source-balanced RSR | 95% bootstrap interval | downstream GT rank |
|---|---:|---:|---:|
| GLM-4.7 | 2.3974 | [2.3771, 2.4179] | 1 |
| Kimi K2.5 | 2.4653 | [2.4486, 2.4825] | 2 |
| GLM-4.6 | 2.4777 | [2.4580, 2.4976] | 3 |
| GPT-5.3-Codex | 2.9404 | [2.9133, 2.9685] | 4 |

The larger source-balanced sample recovers the complete downstream GT order
(Spearman `1.0`, Kendall `1.0`). Five pairwise relations hold in 100% of 20,000
bootstrap draws. The only uncertain relation is Kimi > GLM-4.6 at 83.0%, which
also determines the 83.0% full-order probability. This resolves the 445-slice
point inversion but is not strong evidence that the middle pair is far apart.

Source-specific RSR remains nonstationary:

| source | RSR order (lower is better) | Kendall vs GT |
|---|---|---:|
| SWE-Smith | GLM-4.7 > Kimi > GLM-4.6 > GPT | 1.00 |
| Tezos | GLM-4.7 > GLM-4.6 > Kimi > GPT | 0.67 |
| SuperUser | GLM-4.6 > GLM-4.7 > Kimi > GPT | 0.33 |
| IssueTasks | GPT > Kimi > GLM-4.6 > GLM-4.7 | -0.67 |

Excluding timeouts independently within each teacher or excluding trajectories
with more than 50% command-empty turns retains the complete aggregate GT point
order. The latter moves GPT from `2.9404` to a worse `2.9902`, consistent with
no-op loops making its RSR artificially better rather than causing its last-place
aggregate rank.

## TOR and other trajectory-only controls

TOR is CPU-only and does not use the student model. For every declared action,
it asks whether an earlier observation command inspected the same or a related
path:

`TOR = aligned inspect-before-act actions / all actions`.

The audit did not use the legacy raw-screen count directly. Some Terminus-2
screens contain cumulative prompt history, which can count an old command again.
The corrected parser matches each declared command to its latest ordered prompt
echo and obtains 86.6--96.4% coverage across teachers. It also classifies
redirections such as `cat > file` and `echo > file` as actions rather than
observations. This changes 757/1,317/1,289/179 events for GLM-4.7/Kimi/GLM-4.6/
GPT respectively. The paper's TOR code is not public, so path alignment remains
a documented local operationalization rather than proven upstream equivalence.

On the exact 445-task slice, corrected aligned TOR ranks Kimi > GLM-4.7 > GPT >
GLM-4.6 (Kendall `0.33`), with Kimi top-1 in 87.4% of paired bootstraps. A
separate three-assistant-turn observation-window variant ranks GLM-4.7 > GPT >
Kimi > GLM-4.6, but splits top-1 support almost evenly between GLM-4.7 (49.7%)
and GPT (47.8%). Thus TOR's conclusion is sensitive to a definition that is not
present in the main metric.

The source-balanced 1K sample gives the same aligned TOR point order and Kendall
`0.33`; Kimi is top in 99.98% of independent source-stratified bootstraps. Setting
no-action trajectories to zero leaves the order unchanged and makes Kimi top in
100%. More data therefore stabilizes TOR's disagreement with GT rather than
repairing it. Screen-match coverage is 84.6--97.2% by teacher, so the absolute
scores retain command-reconstruction uncertainty.

Corrected inspect-act-verify/EGS-loop reaches Kendall `0.67` only on 228 tasks
with defined scores; top-1 support is split GLM-4.7/Kimi/GPT = 47.2%/28.5%/24.3%.
Assigning zero rather than dropping no-action trajectories changes the top
teacher to Kimi with 98.0% support. Assistant-token length, declared-command
count, and Error-Retry each have Kendall `0.0`; assistant-turn count is exactly
reversed because GPT's no-op loops dominate it. None is a robust replacement for
the downstream ranking. Proxy definitions and missing-score handling are material,
not implementation details.

## GPT no-op confound

### What an "empty command" means

It does not mean that the assistant message is empty. A simplified real response
looks like:

```json
{
  "analysis": "The repository is still unavailable; waiting for it to be mounted.",
  "commands": []
}
```

The model still generated ordinary text tokens, but requested no shell action.
If `task_complete` is also absent or false, the environment does not change and
the episode does not end. Terminus-2 asks the model again with essentially the
same state, which can produce another nearly identical response.

For example, paired IssueTasks task `issue-0538` asks every teacher to modify
Django's `Model.__init__` and begins with an empty `/app`:

1. GPT runs `ls`, `grep`, `sed`, `pwd`, and `find` to search for the code.
2. It concludes that the Django repository was not mounted and emits
   `commands: []` without completing the task.
3. It repeats variants of "no update; waiting for the codebase" through assistant
   turn 352, still without commands.
4. The trajectory ends in `AgentTimeoutError`, with no patch produced.

GLM-4.7 and Kimi instead install/copy Django into `/app` and continue. This does
not prove that GPT's initial diagnosis was irrational: the exact environment
image is private and refusing to invent a missing mount is conservative. It does
show why this is a poor teacher trajectory for the benchmark: it contains hundreds
of targets that demonstrate waiting rather than task progress, produces no useful
solution, and times out. Plausible causes are a task/environment mismatch, a model
policy that treats missing resources conservatively, the absence of explicit
`wait`/`blocked` actions in Terminus-2, and the harness's lack of a consecutive-
no-op stop condition. The evidence does not isolate one of these as the sole cause.

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

### Does RSR use the commands?

RSR does not parse or count commands. It performs a Qwen3-8B forward pass over the
rendered conversation and scores the tokens inside teacher assistant spans. Task
text and terminal observations are conditioning context; assistant JSON text is
the prediction target. Therefore the words in `analysis` and `plan`, punctuation,
field names, and the literal `"commands": []` are all ordinary scored tokens.
RSR has no semantic signal that no environment action occurred.

The anomaly exposes a ratio failure mode. GPT IssueTasks timeouts receive a better
mean RSR (`1.9917`) than non-timeouts (`2.4940`) despite averaging 10,652 versus
4,337 assistant tokens. In these traces, slightly lower clipped token rank and
higher NLL combine into a lower rank/NLL ratio. Within GPT IssueTasks, no-op
fraction and RSR have Spearman `-0.826`: empirically, more repetitive waiting
looks better to this lower-is-better proxy even though RSR never reads the command
list as a structured action.

### Which tokens reached SFT?

The exact token-level training exposure is not fully known. The checkpoint card
points to the audited dataset revision's `_thinking_preprocessed` variant. Its
public preprocessing code retains all conversation messages and only reformats
thinking tags; the raw and preprocessed copies have identical turn counts. The
matching public LlamaFactory configuration uses a 32,768-token cutoff and trains
assistant targets in prefix order, so it would retain no-op text that falls before
the cutoff. Very long loop tails can be truncated, meaning 286,555 observed no-op
turns must not be equated with 286,555 fully trained turns.

The exact run configuration linked by the model card is private. A hidden filter,
different masking setting, or other override therefore cannot be excluded. The
public provenance strongly suggests that at least early no-op responses entered
the SFT targets, but it does not establish exactly how many no-op tokens received
loss. A tokenizer-level replay of the public configuration could quantify the
public-recipe exposure; only the private run config could close the remaining
provenance gap.

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

No full-10K proxy rescore or SFT retraining was performed for this audit. The
scale-up scored 4 x 1,000 trajectories with the pinned Qwen3-8B student and ran
the trajectory-only controls on the same selected rows.
