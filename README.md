# Teacher-ranking proxy

**Goal:** compare existing proxies from the instruction / reasoning-distillation
literature that aim to predict which teacher produces the best SFT data for a
given student, from trajectories only, without running SFT.

**Relation to OpenThoughts-Agent.** This lives as `data/teacher_ranking_proxy/`
inside the `OpenThoughts-Agent` repo. Trajectory generation (Stage 2,
`generate_trajectories.py`, the `apptainer_patch/` shim, `bridge_deploy/`,
`harbor_patches.py`) imports Harbor and the repo's HPC stack, so it only runs
embedded there. The scoring and evaluation (Stages 3-4:
`compute_proxies.py`, `evaluate_ranking.py`, `judge.py`) are self-contained
(transformers/vLLM plus the pinned author repos under `upstream/`) and run on
existing trajectory files without the parent repo. The public standalone repo
is a snapshot/mirror of this folder.

## Implemented proxies

- TOR / EGS: https://arxiv.org/pdf/2606.03461
- GRAPE: https://arxiv.org/pdf/2502.04194
- LALP (local naturalness): https://arxiv.org/pdf/2510.03988
- ASLEC: https://arxiv.org/pdf/2604.06834
- RSR: https://aclanthology.org/2026.acl-long.1950/
- GRACE: https://arxiv.org/abs/2511.02833
- SCAS: https://arxiv.org/abs/2605.26872
- traj_length, teacher_bench, cmd_error (error rate), error_retry: simple controls / the Terminal-Bench 2.0 command-error taxonomy (https://arxiv.org/abs/2601.11868)

Listed in the spec but not implemented: CAR (https://arxiv.org/pdf/2411.07133,
needs an average reward we do not have with one trajectory per teacher/task),
LARK (https://arxiv.org/pdf/2605.30651), PerSyn (needs router
training).

For what each proxy computes and how, see `docs/PROXY_SPEC.md`.

## Proposed proxy for agentic tasks (SCRF)

Hypothesis: a useful teacher is one whose trajectories demonstrate
recovery behavior in the regions where the target student empirically
struggles.

Pipeline (same for student and teacher trajectories):

1. Extract (command, output) pairs from the trajectory.
2. Drop turns Harbor could not execute because the model's response was not valid Terminus-2 JSON (a parse error, no command ran); those are counted separately, not as command errors.
3. Classify each (command, output) with the Terminal-Bench 2.0 prompt and error taxonomy (11 categories, 91 subcategories) into one error category or "no error".
4. For the pairs classified as errors, a second LLM-as-judge call decides whether the model recovered, given the failed command and its output plus the next 3 commands and outputs (K=3).

This gives, per agent, a distribution over error categories and, within each, a
recovered / not-recovered split. A teacher is scored by how well its
recovered-error demonstrations cover the categories the student fails and does
not recover from.

Notation, per error category e: q_S(e) = fraction of the student's commands
that fail with e (student error density); E_T(e) = fraction of the teacher's
commands that fail with e (teacher error density); R_T(e) = fraction of the
teacher's e-errors judged recovered (a rate within category e; it ignores how
many e-errors there are). E_T(e)·R_T(e) = recovered e-errors / total commands =
how many recovery demonstrations per command the teacher actually provides for
e. 

Variations currently implemented (each at 11-category and 91-subcategory
granularity):

- **SCRF-all** = Σ_e q_S(e)·E_T(e)·R_T(e), with q_S(e) over all the student's failed commands. Weights every category the student errs in.
- **SCRF-unrecovered** = Σ_e q_S(e)·E_T(e)·R_T(e), with q_S(e) counting only the student's failed commands the K=3 judge marked NOT recovered. More targeted than SCRF-all: it weights only the categories the student cannot fix by itself, which is what the hypothesis is about.
- **SCRF-failed** = Σ_e q_S(e)·E_T(e)·R_T(e), with q_S(e) counting only commands from student trajectories that did not solve the task.
- **SCRF-KL** = -KL(P_S‖P_T), where P_S(e) = the student's not-recovered errors of category e / all its not-recovered errors, and P_T(e) = the teacher's recovered errors of category e / all its recovered errors (both normalized over categories, pooled at teacher level). Ranks teachers by how well their recovered-error distribution matches where the student gets stuck (`scrf_distribution_rank.py`, which also prints a plain overlap Σ_e P_S(e)·P_T(e) as a cross-check).



## Preliminary results

How we evaluate: score every teacher's trajectory for a fixed set of matched
Terminal-Lego tasks, aggregate per teacher, and compare the resulting teacher
ranking to the published SFT ranking for that student
(https://arxiv.org/pdf/2606.03461). Two students: Qwen3-8B (headline,
all proxies) and Qwen3-32B (SCRF and the student-independent proxies, plus the
likelihood family at n=200). The paper's ground-truth rankings differ in one
useful way: for the 8B student GLM-5 and Qwen3.5-Plus are tied, for the 32B
student they are not.

    Qwen3-8B:  DeepSeek-V3.2 > {GLM-5 ≈ Qwen3.5-Plus} > Claude Opus 4.6
    Qwen3-32B: DeepSeek-V3.2 > GLM-5 > Qwen3.5-Plus > Claude Opus 4.6

1000 matched tasks (seed 42) is the headline; n=200 is kept below as the
smaller-sample snapshot. Full tables and metric definitions in
`docs/PROXY_RESULTS_n200.md`. tau-b is Kendall correlation with the GT order (+1
identical, -1 reversed); "top" is the predicted best teacher; P = bootstrap
probability the top teacher is correct.

GT (8B): DS > {GLM ≈ Q35} > CL.

### n=1000, Qwen3-8B student (headline)

At n=1000 the field separates into a clear top tier and the rest. All six SCRF
views, cmd_error (gpt-oss judge), and traj_length reach the maximum tau-b
(+0.91) with P(top-1) ~1.00; tor also reaches +0.91 but is less stable on top-1
(P~0.53). These proxies are indistinguishable on this benchmark: because GLM and
Q35 are tied in the ground truth, tau-b saturates at +0.91 and cannot separate a
proxy that predicts GLM > Q35 from one that predicts Q35 > GLM (that pair is a
tie, so it counts neither for nor against). So SCRF matches, but does not beat,
the cheaper cmd_error and traj_length baselines here; separating them needs
ground-truth rankings with more teachers or without the middle tie. All SCRF
numbers use the corrected per-command (JSON-anchored) segmentation and the
gpt-oss-120b judge.

| proxy | tau-b | top | P(top) | predicted order |
|---|---|---|---|---|
| SCRF-all, 11-cat | +0.91 | DS | 1.00 | DS > GLM > Q35 > CL |
| SCRF-all, 91-subcat | +0.91 | DS | 0.97 | DS > GLM > Q35 > CL |
| SCRF-unrecovered, 11-cat | +0.91 | DS | 1.00 | DS > GLM > Q35 > CL |
| SCRF-unrecovered, 91-subcat | +0.91 | DS | 0.96 | DS > GLM > Q35 > CL |
| SCRF-failed, 11-cat | +0.91 | DS | 0.99 | DS > GLM > Q35 > CL |
| SCRF-failed, 91-subcat | +0.91 | DS | 0.90 | DS > GLM > Q35 > CL |
| cmd_error more (gpt-oss) | +0.91 | DS | 1.00 | DS > GLM > Q35 > CL |
| tor | +0.91 | DS | 0.53 | DS > Q35 > GLM > CL |
| traj_length (more, tokens/turns) | +0.91 | DS | 1.00 | DS > Q35 > GLM > CL |
| egs_loop | +0.55 | Q35 | 0.04 | Q35 > DS > GLM > CL |
| egs_post | -0.18 | Q35 | 0.00 | Q35 > GLM > CL > DS |
| teacher_bench, error_retry, cmd_error (fewer) | -0.91 | CL | 0.00 | CL > ... > DS |

The old n=1000 likelihood rows are omitted here: global/local NLL, ASLEC,
RSR, GRACE, and the all-assistant SCAS view all used the assistant-boundary
scanner that later dropped most Claude tokens. They remain historical artifacts
until rescored with the boundary-safe mask.

### Qwen3-32B student: the tie-free ground truth separates SCRF from length

The 32B ground truth has no tie, so tau-b can reach +1.0 and the proxies that
tied at +0.91 above can be told apart. q_S(32B) is built from n=200 Qwen3-32B
student traces (`runs/terminal_lego-n200-s42-q32b`); teacher-side scoring uses
the same 1000 tasks and gpt-oss-120b judge as the 8B run.

| proxy | tau-b | top | P(top) | predicted order | GLM > Q35 middle right? |
|---|---|---|---|---|---|
| SCRF (all six views) | +1.00 | DS | 0.97-0.99 | DS > GLM > Q35 > CL | yes |
| cmd_error more (gpt-oss) | +1.00 | DS | 1.00 | DS > GLM > Q35 > CL | yes |
| traj_length (more, tokens/turns) | +0.67 | DS | 1.00 | DS > Q35 > GLM > CL | no |
| tor | +0.67 | DS | 0.53 | DS > Q35 > GLM > CL | no |
| teacher_bench | -0.67 | CL | 0.00 | CL > GLM > Q35 > DS | - |
| error_retry (raw upstream score) | -1.00 | CL | 0.00 | CL > Q35 > GLM > DS | - |
| error_retry (turn-aware views) | -0.67 | CL | 0.00 | CL > GLM > Q35 > DS | - |

SCRF and cmd_error recover the full 32B ranking; traj_length and tor get the
top and bottom teacher right but swap the middle pair, which is exactly the
pair the 8B tie hid. Caveat on tor: the paper's own TOR values (Table 3:
DeepSeek 13.4%, GLM 7.3%, Qwen3.5-Plus 6.5%, Claude 2.5%) do order the four
teachers exactly like the 32B ground truth, while our original
operationalization gives 56 / 49 / 55 / 33 with GLM and Qwen3.5-Plus swapped.
The paper gives the observation list and three alignment examples but no
action list, and its code is unreleased, so `compute_proxies.py` now reports
TOR in twelve predeclared views: actions {list = edit/install/run commands,
all = every non-observation command}, align {loose = original, strict = same
path or observed directory contains the target, exact = same path}, window
{any = observation anywhere earlier, same turn allowed, prevturn = observation
in an earlier assistant turn, so its output was actually in context}.
Per-teacher means (n=1000, DS / GLM / Q35 / CL, %):

| view | DS | GLM | Q35 | CL | order | tau-b 8B / 32B |
|---|---|---|---|---|---|---|
| paper Table 3 | 13.4 | 7.3 | 6.5 | 2.5 | DS > GLM > Q35 > CL | +0.91 / +1.00 |
| list_loose_any (original) | 56.0 | 48.9 | 55.4 | 33.3 | DS > Q35 > GLM > CL | +0.91 / +0.67 |
| list_strict_prevturn | 32.8 | 20.8 | 30.0 | 6.7 | DS > Q35 > GLM > CL | +0.91 / +0.67 |
| list_exact_prevturn | 28.4 | 16.4 | 25.7 | 4.9 | DS > Q35 > GLM > CL | +0.91 / +0.67 |
| all_strict_prevturn | 27.4 | 22.4 | 30.3 | 6.4 | Q35 > DS > GLM > CL | +0.55 / +0.33 |
| all_exact_prevturn | 23.7 | 17.9 | 25.7 | 4.8 | Q35 > DS > GLM > CL | +0.55 / +0.33 |

The same-turn window is the main inflator (a Terminus-2 turn types several
commands as one batch, so an observation earlier in the same batch was never
seen before the action); requiring a previous turn halves the values and puts
Claude near the paper's 2.5%. Tightening align() changes little. Widening the
action set to all commands makes it worse (Qwen3.5-Plus first). No view
reproduces GLM > Qwen3.5-Plus, and all remain 2-4x above the paper's values,
so something in the paper's unreleased action definition or path parsing is
still different; tor's +0.67 on the 32B ground truth is therefore a property
of our operationalizations, not established for the paper's metric. So on the one benchmark that can separate them, the
recovery-aware proxies beat pure length. SCRF does not beat cmd_error here (both
are perfect); separating those two needs a teacher that errs a lot but does not
recover.

The existing 32B likelihood run is historical rather than decision-grade. It
predates the assistant-boundary repair shared by global/local NLL, ASLEC, RSR,
GRACE, and the all-assistant SCAS view. Its old rows put Qwen3.5-Plus first, but
the analogous 8B conclusion changed after the mask repair, so no same-family
claim should be made until the 32B rows are rescored. SCAS and GRACE for 32B
also remain incomplete; the multi-GPU SCAS mask-device fix is now merged, but
has not yet been rerun.

### n=200, Qwen3-8B student (smaller-sample snapshot)

| proxy | tau-b | top | P(top) | predicted order |
|---|---|---|---|---|
| traj_length (more) | +0.91 | DS | 1.00 | DS > Q35 > GLM > CL |
| tor | +0.91 | DS | 0.82 | DS > Q35 > GLM > CL |
| teacher_bench | -0.91 | CL | 0.00 | CL > GLM > Q35 > DS |
| error_retry | -0.91 | CL | 0.00 | CL > Q35 > GLM > DS |
| cmd_error more (gpt-oss) | +0.55 | GLM | 0.43 | GLM > DS > Q35 > CL |
| cmd_error more (qwen) | +0.91 | DS | 0.50 | DS > GLM > Q35 > CL |
| egs_post | -0.18 | Q35 | 0.00 | Q35 > GLM > CL > DS |
| egs_loop | +0.55 | Q35 | 0.11 | Q35 > DS > GLM > CL |
| global_nll (GRAPE) | -0.18 | Q35 | 0.00 | Q35 > GLM > CL > DS |
| local_nll k1 | -0.55 | Q35 | 0.00 | Q35 > CL > GLM > DS |
| local_nll k2,4,8 | -0.18 | Q35 | 0.00 | Q35 > GLM > CL > DS |
| aslec_drop (legacy skip-1) | +0.18 | Q35 | 0.00 | Q35 > GLM > DS > CL |
| aslec_casl (legacy skip-1) | -0.18 | Q35 | 0.00 | Q35 > GLM > CL > DS |
| rsr (direction + boundary corrected) | +0.55 | DS | 0.999 | DS > GLM > CL > Q35 |
| scas (legacy all-assistant adaptation) | +0.18 | Q35 | 0.00 | Q35 > GLM > DS > CL |
| grace | +0.55 | Q35 | 0.35 | Q35 > DS > GLM > CL |
| SCRF-unrecovered, 11-cat (gpt-oss) | +0.91 | DS | 0.68 | DS > GLM > Q35 > CL |
| SCRF-unrecovered, 11-cat (qwen) | +0.18 | GLM | 0.33 | GLM > DS > CL > Q35 |

## Additional findings

- A fixed-teacher Kimi K2.5 IssueTasks audit provides 1,208 repeated tasks with
  2--3 trajectories each. Boundary-safe Qwen3-8B scoring yields a nontrivial
  median within-task RSR spread of 0.196, but RSR is strongly correlated with
  exact SFT target length within task (centered Spearman +0.662). RSR-low uses
  5.37M target tokens, versus 6.10M for seeded random and 6.89M for RSR-high.
  On the 88 groups with nonconstant outcomes, RSR-low selects 47 passes versus
  45--48 for seeded random. Thus it is a useful mechanism diagnostic, but a
  naive low/high SFT run would confound proxy selection with supervision volume.
- Across Terminal-Lego and OT-Agent, RSR largely recovers the published ordering
  of teachers by downstream SFT performance. Terminal-Lego recovers the top
  teacher and 4/5 ordered pairs; the source-balanced OT-Agent 1K comparison
  recovers the complete four-teacher point order. This is promising
  teacher-selection evidence, not yet evidence that RSR selects better
  individual training trajectories.
- A response-independent Terminal-Lego draw now tests that individual-selection
  question without training yet. After dropping eight tasks with any truncated
  candidate, it retains 1,692 identical tasks with four verifier-passing teacher
  trajectories each. The proposed RSR-low, seeded-random, and RSR-high arms each
  select one complete trajectory per task, use exactly 423 rows per teacher and
  3,797,689 LlamaFactory target tokens. The random control permits natural
  overlap with RSR-low (504/1,692 trajectories, 29.8%); only the deliberately
  high-RSR contrast is disjoint. Mean selected RSR is 2.311/2.494/2.708. Total
  sequence tokens match within 0.25%, squared sequence-length sums within 1%,
  and an analytic Qwen3-8B training-FLOP approximation within 0.08%. These are
  frozen candidate-data artifacts. Per-teacher target-token exposure is audited
  but not forced equal: equal row counts define the practical routing-policy
  comparison, while token-level teacher exposure remains a reported mediator
  and possible alternative explanation. Downstream SFT has not been run or
  approved.
  See the compact
  [machine-readable summary](artifacts/terminal_lego_sft_mix_n1692_summary.json).
- A separate [OT-Agent transfer audit](docs/AGENTIC_TRANSFER_AUDIT.md) evaluates
  the corrected proxies on 445 exact-instruction-matched public teacher
  trajectories from the Qwen3-8B Table-6 ablation. It records the task/source
  controls, RSR's partial ranking transfer, source sensitivity, and a concentrated
  GPT-5.3-Codex no-op-loop confound. This is an artifact-aware case study, not a
  second clean ground-truth benchmark.
- traj_length reproduces the 8B ranking (tau-b +0.91) but that is pure length: DeepSeek writes the longest trajectories. Its GLM/Q35 order differs from SCRF's; for the 8B student that pair is a ground-truth tie, so no verdict, but for the 32B student the pair is ordered (GLM > Q35) and traj_length gets it wrong (+0.67) while SCRF and cmd_error get it right (+1.00). TOR (+0.91) and cmd_error are rates (per action, per command), so they are not mechanically length-driven; TOR reflects DeepSeek inspecting before acting more. cmd_error per command was ambiguous at n=200 (gpt-oss judge said GLM, qwen said DS) but resolves to a clean DS > GLM > Q35 > CL at n=1000 (tau-b +0.91, P=1.00) with the gpt-oss judge; the per-turn "DeepSeek first" seen earlier was a batching artifact.
- Most student-likelihood proxies (GRAPE, LALP at all k, legacy skip-1 ASLEC,
  the legacy all-assistant SCAS adaptation, and GRACE) rank Qwen3.5-Plus first
  for the Qwen3-8B student. Boundary-corrected RSR is the exception: it ranks
  DeepSeek first and recovers 4/5 ordered GT pairs. The n=1000 and Qwen3-32B
  RSR rows on the fork predate the boundary fix and remain stale; they must not
  be used to argue that corrected RSR has same-family bias.
- **SCRF works and stabilizes with n.** With the corrected per-command segmentation and the gpt-oss-120b judge, all SCRF variations reproduce the published ranking (tau-b +0.91, DeepSeek top-1) at both taxonomy granularities. At n=200 it is right but not yet bootstrap-stable (P(top-1) ~0.68 at 11-cat, ~0.50 at 91-subcat); at n=1000 every SCRF view is highly stable (P(top-1) 0.90-1.00, 11-cat > 91-subcat). For the 8B student it ties cmd_error and traj_length at the tau-b ceiling (+0.91): the GLM/Q35 ground-truth tie caps tau-b, so no proxy can be shown to separate there. The 32B ground truth has no tie and does separate them: SCRF +1.00 (all views), traj_length +0.67 (section above). q_S sensitivity to the number of student traces is still open: the 8B run with q_S from 1000 instead of 200 student traces hit the score cache and was not actually recomputed (job 4140407; the file is byte-identical to the 200-trace one), so it has to be rerun with the cache cleared.
- **But it is judge-dependent.** With the Qwen3-32B judge the same proxy is weak at n=200 (tau-b -0.18..+0.55, GLM first). So the signal is real but hinges on judge quality at the error-labelling step. (An earlier version, before the segmentation fix, made SCRF look signal-less; that was a parsing artifact.)

  | SCRF-unrecovered | judge | segmentation | tau-b | top-1 | P(top-1) |
  |---|---|---|---|---|---|
  | 11-category | gpt-oss-120b | JSON-anchored (current) | +0.91 | DS ✓ | 0.68 |
  | 91-subcategory | gpt-oss-120b | JSON-anchored (current) | +0.91 | DS ✓ | 0.50 |
  | 11-category | Qwen3-32B | JSON-anchored (current) | +0.18 | GLM ✗ | 0.33 |
  | (any) | gpt-oss-120b | per-turn (old, batching artifact) | +0.55 | DS | 0.24 |
  | (any) | Qwen3-32B | per-turn (old) | +0.18 | Q35 ✗ | 0.07 |
- **n matters: unstable at 200, stable at 1000.** At n=200, under task-bootstrap resampling most proxies' own teacher ranking is not reproduced in 80% of resamples, i.e. a different sample of tasks would likely give a different ranking (per-trajectory score variance within a teacher is as large as or larger than the variance between teacher means, so teacher identity is a coarse selection unit). Extending to n=1000 resolves this for SCRF and cmd_error: they reach P(top-1) 0.90-1.00, and cmd_error with the gpt-oss judge jumps from tau-b +0.55 (n=200) to +0.91, P=1.00 (n=1000). Corrected RSR is already top-1 stable at n=200; its boundary-safe n=1000 result is still pending.
- **Judge agreement is high on failure detection but low on the fine taxonomy** (two judges, Qwen3-32B vs gpt-oss-120b, identical commands):

  | judgment | n | agreement |
  |---|---|---|
  | is it a failure? | 12,244 commands | 98.5% |
  | recovered? (K=3) | 838 failures | 85.9% |
  | which of the 11 categories | 1,381 failures | 61.0% |
  | which of the 91 subcategories | 1,381 failures | 51.6% |

  so error/recovery signals are judge-robust, but the fine subcategory label is not, which is why SCRF is also reported at the 11-category level.

## Layout

Stage 1-2 (data):
- `prepare_dataset.py`: builds the matched task list and pins the teacher trajectory files.
- `generate_trajectories.py`: runs the student on the sampled tasks to produce its traces.

Stage 3 (score each teacher trajectory):
- `compute_proxies.py`: per (teacher, task) score for every proxy.
- `judge.py`: the LLM judge used by cmd_error and SCRF.

Stage 4 (rank and analyze):
- `evaluate_ranking.py`: teacher rankings, bootstrap, sample-efficiency.

Stage 5 (materialize candidate SFT data; no training):
- `materialize_terminal_lego_subset.py`: draws a response-independent matched
  task subset and audits four-way task-description/environment provenance.
- `build_sft_mixes.py`: selects one complete trajectory per matched task and
  writes RSR/NLL/rank/TOR-selected, random, global-teacher, teacher-balanced,
  target-token-matched, and context-compute-matched controls. It joins exact
  LlamaFactory SFT-token audits by trajectory ID and refuses stale
  assistant-mask semantics.
- `audit_eval_overlap.py`: checks normalized exact and five-token-shingle
  overlap between candidate training instructions and a pinned eval task set.
  These scripts are separate from the original four-stage scoring pipeline
  because their output is candidate training data rather than another ranking
  report; none of them launches training.
- The current candidate-data manifest is the 1,692-task four-way selection on
  the large workspace; it records input/output hashes, exact teacher and token
  controls, sequence-compute diagnostics, proxy distributions, and parser
  audits. The older `artifacts/terminal_lego_sft_mix_n200.json` is retained only
  as the initial smoke. No SFT has been launched.

Supporting scripts:
- `plot_scrf_errors.py`: error-category distributions behind SCRF.
- `scrf_distribution_rank.py`: SCRF-KL / overlap teacher ranking, derived (no rerun) from the per-task scrf components; separate from compute_proxies because it is a teacher-level score over pooled distributions, not a per-(teacher, task) score (like GRACE's teacher-level step).

- `slurm/`: cluster launch scripts. `docs/`: specs and the results write-up.
- `runs/`, `venv/`, `upstream/` (the cloned author repos) are under `$WS_ROOT/teacher_ranking_proxy/`.

## To dos

- TOR still 2-4x the paper's Table 3 values and GLM/Qwen3.5-Plus swapped in every one of the twelve views (see the 32B section). Remaining candidates: per-turn instead of per-command counting, a narrower "state-changing" action definition (file edits only), or stricter path parsing (only explicit file operands). Ask the authors for the TOR script; teacher-only, seconds to rerun. The old single-score file is kept as `proxy_scores/*/tor__v1_listloose.jsonl`.
- Rerun 8B SCRF with q_S built from the 1000-trace student run (clear `proxy_scores/Qwen__Qwen3-8B/scrf@gpt-oss-120b.jsonl` first; the 200-trace result is kept as `scrf@gpt-oss-120b__qS200.jsonl`) to check how many student traces q_S needs.
- Rerun the n=1000 and Qwen3-32B likelihood families with the boundary-safe
  assistant mask. The multi-GPU SCAS mask-device fix is merged; SCAS and GRACE
  still need execution. The earlier 32B n=1000 job was interrupted by the
  Capella storage outage of 2026-09-09.
- Third judge (GLM-4.6-FP8) for 3-way agreement on cmd_error/SCRF labels: needs two nodes (337 GB weights); the alpha 8x40 GB attempt OOMs and the Capella 2-node Ray/vLLM bringup did not become healthy within 40 min.
- Build a pipeline that finds more ground-truth student/teacher rankings on agentic tasks, from open-source trajectories and tasks, so proxies can be validated on more than the one Terminal-Lego ranking. Started in `litmine/` (literature-mining pipeline; spec in `docs/PIPELINE.md`).
- FLOP/token-budget control is currently out of scope for this reproduction because the published SFT runs are external and cannot be retroactively controlled.
