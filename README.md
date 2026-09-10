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
(https://arxiv.org/pdf/2606.03461). Student: Qwen3-8B. Teachers and the
paper's ground-truth ranking for this student:

    DeepSeek-V3.2 > {GLM-5 ≈ Qwen3.5-Plus} > Claude Opus 4.6

200 matched tasks (seed 42), extending to 500 and 1000. n=200 snapshot below;
full tables and metric definitions in `docs/PROXY_RESULTS_n200.md`. tau-b is
Kendall correlation with the GT order (+1 identical, -1 reversed); "top" is the
predicted best teacher; P = bootstrap probability the top teacher is correct.

GT: DS > {GLM ≈ Q35} > CL.

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
| rsr (direction corrected) | -0.18 | CL | n/a* | CL > DS > GLM > Q35 |
| scas (legacy all-assistant adaptation) | +0.18 | Q35 | 0.00 | Q35 > GLM > DS > CL |
| grace | +0.55 | Q35 | 0.35 | Q35 > DS > GLM > CL |
| SCRF-unrecovered (gpt-oss) | +0.55 | Q35 | 0.24 | Q35 > DS > GLM > CL |
| SCRF-unrecovered (qwen) | +0.18 | Q35 | 0.07 | Q35 > GLM > DS > CL |

\* RSR's point result follows exactly by applying the official lower-is-better direction to the stored reported order. Its exact bootstrap probability requires the original score JSONL, which is not committed in this standalone snapshot. See `docs/PROXY_IMPLEMENTATION_AUDIT.md`.

## Additional findings

- A separate [OT-Agent transfer audit](docs/AGENTIC_TRANSFER_AUDIT.md) evaluates
  the corrected proxies on 445 exact-instruction-matched public teacher
  trajectories from the Qwen3-8B Table-6 ablation. It records the task/source
  controls, RSR's partial ranking transfer, source sensitivity, and a concentrated
  GPT-5.3-Codex no-op-loop confound. This is an artifact-aware case study, not a
  second clean ground-truth benchmark.
- traj_length reproduces the ranking (tau-b +0.91) but that is pure length: DeepSeek writes the longest trajectories. TOR (+0.91) and cmd_error are rates (per action, per command), so they are not mechanically length-driven; TOR reflects DeepSeek inspecting before acting more, and cmd_error per command has no clear winner (its per-turn "DeepSeek first" was a batching artifact, and its top teacher depends on the judge).
- Most student-likelihood proxies (GRAPE, LALP at all k, legacy skip-1 ASLEC, the legacy all-assistant SCAS adaptation, and GRACE) rank Qwen3.5-Plus first for the Qwen3-8B student. The earlier RSR entry did so only because its direction was reversed; corrected RSR ranks Claude first and is still negatively correlated with the ground truth.
- SCRF: at the per-command unit it no longer robustly recovers the published top teacher; as currently defined it carries little signal beyond command-count effects. Refinement should use the 11-category level.
- **Not stable at n=200.** Under task-bootstrap resampling most proxies' own teacher ranking is not reproduced in 80% of resamples, i.e. a different sample of tasks would likely give a different ranking. Per-trajectory score variance within a teacher is as large as or larger than the variance between teacher means, so teacher identity is a coarse selection unit. This is why the benchmark is being extended to 500 and 1000 tasks.
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

Supporting scripts:
- `plot_scrf_errors.py`: error-category distributions behind SCRF.
- `scrf_distribution_rank.py`: SCRF-KL / overlap teacher ranking, derived (no rerun) from the per-task scrf components; separate from compute_proxies because it is a teacher-level score over pooled distributions, not a per-(teacher, task) score (like GRACE's teacher-level step).

- `slurm/`: cluster launch scripts. `docs/`: specs and the results write-up.
- `runs/`, `venv/`, `upstream/` (the cloned author repos) are under `$WS_ROOT/teacher_ranking_proxy/`.

## To dos

- Extend to 500 and 1000 tasks to see whether the proxy rankings become stable.
- Build a pipeline that finds more ground-truth student/teacher rankings on agentic tasks, from open-source trajectories and tasks, so proxies can be validated on more than the one Terminal-Lego ranking. Started in `litmine/` (literature-mining pipeline; spec in `docs/PIPELINE.md`).
- FLOP/token-budget control is currently out of scope for this reproduction because the published SFT runs are external and cannot be retroactively controlled.
