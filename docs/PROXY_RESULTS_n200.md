# Teacher-ranking proxies, n=200 Terminal-Lego benchmark (audited 2026-09-08)

Run: `runs/terminal_lego-n200-s42` (seed 42, 200 matched tasks, 4 teachers,
student Qwen/Qwen3-8B). Student traces: 199 trials on the validated Capella
stack (70 solved, 35%), 1476 episodes of which 603 (41%) are agent-format
(parse-retry) episodes; 16 trials never produced a valid command.
Published ranking (Qwen3-8B): DeepSeek-V3.2 > {GLM-5 ≈ Qwen3.5-Plus} > Claude Opus 4.6.

Reports: `ranking_report.md/json` (final, all proxies incl. GRACE and both
judges, 2026-08-26 04:53); `ranking_report_nograce.json` (earlier run without
GRACE).

**Audit correction.** The original report ranked raw RSR descending.  A later
reconstruction also found a response-boundary bug shared with pinned upstream
code: it scored only 207/724 Claude assistant turns.  The boundary-safe rescore
now gives `DeepSeek > GLM > Claude > Qwen3.5`, recovers 4/5 ordered GT pairs,
and puts DeepSeek first in 99.8% of paired bootstraps.  See
`RSR_BOUNDARY_AUDIT.md`.  The ASLEC and SCAS rows remain legacy configurations;
see `PROXY_IMPLEMENTATION_AUDIT.md`.

## 1. Implementation status (PROXY_SPEC.md §6-9)

Where the authors' code exists it is called directly (GRACE aggregation) or its pinned helpers/formulas are reused and checked numerically on real trajectories (`upstream_equivalence.py`, result in `artifacts/upstream_equivalence_n8.json`; RSR and ASLEC exact in that run). The stored SCAS audit predates the corrected `official_final` view and is intentionally retained as evidence that the old all-assistant view was not equivalent.

| § | proxy | status | upstream code | adaptation notes |
|---|---|---|---|---|
| 6.1 | teacher_bench | done | sourced numbers (`artifacts/teacher_bench_scores.json`) | published benchmark score per teacher, constant across tasks |
| 6.2 | traj_length | done | own implementation | 4 predeclared views (turns/tokens, more/less), finding that more is better proxy |
| 6.3 | cmd_error | done | TB2.0 App. E.2–E.4 verbatim (`artifacts/tb2_taxonomy.json`) + one added sentence (`docs/JUDGE_PROMPTS.md`) | as in the paper the judge sees only the executed command and its terminal output, i.e. the `keystrokes` strings Harbor typed into the terminal (extracted from the Terminus-2 JSON's `commands` list; the agent's reasoning is not shown) and that command's own output; segments are cut from the observation Terminus-2 returns after the turn ("New Terminal Output": everything new in the tmux scrollback, head+tail with a 10 KB middle omission for long outputs; a window-only fallback in 0–12% of turns) at the echoed prompt lines (`root@…#` in the Docker-generated teacher data, `Apptainer>` in the student runs), giving one segment per executed command as in the paper (a Terminus-2 turn batches several commands; DeepSeek ≥5 in 36% of turns). 99–100% of typed commands are recovered as segments per teacher; each segment's output is capped at its last 4,000 characters; rate = failed commands / commands; judges: Qwen3-32B (initial prompts) and gpt-oss-120b (final prompts); paper: GPT-5 |
| 6.4 | error_retry | done | hanzunye/swe-trajectory-quality-study @028f154 | their B2 functions (`_compute_b2_error_retry`, keyword set, `B2_MAX_CYCLES=10`) copied verbatim into `compute_proxies.py`; adaptation: in their setting the agent calls named tools; in ours there is one tool, typing into the shell. They call two consecutive steps similar when the tool name is the same (coarse: the arguments may have changed and resolved the error). The equivalent of the tool name in a shell is the program being run, the first word of the command line, so we fill their (tool name, args) slot with (first word, full command line): equivalently coarse: `python a.py` → error → `python b.py` (a different script) counts as an error retry |
| 6.5 | tor (teachers who inspect before acting are better, terminal-lego paper) | done | upstream unreleased, reimplement paper formula | commands = the per-command screen segments (as cmd_error). Classify each as observation / action / other: observation = the paper's list (`cat ls find grep head wc diff stat`, + `pwd`); action = no list in the paper, ours: `sed tee cp mv rm mkdir touch chmod tar pip apt-get npm make gcc python node bash git patch …` plus any line with a `>` redirect (writes/creates a file). As in the paper, an observation is related to an action if they concern the same path: each command's target file/dir is extracted and made absolute with the shell's cwd at that moment (from the prompt line); aligned = same path, or one directory contains the other, or same filename. TOR = actions with an earlier aligned observation / actions |
| 6.6 | egs_post, egs_loop | done | own implementation | extensions of TOR on the same events, reported separately (no composite). tor: teachers who inspect before acting are better (terminal-lego paper). egs_post (act -> verify): teachers who check the result of an action are better; verify = within the next 3 assistant turns an observation on an aligned path or a test/build command (`pytest tox unittest make ctest`). egs_loop (inspect -> act -> verify): both conditions. egs_pre (identical to tor) and egs_adapt (keyword heuristic, optional in the spec, no signal) were removed 2026-08-26 |
| 7.1 | global_nll (GRAPE) | done | own implementation of the paper's score | feed the full history into the student, compute the NLL only over the teacher's assistant tokens (task text and terminal observations are never scored); score = mean log-probability over assistant tokens |
| 7.3 | local_nll_k{1,2,4,8} | done | own implementation of the paper's LALP (local average log-probability) | same scoring, shorter context: for each assistant turn feed the student the task prompt plus only the previous k assistant turns with their observations, compute the NLL over that assistant turn's tokens only; turn means averaged equally (one forward pass per turn). Result: k makes no difference (τ = -0.18 for k = 2, 4, 8; -0.55 for k = 1), all put Qwen3.5-Plus first like global_nll |
| 7.4 | aslec_drop / aslec_casl | implementation corrected; revised scores pending | wangbing1416/ASLEC @5737d69, scoring reimplemented from their `output_drop_score` / `output_causal_score` and verified identical on real trajectories | Steps = assistant turns. The released driver and selection paths use `skip_tokens=2`, now the primary configuration. Results below came from the legacy skip-1 configuration and require a forward rescore. |
| 7.5 | rsr | corrected + boundary-safe | UmeanNever/RankSurprisalRatio @59a7c4c; formulas match upstream, while assistant-span detection intentionally fixes its whitespace-boundary failure | per assistant token: rank of the gold token (clipped at 100) and its NLL; native teacher score = mean rank / mean surprisal (their ratio of means), **lower is better** |
| 7.6 | grace | done | abhishekpanigrahi1996/GRACE @64fc99a cloned under `$WS_ROOT/teacher_ranking_proxy/upstream/`; their `grace()` is imported and called unchanged, the gradient step is re-done in our code because their script reads their own data format | per trajectory: student wrapped with LoRA as in their `--use-lora` option, NLL on the teacher's assistant tokens, backward pass, `lora_B` gradients concatenated (their rule), projected to 512 dims with their library call (TRAK CudaProjector, Rademacher, seed 0; fp32 since this fast_jl build rejects bf16; chunked cross-entropy so 32k-token trajectories fit one GPU). Teacher score = official grace() over its 200 vectors (10 splits, test fraction 0.1, one trajectory per task). Teacher-level only, no per-trajectory score |
| 7.7 | scas | implementation corrected; revised scores pending | ppsmk388/Student-Centric-Answer-Selection @4cec3a6, their `metric_utils` helpers called directly | two explicit views now replace the misleading equivalence claim: `official_final` matches the released final-answer mask; `agent_all_assistant` retains the old multi-turn adaptation. The result below is the legacy all-assistant score. |
| 9 | scrf | done | own proposal (PROXY_SPEC §9) | 4 predeclared q_S views (see §3) |

## 3. SCRF (§9), current formulation

SCRF(T, task) = Σ_e q_S(e) · E_T(e) · R_T(e) over the TB2.0 taxonomy. Student errors are
run through the same K=3 recovery judge as teacher errors (R_S(e)). The student's
agent-format failures (invalid Terminus-2 JSON, 41% of its turns) are reported as
descriptive statistics only: teachers never make them, so a q_S class for them
contributes nothing (a view with it was tried and dropped). Views:

* `qS_all_episodes`: for each error category e, q_S(e) = number of student commands whose failure was classified as e / number of student commands (original)
* `qS_unrecovered`: same, counting only errors the recovery judge marked not recovered
* `qS_failed_trajectories`: same, counting only commands from student trials that failed the task

Each of the three is produced at two granularities and reported separately: the 91 TB2.0 subcategories (view name as above) and the 11 top-level categories (suffix `_cat`), because the two judges agree far better at category level (§8). Six views in total; `score` = `qS_all_episodes` (subcategory).

Controls per row: total error rate, recovery rate, student–teacher error
cosine, unconditioned recovery density.

## 4. Why the judge prompts were changed (PROXY_SPEC §9.3 inspection)

Before freezing the prompts, 20 teacher turns flagged as failures by the verbatim TB2.0 prompt were judged by GPT-5.5 with the initial and the final prompts (`judge_cache/rubric_review_gpt-5.5.md`). The one added sentence to the failure prompt un-flags 5/20, all of them expected failures (deliberately invalid test input, an invalid-domain DNS probe, an SSH-timeout diagnostic). The final recovery rubric, which requires a changed action and visible success, calls 9/20 "recovered" instead of 16/20, rejecting cases where the agent only inspected the error and then declared the task complete. All labels in this report use the final prompts (`docs/JUDGE_PROMPTS.md`).

## 5. Ranking results

(filled from `ranking_report*.json`, see §5.1–5.3 below)

### 5.1 Teacher ranking on the 200-task benchmark (student Qwen3-8B)

**GT order (published SFT ranking): DeepSeek-V3.2 > {GLM-5 ≈ Qwen3.5-Plus} > Claude Opus 4.6**

How to read the table. Every metric compares the proxy's predicted order with the GT order. predicted order = the order of the proxy's mean score per teacher. τ-b = Kendall rank correlation between predicted and published order (+1 identical, -1 reversed, 0 unrelated; with 4 teachers only ±0.91, ±0.55, ±0.18 are possible; the tie is handled). pairwise = fraction of the 5 teacher pairs with a published order that the proxy orders the same way (GLM vs Qwen3.5 is tied and excluded). top-1 = is DeepSeek first. P(top-1) boot = fraction of 10 000 task-bootstrap resamples (200 tasks drawn with replacement, ranking recomputed) in which the top teacher is correct, i.e. how robust the top-1 is to which tasks were sampled. A proxy with no information gets τ-b 0, top-1 correct 25%, pairwise 50%; with 4 teachers there are only 24 orderings, so a perfect match happens by chance with probability 1/24, hence no significance claims (PROXY_SPEC §4).

| proxy | τ-b | pairwise | top-1 | P(top-1) boot | predicted order |
|---|---|---|---|---|---|
| tor, egs_pre | +0.91 | 1.00 | ✓ | 0.95 | DS > Q35 > GLM > CL |
| traj_length (more tokens / more turns) | +0.91 | 1.00 | ✓ | 1.00 | DS > Q35 > GLM > CL |
| cmd_error (**more** errors) | +0.91 | 1.00 | ✓ | 0.95 | DS > GLM > Q35 > CL |
| scrf (3 q_S views) | +0.55 | 0.80 | ✓ | 0.81–0.84 | DS > GLM > CL > Q35 |
| aslec_drop (legacy skip-1), scas (legacy all-assistant), egs_adapt | +0.18 | 0.60 | ✗ | 0–0.26 | Q35 > GLM > DS > CL |
| rsr (**direction + boundary corrected**) | +0.55 | 0.80 | ✓ | 0.998 | DS > GLM > CL > Q35 |
| global_nll, local_nll_k2/4/8, aslec_casl, egs_post | −0.18 | 0.40 | ✗ | 0 | Q35 > GLM > CL > DS |
| local_nll_k1 | −0.55 | 0.20 | ✗ | 0 | Q35 > CL > GLM > DS |
| teacher_bench, error_retry, cmd_error (fewer), traj_length (less) | −0.91 | 0.00 | ✗ | 0 | CL first |
| grace (teacher-level, official) | +0.55 | 0.80 | ✗ | 0.35 (200 replicates) | Q35 > DS > GLM > CL, raw GRACE 3.06 / 3.11 / 3.25 / 23.6 (lower better); τ CI [0.18, 0.91]; DS/Q35/GLM within 6% of each other, only Claude is clearly separated |

(Reading of the final table follows below once the regenerated report is in.)

### 5.2 Teacher-level vs trajectory-level (§10)

Before averaged over trajectories. Now look at the per-trajectory scores, i.e. one score per (teacher, task).

1. Winner share. For each task, compare the four teachers' trajectory scores and note which teacher has the best one. The four numbers are the fraction of the 200 tasks each teacher wins (DS / GLM / Q35 / CL). If teacher identity were what matters, the best teacher would win nearly every task. Instead, e.g. for TOR DeepSeek is first on average but wins only 37% of tasks; GLM wins 21%, Qwen3.5 28%, Claude 13%. Only trajectory length is dominated by one teacher (DeepSeek's trajectories are the longest on 66% of tasks).

2. Within-teacher spread vs. between-teacher spread. std = how much a teacher's scores vary from task to task; spread of means = the gap between the best and the worst teacher average. When the within-teacher std is as large as or larger than the between-teacher gap (true for almost every proxy), the teacher averages differ by less than the task-to-task noise: a "worse" teacher's trajectory beats the "better" teacher's on many tasks.

Both point the same way: a dataset that takes, for each task, the trajectory with the best proxy score regardless of which teacher produced it (DeepSeek's for one task, GLM's for the next, ...) would differ a lot from taking all trajectories from the single best teacher. 

| proxy | winner share | within-teacher std vs. spread of teacher means |
|---|---|---|
| traj_length tokens_more | 0.66/0.13/0.19/0.02 | std 2.4k tokens vs 1.6k spread |
| tor / egs_pre | 0.37/0.21/0.28/0.13 | 0.29 vs 0.19 |
| cmd_error more_errors | 0.34/0.25/0.19/0.22 | 0.14 vs 0.05 |
| global_nll | 0.03/0.33/0.43/0.21 | 0.20 vs 0.25 |
| local_nll_k1…k8 | ≈0.01/0.29/0.45/0.25 | 0.14 vs 0.32–0.37 |
| aslec_drop / casl | 0.02/0.37/0.47/0.14, 0.02/0.31/0.41/0.26 | 0.19 vs 0.27, 0.20 vs 0.25 |
| rsr (boundary-safe) | 0.415/0.205/0.120/0.260 | residual teacher order is DS > GLM > CL > Q35; CL/Q35 is the only wrong GT pair |
| scas | 0.08/0.30/0.42/0.19 | 0.31 vs 0.25 |
| egs_adapt / egs_post | ≈0.25 each | 0.34 vs 0.13 / 0.07 |
| error_retry | 0.21/0.24/0.25/0.30 | 0.08 vs 0.05 |

For most proxies the within-teacher std is of the same order as (or larger
than) the spread between teacher means, and no teacher wins more than ~half
of the tasks except traj_length, consistent with teacher identity being a
coarse selection unit (§10). The legacy likelihood views other than RSR almost
never pick DeepSeek at task level (1–3%); corrected min-RSR task winners cannot
be recovered from the summary statistics alone.

### 5.3 Sample efficiency (§11)

Question: how many matched tasks are needed before a proxy's teacher ranking stops changing? Nested seeded subsets of the 200 tasks (10 ⊂ 25 ⊂ 50 ⊂ 100 ⊂ 200, the same tasks for every teacher and proxy); on each subset the teacher means and the ranking are recomputed, and a 1000-replicate task bootstrap gives P(order) = the probability that the full-200 order comes out. τ-b column = agreement with the GT order using only that many tasks. last column = the smallest subset size at which the ranking equals the 200-task ranking and P(order) ≥ 0.8 there and at every larger n; "not stable" = even at 200 the order is not reproduced in 80% of resamples. Not stable does not mean wrong: it means the proxy's own ranking is uncertain under task resampling.


| proxy | τ-b (10/25/50/100/200) | smallest n with P(order) ≥ 0.8 from there on |
|---|---|---|
| traj_length tokens/turns (more) | +0.91 at every n | 25 (turns) / never ≥0.8 (tokens: 0.60–0.70) |
| tor, egs_pre | +0.55/+0.91/+0.55/+0.91/(+0.91) | not stable (P ≤ 0.61) |
| cmd_error more | +0.18/+0.55/+0.55/+0.55/(+0.91) | not stable (P ≤ 0.16) |
| error_retry, teacher_bench | −0.91 at every n | 100 / 10 |
| global_nll, aslec_casl | −0.18 at every n | 25 |
| local_nll_k1 | −0.91/−0.55/−0.55/−0.55/−0.55 | not stable (0.75–0.79) |
| local_nll_k2 | −0.91/−0.55/−0.55/−0.55/−0.18 | not stable |
| local_nll_k4 | −0.91/−0.55/−0.55/−0.18/−0.18 | 200 |
| local_nll_k8 | −0.91/−0.55/−0.55/−0.18/−0.18 | 200 |
| rsr (old boundary-broken scores) | superseded | see `RSR_BOUNDARY_AUDIT.md`; the corrected n=200 top-1 bootstrap support is 99.8% |
| scas | −0.18/−0.18/−0.18/+0.18/+0.18 | 200 |
| aslec_drop | −0.18 ×4 / +0.18 | not stable (0.69) |

(n=200 entries in parentheses are from the full-pool table; the nested
n=200 subset is undefined for tor/egs/cmd_error because a few tasks have no
scorable actions, so those proxies use 190–199 matched tasks.) The
length-type proxies are stable from n≈25; the likelihood proxies that are
*wrong* are stably wrong from n≈25; the proxies whose sign flips with n
(scas, aslec_drop, local_nll_k≥2) are exactly the ones with |τ| = 0.18 at
n=200, i.e. no real signal.

## 6. Cost (§4)

| proxy family | GPU | judge | wall-clock (n=200, 1×H100) |
|---|---|---|---|
| teacher-only (tor, egs, error_retry, traj_length) | no | no | seconds–minutes |
| cmd_error | no student weights | LLM judge (Qwen3-32B or gpt-oss-120b, served with vLLM on the job's GPU): one failure-detection call per command (~12.7k on n=200), then one error-classification call per command flagged as a failure (~1.7k) | from scratch on n=200: Qwen3-32B ~10 min, gpt-oss-120b ~20 min, plus ~1.5-2.5 min vLLM startup |
| scrf | no student weights | same judge; adds one recovery call per detected failure (~1.8k), reusing cmd_error's failure/classification labels | from scratch after cmd_error: Qwen3-32B ~27 min, gpt-oss-120b ~29 min (run alone it also pays cmd_error's cost) |
| global_nll, rsr, scas | student forward | no | 5–10 min each |
| local_nll (per k) | student forward per turn | no | ~10–17 min each |
| aslec (both) | shared with likelihood | no | 4 min |
| grace | student forward+backward (LoRA) + projection | no | 40 min |

## 7. Open items

1. Qwen3-32B student for the §8 student-specificity test (no runs yet).
2. SCRF refinement (PROXY_SPEC §9.4): start from the 11-category level, where the judges agree (see §8.1).
3. §12 (proxy-selected SFT datasets): STOP, waits for manual go per spec.

## 8. cmd_error and scrf: per-command unit and judge agreement

These two proxies judge each executed command. A Terminus-2 turn can batch several commands (DeepSeek: 36% of turns have 5+, Qwen3.5: only 9% do), so scoring per turn confounds the error rate with batching style; per command (the paper's unit) removes that. Both judges (Qwen3-32B, gpt-oss-120b) labelled the same commands with the final prompts.

Judge agreement, on the identical commands:

| judgment | n | agreement | flagged: Qwen / gpt-oss |
|---|---|---|---|
| is it a failure? | 12,244 commands | 98.5% | 12.1% / 12.1% |
| recovered? (K=3) | 838 failures | 85.9% | 41.2% / 37.8% |
| which of the 11 categories | 1,381 failures | 61.0% | |
| which of the 91 subcategories | 1,381 failures | 51.6% | |

So the two judges agree almost perfectly on whether a command failed, reasonably on whether it recovered, but poorly on the fine subcategory. Since SCRF sums q_S·E_T·R_T per subcategory, that disagreement matters; a refinement (§9.4) should work at the 11-category level.

Rankings (τ-b, top teacher, P(top-1)):

| proxy | unit | Qwen3-32B | gpt-oss-120b |
|---|---|---|---|
| cmd_error (more errors) | per turn | +0.91, DS first (P 0.95) | +0.55, DS first (P 1.00) |
| cmd_error (more errors) | per command | +0.91, DS first (P 0.50) | +0.55, GLM first (P 0.43) |
| scrf (3 q_S views) | per turn | +0.55, DS first (P 0.81-0.84) | +0.55, DS first (P 0.45-0.66) |
| scrf (3 q_S views) | per command | +0.18, Qwen3.5 first (P <= 0.08) | +0.55, Qwen3.5 first (P <= 0.26) |

Takeaway: the per-turn agreement with the published ranking was largely a batching artifact. At the paper's unit, cmd_error's top teacher depends on the judge (DeepSeek vs GLM-5) and SCRF no longer recovers the published top-1, with weak bootstrap support, i.e. SCRF as currently defined has no robust signal beyond command-count/length effects.
