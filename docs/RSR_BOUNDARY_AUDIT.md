# RSR assistant-boundary audit (2026-09-15)

## Finding

The earlier Terminal-Lego RSR result was biased by a response-span bug shared
with the pinned upstream implementation.  The scanner tokenized the fixed Qwen
header `<|im_start|>assistant\n` and searched for that exact token sequence in
the rendered conversation.  Qwen's BPE is context-sensitive at the boundary:
when a response begins with whitespace, the template newline and the response's
leading whitespace can merge into a different token.  The string header is
present, but its separately tokenized ID sequence is not, so the whole assistant
turn is skipped.

This is teacher-style dependent in the public Terminal-Lego release.  On the
exact 200-task, seed-42 sample, 711/724 Claude assistant turns begin with a
newline; none of the DeepSeek, GLM, or Qwen turns do.  The old scanner therefore
found only 207/724 Claude turns and scored an average of 163 assistant tokens per
trajectory, versus 1,771 with the boundary-safe scanner.  The other three
teachers' non-empty assistant turns and token counts are unchanged.

The corrected scanner searches for the stable role prefix without trailing
whitespace, then removes the single whitespace token at the chat-template
boundary.  This intentionally differs from upstream commit `59a7c4c` on the
boundary-merge case.  The earlier upstream-equivalence check established that
our code reproduced upstream; it did not establish that upstream found every
assistant turn.

## Reconstructed n=200 result

Data: public `DCAgent/terminal-lego-traj-8k` revision
`02f33fb252b15308f309e7d10472bb781eeb1ecf`; exact task manifest from the
original seed-42 run.  Student: `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`.  RSR uses rank clipping at 100,
32,768-token context, official ratio-of-trajectory-means aggregation, and lower
is better.

For each teacher token, **rank** is the position of the teacher's actual token
in the student's next-token ordering: rank 1 means it was the student's most
preferred token.  **Surprisal** is `-log P(token | context)`, so it is larger
when the student assigned the token a lower absolute probability.  RSR divides
average clipped rank by average surprisal.  Lower RSR therefore favors tokens
that are relatively high-ranked by the student, but still sufficiently
unexpected to provide a learning signal.

| teacher | boundary-safe RSR | mean scored tokens | mean scored turns | GT |
|---|---:|---:|---:|---:|
| DeepSeek-V3.2 | 2.3911 | 3,302 | 7.42 | 1 |
| GLM-5 | 2.4544 | 2,408 | 5.74 | tied 2 |
| Claude Opus 4.6 | 2.4728 | 1,771 | 3.62 | 4 |
| Qwen3.5-Plus | 2.5195 | 2,435 | 6.32 | tied 2 |

The predicted order changes from `Claude > DeepSeek > GLM > Qwen` to
`DeepSeek > GLM > Claude > Qwen`.  It now recovers the correct top teacher and
4/5 ordered GT pairs (Kendall tau-b 0.55).  In 20,000 paired task bootstraps,
DeepSeek is top in 99.8% of draws.  The remaining error is Claude versus Qwen:
Qwen ranks above Claude in only 1.6% of draws.

The residual is small in proxy space but stable: Claude's RSR is only 1.9%
lower than Qwen's.  Their average surprisals are almost identical (0.7275 versus
0.7324), while Claude has a slightly lower average token rank (1.799 versus
1.845), so RSR prefers Claude.  This metric does not directly represent the
environment-grounded structure highlighted by Terminal-Lego: full-corpus TOR is
2.5% for Claude and 6.5% for Qwen, and Claude also supplies fewer turns and
assistant tokens.  The paper's same-teacher prompt intervention provides
stronger evidence than the length difference alone: increasing Claude's TOR
from 2.5% to 6.6% raises the Qwen3-32B student score from 15.4% to 19.5%.

## Implication for selection

Teacher-level agreement is encouraging but is not yet evidence that ranking
individual agent trajectories by RSR improves SFT.  The original RSR selection
experiment compared multiple correct reasoning solutions for the same problem;
an agentic test should likewise compare candidates for identical tasks and keep
source, outcome validity, and environment fixed.  A defensible first experiment
is a small SFT ablation over random, lowest-RSR, high-TOR, and a predeclared
RSR-plus-EGS rule.  Until then, use RSR as one student-specific signal after
hard validity filters, not as the sole selector.

The OT-Agent scores in `AGENTIC_TRANSFER_AUDIT.md` predate this boundary fix,
but a CPU-only audit of the exact source-balanced 1K datasets shows that a full
four-arm rescore is unnecessary.  Relative to the non-empty assistant spans
actually rendered inside the 32,768-token scoring window, the old and new masks
are identical for GLM-4.7, Kimi, and GPT-5.3.  The smaller apparent coverage
figures previously reported for those arms incorrectly used all raw turns as
the denominator, including later turns removed by right truncation.

Only GLM-4.6 is affected: 106/15,627 rendered non-empty turns (0.68%) begin with
a leading newline and are missed.  They contain 39,557/4,360,790 assistant
tokens (0.91%) and are all the first assistant turn of an affected trajectory,
distributed across SWE-Smith/SuperUser/Tezos/IssueTasks as 29/6/45/26.  This is
far from Terminal-Lego Claude's 90.8% token omission.  It does not challenge the
robust OT-Agent top/bottom result, so no full rescore is warranted.  Because the
Kimi/GLM-4.6 RSR gap is itself small, certifying that one middle pair would at
most require rescoring these 106 GLM-4.6 trajectories and merging them with the
existing rows.

The affected span helper is shared by global/local NLL, ASLEC, GRACE, and the
agent-all-assistant SCAS view.  Their previously reported Terminal-Lego rows
also used incomplete Claude masks and should be regarded as stale until they
are rescored.  SCAS's separate `official_final` view uses its own final-answer
boundary and is not the same all-turn mask.
