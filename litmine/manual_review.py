#!/usr/bin/env python3
"""
Manual review results based on careful reading of each paper.
Verdict format: pass|fail_scan|fail_ablation|fail_ranking|fail_retrieval
"""
import json

results = [
    {
        "key": "arxiv:2502.13092",
        "verdict": "fail_scan",
        "evidence": "Table 2 shows fine-tuning/agent training on single student models, not multi-teacher ablation. No per-teacher SFT results."
    },
    {
        "key": "arxiv:2502.13311",
        "verdict": "fail_ablation",
        "evidence": "Table 1 evaluates different tutor agents (using different models) but not SFT-separate-per-teacher on same student. It's tutor agent comparison, not student SFT ablation."
    },
    {
        "key": "arxiv:2502.13923",
        "verdict": "fail_scan",
        "evidence": "Mentions Qwen2.5-VL capabilities; no evidence of multi-teacher SFT ablation on same student."
    },
    {
        "key": "arxiv:2502.14276",
        "verdict": "fail_ablation",
        "evidence": "STeCa method trains Llama-2-7B with calibration; ablation compares (SFT+DPO vs w/o RT) but not multi-teacher ablation."
    },
    {
        "key": "arxiv:2502.14496",
        "verdict": "fail_scan",
        "evidence": "CollabUIAgents with MARL; mentions multiple models but no same-student multi-teacher SFT pattern found."
    },
    {
        "key": "arxiv:2502.18407",
        "verdict": "fail_scan",
        "evidence": "Reward model training on states from LLaMA-3-8B; no evidence of same-student ablation across different teachers."
    },
    {
        "key": "arxiv:2502.18934",
        "verdict": "fail_scan",
        "evidence": "Fine-tuning strategy on high-quality data per category; no multi-teacher ablation on same student model."
    },
    {
        "key": "arxiv:2503.00401",
        "verdict": "fail_scan",
        "evidence": "No clear pattern of multi-teacher SFT ablation found in scan."
    },
    {
        "key": "arxiv:2503.01490",
        "verdict": "fail_scan",
        "evidence": "Experiments on three tasks (HotpotQA, ALFWorld, etc.); no per-teacher SFT ablation of same student."
    },
    {
        "key": "arxiv:2503.01763",
        "verdict": "fail_scan",
        "evidence": "Tool-use LLMs; no evidence of multi-teacher trajectory generation with same-student SFT ablation."
    },
    {
        "key": "arxiv:2503.01940",
        "verdict": "fail_scan",
        "evidence": "Multiple models mentioned but no same-student multi-teacher SFT pattern identified."
    },
    {
        "key": "arxiv:2503.02197",
        "verdict": "pass",
        "evidence": "Student: LLaMA-3.1-8B-Instruct. Teachers: GPT-4o vs Llama3.1-70B (selector models generating critical steps). Table 5 shows per-teacher downstream scores (Alfworld 83.00 vs 78.50, BabyAI 78.93 vs 67.23, Weather 60.00 vs 55.00). Same student fine-tuned separately on teacher-selected critical steps."
    },
    {
        "key": "arxiv:2503.02878",
        "verdict": "fail_scan",
        "evidence": "Value function approach for lookahead; no multi-teacher ablation pattern found."
    },
    {
        "key": "arxiv:2503.05143",
        "verdict": "fail_scan",
        "evidence": "Model comparisons but not same-student multi-teacher SFT ablation."
    },
    {
        "key": "arxiv:2503.06580",
        "verdict": "fail_scan",
        "evidence": "Introduction section only substantive content; no evidence of multi-teacher SFT ablation."
    },
]

output_file = "work/state/recovery_agents/recout_008.jsonl"

with open(output_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

print(f"Written {len(results)} results to {output_file}")
for r in results:
    print(f"{r['key']}: {r['verdict']}")

# Verify count
pass_count = sum(1 for r in results if r['verdict'] == 'pass')
fail_count = len(results) - pass_count
print(f"\nSummary: {pass_count} pass, {fail_count} fail")
