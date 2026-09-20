from __future__ import annotations

import json
from pathlib import Path

import audit_eval_overlap as overlap


def test_overlap_audit_finds_normalized_exact_match(tmp_path: Path):
    train = tmp_path / "train.json"
    train.write_text(
        json.dumps(
            [
                {
                    "task_id": "train-1",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Task Description:\nCreate THE file.\nCurrent terminal state:\nhost-1",
                        },
                        {"role": "assistant", "content": "done"},
                    ],
                }
            ]
        )
    )
    eval_task = tmp_path / "eval" / "eval-1"
    eval_task.mkdir(parents=True)
    (eval_task / "instruction.md").write_text("create the FILE!")

    report = overlap.audit(
        train,
        tmp_path / "eval",
        eval_repo="example/eval",
        eval_revision="abc123",
    )

    assert report["exact_normalized_match_count"] == 1
    assert report["five_token_shingle_jaccard"]["pairs_at_least_0_8"] == 1
