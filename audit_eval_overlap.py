#!/usr/bin/env python3
"""Audit exact and lexical overlap between SFT tasks and eval instructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _task_description(initial_user_message: str) -> str:
    marker = "Task Description:\n"
    terminal_marker = "\nCurrent terminal state:"
    if marker not in initial_user_message or terminal_marker not in initial_user_message:
        return initial_user_message
    return initial_user_message.split(marker, 1)[1].split(terminal_marker, 1)[0]


def _shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = normalize(text).split()
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def audit(
    train_path: Path,
    eval_root: Path,
    *,
    eval_repo: str,
    eval_revision: str,
) -> dict[str, Any]:
    records = json.loads(train_path.read_text())
    train = {}
    for record in records:
        task_id = str(record["task_id"])
        initial_user = next(
            message["content"]
            for message in record["messages"]
            if message["role"] == "user"
        )
        train[task_id] = _task_description(initial_user)
    eval_paths = sorted(eval_root.rglob("instruction.md"))
    evaluation = {
        str(path.parent.relative_to(eval_root)): path.read_text() for path in eval_paths
    }
    if not train or not evaluation:
        raise ValueError("both training and evaluation instruction sets must be non-empty")

    normalized_train = {task_id: normalize(text) for task_id, text in train.items()}
    normalized_eval = {task_id: normalize(text) for task_id, text in evaluation.items()}
    eval_by_text: dict[str, list[str]] = {}
    for task_id, text in normalized_eval.items():
        eval_by_text.setdefault(text, []).append(task_id)
    exact_matches = [
        {"train_task_id": task_id, "eval_task_ids": eval_by_text[text]}
        for task_id, text in normalized_train.items()
        if text in eval_by_text
    ]

    train_shingles = {task_id: _shingles(text) for task_id, text in train.items()}
    eval_shingles = {task_id: _shingles(text) for task_id, text in evaluation.items()}
    similarities = sorted(
        (
            {
                "train_task_id": train_id,
                "eval_task_id": eval_id,
                "five_token_shingle_jaccard": _jaccard(train_value, eval_value),
            }
            for train_id, train_value in train_shingles.items()
            for eval_id, eval_value in eval_shingles.items()
        ),
        key=lambda row: (
            -row["five_token_shingle_jaccard"],
            row["train_task_id"],
            row["eval_task_id"],
        ),
    )
    return {
        "kind": "train_eval_instruction_overlap_audit",
        "train": {
            "path": str(train_path.resolve()),
            "sha256": _sha256_file(train_path),
            "tasks": len(train),
        },
        "eval": {
            "root": str(eval_root.resolve()),
            "repo": eval_repo,
            "revision": eval_revision,
            "tasks": len(evaluation),
            "instruction_manifest_sha256": hashlib.sha256(
                "\n".join(
                    f"{task_id}\t{_sha256_file(eval_root / task_id / 'instruction.md')}"
                    for task_id in sorted(evaluation)
                ).encode()
            ).hexdigest(),
        },
        "normalization": "NFKC, lowercase, retain ASCII alphanumerics, collapse whitespace",
        "exact_normalized_matches": exact_matches,
        "exact_normalized_match_count": len(exact_matches),
        "five_token_shingle_jaccard": {
            "pairs_at_least_0_5": sum(
                row["five_token_shingle_jaccard"] >= 0.5 for row in similarities
            ),
            "pairs_at_least_0_8": sum(
                row["five_token_shingle_jaccard"] >= 0.8 for row in similarities
            ),
            "top_pairs": similarities[:20],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--eval-repo", required=True)
    parser.add_argument("--eval-revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.train,
        args.eval_root,
        eval_repo=args.eval_repo,
        eval_revision=args.eval_revision,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
