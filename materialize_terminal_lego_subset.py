#!/usr/bin/env python3
"""Materialize a response-independent matched Terminal-Lego trajectory subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open() as handle:
        header = json.loads(handle.readline())
        rows = [json.loads(line) for line in handle if line.strip()]
    if header.get("kind") != "task_manifest_header":
        raise ValueError("manifest first row is not a task_manifest_header")
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("manifest contains duplicate task IDs")
    return header, rows


def _normalize_messages(record: dict[str, Any], *, teacher: str, task_id: str):
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{teacher}/{task_id}: missing conversations")
    messages = []
    expected = "human"
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise TypeError(f"{teacher}/{task_id}: turn {index} is not an object")
        role = message.get("from")
        content = message.get("value")
        if role != expected or not isinstance(content, str):
            raise ValueError(
                f"{teacher}/{task_id}: turn {index} expected {expected!r}, "
                f"got {role!r}"
            )
        messages.append(
            {"role": "user" if role == "human" else "assistant", "content": content}
        )
        expected = "gpt" if expected == "human" else "human"
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{teacher}/{task_id}: trajectory does not end in assistant")
    return messages


def _task_description(initial_user_message: str) -> str:
    """Strip run-specific terminal banners while preserving the assigned task."""
    marker = "Task Description:\n"
    terminal_marker = "\nCurrent terminal state:"
    if marker not in initial_user_message or terminal_marker not in initial_user_message:
        return initial_user_message
    return initial_user_message.split(marker, 1)[1].split(terminal_marker, 1)[0].strip()


def materialize(
    *,
    manifest_path: Path,
    trajectory_dir: Path,
    output_dir: Path,
    report_path: Path,
    n_tasks: int,
    seed: int,
) -> dict[str, Any]:
    header, manifest_rows = _load_manifest(manifest_path)
    if n_tasks < 1 or n_tasks > len(manifest_rows):
        raise ValueError(f"n_tasks must be in [1, {len(manifest_rows)}]")
    manifest_by_task = {str(row["task_id"]): row for row in manifest_rows}
    sampled_task_ids = random.Random(seed).sample(sorted(manifest_by_task), n_tasks)
    selected_task_ids = sorted(sampled_task_ids)
    teachers = [str(teacher) for teacher in header["teachers"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    task_file = output_dir / "sampled_task_ids.json"
    task_file.write_text(
        json.dumps(
            {
                "draw_order": sampled_task_ids,
                "manifest_sha256": _sha256_file(manifest_path),
                "n_tasks": n_tasks,
                "seed": seed,
                "task_ids_sha256": _sha256_lines(selected_task_ids),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    input_reports: dict[str, Any] = {}
    output_reports: dict[str, Any] = {}
    difficulty_by_task: dict[str, str] = {}
    missing_difficulty_by_teacher: Counter[str] = Counter()
    initial_user_by_task: dict[str, dict[str, str]] = {
        task_id: {} for task_id in selected_task_ids
    }
    task_description_by_task: dict[str, dict[str, str]] = {
        task_id: {} for task_id in selected_task_ids
    }
    metadata_fields: set[str] = set()
    for teacher in teachers:
        source_path = trajectory_dir / Path(header["teacher_files"][teacher]).name
        records = json.loads(source_path.read_text())
        if not isinstance(records, list):
            raise TypeError(f"{source_path}: expected a JSON list")
        selected = []
        for task_id in selected_task_ids:
            record_index = int(
                manifest_by_task[task_id]["teacher_record_index"][teacher]
            )
            record = records[record_index]
            metadata = dict(record.get("metadata") or {})
            metadata_fields.update(metadata)
            if metadata.get("oracle_passed_task") != task_id:
                raise ValueError(
                    f"{teacher}/{task_id}: manifest index resolves to "
                    f"{metadata.get('oracle_passed_task')!r}"
                )
            raw_difficulty = metadata.get("difficulty")
            difficulty = (
                raw_difficulty.strip()
                if isinstance(raw_difficulty, str) and raw_difficulty.strip()
                else None
            )
            if difficulty is None:
                missing_difficulty_by_teacher[teacher] += 1
            elif task_id in difficulty_by_task and difficulty_by_task[task_id] != difficulty:
                raise ValueError(
                    f"{task_id}: non-empty difficulty differs across teachers: "
                    f"{difficulty_by_task[task_id]!r} vs {difficulty!r}"
                )
            elif difficulty is not None:
                difficulty_by_task[task_id] = difficulty
            messages = _normalize_messages(record, teacher=teacher, task_id=task_id)
            initial_user = messages[0]["content"]
            initial_user_by_task[task_id][teacher] = initial_user
            task_description_by_task[task_id][teacher] = _task_description(initial_user)
            selected.append(
                {
                    "trajectory_id": f"{teacher}::{task_id}",
                    "task_id": task_id,
                    "teacher": teacher,
                    "messages": messages,
                    "metadata": metadata,
                }
            )

        output_path = output_dir / f"{_slug(teacher)}.json"
        output_path.write_text(json.dumps(selected, ensure_ascii=False) + "\n")
        input_reports[teacher] = {
            "path": str(source_path.resolve()),
            "records": len(records),
            "sha256": _sha256_file(source_path),
        }
        output_reports[teacher] = {
            "path": str(output_path.resolve()),
            "rows": len(selected),
            "sha256": _sha256_file(output_path),
        }

    full_prompt_matches = sum(
        len(set(per_teacher.values())) == 1
        for per_teacher in initial_user_by_task.values()
    )
    task_description_matches = sum(
        len(set(per_teacher.values())) == 1
        for per_teacher in task_description_by_task.values()
    )
    expected_environment_fields = {
        "environment_image",
        "harness_revision",
        "repo_commit",
        "task_config",
        "verifier_revision",
    }
    report = {
        "kind": "terminal_lego_matched_trajectory_subset",
        "dataset": header["traj_repo"],
        "dataset_revision": header["traj_revision"],
        "canonical_task_repo": header.get("task_repo"),
        "canonical_task_revision": header.get("task_revision"),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256_file(manifest_path),
            "matched_tasks": len(manifest_rows),
        },
        "n_tasks": n_tasks,
        "seed": seed,
        "sampling_policy": "uniform without replacement from sorted matched task IDs before reading responses",
        "task_ids_sha256": _sha256_lines(selected_task_ids),
        "sample_file": {
            "path": str(task_file.resolve()),
            "sha256": _sha256_file(task_file),
        },
        "teachers": teachers,
        "candidates_per_task": len(teachers),
        "selection_unit": "later arms must select exactly one complete trajectory per task",
        "source_outcome_semantics": (
            "every released row is keyed by metadata.oracle_passed_task; the "
            "release contains verifier-passing teacher trajectories"
        ),
        "task_input_alignment": {
            "task_description_exact_across_teachers": task_description_matches,
            "initial_user_message_exact_across_teachers": full_prompt_matches,
            "total_tasks": n_tasks,
            "interpretation": (
                "Task descriptions are compared after removing the run-specific "
                "Current terminal state block; full prompts can differ by container "
                "hostname and shell banner."
            ),
        },
        "environment_provenance": {
            "metadata_fields_present": sorted(metadata_fields),
            "expected_revision_fields_missing": sorted(
                expected_environment_fields - metadata_fields
            ),
            "independently_verifiable_from_release": not bool(
                expected_environment_fields - metadata_fields
            ),
        },
        "difficulty_counts": dict(
            Counter(difficulty_by_task.get(task_id, "unknown") for task_id in selected_task_ids)
        ),
        "missing_difficulty_by_teacher": dict(missing_difficulty_by_teacher),
        "inputs": input_reports,
        "outputs": output_reports,
        "training_status": "not_approved",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--n-tasks", type=int, default=1700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = materialize(
        manifest_path=args.manifest,
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        n_tasks=args.n_tasks,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
