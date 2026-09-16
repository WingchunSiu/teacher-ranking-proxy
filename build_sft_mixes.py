#!/usr/bin/env python3
"""Build reproducible per-task multi-teacher SFT candidate mixes.

This is the materialization step described in PROXY_SPEC.md section 12.  It
does not train a model.  Every arm uses the same matched task IDs and emits one
complete trajectory per task.  Both unconstrained selections and exact
teacher-balanced controls are written so a later SFT run can distinguish a
trajectory-score effect from a change in teacher composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import compute_proxies

DEFAULT_STUDENT = "Qwen/Qwen3-8B"
DEFAULT_STUDENT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_TEACHERS = [
    "DeepSeek-V3.2",
    "GLM-5",
    "Qwen3.5-Plus",
    "Claude Opus 4.6",
]


@dataclass(frozen=True)
class Candidate:
    teacher: str
    task_id: str
    trajectory_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    rsr_b: float
    supervised_tokens: int
    tor: float | None
    tor_components: dict[str, Any]


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected TEACHER=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("teacher label cannot be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return label, path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(values: list[str]) -> str:
    payload = "\n".join(values).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalized_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        source = record["messages"]
        role_key, content_key = "role", "content"
        role_map: dict[str, str] = {}
    elif isinstance(record.get("conversations"), list):
        source = record["conversations"]
        role_key, content_key = "from", "value"
        role_map = {"human": "user", "gpt": "assistant"}
    else:
        raise TypeError("trajectory record has neither messages nor conversations")

    messages = []
    for message in source:
        if not isinstance(message, dict):
            raise TypeError("trajectory message is not an object")
        role = role_map.get(str(message.get(role_key)), message.get(role_key))
        content = message.get(content_key)
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(content, str):
            raise TypeError("trajectory message content is not a string")
        messages.append({"role": role, "content": content})
    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("trajectory has no assistant content")
    return messages


def _task_id(record: dict[str, Any]) -> str:
    value = record.get("task_id")
    if not isinstance(value, str):
        value = (record.get("metadata") or {}).get("oracle_passed_task")
    if not isinstance(value, str) or not value:
        raise ValueError("trajectory record has no task_id")
    return value


def _load_trajectories(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"{path}: expected a JSON list")
    by_task = {}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"{path}: trajectory record is not an object")
        task_id = _task_id(record)
        if task_id in by_task:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        by_task[task_id] = record
    return by_task


def _load_rsr_scores(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise SystemExit(
                "Reading parquet scores requires pyarrow; use the project "
                "environment or `uv run --with pyarrow ...`"
            ) from error
        columns = {"task_id", "rsr_b", "num_tokens_masked"}
        table = pq.read_table(path, columns=sorted(columns))
        rows = table.to_pylist()
    else:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    by_task = {}
    for row in rows:
        task_id = row.get("task_id")
        rsr_b = row.get("rsr_b", row.get("score"))
        num_tokens = row.get("num_tokens_masked", row.get("assistant_tokens_scored"))
        if not isinstance(task_id, str):
            raise TypeError(f"{path}: score row has no task_id")
        if task_id in by_task:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        if not isinstance(rsr_b, (int, float)) or not math.isfinite(rsr_b):
            raise ValueError(f"{path}: invalid RSR for {task_id}: {rsr_b!r}")
        if not isinstance(num_tokens, int) or num_tokens < 1:
            raise ValueError(
                f"{path}: invalid supervised token count for {task_id}: {num_tokens!r}"
            )
        by_task[task_id] = {
            "rsr_b": float(rsr_b),
            "supervised_tokens": num_tokens,
        }
    return by_task


def _tor(messages: list[dict[str, str]]) -> tuple[float | None, dict[str, Any]]:
    legacy = {
        "conversations": [
            {
                "from": "human" if message["role"] == "user" else "gpt",
                "value": message["content"],
            }
            for message in messages
            if message["role"] in {"user", "assistant"}
        ]
    }
    components = compute_proxies._trajectory_grounding_components(legacy)
    return components["tor"], components


def load_candidates(
    trajectory_paths: dict[str, Path], score_paths: dict[str, Path]
) -> tuple[dict[str, dict[str, Candidate]], list[str]]:
    if set(trajectory_paths) != set(score_paths):
        raise ValueError(
            "trajectory and RSR score teacher labels differ: "
            f"{sorted(trajectory_paths)} vs {sorted(score_paths)}"
        )
    teachers = [teacher for teacher in DEFAULT_TEACHERS if teacher in trajectory_paths]
    teachers.extend(sorted(set(trajectory_paths) - set(teachers)))
    if len(teachers) < 2:
        raise ValueError("at least two teachers are required")

    raw = {
        teacher: _load_trajectories(trajectory_paths[teacher]) for teacher in teachers
    }
    scores = {teacher: _load_rsr_scores(score_paths[teacher]) for teacher in teachers}
    common_tasks = sorted(
        set.intersection(
            *(set(raw[teacher]) & set(scores[teacher]) for teacher in teachers)
        )
    )
    if not common_tasks:
        raise ValueError("no task has complete trajectory and score coverage")

    candidates: dict[str, dict[str, Candidate]] = {}
    for task_id in common_tasks:
        candidates[task_id] = {}
        for teacher in teachers:
            record = raw[teacher][task_id]
            messages = _normalized_messages(record)
            tor, tor_components = _tor(messages)
            score = scores[teacher][task_id]
            original_id = record.get("trajectory_id", task_id)
            candidates[task_id][teacher] = Candidate(
                teacher=teacher,
                task_id=task_id,
                trajectory_id=f"{teacher}::{original_id}",
                messages=messages,
                metadata=dict(record.get("metadata") or {}),
                rsr_b=score["rsr_b"],
                supervised_tokens=score["supervised_tokens"],
                tor=tor,
                tor_components=tor_components,
            )
    return candidates, teachers


def _stable_tie_break(task_id: str, teacher: str) -> float:
    digest = hashlib.sha256(f"{task_id}\0{teacher}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def _best_per_task(
    candidates: dict[str, dict[str, Candidate]],
    teachers: list[str],
    utility: Callable[[Candidate], float | None],
) -> dict[str, str]:
    selected = {}
    for task_id, options in candidates.items():
        valid = []
        for teacher in teachers:
            value = utility(options[teacher])
            if value is not None and math.isfinite(value):
                valid.append((value, _stable_tie_break(task_id, teacher), teacher))
        if not valid:
            raise ValueError(f"no valid candidate utility for {task_id}")
        selected[task_id] = max(valid)[2]
    return selected


def _hungarian_min_cost(cost: list[list[float]]) -> list[int]:
    """Return the selected column for each row (square minimum assignment)."""
    n = len(cost)
    if n == 0 or any(len(row) != n for row in cost):
        raise ValueError("Hungarian assignment requires a non-empty square matrix")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if not math.isfinite(delta):
                raise ValueError("teacher-balanced assignment is infeasible")
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for column in range(1, n + 1):
        assignment[p[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise ValueError("Hungarian assignment did not cover every task")
    return assignment


def _balanced_selection(
    candidates: dict[str, dict[str, Candidate]],
    teachers: list[str],
    utility: Callable[[Candidate], float | None],
) -> dict[str, str]:
    tasks = sorted(candidates)
    quotient, remainder = divmod(len(tasks), len(teachers))
    quotas = {
        teacher: quotient + int(index < remainder)
        for index, teacher in enumerate(teachers)
    }
    slots = [teacher for teacher in teachers for _ in range(quotas[teacher])]
    missing_cost = 1e12
    cost = []
    for task_id in tasks:
        row = []
        for teacher in slots:
            value = utility(candidates[task_id][teacher])
            if value is None or not math.isfinite(value):
                row.append(missing_cost)
            else:
                tie = _stable_tie_break(task_id, teacher) * 1e-9
                row.append(-value - tie)
        cost.append(row)
    assignment = _hungarian_min_cost(cost)
    selected = {task: slots[column] for task, column in zip(tasks, assignment)}
    for task_id, teacher in selected.items():
        if utility(candidates[task_id][teacher]) is None:
            raise ValueError(
                f"balanced assignment had to select a missing score: "
                f"{task_id}/{teacher}"
            )
    if Counter(selected.values()) != Counter(quotas):
        raise ValueError("balanced assignment violated teacher quotas")
    return selected


def _random_selection(
    candidates: dict[str, dict[str, Candidate]], teachers: list[str], seed: int
) -> dict[str, str]:
    rng = random.Random(seed)
    return {task_id: rng.choice(teachers) for task_id in sorted(candidates)}


def _balanced_random_selection(
    candidates: dict[str, dict[str, Candidate]], teachers: list[str], seed: int
) -> dict[str, str]:
    rng = random.Random(seed)
    random_utility = {
        task_id: {teacher: rng.random() for teacher in teachers}
        for task_id in sorted(candidates)
    }
    return _balanced_selection(
        candidates,
        teachers,
        lambda candidate: random_utility[candidate.task_id][candidate.teacher],
    )


def _summary(values: list[float | int]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        "min": min(numbers),
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": max(numbers),
    }


def _write_arm(
    output_dir: Path,
    arm_name: str,
    proxy: str,
    direction: str,
    selection: dict[str, str],
    candidates: dict[str, dict[str, Candidate]],
) -> dict[str, Any]:
    output = output_dir / f"{arm_name}.jsonl"
    digest = hashlib.sha256()
    selected_candidates = []
    with output.open("wb") as handle:
        for task_id in sorted(selection):
            candidate = candidates[task_id][selection[task_id]]
            selected_candidates.append(candidate)
            candidate_scores = {
                teacher: {
                    "rsr_b": option.rsr_b,
                    "tor": option.tor,
                    "supervised_tokens": option.supervised_tokens,
                }
                for teacher, option in candidates[task_id].items()
            }
            selected_score = (
                candidate.rsr_b
                if proxy == "rsr_b"
                else candidate.tor
                if proxy == "tor"
                else None
            )
            payload = {
                "task_id": task_id,
                "trajectory_id": candidate.trajectory_id,
                "selected_teacher": candidate.teacher,
                "messages": candidate.messages,
                "metadata": candidate.metadata,
                "selection": {
                    "arm": arm_name,
                    "proxy": proxy,
                    "direction": direction,
                    "selected_score": selected_score,
                    "candidate_scores": candidate_scores,
                },
            }
            line = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            handle.write(line)
            digest.update(line)

    teacher_counts = Counter(candidate.teacher for candidate in selected_candidates)
    rsr_values = [candidate.rsr_b for candidate in selected_candidates]
    tor_values = [
        candidate.tor for candidate in selected_candidates if candidate.tor is not None
    ]
    return {
        "path": str(output.resolve()),
        "sha256": digest.hexdigest(),
        "rows": len(selected_candidates),
        "teacher_counts": dict(teacher_counts),
        "supervised_tokens_total": sum(
            candidate.supervised_tokens for candidate in selected_candidates
        ),
        "supervised_tokens_per_trajectory": _summary(
            [candidate.supervised_tokens for candidate in selected_candidates]
        ),
        "selected_rsr_b": _summary(rsr_values),
        "selected_tor": _summary(tor_values) if tor_values else None,
        "selected_tor_missing": len(selected_candidates) - len(tor_values),
    }


def build_mixes(
    candidates: dict[str, dict[str, Candidate]],
    teachers: list[str],
    random_seeds: list[int],
) -> dict[str, tuple[str, str, dict[str, str]]]:
    low_rsr = lambda candidate: -candidate.rsr_b
    high_rsr = lambda candidate: candidate.rsr_b
    high_tor = lambda candidate: candidate.tor
    arms = {
        "rsr_low": (
            "rsr_b",
            "lower_better",
            _best_per_task(candidates, teachers, low_rsr),
        ),
        "rsr_high": (
            "rsr_b",
            "higher_better",
            _best_per_task(candidates, teachers, high_rsr),
        ),
        "tor_high": (
            "tor",
            "higher_better",
            _best_per_task(candidates, teachers, high_tor),
        ),
        "rsr_low_teacher_balanced": (
            "rsr_b",
            "lower_better",
            _balanced_selection(candidates, teachers, low_rsr),
        ),
        "rsr_high_teacher_balanced": (
            "rsr_b",
            "higher_better",
            _balanced_selection(candidates, teachers, high_rsr),
        ),
        "tor_high_teacher_balanced": (
            "tor",
            "higher_better",
            _balanced_selection(candidates, teachers, high_tor),
        ),
    }
    if "DeepSeek-V3.2" in teachers:
        arms["deepseek_all"] = (
            "published_global_teacher",
            "reference",
            {task_id: "DeepSeek-V3.2" for task_id in candidates},
        )
    for seed in random_seeds:
        arms[f"random_s{seed}"] = (
            "random",
            "reference",
            _random_selection(candidates, teachers, seed),
        )
        arms[f"random_teacher_balanced_s{seed}"] = (
            "random",
            "reference",
            _balanced_random_selection(candidates, teachers, seed),
        )
    return arms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        action="append",
        type=_parse_labeled_path,
        required=True,
        metavar="TEACHER=PATH",
    )
    parser.add_argument(
        "--rsr-score",
        action="append",
        type=_parse_labeled_path,
        required=True,
        metavar="TEACHER=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expected-tasks", type=int, default=None)
    parser.add_argument("--random-seeds", default="42,43,44")
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--student-revision", default=DEFAULT_STUDENT_REVISION)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trajectory_paths = dict(args.trajectory)
    score_paths = dict(args.rsr_score)
    if len(trajectory_paths) != len(args.trajectory):
        raise SystemExit("duplicate teacher label in --trajectory")
    if len(score_paths) != len(args.rsr_score):
        raise SystemExit("duplicate teacher label in --rsr-score")
    random_seeds = [
        int(value) for value in args.random_seeds.split(",") if value.strip()
    ]
    if not random_seeds:
        raise SystemExit("--random-seeds cannot be empty")

    candidates, teachers = load_candidates(trajectory_paths, score_paths)
    if args.expected_tasks is not None and len(candidates) != args.expected_tasks:
        raise SystemExit(
            f"expected {args.expected_tasks} complete tasks, found {len(candidates)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = build_mixes(candidates, teachers, random_seeds)
    arm_reports = {
        name: _write_arm(args.output_dir, name, proxy, direction, selection, candidates)
        for name, (proxy, direction, selection) in arms.items()
    }
    task_ids = sorted(candidates)
    report = {
        "kind": "terminal_lego_proxy_selected_sft_mix",
        "schema_version": 1,
        "student": {"model": args.student, "revision": args.student_revision},
        "n_matched_tasks": len(task_ids),
        "task_ids_sha256": _sha256_text(task_ids),
        "teachers": teachers,
        "selection_unit": "one complete teacher trajectory per matched task",
        "rsr_direction": "lower_better",
        "tor_definition": (
            "local paper-formula operationalization in compute_proxies.py; "
            "higher is better; upstream implementation is unreleased"
        ),
        "random_seeds": random_seeds,
        "inputs": {
            "trajectories": {
                teacher: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for teacher, path in trajectory_paths.items()
            },
            "rsr_scores": {
                teacher: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for teacher, path in score_paths.items()
            },
        },
        "arms": arm_reports,
        "training_warning": (
            "The files contain complete matched-task trajectories, but raw "
            "supervised-token totals differ. Match optimizer-token budget in "
            "the trainer before interpreting an SFT comparison."
        ),
    }
    manifest_path = args.output_dir / "selection_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "n_matched_tasks": len(task_ids),
                "arms": {
                    name: {
                        "teacher_counts": arm["teacher_counts"],
                        "supervised_tokens_total": arm["supervised_tokens_total"],
                    }
                    for name, arm in arm_reports.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
