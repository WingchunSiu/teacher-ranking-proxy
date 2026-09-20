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
REQUIRED_SCORE_SEMANTICS = "official_chat_v2_boundary_safe"


@dataclass(frozen=True)
class Candidate:
    teacher: str
    task_id: str
    trajectory_id: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    rsr_b: float
    mean_nll_masked: float
    mean_clipped_rank_masked: float
    supervised_tokens: int
    sft_total_tokens: int
    sft_turns_truncated: int
    proxy_assistant_tokens: int
    tor: float | None
    tor_components: dict[str, Any]
    turn_audit: dict[str, int]


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
        columns = {
            "trajectory_id",
            "task_id",
            "scoring_semantics",
            "rsr_b",
            "mean_nll_masked",
            "mean_clipped_rank_masked",
            "num_tokens_masked",
        }
        table = pq.read_table(path, columns=sorted(columns))
        rows = table.to_pylist()
    else:
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    by_task = {}
    for row in rows:
        task_id = row.get("task_id")
        trajectory_id = row.get("trajectory_id")
        scoring_semantics = row.get("scoring_semantics")
        rsr_b = row.get("rsr_b", row.get("score"))
        mean_nll = row.get("mean_nll_masked")
        mean_clipped_rank = row.get("mean_clipped_rank_masked")
        num_tokens = row.get("num_tokens_masked", row.get("assistant_tokens_scored"))
        if not isinstance(task_id, str):
            raise TypeError(f"{path}: score row has no task_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise TypeError(f"{path}: score row has no trajectory_id")
        if scoring_semantics != REQUIRED_SCORE_SEMANTICS:
            raise ValueError(
                f"{path}: {task_id} uses {scoring_semantics!r}; expected "
                f"{REQUIRED_SCORE_SEMANTICS!r}"
            )
        if task_id in by_task:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        if not isinstance(rsr_b, (int, float)) or not math.isfinite(rsr_b):
            raise ValueError(f"{path}: invalid RSR for {task_id}: {rsr_b!r}")
        if not isinstance(mean_nll, (int, float)) or not math.isfinite(mean_nll):
            raise ValueError(f"{path}: invalid mean NLL for {task_id}: {mean_nll!r}")
        if not isinstance(mean_clipped_rank, (int, float)) or not math.isfinite(
            mean_clipped_rank
        ):
            raise ValueError(
                f"{path}: invalid mean clipped rank for {task_id}: "
                f"{mean_clipped_rank!r}"
            )
        if not isinstance(num_tokens, int) or num_tokens < 1:
            raise ValueError(
                f"{path}: invalid supervised token count for {task_id}: {num_tokens!r}"
            )
        by_task[task_id] = {
            "trajectory_id": trajectory_id,
            "rsr_b": float(rsr_b),
            "mean_nll_masked": float(mean_nll),
            "mean_clipped_rank_masked": float(mean_clipped_rank),
            "proxy_assistant_tokens": num_tokens,
        }
    return by_task


def _load_sft_tokens(path: Path) -> dict[str, dict[str, int | str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit(
            "Reading SFT token audits requires pyarrow; use the project environment"
        ) from error
    table = pq.read_table(
        path,
        columns=[
            "trajectory_id",
            "task_id",
            "sft_total_tokens",
            "sft_trainable_tokens",
            "sft_turns_truncated",
        ],
    )
    by_task = {}
    for row in table.to_pylist():
        task_id = row.get("task_id")
        trajectory_id = row.get("trajectory_id")
        total_tokens = row.get("sft_total_tokens")
        truncated_turns = row.get("sft_turns_truncated")
        tokens = row.get("sft_trainable_tokens")
        if (
            not isinstance(task_id, str)
            or not isinstance(trajectory_id, str)
            or not trajectory_id
            or not isinstance(total_tokens, int)
            or total_tokens < 1
            or not isinstance(truncated_turns, int)
            or truncated_turns < 0
            or not isinstance(tokens, int)
            or tokens < 1
        ):
            raise ValueError(f"{path}: invalid SFT token row: {row!r}")
        if task_id in by_task:
            raise ValueError(f"{path}: duplicate task_id {task_id}")
        by_task[task_id] = {
            "trajectory_id": trajectory_id,
            "sft_total_tokens": total_tokens,
            "sft_trainable_tokens": tokens,
            "sft_turns_truncated": truncated_turns,
        }
    return by_task


def _load_sft_audit_provenance(
    paths: dict[str, Path], expected_model_revision: str
) -> dict[str, Any]:
    required = {
        "cutoff_len",
        "llamafactory_revision",
        "model_revision",
        "semantics",
        "template",
    }
    per_teacher = {}
    shared_values: dict[str, set[str]] = {field: set() for field in required}
    for teacher, path in paths.items():
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"missing SFT token-audit sidecar: {sidecar}")
        payload = json.loads(sidecar.read_text())
        missing = required - set(payload)
        if missing:
            raise ValueError(f"{sidecar}: missing provenance fields {sorted(missing)}")
        per_teacher[teacher] = {
            "path": str(sidecar.resolve()),
            "sha256": _sha256_file(sidecar),
        }
        for field in required:
            shared_values[field].add(json.dumps(payload[field], sort_keys=True))
    inconsistent = {
        field: sorted(values)
        for field, values in shared_values.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"SFT token-audit provenance differs: {inconsistent}")
    shared = {
        field: json.loads(next(iter(values)))
        for field, values in shared_values.items()
    }
    if shared["model_revision"] != expected_model_revision:
        raise ValueError(
            "SFT token-audit model revision differs from selected student: "
            f"{shared['model_revision']} vs {expected_model_revision}"
        )
    return {"shared": shared, "sidecars": per_teacher}


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


def _turn_audit(messages: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter()
    for message in messages:
        if message["role"] != "assistant":
            continue
        counts["assistant_turns"] += 1
        parsed = compute_proxies.parse_gpt_turn(message["content"])
        if parsed is None:
            counts["unparseable_turns"] += 1
            continue
        counts["parseable_turns"] += 1
        commands = compute_proxies._turn_commands(parsed)
        if not commands:
            counts["empty_command_turns"] += 1
        if not commands and not parsed.get("task_complete"):
            counts["noncomplete_noop_turns"] += 1
        if parsed.get("task_complete"):
            counts["task_complete_turns"] += 1
    return dict(counts)


def load_candidates(
    trajectory_paths: dict[str, Path],
    score_paths: dict[str, Path],
    sft_token_paths: dict[str, Path] | None = None,
) -> tuple[dict[str, dict[str, Candidate]], list[str], dict[str, Any]]:
    if set(trajectory_paths) != set(score_paths):
        raise ValueError(
            "trajectory and RSR score teacher labels differ: "
            f"{sorted(trajectory_paths)} vs {sorted(score_paths)}"
        )
    if sft_token_paths is not None and set(trajectory_paths) != set(sft_token_paths):
        raise ValueError(
            "trajectory and SFT-token teacher labels differ: "
            f"{sorted(trajectory_paths)} vs {sorted(sft_token_paths)}"
        )
    teachers = [teacher for teacher in DEFAULT_TEACHERS if teacher in trajectory_paths]
    teachers.extend(sorted(set(trajectory_paths) - set(teachers)))
    if len(teachers) < 2:
        raise ValueError("at least two teachers are required")

    raw = {
        teacher: _load_trajectories(trajectory_paths[teacher]) for teacher in teachers
    }
    scores = {teacher: _load_rsr_scores(score_paths[teacher]) for teacher in teachers}
    sft_tokens = (
        {
            teacher: _load_sft_tokens(sft_token_paths[teacher])
            for teacher in teachers
        }
        if sft_token_paths is not None
        else None
    )
    coverage = [
        set(raw[teacher])
        & set(scores[teacher])
        & (set(sft_tokens[teacher]) if sft_tokens is not None else set(raw[teacher]))
        for teacher in teachers
    ]
    covered_tasks = sorted(set.intersection(*coverage))
    if not covered_tasks:
        raise ValueError("no task has complete trajectory and score coverage")
    incomplete_tasks = (
        [
            task_id
            for task_id in covered_tasks
            if any(
                sft_tokens[teacher][task_id]["sft_turns_truncated"] > 0
                for teacher in teachers
            )
        ]
        if sft_tokens is not None
        else []
    )
    common_tasks = sorted(set(covered_tasks) - set(incomplete_tasks))
    if not common_tasks:
        raise ValueError("no task has four complete SFT trajectories")

    candidates: dict[str, dict[str, Candidate]] = {}
    for task_id in common_tasks:
        candidates[task_id] = {}
        for teacher in teachers:
            record = raw[teacher][task_id]
            messages = _normalized_messages(record)
            tor, tor_components = _tor(messages)
            score = scores[teacher][task_id]
            raw_trajectory_id = record.get("trajectory_id")
            if not isinstance(raw_trajectory_id, str) or not raw_trajectory_id:
                raw_trajectory_id = f"{teacher}::{task_id}"
            if score["trajectory_id"] != raw_trajectory_id:
                raise ValueError(
                    f"trajectory/score mismatch for {task_id}/{teacher}: "
                    f"{raw_trajectory_id!r} vs {score['trajectory_id']!r}"
                )
            if (
                sft_tokens is not None
                and sft_tokens[teacher][task_id]["trajectory_id"] != raw_trajectory_id
            ):
                raise ValueError(
                    f"trajectory/token-audit mismatch for {task_id}/{teacher}: "
                    f"{raw_trajectory_id!r} vs "
                    f"{sft_tokens[teacher][task_id]['trajectory_id']!r}"
                )
            supervised_tokens = (
                sft_tokens[teacher][task_id]["sft_trainable_tokens"]
                if sft_tokens is not None
                else score["proxy_assistant_tokens"]
            )
            sft_total_tokens = (
                sft_tokens[teacher][task_id]["sft_total_tokens"]
                if sft_tokens is not None
                else score["proxy_assistant_tokens"]
            )
            sft_turns_truncated = (
                sft_tokens[teacher][task_id]["sft_turns_truncated"]
                if sft_tokens is not None
                else 0
            )
            candidates[task_id][teacher] = Candidate(
                teacher=teacher,
                task_id=task_id,
                trajectory_id=raw_trajectory_id,
                messages=messages,
                metadata=dict(record.get("metadata") or {}),
                rsr_b=score["rsr_b"],
                mean_nll_masked=score["mean_nll_masked"],
                mean_clipped_rank_masked=score["mean_clipped_rank_masked"],
                supervised_tokens=supervised_tokens,
                sft_total_tokens=sft_total_tokens,
                sft_turns_truncated=sft_turns_truncated,
                proxy_assistant_tokens=score["proxy_assistant_tokens"],
                tor=tor,
                tor_components=tor_components,
                turn_audit=_turn_audit(messages),
            )
    return candidates, teachers, {
        "covered_tasks_before_completeness_filter": len(covered_tasks),
        "dropped_tasks_with_any_truncated_teacher_trajectory": incomplete_tasks,
        "complete_four_way_tasks": len(common_tasks),
        "policy": (
            "drop a task from every arm if any teacher trajectory truncates an "
            "assistant turn under the audited SFT template/cutoff"
        ),
    }


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


def _balanced_selection(
    candidates: dict[str, dict[str, Candidate]],
    teachers: list[str],
    utility: Callable[[Candidate], float | None],
) -> dict[str, str]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:
        raise SystemExit(
            "Teacher-balanced selections require scipy and numpy; use the "
            "project scoring environment."
        ) from error

    tasks = sorted(candidates)
    quotient, remainder = divmod(len(tasks), len(teachers))
    quotas = {
        teacher: quotient + int(index < remainder)
        for index, teacher in enumerate(teachers)
    }
    variables = [(task_id, teacher) for task_id in tasks for teacher in teachers]
    variable_index = {key: index for index, key in enumerate(variables)}
    constraint = lil_matrix((len(tasks) + len(teachers), len(variables)))
    lower = np.ones(len(tasks) + len(teachers), dtype=float)
    upper = np.ones(len(tasks) + len(teachers), dtype=float)
    for row, task_id in enumerate(tasks):
        for teacher in teachers:
            constraint[row, variable_index[(task_id, teacher)]] = 1
    for teacher_index, teacher in enumerate(teachers):
        row = len(tasks) + teacher_index
        for task_id in tasks:
            constraint[row, variable_index[(task_id, teacher)]] = 1
        lower[row] = upper[row] = quotas[teacher]
    objective = []
    variable_upper = np.ones(len(variables), dtype=float)
    for index, (task_id, teacher) in enumerate(variables):
        value = utility(candidates[task_id][teacher])
        if value is None or not math.isfinite(value):
            variable_upper[index] = 0
            objective.append(0.0)
        else:
            tie = _stable_tie_break(task_id, teacher) * 1e-9
            objective.append(-value - tie)
    result = milp(
        np.asarray(objective, dtype=float),
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(np.zeros(len(variables)), variable_upper),
        constraints=LinearConstraint(constraint.tocsr(), lower, upper),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise ValueError(f"teacher-balanced assignment failed: {result.message}")
    selected = {}
    for task_id in tasks:
        chosen = [
            teacher
            for teacher in teachers
            if result.x[variable_index[(task_id, teacher)]] > 0.5
        ]
        if len(chosen) != 1:
            raise ValueError(
                f"teacher-balanced assignment chose {len(chosen)} for {task_id}"
            )
        selected[task_id] = chosen[0]
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


def _balanced_token_matched_selection(
    candidates: dict[str, dict[str, Candidate]],
    teachers: list[str],
    target: dict[str, str],
    seed: int,
    *,
    disjoint: bool = True,
    utility: Callable[[Candidate], float] | None = None,
    sequence_token_tolerance_fraction: float | None = None,
    sequence_square_tolerance_fraction: float | None = None,
) -> dict[str, str]:
    """Seeded control with exact teacher quotas and target token total.

    This is deliberately a control rather than another proxy.  It preserves one
    trajectory per task, uses the same teacher counts as ``_balanced_selection``,
    and matches the target arm's total number of supervised assistant tokens.
    By default it cannot reuse the target trajectory for any task, so an SFT
    comparison is not diluted by identical examples in both arms. Optional
    bounds on total sequence tokens and squared sequence lengths control the
    linear-token and attention-length components of training compute.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as error:
        raise SystemExit(
            "Exact token-matched controls require scipy and numpy; use the "
            "project scoring environment."
        ) from error

    tasks = sorted(candidates)
    if set(target) != set(tasks):
        raise ValueError("token-match target must select every matched task")
    quotient, remainder = divmod(len(tasks), len(teachers))
    quotas = {
        teacher: quotient + int(index < remainder)
        for index, teacher in enumerate(teachers)
    }
    variables = [
        (task_id, teacher) for task_id in tasks for teacher in teachers
    ]
    variable_index = {key: index for index, key in enumerate(variables)}
    target_tokens = sum(
        candidates[task_id][teacher].supervised_tokens
        for task_id, teacher in target.items()
    )
    target_sequence_tokens = sum(
        candidates[task_id][teacher].sft_total_tokens
        for task_id, teacher in target.items()
    )
    target_sequence_squares = sum(
        candidates[task_id][teacher].sft_total_tokens**2
        for task_id, teacher in target.items()
    )

    for name, tolerance in (
        ("sequence_token_tolerance_fraction", sequence_token_tolerance_fraction),
        ("sequence_square_tolerance_fraction", sequence_square_tolerance_fraction),
    ):
        if tolerance is not None and not 0 <= tolerance < 1:
            raise ValueError(f"{name} must be in [0, 1)")

    # One equality per task, one per teacher, one for the exact target-token
    # count, and optional bands for sequence-token compute.
    extra_rows = 1
    extra_rows += int(sequence_token_tolerance_fraction is not None)
    extra_rows += int(sequence_square_tolerance_fraction is not None)
    constraint = lil_matrix(
        (len(tasks) + len(teachers) + extra_rows, len(variables))
    )
    lower: list[float] = []
    upper: list[float] = []
    for row, task_id in enumerate(tasks):
        for teacher in teachers:
            constraint[row, variable_index[(task_id, teacher)]] = 1
        lower.append(1)
        upper.append(1)
    teacher_offset = len(tasks)
    for index, teacher in enumerate(teachers):
        for task_id in tasks:
            constraint[
                teacher_offset + index, variable_index[(task_id, teacher)]
            ] = 1
        lower.append(quotas[teacher])
        upper.append(quotas[teacher])
    token_row = len(tasks) + len(teachers)
    for task_id, teacher in variables:
        constraint[token_row, variable_index[(task_id, teacher)]] = candidates[
            task_id
        ][teacher].supervised_tokens
    lower.append(target_tokens)
    upper.append(target_tokens)
    next_row = token_row + 1
    if sequence_token_tolerance_fraction is not None:
        for task_id, teacher in variables:
            constraint[next_row, variable_index[(task_id, teacher)]] = candidates[
                task_id
            ][teacher].sft_total_tokens
        lower.append(
            math.ceil(
                target_sequence_tokens * (1 - sequence_token_tolerance_fraction)
            )
        )
        upper.append(
            math.floor(
                target_sequence_tokens * (1 + sequence_token_tolerance_fraction)
            )
        )
        next_row += 1
    if sequence_square_tolerance_fraction is not None:
        for task_id, teacher in variables:
            constraint[next_row, variable_index[(task_id, teacher)]] = candidates[
                task_id
            ][teacher].sft_total_tokens**2
        lower.append(
            math.ceil(
                target_sequence_squares
                * (1 - sequence_square_tolerance_fraction)
            )
        )
        upper.append(
            math.floor(
                target_sequence_squares
                * (1 + sequence_square_tolerance_fraction)
            )
        )

    rng = random.Random(seed)
    objective = np.asarray(
        [
            (
                -utility(candidates[task_id][teacher])
                if utility is not None
                else 0.0
            )
            + rng.random() * (1e-9 if utility is not None else 1.0)
            for task_id, teacher in variables
        ],
        dtype=float,
    )
    variable_upper = np.ones(len(variables), dtype=float)
    if disjoint:
        for task_id, teacher in target.items():
            variable_upper[variable_index[(task_id, teacher)]] = 0
    result = milp(
        objective,
        integrality=np.ones(len(variables), dtype=int),
        bounds=Bounds(np.zeros(len(variables)), variable_upper),
        constraints=LinearConstraint(constraint.tocsr(), lower, upper),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "no exact teacher- and token-matched control exists for "
            f"seed {seed}: {result.message}"
        )

    selected = {}
    for task_id in tasks:
        chosen = [
            teacher
            for teacher in teachers
            if result.x[variable_index[(task_id, teacher)]] > 0.5
        ]
        if len(chosen) != 1:
            raise ValueError(f"MILP selected {len(chosen)} teachers for {task_id}")
        selected[task_id] = chosen[0]
    if Counter(selected.values()) != Counter(quotas):
        raise ValueError("token-matched control violated teacher quotas")
    selected_tokens = sum(
        candidates[task_id][teacher].supervised_tokens
        for task_id, teacher in selected.items()
    )
    if selected_tokens != target_tokens:
        raise ValueError(
            f"token-matched control has {selected_tokens}, expected {target_tokens}"
        )
    selected_sequence_tokens = sum(
        candidates[task_id][teacher].sft_total_tokens
        for task_id, teacher in selected.items()
    )
    selected_sequence_squares = sum(
        candidates[task_id][teacher].sft_total_tokens**2
        for task_id, teacher in selected.items()
    )
    for label, selected_value, target_value, tolerance in (
        (
            "sequence tokens",
            selected_sequence_tokens,
            target_sequence_tokens,
            sequence_token_tolerance_fraction,
        ),
        (
            "squared sequence lengths",
            selected_sequence_squares,
            target_sequence_squares,
            sequence_square_tolerance_fraction,
        ),
    ):
        if tolerance is not None and not (
            math.ceil(target_value * (1 - tolerance))
            <= selected_value
            <= math.floor(target_value * (1 + tolerance))
        ):
            raise ValueError(f"token-matched control violated {label} tolerance")
    if disjoint and any(selected[task_id] == target[task_id] for task_id in tasks):
        raise ValueError("token-matched control overlaps its target")
    return selected


def _summary(values: list[float | int]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        "min": min(numbers),
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "max": max(numbers),
    }


def _candidate_diagnostics(
    candidates: dict[str, dict[str, Candidate]], teachers: list[str]
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    fields: dict[str, Callable[[Candidate], float]] = {
        "rsr_b": lambda candidate: candidate.rsr_b,
        "mean_nll_masked": lambda candidate: candidate.mean_nll_masked,
        "mean_clipped_rank_masked": lambda candidate: (
            candidate.mean_clipped_rank_masked
        ),
        "sft_trainable_tokens": lambda candidate: float(
            candidate.supervised_tokens
        ),
        "sft_total_tokens": lambda candidate: float(candidate.sft_total_tokens),
        "proxy_assistant_tokens": lambda candidate: float(
            candidate.proxy_assistant_tokens
        ),
    }
    per_teacher = {}
    for teacher in teachers:
        options = [per_teacher_options[teacher] for per_teacher_options in candidates.values()]
        per_teacher[teacher] = {
            field: _summary([getter(candidate) for candidate in options])
            for field, getter in fields.items()
        }
        tor_values = [candidate.tor for candidate in options if candidate.tor is not None]
        per_teacher[teacher]["tor"] = _summary(tor_values) if tor_values else None
        per_teacher[teacher]["tor_missing"] = len(options) - len(tor_values)
        turn_fields = {
            field
            for candidate in options
            for field in candidate.turn_audit
        }
        per_teacher[teacher]["turn_audit"] = {
            "totals": {
                field: sum(candidate.turn_audit.get(field, 0) for candidate in options)
                for field in sorted(turn_fields)
            },
            "trajectories_with_noncomplete_noop": sum(
                candidate.turn_audit.get("noncomplete_noop_turns", 0) > 0
                for candidate in options
            ),
            "max_noncomplete_noop_turns_per_trajectory": max(
                candidate.turn_audit.get("noncomplete_noop_turns", 0)
                for candidate in options
            ),
        }

    within_task_spreads = {
        field: _summary(
            [
                max(getter(option) for option in options.values())
                - min(getter(option) for option in options.values())
                for options in candidates.values()
            ]
        )
        for field, getter in fields.items()
    }
    centered: dict[str, list[float]] = {field: [] for field in fields}
    for options in candidates.values():
        for field, getter in fields.items():
            values = [getter(option) for option in options.values()]
            mean = statistics.fmean(values)
            centered[field].extend(value - mean for value in values)
    correlations = {}
    for field in fields:
        if field == "rsr_b":
            continue
        result = spearmanr(centered["rsr_b"], centered[field])
        correlation = float(result.statistic)
        correlations[field] = correlation if math.isfinite(correlation) else None
    difficulty_counts = Counter()
    for options in candidates.values():
        values = {
            str(option.metadata.get("difficulty")).strip()
            for option in options.values()
            if str(option.metadata.get("difficulty", "")).strip()
        }
        if len(values) > 1:
            raise ValueError(f"difficulty differs within matched task: {values}")
        difficulty_counts[next(iter(values), "unknown")] += 1
    return {
        "per_teacher": per_teacher,
        "within_task_spreads": within_task_spreads,
        "within_task_centered_spearman_with_rsr_b": correlations,
        "task_difficulty_counts": dict(difficulty_counts),
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
                    "mean_nll_masked": option.mean_nll_masked,
                    "mean_clipped_rank_masked": option.mean_clipped_rank_masked,
                    "tor": option.tor,
                    "supervised_tokens": option.supervised_tokens,
                    "sft_total_tokens": option.sft_total_tokens,
                    "sft_turns_truncated": option.sft_turns_truncated,
                    "proxy_assistant_tokens": option.proxy_assistant_tokens,
                }
                for teacher, option in candidates[task_id].items()
            }
            selected_score = (
                candidate.rsr_b
                if proxy == "rsr_b"
                else candidate.mean_nll_masked
                if proxy == "mean_nll_masked"
                else candidate.mean_clipped_rank_masked
                if proxy == "mean_clipped_rank_masked"
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
    nll_values = [candidate.mean_nll_masked for candidate in selected_candidates]
    rank_values = [
        candidate.mean_clipped_rank_masked for candidate in selected_candidates
    ]
    tor_values = [
        candidate.tor for candidate in selected_candidates if candidate.tor is not None
    ]
    turn_fields = {
        field for candidate in selected_candidates for field in candidate.turn_audit
    }
    return {
        "path": str(output.resolve()),
        "sha256": digest.hexdigest(),
        "rows": len(selected_candidates),
        "teacher_counts": dict(teacher_counts),
        "supervised_tokens_total": sum(
            candidate.supervised_tokens for candidate in selected_candidates
        ),
        "sft_total_tokens_total": sum(
            candidate.sft_total_tokens for candidate in selected_candidates
        ),
        "sft_total_tokens_squared_sum": sum(
            candidate.sft_total_tokens**2 for candidate in selected_candidates
        ),
        "proxy_assistant_tokens_total": sum(
            candidate.proxy_assistant_tokens for candidate in selected_candidates
        ),
        "supervised_tokens_per_trajectory": _summary(
            [candidate.supervised_tokens for candidate in selected_candidates]
        ),
        "selected_rsr_b": _summary(rsr_values),
        "selected_mean_nll_masked": _summary(nll_values),
        "selected_mean_clipped_rank_masked": _summary(rank_values),
        "selected_tor": _summary(tor_values) if tor_values else None,
        "selected_tor_missing": len(selected_candidates) - len(tor_values),
        "selected_turn_audit": {
            "totals": {
                field: sum(
                    candidate.turn_audit.get(field, 0)
                    for candidate in selected_candidates
                )
                for field in sorted(turn_fields)
            },
            "trajectories_with_noncomplete_noop": sum(
                candidate.turn_audit.get("noncomplete_noop_turns", 0) > 0
                for candidate in selected_candidates
            ),
            "max_noncomplete_noop_turns_per_trajectory": max(
                candidate.turn_audit.get("noncomplete_noop_turns", 0)
                for candidate in selected_candidates
            ),
        },
    }


def build_mixes(
    candidates: dict[str, dict[str, Candidate]],
    teachers: list[str],
    random_seeds: list[int],
) -> dict[str, tuple[str, str, dict[str, str]]]:
    low_rsr = lambda candidate: -candidate.rsr_b
    high_rsr = lambda candidate: candidate.rsr_b
    low_nll = lambda candidate: -candidate.mean_nll_masked
    low_rank = lambda candidate: -candidate.mean_clipped_rank_masked
    # TOR is undefined when the parser finds no state-changing action. Keep
    # those candidates last instead of dropping the fixed-task arm. A value
    # of -1 is below every defined TOR (which lies in [0, 1]); if all four are
    # undefined, the stable tie-break decides. The written selected score is
    # still null, so the fallback remains explicit in the audit.
    high_tor = lambda candidate: (
        candidate.tor if candidate.tor is not None else -1.0
    )
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
            "higher_better_undefined_last_stable_tie",
            _best_per_task(candidates, teachers, high_tor),
        ),
        "nll_low": (
            "mean_nll_masked",
            "lower_better",
            _best_per_task(candidates, teachers, low_nll),
        ),
        "rank_low": (
            "mean_clipped_rank_masked",
            "lower_better",
            _best_per_task(candidates, teachers, low_rank),
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
            "higher_better_undefined_last_stable_tie",
            _balanced_selection(candidates, teachers, high_tor),
        ),
        "nll_low_teacher_balanced": (
            "mean_nll_masked",
            "lower_better",
            _balanced_selection(candidates, teachers, low_nll),
        ),
        "rank_low_teacher_balanced": (
            "mean_clipped_rank_masked",
            "lower_better",
            _balanced_selection(candidates, teachers, low_rank),
        ),
    }
    low_balanced = arms["rsr_low_teacher_balanced"][2]
    arms["rsr_high_token_matched_to_rsr_low"] = (
        "rsr_b",
        "higher_better_exact_teacher_and_token_match_to_rsr_low_teacher_balanced",
        _balanced_token_matched_selection(
            candidates,
            teachers,
            low_balanced,
            1042,
            disjoint=True,
            utility=high_rsr,
        ),
    )
    arms["rsr_high_compute_matched_to_rsr_low"] = (
        "rsr_b",
        "higher_better_exact_teacher_target_token_and_compute_match_to_rsr_low_teacher_balanced",
        _balanced_token_matched_selection(
            candidates,
            teachers,
            low_balanced,
            2042,
            disjoint=True,
            utility=high_rsr,
            sequence_token_tolerance_fraction=0.0025,
            sequence_square_tolerance_fraction=0.01,
        ),
    )
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
        arms[f"token_matched_to_rsr_low_s{seed}"] = (
            "length_control",
            "exact_total_match_to_rsr_low_teacher_balanced",
            _balanced_token_matched_selection(
                candidates, teachers, low_balanced, seed, disjoint=True
            ),
        )
        arms[f"compute_matched_to_rsr_low_s{seed}"] = (
            "length_and_compute_control",
            "exact_teacher_and_target_token_match_plus_sequence_compute_bounds_to_rsr_low_teacher_balanced",
            _balanced_token_matched_selection(
                candidates,
                teachers,
                low_balanced,
                seed,
                disjoint=True,
                sequence_token_tolerance_fraction=0.0025,
                sequence_square_tolerance_fraction=0.01,
            ),
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
    parser.add_argument(
        "--sft-token-audit",
        action="append",
        type=_parse_labeled_path,
        default=[],
        metavar="TEACHER=PATH",
        help=(
            "Exact trainable-token audit for the planned SFT template/cutoff. "
            "Required for training-ready mixes; if omitted, historical proxy-mask "
            "counts are retained only for backward compatibility."
        ),
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
    sft_token_paths = dict(args.sft_token_audit)
    if len(trajectory_paths) != len(args.trajectory):
        raise SystemExit("duplicate teacher label in --trajectory")
    if len(score_paths) != len(args.rsr_score):
        raise SystemExit("duplicate teacher label in --rsr-score")
    if len(sft_token_paths) != len(args.sft_token_audit):
        raise SystemExit("duplicate teacher label in --sft-token-audit")
    random_seeds = [
        int(value) for value in args.random_seeds.split(",") if value.strip()
    ]
    if not random_seeds:
        raise SystemExit("--random-seeds cannot be empty")
    sft_audit_provenance = (
        _load_sft_audit_provenance(sft_token_paths, args.student_revision)
        if sft_token_paths
        else None
    )

    candidates, teachers, candidate_coverage = load_candidates(
        trajectory_paths,
        score_paths,
        sft_token_paths or None,
    )
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
    rsr_low_selection = arms["rsr_low"][2]
    for name, (_, _, selection) in arms.items():
        overlap = sum(
            selection[task_id] == rsr_low_selection[task_id]
            for task_id in rsr_low_selection
        )
        arm_reports[name]["trajectory_overlap_with_rsr_low"] = overlap
        arm_reports[name]["trajectory_overlap_fraction_with_rsr_low"] = (
            overlap / len(rsr_low_selection)
        )
    token_match_target = "rsr_low_teacher_balanced"
    target_report = arm_reports[token_match_target]
    target_selection = arms[token_match_target][2]
    token_matched_names = [
        "rsr_high_token_matched_to_rsr_low",
        "rsr_high_compute_matched_to_rsr_low",
        *(f"token_matched_to_rsr_low_s{seed}" for seed in random_seeds),
        *(f"compute_matched_to_rsr_low_s{seed}" for seed in random_seeds),
    ]
    for name in token_matched_names:
        selection = arms[name][2]
        arm_reports[name].update(
            {
                "matched_to": token_match_target,
                "teacher_counts_match": (
                    arm_reports[name]["teacher_counts"]
                    == target_report["teacher_counts"]
                ),
                "supervised_tokens_delta": (
                    arm_reports[name]["supervised_tokens_total"]
                    - target_report["supervised_tokens_total"]
                ),
                "sft_total_tokens_delta": (
                    arm_reports[name]["sft_total_tokens_total"]
                    - target_report["sft_total_tokens_total"]
                ),
                "sft_total_tokens_squared_delta": (
                    arm_reports[name]["sft_total_tokens_squared_sum"]
                    - target_report["sft_total_tokens_squared_sum"]
                ),
                "trajectory_overlap": sum(
                    selection[task_id] == target_selection[task_id]
                    for task_id in selection
                ),
            }
        )
    task_ids = sorted(candidates)
    report = {
        "kind": "terminal_lego_proxy_selected_sft_mix",
        "schema_version": 4,
        "student": {"model": args.student, "revision": args.student_revision},
        "n_matched_tasks": len(task_ids),
        "task_ids_sha256": _sha256_text(task_ids),
        "teachers": teachers,
        "selection_unit": "one complete teacher trajectory per matched task",
        "rsr_direction": "lower_better",
        "rsr_scoring_semantics": REQUIRED_SCORE_SEMANTICS,
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
            "sft_token_audits": {
                teacher: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                }
                for teacher, path in sft_token_paths.items()
            },
            "sft_token_audit_provenance": sft_audit_provenance,
        },
        "supervised_token_semantics": (
            "exact LlamaFactory target tokens from --sft-token-audit"
            if sft_token_paths
            else "fallback proxy-scored assistant tokens; not training-ready"
        ),
        "compute_token_semantics": (
            "sft_total_tokens records non-padding sequence tokens processed by "
            "the audited template/cutoff; squared sequence-length sums audit the "
            "quadratic attention component. The compute_matched arms constrain "
            "these to 0.25% and 1% of rsr_low_teacher_balanced, respectively."
        ),
        "candidate_diagnostics": _candidate_diagnostics(candidates, teachers),
        "candidate_coverage": candidate_coverage,
        "arms": arm_reports,
        "training_warning": (
            "The ordinary arms contain complete matched-task trajectories but "
            "their raw target- and sequence-token totals differ. The token_matched "
            "arms exactly match trainable assistant target tokens and teacher "
            "counts for rsr_low_teacher_balanced. Prefer compute_matched arms for "
            "training: they additionally bound total sequence tokens and squared "
            "sequence lengths; all residual deltas remain reported."
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
                        "sft_total_tokens_total": arm["sft_total_tokens_total"],
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
