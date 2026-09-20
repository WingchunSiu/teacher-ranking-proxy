from __future__ import annotations

import json
from pathlib import Path

import materialize_terminal_lego_subset as subset


def _record(task_id: str, difficulty: str = "easy"):
    return {
        "conversations": [
            {"from": "human", "value": f"solve {task_id}"},
            {"from": "gpt", "value": '{"task_complete": true}'},
        ],
        "metadata": {"oracle_passed_task": task_id, "difficulty": difficulty},
    }


def test_materialize_has_four_candidates_and_identical_task_sets(tmp_path: Path):
    teachers = ["Teacher A", "Teacher B", "Teacher C", "Teacher D"]
    task_ids = [f"task_{index:05d}" for index in range(5)]
    trajectory_dir = tmp_path / "source"
    trajectory_dir.mkdir()
    teacher_files = {}
    for teacher in teachers:
        filename = f"{teacher.lower().replace(' ', '-')}.json"
        teacher_files[teacher] = filename
        (trajectory_dir / filename).write_text(
            json.dumps([_record(task_id) for task_id in task_ids])
        )

    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "task_manifest_header",
                    "traj_repo": "example/data",
                    "traj_revision": "abc123",
                    "teachers": teachers,
                    "teacher_files": teacher_files,
                }
            )
            + "\n"
        )
        for index, task_id in enumerate(task_ids):
            handle.write(
                json.dumps(
                    {
                        "task_id": task_id,
                        "teacher_record_index": {
                            teacher: index for teacher in teachers
                        },
                    }
                )
                + "\n"
            )

    report = subset.materialize(
        manifest_path=manifest,
        trajectory_dir=trajectory_dir,
        output_dir=tmp_path / "out",
        report_path=tmp_path / "report.json",
        n_tasks=3,
        seed=42,
    )

    assert report["n_tasks"] == 3
    assert report["candidates_per_task"] == 4
    assert report["task_input_alignment"] == {
        "task_description_exact_across_teachers": 3,
        "initial_user_message_exact_across_teachers": 3,
        "total_tasks": 3,
        "interpretation": (
            "Task descriptions are compared after removing the run-specific "
            "Current terminal state block; full prompts can differ by container "
            "hostname and shell banner."
        ),
    }
    assert report["environment_provenance"][
        "independently_verifiable_from_release"
    ] is False
    task_sets = []
    for output in report["outputs"].values():
        rows = json.loads(Path(output["path"]).read_text())
        assert len(rows) == 3
        assert all(len(row["messages"]) == 2 for row in rows)
        task_sets.append({row["task_id"] for row in rows})
    assert all(task_set == task_sets[0] for task_set in task_sets)
