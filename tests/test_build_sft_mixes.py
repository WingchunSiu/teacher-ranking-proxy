import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import build_sft_mixes as mixes


def _candidate(
    task: str,
    teacher: str,
    rsr: float,
    tor: float | None,
    supervised_tokens: int = 10,
):
    return mixes.Candidate(
        teacher=teacher,
        task_id=task,
        trajectory_id=f"{teacher}::{task}",
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "answer"},
        ],
        metadata={},
        rsr_b=rsr,
        mean_nll_masked=rsr + 0.25,
        mean_clipped_rank_masked=rsr + 0.5,
        supervised_tokens=supervised_tokens,
        sft_total_tokens=supervised_tokens + 5,
        sft_turns_truncated=0,
        proxy_assistant_tokens=supervised_tokens,
        tor=tor,
        tor_components={},
        turn_audit={},
    )


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.teachers = ["A", "B"]
        self.candidates = {
            f"t{index}": {
                "A": _candidate(f"t{index}", "A", index + 1, 0.1),
                "B": _candidate(f"t{index}", "B", 10 - index, 0.9),
            }
            for index in range(4)
        }

    def test_per_task_rsr_directions_are_opposites(self):
        low = mixes._best_per_task(
            self.candidates, self.teachers, lambda candidate: -candidate.rsr_b
        )
        high = mixes._best_per_task(
            self.candidates, self.teachers, lambda candidate: candidate.rsr_b
        )
        self.assertEqual(low, {"t0": "A", "t1": "A", "t2": "A", "t3": "A"})
        self.assertEqual(high, {"t0": "B", "t1": "B", "t2": "B", "t3": "B"})

    def test_balanced_selection_enforces_exact_quotas(self):
        selected = mixes._balanced_selection(
            self.candidates, self.teachers, lambda candidate: -candidate.rsr_b
        )
        self.assertEqual(sorted(selected), sorted(self.candidates))
        self.assertEqual(list(selected.values()).count("A"), 2)
        self.assertEqual(list(selected.values()).count("B"), 2)

    def test_random_selection_is_seeded(self):
        first = mixes._random_selection(self.candidates, self.teachers, 42)
        second = mixes._random_selection(self.candidates, self.teachers, 42)
        different = mixes._random_selection(self.candidates, self.teachers, 43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_balanced_random_is_seeded_and_balanced(self):
        first = mixes._balanced_random_selection(self.candidates, self.teachers, 42)
        second = mixes._balanced_random_selection(self.candidates, self.teachers, 42)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("A"), 2)
        self.assertEqual(list(first.values()).count("B"), 2)

    def test_token_matched_control_is_exact_balanced_and_disjoint(self):
        target = mixes._balanced_selection(
            self.candidates, self.teachers, lambda candidate: -candidate.rsr_b
        )
        control = mixes._balanced_token_matched_selection(
            self.candidates, self.teachers, target, 42, disjoint=True
        )
        target_tokens = sum(
            self.candidates[task][teacher].supervised_tokens
            for task, teacher in target.items()
        )
        control_tokens = sum(
            self.candidates[task][teacher].supervised_tokens
            for task, teacher in control.items()
        )
        self.assertEqual(target_tokens, control_tokens)
        self.assertEqual(list(control.values()).count("A"), 2)
        self.assertEqual(list(control.values()).count("B"), 2)
        self.assertTrue(all(control[task] != target[task] for task in target))

    def test_token_matched_random_default_permits_target_overlap(self):
        target = mixes._balanced_selection(
            self.candidates, self.teachers, lambda candidate: -candidate.rsr_b
        )
        control = mixes._balanced_token_matched_selection(
            self.candidates, self.teachers, target, 42
        )
        self.assertTrue(any(control[task] == target[task] for task in target))
        self.assertEqual(list(control.values()).count("A"), 2)
        self.assertEqual(list(control.values()).count("B"), 2)

    def test_token_matched_metric_selection_optimizes_subject_to_controls(self):
        target = mixes._balanced_selection(
            self.candidates, self.teachers, lambda candidate: -candidate.rsr_b
        )
        selected = mixes._balanced_token_matched_selection(
            self.candidates,
            self.teachers,
            target,
            42,
            disjoint=True,
            utility=lambda candidate: candidate.rsr_b,
        )
        self.assertEqual(
            sum(
                self.candidates[task][teacher].supervised_tokens
                for task, teacher in selected.items()
            ),
            sum(
                self.candidates[task][teacher].supervised_tokens
                for task, teacher in target.items()
            ),
        )
        self.assertTrue(all(selected[task] != target[task] for task in target))

    def test_compute_matched_control_respects_sequence_bands(self):
        target = mixes._balanced_selection(
            self.candidates, self.teachers, lambda candidate: -candidate.rsr_b
        )
        selected = mixes._balanced_token_matched_selection(
            self.candidates,
            self.teachers,
            target,
            42,
            disjoint=True,
            sequence_token_tolerance_fraction=0.0,
            sequence_square_tolerance_fraction=0.0,
        )
        for transform in (
            lambda candidate: candidate.supervised_tokens,
            lambda candidate: candidate.sft_total_tokens,
            lambda candidate: candidate.sft_total_tokens**2,
        ):
            self.assertEqual(
                sum(
                    transform(self.candidates[task][teacher])
                    for task, teacher in selected.items()
                ),
                sum(
                    transform(self.candidates[task][teacher])
                    for task, teacher in target.items()
                ),
            )

    def test_undefined_tor_is_kept_last_with_auditable_fallback(self):
        teachers = ["A", "B"]
        candidates = {
            "all-missing": {
                "A": _candidate("all-missing", "A", 1.0, None),
                "B": _candidate("all-missing", "B", 2.0, None),
            },
            "one-defined": {
                "A": _candidate("one-defined", "A", 1.0, None),
                "B": _candidate("one-defined", "B", 2.0, 0.0),
            },
        }

        arms = mixes.build_mixes(candidates, teachers, [42])
        tor_selection = arms["tor_high"][2]

        self.assertEqual(tor_selection["one-defined"], "B")
        self.assertIn(tor_selection["all-missing"], teachers)
        self.assertIsNone(
            candidates["all-missing"][tor_selection["all-missing"]].tor
        )

    def test_load_candidates_joins_exact_trajectory_and_sft_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_paths = {}
            score_paths = {}
            token_paths = {}
            for teacher, rsr, tokens in (("A", 1.0, 123), ("B", 2.0, 456)):
                trajectory_id = f"{teacher}::task-1"
                trajectory_path = root / f"{teacher}-trajectory.json"
                trajectory_path.write_text(
                    json.dumps(
                        [
                            {
                                "trajectory_id": trajectory_id,
                                "task_id": "task-1",
                                "messages": [
                                    {"role": "user", "content": "task"},
                                    {"role": "assistant", "content": "answer"},
                                ],
                            }
                        ]
                    )
                )
                score_path = root / f"{teacher}-score.parquet"
                pq.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "trajectory_id": trajectory_id,
                                "task_id": "task-1",
                                "scoring_semantics": mixes.REQUIRED_SCORE_SEMANTICS,
                                "rsr_b": rsr,
                                "mean_nll_masked": rsr + 0.1,
                                "mean_clipped_rank_masked": rsr + 0.2,
                                "num_tokens_masked": 10,
                            }
                        ]
                    ),
                    score_path,
                )
                token_path = root / f"{teacher}-tokens.parquet"
                pq.write_table(
                    pa.Table.from_pylist(
                        [
                            {
                                "trajectory_id": trajectory_id,
                                "task_id": "task-1",
                                "sft_total_tokens": tokens + 50,
                                "sft_trainable_tokens": tokens,
                                "sft_turns_truncated": 0,
                            }
                        ]
                    ),
                    token_path,
                )
                trajectory_paths[teacher] = trajectory_path
                score_paths[teacher] = score_path
                token_paths[teacher] = token_path

            candidates, teachers, coverage = mixes.load_candidates(
                trajectory_paths, score_paths, token_paths
            )

            self.assertEqual(teachers, ["A", "B"])
            self.assertEqual(candidates["task-1"]["A"].trajectory_id, "A::task-1")
            self.assertEqual(candidates["task-1"]["A"].supervised_tokens, 123)
            self.assertEqual(candidates["task-1"]["A"].sft_total_tokens, 173)
            self.assertEqual(candidates["task-1"]["B"].supervised_tokens, 456)
            self.assertEqual(coverage["complete_four_way_tasks"], 1)

            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "trajectory_id": "A::task-1",
                            "task_id": "task-1",
                            "sft_total_tokens": 173,
                            "sft_trainable_tokens": 123,
                            "sft_turns_truncated": 1,
                        }
                    ]
                ),
                token_paths["A"],
            )
            with self.assertRaisesRegex(ValueError, "no task has four complete"):
                mixes.load_candidates(trajectory_paths, score_paths, token_paths)


if __name__ == "__main__":
    unittest.main()
