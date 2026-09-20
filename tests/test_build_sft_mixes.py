import unittest

import build_sft_mixes as mixes


def _candidate(
    task: str, teacher: str, rsr: float, tor: float, supervised_tokens: int = 10
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
        supervised_tokens=supervised_tokens,
        tor=tor,
        tor_components={},
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


if __name__ == "__main__":
    unittest.main()
